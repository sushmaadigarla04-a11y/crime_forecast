"""Crime Forecaster - interactive dashboard (paper Sec. IV).

    streamlit run app.py

Requires the artefacts produced by `python scripts/run_pipeline.py`.
"""
from __future__ import annotations

import json
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import explain as xp
from src import features as fx
from src import forecast as fc
from src import weather as wx
from src.geo import area_name
from src.models import specialist_path

st.set_page_config(page_title="Crime Forecaster", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading data and models...")
def load_assets():
    if not config.REGISTRY_JSON.exists():
        return None
    reg = json.loads(config.REGISTRY_JSON.read_text())
    panel = pd.read_parquet(config.PANEL_PARQUET)
    beats = pd.read_parquet(config.BEATS_PARQUET)
    beats["AreaName"] = beats["CommunityArea"].map(area_name)
    weather = pd.read_csv(config.WEATHER_CSV, parse_dates=["Date"])
    clim = wx.climatology(weather)
    profiles = pd.read_csv(config.REPORTS_DIR / "cluster_profiles.csv")
    models = {(t, c): joblib.load(specialist_path(t, c))
              for t in reg["targets"] for c in range(reg["n_clusters"])}
    return dict(reg=reg, panel=panel, beats=beats, weather=weather, clim=clim,
                profiles=profiles, models=models)


A = load_assets()
if A is None:
    st.error("No trained models found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

reg, panel, beats, weather, clim, models = (A["reg"], A["panel"], A["beats"], A["weather"],
                                            A["clim"], A["models"])
types: list[str] = reg["types"]
beat_codes = {int(k): v for k, v in reg["beat_codes"].items()}
cluster_map = dict(zip(beats["Beat"].astype(int), beats["Cluster"].astype(int)))
cluster_names = dict(zip(A["profiles"]["Cluster"], A["profiles"]["ClusterName"]))
data_end = pd.Timestamp(reg["data_end"])
test_start = pd.Timestamp(reg["test_start"])


@st.cache_data(show_spinner=False)
def run_forecast(beat: int, start: pd.Timestamp, horizon: int) -> dict[str, pd.DataFrame]:
    out = {}
    for t in reg["targets"]:
        bm = {int(k): v for k, v in reg["beat_means"][t].items()}
        out[t] = fc.recursive_forecast(panel, t, [beat], start, horizon,
                                       lambda c, t=t: models[(t, c)], beat_codes, bm,
                                       cluster_map, weather, clim)
    return out


def actual_series(beat: int, target: str, lo, hi) -> pd.DataFrame:
    p = panel[(panel["Beat"] == beat) & (panel["Date"] >= lo) & (panel["Date"] <= hi)]
    return p[["Date", target]].rename(columns={target: "actual"})


def pretty(t: str) -> str:
    return "Overall crime total" if t == "total" else t.title()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Forecasting options")
area_beats = beats.groupby("AreaName")["Beat"].apply(sorted).to_dict()
area = st.sidebar.selectbox("1. Area (community area)", sorted(area_beats),
                            index=sorted(area_beats).index("Austin") if "Austin" in area_beats else 0)
beat = st.sidebar.selectbox(
    "2. Patrol zone (police beat)", area_beats[area],
    format_func=lambda b: f"Beat {b:04d} - cluster {cluster_map[b]}")
start = st.sidebar.date_input(
    "3. Forecast start date", value=(data_end + timedelta(days=1)).date(),
    min_value=test_start.date(), max_value=(data_end + timedelta(days=1)).date(),
    help="Dates up to the end of the data show what actually happened next to the forecast "
         "(a backtest). The last date is a true forecast beyond the data.")
horizon = st.sidebar.slider("4. Number of days to forecast", 1, config.MAX_HORIZON_DAYS, 7)
mode = st.sidebar.radio("5. Prediction type", ["Overall crime total", "Specific crime type"])
target = "total" if mode == "Overall crime total" else st.sidebar.selectbox(
    "Crime type", types, format_func=str.title)
st.sidebar.caption(f"Data: {reg['start_date']} to {reg['data_end']} - test period from {reg['test_start']}")

# ------------------------------------------------------------------ layout
st.title("📈 Crime Forecaster")
st.caption("k-Shape clustering + specialist LightGBM models + SHAP explanations - Chicago police beats")
tab_fc, tab_ov, tab_cl, tab_perf, tab_about = st.tabs(
    ["Forecast", "City overview", "Crime patterns (clusters)", "Model performance", "About"])

# ------------------------------------------------------------------ forecast tab
with tab_fc:
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(days=horizon - 1)
    c = cluster_map[beat]
    with st.spinner("Running specialist models..."):
        fcs = run_forecast(beat, start_ts, horizon)
    f = fcs[target].copy()
    f["low"], f["high"] = fc.poisson_interval(f["pred"].to_numpy(), 0.8)
    act = actual_series(beat, target, start_ts - pd.Timedelta(days=42), end_ts)
    in_window = act[act["Date"] >= start_ts]
    has_actuals = len(in_window) == horizon

    st.subheader(f"{pretty(target)} - Beat {beat:04d}, {area}")
    st.write(f"Crime-pattern cluster **{c}: {cluster_names[c]}**")

    typical = reg["beat_means"][target][str(beat)] * horizon
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Forecast, {horizon} day{'s' * (horizon > 1)}", f"{f['pred'].sum():.1f}",
              delta=f"{100 * (f['pred'].sum() / typical - 1):+.0f}% vs beat's long-run average"
              if typical > 0 else None, delta_color="inverse")
    lo_sum, hi_sum = fc.poisson_interval(np.array([f["pred"].sum()]), 0.8)
    k2.metric("80% range (Poisson)", f"{lo_sum[0]:.0f} - {hi_sum[0]:.0f}")
    if has_actuals:
        k3.metric("Actually recorded", f"{in_window['actual'].sum():.0f}")
    else:
        k3.metric("Actually recorded", "not yet known")

    fig = go.Figure()
    hist = act[act["Date"] < start_ts]
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["actual"], name="History (recorded)",
                             mode="lines+markers", line=dict(color="#8a8f98"), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=pd.concat([f["Date"], f["Date"][::-1]]),
                             y=pd.concat([f["high"], f["low"][::-1]]), fill="toself",
                             fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
                             name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["pred"], name="Forecast", mode="lines+markers",
                             line=dict(color="#2563eb", width=3)))
    if len(in_window):
        fig.add_trace(go.Scatter(x=in_window["Date"], y=in_window["actual"], name="Recorded (test data)",
                                 mode="markers", marker=dict(color="#dc2626", size=9, symbol="x")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="crimes per day", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    if f["weather_is_climatology"].any():
        st.caption("For dates after the weather record ends, the forecast uses the average "
                   "weather for that time of year (climatology).")

    st.subheader(f"Crime type breakdown ({horizon}-day total)")
    br = pd.DataFrame({"Crime type": [t.title() for t in types],
                       "Forecast": [fcs[t]["pred"].sum() for t in types]})
    other = max(fcs["total"]["pred"].sum() - br["Forecast"].sum(), 0)
    br = pd.concat([br, pd.DataFrame({"Crime type": ["All other types"], "Forecast": [other]})])
    br = br.sort_values("Forecast")
    st.plotly_chart(px.bar(br, x="Forecast", y="Crime type", orientation="h", height=420)
                    .update_layout(margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
    st.caption("Each type comes from its own specialist model, so the types do not have to add "
               "up exactly to the total forecast.")

    st.subheader("Why this forecast?")
    day_opts = list(f["Date"].dt.date)
    day = st.select_slider("Explain the forecast for", options=day_opts, value=day_opts[0])
    row = f[f["Date"].dt.date == day]
    table, baseline, pred = xp.explain_single(models[(target, c)], row)
    st.write(f"Typical forecast from this cluster's model: **{baseline:.2f}**. "
             f"Forecast for {day:%a %d %b %Y}: **{pred:.2f}**. The bars show how much each "
             "factor raised (red) or lowered (blue) the forecast for this beat and day.")
    top = table.head(10).iloc[::-1]
    top["shown"] = [f"{l}  (= {v:.1f})" if isinstance(v, (int, float, np.floating)) else l
                    for l, v in zip(top["label"], top["value"])]
    figx = go.Figure(go.Bar(x=top["effect_pct"], y=top["shown"], orientation="h",
                            marker_color=np.where(top["effect_pct"] > 0, "#dc2626", "#2563eb"),
                            text=[f"{v:+.1f}%" for v in top["effect_pct"]], textposition="outside"))
    figx.update_layout(height=420, margin=dict(l=10, r=40, t=10, b=10),
                       xaxis_title="effect on forecast (%)")
    st.plotly_chart(figx, width="stretch")
    with st.expander("All factors (SHAP values)"):
        st.dataframe(table[["label", "value", "shap_log", "effect_pct"]].rename(columns={
            "label": "Feature", "value": "Value", "shap_log": "SHAP (log scale)",
            "effect_pct": "Effect %"}), width="stretch", hide_index=True)

    with st.expander("Daily forecast table"):
        tbl = f[["Date", "pred", "low", "high", "TMAX", "PRCP", "is_holiday"]].merge(
            in_window, on="Date", how="left")
        st.dataframe(tbl.rename(columns={"pred": "Forecast", "low": "80% low", "high": "80% high",
                                         "actual": "Recorded"}),
                     width="stretch", hide_index=True)

# ------------------------------------------------------------------ overview tab
with tab_ov:
    pb = panel.merge(beats[["Beat", "AreaName"]], on="Beat")
    by_area = pb.groupby("AreaName")["total"].sum().sort_values()
    by_type = panel[types].sum().sort_values(ascending=False)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recorded incidents", f"{int(panel['total'].sum()):,}")
    k2.metric("Most active area", by_area.index[-1])
    k3.metric("Quietest area", by_area.index[0])
    k4.metric("Most common crime type", by_type.index[0].title())

    left, right = st.columns([3, 2])
    with left:
        daily = panel.groupby("Date")["total"].sum().rolling(28).mean().reset_index()
        st.plotly_chart(px.line(daily, x="Date", y="total", height=330,
                                title="City-wide daily incidents (28-day average)")
                        .add_vline(x=test_start, line_dash="dot")
                        .update_layout(margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None),
                        width="stretch")
    with right:
        st.plotly_chart(px.bar(x=by_type.values, y=[t.title() for t in by_type.index],
                               orientation="h", height=330, title="Incidents by type (top 10)")
                        .update_layout(margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                                       xaxis_title=None, yaxis_title=None),
                        width="stretch")

    m = beats.dropna(subset=["Latitude", "Longitude"]).copy()
    m["Pattern"] = m["Cluster"].map(lambda k: f"C{k}: {cluster_names[k]}")
    m["Selected"] = np.where(m["Beat"] == beat, "selected", "")
    mapfig = px.scatter_map(m, lat="Latitude", lon="Longitude", color="Pattern", size="Incidents",
                            hover_name=m["Beat"].map(lambda b: f"Beat {b:04d}"),
                            hover_data={"AreaName": True, "Incidents": ":,", "Latitude": False,
                                        "Longitude": False}, zoom=9.3, height=560,
                            center=dict(lat=41.84, lon=-87.68), size_max=14,
                            category_orders={"Pattern": sorted(m["Pattern"].unique())})
    mapfig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title="Police beats coloured by k-Shape cluster",
                         legend=dict(font=dict(size=10)))
    st.plotly_chart(mapfig, width="stretch")

# ------------------------------------------------------------------ clusters tab
with tab_cl:
    st.markdown("Police beats are grouped by the **shape** of their daily crime series "
                "(z-normalised, k-Shape), not by how much crime they have. Each group gets its own "
                "LightGBM models.")
    prof = A["profiles"].copy()
    st.dataframe(prof.rename(columns={
        "ClusterName": "Pattern", "n_beats": "Beats", "mean_daily_crimes": "Avg crimes/day/beat",
        "weekend_ratio": "Weekend / weekday", "summer_winter_ratio": "Summer / winter",
        "day_to_day_cv": "Day-to-day variation (CV)"}).round(3),
        width="stretch", hide_index=True)
    st.image(str(config.FIGURES_DIR / "cluster_shapes.png"), width="stretch")
    if (config.FIGURES_DIR / "elbow_kshape.png").exists():
        st.image(str(config.FIGURES_DIR / "elbow_kshape.png"), width=620)
    for k in range(reg["n_clusters"]):
        with st.expander(f"Cluster {k}: {cluster_names[k]}"):
            bl = beats[beats["Cluster"] == k][["Beat", "AreaName", "District", "Incidents"]]
            st.dataframe(bl.sort_values("Incidents", ascending=False), hide_index=True,
                         width="stretch")

# ------------------------------------------------------------------ performance tab
with tab_perf:
    one_p = config.REPORTS_DIR / "metrics_one_step.csv"
    if not one_p.exists():
        st.info("Run the `evaluate` step of the pipeline to see test-set metrics.")
    else:
        one = pd.read_csv(one_p)
        st.markdown(f"Everything here was measured on **held-out test data** ({reg['test_start']} to "
                    f"{reg['data_end']}), which no model saw during training or clustering.")
        lvl = st.radio("Aggregation level", ["beat-week", "beat-day", "city-day"], horizontal=True)
        tgt = st.selectbox("Target", reg["targets"], format_func=pretty, key="perf_t")
        t = one[(one["target"] == tgt) & (one["level"] == lvl) & (one["cluster"].astype(str) == "all")]
        st.dataframe(t[["model", "MAE", "RMSE", "SMAPE_%", "R2", "Bias", "n"]].round(3),
                     hide_index=True, width="stretch")
        rec_p = config.REPORTS_DIR / "metrics_recursive.csv"
        if rec_p.exists() and tgt == "total" and lvl != "city-day":
            rec = pd.read_csv(rec_p)
            st.markdown("**7-day recursive forecast** (what the dashboard does):")
            st.dataframe(rec[rec["level"] == lvl][["model", "MAE", "RMSE", "SMAPE_%", "R2", "Bias", "n"]]
                         .round(3), hide_index=True, width="stretch")
        for img in ("mae_by_cluster.png", "mae_by_crime_type.png", "shap_importance_heatmap.png"):
            p = config.FIGURES_DIR / img
            if p.exists():
                st.image(str(p), width="stretch")

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown(f"""
### Method
1. **Data**: Chicago Data Portal *Crimes - 2001 to Present*, {reg['start_date']} onward, aggregated
   to daily counts for {len(beat_codes)} police beats. Daily weather is from NOAA GHCN-Daily, Chicago
   O'Hare (USW00094846). Holidays are US + Illinois public holidays.
2. **k-Shape clustering** groups beats by the shape of their daily crime series into
   {reg['n_clusters']} patterns.
3. **Specialist LightGBM models** (Poisson objective): one per cluster for the overall total and for
   each of the top {len(types)} crime types, so {reg['n_clusters'] * len(reg['targets'])} models.
   Features: calendar, holidays, weather, lags (1, 7, 14, 21, 28, 364 days), rolling means, and beat identity.
4. **Recursive forecasting**: each day's forecast is used as input for the next day.
5. **SHAP** (TreeSHAP) explains every individual forecast.

### Limitations
* Recorded crime reflects reporting and police activity, not only underlying crime.
* Daily counts per beat are small and noisy. Forecasts are expected values with wide ranges,
  not certainties.
* For backtests, the model uses the weather that was actually observed. For true future dates it
  uses average weather for that time of year.

### Responsible use
This is a research prototype. Forecasts built from historical police records can repeat and
amplify existing patrol and reporting bias. Do not use them as the only basis for enforcement
decisions about specific neighbourhoods or people.
""")
