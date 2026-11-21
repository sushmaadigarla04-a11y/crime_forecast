"""Stage 4 - specialist LightGBM models + evaluation (paper Sec. III-D).

For every target (overall total + each top-N crime type) we train
  * one specialist model per k-Shape cluster  -> 6 + 6x10 = 66 models (paper)
  * one global model on all beats              -> ablation: does clustering help?
and compare them on the held-out test period against two naive baselines
(same day last week, 28-day moving average).

Training is chronological: fit on data before VALID_START, early-stop on
[VALID_START, TEST_START), then refit on the whole training period with the
chosen number of trees. Nothing from TEST_START onward is ever seen.
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from src import features as fx
from src import forecast as fc
from src import weather as wx

log = logging.getLogger(__name__)
PRED_DIR = config.REPORTS_DIR / "predictions"


# ---------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    denom = np.abs(y) + np.abs(p)
    smape = np.where(denom == 0, 0.0, 2 * np.abs(err) / np.where(denom == 0, 1, denom))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "SMAPE_%": float(100 * smape.mean()),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "Bias": float(err.mean()),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------- paths
def specialist_path(target: str, cluster: int):
    return config.MODELS_DIR / "specialist" / fx.safe_name(target) / f"cluster_{cluster}.joblib"


def global_path(target: str):
    return config.MODELS_DIR / "global" / f"{fx.safe_name(target)}.joblib"


# ---------------------------------------------------------------- training
def _fit(train: pd.DataFrame, target: str) -> tuple[lgb.LGBMRegressor, int]:
    fit = train[train["Date"] < config.VALID_START]
    val = train[train["Date"] >= config.VALID_START]
    m = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    m.fit(fit[fx.FEATURES], fit[target], eval_set=[(val[fx.FEATURES], val[target])],
          eval_metric="poisson", categorical_feature=fx.CATEGORICAL,
          callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])
    best = int(m.best_iteration_ or config.LGBM_PARAMS["n_estimators"])
    n_final = max(50, int(round(best * 1.1)))   # a little more data -> a few more trees
    final = lgb.LGBMRegressor(**{**config.LGBM_PARAMS, "n_estimators": n_final})
    final.fit(train[fx.FEATURES], train[target], categorical_feature=fx.CATEGORICAL)
    return final, n_final


def load_context():
    panel = pd.read_parquet(config.PANEL_PARQUET)
    beats = pd.read_parquet(config.BEATS_PARQUET)
    weather = pd.read_csv(config.WEATHER_CSV, parse_dates=["Date"])
    clim = wx.climatology(weather)
    return panel, beats, weather, clim


def train_all(types: list[str]) -> dict:
    panel, beats, weather, clim = load_context()
    cluster_map = dict(zip(beats["Beat"].astype(int), beats["Cluster"].astype(int)))
    beat_codes = {int(b): i for i, b in enumerate(sorted(cluster_map))}
    panel = panel[panel["Beat"].isin(cluster_map)]
    targets = ["total"] + types
    first_usable = pd.Timestamp(config.START_DATE) + pd.Timedelta(days=max(config.ROLL_WINDOWS))
    registry = {
        "targets": targets, "types": types, "features": fx.FEATURES,
        "n_clusters": config.N_CLUSTERS, "start_date": config.START_DATE,
        "valid_start": config.VALID_START, "test_start": config.TEST_START,
        "data_end": str(panel["Date"].max().date()),
        "beat_codes": {str(k): v for k, v in beat_codes.items()},
        "beat_means": {}, "n_trees": {},
    }
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for target in targets:
        t0 = time.time()
        bm = fx.beat_means(panel, target)
        registry["beat_means"][target] = {str(k): v for k, v in bm.items()}
        ff = fx.build_feature_frame(panel, target, bm, beat_codes, weather, clim, cluster_map)
        ff = ff[ff["Date"] >= first_usable]
        train = ff[ff["Date"] < config.TEST_START]
        test = ff[ff["Date"] >= config.TEST_START].copy()
        trees = {}

        test["pred_specialist"] = np.nan
        for c in range(config.N_CLUSTERS):
            model, n = _fit(train[train["Cluster"] == c], target)
            trees[f"cluster_{c}"] = n
            p = specialist_path(target, c)
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, p)
            idx = test["Cluster"] == c
            test.loc[idx, "pred_specialist"] = model.predict(test.loc[idx, fx.FEATURES])

        gmodel, n = _fit(train, target)
        trees["global"] = n
        global_path(target).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gmodel, global_path(target))
        test["pred_global"] = gmodel.predict(test[fx.FEATURES])
        test["pred_naive_7"] = test["lag_7"]
        test["pred_ma_28"] = test["roll_mean_28"]

        registry["n_trees"][target] = trees
        out = test[["Beat", "Date", "Cluster", target, "pred_specialist", "pred_global",
                    "pred_naive_7", "pred_ma_28"]].rename(columns={target: "actual"})
        out.to_parquet(PRED_DIR / f"one_step_{fx.safe_name(target)}.parquet", index=False)
        log.info("  %-20s trained 7 models in %5.0fs  (trees: %s)", target, time.time() - t0, trees)

    config.REGISTRY_JSON.write_text(json.dumps(registry, indent=2))
    return registry


# ---------------------------------------------------------------- evaluation
MODEL_COLS = {"pred_specialist": "k-Shape + LightGBM (proposed)",
              "pred_global": "Single global LightGBM",
              "pred_naive_7": "Naive: same day last week",
              "pred_ma_28": "Naive: 28-day average"}


def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to 7-day blocks per beat (complete blocks only)."""
    d = df.copy()
    d["block"] = (d["Date"] - pd.Timestamp(config.TEST_START)).dt.days // 7
    counts = d.groupby("block")["Date"].nunique()
    full = counts[counts == 7].index
    d = d[d["block"].isin(full)]
    cols = ["actual"] + [c for c in MODEL_COLS if c in d]
    return d.groupby(["Beat", "block", "Cluster"], as_index=False)[cols].sum()


def evaluate_one_step(registry: dict) -> pd.DataFrame:
    rows = []
    for target in registry["targets"]:
        df = pd.read_parquet(PRED_DIR / f"one_step_{fx.safe_name(target)}.parquet")
        df = df.dropna(subset=list(MODEL_COLS))
        levels = {
            "beat-day": df,
            "beat-week": _weekly(df),
            "city-day": df.groupby("Date", as_index=False)[["actual"] + list(MODEL_COLS)].sum(),
        }
        for level, d in levels.items():
            for col, name in MODEL_COLS.items():
                rows.append({"target": target, "level": level, "model": name,
                             "cluster": "all", **metrics(d["actual"], d[col])})
            if target == "total" and level in ("beat-day", "beat-week"):
                for c, dc in d.groupby("Cluster"):
                    for col, name in MODEL_COLS.items():
                        rows.append({"target": target, "level": level, "model": name,
                                     "cluster": int(c), **metrics(dc["actual"], dc[col])})
    res = pd.DataFrame(rows)
    res.to_csv(config.REPORTS_DIR / "metrics_one_step.csv", index=False)
    return res


def evaluate_recursive(registry: dict, horizon: int = 7) -> pd.DataFrame:
    """Leakage-free 7-day-ahead backtest from weekly origins across the test period."""
    panel, beats, weather, clim = load_context()
    cluster_map = dict(zip(beats["Beat"].astype(int), beats["Cluster"].astype(int)))
    beat_codes = {int(k): v for k, v in registry["beat_codes"].items()}
    beat_list = sorted(cluster_map)
    target = "total"
    bm = {int(k): v for k, v in registry["beat_means"][target].items()}
    spec = {c: joblib.load(specialist_path(target, c)) for c in range(config.N_CLUSTERS)}
    glob = joblib.load(global_path(target))
    data_end = pd.Timestamp(registry["data_end"])
    origins = pd.date_range(config.TEST_START, data_end - pd.Timedelta(days=horizon - 1), freq="7D")
    actual = panel.set_index(["Beat", "Date"])[target]

    parts = []
    for o in origins:
        res = {}
        for name, fn in (("pred_specialist", lambda c: spec[c]), ("pred_global", lambda c: glob)):
            f = fc.recursive_forecast(panel, target, beat_list, o, horizon, fn, beat_codes, bm,
                                      cluster_map, weather, clim)
            res[name] = f.set_index(["Beat", "Date"])["pred"]
        f = f[["Beat", "Date", "Cluster", "lag_7"]].copy()
        f["pred_specialist"] = res["pred_specialist"].to_numpy()
        f["pred_global"] = res["pred_global"].to_numpy()
        # naive baselines frozen at the origin (no peeking into the horizon)
        hist = panel[(panel["Date"] < o) & (panel["Date"] >= o - pd.Timedelta(days=28))]
        ma = hist.groupby("Beat")[target].mean()
        f["pred_ma_28"] = f["Beat"].map(ma)
        last_week = panel[(panel["Date"] >= o - pd.Timedelta(days=7)) & (panel["Date"] < o)]
        lw = last_week.set_index(["Beat", "Date"])[target]
        f["pred_naive_7"] = [lw.get((b, d - pd.Timedelta(days=7 * (1 + (d - o).days // 7))), np.nan)
                             for b, d in zip(f["Beat"], f["Date"])]
        f["actual"] = actual.reindex(pd.MultiIndex.from_frame(f[["Beat", "Date"]])).to_numpy()
        f["origin"] = o
        f["step"] = (f["Date"] - o).dt.days + 1
        parts.append(f.drop(columns="lag_7"))
        log.info("    origin %s done", o.date())
    allp = pd.concat(parts, ignore_index=True)
    allp.to_parquet(PRED_DIR / "recursive_total.parquet", index=False)

    rows = []
    week = allp.groupby(["Beat", "origin", "Cluster"], as_index=False)[["actual"] + list(MODEL_COLS)].sum()
    for level, d in (("beat-day", allp), ("beat-week", week)):
        for col, name in MODEL_COLS.items():
            rows.append({"target": target, "level": level, "model": name, "horizon": horizon,
                         **metrics(d["actual"], d[col])})
    for s, d in allp.groupby("step"):
        for col, name in MODEL_COLS.items():
            rows.append({"target": target, "level": f"beat-day step {s}", "model": name,
                         "horizon": horizon, **metrics(d["actual"], d[col])})
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS_DIR / "metrics_recursive.csv", index=False)
    return out
