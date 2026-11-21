"""Stage 5 - explainability with SHAP (paper Sec. III-E).

LightGBM computes exact TreeSHAP values natively (`pred_contrib=True`), the
same algorithm as `shap.TreeExplainer`. The models use a Poisson (log-link)
objective, so SHAP values are additive on the log scale; we convert each
contribution into an easy-to-read multiplicative effect:
    effect % = (exp(shap) - 1) * 100
i.e. "the warm weather raised this forecast by +12 %".
"""
from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import features as fx

log = logging.getLogger(__name__)


def explain_rows(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (shap values per feature [log scale], baseline expected log-value per row)."""
    contrib = model.predict(X[fx.FEATURES], pred_contrib=True)
    shap_vals = pd.DataFrame(contrib[:, :-1], columns=fx.FEATURES, index=X.index)
    return shap_vals, contrib[:, -1]


def explain_single(model, row: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Explain one forecast row. Returns (table, baseline count, predicted count)."""
    sv, base = explain_rows(model, row)
    s = sv.iloc[0]
    table = pd.DataFrame({
        "feature": s.index,
        "label": [fx.FEATURE_LABELS.get(f, f) for f in s.index],
        "value": row[fx.FEATURES].iloc[0].to_numpy(),
        "shap_log": s.to_numpy(),
    })
    table["effect_pct"] = (np.exp(table["shap_log"]) - 1) * 100
    table = table.reindex(table["shap_log"].abs().sort_values(ascending=False).index)
    baseline = float(np.exp(base[0]))
    prediction = float(np.exp(base[0] + s.sum()))
    return table.reset_index(drop=True), baseline, prediction


def _beeswarm(sv: pd.DataFrame, X: pd.DataFrame, title: str, path, max_display: int = 12) -> None:
    """SHAP summary (beeswarm) plot: one dot per row, colour = feature value (low -> high)."""
    order = sv.abs().mean().sort_values(ascending=False).index[:max_display][::-1]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
    for i, f in enumerate(order):
        v = X[f].astype(float).to_numpy()
        lo, hi = np.nanpercentile(v, [5, 95])
        col = np.clip((v - lo) / (hi - lo if hi > lo else 1), 0, 1)
        ax.scatter(sv[f], i + rng.uniform(-0.3, 0.3, len(v)), c=col, cmap="coolwarm", s=6,
                   alpha=0.6, linewidths=0)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)), [fx.FEATURE_LABELS.get(f, f) for f in order])
    ax.set_xlabel("SHAP value (log scale; +0.1 = about +10% on the forecast)")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_ticks([0, 1], labels=["low", "high"])
    cb.set_label("feature value")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def global_importance(registry: dict, sample: int = 3000) -> pd.DataFrame:
    """Mean |SHAP| per feature for every cluster's total-crime model, on test data."""
    from src.models import PRED_DIR, load_context, specialist_path

    panel, beats, weather, clim = load_context()
    cluster_map = dict(zip(beats["Beat"].astype(int), beats["Cluster"].astype(int)))
    beat_codes = {int(k): v for k, v in registry["beat_codes"].items()}
    bm = {int(k): v for k, v in registry["beat_means"]["total"].items()}
    ff = fx.build_feature_frame(panel[panel["Beat"].isin(cluster_map)], "total", bm,
                                beat_codes, weather, clim, cluster_map)
    test = ff[ff["Date"] >= config.TEST_START]

    rows = []
    for c in range(config.N_CLUSTERS):
        model = joblib.load(specialist_path("total", c))
        X = test[test["Cluster"] == c]
        X = X.sample(min(sample, len(X)), random_state=config.RANDOM_STATE)
        sv, _ = explain_rows(model, X)
        imp = sv.abs().mean()
        rows.append(imp.rename(f"Cluster {c}"))
        _beeswarm(sv, X[fx.FEATURES], f"SHAP summary - total crime, cluster {c} (log scale)",
                  config.FIGURES_DIR / f"shap_summary_cluster_{c}.png")

    imp = pd.concat(rows, axis=1)
    imp.index = [fx.FEATURE_LABELS[f] for f in imp.index]
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
    imp.to_csv(config.REPORTS_DIR / "shap_importance_total.csv")

    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(top.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(top.shape[1]), top.columns)
    ax.set_yticks(range(top.shape[0]), top.index)
    for i in range(top.shape[0]):
        for j in range(top.shape[1]):
            ax.text(j, i, f"{top.iat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean |SHAP| (log scale)")
    ax.set_title("Feature importance by cluster (total-crime specialists)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_importance_heatmap.png", dpi=150)
    plt.close(fig)
    return imp
