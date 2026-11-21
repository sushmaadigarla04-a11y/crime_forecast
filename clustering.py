"""Stage 2 - k-Shape clustering of police beats (paper Sec. III-B).

Each beat's daily total-crime series (training window only) is z-normalised
*per series* - k-Shape compares shape, not volume - and grouped with k-Shape
(shape-based distance, Paparrizos & Gravano 2015; NumPy implementation in src/kshape.py). Cluster IDs are re-ordered
by average volume so that labels are stable and readable, and every cluster
receives an automatic descriptive name from its profile.
"""
from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.kshape import KShape, zscore

import config

log = logging.getLogger(__name__)


def cluster_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    lo, hi = config.CLUSTER_WINDOW
    win = panel[(panel["Date"] >= lo) & (panel["Date"] <= hi)]
    return win.pivot(index="Beat", columns="Date", values="total").sort_index()


def _znorm(mat: pd.DataFrame) -> np.ndarray:
    return zscore(mat.to_numpy(dtype=float))


def elbow(mat: pd.DataFrame) -> pd.DataFrame:
    X = _znorm(mat)
    rows = []
    for k in config.ELBOW_K_RANGE:
        ks = KShape(n_clusters=k, n_init=3, max_iter=50, random_state=config.RANDOM_STATE)
        ks.fit(X)
        rows.append({"k": k, "inertia": float(ks.inertia_)})
        log.info("    k=%d inertia=%.4f", k, ks.inertia_)
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(res["k"], res["inertia"], marker="o", linestyle="--")
    ax.axvline(config.N_CLUSTERS, color="tab:red", alpha=0.4, label=f"chosen k={config.N_CLUSTERS}")
    ax.set(title="Elbow method for k-Shape", xlabel="Number of clusters (k)", ylabel="Inertia")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "elbow_kshape.png", dpi=150)
    plt.close(fig)
    res.to_csv(config.REPORTS_DIR / "elbow_kshape.csv", index=False)
    return res


def _profile(mat: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    dates = pd.to_datetime(mat.columns)
    weekend = dates.dayofweek >= 5
    summer = dates.month.isin([6, 7, 8])
    winter = dates.month.isin([12, 1, 2])
    rows = []
    for c in np.unique(labels):
        m = mat[labels == c].to_numpy(dtype=float)
        mean_series = m.mean(axis=0)
        rows.append({
            "Cluster": int(c),
            "n_beats": int((labels == c).sum()),
            "mean_daily_crimes": float(m.mean()),
            "weekend_ratio": float(mean_series[weekend].mean() / max(mean_series[~weekend].mean(), 1e-9)),
            "summer_winter_ratio": float(mean_series[summer].mean() / max(mean_series[winter].mean(), 1e-9)),
            "day_to_day_cv": float(np.mean(m.std(axis=1) / np.maximum(m.mean(axis=1), 1e-9))),
        })
    return pd.DataFrame(rows)


TRAITS = {  # metric -> (label when high, label when low)
    "mean_daily_crimes": ("busiest", "quietest"),
    "weekend_ratio": ("weekend-leaning", "weekday-leaning"),
    "summer_winter_ratio": ("strong summer peak", "flat across seasons"),
    "day_to_day_cv": ("irregular / spiky", "steady"),
}


def name_clusters(prof: pd.DataFrame) -> list[str]:
    """Name each cluster by its two most distinctive traits *relative to the other clusters*.

    k-Shape groups by shape, not volume, so absolute thresholds give every cluster the
    same name; z-scores across clusters show what actually sets each one apart.
    """
    z = (prof[list(TRAITS)] - prof[list(TRAITS)].mean()) / prof[list(TRAITS)].std(ddof=0)
    names, used = [], set()
    for i in range(len(prof)):
        ranked = list(z.iloc[i].abs().sort_values(ascending=False).index)
        label = lambda m: TRAITS[m][0] if z.iloc[i][m] > 0 else TRAITS[m][1]
        pair = (ranked[0], ranked[1])
        for alt in ranked[2:]:                      # keep names unique across clusters
            if frozenset(pair) not in used:
                break
            pair = (ranked[0], alt)
        used.add(frozenset(pair))
        first, second = label(pair[0]), label(pair[1])
        names.append(first[0].upper() + first[1:] + ", " + second)
    return names


def rename_clusters() -> None:
    """Recompute names from the saved profiles (cosmetic; does not change assignments)."""
    prof = pd.read_csv(config.REPORTS_DIR / "cluster_profiles.csv")
    prof["ClusterName"] = name_clusters(prof)
    prof.to_csv(config.REPORTS_DIR / "cluster_profiles.csv", index=False)
    meta = pd.read_parquet(config.BEATS_PARQUET).drop(columns="ClusterName")
    meta.merge(prof[["Cluster", "ClusterName"]], on="Cluster").to_parquet(config.BEATS_PARQUET, index=False)


def fit_clusters(panel: pd.DataFrame, meta: pd.DataFrame, run_elbow: bool = True) -> pd.DataFrame:
    mat = cluster_matrix(panel)
    log.info("  clustering %d beats on %d daily points", *mat.shape)
    if run_elbow:
        elbow(mat)
    X = _znorm(mat)
    model = KShape(n_clusters=config.N_CLUSTERS, n_init=5, max_iter=100,
                   random_state=config.RANDOM_STATE)
    raw_labels = model.fit_predict(X)

    # Re-order cluster IDs by mean volume (0 = quietest) for stable, readable labels.
    prof = _profile(mat, raw_labels).sort_values("mean_daily_crimes").reset_index(drop=True)
    remap = {int(old): new for new, old in enumerate(prof["Cluster"])}
    labels = np.array([remap[int(l)] for l in raw_labels])
    prof["Cluster"] = range(len(prof))
    prof["ClusterName"] = name_clusters(prof)
    prof.to_csv(config.REPORTS_DIR / "cluster_profiles.csv", index=False)

    joblib.dump({"model": model, "remap": remap, "beats": mat.index.tolist(),
                 "window": config.CLUSTER_WINDOW}, config.MODELS_DIR / "kshape.joblib")

    assign = pd.DataFrame({"Beat": mat.index, "Cluster": labels})
    assign = assign.merge(prof[["Cluster", "ClusterName"]], on="Cluster")
    meta = meta.drop(columns=["Cluster", "ClusterName"], errors="ignore").merge(assign, on="Beat")
    meta.to_parquet(config.BEATS_PARQUET, index=False)
    _plot_centroids(mat, labels, prof)
    log.info("  cluster sizes: %s", meta["Cluster"].value_counts().sort_index().to_dict())
    return meta


def _plot_centroids(mat: pd.DataFrame, labels: np.ndarray, prof: pd.DataFrame) -> None:
    """Average weekly profile (shape) and 4-week-smoothed annual profile per cluster."""
    dates = pd.to_datetime(mat.columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for c in range(config.N_CLUSTERS):
        m = mat[labels == c]
        series = m.mean(axis=0)
        series.index = dates
        name = prof.loc[prof["Cluster"] == c, "ClusterName"].iloc[0]
        lab = f"C{c} ({len(m)} beats): {name}"
        wk = series.groupby(series.index.dayofweek).mean()
        axes[0].plot(days, wk / wk.mean(), marker="o", label=lab)
        axes[1].plot(series.rolling(28, center=True).mean() / series.mean(), label=lab)
    axes[0].set(title="Weekly shape (relative to cluster mean)", ylabel="relative level")
    axes[1].set(title=f"Annual shape {config.CLUSTER_WINDOW[0][:4]} (28-day smoothed)", ylabel="relative level")
    for a in axes:
        a.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cluster_shapes.png", dpi=150)
    plt.close(fig)
