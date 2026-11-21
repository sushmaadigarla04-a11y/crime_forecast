"""Recursive multi-day forecasting shared by evaluation and the dashboard.

Day t+1 is predicted from features that only use data up to day t; the
prediction is then written back into the series so that day t+2's lag and
rolling features can use it, and so on. History is never read past the
forecast origin, so backtests on the test period are leakage-free.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import poisson

import config
from src import features as fx


def recursive_forecast(
    panel: pd.DataFrame,
    target: str,
    beats: list[int],
    start: pd.Timestamp,
    horizon: int,
    model_for_cluster: Callable[[int], object],
    beat_codes: dict,
    beat_mean_map: dict,
    cluster_map: dict,
    weather: pd.DataFrame,
    clim: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast `target` for `beats` on [start, start + horizon - 1].

    Observations on or after `start` are ignored. If the data ends before
    `start - 1`, the gap is forecast recursively as well.
    """
    start = pd.Timestamp(start).normalize()
    end = start + pd.Timedelta(days=horizon - 1)
    lookback = max(config.LAGS) + 35
    hist = panel.loc[panel["Beat"].isin(beats) & (panel["Date"] < start)
                     & (panel["Date"] >= start - pd.Timedelta(days=lookback)),
                     ["Beat", "Date", target]]
    last_obs = hist["Date"].max()
    first_pred = last_obs + pd.Timedelta(days=1)
    pred_dates = pd.date_range(first_pred, end, freq="D")

    fut = pd.MultiIndex.from_product([beats, pred_dates], names=["Beat", "Date"]).to_frame(index=False)
    fut[target] = np.nan
    df = pd.concat([hist, fut], ignore_index=True).sort_values(["Beat", "Date"]).reset_index(drop=True)
    df[target] = df[target].astype("float64")

    df = fx.add_calendar(df)
    df = fx.add_weather(df, weather, clim).sort_values(["Beat", "Date"]).reset_index(drop=True)
    df["beat_code"] = df["Beat"].map(beat_codes).astype("int16")
    df["beat_mean"] = df["Beat"].map(beat_mean_map).astype("float32")
    df["Cluster"] = df["Beat"].map(cluster_map).astype("int8")

    out_rows = []
    for d in pred_dates:
        df = fx.add_lag_features(df, target)
        mask = df["Date"] == d
        rows = df.loc[mask]
        preds = np.empty(len(rows))
        for c, idx in rows.groupby("Cluster").groups.items():
            X = rows.loc[idx, fx.FEATURES]
            preds[rows.index.get_indexer(idx)] = model_for_cluster(int(c)).predict(X)
        preds = np.clip(preds, 0, None)
        df.loc[mask, target] = preds
        snap = rows[["Beat", "Date", "Cluster", "weather_is_climatology"] + fx.FEATURES].copy()
        snap["pred"] = preds
        out_rows.append(snap)

    out = pd.concat(out_rows, ignore_index=True)
    out = out[(out["Date"] >= start) & (out["Date"] <= end)].reset_index(drop=True)
    return out


def poisson_interval(mu: np.ndarray, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Central prediction interval assuming Poisson noise around the forecast mean."""
    a = (1 - level) / 2
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    return poisson.ppf(a, mu), poisson.ppf(1 - a, mu)
