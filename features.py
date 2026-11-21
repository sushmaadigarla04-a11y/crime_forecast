"""Stage 3 - feature engineering (paper Sec. III-C, Table II).

The *same* functions build features for training, evaluation and the
dashboard's recursive forecasts, so there is no train/serve skew.

Feature groups
  calendar : day of week, month, day of month/year, week of year, quarter,
             year, weekend flag, US/Illinois holiday flag
  weather  : TMAX, TMIN (degC), PRCP (mm)
  lags     : target value 1, 7, 14, 21, 28 and 364 days earlier
  rolling  : mean of the previous 7 and 28 days, mean of the same weekday over
             the previous 4 weeks
  spatial  : beat code (categorical) + the beat's training-period average
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import weather as wx

CALENDAR_FEATURES = ["dow", "month", "dom", "doy", "woy", "quarter", "year",
                     "is_weekend", "is_holiday"]
WEATHER_FEATURES = ["TMAX", "TMIN", "PRCP"]
LAG_FEATURES = [f"lag_{l}" for l in config.LAGS]
ROLL_FEATURES = [f"roll_mean_{w}" for w in config.ROLL_WINDOWS] + ["same_wday_mean_4"]
SPATIAL_FEATURES = ["beat_code", "beat_mean"]
FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES + LAG_FEATURES + ROLL_FEATURES + SPATIAL_FEATURES
CATEGORICAL = ["beat_code"]

FEATURE_LABELS = {
    "dow": "Day of week", "month": "Month", "dom": "Day of month", "doy": "Day of year",
    "woy": "Week of year", "quarter": "Quarter", "year": "Year", "is_weekend": "Weekend",
    "is_holiday": "Public holiday", "TMAX": "Max temperature", "TMIN": "Min temperature",
    "PRCP": "Precipitation", "lag_1": "Count yesterday", "lag_7": "Count 7 days ago",
    "lag_14": "Count 14 days ago", "lag_21": "Count 21 days ago", "lag_28": "Count 28 days ago",
    "lag_364": "Count same day last year", "roll_mean_7": "Average of last 7 days",
    "roll_mean_28": "Average of last 28 days", "same_wday_mean_4": "Same weekday, last 4 weeks",
    "beat_code": "Patrol beat identity", "beat_mean": "Beat's long-run average",
}


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["Date"].dt
    df["dow"] = d.dayofweek.astype("int8")
    df["month"] = d.month.astype("int8")
    df["dom"] = d.day.astype("int8")
    df["doy"] = d.dayofyear.astype("int16")
    df["woy"] = d.isocalendar().week.astype("int8").to_numpy()
    df["quarter"] = d.quarter.astype("int8")
    df["year"] = d.year.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_holiday"] = wx.is_holiday(df["Date"])
    return df


def add_weather(df: pd.DataFrame, weather: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(df["Date"].unique())
    w = wx.weather_for_dates(dates, weather, clim)
    return df.drop(columns=WEATHER_FEATURES + ["weather_is_climatology"], errors="ignore").merge(
        w, on="Date", how="left")


def add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """df must be a complete daily grid sorted by (Beat, Date)."""
    g = df.groupby("Beat", sort=False)[target]
    for lag in config.LAGS:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
    prev = g.shift(1)
    prev_g = prev.groupby(df["Beat"], sort=False)
    for w in config.ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = prev_g.transform(
            lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean()).astype("float32")
    df["same_wday_mean_4"] = df[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1).astype("float32")
    return df


def beat_means(panel: pd.DataFrame, target: str) -> dict[int, float]:
    train = panel[panel["Date"] < config.TEST_START]
    return train.groupby("Beat")[target].mean().astype(float).to_dict()


def build_feature_frame(panel: pd.DataFrame, target: str, beat_mean_map: dict,
                        beat_codes: dict, weather: pd.DataFrame, clim: pd.DataFrame,
                        cluster_map: dict) -> pd.DataFrame:
    cols = ["Beat", "Date", target]
    df = panel[cols].sort_values(["Beat", "Date"]).reset_index(drop=True)
    df = add_lag_features(df, target)
    df = add_calendar(df)
    df = add_weather(df, weather, clim)
    df["beat_code"] = df["Beat"].map(beat_codes).astype("int16")
    df["beat_mean"] = df["Beat"].map(beat_mean_map).astype("float32")
    df["Cluster"] = df["Beat"].map(cluster_map).astype("int8")
    return df.sort_values(["Beat", "Date"]).reset_index(drop=True)


def safe_name(target: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in target)
