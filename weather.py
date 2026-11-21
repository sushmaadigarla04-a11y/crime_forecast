"""External data: daily weather + holidays (paper Sec. III-C.3).

Weather comes from NOAA GHCN-Daily for Chicago O'Hare (station USW00094846),
the station named in the paper. If NOAA is unreachable, the Open-Meteo ERA5
archive for the same coordinates is used instead. Units: degC and mm.

For dates beyond the observed record (true future forecasts) we fall back to a
day-of-year climatology computed from the training years only.
"""
from __future__ import annotations

import io
import logging

import holidays
import numpy as np
import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

NOAA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_COLS = ["TMAX", "TMIN", "PRCP"]


def _fetch_noaa(start: str, end: str) -> pd.DataFrame:
    params = dict(dataset="daily-summaries", stations=config.NOAA_STATION,
                  startDate=start, endDate=end, dataTypes="TMAX,TMIN,PRCP",
                  units="metric", format="csv")
    r = requests.get(NOAA_URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"DATE": "Date"})[["Date"] + WEATHER_COLS]
    df["source"] = "NOAA GHCN-D " + config.NOAA_STATION
    return df


def _fetch_open_meteo(start: str, end: str) -> pd.DataFrame:
    params = dict(latitude=config.WEATHER_LAT, longitude=config.WEATHER_LON,
                  start_date=start, end_date=end, timezone="America/Chicago",
                  daily="temperature_2m_max,temperature_2m_min,precipitation_sum")
    r = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"Date": d["time"], "TMAX": d["temperature_2m_max"],
                       "TMIN": d["temperature_2m_min"], "PRCP": d["precipitation_sum"]})
    df["source"] = "Open-Meteo ERA5"
    return df


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Download (or load cached) daily weather covering [start, end]."""
    if config.WEATHER_CSV.exists():
        cached = pd.read_csv(config.WEATHER_CSV, parse_dates=["Date"])
        if cached["Date"].min() <= pd.Timestamp(start) and cached["Date"].max() >= pd.Timestamp(end):
            return cached
    try:
        df = _fetch_noaa(start, end)
        log.info("  weather: %d days from NOAA", len(df))
    except Exception as exc:  # network / service errors
        log.warning("  NOAA failed (%s); falling back to Open-Meteo", exc)
        df = _fetch_open_meteo(start, end)
    df["Date"] = pd.to_datetime(df["Date"])
    full = pd.DataFrame({"Date": pd.date_range(start, end, freq="D")})
    df = full.merge(df, on="Date", how="left")
    n_missing = int(df["TMAX"].isna().sum())
    # Short gaps in station data: linear interpolation (temperature), 0 for rain.
    df[["TMAX", "TMIN"]] = df[["TMAX", "TMIN"]].interpolate(limit_direction="both")
    df["PRCP"] = df["PRCP"].fillna(0.0)
    df["source"] = df["source"].ffill().bfill()
    log.info("  weather gaps filled: %d days", n_missing)
    df.to_csv(config.WEATHER_CSV, index=False)
    return df


def climatology(weather: pd.DataFrame) -> pd.DataFrame:
    """Mean weather per day-of-year, from the training years only."""
    w = weather[weather["Date"] < config.TEST_START].copy()
    w["doy"] = w["Date"].dt.dayofyear
    clim = w.groupby("doy")[WEATHER_COLS].mean()
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    return clim.reset_index()


def weather_for_dates(dates: pd.Series | pd.DatetimeIndex, weather: pd.DataFrame,
                      clim: pd.DataFrame) -> pd.DataFrame:
    """Observed weather where available, climatology otherwise (flagged)."""
    out = pd.DataFrame({"Date": pd.to_datetime(pd.Series(dates).values)})
    out = out.merge(weather[["Date"] + WEATHER_COLS], on="Date", how="left")
    missing = out["TMAX"].isna()
    out["weather_is_climatology"] = missing
    if missing.any():
        doy = out.loc[missing, "Date"].dt.dayofyear
        fill = clim.set_index("doy").loc[doy, WEATHER_COLS].to_numpy()
        out.loc[missing, WEATHER_COLS] = fill
    return out


_HOLIDAYS = holidays.country_holidays("US", subdiv="IL", years=range(2010, 2031))


def is_holiday(dates: pd.Series) -> np.ndarray:
    return np.array([d in _HOLIDAYS for d in pd.to_datetime(dates).dt.date], dtype="int8")
