"""Stage 1 - data collection & preprocessing (paper Sec. III-A).

* read the raw Chicago Data Portal export (only the columns we need)
* clean: drop duplicates, rows without a police beat, and the incomplete last day
* keep beats that were active through the whole study window
* aggregate incidents into a complete beat x day panel of counts
  (total + one column per top-N crime type)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

RAW_COLUMNS = ["ID", "Date", "Primary Type", "Beat", "District", "Community Area",
               "Latitude", "Longitude"]


def load_raw_incidents() -> pd.DataFrame:
    """Read the raw CSV, keep the study window, and cache it as parquet."""
    if config.INCIDENTS_PARQUET.exists():
        return pd.read_parquet(config.INCIDENTS_PARQUET)
    if not config.RAW_CRIMES_CSV.exists():
        raise FileNotFoundError(
            f"Raw crime file not found: {config.RAW_CRIMES_CSV}. "
            "Download 'Crimes - 2001 to Present' from data.cityofchicago.org "
            "or set the CRIMES_CSV environment variable."
        )
    log.info("Reading %s (this takes a minute or two)...", config.RAW_CRIMES_CSV)
    df = pd.read_csv(config.RAW_CRIMES_CSV, usecols=RAW_COLUMNS, engine="pyarrow")
    log.info("  raw rows: %s", f"{len(df):,}")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date", "Beat", "Primary Type"])
    df = df[df["Date"] >= config.START_DATE]
    df = df.drop_duplicates(subset="ID")
    df["Beat"] = df["Beat"].astype(int)
    df["Primary Type"] = df["Primary Type"].astype(str).str.strip().str.upper()
    df = df.rename(columns={"Primary Type": "PrimaryType", "Community Area": "CommunityArea"})

    # The export usually ends part-way through its final day -> drop that day.
    last_day = df["Date"].dt.normalize().max()
    df = df[df["Date"] < last_day]
    df = df.sort_values("Date").reset_index(drop=True)
    log.info("  rows in study window: %s (%s -> %s)", f"{len(df):,}",
             df["Date"].min().date(), df["Date"].max().date())
    df.to_parquet(config.INCIDENTS_PARQUET, index=False)
    return df


def select_active_beats(inc: pd.DataFrame) -> list[int]:
    """Beats with incidents in every calendar year of the window and in the final 90 days."""
    years = inc["Date"].dt.year
    n_years = years.nunique()
    per_beat_years = inc.groupby("Beat")["Date"].apply(lambda s: s.dt.year.nunique())
    recent = inc.loc[inc["Date"] >= inc["Date"].max() - pd.Timedelta(days=90), "Beat"].unique()
    keep = per_beat_years[per_beat_years >= n_years * config.MIN_ACTIVE_YEARS_FRACTION].index
    beats = sorted(set(keep) & set(recent))
    log.info("  active beats kept: %d of %d", len(beats), inc["Beat"].nunique())
    return beats


def top_crime_types(inc: pd.DataFrame) -> list[str]:
    """Top-N primary types, ranked on the training period only."""
    train = inc[inc["Date"] < config.TEST_START]
    return train["PrimaryType"].value_counts().head(config.TOP_N_CRIME_TYPES).index.tolist()


def build_beat_metadata(inc: pd.DataFrame, beats: list[int]) -> pd.DataFrame:
    sub = inc[inc["Beat"].isin(beats)]
    meta = sub.groupby("Beat").agg(
        District=("District", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
        CommunityArea=("CommunityArea", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
        Latitude=("Latitude", "median"),
        Longitude=("Longitude", "median"),
        Incidents=("ID", "size"),
    ).reset_index()
    return meta


def build_panel(inc: pd.DataFrame, beats: list[int], types: list[str]) -> pd.DataFrame:
    """Complete beat x day grid; days with no incidents are explicit zeros."""
    sub = inc[inc["Beat"].isin(beats)].copy()
    sub["Day"] = sub["Date"].dt.normalize()
    dates = pd.date_range(sub["Day"].min(), sub["Day"].max(), freq="D")
    grid = pd.MultiIndex.from_product([beats, dates], names=["Beat", "Date"])

    total = sub.groupby(["Beat", "Day"]).size()
    total.index.names = ["Beat", "Date"]
    panel = pd.DataFrame({"total": total}).reindex(grid, fill_value=0)

    by_type = (sub[sub["PrimaryType"].isin(types)]
               .groupby(["Beat", "Day", "PrimaryType"]).size().unstack(fill_value=0))
    by_type.index.names = ["Beat", "Date"]
    by_type = by_type.reindex(grid, fill_value=0).reindex(columns=types, fill_value=0)
    panel = panel.join(by_type).reset_index()
    count_cols = ["total"] + types
    panel[count_cols] = panel[count_cols].astype("int32")
    return panel


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    inc = load_raw_incidents()
    beats = select_active_beats(inc)
    types = top_crime_types(inc)
    log.info("  top %d crime types: %s", len(types), types)
    panel = build_panel(inc, beats, types)
    meta = build_beat_metadata(inc, beats)
    panel.to_parquet(config.PANEL_PARQUET, index=False)
    meta.to_parquet(config.BEATS_PARQUET, index=False)
    log.info("  panel: %d beats x %d days = %s rows",
             panel["Beat"].nunique(), panel["Date"].nunique(), f"{len(panel):,}")
    return panel, meta, types
