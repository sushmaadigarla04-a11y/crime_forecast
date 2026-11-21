"""Central configuration for the k-Shape + LightGBM + SHAP crime forecasting framework.

Every path, date boundary and hyper-parameter lives here so the whole pipeline
is reproducible from one place. Override the raw-data location with the
CRIMES_CSV environment variable if your copy of the Chicago file lives elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
RAW_CRIMES_CSV = Path(
    os.environ.get("CRIMES_CSV", ROOT.parent / "crime-forecasting-project" / "Crimes.csv")
)
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INCIDENTS_PARQUET = PROCESSED_DIR / "incidents.parquet"
PANEL_PARQUET = PROCESSED_DIR / "panel.parquet"          # beat x day counts
BEATS_PARQUET = PROCESSED_DIR / "beats.parquet"          # beat metadata + cluster
WEATHER_CSV = EXTERNAL_DIR / "weather_daily.csv"
REGISTRY_JSON = MODELS_DIR / "registry.json"

# ---------------------------------------------------------------- time windows
# Chicago redrew its police beats in Dec 2012, so earlier beat IDs are not
# comparable. 2015 onward gives 8 clean training years.
START_DATE = "2015-01-01"
VALID_START = "2022-07-01"   # last 6 months of training -> early stopping only
TEST_START = "2023-01-01"    # everything from here on is held-out test data

# k-Shape is fitted on the last full training year of *daily* series: long enough
# to show weekly + seasonal rhythm, short enough to keep k-Shape tractable, and
# strictly inside the training period (no test leakage).
CLUSTER_WINDOW = ("2022-01-01", "2022-12-31")
N_CLUSTERS = 6
ELBOW_K_RANGE = range(2, 11)

# A beat must have activity in every year of the study window to be modelled
# (removes beats that were retired / created mid-period).
MIN_ACTIVE_YEARS_FRACTION = 1.0

# ---------------------------------------------------------------- features
TOP_N_CRIME_TYPES = 10
LAGS = [1, 7, 14, 21, 28, 364]
ROLL_WINDOWS = [7, 28]
MAX_HORIZON_DAYS = 14

# NOAA GHCN-Daily station used in the paper: Chicago O'Hare International Airport.
NOAA_STATION = "USW00094846"
WEATHER_LAT, WEATHER_LON = 41.9786, -87.9048

# ---------------------------------------------------------------- model
RANDOM_STATE = 42
LGBM_PARAMS = dict(
    objective="poisson",      # crime counts are non-negative integers
    learning_rate=0.05,
    n_estimators=1500,        # upper bound; early stopping picks the real number
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 100

for _d in (PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
