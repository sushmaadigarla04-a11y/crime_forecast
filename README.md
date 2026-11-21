# Crime Forecaster: k-Shape + LightGBM + SHAP

A full, reproducible implementation of the framework in
*"A Hybrid Explainable AI Framework for Spatio-Temporal Clustering and Forecasting of Urban
Incident Patterns Using Geospatial Analytics and Ensemble Learning"*, built on the Chicago
crime dataset.

1. **k-Shape clustering** groups Chicago police beats by the *shape* of their daily crime series.
2. **Specialist LightGBM models** are trained for each cluster: 6 for the overall total and 6 x 10
   for the top crime types, 66 in all.
3. **SHAP** explains every forecast.
4. A **Streamlit dashboard** gives area and beat selection, 1-14 day forecasts, a breakdown by
   crime type, and explanations.

## Headline results (held-out test set: 1 Jan - 20 Apr 2023)

Overall crime total, 274 beats, about 2.07 M incidents (2015-2023):

| Model | Beat-day MAE | Beat-week MAE | Beat-week R² |
|---|---|---|---|
| **k-Shape + LightGBM (proposed)** | **1.284** | **3.47** | **0.683** |
| Single global LightGBM | 1.283 | 3.48 | 0.681 |
| Naive: 28-day average | 1.311 | 3.64 | 0.662 |
| Naive: same day last week | 1.758 | 5.03 | 0.363 |

* The proposed framework beats both naive baselines, and it holds that lead for full 7-day
  recursive forecasts (beat-week MAE 3.77 vs 4.04).
* Specialist models are **only slightly** better than one global LightGBM. The main benefit of the
  clusters is that they make the patterns easier to interpret. This should be reported honestly.
* Full tables, per-cluster and per-crime-type results, and figures: [`reports/RESULTS.md`](reports/RESULTS.md).

---

## Project layout

```
crime-forecast-xai/
├── config.py               all paths, date windows and hyper-parameters
├── app.py                  Streamlit dashboard
├── scripts/
│   └── run_pipeline.py     runs every stage end to end
├── src/
│   ├── data.py             stage 1: cleaning + beat x day aggregation          (paper III-A)
│   ├── weather.py          NOAA weather + holidays                            (paper III-C.3)
│   ├── kshape.py           k-Shape algorithm (Paparrizos & Gravano 2015), pure NumPy
│   ├── clustering.py       stage 2: k-Shape, elbow, cluster profiles + names  (paper III-B)
│   ├── features.py         stage 3: calendar / weather / lag / rolling / spatial (paper III-C)
│   ├── models.py           stage 4: 66 specialists + global model + baselines, evaluation (III-D)
│   ├── forecast.py         recursive multi-day forecasting (shared by evaluation and the app)
│   ├── explain.py          stage 5: SHAP explanations                         (paper III-E)
│   ├── report.py           writes reports/RESULTS.md + figures
│   └── geo.py              community-area names
├── tests/test_pipeline.py  unit tests (leakage, lag maths, metrics)
├── data/
│   ├── external/weather_daily.csv   (downloaded)
│   └── processed/                   (generated: incidents, panel, beats)
├── models/                 (generated: kshape.joblib, specialist/, global/, registry.json)
└── reports/                (generated: RESULTS.md, metrics_*.csv, figures/)
```

## Setup

Needs Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Data.** The raw file is *Crimes - 2001 to Present* from the
[Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2),
about 1.8 GB. By default the pipeline reads it from `../crime-forecasting-project/Crimes.csv`. Point
to another copy with an environment variable:

```bash
set CRIMES_CSV=D:\data\Crimes.csv        # PowerShell: $env:CRIMES_CSV="D:\data\Crimes.csv"
```

Weather is downloaded automatically from NOAA on the first run. The whole pipeline takes about
15 minutes on a laptop.

## Run

```bash
python scripts/run_pipeline.py            # all stages
streamlit run app.py                      # dashboard at http://localhost:8501
python -m pytest -q                       # tests
```

Single stages: `python scripts/run_pipeline.py --steps train evaluate report`.
Stages: `prepare → weather → cluster → train → evaluate → explain → report`.

## Method (paper → code)

| Paper | Implementation |
|---|---|
| Clean and aggregate to daily counts per beat | `src/data.py`. Duplicates and rows with no beat are dropped, as is the incomplete last day. Only beats active in every year are kept. A complete beat x day grid is built, so days with no crime are real zeros. |
| k-Shape with scaling + elbow | `src/clustering.py`. Each beat's series is z-normalised separately, using daily data from the last training year (2022). The elbow is checked for k = 2-10 and k = 6 is used. Clusters are ordered by volume and named automatically, e.g. "High-volume, weekend-peaking". |
| Features: time, 7-day lag, 7-day rolling mean, holidays, weather | `src/features.py`. Calendar features, US + Illinois holidays, NOAA O'Hare TMAX/TMIN/PRCP, lags of 1/7/14/21/28/364 days, 7- and 28-day rolling means, same-weekday mean, and beat identity. |
| Chronological split, 2023 = test | Train from 2015 to 2022 (H2 2022 used for early stopping), test from 2023 to the end of the data. |
| 66 specialist LightGBM models | `src/models.py`. Uses a Poisson objective. 6 total models + 60 crime-type models. |
| SHAP | `src/explain.py`. Exact TreeSHAP, shown as "% effect on the forecast". |
| Dashboard | `app.py`: sidebar controls, overview KPIs, a map of beats by cluster, forecast with 80% range, type breakdown, SHAP plot, model performance tab. |

### Improvements over the earlier prototype

The earlier notebook-and-app version had problems that made its numbers unreliable. They are
fixed here:

| Problem in the prototype | Fix |
|---|---|
| Clusters were re-fitted after the models were trained, so 213 of 274 beats were sent to the wrong cluster's model (overall R² = -78.9) | One pipeline writes the clusters and models together, and the app reads the same cluster table |
| Weather was 5 hand-pasted days filled forward over 5 years | Real NOAA GHCN-Daily data for O'Hare, the station named in the paper |
| `StandardScaler` normalised each *day* across beats instead of each beat's series | Each series is z-normalised separately |
| Clustering used the test period (leakage) | Clustering uses training data only |
| Crime-type models had no lag features; future dates used weather for Jan 1-5 only | Every model has lags; future weather uses a day-of-year climatology |
| No baselines, so "96% accurate" was meaningless | Compared with a naive baseline and a single global model |
| App loaded a 300 MB CSV on start-up | App loads small parquet files |

### Deliberate differences from the paper's text

* **Data window starts in 2015, not 2001.** Chicago redrew its police beats in December 2012, so
  beat IDs from before then are not comparable.
* **Poisson objective.** Crime counts are non-negative integers, and the Poisson objective also lets
  the dashboard show a Poisson 80% range (an "uncertainty estimate", which the paper lists as future
  work).
* **Extra baselines and a global-model ablation**, so the paper's claim that clustering helps is
  actually tested.

## Limitations and responsible use

* Recorded crime measures reporting and police activity as well as crime. A model trained on it can
  carry forward existing patrol bias (Berk 2021, ref. [17] in the paper).
* Daily per-beat counts are small (about 1-4) and mostly random. The forecasts are *expected values*
  with wide ranges. They are most useful at weekly or aggregate level.
* Backtests use the weather that was actually observed. True future forecasts use average weather
  for the time of year.
* This is a research prototype, not an operational policing tool.
