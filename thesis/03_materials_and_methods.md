# CHAPTER THREE — MATERIALS AND METHODS

> **Scope note.** This chapter documents the methodology of the study. All figures quoted here
> are methodological parameters (window sizes, feature counts, protocol settings); measured
> outcomes belong to Chapter Four. Where a measurement is cited here it is because it
> *determined a methodological choice*, and it is marked as such.

---

## 3.1 Research design and scope

### 3.1.1 Statement of what was built

Three distinct projects can be constructed from wind data at an unmetered site, and they
require different evaluation metrics. The distinction is stated explicitly because it governs
every claim made in this thesis.

| Framing | Question answered | Primary metric | Feasible here? |
|:---|:---|:---|:---|
| Operational forecaster | "How much power will this farm produce tomorrow?" | nMAE against metered output | **No** — no metered generation exists |
| Resource assessment | "Is this site worth developing, and how variable is it?" | Capacity factor, exceedance, interannual variability | Yes |
| **Forecast-method study** | **"Does numerical weather prediction improve day-ahead wind prediction at a tropical coastal site?"** | **Wind-speed MAE/R² against baselines** | **Yes** |

**This study is framed as the third, with a resource-assessment component.** The wind-speed
target is directly observable in the reanalysis, the baselines are computable, and the
resulting comparison does not depend on any assumed turbine.

### 3.1.2 The no-SCADA boundary

No supervisory control and data acquisition (SCADA) record exists for the study site, because
no utility-scale wind farm has been constructed there. This is a **scope boundary, not a
limitation to be apologised for**, and it is stated once here so that all downstream claims
can be precise:

- Wind-speed metrics validate the **forecasting step** and are physically meaningful.
- Power metrics validate the forecasting step **given an assumed power curve**. They do not
  validate physical yield: an incorrect turbine selection, wake losses, availability losses
  and electrical losses are all invisible to this evaluation.

This boundary statement is embedded in the automated report generator (`src/report.py`) so
that it is reproduced on every output artefact rather than depending on an author remembering
to include it.

### 3.1.3 Deployment contract

Before any data were acquired, an operational contract was written specifying what the system
must do and what information it is permitted to use at that moment:

```
Issue time          : 00:00 UTC daily
Target period       : the following 24 h (extended to 168 h for the multi-day model)
Effective lead      : 24–47 h for day-ahead (fixed day offset, not fixed hour offset)
Decision served     : day-ahead dispatch planning / resource characterisation
Admissible features : any input whose source is published before the issue time
```

The **latency budget** derived from this contract is reproduced in Table 3.1 and is the single
most consequential artefact in the methodology, because it partitions the feature set into
deployable and non-deployable groups before any model is trained.

**Table 3.1 — Publication latency of candidate inputs**

| Source | Publication latency | Available at issue time? |
|:---|:---|:---|
| ERA5 reanalysis | ~5 days (ERA5T preliminary, then consolidated) | **No** |
| NWP archived forecasts (Open-Meteo Previous Runs) | Available before valid time by construction | **Yes** |
| Calendar / static site properties | Immediate | **Yes** |
| On-site anemometry | Real time, *if hardware exists* | No — no met mast |

The governing rule: **any feature whose source is unavailable at issue time is excluded from
the operational model.** Such features may appear in an offline study model, but the two are
reported as separate systems with separate numbers (§3.11).

---

## 3.2 Study site and turbine specification

### 3.2.1 Site

The study site is **Anloga, Volta Region, Ghana (5.79° N, 0.90° E)**, a low-lying coastal
location on the Gulf of Guinea. The site was selected for its coastal exposure and its
inclusion in Ghana's renewable-energy development discourse.

| Property | Value |
|:---|:---|
| Latitude, longitude | 5.79° N, 0.90° E |
| Hub height analysed | 100 m above ground level |
| Time zone | UTC (Ghana observes UTC+0 year-round; no conversion required) |
| Wind regime | Monsoon-driven, with a pronounced South-West monsoon maximum (July–September) and a Harmattan minimum (October–December) |

Because Ghana observes UTC+0 without daylight saving, all timestamps in this study are UTC
and are simultaneously local time. This removes an entire class of timezone-conversion error
and is asserted at ingest rather than assumed.

### 3.2.2 Reference turbine

Wind speed was converted to electrical power using a single reference turbine:

| Property | Value |
|:---|:---|
| Model | **Enercon E-101/3050** |
| Rated power | **3.05 MW** |
| Hub height | 100 m (matched to the analysed wind height) |
| Power curve source | OpenEnergy Database (`oedb`), retrieved via `windpowerlib` |

The hub height was matched to the 100 m reanalysis wind level deliberately, so that
hub-height extrapolation is an identity operation for the primary data source and cannot
introduce a systematic bias (§3.4.1).

---

## 3.3 Data sources and acquisition

### 3.3.1 Source characterisation audit

A formal data-availability audit preceded all ingest code. The governing principle was:

> **Documentation describes intent; the file describes reality. Assert against the file.**

Every property in Table 3.2 was established from a **downloaded sample**, not from the
provider's documentation. Where the two disagreed, the file was taken as authoritative and the
disagreement recorded, since such disagreements are where silent defects originate.

**Table 3.2 — Source characterisation (nine properties, verified against downloaded files)**

| Property | Why it matters |
|:---|:---|
| Archive start / end | Determines the feasible study window |
| Publication latency | Determines deployability |
| Measurement height | Wind speed scales with height; a 50 m series read as 100 m understates hub wind by ~10.7% |
| Units | Guards against silent 100× errors (hPa vs Pa) |
| Timestamp convention | Instantaneous vs interval-averaged; start- vs end-labelled |
| Time zone | Guards against off-by-hours errors |
| Spatial resolution | Determines what "a point" means; a coarser cell yields a smoother, easier target |
| Actual grid shape returned | Requested dimensions ≠ received dimensions |
| Null pattern | Distinguishes "no data" from "zero" |

Each property was encoded as a runtime assertion in `src/data_ingest.py` rather than recorded
only in prose:

```python
SOURCE_METADATA = {
    "era5":       {"wind_height_m": 100.0, "pressure_unit": "Pa",  "time_offset_h":  0, "latency_days": 5},
    "nasa_power": {"wind_height_m":  50.0, "pressure_unit": "Pa",  "time_offset_h": -1, "latency_days": 1},
    "open_meteo": {"wind_height_m": 100.0, "pressure_unit": "hPa", "time_offset_h":  0, "latency_days": 0},
}
```

Range assertions accompany the unit metadata — surface pressure expressed in pascals must lie
in approximately 90,000–105,000 Pa, so a value of 1013 is immediately identified as
hectopascals carrying a pascal label.

### 3.3.2 ERA5 reanalysis — the truth series

**ERA5** (fifth-generation ECMWF atmospheric reanalysis) provides the target variable and the
contemporaneous atmospheric state. It was retrieved through the Copernicus Climate Data Store
API (`cdsapi`) in three request families, cached per calendar month as NetCDF:

| Family | Dataset | Variables |
|:---|:---|:---|
| Core (`era5_*`) | `reanalysis-era5-single-levels` | 10 m and 100 m wind components (*u*, *v*), 2 m temperature, surface pressure, forecast surface roughness |
| Extras (`era5x_*`) | `reanalysis-era5-single-levels` | 10 m wind gust, 2 m dewpoint, boundary-layer height |
| Pressure levels (`era5pl_*`) | `reanalysis-era5-pressure-levels` | Wind components and temperature at 850 hPa and 925 hPa |

A spatial pad of 0.2° was requested about the site centroid. **The returned grid was verified
by assertion rather than assumed** — a deliberate check, because an earlier iteration of this
work discovered that the requested patch snapped to a 1 × 2 cell array rather than the
intended 3 × 3, rendering ten derived spatial-gradient features mathematically degenerate
(§3.5.3).

ERA5 was selected as the truth series because it is temporally consistent, gap-free and
physically coherent — the appropriate choice for a *study* of forecast skill. Its ~5-day
publication latency makes it unsuitable as an operational input, which is precisely the
distinction Table 3.1 encodes.

**Archive extent obtained:** 156 complete months spanning **2010-01 to 2026-07**, with a
deliberate 41-month gap (2016-08 to 2019-12) that was not backfilled. The gap is tolerated
explicitly by the analysis code rather than being silently interpolated (§3.9.3).

### 3.3.3 Archived NWP forecasts — the predictive information

The central hypothesis of this study is that day-ahead **forecast** information, as distinct
from the analysed atmospheric state, carries the skill needed for day-ahead prediction.
Archived operational forecasts were obtained from the **Open-Meteo Previous Runs API**.

**Endpoint selection is a correctness issue, not a preference.** Two endpoints exist and only
one is admissible:

| Endpoint | Construction | Admissible? |
|:---|:---|:---|
| **Previous Runs API** (`{var}_previous_day{N}`) | The forecast for a valid timestamp issued *N* × 24 h earlier | ✅ **Used** |
| Historical Forecast API | Series built by stitching the *opening hours* of each model run | ❌ **Rejected** — analysis-like, effectively a nowcast; using it as a *t* + 24 h feature constitutes leakage and would produce excellent, invalid results |

Three independent NWP systems were retrieved, providing a multi-model ensemble:

| Model | Provider | Lead days obtained |
|:---|:---|:---|
| `ecmwf_ifs025` | ECMWF IFS, 0.25° | d1–d7 |
| `gfs_global` | NOAA GFS | d1–d7 |
| `icon_global` | DWD ICON | **d1–d6** (d7 is null at this location) |

Six variables were retrieved per model per lead day: 100 m wind speed, 100 m wind direction,
10 m wind speed, 2 m temperature, surface pressure and 10 m wind gusts.

**Archive extent, established by live probing rather than from documentation:** the Previous
Runs archive returns null for every lead at this location before **2024-03**, and is complete
from 2024-03 onward. This was verified by direct API query across candidate windows, and it is
the binding constraint on the study window.

The ICON ceiling at lead day 6 means a 168 h product runs on a **two-model** ensemble while a
144 h product retains three. This trade was resolved by measurement rather than assertion
(Chapter Four).

**Cache invalidation.** Responses are cached per (site, model, year). Because the cache path
encodes nothing about which lead days it holds, a cache hit does not imply a match — widening
the requested lead range would otherwise silently return the narrower cached set. A guard
(`_missing_open_meteo_keys`) compares requested against present series and forces re-download
on mismatch.

### 3.3.4 NASA POWER — cross-source check only

NASA POWER data were retrieved as an independent cross-check. **Its absolute errors are not
comparable to ERA5 errors** and are never reported alongside them, because two confounds run
in opposite directions: its 50 m wind requires upward extrapolation (inflating error ~12%),
while its ~55 km cell and hourly averaging produce a smoother, easier target (deflating
error). Only relative improvements and R² transfer between the two sources.

### 3.3.5 Study window selection

The study window was chosen **from** the availability audit rather than before it:

> The study window is the intersection of every required source's coverage.

| Asset | Coverage |
|:---|:---|
| ERA5 truth | 2010-01→2016-07, 2020-01→2023-12, 2024-03→2026-07 |
| Open-Meteo NWP forecasts | 2024-03→2026-07 |
| **Intersection (primary window)** | **2024-03 → 2025-12** |

This yields **15,912 rows after feature construction** for the day-ahead experiments. Three
consequences constrain the study and are stated rather than left implicit:

1. **No out-of-period holdout is possible** — there is no second era with forecast coverage.
2. Results are not comparable to any analysis run on the 2020–2023 ERA5-only window.
3. Methods requiring long histories (e.g. analogue ensembles) are data-limited by construction,
   and their results are data-constrained measurements rather than fair tests of the method.

A **difficulty indicator** — persistence MAE — is recorded for every window so that window
changes are visible rather than invisible. This guards against the failure mode in which a
result computed on an easier window is compared against a baseline from a harder one.

---

## 3.4 Data preprocessing and feature engineering

### 3.4.1 Hub-height extrapolation

Wind speed at hub height was obtained using `windpowerlib`'s logarithmic wind profile where
surface roughness is available, and the Hellman power law otherwise:

$$v_{hub} = v_{ref}\,\frac{\ln(z_{hub}/z_0)}{\ln(z_{ref}/z_0)}$$

where $z_0$ is the roughness length. Because ERA5 supplies wind at 100 m and the reference
turbine's hub height is 100 m, this operation is an identity for the primary data path; it is
retained for correctness when other sources or hub heights are configured.

### 3.4.2 Feature construction

Features were grouped by provenance so that ablation and latency partitioning operate on
coherent blocks:

| Group | Count | Contents |
|:---|---:|:---|
| **NWP forecast** | 51 | Per (model × variable × lead day) forecast columns, plus derived inter-model **spread** and **ensemble mean** |
| Calendar | 4 | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` — cyclical encodings of the **valid** time |
| Reanalysis surface | — | Wind speed/direction at 10 m and 100 m, temperature, pressure, roughness, gusts, dewpoint |
| Reanalysis upper-air | — | Wind components and temperature at 850 hPa and 925 hPa |
| Spatial derivatives | 20 | Gradients and standard deviations across the retrieved grid patch |
| Lags and rolling statistics | 9 | Target lags at 1, 2, 3, 6, 12 h; rolling means over 3, 6 h |
| **Total** | **104** | of which 51 NWP and 53 reanalysis-derived |

Calendar features encode the **valid** time rather than the feature time, since the diurnal
and seasonal position of the hour being predicted is legitimately known however far ahead the
forecast is issued.

### 3.4.3 The lead-day/horizon contract

A forecast column `nwp_*_d{N}` at valid time τ was issued 24·*N* hours before τ. A training
row at feature time *t* whose target is *t* + *h* may therefore only use lead days satisfying:

$$24N \ge h$$

Violating this constraint admits a forecast issued *after* the moment of prediction. The
constraint is enforced programmatically in `features.usable_nwp_leads`, which drops
insufficient leads and **raises** when none survives:

| Horizon | Admissible leads | Would have been used without the guard |
|---:|:---|:---|
| 24 h | N ≥ 1 | d1, d2, d3 ✅ |
| 48 h | N ≥ 2 | d1 ❌ (issued at *t* + 24 — one day in the future) |
| 72 h | N ≥ 3 | d1, d2 ❌ |
| 168 h | N ≥ 7 | d1, d2, d3 ❌ (d1 issued at *t* + 144 — six days ahead) |

It is emphasised that `d2` and `d3` are **not additional horizons** at a fixed target: they
are the 48 h- and 72 h-old forecasts *for the same valid hour*, included so that the model can
observe whether successive forecast runs agreed. Agreement between runs is itself a confidence
signal.

### 3.4.4 Two indexing conventions

Two framings are used, and the distinction is architectural rather than cosmetic:

**Feature-time framing** (day-ahead model). Rows are indexed by *t*, the moment of prediction,
with target hub(*t* + *h*). This framing is required when observation-side lag features are
used, since those features are naturally stamped at *t*.

**Valid-time framing** (multi-day model). Rows are indexed by τ, the hour being predicted:

$$\big(\text{nwp}_{*}\text{d}N[\tau],\ \text{calendar}(\tau),\ \text{spread}(\tau)\big) \rightarrow \text{hub}(\tau)$$

Beyond approximately day 2 the observation-side lags carry no information — hub(*t* − 1) says
nothing useful about hub(*t* + 168) — which removes the reason for the feature-time framing and
permits the simpler one. The valid-time framing has three methodological advantages:

1. **No horizon shift occurs anywhere**, so the class of alignment error described in §3.5 is
   structurally impossible rather than merely tested for.
2. It yields *seven* models (one per lead day) rather than 168, each trained on the full record.
3. The ICON lead-day ceiling requires no special-casing: the day-7 model simply has two
   members' features instead of three.

### 3.4.5 Operational and study feature sets

Applying the §3.1.3 latency rule partitions the features into two named sets that are never
conflated in reporting:

| Set | Columns | Contents |
|:---|---:|:---|
| `study` | 94 | Every live feature — answers the scientific question |
| `operational` | 55 | NWP + calendar only, **no reanalysis** — answers the engineering question |

Ten columns are excluded from **both** sets as mathematically degenerate: because the ERA5
request returned a 1 × 2 rather than 3 × 3 grid patch, all five `*_grad_ns` columns are
identically zero and all five `*_std` columns are exactly `|*_grad_ew| / 2`. Tree models
tolerate such columns; linear models do not, and neither carries information.

---

## 3.5 Data quality assurance: the alignment harness

Time-series forecasting studies fail most often at time alignment, because the failure is
silent. Forecasts are stamped with the hour they *describe* (valid time τ); training rows are
stamped with the hour the prediction is *made from* (feature time *t*). These differ by the
horizon. An error in one direction leaks the future and yields excellent but worthless models;
an error in the other grades every model against the wrong hour, collapsing all architectures
onto the persistence floor.

The alignment contract, stated once:

> Row *t* receives the forecast **valid at *t* + horizon**, which was issued around *t* and
> was therefore legitimately knowable then.

This is implemented as an explicit reindex on `index + horizon`, never a bare `.shift()`, so
that intent is visible in the code and gaps become `NaN` rather than silently displaced values.

Four guards were implemented **before** any model was trained.

### 3.5.1 Guard 1 — perfect-forecast test

A synthetic forecast column set exactly equal to the future target must, after alignment,
reproduce the target exactly. The underlying signal is deliberately constructed on a period
**not commensurate with the horizon** (a 37-hour period against a 24-hour horizon), because on
a 24-hour-periodic signal a frame shifted by exactly 24 hours would pass by coincidence.

### 3.5.2 Guard 2 — lag-peak against observed truth

Each forecast series is cross-correlated against observed truth over lags −48…+48 h. The peak
must lie at |lag| ≤ 1 h. Failure modes are distinguished rather than aggregated, because they
require different responses:

| Symptom | Diagnosis | Response |
|:---|:---|:---|
| Peak at ±24 h | Horizon shift is wrong | Fix the code, never the criterion |
| Constant 2–3 h offset, one source only | Genuine convention difference | Document and apply an explicit named shift |
| No clear peak | Units, coordinates or wrong variable | Halt; nothing downstream is meaningful |

The 1-hour tolerance is derived from measured physics — genuine disagreement between forecast
models about sea-breeze phase produces ~1 h offsets — rather than being chosen for convenience.

### 3.5.3 Guard 3 — unit and range assertions

Each column's unit is asserted at ingest by asserting its physically plausible range, using the
metadata of §3.3.1.

### 3.5.4 Guard 4 — leakage test via skill decay

Genuine forecast skill must **decay with lead time**. The same pipeline is scored using day-1,
day-2 and day-3 forecast features; flat skill across leads indicates that the feature is not a
forecast but analysis wearing a forecast's name. This guard specifically detects the
Historical-Forecast-API trap described in §3.3.3. A secondary signal is also monitored: a
multi-model ensemble should outperform its individual members.

### 3.5.5 The falsifiability requirement

> **A gate that cannot fail is not a gate.**

Every guard was fed deliberately broken input — a frame shifted by 24 h, a pressure column in
the wrong unit, a leaked analysis column — and **observed to fail** before being trusted in its
passing state. The broken-input cases are retained permanently as regression tests. This
requirement was adopted because a prior four-gate harness in this project's history contained
two gates that returned hardcoded passes, so that the reported statement "all four gates pass"
in fact meant "two passed and two reported".

Two general principles derived from this stage govern the wider codebase:

- **Sanity floor.** Every predictor above the trivial baselines must beat naive persistence;
  anything that does not is treated as a defect until proven otherwise.
- **Assert, never reconcile.** When two objects that should have equal length do not, the
  mismatch *is* the signal. Silently truncating to the shorter is worse than raising.

---

## 3.6 Predictability ceiling analysis

A ceiling analysis was conducted **before** model selection, to determine whether the problem
was variance-limited (curable with more data or regularisation) or information-limited (curable
only with different inputs). Four diagnostics were used:

1. **Autocorrelation ceiling.** The target's autocorrelation at the forecast horizon caps the
   attainable R² for any predictor built solely on the atmospheric state at time *t*. At 24 h
   the measured autocorrelation of 0.571 implies an R² ceiling of **0.326**.
2. **Learning curve.** Test error as a function of training-set length, with the train/test
   error ratio recorded. A ratio near unity indicates that little variance remains to be cured
   by additional data.
3. **Feature-group ablation.** Cumulative addition of coherent feature blocks, to identify
   saturation. *(An ablation can establish that a feature did not help; it cannot establish
   that a feature was broken — hence Guard 3.)*
4. **Nowcast bound.** Error attainable when the model is given the contemporaneous state.

The diagnostic table below governs the response, and is applied before any architecture search:

| Pattern | Diagnosis | Correct response |
|:---|:---|:---|
| Best model far below ceiling, high train/test ratio | Variance-limited | More data, more regularisation |
| Best model at ceiling, flat learning curve | **Information-limited** | **Obtain different information**; more model capacity is wasted effort |
| Full apparatus ≈ one-parameter OLS | Problem may not require ML | State this; it is a legitimate finding |

This analysis is what motivated the acquisition of NWP forecast features: a state-at-*t*
predictor is bounded by the autocorrelation ceiling, whereas a *forecast* of the target hour is
not subject to that bound.

The same procedure was applied independently at monthly scale (§3.9.3) before any monthly model
was attempted.

---

## 3.7 Baselines

Baselines are the denominator of every claim in this thesis and were therefore implemented in
one place, each with a unit test asserting its exact definition and lag, and computed **before**
any machine-learning result existed.

**Table 3.3 — The five mandatory baselines**

| Baseline | Definition | Params | Guards against |
|:---|:---|---:|:---|
| **Persistence** | ŷ(*t*+*h*) = *y*(*t*) — the value now, at exactly the horizon | 0 | Mistaking autocorrelation for skill |
| **Damped persistence** | ŷ = *a*·*y*(*t*) + *b* — persistence shrunk toward the mean | 2 | Being flattered by an opponent that does not shrink |
| **Climatology** | Training mean by month × hour-of-day | — | Mistaking a learned diurnal cycle for skill |
| **Single-model MOS** | OLS of the observation on **one** NWP forecast | 2 | Mistaking *having* a forecast for *learning from* one |
| **Multi-model MOS** | OLS on all admissible forecast columns + calendar | ~21 | Mistaking linear combination for nonlinearity |

The final two are the methodologically decisive additions. A study comparing machine learning
against persistence alone cannot state how much its machine learning contributed, because
classical model output statistics (MOS) capture most of the available skill.

**The exact-lag requirement.** A persistence forecaster predicting from `hub(t−1)` while being
scored against `hub(t+24)` implements a *25-hour* persistence and constitutes an easier bar.
The guard is a test constructing a series where *y*(*t*+*h*) = *y*(*t*) exactly and asserting
that the baseline scores zero error.

**Re-baselining rule.** No baseline computed on one window is ever quoted against a result from
another. When the window changes, all baselines are recomputed before any comparison is made.

**Baseline choice at longer horizons.** Persistence and climatology cross over between 24 h and
48 h; beyond approximately 36 h **climatology is the correct reference**, and quoting
persistence skill at weekly range would flatter the model. At monthly scale persistence is
inadmissible entirely, scoring worse than climatology.

**Climatology construction.** Two variants exist and must be distinguished, since the choice
materially affects reported skill:

- *Training-window climatology* — month × hour-of-day means estimated from the training window
  only.
- *Archive climatology* — month × hour-of-day means estimated from the full multi-year
  reanalysis archive, expanding, using only data preceding the issue time. **This is the
  admissible reference** for aggregate products, since a climatological mean is unaffected by
  ERA5's publication latency and is therefore operationally available.

Archive climatology for energy products is computed in **capacity-factor space** rather than by
passing a climatological wind speed through the power curve, because the curve is convex and
$P(\overline{v}) \neq \overline{P(v)}$.

---

## 3.8 Model development

### 3.8.1 Principle of ascending complexity

Models were developed in ascending order of complexity, stopping when additional complexity
ceased to pay, so that the contribution of each increment is measurable:

1. One-parameter OLS on the current state
2. Single-model MOS (2 parameters)
3. Multi-model MOS (~21 parameters)
4. Full-feature linear model (ridge)
5. Gradient boosting — untuned, then tuned
6. Deep sequence models — **only with a stated justification**

A deep model presented without simple comparators constitutes an unfalsifiable claim.

### 3.8.2 The classical ladder

Eleven predictors were implemented in `src/traditional.py` and evaluated under an identical
protocol:

| Predictor | Description |
|:---|:---|
| Naive persistence | ŷ(*t*+*h*) = hub(*t*) |
| Damped persistence | Shrunk persistence, 2 parameters |
| Climatology | Month × hour-of-day training mean |
| Single-model MOS | OLS on the best-correlated single forecast column |
| Multi-model MOS | OLS on all lead-appropriate forecast columns + calendar |
| Kalman-filtered MOS | Recursively updated MOS coefficients with a 24-step delayed update reflecting observation availability |
| Analogue ensemble (AnEn) | Prediction from historical analogues of the current forecast state |
| SARIMAX with exogenous NWP | Seasonal ARIMA with forecast columns as exogenous regressors |
| Ridge regression | Regularised linear model on the full feature set |
| Gradient boosting | XGBoost / LightGBM |

A methodological requirement was imposed on the MOS family: the **primary lead** is defined as
the shortest lead surviving the §3.4.3 guard, not hardcoded to `d1`. At horizons beyond 24 h a
hardcoded `d1` would be removed by the guard, leaving the model silently fitting calendar terms
alone.

SARIMAX was configured with `trend="c"` and `enforce_stationarity=True`; without these an
unconstrained autoregressive specification produces numerically explosive multi-step forecasts.

### 3.8.3 Machine-learning and deep models

| Class | Models | Library |
|:---|:---|:---|
| Gradient-boosted trees | XGBoost, LightGBM | `xgboost`, `lightgbm` |
| Deep sequence models | LSTM (BlockRNN), Temporal Fusion Transformer (TFT), N-BEATS, DLinear | `darts` (PyTorch backend) |

### 3.8.4 Comparability rules for head-to-head comparison

The following rules were enforced for every architecture comparison. They exist because a
prior comparison in this project's history was invalidated when one model was found to be
solving a different optimisation problem than the others:

| Rule | Rationale |
|:---|:---|
| **Identical objective** — every model optimises MSE | A model trained on a multi-quantile objective and compared against MSE-trained models on a point metric is penalised for solving a different problem |
| **Identical information** — same features, split, target | Removes the most common confound |
| **Comparable training care** — equivalent early stopping and learning-rate selection | Uneven effort otherwise reads as a model-quality difference |
| **Print and assert the resolved configuration** before any run exceeding one hour | Guards against silent default substitution (e.g. epoch counts) |
| **Cheapest model first** in run order | A late crash still leaves earlier results |

Hyperparameter tuning was investigated using Optuna (50 trials) but was **not** applied
per-model in the comparative experiments, on the measured grounds that a tuned and an untuned
XGBoost scored identically to four decimal places on this window. Equal, documented settings
were judged more important than individually optimal ones.

---

## 3.9 Forecast systems developed

Four systems were developed. They operate at different horizons and aggregation scales, and
their metrics are **not mutually comparable**; each is reported on its own basis.

### 3.9.1 Day-ahead system (24 h)

Two-stage architecture, feature-time framing:

**Stage 1** — XGBoost predicts hub-height wind speed at *t* + 24 h from the 104-feature frame.
**Stage 2** — Predicted wind speed is converted to power via the manufacturer power curve
(§3.10).

The two stages are deliberately **decoupled**: the power module runs standalone on any
wind-speed series and imports nothing from the machine-learning side. This decoupling is
justified on **interpretability and modularity** grounds — the wind forecast can be inspected
separately, the turbine can be exchanged without retraining, and failure modes remain
attributable — and explicitly *not* on accuracy grounds, since a direct power model and a
two-stage model were found to be within 0.05 percentage points.

### 3.9.2 Multi-day system (24 h to 168 h)

Seven independent models, one per forecast lead day, in valid-time framing (§3.4.4). To
forecast the week from issue time *T*, valid times τ = *T* + 1 … *T* + 168 are served by the
model for *N* = ⌈(τ − *T*)/24⌉.

### 3.9.3 Monthly resource climatology

Not a forecast. A statistical characterisation of the site computed over the full 156-month
ERA5 archive, comprising:

- **Variance decomposition** into seasonal-cycle and interannual-anomaly components.
- **Anomaly autocorrelation** at lags 1, 2, 3, 6 and 12 months, computed only over **genuinely
  consecutive** month pairs — a naive `shift(1)` would pair 2016-07 with 2020-01 across the
  archive gap and report the result as a lag-1 correlation.
- **Exceedance probabilities** P50/P75/P90 per calendar month.
- **Mean diurnal profile** by calendar month.

**Exceedance convention.** P*XX* is the value **exceeded** *XX*% of the time — hence P90 is the
10th percentile, a low and conservative figure. Reading it as the 90th percentile inverts the
product, so the convention is encoded once (`EXCEEDANCE_QUANTILE`) and unit-tested.

**Estimator selection by validation.** Two estimators were implemented — the empirical order
statistic and a fitted-normal tail — and selected between by **leave-one-year-out calibration**
rather than by preference. With 12–14 years per calendar month, an empirical 10th percentile
lies between the first and second order statistics and shifts by an entire rank when one year
is withheld.

**Completeness guard.** A month enters the climatology only if ≥ 95% of its hours are present;
a month missing its windiest week would otherwise enter as a genuine low year.

### 3.9.4 Thirty-day outlook

A composite rather than a new model: the multi-day system covers days 1–7 and climatology
covers the remainder. The outlook and its baseline are constructed to be **identical except
for the head**, so that any difference in score is attributable to the forecast alone:

$$\hat{y}_{\text{clim}} = \text{clim}(M)$$
$$\hat{y}_{\text{outlook}} = \text{clim}(M) + \frac{h_{\text{head}}}{h_{\text{month}}}\Big(\overline{f}_{\text{head}} - \overline{c}_{\text{head}}\Big)$$

The head enters as an **anomaly correction** rather than a level, because the first seven days
of a month are not climatologically identical to the month as a whole, and comparing a 7-day
forecast level against a 30-day climatological level would score that seasonal drift as
forecast error.

**Aggregation-scale requirement.** Because the head constitutes only ~23% of a monthly total,
scoring the forecast on monthly aggregates dilutes the signal while retaining the tail's noise
— which, being common to both arms under an absolute-error metric, does not cancel. The head is
therefore additionally evaluated on **non-overlapping 7-day blocks**, where it constitutes
100% of the prediction. Forecasts are issued every 7 days so that consecutive issues **tile**
the record without gaps or overlaps; tiling is asserted rather than assumed.

**Training regime.** Expanding-window walk-forward: for a target period beginning at *T*,
training uses only valid times strictly before *T*. Models are refit monthly and issued weekly,
so a mid-month issue uses a model up to four weeks stale — a conservative choice that cannot
leak and that reflects realistic retraining cadence.

---

## 3.10 Wind-to-power conversion

Power was computed from hub-height wind speed using `windpowerlib` with the manufacturer power
curve, incorporating an **air-density correction**. Air density was computed from the ideal gas
law:

$$\rho = \frac{p}{R_{\text{d}} T}$$

where *p* is surface pressure (Pa), *T* absolute temperature (K) and $R_{\text{d}}$ the
specific gas constant for dry air. The correction is material at this site: Ghana's high
ambient temperatures reduce air density and hence available power, and omitting the correction
raises the computed capacity factor by approximately 4%.

Capacity factor is defined throughout as

$$\text{CF} = \frac{P}{P_{\text{rated}}}$$

**Error propagation.** Wind-speed error passes through an approximately cubic mid-section of
the power curve and is therefore amplified nonlinearly. This is the principal weakness of the
two-stage architecture and is measured rather than hidden, by binning absolute power error
against absolute wind-speed error. The analysis is regenerated whenever Stage 1 accuracy
changes materially, since the relative contribution of the amplification changes with it.

---

## 3.11 Probabilistic forecasting

Point forecasts are insufficient for dispatch decisions. Probabilistic forecasts were produced
as follows:

1. **Uncertainty signal.** Per-hour **inter-model spread** (standard deviation across ECMWF,
   GFS and ICON) is included as a feature. Disagreement among independent NWP systems is a
   physically meaningful indicator of atmospheric predictability.
2. **Quantile estimation.** Gradient boosting with a pinball (quantile) objective produces
   P10/P50/P90, **trained as separate models** from the point forecast. This separation is
   required: training a single model on a multi-quantile objective degrades point accuracy
   materially and would confound any architecture comparison.
3. **Calibration assessment.** Over the test set, approximately 10% of observations should fall
   below P10 and 10% above P90. **An uncalibrated interval is worse than no interval**, because
   it invites false confidence.
4. **Sharpness.** Reported alongside calibration, since an interval spanning the full turbine
   range is perfectly calibrated and perfectly useless.

Calibration outcomes are reported as measured, including failures. An interval widened until it
achieves nominal coverage on the test set has been fitted to the test set, not calibrated.

---

## 3.12 Experimental protocol and statistical analysis

### 3.12.1 Pre-registration

The evaluation protocol — split, seeds, baselines, metrics and **falsifier** — was committed in
writing before results were generated. A falsifier was stated in operational terms (e.g. "NWP
features improve MAE by less than 5% against the strongest baseline"), since a statement such
as "the results are disappointing" is not falsifiable.

For the outlook experiment (§3.9.4) the **minimum detectable effect** at 80% power was
computed and recorded before any model result was examined, so that an underpowered design
could be identified as such rather than misread as a negative finding.

### 3.12.2 Cross-validation design

**Rolling-origin (expanding-window) folds**, never shuffled. Four folds with successive
~3-month test blocks were used for the day-ahead comparison, so that **every season appears in
test**:

| Fold | Train rows | Test rows | Test period |
|:---|---:|---:|:---|
| 1 | 7,004 | 2,227 | 2024-12-24 → 2025-03-27 |
| 2 | 9,231 | 2,227 | 2025-03-27 → 2025-06-28 |
| 3 | 11,458 | 2,227 | 2025-06-28 → 2025-09-29 |
| 4 | 13,685 | 2,227 | 2025-09-29 → 2025-12-30 |

Rolling origins are required rather than preferred: persistence MAE across these folds ranges
1.1129–1.6274 m/s, meaning the problem is **46% harder in one quarter than another**. Any
single-split ranking is therefore partly a statement about which quarter was tested.

Random *k*-fold cross-validation is inadmissible throughout, as it leaks future information
into training folds.

For monthly-scale analysis, **leave-one-year-out** folds are used, since months within a year
are not independent.

### 3.12.3 Seeds and replication

| Model class | Seeds | Rationale |
|:---|---:|:---|
| Deterministic (persistence, damped persistence, climatology, MOS, Kalman, AnEn, SARIMAX, ridge) | 1 | Output does not depend on the seed; repetition would imply a precision that does not exist |
| Gradient boosting | 5 — `42, 7, 1234, 2024, 31337` | 20 runs per model per feature set |
| Deep sequence models | 3 — `42, 7, 1234` | Computational cost |

Results are reported as **mean ± standard deviation**, never best-of-*N*, since selecting the
best of several runs is selection rather than measurement. Seed sensitivity was measured
**per model class** rather than assumed uniform, because it is not: deep architectures exhibit
substantially greater seed variance than tree ensembles, and in some cases a single seed could
reverse a reported ranking.

### 3.12.4 Metrics

- **Primary:** Mean absolute error (MAE), in m s⁻¹ for wind speed and capacity-factor points
  for energy.
- **Secondary:** RMSE, R², and normalised MAE/RMSE (nMAE, nRMSE) as a percentage of rated power.
- **Skill score**, computed **per fold**:

$$S = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{reference}}}$$

Skill is computed per fold rather than pooled because fold difficulty varies by up to 46%,
which makes raw MAE non-comparable across folds while skill remains so.

**Every results table states its window, horizon, aggregation scale and reference baseline.**
Metrics computed at different aggregation scales (hourly, 7-day total, monthly total) are
reported as separate quantities even where they share units.

### 3.12.5 Significance testing

Pairwise forecast accuracy was compared using the **Diebold–Mariano test** on loss
differentials $d = |e_A| - |e_B|$, with a **Newey–West HAC variance estimator**. The HAC
correction is load-bearing rather than decorative: overlapping multi-step forecast errors are
serially correlated by construction, and an uncorrected *t*-test substantially overstates
significance. Lag selection follows the aggregation scale — horizon − 1 lags for hourly
multi-step forecasts, and lag 1 for non-overlapping block aggregates, whose adjacent members
still share weather.

Error vectors are averaged across seeds **within each fold** and then concatenated in
chronological order, so that every model is compared on identical timestamps.

**Reporting ties as ties.** Where two models fall within one standard deviation of each other
they are reported as tied, and selection proceeds on secondary grounds — interpretability,
training cost or operational simplicity.

### 3.12.6 Uncertainty quantification for aggregate metrics

For monthly-scale results, where sample sizes are small, **bootstrap confidence intervals**
(10,000 resamples over whole months) are reported. The interval is treated as the deliverable
rather than the point estimate: at small *n* a point estimate alone implies a precision that
does not exist. Where an interval spans zero, the documented conclusion is "no demonstrable
skill", not a rounded-up positive claim.

---

## 3.13 Deployment rehearsal

A deployment rehearsal was conducted, in which the model is re-trained and re-evaluated using
**only information that would genuinely have existed at issue time**.

Features were partitioned by the §3.1.3 latency budget into three tiers:

| Tier | Description | Included in `operational`? |
|:---|:---|:---|
| 1 | Always available — NWP forecasts, calendar, static site properties | Yes |
| 2 | Available with hardware — on-site anemometry, live SCADA | Only if the hardware genuinely exists |
| 3 | Offline only — reanalysis-derived features | No |

The `operational` model was trained and evaluated under an **identical** protocol to `study`
— same window, same folds, same seeds, same baselines — and the two are reported side by side.
The **degradation** is computed explicitly and reported as a finding in its own right, since it
is the honest cost of deployability.

A prediction regarding the expected degradation was recorded **before** the measurement, in the
spirit of §3.12.1, and is retained verbatim in the project record including its hedges, on the
principle that a pre-registration quietly rewritten once the answer is known is worthless.

**Requirement:** every headline number in every document is labelled `study` or `operational`.
An unlabelled number is treated as a defect.

It is stated explicitly that a favourable operational score does **not** by itself constitute
deployability. A deployable system additionally requires a scheduled serving path, input
monitoring, drift monitoring against baselines, a retraining schedule and an explicit
degradation-to-climatology fallback. These are engineering requirements rather than research
outputs, and are identified as future work.

---

## 3.14 Software implementation and reproducibility

### 3.14.1 Implementation

The system was implemented in **Python 3.13** as a modular pipeline. Principal modules:

| Module | Responsibility |
|:---|:---|
| `src/data_ingest.py` | Source retrieval, caching, metadata assertions |
| `src/features.py` | Feature construction, lead/horizon guard |
| `src/stage1_windspeed.py` | Wind-speed models and metrics |
| `src/stage1_deep.py` | Deep sequence architectures |
| `src/stage2_powercurve.py` | Power conversion, density correction |
| `src/traditional.py` | Classical predictor ladder |
| `src/bakeoff.py` | Rolling-origin harness, Diebold–Mariano testing |
| `src/weekly.py` | Multi-day valid-time models |
| `src/monthly.py` | Monthly climatology and exceedance |
| `src/monthly_outlook.py`, `src/outlook_power.py` | Thirty-day outlook and its evaluation |
| `src/evaluate.py`, `src/report.py` | Metrics and automated reporting |

### 3.14.2 Principal libraries

| Purpose | Library |
|:---|:---|
| Numerical / data handling | `numpy`, `pandas`, `scipy`, `xarray`, `netCDF4`, `pyarrow` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep learning | `darts` (PyTorch backend) |
| Classical time series | `statsmodels` |
| Hyperparameter search | `optuna` |
| Wind power modelling | `windpowerlib` |
| Data acquisition | `cdsapi`, `requests` |
| Interpretability | `shap` |
| Testing | `pytest` |

### 3.14.3 Configuration and reproducibility

All experimental parameters are held in version-controlled YAML configuration files
(`config_nwp.yaml`, `config_bakeoff.yaml`, `config_weekly.yaml`, `config_monthly.yaml`) rather
than in code, so that any reported result can be traced to the exact configuration that
produced it. Random seeds are fixed and recorded. Downloads are cached per (source, period) so
that runs are resumable and re-executable without repeated network access.

### 3.14.4 Verification strategy

The project maintains an automated test suite (146 tests) covering:

- The four alignment guards, **each with its deliberately-broken red case**
- Exact baseline definitions and lags
- Classical predictor correctness against analytically known solutions
- The lead/horizon guard
- Completeness and coverage guards
- Tiling and leakage assertions in the walk-forward harness
- Exceedance-convention correctness

In addition, every figure appearing in the results documentation is **re-derived from its
source artefact by script** rather than transcribed, since transcription is itself an error
source.

### 3.14.5 Interpretability analysis

SHAP (SHapley Additive exPlanations) values and correlation analysis were computed
**diagnostically** rather than decoratively, addressing three questions:

1. Does the highest-ranked feature make physical sense?
2. Is anything ranked implausibly high — a possible leakage indicator?
3. Is anything ranked at exactly zero? Such a feature is **broken**, not merely unimportant.

---

## 3.15 Summary of methodological contribution

The methodology's distinguishing characteristic is the **ordering of its stages**. Ceiling
analysis precedes model selection; the deployment contract and latency budget precede source
selection; the alignment harness and baselines precede the first model; and every guard is
observed failing on broken input before being trusted. This ordering was adopted because the
alternative — training models first and validating afterwards — permits plausible, incorrect
results that agree with the investigator's prior to survive review.

The nine-stage protocol, together with a register of the specific defects each stage prevents,
is documented in full in Appendix [X] and in the project repository under `redo/`.
