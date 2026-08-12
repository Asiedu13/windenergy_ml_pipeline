# 06 — Iteration History

Complete chronological record of the four production iterations (v1 → v1.5 → v2
→ v3), including every configuration change, every hyperparameter set, every
result, and every bug encountered and fixed. This is the "engineering journal"
that lets a reader reconstruct how the pipeline reached its current state.

---

> 📐 **Reading the numbers.** Unless stated otherwise, a figure like `1.475` is a **test
> MAE in m/s** — how far off the model typically is when predicting hub-height wind speed
> 24 hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399, guessing the average 1.466. Percentages like `11.25%` are
> power error as a share of the turbine's 3.05 MW maximum. Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** the project's diary — every version, in order, with what
> changed, what it scored, and what broke.
> **Why keep a diary rather than just the final numbers:** the path matters as much as the
> destination here. One version's headline finding was later shown to be a bug, and seeing
> how a wrong conclusion was reached, believed, and eventually caught is the most
> transferable thing in this repository.
> **How to read it:** each version has *Changes*, *Results* and *Diagnosis*. Numbers from
> v2 and v3 deep models are **superseded** — see the warnings on those sections. For the
> current state go to [11_nwp_forecast_features.md](11_nwp_forecast_features.md); for
> definitions, [glossary.md](glossary.md).

## v1 — Initial NASA POWER build

> **What:** the first working version — simplest data source, short 6-hour horizon, six
> months of history.
> **Why it counts as a success despite failing:** the model *lost* to the trivial "nothing
> will change" baseline, and the pipeline said so loudly instead of burying it. That
> guardrail working on the very first run is what made every later result trustworthy.
> **What it taught:** a short horizon plus thin history plus coarse data is a combination
> where machine learning cannot add value — persistence is simply too strong.

**Date range built:** first working pipeline, before ERA5 access.
**Data:** NASA POWER, Anloga, **6 months** (Jan–Jun 2023), hourly, 4344 rows.
**Horizon:** **6 hours** (short — set for a first test with limited data).
**Model:** XGBoost with library defaults (unregularised).

### What we shipped in v1

The complete 8-step scaffold matching the plan file:
- `config.yaml`, `requirements.txt`, `README.md`, `.gitignore`, `data/`, `outputs/`
- All 7 `src/` modules: `utils`, `data_ingest`, `features`,
  `stage1_windspeed`, `stage2_powercurve`, `evaluate`, `pipeline`, `report`
- All 5 test files, 27 tests total, all passing

Feature set (v1):
- Hub extrapolation (log law) with `roughness=NaN` for NASA POWER → default 0.15
- Direction encodings on `wind_dir_10m` and `wind_dir_100m` (using WS50M as 100m
  proxy for NASA POWER)
- Calendar features
- **Lags: [1, 2, 3, 6, 12, 24]** (original set, before trimming)
- **Rolling windows: [3, 6, 24]** (original set)

XGBoost hyperparameters (library defaults):

```python
XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    random_state=42,
    tree_method="hist", n_jobs=-1,
)
```

### v1 Stage 1 result (mode: forecast, horizon: 6h)

| Model | Test MAE (m/s) | Test RMSE | Test MAPE | Test R² |
|:---|---:|---:|---:|---:|
| Persistence | **0.875** | 1.090 | 20.00% | −0.030 |
| Climatology | 0.996 | 1.226 | 21.10% | −0.302 |
| XGBoost (train) | 0.134 | 0.178 | 4.07% | 0.983 |
| XGBoost (test) | 1.045 | 1.283 | 21.63% | −0.427 |

**Verdict:** XGBoost lost to persistence (1.045 > 0.875 MAE), the pipeline
correctly flagged **NO — flag** in the report. Massive overfit (train 0.134 vs
test 1.045 = 8× gap) confirmed too-flexible model on 6 months of data.

### v1 two-stage power result

| Variant | nMAE | nRMSE | Note |
|:---|---:|---:|:---|
| physics_only | 5.89% | 7.55% | Deceptively good — short horizon + short window |
| two_stage | 6.47% | 9.02% | Worst — Stage 1 loses to persistence |
| direct | 5.97% | 8.32% | Middle |

**Why these small numbers are misleading:** at 6h horizon, "reference truth"
(physics-derived from actual observed weather) is very close to "physics from
persistence-forecast wind" because persistence is very strong at 6h. All three
variants look decent. The honest metric — does Stage 1 beat persistence? — says no.

### Key lessons from v1

1. **Short horizon + short data = persistence dominates.** Move to longer horizon
   and more data.
2. **Unregularised XGBoost overfits catastrophically** on this feature set.
3. **The pipeline's "beat persistence" guardrail worked** — it caught the failure
   rather than hiding it.
4. **Two-stage architecture validated at the pipeline level** — everything ran
   end-to-end, three-variant comparison produced a coherent report.

---

## v1.5 — Switch to ERA5, 24h horizon, regularise

> **What:** three changes at once — better weather data, a longer 24-hour horizon, and
> deliberately constraining the model.
> **Why lengthening the horizon *helped*:** counter-intuitively, predicting further ahead
> made the model look better. At 6 hours "nothing will change" is nearly unbeatable; at 24
> hours that assumption breaks down, which is precisely where forecasting earns its place.
> **The other big win:** the model's error on unseen data had been **eight times** its
> error on data it trained on — the signature of memorising. Constraining it brought that
> ratio down to about **1.2×**, where 1.0 would mean identical performance on both. It had
> been memorising; now it was learning.
> **Result:** the first version to beat the baseline.

**Prompted by:** the honest v1 verdict — need better data + longer horizon.

### Changes vs v1

- **`data.source: era5`** (was `nasa_power`) — 0.25° grid, 7 variables, native
  roughness.
- **`start / end: 2020-01-01 / 2023-12-31`** — 4 years (35,064 hourly rows).
- **`forecast.horizon_hours: 24`** (was 6) — persistence gets weaker at day-ahead
  → ML has room to add value.
- **XGBoost regularised:** `n_estimators=200, max_depth=5, min_child_weight=10,
  reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8`.
- **`lags: [1, 2, 3, 6, 12]`** (dropped 24 — was overfitting on lag24 in v1).
- **`rolling_windows: [3, 6]`** (dropped 24).
- **Monthly ERA5 chunking** implemented in `_fetch_era5` — one CDS request per
  (year, month), keeping each under CDS's 121,000-unit cost cap.
- **`pad=0.2°`** for ERA5 bbox (0.4° span) — CDS rejects sub-grid-cell bboxes,
  so must exceed 0.25° (ERA5 grid resolution).

### Bugs caught and fixed during v1.5

1. **CDS "cost limits exceeded"** on first attempt (4-year mega-request). Fixed
   by chunking to per-year, then per-month when per-year also failed.
2. **Licence acceptance required** — CDS returned `400 required licences not
   accepted` until manual click-through on Copernicus site.
3. **`roughness` all-NaN column** killed all test rows in `dropna`. Fixed:
   `build_feature_frame` now drops all-NaN feature columns *before* the row-wise
   `dropna`.

### v1.5 Stage 1 result (mode: forecast, horizon: 24h)

| Model | Test MAE (m/s) | Test RMSE | Test MAPE | Test R² |
|:---|---:|---:|---:|---:|
| Persistence | 1.440 | 1.832 | 32.01% | −0.002 |
| Climatology | 1.282 | 1.617 | 31.74% | 0.220 |
| XGBoost (train) | 0.969 | 1.239 | 22.88% | 0.561 |
| **XGBoost (test)** | **1.171** | **1.494** | 28.59% | **0.334** |

**Verdict:** XGBoost beats persistence (1.171 vs 1.440 MAE) by ~19%. Positive
R² for the first time. Overfit gap collapsed from 8× → 1.2× thanks to
regularisation. All plan guardrails green.

### v1.5 two-stage power result

| Variant | nMAE | nRMSE | R² |
|:---|---:|---:|---:|
| physics_only | 14.65% | 19.92% | −0.071 |
| **two_stage** | **12.02%** | **16.04%** | **0.305** |
| direct | 12.47% | 16.00% | −0.159 (from earlier report; MAPE-dominated) |

Two-stage beats physics-only by ~2.6 percentage points nMAE — architecture
validated at the power stage too. Gap to Energies 2021 benchmark: ~3.2 pp on nMAE.

### v1.5 top 15 features (SHAP-ranked)

Top predictors were: `wind_speed_10m`, `temperature_c`,
`wind_speed_hub_rollmean24` (later removed), `month_cos`, `wind_speed_100m`,
`wind_dir_10m_sin`, `hour_sin`, `wind_dir_100m_sin`, `pressure_pa`,
`wind_speed_hub_lag24`, `wind_speed_hub_lag12`, `wind_speed_hub_lag1`,
`wind_speed_hub_lag2`, `wind_dir_10m_cos`, `wind_speed_hub_lag3`.

Feature story is physically sensible: near-surface wind + temperature
(density effect) + rolling means + diurnal calendar dominate.

---

## v2 — Richer features + spatial context + LSTM sibling

> **What:** many more input variables — winds high above the site, humidity, turbulence
> depth, and measures of how conditions vary across the map — plus the first neural
> network.
> **Why add so much:** the physics is sound. Winds aloft steer surface winds; pressure
> differences across space create wind. Both should carry information about tomorrow.
> **What actually happened:** a modest improvement, later shown to be smaller than it
> looked — the spatial features turned out to be worth almost nothing, partly because the
> downloaded map patch was too small for them to work.
> ⚠️ **The neural network's score here is invalid** — it was measured with a bug not found
> until v4. Do not quote it.

**Prompted by:** v1.5 result was defensible but ~3 pp above benchmark. User chose
three stacked upgrades (from a multi-select AskUserQuestion): richer features,
spatial context, and adding a deep-learning sibling.

### Changes vs v1.5

**Data:**
- Added **v2 extras** single-level ERA5 variables:
  `10m_wind_gust_since_previous_post_processing`, `2m_dewpoint_temperature`,
  `boundary_layer_height`. Cached as `era5x_...`.
- Added **pressure-level dataset** (new CDS request family): `u`, `v`,
  `geopotential` at 850 hPa and 500 hPa. Cached as `era5pl_...`.
- Existing v1 core caches (`era5_...`) retained — no re-download of the 7
  original variables.

**Feature engineering:**
- **Spatial derivatives** computed for 5 primary variables (`wind_speed_100m`,
  `wind_speed_850hpa`, `geopot_850hpa`, `pressure_pa`, `temperature_c`) × 4
  stats (`_mean`, `_std`, `_grad_ns`, `_grad_ew`) = **20 new columns**.
- Grid patch loaded as 3×3 (actually **1×2** — measured 2026-08-10; see
  [03_data_sources.md](03_data_sources.md). This belief was wrong for the whole v2–v4
  period and made 10 of the 20 spatial columns degenerate.)
  rather than collapsing to nearest cell.
- Direction encodings extended to include `wind_dir_850hpa` and `wind_dir_500hpa`.

**Models:**
- New file `src/stage1_deep.py` with `BlockLSTM` (via darts) — same
  `Stage1Result` interface as tree track.
- Pipeline dispatcher (`src/pipeline.py`) accepts `stage1_algo: lstm | tft`.
- `--compare` CLI flag introduced: trains primary + one deep sibling and
  reports both.

**Config:**
- New `deep:` sub-block with `input_chunk_length=48, hidden_size=64, n_epochs=30,
  batch_size=128, dropout=0.2`.
- New dependency: `darts[torch]>=0.30` in `requirements.txt`.

### Bugs caught and fixed during v2

1. **darts `RNNModel` only supports `future_covariates`** — swapped to
   `BlockRNNModel` which supports `past_covariates`.
2. **`pd_series()` renamed to `to_series()`** in newer darts — API drift fixed.
3. **CDS retry on transient failure** — added `_retrieve_with_retry` with
   3 attempts × 60s backoff after job `0b9eff28...` failed mid-run.
4. **CDS returns ZIP for accumulated variables** — the extras family
   (`10m_wind_gust...`) is packaged as ZIP containing multiple NetCDFs, not a
   raw `.nc`. Added ZIP-magic-byte detection + unpack in `_open_dataset`.

### v2 Stage 1 result

> ⚠️ **The LSTM row is invalid** — scored 24 h out of phase. Retained as the
> historical record of what was believed at the time. Correct value: **1.114**.
> See [v4](#v4--the-alignment-fix) below.

| Model | Test MAE (m/s) | Test RMSE | Test R² | Beats persistence? |
|:---|---:|---:|---:|:---:|
| **XGBoost (regularised)** | **1.153** | **1.472** | **0.354** | ✅ |
| ~~LSTM (BlockRNN, 30 epochs, hidden=64)~~ *misaligned* | ~~1.475~~ | ~~1.861~~ | ~~−0.033~~ | ~~❌~~ |
| Persistence (25 h — see v4) | 1.440 | 1.832 | −0.002 | — |

### v2 two-stage power result

| Variant | nMAE | nRMSE | R² |
|:---|---:|---:|---:|
| physics_only | 14.65% | 19.92% | −0.071 |
| **two_stage (XGBoost)** | **11.77%** | **15.81%** | **0.325** |
| direct | 12.17% | 15.71% | 0.334 |

**Improvement vs v1.5:** two-stage nMAE 12.02% → **11.77%**, R² 0.305 →
0.325. Small but real (~0.25 pp), driven by the richer feature set.

### v2 top 15 features

New v2 features already appearing:
- `wind_speed_100m_mean` at #2 (spatial mean)
- `wind_dir_850hpa_sin` at #4 (upper-air direction)
- `bl_height_m` at #11 (v2 extra)
- `wind_dir_100m_sin` at #6
- `pressure_pa` at #8
- `wind_speed_hub_lag12` at #10

Upper-air winds and BL height earn their keep — pipeline extension paid off
qualitatively even if the quantitative improvement was modest.

### Key lessons from v2

1. ~~**LSTM with default hyperparameters and 30 epochs cannot beat persistence**
   at day-ahead on this data volume. Undertrained.~~ **Wrong** — the LSTM was
   misaligned, not undertrained. It converges in 24 epochs and beats persistence
   comfortably (v4).
2. **Richer feature set gives small (~0.25 pp) improvement** — probably close
   to what these features can extract; further gains need more data or better
   models.
3. **Data engineering was harder than modelling** — ERA5 chunking, licence,
   ZIP quirk, retry logic collectively took more time than everything else in v2.

---

## v3 — More architectures + tuned XGBoost benchmark

> **What:** a five-model bake-off — the tree model with automated tuning, against four
> different neural networks.
> **What it appeared to show:** the simple tree model won comfortably; four of five neural
> networks lost even to the trivial baseline. This matched a well-known rule of thumb and
> became the project's headline finding.
> ⚠️ **It was wrong.** Every neural network was scored against the wrong hour. They were
> not being measured at all — the comparison was re-measuring the baseline four times.
> **Why this section is still worth reading:** the conclusion was confident, coherent, and
> survived review. Understanding *why* it was convincing is more instructive than the
> corrected numbers in [10](10_v4_deep_model_retraining.md).

**Prompted by:** the LSTM disappointment in v2. Was it because deep learning
doesn't work here, or because *one particular* undertuned LSTM didn't work?

**User chose (multi-select AskUserQuestion):** add TFT + N-BEATS + DLinear as
siblings, tune XGBoost with Optuna. Skipped: LSTM training fixes, ensembling.

### Changes vs v2

**Deep model expansion** in `src/stage1_deep.py`:
- Added `tft` → `TFTModel(hidden_size=64, add_relative_index=True,
  use_static_covariates=False, output_chunk_length=24)`
- Added `nbeats` → `NBEATSModel(num_stacks=3, num_blocks=3, num_layers=4,
  layer_widths=256)`
- Added `dlinear` → `DLinearModel(kernel_size=25, use_static_covariates=False)`

All four use the same `common` block (input_chunk_length=48, n_epochs=30,
batch_size=128, dropout=0.2, CPU, random_state=42).

**Optuna XGBoost tuning** in `src/stage1_windspeed.py`:
- `tune_xgboost(X_train, y_train, config, n_trials=50)`
- TPE sampler, 3-fold TimeSeriesSplit CV, 50 trials
- Search space per [04_stage1_wind_speed.md](04_stage1_wind_speed.md)
- Config additions: `tune_xgboost: true`, `tune_n_trials: 50`

**Pipeline `--compare` extended:**
- Now iterates over `config.model.compare_algos: [xgboost, lstm, tft, nbeats, dlinear]`
- Per-algo try/except so one failure doesn't kill the whole run
- Report ranks by test MAE and bolds the winner
- Best model (lowest test MAE) used for downstream two-stage evaluation

**Tests:** 28 → 32 (added smokes for TFT, N-BEATS, DLinear + Optuna).

### Bug caught and fixed during v3

**`DEEP_ALGOS` whitelist bug in `src/pipeline.py`:**

```python
# BEFORE (bug):
DEEP_ALGOS = {"lstm", "tft"}   # missing nbeats and dlinear

# AFTER:
DEEP_ALGOS = {"lstm", "tft", "nbeats", "dlinear"}
```

Consequence: first v3 full run trained xgboost + lstm + tft, then silently
skipped nbeats and dlinear with `ERROR pipeline | Model nbeats failed:
Unknown stage1_algo: 'nbeats' (skipping)`. Fixed with a one-line change and
reran with `compare_algos: [nbeats, dlinear]` to fill in the missing two
without re-training the others (TFT had taken 2h 45min).

### v3 wall-time (actual, not estimated)

| Stage | Wall time |
|:---|---:|
| Data load (all 144 cached NetCDFs) | ~1 min |
| Feature build + SHAP | ~5 min |
| Optuna XGBoost sweep (50 trials) | **~4 min** (estimate was 30) |
| XGBoost final 5-fold CV + fit | ~1 min |
| LSTM training | ~9 min |
| **TFT training** | **~2h 45min** (estimate was 30) |
| N-BEATS training (rerun) | ~12 min |
| DLinear training (rerun) | ~2 min |
| Evaluation + report writing | ~1 min |

TFT dominated wall-time by a wide margin because attention on CPU scales
poorly with input × output sequence length × covariate dim.

### v3 Stage 1 consolidated result (all 5 models)

> ⚠️ **All four deep-model rows are invalid** — scored 24 h out of phase, which
> pinned each of them to the persistence error floor. The ranking below is an
> artefact, not a result. XGBoost is unaffected (it never went through darts).
> Corrected values in [v4](#v4--the-alignment-fix). Retained as the historical
> record.

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | Beats persistence? |
|---:|:---|---:|---:|---:|:---:|
| 1 | **XGBoost (Optuna-tuned)** | **1.143** | **1.461** | **0.364** | ✅ |
| 2 | ~~N-BEATS~~ *misaligned* | ~~1.425~~ | ~~1.798~~ | ~~0.036~~ | ~~✅ (barely)~~ |
| — | Persistence baseline (25 h) | 1.440 | 1.832 | −0.002 | — |
| 3 | ~~LSTM~~ *misaligned* | ~~1.475~~ | ~~1.861~~ | ~~−0.033~~ | ~~❌~~ |
| 4 | ~~DLinear~~ *misaligned* | ~~1.519~~ | ~~1.897~~ | ~~−0.074~~ | ~~❌~~ |
| 5 | ~~TFT~~ *misaligned + quantile objective* | ~~1.643~~ | ~~2.088~~ | ~~−0.301~~ | ~~❌~~ |

### v3 best Optuna XGBoost hyperparameters

```python
{
    "max_depth": 4,
    "learning_rate": 0.0185,
    "n_estimators": 450,
    "min_child_weight": 16,
    "subsample": 0.6261,
    "colsample_bytree": 0.6451,
    "reg_alpha": 0.00193,
    "reg_lambda": 0.5571,
}
```

Best 3-fold CV MAE: **1.1117 m/s**.

### v3 two-stage power result (using tuned XGBoost as Stage 1)

| Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|:---|---:|---:|---:|---:|---:|
| physics_only | 446,962 | 607,452 | −0.071 | 14.65% | 19.92% |
| **two_stage** | **355,771** | **478,077** | **0.337** | **11.66%** | **15.67%** |
| direct | 371,247 | 479,103 | 0.334 | 12.17% | 15.71% |

Best two-stage nMAE at the time: **11.66%**. Gap to Energies 2021 benchmark
(~8.8%): 2.86 pp. Progression: v1.5 12.02% → v2 11.77% → v3 11.66%.
Diminishing returns visible — read then as a data-related ceiling.

---

## v4 — the alignment fix

> **What:** the bug was found, fixed, and everything re-run. The previous version's
> headline conclusion reversed completely.
> **How it came to light:** while implementing improvements meant to *help* the neural
> networks, the comparison code was read closely enough to notice the two tables were
> stamped on different clocks, 24 hours apart.
> **The scale of the correction:** the neural networks improved by 0.27–0.36 m/s — more
> than any deliberate design decision in the project. All five models now beat the
> baselines, and they finish within 0.101 m/s of each other.
> **The finding that replaced the old one:** architecture choice barely matters here.
> Measurement correctness mattered enormously.

**Prompted by:** implementing the "what would actually help deep models" list at
the end of [09_why_deep_learning_underperformed.md](09_why_deep_learning_underperformed.md).
The intent was to give the deep models a fair fight. What surfaced instead was
that they had never had a fair *measurement*.

### The bug

`historical_forecasts(..., last_points_only=True)` returns a series indexed by
**target time τ** — `hist[τ]` predicts the wind speed *at* τ from data up to
`τ − 24`. The feature frame is indexed by **feature time t**, where
`y[t] = hub(t + 24)`. `src/stage1_deep.py` did:

```python
pred_test = hist_series.reindex(y_test.index)   # hist[t] vs y[t] = hub(t+24)
```

scoring the model's estimate of `hub(t)` against `hub(t + 24)`. That error is
`|hub(t) − hub(t+24)|` — the definition of 24 h persistence error. Every deep
model was therefore pinned to the persistence floor with its own error on top,
which is why v2/v3 produced a tight, plausible, entirely fictitious ranking.

Verified against darts before changing anything: with `start = 16:00` and
`forecast_horizon = 24`, the first returned point is at `next-day 15:00`, i.e.
`start + horizon − 1`.

### Changes vs v3

- `_to_feature_time()` shifts a backtest result from target time to feature time;
  backtest `start` moved to `split_ts + offset − horizon + 1` so the first point
  lands on `y_test.index[0]`.
- **Rec #1** — `val_fraction` tail of the train slice + `EarlyStopping` on
  `val_loss` + **best-checkpoint restore** via `load_from_checkpoint(best=True)`.
- **Rec #2** — `include_target_lags` (default true): lag/rolling columns as past
  covariates, 39 → 48.
- **Rec #3** — `WarmupCosineLR`, `gradient_clip_val: 1.0`, short-budget `lr_sweep`.
- **Rec #5** — `config_10yr.yaml` + `src/data_ingest_extend.py` to extend ERA5 to
  2010–2023 (resumable; running in the background).
- Rec #4 (multi-site) not actionable — Anloga is the only site.
- `compare_algos` reordered cheapest-first so a TFT crash still leaves four models.

### Bugs caught and fixed during v4

1. **The 24 h alignment bug** (above) — the headline.
2. **`joblib.dump` silently drops darts model weights.**
   `TorchForecastingModel.__getstate__` deliberately omits the LightningModule,
   so `save_stage1` was writing a weightless shell. Never bit before because
   XGBoost always won and the branch was never taken. Now uses darts' own
   serializer with a `stage1.darts.pt` sidecar.
3. **The report hardcoded `config.model.stage1_algo`** rather than the `--compare`
   winner. `write_report` now takes `stage1_algo_label`.
4. **The persistence baseline is a 25 h persistence, not 24 h.** Found, quantified
   (1.399 vs 1.440), and *deliberately left unchanged* so v1–v4 stay comparable.

**Tests:** 32 → 38. New: alignment unit + end-to-end (37-hour period, chosen so a
shifted model cannot pass), deep-config resolution, lag toggle, darts save/load
round-trip, TFT `loss_fn` determinism.

### v4 wall-time

| Stage | Time |
|:---|---:|
| Optuna XGBoost (50 trials) | ~4 min |
| DLinear (sweep + 23 epochs + backtests) | ~2 min |
| LSTM (sweep + 24 epochs) | ~10 min |
| N-BEATS (sweep + 19 epochs) | ~27 min |
| TFT (sweep + 28 epochs, quantile) | 4 h 33 min |
| **Total `--compare` run** | **5 h 17 min** |

### v4 Stage 1 result

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | Early stop | v3 MAE |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** | 24 / 200 | 1.475 |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 | — | 1.143 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 | 19 / 100 | 1.425 |
| 4 | TFT (quantile — fixed in v4b) | 1.551 | 1.974 | −0.163 | 28 / 60 | 1.643 |
| 5 | DLinear | 1.215 | 1.530 | 0.302 | 23 / 200 | 1.519 |
| — | Climatology | 1.282 | 1.617 | 0.220 | — | — |
| — | Persistence (24 h, correct) | 1.399 | 1.791 | 0.043 | — | — |
| — | Persistence (25 h, as reported) | 1.440 | 1.832 | −0.002 | — | — |

XGBoost is unchanged to three decimals — the tree path was untouched, so it is
the control confirming the deep-model movement is real.

### v4 two-stage power result (LSTM as Stage 1)

| Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|:---|---:|---:|---:|---:|---:|
| physics_only | 446,962 | 607,452 | −0.071 | 14.65% | 19.92% |
| **two_stage** | **343,197** | **472,715** | **0.351** | **11.25%** | **15.50%** |
| direct | 371,247 | 479,103 | 0.334 | 12.17% | 15.71% |

Best across all iterations: **11.25%**. Gap to Energies 2021 (~8.8%): 2.45 pp.

### Key lessons from v4

1. **A bug that moves every model toward the same plausible value is far more
   dangerous than one that produces garbage.** The v3 table was internally
   consistent, ordered sensibly by model capacity, and matched a well-known
   prior. The tell was in plain sight: four unrelated architectures clustering
   within 0.2 m/s of persistence is a fingerprint, not a finding.
2. **Smoke tests that assert shape and finiteness cannot catch this.** The v4
   regression test asserts a *quality floor* against a baseline instead.
3. **Doc 09's own recommendations were mostly not the lever.** "Train 200 epochs"
   was moot (convergence at 19–29); "feed the lag features" was neutral-to-harmful
   (DLinear 1.215 with vs 1.196 without). Only the LR sweep helped, and it picked
   a different LR for each model.
4. **What v3 actually lacked was validation-based stopping with best-checkpoint
   restore**, not epochs.

---

## v4b — TFT on the same objective as everyone else

> **What:** a follow-up fixing the one model still losing after the alignment repair.
> **The cause:** an unset library default had it predicting a full range of possible
> outcomes — 17 percentiles — while every other model predicted a single number, from the
> same data and the same time budget. It was then graded on only the middle answer.
> **The fix:** one configuration line, worth 0.380 m/s. More than any architectural
> decision in the project.
> **Why it is filed as its own version:** it was run on its own rather than by repeating
> the seven-hour full comparison, so the auto-generated report still shows the old figure.
> `RESULTS.md` is the authority.

**Prompted by:** TFT being the only model still losing to persistence after v4.

**Cause:** `darts.models.TFTModel` substitutes `likelihood=QuantileRegression()`
— 17 quantiles — whenever *both* `likelihood` and `loss_fn` are left `None`.
BlockRNN, N-BEATS and DLinear all default to plain MSE. TFT was fitting a full
predictive distribution on the same 28k samples while its competitors fitted a
point estimate, and only the median was scored. It is also why its `val_loss`
read ~0.42 against the others' ~0.008 — different objective, not a broken model.

**Change:** `model.deep.loss_fn: mse` (new knob, applied uniformly; a no-op for
the other three). Guarded by `test_loss_fn_makes_tft_deterministic`.

| TFT objective | Test MAE | RMSE | R² | Early stop | Beats persistence? |
|:---|---:|---:|---:|---:|:---:|
| `QuantileRegression` (darts default) | 1.551 | 1.974 | −0.163 | 28 / 60 | ❌ |
| **`MSELoss`** | **1.171** | **1.498** | **0.330** | 29 / 60 | ✅ |

Both early-stopped at effectively the same point, so this is not a budget effect.

**Wall time:** 596 min. Not comparable to the quantile run's 273 min — the
machine was in interactive use and this is a 15 W laptop part, so the difference
is thermal throttling (~8 → ~18.6 min/epoch), not architecture.

### Consolidated final standings (v4 + v4b)

| Rank | Model | Test MAE (m/s) | RMSE | R² |
|---:|:---|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 |
| 4 | TFT (MSE) | 1.171 | 1.498 | 0.330 |
| 5 | DLinear | 1.215 | 1.530 | 0.302 |
| — | Persistence (24 h) | 1.399 | 1.791 | 0.043 |

All five beat both persistence baselines. The spread across architectures
(0.101 m/s) is smaller than the gap from the weakest to persistence (0.184 m/s).

> Single seed, single chronological split — the LSTM/XGBoost ordering is a tie.
> `outputs/reports/report.md` still shows the quantile TFT; the five-model
> report is preserved at `outputs/reports/report_v4_bakeoff.md`.

---

## Full Stage 1 progression summary

> **What:** every wind-speed result from every version, in one table.
> **Why it is worth a careful look:** the honest story is one of diminishing returns.
> The single biggest jump came early, from switching data source and lengthening the
> horizon; everything after that is incremental.
> ⚠️ **Read the superseded markings.** Several rows are invalid measurements retained for
> the record, not results.

| Iteration | Data source | Horizon | Best model | Test MAE | Test R² | Beats persistence? |
|:---|:---|---:|:---|---:|---:|:---:|
| v1 | NASA POWER, 6 mo | 6 h | XGBoost (default) | 1.045 | −0.427 | ❌ |
| v1.5 | ERA5, 4 yr | 24 h | XGBoost (regularised) | 1.171 | 0.334 | ✅ |
| v2 | ERA5 v2, 4 yr | 24 h | XGBoost (regularised) | 1.153 | 0.354 | ✅ |
| v3 | ERA5 v2, 4 yr | 24 h | XGBoost (Optuna) | 1.143 | 0.364 | ✅ |
| **v4** | **ERA5 v2, 4 yr** | **24 h** | **LSTM (aligned, early-stopped)** | **1.114** | **0.390** | ✅ |

Persistence quoted as 1.440 (25 h) throughout; the correct 24 h baseline is
1.399. v3's deep-model rankings were misaligned — v4 is the first iteration whose
deep-model numbers are trustworthy.

## Full two-stage power progression summary

> **What:** the same progression, but measured in electricity rather than wind speed.
> **Why the two tables do not move in step:** a better wind forecast does not convert
> one-for-one into a better power forecast. The power curve produces nothing below cut-in
> and flattens above rated speed, so accuracy gains in those ranges are simply absorbed.
> **What to take from it:** roughly 12.02% error down to 11.25% across all iterations —
> real, steady, unspectacular progress.

| Iteration | physics_only nMAE | **two_stage nMAE** | direct nMAE | physics_only nRMSE | **two_stage nRMSE** | direct nRMSE |
|:---|---:|---:|---:|---:|---:|---:|
| v1 (6h, 6mo, NASA POWER) | 5.89% | 6.47% | 5.97% | 7.55% | 9.02% | 8.32% |
| v1.5 (24h, 4yr, ERA5) | 14.65% | **12.02%** | 12.47% | 19.92% | **16.04%** | 16.00% |
| v2 (24h, 4yr, ERA5 v2) | 14.65% | **11.77%** | 12.17% | 19.92% | **15.81%** | 15.71% |
| v3 (24h, 4yr, ERA5 v2 + Optuna) | 14.65% | 11.66% | 12.17% | 19.92% | 15.67% | 15.71% |
| **v4 (24h, 4yr, ERA5 v2 + LSTM)** | **14.65%** | **11.25%** | **12.17%** | **19.92%** | **15.50%** | **15.71%** |

## Test suite growth

> **What:** how the automated tests grew from 27 to 47 across the project.
> **Why track it here:** the count is less interesting than the change in *kind*. The
> early tests checked that code ran and returned the right shape — and the alignment bug
> passed all of them comfortably.
> **What changed after v4:** newer tests assert *quality*, using signals constructed so
> that a subtly broken model cannot pass. The v5 forecast tests follow the same pattern:
> a deliberately perfect forecast must reproduce the answer exactly, on a signal whose
> cycle does not divide evenly into the forecast horizon.

| Version | Total tests | New in this version |
|:---|---:|:---|
| v1 | 27 | Full baseline: data_ingest, features, stage1, stage2, pipeline |
| v1.5 | 27 | No new tests (regressions still green) |
| v2 | 28 | +1: LSTM smoke test |
| v3 | 32 | +4: TFT smoke, N-BEATS smoke, DLinear smoke, Optuna smoke |
| **v4 / v4b** | **38** | +6: alignment unit (`_to_feature_time`), alignment end-to-end (37 h period), deep-config resolution + overrides, lag-toggle, darts save/load round-trip, TFT `loss_fn` determinism |

All 38 tests green as of v4b.

The v4 additions are deliberately different in kind from the v1–v3 smoke tests.
Those assert *shape and finiteness* — that a model runs and emits the right number
of finite floats — which is precisely why they could not catch the alignment bug.
`test_deep_forecast_predictions_are_horizon_aligned` asserts a **quality floor**
relative to a baseline instead, on a signal engineered so that a misaligned model
must fail it.
