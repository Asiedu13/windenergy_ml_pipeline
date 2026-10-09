# Wind Power Forecasting — Results Log

Chronological record of every training iteration for the Anloga (SE Ghana) site.
Auto-generated report is at [outputs/reports/report.md](outputs/reports/report.md);
this file is the human-maintained journey summary that survives report overwrites.

> **Looking for deeper documentation?** See [docs/README.md](docs/README.md) — full
> design rationale, module map, data-source details, all model hyperparameters,
> per-iteration change log, findings analysis, and next-steps catalogue across 9
> cross-linked files.

> 📐 **Reading the numbers.** In the Stage 1 tables, a figure like `1.114` is a **test MAE
> in m/s** — how far off the model typically is when predicting hub-height wind speed 24
> hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399, guessing the average 1.466. In the Stage 1+2 tables, `nMAE`
> and `nRMSE` are power error as a percentage of the turbine's 3.05 MW maximum, also
> lower-is-better. R² runs the other way — higher is better, 1.0 is perfect, 0 means no
> better than always guessing the average, and negative is worse than that.
> Full explanation: [docs/glossary.md — How to read the numbers](docs/glossary.md#how-to-read-the-numbers).

**Site:** Anloga, Ghana (5.79°N, 0.90°E), 100 m hub, Enercon E-101/3050 (3.05 MW).
**Reference truth:** physics-derived power from observed ERA5 wind speed through the
manufacturer power curve, density-corrected. No real Ghana SCADA available; this
validates the *forecasting* step, not absolute physical yield.

---

> ✅ **v5 has landed (2026-08-11 23:18).** The residual gap was measured to be
> **information-limited, not sample-limited**, so v5 added operational NWP forecasts as
> Stage 1 features. On the full 22-month ERA5 window this cuts Stage 1 test MAE from
> **1.0513 → 0.7295 m/s (−30.6%)** and lifts R² from **0.3300 → 0.6653**, decisively past
> the **0.326** ceiling that bounds any predictor built on the atmospheric state at time
> *t*. End-to-end power reaches **7.48% nMAE / 10.08% nRMSE**, against v4's 11.25% / 15.50%.
> All four verification gates pass. See [v5](#v5--nwp-forecast-features-in-progress-started-2026-08-10).

## Results at a glance (all iterations)

### Stage 1 — hub-height wind-speed forecast

| Iteration | Data source | Horizon | Model | Test MAE (m/s) | Test RMSE | Test R² | Test MAPE | Beats persistence? |
|:---|:---|---:|:---|---:|---:|---:|---:|:---:|
| v1 (initial) | NASA POWER, 6 mo | 6 h | XGBoost (default) | 1.045 | 1.283 | -0.427 | 21.63% | ❌ No |
| v1.5 (ERA5) | ERA5, 4 yr | 24 h | XGBoost (regularised) | 1.171 | 1.494 | 0.334 | 28.59% | ✅ Yes |
| v2 (rich features) | ERA5 v2, 4 yr | 24 h | XGBoost (regularised) | **1.153** | **1.472** | **0.354** | 28.10% | ✅ Yes |
| v2 (rich features) | ERA5 v2, 4 yr | 24 h | LSTM (BlockRNN, 30ep) | 1.475 | 1.861 | -0.033 | — | ❌ No |
| v3 | ERA5 v2, 4 yr | 24 h | Optuna-XGBoost | 1.143 | 1.461 | 0.364 | — | ✅ Yes |
| **v4** (alignment fix) | ERA5 v2, 4 yr | 24 h | **LSTM (aligned, early-stopped)** | **1.114** | **1.429** | **0.390** | 26.41% | ✅ Yes |
| **v5** (+ NWP forecasts) | ERA5, 22 mo ‡ | 24 h | **Optuna-XGBoost + NWP** | **0.738** | **0.950** | **0.660** | 16.93% | ✅ Yes |

> ‡ **v5 runs on a different window** (2024-03 → 2025-12, the span of the Open-Meteo
> forecast archive) so it is not strictly like-for-like with v1–v4 on 2020–2023.
> **The windows are closer than expected, though:** 24 h persistence is **1.336 m/s** on
> v5's window versus **1.399** on v4's — a 4.5% easier bar, not a different problem. An
> earlier 12-month subset of this window *was* markedly calmer (persistence 1.09) and gave
> an inflated +40% gain; the full window does not have that defect.

> ⚠️ **All deep-model rows before v4 are invalid.** They were scored 24 hours out
> of phase — see [v4](#v4--the-alignment-fix-and-what-it-cost-us) below. The
> XGBoost rows are unaffected.

**Persistence baselines** (for reference):
- v1 (6h horizon): MAE 0.875, RMSE 1.090, R² -0.030 — very strong at short horizon
- v1.5 / v2 (24h horizon): MAE 1.440, RMSE 1.832, R² -0.002 — collapses at day-ahead

> ⚠️ **The 24h persistence baseline quoted throughout is actually a 25-hour
> persistence.** `PersistenceForecaster` predicts from `wind_speed_hub_lag1` =
> `hub(t-1)`, scored against `hub(t+24)`. True 24 h persistence is **MAE 1.399,
> RMSE 1.791, R² 0.043** — a 2.9% harder bar. Left unchanged so v1–v4 stay
> comparable; see docs/10 §4.

### Stage 1+2 — end-to-end power forecast

Reference: Energies 2021 two-stage benchmark ≈ **8.8% nMAE / 13% nRMSE**.

| Iteration | Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|:---|:---|---:|---:|---:|---:|---:|
| **v1** (NASA POWER, 6h, 6mo) | physics_only | 179,703 | 230,233 | 0.047 | 5.89% | 7.55% |
| v1 | two_stage | 197,291 | 275,054 | -0.360 | 6.47% | 9.02% |
| v1 | direct | 182,134 | 253,873 | -0.159 | 5.97% | 8.32% |
| **v1.5** (ERA5, 24h, 4yr) | physics_only | 446,962 | 607,452 | -0.071 | 14.65% | 19.92% |
| v1.5 | two_stage | 366,682 | 489,283 | 0.305 | 12.02% | 16.04% |
| v1.5 | direct | 380,204 | 253,873 | -0.159 | 12.47% | 16.00% |
| **v2** (ERA5 v2, richer features) | physics_only | 446,962 | 607,452 | -0.071 | 14.65% | 19.92% |
| v2 | **two_stage** | **358,998** | **482,096** | **0.325** | **11.77%** | **15.81%** |
| v2 | direct | 371,247 | 479,103 | 0.334 | 12.17% | 15.71% |
| **v3** (Optuna + deep zoo) | two_stage | 355,771 | 478,077 | 0.337 | 11.66% | 15.67% |
| **v4** (alignment fix, LSTM) | **two_stage** | **343,197** | **472,715** | **0.351** | **11.25%** | **15.50%** |
| **v5** (+ NWP, ERA5 22 mo ‡) | physics_only | 420,250 | 558,120 | -0.071 | 13.78% | 18.30% |
| **v5** | **two_stage** | **228,048** | **307,296** | **0.675** | **7.48%** | **10.08%** |
| **v5** | direct | 226,728 | 300,333 | 0.690 | 7.43% | 9.85% |

> **Note on v1's tight-looking numbers**: v1 used a 6h horizon on a 6-month NASA POWER
> window, which makes the "reference truth" easier to hit (short horizon + fewer weather
> regimes). It looked good but Stage 1 lost to persistence — the pipeline flagged this
> exactly as designed. The v1.5+ numbers are the honest benchmark.

---

## Journey (chronological)

### v1 — Initial NASA POWER build (short horizon, 6 months of data)

**Config:** NASA POWER, Anloga, 6-hour horizon, Jan–Jun 2023 only, XGBoost defaults.

**What we shipped:**
- Full 8-step pipeline scaffold (`src/{data_ingest,features,stage1_windspeed,stage2_powercurve,pipeline,evaluate,report}.py`)
- 27 passing tests
- Auto-generated markdown report + figures + persisted Stage 1 model
- Two-stage architecture (Stage 1 ML → Stage 2 windpowerlib), fully decoupled

**Result:** XGBoost did **not** beat persistence (MAE 1.045 vs 0.875). The pipeline correctly
flagged this in the report rather than hiding it. Train/test overfit gap was 8× (train MAE
0.134, test 1.045) — clearly too-flexible model on thin data.

**Diagnosis:** short horizon + short history + coarse grid = persistence dominates.

---

### v1.5 — Switch to ERA5, 24h horizon, 4 years, regularised model

**Changes:**
- Switched `data.source: era5` (0.25° grid, 7 variables, roughness included)
- Extended date range to 2020–2023 (4 years, 35k hourly rows)
- Bumped horizon to 24 h (persistence gets weaker → ML has room)
- Regularised XGBoost: `n_estimators=200, max_depth=5, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8`
- Trimmed lag/rolling features (dropped lag24 + roll24 to reduce overfit)
- Chunked ERA5 downloads by month (CDS cost cap = 121k units per request); added `pad=0.2°` (0.25° minimum grid span); auto-retry on CDS glitches

**Result:** Stage 1 XGBoost **beats** persistence for the first time (MAE 1.171 vs 1.440,
R² 0.334). Two-stage nMAE = 12.02%, R² 0.305. Overfit gap collapsed from 8× → 1.2×.

**Diagnosis:** the plan's guardrails were validated — architecture works, feature story is
physically sensible (wind_speed_100m + temperature + rolling features dominate SHAP).
Still ~3–4 points above the Energies 2021 8.8% nMAE benchmark, likely data-limited
(ERA5 is 25 km grid, no real observations).

---

### v2 — Richer features + spatial context + LSTM sibling

**Changes:**
- **ERA5 v2 extras** (extra CDS request family per month): 10m wind gust, 2m dewpoint,
  boundary-layer height
- **ERA5 pressure levels** (third CDS request family): u/v/geopotential at 850 hPa and
  500 hPa
- **Spatial derivatives** across what was believed to be a 3×3 grid patch (it is
  **1×2** — measured 2026-08-10, see v5 notes): `_mean`, `_std`, `_grad_ns`,
  `_grad_ew` for the 5 primary variables (`wind_speed_100m`, `wind_speed_850hpa`,
  `geopot_850hpa`, `pressure_pa`, `temperature_c`) — 20 new columns
- **New deep-model module** `src/stage1_deep.py` using darts (BlockLSTM, TFT stubbed)
- **`--compare` CLI flag** trains both tree and deep model, picks the winner
- Config now supports `stage1_algo: xgboost | lightgbm | lstm | tft`

**Result:**
| Model | Test MAE | Test R² | Beats persistence? |
|---|---:|---:|:---:|
| **XGBoost** | **1.153** | **0.354** | ✅ |
| LSTM (30ep, hidden=64, 1 layer) | 1.475 | -0.033 | ❌ |
| Persistence | 1.440 | -0.002 | — |

Two-stage nMAE: **11.77%** (down from 12.02%). Two-stage R²: 0.325.

Feature ranking now includes the new variables in the top 15: `wind_speed_100m_mean` (#2),
`wind_dir_850hpa_cos` (#6), `dewpoint_c` (#8), `temperature_c_mean` (#9), `bl_height_m`
(#11), `temperature_c_grad_ew` (#13), `wind_dir_850hpa_sin` (#15).

**Diagnosis:** improvement is real but modest (~0.25 pt nMAE). LSTM disappointed — likely
undertrained (30 epochs, no early stopping) and BlockRNN's fixed seq2seq mapping is a
harder task than XGBoost's single-timestep regression with lag features.

---

### v3 — Multi-architecture bakeoff with tuned XGBoost (in progress)

**Changes:**
- Extended `src/stage1_deep.py` with **TFT** (Temporal Fusion Transformer),
  **N-BEATS**, and **DLinear** (Zeng et al. 2022 linear baseline).
- Added **Optuna hyperparameter tuning** for XGBoost (`src/stage1_windspeed.py`):
  50-trial TPE sampler with 3-fold TimeSeriesSplit CV, searches over depth, learning
  rate, n_estimators, min_child_weight, subsample, colsample, L1/L2 regularisation.
- `--compare` now iterates over `config.model.compare_algos` list and trains every
  configured architecture side-by-side; report ranks them by test MAE.
- 32 tests passing (28 v2 + 4 new: 3 deep-model smoke tests + Optuna sweep test).

**Expected wall time:** ~90 min (Optuna 30 min, XGB fit 1 min, LSTM 10 min, TFT 30 min,
N-BEATS 15 min, DLinear 2 min).

**Run complete.** Real timing was dominated by TFT (2h 45min, not the estimated 30 min —
attention on CPU is expensive). Optuna was much faster than estimated (~4 min for 50 trials).

**⚠️ Bug caught mid-run:** `DEEP_ALGOS` set in `pipeline.py` only listed `{lstm, tft}` —
N-BEATS and DLinear branches existed in `stage1_deep.py` but the dispatcher didn't
recognise them, so they were skipped. Bug fixed post-run; N-BEATS + DLinear rerunning
separately.

### v3 head-to-head (3 of 5 models — N-BEATS/DLinear rerun pending)

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | Beats persistence? |
|---:|:---|---:|---:|---:|:---:|
| 1 | **XGBoost (Optuna-tuned)** | **1.143** | **1.461** | **0.364** | ✅ |
| 2 | LSTM | 1.475 | 1.861 | -0.033 | ❌ |
| 3 | TFT | 1.643 | 2.088 | -0.301 | ❌ |
| — | Persistence baseline | 1.440 | 1.832 | -0.002 | — |

**Optuna best hyperparameters** (very different profile from v2 defaults):
- `max_depth=4` (was 5), `learning_rate=0.0185` (was 0.05), `n_estimators=450` (was 200)
- `min_child_weight=16` (was 10), `subsample=0.63`, `colsample_bytree=0.65`
- `reg_alpha=0.002`, `reg_lambda=0.56`
- **Interpretation:** more trees, shallower, slower learning, weaker L1/L2 → a smoother,
  more conservative ensemble than my hand-picked v2 knobs.

### v3 two-stage power result (using tuned XGBoost as Stage 1)

| Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|---|---:|---:|---:|---:|---:|
| physics_only | 446,962 | 607,452 | -0.071 | 14.65% | 19.92% |
| **two_stage** | **355,771** | **478,077** | **0.337** | **11.66%** | **15.67%** |
| direct | 371,247 | 479,103 | 0.334 | 12.17% | 15.71% |

### What v3 tells us

- **Optuna tuning helped, but only marginally.** Test MAE 1.153 → 1.143 (−0.9%),
  Test R² 0.354 → 0.364. Two-stage nMAE 11.77% → 11.66%. Real but small. The pipeline
  is close to what a well-tuned XGBoost can extract from these features.
- **TFT lost to LSTM (1.643 vs 1.475).** Larger and more expressive is not automatically
  better — TFT has more parameters to fit with the same 30 epochs and no early stopping,
  so it likely overfit. Or attention isn't the right inductive bias here.
- **Neither deep model beat persistence at day-ahead.** Both LSTM and TFT under 30 epochs
  with default hidden dims are undertrained. Direction A from Iteration 2's option list
  (dedicated LSTM training fixes: 100+ epochs, early stopping, larger hidden, MAE loss)
  remains untested and is the obvious next lever.
- **XGBoost with rich, engineered features is a genuinely strong baseline** for
  medium-horizon wind forecasting. This is consistent with M5 and other time-series
  competition findings — GBMs with careful features beat vanilla sequence models
  reliably at these data scales.

### N-BEATS + DLinear (bug-fix rerun) — completed

Bug: `pipeline.py` only whitelisted `{lstm, tft}` in `DEEP_ALGOS`; N-BEATS and DLinear
branches in `stage1_deep.py` were unreachable. One-line fix pushed. Reran with
`compare_algos: [nbeats, dlinear]` (no Optuna re-sweep, no XGBoost/LSTM/TFT retrain —
already had those numbers). Total wall time ~15 min (N-BEATS ~12 min, DLinear ~2 min).

Results:
- **N-BEATS: Test MAE 1.425, R² 0.036** — narrowly beats persistence (0.015 m/s better).
- **DLinear: Test MAE 1.519, R² −0.074** — loses to persistence.

---

## v3 — Final consolidated head-to-head (all 5 models)

Combining the two runs into a single ranking, best → worst by test MAE:

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | Beats persistence? |
|---:|:---|---:|---:|---:|:---:|
| 1 | **XGBoost (Optuna-tuned)** | **1.143** | **1.461** | **0.364** | ✅ |
| 2 | N-BEATS | 1.425 | 1.798 | 0.036 | ✅ (barely) |
| — | Persistence baseline | 1.440 | 1.832 | -0.002 | — |
| 3 | LSTM | 1.475 | 1.861 | -0.033 | ❌ |
| 4 | DLinear | 1.519 | 1.897 | -0.074 | ❌ |
| 5 | TFT | 1.643 | 2.088 | -0.301 | ❌ |

### v3 two-stage power result (using XGBoost as Stage 1 — the actual winner)

| Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|---|---:|---:|---:|---:|---:|
| physics_only | 446,962 | 607,452 | -0.071 | 14.65% | 19.92% |
| **two_stage** | **355,771** | **478,077** | **0.337** | **11.66%** | **15.67%** |
| direct | 371,247 | 479,103 | 0.334 | 12.17% | 15.71% |

> **Caveat about `outputs/reports/report.md`:** the latest auto-generated report reflects
> the N-BEATS-only rerun and shows N-BEATS in the head-to-head. The XGBoost/LSTM/TFT rows
> from the earlier full run were **not** re-run and don't appear in the current on-disk
> report. This RESULTS.md file is the source of truth for the complete v3 comparison.

### What v3 tells us in aggregate

- **Only XGBoost and N-BEATS beat persistence.** The other three deep models (LSTM, TFT,
  DLinear) all lose to persistence at day-ahead with default hyperparameters. That's a
  meaningful signal — deep models need more than lucky-defaults to beat a strong tabular
  baseline on this data volume.
- **N-BEATS is the deep model that works.** With 3 stacks × 3 blocks × 256-wide layers,
  it barely edges out persistence (Δ MAE = 0.015 m/s). Interpretation: N-BEATS's
  hierarchical basis decomposition (trend + seasonality) is a good match for wind's
  strong diurnal + synoptic structure, and it doesn't over-parameterise on this
  ~28k-sample training set. But the margin is thin.
- **TFT was the biggest surprise (in a bad way).** Larger and more expressive lost to
  smaller LSTM. Likely cause: 30 epochs is way too few for a transformer to converge, and
  no early stopping meant no way to catch the underfit. TFT with proper training (100+
  epochs, LR scheduling, early stopping) is the natural next experiment.
- **Optuna tuning helped XGBoost only marginally** (test MAE 1.153 → 1.143 m/s, i.e. 0.010 m/s
  better; R² 0.354 → 0.364, where higher R² is better).
  This suggests the v2 hand-tuned hyperparameters were already near-optimal for the data,
  or that we're brushing against a data-limited floor.
- **Two-stage nMAE improved 11.77% → 11.66%** (small but consistent direction). Still
  well above the 8.8% Energies 2021 benchmark — ~~the remaining gap is likely
  data-limited (single 25 km grid cell, no real observations)~~. **Superseded by v5:**
  measured on 2026-08-10, the gap is *information*-limited, not sample-limited. Do not
  quote the struck text.
- **Full 5-model + tuned-XGBoost benchmark is the most defensible result to write up.**
  You can honestly claim: "Among an XGBoost baseline with 50-trial Optuna tuning and
  four state-of-the-art sequence models (LSTM, TFT, N-BEATS, DLinear), an Optuna-tuned
  XGBoost with rich exogenous + spatial features was the best day-ahead wind speed
  forecaster for Anloga. Only N-BEATS matched persistence's headline number. This
  suggests strong tabular baselines remain competitive with sequence models on modest
  data volumes for meteorological forecasting."

---

## v4 — the alignment fix, and what it cost us

Full analysis: [docs/10_v4_deep_model_retraining.md](docs/10_v4_deep_model_retraining.md).

**Trigger:** implementing the "what would actually help deep models" list at the
end of [docs/09](docs/09_why_deep_learning_underperformed.md).

**What we found instead:** every deep-model number in v2 and v3 was scored
**24 hours out of phase**. `historical_forecasts(last_points_only=True)` returns
a series indexed by *target* time τ, where `hist[τ]` predicts the wind speed
**at** τ. The feature frame is indexed by *feature* time t, where
`y[t] = hub(t+24)`. The old code did `hist.reindex(y_test.index)`, comparing the
model's estimate of `hub(t)` against `hub(t+24)`.

That error measures `|hub(t) − hub(t+24)|`, which *is* 24-hour persistence error.
So every deep model landed in a narrow band around persistence (1.425–1.643),
which looked like a meaningful ranking and supported a tidy narrative — while
actually re-measuring the same baseline four times with different noise on top.
Doc 09 then explained that artefact with five plausible mechanisms, none of which
was the cause.

**Changes shipped:**
- `_to_feature_time()` converts backtests from target time to feature time; the
  backtest `start` is offset so the first point lands on `y_test.index[0]`.
  Two regression tests guard it (unit + a 37-hour-period end-to-end test whose
  period is deliberately not commensurate with the 24 h horizon).
- Rec #1: validation tail + `EarlyStopping` on `val_loss` + **best-checkpoint
  restore** (without the restore, early stopping returns weights `patience`
  epochs past the best).
- Rec #2: lag/rolling columns as past covariates (`include_target_lags`).
- Rec #3: `WarmupCosineLR`, gradient clipping, short-budget LR sweep.
- Fixed: `joblib.dump` silently dropped darts model weights (`save_stage1` now
  uses darts' own serializer + a `.darts.pt` sidecar).
- Fixed: report hardcoded the config's algo name instead of the winner's.

### v4 head-to-head (all 5 models, correctly aligned)

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | v3 MAE |
|---:|:---|---:|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** | 1.475 |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 | 1.143 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 | 1.425 |
| 4 | DLinear | 1.215 | 1.530 | 0.302 | 1.519 |
| 5 | TFT (quantile — see v4b) | 1.551 | 1.974 | −0.163 | 1.643 |
| — | Climatology | 1.282 | 1.617 | 0.220 | |
| — | Persistence (25 h, as reported) | 1.440 | 1.832 | −0.002 | |
| — | Persistence (24 h, correct) | 1.399 | 1.791 | 0.043 | |

XGBoost is unchanged to three decimals — the tree path was untouched, so it acts
as the control confirming the deep-model movement is real.

### v4b — TFT on the same objective as everyone else

TFT was the only model still losing to persistence. Cause: `darts.TFTModel`
defaults to `likelihood=QuantileRegression()` over **17 quantiles**, while
BlockRNN, N-BEATS and DLinear all train on plain MSE. TFT was solving a much
harder problem on the same 28k samples, and its point forecast was only the
median. (This is also why its `val_loss` of 0.416 looked absurd next to the
others' ~0.008 — quantile loss and MSE are not the same scale.)

Fix: `model.deep.loss_fn: mse`. darts only substitutes `QuantileRegression()`
when *both* `likelihood` and `loss_fn` are `None`, so naming the loss makes TFT
deterministic. No-op for the other three.

| TFT objective | Test MAE | RMSE | R² | Beats persistence? |
|:---|---:|---:|---:|:---:|
| `QuantileRegression` (darts default) | 1.551 | 1.974 | −0.163 | ❌ |
| **`MSELoss` (deterministic)** | **1.171** | **1.498** | **0.330** | ✅ |

Both runs early-stopped at the same point (28 vs 29 of 60 epochs), so this is
not a budget effect. The objective mismatch accounts for the whole deficit.

### Final v4 standings — all five models, aligned, uniform MSE objective

| Rank | Model | Test MAE (m/s) | RMSE | R² |
|---:|:---|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 |
| 4 | TFT (MSE) | 1.171 | 1.498 | 0.330 |
| 5 | DLinear | 1.215 | 1.530 | 0.302 |
| — | Climatology | 1.282 | 1.617 | 0.220 |
| — | Persistence (24 h, correct) | 1.399 | 1.791 | 0.043 |
| — | Persistence (25 h, as reported) | 1.440 | 1.832 | −0.002 |

**Every model beats both persistence baselines**, and all five sit within
0.101 m/s of each other — a smaller spread than the gap from the weakest model
to persistence (0.184 m/s). In v3, four of five *lost* to persistence and the
spread was 0.50 m/s. The picture is completely inverted.

> **Caveat:** single seed, single chronological split. The LSTM's 0.029 m/s
> margin over XGBoost is inside plausible seed variance. The supported claim is
> "these architectures are near-equivalent and v3's gap was measurement error",
> not "LSTM beats XGBoost".

### What v4 tells us

- **The LSTM beats the Optuna-tuned XGBoost.** Reversing v3's headline finding.
  The margin (0.029 m/s) is small; the honest reading is that these architectures
  are near-equivalent and the earlier gap was measurement error.
- **"Train longer" was moot.** Every model early-stopped well below its cap
  (DLinear 23, LSTM 24, N-BEATS 19, TFT 28 epochs). v3's 30 epochs was already
  past convergence — what was missing was validation-based stopping, not budget.
- **"Feed the lag features" was neutral-to-harmful.** DLinear ablation: 1.215
  with lags vs **1.196** without. The sequence models already have the target
  history in the input chunk.
- **The LR sweep was worth it.** It picked a different LR for each model
  (3e-4 / 1e-3 / 3e-3 / 1e-3); a shared default would have been wrong for two.
- **TFT's failure had a concrete, different cause — now confirmed.** The default
  17-quantile objective, not the architecture. On MSE it goes 1.551 → **1.171**
  and clears persistence (v4b above).
- **Two-stage nMAE 11.66% → 11.25%**, R² 0.337 → 0.351. Best so far; gap to the
  Energies 2021 8.8% benchmark narrows to ~2.5 points.
- **Process lesson worth writing up:** a bug that moves every model toward the
  same plausible value is far more dangerous than one that produces obvious
  garbage. The v3 table was internally consistent, matched a well-known prior
  ("GBMs beat deep nets on small tabular data"), and was wrong. The tell was
  available in plain sight — four unrelated architectures clustering within
  0.2 m/s of persistence is not a result, it's a fingerprint.

---

## v5 — NWP forecast features (in progress, started 2026-08-10)

Full record: [docs/11_nwp_forecast_features.md](docs/11_nwp_forecast_features.md).

> ⚠️ **No v5 model score exists yet.** The ingest and plumbing are complete and tested;
> the ERA5 truth series for the new window is still downloading. Nothing below is an
> accuracy claim.

**Trigger.** Every iteration since v1.5 explained the residual gap to the Energies 2021
benchmark as "data-limited", and the 10-year ERA5 backfill was launched on that basis.
The claim had never been measured. Measured on 2026-08-10, it does not hold.

### The diagnosis (measured on the 2020–2023 parquet)

| Evidence | Result |
|:---|:---|
| Learning curve, 0.4 → 3.2 yr | 1.5017 → 1.1933 → 1.1534 → 1.1479 → **1.1410**; flat after ~1.6 yr, last doubling −0.012 m/s (1.1%) |
| Train/test MAE ratio at full data | **1.16** — no overfitting left, so more data has no variance to cure |
| 24 h autocorrelation of hub wind | **0.571** → R² ceiling **0.326** for any predictor built on the state at *t*; best model reaches 0.390 |
| OLS on `hub(t)` alone, 1 parameter | MAE **1.2388**, R² 0.267 — the whole 53-feature apparatus buys ~0.06 R² over this |
| Feature ablation | lag/roll 1.2564 → +calendar 1.1760 → +surface 1.1615 → +upper-air 1.1444 → +spatial **1.1410** (the 20 spatial columns are worth 0.003 m/s) |
| `mode="nowcast"` | MAE **0.015**, R² 0.9998 — near-tautological, but it localises the error: **100% of it is the 24-hour atmospheric gap** |

**Conclusion:** the constraint is *information*, not sample count. Every exogenous feature
is the analysed state at time *t*; there is no forecast anywhere in the input. The
10-year backfill was **stopped at 2016-07** and the effort redirected.

### ✅ AUTHORITATIVE RESULT (2026-08-11 23:18) — full 22-month ERA5 window

The ERA5 download completed at 66/66 files, so this is the run that goes in the write-up.
`python -m src.validate_nwp --config config_nwp.yaml` — **all four gates pass**.
Feature frame: **15,912 rows × 104 features**, 51 of them NWP.

| Predictor (test MAE, m/s — lower better) | MAE | R² |
|:---|---:|---:|
| climatology (train mean) | 1.3490 | −0.0711 |
| persistence `hub(t) → hub(t+24)` | 1.3097 | −0.0149 |
| OLS on `hub(t)` — strongest simple baseline | 1.1349 | 0.2225 |
| XGBoost **without** NWP (53 feats) | 1.0513 | 0.3300 |
| **XGBoost *with* NWP (104 feats)** | **0.7295** | **0.6653** |

**Headline: −30.6% MAE, R² 0.3300 → 0.6653.** The full pipeline run with Optuna tuning
lands at **0.738 m/s / R² 0.660** on its own holdout — consistent with the gate figure.

**Where the three estimates of the gain now sit.** The 12-month ERA5 subset gave +40.0%,
NASA POWER over 22 months gave +23.2%, and the authoritative full ERA5 window gives
**+30.6%**. The +40% was inflated by short-window baseline overfitting, as predicted; the
+23.2% is on a smoother truth source. **Quote +30.6%, or "roughly 30%".**

**A caveat that got weaker, and should be reported as such.** Earlier v5 notes warned that
the NWP window was "demonstrably calmer" than 2020–2023 — persistence 1.09 vs 1.399 m/s.
That was true of the 12-month subset. On the **full** 22 months persistence is **1.3097**
(pipeline's 25 h variant: 1.336), i.e. within 5% of v4's window. The problem is *not*
materially easier, and the benchmark comparison below no longer needs that hedge.

Gate 1 (alignment) passes with worst |peak lag| = 1 h. Gate 2: ICON is the single best raw
forecast here (rescaled MAE 0.9743 vs ECMWF 1.0410, GFS 1.0465) — note this **reverses the
NASA POWER ordering**, where ECMWF led. Neither margin is large; do not claim a model
ranking. Leakage test passed — skill decays with lead time.

#### End-to-end power — authoritative

| Variant | MAE (W) | R² | nMAE | nRMSE |
|:---|---:|---:|---:|---:|
| physics_only (no ML) | 420,250 | −0.071 | 13.78% | 18.30% |
| **two_stage** | **228,048** | **0.675** | **7.48%** | **10.08%** |
| direct | 226,728 | 0.690 | 7.43% | 9.85% |

**v4 was 11.25% / 15.50%.** The Energies 2021 reference is ~8.8% / ~13%.

⚠️ **What may now be said.** 7.48% nMAE sits below the 8.8% reference on a window whose
persistence baseline (1.31 m/s) is comparable to the literature setting, which is a
stronger position than the earlier pilot allowed. **Still not like-for-like** — different
site, different turbine assumptions, physics-derived truth rather than SCADA, and a
**single seed on a single chronological split**. The multi-seed repeat is the outstanding
item before any headline claim. `two_stage` and `direct` remain within 0.05 pp; that
ordering is not established.

### 🔬 Interim result (2026-08-11, superseded by the above) — 12-month ERA5 subset

Run on **real ERA5 truth** but only 12 of the 22 months (2024-03 → 2025-02, 8,760 h) while
the rest downloads. `python -m src.validate_nwp --config config_nwp_partial.yaml`.

| Predictor (test MAE, m/s — lower better) | MAE | R² |
|:---|---:|---:|
| XGBoost **without** NWP | 1.2695 | 0.2415 |
| 24 h persistence | 1.0928 | 0.3851 |
| OLS on `hub(t)` — strongest simple baseline | 1.0183 | 0.4628 |
| **XGBoost *with* NWP** | **0.7621** | **0.7173** |

**Confirmed independently on NASA POWER truth over the full 22 months** (after fixing its
two defects — see below). Different truth source, longer window, properly-trained
baseline:

| | ERA5, 12 months | NASA POWER, 22 months |
|:---|---:|---:|
| without NWP | 1.2695 m/s · R² 0.2415 | 0.8827 m/s · R² 0.4846 |
| **with NWP** | **0.7621 m/s · R² 0.7173** | **0.6780 m/s · R² 0.7024** |
| improvement | +40.0% | **+23.2%** |

**Headline: ~23% MAE reduction, R² from ~0.48 to ~0.70.** The two runs land on essentially
the same R² from different truth sources — and both confirm the 40% figure was inflated by
the ERA5 arm's short-window overfitting (train/test ratio 1.50 on 0.78 yr). ⚠️ The two
absolute MAE columns are not comparable: NASA POWER's is inflated ~12% by the 50 m → 100 m
extrapolation yet deflated by being coarser and hour-averaged. Only R² and relative gain
transfer.

The result is real: R² **0.7173** against a measured **0.326** ceiling for any predictor
using only the state at time *t*. Only genuine forecast information can break that ceiling,
which is exactly the v5 thesis. Leakage test passed — skill decays with lead time
(d1 0.7684 → d2 0.8504 → d3 0.8457) and the 3-model ensemble beats every single model.

⚠️ Not comparable to the v1–v4 tables: different window, and this one is calmer
(persistence 1.0928 vs 1.399 m/s). Full detail:
[docs/11 §7a](docs/11_nwp_forecast_features.md).

### End-to-end power with NWP — the headline metric

| Variant | ERA5 12 mo | NASA POWER 22 mo |
|:---|---:|---:|
| physics_only (no ML) | 11.21% nMAE / 15.03% nRMSE | 7.53% / 11.03% |
| **two_stage** | **7.66% / 10.55%** | 4.43% / 6.38% |
| direct | 7.49% / 9.94% | 5.29% / 7.86% |

**v4 was 11.25% / 15.50%.** The Energies 2021 reference is ~8.8% / ~13%.

⚠️ **Do not claim the benchmark is beaten.** Different site, a demonstrably calmer window
(persistence 1.09 vs 1.399 m/s), 12 months, single seed. Say *"comparable to or better
than the published reference, though not like-for-like"*. Note too that `two_stage` and
`direct` swap order between the two truth sources — neither margin is established.

### 📅 Weekly forecast — 24 h to 168 h (2026-08-12)

The project's first multi-horizon work. Seven models, one per forecast lead day, on a
**valid-time** framing: for lead day *N* the row is `(nwp_*_d{N}[τ], calendar(τ), spread(τ))
→ hub(τ)`. Full detail, including limitations:
[redo/results/09](redo/results/09_weekly_forecast_results.md).

**A 7-day forecast beats climatology by 20.7%.**

| Lead | Horizon | Members | Climatology | `mos_multi` | **XGBoost** | Skill vs clim |
|---:|---:|---:|---:|---:|---:|---:|
| d1 | 24 h | 3 | 1.2065 | 0.7733 | **0.7322** | **0.393** |
| d2 | 48 h | 3 | 1.2075 | 0.8606 | **0.8251** | 0.317 |
| d3 | 72 h | 3 | 1.2087 | 0.9098 | **0.8729** | 0.278 |
| d4 | 96 h | 3 | 1.2085 | 0.9355 | **0.9166** | 0.242 |
| d5 | 120 h | 3 | 1.2088 | 0.9289 | **0.9147** | 0.243 |
| d6 | 144 h | 3 | 1.2085 | 0.9609 | **0.9354** | 0.226 |
| **d7** | **168 h** | **2** | 1.2075 | 0.9802 | **0.9580** | **0.207** |

⚠️ **Day 7 runs on a two-member ensemble** — ICON's archive stops at d6 (confirmed: zero
non-null hours at d7 across all 16,104). Never place a d7 figure beside a three-member one
unlabelled.

**Weekly energy is the strongest product.** Summing the hourly forecasts through the power
curve over non-overlapping weeks cancels ~2.6× of the hourly error:

| Lead | Hourly nMAE (% rated) | **Weekly nMAE (% rated)** | Weekly MAPE (% of actual) |
|---:|---:|---:|---:|
| d1 | 7.73% | **2.24%** | 9.83% |
| **d7** | **10.32%** | **3.95%** | 18.22% |

*Two normalisations because they are not interchangeable: % of rated is comparable to the
hourly figure and shows the cancellation; % of actual has a denominator ~a quarter the size
(measured capacity factor **0.236–0.240**) and is what an operator feels. Reporting only the
latter made the aggregate look worse than the hourly forecast when it is ~2.6× better.*

**Three findings and one failure:**

1. **The classical/ML gap narrows with lead** — `mos_multi` trails XGBoost by 5.6% at 24 h
   and only **2.3% at 168 h**. The further ahead you forecast, the less machine learning
   adds. At weekly range this is very nearly a linear-regression problem.
2. **By day 7, raw ensemble averaging beats single-model MOS** (1.0418 vs 1.0578) —
   multi-model combination outweighs statistical correction at that range.
3. **Dropping an ensemble member costs nothing.** Scoring d6 with and without ICON:
   0.9354 → **0.9343**, coverage 0.705 → 0.719. This **refuted a documented prior** that
   144 h with three members would beat 168 h with two. Take 168 h.
4. ❌ **The quantiles are not calibrated.** P10–P90 coverage is **0.65–0.74 against 0.80
   nominal** at every lead — bands systematically too narrow. Reported as a failure rather
   than widened, because an interval tuned until it covers is fitted to the test set.
   Interval width does grow correctly with lead (1.81 → 2.70), so the spread feature tracks
   real uncertainty; the model is under-dispersed.

⚠️ **Single split, seed 42 — not the 4-fold × 5-seed protocol.** Treat differences under ~2%
as unresolved, **including the ICON result**. The energy aggregate is scored on the test
fold's **18–19 weeks**, not the ~95 in the full window.

### 🗓️ Monthly scale — climatology and a 30-day outlook (2026-08-12)

⚠️ **Different units and a different sample.** Errors below are **capacity-factor points over
calendar-month aggregates**, not m/s over hours. Nothing in this section is comparable to any
other figure in this log. Detail: [`redo/results/10`](redo/results/10_monthly_ceiling.md) and
[`11`](redo/results/11_monthly_results.md).

**The request was "a monthly model"; the measurement said that cannot mean a 30-day hourly
forecast.** NWP previous-runs stop at lead day 7 (probed live), and at monthly scale:

| Quantity | Value (n = **156 months**, 2010-01→2026-07) |
|:---|---:|
| Seasonal cycle share of monthly-CF variance | **81.6%** |
| Interannual anomaly share | **18.4%** |
| Lag-1 anomaly autocorrelation | **0.229** (gone by lag 2: 0.101) |
| Leave-one-year-out climatology MAE | **0.0402 CF** (4.02 pp of rated) |
| Anomaly persistence MAE | **0.0447** — *worse than climatology* |

So "monthly" was split into a **resource climatology** (which works) and a **30-day outlook**
(which does not, measurably).

**Ceiling first.** A monthly total can only use NWP for its first ~7 days. Granting that head
a *perfect* forecast caps skill at **13.1%** over climatology (7 days), rising to 30.7% at 14
and 51.5% at 21 — the first week is the least valuable week to know.

| Arm | n | MAE (CF) | Skill | 95% CI | Verdict |
|:---|---:|---:|---:|:---|:---|
| Climatology (LOYO) | 22 | 0.0346 | — | — | baseline |
| **30-day outlook** | 22 | 0.0342 | **+1.1%** | **[−19.7%, +17.7%]** | ❌ **no demonstrable skill** |
| Perfect head (ceiling here) | 22 | 0.0262 | +24.2% | [−0.0%, +44.1%] | not significant |

**Findings:**

1. ❌ **The 30-day outlook has no demonstrable skill.** It beat climatology in **12 of 22
   months** — a coin flip. Reported as a null result. This is a **sample-size problem, not a
   modelling one**: Open-Meteo previous-runs begin 2024-03, so n grows one month per month,
   and resolving a true ~5% effect needs 100+ months.
2. ⚠️ **A 22-month estimate is not stable.** Growing the sample 20 → 22 months during this
   work moved the outlook's headline **+6.9% → +1.1%** and flipped the ceiling arm from
   significant to not. Only the interval makes that visible.
3. ✅ **The obvious P90 estimator was wrong, in the optimistic direction.** The empirical 10th
   percentile is exceeded **84.6%** of the time against 90% nominal — with 12–14 years per
   calendar month it moves a whole rank when one year is held out. A fitted-normal tail lands
   at **89.7%**, within 0.3 points. `exceedance_table` now defaults to the normal fit and
   scores both on every run.
4. 🔆 **Wind and solar are close to anti-correlated on the daily cycle here.** Wind troughs
   08:00–11:00 UTC and peaks 16:00–21:00, every month. The diurnal swing (up to 0.207 CF)
   rivals the seasonal one (0.289) — the most operationally useful fact in the monthly work,
   and it comes from the climatology, not a forecast.
5. 📉 **Interannual spread is large:** across 11 complete years, annual energy runs
   **6,246–7,691 MWh** (CF 0.234–0.288). The best year beats the worst by 23%.

**Capacity-factor reconciliation (a defect this closed).** This project was carrying two
capacity factors: **0.2606** here and **0.236–0.240** in the weekly work. Both are correct for
what they measure and neither was labelled. The weekly figure is its **test fold**
(2025-08-19→12-31, a low-wind stretch); it *varies across leads* because each lead's frame
survives `dropna` with a different row count, shifting the fold start — `n_test` 3164–3194
correlates **+0.987** with the quoted capacity factor. Density correction accounts for a
further +4.1%. Any capacity factor quoted from here on carries its period.

### 🔁 The 30-day outlook, re-powered (2026-08-12) — and two findings that reach backwards

The monthly outlook above returned "no demonstrable skill" and was attributed to sample size.
That was incomplete. The design was **diluting** the signal: a 7-day forecast is 23% of a
30-day month, and the other 77% is climatological tail error *identical in both arms* which,
being absolute, inflates variance without cancelling.

**Its minimum detectable effect was 26.1% against a 13.1% ceiling — it could not have resolved
even a perfect forecast.** The null was guaranteed by the design, not the physics.

Fix: score the head where it *is* the prediction — non-overlapping 7-day blocks, forecasts
issued weekly so they tile the era. n = 22 → **114**, effect 4.3× larger, ~10× the power. MDE
was **pre-registered at 8.2% before any result was computed**. Detail:
[`redo/results/12`](redo/results/12_monthly_outlook_power.md).

| Arm | Block MAE (CF) | Skill vs archive clim | DM p |
|:---|---:|---:|---:|
| `climatology_archive` *(reference)* | 0.0728 | — | — |
| `climatology_window` *(what 09 used)* | 0.1110 | **−52.5%** | 1.4×10⁻⁵ |
| `ensemble_mean` | 0.0918 | **−26.1%** | 1.9×10⁻³ |
| `mos_single` | 0.0637 | +12.5% | 0.110 *(ns)* |
| `mos_multi` | 0.0505 | +30.7% | 2.8×10⁻⁴ |
| **`xgboost`** | **0.0441** | **+39.4%** | **1.1×10⁻⁵** |

1. ⚠️ **The baseline outweighed every modelling decision.** Scoring against a ≤2-year
   training-window climatology instead of the 14-year archive is worth **+52.5%** at block
   level and +14.8% hourly — a 21-point swing in XGBoost's headline (39.4% → 60.3%), larger
   than the gap between the best and worst candidate. **The archive is operationally
   available**: ERA5's 5-day latency is irrelevant to a climatological mean.
2. ⚠️ **The winner depends on the objective.** In capacity factor, `mos_multi` beats
   `xgboost` at **5 of 7 leads** — reversing the 7-of-7 result in
   [`redo/results/09`](redo/results/09_weekly_forecast_results.md), which scored **m/s**. The
   power curve is nonlinear, so the model minimising wind-speed error is not the one
   minimising energy error. **The project has been selecting on m/s while the deliverable is
   energy.**
3. **The raw ensemble mean is 98% bias** (−0.0902 CF of a 0.0918 MAE). It beats climatology at
   every lead hourly and loses by 26.1% on weekly aggregates, because a 168-hour sum preserves
   bias while hourly MAE forgives it. This is the case for MOS stated as a measurement.
4. **The monthly product improves but still cannot be called significant:** +1.1%
   [−19.7%, +17.7%] → **+4.4% [−10.5%, +19.1%]**, interval 21% narrower, n 22 → 26. The 13.1%
   ceiling never moved. **A 39.4% gain on the forecastable quarter becomes ~4% on the month.**
   The re-powering did not rescue the monthly outlook — it explained it.

**Both climatology arms and both objectives are reported side by side** rather than the
favourable one being quoted. The gate that caught the tiling bug (86 of 19,488 hours never
predicted, traced to Open-Meteo holes rather than an off-by-one) is in the doc.

### 🏁 Classical vs ML bakeoff (2026-08-12) — one protocol, eleven predictors

The first comparison in this project run on a **defensible protocol**: 4 rolling-origin
folds so every season appears in test (the old single split covered 2025-08→12 only),
5 seeds for stochastic models, and **Diebold–Mariano** significance tests. Full table:
[`outputs/reports/bakeoff.md`](outputs/reports/bakeoff.md).

| Rank | Tier | Model | MAE (m/s) | sd | R² | Skill vs persistence |
|---:|:---|:---|---:|---:|---:|---:|
| 1 | ML | **xgboost** | **0.8037** | 0.102 | 0.589 | +0.386 |
| 2 | ML | lightgbm | 0.8086 | 0.103 | 0.583 | +0.382 |
| 3 | **classical** | **mos_multi** | **0.8412** | 0.119 | 0.551 | +0.355 |
| 4 | ML | ridge | 0.8422 | 0.113 | 0.560 | +0.354 |
| 5 | classical | kalman_mos | 0.8572 | 0.149 | 0.536 | +0.349 |
| 6 | classical | analog_ensemble | 0.8662 | 0.151 | 0.524 | +0.340 |
| 7 | classical | mos_single | 0.9904 | 0.109 | 0.388 | +0.239 |
| 8 | reference | damped_persistence | 1.1861 | 0.145 | 0.111 | +0.094 |
| 9 | classical | sarimax_nwp | 1.2554 | 0.268 | 0.083 | +0.022 |
| 10 | reference | climatology | 1.2725 | 0.096 | 0.003 | +0.021 |
| 11 | reference | persistence | 1.3183 | 0.225 | −0.108 | 0.000 |

**Three findings, in order of importance.**

**1. A 21-parameter linear model gets within 4.7% of gradient boosting.** `mos_multi` —
ordinary least squares on the day-1 forecast columns plus four calendar terms — scores
0.8412 against XGBoost's 0.8037. Diebold–Mariano says the gap is real (p < 0.0001), but it
is *small*, and it means **most of the ~30% headline gain in v5 came from the information,
not from the machine learning**. That is a stronger claim than the original, not a weaker
one: the finding is about day-ahead NWP forecasts being informative at this site, and it
survives being stated without any ML at all.

**2. Dropping every reanalysis feature costs nothing — the model is deployable.**

| Feature set | XGBoost MAE | Features |
|:---|---:|---:|
| `study` (all) | 0.8037 | 94 |
| **`operational`** (NWP + calendar only) | **0.8009** | 55 |

ERA5 publishes with ~5 days of latency, so its 49 derived features cannot exist when a
day-ahead forecast would actually be issued. Removing them **slightly improves** the score.
The deployability concern that motivated [`redo/08`](redo/08_stage8_deployment_rehearsal.md)
is real in principle but **costs nothing in practice here** — an operational system loses
no accuracy. This is measured, not argued.

**3. ARIMA-family methods add almost nothing at 24 h.** `sarimax_nwp` scores 1.2554 —
skill of just +0.022 over persistence — exactly as the autocorrelation ceiling predicts
(24 h autocorrelation 0.571 caps R² at 0.326 for state-based predictors). Its AR terms
decay within a few steps of a 2,200-step-ahead forecast, leaving essentially its exogenous
regression, which is a worse-specified MOS.

⚠️ **Absolute MAEs here are higher than the 0.7295 headline** because rolling-origin folds
include earlier, harder periods with less training data. Compare *within* this table only;
the persistence column is the anchor. The harness reproduces the published persistence
figure (1.3097) exactly on the original single split.

⚠️ **Deep models are not in this table.** They are on a different evaluation basis (single
split, 3 seeds) and are reported separately.

#### Phase B complete (2026-08-12, 12.9 h) — all five architectures × 3 seeds

⚠️ **Single chronological split, `study` features — not the 4-fold protocol above.** XGBoost
is re-run on the same split so the deep scores have a like-for-like anchor. Detail:
[`redo/results/04`](redo/results/04_deep_models.md).

| Model | Mean MAE (m/s) | Seed sd | Range | Min/fit | Seed sensitivity |
|:---|---:|---:|---:|---:|---:|
| **`xgboost`** *(anchor)* | **0.7329** | **0.0035** | 0.0068 | 0.1 | 1× |
| `lstm` | 0.7701 | 0.0276 | 0.0505 | 4.8 | 8× |
| `nbeats` | 0.8234 | 0.0361 | 0.0705 | 20.7 | 10× |
| `dlinear` | 0.8244 | 0.0115 | 0.0215 | 1.1 | 3× |
| **`tft`** | **0.8826** | **0.0614** | **0.1090** | **231.1** | **18×** |

1. **TFT finished last** at ~**2,000× XGBoost's training cost** (3 h 51 m vs 7 s per fit) to
   land **20% worse**. It is also the most seed-sensitive model measured: seeds 42 and 7 agree
   to 0.0056, and seed 1234 falls 0.1090 away.
2. **This inverts the NASA POWER result below**, which had TFT as the *best* deep model on a
   single seed. That run got both the gap size (18% → **5.1%**) and the runner-up's identity
   wrong — and it picked its winner from the noisiest candidate available.
3. **Trees still win, by 5.1% not 18%.** Wherever the 18% figure appears, this belongs with it.

### Five-model bakeoff with NWP (2026-08-11) — the v4 near-tie breaks

⚠️ **NASA POWER truth, 22 months — a cross-source check, not a v5 result.** These MAEs are
not comparable to any ERA5 figure above and must not be read alongside one.

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² |
|---:|:---|---:|---:|---:|
| 1 | **xgboost** | **0.678** | 0.869 | 0.699 |
| 2 | tft | 0.827 | 1.024 | 0.582 |
| 3 | lstm | 0.903 | 1.089 | 0.527 |
| 4 | nbeats | 0.942 | 1.141 | 0.481 |
| 5 | dlinear | 0.969 | 1.158 | 0.466 |
| — | *persistence* | *1.110* | *1.442* | *0.171* |

All five beat persistence, but **the spread is now 0.291 m/s — about 3× the 0.101 m/s of
v4**, and XGBoost leads the best deep model by 0.149 m/s (18%). On state-only features the
five architectures were a tie; with a strong forecast feature they are not.

**The direction was unexpected** — doc 12 anticipated sequence models might exploit a
forecast *trajectory* better than trees. Every deep model instead lost ground.

Three artefact explanations were checked and all fail: the two code paths use the **same
split** (both `int(n*(1-test_split))`; 3183 test points either way), the **same alignment**
(persistence 1.1100 m/s on both), and XGBoost scores **0.6780 under both the v3 Optuna
params and the untuned defaults**. A mechanism — that a pointwise forecast column suits a
tree's splits better than an encoded sequence window — is a **hypothesis, not a finding**.
Needs an ERA5 repeat and multiple seeds before it is claimed.
[docs/11 §7a](docs/11_nwp_forecast_features.md).

### Two bugs found and fixed on the way

1. **NASA POWER hub wind was understated by 10.7%.** Its 50 m wind sat in a column named
   `wind_speed_100m` and the hub extrapolation treated it as already at 100 m, so the log
   law no-opped. **v1's results carry this error.** Fixed via `SOURCE_WIND_HEIGHT_M`;
   ERA5 baseline re-verified unchanged at exactly 1.1410 m/s.
2. **NASA POWER has a ~1-hour timestamp offset vs ERA5.** MERRA-2 hourly values are
   hour-averaged and labelled at the end of the interval. Measured over 8,760 overlapping
   hours: ERA5(t) matches NASA(t+1), and all three NWP models peak exactly +1 h against
   NASA versus ERA5. Corrected via `NASA_POWER_TIME_OFFSET_H = -1`; the alignment peak
   moved from −1 h to 0 h, and the gate that had correctly failed now passes.

Both fixes were found by the alignment gate refusing to pass — it did exactly the job it
was built for.

### What shipped

- Open-Meteo **Previous Runs** ingest (`load_nwp_forecasts`), three models
  (`ecmwf_ifs025`, `gfs_global`, `icon_global`) × 6 variables × 3 lead days.
  **20,448 rows × 54 columns**, 2024-03 → 2026-06, no gaps, ~2.6 MB of JSON.
- `features.add_nwp_features` with an explicit valid-time → feature-time alignment
  contract, plus `nwp_df` threaded through `pipeline.py` and `evaluate.py`.
- `config_nwp.yaml`; **47 tests green** (was 38), including a perfect-forecast alignment
  assertion on a 37-hour signal period that no shifted frame can pass.
- Rejected the GFS-GRIB route on measurement: every GRIB message is a *global* 0.25° field
  (~1 MB), making an hourly multi-year archive 30–400 GB versus 2.6 MB.

### Two constraints on any future v5 claim

1. **The window moves to 2024-03 → 2025-12** (the Open-Meteo archive is null before
   2024-03; the end was shortened from 2026-06 on 2026-08-11 to cut ~8 h of ERA5
   download time, at a measured cost of ~0.004 m/s). v1–v4 numbers are on 2020–2023 and
   are **not** comparable — baselines must be re-run on the new window first.
2. **Do not use Open-Meteo's Historical Forecast API.** It stitches the first hours of
   each run and is analysis-like; as a t+24 feature it is leakage.

---

## v6 — does it generalise? 17 Ghanaian sites (2026-10-09)

⚠️ **Window 2024-03-01 → 2025-12-31, so v6 is comparable to v5 and *not* to v1–v4** (those are
on 2020–2023). Pre-registered before any run in
[`docs/13_multi_site_extension.md`](docs/13_multi_site_extension.md); full result in
[`redo/results/13`](redo/results/13_multi_site_generalisation.md).

Every headline in v1–v5 came from one coastal site. v6 runs the **published 24 h ladder,
unchanged** — same 4 rolling folds, 5 seeds, DM test, report formatter — at Ghana's **16
regional capitals plus Anloga**, off one shared Ghana-wide ERA5 bounding box.

| | Result |
|:---|:---|
| **Sanity floor** | ✅ **34/34 site-arms** (17 sites × 2 feature sets). Every site's best model beats *both* persistence and climatology on its own folds; weakest margin **+24.6%** (damongo, `operational`) |
| **MAE range** | 0.5453 (accra) → 0.9031 (wa), a factor of **1.66** — `study` arm |
| **R² range** | **0.537 → 0.634** — under 0.1 across ~800 km and three climatic zones |
| **Skill vs persistence** | `study` **0.271 → 0.388**; both arms 0.246 → 0.388, median 0.314 |
| **Classical–ML gap** | `mos_multi` within **7.46%** of xgboost at worst; median 5.16% |
| **Best model** | xgboost at 11 sites, **ridge at 6** |
| **Site failures** | 0 |

**The stability is the finding, not any individual MAE.**

### ⚠️ Raw MAE must not be ranked across these sites

Most of the 1.66× MAE spread is how windy a site is, not how predictable. Normalised by mean
wind the ranking **inverts at the top** — Anloga has the second-worst raw MAE (0.8006) and the
**best** normalised error (**0.128** vs 0.211 worst), because its mean wind is 6.24 m/s against
3.3–4.4 elsewhere, and its ERA5 cell is 88% water
([`docs/13` §3a](docs/13_multi_site_extension.md)). A bare MAE ranking reads as "worst at
Anloga", the opposite of the truth.

### Thesis §4.9.3 gets a split verdict

§4.9.3 predicted reanalysis features would matter more where NWP skill is weaker. Dropping
every latency-blocked reanalysis feature:

- **15 of 17 sites show a positive penalty** against Anloga's +0.03%; largest **damongo
  +3.45%** and **tamale +2.55%** — the northern savannah, exactly where §4.9.3 pointed. Sign
  test over the 16 non-zero sites, **p ≈ 0.0003**.
- **0 of 17 penalties exceed that site's own seed sd.**

So the direction is solid and no single magnitude clears noise. **The pre-registration never
quantified "materially more"** — a defect in the criterion's wording, recorded rather than
resolved in whichever direction suits the outcome. v5's deployability claim survives in the
broad form: the penalty is under 1.3% at 15 of 17 sites.

### Exploratory, not pre-registered

- **`ridge` beats both gradient-boosting models at 6 of 17 sites**, including 4 of the 5
  far-northern ones. The same region carries the largest reanalysis penalties *and* the
  narrowest MOS gaps — three signals that the north favours simpler, more linear models.
- **Climatology beats persistence at 13 of 17 sites; the 4 exceptions are the 4 northernmost.**

### Not measured

**Resource and viability are unmeasured.** The long archive the monthly climatology needs is at
**144 of 576 files** — no capacity factor exists for any of these sites and none should be
quoted. Deep models were excluded by pre-registration (12.9 h × 17 ≈ 206 h vs a 5.1% deficit).
Truth is ERA5 everywhere, so this measures agreement with a reanalysis, not with a mast.

### Two defects the v6 build surfaced

1. **A training run could silently start a 432-request CDS download.** `--start` defaulted to
   the archive range on *every* `src.multisite` command, and `site_config` resolves
   `start or config.data.start`, so the template's window was unreachable and
   `fetch_regional_archive` downloads what is missing. Fixed two ways — `resolve_window`, and
   `fetch_regional_archive(download=False)` behind a hard preflight (`b919157`).
2. **`--out` moved the report but overwrote the CSV in place**, so a side-experiment replaced
   two sites' headline CSVs while leaving their reports untouched — a mismatched pair under one
   name. Caught by an mtime that did not match its own report; both sites re-run and all 17
   verified report-to-CSV (`1caf2a8`).

---

## Consistent takeaways across iterations

1. **The two-stage architecture beats physics-only** whenever Stage 1 beats persistence
   — validating the "ML wind speed → power curve" design choice.
2. **Persistence collapses at 24h horizon** — the "tomorrow = right now" baseline goes from
   MAE 0.87 m/s at a 6-hour horizon to 1.44 m/s at 24 hours, i.e. it gets much worse as the
   forecast reaches further ahead — that's where ML earns
   its keep. At 6h persistence is very hard to beat with reanalysis data alone.
3. **Feature story is physically coherent** across iterations: near-surface wind speed
   dominates, then temperature (density effect), diurnal calendar features, and — from v2 —
   upper-air winds and boundary-layer height.
4. **Overfit is controllable** with mild regularisation + trimmed feature set —
   train/test gap stays near 1.2× once tuned.
5. **The gap to published benchmarks (~8.8% nMAE)** is stable at ~3 percentage points.
   Likely floor for a single 25 km grid cell in a data-sparse region. Closing it
   probably needs downscaled data or real met-mast observations.

## Data & pipeline notes worth preserving

- **CDS request chunking is essential** — monthly requests only. Multi-year single
  requests trip the 121k cost cap. `data_ingest._retrieve_with_retry` handles
  transient CDS-side job failures with 3 attempts × 60s backoff.
- **CDS returns `.zip` for accumulated variables** (like 10m wind gust). Loader detects
  ZIP magic bytes and unpacks transparently. See `data_ingest._open_dataset`.
- **ERA5 v2 caches use three prefixes** to avoid re-downloading v1 data:
  `era5_...` (core 7 vars), `era5x_...` (extras 3 vars), `era5pl_...` (pressure levels
  3 vars × 2 levels). All in `data/raw/`.
- **Cache coverage as of 2026-08-10** (file counts drift while a download runs — measure,
  don't quote): 2010-01 → 2016-07 and 2020-01 → 2023-12 complete across all three
  families; 2016-08 → 2019-12 missing (backfill deliberately halted, see v5 above);
  2024-03 → 2025-12 downloading. Only the 2020–2023 parquet is built, and **every result
  above rests on it**. Current inventory: [docs/00_data_card.md §2](docs/00_data_card.md).
- **Never run two ingest processes concurrently.** `target.exists()` is the only
  re-download guard, so two runs pick the same missing month and can write the same path
  at once. This happened on 2026-08-10; check with
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` before launching.
- **The ERA5 grid patch is 1×2, not 3×3** (`pad = 0.2°` snaps to `latitude=[5.75]`,
  `longitude=[0.75, 1.00]`). Consequence: all five `_grad_ns` columns are identically
  zero and all five `_std` columns are exactly `|_grad_ew| / 2`. Harmless to trees, but
  10 of the 36 tidy columns carry no independent information.
- **`geopot_*` is geopotential in m² s⁻², not geopotential height in m** — divide by
  9.80665 for metres.

## Software stack

- Python 3.13, XGBoost 3.4, scikit-learn 1.7, windpowerlib 0.2.2, darts 0.46 (torch
  backend), Optuna 4.9, xarray, cdsapi.
- 32 tests green as of v3 code freeze (before running the v3 comparison).
