# 04 — Stage 1: Wind-Speed Forecasting

Stage 1 is the ML component of the pipeline. It takes the tidy hourly weather
DataFrame from ingestion and predicts hub-height wind speed at `t + horizon_hours`.
This document covers feature engineering, baselines, the tree-model track
(XGBoost + Optuna), the deep-model track (LSTM/TFT/N-BEATS/DLinear via darts), and
the feature-influence analysis (Pearson + SHAP).

> 📐 **Reading the numbers.** Unless stated otherwise, a figure like `1.114` is a **test
> MAE in m/s** — how far off the model typically is when predicting hub-height wind speed
> 24 hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399, guessing the average 1.466. R² is the opposite direction
> (higher is better, 1.0 perfect, 0 = no better than guessing the average). Full
> explanation: [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** Stage 1 in full — how raw weather becomes model inputs, which
> models were tried, and how they were scored.
> **Why read it:** this is where all the machine learning lives. Stage 2
> ([05](05_stage2_and_evaluation.md)) is pure physics with nothing learned. If a term is
> unfamiliar, [glossary.md](glossary.md) defines it.

## Modes: nowcast vs forecast

> **What:** two settings — predict wind *now* from current weather, or predict it *24
> hours ahead*.
> **Why both exist:** forecast mode is the real task. Nowcast mode is a diagnostic: it
> scores almost perfectly (error 0.015 m/s), which proves that converting weather into
> hub-height wind is essentially free. All the difficulty lives in the 24-hour gap, not
> in the conversion — a useful thing to be able to demonstrate rather than assert.

The pipeline supports two operating modes, kept strictly separate:

- **`nowcast`** — target is hub wind speed at the *same* timestep. This is spatial
  downscaling from reanalysis (learn corrections to ERA5 at the site). Used for
  smoke tests where signal is trivial to learn.
- **`forecast`** — target is hub wind speed at `t + horizon_hours` (default 24 h).
  This is the actual operational use case: predict tomorrow's wind from today's
  observations.

All production results in this project use `mode: forecast` with `horizon_hours: 24`.

## Feature engineering

> **What:** turning the raw weather table into the columns a model can actually learn
> from — six transformations, described one per subsection below.
> **Why it matters:** models cannot use raw data directly in useful form. They need wind
> at the right height, directions expressed so that 359° and 1° are recognisably close,
> a sense of time of day, and some recent history. Each step below adds one of those.

Implemented in [`src/features.py`](../src/features.py). The `build_feature_frame`
orchestrator composes several helper functions in a fixed order:

### 1. Hub-height extrapolation

> **What:** converting wind speed at the heights the data provides to wind speed at the
> turbine's 100 m hub.
> **Why:** wind gets faster with height, and the turbine only cares about the height its
> rotor occupies. The conversion uses a standard logarithmic profile, which needs the
> surface roughness — rougher ground slows the lowest air more, making the increase with
> height steeper.

The reference wind measurements come at 10 m (ERA5) or 50 m (NASA POWER), but the
turbine hub sits at 100 m. `extrapolate_hub_wind` uses `windpowerlib`'s logarithmic
profile:

```
u(h) = u_ref × ln(h / z0) / ln(h_ref / z0)
```

where `z0` is the surface roughness length (from ERA5's `fsr` field, or a default
of 0.15 m — a common value for coastal/mixed terrain — when unavailable). The
result is stored as a new `wind_speed_hub` column.

The `hellman` power-law is available as a fallback but not used by default.

### 2. Direction encodings

> **What:** replacing a compass bearing (0–360°) with a pair of sine and cosine values.
> **Why it is necessary:** to a model, 359 and 1 look 358 apart, when in reality they are
> 2° apart — the number wraps around and the model cannot know that. Converting to sine
> and cosine puts the direction on a circle, so values either side of north end up close
> together as they should.

Wind direction is cyclic — 359° and 1° are almost the same physical value but
numerically very different. Feeding raw degrees to an ML model creates a false
discontinuity at 0°. Instead, encode as `(sin, cos)` pairs:

```
dir_sin = sin(dir × π / 180)
dir_cos = cos(dir × π / 180)
```

Applied to four columns when present: `wind_dir_10m`, `wind_dir_100m`,
`wind_dir_850hpa`, `wind_dir_500hpa`. The original degree columns are then
excluded from features (kept only for reporting).

Test: `test_add_direction_encodings` verifies sin² + cos² = 1 to machine precision.

### 3. Calendar features

> **What:** telling the model what time of day and time of year it is — again as sine and
> cosine pairs, for the same wrap-around reason as directions.
> **Why they matter so much here:** Anloga sits on the coast, where land and sea heat at
> different rates and drive a strong, repeating daily sea breeze. That rhythm is highly
> predictable, so the clock alone is genuinely informative. The measured ablation bears
> this out: adding calendar features was the single largest improvement of any feature
> group.

Six features encoding daily and seasonal cycles:

```
hour                     ∈ {0, 1, ..., 23}
month                    ∈ {1, 2, ..., 12}
hour_sin  = sin(2π × hour / 24)
hour_cos  = cos(2π × hour / 24)
month_sin = sin(2π × (month - 1) / 12)
month_cos = cos(2π × (month - 1) / 12)
```

Both raw (hour, month) and sin/cos encodings are computed. Tree models can use
either; the sin/cos pair encodes cyclicity for models that can't discover it (like
linear models and deep sequence models). The raw hour and month columns are
excluded from features by name — only the sin/cos versions feed the ML.

### 4. Lag features

> **What:** columns holding the wind speed 1, 2, 3, 6 and 12 hours ago.
> **Why:** weather has momentum — what happened recently constrains what happens next.
> **Why the list is short:** longer lags were removed because they encouraged the model to
> memorise rather than generalise. At a 24-hour horizon they help less than intuition
> suggests, since the atmosphere has substantially "forgotten" its state by then.

For each lag `k ∈ config.model.lags`, create `wind_speed_hub_lag{k}` = `wind_speed_hub.shift(k)`.

**Current setting: `lags = [1, 2, 3, 6, 12]`** (dropped 24 in v1.5 to reduce
overfit; the SHAP ranking showed lag24 wasn't earning its keep).

Lag features are the single most important predictors at short horizons — for a 24
h forecast, the wind speed 12 h ago carries substantial autocorrelation, and even
1–3 h ago (from t-1 relative to the forecast target at t + 24) helps set the
current atmospheric regime.

**Deep models (v4+):** the deep-model trainer in
[`src/stage1_deep.py`](../src/stage1_deep.py) now *includes* lag and rolling
columns as past covariates by default (`model.deep.include_target_lags: true`).
They are strictly backward-looking — `add_rolling_features` shifts by 1 before
rolling — and sit inside the input chunk, so they cannot leak.

Through v3 they were deliberately excluded, on the reasoning that a sequence model
already sees past target values in its input series. Doc 09 then argued the
opposite: that XGBoost gets an unfair edge because it can place an explicit split
on a *labelled* `lag12` column while an RNN must learn the same operation through
its recurrent weights. Measured directly on DLinear, **neither framing matters** —
including the lags is marginally worse:

| DLinear | Covariates | Test MAE | R² |
|:---|---:|---:|---:|
| lags included (default) | 48 | 1.215 | 0.302 |
| lags withheld (`include_target_lags: false`) | 39 | **1.196** | 0.295 |

The default is `true` because that is what
[09_why_deep_learning_underperformed.md](09_why_deep_learning_underperformed.md)
recommends and what the v4 bakeoff ran; the flag exists so the ablation is
one config line.

### 5. Rolling features

> **What:** the average and the variability of wind speed over the last 3 and 6 hours.
> **Why:** a single past reading can be a blip; an average over several hours describes
> the prevailing conditions. The variability figure separately tells the model whether the
> weather is currently calm and steady or gusty and changeable.
> **The safety detail:** these windows are deliberately shifted back by one hour so they
> can never include the moment being predicted — otherwise the model would be reading its
> own answer.

For each window `w ∈ config.model.rolling_windows`:
- `wind_speed_hub_rollmean{w}` = shifted-by-1 rolling mean over `w` steps
- `wind_speed_hub_rollstd{w}` = shifted-by-1 rolling std over `w` steps

**Current setting: `rolling_windows = [3, 6]`** (dropped 24 in v1.5).

The `.shift(1)` before `.rolling(w)` is deliberate — rolling stats never include
the current timestep, preventing target leakage. Verified in
`test_rolling_features_no_leakage_of_current_step`.

### 6. Target construction and NaN dropping

> **What:** defining the answer the model is trying to produce, and discarding rows that
> cannot be used.
> **Why rows get discarded:** the earliest rows have no 12-hours-ago value to look back
> at, and the last 24 rows have no known answer yet to learn from. Both are unavoidable
> edge effects of using history and predicting ahead.
> **Why this step deserves attention:** the target is wind speed 24 hours *after* the row's
> timestamp. Every column in that row must be knowable at the row's own time — the moment
> that stops being true, the model is cheating. This exact alignment is what the project
> got wrong in v3.

- **`mode='nowcast'`**: target = `wind_speed_hub`
- **`mode='forecast'`**: target = `wind_speed_hub.shift(-horizon_hours)`

The forecast mode's shift means the last `horizon_hours` rows have no valid target
and get dropped by `dropna`. Lag features similarly drop early rows during warmup.

Any feature column that is entirely NaN (e.g., `roughness` for NASA POWER,
pressure-level columns when only v1 core data is available) is dropped **before**
`dropna` on rows, so we don't lose the entire test window just because one column
is empty. This bug was caught during the v1 pipeline smoke test.

## Baselines

> **What:** two deliberately naive predictors the real model is required to beat.
> **Why they are mandatory:** a model can look impressive in isolation and still be
> useless. If a rule as dumb as "tomorrow will be like right now" matches it, the machine
> learning has added nothing. Design principle 5 in [doc 01](01_problem_and_design.md)
> makes reporting these compulsory — and in v1 that rule caught a model that genuinely was
> not working.

Two mandatory baselines. The pipeline flags any ML model that fails to beat
persistence.

### Persistence

> **What:** "the wind in 24 hours will be whatever it is right now."
> **Why it is a serious opponent:** at short horizons it is very hard to beat — weather
> changes slowly over an hour. At 24 hours it weakens considerably, which is precisely why
> this project targets day-ahead: it is the range where forecasting can add real value.
> ⚠️ **Known flaw:** the version used throughout v1–v4 actually looks back 25 hours rather
> than 24, making it slightly easier to beat than it should be. The corrected figure is
> 1.399 m/s; that is the number to quote.

`PersistenceForecaster`: `ŷ_{t+h} = y_t`. The most naive forecast — predict what
you currently see. In feature space (after `build_feature_frame`), the "current
observation" is the `wind_speed_hub_lag1` column, so persistence just returns that.

Persistence is trivial to beat at long horizons (24 h + wind drifts far from the
current value) and extremely hard at short horizons (1 h autocorrelation is very
high). This is why v1 (6 h horizon) had XGBoost losing to persistence but v1.5+
(24 h horizon) had it winning by ~18%.

> ⚠️ **It is actually a 25-hour persistence.** `build_feature_frame` excludes the
> raw `wind_speed_hub` column from `X` (it is the target), so the lag-1 column is
> the closest available stand-in for "the current observation" — but at feature
> time `t` that column holds `hub(t − 1)`, and it is scored against
> `y[t] = hub(t + 24)`. The baseline reported in every iteration is therefore a
> 25 h naive forecast, which is ~2.9% easier than the correct one:
>
> | Baseline | MAE | RMSE | R² |
> |:---|---:|---:|---:|
> | 25 h — `hub(t−1)`, as reported v1–v4 | 1.440 | 1.832 | −0.002 |
> | 24 h — `hub(t)`, correct | **1.399** | **1.791** | **0.043** |
>
> Left unchanged so v1–v4 remain comparable. Quote 1.399 in any write-up.
> Fixing it properly means giving `PersistenceForecaster` access to the unshifted
> hub series and restating the earlier iterations — see
> [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md) §4 and
> [08_next_steps.md](08_next_steps.md).

### Climatology

> **What:** "the wind tomorrow at 3pm will be the historical average for 3pm in this
> month."
> **Why it is the harder baseline to beat here:** it captures the daily sea-breeze cycle
> and the seasonal pattern without any modelling at all. It scores 1.282 m/s, notably
> better than persistence — so it is the more honest yardstick, and any claim about the
> model's skill should be made against both.

`ClimatologyForecaster`: predict the training-set mean of the target grouped by
`(month, hour_of_day)`. Captures the seasonal + diurnal pattern with no other
signal. Falls back to global mean for (month, hour) combinations unseen in training.

Climatology is a fair "seasonal mean" benchmark. Beating it modestly is easy;
beating it substantially requires the model to actually use weather features
beyond the calendar.

## Tree model: XGBoost

> **What:** the project's workhorse model — hundreds of small decision trees, each one
> correcting the errors left by the ones before it.
> **Why this type of model:** it handles tables of mixed-scale numeric features well,
> trains in about a minute, needs no data rescaling, and reports which inputs it relied
> on. That combination made it the natural baseline, and it stayed competitive with every
> neural network tried against it.

XGBoost is the primary Stage 1 model. LightGBM is available via
`stage1_algo: lightgbm` but not currently used in production runs.

### Regularised default hyperparameters (v1.5 → v2)

> **What:** hand-chosen settings that deliberately hold the model back — shallower trees,
> fewer features per split, penalties on complexity.
> **Why hobbling the model helps:** the first version memorised its training data almost
> perfectly while performing badly on unseen data — its error on unseen data was **eight
> times** its error on training data. Constraining it brought that ratio down to about
> **1.2×** (1.0 would mean it performs identically on both) with no loss of accuracy on
> the test set. A model prevented from memorising is forced to learn the pattern instead.

Tuned by hand to combat the v1 overfit (train MAE 0.134 vs test 1.045 = 8× gap):

```python
XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)
```

Key regularisation levers:
- `max_depth=5` limits tree complexity.
- `min_child_weight=10` requires each leaf to represent at least 10 samples' worth
  of gradient — strong regularisation against overfit on rare feature combinations.
- `subsample=0.8` and `colsample_bytree=0.8` add stochastic regularisation.
- `reg_alpha` (L1) and `reg_lambda` (L2) penalise leaf-weight magnitudes.

These defaults reduced the train/test gap from 8× to ~1.2× — the model stopped memorising
and started generalising, exactly what regularisation should do.

> ⚠️ **Do not read v1's 1.045 m/s and v1.5's 1.171 m/s as a regression.** They are not
> comparable: v1 predicted only **6 hours** ahead using NASA POWER data, while v1.5
> predicted **24 hours** ahead using ERA5 — a substantially harder task. The meaningful
> comparison is against each version's own persistence baseline, and on that measure the
> change is decisive:
>
> | Version | Horizon | Model test MAE | Persistence baseline | Beat it? |
> |:---|---:|---:|---:|:---|
> | v1 | 6 h | 1.045 m/s | 0.875 m/s | ❌ no |
> | v1.5 | 24 h | 1.171 m/s | 1.440 m/s | ✅ yes |
>
> v1 had the lower raw error and was still the failure. Raw MAE cannot be compared across
> different forecast horizons.

### Optuna hyperparameter tuning (v3)

> **What:** searching automatically for better settings, 50 attempts, each scored by
> time-aware cross-validation.
> **Why automate it:** hand-tuning explores a handful of guesses; a search explores the
> space systematically and records exactly what it tried, which makes the result
> reproducible rather than a matter of taste.
> **What it actually bought:** very little — about 1% better. That is itself an important
> finding: it says the hand-picked settings were already near the best available, and that
> the limit lies in the data rather than the configuration.

Added in v3 in [`src/stage1_windspeed.py`](../src/stage1_windspeed.py) as
`tune_xgboost`. Runs before the final training if `config.model.tune_xgboost: true`.

**Setup:**
- Sampler: `optuna.samplers.TPESampler(seed=42)` — Tree-structured Parzen Estimator,
  a Bayesian optimisation approach that adaptively focuses on promising regions.
- Objective: 3-fold `TimeSeriesSplit` CV mean MAE (fewer folds than final training's
  5, to keep the sweep fast).
- Trials: 50 (configurable via `config.model.tune_n_trials`).
- Direction: minimise.

**Search space:**

| Hyperparameter | Range | Sampling |
|:---|:---|:---|
| `max_depth` | 3 – 8 | int, uniform |
| `learning_rate` | 0.01 – 0.15 | float, log-uniform |
| `n_estimators` | 100 – 500 | int, step 50 |
| `min_child_weight` | 1 – 20 | int, uniform |
| `subsample` | 0.6 – 1.0 | float, uniform |
| `colsample_bytree` | 0.6 – 1.0 | float, uniform |
| `reg_alpha` | 1e-3 – 1.0 | float, log-uniform |
| `reg_lambda` | 0.1 – 3.0 | float, log-uniform |

**Wall time:** ~4 minutes for 50 trials on Anloga's ~28k training samples (Python
3.13, CPU-only, no parallelism across trials). Much faster than initially
estimated (~30 min).

**v3 best parameters:**

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

Best CV MAE: **1.1117 m/s** (vs untuned regularised CV MAE ≈ 1.15).

**Interpretation:** Optuna converged on a *more conservative* model than my
hand-picked v1.5 defaults — shallower trees (4 vs 5), much slower learning rate
(0.0185 vs 0.05), more trees to compensate (450 vs 200), stronger min_child_weight
(16 vs 10), lower subsampling ratios. The overall effect is a smoother, more
heavily-averaged ensemble. Test MAE: **1.143 m/s** (vs untuned 1.153) — a small
but real ~1% improvement.

### Cross-validation and final training

> **What:** testing the model on several successive slices of history before training the
> final version on everything available.
> **Why the slicing must respect time:** the standard approach shuffles data randomly,
> which here would let the model train on Thursday and be tested on Wednesday — learning
> from the future to predict the past. `TimeSeriesSplit` always keeps training data
> earlier than test data, which is the only honest way to score a forecaster.

`train_stage1` splits the data into 80% train / 20% test by *time order* (last 20%
of the timeline is held out), then runs `TimeSeriesSplit(n_splits=5)` on the train
slice to produce CV metrics per fold, then does a final fit on the full train slice
and evaluates on the held-out test.

The `TimeSeriesSplit` expanding-window design means fold 1 trains on the first
16% of the timeline and validates on the next 16%; fold 2 trains on the first 32%
and validates on the next 16%; etc. No fold ever uses future data to predict the
past.

The final model is persisted to `outputs/models/stage1.pkl` via `joblib.dump` with
its feature list and the raw config dict for reproducibility.

**darts models need a sidecar.** `TorchForecastingModel.__getstate__` deliberately
drops the LightningModule and Trainer, so `joblib.dump` on a darts model writes a
weightless shell — silently. `save_stage1` detects darts models
(`type(model).__module__.startswith("darts.")`), serialises them with darts' own
`.save()` to a `stage1.darts.pt` sidecar, and stores the class path in the pickle;
`load_stage1` reconstructs from it. This never mattered before v4 because XGBoost
always won `--compare` and the branch was never taken with a deep model. Covered
by `test_save_load_round_trip_darts_model`, which asserts the reloaded model both
has a non-`None` torch module and reproduces predictions.

## Deep models (v2+): LSTM, TFT, N-BEATS, DLinear

> **What:** four neural networks designed for sequences, run against the tree model on
> identical data.
> **Why compare four:** they embody genuinely different assumptions — step-by-step memory
> (LSTM), attention across the whole window (TFT), trend-and-season decomposition
> (N-BEATS), and a deliberately simple linear map (DLinear). If they all land in the same
> place, that says something about the data rather than about any one architecture.
> **What happened:** they did all land in the same place — within 0.101 m/s of each other
> and of the tree model. But reaching that conclusion took two attempts, because the first
> comparison was corrupted by a scoring bug
> ([doc 10](10_v4_deep_model_retraining.md)).

Deep sequence models are implemented in [`src/stage1_deep.py`](../src/stage1_deep.py)
using the [`darts`](https://github.com/unit8co/darts) library, which wraps PyTorch
models behind a consistent fit/predict/backtest interface.

### Common training setup (v4)

> **What:** the shared training recipe applied to all four networks — same loss function,
> same early stopping, same learning-rate schedule.
> **Why uniformity is the point:** if each model trains under different conditions you are
> comparing the recipes, not the architectures. One such mismatch was in fact responsible
> for TFT's apparent failure — it was optimising a harder objective than the others purely
> because of a library default.
> **Early stopping** means training halts when performance on held-out data stops
> improving, and the best version seen is restored — otherwise you keep whatever the final
> pass happened to leave behind.

All four models share the `model.deep` block, resolved by `resolve_deep_cfg()`
as `DEEP_DEFAULTS` ← `model.deep` ← `model.deep.overrides.<algo>` (deep-merged,
so per-model overrides can touch a single nested key):

```yaml
input_chunk_length: 48     # 48 h context window
hidden_size: 64
n_rnn_layers: 1
batch_size: 128
dropout: 0.2
learning_rate: 0.001       # starting point; lr_sweep overrides it
gradient_clip_val: 1.0
loss_fn: mse               # uniform objective — see the TFT note below
include_target_lags: true

n_epochs: 200              # a cap, not a target; early stopping ends the run
val_fraction: 0.15         # tail of the *train* slice, held out
early_stopping: {enabled: true, patience: 20, min_delta: 0.0001}
scheduler:  {enabled: true, warmup_epochs: 5, min_lr_factor: 0.02}
lr_sweep:   {enabled: true, candidates: [0.0003, 0.001, 0.003], epochs: 6}

overrides:                 # heavy models get smaller caps / bigger batches
  nbeats: {n_epochs: 100, batch_size: 256, early_stopping: {patience: 15}, lr_sweep: {epochs: 4}}
  tft:    {n_epochs:  60, batch_size: 256, early_stopping: {patience: 10}, lr_sweep: {candidates: [0.0003, 0.001], epochs: 3}}
```

**Validation and early stopping.** The last `val_fraction` of the train slice is
held out (the test slice is never touched). The validation series carries
`input_chunk_length` hours of its own leading context so its first scored sample
sits exactly at the boundary. Training stops on `val_loss` with the configured
patience.

**Best-checkpoint restore is not optional.** Early stopping leaves the *last*
weights loaded, which are by construction `patience` epochs worse than the best
ones — so early stopping without a restore is actively harmful. darts'
`save_checkpoints=True` writes a `best-{epoch}-{val_loss}` checkpoint and
`train_stage1_deep` reloads from it via `load_from_checkpoint(..., best=True)`
after fitting.

**LR schedule.** `WarmupCosineLR` (module-level class in `stage1_deep.py`, so the
fitted model stays picklable): linear warmup for `warmup_epochs`, then cosine
decay to `min_lr_factor × lr`, stepped once per epoch. `lr_sweep` picks the
initial LR by training each candidate for a short budget *with the scheduler
disabled* (a decaying LR would confound the comparison) and taking the lowest
`val_loss`.

Data preparation:
- Target series = raw hourly `wind_speed_hub` (unshifted; darts handles horizon
  via `output_chunk_length`).
- Past covariates = all exogenous features, **including** target-derived
  lag/rolling columns when `include_target_lags` is set (48 columns; 39 without).
- Two scalers (target + covariates) fit on the *fitting* slice only — not on
  validation and not on test.

### Evaluating a darts model against the feature frame

> **What:** converting the sequence library's predictions back onto the same timeline the
> tree model is scored on.
> **Why this small step matters more than anything else in the document:** the library
> returns predictions labelled by the hour being *described*, while the training table is
> labelled by the hour the prediction is *made from*. Those differ by 24 hours. Comparing
> them directly — which the code did for two full iterations — scores each model against
> the wrong hour, and produces results that look plausible and are meaningless. This is
> the single most consequential bug in the project's history.

Evaluation uses `historical_forecasts(retrain=False, forecast_horizon=24,
stride=1, last_points_only=True)`, producing one 24 h-ahead prediction per hour
of the test window. **These predictions cannot be compared to `y_test`
directly** — this was the v2/v3 bug, and it cost two iterations:

- `historical_forecasts(..., last_points_only=True)` returns a series indexed by
  **target time τ**. `hist[τ]` is the prediction of the wind speed *at* τ, made
  from data up to `τ − 24`.
- The feature frame is indexed by **feature time t**, where `y[t] = hub(t + 24)`.

Reindexing one onto the other scores `hub(t)` against `hub(t + 24)` — i.e. it
measures 24-hour persistence error, which is why every deep model in v2/v3 landed
in a narrow band around the persistence baseline. Two guards now exist:

- `_to_feature_time(hist, scaler, offset_hours)` un-scales and shifts the index
  back by the horizon (`offset = horizon` in forecast mode, `0` in nowcast).
- The backtest `start` is set to `split_ts + offset − horizon + 1`, because
  darts' `start` is the first point of the first forecast *window* and the first
  value returned lands at `start + horizon − 1`.

With both applied the information sets match the tree model exactly: at feature
time `t`, XGBoost and the sequence models both see data up to and including `t`
and predict `hub(t + 24)`. Regression tests:
`test_to_feature_time_shifts_index_back_by_horizon` (unit, both offsets) and
`test_deep_forecast_predictions_are_horizon_aligned` (end-to-end on a 37-hour
periodic signal — deliberately not commensurate with the 24 h horizon, so
persistence is weak and a shifted model cannot pass). Full write-up in
[10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md).

### LSTM (BlockRNNModel)

> **What:** a network that reads the last 48 hours one step at a time, carrying a running
> memory of what it has seen.
> **Why it suits this problem:** weather is a sequence, and an LSTM is built to notice
> that the order of events matters.
> **Result:** the best score in the corrected comparison (1.114 m/s) — from the smallest
> network tried, about 30,000 parameters. Its margin over the tree model is small enough
> to be luck, so the fair claim is that they are equivalent.

```python
BlockRNNModel(
    model="LSTM",
    hidden_dim=64,
    n_rnn_layers=1,
    dropout=0.2,
    input_chunk_length=48,
    output_chunk_length=24,
    n_epochs=30,
    batch_size=128,
    random_state=42,
)
```

**Why `BlockRNNModel` and not `RNNModel`?** darts's `RNNModel` only supports
`future_covariates` (values known at prediction time). `BlockRNNModel` supports
`past_covariates` (values known only up to the input window), which matches our
"predict future wind from past weather" setup.

Parameter count ≈ 30,000 — about 1 sample per parameter on ~28,000 training rows.
Through v3 this was read as "likely undertrained". It is the **best-performing
model in the zoo** (test MAE 1.114, R² 0.390), ahead of the Optuna-tuned
XGBoost, and it converges in 24 epochs. The samples-per-parameter heuristic had
no predictive power here — see
[07_findings_and_analysis.md](07_findings_and_analysis.md) §"What did *not* turn
out to be the cause".

Measured cost: ~35 s/epoch on 12 CPU threads.

### TFT (Temporal Fusion Transformer)

> **What:** the largest and most sophisticated model tried — it can attend to any part of
> its input window rather than reading strictly in order.
> **Why it initially looked like a failure, and was not:** the library quietly defaults to
> predicting a full range of possible outcomes (17 percentiles) rather than a single
> number, while every other model predicted a single number. It was being asked a much
> harder question and then scored on only the middle answer. Naming the objective
> explicitly moved it from 1.551 to 1.171 m/s.
> **The lesson:** an undocumented library default cost more accuracy than any architecture
> choice in the whole project.

```python
TFTModel(
    input_chunk_length=48,
    output_chunk_length=24,
    hidden_size=64,
    dropout=0.2,
    add_relative_index=True,       # substitute for missing future_covariates
    use_static_covariates=False,   # we have one site, no static features
    loss_fn=nn.MSELoss(),          # ← critical; see below
    n_epochs=60,                   # override: TFT is ~700 s/epoch on CPU
    batch_size=256,
    random_state=42,
)
```

TFT combines LSTM encoders with an attention mechanism and variable-selection
networks. Parameter count ≈ 100,000+.

> **`loss_fn` is the single most important argument here.** darts substitutes
> `likelihood=QuantileRegression()` — **17 quantiles** — whenever *both*
> `likelihood` and `loss_fn` are left `None`. The other three models default to
> plain MSE. Left unset, TFT is therefore fitting a full predictive distribution
> on the same 28k samples while its competitors fit a point estimate, and only
> the median quantile gets scored. Naming the loss makes it deterministic:
>
> | TFT objective | Test MAE | RMSE | R² | Beats persistence? |
> |:---|---:|---:|---:|:---:|
> | `QuantileRegression` (darts default) | 1.551 | 1.974 | −0.163 | ❌ |
> | `MSELoss` (`model.deep.loss_fn: mse`) | **1.171** | **1.498** | **0.330** | ✅ |
>
> Both runs early-stopped at the same point (28 vs 29 of 60 epochs), so this is
> not a budget effect. Nothing in the logs makes the substitution visible — the
> only hint is a `val_loss` around 0.42 against the other models' ~0.008, which
> reads as a broken model rather than a different objective.

The v3 claim that TFT suffers an *architecture* mismatch (no static covariates,
no future covariates, single series) reached the right conclusion by the wrong
route. Those observations are all true, but the measured deficit was the
objective. Whether attention buys anything here is still open: at 1.171 TFT is
mid-field, within 0.06 m/s of the winner.

**Runtime:** by far the most expensive of the four. ~700 s/epoch at batch 128 on
12 CPU threads; 4 h 33 min for the v4 quantile run and 9 h 56 min for the v4b MSE
run. Do not read anything into that difference — the second ran while the machine
was in interactive use and this is a 15 W laptop part, so it is thermal
throttling, not architecture.

### N-BEATS

> **What:** a deep network that pulls a series apart into trend and repeating-cycle
> components, then rebuilds a forecast from them.
> **Why it was expected to do well:** wind here has an obvious daily cycle, which is
> exactly the kind of structure this design targets.
> **Why its result is interesting:** it has 7.4 million parameters trained on 28,000
> examples — a ratio that conventional wisdom says should fail badly. It placed mid-field.
> That is direct evidence against the "too few samples for deep models" explanation the
> project believed for two iterations.

```python
NBEATSModel(
    input_chunk_length=48,
    output_chunk_length=24,
    num_stacks=3,
    num_blocks=3,
    num_layers=4,
    layer_widths=256,
    dropout=0.2,
    n_epochs=30,
    batch_size=128,
    random_state=42,
)
```

N-BEATS uses hierarchical basis decomposition (trend + seasonality blocks). In v3
it appeared to be the *only* deep model to beat persistence, narrowly (MAE 1.425
vs 1.440) — that reading was an artefact of the alignment bug, and the
"decomposition suits wind" story it supported is not borne out. Corrected, it
sits mid-field at **1.157 / 0.343**, third of five and within 0.043 m/s of the
winner.

**Parameter count is 7.4 million** (measured), not the ~500k estimated in v3 —
with 48 past covariates the first layer of each block takes a
`48 × 49 = 2,352`-wide flattened input, and there are 9 blocks. That is ~0.004
samples per parameter on a 28k-row training set. It places third anyway, which is
the clearest single counterexample to the sample-efficiency argument.

Measured cost: ~103 s/epoch at batch 128; converges in 19 epochs.

### DLinear

> **What:** the deliberately simple entry — essentially a straight-line mapping from the
> input window to the forecast.
> **Why include something this basic:** a 2022 paper showed simple linear models matching
> elaborate transformers on many forecasting benchmarks. Including one tests whether the
> complexity of the others is buying anything.
> **Result:** last of the five, but only 0.101 m/s behind the best — and still clearly
> ahead of both naive baselines. The complexity is buying very little.

```python
DLinearModel(
    input_chunk_length=48,
    output_chunk_length=24,
    kernel_size=25,
    use_static_covariates=False,
    n_epochs=30,
    batch_size=128,
    random_state=42,
)
```

DLinear (Zeng et al. 2022, "Are Transformers Effective for Time Series Forecasting?")
decomposes the input into trend (moving-average with kernel_size=25) and residual
components, then linearly maps each to the output. Fastest of the four by an order
of magnitude: ~5 s/epoch, converging in 23 epochs (~3 min end-to-end including
the LR sweep and both backtests).

In v3 it appeared to lose to persistence (1.519, R² −0.074), supporting a
"decomposition + linear is too simple for rich exogenous features" reading.
Corrected, it scores **1.215 / 0.302** — last of five, but comfortably clear of
both persistence baselines, and only 0.101 m/s behind the winner. For roughly
1/200th of TFT's training cost that is an excellent accuracy-per-second trade, and
it makes DLinear the right model to smoke-test pipeline changes against.

### Deep model dispatcher

> **What:** the small piece of code that maps a model name in the config to the right
> training function.
> **Why it gets its own section:** it once silently ignored two of the four models. They
> were fully implemented, but the dispatcher's list of recognised names was missing them,
> so a "five-model comparison" quietly ran three. Nothing failed and no error appeared —
> the results table was simply short, and that was easy to miss.

`config.model.stage1_algo` in `{lstm, tft, nbeats, dlinear}` routes to
`_make_darts_model` in `stage1_deep.py`. Everything else in the pipeline
(evaluation, three-variant comparison, report writing) treats tree and deep
`Stage1Result` objects identically — the report has no per-model branching.

### The `--compare` mode

> **What:** a command-line flag that trains every configured model in turn and ranks them.
> **Why it is ordered cheapest-first:** the full set takes many hours, dominated by TFT.
> Running the fast models first means a crash late in the run still leaves a usable
> comparison rather than nothing.

`python -m src.pipeline --compare` iterates over `config.model.compare_algos =
[xgboost, dlinear, lstm, nbeats, tft]`, trains each, and produces a head-to-head
ranking table in the report. The winner (lowest test MAE) is used for the
downstream two-stage evaluation, and `write_report` is told the winner's name
explicitly via `stage1_algo_label` (it previously printed
`config.model.stage1_algo`, which is wrong whenever the winner isn't the
configured primary).

The list is ordered **cheapest-first** so that a crash in TFT — ~700 s/epoch,
hours per run — still leaves four models in the report.

**v4 result from this command** (5 h 17 min wall, CPU-only):

| Rank | Model | Test MAE | RMSE | R² |
|---:|:---|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 |
| 4 | TFT (MSE, run separately as v4b) | 1.171 | 1.498 | 0.330 |
| 5 | DLinear | 1.215 | 1.530 | 0.302 |
| — | Climatology | 1.282 | 1.617 | 0.220 |
| — | Persistence (24 h, correct) | 1.399 | 1.791 | 0.043 |

> The on-disk `outputs/reports/report.md` still shows the *quantile* TFT at rank
> 5 (1.551), because the MSE fix was run standalone rather than by regenerating
> the whole bakeoff. The five-model report is preserved at
> `outputs/reports/report_v4_bakeoff.md`;
> [../RESULTS.md](../RESULTS.md) is the source of truth for the consolidated
> table.

## Feature-influence analysis

> **What:** working out which inputs the model actually relies on.
> **Why:** a model that predicts well for physically sensible reasons is far more
> defensible than one that predicts well for unknown reasons — and if the ranking
> contradicted meteorology, that would be a warning sign.
> ⚠️ **Important caveat:** the rankings below show what the model *leaned on*, which is not
> the same as what *improved accuracy*. Removing all twenty spatial columns changes the
> error by about 0.003 m/s despite one of them ranking second. Prefer the ablation
> measurements in [doc 07](07_findings_and_analysis.md) when making claims about value.

`analyze_feature_influence(X, y, outputs_dir)` in
[`src/stage1_windspeed.py`](../src/stage1_windspeed.py) produces two diagnostics
before Stage 1 training begins:

### Pearson correlation heatmap

> **What:** a grid showing which inputs move together, and which move with the answer.
> **Why look at it:** it is the simplest possible check, done before any model is trained.
> It flags features that are near-duplicates of each other — useful context when a
> ranking later says one matters and its twin does not.

Computes `corr(feature, target)` for every column and renders a horizontal bar
chart of absolute correlation, sorted. Saved to
`outputs/figures/pearson_heatmap.png`.

### SHAP summary

> **What:** a method that attributes each individual prediction to the inputs that shaped
> it, then aggregates across all predictions into a ranking.
> **Why it beats simple correlation:** it captures the model's actual reasoning, including
> interactions, rather than just whether two columns move together.
> **What it does not tell you:** whether the feature was *necessary*. A model will happily
> lean on a feature that a duplicate could have replaced entirely — which is why removal
> experiments, not SHAP rank, are the basis for value claims here.

Fits a quick XGBoost (`n_estimators=200`, default depth) purely for interpretability,
then computes SHAP values with `shap.TreeExplainer` on up to 2,000 sampled rows.
The mean absolute SHAP value per feature is the "how much did this feature move
predictions on average" metric.

`shap.summary_plot(..., plot_type="bar", max_display=20)` saved to
`outputs/figures/shap_summary.png`. A ranked markdown table
(feature | Pearson r | mean|SHAP|) is written to `outputs/reports/feature_ranking.md`.

### v3 top 15 features (SHAP-ranked)

> **What:** the fifteen inputs the tuned tree model leaned on hardest.
> **Why it is reassuring:** the ordering matches physical intuition — current wind speed
> first, then time of day and year, then winds high above the site. Nothing inexplicable
> appears near the top, which is what you want.
> ⚠️ Read alongside the caveat above: high rank here did **not** translate into measured
> accuracy for the spatial columns.

| Rank | Feature | Pearson r | mean\|SHAP\| |
|---:|:---|---:|---:|
| 1 | `wind_speed_100m` | +0.571 | 0.4415 |
| 2 | `wind_speed_100m_mean` | +0.571 | 0.2864 |
| 3 | `hour_sin` | -0.169 | 0.1912 |
| 4 | `month_cos` | -0.289 | 0.1570 |
| 5 | `month_sin` | +0.063 | 0.1275 |
| 6 | `wind_dir_850hpa_cos` | +0.010 | 0.1219 |
| 7 | `wind_dir_100m_sin` | -0.228 | 0.1207 |
| 8 | `dewpoint_c` | -0.080 | 0.0953 |
| 9 | `temperature_c_mean` | -0.263 | 0.0910 |
| 10 | `temperature_c` | -0.255 | 0.0818 |
| 11 | `bl_height_m` | +0.186 | 0.0719 |
| 12 | `wind_speed_hub_lag12` | +0.318 | 0.0682 |
| 13 | `temperature_c_grad_ew` | +0.166 | 0.0646 |
| 14 | `pressure_pa` | +0.041 | 0.0577 |
| 15 | `wind_dir_850hpa_sin` | -0.336 | 0.0482 |

**Interpretation:**
- **The centre-cell 100 m wind speed dominates** (0.44 SHAP) — as expected.
- **Its patch mean is the #2 predictor** (0.29 SHAP) — spatial averaging carries
  meaningful information beyond the point value.
- **Calendar features rank high** (hour_sin, month_cos, month_sin) — the diurnal
  sea-breeze and seasonal patterns are strong.
- **Upper-air features earn their keep** — 850 hPa wind direction (cos and sin)
  appear at #6 and #15, confirming that the tropospheric flow drives surface wind
  as meteorology predicts.
- **Boundary-layer height** (#11) and **dewpoint** (#8) — both v2 additions —
  contribute measurably.
- **Spatial gradients** appear (`temperature_c_grad_ew` at #13) — pressure/temperature
  gradients drive wind, so this signal is physically expected.
- **Notably absent from the top 5:** the raw `wind_speed_10m`. The 100 m field is
  a strictly better predictor of 100 m hub wind, and once it's in the model, the
  10 m field adds little.

## Metric definitions

> **What:** the exact formula behind each score reported for Stage 1.
> **Why spell them out:** the same metric name is computed differently in different
> papers, and R² in particular has several competing definitions. Writing the formulas
> down is what lets someone else reproduce or challenge these numbers. Plain-English
> versions are in [glossary.md](glossary.md).

Wind-speed metrics (Stage 1):

| Metric | Formula | Units | Interpretation |
|:---|:---|:---|:---|
| MAE | `mean(\|y - ŷ\|)` | m/s | "Typical error" — robust to outliers. |
| RMSE | `sqrt(mean((y - ŷ)²))` | m/s | Penalises large errors; sensitive to tails. |
| MAPE | `mean(\|y - ŷ\| / \|y\|) × 100%` | % | Relative error; unstable near zero. |
| R² | `1 − sum((y - ŷ)²) / sum((y - ȳ)²)` | dimensionless | Fraction of variance explained; negative if worse than predicting the mean. |

### Sign conventions

> **What:** which direction counts as "better" for each metric.
> **Why it needs stating:** most are lower-is-better, but R² is higher-is-better and can
> go negative — a negative R² means the model is doing worse than always guessing the
> average. Several early results were negative, so this is not hypothetical.

**RMSE and R² are printed plain** — `0.390`, not `+0.390`. Earlier versions used
a `:+.3f` format specifier to force an explicit `+`, on the reasoning that it
made positive and negative values unambiguous. In practice it did the opposite:
the results tables also carried Δ columns comparing iterations, so a reader saw
signed numbers in two different roles at once and could not tell a measurement
(`+0.390`, an absolute R²) from a difference (`−0.361`, a change in MAE). The Δ
columns are gone and the `+` with them.

A **negative R² is a real measurement**, not a formatting artefact: it means the
model does worse than predicting the training mean. Several baselines and the
pre-fix quantile TFT are legitimately negative, so minus signs in an R² column
should be read as results, not as deltas.

**Pearson r keeps its explicit sign** in the feature-ranking tables
(`analyze_feature_influence`, `outputs/reports/feature_ranking.md`). There the
sign carries the information — it says whether a feature correlates positively or
negatively with hub wind speed — so `+0.571` versus `-0.336` is a meaningful
contrast rather than noise. This asymmetry with R² is deliberate.

Downstream power metrics (nMAE, nRMSE) are covered in
[05_stage2_and_evaluation.md](05_stage2_and_evaluation.md).
