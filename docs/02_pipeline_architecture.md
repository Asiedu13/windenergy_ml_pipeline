# 02 — Pipeline Architecture

The pipeline is a small, focused Python package with clear module boundaries.
This document describes the code layout, how data flows through it, the configuration
schema, and the testing strategy.

> **What this document is:** the map of the code — which file does what, how data moves
> between them, and how to run and test it all.
> **Why read it:** doc 01 explains *what* was built and why; this explains *where it
> lives*. Read it before changing any code. Jargon: [glossary.md](glossary.md).

## Repo layout

> **What:** the folder structure, top to bottom.
> **Why it matters:** the split between `src/` (code), `data/` (downloads, never edited by
> hand) and `outputs/` (generated results, safe to delete and regenerate) is the
> convention everything else assumes.

```
windenergy_ml_pipeline/
  config.yaml             # single source of truth for all knobs
  requirements.txt        # pinned dependencies
  README.md               # short user-facing README
  RESULTS.md              # concise chronological result log (this doc's sibling)
  .gitignore              # data/, outputs/, __pycache__/, .venv/ ignored
  docs/                   # detailed documentation (you are here)
  data/
    raw/                  # cached NetCDF / JSON downloads (gitignored)
    processed/            # tidy hourly parquet files (gitignored)
  src/
    __init__.py
    utils.py              # config loader (dataclasses), seeding, logger
    data_ingest.py        # ERA5 + NASA POWER fetchers → tidy hourly DataFrame
    data_ingest_extend.py # resumable download-only CLI for extending ERA5 (v4)
    features.py           # hub-height extrapolation, encodings, lags, rolling
    stage1_windspeed.py   # baselines, XGBoost/LightGBM, Optuna, SHAP, save/load
    stage1_deep.py        # LSTM / TFT / N-BEATS / DLinear via darts (v2+)
    stage2_powercurve.py  # turbine load, power curve, air-density correction
    evaluate.py           # three-variant comparison + error-propagation plot
    report.py             # markdown report assembly
    pipeline.py           # CLI orchestrator, --compare flag, dispatch
  tests/
    __init__.py
    test_data_ingest.py   # mocked NASA POWER, schema tests
    test_features.py      # lag/roll/encoding round-trips
    test_stage1.py        # baselines + XGBoost + deep-model smokes + Optuna
    test_stage2.py        # power curve edge cases + density correction
    test_pipeline.py      # end-to-end smoke on synthetic data
  outputs/
    models/               # persisted stage1.pkl (via joblib)
    figures/              # PNGs from feature analysis + variants + error prop
    reports/              # auto-generated markdown report per run
```

## Module map

> **What:** one paragraph per Python module — its job and its public functions.
> **Why it matters:** each module has a single responsibility (fetch data, build features,
> train, simulate power, evaluate, report). If you know which of those six things you are
> changing, this tells you the one file to open.

Each `src/*.py` file has one clearly delineated responsibility:

**[`src/utils.py`](../src/utils.py)** — dataclasses that mirror the YAML config
(`SiteConfig`, `DataConfig`, `TurbineConfig`, `ForecastConfig`, `ModelConfig`,
`PathsConfig`, `Config`), plus `load_config(path) → Config`, `set_seed(seed)`, and
`get_logger(name)`. Every other module imports from here for typed config access —
no raw dict-key lookups anywhere else.

**[`src/data_ingest.py`](../src/data_ingest.py)** — the single public entry point
`load_weather(config) → pd.DataFrame` dispatches on `config.data.source` to either
`_fetch_era5` or `_fetch_nasa_power`. Handles monthly chunking, per-request retry,
ZIP unpacking, spatial derivative computation, and parquet caching. Output is a
UTC-tz-aware hourly DataFrame with a fixed column schema (see
[03_data_sources.md](03_data_sources.md)).

**[`src/features.py`](../src/features.py)** — `build_feature_frame(weather_df, config, mode)
→ (X, y, feature_cols)`. Composes `extrapolate_hub_wind`, `add_direction_encodings`,
`add_calendar_features`, `add_lag_features`, `add_rolling_features`, then constructs the
target (same-step for `nowcast`, `t + horizon` for `forecast`), drops all-NaN feature
columns (e.g., `roughness` for NASA POWER), and drops rows with NaNs from lag/rolling
warmup.

**[`src/stage1_windspeed.py`](../src/stage1_windspeed.py)** — everything for the tree
Stage 1: `PersistenceForecaster` and `ClimatologyForecaster` baselines,
`compute_metrics` (MAE, RMSE, MAPE, R²), `analyze_feature_influence` (Pearson heatmap
+ SHAP summary + ranked table), `tune_xgboost` (Optuna 3-fold TSCV sweep),
`train_stage1` (TimeSeriesSplit CV + final fit), `save_stage1` / `load_stage1` via
joblib. Owns the `Stage1Result` dataclass that both tree and deep pipelines return.

**[`src/stage1_deep.py`](../src/stage1_deep.py)** — deep Stage 1 via `darts`
(BlockRNN/LSTM, TFTModel, NBEATSModel, DLinearModel). `_make_darts_model` dispatches
on `config.model.stage1_algo`; `resolve_deep_cfg` merges the config layers.
`train_stage1_deep` reuses `Stage1Result` so the report generator can treat tree
and deep results identically. Handles scaling (`darts.Scaler`, fit on the fitting
slice only), the train/validation/test three-way split, `EarlyStopping` with
best-checkpoint restore, `WarmupCosineLR`, the LR sweep, and the
`historical_forecasts(last_points_only=True)` backtest.

> **`_to_feature_time()` is the load-bearing function here.** darts returns
> backtests indexed at *target* time τ; the feature frame is indexed at *feature*
> time t with `y[t] = hub(t + horizon)`. Comparing them without converting scores
> the whole test set `horizon` hours out of phase — the v2/v3 bug. See
> [04_stage1_wind_speed.md](04_stage1_wind_speed.md) §"Evaluating a darts model
> against the feature frame".

**[`src/data_ingest_extend.py`](../src/data_ingest_extend.py)** — download-only
CLI (`python -m src.data_ingest_extend --config config_10yr.yaml`) that fetches
missing ERA5 months for a config's date range and *reports* what is still missing
rather than raising. Separate from `load_weather` because that function assembles
a parquet at the end and aborts on any missing core month, which is unhelpful for
a resumable multi-hour job. Reuses `data_ingest`'s request builders and
`_retrieve_with_retry`.

**[`src/stage2_powercurve.py`](../src/stage2_powercurve.py)** — `load_turbine(name,
hub_height_m) → WindTurbine`, `turbine_spec` (extracts cut-in/rated/cut-out from the
curve programmatically), `air_density(temperature_c, pressure_pa)` via ideal gas law,
`windspeed_to_power(ws_series, turbine, density=None)` with optional density
correction, `model_chain_from_weather` (thin ModelChain wrapper for the physics-only
baseline path).

**[`src/evaluate.py`](../src/evaluate.py)** — `evaluate_variants` produces the three
`VariantResult` objects (physics_only, two_stage, direct), each with true power series,
predicted power series, and metrics. `error_propagation_plot` bins by |ws error| and
plots mean |power error|. `timeseries_plot` produces the test-week overlays.
`summarize_variants` returns the rows for the report table.

**[`src/report.py`](../src/report.py)** — `write_report` assembles
`outputs/reports/report.md` from the site, turbine, Stage 1 result, feature ranking,
variants summary, and figure paths. Sorts the head-to-head comparison table by test
MAE ascending and bolds the winner. Includes a reproducibility footer with seed,
config hash, and git SHA.

**[`src/pipeline.py`](../src/pipeline.py)** — CLI entry
(`python -m src.pipeline --config config.yaml --mode forecast --compare`). Loads
config, seeds RNGs, calls `load_weather`, `build_feature_frame`,
`analyze_feature_influence`, dispatches to `train_stage1` or `train_stage1_deep` per
algo (iterating over `config.model.compare_algos` under `--compare`), picks the best
model by test MAE for downstream two-stage evaluation, calls `evaluate_variants` +
`error_propagation_plot` + `timeseries_plot`, then `write_report`.

## Data flow

> **What:** the journey of a number, from a downloaded weather file to a figure in the
> final report.
> **Why it matters:** this is the mental model to hold while debugging. When a result
> looks wrong, you work backwards along this chain to find the step that broke — which is
> exactly how the v3 measurement bug was eventually located.

```
config.yaml
    │
    ▼
utils.load_config ─────► Config (dataclasses)
    │
    ▼
data_ingest.load_weather(config)
    │  (ERA5 CDS ← ~/.cdsapirc, or NASA POWER HTTP)
    │  monthly chunks → data/raw/*.nc | *.json
    │  loader → tidy hourly DataFrame (UTC index, ~35k rows for 2020-2023)
    │  parquet → data/processed/*.parquet
    ▼
features.build_feature_frame(weather_df, config, mode)
    │  hub extrapolation (log law + roughness)
    │  dir sin/cos, calendar sin/cos, lag features, rolling
    │  target = wind_speed_hub  (nowcast) or  wind_speed_hub.shift(-horizon)  (forecast)
    │  drop all-NaN columns; dropna rows
    ▼
    (X, y, feature_cols)
    │
    ├─► stage1_windspeed.analyze_feature_influence
    │        Pearson corr + SHAP → outputs/figures/*.png + feature_ranking.md
    │
    ├─► [per algo in config.model.compare_algos]
    │        stage1_windspeed.train_stage1  OR  stage1_deep.train_stage1_deep
    │            ├── Optuna tune (if tune_xgboost=true)
    │            ├── TimeSeriesSplit CV
    │            ├── Final fit on all train
    │            └── Stage1Result (metrics + predictions + persistence/climatology)
    │
    ▼
Best Stage1Result (by test MAE)
    │
    ▼
stage2_powercurve.load_turbine  +  turbine_spec
    │
    ▼
evaluate.evaluate_variants(weather, stage1, turbine, spec, config)
    │  physics_only : raw hub wind → windspeed_to_power
    │  two_stage    : stage1.pred_test → windspeed_to_power
    │  direct       : train XGBoost on features → power directly (skip curve)
    │  reference    : actual hub wind → windspeed_to_power  (with air density)
    ▼
evaluate.error_propagation_plot + timeseries_plot per variant
    │
    ▼
report.write_report → outputs/reports/report.md
```

## Config schema

> **What:** every setting in `config.yaml` and what it controls.
> **Why it matters:** the config is the only thing you should normally edit. Site,
> turbine, date range, forecast horizon and model choice are all changed here rather than
> in code — that is what makes a run reproducible from a single file.

Single YAML file, [`config.yaml`](../config.yaml). Every knob lives here; the code
never reads environment variables or hard-codes site-specific values.

```yaml
site:
  name: anloga                  # short slug used in cache filenames
  latitude: 5.79                # WGS84 decimal degrees, prime meridian → positive
  longitude: 0.90               # small (Ghana near Greenwich)
  hub_height_m: 100             # target height for extrapolation + Stage 2

data:
  source: era5                  # era5 | nasa_power
  start: "2020-01-01"           # inclusive
  end: "2023-12-31"             # inclusive
  cache_dir: data/raw

turbine:
  name: E-101/3050              # must exist in oedb turbine library
  source: oedb                  # only oedb currently supported

forecast:
  horizon_hours: 24             # forecast lead time (used when mode=forecast)
  timestep: hourly              # currently only hourly supported

model:
  stage1_algo: xgboost          # xgboost | lightgbm | lstm | tft | nbeats | dlinear
  test_split: 0.2               # last 20% of timeline held out
  cv_folds: 5                   # TimeSeriesSplit folds for final training
  lags: [1, 2, 3, 6, 12]        # hours of lag features for wind_speed_hub
  rolling_windows: [3, 6]       # hours of rolling mean/std features
  seed: 42                      # propagates to numpy, XGBoost, Optuna, torch

  # Optuna XGBoost sweep (only applies when stage1_algo=xgboost)
  tune_xgboost: true
  tune_n_trials: 50

  # Models trained under --compare (empty → primary + LSTM sibling).
  # Ordered cheapest-first so a crash in TFT still leaves four models reported.
  compare_algos: [xgboost, dlinear, lstm, nbeats, tft]

  # Deep-model knobs (ignored by tree algos). Resolved by resolve_deep_cfg() as
  # DEEP_DEFAULTS ← model.deep ← model.deep.overrides.<algo>, deep-merged, so a
  # per-model override can touch one nested key without restating the block.
  deep:
    input_chunk_length: 48      # look-back window in hours
    hidden_size: 64             # hidden dim / attention hidden size
    n_rnn_layers: 1
    batch_size: 128
    dropout: 0.2
    learning_rate: 0.001        # starting point; lr_sweep overrides it
    gradient_clip_val: 1.0
    loss_fn: mse                # uniform objective; null → each model's darts
                                # default, which for TFT is a 17-quantile
                                # QuantileRegression likelihood
    include_target_lags: true   # lag/rolling cols as past covariates (39 → 48)

    n_epochs: 200               # a cap, not a target — early stopping ends it
    val_fraction: 0.15          # tail of the *train* slice, held out
    early_stopping:
      enabled: true
      patience: 20
      min_delta: 0.0001
    scheduler:                  # WarmupCosineLR, stepped per epoch
      enabled: true
      warmup_epochs: 5
      min_lr_factor: 0.02
    lr_sweep:                   # short-budget, scheduler off, picks on val_loss
      enabled: true
      candidates: [0.0003, 0.001, 0.003]
      epochs: 6

    nbeats:
      num_stacks: 3
      num_blocks: 3
      layer_widths: 256
    dlinear:
      kernel_size: 25

    overrides:                  # heavy models: smaller caps, larger batches
      nbeats:
        n_epochs: 100
        batch_size: 256
        early_stopping: {patience: 15}
        lr_sweep: {epochs: 4}
      tft:
        n_epochs: 60
        batch_size: 256
        early_stopping: {patience: 10}
        lr_sweep: {candidates: [0.0003, 0.001], epochs: 3}

paths:
  outputs: outputs
  processed: data/processed
```

## Testing strategy

> **What:** what the automated test suite checks, and how it is organised.
> **Why it matters:** in ML the dangerous bugs do not crash — they return a plausible
> wrong number. So several tests assert *quality*, not just that code runs: they use
> signals crafted so that a mis-aligned or subtly broken model cannot pass. That style
> exists because tests which only checked shapes let a major bug through two whole
> iterations (see [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md)).

38 tests across 5 files, all `pytest` — no test framework beyond the standard library.

**[`tests/test_stage2.py`](../tests/test_stage2.py)** (8 tests) — Stage 2 is
deterministic and cheap to test. Below cut-in → 0 kW. Above cut-out → 0 kW. Above
rated → ~nameplate. Monotonic non-decreasing between cut-in and rated. Density
correction: lower density → lower power at mid-range. `load_turbine('E-101/3050')`
returns non-empty curve. `air_density(15, 101325)` ≈ 1.225 kg/m³.

**[`tests/test_data_ingest.py`](../tests/test_data_ingest.py)** (5 tests) — u/v to
speed/dir round-trip, NASA POWER payload parsing, end-to-end `load_weather` with a
mocked `requests.get`, cache hit on second call, unknown source raises ValueError.
CDS is *not* mocked because tests never touch it — real fetches only via
`python -m src.pipeline`.

**[`tests/test_features.py`](../tests/test_features.py)** (8 tests) — hub
extrapolation strictly increases wind speed from 100 m to 140 m; equals the reference
at the reference height; direction encodings satisfy sin² + cos² = 1; calendar features
cycle correctly; lag features match pandas `.shift(k)`; rolling features have no
current-step leakage; `build_feature_frame` returns valid shapes for nowcast and
forecast modes.

**[`tests/test_stage1.py`](../tests/test_stage1.py)** (16 tests) — persistence
predicts lag1; climatology gives correct shapes; `compute_metrics` on perfect
prediction returns zeros / R²=1; XGBoost beats persistence on a synthetic sinusoid
+ noise series; save/load round-trip preserves predictions bit-exactly; smoke tests
for LSTM, TFT, N-BEATS, DLinear on tiny synthetic data (2 epochs, tiny hidden dim);
Optuna 3-trial sweep returns a valid params dict.

Six added in v4/v4b, and they are deliberately a *different kind* of test:

- `test_to_feature_time_shifts_index_back_by_horizon` — the τ→t conversion, both
  forecast and nowcast offsets.
- `test_deep_forecast_predictions_are_horizon_aligned` — end-to-end guard on a
  37-hour periodic signal, chosen because it is not commensurate with the 24 h
  horizon: persistence is weak, so a correctly aligned model must beat it by a
  wide margin and a phase-shifted one cannot. **This asserts a quality floor, not
  a shape** — which is precisely why the v1–v3 smoke tests (shape + finiteness)
  could not catch the alignment bug.
- `test_resolve_deep_cfg_applies_defaults_and_overrides` — three-layer config
  merge, including nested deep-merge.
- `test_exogenous_cols_lag_toggle` — `include_target_lags` both ways.
- `test_save_load_round_trip_darts_model` — the reloaded darts model has a
  non-`None` torch module and reproduces predictions (plain `joblib` does not).
- `test_loss_fn_makes_tft_deterministic` — TFT has `likelihood is None` with an
  explicit `loss_fn`, and a non-`None` likelihood without one.

**[`tests/test_pipeline.py`](../tests/test_pipeline.py)** (1 end-to-end test) —
mocks NASA POWER (no CDS), writes a temporary config, runs `pipeline.run()` in
forecast mode, asserts the report file exists and contains all three variant names,
and that Stage 1 pickle and all figures are produced.

## Running the pipeline

> **What:** the commands to actually execute a run, and what each flag does.
> **Why it matters:** a first run downloads years of weather data and can take hours, so
> it is worth knowing what you are starting before you start it. Downloads are cached and
> resumable — re-running never re-fetches what is already on disk.

```bash
# Full comparison run (all 5 models + tuned XGBoost) — used for v4
# ~5-7 h on CPU; TFT dominates. Log to a file and background it.
python -m src.pipeline --config config.yaml --mode forecast --compare

# Single primary model only (matches config.model.stage1_algo)
python -m src.pipeline --config config.yaml --mode forecast

# Nowcast (same-timestep, i.e., spatial downscaling from reanalysis) — sanity check
python -m src.pipeline --config config.yaml --mode nowcast

# Extend the ERA5 cache to 2010–2023 (resumable; re-run until zero missing)
python -m src.data_ingest_extend --config config_10yr.yaml

# Tests
pytest -q                       # all 38 tests
pytest tests/test_stage1.py -v  # single file, verbose
```

The default `config.yaml` shipped in the repo produces the v4 comparison result
documented in [06_iteration_history.md](06_iteration_history.md), except that TFT
now trains with `loss_fn: mse` (v4b) rather than the quantile objective that was
still in place when `outputs/reports/report.md` was last generated.

## Reproducibility

> **What:** the machinery that makes the same config produce the same numbers every time.
> **Why it matters:** machine learning uses randomness in many places (data shuffling,
> model initialisation, hyperparameter search). Unless every source is pinned to a fixed
> seed, two runs of identical code disagree — and you cannot tell a real improvement from
> noise. That makes this section a precondition for trusting any comparison in these docs.

- **Seed:** `config.model.seed` (default 42) propagates through:
  `numpy.random.seed`, `os.environ["PYTHONHASHSEED"]`, `python.random.seed`,
  `XGBRegressor(random_state=...)`, `optuna.samplers.TPESampler(seed=...)`,
  darts models' `random_state`. Same config in → same numbers out.
- **Config hash:** the report footer includes `sha256(config.yaml)[:12]` so a
  reader can verify which config produced which report.
- **Git SHA:** the report footer includes `git rev-parse --short HEAD` for the
  commit the pipeline ran against.
- **Cache filenames encode the site slug and date range** so switching sites or
  windows never contaminates previously-cached data.
