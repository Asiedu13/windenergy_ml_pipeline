# Ghana Wind Power Forecasting Pipeline

Two-stage wind power forecasting pipeline for Ghanaian sites.

- **Stage 1 (ML):** predicts hub-height wind speed some hours ahead from reanalysis weather.
- **Stage 2 (Simulator):** converts predicted wind speed to power using a manufacturer power curve via `windpowerlib`.

Ghana has no operating utility wind farm publishing SCADA data, so this pipeline forecasts *potential* output from reanalysis weather (ERA5 or NASA POWER) plus a manufacturer power curve. See `outputs/reports/report.md` after a run for feature rankings, three-model comparison, and error-propagation analysis.

## Quick start

```bash
pip install -r requirements.txt
python -m src.pipeline --config config.yaml
```

Outputs land in `outputs/models/`, `outputs/figures/`, `outputs/reports/`.

## Data sources

The pipeline supports two reanalysis sources (choose via `data.source` in `config.yaml`):

### ERA5 (primary, higher quality)

Requires a free Copernicus Climate Data Store (CDS) account.

1. Register at https://cds.climate.copernicus.eu/
2. Copy your API key from your CDS profile.
3. Create `~/.cdsapirc` with:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-KEY>
   ```
4. Accept the ERA5 licence once via the CDS web UI.

First run downloads ~decade of hourly variables for a small bbox around the site — expect a slow first run (minutes to hours depending on date range) and cached NetCDF in `data/raw/`.

### NASA POWER (no auth, coarser)

Set `data.source: nasa_power` in `config.yaml`. Uses the public POWER hourly API, no credentials needed.

## Changing site or turbine

Edit `config.yaml`. Retraining on a new site is a one-line change:

```yaml
site:
  name: takoradi          # any short slug
  latitude: 4.90
  longitude: -1.75
  hub_height_m: 100
```

Turbine name must exist in the `oedb` library (see `windpowerlib.get_turbine_types()`).

## Retraining note

Wind-speed models are **site-specific** — a model trained at Anloga will not generalize to Ada Foah. Rerun the pipeline for each new site. This is a documented property of the architecture, not a bug.

## Layout

- `src/data_ingest.py` — ERA5 + NASA POWER fetchers
- `src/features.py` — hub-height extrapolation, encodings, lags, rolling stats
- `src/stage1_windspeed.py` — persistence/climatology baselines + XGBoost/LightGBM + SHAP
- `src/stage2_powercurve.py` — turbine power curve + air-density correction
- `src/evaluate.py` — three-model comparison + error propagation
- `src/pipeline.py` — CLI orchestrator
- `src/report.py` — auto-generated markdown report

Run `pytest -q` to execute the test suite.
