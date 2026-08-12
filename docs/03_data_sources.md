# 03 — Data Sources

The pipeline supports two reanalysis feeds and produces a single tidy hourly
DataFrame that downstream feature engineering consumes regardless of source. This
document covers both sources, the trade-offs between them, all fetched variables,
CDS-specific quirks (chunking + cost cap, licence acceptance, ZIP unpacking), the
grid patch and spatial derivatives, and the tidy output schema.

> **What this document is:** everything about getting weather data in — which archives,
> which variables, the download quirks that cost real time, and the exact shape of the
> result.
> **Why read it:** [00_data_card.md](00_data_card.md) is the quick lookup table; this is
> the reasoning and the war stories behind it. Terms: [glossary.md](glossary.md).

## ERA5 vs NASA POWER

> **What:** a head-to-head of the two historical weather archives available.
> **Why it matters:** ERA5 was chosen as primary because it is finer-grained (~25 km vs
> ~55 km cells) and carries upper-air winds, which matter for next-day prediction. The
> cost is that ERA5 needs an account and hours of queueing, while NASA POWER is instant
> and needs nothing — which is why the slower one had to be justified.

| Property | ERA5 (primary) | NASA POWER (fallback) |
|:---|:---|:---|
| Provider | ECMWF via CDS API | NASA Langley POWER API |
| Grid resolution | 0.25° × 0.25° (~25 km) | 0.5° × 0.625° (~55 km) |
| Time resolution | Hourly | Hourly |
| Upper-air data | Yes (via pressure-levels dataset) | No — 10 m + 50 m only |
| Roughness | Yes (`forecast_surface_roughness`) | No — assume constant 0.15 m |
| Boundary-layer height | Yes | No |
| Wind gusts | Yes | No |
| Authentication | CDS account + `~/.cdsapirc` API key required | None (public) |
| Licence acceptance | Manual one-time click on CDS site | None |
| Cost cap per request | 121,000 cost units (their unit) | No formal cap |
| Payload per year (Anloga) | ~200 KB × 3 families × 12 months ≈ 7 MB | ~5 MB JSON |
| Typical wall time | 5-8 hours for 4 years (dominated by CDS queue) | <30 seconds for 4 years |

**Choice:** ERA5 is the primary data source because of higher spatial resolution,
richer variables (particularly upper-air winds, which the feature-influence analysis
shows are meaningful predictors), and native roughness field. NASA POWER is retained
for two reasons: (1) it's a network-only fallback if CDS is unreachable, and (2) it
provides an independent cross-source sanity check ("do the same broad conclusions
hold when the input data changes?").

All ERA5 v1 → v3 results in this project use `data.source: era5` in `config.yaml`.

## ERA5 variable catalogue

> **What:** the full list of weather variables requested from ERA5, in three batches.
> **Why three batches:** they were added at different times. Splitting them into separate
> download families means adding new variables never invalidates data already on disk —
> a practical necessity when a full download takes many hours.

ERA5 data is split across three CDS request families, stored in three separate cache
prefixes so newer variables can be added without invalidating older downloads.

### v1 core (single-level, 7 variables)

> **What:** the original seven near-surface variables, downloaded first.
> **Why these seven:** wind at two heights (to work out how wind changes with altitude),
> temperature and pressure (to compute air density for the power conversion), and surface
> roughness (needed to extrapolate wind up to the 100 m hub).
> **Note:** ERA5 stores wind as two components, eastward and northward, rather than as a
> speed and a direction — the conversion is at the end of this document.

Cache prefix: `data/raw/era5_{site}_{YYYY}-{MM}.nc`

| CDS variable name | Short | Meaning | Used for |
|:---|:---|:---|:---|
| `10m_u_component_of_wind` | `u10` | Eastward wind at 10 m (m/s) | wind speed + direction at 10 m |
| `10m_v_component_of_wind` | `v10` | Northward wind at 10 m (m/s) | wind speed + direction at 10 m |
| `100m_u_component_of_wind` | `u100` | Eastward wind at 100 m (m/s) | wind speed + direction at 100 m (primary Stage 1 input) |
| `100m_v_component_of_wind` | `v100` | Northward wind at 100 m (m/s) | wind speed + direction at 100 m |
| `2m_temperature` | `t2m` | 2 m temperature (K → converted to °C) | air-density correction, feature |
| `surface_pressure` | `sp` | Surface pressure (Pa) | air-density correction, feature |
| `forecast_surface_roughness` | `fsr` | Roughness length (m) | log-law hub extrapolation |

### v2 single-level extras (3 variables)

> **What:** three more surface variables added in the second iteration.
> **Why:** gusts, dewpoint and boundary-layer height all describe how turbulent and
> well-mixed the lower atmosphere is — which governs how much of the faster wind higher up
> actually reaches the turbine.

Cache prefix: `data/raw/era5x_{site}_{YYYY}-{MM}.nc`. Delivered by CDS as a ZIP
containing multiple NetCDFs (see "ZIP unpacking" below).

| CDS variable name | Short | Meaning | Used for |
|:---|:---|:---|:---|
| `10m_wind_gust_since_previous_post_processing` | `fg10` | Max 10 m gust since last output (m/s) | gust feature |
| `2m_dewpoint_temperature` | `d2m` | 2 m dewpoint (K → converted to °C) | humidity feature |
| `boundary_layer_height` | `blh` | PBL height (m) | atmospheric-mixing feature |

### v2 pressure levels (3 variables × 2 levels = 6 columns)

> **What:** wind and geopotential high above the site, from a *different* ERA5 dataset
> that needs its own download request.
> **Why it matters most for this project:** surface wind tomorrow is largely set by
> weather systems moving overhead today. Since the goal is a 24-hour-ahead forecast, these
> upper-air columns are among the few features carrying genuine information about the
> future rather than the present.

Cache prefix: `data/raw/era5pl_{site}_{YYYY}-{MM}.nc`. This is a different CDS
dataset (`reanalysis-era5-pressure-levels`, not `reanalysis-era5-single-levels`)
requiring its own request per month.

| CDS variable | Levels | Yields tidy columns | Unit |
|:---|:---|:---|:---|
| `u_component_of_wind` | 850, 500 hPa | `wind_speed_{level}hpa`, `wind_dir_{level}hpa` | m/s, degrees |
| `v_component_of_wind` | 850, 500 hPa | (paired with u for speed/dir) | m/s |
| `geopotential` | 850, 500 hPa | `geopot_{level}hpa` | **m²/s²** |

**`geopot_*` is geopotential, not geopotential height.** ERA5's `z` is stored
verbatim in units of m² s⁻², confirmed from the cached NetCDF metadata. Divide by
g = 9.80665 to get metres. Over Anloga the 2020 means are 14,886 m²/s² (→ 1518 m)
at 850 hPa and 57,677 m²/s² (→ 5882 m) at 500 hPa.

850 hPa (~1500 m altitude over Anloga) represents the low-troposphere flow driving
the surface pressure-gradient wind. 500 hPa (~5500 m) represents the mid-troposphere
steering flow. Both are well-established predictors of surface wind in operational
meteorology.

## CDS chunking and cost math

> **What:** why the download asks for one month at a time instead of four years at once.
> **Why it matters:** the Copernicus service rejects requests it judges too large, and its
> limit is stricter than the documentation implies. Asking month by month keeps every
> request comfortably under the cap, and means a failure loses one month rather than the
> whole job. This was learned by having larger requests rejected repeatedly.

CDS enforces a per-request cost cap (currently 121,000 cost units per request). Each
request's cost roughly scales with `variables × months × days × hours × grid_cells`.
Two lessons learned the hard way:

**Single request for 4 years × 7 core variables:** ≈ 250,000 units → **rejected**
with `400 "cost limits exceeded — please reduce your selection"`. This was the first
CDS failure encountered.

**Single request for 1 year × 7 core variables:** ≈ 60,000 units → still rejected on
the first try, worked on some retries but unreliable. CDS's cost estimator is
conservative and depends on server load.

**Per-month request for 7 core variables:** ≈ 3,225 units → **safely under the cap
every time**, cache accumulates incrementally, and if any single month fails the rest
continue with retry.

Chunking strategy in [`src/data_ingest.py`](../src/data_ingest.py):

```python
months = _year_month_range(start, end)   # list of (year, month) tuples
for year, month in months:
    for family in [core, extras, pressure]:
        if cached: continue
        _retrieve_with_retry(dataset, request, cache_path, max_attempts=3)
```

For 2020–2023 (48 months × 3 families = 144 requests total):
- **Wall time:** ~7 min per request through the CDS queue → ~17 h sequential for a
  cold cache. In practice, most were served in 3–8 min; a few outliers hit 30–60
  min during European business hours.
- **Payload:** each monthly NetCDF is ~120–170 KB.
- **Cache size on disk after v3:** ~24 MB across 144 files.

## CDS licence acceptance

> **What:** a one-time manual step — you must click "accept" on the Copernicus website
> before any download works.
> **Why it is documented:** the API's error message for this is unhelpful, so without
> this note it looks like a broken setup rather than a missing click. It cannot be
> automated; the service deliberately requires a human.

**One-time manual step** — the API returns `400 required licences not accepted` until
the user visits the dataset page on the CDS website and clicks Accept on the
"Licence to use Copernicus Products" section at the bottom of the download form.

The pipeline documents this in the README but cannot automate it — CDS deliberately
requires human acknowledgement. Same licence covers both `reanalysis-era5-single-levels`
and `reanalysis-era5-pressure-levels`.

## ZIP unpacking (v2 extras quirk)

> **What:** sometimes Copernicus returns a ZIP archive of several files where a single
> weather file was expected — even though the filename says otherwise.
> **Why it matters:** the downloads all succeeded but the loading code crashed, costing
> about eight hours. The fix detects the real file type by inspecting the first four bytes
> rather than trusting the extension, so the same surprise cannot recur.

When CDS is asked for the accumulated/forecast-family variables (like
`10m_wind_gust_since_previous_post_processing`) mixed with instantaneous variables,
it returns a **ZIP archive** containing multiple NetCDF files rather than a single
NetCDF. The file we saved as `era5x_anloga_2020-01.nc` is actually a ZIP.

The loader detects this transparently via ZIP magic bytes and unpacks in a temporary
directory:

```python
def _open_dataset(path: Path):
    if _is_zip(path):
        # extract all NetCDFs inside the ZIP into a temp dir,
        # open each with xarray, merge into a single Dataset
        ...
    return xr.open_dataset(path)
```

Detection uses the 4-byte header `b"PK\x03\x04"`, not filename extension, so we're
robust to CDS changing the packaging in the future. Merged datasets are loaded
in-memory (small monthly payloads), and the temporary extraction dir is deleted after.

This quirk cost about 8 hours of pipeline runtime — the v2 downloads all succeeded
but the loader crashed. Post-fix, no re-download was needed; we just re-ran the load
step.

## Grid patch and spatial derivatives

> **What:** instead of downloading one map square, the code downloads a small patch of
> neighbouring squares and computes how each variable changes across it.
> **Why the idea is sound:** wind is caused by pressure differences *between* places, so a
> difference across the map should be a better predictor than a single point value.
> **Why it failed in practice:** the requested area was too small and snapped to just
> 1 × 2 cells. Half the resulting columns are therefore mathematically dead — a
> north-south difference cannot exist across a patch only one cell tall. Full detail
> below; this is one of the project's more instructive mistakes.

The bbox around Anloga is `pad = 0.2°` (0.4° span in each dimension). The intent was
a 3×3 patch on ERA5's 0.25° grid, but that is **not** what the requests actually
return.

**Actual patch: 1 latitude × 2 longitudes.** Read directly from the cached NetCDFs
(`era5_`, `era5x_` and `era5pl_` alike, `era5*_anloga_2020-01.nc`):

```
latitude  = [5.75]
longitude = [0.75, 1.00]
```

Anloga sits at 5.79 °N, 0.90 °E, so the latitude window [5.59, 5.99] contains only
the 5.75 grid row, and the longitude window [0.70, 1.10] contains 0.75 and 1.00.
An earlier version of this document claimed "1×2 for the extras, 3×3 for core" —
that is wrong; all three families are 1×2.

The loader keeps the full patch rather than collapsing to the nearest cell, and
computes four spatial statistics per variable. All four carry the **parent
variable's own unit** — they are plain differences across the patch, *not* divided
by the span, so they are not per-degree gradients:

- **`_mean`** — patch mean over lat/lon (a smoothed representation)
- **`_std`** — population spatial standard deviation (proxy for local heterogeneity)
- **`_grad_ns`** — north-edge mean minus south-edge mean
- **`_grad_ew`** — east-edge mean minus west-edge mean

Applied to five primary drivers to avoid feature explosion:
`wind_speed_100m`, `wind_speed_850hpa`, `geopot_850hpa`, `pressure_pa`,
`temperature_c`. That's **5 × 4 = 20 new columns** added to the tidy DataFrame.

### Consequence: half of those 20 columns are degenerate

> **What:** ten of the twenty spatial columns contain no usable information — five are
> always exactly zero, and five are just a rescaled copy of another column.
> **Why:** with only one row of cells there is no north-south direction to measure, and
> with only two columns of cells the "spread" is arithmetically half the difference
> between them. Both follow from the patch shape, not from the weather.
> **Why the impact is small:** decision-tree models ignore a column that never changes,
> and the deep-learning path already filters constant columns out. The measured cost is
> about 0.003 m/s — but any write-up should describe these features honestly.

The 1×2 patch breaks two of the four statistics. Measured over the full
2020–2023 parquet (35,064 rows):

| Statistic | State on a 1×2 patch | Evidence |
|:---|:---|:---|
| `_mean` | Informative, but ≈ the centre cell | corr 0.9967–0.9999 vs the centre-cell column |
| `_std` | **Exactly `\|_grad_ew\| / 2`** — perfectly collinear | max abs difference 0.0 to 1.2e-7 across all five |
| `_grad_ns` | **Identically zero** — n_lat = 1, so the north and south edges are the same row | `nunique == 1`, min = max = std = 0 for all five |
| `_grad_ew` | Informative — the only genuine spatial signal | std 0.13–5.6 depending on variable |

So of the 20 spatial columns, only ~10 carry independent information, and the five
`_grad_ns` columns carry none at all.

**Impact is limited but real.** The XGBoost path in
[`src/features.py`](../src/features.py) drops all-NaN columns but not constant ones,
so the zero columns are handed to the model; gradient-boosted trees ignore
zero-variance features, so results are unaffected. The deep path already guards
itself — [`src/stage1_deep.py`](../src/stage1_deep.py) filters to
`nunique(dropna=True) > 1` before scaling, precisely because constant columns wreck
the scaler's conditioning. The claim in earlier analysis that spatial derivatives
"pay off" still holds, but it rests on `_grad_ew` and `_mean` only.

**Fix, deliberately not applied:** widening `pad` to ~0.35° gives latitude
[5.44, 6.14] → 5.50/5.75/6.00 and longitude [0.55, 1.25] → 0.75/1.00/1.25, i.e. the
intended 3×3. This changes every cached download and would invalidate the parquet
and every published result, so it belongs in a planned re-ingest, not a patch. See
[08_next_steps.md](08_next_steps.md).

**Why spatial derivatives at all?** Local wind is driven by pressure gradients
(geostrophic approximation), which are spatial derivatives of pressure. Feeding
`pressure_pa` alone lets the model see the average, but `pressure_pa_grad_ew`
directly encodes a pressure-gradient magnitude that creates wind.

The centre-cell value of each variable is kept as the primary scalar (e.g.,
`wind_speed_100m`), so backwards compatibility with the v1 API is preserved.

## Open-Meteo Previous Runs — archived NWP forecasts

> **What:** an archive of *old weather forecasts* — what the major forecast models
> predicted for each hour, saved as they were issued a day or more earlier.
> **Why this is the important one:** every other source in this document describes weather
> that had already happened. To predict tomorrow you want yesterday's *forecast of*
> tomorrow, and until v5 the project had nothing of the kind. That gap turned out to be
> its main limitation.
> **The catch:** two of Open-Meteo's APIs look interchangeable and are not. Using the
> wrong one silently cheats — see the warning immediately below.

Added 2026-08. This is the only source in the project that carries an actual **forecast**;
ERA5 and NASA POWER both describe the analysed atmospheric state at time *t*. The
motivation is measured, not assumed: hub wind at t+24 is only 0.571 correlated with the
state at *t*, which caps R² at 0.326 for any model built on the current features. See
[07_findings_and_analysis.md](07_findings_and_analysis.md).

Endpoint: `https://previous-runs-api.open-meteo.com/v1/forecast`, one request per
(model, year), cached as `data/raw/openmeteo_{site}_{model}_{year}.json`.

### ⚠️ Endpoint choice is a correctness issue

> **What:** one of the two APIs gives real day-ahead forecasts; the other gives something
> that only *looks* like one.
> **Why it matters:** the Historical Forecast API assembles its record from the opening
> hours of each model run — the part that is closest to observed reality. Using that as a
> "prediction of tomorrow" hands the model a near-perfect answer, so scores look
> outstanding and mean nothing. This is a textbook case of *leakage*
> ([glossary](glossary.md)).

| API | What it returns | Usable as a t+24 feature? |
|:---|:---|:---|
| **Previous Runs** (`previous-runs-api`) | `{var}_previous_day{N}` — the forecast for the valid timestamp, issued N×24 h earlier | ✅ **Yes** — this is what we use |
| Historical Forecast (`archive-api`) | A series stitched from the **first hours of each model run** | ❌ **No** — analysis-like, effectively a nowcast. Using it is leakage. |

The Historical Forecast API is the more obvious-looking endpoint and would produce
excellent, invalid numbers. This is the same failure mode as the v3 alignment bug
(docs/10): plausible results, silently wrong, agreeing with what you hoped.

### Coverage and content (verified 2026-08-10 for Anloga)

> **What:** which dates, models and variables are actually available.
> **Why it matters:** the archive only starts in March 2024, which forced the study period
> to move. That means v5's results sit on a different stretch of time from v1–v4 and
> cannot be compared to them directly without re-running the baselines.

- **Starts 2024-03.** 2024-02 and every earlier month return all-null for wind. This is
  why [`config_nwp.yaml`](../config_nwp.yaml) uses a 2024-03 → 2025-12 window and why its
  results are **not** comparable to the v1–v4 tables without re-running baselines.
- **Three independent global models**, all confirmed working: `ecmwf_ifs025`,
  `gfs_global`, `icon_global`.
- **Variables**: `wind_speed_100m`, `wind_direction_100m`, `wind_speed_10m`,
  `temperature_2m`, `surface_pressure`, `wind_gusts_10m`. ECMWF does not serve gusts
  (100% null) — the feature builder drops all-NaN columns automatically.
- **Leads**: `_previous_day1..3`. d1 is the 24 h horizon; d2/d3 come free on the same
  request and let the model see forecast consistency across lead times.
- Fetched frame: **20,448 rows × 54 columns**, hourly, UTC, no gaps, ~2.6 MB of JSON.

Columns are named `nwp_{model}_{variable}_d{lead}` by `data_ingest.nwp_column`.

### Alignment contract (the load-bearing part)

> **What:** the rule for matching each forecast to the right row of the training table.
> **Why it is "load-bearing":** the forecast table is stamped with the time being
> *described*, while the training table is stamped with the time the prediction is *made*.
> Those differ by 24 hours. Line them up wrongly and the model either sees the future
> (cheating) or is scored against the wrong hour — the second of which already happened
> once and invalidated two full iterations
> ([10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md)).

`nwp_df` is indexed by **valid time τ**; the feature frame is indexed by **feature time
t** with target `hub(t + horizon)`. Row *t* must receive the forecast **valid at
t + horizon**, which was issued around *t* and so is legitimately known then.
`features.add_nwp_features` does this by reindexing on `df.index + horizon`, and four
tests in `tests/test_features.py` pin it — including a perfect-forecast test on a 37-hour
signal period chosen so no shifted frame can pass by coincidence.

### Known inter-model differences (not bugs)

> **What:** the three forecast models disagree with each other, substantially — one is
> 47% windier on average than another.
> **Why it is not a problem:** each model has its own systematic biases, and learning to
> correct them is precisely what the machine-learning step is for. Three disagreeing
> models are more informative than one, because their spread hints at how uncertain the
> forecast is.
> **Why this section exists anyway:** one difference looked like a code bug and was
> checked five different ways before being accepted as real. The reasoning is recorded so
> nobody "fixes" a genuine physical signal later.

Measured over the full 2024-03 → 2026-06 fetch:

| Model | mean 100 m wind | diurnal peak (UTC) | diurnal trough |
|:---|---:|---:|---:|
| `ecmwf_ifs025` | 4.50 m/s | 18.0 | 11.5 |
| `icon_global` | 5.12 m/s | 17.7 | 8.3 |
| `gfs_global` | 6.62 m/s | 17.0 | 9.0 |

GFS is ~47% windier than ECMWF in the mean, and the models disagree on the sea-breeze
trough by up to 3 hours. Cross-correlating ECMWF against the other two peaks at lag +1 h
rather than 0, but the correlation curve is flat-topped (r 0.6420 at lag 0 vs 0.6553 at
+1) and ECMWF's peak and trough are displaced by *different* amounts — a uniform
timestamp offset would displace both equally. This is genuine model disagreement about
diurnal phase, and it is precisely the systematic error the ML bias-correction step
exists to absorb.

## NASA POWER

> **What:** the backup weather source — one web request, no account needed.
> **Why keep it:** it works if Copernicus is unreachable, and it provides an independent
> check that conclusions are not an artefact of one particular dataset. Its limits are
> real though: coarser cells, wind at 50 m instead of 100 m, and none of the extras.

Simpler fetcher — single HTTP request per site+date range, no chunking, no auth.
Only used as a cross-source sanity check now (v1 used it briefly).

Endpoint: `https://power.larc.nasa.gov/api/temporal/hourly/point`

Parameters requested: `WS10M, WS50M, WD10M, WD50M, T2M, PS`. NASA POWER provides wind at
10 m and 50 m only — verified 2026-08-11 that `WS80M` and `WS100M` do not exist. WS50M is
stored in the `wind_speed_100m` column so the tidy schema stays uniform across sources.

> ⚠️ **Corrected 2026-08-11.** This section previously claimed the log-law extrapolation
> "reaches the 100 m hub height". **It did not.** `extrapolate_hub_wind` defaulted to a
> 100 m reference height for every source, so NASA POWER's 50 m wind was treated as
> already being at hub height and no correction was applied — understating hub wind by
> **10.7%** (7.00 m/s at 50 m stayed 7.00 instead of becoming 7.8352).
> **v1's published results rest on that understated target.**
> Fixed by `SOURCE_WIND_HEIGHT_M` in `src/data_ingest.py`, which the feature builder now
> consults instead of assuming. ERA5 is unaffected — its column really is at 100 m, and
> its 2020–2023 baseline was re-verified at exactly 1.1410 m/s after the change.

> ⚠️ **NASA POWER also carries a ~1-hour timestamp offset relative to ERA5.** Measured
> over 8,760 overlapping hours: the two truth sources cross-correlate at peak lag −1 h,
> and every NWP model's lag against NASA POWER is exactly 1 h more negative than against
> ERA5. MERRA-2 hourly values are hour-averaged rather than instantaneous. This makes
> NASA POWER unsuitable as a verification target for forecast alignment without
> correcting for it.

NASA POWER returns pressure in kPa; the code converts to Pa. Sentinel `-999.0`
values are converted to NaN. No roughness field, no upper-air, no gusts, no BL
height — all v2 extras are NaN in the tidy DataFrame for NASA POWER runs, and the
feature builder drops these all-NaN columns automatically.

## Tidy DataFrame schema

> **What:** the single table every download is converted into, whichever source it came
> from.
> **Why it matters:** downstream code never needs to know where the data originated. If a
> source cannot supply a column it is filled with blanks and dropped automatically, so
> switching sources requires no code changes anywhere else.

Regardless of source, `load_weather(config)` returns a `pd.DataFrame` with:

- **Index:** `pd.DatetimeIndex` in UTC, hourly frequency, name `timestamp`, no
  duplicates.
- **Columns** (in this order):

```
CORE_COLUMNS = [
    "wind_speed_10m", "wind_speed_100m",
    "wind_dir_10m", "wind_dir_100m",
    "temperature_c", "pressure_pa", "roughness",
]

EXTRA_SINGLE_COLUMNS = [                 # NaN for NASA POWER
    "wind_gust_10m", "dewpoint_c", "bl_height_m",
]

PRESSURE_LEVEL_COLUMNS = [               # NaN for NASA POWER
    "wind_speed_850hpa", "wind_dir_850hpa", "geopot_850hpa",
    "wind_speed_500hpa", "wind_dir_500hpa", "geopot_500hpa",
]

# Spatial derivatives for 5 primary vars × 4 stats
SPATIAL_COLUMNS = [
    "wind_speed_100m_mean", "wind_speed_100m_std",
    "wind_speed_100m_grad_ns", "wind_speed_100m_grad_ew",
    ... (same pattern for wind_speed_850hpa, geopot_850hpa,
         pressure_pa, temperature_c)
]
```

Total columns for ERA5 v2: **7 core + 3 extras + 6 pressure-level + 20 spatial = 36
columns**. For NASA POWER: 7 core (rest NaN → dropped by feature builder).

Written to `data/processed/{source}_{site}_{start}_{end}.parquet` after ingestion,
so subsequent pipeline runs skip the parquet write and go straight to loading.

## u/v to speed/direction conversion

> **What:** turning ERA5's two wind components (how fast east, how fast north) into a
> speed and a compass direction.
> **Why it needs its own section:** meteorology names a wind by where it *comes from*, not
> where it is going — a "westerly" blows *from* the west. Getting that backwards flips
> every direction by 180°, which is easy to do and hard to notice, so the formula is
> written out and pinned by a test.

Wind is stored in ERA5 as eastward (u) and northward (v) vector components. To get
scalar wind speed and meteorological direction (direction the wind is *from*):

```python
speed = sqrt(u² + v²)
direction = (270° - atan2(v, u) in degrees) mod 360°
```

The `270° -` and mod 360° convert from mathematical convention (angle east of x-axis,
counterclockwise, pointing to *where the wind is going*) to meteorological convention
(angle from north, clockwise, indicating *where the wind comes from*). This is the
standard used by WMO and all downstream libraries including `windpowerlib`.

Verified in [tests/test_data_ingest.py](../tests/test_data_ingest.py): pure eastward
wind (u>0, v=0) → direction 270° ("wind from the west"), pure northward wind (v>0)
→ direction 180° ("wind from the south").
