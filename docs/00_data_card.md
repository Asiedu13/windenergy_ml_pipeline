# 00 — Data Card

Single-page answer to "what data is in this project, where did it come from, and what
are the units?" Every figure below was read from the code or measured from the files on
disk, not copied from prose.

For the *narrative* — CDS cost math, chunking strategy, the ZIP-unpacking quirk, why
ERA5 over NASA POWER — see [03_data_sources.md](03_data_sources.md). This page is the
lookup table.

**Cache state as of 2026-08-10.** Re-measure after any ingest run and update the dates.

---

## 1. Sources

> **What:** where every number in this project originally came from — four external
> sources in total.
> **Why it matters:** none of this data was measured at the site. It is all downloaded
> from public archives, and knowing which archive a column came from tells you how much
> to trust it. Terms like *reanalysis* and *NWP* are defined in [glossary.md](glossary.md).

| # | Source | Provider / endpoint | Resolution | Auth | Role |
|---:|:---|:---|:---|:---|:---|
| 1 | **ERA5 reanalysis** | ECMWF via Copernicus CDS API (`cdsapi`) | 0.25°, hourly, UTC | CDS account + `~/.cdsapirc`, one-time licence click | **Primary.** All v1–v4 results. |
| 2 | **NASA POWER** | NASA Langley, `https://power.larc.nasa.gov/api/temporal/hourly/point` | 0.5° × 0.625°, hourly, UTC | None | Fallback + cross-source sanity check. |
| 3 | **Turbine power curve** | OpenEnergy Database (oedb) via `windpowerlib.WindTurbine` | — | None | Stage 2 simulator. Fetched by the library, not stored in-repo. |
| 4 | **Archived NWP forecasts** | Open-Meteo Previous Runs, `previous-runs-api.open-meteo.com/v1/forecast` | Point, hourly, UTC; **2024-03 onward** | None | **The only genuine forecast in the project.** Feeds `config_nwp.yaml`. |

**Site:** Anloga, Ghana — **5.79 °N, 0.90 °E**, hub height **100 m**
([`config.yaml`](../config.yaml)).
**Turbine:** Enercon **E-101/3050**, rated 3.05 MW, hub 100 m.

ERA5 is fetched as three separate CDS request families so new variables can be added
without invalidating older downloads:

| Cache prefix | CDS dataset | Contents |
|:---|:---|:---|
| `data/raw/era5_{site}_{YYYY}-{MM}.nc` | `reanalysis-era5-single-levels` | 7 core variables (v1) |
| `data/raw/era5x_{site}_{YYYY}-{MM}.nc` | `reanalysis-era5-single-levels` | 3 extras (v2) — arrives ZIP-wrapped |
| `data/raw/era5pl_{site}_{YYYY}-{MM}.nc` | `reanalysis-era5-pressure-levels` | 3 variables × 850/500 hPa (v2) |

NASA POWER is one HTTP request, parameters `WS10M,WS50M,WD10M,WD50M,T2M,PS`, cached as
a single JSON.

---

## 2. What is actually on disk

> **What:** which time periods have actually been downloaded, and which have not.
> **Why it matters:** the intended coverage and the real coverage differ. There is a
> deliberate hole in the record, and one config file points at a range that cannot
> currently run. Check here before trusting any date range.

> **File counts below are a snapshot taken 2026-08-10 and grow while a download runs.**
> An earlier revision of this page quoted a live-growing cache as a static fact and was
> wrong within minutes. Re-measure before quoting; the live check is in
> [11_nwp_forecast_features.md §8](11_nwp_forecast_features.md).

| Location | Contents | Size |
|:---|:---|---:|
| `data/raw/` | ~385 monthly NetCDFs (127 complete months × 3 families, plus the 2024–2026 pull in progress) + 1 NASA POWER JSON + 9 Open-Meteo forecast JSONs | ~56 MB |
| `data/processed/era5_anloga_2020-01-01_2023-12-31.parquet` | **35,064 rows × 36 cols**, 2020-01-01 00:00 → 2023-12-31 23:00 UTC, zero all-NaN columns | 5.3 MB |
| `data/processed/nasa_power_anloga_2023-01-01_2023-06-30.parquet` | Cross-check slice, 2023 H1 | 0.09 MB |

**ERA5 month coverage — per-site cache (`era5_anloga_*`, the v1–v5 path):**

```
2010-01 ──────────────► 2016-07     ✓ complete (backfill halted here, deliberately)
2016-08 ──────────────► 2019-12     ✗ MISSING (41 months, 123 requests)
2020-01 ──────────────► 2023-12     ✓ complete (48 months, all 3 families)
2024-03 ──────────────► 2025-12     ⟳ downloading for config_nwp.yaml (22 months)
```

### 2.1 Regional cache (`era5gh_*`, v6 multi-site) — a fourth family

Added 2026-10-01 for the 16-regional-capital extension
([13](13_multi_site_extension.md)). Keyed by **month only, not by site**, because one
bounding-box download serves every site inside it — CDS charges per field rather than per
grid cell, so 17 sites cost one site's requests instead of seventeen.

| File | Contents |
|:---|:---|
| `era5gh_core_{YYYY}-{MM}.nc` | the 7 v1 single-level variables over the whole bbox |
| `era5gh_extra_{YYYY}-{MM}.nc` | the 3 v2 extras over the bbox |
| `era5gh_pressure_{YYYY}-{MM}.nc` | 3 variables × 2 pressure levels over the bbox |
| `era5gh_lsm.nc` | land-sea mask, one hour, ~25 KB, reused for the whole archive |
| `era5gh_manifest.json` | the bbox these files were built for |

```
bbox (N, W, S, E) = 11.5, -3.5, 4.25, 1.5   →  30 lat × 21 lon = 630 cells
target range      = 2010-01 ──► 2025-12      (192 months × 3 families = 576 requests)
```

**Measured, not estimated:** 6.3 MB per core month, ~4.6–5.8 min per request, so ≈2.7 GB and
≈2 days for the full archive. **This range has no hole** — it covers the 41 months
(2016-08 → 2019-12) the per-site backfill deliberately skipped.

⚠️ **The filenames do not encode the bbox.** A cache built for one box would otherwise be
silently reused for another and every spatial column would describe the wrong neighbourhood —
the same class of defect as the Open-Meteo `lead_days` leak
([register entry B4 / defect 9](../redo/results/06_what_the_protocol_caught.md)), where a cache
key encoded less than the request. `era5gh_manifest.json` is the guard and
`_check_regional_manifest` raises rather than mixing grids.

**Live coverage check** (file counts are never written into this page as facts — see the
warning above):

```bash
powershell -c "(Get-ChildItem data/raw/era5gh_core_* | Measure-Object).Count"   # of 192
powershell -c "Get-Content outputs/logs/era5gh_archive.log -Tail 5"
python -m src.multisite cells      # which cell each site resolved to, and its land fraction
```

**Status 2026-10-09 — this is now a *used* input, not a planned one.** The **forecast window
(2024-03 → 2025-12) is complete** and the 24 h ladder has run at all 17 sites off it
([`../redo/results/13`](../redo/results/13_multi_site_generalisation.md)). The **long archive is
not complete**, so the monthly resource climatology has not run and **no capacity factor exists
for any of the 16 capitals.** Check coverage with the commands above rather than trusting a
count written here.

⚠️ **A forecast run will never download.** `fetch_regional_archive(download=False)` plus a
preflight in `multisite.cmd_forecast` make a missing month a hard stop, because the default
`--start` previously reached past the cached window and silently turned a training run into a
432-request CDS ingest. See [13 §10](13_multi_site_extension.md).

**Why the backfill was stopped (2026-08-10).** It ran from 2026-08-08 to 2026-08-10 and
reached 2016-07. It was then halted because the learning curve in
[07_findings_and_analysis.md](07_findings_and_analysis.md) showed more history is worth
~1% MAE, and the CDS queue was more valuable pointed at the 2024–2026 window that
`config_nwp.yaml` needs. Nothing is corrupted: the halt landed on a complete month across
all three families, and `python -m src.data_ingest_extend --config config_10yr.yaml`
resumes from 2016-08 whenever a multi-year scaling study is wanted.

Consequences:

- **Every result in [`../RESULTS.md`](../RESULTS.md) rests on the 2020–2023 parquet.**
- [`config_10yr.yaml`](../config_10yr.yaml) declares `2010-01-01 → 2023-12-31` and
  **cannot be handed to `python -m src.pipeline`** — `_fetch_era5` raises as soon as a
  *core* month is missing.
- [`config_nwp.yaml`](../config_nwp.yaml) uses **2024-03 → 2025-12** (22 months), a
  different window, so its results are **not** comparable to the v1–v4 tables without
  re-running the baselines on that window. *(Shortened from 2026-06 on 2026-08-11 — the
  ERA5 download for this period runs at ~27 min/request, and the learning curve in
  [07](07_findings_and_analysis.md) flattens well before the extra months would matter.)*

**Never run two ingest processes at once.** Existence of the target file is the only
guard against re-downloading, so two concurrent runs pick the same missing month, issue
duplicate CDS requests, and can write the same path simultaneously. This happened on
2026-08-10. Check first:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
```

---

## 3. Tidy schema — 36 columns

> **What:** the cleaned-up weather table the rest of the pipeline reads — one row per
> hour, 36 columns, with the unit of every column.
> **Why it matters:** this is the reference you will come back to most. Getting a unit
> wrong here silently corrupts everything downstream, and two columns in particular
> (§3.3 geopotential, §3.4 gradients) are easy to misread.

**How to read a row:** each row is one hour, timestamped in UTC. The columns describe the
atmosphere over Anloga during that hour — wind at two heights, temperature, pressure, and
so on. "Tidy" just means one observation per row with one variable per column.

`load_weather(config)` returns the same frame regardless of source: a
`pd.DatetimeIndex` named `timestamp`, **UTC**, hourly, de-duplicated. Column order is
`TIDY_COLUMNS` in [`src/data_ingest.py`](../src/data_ingest.py).

"POWER" column below = available from NASA POWER; ✗ means the column exists but is
all-NaN, which the feature builder drops automatically.

### 3.1 Core (7)

> **What:** the seven basic weather measurements available from every source.
> **Why:** `wind_speed_100m` is the single most important column in the project — it is
> the closest thing to what the turbine actually experiences, and the model leans on it
> more than anything else.

| Column | Meaning | Unit | ERA5 var | POWER |
|:---|:---|:---|:---|:---:|
| `wind_speed_10m` | Wind speed at 10 m | m/s | `u10`,`v10` → √(u²+v²) | ✓ `WS10M` |
| `wind_speed_100m` | Wind speed at 100 m — primary Stage 1 input | m/s | `u100`,`v100` | ~ `WS50M` (50 m, not 100 m) |
| `wind_dir_10m` | Direction at 10 m, meteorological (*from*) | degrees, 0–360 | `u10`,`v10` | ✓ `WD10M` |
| `wind_dir_100m` | Direction at 100 m, meteorological | degrees, 0–360 | `u100`,`v100` | ~ `WD50M` |
| `temperature_c` | Air temperature at 2 m | **°C** (converted from K) | `t2m` | ✓ `T2M` |
| `pressure_pa` | Surface pressure | **Pa** | `sp` | ✓ `PS` (kPa → ×1000) |
| `roughness` | Surface roughness length | m | `fsr` | ✗ |

### 3.2 ERA5 extras (3)

> **What:** three extra surface variables ERA5 provides but NASA POWER does not.
> **Why:** gusts, humidity and boundary-layer depth all describe how turbulent and
> well-mixed the air is — conditions that change how much of the wind aloft reaches
> turbine height.

| Column | Meaning | Unit | ERA5 var | POWER |
|:---|:---|:---|:---|:---:|
| `wind_gust_10m` | Max 10 m gust since previous post-processing | m/s | `fg10` | ✗ |
| `dewpoint_c` | Dewpoint at 2 m | **°C** (converted from K) | `d2m` | ✗ |
| `bl_height_m` | Boundary-layer height | m | `blh` | ✗ |

### 3.3 Pressure levels (6)

> **What:** wind and geopotential high above the site — roughly 1.5 km and 5.9 km up.
> **Why:** surface wind is largely driven from above. Weather systems steering the
> atmosphere at altitude today shape the surface wind tomorrow, which is exactly the
> 24-hour-ahead signal this project needs.
> ⚠️ **The trap:** `geopot_*` is geopotential in **m²/s²**, not a height in metres.

850 hPa ≈ 1.5 km (low-troposphere flow driving the surface pressure gradient);
500 hPa ≈ 5.9 km (mid-troposphere steering flow).

| Column | Meaning | Unit | ERA5 var | POWER |
|:---|:---|:---|:---|:---:|
| `wind_speed_850hpa` / `wind_speed_500hpa` | Wind speed at level | m/s | `u`,`v` | ✗ |
| `wind_dir_850hpa` / `wind_dir_500hpa` | Direction at level, meteorological | degrees, 0–360 | `u`,`v` | ✗ |
| `geopot_850hpa` / `geopot_500hpa` | **Geopotential** | **m² s⁻²** | `z` | ✗ |

> ⚠️ `geopot_*` is **geopotential, not geopotential height.** ERA5's `z` is passed
> through unchanged; the NetCDF attribute reads `units = m**2 s**-2`. Divide by
> g = 9.80665 for metres. Measured 2020 means over Anloga: 14,886 m²/s² (→ 1518 m) at
> 850 hPa, 57,677 m²/s² (→ 5882 m) at 500 hPa.

### 3.4 Spatial statistics (20)

> **What:** twenty columns describing how each weather variable *varies across the map*
> around the site, rather than just its value at one point.
> **Why the idea is sound:** wind is created by pressure differences between places, so
> a difference across the map should predict wind better than a single reading.
> **Why it did not work here:** the downloaded map patch turned out to be only 1 cell tall
> and 2 wide, which is too small for the idea to function. Ten of these twenty columns
> carry no usable information at all — the details below are worth reading before you
> quote any of them.

Five drivers × four statistics: `{wind_speed_100m, wind_speed_850hpa, geopot_850hpa,
pressure_pa, temperature_c}` × `{_mean, _std, _grad_ns, _grad_ew}`.

**Units:** each statistic carries its parent variable's unit (m/s, Pa, °C, m² s⁻²).
They are plain differences across the grid patch — **not** divided by the patch span,
so they are *not* per-degree gradients despite the name.

| Suffix | Definition | Status on the current 1×2 patch |
|:---|:---|:---|
| `_mean` | Patch mean over lat/lon | Informative, but corr **0.9967–0.9999** with the centre-cell column |
| `_std` | Population spatial std | **Exactly `\|_grad_ew\| / 2`** — perfectly collinear (max abs diff 0 to 1.2e-7) |
| `_grad_ns` | North-edge mean − south-edge mean | **Identically zero** — `nunique == 1`, min = max = std = 0 |
| `_grad_ew` | East-edge mean − west-edge mean | Informative — the only genuine spatial signal |

**Why:** the `pad = 0.2°` bbox snaps to `latitude = [5.75]`, `longitude = [0.75, 1.00]`
— a **1 × 2** patch, not the 3 × 3 the code was written for. With one latitude row the
north and south edges are the same row, so `_grad_ns` is structurally zero; with two
longitude cells the population std is algebraically half the east–west difference.
Verified against all 35,064 rows of the 2020–2023 parquet.

Practical impact is small: the XGBoost path passes the zero columns to the model and
gradient-boosted trees ignore zero-variance features, while
[`src/stage1_deep.py`](../src/stage1_deep.py) already filters to `nunique > 1` before
scaling. But **only ~10 of the 20 spatial columns carry independent information**, and
any write-up should say so. Widening `pad` to ~0.35° would give a true 3×3 patch at the
cost of a full re-ingest — see [03_data_sources.md](03_data_sources.md) and
[08_next_steps.md](08_next_steps.md).

> ✅ **Fixed for `era5_regional` (v6), measured 2026-10-01.** The multi-site path downloads a
> Ghana-wide bbox and extracts a genuine **3 × 3** window per site, so on that path:
>
> | Suffix | Status on the 1×2 per-site patch | Status on the 3×3 regional patch |
> |:---|:---|:---|
> | `_grad_ns` | identically zero | **alive** — sd 0.71 (wind 100 m) to 14.6 (pressure) |
> | `_std` | `= \|_grad_ew\|/2` | **no longer collinear** |
>
> **All twenty columns carry information on the regional path.** Verified on 2024-06 against
> the per-site cache ([13 §7a](13_multi_site_extension.md)).
>
> Two caveats. (1) **It is a different feature set**, so regional and per-site results are not
> directly comparable on anything spatial. (2) **The "exactly" above is slightly too strong**
> even for the 1×2 patch: three of the five `_std` columns match `|_grad_ew|/2` to 0.0e+00,
> but the two *derived* wind-magnitude columns (`wind_speed_100m_std`,
> `wind_speed_850hpa_std`) agree only to 8.0e-05 and 8.9e-06 relative, because `np.std` over a
> 2-cell float32 array of `sqrt(u²+v²)` rounds where a raw field does not. Functionally
> collinear either way — the substantive claim is unchanged.

> ⚠️ **"Fixed" overstates it — superseded 2026-10-09.** Everything in the box above is true of
> the **tidy frame** and false of **what any model receives**. `is_dead_column`
> ([`../src/traditional.py`](../src/traditional.py)) matches on the `_grad_ns` / `_std` **name
> suffix** and withholds all ten unconditionally, so the `study` feature set keeps 26 columns on
> the regional path exactly as on the per-site one. The revived columns were never reaching a
> model, and nothing in the outputs said so — both paths even report the same 104-column
> feature frame.
>
> Measured rather than argued: a `features.dead_columns: value` mode was added and run at two
> sites. **Every model's MAE moved less than its own seed sd (22 of 22 rows, both sites), with
> inconsistent sign**, and the `operational` arm moved by exactly 0.0000 — as it must, having no
> spatial columns at all. So the 630-cell download bought spatial realism in the data and **no
> measurable forecast skill at 24 h**, and `name` remains the default so published figures stay
> comparable. See [13 §4d](13_multi_site_extension.md).

---

## 3.5 NWP forecast columns (`config_nwp.yaml` only)

> **What:** actual weather *forecasts* — what three major forecast models predicted for
> each hour, 1, 2 and 3 days beforehand.
> **Why it matters:** everything in §3.1–3.4 describes weather that had already happened.
> These are the first columns in the project that contain a genuine prediction of the
> future, which is what a day-ahead forecast fundamentally needs. Added in v5; see
> [11_nwp_forecast_features.md](11_nwp_forecast_features.md).

Separate parquet: `data/processed/nwp_{site}_{start}_{end}.parquet`.

The **fetched** Open-Meteo data spans **2024-03-01 → 2026-06-30** — 20,448 rows × 54
columns, hourly UTC, indexed by **valid time**, no gaps. That full fetch stays on disk.
The **study window** is shorter (2024-03 → 2025-12); `load_nwp_forecasts` slices to the
config's date range, so a run writes a 22-month parquet from the same cached JSON with no
re-fetch. Do not confuse the two numbers.

Naming: `nwp_{model}_{variable}_d{lead}` (`data_ingest.nwp_column`).

| Component | Values |
|:---|:---|
| `model` | `ecmwf_ifs025`, `gfs_global`, `icon_global` |
| `variable` | `wind_speed_100m` (m/s), `wind_direction_100m` (deg), `wind_speed_10m` (m/s), `temperature_2m` (°C), `surface_pressure` (hPa), `wind_gusts_10m` (m/s) |
| `lead` | `d1`, `d2`, `d3` — forecast issued 1/2/3 × 24 h **before** the valid time |

Wind speeds are requested with `wind_speed_unit=ms`, so they are **m/s, not km/h**.
Note `surface_pressure` here is **hPa** (Open-Meteo's unit), whereas ERA5's
`pressure_pa` is **Pa** — a factor of 100 apart. Do not mix them.

`nwp_ecmwf_ifs025_wind_gusts_10m_d{1,2,3}` are **100% null** (ECMWF IFS does not serve
gusts through this API); `build_feature_frame` drops all-NaN columns automatically. Every
other column is 0–1% null.

**Alignment:** row *t* of the feature frame receives the forecast valid at
*t + horizon_hours*, issued around *t*. See `features.add_nwp_features` and
[03_data_sources.md](03_data_sources.md).

---

## 4. Derived quantities (not in the parquet)

> **What:** columns the code calculates on the fly at training time rather than storing.
> **Why they exist:** the raw data gives wind at 10 m and 100 m, but the turbine sits at
> 100 m and the model needs a sense of recent history and time of day. These are those
> conversions — height extrapolation, recent-past values, and clock/calendar position.

Built at train time by [`src/features.py`](../src/features.py) and
[`src/stage2_powercurve.py`](../src/stage2_powercurve.py).

| Quantity | Meaning | Unit | Notes |
|:---|:---|:---|:---|
| `wind_speed_hub` | Wind speed at 100 m hub | m/s | `windpowerlib` log-law from `wind_speed_100m`; `roughness` with fallback **0.15 m** (`DEFAULT_ROUGHNESS_M`) |
| `wind_dir_*_sin` / `_cos` | Circular direction encoding | dimensionless, [−1, 1] | Raw `wind_dir_*` columns are excluded from `X` |
| `hour`, `month` | Calendar position | 0–23, 1–12 | Excluded from `X`; only their sin/cos are used |
| `hour_sin/cos`, `month_sin/cos` | Cyclical calendar encoding | dimensionless, [−1, 1] | |
| `wind_speed_hub_lag{1,2,3,6,12}` | Lagged hub wind speed | m/s | `model.lags` in `config.yaml` |
| `wind_speed_hub_rollmean{3,6}` | Rolling mean, shifted 1 h | m/s | Shift prevents leaking the current timestep |
| `wind_speed_hub_rollstd{3,6}` | Rolling std, shifted 1 h | m/s | `min_periods = w`, so no partial windows |
| **Target** `wind_speed_hub_target` | Hub wind speed at t (nowcast) or t + 24 h (forecast) | m/s | `forecast.horizon_hours = 24` |
| `power_w` | Stage 2 turbine power | **W** | oedb power curve, density-corrected |
| air density | Ideal-gas dry air | kg/m³ | `p / (R·T)`, R = 287.058 J·kg⁻¹·K⁻¹ |

**Wind-direction convention:** meteorological throughout — the direction the wind comes
*from*, degrees clockwise from north. Converted from ERA5's u/v components by
`direction = (270° − atan2(v, u)) mod 360°`. Pure eastward wind → 270°; pure northward
wind → 180°. Asserted in `tests/test_data_ingest.py`.

---

## 5. Known data limitations

> **What:** six honest weaknesses in the data, stated plainly.
> **Why it matters:** these bound what the project can legitimately claim. Anyone
> assessing this work will look for exactly this list, and it is far stronger to state
> the limits yourself than to have them found for you.

Carry these into any write-up:

1. **No ground truth.** No operating wind farm in Ghana publishes SCADA data, so the
   target is ERA5-derived hub wind speed, not measured output. The pipeline forecasts
   *potential* generation. Errors are relative to reanalysis, not to reality.
2. **Effectively a single grid cell.** 1 × 2 ERA5 cells at 0.25° — no meaningful
   spatial diversity, which is the direct cause of the degenerate columns in §3.4.
   *(Fixed on the v6 regional path, which extracts a true 3 × 3 — see §3.4.)*
3. **Four usable years, not ten.** 2020–2023 only; the backfill stalled at 2016-03 (§2).
4. **Reanalysis is not a forecast.** ERA5 assimilates observations after the fact, so
   a "day-ahead" forecast here has access to a cleaner atmospheric state than an
   operational NWP forecast would. This flatters the reported scores.
5. **`geopot_*` and the `_grad_*` columns are easy to misreport** — see the unit
   warnings in §3.3 and §3.4.
6. ⚠️ **Anloga's ERA5 cell is 88% water** (land fraction **0.124**, measured 2026-10-01
   against `era5gh_lsm.nc`). Surface roughness in the parquet confirms it independently:
   mean **z₀ = 0.000063 m**, where open sea is 0.0001–0.0002 and cropland 0.05–0.10 — about
   a thousand times rougher. ERA5 at 25 km cannot resolve the ~2 km sandspit between the
   Keta Lagoon and the Gulf of Guinea, so the cell describes the water around the site
   rather than the site.
   **What this bounds:** *forecast* metrics stay valid (truth and features are the same
   cell, so the comparisons are internally sound), but **resource and power levels are
   overstated with a known sign** — lower roughness, faster wind, larger apparent resource
   — and the benchmark comparison is **flattered**, because marine wind is smoother and
   more persistent than the land wind the reference study measured. How much is **not
   quantified**; that needs a land-cell or mast comparison. The cell is deliberately left
   unchanged so published figures keep the data they were computed on.
   → [13 §3a](13_multi_site_extension.md)
