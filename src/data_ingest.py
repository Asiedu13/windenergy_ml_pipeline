"""Data ingestion for ERA5 (CDS) and NASA POWER.

Both fetchers return the same tidy hourly DataFrame with a UTC-aware
DatetimeIndex. Core columns:

    wind_speed_10m, wind_speed_100m,
    wind_dir_10m, wind_dir_100m,
    temperature_c, pressure_pa, roughness

ERA5 additionally provides (v2 extension):
    wind_gust_10m, dewpoint_c, bl_height_m,
    wind_speed_850hpa, wind_dir_850hpa, geopot_850hpa,
    wind_speed_500hpa, wind_dir_500hpa, geopot_500hpa,
    <var>_mean, <var>_std, <var>_grad_ns, <var>_grad_ew for a handful of
        primary drivers (spatial statistics over the grid patch).

Units are the ERA5 native ones except where converted here: temperature and
dewpoint are K->degC, everything else is passed through (m/s for winds, Pa for
pressure, m for roughness / BL height). See docs/00_data_card.md for the full
per-column table.

Extras and pressure-levels are missing (all-NaN) for NASA POWER, which the
downstream feature builder drops automatically.

Caches are split into three families so v1 data is not re-downloaded:
    era5_{site}_{YYYY}-{MM}.nc    — original 7 single-level variables (v1)
    era5x_{site}_{YYYY}-{MM}.nc   — extra single-level variables (v2)
    era5pl_{site}_{YYYY}-{MM}.nc  — pressure-level variables (v2)

The multi-site extension (v6, `source: era5_regional`) adds a fourth family keyed
by month rather than site, because one bounding-box download serves every site in
it — CDS charges per field, not per grid cell, so 17 sites cost one site's
requests. See `_fetch_era5_regional` and docs/13_multi_site_extension.md.
    era5gh_core_{YYYY}-{MM}.nc      — the 7 v1 variables over the whole bbox
    era5gh_extra_{YYYY}-{MM}.nc     — v2 extras over the bbox
    era5gh_pressure_{YYYY}-{MM}.nc  — pressure levels over the bbox
    era5gh_lsm.nc                   — land-sea mask, one hour, reused forever
    era5gh_manifest.json            — the bbox these files were built for
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .utils import Config, SiteConfig, ensure_dir, get_logger

logger = get_logger(__name__)

# Height at which each source's "upper level" wind is actually measured.
#
# ⚠️ This exists because the tidy schema names the column `wind_speed_100m`
# regardless of source, which is only literally true for ERA5. NASA POWER's
# highest wind level is **50 m** (verified 2026-08-11: WS50M/WD50M exist,
# WS80M/WS100M do not), and its values are stored in that same column.
#
# `features.extrapolate_hub_wind` previously defaulted to a 100 m reference
# height for every source, so NASA POWER's 50 m wind was treated as already
# being at hub height and no log-law correction was applied — understating hub
# wind by ~10.7%. v1's published results carry that error. Look the height up
# here rather than assuming.
SOURCE_WIND_HEIGHT_M = {
    "era5": 100.0,
    # Same product, different download geometry — listed explicitly rather than
    # left to DEFAULT_SOURCE_WIND_HEIGHT_M, because an implicit default is what
    # caused the v1 error this table exists to prevent.
    "era5_regional": 100.0,
    "nasa_power": 50.0,
}
DEFAULT_SOURCE_WIND_HEIGHT_M = 100.0


def source_wind_height_m(source: str) -> float:
    """Measurement height of the `wind_speed_100m` column for a given source."""
    return SOURCE_WIND_HEIGHT_M.get(source.lower(), DEFAULT_SOURCE_WIND_HEIGHT_M)


# Base columns (present in every source; NaN where unavailable).
CORE_COLUMNS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_10m",
    "wind_dir_100m",
    "temperature_c",
    "pressure_pa",
    "roughness",
]

# v2 extras — populated by ERA5 only.
EXTRA_SINGLE_COLUMNS = [
    "wind_gust_10m",
    "dewpoint_c",
    "bl_height_m",
]

# NOTE: geopot_* is ERA5's `z` passed through unchanged — geopotential in
# m**2 s**-2, NOT geopotential height in m. Divide by g = 9.80665 for metres
# (850 hPa ~ 14,900 -> ~1520 m; 500 hPa ~ 57,700 -> ~5880 m over Anloga).
PRESSURE_LEVEL_COLUMNS = [
    "wind_speed_850hpa",
    "wind_dir_850hpa",
    "geopot_850hpa",
    "wind_speed_500hpa",
    "wind_dir_500hpa",
    "geopot_500hpa",
]

# Which variables get spatial-derivative expansion, and what stats.
SPATIAL_VARS = [
    "wind_speed_100m",
    "wind_speed_850hpa",
    "geopot_850hpa",
    "pressure_pa",
    "temperature_c",
]
SPATIAL_STATS = ["mean", "std", "grad_ns", "grad_ew"]


def spatial_columns() -> list[str]:
    return [f"{v}_{s}" for v in SPATIAL_VARS for s in SPATIAL_STATS]


def all_columns() -> list[str]:
    return CORE_COLUMNS + EXTRA_SINGLE_COLUMNS + PRESSURE_LEVEL_COLUMNS + spatial_columns()


TIDY_COLUMNS = all_columns()


# ---------------------------------------------------------------------------
# u,v conversion + generic helpers
# ---------------------------------------------------------------------------

def _uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u,v (east/north components in m/s) -> speed, meteorological direction (deg)."""
    speed = np.sqrt(u * u + v * v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


def _year_month_range(start: str, end: str) -> list[tuple[int, int]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    months = pd.period_range(start=s.to_period("M"), end=e.to_period("M"), freq="M")
    return [(p.year, p.month) for p in months]


def _nasa_cache_path(cache_dir: Path, site: SiteConfig, start: str, end: str) -> Path:
    return cache_dir / f"nasa_power_{site.name}_{start}_{end}.json"


# ---------------------------------------------------------------------------
# ERA5 — single-level + extras + pressure-levels, chunked per month
# ---------------------------------------------------------------------------

# v1 single-level variables (kept in the era5_ cache family so old downloads
# are not invalidated).
ERA5_CORE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "forecast_surface_roughness",
]

# v2 extras — separate cache family so we can add them without touching v1 files.
ERA5_EXTRA_VARS = [
    "10m_wind_gust_since_previous_post_processing",
    "2m_dewpoint_temperature",
    "boundary_layer_height",
]

ERA5_PRESSURE_VARS = ["u_component_of_wind", "v_component_of_wind", "geopotential"]
ERA5_PRESSURE_LEVELS = ["850", "500"]

# Regional (multi-site) cache family. Keyed by MONTH ONLY, deliberately: these
# files are shared by every site in the bbox, which is the entire reason 17 sites
# cost one site's download. The per-site `era5_{name}_*` families above are
# untouched and still valid.
#
# ⚠️ The bbox is NOT in the filename, so a cache built for one bbox would be
# silently reused for another. `_fetch_era5_regional` therefore writes a sidecar
# manifest and refuses to mix bboxes — the same class of defect as the Open-Meteo
# `lead_days` leak (defect 9), where a cache key encoded less than the request.
REGIONAL_CORE_PREFIX = "era5gh_core"
REGIONAL_EXTRA_PREFIX = "era5gh_extra"
REGIONAL_PRESSURE_PREFIX = "era5gh_pressure"
REGIONAL_LSM_NAME = "era5gh_lsm.nc"
REGIONAL_MANIFEST_NAME = "era5gh_manifest.json"

# Truncation floors for the regional families, passed to `_retrieve_with_retry`
# as `min_bytes`. A download killed mid-request can leave a short file behind, and
# `path.exists()` is the only re-download guard in this module — so without a floor
# a partial month would be treated as complete forever.
#
# Calibrated against the smallest legitimate files observed (February, 28 days):
# core 5.87 MB, extra 2.68 MB, pressure 4.71 MB. Month-length variation is only
# ~10% (28/31 days), so a floor near half the February size cannot false-positive
# on a real month while still catching a gross truncation or an error page written
# to the target.
#
# Subtler truncation is caught elsewhere and harder: a partially-written NetCDF
# fails to open in `_open_dataset` rather than yielding plausible wrong numbers.
REGIONAL_MIN_BYTES = {
    REGIONAL_CORE_PREFIX: 2_500_000,
    REGIONAL_EXTRA_PREFIX: 1_200_000,
    REGIONAL_PRESSURE_PREFIX: 2_000_000,
}


@dataclass(frozen=True)
class RegionalExtract:
    """How to crop a regional ERA5 grid down to a single site.

    Passed to the monthly loaders by the regional path only. When it is None they
    behave exactly as they always have — whole grid into `_centre_and_spatial` —
    which is what keeps the published Anloga results byte-reproducible.
    """
    half: int = 1
    min_land_fraction: float = 0.5
    lsm: np.ndarray | None = None
    mask_lats: np.ndarray | None = None
    mask_lons: np.ndarray | None = None


def _fetch_era5(site: SiteConfig, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch ERA5 single-level + extras + pressure-levels for the site.

    Three cache families keep v1 downloads valid while the v2 extension pulls
    only the newly-added variables:
        era5_...  (v1 core, 7 vars)
        era5x_... (v2 extras, 3 vars)
        era5pl_... (pressure levels, 3 vars × 2 levels)

    Chunked per (year, month) so each CDS request sits under the cost cap.
    """
    import cdsapi

    ensure_dir(cache_dir)
    client = cdsapi.Client()
    # ERA5 grid = 0.25°; CDS rejects bbox smaller than one cell. 0.2° pad gives
    # a 0.4° span which, after CDS snaps to the grid, lands on a 1(lat)×2(lon)
    # patch for Anloga — not the 3×3 originally intended. See the caveat in
    # `_centre_and_spatial` for what that does to the spatial-derivative columns.
    pad = 0.2
    area = [site.latitude + pad, site.longitude - pad,
            site.latitude - pad, site.longitude + pad]  # N, W, S, E

    months = _year_month_range(start, end)
    n = len(months)

    core_paths: list[Path] = []
    extra_paths: list[Path] = []
    pressure_paths: list[Path] = []
    failures: list[tuple[str, str]] = []  # (family, "YYYY-MM")

    for i, (year, month) in enumerate(months, start=1):
        core = cache_dir / f"era5_{site.name}_{year}-{month:02d}.nc"
        extra = cache_dir / f"era5x_{site.name}_{year}-{month:02d}.nc"
        press = cache_dir / f"era5pl_{site.name}_{year}-{month:02d}.nc"
        core_paths.append(core)
        extra_paths.append(extra)
        pressure_paths.append(press)
        tag = f"{year}-{month:02d}"

        if not core.exists():
            if not _retrieve_with_retry(
                client, "reanalysis-era5-single-levels",
                _era5_single_request(ERA5_CORE_VARS, year, month, area),
                str(core), label=f"core {tag} ({i}/{n})",
            ):
                failures.append(("core", tag))
        if not extra.exists():
            if not _retrieve_with_retry(
                client, "reanalysis-era5-single-levels",
                _era5_single_request(ERA5_EXTRA_VARS, year, month, area),
                str(extra), label=f"extras {tag} ({i}/{n})",
            ):
                failures.append(("extras", tag))
        if not press.exists():
            if not _retrieve_with_retry(
                client, "reanalysis-era5-pressure-levels",
                _era5_pressure_request(
                    ERA5_PRESSURE_VARS, ERA5_PRESSURE_LEVELS, year, month, area,
                ),
                str(press), label=f"pressure {tag} ({i}/{n})",
            ):
                failures.append(("pressure", tag))

    if failures:
        by_family: dict[str, list[str]] = {}
        for fam, tag in failures:
            by_family.setdefault(fam, []).append(tag)
        for fam, tags in by_family.items():
            logger.warning(
                "ERA5 %s: %d/%d months failed and will be missing from features: %s",
                fam, len(tags), n, ", ".join(tags),
            )
        # Only abort if the CORE (v1) data has holes — pipeline can't run without it.
        # Missing extras / pressure-levels just become NaN columns → dropped by feature builder.
        if any(fam == "core" for fam, _ in failures):
            raise RuntimeError(
                f"ERA5 core downloads failed for {len([f for f in failures if f[0] == 'core'])} "
                f"months; pipeline cannot continue. Re-run to retry."
            )

    return _load_era5_all(core_paths, extra_paths, pressure_paths, site, start, end)


def _bbox_area(bbox: dict[str, float]) -> list[float]:
    """{north, west, south, east} -> the [N, W, S, E] list CDS wants."""
    missing = {"north", "west", "south", "east"} - set(bbox)
    if missing:
        raise ValueError(
            f"data.bbox is missing {sorted(missing)}; needed for source "
            f"'era5_regional'. See sites/ghana_16.yaml for the Ghana box."
        )
    return [float(bbox["north"]), float(bbox["west"]),
            float(bbox["south"]), float(bbox["east"])]


def _check_regional_manifest(cache_dir: Path, area: list[float]) -> None:
    """Refuse to mix bboxes inside one regional cache.

    ⚠️ The regional filenames carry only the month, because the files are shared
    by every site in the box — that sharing is the whole point. The cost is that
    nothing in the name says *which* box, so a cache built for one bbox would be
    silently reused for another and every spatial column would quietly describe
    the wrong neighbourhood.

    This is exactly the shape of defect 9 (`lead_days` stopped being binding once
    the cache widened): **a cache key that encodes less than the request does**.
    That one was found only because a re-run moved 20 published figures. Having
    paid for the lesson once, the manifest is written here rather than after.
    """
    path = Path(cache_dir) / REGIONAL_MANIFEST_NAME
    if not path.exists():
        path.write_text(json.dumps({"area": area}, indent=2), encoding="utf-8")
        logger.info("Wrote regional cache manifest %s for area %s", path, area)
        return
    recorded = json.loads(path.read_text(encoding="utf-8")).get("area")
    if recorded != area:
        raise RuntimeError(
            f"Regional ERA5 cache in {cache_dir} was built for area {recorded}, "
            f"but this run asks for {area}. The `era5gh_*` filenames do not "
            f"encode the bbox, so reusing them would silently mix grids. Either "
            f"restore the original bbox or move the existing era5gh_* files "
            f"aside."
        )


def _drop_truncated_regional_files(cache_dir: Path) -> list[Path]:
    """Delete any regional cache file below its family's floor, before resuming.

    ⚠️ **This is the hole `min_bytes` alone does not close.** That check runs after
    a `retrieve` returns, so it catches a short file this process downloaded — but
    a process **killed mid-write** leaves a partial file behind that no later run
    ever looks at, because `path.exists()` short-circuits first. The partial month
    then feeds every site that reads it, and nothing about the output looks wrong.

    That is not hypothetical here. The archive download was killed at 2010-07 on
    2026-10-01 when its parent session ended, and escaped damage only because
    cdsapi had not yet opened the target. A multi-day download crossing several
    session boundaries will be killed again.

    So every resume re-validates what is already on disk and deletes anything
    short, letting the normal request loop fetch it again. Returns what it removed
    so callers can log it.
    """
    removed: list[Path] = []
    for prefix, floor in REGIONAL_MIN_BYTES.items():
        for path in sorted(Path(cache_dir).glob(f"{prefix}_*.nc")):
            size = path.stat().st_size
            if size < floor:
                logger.warning(
                    "Regional cache: %s is %d bytes (floor %d) — truncated, "
                    "deleting so it is re-requested", path.name, size, floor,
                )
                path.unlink()
                removed.append(path)
    if removed:
        logger.warning("Regional cache: removed %d truncated file(s) before "
                       "resuming", len(removed))
    else:
        logger.info("Regional cache: all existing files pass their size floor")
    return removed


def fetch_regional_archive(
    start: str, end: str, cache_dir: Path, bbox: dict[str, float],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Download the shared bbox archive once. Site-independent, idempotent.

    Separate from `_fetch_era5_regional` because the download is **not** per-site
    — every site reads the same files — and the multi-site runner needs to do it
    exactly once, before any site is processed, rather than discovering it
    mid-way through site 1.

    Returns the (core, extra, pressure) path lists for the month range, whether or
    not each file was downloaded on this call. Safe to re-run: existing files are
    skipped, which is also how a part-finished multi-day download resumes.

    ⚠️ **Never run two of these at once.** `path.exists()` is the only
    re-download guard, so two processes pick the same missing month and can write
    the same path simultaneously. This happened on 2026-08-10 (RESULTS.md).
    """
    import cdsapi

    ensure_dir(cache_dir)
    area = _bbox_area(bbox)
    _check_regional_manifest(cache_dir, area)
    _drop_truncated_regional_files(cache_dir)
    client = cdsapi.Client()

    # The mask is constant in value, so one hour covers the whole archive.
    lsm_path = Path(cache_dir) / REGIONAL_LSM_NAME
    if not lsm_path.exists():
        _retrieve_with_retry(
            client, "reanalysis-era5-single-levels",
            {"product_type": "reanalysis", "variable": ["land_sea_mask"],
             "year": "2024", "month": "06", "day": "01", "time": ["12:00"],
             "area": area, "format": "netcdf"},
            str(lsm_path), label="land-sea mask (one-off)",
        )

    months = _year_month_range(start, end)
    n = len(months)
    core_paths: list[Path] = []
    extra_paths: list[Path] = []
    pressure_paths: list[Path] = []
    failures: list[tuple[str, str]] = []

    for i, (year, month) in enumerate(months, start=1):
        tag = f"{year}-{month:02d}"
        core = Path(cache_dir) / f"{REGIONAL_CORE_PREFIX}_{tag}.nc"
        extra = Path(cache_dir) / f"{REGIONAL_EXTRA_PREFIX}_{tag}.nc"
        press = Path(cache_dir) / f"{REGIONAL_PRESSURE_PREFIX}_{tag}.nc"
        core_paths.append(core)
        extra_paths.append(extra)
        pressure_paths.append(press)

        if not core.exists():
            if not _retrieve_with_retry(
                client, "reanalysis-era5-single-levels",
                _era5_single_request(ERA5_CORE_VARS, year, month, area),
                str(core), label=f"regional core {tag} ({i}/{n})",
                min_bytes=REGIONAL_MIN_BYTES[REGIONAL_CORE_PREFIX],
            ):
                failures.append(("core", tag))
        if not extra.exists():
            if not _retrieve_with_retry(
                client, "reanalysis-era5-single-levels",
                _era5_single_request(ERA5_EXTRA_VARS, year, month, area),
                str(extra), label=f"regional extras {tag} ({i}/{n})",
                min_bytes=REGIONAL_MIN_BYTES[REGIONAL_EXTRA_PREFIX],
            ):
                failures.append(("extras", tag))
        if not press.exists():
            if not _retrieve_with_retry(
                client, "reanalysis-era5-pressure-levels",
                _era5_pressure_request(
                    ERA5_PRESSURE_VARS, ERA5_PRESSURE_LEVELS, year, month, area,
                ),
                str(press), label=f"regional pressure {tag} ({i}/{n})",
                min_bytes=REGIONAL_MIN_BYTES[REGIONAL_PRESSURE_PREFIX],
            ):
                failures.append(("pressure", tag))

    if failures:
        by_family: dict[str, list[str]] = {}
        for fam, tag in failures:
            by_family.setdefault(fam, []).append(tag)
        for fam, tags in by_family.items():
            logger.warning(
                "Regional ERA5 %s: %d/%d months failed and will be missing from "
                "features: %s", fam, len(tags), n, ", ".join(tags),
            )
        if any(fam == "core" for fam, _ in failures):
            raise RuntimeError(
                f"Regional ERA5 core downloads failed for "
                f"{len([f for f in failures if f[0] == 'core'])} months; cannot "
                f"continue. Re-run to retry."
            )
    return core_paths, extra_paths, pressure_paths


def _fetch_era5_regional(site: SiteConfig, start: str, end: str, cache_dir: Path,
                         bbox: dict[str, float], window: int = 1,
                         min_land_fraction: float = 0.5) -> pd.DataFrame:
    """ERA5 for one site, extracted from a cache shared across the whole bbox.

    Why this exists rather than a wider `pad` on `_fetch_era5`
    ---------------------------------------------------------
    The per-site fetcher issues one CDS request per (month x family) around one
    point, so each site costs its own download: 468 requests for the 2010-2025
    archive, and the v5 logs measured ~19 min per request end to end. Seventeen
    sites that way is roughly **99 days** of continuous downloading, which does
    not merely inconvenience the study, it ends it.

    **CDS charges per field — variables x times x levels — not per grid cell.**
    So one request covering all of Ghana costs the same as one covering two
    cells; only the payload grows. The regional cache is therefore keyed by month
    alone and every site reads from it, turning 17 downloads into 1.

    Two consequences worth stating plainly:

    * **The spatial columns change.** The per-site `pad = 0.2` collapses to a
      1(lat)x2(lon) patch, which makes every `_grad_ns` identically zero and
      every `_std` exactly `|_grad_ew| / 2` — ten of 36 tidy columns carrying no
      information (see `_centre_and_spatial`). A regional grid gives a genuine
      3x3, so those columns become real. That is a fix, but it means this path
      produces a *different feature set*, not a drop-in replacement.
    * **Cell selection is not just nearest.** Coastal sites can land offshore,
      where lower roughness inflates wind speed invisibly. `select_site_cell`
      snaps onto land; see its docstring and docs/13 section 3.
    """
    core_paths, extra_paths, pressure_paths = fetch_regional_archive(
        start, end, cache_dir, bbox)

    mask = load_land_sea_mask(Path(cache_dir))
    extract = RegionalExtract(
        half=window, min_land_fraction=min_land_fraction,
        lsm=mask[0] if mask else None,
        mask_lats=mask[1] if mask else None,
        mask_lons=mask[2] if mask else None,
    )
    return _load_era5_all(core_paths, extra_paths, pressure_paths, site,
                          start, end, extract=extract)


def _retrieve_with_retry(client, dataset: str, request: dict, target_path: str,
                         label: str, max_attempts: int = 5, wait_seconds: int = 300,
                         min_bytes: int = 0) -> bool:
    """Retry a CDS retrieve on transient/CDS-side failures.

    Returns True on success, False if all attempts failed (caller decides
    whether to abort or continue). CDS jobs sometimes fail server-side with
    a terse 400 — a retry usually clears them.

    Budget widened from 3×60 s to 5×300 s on 2026-08-11. The old ~3-minute
    tolerance was shorter than a real outage: on 2026-08-11 an SSL/DNS drop
    lasting ~6 minutes exhausted all three attempts for *every* family of
    2024-07, so the month was skipped entirely and the run carried on. A missing
    core month makes `_fetch_era5` raise at the end of a multi-day download,
    which is an expensive way to find out. ~20 minutes of tolerance covers the
    transient drops actually observed on this connection.

    Note this wraps cdsapi's own retry loop, which already handles connection
    errors internally; this layer exists for failures that surface as
    exceptions from `client.retrieve`.

    ⚠️ **`min_bytes` guards against a truncated file becoming permanent.** A
    download interrupted after the target is opened but before it is complete
    leaves a short file behind, and since `path.exists()` is the only
    re-download guard anywhere in this module, that file is treated as finished
    *forever* — silently feeding a partial month into every result that reads it.
    Nothing about the resulting series looks broken.

    This nearly happened on 2026-10-01: the regional archive download was killed
    mid-request at 2010-07 and only escaped damage because cdsapi happened not to
    have written the target yet. Over a multi-day download crossing several
    session boundaries, that is luck rather than a guarantee.

    The floor is **per call**, not global, because the families differ by two
    orders of magnitude: a regional monthly file is 2.8-6.5 MB while a per-site
    one is ~150 KB. `min_bytes=0` (the default) keeps the per-site path exactly
    as it was. A short file is deleted so the retry loop re-requests it rather
    than skipping it next run.

    Precedent: the 10-year backfill was validated by hand with "verified no file
    under 50 KB" (docs/11 section 8). This is that check, automated.
    """
    import time
    target = Path(target_path)
    for attempt in range(1, max_attempts + 1):
        logger.info("Downloading ERA5 %s (attempt %d/%d)", label, attempt, max_attempts)
        try:
            client.retrieve(dataset, request, target_path)
        except Exception as exc:  # cdsapi wraps requests.HTTPError etc.
            logger.warning("ERA5 %s attempt %d failed: %s", label, attempt, exc)
        else:
            size = target.stat().st_size if target.exists() else 0
            if size >= min_bytes:
                return True
            # Delete, or `path.exists()` would skip this month on the next run.
            logger.warning(
                "ERA5 %s: retrieve reported success but the file is %d bytes "
                "(floor %d) — treating as truncated and deleting",
                label, size, min_bytes,
            )
            target.unlink(missing_ok=True)
        if attempt < max_attempts:
            time.sleep(wait_seconds)
    logger.error("ERA5 %s: giving up after %d attempts", label, max_attempts)
    return False


def _era5_single_request(variables: list[str], year: int, month: int,
                         area: list[float]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "format": "netcdf",
    }


def _era5_pressure_request(variables: list[str], levels: list[str], year: int,
                            month: int, area: list[float]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "pressure_level": levels,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "format": "netcdf",
    }


# ---------------------------------------------------------------------------
# NetCDF → tidy DataFrame with spatial derivatives
# ---------------------------------------------------------------------------

def _load_era5_all(
    core_paths: list[Path], extra_paths: list[Path], pressure_paths: list[Path],
    site: SiteConfig, start: str, end: str,
    extract: RegionalExtract | None = None,
) -> pd.DataFrame:
    core = pd.concat([_load_era5_single_month(p, site, extract)
                      for p in core_paths]).sort_index()
    core = core[~core.index.duplicated(keep="first")]

    extras_frames = [_load_era5_extra_month(p, site, extract)
                     for p in extra_paths if p.exists()]
    if extras_frames:
        extras = pd.concat(extras_frames).sort_index()
        extras = extras[~extras.index.duplicated(keep="first")]
        core = core.join(extras, how="left")

    pressure_frames = [_load_era5_pressure_month(p, site, extract)
                       for p in pressure_paths if p.exists()]
    if pressure_frames:
        press = pd.concat(pressure_frames).sort_index()
        press = press[~press.index.duplicated(keep="first")]
        core = core.join(press, how="left")

    core = core.loc[start:end]

    # Ensure every declared column exists (fill NaN if a family didn't run)
    for col in TIDY_COLUMNS:
        if col not in core.columns:
            core[col] = np.nan
    return core[TIDY_COLUMNS]


def _is_zip(path: Path) -> bool:
    """Detect ZIP magic bytes (`PK\\x03\\x04`) regardless of file extension."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _open_dataset(path: Path):
    """Open a NetCDF, transparently unpacking ZIP-wrapped bundles.

    CDS sometimes returns a ZIP containing one NetCDF per variable-stream
    (e.g. accumulated-flux variables like `10m_wind_gust` ship separately
    from instantaneous ones). We open all NetCDFs inside and merge them.
    """
    import xarray as xr

    if not _is_zip(path):
        return xr.open_dataset(path)

    import shutil
    import tempfile
    import zipfile

    extract_dir = Path(tempfile.mkdtemp(prefix="era5unzip_"))
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        nc_files = sorted(extract_dir.rglob("*.nc"))
        if not nc_files:
            raise ValueError(f"ZIP {path} contains no .nc files")
        # Open each and merge into a single Dataset. Different sub-files can
        # have different variables (that's the whole point of the split).
        datasets = []
        for f in nc_files:
            ds = xr.open_dataset(f)
            # Load into memory so we can delete the temp files afterwards
            datasets.append(ds.load())
            ds.close()
        merged = xr.merge(datasets, compat="override")
        return merged
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _coord_names(ds) -> tuple[str, str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "time" if "time" in ds.coords else "valid_time"
    return lat_name, lon_name, time_name


def _times_utc(ds, time_name: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ds[time_name].values), tz="UTC")


def _grid_arr(ds, var: str) -> np.ndarray | None:
    """Extract a (time, lat, lon) array — squeeze away only unit dims other than lat/lon."""
    if var not in ds.variables:
        return None
    arr = np.asarray(ds[var].values)
    # Some ERA5 downloads add an extra unit dim (e.g. expver). Drop unit dims
    # that are NOT the last three.
    while arr.ndim > 3 and 1 in arr.shape[:-3]:
        for axis in range(arr.ndim - 3):
            if arr.shape[axis] == 1:
                arr = np.squeeze(arr, axis=axis)
                break
    return arr


# ---------------------------------------------------------------------------
# Regional ERA5: site cell selection and window extraction
# ---------------------------------------------------------------------------

def load_land_sea_mask(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (lsm, lats, lons) from the one-off regional mask, or None.

    `land_sea_mask` is constant in value, so it is fetched once for a single hour
    over the bbox (`era5gh_lsm.nc`, a few KB) rather than per month. Returns None
    when the file is absent, which callers must treat as "cannot verify" rather
    than "it is land".
    """
    path = Path(cache_dir) / REGIONAL_LSM_NAME
    if not path.exists():
        return None
    ds = _open_dataset(path)
    lat_name, lon_name, _ = _coord_names(ds)
    lsm = np.asarray(ds["lsm"].values)
    while lsm.ndim > 2:
        lsm = lsm[0]
    return lsm, np.asarray(ds[lat_name].values), np.asarray(ds[lon_name].values)


def select_site_cell(lats: np.ndarray, lons: np.ndarray, site: SiteConfig,
                     lsm: np.ndarray | None = None,
                     mask_lats: np.ndarray | None = None,
                     mask_lons: np.ndarray | None = None,
                     min_land_fraction: float = 0.5) -> tuple[int, int]:
    """Pick the grid cell representing `site`: nearest, then snapped onto land.

    ⚠️ **Nearest is not always right, and the failure is silent.** ERA5 sea cells
    carry far lower surface roughness and correspondingly higher wind speeds, so a
    coastal site whose nearest cell lands offshore reports an *inflated* resource
    while looking entirely self-consistent. Cape Coast is the live case: at
    5.11 N it sits 0.11 deg from the ERA5 row at 5.00 and 0.14 deg from 5.25 — a
    margin of 0.030 deg — and the coastline there runs near 5.10 N, so plain
    nearest-cell selection puts it in the Gulf of Guinea.

    So when a mask is supplied and the nearest cell is below
    `min_land_fraction`, this snaps to the closest cell that qualifies, searching
    outward, and logs the snap. With no mask it returns the nearest cell and warns
    once — "cannot verify" is not "verified".

    The threshold is pre-registered in docs/13 §3 and fixed before any resource
    figure exists; choosing it after seeing which sites rank well would be
    fitting the geography to the result.
    """
    i = int(np.abs(lats - site.latitude).argmin())
    j = int(np.abs(lons - site.longitude).argmin())
    if lsm is None or mask_lats is None or mask_lons is None:
        logger.warning(
            "%s: no land-sea mask available, using nearest cell (%.2f, %.2f) "
            "unverified — a coastal site may be sitting on water",
            site.name, lats[i], lons[j],
        )
        return i, j

    mi = int(np.abs(mask_lats - site.latitude).argmin())
    mj = int(np.abs(mask_lons - site.longitude).argmin())
    if float(lsm[mi, mj]) >= min_land_fraction:
        return i, j

    best: tuple[int, int] | None = None
    best_dist = np.inf
    for di in range(-3, 4):
        for dj in range(-3, 4):
            mii, mjj = mi + di, mj + dj
            if not (0 <= mii < len(mask_lats) and 0 <= mjj < len(mask_lons)):
                continue
            if float(lsm[mii, mjj]) < min_land_fraction:
                continue
            dist = float(np.hypot(
                (mask_lats[mii] - site.latitude) * 111.0,
                (mask_lons[mjj] - site.longitude) * 111.0
                * np.cos(np.radians(mask_lats[mii])),
            ))
            if dist < best_dist:
                best, best_dist = (mii, mjj), dist
    if best is None:
        raise RuntimeError(
            f"{site.name}: no cell within 3 cells of ({site.latitude}, "
            f"{site.longitude}) has land fraction >= {min_land_fraction}. "
            f"Check the coordinate — this is an ocean location."
        )

    target_lat, target_lon = mask_lats[best[0]], mask_lons[best[1]]
    logger.info(
        "%s: nearest cell (%.2f, %.2f) has land fraction %.3f < %.2f — snapped "
        "to (%.2f, %.2f) at %.0f km, land fraction %.3f",
        site.name, lats[i], lons[j], float(lsm[mi, mj]), min_land_fraction,
        target_lat, target_lon, best_dist, float(lsm[best]),
    )
    return (int(np.abs(lats - target_lat).argmin()),
            int(np.abs(lons - target_lon).argmin()))


def _cropper(ds, site: SiteConfig, extract: RegionalExtract | None):
    """Build the per-dataset crop function used by the monthly loaders.

    Returns identity when `extract` is None, which is the per-site path and must
    stay bit-for-bit what it was — every published Anloga figure depends on it.
    Otherwise returns a closure that crops any (time, lat, lon) array to the
    site's window, resolving the site cell **once** per file rather than per
    variable.
    """
    if extract is None:
        return lambda arr: arr

    lat_name, lon_name, _ = _coord_names(ds)
    lats = np.asarray(ds[lat_name].values)
    lons = np.asarray(ds[lon_name].values)
    i, j = select_site_cell(
        lats, lons, site,
        lsm=extract.lsm, mask_lats=extract.mask_lats, mask_lons=extract.mask_lons,
        min_land_fraction=extract.min_land_fraction,
    )
    n_lat, n_lon = len(lats), len(lons)

    def crop(arr: np.ndarray | None) -> np.ndarray | None:
        return _window_slice(arr, i, j, extract.half, n_lat, n_lon)

    return crop


def _window_slice(arr: np.ndarray | None, i: int, j: int, half: int,
                  n_lat: int, n_lon: int) -> np.ndarray | None:
    """Crop a (time, lat, lon) array to the (2*half+1)^2 window centred on (i, j).

    Clamped at the grid edge, so a site near the bbox boundary yields a smaller
    window rather than an error — but the bbox in `sites/ghana_16.yaml` is sized
    so that every Ghanaian site has room for a full 3x3, and that is asserted in
    the tests rather than left to chance.

    The centre of the returned window is the site's own cell, which is what makes
    `_centre_and_spatial`'s `arr[:, n_lat // 2, n_lon // 2]` pick the right cell
    downstream without changing that function at all.
    """
    if arr is None or arr.ndim < 3:
        return arr
    i0, i1 = max(0, i - half), min(n_lat, i + half + 1)
    j0, j1 = max(0, j - half), min(n_lon, j + half + 1)
    # Keep the site cell at the window centre even when clamped, otherwise
    # `_centre_and_spatial` would silently return a neighbour.
    if i - i0 != i1 - 1 - i:
        k = min(i - i0, i1 - 1 - i)
        i0, i1 = i - k, i + k + 1
    if j - j0 != j1 - 1 - j:
        k = min(j - j0, j1 - 1 - j)
        j0, j1 = j - k, j + k + 1
    return arr[:, i0:i1, j0:j1]


def _centre_and_spatial(arr: np.ndarray, lat_axis: int = -2,
                        lon_axis: int = -1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Given a (time, lat, lon) array, return (centre-cell series, spatial stats).

    Spatial stats, all carrying the parent variable's own unit (these are plain
    differences across the patch, NOT divided by the patch span — they are not
    per-degree gradients):
        mean    — patch mean over lat/lon
        std     — patch spatial std (population)
        grad_ns — north edge mean − south edge mean
        grad_ew — east edge mean − west edge mean

    CAVEAT — degenerate on the current 1(lat)x2(lon) patch. `pad = 0.2` around
    Anloga survives only one ERA5 latitude row (5.75) and two longitudes (0.75,
    1.00), so with n_lat == 1 the north and south edges are the same row and
    every `grad_ns` column is identically zero; with n_lon == 2 the population
    std over two cells is exactly |grad_ew| / 2, making those columns perfectly
    collinear. Verified against the 2020-2023 parquet. Widening `pad` to ~0.35
    would yield a true 3x3 patch, but that changes every cached download and all
    published results, so it is deliberately left alone here — see
    docs/00_data_card.md.
    """
    if arr.ndim == 1:
        # No spatial extent — treat as centre only, no spatial stats
        return arr, {}
    if arr.ndim == 2:
        # Just (time, one_spatial); can't do spatial stats meaningfully
        return arr.mean(axis=-1), {}

    n_lat, n_lon = arr.shape[lat_axis], arr.shape[lon_axis]
    centre = arr[:, n_lat // 2, n_lon // 2]
    mean = arr.mean(axis=(lat_axis, lon_axis))
    std = arr.std(axis=(lat_axis, lon_axis))
    # ERA5 latitude is typically decreasing (N→S): index 0 is north edge.
    north_edge = arr[:, 0, :].mean(axis=-1)
    south_edge = arr[:, -1, :].mean(axis=-1)
    west_edge = arr[:, :, 0].mean(axis=-1)
    east_edge = arr[:, :, -1].mean(axis=-1)
    # Plain edge differences in the parent variable's unit — no division by the
    # patch span. Both collapse to zero when the corresponding axis has length 1
    # (which is the case for grad_ns today; see the caveat above).
    grad_ns = (north_edge - south_edge)
    grad_ew = (east_edge - west_edge)
    stats = {"mean": mean, "std": std, "grad_ns": grad_ns, "grad_ew": grad_ew}
    return centre, stats


def _load_era5_single_month(path: Path, site: SiteConfig,
                            extract: RegionalExtract | None = None) -> pd.DataFrame:
    """Load one monthly NetCDF of v1 core variables → per-cell + spatial stats.

    `extract=None` (the per-site path) reads the whole grid, exactly as before.
    Supplied (the regional path), the grid is first cropped to the site's window
    so `_centre_and_spatial` sees a patch centred on that site.
    """
    ds = _open_dataset(path)
    _, _, time_name = _coord_names(ds)
    times = _times_utc(ds, time_name)
    crop = _cropper(ds, site, extract)

    def _grid(var: str) -> np.ndarray | None:
        return crop(_grid_arr(ds, var))

    u10 = _grid("u10"); v10 = _grid("v10")
    u100 = _grid("u100"); v100 = _grid("v100")
    t2m = _grid("t2m")
    sp = _grid("sp")
    fsr = _grid("fsr")

    # Centre-cell derived scalars
    u10_c, _ = _centre_and_spatial(u10)
    v10_c, _ = _centre_and_spatial(v10)
    u100_c, sp_u100 = _centre_and_spatial(u100)
    v100_c, sp_v100 = _centre_and_spatial(v100)
    t2m_c, sp_t2m = _centre_and_spatial(t2m)
    sp_c, sp_sp = _centre_and_spatial(sp)
    fsr_c = fsr[:, fsr.shape[-2] // 2, fsr.shape[-1] // 2] if fsr is not None else None

    ws10, dir10 = _uv_to_speed_dir(u10_c, v10_c)
    ws100, dir100 = _uv_to_speed_dir(u100_c, v100_c)

    # Spatial derivatives for wind_speed_100m: compute on the full (time,lat,lon)
    # magnitude, not on u/v components (magnitude is the interesting field).
    ws100_grid = np.sqrt(u100 * u100 + v100 * v100)
    _, sp_ws100 = _centre_and_spatial(ws100_grid)

    data = {
        "wind_speed_10m": ws10,
        "wind_speed_100m": ws100,
        "wind_dir_10m": dir10,
        "wind_dir_100m": dir100,
        "temperature_c": t2m_c - 273.15,
        "pressure_pa": sp_c,
        "roughness": fsr_c if fsr_c is not None else np.nan,
    }
    # Spatial derivatives for the vars declared in SPATIAL_VARS that we have
    for name, stats in [
        ("wind_speed_100m", sp_ws100),
        ("pressure_pa", sp_sp),
        ("temperature_c", {k: v - 273.15 if k == "mean" else v for k, v in sp_t2m.items()}),
    ]:
        for stat_name, arr in stats.items():
            data[f"{name}_{stat_name}"] = arr

    return pd.DataFrame(data, index=times).rename_axis("timestamp")


def _load_era5_extra_month(path: Path, site: SiteConfig,
                           extract: RegionalExtract | None = None) -> pd.DataFrame:
    ds = _open_dataset(path)
    _, _, time_name = _coord_names(ds)
    times = _times_utc(ds, time_name)
    crop = _cropper(ds, site, extract)

    gust = crop(_grid_arr(ds, "fg10"))  # 10m wind gust
    d2m = crop(_grid_arr(ds, "d2m"))    # 2m dewpoint (K)
    blh = crop(_grid_arr(ds, "blh"))    # boundary layer height (m)

    def _centre(x):
        if x is None:
            return np.full(len(times), np.nan)
        return x[:, x.shape[-2] // 2, x.shape[-1] // 2]

    return pd.DataFrame({
        "wind_gust_10m": _centre(gust),
        "dewpoint_c": _centre(d2m) - 273.15 if d2m is not None else np.nan,
        "bl_height_m": _centre(blh),
    }, index=times).rename_axis("timestamp")


def _load_era5_pressure_month(path: Path, site: SiteConfig,
                              extract: RegionalExtract | None = None) -> pd.DataFrame:
    ds = _open_dataset(path)
    _, _, time_name = _coord_names(ds)
    times = _times_utc(ds, time_name)
    crop = _cropper(ds, site, extract)

    if "pressure_level" in ds.coords:
        level_name = "pressure_level"
    elif "level" in ds.coords:
        level_name = "level"
    else:
        level_name = None

    def _grid_at_level(var: str, level: int) -> np.ndarray | None:
        if var not in ds.variables or level_name is None:
            return None
        return crop(np.asarray(ds[var].sel({level_name: level}).values))

    def _centre(x):
        if x is None:
            return np.full(len(times), np.nan)
        return x[:, x.shape[-2] // 2, x.shape[-1] // 2] if x.ndim >= 3 else x

    out: dict[str, np.ndarray] = {}
    for level in (850, 500):
        u = _grid_at_level("u", level)
        v = _grid_at_level("v", level)
        z = _grid_at_level("z", level)
        if u is not None and v is not None:
            uc = _centre(u); vc = _centre(v)
            ws, wd = _uv_to_speed_dir(uc, vc)
        else:
            ws = wd = np.full(len(times), np.nan)
        gp = _centre(z) if z is not None else np.full(len(times), np.nan)
        out[f"wind_speed_{level}hpa"] = ws
        out[f"wind_dir_{level}hpa"] = wd
        out[f"geopot_{level}hpa"] = gp

    df = pd.DataFrame(out, index=times).rename_axis("timestamp")

    # Spatial derivatives for wind_speed_850hpa and geopot_850hpa
    u850 = _grid_at_level("u", 850)
    v850 = _grid_at_level("v", 850)
    z850 = _grid_at_level("z", 850)
    if u850 is not None and v850 is not None:
        ws850_grid = np.sqrt(u850 * u850 + v850 * v850)
        _, sp = _centre_and_spatial(ws850_grid)
        for k, arr in sp.items():
            df[f"wind_speed_850hpa_{k}"] = arr
    if z850 is not None:
        _, sp = _centre_and_spatial(z850)
        for k, arr in sp.items():
            df[f"geopot_850hpa_{k}"] = arr

    return df


# ---------------------------------------------------------------------------
# NASA POWER (unchanged)
# ---------------------------------------------------------------------------

def _fetch_nasa_power(site: SiteConfig, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    ensure_dir(cache_dir)
    cache = _nasa_cache_path(cache_dir, site, start, end)
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = _download_nasa_power(site, start, end)
        cache.write_text(json.dumps(payload))
    return _parse_nasa_power(payload)


def _download_nasa_power(site: SiteConfig, start: str, end: str) -> dict[str, Any]:
    logger.info("Downloading NASA POWER for %s (%s..%s)", site.name, start, end)
    params = {
        "parameters": "WS10M,WS50M,WD10M,WD50M,T2M,PS",
        "community": "RE",
        "longitude": site.longitude,
        "latitude": site.latitude,
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


# NASA POWER hourly timestamps run one hour later than the equivalent ERA5 stamp.
#
# POWER's hourly fields come from MERRA-2, whose values are hour-*averaged* rather than
# instantaneous, and the label lands at the end of the averaging interval. Measured over
# 8,760 overlapping hours (2024-03 -> 2025-02) on 2026-08-11:
#
#   * ERA5(t) correlates best with NASA(t+1)      r 0.6892 vs 0.6790 at lag 0
#   * every NWP model's cross-correlation peak against NASA sits exactly +1 h from its
#     peak against ERA5 — three independent confirmations of the same 1 h shift
#
# The ERA5-vs-NASA curve is nearly flat between -1 and -2 h (0.68921 vs 0.68916), which is
# what hour-averaging does to a correlation peak; the NWP cross-check is the sharper
# evidence and puts it at 1 h.
#
# Subtracting an hour puts NASA POWER on ERA5's clock so the two are interchangeable as
# truth sources. Without it, forecasts joined against NASA POWER truth are misaligned by
# an hour and the alignment gate in src/validate_nwp.py fails (as it did, correctly).
NASA_POWER_TIME_OFFSET_H = -1


def _parse_nasa_power(payload: dict[str, Any]) -> pd.DataFrame:
    params = payload["properties"]["parameter"]

    def _series(key: str) -> pd.Series:
        raw = params[key]
        idx = pd.to_datetime(list(raw.keys()), format="%Y%m%d%H", utc=True)
        s = pd.Series(list(raw.values()), index=idx, dtype=float)
        return s.replace(-999.0, np.nan)

    df = pd.DataFrame({
        "wind_speed_10m": _series("WS10M"),
        # NASA POWER's highest wind level is 50 m — verified 2026-08-11 that WS80M and
        # WS100M do not exist. It lands in `wind_speed_100m` to keep the tidy schema
        # uniform; SOURCE_WIND_HEIGHT_M records the real height so the hub extrapolation
        # uses 50 m rather than silently assuming 100 m.
        "wind_speed_100m": _series("WS50M"),
        "wind_dir_10m": _series("WD10M"),
        "wind_dir_100m": _series("WD50M"),
        "temperature_c": _series("T2M"),
        "pressure_pa": _series("PS") * 1000.0,
        "roughness": np.nan,
    })
    # Put NASA POWER on ERA5's clock — see NASA_POWER_TIME_OFFSET_H above.
    df.index = df.index + pd.Timedelta(hours=NASA_POWER_TIME_OFFSET_H)
    df.index.name = "timestamp"
    # Fill v2 columns with NaN — the feature builder will drop them.
    for col in EXTRA_SINGLE_COLUMNS + PRESSURE_LEVEL_COLUMNS + spatial_columns():
        df[col] = np.nan
    return df[TIDY_COLUMNS]


# ---------------------------------------------------------------------------
# Open-Meteo Previous Runs — archived day-ahead NWP forecasts
# ---------------------------------------------------------------------------
#
# This is the ONLY source in this module that carries a genuine *forecast*.
# Everything else (ERA5, NASA POWER) is the analysed atmospheric state at time
# t; those describe what the weather WAS, and a 24 h-ahead target is only 0.57
# correlated with them. See docs/00_data_card.md.
#
# ⚠️ Endpoint choice is a correctness issue, not a preference:
#
#   previous-runs-api.open-meteo.com  — `{var}_previous_day{N}` is the forecast
#       for the valid timestamp issued N*24 h earlier. This is what we want.
#
#   archive-api / "Historical Forecast API" — builds its series by stitching
#       the FIRST hours of each model run. That is analysis-like (essentially a
#       nowcast) and using it as a t+24 feature is leakage: it would produce an
#       excellent and completely invalid result.
#
# Coverage verified 2026-08-10 for Anloga: all-null through 2024-02, complete
# from 2024-03 onward.

OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def nwp_column(model: str, variable: str, lead_day: int) -> str:
    """Tidy column name for one (model, variable, lead) forecast series."""
    return f"nwp_{model}_{variable}_d{lead_day}"


def _open_meteo_cache_path(cache_dir: Path, site: SiteConfig, model: str,
                           year: int) -> Path:
    return cache_dir / f"openmeteo_{site.name}_{model}_{year}.json"


def _missing_open_meteo_keys(payload: dict[str, Any], lead_days: list[int],
                             variables: list[str]) -> list[str]:
    """Requested `{var}_previous_day{N}` series absent from a cached payload.

    The cache path keys on (site, model, year) and says nothing about which
    leads or variables it holds, so a hit does not imply a match. Without this
    check, widening `lead_days` in a config is a silent no-op.
    """
    hourly = payload.get("hourly", {})
    return [
        key for v in variables for d in lead_days
        if (key := f"{v}_previous_day{d}") not in hourly
    ]


def _fetch_open_meteo(
    site: SiteConfig,
    start: str,
    end: str,
    cache_dir: Path,
    models: list[str],
    lead_days: list[int],
    variables: list[str],
) -> pd.DataFrame:
    """Fetch archived day-ahead NWP forecasts, cached per (model, year).

    Returns a DataFrame indexed by **valid time** (UTC), one column per
    (model, variable, lead_day). Chunked by calendar year so a long range is
    resumable and each response stays small.
    """
    ensure_dir(cache_dir)
    years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    frames: list[pd.DataFrame] = []

    for model in models:
        model_frames: list[pd.DataFrame] = []
        for year in years:
            cache = _open_meteo_cache_path(cache_dir, site, model, year)
            y_start = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01"))
            y_end = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31"))
            if y_start > y_end:
                continue
            payload = None
            if cache.exists():
                payload = json.loads(cache.read_text())
                missing = _missing_open_meteo_keys(payload, lead_days, variables)
                if missing:
                    # The cache filename keys on (site, model, year) only, so a
                    # file written for lead_days [1,2,3] would otherwise be
                    # served silently to a request for [1..7] — a weekly run
                    # would quietly become a 3-day run. Re-fetch instead.
                    logger.info(
                        "Open-Meteo cache %s is missing %d requested series "
                        "(e.g. %s); re-downloading.",
                        cache.name, len(missing), missing[0],
                    )
                    payload = None
            if payload is None:
                payload = _download_open_meteo(
                    site, y_start.strftime("%Y-%m-%d"), y_end.strftime("%Y-%m-%d"),
                    model, lead_days, variables,
                )
                cache.write_text(json.dumps(payload))
            model_frames.append(
                _parse_open_meteo(payload, model, lead_days, variables))
        if model_frames:
            frames.append(pd.concat(model_frames).sort_index())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.loc[start:end]


def _download_open_meteo(site: SiteConfig, start: str, end: str, model: str,
                         lead_days: list[int], variables: list[str]) -> dict[str, Any]:
    logger.info("Downloading Open-Meteo previous-runs %s for %s (%s..%s)",
                model, site.name, start, end)
    hourly = [f"{v}_previous_day{d}" for v in variables for d in lead_days]
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(hourly),
        "models": model,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    resp = requests.get(OPEN_METEO_PREVIOUS_RUNS_URL, params=params, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _parse_open_meteo(payload: dict[str, Any], model: str,
                      lead_days: list[int] | None = None,
                      variables: list[str] | None = None) -> pd.DataFrame:
    """Payload -> DataFrame indexed by valid time, columns `nwp_{model}_{var}_d{N}`.

    ⚠️ **The payload is filtered to what was requested, not served whole.**
    A cache file holds whatever the *widest* previous request asked for, so once
    the weekly work widened these caches to leads 1-7, a config asking for
    `lead_days: [1, 2, 3]` was silently handed all seven — 126 columns instead of
    54. That is not a small difference: it re-ran the 24 h bakeoff on an
    undocumented feature set and moved every published MAE (xgboost 0.8037 ->
    0.7998), because at a 24 h horizon the lead guard admits every lead and the
    extra four are genuine additional information.

    This is the mirror of the earlier cache defect. That one was *request wider,
    receive narrower*; this is *request narrower, receive wider*. Both come from
    the same root cause — a cache key that encodes less than the request does —
    so both are guarded, one at fetch (`_missing_open_meteo_keys`) and one here.
    """
    hourly = payload["hourly"]
    idx = pd.to_datetime(hourly["time"], utc=True)
    wanted_leads = None if lead_days is None else set(lead_days)
    wanted_vars = None if variables is None else set(variables)

    data: dict[str, Any] = {}
    dropped = 0
    for key, values in hourly.items():
        if key == "time":
            continue
        # keys arrive as "{variable}_previous_day{N}"
        variable, _, lead = key.rpartition("_previous_day")
        if not variable or not lead.isdigit():
            logger.warning("Unexpected Open-Meteo key %r — skipped", key)
            continue
        if ((wanted_leads is not None and int(lead) not in wanted_leads)
                or (wanted_vars is not None and variable not in wanted_vars)):
            dropped += 1
            continue
        data[nwp_column(model, variable, int(lead))] = pd.Series(
            values, index=idx, dtype=float
        )
    if dropped:
        logger.info("%s: dropped %d cached series outside the requested "
                    "leads/variables", model, dropped)

    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


def load_nwp_forecasts(config: Config) -> pd.DataFrame:
    """Fetch archived NWP forecasts for the config's site and window.

    Returns an empty frame when `nwp.enabled` is false, so callers can join
    unconditionally.
    """
    if not config.nwp.enabled:
        return pd.DataFrame()

    df = _fetch_open_meteo(
        config.site, config.data.start, config.data.end,
        Path(config.data.cache_dir),
        models=config.nwp.models,
        lead_days=config.nwp.lead_days,
        variables=config.nwp.variables,
    )
    if df.empty:
        logger.warning("Open-Meteo returned no forecast columns")
        return df

    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        logger.warning(
            "Open-Meteo: %d/%d columns are all-NaN (archive starts 2024-03; "
            "earlier windows return nulls): %s",
            len(all_nan), df.shape[1], ", ".join(all_nan[:4]),
        )

    processed_dir = ensure_dir(config.paths.processed)
    out = (processed_dir /
           f"nwp_{config.site.name}_{config.data.start}_{config.data.end}.parquet")
    df.to_parquet(out)
    logger.info("Wrote NWP forecasts: %s (%d rows, %d cols)", out, len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_weather(config: Config) -> pd.DataFrame:
    """Fetch tidy hourly weather for the site, cached to disk."""
    cache_dir = Path(config.data.cache_dir)
    source = config.data.source.lower()
    if source == "era5":
        df = _fetch_era5(config.site, config.data.start, config.data.end, cache_dir)
    elif source == "era5_regional":
        # Multi-site path: one bbox download shared by every site. See
        # `_fetch_era5_regional` for why, and docs/13 for the measurements.
        df = _fetch_era5_regional(
            config.site, config.data.start, config.data.end, cache_dir,
            bbox=config.data.bbox, window=config.data.window,
            min_land_fraction=config.data.min_land_fraction,
        )
    elif source == "nasa_power":
        df = _fetch_nasa_power(config.site, config.data.start, config.data.end, cache_dir)
    else:
        raise ValueError(f"Unknown data.source: {config.data.source!r}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    processed_dir = ensure_dir(config.paths.processed)
    out = processed_dir / f"{source}_{config.site.name}_{config.data.start}_{config.data.end}.parquet"
    df.to_parquet(out)
    logger.info("Wrote tidy weather: %s (%d rows, %d cols)", out, len(df), df.shape[1])
    return df
