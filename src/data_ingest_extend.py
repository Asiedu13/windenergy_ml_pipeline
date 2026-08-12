"""Download-only CLI for extending the ERA5 cache backwards in time.

Recommendation #5 of ``docs/09_why_deep_learning_underperformed.md`` is "10+
years of data". :func:`src.data_ingest.load_weather` would do the downloading,
but it also assembles every cached month into a parquet at the end and aborts
if any *core* month is missing — unhelpful for a long, resumable background job
that may need several passes to clear the CDS queue.

This module fetches the three request families month by month, skips whatever
is already on disk, and reports what is still missing instead of raising:

    python -m src.data_ingest_extend --config config_10yr.yaml

Re-run until it reports zero missing months, then point the pipeline at the
same config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_ingest import (
    ERA5_CORE_VARS,
    ERA5_EXTRA_VARS,
    ERA5_PRESSURE_LEVELS,
    ERA5_PRESSURE_VARS,
    _era5_pressure_request,
    _era5_single_request,
    _retrieve_with_retry,
    _year_month_range,
)
from .utils import Config, ensure_dir, get_logger, load_config

logger = get_logger("data_ingest_extend")


def download_era5_range(config: Config) -> dict[str, list[str]]:
    """Fetch every missing monthly NetCDF in ``config.data``'s date range.

    Returns a ``{family: [YYYY-MM, ...]}`` map of months that are still absent
    after this pass (empty when the cache is complete).
    """
    import cdsapi

    cache_dir = ensure_dir(Path(config.data.cache_dir))
    site = config.site
    client = cdsapi.Client()

    # Same 0.2° pad as data_ingest._fetch_era5 — must stay identical or the
    # backfilled months would have a different grid patch from the cached ones.
    pad = 0.2
    area = [site.latitude + pad, site.longitude - pad,
            site.latitude - pad, site.longitude + pad]  # N, W, S, E

    months = _year_month_range(config.data.start, config.data.end)
    n = len(months)
    logger.info("ERA5 extension: %d months (%s → %s), cache=%s",
                n, config.data.start, config.data.end, cache_dir)

    families = {
        "core": (
            "era5", "reanalysis-era5-single-levels",
            lambda y, m: _era5_single_request(ERA5_CORE_VARS, y, m, area),
        ),
        "extras": (
            "era5x", "reanalysis-era5-single-levels",
            lambda y, m: _era5_single_request(ERA5_EXTRA_VARS, y, m, area),
        ),
        "pressure": (
            "era5pl", "reanalysis-era5-pressure-levels",
            lambda y, m: _era5_pressure_request(
                ERA5_PRESSURE_VARS, ERA5_PRESSURE_LEVELS, y, m, area,
            ),
        ),
    }

    missing: dict[str, list[str]] = {fam: [] for fam in families}
    skipped = downloaded = 0

    for i, (year, month) in enumerate(months, start=1):
        tag = f"{year}-{month:02d}"
        for family, (prefix, dataset, build_request) in families.items():
            target = cache_dir / f"{prefix}_{site.name}_{tag}.nc"
            if target.exists():
                skipped += 1
                continue
            ok = _retrieve_with_retry(
                client, dataset, build_request(year, month), str(target),
                label=f"{family} {tag} ({i}/{n})",
            )
            if ok:
                downloaded += 1
            else:
                missing[family].append(tag)

    logger.info("ERA5 extension pass done: %d already cached, %d downloaded, "
                "%d still missing", skipped, downloaded,
                sum(len(v) for v in missing.values()))
    for family, tags in missing.items():
        if tags:
            logger.warning("still missing %s: %s", family, ", ".join(tags))
    return {fam: tags for fam, tags in missing.items() if tags}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERA5 months for a config's date range (resumable)."
    )
    parser.add_argument("--config", default="config_10yr.yaml")
    args = parser.parse_args()
    missing = download_era5_range(load_config(args.config))
    if missing:
        logger.warning("Re-run to retry the months listed above.")
    else:
        logger.info("Cache complete for this date range.")


if __name__ == "__main__":
    main()
