"""v6 — run the pipeline across Ghana's 16 regional capitals plus Anloga.

What this module is for
----------------------
Not "run the pipeline 16 times". The project has four forecast systems, all
measured at one site, and three documents record the generalisation question as
deferred — `docs/07` threat 2, thesis sections 4.14.4 and 4.9.3. The last of
those makes a *falsifiable* prediction this module exists to test:

    "It is one site and one horizon. At a location with weaker NWP skill, or at a
    shorter horizon where persistence is strong, reanalysis features could matter
    considerably more."

That is the claim behind the deployability result (dropping latency-blocked
reanalysis features cost nothing at Anloga). Twelve of the sixteen capitals are
low-wind interior sites, which is exactly where it should break if it breaks.

The pre-registration — predictions, the viability floor, gates with pass criteria
— is `docs/13_multi_site_extension.md`, written before any of this ran.

Design notes
------------
**Coordinates live in `sites/ghana_16.yaml`, never here.** The nine hand-written
`config_*.yaml` files in the repo root each carry their own copy of the Anloga
site block, which is how copies drift. Per-site configs are generated from the
registry at run time and are not committed.

**The ERA5 download happens once, not per site.** `fetch_regional_archive` pulls
one bounding box covering all 17 sites; CDS charges per field rather than per grid
cell, so that costs one site's requests instead of seventeen. Measured: 6.3 MB and
3.0 min per monthly request, so the 2010-2025 archive is ~2.9 GB and ~1 day,
against ~99 days if each site were fetched separately.

**Per-site outputs go to `outputs/sites/<name>/`** via `paths.outputs`, so the
existing artefact names are reused unchanged and 17 runs cannot overwrite each
other.

Usage
-----
    python -m src.multisite fetch       # one-time bbox archive download
    python -m src.multisite cells       # report the selected cell per site
    python -m src.multisite resource    # monthly climatology, all sites
    python -m src.multisite forecast    # 24 h ladder, all sites
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .bakeoff import DEFAULT_N_FOLDS as BAKEOFF_N_FOLDS
from .data_ingest import (fetch_regional_archive, load_land_sea_mask,
                          select_site_cell)
from .utils import Config, SiteConfig, ensure_dir, get_logger, load_config

logger = get_logger(__name__)

REGISTRY_PATH = Path("sites/ghana_16.yaml")
# Template supplying everything that is not site-specific — model knobs, NWP
# settings, horizon. The site block in it is ignored and replaced per site.
DEFAULT_TEMPLATE = "config_bakeoff.yaml"
# The resource product needs the long record: LOYO climatology, interannual
# spread and exceedance are only meaningful over many years. The forecast window
# (2024-03 onward, where the Open-Meteo archive starts) is a subset of this.
ARCHIVE_START, ARCHIVE_END = "2010-01-01", "2025-12-31"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate `sites/ghana_16.yaml`."""
    reg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("bbox", "defaults", "sites"):
        if key not in reg:
            raise ValueError(f"{path}: missing top-level '{key}'")
    names = [s["name"] for s in reg["sites"]]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate site names {sorted(dupes)}")
    for s in reg["sites"]:
        missing = {"name", "region", "latitude", "longitude", "coord_source",
                   "verified"} - set(s)
        if missing:
            raise ValueError(
                f"{path}: site {s.get('name', '?')} missing {sorted(missing)}. "
                f"Provenance fields are mandatory — an unsourced coordinate is "
                f"the quietest defect available here."
            )
    return reg


def site_config(reg: dict[str, Any], entry: dict[str, Any],
                template: str = DEFAULT_TEMPLATE,
                start: str | None = None, end: str | None = None) -> Config:
    """Build a Config for one registry entry, from the shared template."""
    config = copy.deepcopy(load_config(template))
    defaults = reg.get("defaults", {})
    config.site = SiteConfig(
        name=entry["name"],
        latitude=float(entry["latitude"]),
        longitude=float(entry["longitude"]),
        hub_height_m=float(entry.get("hub_height_m",
                                     defaults.get("hub_height_m", 100))),
    )
    config.data = replace(
        config.data,
        source="era5_regional",
        start=start or config.data.start,
        end=end or config.data.end,
        bbox=reg["bbox"],
        window=int(reg.get("defaults", {}).get("window", 1)),
        # Per-site override exists for exactly one site: Anloga, whose cell is
        # 88% water but must not move or Gate 1 becomes untestable and every
        # published v5 figure is silently re-based. See docs/13 section 3a.
        min_land_fraction=float(
            entry.get("min_land_fraction",
                      defaults.get("min_land_fraction", 0.5))),
    )
    if "turbine" in defaults:
        config.turbine = replace(config.turbine, name=defaults["turbine"])
    config.paths = replace(config.paths,
                           outputs=str(Path("outputs/sites") / entry["name"]))
    return config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch(reg: dict[str, Any], args: argparse.Namespace) -> None:
    """One-time shared archive download."""
    logger.info("Regional ERA5 archive %s -> %s, bbox %s",
                args.start, args.end, reg["bbox"])
    logger.info("This is ONE download shared by all %d sites. Do not start a "
                "second ingest process.", len(reg["sites"]))
    core, extra, press = fetch_regional_archive(
        args.start, args.end, Path(args.cache_dir), reg["bbox"])
    have = sum(p.exists() for p in core + extra + press)
    logger.info("Archive complete: %d/%d files present",
                have, len(core) + len(extra) + len(press))


def cmd_cells(reg: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    """Report which ERA5 cell each site resolves to, and its land fraction.

    Run this before any resource figure is quoted. A coastal site sitting on a
    sea cell reports an inflated resource and nothing downstream would notice —
    see docs/13 section 3.
    """
    mask = load_land_sea_mask(Path(args.cache_dir))
    if mask is None:
        raise SystemExit(
            f"No {Path(args.cache_dir)}/era5gh_lsm.nc — run "
            f"`python -m src.multisite fetch` first. Land fraction cannot be "
            f"verified without it, and 'unverified' is not 'land'."
        )
    lsm, mlats, mlons = mask
    defaults = reg.get("defaults", {})
    rows = []
    for entry in reg["sites"]:
        site = SiteConfig(name=entry["name"], latitude=float(entry["latitude"]),
                          longitude=float(entry["longitude"]),
                          hub_height_m=float(defaults.get("hub_height_m", 100)))
        thresh = float(entry.get("min_land_fraction",
                                 defaults.get("min_land_fraction", 0.5)))
        i, j = select_site_cell(mlats, mlons, site, lsm=lsm, mask_lats=mlats,
                                mask_lons=mlons, min_land_fraction=thresh)
        ni = int(np.abs(mlats - site.latitude).argmin())
        nj = int(np.abs(mlons - site.longitude).argmin())
        rows.append(dict(
            site=site.name, region=entry["region"],
            latitude=site.latitude, longitude=site.longitude,
            coastal=bool(entry.get("coastal", False)),
            nearest_lat=float(mlats[ni]), nearest_lon=float(mlons[nj]),
            nearest_land_fraction=round(float(lsm[ni, nj]), 4),
            cell_lat=float(mlats[i]), cell_lon=float(mlons[j]),
            cell_land_fraction=round(float(lsm[i, j]), 4),
            snapped=bool((i, j) != (ni, nj)),
            threshold=thresh,
        ))
    df = pd.DataFrame(rows)
    out = ensure_dir("outputs/reports") / "multisite_cells.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s", out)
    print(df.to_string(index=False))
    return df


def _site_entries(reg: dict[str, Any], only: list[str] | None) -> list[dict]:
    entries = reg["sites"]
    if only:
        wanted = set(only)
        unknown = wanted - {s["name"] for s in entries}
        if unknown:
            raise SystemExit(f"Unknown site(s): {sorted(unknown)}")
        entries = [s for s in entries if s["name"] in wanted]
    return entries


def _run_per_site(reg: dict[str, Any], args: argparse.Namespace, label: str,
                  marker: str, fn) -> pd.DataFrame:
    """Drive `fn(config)` across sites, resuming and never aborting the batch.

    One site failing must not cost the other sixteen — with a multi-hour batch
    that is the difference between a partial result and none.
    """
    entries = _site_entries(reg, args.sites)
    results, failures = [], []
    for n, entry in enumerate(entries, start=1):
        name = entry["name"]
        config = site_config(reg, entry, template=args.template,
                             start=args.start, end=args.end)
        done = Path(config.paths.outputs) / "reports" / marker
        if done.exists() and not args.force:
            logger.info("[%d/%d] %s: %s already present, skipping (--force to "
                        "redo)", n, len(entries), name, marker)
            continue
        logger.info("[%d/%d] %s — %s", n, len(entries), name, label)
        try:
            results.append({"site": name, **(fn(config) or {})})
        except Exception as exc:          # noqa: BLE001 — batch must continue
            logger.exception("%s FAILED: %s", name, exc)
            failures.append({"site": name, "error": repr(exc)})

    if failures:
        out = ensure_dir("outputs/reports") / f"multisite_{marker}_failures.json"
        out.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        logger.warning("%d/%d sites failed; see %s",
                       len(failures), len(entries), out)
    return pd.DataFrame(results)


def cmd_resource(reg: dict[str, Any], args: argparse.Namespace) -> None:
    from . import monthly

    def one(config: Config) -> dict:
        ensure_dir(Path(config.paths.outputs) / "reports")
        cfg_path = Path(config.paths.outputs) / "config.resolved.yaml"
        _dump_config(config, cfg_path)
        monthly.run(str(cfg_path))
        return {}

    _run_per_site(reg, args, "monthly resource climatology",
                  "monthly_climatology.csv", one)


def cmd_forecast(reg: dict[str, Any], args: argparse.Namespace) -> None:
    """The 24 h ladder at every site, on the published protocol.

    Deliberately identical to the single-site run in every respect except the
    site: same 4 rolling-origin folds, same seeds, same DM test, same report
    formatter (`bakeoff.run_phase_a_and_report`). Sites are only comparable to
    each other — and to the published Anloga table — if nothing else moves.

    Both feature sets always, because `operational` vs `study` is what tests
    prediction 3 (docs/13 §6b): thesis §4.9.3 predicts reanalysis features should
    matter more where NWP skill is weaker, and twelve of these sites are
    low-wind interior locations.

    Deep models are not run here. Phase B cost 12.9 h at one site; see docs/13 §5.
    """
    from .bakeoff import DEFAULT_SEEDS, run_phase_a_and_report

    feature_sets = (["study", "operational"] if args.features == "both"
                    else [args.features])

    def one(config: Config) -> dict:
        ensure_dir(Path(config.paths.outputs) / "reports")
        _dump_config(config, Path(config.paths.outputs) / "config.resolved.yaml")
        if args.horizon is not None:
            config.forecast.horizon_hours = int(args.horizon)
        # The bakeoff protocol already runs 5 seeds (bakeoff.DEFAULT_SEEDS) for
        # the stochastic arms, so every site inherits multi-seed coverage without
        # extra cost. Note this does NOT retire docs/12 item 4: that concerns the
        # `validate_nwp` headline run (the 0.7295 figure), a different
        # single-split experiment this does not touch.
        seeds = args.seeds or config.raw.get("bakeoff", {}).get(
            "seeds", DEFAULT_SEEDS)
        agg, out = run_phase_a_and_report(
            config, feature_sets, list(seeds), args.folds, per_fold=True)
        best = (agg[agg["tier"].str.startswith("2")]
                .sort_values("mae_mean").head(1))
        row: dict[str, Any] = {"report": str(out), "n_rows": len(agg)}
        if not best.empty:
            row["best_ml"] = best.iloc[0]["model"]
            row["best_mae"] = round(float(best.iloc[0]["mae_mean"]), 4)
        return row

    df = _run_per_site(reg, args, "24 h forecast ladder", "bakeoff.csv", one)
    if not df.empty:
        out = ensure_dir("outputs/reports") / "multisite_forecast.csv"
        df.to_csv(out, index=False)
        logger.info("Wrote %s", out)
        print(df.to_string(index=False))


def _dump_config(config: Config, path: Path) -> None:
    """Write the resolved per-site config so every run is reproducible.

    The generated configs are deliberately not committed — the registry plus this
    module is the source of truth — but each run leaves its own resolved copy
    beside its outputs, so a result can always be traced to the exact settings
    that produced it.
    """
    raw = copy.deepcopy(config.raw)
    raw["site"] = {
        "name": config.site.name, "latitude": config.site.latitude,
        "longitude": config.site.longitude,
        "hub_height_m": config.site.hub_height_m,
    }
    raw["data"] = {
        "source": config.data.source, "start": config.data.start,
        "end": config.data.end, "cache_dir": config.data.cache_dir,
        "bbox": config.data.bbox, "window": config.data.window,
        "min_land_fraction": config.data.min_land_fraction,
    }
    raw["turbine"] = {"name": config.turbine.name, "source": config.turbine.source}
    raw["paths"] = {"outputs": config.paths.outputs,
                    "processed": config.paths.processed}
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command",
                    choices=["fetch", "cells", "resource", "forecast"])
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--start", default=ARCHIVE_START)
    ap.add_argument("--end", default=ARCHIVE_END)
    ap.add_argument("--sites", nargs="*", default=None,
                    help="Restrict to these registry names (default: all).")
    ap.add_argument("--force", action="store_true",
                    help="Redo sites whose output marker already exists.")
    # forecast-arm knobs, mirroring src.bakeoff so the two cannot diverge
    ap.add_argument("--features", default="both",
                    choices=["study", "operational", "both"])
    ap.add_argument("--folds", type=int, default=BAKEOFF_N_FOLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()

    reg = load_registry(Path(args.registry))
    logger.info("Registry: %d sites, bbox %s", len(reg["sites"]), reg["bbox"])
    {"fetch": cmd_fetch, "cells": cmd_cells,
     "resource": cmd_resource, "forecast": cmd_forecast}[args.command](reg, args)


if __name__ == "__main__":
    main()
