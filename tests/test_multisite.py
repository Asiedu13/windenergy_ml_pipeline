"""Tests for the v6 regional (multi-site) ERA5 path and the site registry.

Two things here are load-bearing rather than incidental, and are tested as such:

* **`extract=None` must leave the per-site path bit-identical.** Every published
  Anloga figure comes through it. If a default changes, published results move
  silently — which is defect 9's exact failure mode, found only because a re-run
  happened to be compared against the published table.
* **Cell selection must be verifiable, not plausible.** A site resolved to the
  wrong grid cell produces a physically self-consistent series describing
  somewhere else, and nothing downstream can detect it. The land-mask snap exists
  because Cape Coast's nearest cell is open ocean (docs/13 section 3).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_ingest import (
    ERA5_CORE_VARS,
    ERA5_EXTRA_VARS,
    ERA5_PRESSURE_LEVELS,
    ERA5_PRESSURE_VARS,
    REGIONAL_CORE_PREFIX,
    REGIONAL_EXTRA_PREFIX,
    REGIONAL_LSM_NAME,
    REGIONAL_MIN_BYTES_PER_MONTH,
    REGIONAL_PRESSURE_PREFIX,
    RegionalExtract,
    _bbox_area,
    _check_regional_manifest,
    _era5_pressure_request,
    _era5_single_request,
    _months_arg,
    _regional_chunks,
    _retrieve_with_retry,
    _window_slice,
    regional_min_bytes,
    select_site_cell,
)
from src.multisite import load_registry, site_config
from src.utils import SiteConfig

GHANA_BBOX = {"north": 11.5, "west": -3.5, "south": 4.25, "east": 1.5}


def _grid(bbox: dict[str, float] = GHANA_BBOX) -> tuple[np.ndarray, np.ndarray]:
    """The ERA5 0.25 deg grid CDS returns for a bbox: latitudes descending."""
    lats = np.arange(bbox["north"], bbox["south"] - 1e-9, -0.25)
    lons = np.arange(bbox["west"], bbox["east"] + 1e-9, 0.25)
    return lats, lons


def _site(lat: float, lon: float, name: str = "t") -> SiteConfig:
    return SiteConfig(name=name, latitude=lat, longitude=lon, hub_height_m=100)


# ---------------------------------------------------------------------------
# bbox request building
# ---------------------------------------------------------------------------

def test_bbox_area_orders_as_cds_expects():
    assert _bbox_area(GHANA_BBOX) == [11.5, -3.5, 4.25, 1.5]


@pytest.mark.parametrize("missing", ["north", "west", "south", "east"])
def test_bbox_area_rejects_incomplete_box(missing):
    bbox = {k: v for k, v in GHANA_BBOX.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        _bbox_area(bbox)


def test_ghana_bbox_grid_shape_is_630_cells():
    """The figure docs/13 quotes, and the basis of the cost argument."""
    lats, lons = _grid()
    assert (len(lats), len(lons)) == (30, 21)
    assert len(lats) * len(lons) == 630


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

def test_nearest_cell_when_site_is_not_at_grid_centre():
    """Anloga resolves to (5.75, 1.00) — the cell its per-site cache uses.

    This is the identity half of gate 1: the regional lookup must independently
    land on the same cell the 1x2 per-site patch calls its centre.
    """
    lats, lons = _grid()
    i, j = select_site_cell(lats, lons, _site(5.79, 0.90))
    assert (lats[i], lons[j]) == (5.75, 1.00)


def test_every_registry_site_has_room_for_a_full_3x3_window():
    """A site at the bbox edge would get a clipped patch and asymmetric gradients."""
    reg = load_registry()
    lats, lons = _grid(reg["bbox"])
    for entry in reg["sites"]:
        i, j = select_site_cell(lats, lons, _site(entry["latitude"],
                                                 entry["longitude"]))
        assert 0 < i < len(lats) - 1, f"{entry['name']}: window hits a lat edge"
        assert 0 < j < len(lons) - 1, f"{entry['name']}: window hits a lon edge"


def _mask_all_land(lats, lons):
    return np.ones((len(lats), len(lons)), dtype=float)


def test_no_snap_when_nearest_cell_is_land():
    lats, lons = _grid()
    i0, j0 = select_site_cell(lats, lons, _site(5.79, 0.90))
    i, j = select_site_cell(lats, lons, _site(5.79, 0.90),
                            lsm=_mask_all_land(lats, lons),
                            mask_lats=lats, mask_lons=lons)
    assert (i, j) == (i0, j0)


def test_snaps_to_nearest_land_cell_when_nearest_is_water():
    """The Cape Coast case: nearest row is ocean, the row above is land."""
    lats, lons = _grid()
    lsm = _mask_all_land(lats, lons)
    i_near = int(np.abs(lats - 5.11).argmin())
    j_near = int(np.abs(lons - (-1.25)).argmin())
    assert lats[i_near] == 5.00          # the offshore row, as measured
    lsm[i_near, j_near] = 0.178          # the measured land fraction

    i, j = select_site_cell(lats, lons, _site(5.11, -1.25), lsm=lsm,
                            mask_lats=lats, mask_lons=lons,
                            min_land_fraction=0.5)
    assert (lats[i], lons[j]) == (5.25, -1.25)


def test_threshold_zero_disables_snapping():
    """The Anloga exemption (docs/13 section 3a).

    Its cell is 12% land. The rule would move it 26 km inland, which would make
    gate 1 untestable and silently re-base every published v5 figure, so the
    registry sets min_land_fraction: 0.0. That exemption must actually work.
    """
    lats, lons = _grid()
    lsm = _mask_all_land(lats, lons)
    i_near = int(np.abs(lats - 5.79).argmin())
    j_near = int(np.abs(lons - 0.90).argmin())
    lsm[i_near, j_near] = 0.124

    i, j = select_site_cell(lats, lons, _site(5.79, 0.90), lsm=lsm,
                            mask_lats=lats, mask_lons=lons,
                            min_land_fraction=0.0)
    assert (lats[i], lons[j]) == (5.75, 1.00)


def test_raises_when_no_land_cell_is_within_reach():
    """An ocean coordinate must fail loudly, not silently return seawater."""
    lats, lons = _grid()
    lsm = np.zeros((len(lats), len(lons)), dtype=float)
    with pytest.raises(RuntimeError, match="ocean"):
        select_site_cell(lats, lons, _site(5.11, -1.25, name="atlantic"),
                         lsm=lsm, mask_lats=lats, mask_lons=lons,
                         min_land_fraction=0.5)


def test_missing_mask_warns_rather_than_assuming_land(caplog):
    lats, lons = _grid()
    with caplog.at_level("WARNING"):
        select_site_cell(lats, lons, _site(5.11, -1.25, name="capecoast"))
    assert "no land-sea mask" in caplog.text


# ---------------------------------------------------------------------------
# Window slicing
# ---------------------------------------------------------------------------

def test_window_slice_returns_3x3_in_the_interior():
    arr = np.arange(4 * 30 * 21, dtype=float).reshape(4, 30, 21)
    out = _window_slice(arr, 10, 10, 1, 30, 21)
    assert out.shape == (4, 3, 3)


def test_window_slice_keeps_the_site_cell_at_the_window_centre():
    """`_centre_and_spatial` takes [n_lat//2, n_lon//2].

    If a clamped window were not symmetric about the site cell, that index would
    quietly return a neighbouring cell — the silent-wrong-location failure this
    module's docstring is about.
    """
    arr = np.arange(2 * 7 * 7, dtype=float).reshape(2, 7, 7)
    for i, j in [(3, 3), (0, 3), (6, 3), (3, 0), (3, 6), (0, 0), (6, 6)]:
        out = _window_slice(arr, i, j, 1, 7, 7)
        centre = out[:, out.shape[-2] // 2, out.shape[-1] // 2]
        np.testing.assert_array_equal(centre, arr[:, i, j])


def test_window_slice_passes_through_arrays_without_spatial_dims():
    assert _window_slice(None, 1, 1, 1, 3, 3) is None
    flat = np.arange(5, dtype=float)
    np.testing.assert_array_equal(_window_slice(flat, 1, 1, 1, 3, 3), flat)


def test_larger_window_half_widens_the_patch():
    arr = np.arange(2 * 30 * 21, dtype=float).reshape(2, 30, 21)
    assert _window_slice(arr, 10, 10, 2, 30, 21).shape == (2, 5, 5)


# ---------------------------------------------------------------------------
# The guard on published results
# ---------------------------------------------------------------------------

def test_regional_extract_defaults_match_the_documented_contract():
    e = RegionalExtract()
    assert e.half == 1
    assert e.min_land_fraction == 0.5
    assert e.lsm is None


def test_cropper_is_identity_when_extract_is_none():
    """`extract=None` must not touch the grid.

    The per-site path produces every published Anloga figure. This pins the
    default so a future change to the regional path cannot move those numbers.
    """
    from src.data_ingest import _cropper

    crop = _cropper(ds=None, site=_site(5.79, 0.90), extract=None)
    arr = np.arange(2 * 30 * 21, dtype=float).reshape(2, 30, 21)
    assert crop(arr) is arr


# ---------------------------------------------------------------------------
# Regional cache manifest — the defect-9 lesson applied in advance
# ---------------------------------------------------------------------------

def test_manifest_is_written_on_first_use(tmp_path):
    _check_regional_manifest(tmp_path, _bbox_area(GHANA_BBOX))
    recorded = json.loads((tmp_path / "era5gh_manifest.json").read_text())
    assert recorded["area"] == [11.5, -3.5, 4.25, 1.5]


def test_manifest_accepts_the_same_bbox_again(tmp_path):
    area = _bbox_area(GHANA_BBOX)
    _check_regional_manifest(tmp_path, area)
    _check_regional_manifest(tmp_path, area)        # must not raise


def test_manifest_refuses_a_different_bbox(tmp_path):
    """The regional filenames carry only the month, not the bbox.

    So a cache built for one box would be silently reused for another and every
    spatial column would describe the wrong neighbourhood — a cache key encoding
    less than the request, which is exactly defect 9.
    """
    _check_regional_manifest(tmp_path, _bbox_area(GHANA_BBOX))
    other = dict(GHANA_BBOX, north=12.0)
    with pytest.raises(RuntimeError, match="built for area"):
        _check_regional_manifest(tmp_path, _bbox_area(other))


# ---------------------------------------------------------------------------
# Download chunking
# ---------------------------------------------------------------------------

def test_months_arg_accepts_a_single_month_or_a_list():
    assert _months_arg(7) == ["07"]
    assert _months_arg([1, 2, 12]) == ["01", "02", "12"]


def test_monthly_chunks_are_one_unit_per_calendar_month():
    units = _regional_chunks("2024-03-01", "2024-05-31", "month")
    assert [u[0] for u in units] == ["2024-03", "2024-04", "2024-05"]
    assert [u[2] for u in units] == [3, 4, 5]


def test_yearly_chunks_group_by_year_and_do_not_widen_the_range():
    """A range starting mid-year must ask for that year's months only.

    Widening to a full year would quietly download data outside the configured
    window and, worse, make the file's coverage disagree with its name.
    """
    units = _regional_chunks("2024-03-01", "2025-02-28", "year")
    assert [u[0] for u in units] == ["2024", "2025"]
    assert units[0][2] == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert units[1][2] == [1, 2]


def test_chunk_mode_must_be_month_or_year():
    with pytest.raises(ValueError, match="month.*year|chunk"):
        _regional_chunks("2024-01-01", "2024-02-28", "fortnight")


def test_request_builders_pass_a_month_list_through():
    req = _era5_single_request(ERA5_CORE_VARS, 2012, [1, 2, 3], _bbox_area(GHANA_BBOX))
    assert req["year"] == "2012"
    assert req["month"] == ["01", "02", "03"]
    assert len(req["time"]) == 24 and len(req["day"]) == 31
    preq = _era5_pressure_request(ERA5_PRESSURE_VARS, ERA5_PRESSURE_LEVELS,
                                  2012, 5, _bbox_area(GHANA_BBOX))
    assert preq["month"] == ["05"]
    assert preq["pressure_level"] == ERA5_PRESSURE_LEVELS


def test_floors_scale_with_coverage():
    """A flat monthly floor would pass a yearly file that is 92% truncated."""
    one = regional_min_bytes(REGIONAL_CORE_PREFIX, 1)
    twelve = regional_min_bytes(REGIONAL_CORE_PREFIX, 12)
    assert one == REGIONAL_MIN_BYTES_PER_MONTH[REGIONAL_CORE_PREFIX]
    assert twelve == one * 12


def test_startup_validation_infers_coverage_from_the_filename(tmp_path):
    """`..._2010-07.nc` is one month; `..._2010.nc` is twelve.

    A multi-month file just above the *monthly* floor is badly truncated and must
    be caught; the same size in a monthly file is fine.
    """
    from src.data_ingest import _drop_truncated_regional_files

    monthly = tmp_path / f"{REGIONAL_CORE_PREFIX}_2010-07.nc"
    yearly = tmp_path / f"{REGIONAL_CORE_PREFIX}_2011.nc"
    size = regional_min_bytes(REGIONAL_CORE_PREFIX, 1) + 1_000
    monthly.write_bytes(b"x" * size)
    yearly.write_bytes(b"x" * size)

    removed = _drop_truncated_regional_files(tmp_path)

    assert removed == [yearly], "yearly file above only the monthly floor must go"
    assert monthly.exists(), "monthly file above its own floor must stay"


# ---------------------------------------------------------------------------
# Truncation guard — a short file must not become permanent
# ---------------------------------------------------------------------------

class _FakeClient:
    """Stands in for cdsapi.Client, writing a file of a chosen size."""

    def __init__(self, sizes: list[int]):
        self.sizes = list(sizes)
        self.calls = 0

    def retrieve(self, dataset, request, target):      # noqa: ARG002
        size = self.sizes[min(self.calls, len(self.sizes) - 1)]
        self.calls += 1
        Path(target).write_bytes(b"x" * size)


def test_retrieve_accepts_a_file_above_the_floor(tmp_path):
    client = _FakeClient([3_000_000])
    target = tmp_path / "era5gh_core_2010-07.nc"
    assert _retrieve_with_retry(client, "ds", {}, str(target), label="t",
                                min_bytes=2_500_000) is True
    assert target.exists()
    assert client.calls == 1


def test_retrieve_deletes_a_truncated_file_and_retries(tmp_path):
    """The failure this exists to prevent.

    `path.exists()` is the only re-download guard in the module, so a short file
    left behind would be treated as a complete month forever. It must be removed,
    not merely reported.
    """
    client = _FakeClient([1_000, 3_000_000])      # truncated, then good
    target = tmp_path / "era5gh_core_2010-07.nc"
    assert _retrieve_with_retry(client, "ds", {}, str(target), label="t",
                                min_bytes=2_500_000, wait_seconds=0) is True
    assert client.calls == 2
    assert target.stat().st_size == 3_000_000


def test_retrieve_gives_up_and_leaves_no_file_behind(tmp_path):
    """After exhausting attempts the target must be absent, so the next run retries."""
    client = _FakeClient([1_000])
    target = tmp_path / "era5gh_core_2010-07.nc"
    assert _retrieve_with_retry(client, "ds", {}, str(target), label="t",
                                max_attempts=3, wait_seconds=0,
                                min_bytes=2_500_000) is False
    assert not target.exists()
    assert client.calls == 3


def test_retrieve_default_floor_is_zero_so_the_per_site_path_is_unchanged(tmp_path):
    """Per-site monthly files are ~150 KB against the regional 2.8-6.5 MB.

    A global floor would reject every per-site download, so the floor is per call
    and defaults to 0 — which also keeps every published v1-v5 figure's data path
    byte-identical.
    """
    client = _FakeClient([150_000])
    target = tmp_path / "era5_anloga_2024-06.nc"
    assert _retrieve_with_retry(client, "ds", {}, str(target), label="t") is True
    assert client.calls == 1


def test_retrieve_treats_a_missing_file_as_truncated(tmp_path):
    """A retrieve that raises nothing but writes nothing must not count as success."""
    class _Silent:
        calls = 0

        def retrieve(self, dataset, request, target):      # noqa: ARG002
            _Silent.calls += 1

    target = tmp_path / "era5gh_core_2010-07.nc"
    assert _retrieve_with_retry(_Silent(), "ds", {}, str(target), label="t",
                                max_attempts=2, wait_seconds=0,
                                min_bytes=1) is False
    assert not target.exists()


def test_startup_validation_removes_a_partial_file_left_by_a_killed_process(tmp_path):
    """The hole `min_bytes` alone does not close.

    `min_bytes` runs after a retrieve returns. A process killed mid-write leaves a
    partial file that no later run inspects, because `path.exists()` short-circuits
    first — so the partial month silently feeds every site that reads it. Resumes
    must therefore re-validate what is already on disk.
    """
    from src.data_ingest import _drop_truncated_regional_files

    good = tmp_path / f"{REGIONAL_CORE_PREFIX}_2010-01.nc"
    partial = tmp_path / f"{REGIONAL_CORE_PREFIX}_2010-07.nc"
    other_good = tmp_path / f"{REGIONAL_EXTRA_PREFIX}_2010-01.nc"
    good.write_bytes(b"x" * 6_400_000)
    partial.write_bytes(b"x" * 900_000)
    other_good.write_bytes(b"x" * 2_900_000)

    removed = _drop_truncated_regional_files(tmp_path)

    assert removed == [partial]
    assert not partial.exists()
    assert good.exists() and other_good.exists()


def test_startup_validation_leaves_a_healthy_cache_alone(tmp_path):
    from src.data_ingest import _drop_truncated_regional_files

    for prefix, size in [(REGIONAL_CORE_PREFIX, 6_400_000),
                         (REGIONAL_EXTRA_PREFIX, 2_900_000),
                         (REGIONAL_PRESSURE_PREFIX, 5_100_000)]:
        (tmp_path / f"{prefix}_2010-01.nc").write_bytes(b"x" * size)
    assert _drop_truncated_regional_files(tmp_path) == []
    assert len(list(tmp_path.glob("*.nc"))) == 3


def test_startup_validation_ignores_the_mask_and_manifest(tmp_path):
    """`era5gh_lsm.nc` is ~25 KB by design and must not be mistaken for truncated."""
    from src.data_ingest import _drop_truncated_regional_files

    (tmp_path / REGIONAL_LSM_NAME).write_bytes(b"x" * 25_000)
    (tmp_path / "era5gh_manifest.json").write_text("{}")
    assert _drop_truncated_regional_files(tmp_path) == []
    assert (tmp_path / REGIONAL_LSM_NAME).exists()
    assert (tmp_path / "era5gh_manifest.json").exists()


def test_regional_floors_sit_below_the_smallest_observed_real_files():
    """Floors must not trip on a legitimate short month.

    Smallest observed are February (28 days): core 5.87 MB, extra 2.68 MB,
    pressure 4.71 MB. Month-length variation is only ~10%, so the floors should
    sit comfortably under those and still be non-trivial.
    """
    observed_feb = {
        REGIONAL_CORE_PREFIX: 5_871_500,
        REGIONAL_EXTRA_PREFIX: 2_678_400,
        REGIONAL_PRESSURE_PREFIX: 4_711_800,
    }
    for prefix, floor in REGIONAL_MIN_BYTES_PER_MONTH.items():
        smallest = observed_feb[prefix]
        assert floor < smallest * 0.75, f"{prefix}: floor too close to a real file"
        assert floor > smallest * 0.25, f"{prefix}: floor too low to catch much"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_holds_16_capitals_plus_anloga():
    reg = load_registry()
    sites = reg["sites"]
    assert len(sites) == 17
    refs = [s for s in sites if s.get("reference")]
    assert [s["name"] for s in refs] == ["anloga"]
    assert len(sites) - len(refs) == 16


def test_registry_regions_are_the_16_distinct_regions():
    reg = load_registry()
    regions = [s["region"] for s in reg["sites"] if not s.get("reference")]
    assert len(set(regions)) == 16


def test_registry_coordinates_fall_inside_the_bbox():
    reg = load_registry()
    bb = reg["bbox"]
    for s in reg["sites"]:
        assert bb["south"] < s["latitude"] < bb["north"], s["name"]
        assert bb["west"] < s["longitude"] < bb["east"], s["name"]


def test_registry_anloga_carries_its_exemption():
    """If this ever silently flips to 0.5, Anloga moves 26 km and gate 1 breaks."""
    reg = load_registry()
    anloga = next(s for s in reg["sites"] if s["name"] == "anloga")
    assert anloga["min_land_fraction"] == 0.0


def _write_registry(tmp_path: Path, sites: list[dict]) -> Path:
    path = tmp_path / "reg.yaml"
    path.write_text(yaml.safe_dump(
        {"bbox": GHANA_BBOX, "defaults": {"hub_height_m": 100}, "sites": sites},
    ), encoding="utf-8")
    return path


def _entry(name: str, **kw) -> dict:
    base = dict(name=name, region="R", latitude=6.0, longitude=-1.0,
                coord_source="test", verified="2026-10-01")
    base.update(kw)
    return base


def test_registry_rejects_duplicate_site_names(tmp_path):
    path = _write_registry(tmp_path, [_entry("a"), _entry("a")])
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(path)


@pytest.mark.parametrize("field", ["region", "coord_source", "verified"])
def test_registry_requires_provenance_fields(tmp_path, field):
    """An unsourced coordinate is the quietest defect available here."""
    entry = _entry("a")
    del entry[field]
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(ValueError, match=field):
        load_registry(path)


@pytest.mark.parametrize("key", ["bbox", "defaults", "sites"])
def test_registry_requires_top_level_keys(tmp_path, key):
    doc = {"bbox": GHANA_BBOX, "defaults": {}, "sites": [_entry("a")]}
    del doc[key]
    path = tmp_path / "reg.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        load_registry(path)


# ---------------------------------------------------------------------------
# Per-site config generation
# ---------------------------------------------------------------------------

def test_site_config_is_generated_from_the_registry():
    reg = load_registry()
    entry = next(s for s in reg["sites"] if s["name"] == "tamale")
    config = site_config(reg, entry)

    assert config.site.name == "tamale"
    assert config.site.latitude == pytest.approx(9.40)
    assert config.data.source == "era5_regional"
    assert config.data.bbox == reg["bbox"]
    assert config.data.window == 1
    assert config.data.min_land_fraction == 0.5
    assert config.turbine.name == reg["defaults"]["turbine"]


def test_site_config_namespaces_outputs_per_site():
    """Without this, 17 runs overwrite each other's bakeoff.csv."""
    reg = load_registry()
    names = [s["name"] for s in reg["sites"]]
    outs = {site_config(reg, s).paths.outputs for s in reg["sites"]}
    assert len(outs) == len(names)
    for name in names:
        assert str(Path("outputs/sites") / name) in outs


def test_site_config_passes_through_the_anloga_exemption():
    reg = load_registry()
    entry = next(s for s in reg["sites"] if s["name"] == "anloga")
    assert site_config(reg, entry).data.min_land_fraction == 0.0


def test_site_config_keeps_one_turbine_and_hub_height_everywhere():
    """Comparability across sites is the point; a per-site machine is a different
    study, and a low-wind turbine would flatter the interior capitals."""
    reg = load_registry()
    configs = [site_config(reg, s) for s in reg["sites"]]
    assert len({c.turbine.name for c in configs}) == 1
    assert len({c.site.hub_height_m for c in configs}) == 1


def test_resolved_config_round_trips_through_load_config(tmp_path):
    """Every per-site run writes its resolved config and the stages re-read it.

    If the dump is not loadable, all 17 sites fail at step one — and in a batch
    that takes hours, finding out late is expensive. Also pins that the template's
    NWP block survives, since the forecast arm is worthless without it.
    """
    from src.multisite import _dump_config
    from src.utils import load_config

    reg = load_registry()
    entry = next(s for s in reg["sites"] if s["name"] == "cape_coast")
    config = site_config(reg, entry)
    path = tmp_path / "resolved.yaml"
    _dump_config(config, path)

    back = load_config(str(path))
    assert back.site.name == "cape_coast"
    assert back.site.latitude == pytest.approx(5.11)
    assert back.data.source == "era5_regional"
    assert back.data.bbox == reg["bbox"]
    assert back.data.window == config.data.window
    assert back.data.min_land_fraction == config.data.min_land_fraction
    assert back.paths.outputs == config.paths.outputs
    assert back.turbine.name == config.turbine.name
    assert back.nwp.enabled is True
    assert back.nwp.models == config.nwp.models


# ---------------------------------------------------------------------------
# Window resolution and the no-download guard
#
# These two exist because a *training* run must never be able to start an ingest.
# `--start` used to default to ARCHIVE_START for every command, and since
# `site_config` resolves `start or config.data.start`, that non-None default made
# the template's 2024-03 unreachable: a bare `multisite forecast` asked for 2010
# onward, and `_fetch_era5_regional` -> `fetch_regional_archive` downloads what is
# missing. Measured against the real cache, that was 432 of 576 files — over a
# week of CDS requests in place of training, with nothing in the output saying so.
# ---------------------------------------------------------------------------

def test_forecast_window_defaults_to_the_template_not_the_archive():
    """The pre-registered study window lives in the template, not in this module.

    The window is not a free choice: the Open-Meteo previous-runs archive is
    all-null before 2024-03, so the overlap with ERA5 truth *is* the study period.
    """
    from src.multisite import resolve_window

    assert resolve_window("forecast", None, None) == (None, None)

    reg = load_registry()
    entry = next(s for s in reg["sites"] if s["name"] == "anloga")
    start, end = resolve_window("forecast", None, None)
    config = site_config(reg, entry, start=start, end=end)
    assert (config.data.start, config.data.end) == ("2024-03-01", "2025-12-31")


@pytest.mark.parametrize("command", ["fetch", "resource"])
def test_archive_commands_still_default_to_the_long_record(command):
    """LOYO climatology, interannual spread and exceedance need the full record."""
    from src.multisite import ARCHIVE_END, ARCHIVE_START, resolve_window

    assert resolve_window(command, None, None) == (ARCHIVE_START, ARCHIVE_END)


@pytest.mark.parametrize("command", ["fetch", "resource", "forecast", "cells"])
def test_an_explicit_window_always_wins(command):
    from src.multisite import resolve_window

    assert resolve_window(command, "2019-01-01", "2019-12-31") == (
        "2019-01-01", "2019-12-31")


def test_coverage_query_issues_no_request_and_creates_no_file(tmp_path, monkeypatch):
    """`download=False` must be a pure question about disk.

    `cdsapi` is made un-importable for the duration: if this path ever grows a
    client the test fails rather than quietly submitting requests from a test run.
    """
    import builtins

    from src.data_ingest import fetch_regional_archive

    real_import = builtins.__import__

    def no_cdsapi(name, *args, **kwargs):
        if name == "cdsapi":
            raise AssertionError("download=False must not import cdsapi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cdsapi)

    core, extra, press = fetch_regional_archive(
        "2024-03-01", "2024-05-31", tmp_path, GHANA_BBOX, download=False)

    assert [p.name for p in core] == [
        f"{REGIONAL_CORE_PREFIX}_2024-0{m}.nc" for m in (3, 4, 5)]
    assert len(extra) == len(press) == 3
    assert not any(p.exists() for p in (*core, *extra, *press))
    # only the manifest the guard writes on first use — no .nc files
    assert sorted(p.name for p in tmp_path.iterdir()) == ["era5gh_manifest.json"]


def test_coverage_query_returns_the_same_paths_as_a_download_would(tmp_path):
    """The guard is only meaningful if it asks about the files training reads."""
    from src.data_ingest import fetch_regional_archive

    a = fetch_regional_archive("2024-03-01", "2024-04-30", tmp_path,
                               GHANA_BBOX, download=False)
    b = fetch_regional_archive("2024-03-01", "2024-04-30", tmp_path,
                               GHANA_BBOX, download=False)
    assert a == b
    tags = {p.name.rsplit("_", 1)[1] for group in a for p in group}
    assert tags == {"2024-03.nc", "2024-04.nc"}


def test_training_refuses_a_window_it_would_have_to_download(tmp_path):
    """A hard stop, not a warning.

    A partially-covered window still produces a report, and a report over a
    quietly different record length is not comparable to the published table or to
    the other sites — which is the whole point of the multi-site study.
    """
    from src.multisite import require_cached_window

    reg = load_registry()
    entry = next(s for s in reg["sites"] if s["name"] == "anloga")
    config = site_config(reg, entry, start="2024-03-01", end="2024-04-30")

    with pytest.raises(SystemExit) as excinfo:
        require_cached_window(reg, config, cache_dir=str(tmp_path))

    msg = str(excinfo.value)
    assert "6 of 6" in msg
    assert "era5gh_core_2024-03.nc" in msg
    assert "will NOT download" in msg
    assert "multisite fetch" in msg          # tells you how to fix it
    assert not list(tmp_path.glob("*.nc"))


def test_training_accepts_a_fully_cached_window(tmp_path):
    from src.multisite import require_cached_window

    reg = load_registry()
    entry = next(s for s in reg["sites"] if s["name"] == "anloga")
    config = site_config(reg, entry, start="2024-03-01", end="2024-03-31")

    for prefix, floor in REGIONAL_MIN_BYTES_PER_MONTH.items():
        (tmp_path / f"{prefix}_2024-03.nc").write_bytes(b"\0" * (floor + 1))

    require_cached_window(reg, config, cache_dir=str(tmp_path))   # must not raise


# ---------------------------------------------------------------------------
# Report artefacts must travel together
# ---------------------------------------------------------------------------

def test_out_moves_the_csv_with_the_report(tmp_path, monkeypatch):
    """`--out` used to redirect only the markdown.

    The CSVs kept going to `paths.outputs/reports/{stem}.csv`, so pointing
    `--out` elsewhere while passing a per-site resolved config overwrote that
    site's headline `bakeoff.csv` in place — leaving a report and a CSV from
    different runs under the same name with nothing saying so. It happened to
    anloga and cape_coast on 2026-10-09 and was caught only by an mtime that did
    not match its own report's.
    """
    import pandas as pd

    from src import bakeoff
    from src.utils import load_config

    config = load_config("config_bakeoff.yaml")
    config.paths = replace(config.paths, outputs=str(tmp_path / "site"))

    records = [
        bakeoff.RunRecord(model="xgboost", tier="2 ML", feature_set="study",
                          fold="fold1", seed=42, horizon=24, mae=0.8,
                          rmse=1.0, r2=0.5, skill=0.3, seconds=0.1),
    ]
    monkeypatch.setattr(bakeoff, "run_phase_a", lambda *a, **k: records)

    elsewhere = tmp_path / "elsewhere" / "anloga.md"
    _, out_path = bakeoff.run_phase_a_and_report(
        config, ["study"], [42], n_folds=4, per_fold=True, out=str(elsewhere))

    assert out_path == elsewhere
    # every artefact lands beside the report...
    assert (elsewhere.parent / "anloga.csv").exists()
    assert (elsewhere.parent / "anloga_folds.csv").exists()
    # ...and nothing is written under the config's own reports dir
    site_reports = tmp_path / "site" / "reports"
    assert not list(site_reports.glob("*.csv"))

    df = pd.read_csv(elsewhere.parent / "anloga.csv")
    assert df.model.tolist() == ["xgboost"]


def test_without_out_the_artefacts_stay_under_paths_outputs(tmp_path, monkeypatch):
    """The per-site default: 17 sites must not write over each other."""
    from src import bakeoff
    from src.utils import load_config

    config = load_config("config_bakeoff.yaml")
    config.paths = replace(config.paths, outputs=str(tmp_path / "site"))
    monkeypatch.setattr(bakeoff, "run_phase_a", lambda *a, **k: [
        bakeoff.RunRecord(model="xgboost", tier="2 ML", feature_set="study",
                          fold="fold1", seed=42, horizon=24, mae=0.8,
                          rmse=1.0, r2=0.5, skill=0.3, seconds=0.1)])

    _, out_path = bakeoff.run_phase_a_and_report(config, ["study"], [42])

    reports = tmp_path / "site" / "reports"
    assert out_path == reports / "bakeoff.md"
    assert (reports / "bakeoff.csv").exists()
