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
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_ingest import (
    REGIONAL_LSM_NAME,
    RegionalExtract,
    _bbox_area,
    _check_regional_manifest,
    _window_slice,
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
