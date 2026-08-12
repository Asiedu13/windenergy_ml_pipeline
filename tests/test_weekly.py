"""Tests for the valid-time weekly forecasting frame.

The reframing in `src/weekly.py` removes the horizon shift entirely — forecast
and truth share an index — so the class of alignment bug that cost this project
two iterations cannot arise. These tests pin that, and pin the lead isolation
that makes the leak guard structural rather than a filter.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    Config,
    DataConfig,
    ForecastConfig,
    ModelConfig,
    PathsConfig,
    SiteConfig,
    TurbineConfig,
)
from src.weekly import (
    available_leads,
    build_valid_time_frame,
    climatology_predict,
    lead_columns,
    models_present,
)

HOURS = 24 * 120


def _config() -> Config:
    return Config(
        site=SiteConfig(name="t", latitude=5.79, longitude=0.9, hub_height_m=100),
        data=DataConfig(source="era5", start="2024-03-01", end="2024-06-28",
                        cache_dir="/tmp"),
        turbine=TurbineConfig(name="E-101/3050"),
        forecast=ForecastConfig(horizon_hours=168),
        model=ModelConfig(seed=42),
        paths=PathsConfig(),
        raw={},
    )


def _weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    t = np.arange(len(index))
    return pd.DataFrame(
        {
            "wind_speed_10m": 4 + 1.5 * np.sin(2 * np.pi * t / 24),
            "wind_speed_100m": 7 + 2.5 * np.sin(2 * np.pi * t / 24),
            "roughness": 0.15,
        },
        index=index,
    )


def _nwp(index: pd.DatetimeIndex, leads=(1, 2, 3), models=("ecmwf_ifs025", "gfs_global")):
    data = {}
    for i, m in enumerate(models):
        for d in leads:
            # Each (model, lead) gets a distinguishable series so tests can tell
            # which one a frame actually picked up.
            data[f"nwp_{m}_wind_speed_100m_d{d}"] = (
                7 + 2.5 * np.sin(2 * np.pi * np.arange(len(index)) / 24) + i + d
            )
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# Lead isolation — the structural version of the leak guard
# ---------------------------------------------------------------------------

def test_a_lead_frame_contains_only_that_leads_columns():
    """The lead-N model must never see a shorter (future-issued) lead.

    In the feature-time pipeline this needs an explicit filter. Here it falls
    out of the construction, and this test is what pins that.
    """
    idx = pd.date_range("2024-03-01", periods=HOURS, freq="h", tz="UTC")
    X, y, _ = build_valid_time_frame(_weather(idx), _nwp(idx), 3, _config())
    forecast_cols = [c for c in X.columns if c.startswith("nwp_") and "_d" in c]
    assert forecast_cols, "expected some forecast columns"
    assert all(c.endswith("_d3") for c in forecast_cols), forecast_cols


def test_lead_columns_and_available_leads_agree():
    idx = pd.date_range("2024-03-01", periods=200, freq="h", tz="UTC")
    nwp = _nwp(idx, leads=(1, 4, 7))
    assert available_leads(nwp) == [1, 4, 7]
    assert all(c.endswith("_d4") for c in lead_columns(nwp, 4))
    assert lead_columns(nwp, 2) == []


def test_missing_lead_raises_rather_than_returning_an_empty_frame():
    idx = pd.date_range("2024-03-01", periods=200, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="lead day 7"):
        build_valid_time_frame(_weather(idx), _nwp(idx, leads=(1, 2)), 7, _config())


# ---------------------------------------------------------------------------
# Alignment — no shift, by construction
# ---------------------------------------------------------------------------

def test_a_perfect_forecast_reproduces_the_target_exactly():
    """A forecast column equal to the truth must round-trip with zero error.

    The signal period is 37 h — deliberately not commensurate with 24 h — so no
    accidentally shifted frame can pass by coincidence. Same trick as the
    feature-time alignment test, applied to the new framing.
    """
    idx = pd.date_range("2024-03-01", periods=HOURS, freq="h", tz="UTC")
    t = np.arange(len(idx))
    weather = pd.DataFrame(
        {"wind_speed_100m": 8 + 3 * np.sin(2 * np.pi * t / 37), "roughness": 0.15},
        index=idx,
    )
    cfg = _config()
    from src.features import extrapolate_hub_wind
    hub = extrapolate_hub_wind(weather, cfg.site.hub_height_m,
                               reference_height_m=100.0)
    nwp = pd.DataFrame({"nwp_perfect_wind_speed_100m_d5": hub.to_numpy()}, index=idx)

    X, y, _ = build_valid_time_frame(weather, nwp, 5, cfg)
    np.testing.assert_allclose(
        X["nwp_perfect_wind_speed_100m_d5"].to_numpy(), y.to_numpy(), atol=1e-9
    )


def test_target_is_the_observation_at_the_same_timestamp():
    """Valid-time framing means y[τ] is hub(τ) — no offset anywhere."""
    idx = pd.date_range("2024-03-01", periods=HOURS, freq="h", tz="UTC")
    weather = _weather(idx)
    cfg = _config()
    X, y, _ = build_valid_time_frame(weather, _nwp(idx), 2, cfg)

    from src.features import extrapolate_hub_wind
    hub = extrapolate_hub_wind(weather, cfg.site.hub_height_m,
                               reference_height_m=100.0)
    np.testing.assert_allclose(y.to_numpy(), hub.reindex(y.index).to_numpy())


# ---------------------------------------------------------------------------
# Ensemble bookkeeping
# ---------------------------------------------------------------------------

def test_ensemble_membership_is_recorded_per_lead():
    """ICON stops at d6, so the day-7 model runs on two members, not three.

    That fact has to travel with any day-7 number rather than being discovered
    afterwards, so the frame reports which models it actually had.
    """
    idx = pd.date_range("2024-03-01", periods=HOURS, freq="h", tz="UTC")
    three = _nwp(idx, leads=(6,), models=("ecmwf_ifs025", "gfs_global", "icon_global"))
    two = _nwp(idx, leads=(7,), models=("ecmwf_ifs025", "gfs_global"))
    nwp = pd.concat([three, two], axis=1)

    _, _, m6 = build_valid_time_frame(_weather(idx), nwp, 6, _config())
    _, _, m7 = build_valid_time_frame(_weather(idx), nwp, 7, _config())
    assert m6 == ["ecmwf_ifs025", "gfs_global", "icon_global"]
    assert m7 == ["ecmwf_ifs025", "gfs_global"]


def test_spread_feature_requires_more_than_one_member():
    idx = pd.date_range("2024-03-01", periods=HOURS, freq="h", tz="UTC")
    solo = _nwp(idx, leads=(1,), models=("ecmwf_ifs025",))
    X, _, _ = build_valid_time_frame(_weather(idx), solo, 1, _config())
    assert "nwp_spread" not in X.columns

    duo = _nwp(idx, leads=(1,), models=("ecmwf_ifs025", "gfs_global"))
    X2, _, _ = build_valid_time_frame(_weather(idx), duo, 1, _config())
    assert "nwp_spread" in X2.columns
    assert (X2["nwp_spread"] >= 0).all()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def test_climatology_recovers_a_pure_diurnal_cycle():
    """With a signal that repeats exactly by hour, climatology should nail it."""
    idx = pd.date_range("2024-03-01", periods=24 * 60, freq="h", tz="UTC")
    y = pd.Series(5 + 2 * np.sin(2 * np.pi * idx.hour / 24), index=idx)
    pred = climatology_predict(y.iloc[:24 * 40], idx[24 * 40:])
    np.testing.assert_allclose(pred, y.iloc[24 * 40:].to_numpy(), atol=1e-9)


def test_climatology_falls_back_when_a_month_hour_is_unseen():
    idx = pd.date_range("2024-03-01", periods=24 * 30, freq="h", tz="UTC")
    y = pd.Series(np.linspace(4, 8, len(idx)), index=idx)
    future = pd.date_range("2024-09-01", periods=48, freq="h", tz="UTC")
    pred = climatology_predict(y, future)          # September never seen
    np.testing.assert_allclose(pred, np.full(48, y.mean()))
