import numpy as np
import pandas as pd
import pytest

from src.stage2_powercurve import (
    air_density,
    load_turbine,
    turbine_spec,
    windspeed_to_power,
)

TURBINE_NAME = "E-101/3050"
HUB = 100.0


@pytest.fixture(scope="module")
def turbine():
    return load_turbine(TURBINE_NAME, HUB)


@pytest.fixture(scope="module")
def spec(turbine):
    return turbine_spec(turbine, TURBINE_NAME)


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, name="ws")


def test_spec_reasonable(spec):
    assert spec.nominal_power_w == pytest.approx(3_050_000, rel=0.05)
    assert 1.0 <= spec.cut_in_ms <= 4.0
    assert 10.0 <= spec.rated_ms <= 16.0
    assert 20.0 <= spec.cut_out_ms <= 30.0


def test_power_below_cut_in_is_zero(turbine, spec):
    ws = _series([0.0, 0.5, 1.0, spec.cut_in_ms * 0.5])
    p = windspeed_to_power(ws, turbine)
    assert (p == 0).all()


def test_power_above_cut_out_is_zero(turbine, spec):
    ws = _series([spec.cut_out_ms + 1.0, spec.cut_out_ms + 5.0])
    p = windspeed_to_power(ws, turbine)
    assert (p == 0).all()


def test_power_at_rated_is_near_nominal(turbine, spec):
    ws = _series([spec.rated_ms, spec.rated_ms + 1.0, spec.rated_ms + 3.0])
    p = windspeed_to_power(ws, turbine)
    # Curve max may differ slightly from nameplate — allow 5% band.
    assert (p >= 0.95 * spec.nominal_power_w * 0.98).all()


def test_monotonic_between_cut_in_and_rated(turbine, spec):
    ws_values = np.linspace(spec.cut_in_ms, spec.rated_ms, 20)
    p = windspeed_to_power(_series(ws_values), turbine).to_numpy()
    diffs = np.diff(p)
    assert (diffs >= -1e-6).all(), f"Non-monotonic: {diffs}"


def test_density_correction_lowers_power_in_midrange(turbine):
    ws_mid = (turbine.power_curve["wind_speed"]
              [turbine.power_curve["value"].between(
                  0.2 * turbine.power_curve["value"].max(),
                  0.8 * turbine.power_curve["value"].max())].mean())
    ws = _series([ws_mid] * 4)
    # Warm, humid Ghana coast ~ 1.15 kg/m^3; standard is 1.225.
    p_std = windspeed_to_power(ws, turbine,
                               density=pd.Series([1.225] * 4, index=ws.index))
    p_hot = windspeed_to_power(ws, turbine,
                               density=pd.Series([1.10] * 4, index=ws.index))
    assert (p_hot < p_std).all()


def test_air_density_ideal_gas():
    # 15C, 101325 Pa → ~1.225 kg/m^3
    d = air_density(15.0, 101_325.0)
    assert d == pytest.approx(1.225, abs=0.01)
    # Hotter → less dense
    d_hot = air_density(35.0, 101_325.0)
    assert d_hot < d


def test_turbine_curve_nonempty(turbine):
    assert not turbine.power_curve.empty
    assert (turbine.power_curve["value"] >= 0).all()
