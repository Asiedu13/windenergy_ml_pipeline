"""Tests for the 30-day outlook (M1).

The outlook is the arm most likely to produce a misleading number, so the tests
concentrate on the three ways it could:

* asking the forecast API for years it does not cover (a live HTTP 400, invisible
  to any test that does not exercise the clamp);
* letting the head see data from the month it is predicting;
* reporting a point estimate whose interval would have spanned zero.
"""

import numpy as np
import pandas as pd
import pytest

from src.monthly_outlook import (
    HEAD_DAYS,
    NWP_HISTORY_START,
    _nwp_window,
    bootstrap_skill,
    head_climatology,
)
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config("config_monthly.yaml")


# ---------------------------------------------------------------------------
# NWP window clamping
# ---------------------------------------------------------------------------

def test_nwp_window_clamps_to_the_forecast_era(config):
    """config.data.start is 2010 for the climatology; the API 400s on that."""
    assert pd.Timestamp(config.data.start) < pd.Timestamp(NWP_HISTORY_START)
    assert _nwp_window(config).data.start == NWP_HISTORY_START


def test_nwp_window_leaves_a_later_start_alone(config):
    import dataclasses
    late = dataclasses.replace(
        config, data=dataclasses.replace(config.data, start="2025-01-01"))
    assert _nwp_window(late).data.start == "2025-01-01"


def test_nwp_window_preserves_the_end_date(config):
    assert _nwp_window(config).data.end == config.data.end


# ---------------------------------------------------------------------------
# Head climatology
# ---------------------------------------------------------------------------

def _cf_series(years=range(2020, 2026), value_by_year=None):
    idx = pd.date_range("2020-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    s = pd.Series(0.0, index=idx)
    for year in years:
        s[s.index.year == year] = (value_by_year or {}).get(year, 0.3)
    return s


def test_head_climatology_excludes_the_target_year():
    """The held-out year must not contribute to the level its head is judged against."""
    cf = _cf_series(value_by_year={2020: 0.1, 2021: 0.2, 2022: 0.3,
                                   2023: 0.4, 2024: 0.5, 2025: 0.6})
    got = head_climatology(cf, month=6, year=2023)
    expected = np.mean([0.1, 0.2, 0.3, 0.5, 0.6])  # 2023 omitted
    assert got == pytest.approx(expected)


def test_head_climatology_uses_only_the_first_seven_days():
    idx = pd.date_range("2020-06-01", "2021-06-30 23:00", freq="h", tz="UTC")
    cf = pd.Series(np.where(idx.day <= HEAD_DAYS, 0.8, 0.1), index=idx)
    # 2021 is the target year, so only 2020's June head counts.
    assert head_climatology(cf, month=6, year=2021) == pytest.approx(0.8)


def test_head_climatology_is_nan_when_only_the_target_year_exists():
    idx = pd.date_range("2024-06-01", "2024-06-30 23:00", freq="h", tz="UTC")
    cf = pd.Series(0.3, index=idx)
    assert np.isnan(head_climatology(cf, month=6, year=2024))


# ---------------------------------------------------------------------------
# Bootstrap interval
# ---------------------------------------------------------------------------

def _frame(clim_err, outlook_err):
    return pd.DataFrame({
        "clim_abs_err": clim_err,
        "outlook_abs_err": outlook_err,
    }, index=[f"2024-{i:02d}" for i in range(1, len(clim_err) + 1)])


def test_bootstrap_reports_positive_skill_when_the_outlook_is_better():
    rng = np.random.default_rng(0)
    clim = rng.uniform(0.03, 0.05, 40)
    frame = _frame(clim, clim * 0.5)  # unambiguously better
    out = bootstrap_skill(frame, "outlook", n_samples=2000, seed=1)

    assert out["skill"] == pytest.approx(0.5, abs=0.02)
    assert out["ci_low"] > 0
    assert out["p_better"] == 1.0


def test_bootstrap_interval_spans_zero_for_an_indistinguishable_arm():
    """The expected M1 outcome, and it must be reported as such rather than
    rounded up into a claim."""
    rng = np.random.default_rng(3)
    clim = rng.uniform(0.02, 0.06, 20)
    frame = _frame(clim, clim * rng.uniform(0.97, 1.03, 20))
    out = bootstrap_skill(frame, "outlook", n_samples=4000, seed=2)

    assert out["ci_low"] < 0 < out["ci_high"]
    assert 0.05 < out["p_better"] < 0.95


def test_bootstrap_n_months_matches_the_frame():
    frame = _frame(np.full(11, 0.04), np.full(11, 0.03))
    assert bootstrap_skill(frame, "outlook", n_samples=500)["n_months"] == 11


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(5)
    clim = rng.uniform(0.02, 0.06, 25)
    frame = _frame(clim, clim * rng.uniform(0.9, 1.1, 25))
    a = bootstrap_skill(frame, "outlook", n_samples=1000, seed=7)
    b = bootstrap_skill(frame, "outlook", n_samples=1000, seed=7)
    assert a == b
