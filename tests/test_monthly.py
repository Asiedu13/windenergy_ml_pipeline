"""Tests for the monthly resource climatology.

Two things here are worth more than the rest combined, because getting either
wrong inverts a published number rather than merely degrading it:

* **The exceedance convention.** P90 is the value *exceeded* 90% of the time —
  the 10th percentile, a low and conservative figure. Reading it as the 90th
  percentile turns a conservative estimate into an optimistic one.
* **The gap handling.** The ERA5 archive has a 41-month hole, so a naive
  ``shift(1)`` would pair 2016-07 with 2020-01 and report it as a lag-1
  correlation between adjacent months.
"""

import numpy as np
import pandas as pd
import pytest

from src.monthly import (
    EXCEEDANCE_QUANTILE,
    MIN_COMPLETENESS,
    anomaly_autocorrelation,
    annual_energy_mwh,
    capacity_factor_from_weather,
    diurnal_profile,
    exceedance_table,
    exceedance_validation,
    loyo_climatology,
    monthly_frame,
    score_baselines,
    variance_decomposition,
)
from src.utils import load_config


def _hourly_cf(start: str = "2010-01-01", months: int = 24,
               amplitude: float = 0.1, seed: int = 0) -> pd.Series:
    """Hourly capacity factor with a clean seasonal cycle and no trend."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(start) + pd.DateOffset(months=months)
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    seasonal = 0.26 + amplitude * np.sin(2 * np.pi * (idx.month - 1) / 12)
    return pd.Series(seasonal + rng.normal(0, 0.02, len(idx)), index=idx, name="cf")


def _monthly(months: int = 48, seed: int = 0) -> pd.DataFrame:
    return monthly_frame(_hourly_cf(months=months, seed=seed))


# ---------------------------------------------------------------------------
# Completeness guard
# ---------------------------------------------------------------------------

def test_complete_months_are_kept():
    monthly = _monthly(months=24)
    assert len(monthly) == 24
    assert (monthly["hours"] == monthly.index.days_in_month * 24).all()


def test_truncated_month_is_dropped():
    """A month missing a third of its hours must not enter the climatology."""
    cf = _hourly_cf(months=3)
    march = cf.index.month == 3
    keep = ~(march & (cf.index.day > 20))  # drop the last ~third of March
    monthly = monthly_frame(cf[keep])

    assert 3 not in monthly["month"].to_numpy(), "truncated March should be dropped"
    assert len(monthly) == 2


def test_month_just_above_threshold_is_kept():
    """The guard is a threshold, not a demand for perfection."""
    cf = _hourly_cf(months=2)
    jan = cf.index.month == 1
    # Remove 2% of January's hours — comfortably above the 95% floor.
    drop = np.flatnonzero(jan)[:15]
    monthly = monthly_frame(cf.drop(cf.index[drop]))

    assert 1 in monthly["month"].to_numpy()
    assert monthly.loc[monthly["month"] == 1, "hours"].iloc[0] >= MIN_COMPLETENESS * 31 * 24


def test_empty_input_returns_empty_frame():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    assert monthly_frame(empty).empty


# ---------------------------------------------------------------------------
# Exceedance convention — P90 must be LOW
# ---------------------------------------------------------------------------

def test_exceedance_quantile_mapping_is_inverted():
    """PXX is exceeded XX% of the time, so it is the (1-XX/100) quantile."""
    assert EXCEEDANCE_QUANTILE[50] == 0.50
    assert EXCEEDANCE_QUANTILE[75] == 0.25
    assert EXCEEDANCE_QUANTILE[90] == 0.10


def test_p90_is_below_p50_is_below_mean_ish():
    monthly = _monthly(months=120)
    table = exceedance_table(monthly)
    assert (table["p90_cf"] < table["p75_cf"]).all()
    assert (table["p75_cf"] < table["p50_cf"]).all()


def test_empirical_p90_matches_tenth_percentile_exactly():
    monthly = _monthly(months=120)
    table = exceedance_table(monthly, method="empirical")
    for month in table.index:
        observed = monthly.loc[monthly["month"] == month, "cf"]
        assert table.loc[month, "p90_cf"] == pytest.approx(observed.quantile(0.10))


@pytest.mark.parametrize("method", ["empirical", "normal"])
def test_exceedance_validation_hits_are_near_nominal_for_a_stationary_series(method):
    """With a stationary series and enough years, LOYO hit rates track nominal."""
    monthly = _monthly(months=12 * 30, seed=3)
    validation = exceedance_validation(monthly, method=method)
    for level in ("P50", "P75", "P90"):
        nominal = validation.loc[level, "nominal_exceedance"]
        observed = validation.loc[level, "observed_exceedance"]
        assert abs(observed - nominal) < 0.10, f"{level}: {observed} vs {nominal}"


def test_normal_fit_beats_empirical_in_the_tail_at_small_n():
    """The reason exceedance_table defaults to the normal fit.

    At ~12 years per calendar month the empirical 10th percentile sits between
    the first and second order statistics, so holding a year out shifts it by a
    whole rank. The fitted tail uses all 12 points and should track nominal more
    closely.
    """
    monthly = _monthly(months=12 * 12, seed=11)  # 12 years -> the real regime
    errors = {}
    for method in ("empirical", "normal"):
        v = exceedance_validation(monthly, method=method)
        errors[method] = abs(v.loc["P90", "observed_exceedance"]
                             - v.loc["P90", "nominal_exceedance"])
    assert errors["normal"] <= errors["empirical"]


def test_normal_p90_is_below_p50():
    """The fitted estimator must respect the same ordering as the empirical one."""
    table = exceedance_table(_monthly(months=120), method="normal")
    assert (table["p90_cf"] < table["p50_cf"]).all()


def test_normal_p50_equals_the_mean():
    """For a Gaussian fit the median is the mean — a cheap check that loc/scale
    are not transposed."""
    monthly = _monthly(months=120)
    table = exceedance_table(monthly, method="normal")
    for month in table.index:
        observed = monthly.loc[monthly["month"] == month, "cf"]
        assert table.loc[month, "p50_cf"] == pytest.approx(observed.mean())


def test_unknown_exceedance_method_raises():
    with pytest.raises(ValueError, match="unknown exceedance method"):
        exceedance_table(_monthly(months=24), method="weibull")


# ---------------------------------------------------------------------------
# Climatology and baselines
# ---------------------------------------------------------------------------

def test_loyo_climatology_excludes_the_target_year():
    monthly = _monthly(months=60)
    clim = loyo_climatology(monthly)

    row = monthly.iloc[7]
    others = monthly[(monthly["month"] == row["month"])
                     & (monthly["year"] != row["year"])]["cf"]
    assert clim.iloc[7] == pytest.approx(others.mean())


def test_loyo_climatology_is_never_better_than_in_sample():
    """The honest baseline must score worse — that is why it is the honest one."""
    monthly = _monthly(months=60)
    scores = score_baselines(monthly)
    assert scores.clim_loyo > scores.clim_in_sample


def test_single_year_gives_undefined_loyo_climatology():
    """With one year there are no 'other years' — must be NaN, not silently 0."""
    monthly = _monthly(months=12)
    assert loyo_climatology(monthly).isna().all()


def test_variance_decomposition_shares_sum_to_one():
    stats = variance_decomposition(_monthly(months=120))
    assert stats["seasonal_share"] + stats["anomaly_share"] == pytest.approx(1.0, abs=0.02)


def test_strong_seasonal_cycle_dominates_variance():
    """A large seasonal amplitude must show up as a large seasonal share."""
    strong = monthly_frame(_hourly_cf(months=120, amplitude=0.15, seed=1))
    weak = monthly_frame(_hourly_cf(months=120, amplitude=0.01, seed=1))
    assert (variance_decomposition(strong)["seasonal_share"]
            > variance_decomposition(weak)["seasonal_share"])


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------

def test_autocorrelation_ignores_pairs_across_a_gap():
    """The 41-month archive hole must not be treated as a one-month step."""
    monthly = _monthly(months=48)
    gapped = pd.concat([monthly.iloc[:12], monthly.iloc[36:]])

    full_n = anomaly_autocorrelation(monthly, lags=(1,))[1][1]
    gapped_n = anomaly_autocorrelation(gapped, lags=(1,))[1][1]

    assert full_n == len(monthly) - 1          # every pair is consecutive
    # 24 months are present but only 22 adjacent pairs: the 2010-12 -> 2013-01
    # join spans the gap and must not be counted. A naive shift(1) would give 23.
    assert gapped_n == len(gapped) - 2
    assert gapped_n < len(gapped) - 1


def test_autocorrelation_reports_nan_when_too_few_pairs():
    monthly = _monthly(months=6)
    r, n = anomaly_autocorrelation(monthly, lags=(1,))[1]
    assert np.isnan(r)
    assert n < 10


def test_persistence_is_scored_only_on_consecutive_months():
    monthly = _monthly(months=48)
    gapped = pd.concat([monthly.iloc[:12], monthly.iloc[36:]])
    assert score_baselines(gapped).persistence_n == len(gapped) - 2


# ---------------------------------------------------------------------------
# Derived products
# ---------------------------------------------------------------------------

def test_diurnal_profile_shape():
    profile = diurnal_profile(_hourly_cf(months=24))
    assert profile.shape == (12, 24)
    assert profile.index[0] == "Jan"


def test_annual_energy_flags_incomplete_years():
    monthly = _monthly(months=18)  # 2010 complete, 2011 has 6 months
    annual = annual_energy_mwh(monthly, rated_w=3.05e6)
    assert bool(annual.loc[2010, "complete"]) is True
    assert bool(annual.loc[2011, "complete"]) is False


# ---------------------------------------------------------------------------
# Capacity factor from weather
# ---------------------------------------------------------------------------

def _weather(n: int = 720, wind: float = 9.0) -> pd.DataFrame:
    idx = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "wind_speed_100m": np.full(n, wind),
        "temperature_c": np.full(n, 28.0),   # Ghana, not a standard atmosphere
        "pressure_pa": np.full(n, 101_000.0),
    }, index=idx)


def test_capacity_factor_runs_end_to_end():
    """Guards the whole path — an earlier refactor left `power_w` undefined here
    and every existing test still passed, because none of them called it."""
    cfg = load_config("config_monthly.yaml")
    cf = capacity_factor_from_weather(_weather(), cfg)

    assert len(cf) == 720
    assert cf.name == "cf"
    assert (cf >= 0).all() and (cf <= 1).all()
    assert cf.std() == pytest.approx(0.0, abs=1e-9)  # constant wind -> constant CF


def test_capacity_factor_is_monotonic_in_wind_below_rated():
    cfg = load_config("config_monthly.yaml")
    low = capacity_factor_from_weather(_weather(n=24, wind=6.0), cfg).mean()
    high = capacity_factor_from_weather(_weather(n=24, wind=10.0), cfg).mean()
    assert 0 < low < high


def test_density_correction_lowers_output_in_the_heat():
    """Ghana's heat lowers air density, which must lower power — the correction
    is not cosmetic. Hot air is compared against a cooler reference."""
    cfg = load_config("config_monthly.yaml")
    hot = _weather(n=24, wind=8.0)
    cool = hot.copy()
    cool["temperature_c"] = 5.0
    assert (capacity_factor_from_weather(hot, cfg).mean()
            < capacity_factor_from_weather(cool, cfg).mean())


def test_capacity_factor_skips_missing_wind_hours():
    cfg = load_config("config_monthly.yaml")
    weather = _weather(n=48)
    weather.iloc[:5, weather.columns.get_loc("wind_speed_100m")] = np.nan
    assert len(capacity_factor_from_weather(weather, cfg)) == 43


def test_annual_energy_matches_hand_calculation():
    monthly = _monthly(months=12)
    annual = annual_energy_mwh(monthly, rated_w=3.05e6)
    expected = (monthly["cf"] * monthly["hours"]).sum() * 3.05e6 / 1e6
    assert annual.loc[2010, "energy_mwh"] == pytest.approx(expected)
