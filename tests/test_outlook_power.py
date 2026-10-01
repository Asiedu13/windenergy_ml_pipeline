"""Tests for the re-powered outlook evaluation.

The whole claim of this module is that it scores a 7-day forecast on 7-day
blocks that **tile** the record with no gaps and no overlaps, using only past
data. Two things can silently break that and produce a plausible-looking number:

* an off-by-one in the lead-to-day mapping, which shows up as an overlap or a
  hole and nowhere else;
* a training window that reaches past the block start.

Both are asserted in the module and tested here on inputs that deliberately
violate them.
"""

import numpy as np
import pandas as pd
import pytest

from src.outlook_power import (
    ISSUE_DAYS,
    LADDER,
    MIN_BLOCK_COVERAGE,
    REFERENCE,
    archive_climatology,
    assert_tiles,
    block_scores,
    block_summary,
    fit_candidate,
    issue_origins,
    lead_summary,
    month_start_origins,
)

TZ = "UTC"


def _origins(n=4, start="2024-05-01"):
    return issue_origins(
        pd.Timestamp(start, tz=TZ),
        pd.Timestamp(start, tz=TZ) + pd.Timedelta(days=7 * n),
        ISSUE_DAYS,
    )


def _tiled_frame(origins, ladder=("mos_multi",)):
    """A frame that correctly tiles the origins, one row per hour."""
    parts = []
    rng = np.random.default_rng(0)
    for block, origin in enumerate(origins):
        for lead in range(1, ISSUE_DAYS + 1):
            idx = pd.date_range(origin + pd.Timedelta(days=lead - 1),
                                periods=24, freq="h", tz=TZ)
            f = pd.DataFrame(index=idx)
            f["actual"] = rng.uniform(0.1, 0.5, 24)
            f[REFERENCE] = 0.26
            for name in ladder:
                f[name] = f["actual"] + rng.normal(0, 0.02, 24)
            f["lead_day"] = lead
            f["origin"] = origin
            f["block"] = block
            parts.append(f)
    return pd.concat(parts).sort_index()


def _cf_archive(years=6, start="2019-01-01"):
    idx = pd.date_range(start, periods=years * 365 * 24, freq="h", tz=TZ)
    seasonal = 0.26 + 0.10 * np.sin(2 * np.pi * (idx.month - 1) / 12)
    diurnal = 0.05 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.Series(seasonal + diurnal, index=idx, name="cf")


# ---------------------------------------------------------------------------
# Origins
# ---------------------------------------------------------------------------

def test_origins_are_evenly_spaced():
    origins = _origins(5)
    gaps = {(b - a).days for a, b in zip(origins, origins[1:])}
    assert gaps == {ISSUE_DAYS}


def test_partial_trailing_block_is_dropped():
    """A 3-day block and a 7-day block are not exchangeable samples."""
    start = pd.Timestamp("2024-05-01", tz=TZ)
    origins = issue_origins(start, start + pd.Timedelta(days=17), ISSUE_DAYS)
    assert len(origins) == 2  # 14 days fit; the trailing 3 do not
    assert origins[-1] + pd.Timedelta(days=7) <= start + pd.Timedelta(days=17)


def test_no_origins_when_window_is_shorter_than_one_block():
    start = pd.Timestamp("2024-05-01", tz=TZ)
    assert issue_origins(start, start + pd.Timedelta(days=5), ISSUE_DAYS) == []


# ---------------------------------------------------------------------------
# Tiling gate
# ---------------------------------------------------------------------------

def test_complete_tiling_reports_full_coverage():
    origins = _origins(4)
    coverage = assert_tiles(_tiled_frame(origins), origins)
    assert len(coverage) == 4
    assert np.allclose(coverage.to_numpy(), 1.0)


def test_overlap_is_caught():
    """Duplicating one lead's day is what an off-by-one produces. Overlap is
    always a bug, so it raises rather than being reported."""
    origins = _origins(3)
    frame = _tiled_frame(origins)
    doubled = pd.concat([frame, frame.iloc[:24]]).sort_index()
    with pytest.raises(AssertionError, match="more than once"):
        assert_tiles(doubled, origins)


def test_gaps_are_reported_as_coverage_not_raised():
    """Open-Meteo has small holes; asserting zero gaps would assert something
    about its uptime rather than about this code."""
    origins = _origins(3)
    frame = _tiled_frame(origins)
    holed = frame.iloc[12:]  # remove half of block 0's first day

    coverage = assert_tiles(holed, origins)
    assert coverage[0] < 1.0
    assert np.allclose(coverage.loc[1:].to_numpy(), 1.0)


def test_block_below_threshold_is_identifiable():
    origins = _origins(3)
    frame = _tiled_frame(origins)
    # Drop 20% of block 1 — well under MIN_BLOCK_COVERAGE.
    drop = frame[(frame.block == 1)].index[:34]
    coverage = assert_tiles(frame.drop(drop), origins)
    assert coverage[1] < MIN_BLOCK_COVERAGE
    assert coverage[0] >= MIN_BLOCK_COVERAGE


def test_empty_origins_raise():
    with pytest.raises(ValueError, match="no issue origins"):
        assert_tiles(pd.DataFrame(), [])


def test_tiling_covers_exactly_the_expected_hours():
    origins = _origins(4)
    frame = _tiled_frame(origins)
    assert len(frame) == 4 * ISSUE_DAYS * 24
    assert frame.index.is_monotonic_increasing
    assert not frame.index.duplicated().any()


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def _training(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=n, freq="h", tz=TZ)
    ecmwf = 6 + 2 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.3, n)
    gfs = ecmwf * 1.2 + rng.normal(0, 0.4, n)
    X = pd.DataFrame({
        "nwp_ecmwf_ifs025_wind_speed_100m_d1": ecmwf,
        "nwp_gfs_global_wind_speed_100m_d1": gfs,
        "nwp_ensemble_mean": (ecmwf + gfs) / 2,
    }, index=idx)
    y = pd.Series(2.0 * ecmwf + 0.5, index=idx, name="wind_speed_hub")
    return X, y


def test_mos_multi_recovers_an_exact_linear_target():
    X, y = _training()
    predict = fit_candidate("mos_multi", X, y, lead=1, seed=0)
    assert np.allclose(predict(X), y.to_numpy(), atol=1e-6)


def test_mos_single_picks_the_correlated_column():
    X, y = _training()
    predict = fit_candidate("mos_single", X, y, lead=1, seed=0)
    # y is an exact affine function of ecmwf, so a single-column fit nails it.
    assert np.corrcoef(predict(X), y.to_numpy())[0, 1] > 0.99


def test_ensemble_mean_does_not_fit_anything():
    X, y = _training()
    predict = fit_candidate("ensemble_mean", X, y, lead=1, seed=0)
    assert np.allclose(predict(X), X["nwp_ensemble_mean"].to_numpy())


def test_ensemble_mean_absent_returns_none():
    X, y = _training()
    predict = fit_candidate("ensemble_mean", X.drop(columns=["nwp_ensemble_mean"]),
                            y, lead=1, seed=0)
    assert predict is None


def test_mos_single_with_no_matching_lead_returns_none():
    X, y = _training()
    assert fit_candidate("mos_single", X, y, lead=5, seed=0) is None


def test_unknown_candidate_raises():
    X, y = _training()
    with pytest.raises(ValueError, match="unknown candidate"):
        fit_candidate("magic", X, y, lead=1, seed=0)


def test_candidates_return_predictors_not_predictions():
    """One fit is reused across every issue in its refit month; caching a
    prediction instead would serve one block's values to the next."""
    X, y = _training()
    predict = fit_candidate("mos_multi", X, y, lead=1, seed=0)
    assert callable(predict)
    first, second = predict(X.iloc[:24]), predict(X.iloc[24:48])
    assert len(first) == len(second) == 24
    assert not np.allclose(first, second)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_block_scores_average_to_one_row_per_block():
    origins = _origins(4)
    blocks = block_scores(_tiled_frame(origins), ladder=("mos_multi",))
    assert len(blocks) == 4
    assert set(blocks.columns) == {"actual", REFERENCE, "mos_multi"}


def test_perfect_arm_scores_full_skill():
    origins = _origins(3)
    frame = _tiled_frame(origins, ladder=("mos_multi",))
    frame["mos_multi"] = frame["actual"]
    summary = block_summary(block_scores(frame, ("mos_multi",)), ("mos_multi",))
    assert summary.loc["mos_multi", "mae_cf"] == pytest.approx(0.0, abs=1e-12)
    assert summary.loc["mos_multi", "skill_vs_clim"] == pytest.approx(1.0)


def test_reference_arm_has_zero_skill_against_itself():
    origins = _origins(3)
    summary = block_summary(block_scores(_tiled_frame(origins), ("mos_multi",)),
                            ("mos_multi",))
    assert summary.loc[REFERENCE, "skill_vs_clim"] == pytest.approx(0.0)
    assert "dm_p" not in summary.columns or np.isnan(summary.loc[REFERENCE, "dm_p"])


def test_lead_summary_has_one_row_per_lead_and_arm():
    origins = _origins(3)
    leads = lead_summary(_tiled_frame(origins), ladder=("mos_multi",))
    assert len(leads) == ISSUE_DAYS * 2  # climatology + one candidate
    assert leads.index.get_level_values("lead_day").nunique() == ISSUE_DAYS


def test_ladder_is_the_documented_set():
    assert LADDER == ("ensemble_mean", "mos_single", "mos_multi", "xgboost")


# ---------------------------------------------------------------------------
# Archive climatology — the admissible baseline
# ---------------------------------------------------------------------------

def test_archive_climatology_uses_only_the_past():
    """An outlook issued in 2022 may not consult 2023."""
    cf = _cf_archive()
    before = pd.Timestamp("2022-01-01", tz=TZ)
    idx = pd.date_range("2022-06-01", periods=48, freq="h", tz=TZ)

    got = archive_climatology(cf, before, idx)
    # Recompute by hand from the permitted slice only.
    past = cf[cf.index < before]
    expected = [past[(past.index.month == m) & (past.index.hour == h)].mean()
                for m, h in zip(idx.month, idx.hour)]
    assert np.allclose(got, expected)


def test_archive_climatology_tracks_the_seasonal_cycle():
    """The fixture's sin((month-1)/12) peaks in April and troughs in October —
    Jan and Jul both sit on a zero crossing, so they are not the pair to test."""
    cf = _cf_archive()
    before = pd.Timestamp("2023-01-01", tz=TZ)
    peak = archive_climatology(
        cf, before, pd.date_range("2023-04-01", periods=24, freq="h", tz=TZ))
    trough = archive_climatology(
        cf, before, pd.date_range("2023-10-01", periods=24, freq="h", tz=TZ))
    assert peak.mean() > trough.mean()
    assert peak.mean() == pytest.approx(0.36, abs=1e-9)
    assert trough.mean() == pytest.approx(0.16, abs=1e-9)


def test_archive_climatology_is_richer_than_a_two_year_window():
    """The whole reason REFERENCE is the archive: a short window is a weak
    baseline, and scoring against it inflates every candidate's skill."""
    cf = _cf_archive()
    idx = pd.date_range("2024-06-01", periods=168, freq="h", tz=TZ)
    long_n = len(cf[cf.index < pd.Timestamp("2024-01-01", tz=TZ)])
    short_n = len(cf[(cf.index >= pd.Timestamp("2022-01-01", tz=TZ))
                     & (cf.index < pd.Timestamp("2024-01-01", tz=TZ))])
    assert long_n > short_n * 2


def test_archive_climatology_raises_without_history():
    cf = _cf_archive()
    with pytest.raises(ValueError, match="no archive data before"):
        archive_climatology(cf, pd.Timestamp("2018-01-01", tz=TZ),
                            pd.date_range("2018-01-01", periods=3, freq="h", tz=TZ))


# ---------------------------------------------------------------------------
# Month-start origins — the product grid
# ---------------------------------------------------------------------------

def test_month_start_origins_land_on_the_first():
    origins = month_start_origins(pd.Timestamp("2024-05-06", tz=TZ),
                                  pd.Timestamp("2024-12-31", tz=TZ))
    assert all(o.day == 1 for o in origins)
    assert origins[0] == pd.Timestamp("2024-06-01", tz=TZ)


def test_month_start_origins_differ_from_the_tiled_grid():
    """The distinction the composite depends on: on a weekly grid the forecast
    covering day 3 may have been issued five days earlier."""
    start = pd.Timestamp("2024-05-06", tz=TZ)
    end = pd.Timestamp("2024-12-31", tz=TZ)
    assert set(month_start_origins(start, end)) != set(
        issue_origins(start, end, ISSUE_DAYS))


def test_month_start_origins_need_a_whole_head():
    origins = month_start_origins(pd.Timestamp("2024-05-06", tz=TZ),
                                  pd.Timestamp("2024-07-04", tz=TZ))
    # July needs 1-7 July; only 1-4 are available, so July is excluded.
    assert origins == [pd.Timestamp("2024-06-01", tz=TZ)]
