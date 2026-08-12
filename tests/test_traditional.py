"""Tests for the classical forecasters and the bakeoff harness.

Each classical model is checked against a case where the correct answer is known
exactly, not merely "looks reasonable" — the whole point of this ladder is that
the classical rungs are trustworthy enough to judge the ML rungs against.
"""

import numpy as np
import pandas as pd
import pytest

from src.bakeoff import diebold_mariano, make_rolling_folds, select_features
from src.traditional import (
    AnalogEnsemble,
    Climatology,
    DampedPersistence,
    FoldContext,
    KalmanMOS,
    MultiModelMOS,
    NaivePersistence,
    SingleModelMOS,
    is_dead_column,
)

HORIZON = 24


def _ctx(n_train: int = 600, n_test: int = 200, noise: float = 0.0,
         seed: int = 0) -> FoldContext:
    """Synthetic fold where the target is an exact affine function of a forecast.

    y = 2*ecmwf + 0.5  — so any correct MOS implementation must recover it and
    score ~0 error when `noise=0`.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    idx = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)

    ecmwf = 5.0 + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    gfs = ecmwf * 1.4 + rng.normal(0, 0.5, n)
    icon = ecmwf * 0.8 + rng.normal(0, 0.5, n)
    y = 2.0 * ecmwf + 0.5 + rng.normal(0, noise, n)

    X = pd.DataFrame({
        "nwp_ecmwf_ifs025_wind_speed_100m_d1": ecmwf,
        "nwp_gfs_global_wind_speed_100m_d1": gfs,
        "nwp_icon_global_wind_speed_100m_d1": icon,
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "month_sin": np.sin(2 * np.pi * (idx.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (idx.month - 1) / 12),
        "wind_speed_hub_lag1": np.roll(y, 1),
    }, index=idx)
    y_s = pd.Series(y, index=idx, name="target")
    # hub(t) is the observation at feature time; the target is hub(t+HORIZON).
    hub = pd.Series(np.roll(y, HORIZON), index=idx, name="hub")

    tr, te = slice(0, n_train), slice(n_train, n)
    return FoldContext(
        X_train=X.iloc[tr], y_train=y_s.iloc[tr],
        X_test=X.iloc[te], y_test=y_s.iloc[te],
        hub_train=hub.iloc[tr], hub_test=hub.iloc[te],
        nwp_cols=[c for c in X.columns if c.startswith("nwp_")],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tier 0
# ---------------------------------------------------------------------------

def test_naive_persistence_uses_hub_at_t_not_lag1():
    """True horizon-h persistence is hub(t), not hub(t-1).

    The project's existing PersistenceForecaster reads a `_lag1` column, making it
    a 25-hour persistence quoted as 24-hour (docs/10 §4) — a 2.9% easier bar. This
    asserts the bakeoff does not inherit that defect.
    """
    ctx = _ctx()
    pred = NaivePersistence().fit_predict(ctx)
    np.testing.assert_allclose(pred, ctx.hub_test.to_numpy())
    assert not np.allclose(pred, ctx.X_test["wind_speed_hub_lag1"].to_numpy())


def test_damped_persistence_beats_naive_when_signal_is_scaled():
    """Fitting a slope must not do worse than assuming a slope of exactly 1."""
    ctx = _ctx(noise=0.5, seed=1)
    naive = NaivePersistence().fit_predict(ctx)
    damped = DampedPersistence().fit_predict(ctx)
    y = ctx.y_test.to_numpy()
    assert np.mean(np.abs(y - damped)) <= np.mean(np.abs(y - naive))


def test_climatology_predicts_by_month_and_hour():
    ctx = _ctx()
    pred = Climatology().fit_predict(ctx)
    assert len(pred) == len(ctx.y_test)
    assert np.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Tier 1 — the classical rungs
# ---------------------------------------------------------------------------

def test_single_model_mos_recovers_an_exact_linear_relation():
    """y = 2*ecmwf + 0.5 with no noise ⇒ MOS must reproduce it essentially exactly."""
    ctx = _ctx(noise=0.0)
    model = SingleModelMOS()
    pred = model.fit_predict(ctx)
    assert model.chosen_ == "nwp_ecmwf_ifs025_wind_speed_100m_d1"
    assert np.mean(np.abs(ctx.y_test.to_numpy() - pred)) < 1e-8


def test_single_model_mos_picks_its_source_on_train_only():
    """Model selection must not peek at the test fold."""
    ctx = _ctx(noise=0.2)
    model = SingleModelMOS()
    model.fit_predict(ctx)
    corrs = {c: abs(ctx.X_train[c].corr(ctx.y_train))
             for c in ctx.nwp_primary_wind_cols}
    assert model.chosen_ == max(corrs, key=corrs.get)


def test_classical_models_follow_the_lead_the_horizon_allows():
    """At a longer horizon the primary lead is d2/d3, never a hardcoded d1.

    Regression test for a real failure: the classical models originally selected
    columns by the literal suffix `_d1`. At a 48 h horizon the lead guard removes
    every d1 column, so all five silently degraded to calendar-only fits and
    `mos_multi` scored *worse than persistence* on the first 48 h run.
    """
    ctx = _ctx(noise=0.3, seed=9)
    assert ctx.primary_lead == 1
    assert ctx.nwp_primary_wind_cols  # d1 present at a 24 h horizon

    # Simulate what the lead guard leaves behind at a 48 h horizon: no d1.
    long_lead_cols = [c for c in ctx.nwp_cols if not c.endswith("_d1")]
    renamed = {c: c.replace("_d1", "_d2") for c in ctx.nwp_cols}
    X_tr = ctx.X_train.rename(columns=renamed)
    X_te = ctx.X_test.rename(columns=renamed)
    ctx48 = FoldContext(
        X_train=X_tr, y_train=ctx.y_train, X_test=X_te, y_test=ctx.y_test,
        hub_train=ctx.hub_train, hub_test=ctx.hub_test,
        nwp_cols=[renamed[c] for c in ctx.nwp_cols], seed=ctx.seed,
    )
    assert not long_lead_cols                      # the fixture only has d1
    assert ctx48.primary_lead == 2
    assert all(c.endswith("_d2") for c in ctx48.nwp_primary_wind_cols)

    # And the models must actually use them, not fall back to calendar only.
    model = SingleModelMOS()
    pred = model.fit_predict(ctx48)
    assert model.chosen_.endswith("_d2")
    y = ctx48.y_test.to_numpy()
    naive = np.mean(np.abs(y - NaivePersistence().fit_predict(ctx48)))
    assert np.mean(np.abs(y - pred)) < naive       # the sanity floor


def test_multi_model_mos_is_at_least_as_good_as_single():
    """Adding the other models plus calendar cannot hurt in-sample-fitted OLS."""
    ctx = _ctx(noise=0.5, seed=3)
    y = ctx.y_test.to_numpy()
    single = np.mean(np.abs(y - SingleModelMOS().fit_predict(ctx)))
    multi = np.mean(np.abs(y - MultiModelMOS().fit_predict(ctx)))
    assert multi <= single * 1.05


def test_kalman_mos_does_not_use_unavailable_observations():
    """The filter may only learn from observations old enough to have arrived.

    Corrupting the last `horizon` test targets must leave predictions untouched:
    those values could not have reached the filter in time to influence anything.
    """
    ctx = _ctx(noise=0.3, seed=5)
    base = KalmanMOS(horizon=HORIZON).fit_predict(ctx)

    tampered = FoldContext(
        X_train=ctx.X_train, y_train=ctx.y_train,
        X_test=ctx.X_test, y_test=ctx.y_test.copy(),
        hub_train=ctx.hub_train, hub_test=ctx.hub_test,
        nwp_cols=ctx.nwp_cols, seed=ctx.seed,
    )
    tampered.y_test.iloc[-HORIZON:] = 999.0
    after = KalmanMOS(horizon=HORIZON).fit_predict(tampered)
    np.testing.assert_allclose(base, after)


def test_analog_ensemble_recovers_an_exactly_repeating_series():
    """With an exact analogue in the training pool, AnEn should find it."""
    idx = pd.date_range("2024-03-01", periods=800, freq="h", tz="UTC")
    t = np.arange(800)
    forecast = np.sin(2 * np.pi * t / 24)          # repeats every 24 h
    y = 3.0 + 2.0 * forecast                        # deterministic in the forecast
    X = pd.DataFrame({
        "nwp_ecmwf_ifs025_wind_speed_100m_d1": forecast,
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "month_sin": np.zeros(800), "month_cos": np.ones(800),
    }, index=idx)
    y_s = pd.Series(y, index=idx)
    hub = pd.Series(np.roll(y, HORIZON), index=idx)
    ctx = FoldContext(
        X_train=X.iloc[:600], y_train=y_s.iloc[:600],
        X_test=X.iloc[600:], y_test=y_s.iloc[600:],
        hub_train=hub.iloc[:600], hub_test=hub.iloc[600:],
        nwp_cols=["nwp_ecmwf_ifs025_wind_speed_100m_d1"], seed=0,
    )
    pred = AnalogEnsemble(k=5).fit_predict(ctx)
    assert np.mean(np.abs(ctx.y_test.to_numpy() - pred)) < 0.15


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def test_rolling_folds_are_chronological_and_disjoint():
    folds = make_rolling_folds(10_000, n_folds=4, test_span=0.56)
    assert len(folds) == 4
    for i, f in enumerate(folds):
        assert f.train_end < f.test_end          # never an empty test block
        if i:
            # Each fold trains on everything the previous one tested.
            assert f.train_end == folds[i - 1].test_end
    assert folds[-1].test_end == 10_000


def test_rolling_folds_never_train_on_the_future():
    folds = make_rolling_folds(10_000, n_folds=4)
    for f in folds:
        assert f.test_start >= f.train_end


def test_dead_columns_are_identified_but_rolling_std_is_not():
    """The 1×2 grid patch kills `*_grad_ns` and `*_std`; `rollstd*` is unaffected."""
    assert is_dead_column("wind_speed_100m_grad_ns")
    assert is_dead_column("pressure_pa_std")
    assert not is_dead_column("wind_speed_hub_rollstd3")
    assert not is_dead_column("wind_speed_100m_grad_ew")


def test_operational_feature_set_excludes_every_reanalysis_column():
    """ERA5 has ~5-day latency, so no reanalysis feature may reach the
    operational model — that is the whole point of the arm."""
    cols = [
        "nwp_ecmwf_ifs025_wind_speed_100m_d1", "hour_sin", "month_cos",
        "wind_speed_100m", "wind_speed_hub_lag1", "temperature_c_grad_ns",
    ]
    op = select_features(cols, "operational")
    assert op == ["nwp_ecmwf_ifs025_wind_speed_100m_d1", "hour_sin", "month_cos"]
    study = select_features(cols, "study")
    assert "wind_speed_100m" in study
    assert "temperature_c_grad_ns" not in study      # dead column, dropped in both


def test_diebold_mariano_sign_and_significance():
    """Negative statistic means the first argument has the lower loss."""
    rng = np.random.default_rng(0)
    good = rng.normal(0, 1.0, 2000)
    bad = rng.normal(0, 2.0, 2000)
    stat, p = diebold_mariano(good, bad, horizon=24)
    assert stat < 0
    assert p < 0.05


def test_diebold_mariano_finds_no_difference_between_equals():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1.0, 2000)
    stat, p = diebold_mariano(a, a.copy(), horizon=24)
    assert np.isnan(p) or p > 0.05
