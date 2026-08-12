import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_direction_encodings,
    add_lag_features,
    add_nwp_features,
    add_rolling_features,
    build_feature_frame,
    extrapolate_hub_wind,
)
from src.utils import (
    Config,
    DataConfig,
    ForecastConfig,
    ModelConfig,
    PathsConfig,
    SiteConfig,
    TurbineConfig,
)


def _make_weather(hours: int = 24 * 7) -> pd.DataFrame:
    idx = pd.date_range("2023-06-01", periods=hours, freq="h", tz="UTC")
    t = np.arange(hours)
    return pd.DataFrame(
        {
            "wind_speed_10m": 4 + 1.5 * np.sin(2 * np.pi * t / 24),
            "wind_speed_100m": 7 + 2.5 * np.sin(2 * np.pi * t / 24),
            "wind_dir_10m": (180 + 30 * np.sin(t / 12)) % 360,
            "wind_dir_100m": (200 + 30 * np.sin(t / 12)) % 360,
            "temperature_c": 28 + 3 * np.sin(2 * np.pi * t / 24),
            "pressure_pa": 101_300 + 100 * np.sin(t / 48),
            "roughness": 0.15,
        },
        index=idx,
    )


def _make_config() -> Config:
    return Config(
        site=SiteConfig(name="t", latitude=5.79, longitude=0.9, hub_height_m=100),
        data=DataConfig(source="nasa_power", start="2023-06-01", end="2023-06-08",
                        cache_dir="/tmp"),
        turbine=TurbineConfig(name="E-101/3050"),
        forecast=ForecastConfig(horizon_hours=6),
        model=ModelConfig(lags=[1, 2, 3], rolling_windows=[3, 6]),
        paths=PathsConfig(),
        raw={},
    )


def test_extrapolate_hub_wind_increases_from_100m_to_taller_hub():
    df = _make_weather()
    hub = extrapolate_hub_wind(df, hub_height_m=140.0, method="logarithmic")
    assert (hub > df["wind_speed_100m"]).all()


def test_extrapolate_hub_wind_equals_reference_at_reference_height():
    df = _make_weather()
    hub = extrapolate_hub_wind(df, hub_height_m=100.0, method="logarithmic")
    np.testing.assert_allclose(hub.to_numpy(), df["wind_speed_100m"].to_numpy(), rtol=1e-6)


def test_add_direction_encodings():
    df = _make_weather().iloc[:5]
    out = add_direction_encodings(df)
    for col in ("wind_dir_10m", "wind_dir_100m"):
        assert f"{col}_sin" in out.columns
        assert f"{col}_cos" in out.columns
        # unit-circle identity
        s = out[f"{col}_sin"].to_numpy()
        c = out[f"{col}_cos"].to_numpy()
        np.testing.assert_allclose(s * s + c * c, np.ones_like(s), atol=1e-9)


def test_add_calendar_features_cycles():
    df = _make_weather()
    out = add_calendar_features(df)
    assert out["hour"].min() == 0 and out["hour"].max() == 23
    np.testing.assert_allclose(
        out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0, atol=1e-9
    )


def test_lag_shift_matches_pandas_shift():
    df = _make_weather()
    out = add_lag_features(df, "wind_speed_100m", [1, 3])
    pd.testing.assert_series_equal(
        out["wind_speed_100m_lag1"].dropna(),
        df["wind_speed_100m"].shift(1).dropna(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out["wind_speed_100m_lag3"].dropna(),
        df["wind_speed_100m"].shift(3).dropna(),
        check_names=False,
    )


def test_rolling_features_no_leakage_of_current_step():
    df = _make_weather()
    out = add_rolling_features(df, "wind_speed_100m", [3])
    # At time t, rolling mean uses values [t-3, t-2, t-1] — not t itself.
    expected = df["wind_speed_100m"].shift(1).rolling(3, min_periods=3).mean()
    pd.testing.assert_series_equal(
        out["wind_speed_100m_rollmean3"].dropna(),
        expected.dropna(),
        check_names=False,
    )


def test_build_feature_frame_nowcast_shapes():
    cfg = _make_config()
    weather = _make_weather()
    X, y, cols = build_feature_frame(weather, cfg, mode="nowcast")
    assert list(X.columns) == cols
    assert len(X) == len(y)
    # Warmup rows dropped for lags and rolling windows (max lag/rolling = 6)
    assert len(X) <= len(weather) - 6
    # No NaNs in X or y
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_build_feature_frame_forecast_horizon_shifts_target():
    cfg = _make_config()  # horizon=6
    weather = _make_weather(hours=48)
    _, y_fc, _ = build_feature_frame(weather, cfg, mode="forecast")
    _, y_nc, _ = build_feature_frame(weather, cfg, mode="nowcast")
    # Forecast dataset should be strictly shorter (loses horizon at the tail)
    assert len(y_fc) < len(y_nc)


# ---------------------------------------------------------------------------
# NWP forecast alignment
#
# These guard the same class of defect that invalidated every deep-model score
# in v2/v3 (docs/10): two frames indexed on different clocks, joined without
# accounting for the offset. The failure is silent and the numbers stay
# plausible, so the assertions below are deliberately exact rather than
# tolerance-based.
# ---------------------------------------------------------------------------


def _make_nwp(index: pd.DatetimeIndex, values, col: str = "nwp_test_ws100_d1"):
    return pd.DataFrame({col: np.asarray(values, dtype=float)}, index=index)


def test_add_nwp_features_takes_the_row_valid_at_t_plus_offset():
    idx = pd.date_range("2023-06-01", periods=12, freq="h", tz="UTC")
    # Encode the valid time in the value so misalignment is unambiguous.
    nwp = _make_nwp(idx, range(12))
    df = pd.DataFrame({"x": 0.0}, index=idx)

    out = add_nwp_features(df, nwp, offset_hours=3)

    # Row t must carry the forecast valid at t+3, i.e. value (t_index + 3).
    assert out["nwp_test_ws100_d1"].iloc[0] == 3.0
    assert out["nwp_test_ws100_d1"].iloc[5] == 8.0
    # The tail has no t+3 row to draw from.
    assert out["nwp_test_ws100_d1"].iloc[-3:].isna().all()
    # Index is preserved (feature time, not valid time).
    assert out.index.equals(df.index)


def test_add_nwp_features_zero_offset_is_identity_join():
    idx = pd.date_range("2023-06-01", periods=6, freq="h", tz="UTC")
    nwp = _make_nwp(idx, [10, 11, 12, 13, 14, 15])
    out = add_nwp_features(pd.DataFrame({"x": 0.0}, index=idx), nwp, offset_hours=0)
    assert out["nwp_test_ws100_d1"].tolist() == [10, 11, 12, 13, 14, 15]


def test_add_nwp_features_survives_gaps_in_the_forecast_index():
    idx = pd.date_range("2023-06-01", periods=10, freq="h", tz="UTC")
    nwp = _make_nwp(idx, range(10)).drop(index=idx[4])  # hole at t=4
    out = add_nwp_features(pd.DataFrame({"x": 0.0}, index=idx), nwp, offset_hours=2)
    # Row 2 wanted valid time index 4, which is missing -> NaN, not a shifted value.
    assert np.isnan(out["nwp_test_ws100_d1"].iloc[2])
    assert out["nwp_test_ws100_d1"].iloc[3] == 5.0


def test_perfect_forecast_column_reproduces_the_target_exactly():
    """The sharp test: a *perfect* forecast must equal y after alignment.

    The signal period (37 h) is deliberately not commensurate with the 6 h
    horizon, so a frame that is shifted by any non-zero amount cannot pass by
    coincidence — the same trick docs/10 uses for the deep-model backtest.
    """
    cfg = _make_config()  # horizon=6
    hours = 24 * 10
    idx = pd.date_range("2023-06-01", periods=hours, freq="h", tz="UTC")
    t = np.arange(hours)
    weather = _make_weather(hours)
    weather["wind_speed_100m"] = 7 + 2.5 * np.sin(2 * np.pi * t / 37)

    # A forecast that is exactly right, indexed by valid time. It must be built
    # on the *same* reference height build_feature_frame uses, otherwise this
    # test measures the height conversion rather than the alignment it targets.
    from src.data_ingest import source_wind_height_m
    hub = extrapolate_hub_wind(
        weather, cfg.site.hub_height_m,
        reference_height_m=source_wind_height_m(cfg.data.source),
    )
    nwp = _make_nwp(idx, hub.to_numpy(), col="nwp_perfect_ws100_d1")

    X, y, cols = build_feature_frame(weather, cfg, mode="forecast", nwp_df=nwp)

    assert "nwp_perfect_ws100_d1" in cols
    np.testing.assert_allclose(X["nwp_perfect_ws100_d1"].to_numpy(),
                               y.to_numpy(), rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Source-dependent reference height
#
# `wind_speed_100m` is only literally at 100 m for ERA5. NASA POWER's highest
# level is 50 m and lands in the same column, so the hub extrapolation must use
# the source's real height. Before this was fixed the log law no-opped and hub
# wind was understated by ~10.7% on every NASA POWER run, v1's included.
# ---------------------------------------------------------------------------


def test_nasa_power_wind_is_extrapolated_from_50m_not_treated_as_100m():
    from windpowerlib.wind_speed import logarithmic_profile

    cfg = _make_config()          # DataConfig(source="nasa_power")
    assert cfg.data.source == "nasa_power"
    weather = _make_weather(48)
    weather["roughness"] = np.nan  # NASA POWER has no roughness -> default 0.15

    X, y, _ = build_feature_frame(weather, cfg, mode="nowcast")

    raw50 = weather["wind_speed_100m"].to_numpy(dtype=float)
    expected = logarithmic_profile(
        wind_speed=raw50, wind_speed_height=50.0, hub_height=100.0,
        roughness_length=np.full_like(raw50, 0.15),
    )
    hub = y.to_numpy()  # nowcast target is hub wind itself
    idx = weather.index.get_indexer(y.index)
    np.testing.assert_allclose(hub, expected[idx], rtol=0, atol=1e-9)
    # And it must actually be larger than the raw 50 m reading.
    assert (hub > raw50[idx]).all()
    np.testing.assert_allclose(hub / raw50[idx], 1.1193, rtol=1e-3)


def test_era5_reference_height_is_unchanged_by_the_fix():
    """ERA5 is genuinely at 100 m, so 100 -> 100 must remain an exact no-op."""
    cfg = _make_config()
    cfg.data.source = "era5"
    weather = _make_weather(48)
    _, y, _ = build_feature_frame(weather, cfg, mode="nowcast")
    raw100 = weather["wind_speed_100m"]
    np.testing.assert_allclose(y.to_numpy(), raw100.loc[y.index].to_numpy(),
                               rtol=0, atol=1e-12)


def test_unknown_source_falls_back_to_100m():
    from src.data_ingest import source_wind_height_m
    assert source_wind_height_m("era5") == 100.0
    assert source_wind_height_m("nasa_power") == 50.0
    assert source_wind_height_m("ERA5") == 100.0          # case-insensitive
    assert source_wind_height_m("something_new") == 100.0  # safe default


def test_nwp_columns_absent_when_not_supplied():
    cfg = _make_config()
    weather = _make_weather()
    _, _, cols_off = build_feature_frame(weather, cfg, mode="forecast")
    _, _, cols_empty = build_feature_frame(weather, cfg, mode="forecast",
                                           nwp_df=pd.DataFrame())
    assert not any(c.startswith("nwp_") for c in cols_off)
    assert cols_off == cols_empty


# ---------------------------------------------------------------------------
# Lead-day / horizon leakage guard
# ---------------------------------------------------------------------------
#
# `nwp_{model}_{var}_d{N}[tau]` is the forecast for tau, issued 24*N hours
# BEFORE tau. Row t receives the forecast valid at t + horizon, so that value
# was issued at (t + horizon) - 24N. It is only legitimately knowable at time t
# when that issue time is <= t, i.e. when:
#
#       24 * N >= horizon
#
# At a 24 h horizon every lead qualifies, which is why this never mattered
# before. At 168 h, a d1 column attached to row t was issued at t + 144 — six
# days into the future. Nothing checked this until these tests existed.

def _nwp_frame(index: pd.DatetimeIndex, leads=(1, 2, 3)) -> pd.DataFrame:
    return pd.DataFrame(
        {f"nwp_ecmwf_ifs025_wind_speed_100m_d{d}": np.linspace(5, 9, len(index))
         for d in leads},
        index=index,
    )


def test_all_leads_survive_at_the_24h_horizon():
    """h=24 needs 24N >= 24, so d1/d2/d3 all qualify — the guard is a no-op."""
    idx = pd.date_range("2024-03-01", periods=300, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(300.0)}, index=idx)
    out = add_nwp_features(df, _nwp_frame(idx), offset_hours=24)
    kept = sorted(c for c in out.columns if c.startswith("nwp_"))
    assert kept == [
        "nwp_ecmwf_ifs025_wind_speed_100m_d1",
        "nwp_ecmwf_ifs025_wind_speed_100m_d2",
        "nwp_ecmwf_ifs025_wind_speed_100m_d3",
    ]


def test_only_d3_survives_at_the_72h_horizon():
    """h=72 needs 24N >= 72, so d1 and d2 would both be issued in the future."""
    idx = pd.date_range("2024-03-01", periods=300, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(300.0)}, index=idx)
    out = add_nwp_features(df, _nwp_frame(idx), offset_hours=72)
    kept = [c for c in out.columns if c.startswith("nwp_")]
    assert kept == ["nwp_ecmwf_ifs025_wind_speed_100m_d3"]


def test_weekly_horizon_raises_when_no_lead_is_long_enough():
    """h=168 needs d7+. With only d1..d3 the honest response is to refuse.

    Returning a frame here would hand the model forecasts issued up to six days
    after the moment they are supposed to be known — excellent, invalid numbers.
    """
    idx = pd.date_range("2024-03-01", periods=400, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(400.0)}, index=idx)
    with pytest.raises(ValueError, match="lead"):
        add_nwp_features(df, _nwp_frame(idx), offset_hours=168)


def test_weekly_horizon_accepts_a_long_enough_lead():
    """With d7 present, a 168 h horizon is legitimate and d1..d6 are dropped."""
    idx = pd.date_range("2024-03-01", periods=400, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(400.0)}, index=idx)
    out = add_nwp_features(df, _nwp_frame(idx, leads=(1, 3, 7)), offset_hours=168)
    kept = [c for c in out.columns if c.startswith("nwp_")]
    assert kept == ["nwp_ecmwf_ifs025_wind_speed_100m_d7"]


def test_unparseable_lead_suffix_is_kept_rather_than_silently_dropped():
    """A column that does not match the `_d{N}` convention must not vanish.

    Dropping something we failed to parse would be a silent feature loss; the
    guard is only allowed to remove columns it positively identified as too
    short a lead.
    """
    idx = pd.date_range("2024-03-01", periods=300, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(300.0)}, index=idx)
    nwp = _nwp_frame(idx)
    nwp["nwp_ecmwf_ifs025_ensemble_spread"] = 1.0
    out = add_nwp_features(df, nwp, offset_hours=72)
    assert "nwp_ecmwf_ifs025_ensemble_spread" in out.columns
