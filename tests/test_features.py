import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_direction_encodings,
    add_lag_features,
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
