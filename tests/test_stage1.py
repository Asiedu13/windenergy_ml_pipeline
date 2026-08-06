import numpy as np
import pandas as pd
import pytest

from src.features import build_feature_frame
from src.stage1_windspeed import (
    ClimatologyForecaster,
    PersistenceForecaster,
    compute_metrics,
    load_stage1,
    save_stage1,
    train_stage1,
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


def _make_config(tmp_path=None) -> Config:
    return Config(
        site=SiteConfig(name="t", latitude=5.79, longitude=0.9, hub_height_m=100),
        data=DataConfig(source="nasa_power", start="2023-01-01", end="2023-12-31",
                        cache_dir="/tmp"),
        turbine=TurbineConfig(name="E-101/3050"),
        forecast=ForecastConfig(horizon_hours=6),
        model=ModelConfig(lags=[1, 2, 3, 6], rolling_windows=[3, 6],
                          cv_folds=3, test_split=0.2, seed=42),
        paths=PathsConfig(outputs=str(tmp_path) if tmp_path else "outputs"),
        raw={},
    )


def _synthetic_weather(hours: int = 24 * 90) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=hours, freq="h", tz="UTC")
    t = np.arange(hours)
    # Deterministic diurnal + seasonal + AR(1) noise → learnable pattern
    diurnal = 2.0 * np.sin(2 * np.pi * t / 24)
    seasonal = 1.0 * np.sin(2 * np.pi * t / (24 * 30))
    noise = np.zeros(hours)
    for i in range(1, hours):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.5)
    wind = 7.0 + diurnal + seasonal + noise

    return pd.DataFrame(
        {
            "wind_speed_10m": wind * 0.7,
            "wind_speed_100m": wind,
            "wind_dir_10m": (180 + 30 * np.sin(t / 12)) % 360,
            "wind_dir_100m": (200 + 30 * np.sin(t / 12)) % 360,
            "temperature_c": 28 + 3 * np.sin(2 * np.pi * t / 24),
            "pressure_pa": 101_300.0,
            "roughness": 0.15,
        },
        index=idx,
    )


def test_persistence_predicts_lag1():
    cfg = _make_config()
    X, y, _ = build_feature_frame(_synthetic_weather(200), cfg, mode="nowcast")
    pers = PersistenceForecaster().fit(X, y)
    pred = pers.predict(X)
    assert pred.shape == y.shape
    # Persistence output equals whatever the *_lag1 column holds
    np.testing.assert_allclose(pred, X[pers.reference_col_].to_numpy())


def test_climatology_shape_and_no_nan():
    cfg = _make_config()
    X, y, _ = build_feature_frame(_synthetic_weather(500), cfg, mode="nowcast")
    clim = ClimatologyForecaster().fit(X, y)
    pred = clim.predict(X)
    assert pred.shape == y.shape
    assert not np.isnan(pred).any()


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y.copy())
    assert m.mae == 0
    assert m.rmse == 0
    assert m.r2 == 1


def test_train_stage1_beats_persistence(tmp_path):
    cfg = _make_config(tmp_path)
    weather = _synthetic_weather(hours=24 * 120)  # 120 days hourly
    X, y, _ = build_feature_frame(weather, cfg, mode="nowcast")
    result = train_stage1(X, y, cfg, mode="nowcast")
    # On a signal with strong diurnal + seasonal structure, xgboost must beat
    # naive persistence on test MAE.
    assert result.beats_persistence, (
        f"test MAE {result.metrics_test.mae:.3f} !< persistence "
        f"{result.metrics_persistence.mae:.3f}"
    )


def test_save_load_round_trip(tmp_path):
    cfg = _make_config(tmp_path)
    weather = _synthetic_weather(hours=24 * 60)
    X, y, _ = build_feature_frame(weather, cfg, mode="nowcast")
    result = train_stage1(X, y, cfg, mode="nowcast")
    p = save_stage1(result, tmp_path / "stage1.pkl", cfg)
    loaded = load_stage1(p)
    pred_before = result.model.predict(X.iloc[:10])
    pred_after = loaded["model"].predict(X.iloc[:10])
    np.testing.assert_allclose(pred_before, pred_after)
    assert loaded["features"] == result.features
