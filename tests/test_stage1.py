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


def test_train_stage1_deep_lstm_runs_end_to_end(tmp_path):
    """Smoke test: LSTM trains without error on a small synthetic series and
    produces test predictions aligned with y_test."""
    pytest.importorskip("darts")
    from src.stage1_deep import train_stage1_deep

    cfg = _make_config(tmp_path)
    cfg.model.stage1_algo = "lstm"
    cfg.model.deep = {
        "input_chunk_length": 24,
        "hidden_size": 8,
        "n_epochs": 2,
        "batch_size": 32,
        "dropout": 0.0,
    }
    weather = _synthetic_weather(hours=24 * 60)  # 60 days
    X, y, _ = build_feature_frame(weather, cfg, mode="forecast")
    result = train_stage1_deep(X, y, cfg, mode="forecast", weather_df=weather)
    # Predictions should be finite and cover the test window
    assert np.isfinite(result.pred_test.to_numpy()).all()
    assert len(result.pred_test) == len(result.y_test)
    # Metric objects populated
    assert result.metrics_test.mae > 0
    assert result.metrics_persistence.mae > 0


@pytest.mark.parametrize("algo", ["tft", "nbeats", "dlinear"])
def test_darts_new_models_smoke(tmp_path, algo):
    """Smoke test: TFT, N-BEATS, DLinear each train + predict end-to-end."""
    pytest.importorskip("darts")
    from src.stage1_deep import train_stage1_deep

    cfg = _make_config(tmp_path)
    cfg.model.stage1_algo = algo
    cfg.model.deep = {
        "input_chunk_length": 24,
        "hidden_size": 8,
        "n_epochs": 2,
        "batch_size": 32,
        "dropout": 0.0,
        "nbeats": {"num_stacks": 2, "num_blocks": 1, "layer_widths": 32},
        "dlinear": {"kernel_size": 5},
    }
    weather = _synthetic_weather(hours=24 * 60)
    X, y, _ = build_feature_frame(weather, cfg, mode="forecast")
    result = train_stage1_deep(X, y, cfg, mode="forecast", weather_df=weather)
    assert np.isfinite(result.pred_test.to_numpy()).all()
    assert len(result.pred_test) == len(result.y_test)
    assert result.metrics_test.mae > 0


def test_save_load_round_trip_darts_model(tmp_path):
    """darts models must survive save/load — plain joblib drops their weights."""
    pytest.importorskip("darts")
    from src.stage1_deep import train_stage1_deep

    cfg = _make_config(tmp_path)
    cfg.model.stage1_algo = "dlinear"
    cfg.model.deep = {
        "input_chunk_length": 24, "n_epochs": 2, "batch_size": 32,
        "dropout": 0.0, "dlinear": {"kernel_size": 5},
    }
    weather = _synthetic_weather(hours=24 * 60)
    X, y, _ = build_feature_frame(weather, cfg, mode="forecast")
    result = train_stage1_deep(X, y, cfg, mode="forecast", weather_df=weather)

    p = save_stage1(result, tmp_path / "stage1.pkl", cfg)
    loaded = load_stage1(p)
    assert loaded["features"] == result.features
    # The torch module must come back, not None — that's the bug being guarded.
    assert loaded["model"] is not None
    assert loaded["model"].model is not None

    from darts import TimeSeries
    series = TimeSeries.from_series(
        pd.Series(np.linspace(5.0, 9.0, 120),
                  index=pd.date_range("2023-01-01", periods=120, freq="h"))
    )
    np.testing.assert_allclose(
        result.model.predict(n=6, series=series).values(),
        loaded["model"].predict(n=6, series=series).values(),
        rtol=1e-5, atol=1e-6,
    )


def test_resolve_deep_cfg_applies_defaults_and_overrides(tmp_path):
    """model.deep merges over DEEP_DEFAULTS, then overrides.<algo> over that."""
    pytest.importorskip("darts")
    from src.stage1_deep import DEEP_DEFAULTS, resolve_deep_cfg

    cfg = _make_config(tmp_path)
    cfg.model.deep = {
        "n_epochs": 150,
        "early_stopping": {"patience": 5},
        "overrides": {"tft": {"n_epochs": 60, "batch_size": 256}},
    }
    lstm = resolve_deep_cfg(cfg, "lstm")
    tft = resolve_deep_cfg(cfg, "tft")

    assert lstm["n_epochs"] == 150
    assert tft["n_epochs"] == 60           # per-algo override wins
    assert tft["batch_size"] == 256
    assert lstm["batch_size"] == DEEP_DEFAULTS["batch_size"]  # untouched default
    # Nested dicts deep-merge rather than replace wholesale
    assert lstm["early_stopping"]["patience"] == 5
    assert lstm["early_stopping"]["min_delta"] == DEEP_DEFAULTS["early_stopping"]["min_delta"]


def test_exogenous_cols_lag_toggle():
    """Recommendation #2 — target lag/rolling columns are covariates by default."""
    pytest.importorskip("darts")
    from src.stage1_deep import _exogenous_cols

    cols = ["wind_speed_100m", "hour_sin",
            "wind_speed_hub_lag12", "wind_speed_hub_rollmean6"]
    assert _exogenous_cols(cols, include_target_lags=True) == cols
    assert _exogenous_cols(cols, include_target_lags=False) == [
        "wind_speed_100m", "hour_sin",
    ]


def test_loss_fn_makes_tft_deterministic():
    """darts' TFTModel silently defaults to a 17-quantile QuantileRegression
    likelihood unless an explicit loss_fn is given. Naming the loss must put it
    on the same MSE objective as the rest of the zoo."""
    pytest.importorskip("darts")
    import torch.nn as nn

    from src.stage1_deep import DEEP_DEFAULTS, _make_darts_model, _resolve_loss_fn

    assert DEEP_DEFAULTS["loss_fn"] == "mse"
    assert isinstance(_resolve_loss_fn("mae"), nn.L1Loss)
    assert _resolve_loss_fn(None) is None
    with pytest.raises(ValueError, match="Unknown deep.loss_fn"):
        _resolve_loss_fn("cross_entropy")

    cfg = dict(DEEP_DEFAULTS, input_chunk_length=8, hidden_size=4, batch_size=8)
    tft = _make_darts_model("tft", cfg, 2, 42, n_epochs=1, learning_rate=1e-3)
    assert tft.likelihood is None, "TFT should be deterministic when loss_fn is set"
    assert isinstance(tft.model_params["loss_fn"], nn.MSELoss)

    # ...and leaving loss_fn unset restores darts' probabilistic default.
    quantile_cfg = dict(cfg, loss_fn=None)
    tft_q = _make_darts_model("tft", quantile_cfg, 2, 42, n_epochs=1, learning_rate=1e-3)
    assert tft_q.likelihood is not None


def test_to_feature_time_shifts_index_back_by_horizon():
    """historical_forecasts is indexed at target time τ; features are indexed at
    t where y[t] = hub(t + horizon). The conversion must subtract the horizon."""
    pytest.importorskip("darts")
    from darts import TimeSeries

    from src.stage1_deep import _to_feature_time

    idx = pd.date_range("2023-03-01", periods=48, freq="h")
    hist = TimeSeries.from_series(pd.Series(np.arange(48.0), index=idx))

    class _Identity:
        def inverse_transform(self, ts):
            return ts

    out = _to_feature_time(hist, _Identity(), offset_hours=24)
    assert out.index[0] == idx[0].tz_localize("UTC") - pd.Timedelta(hours=24)
    np.testing.assert_allclose(out.to_numpy(), np.arange(48.0))
    # offset 0 (nowcast) leaves the index alone apart from localisation
    same = _to_feature_time(hist, _Identity(), offset_hours=0)
    assert same.index[0] == idx[0].tz_localize("UTC")


def test_deep_forecast_predictions_are_horizon_aligned(tmp_path):
    """End-to-end guard against the τ-vs-t misalignment.

    On a clean 37-hour periodic signal, 24 h-ahead persistence is a poor
    baseline but the true value is highly learnable. A correctly aligned model
    must beat persistence by a wide margin; a model whose predictions are
    `horizon` hours out of phase cannot.
    """
    pytest.importorskip("darts")
    from src.stage1_deep import train_stage1_deep

    cfg = _make_config(tmp_path)
    cfg.model.stage1_algo = "dlinear"
    cfg.forecast.horizon_hours = 24
    cfg.model.deep = {
        "input_chunk_length": 48,
        "n_epochs": 40,
        "batch_size": 64,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "val_fraction": 0.15,
        "early_stopping": {"patience": 8},
        "scheduler": {"enabled": True, "warmup_epochs": 2},
        "dlinear": {"kernel_size": 5},
    }

    hours = 24 * 120
    t = np.arange(hours)
    idx = pd.date_range("2023-01-01", periods=hours, freq="h", tz="UTC")
    # Period 37 h → not commensurate with the 24 h horizon, so persistence is weak.
    wind = 8.0 + 2.5 * np.sin(2 * np.pi * t / 37)
    weather = pd.DataFrame(
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
    X, y, _ = build_feature_frame(weather, cfg, mode="forecast")
    result = train_stage1_deep(X, y, cfg, mode="forecast", weather_df=weather)

    assert np.isfinite(result.pred_test.to_numpy()).all()
    assert result.metrics_test.mae < 0.5 * result.metrics_persistence.mae, (
        f"deep MAE {result.metrics_test.mae:.3f} vs persistence "
        f"{result.metrics_persistence.mae:.3f} — predictions look misaligned"
    )


def test_tune_xgboost_returns_valid_params(tmp_path):
    """3-trial Optuna sweep on synthetic data returns usable params."""
    pytest.importorskip("optuna")
    from src.stage1_windspeed import _make_regressor, tune_xgboost

    cfg = _make_config(tmp_path)
    weather = _synthetic_weather(hours=24 * 60)
    X, y, _ = build_feature_frame(weather, cfg, mode="nowcast")
    split = int(len(X) * 0.8)
    params = tune_xgboost(X.iloc[:split], y.iloc[:split], cfg, n_trials=3)
    # All expected search-space keys present
    expected = {"max_depth", "learning_rate", "n_estimators", "min_child_weight",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}
    assert expected.issubset(params.keys())
    # Params are usable to build a regressor
    model = _make_regressor("xgboost", cfg.model.seed, params=params)
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])
    assert len(pred) == len(X) - split
    assert np.isfinite(pred).all()
