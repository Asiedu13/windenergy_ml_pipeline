"""Stage 1 — sequence deep-learning models (LSTM / TFT / N-BEATS / DLinear via darts).

Sibling module to :mod:`src.stage1_windspeed`. Same public shape:
:func:`train_stage1_deep` returns a :class:`Stage1Result` so the rest of the
pipeline (evaluate / report) treats tree and deep models identically.

Design choices:
- Target series = raw hourly hub wind speed (unshifted). darts handles the
  horizon via ``output_chunk_length``.
- Past covariates = exogenous weather + calendar features, **plus** the
  target-derived lag / rolling columns when ``include_target_lags`` is set
  (default). These are strictly backward-looking and sit inside the input
  chunk, so they cannot leak; giving the network the same labelled lag
  variables XGBoost gets is recommendation #2 of
  ``docs/09_why_deep_learning_underperformed.md``.
- Training budget follows recommendations #1 and #3 of that document: a
  held-out validation tail carved from the train slice, early stopping on
  ``val_loss`` with best-checkpoint restore, linear warmup → cosine decay of
  the learning rate, gradient clipping, and an optional short-budget LR sweep.
- One-shot train on the train slice, prediction via ``historical_forecasts``
  with ``retrain=False`` for the test slice.
- SHAP is not supported for deep models; report notes this.

Alignment note (important)
--------------------------
``historical_forecasts(..., last_points_only=True)`` returns a series indexed
by the *target* timestamp τ, i.e. ``hist[τ]`` is the prediction of the hub wind
speed **at** τ, produced from data up to ``τ - horizon``. The feature frame, by
contrast, is indexed by the *feature* timestamp t with ``y[t] = hub(t + horizon)``.
Comparing the two directly puts the whole test set ``horizon`` hours out of
phase. :func:`_to_feature_time` performs the conversion.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR

from .stage1_windspeed import (
    ClimatologyForecaster,
    PersistenceForecaster,
    Stage1Result,
    compute_metrics,
)
from .utils import Config, ensure_dir, get_logger

logger = get_logger(__name__)

TARGET_DERIVED_PREFIXES = ("wind_speed_hub_lag", "wind_speed_hub_roll")


# ---------------------------------------------------------------------------
# Deep-model configuration
# ---------------------------------------------------------------------------

#: Defaults for the ``model.deep`` config block. Anything the YAML omits falls
#: back to these; ``model.deep.overrides.<algo>`` deep-merges on top per model.
DEEP_DEFAULTS: dict[str, Any] = {
    "input_chunk_length": 48,
    "hidden_size": 64,
    "n_rnn_layers": 1,
    "n_epochs": 200,
    "batch_size": 128,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "gradient_clip_val": 1.0,
    # Training objective, applied uniformly across the zoo. BlockRNN, N-BEATS
    # and DLinear already default to MSE, so this is a no-op for them — but
    # `TFTModel` defaults to `likelihood=QuantileRegression()` over 17
    # quantiles, a substantially harder problem whose point forecast is only
    # the median. Naming the loss here makes TFT deterministic and puts every
    # model on the same objective. Set to null to restore each model's own
    # darts default (i.e. give TFT its quantile likelihood back).
    "loss_fn": "mse",
    # Recommendation #2 — hand the network the same labelled lag/rolling
    # variables the tree model gets, as past covariates.
    "include_target_lags": True,
    # Recommendation #1 — validation tail + early stopping.
    "val_fraction": 0.15,
    "early_stopping": {"enabled": True, "patience": 20, "min_delta": 1e-4},
    # Recommendation #3 — warmup + cosine decay, and an optional LR sweep.
    "scheduler": {"enabled": True, "warmup_epochs": 5, "min_lr_factor": 0.02},
    "lr_sweep": {"enabled": False, "candidates": [3e-4, 1e-3, 3e-3], "epochs": 8},
    "nbeats": {"num_stacks": 3, "num_blocks": 3, "num_layers": 4, "layer_widths": 256},
    "dlinear": {"kernel_size": 25},
    "overrides": {},
}


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    return dict(getattr(obj, "__dict__", {}))


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_deep_cfg(config: Config, algo: str) -> dict[str, Any]:
    """Merge ``DEEP_DEFAULTS`` ← ``model.deep`` ← ``model.deep.overrides[algo]``."""
    cfg = _deep_merge(DEEP_DEFAULTS, _as_dict(getattr(config.model, "deep", None)))
    overrides = _as_dict(cfg.get("overrides")).get(algo.lower(), {})
    if overrides:
        cfg = _deep_merge(cfg, dict(overrides))
    return cfg


# ---------------------------------------------------------------------------
# Learning-rate schedule (recommendation #3)
# ---------------------------------------------------------------------------

def _warmup_cosine_factor(epoch: int, warmup_epochs: int, max_epochs: int,
                          min_lr_factor: float) -> float:
    """Multiplicative LR factor: linear warmup, then cosine decay to the floor."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    span = max(1, max_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / span))
    return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class WarmupCosineLR(LambdaLR):
    """Linear warmup → cosine decay, stepped once per epoch.

    Kept as a module-level class (rather than an inline lambda) so darts can
    pickle the model after fitting.
    """

    def __init__(self, optimizer, warmup_epochs: int = 5, max_epochs: int = 200,
                 min_lr_factor: float = 0.02, last_epoch: int = -1):
        super().__init__(
            optimizer,
            partial(
                _warmup_cosine_factor,
                warmup_epochs=int(warmup_epochs),
                max_epochs=int(max_epochs),
                min_lr_factor=float(min_lr_factor),
            ),
            last_epoch=last_epoch,
        )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

#: Accepted ``model.deep.loss_fn`` values → ``torch.nn`` class names.
_LOSS_FNS = {"mse": "MSELoss", "mae": "L1Loss", "l1": "L1Loss", "huber": "HuberLoss"}


def _resolve_loss_fn(name: str | None):
    """Build a torch loss module from a config string, or None to leave it alone."""
    if name is None:
        return None
    key = str(name).lower()
    if key not in _LOSS_FNS:
        raise ValueError(
            f"Unknown deep.loss_fn {name!r}; expected one of "
            f"{sorted(_LOSS_FNS)} or null."
        )
    import torch.nn as nn
    return getattr(nn, _LOSS_FNS[key])()


def _make_darts_model(
    algo: str,
    cfg: dict[str, Any],
    horizon: int,
    seed: int,
    *,
    n_epochs: int,
    learning_rate: float,
    use_scheduler: bool = True,
    callbacks: list | None = None,
    model_name: str | None = None,
    work_dir: str | Path | None = None,
    save_checkpoints: bool = False,
):
    """Build a darts forecaster from a resolved deep-config dict."""
    algo = algo.lower()

    trainer_kwargs: dict[str, Any] = {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "gradient_clip_val": float(cfg["gradient_clip_val"]),
    }
    if callbacks:
        trainer_kwargs["callbacks"] = list(callbacks)

    common: dict[str, Any] = dict(
        input_chunk_length=int(cfg["input_chunk_length"]),
        output_chunk_length=horizon,
        n_epochs=int(n_epochs),
        batch_size=int(cfg["batch_size"]),
        random_state=seed,
        optimizer_kwargs={"lr": float(learning_rate)},
        pl_trainer_kwargs=trainer_kwargs,
        force_reset=True,
    )
    # Passing an explicit loss_fn also stops TFTModel from falling back to its
    # QuantileRegression likelihood (darts only applies that when *both*
    # `likelihood` and `loss_fn` are None).
    loss_fn = _resolve_loss_fn(cfg.get("loss_fn"))
    if loss_fn is not None:
        common["loss_fn"] = loss_fn

    if save_checkpoints:
        common.update(save_checkpoints=True, model_name=model_name,
                      work_dir=str(work_dir))
    elif model_name is not None:
        common.update(model_name=model_name, work_dir=str(work_dir))

    scheduler_cfg = cfg["scheduler"]
    if use_scheduler and scheduler_cfg.get("enabled", True):
        common.update(
            lr_scheduler_cls=WarmupCosineLR,
            lr_scheduler_kwargs={
                "warmup_epochs": int(scheduler_cfg.get("warmup_epochs", 5)),
                "max_epochs": int(n_epochs),
                "min_lr_factor": float(scheduler_cfg.get("min_lr_factor", 0.02)),
                "interval": "epoch",
            },
        )

    hidden = int(cfg["hidden_size"])
    dropout = float(cfg["dropout"])

    if algo == "lstm":
        from darts.models import BlockRNNModel
        # BlockRNN (seq-to-seq) supports past_covariates — the right shape for
        # our "past weather → future wind speed" problem. `RNNModel` only
        # supports future_covariates and would be a semantic mismatch.
        return BlockRNNModel(
            model="LSTM",
            hidden_dim=hidden,
            n_rnn_layers=int(cfg["n_rnn_layers"]),
            dropout=dropout,
            **common,
        )
    if algo == "tft":
        from darts.models import TFTModel
        return TFTModel(
            hidden_size=hidden,
            dropout=dropout,
            add_relative_index=True,  # needed if no future covariates supplied
            use_static_covariates=False,
            **common,
        )
    if algo == "nbeats":
        from darts.models import NBEATSModel
        nbeats_cfg = cfg["nbeats"]
        return NBEATSModel(
            num_stacks=int(nbeats_cfg.get("num_stacks", 3)),
            num_blocks=int(nbeats_cfg.get("num_blocks", 3)),
            num_layers=int(nbeats_cfg.get("num_layers", 4)),
            layer_widths=int(nbeats_cfg.get("layer_widths", 256)),
            dropout=dropout,
            **common,
        )
    if algo == "dlinear":
        from darts.models import DLinearModel
        dl_cfg = cfg["dlinear"]
        return DLinearModel(
            kernel_size=int(dl_cfg.get("kernel_size", 25)),
            use_static_covariates=False,
            **common,
        )
    raise ValueError(f"Unknown deep algo: {algo!r}")


def _early_stopping_callback(cfg: dict[str, Any]):
    """Fresh EarlyStopping instance (callbacks are stateful — never reuse)."""
    es_cfg = cfg["early_stopping"]
    if not es_cfg.get("enabled", True):
        return None
    from pytorch_lightning.callbacks import EarlyStopping
    return EarlyStopping(
        monitor="val_loss",
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 1e-4)),
        mode="min",
    )


# ---------------------------------------------------------------------------
# Series plumbing
# ---------------------------------------------------------------------------

def _exogenous_cols(feature_cols: list[str], include_target_lags: bool = True) -> list[str]:
    """Select past-covariate columns.

    Target-derived lag / rolling columns are backward-looking (``add_rolling_features``
    shifts by 1 before rolling), so they are safe inside the input chunk. Set
    ``include_target_lags=False`` to reproduce the pre-v4 behaviour where they
    were withheld from the sequence models.
    """
    if include_target_lags:
        return list(feature_cols)
    return [c for c in feature_cols if not c.startswith(TARGET_DERIVED_PREFIXES)]


def _to_timeseries(series: pd.Series, name: str):
    from darts import TimeSeries
    s = series.copy()
    # darts needs tz-naive index
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.name = name
    return TimeSeries.from_series(s, fill_missing_dates=True, freq="h")


def _to_timeseries_df(df: pd.DataFrame, name: str = "cov"):
    from darts import TimeSeries
    d = df.copy()
    if d.index.tz is not None:
        d.index = d.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_dataframe(d, fill_missing_dates=True, freq="h")


def _to_feature_time(hist, scaler, offset_hours: int) -> pd.Series:
    """Un-scale a ``historical_forecasts`` result and re-index onto feature time.

    ``hist`` is indexed by the *target* timestamp τ. The feature frame is
    indexed by t with ``y[t] = hub(t + offset_hours)``, so shifting the index
    back by ``offset_hours`` makes the two directly comparable.
    """
    s = scaler.inverse_transform(hist).to_series()
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC") - pd.Timedelta(hours=offset_hours)
    return s


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _fit_and_val_loss(model, target_train, cov_train, target_val, cov_val) -> float:
    """Fit a model with a validation series and return its final ``val_loss``."""
    model.fit(
        target_train,
        past_covariates=cov_train,
        val_series=target_val,
        val_past_covariates=cov_val,
        verbose=False,
    )
    metrics = getattr(model.trainer, "callback_metrics", {}) or {}
    val = metrics.get("val_loss")
    return float(val) if val is not None else float("inf")


def _sweep_learning_rate(algo, cfg, horizon, seed, target_train, cov_train,
                         target_val, cov_val) -> float:
    """Short-budget LR sweep — train each candidate briefly, keep the best val_loss."""
    sweep_cfg = cfg["lr_sweep"]
    candidates = [float(c) for c in sweep_cfg.get("candidates", [])]
    sweep_epochs = int(sweep_cfg.get("epochs", 8))
    if not candidates:
        return float(cfg["learning_rate"])

    best_lr, best_loss = float(cfg["learning_rate"]), float("inf")
    for lr in candidates:
        probe = _make_darts_model(
            algo, cfg, horizon, seed,
            n_epochs=sweep_epochs, learning_rate=lr,
            use_scheduler=False,  # a decayed LR would confound the comparison
            model_name=f"{algo}_lrsweep", work_dir=cfg["_work_dir"],
        )
        loss = _fit_and_val_loss(probe, target_train, cov_train, target_val, cov_val)
        logger.info("  LR sweep %s: lr=%.5g → val_loss=%.5f", algo, lr, loss)
        if loss < best_loss:
            best_lr, best_loss = lr, loss
    logger.info("LR sweep %s: best lr=%.5g (val_loss=%.5f)", algo, best_lr, best_loss)
    return best_lr


def train_stage1_deep(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    mode: Literal["nowcast", "forecast"] = "forecast",
    weather_df: pd.DataFrame | None = None,
) -> Stage1Result:
    """Train a deep sequence model on hourly hub wind speed.

    Parameters mirror :func:`src.stage1_windspeed.train_stage1`. The ``X, y``
    are used for baseline metrics, covariate construction and test-window
    alignment; the deep model rebuilds the raw (unshifted) target series from
    ``weather_df``.
    """
    algo = config.model.stage1_algo.lower()
    cfg = resolve_deep_cfg(config, algo)
    horizon = int(config.forecast.horizon_hours) if mode == "forecast" else 1
    # y[t] = hub(t + offset); nowcast targets the current step, so offset 0.
    offset = horizon if mode == "forecast" else 0

    n = len(X)
    if n < 100:
        raise ValueError(f"Not enough samples for deep Stage 1 (got {n}).")

    split_at = int(n * (1 - config.model.test_split))
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    # Baselines (same as tree model — the plan's "beat persistence" guardrail)
    persistence = PersistenceForecaster().fit(X_train, y_train)
    climatology = ClimatologyForecaster().fit(X_train, y_train)
    pred_pers = persistence.predict(X_test)
    pred_clim = climatology.predict(X_test)
    m_pers = compute_metrics(y_test.to_numpy(), pred_pers)
    m_clim = compute_metrics(y_test.to_numpy(), pred_clim)

    # Build the raw target series (unshifted hub wind speed).
    from .features import extrapolate_hub_wind
    if weather_df is None:
        raise ValueError("weather_df is required for deep-model training.")
    hub = extrapolate_hub_wind(weather_df, config.site.hub_height_m)
    # Cover the feature window *plus* the horizon, so the last test target exists.
    hub = hub.loc[X.index.min():X.index.max() + pd.Timedelta(hours=offset)]
    target_ts = _to_timeseries(hub, "wind_speed_hub")

    # Past covariates. Recommendation #2: keep the lag/rolling columns.
    exog_cols = _exogenous_cols(list(X.columns),
                                include_target_lags=bool(cfg["include_target_lags"]))
    exog_df = X[exog_cols].copy()
    # Constant / all-NaN columns are useless and hurt the scaler's conditioning.
    keep = [c for c in exog_df.columns if exog_df[c].nunique(dropna=True) > 1]
    exog_df = exog_df[keep]
    # Reindex to the full hourly index of the target and fill any gaps.
    exog_df = exog_df.reindex(hub.index).ffill().bfill()
    covariates_ts = _to_timeseries_df(exog_df, "past_covariates")
    logger.info("%s: %d past-covariate columns (target lags %s)", algo,
                len(exog_df.columns),
                "included" if cfg["include_target_lags"] else "excluded")

    # --- three-way split: fit / validation tail / test ----------------------
    input_chunk = int(cfg["input_chunk_length"])
    min_series_len = input_chunk + horizon + 1
    val_len = int(round(split_at * float(cfg["val_fraction"])))
    # Validation needs its own context window on top of the samples it scores.
    val_len = max(val_len, min_series_len) if val_len > 0 else 0
    if split_at - val_len < min_series_len:
        logger.warning(
            "%s: train slice too short for a validation tail (%d rows); "
            "disabling early stopping.", algo, split_at,
        )
        val_len = 0

    split_ts = pd.Timestamp(X.index[split_at]).tz_localize(None)
    target_train_full, _ = target_ts.split_before(split_ts)
    cov_train_full, _ = covariates_ts.split_before(split_ts)

    # Scale on the fitting slice only (never on validation or test).
    from darts.dataprocessing.transformers import Scaler
    target_scaler, cov_scaler = Scaler(), Scaler()

    if val_len > 0:
        val_start_ts = pd.Timestamp(X.index[split_at - val_len]).tz_localize(None)
        # The validation series carries `input_chunk` hours of its own context
        # so its first scored sample sits exactly at `val_start_ts`.
        val_ctx_ts = val_start_ts - pd.Timedelta(hours=input_chunk)
        target_fit, _ = target_ts.split_before(val_start_ts)
        cov_fit, _ = cov_train_full.split_before(val_start_ts)
        target_val = target_train_full.slice(val_ctx_ts, split_ts)
        cov_val = cov_train_full.slice(val_ctx_ts, split_ts)
    else:
        target_fit, cov_fit = target_train_full, cov_train_full
        target_val = cov_val = None

    target_fit_s = target_scaler.fit_transform(target_fit)
    cov_fit_s = cov_scaler.fit_transform(cov_fit)
    target_val_s = target_scaler.transform(target_val) if target_val is not None else None
    cov_val_s = cov_scaler.transform(cov_val) if cov_val is not None else None
    target_all_s = target_scaler.transform(target_ts)
    cov_all_s = cov_scaler.transform(covariates_ts)

    work_dir = ensure_dir(Path(config.paths.outputs) / "darts")
    cfg["_work_dir"] = str(work_dir)

    # --- recommendation #3: optional LR sweep -------------------------------
    learning_rate = float(cfg["learning_rate"])
    if cfg["lr_sweep"].get("enabled", False) and target_val_s is not None:
        learning_rate = _sweep_learning_rate(
            algo, cfg, horizon, config.model.seed,
            target_fit_s, cov_fit_s, target_val_s, cov_val_s,
        )

    # --- recommendation #1: long budget + early stopping --------------------
    n_epochs = int(cfg["n_epochs"])
    early_stop = _early_stopping_callback(cfg) if target_val_s is not None else None
    model_name = f"{algo}_stage1"
    model = _make_darts_model(
        algo, cfg, horizon, config.model.seed,
        n_epochs=n_epochs, learning_rate=learning_rate,
        callbacks=[early_stop] if early_stop else None,
        model_name=model_name, work_dir=work_dir,
        save_checkpoints=early_stop is not None,
    )

    logger.info(
        "Fitting %s — %d epochs max, lr=%.5g, batch=%d, %d fit points, "
        "%s validation points",
        algo, n_epochs, learning_rate, int(cfg["batch_size"]), len(target_fit_s),
        len(target_val_s) if target_val_s is not None else "no",
    )
    if target_val_s is not None:
        model.fit(target_fit_s, past_covariates=cov_fit_s,
                  val_series=target_val_s, val_past_covariates=cov_val_s,
                  verbose=False)
        stopped_at = getattr(model.trainer, "current_epoch", None)
        logger.info("%s: training stopped after %s epochs (max %d)",
                    algo, stopped_at, n_epochs)
        # Early stopping leaves the *last* weights loaded, not the best ones.
        model = type(model).load_from_checkpoint(
            model_name=model_name, work_dir=str(work_dir), best=True,
        )
    else:
        model.fit(target_fit_s, past_covariates=cov_fit_s, verbose=False)

    # --- test-window backtest ----------------------------------------------
    # `start` is the first point of the first forecast *window*; with
    # last_points_only the first value returned lands at `start + horizon - 1`.
    # We want that first value at target time `split_ts + offset`, which maps
    # back to feature time `split_ts` — i.e. y_test.index[0].
    backtest_start = split_ts + pd.Timedelta(hours=offset - horizon + 1)
    logger.info("Backtesting %s on the test window (%d points)", algo, len(y_test))
    hist = model.historical_forecasts(
        series=target_all_s,
        past_covariates=cov_all_s,
        start=backtest_start,
        forecast_horizon=horizon,
        stride=1,
        retrain=False,
        verbose=False,
        last_points_only=True,  # one prediction (h steps ahead) per stride
    )
    hist_series = _to_feature_time(hist, target_scaler, offset)

    common = hist_series.index.intersection(y_test.index)
    if len(common) < 10:
        raise RuntimeError(
            f"Deep model produced only {len(common)} aligned test points."
        )
    pred_test = hist_series.reindex(y_test.index)
    # Fill any gaps at the edges so the metrics stay defined.
    pred_test = pred_test.interpolate(limit_direction="both")
    m_test = compute_metrics(y_test.to_numpy(), pred_test.to_numpy())

    # Train metrics (in-sample, informational)
    hist_train = model.historical_forecasts(
        series=target_scaler.transform(target_train_full),
        past_covariates=cov_scaler.transform(cov_train_full),
        start=0.5,  # skip the warmup half of train
        forecast_horizon=horizon,
        stride=max(1, horizon // 2),
        retrain=False,
        verbose=False,
        last_points_only=True,
    )
    ht = _to_feature_time(hist_train, target_scaler, offset)
    train_common = ht.index.intersection(y_train.index)
    m_train = compute_metrics(y_train.loc[train_common].to_numpy(),
                              ht.loc[train_common].to_numpy())

    beats = m_test.mae < m_pers.mae
    logger.info("%s: test MAE=%.3f RMSE=%.3f R2=%.3f (persistence MAE=%.3f)",
                algo, m_test.mae, m_test.rmse, m_test.r2, m_pers.mae)
    if not beats:
        logger.warning(
            "Deep model does NOT beat persistence on test MAE (%.3f vs %.3f).",
            m_test.mae, m_pers.mae,
        )

    return Stage1Result(
        model=model,
        features=list(X.columns),
        metrics_train=m_train,
        metrics_test=m_test,
        metrics_persistence=m_pers,
        metrics_climatology=m_clim,
        cv_metrics=[],
        y_test=y_test,
        pred_test=pd.Series(pred_test.to_numpy(), index=y_test.index, name="pred_ws"),
        pred_test_persistence=pd.Series(pred_pers, index=y_test.index, name="pers"),
        pred_test_climatology=pd.Series(pred_clim, index=y_test.index, name="clim"),
        mode=mode,
        beats_persistence=beats,
    )
