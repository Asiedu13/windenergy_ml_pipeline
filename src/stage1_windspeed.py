"""Stage 1 — hub-height wind-speed prediction.

Includes:
- Persistence + climatology baselines (the model must beat persistence).
- Feature-influence analysis (Pearson + SHAP) with saved figures + ranked table.
- XGBoost / LightGBM regressor with TimeSeriesSplit CV.
- save/load helpers backed by joblib.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .utils import Config, ensure_dir, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class PersistenceForecaster:
    """ŷ_{t+h} = y_t. Fit is a no-op."""

    def __init__(self, horizon_hours: int = 0):
        self.horizon_hours = horizon_hours
        self.reference_col_ = "wind_speed_hub_lag1"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceForecaster":
        # Pick a column that best represents the current observation:
        # after feature-building, the raw hub wind-speed is no longer in X,
        # so we fall back to the lag-1 column (which IS the current obs before
        # the shift-into-target).
        candidates = [c for c in X.columns if c.endswith("_lag1")]
        if not candidates:
            raise ValueError("PersistenceForecaster needs a *_lag1 column in X.")
        # Prefer the hub-speed lag if present
        hub = [c for c in candidates if "hub" in c]
        self.reference_col_ = hub[0] if hub else candidates[0]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.reference_col_].to_numpy(dtype=float)


class ClimatologyForecaster:
    """Predicts the training mean of the target grouped by (month, hour_of_day)."""

    def __init__(self):
        self.lookup_: dict[tuple[int, int], float] = {}
        self.global_mean_: float = float("nan")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyForecaster":
        idx = y.index
        key = pd.MultiIndex.from_arrays([idx.month, idx.hour], names=["month", "hour"])
        grouped = y.groupby(key).mean()
        self.lookup_ = {(m, h): float(v) for (m, h), v in grouped.items()}
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        idx = X.index
        out = np.array([
            self.lookup_.get((m, h), self.global_mean_)
            for m, h in zip(idx.month, idx.hour)
        ], dtype=float)
        return out


# ---------------------------------------------------------------------------
# ML model
# ---------------------------------------------------------------------------

def _make_regressor(algo: str, seed: int, params: dict | None = None):
    algo = algo.lower()
    if algo == "xgboost":
        from xgboost import XGBRegressor
        defaults = dict(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, tree_method="hist", n_jobs=-1,
        )
        if params:
            defaults.update(params)
        return XGBRegressor(**defaults)
    if algo == "lightgbm":
        from lightgbm import LGBMRegressor
        defaults = dict(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        if params:
            defaults.update(params)
        return LGBMRegressor(**defaults)
    raise ValueError(f"Unknown stage1 algo: {algo!r}")


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 config: "Config", n_trials: int = 50) -> dict:
    """Optuna sweep with 3-fold TimeSeriesSplit CV — returns best params dict.

    Search space is deliberately narrower than an unconstrained grid so the
    50-trial default finishes in ~30 min on CPU. The final model still trains
    with `TimeSeriesSplit(cv_folds)` from config; this function only picks
    the hyperparameters.
    """
    import optuna
    from optuna.samplers import TPESampler
    from xgboost import XGBRegressor

    tscv = TimeSeriesSplit(n_splits=3)

    def _objective(trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0, log=True),
        }
        maes = []
        for tr, va in tscv.split(X_train):
            model = XGBRegressor(
                **params,
                random_state=config.model.seed,
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(X_train.iloc[tr], y_train.iloc[tr])
            pred = model.predict(X_train.iloc[va])
            maes.append(float(np.mean(np.abs(y_train.iloc[va].to_numpy() - pred))))
        return float(np.mean(maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=config.model.seed),
    )
    logger.info("Starting Optuna XGBoost sweep (%d trials, 3-fold TSCV)", n_trials)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best MAE=%.4f with params=%s", study.best_value, study.best_params)
    return study.best_params


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"MAE": self.mae, "RMSE": self.rmse, "MAPE": self.mape, "R2": self.r2}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return RegressionMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2)


# ---------------------------------------------------------------------------
# Feature-influence analysis
# ---------------------------------------------------------------------------

def analyze_feature_influence(
    X: pd.DataFrame,
    y: pd.Series,
    outputs_dir: str | Path,
    seed: int = 42,
    top_k_for_shap: int = 20,
) -> pd.DataFrame:
    """Save Pearson heatmap + SHAP summary; return ranked feature table."""
    from xgboost import XGBRegressor

    figures = ensure_dir(Path(outputs_dir) / "figures")
    reports = ensure_dir(Path(outputs_dir) / "reports")

    # Pearson
    corr_df = X.copy()
    corr_df["_target"] = y.values
    pearson = corr_df.corr(numeric_only=True)["_target"].drop("_target").sort_values(
        key=lambda s: s.abs(), ascending=False
    )

    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(pearson))))
    pearson.iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Pearson r vs. hub wind speed")
    ax.axvline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(figures / "pearson_heatmap.png", dpi=140)
    plt.close(fig)

    # SHAP on a quick XGBoost fit
    model = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        random_state=seed, tree_method="hist", n_jobs=-1,
    )
    model.fit(X, y)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # Cap at 2000 rows for speed
        sample = X.sample(min(2000, len(X)), random_state=seed)
        shap_values = explainer.shap_values(sample)
        mean_abs_shap = pd.Series(
            np.abs(shap_values).mean(axis=0), index=X.columns, name="mean_abs_shap"
        )
        # SHAP summary bar
        fig = plt.figure(figsize=(6, max(4, 0.25 * min(len(X.columns), top_k_for_shap))))
        shap.summary_plot(
            shap_values, sample, plot_type="bar",
            max_display=top_k_for_shap, show=False,
        )
        plt.tight_layout()
        plt.savefig(figures / "shap_summary.png", dpi=140)
        plt.close("all")
    except Exception as exc:  # pragma: no cover - depends on optional stack
        warnings.warn(f"SHAP analysis failed: {exc}")
        mean_abs_shap = pd.Series(model.feature_importances_,
                                  index=X.columns, name="mean_abs_shap")

    ranking = pd.DataFrame({
        "feature": X.columns,
        "pearson_r": pearson.reindex(X.columns).values,
        "mean_abs_shap": mean_abs_shap.reindex(X.columns).values,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Markdown table
    md_lines = [
        "# Stage 1 feature ranking",
        "",
        "| Rank | Feature | Pearson r | mean\\|SHAP\\| |",
        "|---:|:---|---:|---:|",
    ]
    for i, row in ranking.iterrows():
        md_lines.append(
            f"| {i + 1} | `{row['feature']}` | {row['pearson_r']:+.3f} | {row['mean_abs_shap']:.4f} |"
        )
    (reports / "feature_ranking.md").write_text("\n".join(md_lines))

    return ranking


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class Stage1Result:
    model: object
    features: list[str]
    metrics_train: RegressionMetrics
    metrics_test: RegressionMetrics
    metrics_persistence: RegressionMetrics
    metrics_climatology: RegressionMetrics
    cv_metrics: list[RegressionMetrics]
    y_test: pd.Series
    pred_test: pd.Series
    pred_test_persistence: pd.Series
    pred_test_climatology: pd.Series
    mode: str
    beats_persistence: bool


def train_stage1(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    mode: Literal["nowcast", "forecast"] = "forecast",
) -> Stage1Result:
    n = len(X)
    if n < 50:
        raise ValueError(f"Not enough samples for Stage 1 (got {n}).")

    split_at = int(n * (1 - config.model.test_split))
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    # Baselines
    persistence = PersistenceForecaster().fit(X_train, y_train)
    climatology = ClimatologyForecaster().fit(X_train, y_train)
    pred_pers = persistence.predict(X_test)
    pred_clim = climatology.predict(X_test)
    m_pers = compute_metrics(y_test.to_numpy(), pred_pers)
    m_clim = compute_metrics(y_test.to_numpy(), pred_clim)

    # Optional: Optuna hyperparameter tuning before final training
    tuned_params: dict = {}
    if (config.model.stage1_algo.lower() == "xgboost"
            and getattr(config.model, "tune_xgboost", False)):
        n_trials = int(getattr(config.model, "tune_n_trials", 50))
        tuned_params = tune_xgboost(X_train, y_train, config, n_trials=n_trials)

    # ML with time-series CV
    model = _make_regressor(config.model.stage1_algo, config.model.seed,
                            params=tuned_params or None)
    cv_metrics: list[RegressionMetrics] = []
    tscv = TimeSeriesSplit(n_splits=config.model.cv_folds)
    for fold, (tr, va) in enumerate(tscv.split(X_train), start=1):
        model.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred_va = model.predict(X_train.iloc[va])
        m = compute_metrics(y_train.iloc[va].to_numpy(), pred_va)
        cv_metrics.append(m)
        logger.info("CV fold %d: MAE=%.3f RMSE=%.3f R2=%.3f", fold, m.mae, m.rmse, m.r2)

    # Final fit on all training data
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    m_train = compute_metrics(y_train.to_numpy(), pred_train)
    m_test = compute_metrics(y_test.to_numpy(), pred_test)

    beats = m_test.mae < m_pers.mae
    if not beats:
        logger.warning(
            "Stage 1 model does NOT beat persistence on test MAE (%.3f vs %.3f). "
            "Report will flag this.", m_test.mae, m_pers.mae,
        )

    return Stage1Result(
        model=model,
        features=list(X.columns),
        metrics_train=m_train,
        metrics_test=m_test,
        metrics_persistence=m_pers,
        metrics_climatology=m_clim,
        cv_metrics=cv_metrics,
        y_test=y_test,
        pred_test=pd.Series(pred_test, index=y_test.index, name="pred_ws"),
        pred_test_persistence=pd.Series(pred_pers, index=y_test.index, name="pers"),
        pred_test_climatology=pd.Series(pred_clim, index=y_test.index, name="clim"),
        mode=mode,
        beats_persistence=beats,
    )


# ---------------------------------------------------------------------------
# Persistence (of the model itself)
# ---------------------------------------------------------------------------

def _is_darts_model(model: object) -> bool:
    return type(model).__module__.startswith("darts.")


def save_stage1(result: Stage1Result, path: str | Path, config: Config) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    payload: dict = {
        "model": result.model,
        "features": result.features,
        "mode": result.mode,
        "config_raw": config.raw,
    }
    if _is_darts_model(result.model):
        # darts' __getstate__ drops the LightningModule and Trainer, so a plain
        # joblib.dump would write a model with no weights — and fail silently.
        # Use darts' own serializer and keep a sidecar reference.
        sidecar = p.with_suffix(".darts.pt")
        result.model.save(str(sidecar))
        payload["model"] = None
        payload["darts_model_file"] = sidecar.name
        payload["darts_model_class"] = (
            f"{type(result.model).__module__}:{type(result.model).__qualname__}"
        )
    joblib.dump(payload, p)
    return p


def load_stage1(path: str | Path) -> dict:
    p = Path(path)
    payload = joblib.load(p)
    sidecar = payload.get("darts_model_file")
    if sidecar:
        import importlib
        module_name, qualname = payload["darts_model_class"].split(":")
        cls = getattr(importlib.import_module(module_name), qualname)
        payload["model"] = cls.load(str(p.parent / sidecar))
    return payload
