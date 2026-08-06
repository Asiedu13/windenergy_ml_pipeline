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

def _make_regressor(algo: str, seed: int):
    algo = algo.lower()
    if algo == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            tree_method="hist", n_jobs=-1,
        )
    if algo == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"Unknown stage1 algo: {algo!r}")


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

    # ML with time-series CV
    model = _make_regressor(config.model.stage1_algo, config.model.seed)
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

def save_stage1(result: Stage1Result, path: str | Path, config: Config) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(
        {
            "model": result.model,
            "features": result.features,
            "mode": result.mode,
            "config_raw": config.raw,
        },
        p,
    )
    return p


def load_stage1(path: str | Path) -> dict:
    return joblib.load(path)
