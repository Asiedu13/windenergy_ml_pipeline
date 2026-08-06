"""Three-model comparison + error-propagation analysis.

Variants (mirroring Donadio/Fang/Porté-Agel, Energies 2021):
- **physics_only**: reanalysis hub wind speed → power curve, no ML.
- **two_stage**:    Stage 1 predicted hub wind speed → power curve. Target variant.
- **direct**:       ML predicts power directly from the same features.

Reference "truth" for all three = physics-derived power from the *actual* observed
weather. This validates the forecasting step, not absolute physical yield —
Ghana has no operating utility SCADA to validate against yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from windpowerlib import WindTurbine

from .features import build_feature_frame
from .stage1_windspeed import (
    RegressionMetrics,
    Stage1Result,
    _make_regressor,
    compute_metrics,
)
from .stage2_powercurve import (
    TurbineSpec,
    air_density,
    windspeed_to_power,
)
from .utils import Config, ensure_dir, get_logger

logger = get_logger(__name__)


@dataclass
class VariantResult:
    name: str
    y_true_power: pd.Series
    y_pred_power: pd.Series
    metrics: RegressionMetrics
    nmae_pct: float
    nrmse_pct: float


def _normalized(metrics: RegressionMetrics, rated_w: float) -> tuple[float, float]:
    return (
        metrics.mae / rated_w * 100.0,
        metrics.rmse / rated_w * 100.0,
    )


def evaluate_variants(
    weather_df: pd.DataFrame,
    stage1: Stage1Result,
    turbine: WindTurbine,
    spec: TurbineSpec,
    config: Config,
) -> dict[str, VariantResult]:
    """Compute all three variants aligned on stage1's held-out test index."""
    test_idx = stage1.y_test.index

    # Density on the test window (from actual observed weather)
    density = pd.Series(
        air_density(
            weather_df.loc[test_idx, "temperature_c"].to_numpy(),
            weather_df.loc[test_idx, "pressure_pa"].to_numpy(),
        ),
        index=test_idx,
        name="density",
    )

    # Reference "truth": actual observed hub wind speed → power (with density)
    truth_ws = stage1.y_test  # actual hub wind speed on the test window
    y_true_power = windspeed_to_power(truth_ws, turbine, density=density)

    # Variant A — physics-only: use the raw (unshifted) hub wind speed from
    # weather at test timestamps (same as truth for nowcast; differs for
    # forecast mode because truth is shifted forward).
    # For "forecast" mode this measures what happens if you naively believed
    # persistence of the current reading.
    variants: dict[str, VariantResult] = {}

    physics_only_ws = truth_ws  # by construction reproduces the reference
    if stage1.mode == "forecast":
        # For forecast, "physics-only" is the ERA5/POWER hub wind at t (which
        # is what you'd know at prediction time), then curve-mapped.
        # We approximate this by using the persistence baseline series.
        physics_only_ws = stage1.pred_test_persistence
    physics_power = windspeed_to_power(physics_only_ws, turbine, density=density)
    m = compute_metrics(y_true_power.to_numpy(), physics_power.to_numpy())
    nmae, nrmse = _normalized(m, spec.nominal_power_w)
    variants["physics_only"] = VariantResult(
        "physics_only", y_true_power, physics_power, m, nmae, nrmse,
    )

    # Variant B — two-stage: Stage 1 predicted wind speed → curve
    two_stage_ws = stage1.pred_test
    two_stage_power = windspeed_to_power(two_stage_ws, turbine, density=density)
    m = compute_metrics(y_true_power.to_numpy(), two_stage_power.to_numpy())
    nmae, nrmse = _normalized(m, spec.nominal_power_w)
    variants["two_stage"] = VariantResult(
        "two_stage", y_true_power, two_stage_power, m, nmae, nrmse,
    )

    # Variant C — direct: retrain ML on the same features but predicting power
    X, _, _ = build_feature_frame(weather_df, config, mode=stage1.mode)
    # Align to Stage 1's train/test split by index
    X_aligned = X.loc[X.index.intersection(stage1.y_test.index.union(stage1.pred_test.index))]
    # Build direct target = physics-derived power from actual weather (density-adjusted)
    density_full = pd.Series(
        air_density(
            weather_df.loc[X.index, "temperature_c"].to_numpy(),
            weather_df.loc[X.index, "pressure_pa"].to_numpy(),
        ),
        index=X.index,
    )
    # For direct target, we need actual hub wind speed at t+h (forecast) or t (nowcast)
    hub_series = weather_df.loc[X.index, :]  # placeholder
    from .features import extrapolate_hub_wind
    hub_all = extrapolate_hub_wind(weather_df, config.site.hub_height_m)
    if stage1.mode == "forecast":
        target_ws = hub_all.shift(-int(config.forecast.horizon_hours)).loc[X.index]
    else:
        target_ws = hub_all.loc[X.index]
    y_power_all = windspeed_to_power(target_ws.dropna(), turbine,
                                     density=density_full.loc[target_ws.dropna().index])
    y_power_all = y_power_all.reindex(X.index).dropna()
    X_direct = X.loc[y_power_all.index]

    # Split direct at same fraction as Stage 1
    split = int(len(X_direct) * (1 - config.model.test_split))
    Xd_tr, Xd_te = X_direct.iloc[:split], X_direct.iloc[split:]
    yd_tr, yd_te = y_power_all.iloc[:split], y_power_all.iloc[split:]

    direct_model = _make_regressor(config.model.stage1_algo, config.model.seed)
    direct_model.fit(Xd_tr, yd_tr)
    direct_pred = pd.Series(direct_model.predict(Xd_te), index=Xd_te.index, name="direct_pred")

    # Align to y_true_power (test_idx used above)
    common = direct_pred.index.intersection(y_true_power.index)
    m = compute_metrics(y_true_power.loc[common].to_numpy(),
                        direct_pred.loc[common].to_numpy())
    nmae, nrmse = _normalized(m, spec.nominal_power_w)
    variants["direct"] = VariantResult(
        "direct", y_true_power.loc[common], direct_pred.loc[common], m, nmae, nrmse,
    )

    return variants


def error_propagation_plot(
    stage1: Stage1Result,
    two_stage_variant: VariantResult,
    output_path: str | Path,
    n_bins: int = 10,
) -> Path:
    """Bin by wind-speed prediction error and plot mean |power error| per bucket."""
    ws_err = (stage1.pred_test - stage1.y_test).abs()
    # Align power error to the two-stage variant series
    p_err = (two_stage_variant.y_pred_power - two_stage_variant.y_true_power).abs()
    common = ws_err.index.intersection(p_err.index)
    df = pd.DataFrame({"ws_err": ws_err.loc[common], "p_err": p_err.loc[common]})
    if df.empty:
        raise ValueError("No aligned samples for error propagation.")
    df["bucket"] = pd.qcut(df["ws_err"], q=min(n_bins, df["ws_err"].nunique()),
                           duplicates="drop")
    grouped = df.groupby("bucket", observed=True)["p_err"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    bucket_centers = [interval.mid for interval in grouped.index]
    ax.plot(bucket_centers, grouped.values / 1000.0, marker="o")
    ax.set_xlabel("Wind-speed prediction |error| (m/s)")
    ax.set_ylabel("Mean |power error| (kW)")
    ax.set_title("Error propagation: wind-speed error → power error")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(output_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def timeseries_plot(
    variant: VariantResult,
    output_path: str | Path,
    window_hours: int = 24 * 7,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    slice_ = slice(0, min(window_hours, len(variant.y_true_power)))
    y_true = variant.y_true_power.iloc[slice_] / 1000.0
    y_pred = variant.y_pred_power.iloc[slice_] / 1000.0
    ax.plot(y_true.index, y_true.values, label="reference (physics from obs)", lw=1.2)
    ax.plot(y_pred.index, y_pred.values, label=f"{variant.name} predicted",
            lw=1.2, alpha=0.85)
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Test-week power: {variant.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = Path(output_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def summarize_variants(variants: dict[str, VariantResult], spec: TurbineSpec) -> dict[str, Any]:
    rows = []
    for name, v in variants.items():
        rows.append({
            "variant": name,
            **v.metrics.as_dict(),
            "nMAE_%": v.nmae_pct,
            "nRMSE_%": v.nrmse_pct,
        })
    return {"rated_w": spec.nominal_power_w, "rows": rows}
