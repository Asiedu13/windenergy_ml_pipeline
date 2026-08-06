"""Feature engineering for Stage 1.

Everything the ML model sees is derived here from the tidy weather DataFrame:
- hub-height wind speed (log or Hellman extrapolation),
- direction sin/cos encodings,
- calendar features + sin/cos,
- lag features,
- rolling mean/std.

The orchestrator :func:`build_feature_frame` returns `(X, y, feature_cols)` and
drops warmup rows produced by lags / rolling windows.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from windpowerlib.wind_speed import hellman, logarithmic_profile

from .utils import Config

DEFAULT_ROUGHNESS_M = 0.15  # coastal/agricultural default when ERA5 fsr missing


def extrapolate_hub_wind(
    df: pd.DataFrame,
    hub_height_m: float,
    method: str = "logarithmic",
    reference_height_m: float = 100.0,
    reference_col: str = "wind_speed_100m",
    roughness_default: float = DEFAULT_ROUGHNESS_M,
) -> pd.Series:
    """Extrapolate reference wind speed to the turbine hub height.

    Uses `windpowerlib.wind_speed.logarithmic_profile` (needs roughness) or
    `hellman` (rougness optional). Keeps the same convention as Stage 2's
    ModelChain so numbers line up.
    """
    ws_ref = df[reference_col].to_numpy(dtype=float)
    roughness = df.get("roughness")
    if roughness is None or roughness.isna().all():
        roughness_arr = np.full_like(ws_ref, roughness_default)
    else:
        roughness_arr = roughness.fillna(roughness_default).to_numpy(dtype=float)

    if method == "logarithmic":
        hub = logarithmic_profile(
            wind_speed=ws_ref,
            wind_speed_height=reference_height_m,
            hub_height=hub_height_m,
            roughness_length=roughness_arr,
        )
    elif method == "hellman":
        hub = hellman(
            wind_speed=ws_ref,
            wind_speed_height=reference_height_m,
            hub_height=hub_height_m,
            roughness_length=roughness_arr,
        )
    else:
        raise ValueError(f"Unknown extrapolation method: {method!r}")
    return pd.Series(np.asarray(hub), index=df.index, name="wind_speed_hub")


def add_direction_encodings(df: pd.DataFrame,
                            direction_cols: Iterable[str] = ("wind_dir_10m", "wind_dir_100m")) -> pd.DataFrame:
    out = df.copy()
    for col in direction_cols:
        if col not in out.columns:
            continue
        rad = np.deg2rad(out[col].to_numpy(dtype=float))
        out[f"{col}_sin"] = np.sin(rad)
        out[f"{col}_cos"] = np.cos(rad)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    hour = idx.hour.to_numpy()
    month = idx.month.to_numpy()
    out["hour"] = hour
    out["month"] = month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return out


def add_lag_features(df: pd.DataFrame, target_col: str,
                     lags: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str,
                         windows: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        # `min_periods=w` so we don't leak early with tiny partial windows.
        # Shift by 1 so rolling stats never include the current timestep.
        base = out[target_col].shift(1)
        out[f"{target_col}_rollmean{w}"] = base.rolling(w, min_periods=w).mean()
        out[f"{target_col}_rollstd{w}"] = base.rolling(w, min_periods=w).std()
    return out


def build_feature_frame(
    weather_df: pd.DataFrame,
    config: Config,
    mode: str = "nowcast",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_cols) for Stage 1.

    mode='nowcast'  : target is hub wind speed at the same timestep
                      (spatial downscaling from reanalysis).
    mode='forecast' : target is hub wind speed at t + horizon_hours.
    """
    df = weather_df.copy()
    hub = extrapolate_hub_wind(df, config.site.hub_height_m)
    df["wind_speed_hub"] = hub

    df = add_direction_encodings(df)
    df = add_calendar_features(df)
    df = add_lag_features(df, "wind_speed_hub", config.model.lags)
    df = add_rolling_features(df, "wind_speed_hub", config.model.rolling_windows)

    if mode == "nowcast":
        target = df["wind_speed_hub"]
    elif mode == "forecast":
        target = df["wind_speed_hub"].shift(-int(config.forecast.horizon_hours))
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    excluded = {
        "wind_speed_hub", "wind_dir_10m", "wind_dir_100m",
        "hour", "month",
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    combined = df[feature_cols].copy()
    combined["_target"] = target
    # Drop feature columns that are entirely NaN (e.g. `roughness` from NASA POWER)
    all_nan = [c for c in feature_cols if combined[c].isna().all()]
    if all_nan:
        combined = combined.drop(columns=all_nan)
        feature_cols = [c for c in feature_cols if c not in all_nan]
    combined = combined.dropna()

    X = combined[feature_cols]
    y = combined["_target"].rename("wind_speed_hub_target")
    return X, y, feature_cols
