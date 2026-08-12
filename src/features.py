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

import re
from typing import Iterable

import numpy as np
import pandas as pd
from windpowerlib.wind_speed import hellman, logarithmic_profile

from .utils import Config, get_logger

logger = get_logger(__name__)

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


def add_direction_encodings(
    df: pd.DataFrame,
    direction_cols: Iterable[str] = (
        "wind_dir_10m", "wind_dir_100m",
        "wind_dir_850hpa", "wind_dir_500hpa",
    ),
) -> pd.DataFrame:
    """Add sin/cos encodings for any direction columns present in df."""
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


_NWP_LEAD_RE = re.compile(r"_d(\d+)$")


def usable_nwp_leads(columns: Iterable[str], offset_hours: int) -> tuple[list[str], list[str]]:
    """Split `nwp_*` columns into (usable, too_short) for a given horizon.

    A column named `nwp_{model}_{var}_d{N}` holds, at valid time τ, the forecast
    for τ **issued 24·N hours before τ**. Row *t* receives the value at
    `τ = t + offset_hours`, so that value was issued at::

        issue_time = (t + offset_hours) - 24·N

    which is only knowable at time *t* when ``issue_time <= t``, i.e. when::

        24 · N >= offset_hours

    At the 24 h horizon every lead qualifies, which is why this never bit before.
    At 168 h a `_d1` column would have been issued at ``t + 144`` — six days into
    the future. Columns whose name does not match the `_d{N}` convention are kept
    untouched: the guard may only drop what it positively identified.
    """
    usable, too_short = [], []
    for col in columns:
        if not col.startswith("nwp_"):
            continue
        m = _NWP_LEAD_RE.search(col)
        if m is None:
            usable.append(col)          # unparseable — never silently dropped
            continue
        (too_short if 24 * int(m.group(1)) < offset_hours else usable).append(col)
    return usable, too_short


def add_nwp_features(df: pd.DataFrame, nwp_df: pd.DataFrame,
                     offset_hours: int) -> pd.DataFrame:
    """Join archived NWP forecasts onto the feature frame, correctly aligned.

    The two frames are indexed differently, and getting this wrong is the exact
    bug that invalidated every deep-model score in v2/v3 (docs/10):

    - `nwp_df` is indexed by **valid time** τ. `nwp_..._d1[τ]` is the forecast
      *for* τ, issued ~24 h *before* τ.
    - `df` is indexed by **feature time** t, and the target is
      `hub(t + offset_hours)`.

    So row *t* must receive the forecast whose valid time is `t + offset_hours`
    — which was issued around t and is therefore legitimately known at t. There
    is no lookahead: a `_d1` column at valid time t+24 was published at t.

    Reindexing on the shifted timestamps (rather than `.shift()`) makes the
    intent explicit and stays correct if either index has gaps.

    **Lead-day guard.** The "issued around t" step above only holds while
    ``24·N >= offset_hours``. Beyond a 24 h horizon it stops holding for the
    shorter leads, so those columns are dropped here rather than quietly handing
    the model the future. See :func:`usable_nwp_leads`. Raises if the horizon
    outruns every available lead — a weekly forecast needs `d7`, and refusing is
    the only honest response to not having it.
    """
    if nwp_df.empty:
        return df

    usable, too_short = usable_nwp_leads(nwp_df.columns, offset_hours)
    if too_short and not usable:
        raise ValueError(
            f"No NWP lead is long enough for a {offset_hours} h horizon: every "
            f"column would have been issued after the forecast time. Need "
            f"`_d{-(-offset_hours // 24)}` or longer; have "
            f"{sorted(too_short)}. Fetch more lead days rather than relaxing this."
        )
    if too_short:
        logger.info(
            "NWP lead guard: dropped %d of %d columns too short for a %d h "
            "horizon (need 24*N >= %d); kept %d.",
            len(too_short), len(too_short) + len(usable), offset_hours,
            offset_hours, len(usable),
        )

    target_times = df.index + pd.Timedelta(hours=offset_hours)
    aligned = nwp_df[usable].reindex(target_times)
    aligned.index = df.index
    return df.join(aligned)


def build_feature_frame(
    weather_df: pd.DataFrame,
    config: Config,
    mode: str = "nowcast",
    nwp_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_cols) for Stage 1.

    mode='nowcast'  : target is hub wind speed at the same timestep
                      (spatial downscaling from reanalysis).
    mode='forecast' : target is hub wind speed at t + horizon_hours.

    `nwp_df` (optional) holds archived day-ahead NWP forecasts indexed by valid
    time; see :func:`add_nwp_features` for the alignment contract.
    """
    df = weather_df.copy()
    # The `wind_speed_100m` column is only literally at 100 m for ERA5; NASA
    # POWER tops out at 50 m and stores it in the same column. Look the real
    # height up instead of assuming, or the log law silently no-ops and hub
    # wind is understated (~10.7% for NASA POWER). See SOURCE_WIND_HEIGHT_M.
    from .data_ingest import source_wind_height_m
    ref_height = source_wind_height_m(config.data.source)
    hub = extrapolate_hub_wind(df, config.site.hub_height_m,
                               reference_height_m=ref_height)
    df["wind_speed_hub"] = hub

    df = add_direction_encodings(df)
    df = add_calendar_features(df)
    df = add_lag_features(df, "wind_speed_hub", config.model.lags)
    df = add_rolling_features(df, "wind_speed_hub", config.model.rolling_windows)

    if mode == "nowcast":
        offset_hours = 0
        target = df["wind_speed_hub"]
    elif mode == "forecast":
        offset_hours = int(config.forecast.horizon_hours)
        target = df["wind_speed_hub"].shift(-offset_hours)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Must use the same offset as the target: the forecast valid when the
    # target is valid. See add_nwp_features for why this is load-bearing.
    if nwp_df is not None and not nwp_df.empty:
        df = add_nwp_features(df, nwp_df, offset_hours)

    excluded = {
        "wind_speed_hub",
        "wind_dir_10m", "wind_dir_100m",
        "wind_dir_850hpa", "wind_dir_500hpa",
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
