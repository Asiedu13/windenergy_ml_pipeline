"""Stage 2 — deterministic power-curve simulator.

Wraps windpowerlib so that a wind-speed series (from Stage 1 or from reanalysis)
can be converted to turbine power, with an ideal-gas air-density correction that
matters in Ghana's heat.

This module must remain usable standalone — nothing here depends on Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from windpowerlib import ModelChain, WindTurbine
from windpowerlib.power_output import power_curve as _wpl_power_curve

R_SPECIFIC_DRY_AIR = 287.058  # J/(kg*K)


@dataclass
class TurbineSpec:
    """Compact metadata for the report / sanity checks."""

    name: str
    hub_height_m: float
    rotor_diameter_m: float
    nominal_power_w: float
    cut_in_ms: float
    rated_ms: float
    cut_out_ms: float


def load_turbine(name: str, hub_height_m: float) -> WindTurbine:
    """Load a turbine from the oedb library."""
    return WindTurbine(hub_height=hub_height_m, turbine_type=name)


def turbine_spec(turbine: WindTurbine, name: str) -> TurbineSpec:
    pc = turbine.power_curve
    ws = pc["wind_speed"].to_numpy()
    val = pc["value"].to_numpy()
    nonzero = np.flatnonzero(val > 0)
    if nonzero.size == 0:
        raise ValueError(f"Turbine {name!r} has an all-zero power curve.")
    cut_in = float(ws[nonzero[0]])
    peak = float(val.max())
    rated = float(ws[np.argmax(val >= peak)])
    trailing_zeros = np.flatnonzero((val == 0) & (ws > rated))
    cut_out = float(ws[trailing_zeros[0]]) if trailing_zeros.size else float(ws[-1])
    return TurbineSpec(
        name=name,
        hub_height_m=float(turbine.hub_height),
        rotor_diameter_m=float(turbine.rotor_diameter),
        nominal_power_w=float(turbine.nominal_power),
        cut_in_ms=cut_in,
        rated_ms=rated,
        cut_out_ms=cut_out,
    )


def air_density(temperature_c: pd.Series | np.ndarray | float,
                pressure_pa: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Ideal-gas density of dry air (kg/m^3).

    Ghana's heat lowers density and measurably reduces turbine output — do not
    skip this correction when temperature and pressure are available.
    """
    temperature_k = np.asarray(temperature_c) + 273.15
    return np.asarray(pressure_pa) / (R_SPECIFIC_DRY_AIR * temperature_k)


def windspeed_to_power(
    ws_series: pd.Series,
    turbine: WindTurbine,
    density: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """Convert a hub-height wind-speed series to power (W).

    When `density` is provided, applies windpowerlib's density correction
    (cube-root scaling on the effective wind speed).
    """
    pc = turbine.power_curve
    ws_arr = np.asarray(ws_series, dtype=float)
    density_arr = None if density is None else np.asarray(density, dtype=float)
    power = _wpl_power_curve(
        wind_speed=ws_arr,
        power_curve_wind_speeds=pc["wind_speed"].to_numpy(),
        power_curve_values=pc["value"].to_numpy(),
        density=density_arr,
        density_correction=density_arr is not None,
    )
    return pd.Series(np.asarray(power), index=ws_series.index, name="power_w")


def model_chain_from_weather(
    weather_df: pd.DataFrame,
    turbine: WindTurbine,
    reference_height_ws: float = 100.0,
    reference_height_t: float = 2.0,
    reference_height_p: float = 0.0,
    roughness_default: float = 0.15,
) -> pd.Series:
    """Physics-only path: raw weather DataFrame → power (W).

    Expects columns from :func:`src.data_ingest.load_weather` in tidy form:
    `wind_speed_100m`, `temperature_c`, `pressure_pa`, and optional `roughness`.
    Builds the MultiIndex frame that windpowerlib.ModelChain expects.
    """
    required = {"wind_speed_100m", "temperature_c", "pressure_pa"}
    missing = required - set(weather_df.columns)
    if missing:
        raise ValueError(f"weather_df missing required columns: {sorted(missing)}")

    temperature_k = weather_df["temperature_c"] + 273.15
    roughness = weather_df.get("roughness")
    if roughness is None or roughness.isna().all():
        roughness = pd.Series(roughness_default, index=weather_df.index)
    roughness = roughness.fillna(roughness_default)

    frame = pd.DataFrame(
        {
            ("wind_speed", reference_height_ws): weather_df["wind_speed_100m"].to_numpy(),
            ("temperature", reference_height_t): temperature_k.to_numpy(),
            ("pressure", reference_height_p): weather_df["pressure_pa"].to_numpy(),
            ("roughness_length", 0): roughness.to_numpy(),
        },
        index=weather_df.index,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    mc = ModelChain(turbine).run_model(frame)
    return pd.Series(np.asarray(mc.power_output), index=weather_df.index, name="power_w")
