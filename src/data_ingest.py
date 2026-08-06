"""Data ingestion for ERA5 (CDS) and NASA POWER.

Both fetchers return the same tidy hourly DataFrame with a UTC-aware
DatetimeIndex and columns:

    wind_speed_10m, wind_speed_100m,
    wind_dir_10m, wind_dir_100m,
    temperature_c, pressure_pa, roughness

`roughness` is NaN for NASA POWER (not provided by the API).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .utils import Config, SiteConfig, ensure_dir, get_logger

logger = get_logger(__name__)

TIDY_COLUMNS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_10m",
    "wind_dir_100m",
    "temperature_c",
    "pressure_pa",
    "roughness",
]


def _uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u,v (east/north components in m/s) -> speed, meteorological direction (deg)."""
    speed = np.sqrt(u * u + v * v)
    # meteorological convention: direction FROM which wind blows
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


def _era5_cache_path(cache_dir: Path, site: SiteConfig, start: str, end: str) -> Path:
    return cache_dir / f"era5_{site.name}_{start}_{end}.nc"


def _nasa_cache_path(cache_dir: Path, site: SiteConfig, start: str, end: str) -> Path:
    return cache_dir / f"nasa_power_{site.name}_{start}_{end}.json"


def _fetch_era5(site: SiteConfig, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    import cdsapi  # imported lazily so tests don't require it

    ensure_dir(cache_dir)
    nc_path = _era5_cache_path(cache_dir, site, start, end)
    if not nc_path.exists():
        logger.info("Downloading ERA5 to %s (this can be slow on first run)", nc_path)
        client = cdsapi.Client()
        pad = 0.1
        area = [site.latitude + pad, site.longitude - pad,
                site.latitude - pad, site.longitude + pad]  # N, W, S, E
        request = {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "2m_temperature",
                "surface_pressure",
                "forecast_surface_roughness",
            ],
            "year": _year_range(start, end),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "format": "netcdf",
        }
        client.retrieve("reanalysis-era5-single-levels", request, str(nc_path))
    return _load_era5_netcdf(nc_path, site, start, end)


def _year_range(start: str, end: str) -> list[str]:
    y0 = pd.Timestamp(start).year
    y1 = pd.Timestamp(end).year
    return [str(y) for y in range(y0, y1 + 1)]


def _load_era5_netcdf(path: Path, site: SiteConfig, start: str, end: str) -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(path)
    # Pick the nearest grid cell to the site
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    ds = ds.sel({lat_name: site.latitude, lon_name: site.longitude}, method="nearest")

    time_name = "time" if "time" in ds.coords else "valid_time"
    times = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values), tz="UTC")

    def _arr(name: str) -> np.ndarray | None:
        if name not in ds.variables:
            return None
        return np.asarray(ds[name].values).squeeze()

    u10, v10 = _arr("u10"), _arr("v10")
    u100, v100 = _arr("u100"), _arr("v100")
    t2m = _arr("t2m")
    sp = _arr("sp")
    fsr = _arr("fsr")

    ws10, dir10 = _uv_to_speed_dir(u10, v10)
    ws100, dir100 = _uv_to_speed_dir(u100, v100)

    df = pd.DataFrame(
        {
            "wind_speed_10m": ws10,
            "wind_speed_100m": ws100,
            "wind_dir_10m": dir10,
            "wind_dir_100m": dir100,
            "temperature_c": t2m - 273.15,
            "pressure_pa": sp,
            "roughness": fsr if fsr is not None else np.nan,
        },
        index=times,
    )
    df.index.name = "timestamp"
    df = df.loc[start:end]
    return df[TIDY_COLUMNS]


def _fetch_nasa_power(site: SiteConfig, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    ensure_dir(cache_dir)
    cache = _nasa_cache_path(cache_dir, site, start, end)
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = _download_nasa_power(site, start, end)
        cache.write_text(json.dumps(payload))
    return _parse_nasa_power(payload)


def _download_nasa_power(site: SiteConfig, start: str, end: str) -> dict[str, Any]:
    logger.info("Downloading NASA POWER for %s (%s..%s)", site.name, start, end)
    params = {
        "parameters": "WS10M,WS50M,WD10M,WD50M,T2M,PS",
        "community": "RE",
        "longitude": site.longitude,
        "latitude": site.latitude,
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _parse_nasa_power(payload: dict[str, Any]) -> pd.DataFrame:
    params = payload["properties"]["parameter"]

    def _series(key: str) -> pd.Series:
        raw = params[key]
        idx = pd.to_datetime(list(raw.keys()), format="%Y%m%d%H", utc=True)
        s = pd.Series(list(raw.values()), index=idx, dtype=float)
        return s.replace(-999.0, np.nan)

    ws10 = _series("WS10M")
    ws50 = _series("WS50M")
    wd10 = _series("WD10M")
    wd50 = _series("WD50M")
    t2m = _series("T2M")
    ps_kpa = _series("PS")

    df = pd.DataFrame(
        {
            "wind_speed_10m": ws10,
            # NASA POWER exposes 50m rather than 100m; upstream feature code
            # extrapolates to hub height, so we treat WS50M as the upper level.
            "wind_speed_100m": ws50,
            "wind_dir_10m": wd10,
            "wind_dir_100m": wd50,
            "temperature_c": t2m,
            "pressure_pa": ps_kpa * 1000.0,
            "roughness": np.nan,
        }
    )
    df.index.name = "timestamp"
    return df[TIDY_COLUMNS]


def load_weather(config: Config) -> pd.DataFrame:
    """Fetch tidy hourly weather for the site, cached to disk."""
    cache_dir = Path(config.data.cache_dir)
    source = config.data.source.lower()
    if source == "era5":
        df = _fetch_era5(config.site, config.data.start, config.data.end, cache_dir)
    elif source == "nasa_power":
        df = _fetch_nasa_power(config.site, config.data.start, config.data.end, cache_dir)
    else:
        raise ValueError(f"Unknown data.source: {config.data.source!r}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    processed_dir = ensure_dir(config.paths.processed)
    out = processed_dir / f"{source}_{config.site.name}_{config.data.start}_{config.data.end}.parquet"
    df.to_parquet(out)
    logger.info("Wrote tidy weather: %s (%d rows)", out, len(df))
    return df
