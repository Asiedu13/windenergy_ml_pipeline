from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class SiteConfig:
    name: str
    latitude: float
    longitude: float
    hub_height_m: float


@dataclass
class DataConfig:
    source: str
    start: str
    end: str
    cache_dir: str
    # Regional ERA5 (source: era5_regional) — one bounding box shared by every
    # site, as {north, west, south, east}. The point of it is cost: CDS charges
    # per field, not per grid cell, so one Ghana-wide request costs the same as
    # one two-cell request and 17 sites come out of a single download pass.
    # See docs/13_multi_site_extension.md §4.
    bbox: dict[str, float] = field(default_factory=dict)
    # Half-width of the extraction window around a site's cell, in cells.
    # 1 -> 3x3. Only used by era5_regional; the per-site path is unaffected.
    window: int = 1
    # Minimum land fraction for a site's selected ERA5 cell. Coastal sites whose
    # nearest cell lands offshore otherwise report an inflated resource, because
    # sea cells carry much lower roughness. See docs/13 §3.
    min_land_fraction: float = 0.5
    # Download unit for era5_regional: "month" or "year".
    #
    # ⚠️ **"month" is the default because it is the only size CDS actually
    # accepts for this bbox.** A yearly request was rejected with "cost limits
    # exceeded" (measured 2026-10-01): CDS's cost includes spatial extent, and at
    # 630 cells a year is far over the limit — 6 and 4 months are rejected too.
    # "year" is retained only for a smaller bbox or a future quota change; do not
    # set it for the Ghana box without re-probing. See docs/13 §4a.
    chunk: str = "month"


@dataclass
class TurbineConfig:
    name: str
    source: str = "oedb"


@dataclass
class ForecastConfig:
    horizon_hours: int
    timestep: str = "hourly"


@dataclass
class ModelConfig:
    stage1_algo: str = "xgboost"
    test_split: float = 0.2
    cv_folds: int = 5
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 6, 24])
    seed: int = 42
    # deep-model knobs (ignored for tree algos)
    deep: dict[str, Any] = field(default_factory=dict)
    # Optuna XGBoost sweep
    tune_xgboost: bool = False
    tune_n_trials: int = 50
    # Compare-all algo list — used when --compare is passed
    compare_algos: list[str] = field(default_factory=list)


@dataclass
class NWPConfig:
    """Operational NWP forecast features (Open-Meteo Previous Runs API).

    Disabled by default so the ERA5-only path stays the control. `lead_days`
    are `_previous_day{N}` offsets: N=1 is the forecast issued ~24 h before the
    valid time, which is what a 24 h horizon needs.
    """

    enabled: bool = False
    models: list[str] = field(
        default_factory=lambda: ["ecmwf_ifs025", "gfs_global", "icon_global"]
    )
    lead_days: list[int] = field(default_factory=lambda: [1, 2, 3])
    variables: list[str] = field(
        default_factory=lambda: [
            "wind_speed_100m",
            "wind_direction_100m",
            "wind_speed_10m",
            "temperature_2m",
            "surface_pressure",
            "wind_gusts_10m",
        ]
    )


@dataclass
class FeaturesConfig:
    """How the feature mask decides which columns are degenerate.

    ``name``  — suffix match on `_grad_ns`/`_std`. The default, and what every
                published figure used. Correct for the per-site ERA5 request,
                whose 1x2 patch really does make those columns degenerate.
    ``value`` — judge each column on the training data instead. The v6 regional
                3x3 patch revives all ten (measured: 5/5 `_grad_ns` non-zero,
                0/5 `_std` still collinear), which `name` discards anyway.

    Off by default because flipping it re-bases published results; see
    docs/13 section 3 and redo/09 section 9.3.
    """

    dead_columns: str = "name"


@dataclass
class PathsConfig:
    outputs: str = "outputs"
    processed: str = "data/processed"


@dataclass
class Config:
    site: SiteConfig
    data: DataConfig
    turbine: TurbineConfig
    forecast: ForecastConfig
    model: ModelConfig
    paths: PathsConfig
    nwp: NWPConfig = field(default_factory=NWPConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        return hashlib.sha256(
            yaml.safe_dump(self.raw, sort_keys=True).encode()
        ).hexdigest()[:12]


def load_config(path: str | os.PathLike) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        site=SiteConfig(**raw["site"]),
        data=DataConfig(**raw["data"]),
        turbine=TurbineConfig(**raw["turbine"]),
        forecast=ForecastConfig(**raw["forecast"]),
        model=ModelConfig(**raw["model"]),
        paths=PathsConfig(**raw.get("paths", {})),
        nwp=NWPConfig(**raw.get("nwp", {})),
        features=FeaturesConfig(**raw.get("features", {})),
        raw=raw,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "windpipe") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
