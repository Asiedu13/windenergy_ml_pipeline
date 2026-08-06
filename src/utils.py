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
