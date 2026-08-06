"""Tests for data ingestion — both fetchers mocked at the network boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_ingest import (
    TIDY_COLUMNS,
    _parse_nasa_power,
    _uv_to_speed_dir,
    load_weather,
)
from src.utils import (
    Config,
    DataConfig,
    ForecastConfig,
    ModelConfig,
    PathsConfig,
    SiteConfig,
    TurbineConfig,
)


def _make_config(tmp_path: Path, source: str) -> Config:
    return Config(
        site=SiteConfig(name="test", latitude=5.79, longitude=0.9, hub_height_m=100),
        data=DataConfig(source=source, start="2023-01-01", end="2023-01-02",
                        cache_dir=str(tmp_path / "raw")),
        turbine=TurbineConfig(name="E-101/3050"),
        forecast=ForecastConfig(horizon_hours=24),
        model=ModelConfig(),
        paths=PathsConfig(outputs=str(tmp_path / "out"),
                          processed=str(tmp_path / "proc")),
        raw={},
    )


def _fake_nasa_payload(hours: int = 48) -> dict:
    idx = pd.date_range("2023-01-01", periods=hours, freq="h", tz="UTC")

    def _vals(base: float, amp: float) -> dict:
        return {
            t.strftime("%Y%m%d%H"): float(base + amp * np.sin(i / 6))
            for i, t in enumerate(idx)
        }

    return {
        "properties": {
            "parameter": {
                "WS10M": _vals(5.0, 2.0),
                "WS50M": _vals(7.0, 2.5),
                "WD10M": _vals(180.0, 90.0),
                "WD50M": _vals(190.0, 80.0),
                "T2M": _vals(28.0, 3.0),
                "PS": _vals(101.3, 0.5),  # kPa
            }
        }
    }


def test_uv_to_speed_dir_basics():
    # pure eastward (u>0, v=0) → speed=u, dir=270 (wind FROM the west)
    speed, direction = _uv_to_speed_dir(np.array([3.0]), np.array([0.0]))
    assert speed[0] == pytest.approx(3.0)
    assert direction[0] == pytest.approx(270.0)
    # pure northward (v>0) → dir=180 (wind FROM the south)
    speed, direction = _uv_to_speed_dir(np.array([0.0]), np.array([4.0]))
    assert direction[0] == pytest.approx(180.0)


def test_parse_nasa_power_schema():
    payload = _fake_nasa_payload(24)
    df = _parse_nasa_power(payload)
    assert list(df.columns) == TIDY_COLUMNS
    assert df.index.tz is not None
    assert df["pressure_pa"].iloc[0] == pytest.approx(101_300, rel=0.02)
    assert df["roughness"].isna().all()


def test_load_weather_nasa_power_end_to_end(tmp_path):
    cfg = _make_config(tmp_path, "nasa_power")
    payload = _fake_nasa_payload(48)

    with patch("src.data_ingest.requests.get") as mocked:
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        mocked.return_value = resp
        df = load_weather(cfg)

    # Schema, ordering, and no-duplicate index
    assert list(df.columns) == TIDY_COLUMNS
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    # Cache written
    cache = Path(cfg.data.cache_dir) / f"nasa_power_test_{cfg.data.start}_{cfg.data.end}.json"
    assert cache.exists()
    # Parquet written
    proc = Path(cfg.paths.processed) / f"nasa_power_test_{cfg.data.start}_{cfg.data.end}.parquet"
    assert proc.exists()


def test_load_weather_uses_cache_on_second_call(tmp_path):
    cfg = _make_config(tmp_path, "nasa_power")
    payload = _fake_nasa_payload(24)
    cache = Path(cfg.data.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"nasa_power_test_{cfg.data.start}_{cfg.data.end}.json").write_text(json.dumps(payload))

    with patch("src.data_ingest.requests.get") as mocked:
        df = load_weather(cfg)
        mocked.assert_not_called()

    assert len(df) == 24


def test_load_weather_unknown_source(tmp_path):
    cfg = _make_config(tmp_path, "bogus")
    with pytest.raises(ValueError, match="Unknown data.source"):
        load_weather(cfg)
