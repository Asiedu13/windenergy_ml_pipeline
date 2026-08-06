"""End-to-end smoke test with mocked ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


def _fake_nasa_payload(hours: int = 24 * 120) -> dict:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=hours, freq="h", tz="UTC")
    t = np.arange(hours)
    diurnal = 2.0 * np.sin(2 * np.pi * t / 24)
    seasonal = 1.0 * np.sin(2 * np.pi * t / (24 * 30))
    noise = np.zeros(hours)
    for i in range(1, hours):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.4)
    wind50 = 8.0 + diurnal + seasonal + noise
    wind10 = wind50 * 0.7

    def _dict(arr):
        return {t_.strftime("%Y%m%d%H"): float(v) for t_, v in zip(idx, arr)}

    return {
        "properties": {
            "parameter": {
                "WS10M": _dict(wind10),
                "WS50M": _dict(wind50),
                "WD10M": _dict((180 + 30 * np.sin(t / 12)) % 360),
                "WD50M": _dict((200 + 30 * np.sin(t / 12)) % 360),
                "T2M": _dict(28 + 3 * np.sin(2 * np.pi * t / 24)),
                "PS": _dict(np.full(hours, 101.3)),
            }
        }
    }


def _write_config(tmp_path: Path, cache_dir: Path, out_dir: Path,
                  processed_dir: Path) -> Path:
    cfg = {
        "site": {"name": "smoke", "latitude": 5.79, "longitude": 0.9, "hub_height_m": 100},
        "data": {"source": "nasa_power", "start": "2023-01-01",
                 "end": "2023-05-01", "cache_dir": str(cache_dir)},
        "turbine": {"name": "E-101/3050", "source": "oedb"},
        "forecast": {"horizon_hours": 6, "timestep": "hourly"},
        "model": {"stage1_algo": "xgboost", "test_split": 0.2, "cv_folds": 3,
                  "lags": [1, 2, 3, 6], "rolling_windows": [3, 6], "seed": 42},
        "paths": {"outputs": str(out_dir), "processed": str(processed_dir)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_pipeline_end_to_end(tmp_path):
    from src.pipeline import run

    cache = tmp_path / "raw"
    out = tmp_path / "outputs"
    proc = tmp_path / "proc"
    cfg_path = _write_config(tmp_path, cache, out, proc)

    # Pre-populate the NASA POWER cache so ingestion doesn't hit the network
    cache.mkdir(parents=True, exist_ok=True)
    payload = _fake_nasa_payload(hours=24 * 120)
    (cache / "nasa_power_smoke_2023-01-01_2023-05-01.json").write_text(json.dumps(payload))

    report_path = run(str(cfg_path), mode="forecast")

    assert report_path.exists()
    body = report_path.read_text()
    for variant in ("physics_only", "two_stage", "direct"):
        assert variant in body
    # Model persisted
    assert (out / "models" / "stage1.pkl").exists()
    # Figures produced
    for fig in ("pearson_heatmap", "error_propagation",
                "timeseries_two_stage", "timeseries_physics_only",
                "timeseries_direct"):
        assert (out / "figures" / f"{fig}.png").exists(), f"missing {fig}.png"
