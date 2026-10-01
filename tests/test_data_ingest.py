"""Tests for data ingestion — both fetchers mocked at the network boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_ingest import (
    NASA_POWER_TIME_OFFSET_H,
    TIDY_COLUMNS,
    _parse_nasa_power,
    _parse_open_meteo,
    _uv_to_speed_dir,
    load_nwp_forecasts,
    load_weather,
    nwp_column,
)
from src.utils import (
    Config,
    DataConfig,
    ForecastConfig,
    ModelConfig,
    NWPConfig,
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


# ---------------------------------------------------------------------------
# Open-Meteo previous-runs (archived NWP forecasts)
# ---------------------------------------------------------------------------


def _fake_open_meteo_payload(hours: int = 48) -> dict:
    idx = pd.date_range("2024-03-01", periods=hours, freq="h", tz="UTC")
    t = np.arange(hours)
    return {
        "hourly": {
            "time": [x.strftime("%Y-%m-%dT%H:%M") for x in idx],
            "wind_speed_100m_previous_day1": (8 + 2 * np.sin(t / 6)).tolist(),
            "wind_speed_100m_previous_day2": (8 + 2 * np.sin(t / 6 + 0.3)).tolist(),
            "temperature_2m_previous_day1": (28 + 3 * np.sin(t / 12)).tolist(),
        }
    }


def test_nasa_power_timestamps_shifted_onto_era5_clock():
    """NASA POWER hourly stamps run 1 h late; the parser must correct that.

    MERRA-2 hourly values are hour-averaged and labelled at the end of the interval.
    Measured over 8,760 overlapping hours: ERA5(t) matches NASA(t+1), and every NWP
    model's cross-correlation peak sits exactly +1 h against NASA versus ERA5. Without
    this correction the alignment gate in src.validate_nwp fails.
    """
    payload = _fake_nasa_payload(24)
    raw_first = pd.to_datetime(
        sorted(payload["properties"]["parameter"]["WS10M"].keys())[0],
        format="%Y%m%d%H", utc=True,
    )
    df = _parse_nasa_power(payload)
    assert df.index[0] == raw_first + pd.Timedelta(hours=NASA_POWER_TIME_OFFSET_H)
    assert NASA_POWER_TIME_OFFSET_H == -1
    # values themselves must be untouched — this is a relabelling, not a resample
    assert len(df) == 24
    assert df["wind_speed_10m"].notna().all()


def test_parse_open_meteo_filters_to_requested_leads():
    """A cache holds whatever the *widest* previous request asked for.

    Once the weekly work widened these caches to leads 1-7, a config asking for
    `lead_days: [1, 2, 3]` was silently handed all seven — 126 columns instead of
    54 — which re-ran the 24 h bakeoff on an undocumented feature set and moved
    every published MAE. The parser must honour the request, not the file.
    """
    payload = _fake_open_meteo_payload(12)
    payload["hourly"]["wind_speed_100m_previous_day7"] = [9.9] * 12

    served_all = _parse_open_meteo(payload, "ecmwf_ifs025")
    assert "nwp_ecmwf_ifs025_wind_speed_100m_d7" in served_all.columns

    filtered = _parse_open_meteo(payload, "ecmwf_ifs025", lead_days=[1, 2])
    assert "nwp_ecmwf_ifs025_wind_speed_100m_d7" not in filtered.columns
    assert "nwp_ecmwf_ifs025_wind_speed_100m_d1" in filtered.columns
    assert "nwp_ecmwf_ifs025_wind_speed_100m_d2" in filtered.columns


def test_parse_open_meteo_filters_to_requested_variables():
    payload = _fake_open_meteo_payload(12)
    filtered = _parse_open_meteo(payload, "ecmwf_ifs025",
                                 variables=["wind_speed_100m"])
    assert all("temperature_2m" not in c for c in filtered.columns)
    assert any("wind_speed_100m" in c for c in filtered.columns)


def test_parse_open_meteo_unfiltered_by_default():
    """Absent an explicit request the parser stays permissive, so existing
    callers that pass no lead list are unaffected."""
    payload = _fake_open_meteo_payload(6)
    assert len(_parse_open_meteo(payload, "gfs_global").columns) == 3


def test_parse_open_meteo_column_naming_and_index():
    df = _parse_open_meteo(_fake_open_meteo_payload(24), "ecmwf_ifs025")
    assert list(df.columns) == [
        "nwp_ecmwf_ifs025_wind_speed_100m_d1",
        "nwp_ecmwf_ifs025_wind_speed_100m_d2",
        "nwp_ecmwf_ifs025_temperature_2m_d1",
    ]
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert len(df) == 24
    assert not df.isna().any().any()


def test_nwp_column_helper_roundtrips():
    # Variable names contain underscores and digits; the parser splits on the
    # "_previous_day" marker, not on underscores, so the two must agree.
    assert (nwp_column("gfs_global", "wind_speed_100m", 1)
            == "nwp_gfs_global_wind_speed_100m_d1")
    df = _parse_open_meteo(_fake_open_meteo_payload(6), "gfs_global")
    assert nwp_column("gfs_global", "wind_speed_100m", 1) in df.columns


def test_load_nwp_forecasts_disabled_returns_empty(tmp_path):
    cfg = _make_config(tmp_path, "era5")
    assert cfg.nwp.enabled is False
    with patch("src.data_ingest.requests.get") as mocked:
        assert load_nwp_forecasts(cfg).empty
        mocked.assert_not_called()


def test_load_nwp_forecasts_caches_per_model_and_year(tmp_path):
    cfg = _make_config(tmp_path, "era5")
    cfg.data.start, cfg.data.end = "2024-03-01", "2024-03-02"
    cfg.nwp = NWPConfig(enabled=True, models=["ecmwf_ifs025", "gfs_global"],
                        lead_days=[1], variables=["wind_speed_100m"])

    resp = MagicMock()
    resp.json.return_value = _fake_open_meteo_payload(48)
    resp.raise_for_status.return_value = None

    with patch("src.data_ingest.requests.get", return_value=resp) as mocked:
        df = load_nwp_forecasts(cfg)
        assert mocked.call_count == 2  # one request per model, one year

    # Both models present, side by side on a shared valid-time index.
    assert "nwp_ecmwf_ifs025_wind_speed_100m_d1" in df.columns
    assert "nwp_gfs_global_wind_speed_100m_d1" in df.columns
    for model in ("ecmwf_ifs025", "gfs_global"):
        assert (Path(cfg.data.cache_dir) / f"openmeteo_test_{model}_2024.json").exists()

    # Second call is served entirely from cache.
    with patch("src.data_ingest.requests.get") as mocked:
        again = load_nwp_forecasts(cfg)
        mocked.assert_not_called()
    pd.testing.assert_frame_equal(df, again)
