"""Monthly-scale wind resource: climatology, exceedance, and the outlook ceiling.

Why this module is not "the 24 h model with a bigger horizon"
-------------------------------------------------------------
It cannot be. Open-Meteo previous-runs stop at lead day 7 (probed live
2026-08-12: nothing before 2024-03, nothing past d7), and deterministic point
skill at 30 days is zero for physical reasons rather than data reasons. A model
trained to predict hour-by-hour power 30 days out would return a confident
artifact with no skill — the failure mode `redo/00_mistake_register.md` exists
to prevent.

So "monthly" is split into two products with very different prospects:

**M2 — resource climatology (this module's main output).** Per-calendar-month
P50/P75/P90 exceedance energy, interannual spread, and the mean diurnal profile,
from the ERA5 months on disk. Not a forecast; it is what wind resource
assessment actually delivers for siting and PPA sizing, and its value does not
depend on beating a baseline.

**M1 — 30-day energy outlook.** Weekly model for days 1-7, climatology for the
rest, summed to a monthly total. See :mod:`src.monthly_outlook`.

What the measurements say about the design
------------------------------------------
Measured here over 149 complete months (2010-01..2025-12, density-corrected):

* The **seasonal cycle is 81.9% of monthly-CF variance**; the interannual
  anomaly is only 18.1%. The calendar is the model.
* **Lag-1 anomaly autocorrelation is 0.255 and dead by lag 2** (0.085). Anomaly
  persistence scores *worse* than climatology (MAE 0.0439 vs 0.0369), so
  persistence must not be used as a monthly baseline at all.
* Leave-one-year-out climatology MAE is **0.0403 CF**; the in-sample figure is
  0.0371. Quoting the in-sample number as the baseline would silently flatter
  every model by **8.6%**, so :func:`loyo_climatology` is the only baseline this
  module exposes.

Exceedance convention
---------------------
P90 is the value **exceeded 90% of the time** — a *low*, conservative number,
i.e. the 10th percentile. P50 is the median. Getting this backwards inverts the
entire product, so it is encoded once in :data:`EXCEEDANCE_QUANTILE` and tested.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_ingest import _load_era5_single_month
from .stage2_powercurve import air_density, load_turbine, windspeed_to_power
from .utils import Config, ensure_dir, get_logger, load_config

logger = get_logger(__name__)

#: A month is usable only if this fraction of its hours are present. ERA5 months
#: are whole-month files, so in practice this rejects a truncated final month
#: (e.g. the current, in-progress one) rather than scattered gaps.
MIN_COMPLETENESS = 0.95

#: Exceedance level -> the quantile that produces it. PXX is the value exceeded
#: XX% of the time, so it is the (1 - XX/100) quantile. P90 = q0.10 = a low,
#: conservative estimate. See the module docstring.
EXCEEDANCE_QUANTILE = {50: 0.50, 75: 0.25, 90: 0.10}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def cached_era5_months(cache_dir: Path, site_name: str) -> list[Path]:
    """Every cached core-variable ERA5 month, ascending.

    Reads the cache directly rather than going through
    :func:`src.data_ingest.load_weather`, which would try to *download* any
    missing month and aborts when a core month is absent. The archive has a
    41-month hole (2016-08..2019-12) that is deliberately not being filled, so
    the monthly work must tolerate gaps rather than trip over them.
    """
    return sorted(cache_dir.glob(f"era5_{site_name}_*.nc"))


def _hourly_cache_key(paths: list[Path]) -> str:
    """Fingerprint of the source months: count plus the newest modification time.

    Reading 150+ NetCDFs takes minutes, and the monthly work re-runs every time
    another month lands. This key changes exactly when the inputs do, so a new
    download invalidates the cache and nothing else does.
    """
    newest = max(p.stat().st_mtime_ns for p in paths)
    return f"{len(paths)}_{newest}"


def load_cached_weather(config: Config, use_cache: bool = True) -> pd.DataFrame:
    """Hourly ERA5 from the cache, without triggering any download.

    :func:`src.data_ingest.load_weather` would try to fetch the archive's
    deliberate 41-month hole and abort. This reads whatever is on disk and
    tolerates the gap.

    Cached to parquet: reading 150+ NetCDFs takes minutes, and both the
    climatology and the outlook need this same frame.
    """
    cache_dir = Path(config.data.cache_dir)
    paths = cached_era5_months(cache_dir, config.site.name)
    if not paths:
        raise FileNotFoundError(
            f"No cached ERA5 months in {cache_dir} for site {config.site.name!r}."
        )

    processed = ensure_dir(Path(config.paths.processed))
    cache_file = processed / (
        f"monthly_weather_{config.site.name}_{_hourly_cache_key(paths)}.parquet")
    if use_cache and cache_file.exists():
        logger.info("reusing hourly weather cache %s", cache_file.name)
        return pd.read_parquet(cache_file)

    frames = []
    for path in paths:
        try:
            frames.append(_load_era5_single_month(path, config.site))
        except Exception as exc:  # noqa: BLE001 — one bad month must not kill the run
            logger.warning("skipping unreadable month %s: %s", path.name, exc)
    if not frames:
        raise ValueError(f"None of the {len(paths)} cached months could be read.")

    hourly = pd.concat(frames).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    logger.info("loaded %d months -> %d hourly rows (%s .. %s)",
                len(frames), len(hourly), hourly.index[0], hourly.index[-1])

    if use_cache:
        # Stale keys describe older month-sets and are never asked for again.
        for old in processed.glob(f"monthly_weather_{config.site.name}_*.parquet"):
            old.unlink()
        hourly.to_parquet(cache_file)
        logger.info("cached hourly weather -> %s", cache_file.name)
    return hourly


def capacity_factor_from_weather(weather: pd.DataFrame, config: Config) -> pd.Series:
    """Hourly capacity factor for the configured turbine.

    Applies the density correction — Ghana's heat lowers air density enough to
    matter. Measured on the weekly test fold, omitting it raises the capacity
    factor by +4.1% (0.2327 -> 0.2422), which is one of the two reasons this
    project briefly carried two different capacity factors. The other was an
    unlabelled evaluation period; see `redo/results/10`.
    """
    ws = weather["wind_speed_100m"].dropna()
    turbine = load_turbine(config.turbine.name, float(config.site.hub_height_m))
    density = air_density(weather.loc[ws.index, "temperature_c"],
                          weather.loc[ws.index, "pressure_pa"])
    power_w = windspeed_to_power(ws, turbine, density=density)
    return (power_w / float(turbine.nominal_power)).rename("cf")


def load_hourly_capacity_factor(config: Config, use_cache: bool = True) -> pd.Series:
    """Hourly capacity factor from every cached ERA5 month."""
    return capacity_factor_from_weather(
        load_cached_weather(config, use_cache=use_cache), config)


# ---------------------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------------------

def monthly_frame(cf_hourly: pd.Series,
                  min_completeness: float = MIN_COMPLETENESS) -> pd.DataFrame:
    """Aggregate hourly capacity factor to complete calendar months.

    Returns a frame indexed by ``Period[M]`` with columns ``cf``, ``hours``,
    ``year``, ``month``. Incomplete months are dropped, not partially credited:
    a month missing its windiest week would otherwise enter the climatology as a
    genuine low year.
    """
    if cf_hourly.empty:
        return pd.DataFrame(columns=["cf", "hours", "year", "month"])

    # Drop the tz before periodising: pandas warns that Period has no timezone,
    # and Anloga is UTC+0 so there is nothing to preserve.
    index = cf_hourly.index
    period = (index.tz_localize(None) if index.tz is not None else index).to_period("M")
    grouped = cf_hourly.groupby(period)
    agg = pd.DataFrame({"cf": grouped.mean(), "hours": grouped.size()})
    agg.index.name = "month_period"

    expected = agg.index.days_in_month * 24
    complete = agg["hours"] >= min_completeness * expected
    dropped = (~complete).sum()
    if dropped:
        logger.info("dropped %d incomplete month(s): %s", dropped,
                    ", ".join(str(p) for p in agg.index[~complete]))
    agg = agg[complete].copy()
    agg["year"] = agg.index.year
    agg["month"] = agg.index.month
    return agg


def loyo_climatology(monthly: pd.DataFrame) -> pd.Series:
    """Leave-one-year-out month-of-year climatology.

    For each row, the mean of the *same calendar month in every other year*. The
    in-sample alternative includes the target month in its own baseline, which
    measured 0.0371 CF against this function's 0.0403 — an 8.6% flattery applied
    to every model compared against it.
    """
    out = pd.Series(index=monthly.index, dtype=float)
    for idx, row in monthly.iterrows():
        others = monthly[(monthly["month"] == row["month"])
                         & (monthly["year"] != row["year"])]
        out[idx] = others["cf"].mean() if len(others) else np.nan
    return out.rename("clim_loyo")


def variance_decomposition(monthly: pd.DataFrame) -> dict[str, float]:
    """How much of monthly-CF variance is the calendar, and how much is weather?

    The seasonal share being ~82% is the single fact that determines what a
    monthly model can be: only the remaining ~18% is available to predict.
    """
    clim_in_sample = monthly.groupby("month")["cf"].transform("mean")
    anomaly = monthly["cf"] - clim_in_sample
    total_var = monthly["cf"].var()
    seasonal_var = clim_in_sample.var()
    anomaly_var = anomaly.var()
    return {
        "n_months": float(len(monthly)),
        "mean_cf": float(monthly["cf"].mean()),
        "sd_cf": float(monthly["cf"].std()),
        "seasonal_share": float(seasonal_var / total_var),
        "anomaly_share": float(anomaly_var / total_var),
        "anomaly_sd": float(anomaly.std()),
    }


def anomaly_autocorrelation(monthly: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 6, 12)
                            ) -> dict[int, tuple[float, int]]:
    """Autocorrelation of the seasonal anomaly at each lag -> ``{lag: (r, n)}``.

    Only genuinely consecutive month pairs are used. The archive has a 41-month
    hole, so naive ``.shift(lag)`` would pair 2016-07 with 2020-01 and report a
    correlation between months four years apart.
    """
    clim = monthly.groupby("month")["cf"].transform("mean")
    anomaly = (monthly["cf"] - clim).to_numpy()
    periods = monthly.index

    out: dict[int, tuple[float, int]] = {}
    for lag in lags:
        pairs = [(anomaly[i], anomaly[i - lag]) for i in range(lag, len(periods))
                 if periods[i] == periods[i - lag] + lag]
        if len(pairs) < 10:
            out[lag] = (float("nan"), len(pairs))
            continue
        a, b = np.array(pairs).T
        out[lag] = (float(np.corrcoef(a, b)[0, 1]), len(pairs))
    return out


# ---------------------------------------------------------------------------
# M2 — exceedance product
# ---------------------------------------------------------------------------

#: Exceedance estimators. "empirical" reads the quantile straight off the
#: order statistics; "normal" fits mean/sd and reads the quantile off the fitted
#: Gaussian. The choice matters far more than it looks: with 12-13 years per
#: calendar month, the empirical 10th percentile is essentially the second-lowest
#: year, so it moves by a whole rank when one year is held out. The normal fit
#: uses all 12 points to place the tail. Which one is right is settled by
#: `exceedance_validation`, not by preference.
EXCEEDANCE_METHODS = ("empirical", "normal")


def _exceedance_value(sample: pd.Series, quantile: float, method: str) -> float:
    if method == "empirical":
        return float(sample.quantile(quantile))
    if method == "normal":
        from scipy.stats import norm
        return float(norm.ppf(quantile, loc=sample.mean(), scale=sample.std(ddof=1)))
    raise ValueError(f"unknown exceedance method {method!r}; expected one of "
                     f"{EXCEEDANCE_METHODS}")


def exceedance_table(monthly: pd.DataFrame, method: str = "normal") -> pd.DataFrame:
    """Per-calendar-month P50/P75/P90 capacity factor — the deliverable.

    Uses every available year for each calendar month. This is the product;
    :func:`exceedance_validation` is the check that it holds out of sample.

    Defaults to the ``normal`` fit because that is what the validation selected —
    the empirical estimator's P90 was exceeded 83.3% of the time against a
    nominal 90%, i.e. it published a P90 that was too optimistic.
    """
    rows = []
    for month in range(1, 13):
        sub = monthly[monthly["month"] == month]["cf"]
        if sub.empty:
            continue
        row = {
            "month": month,
            "month_name": MONTH_NAMES[month - 1],
            "n_years": len(sub),
            "mean_cf": sub.mean(),
            "sd_cf": sub.std(),
        }
        for level, q in EXCEEDANCE_QUANTILE.items():
            row[f"p{level}_cf"] = _exceedance_value(sub, q, method)
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def exceedance_validation(monthly: pd.DataFrame, method: str = "normal") -> pd.DataFrame:
    """Do the exceedance levels hold out of sample? Leave-one-year-out hit rates.

    For each held-out month, the PXX is estimated from the *other* years and we
    ask whether the held-out value exceeded it. A well-calibrated P90 is exceeded
    ~90% of the time.

    This gate can fail, and for the empirical estimator it does: P90 came back at
    0.833 against a nominal 0.90, meaning the published P90 was **too high** —
    the optimistic direction, which is the dangerous one for a bankability
    figure. Hence the ``normal`` default in :func:`exceedance_table`.
    """
    records: dict[int, list[bool]] = {level: [] for level in EXCEEDANCE_QUANTILE}
    for idx, row in monthly.iterrows():
        others = monthly[(monthly["month"] == row["month"])
                         & (monthly["year"] != row["year"])]["cf"]
        if len(others) < 3:
            continue
        for level, q in EXCEEDANCE_QUANTILE.items():
            records[level].append(bool(row["cf"] > _exceedance_value(others, q, method)))

    return pd.DataFrame([
        {
            "level": f"P{level}",
            "method": method,
            "nominal_exceedance": level / 100,
            "observed_exceedance": float(np.mean(hits)) if hits else float("nan"),
            "error": (float(np.mean(hits)) - level / 100) if hits else float("nan"),
            "n": len(hits),
        }
        for level, hits in records.items()
    ]).set_index("level")


def diurnal_profile(cf_hourly: pd.Series) -> pd.DataFrame:
    """Mean capacity factor by (calendar month, hour of day), UTC.

    Anloga is UTC+0, so these hours are local time without conversion.
    """
    frame = pd.DataFrame({"cf": cf_hourly})
    frame["month"] = frame.index.month
    frame["hour"] = frame.index.hour
    pivot = frame.pivot_table(index="month", columns="hour", values="cf", aggfunc="mean")
    pivot.index = [MONTH_NAMES[m - 1] for m in pivot.index]
    return pivot


def annual_energy_mwh(monthly: pd.DataFrame, rated_w: float) -> pd.DataFrame:
    """Energy per calendar year (MWh), flagging years that are not complete."""
    rows = []
    for year, sub in monthly.groupby("year"):
        hours = float(sub["hours"].sum())
        rows.append({
            "year": int(year),
            "months": len(sub),
            "complete": len(sub) == 12,
            "mean_cf": float(sub["cf"].mean()),
            "energy_mwh": float((sub["cf"] * sub["hours"]).sum() * rated_w / 1e6),
            "hours": hours,
        })
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Baseline scoring
# ---------------------------------------------------------------------------

@dataclass
class BaselineScores:
    """MAE of each monthly baseline, in capacity-factor units."""

    n: int
    clim_loyo: float
    clim_in_sample: float
    persistence: float
    persistence_n: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"baseline": "climatology (leave-one-year-out)", "mae_cf": self.clim_loyo,
             "pp_of_rated": self.clim_loyo * 100, "n": self.n},
            {"baseline": "climatology (in-sample, DO NOT USE)", "mae_cf": self.clim_in_sample,
             "pp_of_rated": self.clim_in_sample * 100, "n": self.n},
            {"baseline": "anomaly persistence", "mae_cf": self.persistence,
             "pp_of_rated": self.persistence * 100, "n": self.persistence_n},
        ]).set_index("baseline")


def score_baselines(monthly: pd.DataFrame) -> BaselineScores:
    """Score the monthly baselines against each other.

    Persistence is scored so it can be *ruled out* on evidence: it loses to
    climatology at this scale, which is the opposite of the 24 h regime.
    """
    clim_loyo = loyo_climatology(monthly)
    valid = clim_loyo.notna()
    mae_loyo = float((monthly.loc[valid, "cf"] - clim_loyo[valid]).abs().mean())

    clim_in = monthly.groupby("month")["cf"].transform("mean")
    mae_in = float((monthly["cf"] - clim_in).abs().mean())

    # Persistence of the anomaly, over genuinely consecutive months only.
    anomaly = monthly["cf"] - clim_in
    periods = monthly.index
    diffs = [abs(anomaly.iloc[i] - anomaly.iloc[i - 1]) for i in range(1, len(periods))
             if periods[i] == periods[i - 1] + 1]
    return BaselineScores(
        n=int(valid.sum()),
        clim_loyo=mae_loyo,
        clim_in_sample=mae_in,
        persistence=float(np.mean(diffs)) if diffs else float("nan"),
        persistence_n=len(diffs),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(config_path: str) -> dict[str, pd.DataFrame]:
    """Build every M2 artifact and write it to ``outputs/reports/``."""
    config = load_config(config_path)
    cf_hourly = load_hourly_capacity_factor(config)
    monthly = monthly_frame(cf_hourly)

    turbine = load_turbine(config.turbine.name, float(config.site.hub_height_m))
    rated_w = float(turbine.nominal_power)

    stats = variance_decomposition(monthly)
    autocorr = anomaly_autocorrelation(monthly)
    baselines = score_baselines(monthly)
    table = exceedance_table(monthly)
    # Score BOTH estimators so the choice of the normal fit is recorded as a
    # measurement rather than a preference.
    validation = pd.concat([exceedance_validation(monthly, method=m)
                            for m in EXCEEDANCE_METHODS])
    diurnal = diurnal_profile(cf_hourly)
    annual = annual_energy_mwh(monthly, rated_w)

    table["p50_energy_mwh"] = table["p50_cf"] * table.index.map(
        lambda m: 24 * pd.Period(f"2001-{m:02d}", freq="M").days_in_month) * rated_w / 1e6

    reports = ensure_dir(Path(config.paths.outputs) / "reports")
    monthly_out = monthly.copy()
    monthly_out.index = monthly_out.index.astype(str)
    monthly_out["clim_loyo"] = loyo_climatology(monthly).to_numpy()
    monthly_out["anomaly"] = monthly_out["cf"] - monthly_out["clim_loyo"]

    artifacts = {
        "monthly_series": monthly_out,
        "monthly_climatology": table,
        "monthly_exceedance_validation": validation,
        "monthly_diurnal": diurnal,
        "monthly_annual_energy": annual,
        "monthly_baselines": baselines.as_frame(),
    }
    for name, frame in artifacts.items():
        frame.to_csv(reports / f"{name}.csv")
        logger.info("wrote %s", reports / f"{name}.csv")

    logger.info("n=%d months | mean CF %.4f | seasonal share %.1f%% | anomaly share %.1f%%",
                stats["n_months"], stats["mean_cf"],
                stats["seasonal_share"] * 100, stats["anomaly_share"] * 100)
    logger.info("LOYO climatology MAE %.4f CF (%.2f pp of rated); in-sample %.4f",
                baselines.clim_loyo, baselines.clim_loyo * 100, baselines.clim_in_sample)
    for lag, (r, n) in autocorr.items():
        logger.info("anomaly autocorr lag %-2d: %+.3f (n=%d)", lag, r, n)
    for method in EXCEEDANCE_METHODS:
        sub = validation[validation["method"] == method]
        logger.info("exceedance (%s): %s", method, ", ".join(
            f"{lvl} {row.observed_exceedance:.3f} vs {row.nominal_exceedance:.2f}"
            for lvl, row in sub.iterrows()))

    artifacts["_stats"] = pd.DataFrame([stats])
    artifacts["_stats"].to_csv(reports / "monthly_stats.csv", index=False)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly wind resource climatology (M2)."
    )
    parser.add_argument("--config", default="config_monthly.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
