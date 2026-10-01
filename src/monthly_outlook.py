"""M1 — the 30-day energy outlook: NWP for days 1-7, climatology for the rest.

What this can and cannot be
---------------------------
Open-Meteo previous-runs stop at lead day 7 (probed live 2026-08-12: all-null
before 2024-03, nothing past d7). So a monthly outlook issued on day 0 knows the
first ~7 days and must fall back to climatology for the other ~23. Measured on
150 months, assuming a **perfect** 7-day forecast, that caps skill against
leave-one-year-out climatology at **13.1%**. The real weekly model is not
perfect, so the achievable figure is lower — and this module's job is to
measure it honestly, not to reach for it.

Experimental design: isolate the head
-------------------------------------
The outlook and its baseline are made *identical except for the first seven
days*::

    baseline_cf = clim(M)
    outlook_cf  = clim(M) + (head_hours / month_hours) * (head_forecast - head_clim)

Both use the same climatology for the tail, so any difference in score is
attributable to the head and nothing else. The head term is an **anomaly**
correction rather than a level, which removes a subtle bias: the first seven
days of a month are not climatologically identical to the month as a whole, and
comparing a 7-day forecast level against a 30-day climatological level would
score that seasonal drift as forecast error.

Training is an expanding window
-------------------------------
For target month *M*, every lead model is trained only on valid times **before**
*M* begins. This is a deployment rehearsal, not a study: on the first of the
month you have the past and nothing else. It costs sample size — the earliest
months are unevaluable — and that cost is reported rather than avoided by
training on the whole window.

⚠️ **Expect the confidence interval to span zero.** With a 13.1% ceiling and
~20 evaluable months, this experiment does not have the statistical power to
separate a small real improvement from noise. Reporting "no demonstrable skill"
is the correct outcome in that case, and the interval is the deliverable — not
the point estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data_ingest import load_nwp_forecasts
from .monthly import (
    capacity_factor_from_weather,
    load_cached_weather,
    loyo_climatology,
    monthly_frame,
)
from .stage2_powercurve import load_turbine, windspeed_to_power
from .utils import Config, ensure_dir, get_logger, load_config, set_seed
from .weekly import available_leads, build_valid_time_frame

logger = get_logger(__name__)

#: Days of a month covered by the NWP head, one per available lead day.
HEAD_DAYS = 7

#: First date for which Open-Meteo previous-runs returns anything at this site.
#: Probed live 2026-08-12: every lead is all-null through 2024-02 and populated
#: from 2024-03. The monthly config's `data.start` is 2010-01-01 because the
#: climatology wants the whole ERA5 archive — asking the forecast API for 2010
#: returns HTTP 400, so the NWP window has to be clamped separately rather than
#: inherited.
NWP_HISTORY_START = "2024-03-01"

#: A month is only evaluable if at least this many months of NWP precede it,
#: otherwise the lead models are fitted on too little to mean anything.
MIN_TRAIN_MONTHS = 6


@dataclass
class MonthOutlook:
    """One month's outlook, its baseline, and the ceiling on the same month."""

    period: str
    actual_cf: float
    clim_cf: float
    outlook_cf: float
    perfect_head_cf: float
    head_hours: int
    month_hours: int
    n_train: int
    leads_used: list[int] = field(default_factory=list)


def _nwp_window(config: Config) -> Config:
    """Config copy whose data window is clamped to the NWP era.

    The climatology wants every ERA5 month back to 2010; the forecast API has
    nothing before 2024-03 and returns HTTP 400 rather than an empty frame when
    asked. Clamping here keeps one config serving both products.
    """
    import dataclasses

    start = config.raw.get("monthly", {}).get("nwp_start", NWP_HISTORY_START)
    clamped = max(pd.Timestamp(config.data.start), pd.Timestamp(start))
    return dataclasses.replace(
        config, data=dataclasses.replace(
            config.data, start=clamped.strftime("%Y-%m-%d")))


def head_climatology(cf_hourly: pd.Series, month: int, year: int,
                     head_days: int = HEAD_DAYS) -> float:
    """Mean capacity factor over days 1..head_days of this calendar month,
    across **other** years. The climatological expectation the head replaces."""
    idx = cf_hourly.index
    mask = (idx.month == month) & (idx.day <= head_days) & (idx.year != year)
    return float(cf_hourly[mask].mean())


def _fit_head(frames: dict[int, tuple[pd.DataFrame, pd.Series]],
              month_start: pd.Timestamp, month_end: pd.Timestamp,
              turbine, seed: int) -> tuple[pd.Series, pd.Series, int, list[int]]:
    """Predicted and actual hourly capacity factor over the head of one month.

    Returns ``(predicted_cf, actual_cf, n_train, leads_used)``. Lead day *N*
    supplies day *N* of the month, which is exactly what an outlook issued at
    month start would have.
    """
    from sklearn.linear_model import LinearRegression

    pred_parts, actual_parts = [], []
    n_train = 0
    leads_used: list[int] = []

    for lead, (X, y) in sorted(frames.items()):
        day_start = month_start + pd.Timedelta(days=lead - 1)
        day_end = min(day_start + pd.Timedelta(days=1), month_end)
        if day_start >= month_end:
            continue

        train = X.index < month_start
        target = (X.index >= day_start) & (X.index < day_end)
        if train.sum() < 24 * 30 or target.sum() == 0:
            continue

        # A 21-parameter linear MOS stays within ~5% of gradient boosting at
        # every horizon this project has tested, and at this sample size the
        # simpler estimator is the defensible one.
        model = LinearRegression().fit(X[train], y[train])
        pred_ws = pd.Series(model.predict(X[target]), index=X.index[target])

        pred_parts.append(windspeed_to_power(pred_ws, turbine)
                          / float(turbine.nominal_power))
        actual_parts.append(windspeed_to_power(y[target], turbine)
                            / float(turbine.nominal_power))
        n_train = max(n_train, int(train.sum()))
        leads_used.append(lead)

    if not pred_parts:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0, []
    return (pd.concat(pred_parts).sort_index(),
            pd.concat(actual_parts).sort_index(), n_train, leads_used)


def build_outlooks(config: Config, seed: int = 42) -> pd.DataFrame:
    """Score the outlook against climatology for every evaluable month."""
    set_seed(seed)
    # One NetCDF read shared by both sides: the frame the lead models train on
    # and the capacity factor the months are scored in.
    weather = load_cached_weather(config)
    nwp = load_nwp_forecasts(_nwp_window(config))
    if nwp.empty:
        raise ValueError("No NWP forecasts loaded; the outlook needs day 1-7 data.")

    cf_hourly = capacity_factor_from_weather(weather, config)
    monthly = monthly_frame(cf_hourly)
    clim = loyo_climatology(monthly)

    turbine = load_turbine(config.turbine.name, float(config.site.hub_height_m))

    leads = [l for l in available_leads(nwp) if l <= HEAD_DAYS]
    frames: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    for lead in leads:
        try:
            X, y, models = build_valid_time_frame(weather, nwp, lead, config)
        except ValueError as exc:
            logger.warning("lead %d unusable: %s", lead, exc)
            continue
        if X.empty:
            continue
        frames[lead] = (X, y)
        logger.info("lead %d: %d rows, %d features, members=%s",
                    lead, len(X), X.shape[1], ",".join(models))

    if not frames:
        raise ValueError("No usable lead frames were built.")

    nwp_start = min(X.index.min() for X, _ in frames.values())
    results: list[MonthOutlook] = []

    for period, row in monthly.iterrows():
        month_start = pd.Timestamp(period.start_time, tz="UTC")
        month_end = pd.Timestamp(period.end_time, tz="UTC").ceil("h")
        if month_start < nwp_start + pd.DateOffset(months=MIN_TRAIN_MONTHS):
            continue
        if pd.isna(clim[period]):
            continue

        pred_cf, actual_cf, n_train, leads_used = _fit_head(
            frames, month_start, month_end, turbine, seed)
        if pred_cf.empty:
            continue

        head_clim = head_climatology(cf_hourly, period.month, period.year)
        if not np.isfinite(head_clim):
            continue

        month_hours = int(row["hours"])
        head_hours = len(pred_cf)
        weight = head_hours / month_hours

        results.append(MonthOutlook(
            period=str(period),
            actual_cf=float(row["cf"]),
            clim_cf=float(clim[period]),
            outlook_cf=float(clim[period] + weight * (pred_cf.mean() - head_clim)),
            perfect_head_cf=float(clim[period] + weight * (actual_cf.mean() - head_clim)),
            head_hours=head_hours,
            month_hours=month_hours,
            n_train=n_train,
            leads_used=leads_used,
        ))

    if not results:
        raise ValueError("No month had enough preceding NWP data to evaluate.")

    frame = pd.DataFrame([{
        "period": r.period,
        "actual_cf": r.actual_cf,
        "clim_cf": r.clim_cf,
        "outlook_cf": r.outlook_cf,
        "perfect_head_cf": r.perfect_head_cf,
        "head_hours": r.head_hours,
        "month_hours": r.month_hours,
        "n_train": r.n_train,
        "n_leads": len(r.leads_used),
    } for r in results]).set_index("period")

    for name in ("clim", "outlook", "perfect_head"):
        frame[f"{name}_abs_err"] = (frame["actual_cf"] - frame[f"{name}_cf"]).abs()
    return frame


def bootstrap_skill(frame: pd.DataFrame, column: str = "outlook",
                    n_samples: int = 10000, seed: int = 42) -> dict[str, float]:
    """Skill against climatology with a percentile confidence interval.

    Months are resampled whole. The interval is the deliverable: at this sample
    size a point estimate on its own would imply a precision that does not exist.
    """
    rng = np.random.default_rng(seed)
    base = frame["clim_abs_err"].to_numpy()
    test = frame[f"{column}_abs_err"].to_numpy()
    n = len(base)

    point = 1.0 - test.mean() / base.mean()
    draws = rng.integers(0, n, size=(n_samples, n))
    skills = 1.0 - test[draws].mean(axis=1) / base[draws].mean(axis=1)

    return {
        "n_months": float(n),
        "mae_clim": float(base.mean()),
        f"mae_{column}": float(test.mean()),
        "skill": float(point),
        "ci_low": float(np.percentile(skills, 2.5)),
        "ci_high": float(np.percentile(skills, 97.5)),
        "p_better": float((skills > 0).mean()),
    }


def run(config_path: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    frame = build_outlooks(config, seed=seed)

    n_boot = int(config.raw.get("monthly", {}).get("bootstrap_samples", 10000))
    summary = pd.DataFrame([
        {"arm": "outlook (real 7-day forecast)",
         **bootstrap_skill(frame, "outlook", n_boot, seed)},
        {"arm": "perfect head (ceiling on these months)",
         **bootstrap_skill(frame, "perfect_head", n_boot, seed)},
    ]).set_index("arm")

    reports = ensure_dir(Path(config.paths.outputs) / "reports")
    frame.to_csv(reports / "monthly_outlook.csv")
    summary.to_csv(reports / "monthly_outlook_summary.csv")
    logger.info("wrote %s", reports / "monthly_outlook.csv")

    for (arm, column), row in zip(
            [("outlook (real 7-day forecast)", "outlook"),
             ("perfect head (ceiling on these months)", "perfect_head")],
            summary.itertuples()):
        verdict = ("no demonstrable skill" if row.ci_low <= 0 <= row.ci_high
                   else "significant")
        logger.info("%-40s n=%2d  MAE %.4f vs clim %.4f  skill %+.1f%% "
                    "[%.1f%%, %.1f%%]  -> %s",
                    arm, int(row.n_months), getattr(row, f"mae_{column}"),
                    row.mae_clim, row.skill * 100,
                    row.ci_low * 100, row.ci_high * 100, verdict)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="30-day energy outlook (M1): NWP head + climatological tail."
    )
    parser.add_argument("--config", default="config_monthly.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
