"""Re-powered evaluation of the 30-day outlook's forecast head.

Why this module exists rather than a flag on `src.monthly_outlook`
------------------------------------------------------------------
:mod:`src.monthly_outlook` scores the outlook on **calendar-month totals** and
returned +1.1% skill with a 95% interval of [-19.7%, +17.7%]. That null was
reported honestly, but the design was not merely short of samples — it was
*diluting* the signal, and dilution is fixable where waiting is not:

* A 7-day forecast is **23% of a 30-day month**. The other 77% is climatological
  tail error, **identical in both arms** — and because the metric is absolute
  error it does not cancel, it only inflates the variance the comparison is
  measured against. The perfect-head floor is 0.0262 CF against a climatology
  MAE of 0.0346, so the head can move at most 24% of the total.
* Consequence: the monthly design's minimum detectable effect is **26.1%**,
  against a ceiling of **13.1%**. It was structurally incapable of resolving even
  a *perfect* forecast. A null was guaranteed by the design.

So this module changes **what the head is scored against**, not the model:

    monthly totals  ->  non-overlapping 7-day energy blocks
    n = 22          ->  n = 117          (5.3x)
    head = 23%      ->  head = 100%      (4.3x effect)
                                          ~10x power

`src.monthly_outlook.build_outlooks` is deliberately left alone: its n=22 result
is the consistency check that the composite here must reproduce.

Tiling, not sampling
--------------------
The monthly design issued once per month and forecast only days 1-7, leaving
days 8-30 unforecast — one scalar per month, most of the archive unused. Here a
forecast is issued **every 7 days**, so consecutive issues *tile* the era with no
gaps and no overlaps. Every hour of the evaluable record is predicted exactly
once, which :func:`assert_tiles` checks rather than assumes.

Issue weekly, refit monthly
---------------------------
Refitting at every issue would be 7 leads x 117 origins ~ 819 gradient-boosting
fits. Refitting monthly cuts that to ~203 and is the more operationally
realistic choice — nobody retrains weekly. Training for an origin always ends at
the **refit boundary at or before it**, so a mid-month issue uses a model that is
up to four weeks stale. That is conservative and cannot leak.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bakeoff import diebold_mariano
from .data_ingest import load_nwp_forecasts
from .monthly import (
    capacity_factor_from_weather,
    load_cached_weather,
    loyo_climatology,
    monthly_frame,
)
from .monthly_outlook import HEAD_DAYS, NWP_HISTORY_START, _nwp_window, head_climatology
from .stage2_powercurve import load_turbine, windspeed_to_power
from .utils import Config, ensure_dir, get_logger, load_config, set_seed
from .weekly import available_leads, build_valid_time_frame, climatology_predict

logger = get_logger(__name__)

#: Days between forecast issues. Equal to HEAD_DAYS so issues tile the timeline.
ISSUE_DAYS = HEAD_DAYS

#: Months of NWP required before the first issue. The monthly design used 6,
#: which cost seven months of record; the MOS heads are 17-23 parameters and two
#: months is ~1,400 hourly rows, so 6 was over-cautious.
WARMUP_MONTHS = 2

#: Candidate heads, cheapest first. Every one already exists elsewhere in the
#: project; this module only runs them in an expanding-origin loop instead of
#: `weekly.evaluate_lead`'s single 80/20 split.
LADDER = ("ensemble_mean", "mos_single", "mos_multi", "xgboost")

#: Reference arm. Beyond ~36 h climatology is the right baseline, not
#: persistence — measured in redo/results/07.
#:
#: ⚠️ **Which climatology matters enormously.** `weekly.climatology_predict`
#: builds a (month, hour) mean from the *training window*, which in the NWP era
#: is at most two years — a weak estimate, and scoring against it inflates every
#: candidate's skill. `redo/results/10` requires leave-one-year-out climatology
#: from the full archive, and that archive is operationally available: ERA5's
#: ~5-day latency is irrelevant to a climatological mean. So the reference is
#: built from every ERA5 hour **before the issue time** (~14 years), and the
#: training-window version is retained as a second arm purely to measure how much
#: the choice is worth.
REFERENCE = "climatology_archive"
WINDOW_CLIMATOLOGY = "climatology_window"


# ---------------------------------------------------------------------------
# Origins and tiling
# ---------------------------------------------------------------------------

def issue_origins(start: pd.Timestamp, end: pd.Timestamp,
                  issue_days: int = ISSUE_DAYS) -> list[pd.Timestamp]:
    """Issue timestamps that tile ``[start, end)`` in whole blocks.

    A partial trailing block is dropped rather than scored short: a 3-day block
    and a 7-day block are not exchangeable samples, and mixing them would let
    block length leak into the variance estimate.
    """
    origins = []
    cursor = start
    step = pd.Timedelta(days=issue_days)
    while cursor + step <= end:
        origins.append(cursor)
        cursor += step
    return origins


#: A block is scored only if this fraction of its 168 hours were predicted.
MIN_BLOCK_COVERAGE = 0.95


def assert_tiles(frame: pd.DataFrame, origins: list[pd.Timestamp],
                 issue_days: int = ISSUE_DAYS) -> pd.Series:
    """No hour predicted twice; returns per-block coverage.

    **Overlap is a bug and raises.** An off-by-one in the lead-to-day mapping
    shows up here as a duplicate and nowhere else, so this stays strict.

    **Gaps are data, and are reported rather than raised.**
    `weekly.build_valid_time_frame` drops rows where any NWP column is null, so a
    handful of hours are genuinely absent from the record — measured at 86 of
    19,488 (0.44%) on the first run of this module, concentrated in 2026-04.
    Asserting zero gaps would therefore be asserting something about Open-Meteo's
    uptime, not about this code. Blocks below `MIN_BLOCK_COVERAGE` are dropped by
    the caller instead, on the same reasoning as `monthly.monthly_frame`'s
    completeness guard: a block missing its windiest day is not a short block,
    it is a biased one.
    """
    if not origins:
        raise ValueError("no issue origins")

    counts = frame.index.value_counts()
    duplicated = counts[counts > 1]
    if len(duplicated):
        raise AssertionError(
            f"{len(duplicated)} timestamps predicted more than once, "
            f"first={duplicated.index[0]}")

    hours = issue_days * 24
    coverage = frame.groupby("block").size() / hours
    return coverage.rename("coverage")


# ---------------------------------------------------------------------------
# Candidate heads
# ---------------------------------------------------------------------------

def fit_candidate(name: str, X_tr: pd.DataFrame, y_tr: pd.Series,
                  lead: int, seed: int):
    """Fit one candidate and return a ``predict(X) -> ndarray`` callable.

    Returns a *fitted predictor*, not a prediction, because one fit is reused
    across every issue in its refit month — caching predictions instead would
    silently serve one block's values to the next.
    """
    from sklearn.linear_model import LinearRegression

    if name == "ensemble_mean":
        # The 'no statistics at all' bar — raw multi-model mean, nothing fitted.
        if "nwp_ensemble_mean" not in X_tr.columns:
            return None
        return lambda X: X["nwp_ensemble_mean"].to_numpy(dtype=float)

    if name == "mos_single":
        cols = [c for c in X_tr.columns if c.endswith(f"wind_speed_100m_d{lead}")]
        if not cols:
            return None
        corrs = {c: abs(X_tr[c].corr(y_tr)) for c in cols}
        best = max(corrs, key=corrs.get)
        model = LinearRegression().fit(X_tr[[best]], y_tr)
        return lambda X: np.asarray(model.predict(X[[best]]), dtype=float)

    if name == "mos_multi":
        model = LinearRegression().fit(X_tr, y_tr)
        return lambda X: np.asarray(model.predict(X), dtype=float)

    if name == "xgboost":
        from .stage1_windspeed import _make_regressor
        model = _make_regressor("xgboost", seed)
        model.fit(X_tr, y_tr)
        return lambda X: np.asarray(model.predict(X), dtype=float)

    raise ValueError(f"unknown candidate {name!r}; expected one of {LADDER}")


def archive_climatology(cf_hourly: pd.Series, before: pd.Timestamp,
                        index: pd.DatetimeIndex) -> np.ndarray:
    """(month, hour) mean capacity factor from every archive hour before `before`.

    Built directly in **capacity-factor space** rather than by pushing a
    climatological wind speed through the power curve: the power curve is
    convex, so ``power(mean(ws)) != mean(power(ws))``, and an energy baseline
    wants the latter.

    Expanding window — an outlook issued in 2025 may use 2010-2024 but not 2026.
    """
    past = cf_hourly[cf_hourly.index < before]
    if past.empty:
        raise ValueError(f"no archive data before {before}")
    key = pd.MultiIndex.from_arrays([past.index.month, past.index.hour],
                                    names=["m", "h"])
    table = {(m, h): float(v) for (m, h), v in past.groupby(key).mean().items()}
    fallback = float(past.mean())
    return np.array([table.get((m, h), fallback)
                     for m, h in zip(index.month, index.hour)], dtype=float)


@dataclass
class _LeadFrame:
    X: pd.DataFrame
    y: pd.Series
    models: list[str]


def _lead_frames(config: Config, weather: pd.DataFrame,
                 nwp: pd.DataFrame) -> dict[int, _LeadFrame]:
    frames: dict[int, _LeadFrame] = {}
    for lead in [l for l in available_leads(nwp) if l <= HEAD_DAYS]:
        try:
            X, y, models = build_valid_time_frame(weather, nwp, lead, config)
        except ValueError as exc:
            logger.warning("lead %d unusable: %s", lead, exc)
            continue
        if not X.empty:
            frames[lead] = _LeadFrame(X, y, models)
            logger.info("lead %d: %d rows, %d features, members=%s",
                        lead, len(X), X.shape[1], ",".join(models))
    return frames


# ---------------------------------------------------------------------------
# The walk-forward run
# ---------------------------------------------------------------------------

def month_start_origins(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Issue timestamps at the first of each month — the outlook *product*.

    The tiled weekly grid is for **measuring** the head; the actual product is
    issued on the first of the month. The two must not be confused: on a weekly
    grid the forecast covering day 3 of a month may have been issued five days
    earlier (lead 5), whereas the product would use lead 3. Composing the monthly
    number from the tiled grid therefore scores the wrong leads.
    """
    first = pd.Timestamp(year=start.year, month=start.month, day=1, tz=start.tz)
    if first < start:
        first += pd.DateOffset(months=1)
    origins = []
    cursor = first
    while cursor + pd.Timedelta(days=HEAD_DAYS) <= end:
        origins.append(cursor)
        cursor += pd.DateOffset(months=1)
    return origins


def build_head_forecasts(config: Config, ladder: tuple[str, ...] = LADDER,
                         issue_days: int = ISSUE_DAYS,
                         warmup_months: int = WARMUP_MONTHS,
                         origins: list[pd.Timestamp] | None = None,
                         tile: bool = True,
                         seed: int = 42) -> pd.DataFrame:
    """Hourly predictions from every candidate over a grid of forecast issues.

    Returns a frame indexed by valid time with ``actual``, both climatology arms,
    one column per candidate (all in capacity-factor units), plus ``lead_day``,
    ``origin`` and ``block``.

    ``origins=None`` uses the weekly grid that tiles the era (for measurement);
    pass :func:`month_start_origins` for the product. ``tile`` controls whether
    full coverage is expected — month-start issues leave 23-day gaps by design.
    """
    set_seed(seed)
    weather = load_cached_weather(config)
    cf_archive = capacity_factor_from_weather(weather, config)
    nwp = load_nwp_forecasts(_nwp_window(config))
    if nwp.empty:
        raise ValueError("No NWP forecasts loaded; the head needs day 1-7 data.")

    frames = _lead_frames(config, weather, nwp)
    if not frames:
        raise ValueError("No usable lead frames were built.")

    turbine = load_turbine(config.turbine.name, float(config.site.hub_height_m))
    rated = float(turbine.nominal_power)

    nwp_start = min(f.X.index.min() for f in frames.values())
    nwp_end = min(f.X.index.max() for f in frames.values())
    start = (nwp_start + pd.DateOffset(months=warmup_months)).normalize()
    if origins is None:
        origins = issue_origins(start, nwp_end, issue_days)
    if not origins:
        raise ValueError("no issue origins in the evaluable window")
    logger.info("issuing %d forecasts: %s .. %s (tile=%s)",
                len(origins), origins[0].date(), origins[-1].date(), tile)

    # Models are refit at month boundaries; every issue inside a month reuses
    # that month's fit. Keyed by ((year, month), lead, candidate) — a plain tuple
    # rather than a Period, which warns about dropping the timezone.
    fitted: dict[tuple[tuple[int, int], int, str], object] = {}
    rows: list[pd.DataFrame] = []

    for block, origin in enumerate(origins):
        refit_key = (origin.year, origin.month)
        refit_at = pd.Timestamp(year=origin.year, month=origin.month, day=1,
                                tz=origin.tz)
        for lead, frame in sorted(frames.items()):
            day_start = origin + pd.Timedelta(days=lead - 1)
            day_end = day_start + pd.Timedelta(days=1)

            train = frame.X.index < refit_at
            target = (frame.X.index >= day_start) & (frame.X.index < day_end)
            if target.sum() == 0:
                continue
            if train.sum() < 24 * 30:
                continue

            # Leakage gate: training must end strictly before the block starts.
            last_train = frame.X.index[train].max()
            if last_train >= origin:
                raise AssertionError(
                    f"leak at lead {lead} origin {origin}: train ends {last_train}")

            X_tr, y_tr = frame.X[train], frame.y[train]
            X_te, y_te = frame.X[target], frame.y[target]

            out = pd.DataFrame(index=X_te.index)
            out["actual"] = windspeed_to_power(y_te, turbine).to_numpy() / rated

            # The admissible baseline: ~14 years of archive, expanding, in CF space.
            out[REFERENCE] = archive_climatology(cf_archive, origin, X_te.index)
            # The weak baseline 09 used, kept only to price the difference.
            clim_ws = climatology_predict(y_tr, X_te.index)
            out[WINDOW_CLIMATOLOGY] = windspeed_to_power(
                pd.Series(clim_ws, index=X_te.index), turbine).to_numpy() / rated

            for name in ladder:
                key = (refit_key, lead, name)
                if key not in fitted:
                    fitted[key] = fit_candidate(name, X_tr, y_tr, lead, seed)
                predictor = fitted[key]
                if predictor is None:
                    out[name] = np.nan
                    continue
                out[name] = windspeed_to_power(
                    pd.Series(predictor(X_te), index=X_te.index),
                    turbine).to_numpy() / rated

            out["lead_day"] = lead
            out["origin"] = origin
            out["block"] = block
            rows.append(out)

        if (block + 1) % 20 == 0:
            logger.info("  %d/%d issues done", block + 1, len(origins))

    frame = pd.concat(rows).sort_index()
    coverage = assert_tiles(frame, origins, issue_days)

    incomplete = coverage[coverage < MIN_BLOCK_COVERAGE]
    if len(incomplete):
        logger.info("dropping %d block(s) below %.0f%% coverage: %s",
                    len(incomplete), MIN_BLOCK_COVERAGE * 100,
                    ", ".join(f"#{b}({c:.2f})" for b, c in incomplete.items()))
        frame = frame[~frame["block"].isin(incomplete.index)]

    logger.info("%d hours across %d blocks, no overlaps; mean coverage %.4f",
                len(frame), frame["block"].nunique(),
                coverage[coverage >= MIN_BLOCK_COVERAGE].mean())
    if tile:
        span = (origins[-1] + pd.Timedelta(days=issue_days) - origins[0])
        logger.info("tiling check: %d blocks x %d h vs %d h span",
                    len(origins), issue_days * 24, int(span.total_seconds() // 3600))
    return frame


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def block_scores(frame: pd.DataFrame, ladder: tuple[str, ...] = LADDER
                 ) -> pd.DataFrame:
    """Mean capacity factor per non-overlapping 7-day block, per arm."""
    cols = ["actual", REFERENCE, *([WINDOW_CLIMATOLOGY]
                                   if WINDOW_CLIMATOLOGY in frame.columns else []),
            *ladder]
    return frame.groupby("block")[cols].mean()


def block_summary(blocks: pd.DataFrame, ladder: tuple[str, ...] = LADDER
                  ) -> pd.DataFrame:
    """Block-level MAE, skill against climatology, and a paired DM test.

    ⚠️ **Block level: the forecast is 100% of the prediction.** These skills are
    not comparable to the monthly figures in `redo/results/11`, where the same
    forecast covers 23% of the target.
    """
    ref_err = (blocks["actual"] - blocks[REFERENCE]).to_numpy()
    arms = [REFERENCE] + ([WINDOW_CLIMATOLOGY]
                          if WINDOW_CLIMATOLOGY in blocks.columns else []) + list(ladder)
    rows = []
    for name in arms:
        err = (blocks["actual"] - blocks[name]).to_numpy()
        row = {
            "arm": name,
            "n_blocks": len(blocks),
            "mae_cf": float(np.abs(err).mean()),
            "skill_vs_clim": float(1 - np.abs(err).mean() / np.abs(ref_err).mean()),
        }
        if name != REFERENCE:
            # Adjacent blocks do not overlap but do share weather, so HAC lag 1.
            stat, p = diebold_mariano(err, ref_err, horizon=2)
            row["dm_stat"] = stat
            row["dm_p"] = p
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def lead_summary(frame: pd.DataFrame, ladder: tuple[str, ...] = LADDER
                 ) -> pd.DataFrame:
    """Per-lead hourly MAE in capacity factor, for ranking the ladder.

    Hourly rather than block-aggregated because ranking wants every sample it can
    get; the DM test uses HAC lag 23 for within-day autocorrelation.
    """
    arms = [REFERENCE] + ([WINDOW_CLIMATOLOGY]
                          if WINDOW_CLIMATOLOGY in frame.columns else []) + list(ladder)
    rows = []
    for lead, g in frame.groupby("lead_day"):
        ref_err = (g["actual"] - g[REFERENCE]).to_numpy()
        for name in arms:
            err = (g["actual"] - g[name]).to_numpy()
            row = {
                "lead_day": int(lead), "arm": name, "n_hours": len(g),
                "mae_cf": float(np.abs(err).mean()),
                "skill_vs_clim": float(
                    1 - np.abs(err).mean() / np.abs(ref_err).mean()),
            }
            if name != REFERENCE:
                stat, p = diebold_mariano(err, ref_err, horizon=24)
                row["dm_stat"], row["dm_p"] = stat, p
            rows.append(row)
    return pd.DataFrame(rows).set_index(["lead_day", "arm"])


# ---------------------------------------------------------------------------
# Monthly composite
# ---------------------------------------------------------------------------

def compose_monthly(product: pd.DataFrame, config: Config, best: str,
                    n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Monthly outlook from **month-start** issues of the selected head.

    ``product`` must come from :func:`build_head_forecasts` called with
    :func:`month_start_origins` — not from the tiled weekly grid. On the weekly
    grid the forecast covering day 3 of a month may have been issued five days
    earlier, so composing from it scores leads the real product would never use.

    The monthly product survives the re-powering; it is simply no longer *tested*
    at n=22. The **13.1% ceiling does not move** — the forecast still covers only
    ~23% of a month, and no amount of head skill changes that arithmetic.
    """
    cf = capacity_factor_from_weather(load_cached_weather(config), config)
    monthly = monthly_frame(cf)
    clim = loyo_climatology(monthly)
    rng = np.random.default_rng(seed)

    rows = []
    for period, row in monthly.iterrows():
        if pd.isna(clim[period]):
            continue
        month_start = pd.Timestamp(period.start_time, tz="UTC")
        head_end = month_start + pd.Timedelta(days=HEAD_DAYS)
        head = product[(product.index >= month_start) & (product.index < head_end)]
        if len(head) < HEAD_DAYS * 24 * 0.95:
            continue
        # Every hour of the head must come from this month's own issue.
        if head["origin"].nunique() != 1 or head["origin"].iloc[0] != month_start:
            continue
        hclim = head_climatology(cf, period.month, period.year)
        if not np.isfinite(hclim):
            continue

        w = len(head) / row["hours"]
        rows.append({
            "period": str(period),
            "actual_cf": float(row["cf"]),
            "clim_cf": float(clim[period]),
            "outlook_cf": float(clim[period] + w * (head[best].mean() - hclim)),
            "head_w": w,
        })

    out = pd.DataFrame(rows).set_index("period")
    out["clim_abs_err"] = (out.actual_cf - out.clim_cf).abs()
    out["outlook_abs_err"] = (out.actual_cf - out.outlook_cf).abs()

    base = out.clim_abs_err.to_numpy()
    test = out.outlook_abs_err.to_numpy()
    draws = rng.integers(0, len(base), size=(n_boot, len(base)))
    skills = 1 - test[draws].mean(axis=1) / base[draws].mean(axis=1)
    out.attrs["summary"] = {
        "head_model": best,
        "n_months": float(len(out)),
        "mae_clim": float(base.mean()),
        "mae_outlook": float(test.mean()),
        "skill": float(1 - test.mean() / base.mean()),
        "ci_low": float(np.percentile(skills, 2.5)),
        "ci_high": float(np.percentile(skills, 97.5)),
    }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(config_path: str, seed: int = 42) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)

    # (1) Measurement: the weekly grid that tiles the era.
    frame = build_head_forecasts(config, seed=seed)

    blocks = block_scores(frame)
    summary = block_summary(blocks)
    leads = lead_summary(frame)

    ranked = summary.drop(index=[REFERENCE, WINDOW_CLIMATOLOGY]).sort_values("mae_cf")
    best = str(ranked.index[0])
    logger.info("best head at block level: %s (MAE %.4f, skill %.1f%%)",
                best, ranked.iloc[0]["mae_cf"], ranked.iloc[0]["skill_vs_clim"] * 100)

    # (2) Product: month-start issues, so the leads match what would be served.
    weather_end = frame.index.max()
    product = build_head_forecasts(
        config, ladder=(best,), tile=False, seed=seed,
        origins=month_start_origins(frame.index.min(), weather_end))
    monthly = compose_monthly(product, config, best, seed=seed)
    s = monthly.attrs["summary"]
    logger.info("monthly composite via %s: n=%d MAE %.4f vs clim %.4f  "
                "skill %+.1f%% [%.1f%%, %.1f%%]",
                s["head_model"], int(s["n_months"]), s["mae_outlook"],
                s["mae_clim"], s["skill"] * 100, s["ci_low"] * 100, s["ci_high"] * 100)

    reports = ensure_dir(Path(config.paths.outputs) / "reports")
    artifacts = {
        "outlook_blocks": blocks,
        "outlook_block_summary": summary,
        "outlook_lead_summary": leads.reset_index(),
        "outlook_monthly_composite": monthly,
        "outlook_composite_summary": pd.DataFrame([s]),
    }
    for name, f in artifacts.items():
        f.to_csv(reports / f"{name}.csv")
        logger.info("wrote %s", reports / f"{name}.csv")

    for arm, r in summary.iterrows():
        logger.info("  %-15s MAE %.4f  skill %+6.1f%%%s", arm, r["mae_cf"],
                    r["skill_vs_clim"] * 100,
                    "" if arm == REFERENCE else f"  DM p={r['dm_p']:.2e}")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-powered evaluation of the outlook head (7-day blocks).")
    parser.add_argument("--config", default="config_monthly.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
