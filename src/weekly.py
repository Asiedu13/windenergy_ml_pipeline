"""Weekly (multi-day) forecasting — one model per forecast lead day.

Why this module exists rather than a horizon parameter
-----------------------------------------------------
The rest of the pipeline is indexed by **feature time** *t*, with target
``hub(t + h)``, because it needed observation-side lag features. Beyond about
day 2 those lags are dead — ``hub(t-1)`` says nothing useful about ``hub(t+168)``
— which removes the reason for that framing and permits a much simpler one:

    Index by **valid time τ**. For lead day *N* the row is

        (nwp_*_d{N}[τ], calendar(τ), spread(τ))  ->  hub(τ)

    which is exactly a lead-*N* Model Output Statistics problem, trained on
    every τ in the window.

Consequences, all of which we want:

* **Seven models, one per lead day**, each on ~15.9k rows — not 168 models, and
  not one pooled frame with 168x redundancy.
* Serving is trivial: to forecast the week from issue time *T*, take
  τ = T+1 … T+168 and apply the model for ``N = ceil((τ - T) / 24)``.
* **The lead guard becomes structural.** Each model can only see its own lead's
  columns, so the horizon/lead leak that `features.add_nwp_features` has to
  filter against cannot arise here by construction.
* The ICON gap handles itself — the day-7 model simply has two models' features
  instead of three, with no special-casing.

What the measurements say about the design
------------------------------------------
From the horizon sweep (redo/results/07):

* Skill decays but **decelerates** — 0.371 -> 0.320 -> 0.289 against climatology
  at 24/48/72 h, so a week-ahead forecast is worth building.
* **Climatology is the reference baseline beyond ~36 h**, not persistence.
  Damped persistence beats climatology at 24 h and loses to it by 48 h.
* A 21-parameter linear MOS stays within ~5% of gradient boosting at every
  horizon tested, so this is not a problem that needs a large model.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_ingest import load_nwp_forecasts, load_weather, source_wind_height_m
from .features import extrapolate_hub_wind
from .stage1_windspeed import compute_metrics
from .utils import Config, ensure_dir, get_logger, load_config, set_seed

logger = get_logger(__name__)

_LEAD_RE = re.compile(r"_d(\d+)$")

#: Hours in a forecast day. A lead day N covers valid times (24(N-1), 24N]
#: after the issue time.
HOURS_PER_DAY = 24


# ---------------------------------------------------------------------------
# Valid-time frame
# ---------------------------------------------------------------------------

def lead_columns(nwp_df: pd.DataFrame, lead_day: int) -> list[str]:
    """Forecast columns belonging to exactly this lead day."""
    suffix = f"_d{lead_day}"
    return [c for c in nwp_df.columns if c.endswith(suffix)]


def available_leads(nwp_df: pd.DataFrame) -> list[int]:
    """Lead days present in the frame, ascending."""
    return sorted({
        int(m.group(1)) for c in nwp_df.columns
        if (m := _LEAD_RE.search(c)) is not None
    })


def models_present(cols: list[str], lead_day: int) -> list[str]:
    """NWP model names contributing 100 m wind at this lead.

    Used to record *which* ensemble a lead actually had. ICON stops at d6, so
    the day-7 model runs on two members rather than three — a fact that must
    travel with any day-7 number rather than being discovered later.
    """
    tag = f"_wind_speed_100m_d{lead_day}"
    return sorted(c[len("nwp_"):-len(tag)] for c in cols if c.endswith(tag))


def build_valid_time_frame(
    weather: pd.DataFrame,
    nwp_df: pd.DataFrame,
    lead_day: int,
    config: Config,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, model_names) for one lead day, indexed by valid time τ.

    There is **no horizon shift here at all** — that is the point of the
    reframing. ``nwp_*_d{N}[τ]`` is already the forecast *for* τ, issued 24·N
    hours earlier, and the target is the observation *at* τ. Both sides share an
    index, so the class of alignment bug that cost this project two iterations
    cannot occur.
    """
    cols = lead_columns(nwp_df, lead_day)
    if not cols:
        raise ValueError(f"No forecast columns for lead day {lead_day}.")

    hub = extrapolate_hub_wind(
        weather, config.site.hub_height_m,
        reference_height_m=source_wind_height_m(config.data.source),
    )

    X = nwp_df[cols].copy()

    # Calendar of the VALID time — the diurnal and seasonal cycle of the hour
    # being predicted, which is legitimately known however far ahead it is.
    idx = X.index
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    # Inter-model disagreement: free, and the only uncertainty signal available.
    # Wide spread means the atmosphere was hard to pin down for that hour.
    wind_cols = [c for c in cols if c.endswith(f"wind_speed_100m_d{lead_day}")]
    if len(wind_cols) > 1:
        X["nwp_spread"] = nwp_df[wind_cols].std(axis=1)
        X["nwp_ensemble_mean"] = nwp_df[wind_cols].mean(axis=1)

    combined = X.copy()
    combined["_target"] = hub.reindex(X.index)
    # Drop all-NaN columns first (e.g. ECMWF serves no gusts here) so they do
    # not take every row with them.
    all_nan = [c for c in X.columns if combined[c].isna().all()]
    if all_nan:
        combined = combined.drop(columns=all_nan)
    combined = combined.dropna()

    feature_cols = [c for c in combined.columns if c != "_target"]
    return (combined[feature_cols],
            combined["_target"].rename("wind_speed_hub"),
            models_present(cols, lead_day))


# ---------------------------------------------------------------------------
# Baselines appropriate to this range
# ---------------------------------------------------------------------------

def climatology_predict(y_train: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Training mean by (month, hour-of-day).

    **This is the reference baseline for multi-day forecasting**, not
    persistence. Measured at 24/48/72 h, climatology moves only +2.6% while
    persistence degrades +20.4%, and the two cross over between 24 h and 48 h.
    Quoting persistence skill at weekly range would flatter the model by ~20%.
    """
    key = pd.MultiIndex.from_arrays(
        [y_train.index.month, y_train.index.hour], names=["m", "h"]
    )
    lookup = y_train.groupby(key).mean()
    table = {(m, h): float(v) for (m, h), v in lookup.items()}
    fallback = float(y_train.mean())
    return np.array([table.get((m, h), fallback)
                     for m, h in zip(index.month, index.hour)], dtype=float)


def ensemble_mean_predict(X_test: pd.DataFrame) -> np.ndarray | None:
    """Raw multi-model mean, uncorrected — the 'no statistics at all' bar."""
    if "nwp_ensemble_mean" not in X_test.columns:
        return None
    return X_test["nwp_ensemble_mean"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Per-lead training
# ---------------------------------------------------------------------------

@dataclass
class LeadResult:
    lead_day: int
    n_models: int
    models: list[str]
    n_features: int
    n_train: int
    n_test: int
    scores: dict[str, float] = field(default_factory=dict)


def evaluate_lead(X: pd.DataFrame, y: pd.Series, config: Config,
                  lead_day: int, models: list[str],
                  test_split: float = 0.2, seed: int = 42) -> LeadResult:
    """Chronological holdout for one lead day: MOS, gradient boosting, baselines."""
    from sklearn.linear_model import LinearRegression

    from .stage1_windspeed import _make_regressor

    set_seed(seed)
    s = int(len(X) * (1 - test_split))
    X_tr, X_te = X.iloc[:s], X.iloc[s:]
    y_tr, y_te = y.iloc[:s], y.iloc[s:]
    yt = y_te.to_numpy(dtype=float)

    res = LeadResult(lead_day=lead_day, n_models=len(models), models=models,
                     n_features=X.shape[1], n_train=len(X_tr), n_test=len(X_te))

    res.scores["climatology"] = compute_metrics(
        yt, climatology_predict(y_tr, X_te.index)).mae

    raw = ensemble_mean_predict(X_te)
    if raw is not None:
        res.scores["ensemble_mean_raw"] = compute_metrics(yt, raw).mae

    wind_cols = [c for c in X.columns
                 if c.endswith(f"wind_speed_100m_d{lead_day}")]
    if wind_cols:
        corrs = {c: abs(X_tr[c].corr(y_tr)) for c in wind_cols}
        best = max(corrs, key=corrs.get)
        m = LinearRegression().fit(X_tr[[best]], y_tr)
        res.scores["mos_single"] = compute_metrics(yt, m.predict(X_te[[best]])).mae

    m = LinearRegression().fit(X_tr, y_tr)
    res.scores["mos_multi"] = compute_metrics(yt, m.predict(X_te)).mae

    gbm = _make_regressor(config.model.stage1_algo, seed)
    gbm.fit(X_tr, y_tr)
    res.scores["xgboost"] = compute_metrics(yt, gbm.predict(X_te)).mae
    return res


def run(config_path: str, test_split: float = 0.2, seed: int = 42) -> pd.DataFrame:
    config = load_config(config_path)
    weather = load_weather(config)
    nwp = load_nwp_forecasts(config)
    if nwp.empty:
        raise SystemExit("No NWP data — set nwp.enabled: true in the config.")

    leads = config.raw.get("weekly", {}).get("lead_days") or available_leads(nwp)
    rows = []
    for lead in leads:
        try:
            X, y, models = build_valid_time_frame(weather, nwp, lead, config)
        except ValueError as exc:
            logger.warning("lead d%d skipped: %s", lead, exc)
            continue
        r = evaluate_lead(X, y, config, lead, models, test_split, seed)
        logger.info(
            "d%d (%dh) | %d ensemble members %s | %d feats | clim=%.4f "
            "raw=%.4f mos_single=%.4f mos_multi=%.4f xgb=%.4f",
            lead, lead * 24, r.n_models, ",".join(r.models), r.n_features,
            r.scores.get("climatology", float("nan")),
            r.scores.get("ensemble_mean_raw", float("nan")),
            r.scores.get("mos_single", float("nan")),
            r.scores.get("mos_multi", float("nan")),
            r.scores.get("xgboost", float("nan")),
        )
        rows.append(dict(lead_day=lead, horizon_h=lead * 24,
                         members=r.n_models, models="+".join(r.models),
                         n_features=r.n_features, n_train=r.n_train,
                         n_test=r.n_test, **r.scores))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["skill_vs_clim"] = 1.0 - df["xgboost"] / df["climatology"]
        out = ensure_dir("outputs/reports") / "weekly_leads.csv"
        df.to_csv(out, index=False)
        logger.info("Wrote %s", out)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-lead-day weekly forecasting.")
    ap.add_argument("--config", default="config_weekly.yaml")
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = run(args.config, args.test_split, args.seed)
    if df.empty:
        raise SystemExit("No leads evaluated.")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
