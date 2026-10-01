"""Head-to-head bakeoff: classical vs ML vs deep, on one protocol.

Why this exists
---------------
Every previous comparison in this project has a flaw the results table did not
carry: a single seasonally-biased split, one seed, no significance test, and
naive persistence as the only classical opponent. This module runs every
predictor through *identical* folds, seeds, features and objective, so the
resulting ranking means something.

Protocol (fixed before running, per docs/12 and redo/05):

- **Rolling-origin folds.** Expanding train, successive test blocks, so every
  season appears in test. The old single split covered 2025-08-20 -> 2025-12-30
  only — four months of one season, a thin basis for ranking a dozen predictors.
- **Seeds.** Deterministic predictors run once (repeating them would fake a
  precision they do not have); stochastic ones run over the seed list. Reported
  as mean +/- sd, never best-of-N.
- **Identical everything else.** Same folds, same feature masks, same target.
- **Skill score vs persistence** alongside raw MAE, because folds differ in
  difficulty and raw MAE is not comparable across them.
- **Diebold-Mariano** for pairwise significance. Without it, "A beats B by 0.15"
  is an observation, not a result.

Feature sets
------------
``study``        every live feature (~104). What the project has always measured.
``operational``  NWP + calendar only (~55). **Never measured before.** ERA5 is
                 reanalysis with ~5-day latency, so at the moment a forecast for
                 t+24h would actually be issued, none of the 49 reanalysis-derived
                 features exist. This arm is what a deployed system would score.

Usage
-----
    python -m src.bakeoff --config config_bakeoff.yaml                  # phase A
    python -m src.bakeoff --config config_bakeoff.yaml --features study
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data_ingest import load_nwp_forecasts, load_weather, source_wind_height_m
from .features import build_feature_frame, extrapolate_hub_wind
from .stage1_windspeed import compute_metrics
from .traditional import (
    CALENDAR_COLS,
    DETERMINISTIC,
    FoldContext,
    build_registry,
    is_dead_column,
)
from .utils import Config, ensure_dir, get_logger, load_config, set_seed

logger = get_logger(__name__)

DEFAULT_SEEDS = [42, 7, 1234, 2024, 31337]
DEFAULT_N_FOLDS = 4
#: Fraction of the record reserved for the rolling-origin test blocks. The first
#: fold therefore trains on 1 - TEST_SPAN and each later fold gains one block.
TEST_SPAN = 0.56


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    name: str
    train_end: int
    test_end: int

    @property
    def test_start(self) -> int:
        return self.train_end


def make_rolling_folds(n_rows: int, n_folds: int = DEFAULT_N_FOLDS,
                       test_span: float = TEST_SPAN) -> list[Fold]:
    """Expanding-origin folds covering the last `test_span` of the record.

    Chronological and never shuffled. Each fold trains on everything before its
    test block, so no future information reaches the model.
    """
    block = int(n_rows * test_span / n_folds)
    first_train_end = n_rows - block * n_folds
    if first_train_end < block:
        raise ValueError(
            f"Not enough rows ({n_rows}) for {n_folds} folds at span {test_span}."
        )
    folds = []
    for i in range(n_folds):
        train_end = first_train_end + i * block
        test_end = train_end + block
        folds.append(Fold(name=f"fold{i + 1}", train_end=train_end, test_end=test_end))
    return folds


# ---------------------------------------------------------------------------
# Feature masks
# ---------------------------------------------------------------------------

def select_features(cols: list[str], feature_set: str) -> list[str]:
    """Apply the study/operational mask and drop the degenerate spatial columns."""
    live = [c for c in cols if not is_dead_column(c)]
    if feature_set == "study":
        return live
    if feature_set == "operational":
        return [c for c in live if c.startswith("nwp_") or c in CALENDAR_COLS]
    raise ValueError(f"Unknown feature set: {feature_set!r}")


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray,
                    horizon: int = 24) -> tuple[float, float]:
    """DM test on absolute-error differentials. Returns (statistic, p-value).

    Negative statistic => A has the lower loss. The HAC (Newey-West) variance with
    `horizon - 1` lags is what makes this valid for overlapping multi-step
    forecasts, whose errors are serially correlated by construction; a plain
    t-test here would badly overstate significance.
    """
    d = np.abs(errors_a) - np.abs(errors_b)
    n = len(d)
    d_bar = float(d.mean())
    d_dev = d - d_bar

    gamma0 = float(d_dev @ d_dev) / n
    var = gamma0
    for lag in range(1, min(horizon, n - 1)):
        cov = float(d_dev[lag:] @ d_dev[:-lag]) / n
        var += 2.0 * (1.0 - lag / horizon) * cov
    if var <= 0:
        return float("nan"), float("nan")

    stat = d_bar / np.sqrt(var / n)
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    model: str
    tier: str
    feature_set: str
    fold: str
    seed: int
    mae: float
    rmse: float
    r2: float
    skill: float
    seconds: float
    horizon: int = 24
    errors: np.ndarray = field(repr=False, default=None)


def _hub_series(weather: pd.DataFrame, config: Config) -> pd.Series:
    return extrapolate_hub_wind(
        weather, config.site.hub_height_m,
        reference_height_m=source_wind_height_m(config.data.source),
    )


def run_phase_a(config: Config, feature_sets: list[str], seeds: list[int],
                n_folds: int) -> list[RunRecord]:
    horizon = int(config.forecast.horizon_hours)
    weather = load_weather(config)
    nwp = load_nwp_forecasts(config)
    if nwp.empty:
        raise SystemExit("No NWP data — set nwp.enabled: true in the config.")

    X_all, y_all, cols = build_feature_frame(
        weather, config, mode="forecast", nwp_df=nwp
    )
    hub_all = _hub_series(weather, config).reindex(X_all.index)
    nwp_cols_all = [c for c in cols if c.startswith("nwp_")]

    folds = make_rolling_folds(len(X_all), n_folds)
    logger.info(
        "horizon %d h | feature frame: %d rows x %d cols (%d NWP); %d rolling folds",
        horizon, len(X_all), len(cols), len(nwp_cols_all), len(folds),
    )
    for f in folds:
        logger.info(
            "  %s: train %d rows -> test %d rows [%s .. %s]",
            f.name, f.train_end, f.test_end - f.test_start,
            X_all.index[f.test_start].date(), X_all.index[f.test_end - 1].date(),
        )

    records: list[RunRecord] = []
    for feature_set in feature_sets:
        keep = select_features(cols, feature_set)
        X = X_all[keep]
        nwp_cols = [c for c in keep if c.startswith("nwp_")]
        logger.info("=== feature set '%s': %d columns ===", feature_set, len(keep))

        for fold in folds:
            tr = slice(0, fold.train_end)
            te = slice(fold.test_start, fold.test_end)
            base_ctx = dict(
                X_train=X.iloc[tr], y_train=y_all.iloc[tr],
                X_test=X.iloc[te], y_test=y_all.iloc[te],
                hub_train=hub_all.iloc[tr], hub_test=hub_all.iloc[te],
                nwp_cols=nwp_cols,
            )
            y_te = y_all.iloc[te].to_numpy(dtype=float)

            # Persistence defines the skill-score denominator for this fold.
            pers_mae = float(np.mean(np.abs(y_te - hub_all.iloc[te].to_numpy())))

            for model in build_registry():
                model_seeds = [seeds[0]] if model.name in DETERMINISTIC else seeds
                for seed in model_seeds:
                    set_seed(seed)
                    ctx = FoldContext(seed=seed, **base_ctx)
                    t0 = time.time()
                    try:
                        pred = model.fit_predict(ctx)
                    except Exception as exc:  # a failing model must not kill the run
                        logger.warning("%s failed on %s/%s: %s",
                                       model.name, feature_set, fold.name, exc)
                        continue
                    elapsed = time.time() - t0
                    m = compute_metrics(y_te, pred)
                    records.append(RunRecord(
                        model=model.name, tier=model.tier, feature_set=feature_set,
                        fold=fold.name, seed=seed, mae=m.mae, rmse=m.rmse, r2=m.r2,
                        skill=1.0 - m.mae / pers_mae, seconds=elapsed,
                        horizon=horizon, errors=y_te - pred,
                    ))
                    logger.info(
                        "h=%-3d %-18s %-12s %-6s seed=%-5d MAE=%.4f R2=%+.4f "
                        "skill=%+.3f (%.1fs)",
                        horizon, model.name, feature_set, fold.name, seed,
                        m.mae, m.r2, records[-1].skill, elapsed,
                    )
    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(records: list[RunRecord]) -> pd.DataFrame:
    """Mean +/- sd across folds and seeds, per model and feature set."""
    df = pd.DataFrame([
        dict(model=r.model, tier=r.tier, feature_set=r.feature_set,
             horizon=r.horizon, fold=r.fold, seed=r.seed, mae=r.mae,
             rmse=r.rmse, r2=r.r2, skill=r.skill, seconds=r.seconds)
        for r in records
    ])
    if df.empty:
        return df
    agg = df.groupby(["tier", "model", "feature_set", "horizon"]).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
        skill_mean=("skill", "mean"), skill_sd=("skill", "std"),
        n_runs=("mae", "size"), seconds=("seconds", "sum"),
    ).reset_index()
    return agg.sort_values(["feature_set", "mae_mean"])


def render_table(agg: pd.DataFrame, feature_set: str) -> str:
    sub = agg[agg.feature_set == feature_set]
    horizon = int(sub.horizon.iloc[0]) if len(sub) else 0
    lines = [
        f"### {feature_set} — horizon {horizon} h",
        "",
        "| Rank | Tier | Model | MAE (m/s) | sd | RMSE | R² | Skill vs persistence |",
        "|---:|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(sub.itertuples(), start=1):
        sd = "—" if pd.isna(row.mae_sd) else f"{row.mae_sd:.4f}"
        lines.append(
            f"| {i} | {row.tier} | `{row.model}` | **{row.mae_mean:.4f}** | {sd} | "
            f"{row.rmse_mean:.4f} | {row.r2_mean:.4f} | {row.skill_mean:+.3f} |"
        )
    return "\n".join(lines)


def significance_matrix(records: list[RunRecord], feature_set: str,
                        horizon: int) -> str:
    """Pairwise DM tests against the best model on pooled fold errors."""
    # Deterministic models produce one error vector per fold; stochastic ones
    # produce one per seed. Averaging across seeds *within* each fold, then
    # concatenating folds in chronological order, gives every model an error
    # vector of identical length covering identical timestamps — without which
    # the test silently compares different periods.
    by_model_fold: dict[str, dict[str, list[np.ndarray]]] = {}
    for r in records:
        if r.feature_set != feature_set or r.errors is None:
            continue
        by_model_fold.setdefault(r.model, {}).setdefault(r.fold, []).append(r.errors)
    if not by_model_fold:
        return ""

    fold_order = sorted({r.fold for r in records if r.feature_set == feature_set})
    pooled: dict[str, np.ndarray] = {}
    for model, folds in by_model_fold.items():
        if set(folds) != set(fold_order):
            continue  # a model that failed a fold cannot be compared fairly
        pooled[model] = np.concatenate(
            [np.mean(np.vstack(folds[f]), axis=0) for f in fold_order]
        )
    if not pooled:
        return ""
    assert len({len(v) for v in pooled.values()}) == 1, "DM error vectors misaligned"

    best = min(pooled, key=lambda m: float(np.mean(np.abs(pooled[m]))))
    lines = [
        "",
        f"### Diebold–Mariano vs `{best}` ({feature_set}, horizon {horizon} h)",
        "",
        "Negative statistic means the row model has **lower** loss than the "
        f"reference. `p < 0.05` is a real difference in forecast accuracy.",
        "",
        "| Model | DM statistic | p-value | verdict |",
        "|:---|---:|---:|:---|",
    ]
    for m, err in sorted(pooled.items(), key=lambda kv: np.mean(np.abs(kv[1]))):
        if m == best:
            continue
        stat, p = diebold_mariano(pooled[best], err, horizon=horizon)
        verdict = "tie" if (np.isnan(p) or p >= 0.05) else f"`{best}` better"
        lines.append(f"| `{m}` | {stat:+.3f} | {p:.4f} | {verdict} |")
    return "\n".join(lines)


def run_phase_b(config: Config, algos: list[str], seeds: list[int],
                feature_sets: list[str]) -> pd.DataFrame:
    """Deep models on the single chronological split, several seeds.

    **A different evaluation basis from phase A**, and that must travel with the
    numbers: one split instead of four rolling folds, and 3 seeds instead of 5.
    The reason is cost — TFT alone took 5 h 54 m for a single fit, so four
    rolling folds would be ~24 h for that model per seed. XGBoost is re-run here
    on the *same* single split so the deep scores have a like-for-like anchor
    rather than being silently compared against phase A's fold means.
    """
    from .stage1_deep import resolve_deep_cfg, train_stage1_deep
    from .stage1_windspeed import train_stage1

    weather = load_weather(config)
    nwp = load_nwp_forecasts(config)
    X_all, y_all, cols = build_feature_frame(
        weather, config, mode="forecast", nwp_df=nwp
    )

    # Assert the resolved deep config before committing to an overnight run.
    # A config with no `deep:` block silently gives TFT 200 epochs instead of 60.
    for algo in algos:
        if algo in {"xgboost", "lightgbm"}:
            continue
        cfg = resolve_deep_cfg(config, algo)
        logger.info("resolved %s: n_epochs=%s loss_fn=%s",
                    algo, cfg.get("n_epochs"), cfg.get("loss_fn"))
        assert cfg.get("loss_fn") == "mse", f"{algo} is not on the MSE objective"
        if algo == "tft":
            assert int(cfg.get("n_epochs")) == 60, "TFT epoch budget is not 60"

    rows = []
    for feature_set in feature_sets:
        keep = select_features(cols, feature_set)
        X = X_all[keep]
        for algo in algos:
            for seed in seeds:
                config.model.stage1_algo = algo
                config.model.seed = seed
                set_seed(seed)
                t0 = time.time()
                try:
                    if algo in {"xgboost", "lightgbm"}:
                        res = train_stage1(X, y_all, config, mode="forecast")
                    else:
                        res = train_stage1_deep(X, y_all, config, mode="forecast",
                                                weather_df=weather)
                except Exception as exc:
                    logger.warning("%s/%s seed=%d failed: %s",
                                   algo, feature_set, seed, exc)
                    continue
                m = res.metrics_test
                p = res.metrics_persistence
                rows.append(dict(
                    model=algo, feature_set=feature_set, seed=seed,
                    mae=m.mae, rmse=m.rmse, r2=m.r2,
                    skill=1.0 - m.mae / p.mae, minutes=(time.time() - t0) / 60,
                ))
                logger.info("PHASE-B %-9s %-12s seed=%-5d MAE=%.4f R2=%+.4f (%.1f min)",
                            algo, feature_set, seed, m.mae, m.r2, rows[-1]["minutes"])
                pd.DataFrame(rows).to_csv(
                    ensure_dir("outputs/reports") / "bakeoff_deep.csv", index=False
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Traditional vs ML vs deep bakeoff.")
    ap.add_argument("--config", default="config_bakeoff.yaml")
    ap.add_argument("--phase", default="a", choices=["a", "b"])
    ap.add_argument("--features", default="both",
                    choices=["study", "operational", "both"])
    ap.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--horizon", type=int, default=None,
                    help="Override forecast.horizon_hours. Leads shorter than "
                         "the horizon are dropped by the lead guard.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-fold", action="store_true",
                    help="Also write {stem}_folds.csv, one row per "
                         "(model, feature_set, fold, seed). Phase A only.")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.horizon is not None:
        config.forecast.horizon_hours = int(args.horizon)
    bk = config.raw.get("bakeoff", {})
    seeds = args.seeds or bk.get("seeds", DEFAULT_SEEDS)
    feature_sets = (["study", "operational"] if args.features == "both"
                    else [args.features])

    if args.phase == "b":
        deep_seeds = args.seeds or bk.get("deep_seeds", [42, 7, 1234])
        algos = ["xgboost"] + [
            a for a in config.model.compare_algos
            if a not in {"xgboost", "lightgbm"}
        ]
        t0 = time.time()
        df = run_phase_b(config, algos, deep_seeds, feature_sets)
        logger.info("Phase B done in %.1f h", (time.time() - t0) / 3600)
        print(df.to_string(index=False))
        return

    t0 = time.time()
    records = run_phase_a(config, feature_sets, seeds, args.folds)
    agg = summarise(records)

    horizon = int(config.forecast.horizon_hours)
    parts = [
        f"# Bakeoff — classical vs ML, horizon {horizon} h",
        "",
        f"- window: **{config.data.start} → {config.data.end}** "
        f"({config.data.source} truth), horizon **{horizon} h**",
        f"- folds: **{args.folds} rolling-origin**, seeds: **{seeds}** "
        "(deterministic models run once per fold)",
        "- lower MAE is better; skill is `1 − MAE/MAE_persistence`, higher better",
        f"- **NWP lead guard active:** only `_d{{N}}` columns with `24·N ≥ {horizon}` "
        "are used, so every forecast feature was issued at or before the feature time",
        "",
    ]
    for fs in feature_sets:
        parts.append(render_table(agg, fs))
        parts.append(significance_matrix(records, fs, horizon))
        parts.append("")

    # Horizon-tagged filenames so a sweep does not overwrite its own earlier runs.
    stem = "bakeoff" if horizon == 24 else f"bakeoff_h{horizon}"
    reports = ensure_dir("outputs/reports")
    out = Path(args.out) if args.out else reports / f"{stem}.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    agg.to_csv(reports / f"{stem}.csv", index=False)

    if args.per_fold:
        # `summarise` averages folds away, which hides that fold difficulty varies
        # by ~46% across the year — the fact that motivates rolling origins at all.
        # RunRecord already carries `fold`, so this is an output path, not new work.
        pd.DataFrame([{
            "model": r.model, "tier": r.tier, "feature_set": r.feature_set,
            "fold": r.fold, "seed": r.seed, "horizon": r.horizon,
            "mae": r.mae, "rmse": r.rmse, "r2": r.r2, "skill": r.skill,
        } for r in records]).to_csv(reports / f"{stem}_folds.csv", index=False)
        logger.info("Wrote %s_folds.csv (%d rows)", stem, len(records))

    logger.info("Wrote %s and %s.csv (%.1f min)", out, stem, (time.time() - t0) / 60)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
