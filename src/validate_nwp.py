"""Verification gates for the NWP-forecast feature work (docs/11 §7).

Four checks that must pass, in order, before any result from the NWP iteration
can be believed. Each has a pass criterion fixed *before* the data was seen, and
a defined response to failure — the point is to decide in advance what would
count as evidence that something is broken, since this project has already spent
two iterations trusting plausible wrong numbers (docs/10).

    python -m src.validate_nwp --config config_nwp_pilot.yaml   # NASA POWER pilot
    python -m src.validate_nwp --config config_nwp.yaml         # ERA5, the real run

The same code serves both, which is the point: the pilot exercises the gates on
live data today, and the authoritative run reuses them unchanged.

Gates
-----
1. lag-peak   — is the forecast joined to the right hour?
2. raw skill  — does the forecast alone beat persistence, after bias correction?
3. baselines  — what are the legitimate comparators on *this* window?
4. experiment — does adding the NWP block actually help?

Exit code is non-zero if a gate fails, so this can front a longer pipeline run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .data_ingest import load_nwp_forecasts, load_weather, nwp_column
from .features import build_feature_frame, extrapolate_hub_wind
from .stage1_windspeed import compute_metrics
from .utils import Config, get_logger, load_config, set_seed

logger = get_logger("validate_nwp")

# Same hyperparameters in both arms of gate 4 — otherwise the comparison measures
# tuning rather than the feature block. These are the v3 Optuna values.
XGB_PARAMS = dict(
    n_estimators=450, max_depth=4, learning_rate=0.0185, min_child_weight=16,
    subsample=0.63, colsample_bytree=0.65, reg_alpha=0.002, reg_lambda=0.56,
    n_jobs=-1, tree_method="hist",
)

MAX_ACCEPTABLE_LAG_H = 1   # gate 1: ECMWF genuinely leads the others ~1 h (docs/11 §6)
LAG_SEARCH_H = 48


@dataclass
class GateResult:
    number: int
    name: str
    passed: bool
    detail: str
    rows: list[tuple] = field(default_factory=list)
    headers: tuple = ()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _hub_truth(weather: pd.DataFrame, config: Config) -> pd.Series:
    """Observed hub-height wind speed — the target these gates score against."""
    from .data_ingest import source_wind_height_m
    return extrapolate_hub_wind(
        weather, config.site.hub_height_m,
        reference_height_m=source_wind_height_m(config.data.source),
    )


def _cross_correlation_peak(a: pd.Series, b: pd.Series,
                            span: int = LAG_SEARCH_H) -> tuple[int, float, float]:
    """Return (peak_lag_hours, r_at_peak, r_at_lag_0) for a vs b.shift(lag)."""
    scores = {}
    for lag in range(-span, span + 1):
        r = a.corr(b.shift(lag))
        if pd.notna(r):
            scores[lag] = float(r)
    if not scores:
        return 0, float("nan"), float("nan")
    peak = max(scores, key=scores.get)
    return peak, scores[peak], scores.get(0, float("nan"))


def _fit_predict_xgb(X_tr, y_tr, X_te, seed: int):
    import xgboost as xgb
    model = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _split_index(n: int, test_split: float) -> int:
    return int(n * (1.0 - test_split))


# ---------------------------------------------------------------------------
# Gate 1 — lag-peak against truth
# ---------------------------------------------------------------------------

def gate_1_lag_peak(nwp: pd.DataFrame, truth_ref: pd.Series) -> GateResult:
    """Each model's day-ahead wind forecast must peak against truth at |lag| <= 1 h.

    Failure modes are distinct and must not be conflated:
      * peak at +/-24 h -> the horizon shift is wrong; fix the code, not the test
      * a constant 2-3 h offset on one model -> a real convention difference;
        document and correct it explicitly rather than absorbing it silently
      * no clear peak -> units, coordinates, or the wrong variable entirely
    """
    rows, worst = [], 0
    for col in [c for c in nwp.columns if c.endswith("_wind_speed_100m_d1")]:
        model = col[len("nwp_"):-len("_wind_speed_100m_d1")]
        lag, r_peak, r0 = _cross_correlation_peak(nwp[col], truth_ref)
        ok = abs(lag) <= MAX_ACCEPTABLE_LAG_H
        worst = max(worst, abs(lag))
        rows.append((model, lag, round(r_peak, 4), round(r0, 4), "PASS" if ok else "FAIL"))
    passed = bool(rows) and all(r[-1] == "PASS" for r in rows)
    if not rows:
        return GateResult(1, "lag-peak vs truth", False,
                          "no `*_wind_speed_100m_d1` columns found — nothing to check")
    detail = (f"worst |peak lag| = {worst} h (allowed <= {MAX_ACCEPTABLE_LAG_H} h)"
              if passed else
              f"worst |peak lag| = {worst} h EXCEEDS {MAX_ACCEPTABLE_LAG_H} h — "
              "join is wrong, do not proceed")
    return GateResult(1, "lag-peak vs truth", passed, detail, rows,
                      ("model", "peak lag (h)", "r@peak", "r@0", ""))


# ---------------------------------------------------------------------------
# Gate 2 — raw forecast skill, after bias correction
# ---------------------------------------------------------------------------

def gate_2_raw_skill(nwp: pd.DataFrame, hub: pd.Series, horizon: int) -> GateResult:
    """The forecast alone, no ML, must beat 24 h persistence.

    Scored on a single-parameter linear rescale, NOT on raw values. The models
    carry large mean biases (docs/11 §6), so a raw comparison can lose to
    persistence while the alignment is perfectly correct. Both are reported; the
    rescaled figure is the alignment test, the raw one documents the bias.
    """
    target = hub.shift(-horizon).dropna()
    persistence = hub.reindex(target.index)
    pm = compute_metrics(target.to_numpy(), persistence.to_numpy())

    rows = []
    best_ok = False
    for col in [c for c in nwp.columns if c.endswith("_wind_speed_100m_d1")]:
        model = col[len("nwp_"):-len("_wind_speed_100m_d1")]
        fc = nwp[col].reindex(target.index + pd.Timedelta(hours=horizon))
        fc.index = target.index
        mask = fc.notna() & target.notna()
        if mask.sum() < 100:
            rows.append((model, "-", "-", "-", "SKIP (too few overlapping hours)"))
            continue
        t, f = target[mask], fc[mask]
        raw = compute_metrics(t.to_numpy(), f.to_numpy())
        lr = LinearRegression().fit(f.to_numpy().reshape(-1, 1), t.to_numpy())
        adj = compute_metrics(t.to_numpy(), lr.predict(f.to_numpy().reshape(-1, 1)))
        pm_masked = compute_metrics(t.to_numpy(), persistence[mask].to_numpy())
        ok = adj.mae < pm_masked.mae
        best_ok = best_ok or ok
        rows.append((model, f"{raw.mae:.4f}", f"{adj.mae:.4f}",
                     f"{pm_masked.mae:.4f}", "PASS" if ok else "FAIL"))
    detail = (f"persistence MAE {pm.mae:.4f} m/s; at least one model beats it after "
              "bias correction" if best_ok else
              "NO model beats persistence after bias correction — alignment or units "
              "are wrong, stop and fix before modelling")
    return GateResult(2, "raw forecast skill", best_ok, detail, rows,
                      ("model", "raw MAE", "rescaled MAE", "persistence MAE", ""))


# ---------------------------------------------------------------------------
# Gate 3 — baselines on this window
# ---------------------------------------------------------------------------

def gate_3_baselines(X: pd.DataFrame, y: pd.Series, hub: pd.Series,
                     config: Config) -> GateResult:
    """Legitimate comparators for *this* window. Never quote another window's."""
    s = _split_index(len(X), config.model.test_split)
    y_tr, y_te = y.iloc[:s], y.iloc[s:]
    h = hub.reindex(X.index)
    h_tr, h_te = h.iloc[:s].to_numpy().reshape(-1, 1), h.iloc[s:].to_numpy().reshape(-1, 1)

    rows = []
    naive = compute_metrics(y_te.to_numpy(), h_te.ravel())
    rows.append(("persistence  hub(t) -> hub(t+h)", f"{naive.mae:.4f}", f"{naive.r2:.4f}"))

    ols = LinearRegression().fit(h_tr, y_tr.to_numpy())
    m = compute_metrics(y_te.to_numpy(), ols.predict(h_te))
    rows.append(("OLS on hub(t)  [1 parameter]", f"{m.mae:.4f}", f"{m.r2:.4f}"))

    clim = compute_metrics(y_te.to_numpy(), np.full(len(y_te), y_tr.mean()))
    rows.append(("climatology (train mean)", f"{clim.mae:.4f}", f"{clim.r2:.4f}"))

    no_nwp = [c for c in X.columns if not c.startswith("nwp_")]
    pred = _fit_predict_xgb(X.iloc[:s][no_nwp], y_tr, X.iloc[s:][no_nwp], config.model.seed)
    m = compute_metrics(y_te.to_numpy(), pred)
    rows.append((f"XGBoost WITHOUT nwp ({len(no_nwp)} feats)", f"{m.mae:.4f}", f"{m.r2:.4f}"))

    band = 1.2 <= naive.mae <= 1.6
    detail = (f"persistence {naive.mae:.4f} m/s is inside the 1.2-1.6 sanity band"
              if band else
              f"persistence {naive.mae:.4f} m/s is OUTSIDE the 1.2-1.6 band seen on "
              "2020-2023 — this window has a different wind climatology, note it before "
              "comparing anything")
    return GateResult(3, "baselines on this window", True, detail, rows,
                      ("predictor", "test MAE", "test R2"))


# ---------------------------------------------------------------------------
# Gate 4 — the experiment
# ---------------------------------------------------------------------------

def gate_4_experiment(X: pd.DataFrame, y: pd.Series, config: Config) -> GateResult:
    """XGBoost with vs without the NWP block, identical split and hyperparameters."""
    s = _split_index(len(X), config.model.test_split)
    y_te = y.iloc[s:]
    no_nwp = [c for c in X.columns if not c.startswith("nwp_")]
    with_nwp = list(X.columns)
    if len(with_nwp) == len(no_nwp):
        return GateResult(4, "with vs without NWP", False,
                          "no nwp_* columns in the feature frame — nothing to compare")

    a = compute_metrics(y_te.to_numpy(),
                        _fit_predict_xgb(X.iloc[:s][no_nwp], y.iloc[:s],
                                         X.iloc[s:][no_nwp], config.model.seed))
    b = compute_metrics(y_te.to_numpy(),
                        _fit_predict_xgb(X.iloc[:s][with_nwp], y.iloc[:s],
                                         X.iloc[s:][with_nwp], config.model.seed))
    delta = a.mae - b.mae
    pct = delta / a.mae * 100.0
    band = "LARGE (>=20%)" if pct >= 20 else "MODEST (5-20%)" if pct >= 5 else "NONE (<5%)"
    rows = [
        (f"without NWP ({len(no_nwp)} feats)", f"{a.mae:.4f}", f"{a.r2:.4f}"),
        (f"with NWP ({len(with_nwp)} feats)", f"{b.mae:.4f}", f"{b.r2:.4f}"),
        ("improvement", f"{delta:+.4f} ({pct:+.1f}%)", f"{b.r2 - a.r2:+.4f}"),
    ]
    return GateResult(4, "with vs without NWP", True,
                      f"outcome band: {band} — see docs/11 §7b for the pre-registered "
                      "response to each band", rows,
                      ("model", "test MAE", "test R2"))


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _render(g: GateResult) -> None:
    status = "PASS" if g.passed else "FAIL"
    print(f"\n{'='*74}\nGATE {g.number}: {g.name}   [{status}]\n{'='*74}")
    if g.rows:
        widths = [max(len(str(r[i])) for r in ([g.headers] + g.rows) if i < len(r))
                  for i in range(len(g.headers))]
        print("  " + "  ".join(str(h).ljust(w) for h, w in zip(g.headers, widths)))
        print("  " + "  ".join("-" * w for w in widths))
        for r in g.rows:
            print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(f"\n  -> {g.detail}")


def run(config_path: str) -> int:
    config = load_config(config_path)
    set_seed(config.model.seed)
    is_pilot = config.data.source.lower() != "era5"

    print("=" * 74)
    print("NWP VERIFICATION GATES")
    print(f"  config : {config_path}")
    print(f"  truth  : {config.data.source}   window {config.data.start} -> {config.data.end}")
    print(f"  horizon: {config.forecast.horizon_hours} h")
    if is_pilot:
        print("\n  *** PILOT RUN — NASA POWER truth, NOT v5 results. ***")
        print("  Absolute MAE is inflated ~12% by the 50 m -> 100 m extrapolation and is")
        print("  NOT comparable to ERA5 figures. Use R2 and relative improvement only.")
    print("=" * 74)

    weather = load_weather(config)
    nwp = load_nwp_forecasts(config)
    if nwp.empty:
        print("\nNo NWP data — set nwp.enabled: true in the config.")
        return 2

    hub = _hub_truth(weather, config)
    X, y, cols = build_feature_frame(weather, config, mode="forecast", nwp_df=nwp)
    print(f"\nfeature frame: {len(X)} rows x {len(cols)} features "
          f"({sum(c.startswith('nwp_') for c in cols)} of them NWP)")

    gates = [
        gate_1_lag_peak(nwp, hub.reindex(nwp.index)),
        gate_2_raw_skill(nwp, hub, int(config.forecast.horizon_hours)),
    ]
    for g in gates:
        _render(g)
        if not g.passed:
            print(f"\nSTOPPED at gate {g.number}. Later gates are meaningless until this "
                  "passes — fix the cause rather than the criterion.")
            return 1

    for g in (gate_3_baselines(X, y, hub, config), gate_4_experiment(X, y, config)):
        _render(g)
        gates.append(g)

    print(f"\n{'='*74}")
    print("SUMMARY: " + " | ".join(
        f"gate {g.number} {'PASS' if g.passed else 'FAIL'}" for g in gates))
    if is_pilot:
        print("\nPILOT — do not quote these numbers as v5 results, and do not place this")
        print("MAE in a table beside an ERA5 figure. Re-run against config_nwp.yaml once")
        print("the ERA5 download completes.")
    print("=" * 74)
    return 0 if all(g.passed for g in gates) else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Run the NWP verification gates.")
    p.add_argument("--config", default="config_nwp.yaml")
    raise SystemExit(run(p.parse_args().config))


if __name__ == "__main__":
    main()
