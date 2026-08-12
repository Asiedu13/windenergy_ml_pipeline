# 01 — The protocol

> **What:** exactly how every number in this folder was produced.
> **Why it comes first:** the protocol *is* the contribution. The project's earlier
> comparisons used one seasonally-biased split, one seed, no significance test, and naive
> persistence as the only classical opponent. Changing the protocol changed the conclusions
> — see [02](02_classical_vs_ml.md) and [04](04_deep_models.md).

## The horizon: t + 24 h, and only that

Every model here predicts hub-height wind speed **exactly 24 hours ahead**.
`forecast.horizon_hours: 24`; the target is built once as `y[t] = hub(t + 24)`.

Three different things get called "24" in this project and conflating them causes real
confusion:

| Concept | Value | What it actually is |
|:---|:---|:---|
| **Forecast horizon** | **24 h, single** | What is predicted: `hub(t+24)` |
| **NWP lead days** `d1`/`d2`/`d3` | 24 / 48 / 72 h | How *old* the forecast feature is — all three describe the **same valid time t+24** |
| **Feature lags** | 1, 2, 3, 6, 12 h + rolling 3, 6 h | How far *back* the model looks |

**`d2` and `d3` are not extra horizons.** They are the 48- and 72-hour-old forecasts *for
the same target hour*, included so the model can see whether successive forecast runs
agreed — agreement is a confidence signal. All 51 NWP features describe t + 24 h.

**Limitation, stated rather than left implicit:** there is no horizon-vs-error curve. The
`d2`/`d3` columns are on disk and `add_nwp_features` already accepts an `offset_hours`
argument, so a sweep is roughly half a day of work and needs no new data. Until it is run,
nothing here supports a claim about how skill decays with lead time.

**One caveat on the "24 h" label.** Open-Meteo's `previous_day1` is a fixed **day** offset,
not a fixed hour offset, so the true issue-to-valid gap varies through the target day
(roughly 24–47 h). It passed the ±1 h lag-peak gate and behaves consistently with ~24 h, but
has never been verified hour by hour — [`docs/12` item 9](../../docs/12_next_steps_v2.md),
still open.

## The window is fixed by data, not chosen

| Asset | Coverage |
|:---|:---|
| ERA5 truth | 2010-01→2016-07, 2020-01→2023-12, **2024-03→2025-12** |
| Open-Meteo NWP forecasts | **2024-03→2026-06** |

The comparison can only run where truth and forecasts overlap: **2024-03 → 2025-12**,
15,912 rows after feature building. Consequences that constrain every result here:

- **No out-of-period holdout is possible.** There is no second era with forecast data.
- **No test on the 2020–2023 window**, so nothing here is comparable to the v1–v4 tables.
- **The analog ensemble is data-limited.** AnEn wants several years so close analogues exist
  for unusual situations; it gets at most ~13.7k training hours. Its result is a
  data-constrained measurement, not a fair test of the method.

## Rolling-origin folds

Four folds, expanding train, successive ~3-month test blocks — so **every season appears in
test**. The single split it replaces covered 2025-08-20 → 2025-12-30 only.

| Fold | Train rows | Test rows | Test period (24 h horizon) |
|:---|---:|---:|:---|
| fold1 | 7,004 | 2,227 | 2024-12-24 → 2025-03-27 |
| fold2 | 9,231 | 2,227 | 2025-03-27 → 2025-06-28 |
| fold3 | 11,458 | 2,227 | 2025-06-28 → 2025-09-29 |
| fold4 | 13,685 | 2,227 | 2025-09-29 → 2025-12-30 |

Chronological, never shuffled; each fold trains on everything before its test block.

**Why this matters more than it sounds.** Persistence MAE across the four folds ranges
**1.1129 → 1.6274 m/s** — the problem is 46% harder in one quarter than another. Any
single-split ranking is partly a statement about which quarter you happened to test on.

## Seeds

| Model class | Seeds | Rationale |
|:---|:---|:---|
| Deterministic (persistence, damped persistence, climatology, MOS ×2, Kalman, AnEn, SARIMAX, ridge) | 1 | Output does not depend on the seed; repeating would fake a precision they do not have |
| Gradient boosting | 5 — `42, 7, 1234, 2024, 31337` | 20 runs per model per feature set |
| Deep (phase B) | 3 — `42, 7, 1234` | Cost: TFT alone is ~6 h per fit |

Reported as **mean ± sd**, never best-of-N.

## Metrics

- **Primary: MAE (m/s)**, lower better.
- Secondary: RMSE, R².
- **Skill score** = `1 − MAE_model / MAE_persistence`, computed **per fold**. Because fold
  difficulty varies by 46%, raw MAE is not comparable across folds and skill is.

## Significance: Diebold–Mariano

The standard test for comparing *forecast accuracy*. Loss differential `d = |e_A| − |e_B|`,
with a **Newey–West HAC variance using horizon−1 = 23 lags**. The HAC correction is
load-bearing: overlapping 24-hour-ahead forecasts have serially correlated errors by
construction, and a plain t-test would badly overstate significance.

Error vectors are averaged across seeds **within each fold**, then concatenated in
chronological order, so every model is compared on identical timestamps. The first
implementation was not — see [06](06_what_the_protocol_caught.md), defect 2.

## Feature sets

| Set | Columns | Contents |
|:---|---:|:---|
| `study` | 94 | Every live feature — what the project has always measured |
| `operational` | 55 | NWP + calendar only. **No reanalysis**, because ERA5's ~5-day latency means none of it exists at issue time |

**Ten columns are excluded from both** as mathematically dead: the ERA5 request snaps to a
1 lat × 2 lon patch rather than the intended 3×3, so all five `*_grad_ns` columns are
identically zero and all five `*_std` columns are exactly `|*_grad_ew| / 2`. Trees tolerate
them; linear models do not, and neither carries information.

## Comparability rules enforced

- Identical folds, features, target and objective (MSE) for every model.
- No per-model hyperparameter tuning. On this window a 50-trial Optuna sweep and the untuned
  defaults both scored 0.6780 m/s — identical to four decimals — so equal documented
  settings matter more than optimal ones.
- The resolved deep config is printed and asserted before any run over an hour
  (`n_epochs == 60` and `loss_fn == "mse"` for TFT).

## Reproduction

```bash
# Phase A — 11 predictors, 4 folds, both feature sets (~3 min)
python -m src.bakeoff --config config_bakeoff.yaml --features both

# Phase B — deep models, single split, 3 seeds (~19 h, TFT dominates)
python -m src.bakeoff --config config_bakeoff.yaml --phase b --features study

# Tests, including the classical-model correctness checks
python -m pytest -q
```

**Harness validation.** On the original single split the harness reproduces the published
persistence figure **1.3097 m/s** exactly, and XGBoost at **0.7299** against the recorded
gate-4 value of 0.7295. If the harness disagreed with figures already published, the harness
would be the thing at fault.
