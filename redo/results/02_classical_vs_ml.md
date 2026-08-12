# 02 — Classical vs ML: the ladder

> **What:** eleven predictors, from naive persistence to gradient boosting, on one protocol.
> **Why it matters:** the project had never compared against classical NWP post-processing —
> only against naive persistence, which is a weak opponent. The result changes how the
> headline should be stated.
> **Protocol:** [01](01_protocol.md). **Source:** `outputs/reports/bakeoff.csv`.

## Results — 24 h horizon, `study` features, 4 rolling folds

| Rank | Tier | Model | MAE (m/s) | sd | RMSE | R² | Skill | Params (approx) |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
| 1 | ML | **`xgboost`** | **0.8037** | 0.1023 | 1.0337 | 0.5886 | +0.386 | ~10⁵ |
| 2 | ML | `lightgbm` | 0.8086 | 0.1026 | 1.0407 | 0.5831 | +0.383 | ~10⁵ |
| 3 | **classical** | **`mos_multi`** | **0.8412** | 0.1193 | 1.0821 | 0.5506 | +0.355 | **21** |
| 4 | ML | `ridge` | 0.8422 | 0.1129 | 1.0723 | 0.5595 | +0.354 | 94 |
| 5 | classical | `kalman_mos` | 0.8572 | 0.1489 | 1.0912 | 0.5362 | +0.349 | 4 (time-varying) |
| 6 | classical | `analog_ensemble` | 0.8662 | 0.1509 | 1.1114 | 0.5238 | +0.340 | 0 (k=25) |
| 7 | classical | `mos_single` | 0.9904 | 0.1094 | 1.2626 | 0.3883 | +0.239 | **2** |
| 8 | reference | `damped_persistence` | 1.1861 | 0.1447 | 1.5035 | 0.1111 | +0.094 | 2 |
| 9 | classical | `sarimax_nwp` | 1.2554 | 0.2675 | 1.5539 | 0.0831 | +0.022 | ~10 |
| 10 | reference | `climatology` | 1.2725 | 0.0959 | 1.6021 | 0.0031 | +0.021 | 288 |
| 11 | reference | `persistence` | 1.3183 | 0.2250 | 1.6785 | −0.1078 | 0.000 | 0 |

*sd is across folds and seeds; skill is `1 − MAE/MAE_persistence` per fold.*

## Finding 1 — the gap between classical and ML is 4.7%

`mos_multi` is ordinary least squares on the seventeen day-1 forecast columns plus four
calendar terms. **Twenty-one parameters.** It lands 0.0375 m/s behind a gradient-boosted
ensemble with roughly five orders of magnitude more.

Diebold–Mariano says the difference is real, not noise:

| Model | DM statistic | p-value | verdict |
|:---|---:|---:|:---|
| `lightgbm` | −2.209 | 0.0272 | `xgboost` better |
| `mos_multi` | −4.816 | 0.0000 | `xgboost` better |
| `ridge` | −5.202 | 0.0000 | `xgboost` better |
| `kalman_mos` | −4.021 | 0.0001 | `xgboost` better |
| `analog_ensemble` | −6.692 | 0.0000 | `xgboost` better |
| `mos_single` | −14.453 | 0.0000 | `xgboost` better |
| `damped_persistence` | −15.174 | 0.0000 | `xgboost` better |
| `sarimax_nwp` | −16.527 | 0.0000 | `xgboost` better |
| `climatology` | −14.887 | 0.0000 | `xgboost` better |
| `persistence` | −16.368 | 0.0000 | `xgboost` better |

*Reference: `xgboost`. Negative statistic = row model has lower loss than the reference —
here every statistic is negative and every p-value small, i.e. XGBoost wins each comparison.*

**So ML does win, and defensibly.** The point is the *size*: real, and 4.7%.

### What this means for the project's headline

v5's claim was "adding NWP forecast features cuts MAE ~30%". That claim is intact, but its
attribution changes:

> **The gain is overwhelmingly an information effect, not a machine-learning effect.**
> Most of it is available to a twenty-one-parameter linear model. Gradient boosting adds a
> real but small 4.7% on top.

This is the *stronger* framing, not the weaker one. It survives the obvious examiner
question — "did you need machine learning at all?" — with a precise answer: for the last
4.7%, yes, and here is the significance test.

## Finding 2 — the classical ladder shows where the skill enters

Reading down the tiers, each rung isolates one addition:

| Step | MAE | What it bought |
|:---|---:|:---|
| persistence → damped persistence | 1.3183 → 1.1861 | **0.13** — just shrinking toward the mean |
| damped persistence → `mos_single` | 1.1861 → 0.9904 | **0.20** — *having a forecast at all*, via 2 parameters |
| `mos_single` → `mos_multi` | 0.9904 → 0.8412 | **0.15** — combining three NWP models + calendar |
| `mos_multi` → `xgboost` | 0.8412 → 0.8037 | **0.04** — everything machine learning adds |

**The two largest steps are both classical**, and neither involves learning anything beyond
a linear fit. That is the shape of this problem.

## Finding 3 — naive persistence is a weak opponent

`persistence` scores R² **−0.108** — worse than predicting the training mean. `damped
persistence` is the same information with one extra parameter and scores +0.111.

Comparing a model against naive persistence therefore flatters it by roughly 0.13 m/s.
Every v1–v4 claim in this project used naive persistence as the bar. Use damped persistence
as the reference instead — it is the honest one.

## Finding 4 — SARIMAX adds almost nothing, exactly as predicted

`sarimax_nwp` scores 1.2554, a skill of **+0.022** over persistence. This is the
autocorrelation ceiling made visible: the target's 24 h autocorrelation of 0.571 caps R² at
**0.326** for any predictor built on the state at time *t*
([05](05_ceiling_and_information.md)). SARIMAX's AR terms decay within a few steps of a
2,227-step-ahead forecast, leaving essentially its exogenous regression — a worse-specified
MOS.

**Reporting an ARIMA-family model that barely beats persistence is the point**, not an
embarrassment. It is direct evidence for the ceiling argument.

## Per-fold detail — why rolling origin matters

Seed 42, `study`, 24 h horizon:

| Fold | Test period | persistence | damped | `mos_multi` | `xgboost` |
|:---|:---|---:|---:|---:|---:|
| fold1 | 2024-12-24 → 2025-03-27 | 1.1129 | 1.0345 | 0.8743 | 0.7793 |
| fold2 | 2025-03-27 → 2025-06-28 | **1.6274** | 1.3827 | 0.9866 | 0.9603 |
| fold3 | 2025-06-28 → 2025-09-29 | 1.2001 | 1.1725 | **0.7043** | **0.7031** |
| fold4 | 2025-09-29 → 2025-12-30 | 1.3330 | 1.1546 | 0.7997 | 0.7656 |

Two things only visible fold by fold:

1. **Difficulty swings 46%.** Persistence ranges 1.1129 → 1.6274. A single split lands you
   somewhere in that range at random, and the resulting MAE is partly a statement about
   which quarter you tested on.
2. **In fold 3, `mos_multi` and `xgboost` are separated by 0.0012 m/s** — 0.2%, a dead heat.
   The ML advantage is not uniform across the year; it concentrates in the harder quarters.

## Caveats

- **Absolute MAEs here are higher than the 0.7295 v5 headline** because rolling folds include
  earlier periods with less training data. Compare *within* this table; persistence is the
  anchor.
- **One site, one horizon, one truth source.** Nothing here transfers to another location or
  lead time without re-measurement.
- Deep models are **not** in this table — different evaluation basis, see
  [04](04_deep_models.md).
