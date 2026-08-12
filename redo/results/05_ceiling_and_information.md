# 05 — The ceiling, and where the skill comes from

> **What:** the measurements that established the project's central constraint — that the
> accuracy limit was *information*, not sample count or model class — and how the bakeoff
> corroborates them.
> **Why it matters:** this is the evidence for [stage 4](../04_stage4_ceiling_analysis.md),
> which argues ceiling analysis must precede model selection. The original project ran it
> in iteration **five**, and it retroactively explained everything that happened in the
> first four.
> **Primary source:** [`docs/11 §1`](../../docs/11_nwp_forecast_features.md) — summarised
> here with citations rather than restated in full, per [stage 9](../09_stage9_writeup_and_hygiene.md).

## The four measurements

All at a **24 h horizon**, on the 2020–2023 ERA5 window (35,028 rows, 53 features).

### 1. Autocorrelation ceiling

The target's autocorrelation at 24 hours is **0.571**, which caps R² at **0.326** for *any*
predictor built only on the atmospheric state at time *t*.

The best v4 model reached R² 0.390 on a slightly richer information set — **already at the
ceiling**. There was nothing left to win by tuning.

### 2. Learning curve — flat after ~1.6 years

| Train years | Test MAE (m/s) | Δ | Train/test ratio |
|---:|---:|---:|---:|
| 0.4 | 1.5017 | — | 2.36 |
| 0.8 | 1.1933 | −0.3085 | 1.52 |
| 1.6 | 1.1534 | −0.0399 | 1.25 |
| 2.4 | 1.1479 | −0.0055 | 1.19 |
| 3.2 | **1.1410** | −0.0069 | **1.16** |

The final doubling of training data bought **1.1%**. The train/test ratio of 1.16 is the
decisive figure: more data cures *variance*, and there was almost none left to cure.

A 44.8-hour ten-year backfill had been launched on the assumption that this curve was steep.
It was stopped mid-run when the curve was finally measured.

### 3. Feature ablation — saturated

| Feature set | n | Test MAE |
|:---|---:|---:|
| lag/rolling only | 9 | 1.2564 |
| + calendar | 13 | 1.1760 |
| + surface weather | 25 | 1.1615 |
| + upper-air | 33 | 1.1444 |
| + spatial derivatives | 53 | **1.1410** |

The last two groups — 28 columns including the expensive upper-air download — bought
**0.02 m/s combined**. (Ten of the twenty spatial columns were later found to be
mathematically dead; see [06](06_what_the_protocol_caught.md), defect 4. An ablation can
tell you a feature did not help, never that it was broken.)

### 4. Nowcast decomposition — the error is all in the gap

At horizon 0 the pipeline scores MAE **0.015**, R² **0.9998**. Near-tautological, but it
localises the error exactly:

> **Converting weather to hub wind is free. One hundred percent of the error is the 24-hour
> atmospheric gap.**

Since every exogenous feature was the analysed state at time *t*, there was no forecast
anywhere in the input. That identified forecast information as the only remaining lever.

## What breaking the ceiling looked like

Adding day-ahead NWP forecasts lifted R² from **0.3300 → 0.6653** on the full 22-month ERA5
window — straight through the 0.326 cap, because a forecast is not "the state at time *t*"
and the ceiling does not bind it. Details and gates:
[`RESULTS.md`](../../RESULTS.md).

## How the bakeoff corroborates all of this

Three independent lines of evidence from [02](02_classical_vs_ml.md), at a 24 h horizon:

**1. SARIMAX barely beats persistence.** `sarimax_nwp` scores 1.2554 — skill **+0.022**.
An ARIMA-family model is a state-at-*t* predictor by construction, so the ceiling binds it.
Its AR terms decay within a few steps of a 2,227-step forecast, and it collapses toward its
exogenous regression. **This is the ceiling made directly visible in a model's score.**

**2. A 21-parameter linear model captures almost all the skill.** `mos_multi` reaches 0.8412
against gradient boosting's 0.8037. If the problem were model-limited, that gap would be
large. It is 4.7%. The binding constraint is what information reaches the model, not what
the model does with it.

**3. Removing 39 reanalysis features changes nothing.** [03](03_operational_vs_study.md)
finds `operational` (55 features) marginally *better* than `study` (94). Once a forecast of
the target hour is present, the analysed state at *t* is largely redundant — which is the
ceiling argument restated from the opposite direction.

## The generalisable lesson

Three of these measurements take **under a day** in total. They would have redirected the
project at iteration one instead of iteration five, and would have prevented a 44.8-hour
download launched on an unmeasured belief.

That is why [stage 4](../04_stage4_ceiling_analysis.md) places ceiling analysis **before**
model selection, with an exit gate that cannot be passed without stating a number:

> *"On this information set, the maximum achievable R² is approximately X, and we are
> currently at Y."*

If you cannot fill in X, you are about to repeat this project's first four iterations.
