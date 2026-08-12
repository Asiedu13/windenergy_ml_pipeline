# Stage 4 — Ceiling analysis

> **What this stage is:** measuring how much predictability the available information
> actually contains, **before** choosing a model.
> **Why it comes fourth:** this is the stage the original project ran in iteration *five*,
> and it retroactively explained everything that had happened in the first four. It showed
> the model was pinned at R² 0.326 by the information set, not by the algorithm — which
> means four iterations of feature engineering, hyperparameter tuning and architecture
> search had been optimising against a hard ceiling ([register A1](00_mistake_register.md)).
> **Effort:** 1 day. It is the highest-return day in the whole plan.

## 4.1 The question this stage answers

> **Is the gap between where I am and where I want to be caused by the model, or by the
> information?**

These have opposite remedies. If it is the model, tune, tune, search architectures. If it
is the information, *none of that will work* and the only fix is a different input. The
original spent four iterations applying the first remedy to the second problem.

## 4.2 The four measurements

Run all four on the **operational feature set** from
[stage 2](02_stage2_data_availability_audit.md) — the features that survive the latency
budget. This matters: the original ran its ceiling analysis on the offline feature set, so
the ceiling it computed is not the one a deployed system would face.

### Measurement 1 — the autocorrelation ceiling

Compute the autocorrelation of the target at the forecast horizon. For any predictor built
purely on the state at time *t*, that correlation squared is approximately the maximum
achievable R².

*Original result:* 24 h autocorrelation **0.571** → ceiling **R² 0.326**. The best v4 model
reached 0.390 — already at the ceiling, using a slightly richer information set. There was
nothing left to win.

**How to read it:** if your best model is near the ceiling, stop tuning. If it is far below,
tuning is legitimate.

### Measurement 2 — the learning curve

Retrain on 10%, 20%, 40%, 80%, 100% of the training window against a fixed test set. Record
test error **and the train/test error ratio** at each point.

*Original result:*

| Train years | Test MAE | Δ | Train/test ratio |
|---:|---:|---:|---:|
| 0.4 | 1.5017 | — | 2.36 |
| 0.8 | 1.1933 | −0.3085 | 1.52 |
| 1.6 | 1.1534 | −0.0399 | 1.25 |
| 3.2 | 1.1410 | −0.0069 | **1.16** |

The final doubling bought **1.1%**. The train/test ratio of 1.16 is the decisive number:
more data cures *variance*, and there was almost none left to cure.

**How to read it:** a flat tail plus a low train/test ratio means more data is worthless.
A steep tail or a high ratio means more data is the cheapest available win. The original
launched a 44.8-hour ten-year backfill on an *assumption* about this curve, and stopped it
mid-run when the curve was finally measured.

### Measurement 3 — feature ablation

Add feature groups cumulatively and record where the curve flattens.

*Original result:* lag/rolling only 1.2564 → +calendar 1.1760 → +surface 1.1615 → +upper-air
1.1444 → +spatial derivatives 1.1410. The last two groups — 28 additional columns, including
the expensive upper-air download — bought **0.02 m/s combined**.

**A caution from the original:** ten of those twenty spatial columns were mathematically
dead (a 1×2 grid patch instead of the intended 3×3, so all `_grad_ns` columns are
identically zero — [register B4](00_mistake_register.md)). An ablation cannot tell you a
feature is broken; it only tells you it did not help. Pair this with
[stage 3](03_stage3_alignment_harness.md)'s shape assertions.

### Measurement 4 — the one-parameter baseline, and the nowcast decomposition

Two cheap, brutal checks:

**Against one parameter.** Fit an OLS on the single most obvious predictor and compare.

*Original result:* OLS on `hub(t)` alone — one parameter — scored 1.2388 / R² 0.267. The
full 53-feature Optuna-tuned XGBoost scored 1.1410 / R² 0.365. **The entire apparatus bought
0.06 R² over one parameter.** If that is your situation, you do not have a modelling problem.

**The nowcast decomposition.** Re-run the pipeline at horizon 0. If it scores near-perfectly,
the mapping from features to target is free and **100% of the error lives in the temporal
gap**.

*Original result:* nowcast MAE **0.015**, R² 0.9998. Near-tautological, but it localised
the error precisely: converting weather to hub wind costs nothing; the entire error is the
24-hour atmospheric gap. That is what identified forecast information as the only remaining
lever.

## 4.3 What to do with the answer

| Finding | What it means | What to do |
|:---|:---|:---|
| Best model far below ceiling, train/test ratio high | Variance-limited | More data, more regularisation |
| Best model at ceiling, learning curve flat, nowcast near-perfect | **Information-limited** | **Get different information.** More model is wasted effort |
| Ablation flat after group 2 | Later features are noise or broken | Check shapes before concluding they're useless |
| Full apparatus ≈ one-parameter OLS | The problem may not need ML at all | Say so; it is a legitimate finding |

**In the original's case the answer was "information-limited", and the remedy was
day-ahead NWP forecasts** — which lifted R² from 0.330 to 0.665, straight through the 0.326
ceiling, because forecast information is not "the state at time *t*" and the ceiling
therefore does not bind it.

### The bakeoff corroborates the ceiling three ways

Measured later, at a 24 h horizon ([results/05](results/05_ceiling_and_information.md)):

1. **SARIMAX barely beats persistence** — skill **+0.022**. An ARIMA-family model is a
   state-at-*t* predictor by construction, so the ceiling binds it. **This is the ceiling
   made directly visible in a model's score.**
2. **A 21-parameter linear model reaches within 4.7% of gradient boosting.** If the problem
   were model-limited that gap would be large. The constraint is what information reaches
   the model, not what the model does with it.
3. **Removing 39 reanalysis features changes nothing** — once a forecast of the target hour
   is present, the analysed state at *t* is largely redundant. The same argument from the
   opposite direction.

## 4.4 Then, and only then, re-plan

Take the finding back to the [stage 1](01_stage1_deployment_contract.md) contract. If the
ceiling is low on the operational feature set, the honest options are: find information that
breaks it, change the horizon, or reframe the project. All three are better than four
iterations of tuning.

## Exit gate

You may not proceed to stage 5 until:

1. All four measurements are run **on the operational feature set**, and recorded.
2. A single sentence is written: *"On this information set, the maximum achievable R² is
   approximately X, and we are currently at Y."*
3. The bottleneck is named — variance, information, or neither — with the measurement that
   says so.
4. If the answer is "information-limited", the modelling stages are **not** to be entered
   until a new information source is identified.

**How this gate can fail:** being unable to state the ceiling number. If you can't, you are
about to repeat the original project's first four iterations.

---

Previous: [Stage 3](03_stage3_alignment_harness.md) ·
Next: [Stage 5 — Baselines and protocol](05_stage5_baselines_and_protocol.md)
