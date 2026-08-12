# Stage 6 — Modelling

> **What this stage is:** actually training models — which in the original project was
> stage one of one, for four iterations.
> **Why it comes sixth:** everything before it determines whether this stage can succeed at
> all. If [stage 4](04_stage4_ceiling_analysis.md) says the problem is information-limited,
> no model here will help, and the honest move is to go back and find information rather
> than search architectures.
> **Effort:** days, not weeks — and deliberately less than the original spent.

## 6.1 The order: simplest first, and stop when it stops paying

> **Why not start with the sophisticated thing:** because you need to know what the
> sophistication *bought*. A deep model with no simple comparator is an unfalsifiable claim.

1. **One-parameter OLS** — already built in [stage 5](05_stage5_baselines_and_protocol.md).
2. **Single-model MOS** — two parameters, a linear rescale of one NWP forecast.
3. **Multi-model MOS** — OLS on every day-1 forecast column plus calendar. ~21 parameters.
4. **Full-feature linear model.** In the original this scored 1.1591 against the tuned
   XGBoost's 1.1410 — within 1.6%. Worth knowing before spending a week on trees.
5. **Gradient boosting**, untuned, then tuned. The original's tuning bought very little:
   on the final ERA5 window, XGBoost scored **0.6780 under both** a 50-trial Optuna
   parameter set and the untuned defaults. Identical to four decimal places.
6. **Deep models — only if there is a stated reason.**

**Rungs 2 and 3 were added after measurement, and they are where most of the skill enters.**
The bakeoff ([results/02](results/02_classical_vs_ml.md), 24 h horizon, 4 rolling folds)
gives the step sizes:

| Step | MAE (m/s) | What it bought |
|:---|---:|---:|
| persistence → damped persistence | 1.3183 → 1.1861 | 0.13 |
| damped → **single-model MOS** (2 params) | 1.1861 → 0.9904 | **0.20** |
| single → **multi-model MOS** (21 params) | 0.9904 → 0.8412 | **0.15** |
| multi-model MOS → **gradient boosting** | 0.8412 → 0.8037 | **0.04** |

The two largest steps are both classical, and neither involves learning anything beyond a
linear fit. **Gradient boosting adds a real but small 4.7%** (Diebold–Mariano p < 0.0001 —
the gap is not noise, it is just small). Any project of this shape that skips rungs 2 and 3
cannot tell how much its machine learning actually contributed.

## 6.2 Deep learning is the thing to justify, not the default

The original ran a five-model bakeoff three times and the answer moved each time. All at a
**24 h horizon**:

| Run | Truth source | Spread across architectures | Winner |
|:---|:---|---:|:---|
| State-only (v4) | ERA5, 2020–2023 | 0.101 m/s — effectively a tie | LSTM, by a nose |
| + NWP forecasts (v5) | **NASA POWER**, 22 mo | 0.291 m/s — not a tie | XGBoost, **by 18%** over the best deep model |
| **+ NWP forecasts** | **ERA5**, 22 mo | *pending TFT* | **XGBoost, by 5.1%** |

> ⚠️ **The 18% figure does not transfer, and it was written here before that was known.**
> On NASA POWER truth the tree led the best deep model by 18% (0.678 vs 0.827). On **ERA5**
> truth — the project's actual truth series — the same comparison gives **0.7329 vs 0.7701,
> a 5.1% gap** ([results/04](results/04_deep_models.md)). Trees still win; by roughly a
> third of what the NASA POWER run suggested. **Provisional: TFT is 0 of 3 seeds done.**
>
> Why the two disagree is not established. NASA POWER is coarser (~55 km) and hour-averaged,
> so its target is smoother; whether that favours one model class is a hypothesis.

On NASA POWER truth the ranking was XGBoost 0.678, TFT 0.827, LSTM 0.903, N-BEATS 0.942,
DLinear 0.969. Three artefact explanations were tested and all three failed — same split
(3183 test points on both paths), same alignment (persistence 1.1100 on both), and the
tree's score was insensitive to tuning. So the *direction* is solid on both sources; only
the magnitude moved.

**The mechanism is a hypothesis, not a finding.** A plausible reading is that a forecast
column whose value at time *t* is close to the answer at *t+24 h* is exactly the structure a
tree splits on directly, while sequence models must carry it through an encoded input
window. That has not been tested and should not be stated as the explanation.

**The practical rule:** on this problem, budget days for gradient boosting and treat deep
models as an experiment requiring justification — not the reverse.

The cost side of that judgement, measured on ERA5 truth at a 24 h horizon
([results/04](results/04_deep_models.md)):

| Model | Minutes per fit | MAE (m/s) | Cost relative to XGBoost |
|:---|---:|---:|---:|
| `xgboost` | 0.1 | 0.7329 | 1× |
| `dlinear` | 1.1 | 0.8244 | ~10× |
| `lstm` | 4.8 | 0.7701 | ~42× |
| `nbeats` | 20.7 | 0.8234 | ~185× |
| `tft` | ~350 | *pending* | ~3,000× |

**The best deep model costs ~42× the training time to finish 5% behind.** For a single-site
day-ahead forecast that is not a close decision.

## 6.3 Comparability rules for any head-to-head

> **Why this section exists:** the original's first bakeoff was invalid because one model
> was solving a different problem, and nobody noticed until the doc explaining its poor
> performance had already been written.

| Rule | What went wrong without it |
|:---|:---|
| **Identical objective.** Every model optimises the same loss | TFT trained on darts' default **17-quantile** objective while the rest used MSE. Correcting it recovered **0.380 m/s** and moved TFT from last to mid-field ([register C1](00_mistake_register.md)) |
| **Identical information.** Same features, same split, same target | — |
| **Comparable training care.** Equivalent early stopping and LR selection | Uneven effort reads as a model-quality difference |
| **Print and assert the resolved config before any run over an hour** | A config with no `deep:` block silently gave TFT **200 epochs instead of 60** — roughly 12 hours instead of 4 ([register C2](00_mistake_register.md)) |
| **Cheapest model first in the run order** | A crash five hours in still leaves four results |

That last pair is worth stating as a habit: **before launching anything long, print the
resolved configuration object — not the config file — and assert the two or three values
that would silently ruin the run.**

## 6.4 Report ties as ties

With the [stage 5](05_stage5_baselines_and_protocol.md) protocol you have a standard
deviation. Use it. If two models are within one, they are tied — say so, and pick on
secondary grounds: interpretability, training cost, operational simplicity.

The original got this right for v4 ("treat the LSTM/XGBoost ordering as a tie") and it aged
well. What did *not* age well was the reverse error: a 0.291 m/s spread reported after only
a single seed, so the size of the gap was known but its uncertainty was not.

**The multi-seed run shows exactly how dangerous that is.** Across three seeds on ERA5 truth:

| Model | Seed sd | Range across seeds | vs XGBoost |
|:---|---:|---:|---:|
| `xgboost` | **0.0035** | 0.0068 | 1× |
| `dlinear` | 0.0115 | 0.0215 | 3× |
| `lstm` | 0.0276 | **0.0505** | **8×** |
| `nbeats` | **0.0361** | **0.0705** | **10×** |

**Deep models are 3–10× more seed-sensitive than the tree**, and LSTM's seed range (0.0505)
is *larger than its entire mean gap to XGBoost* (0.0372). Change the seed and a single-seed
run could report the ranking either way. N-BEATS and DLinear differ by 0.0010 m/s on the
mean and overlap heavily — on this evidence they are **tied**, and any ordering between them
is noise.

No single-seed experiment can see this, and every architecture conclusion in the original
project was single-seed. This is the strongest available argument for putting multi-seed in
the protocol ([stage 5](05_stage5_baselines_and_protocol.md)) rather than on a follow-up
list — where the original left it for five iterations.

## 6.5 Interpretability as a check, not decoration

Run SHAP and correlation analysis, but use them **diagnostically**:

- Does the top feature make physical sense? In the original, the top feature by a factor of
  five was the day-ahead 100 m wind forecast — expected, and reassuring.
- Is anything ranked high that shouldn't be? That is a leakage smell.
- Is anything ranked at exactly zero? That is a **broken** feature, not an unimportant one —
  the ten dead spatial columns would have shown up here ([register B4](00_mistake_register.md)).

## Exit gate

You may not proceed to stage 7 until:

1. Every model beats all three baselines, on the same window and horizon, with intervals.
2. Any head-to-head satisfies all five comparability rules, verified by the printed resolved
   config for each model.
3. Results are mean ± sd across seeds; ties reported as ties.
4. The chosen model is chosen for a stated reason — accuracy *with* its interval, or
   cost/interpretability where the accuracy is tied.

**How this gate can fail:** a model that does not beat persistence; a comparison where two
models used different objectives; or a headline gap quoted without a standard deviation.

---

Previous: [Stage 5](05_stage5_baselines_and_protocol.md) ·
Next: [Stage 7 — Power and uncertainty](07_stage7_power_and_uncertainty.md)
