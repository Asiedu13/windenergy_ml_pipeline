# Stage 5 — Baselines and the measurement protocol

> **What this stage is:** building the comparators every future number will be judged
> against, and fixing the statistical protocol, before any machine-learning result exists.
> **Why it comes fifth:** the original project quoted a **25-hour** persistence baseline as
> "24-hour persistence" for its entire duration ([register C3](00_mistake_register.md)),
> once compared a 6-hour horizon against a 24-hour one and called a regression an
> improvement ([register C4](00_mistake_register.md)), and never once ran more than a single
> seed ([register C5](00_mistake_register.md)). Baselines are not a formality — they are the
> denominator of every claim.
> **Effort:** half a day, plus about an hour of compute.

## 5.1 Define each baseline once, in one place, with a test

> **What:** the five mandatory comparators.
> **Why they are mandatory:** an ML model that cannot beat "tomorrow will be like today" is
> not a result, and it is remarkably easy to produce one without noticing.
> **Why five and not three:** this section originally listed three. The bakeoff
> ([results/02](results/02_classical_vs_ml.md)) showed that the two missing ones —
> classical NWP post-processing — are **where most of the skill actually enters**. A project
> comparing ML against persistence alone cannot say what its ML contributed.

| Baseline | Definition | Params | Guards against |
|:---|:---|---:|:---|
| **Persistence** | ŷ(t+h) = y(t) — the value *now*, at exactly the horizon | 0 | Mistaking autocorrelation for skill |
| **Damped persistence** | ŷ = a·y(t) + b — persistence shrunk toward the mean | 2 | Being flattered by an opponent that does not shrink |
| **Climatology** | training mean by month × hour-of-day | — | Mistaking a learned diurnal cycle for skill |
| **Single-model MOS** | OLS of the observation on **one** NWP forecast | 2 | Mistaking *having a forecast* for *learning from one* |
| **Multi-model MOS** | OLS on every day-1 forecast column + calendar | ~21 | Mistaking linear combination for nonlinearity |

**The last two are the ones that change conclusions.** Measured at a 24 h horizon over four
rolling folds:

| Step | MAE (m/s) | Bought |
|:---|---:|---:|
| persistence → damped persistence | 1.3183 → 1.1861 | 0.13 |
| damped → **single-model MOS** | 1.1861 → 0.9904 | **0.20** |
| single → **multi-model MOS** | 0.9904 → 0.8412 | **0.15** |
| multi-model MOS → gradient boosting | 0.8412 → 0.8037 | **0.04** |

A 21-parameter linear model lands within **4.7%** of gradient boosting. Without the MOS
rungs, that 4.7% would have been reported as the full value of the machine learning.

**The exact-lag trap.** The original's `PersistenceForecaster` predicted from
`hub(t-1)` while scoring against `hub(t+24)` — a 25-hour persistence. True 24-hour
persistence is **1.399 m/s**; the quoted figure was 1.440, a 2.9% easier bar, and it was
never corrected because it would have broken comparability with earlier iterations.

**The guard:** one function, one definition, and a test that asserts the exact lag —
construct a series where `y(t+h) = y(t)` exactly and assert the baseline scores zero error.

**Naive persistence is a weak opponent — now measured, not asserted.** At a 24 h horizon
over four rolling folds it scores R² **−0.108**: *worse than predicting the training mean*.
Damped persistence is the same information with one extra parameter and scores **+0.111**.

> Comparing a model against naive persistence therefore flatters it by ~0.13 m/s. Every
> v1–v4 claim in the original used naive persistence as the bar. **Quote damped persistence
> as the reference** — it is the honest one.

**Include the OLS baseline.** It is the most under-rated of the set. In the original it
scored R² 0.267 with a single parameter against 0.365 for a 53-feature tuned XGBoost. A
baseline that captures 73% of your model's skill with one coefficient is telling you
something important.

## 5.2 Re-baseline on *your* window, always

> **Never quote a baseline computed on a different window than the result it is compared to.**

The original moved its study window in iteration five and had to recompute everything;
until it did, the temptation to compare a new result against an old baseline was constant.
The two windows differed materially in difficulty — persistence 1.09 m/s on a 12-month
subset versus 1.399 on the original window, which made an early result look like a **+40%**
gain when the honest figure on the full window was **+30.6%**.

Rules:

1. Every result table states its window and its horizon.
2. Every result is compared only against baselines from the same window and horizon.
3. When the window changes, **all** baselines are recomputed before any comparison.
4. Record a difficulty indicator — persistence MAE is the natural one — so window changes
   are visible rather than invisible.

## 5.3 The statistical protocol

> **What:** how many times you run things, and what you report.
> **Why:** a single seed on a single split is a point estimate with unknown variance. The
> original's headline claim — a 30.6% improvement — has no interval around it to this day,
> and a large claimed effect attracts more scrutiny than a small one.

Fix in advance:

- **Seeds:** at least five, fixed values, decided in [stage 1](01_stage1_deployment_contract.md).
  Report **mean ± sd**. Never best-of-N — that is selection, not measurement.
- **Splits:** chronological, never shuffled. Ideally more than one origin, so the result is
  not an artefact of where the split happens to fall.
- **Cross-validation:** expanding-window (`TimeSeriesSplit`), never random. Random shuffling
  leaks the future into training folds.
- **What counts as a difference:** decide the threshold before seeing the numbers. If two
  models are within one standard deviation, **report it as a tie** — the original correctly
  did this for its v4 architecture comparison and it was one of its better calls.

The cost is about an hour for tree models. It converts every subsequent claim from a number
into a number with an uncertainty, which is the difference between a result and an anecdote.

**Measured justification, and it is stronger than expected.** Across three seeds on ERA5
truth at a 24 h horizon ([results/04](results/04_deep_models.md)):

| Model | Seed sd | Range across seeds |
|:---|---:|---:|
| `xgboost` | 0.0035 | 0.0068 |
| `lstm` | 0.0276 | **0.0505** |
| `nbeats` | **0.0361** | **0.0705** |

**Deep models are 3–10× more seed-sensitive than trees**, and LSTM's seed range *exceeds its
entire mean gap to XGBoost* (0.0372). A single-seed run could report that ranking either
way and no reader could tell. Seed variance is not uniform across model classes, so it
cannot be assumed away — it has to be measured, per class.

**Also fix the significance test here, not later.** Mean ± sd tells you the spread; it does
not tell you whether one model genuinely forecasts better. Use **Diebold–Mariano** on the
loss differentials with a HAC (Newey–West) variance — overlapping multi-step forecast errors
are serially correlated by construction, and a plain t-test overstates significance badly.
Details and a worked implementation trap: [results/01](results/01_protocol.md) and
[results/06](results/06_what_the_protocol_caught.md).

## 5.4 Report baselines before any ML number exists

Produce and commit the baseline table on its own, before the first model is trained. Two
reasons:

- It cannot then be tuned, consciously or otherwise, to flatter a result you have already
  seen.
- If a baseline is surprisingly strong, that is itself a finding — and one you want to
  encounter *before* you have invested in an architecture.

## Exit gate

You may not proceed to stage 6 until:

1. All five baselines are implemented in one place, each with a test asserting its exact
   definition and lag.
2. The baseline table for the study window is computed, committed, and states window and
   horizon.
3. The multi-seed, multi-split protocol is implemented and demonstrated on the baselines
   themselves — so the machinery is known to work before it matters.
4. A difficulty indicator for the window is recorded.

**How this gate can fail:** a persistence baseline whose lag has not been asserted by a
test. That is the exact defect the original carried from beginning to end.

---

Previous: [Stage 4](04_stage4_ceiling_analysis.md) ·
Next: [Stage 6 — Modelling](06_stage6_modelling.md)
