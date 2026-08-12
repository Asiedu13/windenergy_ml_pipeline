# `redo/` — building this project again, in the right order

> **What this folder is:** a staged plan for rebuilding the Anloga wind-power forecasting
> project from nothing, written *after* finishing it once.
> **What it is not:** a correction to [`../docs/`](../docs/). Those thirteen documents are
> the true record of what actually happened, mistakes included, and this folder is only
> meaningful read against them. Nothing here deletes or supersedes anything there.
> **Who it is for:** you in six months, a supervisor asking "what would you do
> differently", or anyone starting a similar forecasting project.

## The short version

The project finished with a genuinely good result — **7.48% nMAE, R² 0.665**, at or below
a published benchmark. It also finished with two problems that no further modelling can
fix:

1. **The target is synthetic.** No wind farm in Ghana publishes SCADA, so "true" power is
   ERA5 wind pushed through a manufacturer power curve. Every power metric measures how
   well the model forecasts *its own physics model*.
2. **Half the features cannot exist at forecast time.** ERA5 is *reanalysis*, published
   with roughly five days of latency. Of the final model's **104 features, 53 are
   ERA5-derived** — the atmospheric state at time *t*, which is not available at the moment
   you would actually issue a forecast for *t+24 h*.

Both were knowable on day one from the sources' own documentation. Neither was noticed
until the project was essentially finished.

## What the evidence added (2026-08-12)

This folder was written before the classical-vs-ML bakeoff ran. That run
([results/](results/README.md)) has since measured several of its claims, and two came out
differently enough to change what the documents say:

| Finding | Where |
|:---|:---|
| **A 21-parameter linear model lands within 4.7% of gradient boosting.** Most of the project's ~30% NWP gain is an *information* effect, not a machine-learning one — and the redo's original three-baseline list was missing the two rungs that show it | [05](05_stage5_baselines_and_protocol.md), [results/02](results/02_classical_vs_ml.md) |
| **Dropping every latency-blocked reanalysis feature costs nothing** — degradation is *negative*. Stage 8's pre-registered prediction, measured and right | [08](08_stage8_deployment_rehearsal.md), [results/03](results/03_operational_vs_study.md) |
| **The "XGBoost beats deep models by 18%" figure does not transfer.** That was NASA POWER truth; on ERA5 it is **5.1%**. Direction holds, magnitude does not | [06](06_stage6_modelling.md), [results/04](results/04_deep_models.md) |
| **Deep models are ~8× more seed-sensitive than trees** — LSTM's seed range exceeds its entire mean gap to XGBoost | [06](06_stage6_modelling.md), [results/04](results/04_deep_models.md) |
| **The guards caught four real defects in one day**, two of them introduced by the redo's own bakeoff | [00 group F](00_mistake_register.md), [results/06](results/06_what_the_protocol_caught.md) |

None of it changes the nine-stage structure or the ordering argument — the evidence supports
both. It changes specific figures, and it adds two baselines that should have been there
from the start.

## The actual lesson: three inverted orderings

The individual bugs — and there were many, catalogued in
[00_mistake_register.md](00_mistake_register.md) — matter less than the order the work was
done in. Three orderings were backwards:

| The project did this… | …before doing this | What it cost |
|:---|:---|:---|
| Optimised models across four iterations | Measuring the information ceiling | v1–v4 chased accuracy against a hard cap of **R² 0.326**. Four iterations of model work bought 0.06 R²; one feature change in v5 bought 0.34. |
| Chose data sources and built on them | Auditing coverage, latency and conventions | Study window forced to move to 2024-03 late in the project, invalidating v1–v4 baselines. Two silent data defects. The deployability blocker found last. |
| Wrote conclusions into documentation | Confirming the numbers were stable | [`09_why_deep_learning_underperformed.md`](../docs/09_why_deep_learning_underperformed.md) is a complete, well-argued essay explaining a **measurement artefact**. [`07_findings_and_analysis.md`](../docs/07_findings_and_analysis.md) had to be rewritten once and is stale again today. |

**So the redo is not a different model. It is the same work in a different order**, with
modelling sixth instead of first, and an exit gate on every stage that can actually fail.

## The stages

Each stage has an **exit gate** — a condition stated so that it is possible to fail it. A
gate you cannot fail is not a gate, which is a lesson this project learned twice
(see the register, entry 14).

| # | Stage | Exit gate |
|---:|:---|:---|
| — | [00 Mistake register](00_mistake_register.md) | *(reference — read first)* |
| — | [results/](results/README.md) | *(measured evidence for the stages below — all at a **24 h horizon**)* |
| 1 | [Deployment contract](01_stage1_deployment_contract.md) | A one-page contract, plus a git repo with a first commit |
| 2 | [Data availability audit](02_stage2_data_availability_audit.md) | Every source characterised; the study window chosen *from* that table |
| 3 | [Alignment harness](03_stage3_alignment_harness.md) | Every guard demonstrated **failing** on broken input before it is trusted |
| 4 | [Ceiling analysis](04_stage4_ceiling_analysis.md) | A stated maximum achievable R² on the deployable feature set |
| 5 | [Baselines and protocol](05_stage5_baselines_and_protocol.md) | Baselines with intervals, before any ML number exists |
| 6 | [Modelling](06_stage6_modelling.md) | Every model beats every baseline, same split, with intervals |
| 7 | [Power and uncertainty](07_stage7_power_and_uncertainty.md) | Calibrated P10/P50/P90, not just a point forecast |
| 8 | [Deployment rehearsal](08_stage8_deployment_rehearsal.md) | An operational score reported beside the offline score |
| 9 | [Write-up and hygiene](09_stage9_writeup_and_hygiene.md) | No document states a number it did not compute |

**Modelling is stage 6 of 9.** That is the single biggest structural change. In the original
project it was, effectively, stage 1 of 1 for four iterations.

## What the redo does *not* change

Not everything went wrong, and a plan that pretends otherwise is not useful.

- **The site, the turbine and the two-stage architecture were sound.** Each traces to a real
  constraint and is documented in [`01_problem_and_design.md`](../docs/01_problem_and_design.md).
  The redo keeps all three.
- **The six non-negotiable design principles were right** — never shuffle a time series,
  always beat persistence and climatology, seed everything, decouple the stages, report
  error propagation, keep the model site-specific. They were followed, and they held.
- **The documentation habit was right**, even where individual documents went stale. A
  project that recorded nothing would not have been able to diagnose its own ceiling.
- **The verification-gate idea was right.** It caught two real data defects. It was only
  the *implementation* that was weak — see register entry 14.

The redo changes **order and evidence standards**, not the engineering.

## Reading order

If you are picking this up cold: this page → [00 mistake register](00_mistake_register.md)
→ [stage 1](01_stage1_deployment_contract.md), then in sequence. If you only read two
files, read the register and [stage 8](08_stage8_deployment_rehearsal.md) — the register is
where the pain is, and stage 8 is the one the original project never had at all.

For any unfamiliar term, [`../docs/glossary.md`](../docs/glossary.md) still applies
unchanged.
