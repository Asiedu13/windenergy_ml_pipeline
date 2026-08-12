# `redo/results/` — measured evidence for the redo methodology

> ⏱️ **Horizons covered: 24 h, 48 h and 72 h.** Unless a document says otherwise its numbers
> are **t + 24 h** — that is the project's primary horizon and the basis of documents
> 01–06. [07](07_horizon_sweep.md) is the exception: it sweeps 24/48/72 h to measure how
> skill decays with lead time. **Nothing here covers 96 h or beyond**; a weekly (168 h)
> model would need forecast leads that are available but not yet fetched.

> **What this folder is:** the measurements that justify the staged plan in
> [`../`](../README.md). Each document ties a result to the stage it evidences.
> **What it is not:** a second source of truth for the project's numbers.
> [`../../RESULTS.md`](../../RESULTS.md) remains authoritative for the chronological record.
> Where a result is already recorded there, this folder gives a one-line summary and a
> citation rather than restating it — the rule
> [stage 9](../09_stage9_writeup_and_hygiene.md) sets for the project applies to the redo
> folder too.

## Why this folder exists at all

The classical-vs-ML bakeoff was written to `outputs/reports/bakeoff.md`, which is
**auto-generated and overwritten by the next pipeline run**. That is by design — but it
means the evidence for several of the redo's central claims was sitting in a disposable
file. These documents are the durable copy.

## The four findings

**1. A 21-parameter linear model gets within 4.7% of gradient boosting.**
`mos_multi` — ordinary least squares on the day-1 forecast columns plus four calendar
terms — scores **0.8412 m/s** against XGBoost's **0.8037**, at a 24 h horizon over four
rolling-origin folds. The gap is statistically real (Diebold–Mariano p < 0.0001) and small.
**The project's ~30% v5 gain is therefore overwhelmingly an information effect, not a
machine-learning effect.** → [02](02_classical_vs_ml.md)

**2. Dropping every latency-blocked feature costs nothing.**
XGBoost on NWP + calendar only (55 features) scores **0.8009** — marginally *better* than
all 94 features (0.8037). ERA5 publishes ~5 days late, so its 49 derived features cannot
exist when a day-ahead forecast is issued. Removing them does not hurt. The deployability
concern is real in principle and **free in practice here.** → [03](03_operational_vs_study.md)

**3. Deep models are far more seed-sensitive than trees.**
LSTM's spread across seeds (sd 0.0281) is **~8× XGBoost's** (0.0034). Every prior
architecture conclusion in this project rested on single-seed runs, which could not have
seen this. → [04](04_deep_models.md)

**4. The protocol caught six defects that inspection had not.**
Including a SARIMAX configuration that produced numerically explosive forecasts, a
significance test silently comparing different time periods, and a **latent leak that would
have made any multi-day forecast look superb and mean nothing**.
→ [06](06_what_the_protocol_caught.md)

**5. Forecast skill decays, but decelerates — a weekly model looks viable.**
Across 24/48/72 h, XGBoost's skill against climatology falls 0.371 → 0.320 → 0.289, with the
loss per extra day shrinking. Extrapolated, a 7-day forecast would still beat climatology by
~25%. Also: **climatology overtakes persistence as the right baseline between 24 h and
48 h.** → [07](07_horizon_sweep.md)

## Contents

| File | What it holds | Redo stage evidenced |
|:---|:---|:---|
| [01_protocol.md](01_protocol.md) | The exact protocol — horizon, folds with dates, seeds, DM test, feature masks. **Read first**; the other files assume it. | [05](../05_stage5_baselines_and_protocol.md) |
| [02_classical_vs_ml.md](02_classical_vs_ml.md) | The 11-predictor ladder and both significance matrices | [05](../05_stage5_baselines_and_protocol.md), [06](../06_stage6_modelling.md) |
| [03_operational_vs_study.md](03_operational_vs_study.md) | The deployment-rehearsal measurement | [08](../08_stage8_deployment_rehearsal.md) |
| [04_deep_models.md](04_deep_models.md) | Deep architectures, seed sensitivity — ⚠️ **partial, TFT still running** | [06](../06_stage6_modelling.md) |
| [05_ceiling_and_information.md](05_ceiling_and_information.md) | The ceiling measurements, and how the bakeoff corroborates them | [04](../04_stage4_ceiling_analysis.md) |
| [06_what_the_protocol_caught.md](06_what_the_protocol_caught.md) | **Six** defects and the guard that surfaced each | [03](../03_stage3_alignment_harness.md) |
| [07_horizon_sweep.md](07_horizon_sweep.md) | 24/48/72 h decay; the baseline crossover; whether a weekly model is viable | [04](../04_stage4_ceiling_analysis.md), [05](../05_stage5_baselines_and_protocol.md) |

## Provenance

Every figure traces to one of these artifacts:

| Artifact | Produced by | Covers |
|:---|:---|:---|
| `outputs/reports/bakeoff.csv` / `.md` | `python -m src.bakeoff --config config_bakeoff.yaml` | Phase A, 11 predictors, 24 h |
| `outputs/reports/bakeoff_h48.*`, `bakeoff_h72.*` | same, `--horizon 48` / `72` | the horizon sweep |
| `outputs/logs/bakeoff_phase_a.log` | same run | per-fold detail |
| `outputs/reports/bakeoff_deep.csv` | `python -m src.bakeoff --phase b` | Phase B, deep models |
| [`../../RESULTS.md`](../../RESULTS.md) | the v5 pipeline run | the authoritative v5 headline |
| [`../../docs/11_nwp_forecast_features.md`](../../docs/11_nwp_forecast_features.md) | v5 iteration | gates, ceiling analysis |

**Run dates:** Phase A 2026-08-12 02:33–02:35 · Phase B started 2026-08-12 06:11.
