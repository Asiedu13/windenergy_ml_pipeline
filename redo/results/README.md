# `redo/results/` — measured evidence for the redo methodology

> ⏱️ **Horizons covered: 24 h through 168 h.** Unless a document says otherwise its numbers
> are **t + 24 h** — the project's primary horizon and the basis of documents 01–06.
> [07](07_horizon_sweep.md) sweeps 24/48/72 h; [09](09_weekly_forecast_results.md) covers
> **24 h to 168 h** at daily resolution.
>
> 🔮 **[08](08_weekly_forecast_design.md) is a design, not a measurement** — the only
> forward-looking document here, left unedited after the build so its refuted projection and
> prior stay visible. Its results live in [09](09_weekly_forecast_results.md).

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

**4. The protocol caught seven defects that inspection had not.**
Including a SARIMAX configuration producing numerically explosive forecasts, a significance
test silently comparing different time periods, a **latent leak that would have made any
multi-day forecast look superb and mean nothing**, and a cache that would have turned a
7-day model into a 3-day one. **Three of the seven surfaced within a day of starting the
weekly work** — novel configurations are where dormant bugs live.
→ [06](06_what_the_protocol_caught.md)

**5. Forecast skill decays, but decelerates — and a 7-day model works.**
Across 24/48/72 h, skill against climatology falls 0.371 → 0.320 → 0.289. Built out to
168 h, a **7-day forecast beats climatology by 20.7%** and predicts **weekly energy to ~4%
of rated capacity**. Also: **climatology overtakes persistence as the right baseline between
24 h and 48 h**, and the classical/ML gap *narrows* with lead (5.6% → 2.3%).
→ [07](07_horizon_sweep.md), [09](09_weekly_forecast_results.md)

**6. Two documented predictions were tested — one held, one was refuted.**
The 168 h skill projection of ~0.25 measured **0.207** (decay is faster than geometric). The
prior that losing an ensemble member would hurt was **wrong** — dropping ICON costs nothing,
so 168 h beats 144 h. And the quantile arm **fails calibration** (0.65–0.74 vs 0.80 nominal),
reported as a failure rather than fixed by widening.
→ [09](09_weekly_forecast_results.md)

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
| [08_weekly_forecast_design.md](08_weekly_forecast_design.md) | 🔮 **Design, written before the build.** Valid-time reframing, the ICON constraint, probabilistic output. Left unedited — including the projection and prior that 09 refutes | [08](../08_stage8_deployment_rehearsal.md) |
| [09_weekly_forecast_results.md](09_weekly_forecast_results.md) | **24 h → 168 h measured.** 7-day skill vs climatology, the ICON ablation, weekly energy, and a failing calibration check | [05](../05_stage5_baselines_and_protocol.md), [08](../08_stage8_deployment_rehearsal.md) |

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
