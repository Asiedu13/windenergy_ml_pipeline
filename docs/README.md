# Ghana Wind Power Forecasting Pipeline — Documentation

> ✅ **v5 has landed (2026-08-11 23:18).** Adding operational day-ahead NWP forecasts as
> Stage 1 features cuts test MAE by **30.6%** (1.0513 → 0.7295 m/s) and lifts R² from
> **0.3300 → 0.6653** on the full 22-month ERA5 window, past the **0.326** ceiling that
> bounds any predictor built on the atmospheric state at time *t*. End-to-end power reaches
> **7.48% nMAE / 10.08% nRMSE** against v4's 11.25% / 15.50%. All four verification gates
> pass. Full detail: [11 §7a‑final](11_nwp_forecast_features.md).
>
> **One thing still outstanding before this is a headline claim:** it is a **single seed on
> a single split** ([12 item 4](12_next_steps_v2.md)). Quote the direction and rough
> magnitude, not a precise figure, until the multi-seed repeat is done.
>
> ✅ **The second caveat is retired.** The five-model bakeoff that broke v4's architecture
> near-tie *was* run on NASA POWER truth, and has since been repeated on ERA5 over 3 seeds
> (2026-08-12): XGBoost still wins, but by **5.1%** rather than 18%, and TFT went from best
> of the deep bracket to **last**. → [`../redo/results/04`](../redo/results/04_deep_models.md)
>
> 📘 **For interpretation, read [`../thesis/04_results_and_discussion.md`](../thesis/04_results_and_discussion.md).**
> It supersedes [07_findings_and_analysis.md](07_findings_and_analysis.md), which remains the
> v4-era record.

Detailed documentation for the two-stage wind power forecasting pipeline built for
Anloga, Ghana. Every design decision, every hyperparameter, every result — written
so the pipeline can be read, defended, and extended.

For a short chronological summary of results across iterations, see
[../RESULTS.md](../RESULTS.md). For an auto-generated report from the most recent
pipeline run, see [../outputs/reports/report.md](../outputs/reports/report.md).

> ⚠️ **`report.md` is not a v4 or v5 report.** It is overwritten by whichever pipeline run
> finished last — currently the **NASA POWER NWP pilot** (2026-08-11 18:10), whose MAEs are
> a cross-source check and are **not comparable to any ERA5 figure**. The full v4
> five-model report is preserved separately at `../outputs/reports/report_v4_bakeoff.md`.
> **`RESULTS.md` is the source of truth for every consolidated table**, not the
> auto-generated report.

> 🆕 **New to this project?** Read [glossary.md](glossary.md) first. It defines every term
> these docs use without explaining — reanalysis, hub height, persistence baseline, nMAE,
> power curve, SHAP — in plain English, with this project's own numbers as examples.
>
> Then read this page's executive summary, then [01_problem_and_design.md](01_problem_and_design.md).

## Executive summary

> **What:** the whole project in one page — the problem, the two-stage design, and the
> best numbers so far.
> **Why:** so you can decide whether you need any of the other twelve documents.

**Goal:** Forecast day-ahead wind power output for a Ghanaian site using historical
reanalysis weather. No operating utility wind farm publishes SCADA data in Ghana, so
the pipeline forecasts *potential* output from ERA5 weather feeds through a
manufacturer power curve — the "physical/indirect" two-stage architecture from
Donadio, Fang & Porté-Agel (Energies 2021).

**Architecture:**
1. **Stage 1 (ML)** predicts hub-height wind speed at t+24h from ERA5 features
   (surface + upper-air + spatial gradients + calendar).
2. **Stage 2 (Simulator)** converts predicted wind speed to power via
   `windpowerlib`'s manufacturer power curve for an Enercon E-101/3050 (3.05 MW),
   density-corrected for Ghana's heat.

**Best result (v5, ERA5 2024-03 → 2025-12):**
- **Stage 1 (Optuna-XGBoost + NWP forecast features):** test MAE **0.7295 m/s**,
  R² **0.6653**, against 1.0513 / 0.3300 for the same model without the NWP block —
  a **30.6%** reduction.
- **Two-stage power forecast:** **nMAE 7.48%, nRMSE 10.08%** (rated 3.05 MW).
  Reference: Energies 2021 two-stage benchmark ~8.8% nMAE / 13% nRMSE. Single seed,
  single split, different site — see [12](12_next_steps_v2.md) for what may be claimed.

**How v4 got here.** v4 peaked at **1.114 m/s / nMAE 11.25%** and stalled, because the gap
was **information-limited, not sample-limited**: the learning curve flattens after ~1.6
years at a 1.16 train/test ratio, and a 24 h autocorrelation of 0.571 caps R² at **0.326**
for *any* predictor built on the analysed state at time *t*. Only genuine forecast
information can break that ceiling — which is what v5 added, and R² 0.6653 is the evidence
it worked. See [11_nwp_forecast_features.md §1](11_nwp_forecast_features.md).

**Key finding:** with identical information sets, a uniform MSE objective and equivalent
training care, a single-layer LSTM (1.114 m/s) modestly outperforms a 50-trial
Optuna-tuned XGBoost (1.143), with N-BEATS (1.157), TFT (1.171) and DLinear (1.215)
close behind. **All five beat 24 h persistence (1.399).** The spread across
architectures (0.101 m/s) is smaller than the gap from the weakest of them to
persistence (0.184 m/s) — architecture choice matters far less here than correct
temporal alignment, a comparable training objective, and validation-based stopping.
(Single seed, single split: treat the LSTM/XGBoost ordering as a tie.)

> ⚠️ **This reverses the v3 conclusion**, which was that deep models lose to XGBoost on
> this problem. That conclusion rested on deep-model scores that were **24 hours out of
> phase** — an artefact that pinned every sequence model to the persistence error floor.
> See [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md) for the bug, the
> fix, and the corrected results.
>
> Doc 07 has been **rewritten** against the corrected numbers (it is the doc written to
> be quoted, so leaving refuted figures in it was too dangerous). Doc 09 is kept
> **verbatim** behind a supersede banner, and doc 06 retains the v2/v3 tables struck
> through — between them the original reasoning is fully preserved.

## How to read these docs

> **What:** an index of all thirteen documents and what each one answers.
> **Why:** they are self-contained and can be read in any order — this table saves you
> opening the wrong one.

**Suggested path for a newcomer:** [glossary](glossary.md) → [01 problem & design](01_problem_and_design.md)
→ [00 data card](00_data_card.md) → [02 architecture](02_pipeline_architecture.md) →
[11 §7a‑final](11_nwp_forecast_features.md) for the current result →
[`../thesis/04_results_and_discussion.md`](../thesis/04_results_and_discussion.md) for what
it means. The rest are reference. Read [07 findings](07_findings_and_analysis.md) for the
v4-era interpretation and the methodological findings, **not** for current figures.

| # | File | What it covers |
|---:|:---|:---|
| — | [glossary.md](glossary.md) | **Start here if any term is unfamiliar.** Plain-English definitions of every piece of jargon in these docs, grouped by topic. |
| 0 | [00_data_card.md](00_data_card.md) | **Lookup table:** every source, every column with its unit, what is actually cached on disk, derived features, and known data limitations. Start here when you need a fact about the data. |
| 1 | [01_problem_and_design.md](01_problem_and_design.md) | Why this problem, why the two-stage architecture, non-negotiable design principles, alternatives rejected. |
| 2 | [02_pipeline_architecture.md](02_pipeline_architecture.md) | Module map, data flow, config schema, testing strategy. |
| 3 | [03_data_sources.md](03_data_sources.md) | ERA5 vs NASA POWER, complete variable list, CDS chunking + cost math, ZIP-unpacking quirk, spatial derivatives. |
| 4 | [04_stage1_wind_speed.md](04_stage1_wind_speed.md) | Feature engineering, baselines (persistence + climatology), XGBoost + Optuna, deep models (LSTM/TFT/N-BEATS/DLinear), SHAP + Pearson analysis. |
| 5 | [05_stage2_and_evaluation.md](05_stage2_and_evaluation.md) | Turbine, power curve, air-density correction, error propagation, three-variant comparison, metric definitions. |
| 6 | [06_iteration_history.md](06_iteration_history.md) | v1 → v1.5 → v2 → v3 → v4 → v4b, every decision, every hyperparameter change, every result, every bug and fix. |
| 7 | [07_findings_and_analysis.md](07_findings_and_analysis.md) | **v4-era interpretation; superseded as the quotable layer by [`../thesis/04`](../thesis/04_results_and_discussion.md).** Why the deep models *appeared* to underperform, feature-ranking interpretation, positioning vs literature, threats to validity. Its findings 2 and 4 and its "defensible conclusions" 1–3 carry v5 supersede notes — read them, don't quote around them. |
| 8 | [08_next_steps.md](08_next_steps.md) | Ranked improvements with effort estimates and expected ROI. **Largely superseded by doc 12** — its tiers assume a data-volume constraint that has since been refuted. |
| 9 | [09_why_deep_learning_underperformed.md](09_why_deep_learning_underperformed.md) | Original word-for-word explanation of why LSTM/TFT/N-BEATS/DLinear lost to XGBoost. **Superseded by doc 10 — the premise was a measurement artefact.** |
| 10 | [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md) | The 24-hour alignment bug, doc 09's recommendations implemented and evaluated, corrected 5-model bakeoff, the 25 h persistence-baseline issue. |
| 12 | [12_next_steps_v2.md](12_next_steps_v2.md) | **Current priorities**, rewritten after the NWP result. What is now settled, four tiers of remaining work, threats to the new result, and the results table so far. Replaces doc 08. |
| 11 | [11_nwp_forecast_features.md](11_nwp_forecast_features.md) | **v5, landed — the authoritative result (§7a‑final) and the boundary list (§9).** Measured refutation of the "data-limited" claim (learning curve, ablation, autocorrelation ceiling), Open-Meteo vs GFS-GRIB source selection, the alignment contract, and a dated decision log. §7 (verification gates with pass criteria and failure actions) and §7b (pre-registered outcome branches) are kept **unedited as pre-registration** — all four gates passed; read their outcomes in §7a‑final. |

## Quick reference

**Running the pipeline:**
```bash
python -m src.pipeline --config config.yaml --mode forecast --compare
```

**Extending ERA5 to 2010–2023** (resumable; re-run until it reports zero missing):
```bash
python -m src.data_ingest_extend --config config_10yr.yaml
```
> ⚠️ **Deliberately halted on 2026-08-10 at 2016-07.** The 10-year extension was stopped
> once the learning curve showed more history buys ~1% MAE
> ([07_findings_and_analysis.md](07_findings_and_analysis.md)); the CDS queue went to the
> 2024–2026 window that `config_nwp.yaml` needs instead. 2010-01→2016-07 is complete and
> resumable. `config_10yr.yaml` still aborts in `src.pipeline` on the missing
> 2016-08→2019-12 core months. See [00_data_card.md §2](00_data_card.md).
>
> Before launching any ingest, check nothing is already running — existence of the cached
> file is the only re-download guard, so two concurrent runs issue duplicate CDS requests
> and can write the same path twice:
> `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select ProcessId, CommandLine`

**Adding NWP forecast features** (the current line of work):
```bash
python -m src.pipeline --config config_nwp.yaml --mode forecast
```

**Test suite (38 tests):**
```bash
pytest -q
```

**Key files in the codebase** (linked to full paths from each doc):
- Config: [`config.yaml`](../config.yaml)
- Data ingestion: [`src/data_ingest.py`](../src/data_ingest.py)
- Features: [`src/features.py`](../src/features.py)
- Stage 1 tree models: [`src/stage1_windspeed.py`](../src/stage1_windspeed.py)
- Stage 1 deep models: [`src/stage1_deep.py`](../src/stage1_deep.py)
- Stage 2 power curve: [`src/stage2_powercurve.py`](../src/stage2_powercurve.py)
- Evaluation: [`src/evaluate.py`](../src/evaluate.py)
- Pipeline orchestrator: [`src/pipeline.py`](../src/pipeline.py)
- Report generation: [`src/report.py`](../src/report.py)
