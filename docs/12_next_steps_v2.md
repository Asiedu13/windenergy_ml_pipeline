# 12 — Next Steps v2 (post-NWP)

> 📐 **Reading the numbers.** Figures like `0.7621` are **test MAE in m/s** — typical error
> predicting hub-height wind speed 24 h ahead, **lower is better**. Percentages like
> `7.66%` are power error as a share of the turbine's 3.05 MW rating, also lower-is-better.
> R² runs the other way. Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** the replacement for [08_next_steps.md](08_next_steps.md),
> rewritten after the NWP-forecast result changed the project's central constraint.
> **Why a v2 rather than an edit:** doc 08's entire tier structure assumes the bottleneck
> is data volume. That assumption has been measured and refuted, so patching it would
> leave the reasoning incoherent. Doc 08 is kept for provenance.

## What is now settled — do not re-litigate

Each of these was an open question in doc 08 and is now measured:

| Question | Answer | Evidence |
|:---|:---|:---|
| Is more ERA5 history the lever? | **No** — worth ~1% MAE | Learning curve flattens after ~1.6 yr; train/test ratio 1.16 ([11 §1](11_nwp_forecast_features.md)) |
| Is architecture choice the lever? | **No** on state-only features | All five models within 0.101 m/s ([10](10_v4_deep_model_retraining.md)) |
| Is the gap information or samples? | **Information** | 24 h autocorrelation 0.571 caps R² at 0.326; nowcast MAE 0.015 localises all error to the 24 h gap |
| Do NWP forecasts help? | **Yes, materially — ~23%** | Cross-validated on two truth sources ([11 §7a](11_nwp_forecast_features.md)) |

**The headline.** Adding archived day-ahead forecasts lifts R² from ~0.48 to **~0.70**,
decisively past the 0.326 ceiling that bounds any predictor built on the atmospheric state
at time *t*. Two-stage power error drops from **11.25% nMAE (v4)** to **7.66%** on ERA5
truth over 2024-03 → 2025-02.

## Tier 1 — finish what the result opens up

### 1. Authoritative ERA5 run on the full 22-month window — ✅ *done 2026-08-11 23:18*

**Result:** all four gates pass on 15,912 rows × 104 features. Stage 1 **1.0513 → 0.7295
m/s (−30.6%)**, R² **0.3300 → 0.6653**. End-to-end **7.48% nMAE / 10.08% nRMSE**.

**The prediction here was close but low.** This item forecast "near R² 0.70 with a ~23%
gain"; the actual is R² 0.665 with a **30.6%** gain. The three estimates now bracket the
truth from both sides — 12-month ERA5 +40.0% (inflated), NASA POWER +23.2% (smoother
truth), full ERA5 **+30.6%**. Quote the last one.

### 2. Five-model bakeoff on NWP features — ✅ *done 2026-08-11; the near-tie broke*

**Result:** the near-tie does **not** survive. XGBoost 0.678 → TFT 0.827 → LSTM 0.903 →
N-BEATS 0.942 → DLinear 0.969 m/s (NASA POWER truth, 22 months). Spread **0.291 m/s, ~3×
v4's 0.101**; XGBoost leads the best deep model by 18%. All five still beat persistence
(1.110). Split, alignment and hyperparameters were each ruled out as artefacts —
[11 §7a](11_nwp_forecast_features.md).

**Note the direction was the opposite of what this item predicted.** The expectation
written here was that sequence models might exploit a forecast trajectory better than
trees; every deep model instead lost ground. Recording the failed prediction rather than
quietly rewriting it is the point of pre-registering.

**Still owed before this is claimed:** an ERA5 repeat (item 1) and multiple seeds (item 4).
One source, one seed, one split.

### 3. Two-stage power evaluation with NWP — ✅ *done, see §Results below*

### 4. Multi-seed repeat — **now the highest-value outstanding item**

**Why:** carried from doc 08 tier 1 and still not done. It matters *more* now, not less: a
large claimed effect attracts more scrutiny than a small one, and every NWP number is a
single seed on a single chronological split.
**What to change:** loop `model.seed` over ≥5 values in `src/validate_nwp.py`; report
mean ± sd for the with/without gap.
**Effort:** ~1 h for the XGBoost arms.
**Expected ROI:** converts "NWP improves MAE by 23%" into a defensible interval. Cheap
insurance on the project's headline claim.

### 4b. Classical baselines — ✅ *done 2026-08-12, and it reframes the headline*

**Why it was added:** doc 08 and this document both assumed the interesting comparison was
between ML architectures. It was not. The project had never once compared against classical
NWP post-processing, only against naive persistence.

**Result:** `mos_multi` — OLS on the day-1 forecast columns plus calendar, **21
parameters** — scores 0.8412 m/s against XGBoost's 0.8037 on a 4-fold rolling-origin
protocol. Diebold–Mariano confirms the gap is real (p < 0.0001) but it is 4.7%.

**What that means for the headline.** The ~30% gain from adding NWP features is
**overwhelmingly an information effect, not a machine-learning effect**. Say it that way:
it is more defensible, and it survives an examiner asking "did you need ML at all?" —
the answer is "for the last 4.7%, yes, and here is the significance test."

**And the operational arm is free.** XGBoost on NWP + calendar only (55 features, no
reanalysis) scores **0.8009** — marginally *better* than the full 94-feature set. The ERA5
latency problem that [redo stage 8](../redo/08_stage8_deployment_rehearsal.md) raises is
real in principle and **costs nothing here**. Threat 5 below is unaffected; this is a
separate, better outcome than expected.

Full table and DM matrices: [`../outputs/reports/bakeoff.md`](../outputs/reports/bakeoff.md).

## Tier 2 — newly attractive because of the result

### 5. Multi-horizon forecasting — ✅ *done 2026-08-12, and it went to 7 days*

**Result:** built out to **168 h**, seven models on a valid-time framing (one per forecast
lead day). A 7-day forecast beats climatology by **20.7%**; weekly energy lands within **~4%
of rated capacity**. Full table in [`../RESULTS.md`](../RESULTS.md), detail and limitations
in [`redo/results/09`](../redo/results/09_weekly_forecast_results.md).

**One correctness note that matters more than the result.** Setting a longer horizon on the
old feature-time path would have **leaked the future**: `nwp_*_d{N}` at valid time τ was
issued 24·N hours before τ, so row *t* may only use leads with `24·N ≥ h`. At 168 h the `d1`
column would have carried a forecast issued at `t + 144` — six days ahead. Nothing checked
this until it was looked for. `add_nwp_features` now drops short leads and raises when none
survives.

**What this item under-predicted.** It scoped a horizon curve from the `d2`/`d3` columns
already on disk. Leads `d4`–`d7` turned out to be available from the same API for ~6 MB, so
the ceiling was seven days rather than three — and the interesting findings were at the far
end: the classical/ML gap **narrows** with lead (5.6% → 2.3%), and by day 7 raw ensemble
averaging beats single-model MOS.

### 6. Probabilistic forecasting — ⚠️ *first attempt built 2026-08-12; calibration FAILS*

**Why:** doc 08 ranked this mid-table partly because there was no natural uncertainty
signal. There is one now: **three independent NWP models whose disagreement is itself
informative**. Wide spread between ECMWF, GFS and ICON on a given hour is a physically
meaningful "low confidence" indicator, and it is already fetched.

**Built:** per-hour ensemble spread as a feature, plus quantile gradient boosting (pinball
objective) for P10/P50/P90 per lead day, trained **separately** from the point model.

**Result: not calibrated.** Observed P10–P90 coverage is **0.65–0.74 against 0.80 nominal**
at every lead from 24 h to 168 h — the bands are systematically too narrow, roughly
symmetrically, so it is a width problem rather than bias. **Reported as a failure rather
than fixed by widening**, because an interval tuned until it covers is fitted to the test
set, not calibrated.

**What does work:** interval width grows monotonically with lead (1.81 → 2.70 m/s) and
coverage improves with it, so the spread feature is tracking real uncertainty. The model is
simply under-dispersed throughout.

**Remaining work:** recalibrate on a held-out validation fold, or wrap in conformal
prediction, which guarantees coverage by construction rather than by fitting.
[`redo/results/09`](../redo/results/09_weekly_forecast_results.md).

### 7. Richer NWP inputs via GFS GRIB — **upgraded from "only on a null result"**

**Why:** doc 11 §7b reserved this for a null result. That reservation is void — five
Open-Meteo variables bought ~23%, which makes *more* forecast variables a plausible route
to more. GFS carries the full 10/20/30/40/50/80/100 m wind profile plus `HPBL` and
`FRICV`, letting a model see shear and stability directly instead of inferring them.
**What to change:** the GRIB fetcher scoped in [11 §2](11_nwp_forecast_features.md).
**Effort:** ~a week; 30–400 GB (every GRIB message is a global field).
**ROI:** genuinely uncertain, but the prior has moved favourably.

## Tier 3 — consequences of a better Stage 1

### 8. Error-propagation re-analysis

**Why:** the power curve's roughly cubic mid-section amplifies wind error. As Stage 1 error
shrinks, the *relative* contribution of that amplification grows, so the existing
error-propagation figure (design principle 3, [01](01_problem_and_design.md)) is now
measuring a different regime and should be regenerated.
**Effort:** hours. **ROI:** keeps an existing analysis honest.

### 9. Operational realism of the lead time

**Why:** Open-Meteo's `previous_day1` is a fixed **day** offset, so the effective lead
varies through the day rather than being a constant 24 h. A genuinely operational setup
fixes the *issue* time (e.g. always the 00Z run) and lets the lead vary accordingly.
**What to change:** verify the per-hour lead empirically, then decide whether to restrict
to a fixed issue cycle.
**Effort:** a day. **ROI:** removes a caveat rather than adding accuracy.

## Threats to the new result — state these before someone else does

1. **Single seed, single split.** Item 4 addresses it. Until then, quote the direction and
   rough magnitude, not a precise figure.
2. **ERA5 and ECMWF IFS share lineage.** ERA5 is produced *with* ECMWF's own IFS, so
   verifying an ECMWF forecast against ERA5 truth is not fully independent — shared
   dynamics and physics flatter the forecast.
   **Partially mitigated:** on NASA POWER truth (MERRA-2 lineage, NASA/GMAO, independent
   of ECMWF), ECMWF still scored best of the three (rescaled MAE 0.8301 vs GFS 0.9397,
   ICON 0.8890). Its edge is therefore not purely kinship — but the caveat stands and
   belongs in any write-up.
3. **Lead-time semantics assumed.** `previous_day1` is documented as ~24 h before valid
   time and behaves consistently with that (skill decays d1 → d2 → d3), but it has not
   been verified hour-by-hour. Item 9.
4. ~~**Both NWP windows are calmer than 2020–2023.**~~ **RETRACTED 2026-08-11** — this was
   true of the 12-month ERA5 subset (persistence 1.09 m/s) and of NASA POWER, but **not of
   the full 22-month ERA5 window, where persistence is 1.3097 m/s** against 1.399 on
   2020–2023. A 4.5% easier bar is not "a genuinely easier problem". The benchmark
   comparison no longer needs this hedge; the remaining hedges (different site, single
   seed, physics-derived truth) still stand.
5. **`model_chain_from_weather` carries a latent 100 m assumption.** It would treat NASA
   POWER's 50 m wind as hub height, the same defect fixed in the feature path. Verified
   called from nowhere in `src/` or `tests/`, so it affects no result — but fix it before
   anyone uses that entry point.

## Results (2026-08-11, final)

**Stage 1 — hub wind speed at t+24 h.** The ERA5 22-month column is authoritative; the
other two are kept to show how the estimate moved.

| | **ERA5, 22 mo ✅** | ERA5, 12 mo | NASA POWER, 22 mo |
|:---|---:|---:|---:|
| without NWP | **1.0513 · R² 0.3300** | 1.2695 · R² 0.2415 | 0.8827 · R² 0.4846 |
| **with NWP** | **0.7295 · R² 0.6653** | 0.7621 · R² 0.7173 | 0.6780 · R² 0.7024 |
| gain | **+30.6%** | +40.0% *(inflated)* | +23.2% *(smoother truth)* |

**Stage 1+2 — end-to-end power, with NWP**

| Variant | **ERA5 22 mo ✅** | ERA5 12 mo | NASA POWER 22 mo |
|:---|---:|---:|---:|
| physics_only | **13.78% / 18.30%** | 11.21% / 15.03% | 7.53% / 11.03% |
| **two_stage** | **7.48% nMAE / 10.08% nRMSE** | 7.66% / 10.55% | 4.43% / 6.38% |
| direct | **7.43% / 9.85%** | 7.49% / 9.94% | 5.29% / 7.86% |

**On the benchmark comparison.** The Energies 2021 reference is ~8.8% nMAE / ~13% nRMSE,
and **7.48% / 10.08%** sits below it — now on a window whose persistence baseline
(1.31 m/s) is comparable to the literature setting rather than markedly calmer, which
removes the hedge that most weakened the earlier pilot figure.

**It is still not like-for-like**, and the claim must carry those limits: different site,
different turbine assumptions, physics-derived truth rather than SCADA, and a **single seed
on a single chronological split**. Defensible phrasing: *"with day-ahead NWP forecasts as
features, the two-stage pipeline reaches 7.48% nMAE on a 22-month ERA5 window — at or
below the published two-stage reference, on a window of comparable difficulty, though not
under like-for-like conditions."* Item 4 is what upgrades this further.

`two_stage` and `direct` are within **0.05 pp** here and swap order across truth sources.
That ordering is not established and should not be argued from.

## Retired from doc 08

- **"Extend the training window to 10+ years"** — measured worthless (~1% MAE); the
  backfill was deliberately halted at 2016-07.
- **Architecture search as a primary lever** — settled in v4 and unchanged by this work
  unless the in-flight bakeoff says otherwise.
- **GFS GRIB gated behind a null result** — that gate is void; see item 7.
