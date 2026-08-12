# 00 — Mistake register

> **What this is:** every significant mistake the original project made, what it cost, and
> the **specific structural guard** that would have caught it.
> **Why "structural":** "be more careful" is not a guard. A guard is a test, an ordering,
> or an assertion — something that fails on its own without anyone remembering to check.
> Every mistake below was made by someone who was already being careful.
> **Why it is written down at all:** four of these cost more than a week each, and two of
> them survived multiple review cycles precisely because the wrong output looked reasonable.

Each entry cites where the evidence lives in [`../docs/`](../docs/) so nothing here has to
be taken on trust.

---

## Group A — the expensive ones

These are the mistakes that cost iterations rather than hours.

### A1. Four iterations of modelling before measuring the ceiling

**What happened.** v1 through v4 tuned features, hyperparameters and architectures against
a feature set consisting entirely of the *analysed atmospheric state at time t*. In v5 the
24-hour autocorrelation of hub wind was finally measured at **0.571**, which caps R² at
**0.326** for any predictor built on that information. Four iterations of model work bought
about 0.06 R². One feature change — adding day-ahead NWP forecasts — bought 0.34.

**Cost.** Effectively the first four iterations, as a route to accuracy.
**Evidence.** [`11_nwp_forecast_features.md` §1](../docs/11_nwp_forecast_features.md).
**Guard.** Ceiling analysis is **stage 4**, and gates all modelling. See
[stage 4](04_stage4_ceiling_analysis.md).

### A2. "Not enough data" believed for months without ever being measured

**What happened.** Docs 01–10 attributed the residual gap to data volume, and a **10-year
ERA5 backfill** was launched on that basis. When the learning curve was finally run, it
flattened after ~1.6 years with a train/test ratio of 1.16 — more data cures variance, and
there was almost none left. The final doubling of training data bought **1.1%**.

**Cost.** A 44.8-hour download, halted mid-run, plus every decision that deferred to it.
**Evidence.** [`11 §1`](../docs/11_nwp_forecast_features.md), decision log entry 2026-08-10.
**Guard.** No expensive action may be justified by an unmeasured belief. If a claim is
driving a multi-day job, it is worth the two hours to measure it first.

### A3. The 24-hour alignment bug, and the essay written to explain it

**What happened.** The deep models were scored 24 hours out of phase, which pinned every
sequence model to the persistence error floor. All four clustered just above persistence —
which *looked* like a finding about deep learning and was written up as one. Doc 09 is a
complete, carefully argued essay explaining why LSTM/TFT/N-BEATS/DLinear underperform on
this problem. The premise was a measurement artefact. Fixing the alignment moved the LSTM
from 1.475 to 1.114 m/s and made it the **best** model.

**Cost.** Two iterations, one entire document invalidated, and a rewrite of the findings doc.
**Why it survived review.** The wrong answer was plausible *and agreed with a prior everyone
already held* ("deep learning needs more data than we have"). That combination is the single
most dangerous thing in this register.
**Evidence.** [`10_v4_deep_model_retraining.md`](../docs/10_v4_deep_model_retraining.md);
[`09_why_deep_learning_underperformed.md`](../docs/09_why_deep_learning_underperformed.md)
is preserved verbatim behind a supersede banner.
**Guard.** The perfect-forecast test, written **before the first model** — a deliberately
perfect forecast column must reproduce the target exactly, on a signal period
deliberately *not* commensurate with the horizon so no shifted frame can pass by
coincidence. See [stage 3](03_stage3_alignment_harness.md).

> **The generalisable rule:** four architectures clustering within 0.2 m/s of persistence is
> not a result, it is a *fingerprint* of measuring persistence four times. When independent
> methods agree suspiciously well, suspect the measurement, not the methods.

### A4. Deployability never checked until the project was finished

**What happened.** The final model uses 104 features, of which **53 are ERA5-derived**.
ERA5 is reanalysis, published with roughly five days of latency — so at the moment a
forecast for *t+24 h* would actually be issued, the state at *t* does not exist. The
deployed model would be missing half its inputs, and **no one has measured what it scores
without them**.

**Cost.** Not yet paid, but it undermines every operational claim the project could make.
**Evidence.** Not in `docs/` — found only when the question "is this deployable" was asked
directly, after v5 was complete.
**Guard.** The **deployment contract is stage 1**, and includes a per-source latency
budget. A feature that cannot exist at issue time is not a feature. See
[stage 1](01_stage1_deployment_contract.md) and
[stage 8](08_stage8_deployment_rehearsal.md).

---

## Group B — silent data defects

Every one of these produced plausible numbers. None announced itself.

| # | Defect | Cost | Guard |
|---:|:---|:---|:---|
| B1 | **NASA POWER's 50 m wind treated as 100 m.** `WS50M` was written into a column named `wind_speed_100m`, and the hub extrapolation defaulted to a 100 m reference — so the log law no-opped and hub wind was **understated by 10.7%**. v1's published results carry this error. | An invalid iteration | Source metadata table (height, units, timestamp convention) asserted **in code**, not remembered |
| B2 | **NASA POWER carries a ~1-hour timestamp offset.** MERRA-2 hourly values are hour-*averaged* and labelled at the end of the interval, so ERA5(t) matches NASA(t+1). | Caught by a gate before it did damage | Cross-source lag check at ingest |
| B3 | **hPa vs Pa in the same feature frame.** Open-Meteo reports surface pressure in hPa; the existing ERA5 table uses Pa. 100× apart, both in the same frame. | Caught before it did damage | Unit assertion at ingest, with the expected range per column |
| B4 | **The grid patch was never the shape it was written for.** `pad = 0.2°` was intended to fetch a 3×3 patch; it actually snaps to **1 latitude × 2 longitudes**. Consequence: all five `_grad_ns` columns are **identically zero**, and all five `_std` columns are exactly `\|_grad_ew\|/2`. Ten of twenty spatial features are mathematically dead, and were described as working features for three iterations. | Ten dead features, a false claim in the "defensible conclusions" section, **and — measured later — a comparison that would have been rigged**: trees tolerate zero-variance and perfectly-collinear columns, linear models do not, so the classical arm of the bakeoff would have been handicapped in exactly the ML-vs-classical comparison it was built to settle ([results/06](results/06_what_the_protocol_caught.md)) | **Assert the shape of what arrived, never the shape you requested.** One `assert ds.latitude.size == 3` at ingest would have caught it on day one |

**Evidence.** B1, B2: [`11 §10` decision log](../docs/11_nwp_forecast_features.md).
B3: [`11 §6`](../docs/11_nwp_forecast_features.md). B4:
[`03_data_sources.md`](../docs/03_data_sources.md) and
[`08_next_steps.md` §6b](../docs/08_next_steps.md).

> **The pattern across all four:** each was an assumption about *someone else's data* that
> was never checked against the data itself. Documentation describes intent; the file
> describes reality. Assert against the file.

---

## Group C — experiment-hygiene failures

### C1. A five-model bakeoff where one model solved a different problem

TFT was trained on darts' default **17-quantile objective** while BlockRNN, N-BEATS and
DLinear all trained on plain MSE. TFT was therefore fitting a substantially harder problem
and was ranked last for it. Putting it on the same objective recovered **0.380 m/s** —
1.551 → 1.171 — and moved it mid-field. A "head-to-head comparison" in which the models
optimise different losses is not a comparison.
**Evidence.** [`10 §3b`](../docs/10_v4_deep_model_retraining.md).

### C2. A config with no `deep:` block silently triples the training budget

`resolve_deep_cfg` falls back to `DEEP_DEFAULTS`, whose `overrides` are empty — so TFT runs
**200 epochs instead of 60**, and the learning-rate sweep is silently off. Caught only
because the resolved config was asserted before launching a multi-hour job.
**Guard for C1 and C2 alike.** **Print and assert the resolved config before any run longer
than an hour.** Not the config file — the resolved object the code will actually use.

### C3. The persistence baseline was never 24-hour persistence

`PersistenceForecaster` predicts from `hub(t-1)` and is scored against `hub(t+24)` — a
**25-hour** persistence, quoted throughout the project as "24 h persistence". True 24 h
persistence is 1.399 m/s, a 2.9% harder bar than the 1.440 quoted for v1–v4.
**Evidence.** [`RESULTS.md`](../RESULTS.md), [`10 §4`](../docs/10_v4_deep_model_retraining.md).
**Guard.** Baselines defined **once**, in one function, with a test asserting the exact lag.

### C4. A regression reported as an improvement

[`04_stage1_wind_speed.md`](../docs/04_stage1_wind_speed.md) presented v1 → v1.5 as
"1.045 → 1.171, better test performance". Those are a **6-hour horizon on NASA POWER** and a
**24-hour horizon on ERA5** — different problems. Against its own baseline v1 actually *lost*
to persistence (1.045 vs 0.875) while v1.5 beat it. The doc now carries a corrected
comparison against each version's own baseline.
**Guard.** Never compare two numbers without restating the window and the horizon for both.
If they differ, the comparison is invalid regardless of which direction it points.

### C5. Single seed, single split, from the first iteration to the last

Every number in the project, including the 30.6% headline, is one seed on one chronological
split. It is still not fixed.
**Guard.** Multi-seed is in the protocol from [stage 5](05_stage5_baselines_and_protocol.md),
reported as mean ± sd. It costs about an hour.

---

## Group D — process and infrastructure

| # | Mistake | Guard |
|---:|:---|:---|
| D1 | **Two ingest processes ran concurrently.** File existence was the only re-download guard, so both picked the same missing month and could write the same path. | A lockfile, and a "check for a running process" step that is part of the command, not the docs |
| D2 | **Parallel downloading was attempted on the assumption it would help.** It doesn't — CDS grants roughly one running job per user, and ~85 of every ~89 minutes is queueing. 75 minutes wasted proving it. | Measure the bottleneck before engineering around it. The timing was already in the logs |
| D3 | **Live-growing counts written into docs as static facts.** A cache count in `00_data_card.md` was wrong within minutes of being written. | Docs contain **check commands**, never snapshots of moving state |
| D4 | **Two commits, months of work untracked.** All of v2–v5 — every doc, every source change, every config — existed only in one machine's working tree, through multiple multi-day experiments. | Git is a **stage 1 exit criterion**, and a commit precedes every long-running job |

**Evidence.** D1, D2: [`11 §8` and §10](../docs/11_nwp_forecast_features.md).
D3: [`11 §8`](../docs/11_nwp_forecast_features.md). D4:
[`11 §11`](../docs/11_nwp_forecast_features.md).

---

## Group E — the guards themselves were weak

### E1. Two of the four verification gates could not fail

The four-gate harness in `src/validate_nwp.py` was one of the project's better ideas, and
it did real work — it caught B1 and B2. But
[`gate_3_baselines`](../src/validate_nwp.py) and
[`gate_4_experiment`](../src/validate_nwp.py) both **hardcode `passed=True`**. Gate 3's
"sanity band" only changes the wording of its own detail line; gate 4 passes as long as any
`nwp_*` column exists.

So "all four gates pass" — a phrase that appears in several documents — actually means "two
gates passed and two reported". Only gates 1 and 2 can fail.

**Guard.** **A gate that cannot fail is not a gate.** Every gate must be demonstrated red
on deliberately broken input before it is trusted green. See
[stage 3](03_stage3_alignment_harness.md).

**This was subsequently vindicated four times over.** The bakeoff's guards — a sanity floor,
a length assertion, an exact-lag test, a dead-column test — surfaced four defects in a single
day's work, none of which crashed and three of which produced numbers that agreed with a
reasonable prior. Worked examples:
[results/06](results/06_what_the_protocol_caught.md).

### E2. Documentation written before the numbers were stable

Doc 09 explains an artefact. Doc 07 — explicitly "the doc written to be quoted" — has been
rewritten once and is stale again as of today, because two of its five headline findings
were overtaken by later measurements.

**Guard.** One results file is the single source of truth. Interpretation documents are
written **last**, and every number in them cites the run that produced it. See
[stage 9](09_stage9_writeup_and_hygiene.md).

---

---

## Group F — mistakes made *building the redo itself*

> **Why these belong here.** This register would be dishonest if it only catalogued the
> original project's errors. The bakeoff written to test the redo's own claims introduced two
> defects of exactly the kinds described above — which is the point: the failure modes are
> structural, not a matter of who is being careful. Full detail:
> [results/06](results/06_what_the_protocol_caught.md).

### F1. A SARIMAX configuration that produced numerically explosive forecasts

`sarimax_nwp` scored **MAE 4.4619** against persistence's 1.3183 — R² −7.82. Cause:
`enforce_stationarity=False` with no constant term, so an AR root outside the unit circle
made a 2,227-step-ahead forecast diverge.

**Caught by** the sanity floor written into the plan before the run: *every predictor above
tier 0 must beat naive persistence; any that doesn't is a bug until proven otherwise.*
Fixed → **1.2554**.

**Why it was dangerous:** the buggy number and the correct number support the *same
qualitative conclusion* — ARIMA adds little at a 24 h horizon. Without the explicit floor it
would have been accepted as a finding. Same mechanism as [A3](#a3-the-24-hour-alignment-bug-and-the-essay-written-to-explain-it).

### F2. A significance test silently comparing different time periods

Deterministic models emit one error vector per fold (4); stochastic models emit one per seed
per fold (20). The first Diebold–Mariano implementation noticed the mismatch and *truncated
to the shortest*, judging stochastic models on **fold 1 alone** while classical models were
judged on all four — in a window where fold difficulty varies by 46%.

**Caught by** the inconsistency it produced downstream (two tables nominating different
"best" models). **Fixed** by averaging within fold, concatenating chronologically, and
asserting equal lengths instead of reconciling them.

**The rule:** when two things should be the same length and are not, **assert — never
truncate**. The mismatch *was* the signal.

---

## The one-sentence summary

Almost every entry above is a case of **trusting a plausible number, a documented intent, or
an unexamined belief, in place of a measurement that would have taken under a day** — which
is why the stages that follow are ordered to force the measurement first.
