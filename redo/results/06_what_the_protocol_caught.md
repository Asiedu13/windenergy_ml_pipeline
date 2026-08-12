# 06 — What the protocol caught

> **What:** four defects surfaced by the bakeoff's own guards during a single day's work.
> **Why this file exists:** [stage 3](../03_stage3_alignment_harness.md) argues that guards
> must be built before models and must be able to *fail*. That argument is cheap to make and
> easy to ignore. This is what it bought in practice — every one of these produced a
> plausible-looking number, and none announced itself.
> **All results at a 24 h horizon.** Protocol: [01](01_protocol.md).

---

## Defect 1 — SARIMAX produced numerically explosive forecasts

**Symptom.** First Phase A run: `sarimax_nwp` scored **MAE 4.4619**, against persistence's
1.3183 — more than three times worse than "tomorrow will be like today", with R² **−7.82**.

**What caught it.** The **sanity floor**, written into the plan before the run:

> *Every predictor above tier 0 must beat naive persistence. Any that doesn't is a bug until
> proven otherwise.*

Without that rule the number would have been reported as "ARIMA-family methods perform
poorly on this problem" — which is *directionally true* (see below), and would therefore
have been very easy to accept.

**Cause.** Two settings, both wrong together:

- `enforce_stationarity=False` allowed an AR root outside the unit circle, so a
  2,227-step-ahead forecast **diverged**;
- no constant term (`trend` defaulted off with exog present), forcing the exogenous
  regression through the origin.

**Fix.** `trend="c"`, `enforce_stationarity=True`, `enforce_invertibility=True`, plus an
`ar_roots_stable_` attribute recording whether the fitted roots were in fact stable.

**After the fix: 1.2554** — beating persistence, with a skill of +0.022.

**The lesson.** The corrected result and the buggy result support the *same qualitative
conclusion*: ARIMA adds almost nothing at this horizon. That is exactly what made the bug
dangerous. A wrong number that agrees with your expectation is the hardest kind to catch,
and it is the same mechanism that let a 24-hour alignment bug survive two review cycles
earlier in this project ([register A3](../00_mistake_register.md)).

---

## Defect 2 — the significance test was comparing different time periods

**Symptom.** In the first run the `study` table ranked `xgboost` first while the
`operational` Diebold–Mariano section used `lightgbm` as its reference "best" model. A
small inconsistency — and a symptom of something worse.

**Cause.** Deterministic models emit **one error vector per fold** (4 total); stochastic
models emit **one per seed per fold** (20 total). The first implementation noticed the
length mismatch and "fixed" it by truncating everything to the shortest:

```python
n = min(lengths)
pooled = {m: v[:n] for m, v in pooled.items()}
```

That silently judged stochastic models on **fold 1 alone** while classical models were
judged on all four — comparing models across *different quarters of the year*, in a window
where fold difficulty varies by 46%.

**Fix.** Average errors across seeds **within** each fold, concatenate folds in chronological
order, and assert equal lengths rather than papering over inequality:

```python
assert len({len(v) for v in pooled.values()}) == 1, "DM error vectors misaligned"
```

**The lesson.** The original code did not crash and produced p-values that looked entirely
reasonable. A silent reconciliation of mismatched data is worse than an error, because the
mismatch *was* the signal. When two things should be the same length and aren't,
**assert — never truncate.**

---

## Defect 3 — "24-hour persistence" was a 25-hour persistence

**Symptom.** The project's `PersistenceForecaster` predicts from the `wind_speed_hub_lag1`
column — `hub(t−1)` — while being scored against `hub(t+24)`. That is **25-hour**
persistence, quoted as 24-hour throughout v1–v4.

**Scale.** True 24 h persistence is **1.399 m/s** against the 1.440 quoted — a **2.9%**
easier bar than the one advertised, applied to every claim of the form "the model beats
persistence by X%".

**Why it persisted.** It was known and documented
([`docs/10 §4`](../../docs/10_v4_deep_model_retraining.md)) but left unchanged so that
v1–v4 stayed mutually comparable. A defensible call that quietly hardened into a permanent
one.

**Fix.** `NaivePersistence` in `src/traditional.py` reads `hub(t)` directly, and a test
fails if the `_lag1` column is ever used:

```python
np.testing.assert_allclose(pred, ctx.hub_test.to_numpy())
assert not np.allclose(pred, ctx.X_test["wind_speed_hub_lag1"].to_numpy())
```

**Verified**: the harness reproduces the published true-persistence figure **1.3097 m/s**
exactly on the original single split.

**The lesson.** [Stage 5](../05_stage5_baselines_and_protocol.md): baselines get defined
once, in one function, with a test asserting the exact lag. A baseline is the denominator of
every claim you will make, and a 2.9% error in it propagates to all of them.

---

## Defect 4 — ten features that were mathematically dead

**Symptom.** Five columns with zero variance and five that are exact linear functions of
other columns.

**Cause.** The ERA5 request uses `pad = 0.2°`, written intending a **3×3** grid patch. It
actually snaps to **1 latitude × 2 longitudes**. Consequently:

- all five `*_grad_ns` columns are **identically zero** — with one latitude row, the north
  and south edges are the same row;
- all five `*_std` columns are exactly `|*_grad_ew| / 2`.

Ten of twenty spatial features carry no information, and were described as working features
for three iterations.

**Impact on this bakeoff.** Trees tolerate both. Linear models do not — zero-variance and
perfectly-collinear columns are precisely what breaks OLS — so the classical rungs would
have been unfairly handicapped in a comparison against trees.

**Fix.** Excluded from both feature sets via `is_dead_column`, with a test that distinguishes
them from the rolling-statistic columns whose names look similar:

```python
assert is_dead_column("pressure_pa_std")
assert not is_dead_column("wind_speed_hub_rollstd3")
```

**The lesson.** [Stage 2](../02_stage2_data_availability_audit.md): **assert the shape of
what arrived, never the shape you requested.** One `assert ds.latitude.size == 3` at ingest
would have caught this on the first download, three iterations earlier.

---

---

## Defect 5 — no guard stopped a longer horizon from reading the future

**Symptom.** None, until it was looked for. `add_nwp_features` joined *every* `nwp_*_d{N}`
column regardless of horizon. At the 24 h horizon that is correct — every lead qualifies —
which is why it never bit.

**The latent bug.** A column `_d{N}` at valid time τ was issued `24N` hours before τ. Row *t*
takes the value at `t + h`, issued at `(t + h) − 24N`, which is only knowable at *t* when
`24·N ≥ h`. So the moment anyone set `horizon_hours: 168`, the `d1` column would have carried
a forecast **issued six days after the moment it was supposed to be known** — and produced
excellent, invalid numbers.

**What caught it.** Writing the tests before the feature. Three of five failed red against
the existing code, including the 168 h case returning `d1` and `d3` alongside `d7`.

**Fix.** `usable_nwp_leads` drops every too-short lead, logs what went, and **raises** when
none survives — refusing to run rather than quietly returning a leaking frame.

**The lesson.** This is the same shape as the v3 alignment bug and the Historical-Forecast-API
trap, and it was **dormant and invisible** because the project had only ever run one horizon.
A guard is worth writing for the configuration you have not tried yet.

---

## Defect 6 — the classical models silently degraded when the guard fired

**Symptom.** First 48 h run: `mos_multi` scored **1.6207** against persistence's 1.6162 —
worse than "tomorrow is like today" — with a large spread (sd 0.4097). `mos_single` was
missing from the results table entirely.

**Cause.** All five classical models selected their predictors by the literal suffix `_d1`.
The new lead guard had just removed every d1 column at a 48 h horizon, so they fell back to
whatever was left: `mos_multi` fitted on four calendar terms, `kalman_mos` on an intercept
alone, and `mos_single` raised and was skipped.

**What caught it.** The **sanity floor** again, one defect after it caught the SARIMAX
explosion. A model scoring worse than persistence is a bug until proven otherwise.

**Fix.** `FoldContext.primary_lead` now derives the **shortest surviving lead** rather than
assuming `d1` — which is exactly the freshest legitimate forecast at any horizon. A
regression test pins it.

**The lesson.** Fixing one thing exposed another: the guard was correct, and it broke five
models that had an unstated dependency on the configuration it changed. **Tree models
adapted automatically** (they consume whatever columns exist); only the models that select
columns *by name* were affected. Name-based feature selection is an implicit contract, and
implicit contracts break silently.

---

## Defect 7 — a cache that would have turned a 7-day model into a 3-day one

**Symptom.** None available. This one produced no error, no warning, and no visibly wrong
number — it would have produced *suspiciously good* day-7 scores.

**Cause.** Open-Meteo responses are cached at
`data/raw/openmeteo_{site}_{model}_{year}.json` — a path that keys on site, model and year
and **says nothing about which lead days or variables the file holds**. Widening
`lead_days` from `[1,2,3]` to `[1..7]` for the weekly work would have hit those existing
files, found them present, and returned a three-lead frame. Every "day 7" model would in
fact have been running on a day-1/2/3 forecast — i.e. reading 4–6 days into the future — and
scoring beautifully for it.

**What caught it.** Reading the cache path before trusting the fetch. A cache hit is not a
match: the filename encodes less than the request does.

**Fix.** `_missing_open_meteo_keys` compares the requested `{var}_previous_day{N}` series
against what the payload actually contains, and re-downloads on any mismatch. Confirmed
firing on every stale file:

```
Open-Meteo cache openmeteo_anloga_ecmwf_ifs025_2024.json is missing 24 requested
series (e.g. wind_speed_100m_previous_day4); re-downloading.
```

**The lesson.** **A cache key must encode everything that changes the answer.** This one
encoded three of five dimensions. The general form — *silent staleness when a request widens*
— is the same failure as defect 5: a configuration nobody had tried yet, sitting behind code
that looked correct for the configuration everybody had.

---

## Defect 8 — a day-7 model labelled as a three-member ensemble

**Symptom.** The first per-lead run reported d7 with `members: 3` and
`models: ecmwf+gfs+icon`, while the same row showed **17 features against 23** for every
other lead. The two numbers could not both be right.

**Cause.** `models_present` named the ensemble from the *requested* columns. ICON's d7
columns exist in the frame and are entirely null, so they were counted — then dropped moments
later by the all-NaN filter. The scores were correct; the label was not.

**What caught it.** Cross-reading two fields that had to agree. Nothing failed, and the
mislabelled row would have travelled into a results table where a two-member day-7 number sat
unlabelled beside three-member ones — **exactly the failure that verification gate 6 in
[08](08_weekly_forecast_design.md) was written to prevent**, occurring in the same session
the gate was written.

**Fix.** Membership is now reported from the columns that *survive*, so an all-null member is
never counted.

**The lesson.** Metadata drifts from data unless it is derived from the same object. Naming
the ensemble from the request rather than the result is the same class of error as asserting
the grid shape you asked for rather than the one that arrived ([register B4](../00_mistake_register.md)).

---

## What these eight have in common

**None of them crashed.** Several produced numbers that *agreed with a reasonable prior* —
ARIMA is weak at day-ahead, spatial gradients are marginal, deep models are seed-sensitive —
which is exactly why a protocol with explicit, falsifiable checks is worth more than careful
reading.

Two patterns are worth naming because they recur:

- **The dangerous bug is the one that confirms what you expected.** Defects 1 and 5 both
  produced output consistent with a held belief. Defect 5 was invisible for the entire life
  of the project because only one horizon had ever been run.
- **Fixing one thing exposes the next.** Defect 6 was *created* by the fix for defect 5 —
  five models had an unstated, name-based dependency on the configuration that changed. The
  sanity floor caught it within one run.

**Cost of the guards:** a sanity-floor rule, three assertions, and about thirty test cases.

**Cost of the seven defects had they gone unreported:** one invalid published number; one
invalid significance analysis; a 2.9%-inflated baseline under every historical claim; a
handicapped classical arm that would have overstated the advantage of machine learning; a
latent leak that would have made any multi-day forecast look superb and mean nothing; a 48 h
comparison in which five of eleven models were silently fitting calendar terms; and a weekly
model that was secretly a three-day model.

**Three of the seven (5, 6, 7) surfaced within a single day of starting the weekly work** —
because that work was the first to use a configuration the code had never run. Novel
configurations are where dormant bugs live.
