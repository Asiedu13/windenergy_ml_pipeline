# 11 — NWP Forecast Features (v5, in progress)

Record of the v5 iteration, started 2026-08-10 — both what has happened and **what is
planned next**. Written **mid-flight**: the ingest and plumbing are complete and tested,
the ERA5 truth series for the new window is still downloading, and **no accuracy result
exists yet**.

Reading order if you are picking this up cold: **§8** for what is running and how to
resume it, **§7** for the gates still to run and their pass criteria, **§7b** for what
each outcome commits you to, **§10** for why every decision was taken.

> 📐 **Reading the numbers.** Unless stated otherwise, a figure like `1.1410` is a **test
> MAE in m/s** — how far off the model typically is when predicting hub-height wind speed
> 24 hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399, guessing the average 1.466. Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).
> ⚠️ Every figure in this document is from the **2020–2023** period. The v5 study window
> is 2024-03 → 2025-12 and its results, once they exist, will not be comparable to these.

> **Status at last edit: 2026-08-11 02:40.** ERA5 for the study window
> (**2024-03 → 2025-12**, shortened from 2026-06 — see §10) is downloading; §8 has a live
> progress check, since the count in any document is stale on arrival.
> Nothing in this iteration has produced a model score yet. Do not quote v5 numbers —
> there are none.

## 1. Why this iteration exists

> **What:** four experiments that overturned the project's long-standing explanation for
> why the model stopped improving.
> **The old belief:** "not enough data" — which is why a ten-year download was running.
> **What the measurements showed:** performance plateaus after about 1.6 years, and the
> model is barely memorising, so more of the same data has nothing left to fix. The
> shortfall is not the *quantity* of data but its *kind* — the model had only ever seen
> weather that had already happened, never a forecast of weather to come.
> **Why it changed the plan:** the download was stopped mid-run and the effort redirected.

Docs 01–10 attributed the residual ~2.5 pp nMAE gap to the Energies 2021 benchmark to
"data volume", and the 10-year ERA5 backfill was launched on that basis. That claim had
never been measured. On 2026-08-10 it was, and it does not hold.

### Learning curve — flat after ~1.6 years

XGBoost, v3 Optuna hyperparameters, fixed test set, trained on the most-recent fraction
of the training slice (2020–2023 parquet, 35,028 rows, 53 features):

| Train rows | Years | Test MAE | Δ | Train/test gap |
|---:|---:|---:|---:|---:|
| 3,502 | 0.4 | 1.5017 | — | 2.36 |
| 7,005 | 0.8 | 1.1933 | −0.3085 | 1.52 |
| 14,011 | 1.6 | 1.1534 | −0.0399 | 1.25 |
| 21,016 | 2.4 | 1.1479 | −0.0055 | 1.19 |
| 28,022 | 3.2 | **1.1410** | −0.0069 | **1.16** |

The final doubling bought **0.012 m/s (1.1%)**. Extrapolating, 3.2 → 12.8 years is worth
~0.02 m/s ≈ 0.2 pp nMAE. The train/test ratio of **1.16** is the decisive figure: more
data cures variance, and there is almost none left.

### Features saturated

| Feature set | n | Test MAE |
|:---|---:|---:|
| lag/rolling only | 9 | 1.2564 |
| + calendar | 13 | 1.1760 |
| + surface weather | 25 | 1.1615 |
| + upper-air | 33 | 1.1444 |
| + spatial derivatives | 53 | **1.1410** |

### Against a one-parameter baseline

| Predictor | Test MAE | R² |
|:---|---:|---:|
| Naive persistence `hub(t)` | 1.3990 | 0.043 |
| **OLS on `hub(t)` alone (1 param)** | **1.2388** | **0.267** |
| OLS, lag1 + calendar (5 params) | 1.2216 | 0.284 |
| OLS, all 53 features | 1.1591 | 0.348 |
| XGBoost, Optuna, 53 features | 1.1410 | 0.365 |
| LSTM (v4 best) | 1.114 | 0.390 |

24 h autocorrelation of hub wind is **0.571**, capping R² at **0.326** for any predictor
built on the state at time *t*. The full apparatus buys ~0.06 R² over one parameter.

### The diagnosis

`mode="nowcast"` scores MAE **0.015**, R² **0.9998**. That is near-tautological — hub
speed is a deterministic function of `wind_speed_100m`, a feature — but it localises the
error exactly: **converting weather to hub wind is free; 100% of the error is the
24-hour atmospheric gap.** Every exogenous feature is the analysed state at time *t*;
there is no forecast in the input. That is the binding constraint.

**Consequence:** the 10-year backfill was stopped (§8) and the effort redirected here.

## 2. Source selection

> **What:** choosing where to get archived weather forecasts from — two realistic options,
> compared before committing.
> **Why the choice was lopsided:** both are free and both work. One is a small web request
> returning a few megabytes of text; the other requires downloading whole-planet weather
> files to read a single point, running to hundreds of gigabytes for the same period.
> **The trade-off accepted:** the easy option's archive only reaches back to March 2024,
> which forced the study period to move — the main cost of this iteration.

| | **Open-Meteo Previous Runs (chosen)** | GFS GRIB from AWS S3 (rejected) |
|:---|:---|:---|
| Coverage | 2024-03 → present (verified) | 2021-02 → present |
| Transport | Point JSON over HTTPS | GRIB2 + `.idx` byte-ranges on S3 |
| Volume | **2.6 MB** for 2.3 years | 30–400 GB (every message is a *global* 0.25° field, ~1 MB) |
| Effort | Hours | ~a week |
| Models | `ecmwf_ifs025`, `gfs_global`, `icon_global` | GFS only |
| Extra variables | — | 10/20/30/40/50/80/100 m wind profile, `HPBL`, `FRICV` |

Verified by direct S3 probe on 2026-08-10: `noaa-gfs-bdp-pds` has no `gfs.2015`/`2018`/
`2020` prefixes; usable `atmos/gfs.t00z.pgrb2.0p25.f024` starts 2021-02. `gfs.20220601`
f024 `.idx` confirms UGRD/VGRD at 100 m, GUST, HPBL, FRICV, and 850/500 mb winds — so the
GRIB route remains a genuine fallback if the wind profile turns out to matter.
`noaa-gefs-retrospective` (GEFSv12 reforecast, 2000–2020, 5 members, 00Z only) was also
mapped and is a third option, best paired with a completed 2010–2020 ERA5 record.

**Cost of the choice: the study window moves to 2024-03 → 2025-12** (22 months; initially
2026-06, shortened 2026-08-11 — see §10). v1–v4 numbers are on 2020–2023 and are **not**
comparable. Re-baselining on the new window is a gating step (§7), not an optional extra.

## 3. ⚠️ The endpoint trap

> **What:** the chosen provider offers two APIs that look interchangeable. One gives real
> day-ahead forecasts; the other gives something that only resembles them.
> **Why the wrong one would be catastrophic:** its archive is stitched from the opening
> hours of each forecast run — the part closest to observed reality. Feeding that in as
> "yesterday's prediction of today" effectively hands the model the answer. Scores would
> look superb and mean nothing.
> **Why it is called a trap:** the misleading endpoint is the more prominent and
> obvious-sounding of the two.

| API | Returns | Usable as a t+24 feature? |
|:---|:---|:---|
| **Previous Runs** (`previous-runs-api`) | `{var}_previous_day{N}` — forecast for the valid timestamp, issued N×24 h earlier | ✅ yes |
| Historical Forecast (`archive-api`) | series stitched from the **first hours of each run** | ❌ no — analysis-like, leakage |

The Historical Forecast API is the more obvious endpoint and would produce excellent,
invalid results. Same failure mode as the v3 alignment bug (doc 10): plausible numbers,
silently wrong, agreeing with the hypothesis.

## 4. What shipped

> **What:** the code written for this iteration, file by file.
> **Design principle throughout:** the forecast features are switched off by default. The
> original no-forecast path is untouched and still runs exactly as before, so it remains a
> valid control to measure the new version against.

| File | Change |
|:---|:---|
| `src/data_ingest.py` | `_fetch_open_meteo`, `_download_open_meteo`, `_parse_open_meteo`, `load_nwp_forecasts`, `nwp_column`; cached per (model, year) as `data/raw/openmeteo_{site}_{model}_{year}.json`, modelled on the existing NASA POWER trio |
| `src/features.py` | `add_nwp_features`; `build_feature_frame` gains `nwp_df=` and applies the same offset as the target |
| `src/utils.py` | `NWPConfig` (`enabled`, `models`, `lead_days`, `variables`), wired into `load_config` |
| `src/pipeline.py` | calls `load_nwp_forecasts`, passes `nwp_df` to `build_feature_frame` **and** to `evaluate_variants` |
| `src/evaluate.py` | `evaluate_variants(..., nwp_df=None)` — the `direct` variant rebuilds its own feature frame and would otherwise be fitted on a different feature set |
| `config_nwp.yaml` | new; 2024-03 → 2025-12, `nwp.enabled: true`, model block otherwise identical to `config.yaml` |
| `tests/` | 9 new tests, **47 green** (was 38) |

Columns are `nwp_{model}_{variable}_d{lead}`.

## 5. The alignment contract

> **What:** the rule for matching each forecast to the correct row of the training table.
> **Why it is the riskiest part of the whole iteration:** the two tables are stamped on
> different clocks. Forecasts are labelled with the hour they *describe*; training rows are
> labelled with the hour the prediction is *made from*. Those differ by 24 hours.
> **The two ways to get it wrong:** shift one way and the model sees the future — free
> marks, worthless model. Shift the other and it is graded against the wrong hour, which
> is exactly what invalidated two earlier iterations.
> **Why it should be safe this time:** four automated tests pin it, including one where a
> deliberately perfect forecast must reproduce the answer exactly.

This is the part that repeats v3's mistake if you get it wrong.

- `nwp_df` is indexed by **valid time τ**. `nwp_..._d1[τ]` is the forecast *for* τ, issued
  ~24 h *before* τ.
- The feature frame is indexed by **feature time t**, target `hub(t + horizon)`.
- Row *t* must receive the forecast **valid at t + horizon** — issued around *t*, so
  legitimately known then. No lookahead.

`add_nwp_features` reindexes on `df.index + horizon` rather than using `.shift()`, so the
intent is explicit and gaps in either index produce NaN instead of a silently shifted
value.

Guarded by four tests in `tests/test_features.py`, the sharpest being
`test_perfect_forecast_column_reproduces_the_target_exactly`: a *perfect* forecast column
must equal `y` exactly after alignment, on a **37-hour** signal period deliberately not
commensurate with the horizon so no shifted frame can pass by coincidence.

## 6. Data fetched

> **What:** what actually arrived — 2.3 years of hourly forecasts from three models, with
> their units verified against the data rather than assumed.
> **Why the unit check is not pedantry:** this provider reports air pressure in hPa while
> the existing weather table uses Pa. The two differ by a factor of 100 and now sit in the
> same feature frame. Mixing them would be silent and damaging.

`data/processed/nwp_anloga_2024-03-01_2026-06-30.parquet` — **20,448 rows × 54 columns**,
hourly UTC, 2024-03-01 00:00 → 2026-06-30 23:00, **no gaps** (852 days × 24 = 20,448).

> ℹ️ **This is the full fetch, which is wider than the study window.** The forecast data
> was pulled before the window was shortened to 2025-12 (§10), and it stays on disk
> untouched — `load_nwp_forecasts` slices to whatever `config_nwp.yaml` specifies, so a
> run produces a 22-month parquet from this same cached JSON with no re-fetch. If the
> window is ever widened again, the forecast side needs no work.
9 HTTP requests, ~2.6 MB of cached JSON.

Units (verified from the data, not assumed): wind **m/s** (`wind_speed_unit=ms`),
direction **degrees**, temperature **°C**, surface pressure **hPa**.
⚠️ ERA5's `pressure_pa` is **Pa** — the two are 100× apart and both are in the feature
frame.

`nwp_ecmwf_ifs025_wind_gusts_10m_d{1,2,3}` are **100% null** (ECMWF IFS does not serve
gusts here); `build_feature_frame` drops all-NaN columns automatically. Everything else is
0–1% null.

### Inter-model differences (investigated, not bugs)

> **What:** the three forecast models disagree substantially — one is 47% windier than
> another on average, and they place the daily wind minimum hours apart.
> **Why that is fine, even useful:** every forecast model has its own systematic biases,
> and learning to cancel them is exactly what the machine-learning step does. Three
> disagreeing models also hint at how confident to be on any given day.
> **Why this section is long:** one difference looked like a coding error. Rather than
> assume either way, it was checked five different ways before being accepted as genuine
> physics. The reasoning is recorded so nobody later "corrects" a real signal.

| Model | mean 100 m wind | diurnal peak (UTC) | trough |
|:---|---:|---:|---:|
| `ecmwf_ifs025` | 4.50 m/s | 18.0 | 11.5 |
| `icon_global` | 5.12 m/s | 17.7 | 8.3 |
| `gfs_global` | 6.62 m/s | 17.0 | 9.0 |

GFS is ~47% windier than ECMWF in the mean. Cross-correlating ECMWF against the other two
peaks at lag **+1 h** rather than 0. Investigated rather than assumed benign:

- ECMWF vs its **own** longer leads peaks at lag 0 → not a parse/index error.
- GFS vs ICON, and both temperature and pressure pairs, peak at lag 0.
- The correlation curve is **flat-topped**: r = 0.6420 at lag 0 vs 0.6553 at +1.
- Ruled out linear interpolation from a coarser native step (only 2.1% of points are
  exact midpoints, with no hour-mod-3 pattern).
- A uniform timestamp offset would displace peak and trough **equally**; ECMWF's peak
  moves 1 h and its trough 2.5 h.

Conclusion: genuine model disagreement about the coastal sea-breeze phase, which is
precisely the systematic error the ML bias-correction absorbs. **Still to be confirmed
per-model against ERA5 truth (§7 gate 1).**

## 7. Verification gates — none of these have run yet

> **What:** four checks that must pass, in order, before any result from this iteration
> can be believed. Each has an exact pass mark and a defined response to failure.
> **Why gates rather than a checklist:** having just spent two iterations trusting
> plausible wrong numbers, the point is to decide *in advance* what would count as
> evidence that something is broken. A test you design after seeing the result is not a
> test.
> **Why gate 1's tolerance is 1 hour and not zero:** §6 established that one model
> genuinely leads the others by about an hour. A zero-tolerance rule would fail on
> known-correct behaviour.

All four need the ERA5 truth series for 2024-03 → 2025-12, which is still downloading
(§8). Run in order and stop at the first failure.

| # | Gate | Pass criterion | Status |
|---:|:---|:---|:---|
| 1 | Lag-peak vs ERA5 truth | per-model cross-correlation peak at **\|lag\| ≤ 1 h** | ⏳ pending |
| 2 | Raw skill | **bias-corrected** forecast beats 24 h persistence on the same window | ⏳ pending |
| 3 | Re-baseline | comparators produced; persistence MAE inside a 1.2–1.6 sanity band | ⏳ pending |
| 4 | Experiment | produces a number, whatever it is | ⏳ pending |

### Gate 1 — lag-peak against truth

Cross-correlate each model's `nwp_{model}_wind_speed_100m_d1[τ]` against ERA5
`wind_speed_100m[τ]` over lags −48…+48 h.

**Pass: |peak lag| ≤ 1 h.** Not "peak at exactly 0" — §6 establishes that ECMWF leads the
other two by ~1 h for physical reasons, so a zero-tolerance criterion would fire on
already-understood behaviour.

Failure modes are distinct and must not be conflated:

- **Peak at ±24 h** → the horizon shift in `add_nwp_features` is wrong. This is the v3 bug
  recurring. Fix the code; do not adjust the criterion.
- **Peak at a constant offset of 2–3 h or more, one model only** → a real per-model
  convention difference. Document it and apply an explicit per-model shift. Do **not**
  silently absorb it into the join.
- **No clear peak at all** → units, site coordinates, or the wrong variable. Stop.

This is the check that would have caught the v3 alignment bug in minutes.

### Gate 2 — raw skill, evaluated after bias correction

The forecast alone, with no ML, must beat 24 h persistence on the new window.

**Evaluate on a single-parameter linear rescale (OLS of ERA5 hub wind on the forecast),
not on raw values.** §6 measured mean 100 m wind of 4.50 (ECMWF), 5.12 (ICON) and 6.62
(GFS) m/s — spreads of that size mean a *raw* comparison can lose to persistence on bias
alone while the alignment is perfectly correct. Report both numbers; the rescaled one is
the alignment test, the raw one documents the bias.

**On failure:** alignment or unit error. Stop — do not proceed to modelling. Check the
hPa/Pa distinction (§6) and gate 1 first.

### Gate 3 — re-baseline the new window

Persistence, climatology, OLS-on-`hub(t)`, and XGBoost-**without**-NWP on
2024-03 → 2025-12. These are the only legitimate comparators; quoting the 2020–2023
numbers against a v5 result is a window confound.

**Sanity band, not a threshold:** 24 h persistence measured **1.399 m/s** on 2020–2023
(§1), so anything in roughly 1.2–1.6 is unremarkable. Outside that band the new window has
a materially different wind climatology — note it explicitly before comparing anything,
because it would also shift every other baseline.

### Gate 4 — the experiment

XGBoost with vs without the NWP block, identical split, identical hyperparameters.
Published NWP+ML comparisons suggest 2–4 pp nMAE. **That is a hypothesis, not a result.**
Report whatever comes out, including no improvement. Branches in §7b.

Then `python -m pytest -q`, and only if gate 4 justifies the compute,
`python -m src.pipeline --config config_nwp.yaml --mode forecast --compare`.

## 7a‑final. ✅ AUTHORITATIVE RESULT — full 22-month ERA5 window (2026-08-11 23:18)

> **What:** the run this whole iteration was building toward. The ERA5 download reached
> 66/66 files at ~23:00; gates and pipeline were run immediately after.
> **Why it supersedes everything below:** §7a's numbers are a 12-month subset and §7a's
> NASA POWER numbers are a different truth source. **This section is what to quote.**

`python -m src.validate_nwp --config config_nwp.yaml` — **all four gates pass**.
Feature frame **15,912 rows × 104 features**, 51 of them NWP.

| Predictor (test MAE, m/s — lower is better) | MAE | R² |
|:---|---:|---:|
| climatology (train mean) | 1.3490 | −0.0711 |
| persistence `hub(t) → hub(t+24)` | 1.3097 | −0.0149 |
| OLS on `hub(t)` — strongest simple baseline | 1.1349 | 0.2225 |
| XGBoost **without** NWP (53 feats) | 1.0513 | 0.3300 |
| **XGBoost *with* NWP (104 feats)** | **0.7295** | **0.6653** |

**Headline: −30.6% MAE; R² 0.3300 → 0.6653**, against the measured **0.326** ceiling for
any predictor using only the state at time *t*. The tuned pipeline run agrees at
**0.738 m/s / R² 0.660**.

| Variant | MAE (W) | R² | nMAE | nRMSE |
|:---|---:|---:|---:|---:|
| physics_only (no ML) | 420,250 | −0.071 | 13.78% | 18.30% |
| **two_stage** | **228,048** | **0.675** | **7.48%** | **10.08%** |
| direct | 226,728 | 0.690 | 7.43% | 9.85% |

### Two things this run changed about the story

**1. The "calmer window" caveat is retracted for the ERA5 arm.** §7a and doc 12 both warned
that the NWP window was demonstrably easier than 2020–2023 — persistence 1.09 vs 1.399 m/s.
That held for the 12-month subset. On the **full** 22 months, persistence is **1.3097 m/s**
(the pipeline's 25 h variant, 1.336), within 5% of v4's window. The problem is not
materially easier, so the benchmark comparison no longer rests on that hedge.

**2. The best raw NWP model is different here.** Gate 2 ranks **ICON first** (rescaled MAE
0.9743) ahead of ECMWF (1.0410) and GFS (1.0465) — *reversing* the NASA POWER ordering,
where ECMWF led. Two consequences:

- **Do not claim a model ranking.** The ordering is not stable across truth sources and the
  margins are small.
- It weakens the **shared-lineage concern** ([12, threat 2](12_next_steps_v2.md)) from the
  other direction: if ERA5/ECMWF kinship were flattering ECMWF, ECMWF should have won *most*
  clearly on ERA5 truth. It came second.

### Still outstanding before this is a headline claim

Single seed, single chronological split. The multi-seed repeat
([12 item 4](12_next_steps_v2.md)) is the one thing standing between "measured on one run"
and a defensible interval. The five-model bakeoff below also needs repeating on ERA5 truth.

## 7a. INTERIM RESULT — 12 of 22 months (2026-08-11, superseded by 7a‑final)

> ⚠️ **Interim, not final.** Real ERA5 truth, but only 2024-03 → 2025-02 (8,760 h) of the
> 22-month window. Config: [`config_nwp_partial.yaml`](../config_nwp_partial.yaml).
> Re-run `config_nwp.yaml` when the download completes.

**All four gates PASS.** Run with `python -m src.validate_nwp --config config_nwp_partial.yaml`.

| Gate | Result |
|:---|:---|
| 1 lag-peak | **PASS** — ecmwf 0 h, gfs −1 h, icon −1 h (allowed ≤1 h); r 0.71–0.77 |
| 2 raw skill | **PASS** — all three beat persistence after bias correction |
| 3 baselines | PASS, with a flag: persistence is **1.0928 m/s**, outside the 1.2–1.6 band from 2020–2023. This window is calmer and more predictable. |
| 4 experiment | **PASS — LARGE band** |

### Gate 4 — the headline, and why it is smaller than it first looks

| Comparison (test MAE, m/s, lower better) | MAE | R² | with-NWP gain |
|:---|---:|---:|---:|
| XGBoost **without** NWP | 1.2695 | 0.2415 | *40.0% — inflated, see below* |
| 24 h persistence | 1.0928 | 0.3851 | 30.3% |
| **OLS on `hub(t)` — strongest simple baseline** | **1.0183** | 0.4628 | **25.2%** |
| Best single rescaled forecast (icon) | 0.9734 | — | 21.7% |
| **XGBoost *with* NWP** | **0.7621** | **0.7173** | — |

**The 40% figure is inflated and should not be quoted.** The without-NWP arm has a
train/test MAE ratio of **1.50** — it is overfitting on 0.78 yr of training data, well
inside the steep part of the learning curve (§1). On the full 22 months that arm improves
and the gap narrows. **Quote ~25%, against the strongest baseline.**

### Why this is nonetheless a real result

R² reaches **0.7173**, against a measured ceiling of **0.326** for any predictor built on
the atmospheric state at time *t* (§1). Breaking decisively through that ceiling is only
possible with genuine forecast information, and it is exactly what the v5 diagnosis
predicted.

### Leakage test — passed

Forecast skill must decay as the forecast is issued further ahead. It does:

| Lead | Issued before valid time | Test MAE | R² |
|:---|---:|---:|---:|
| d1 | 24 h | **0.7684** | 0.7138 |
| d2 | 48 h | 0.8504 | 0.6507 |
| d3 | 72 h | 0.8457 | 0.6565 |

Single models are each worse than the three combined (ecmwf 0.8407, icon 0.9006, gfs
0.9519), which is the expected multi-model ensemble benefit. A leaking feature would show
flat skill across leads and no ensemble gain.

### Cross-source confirmation on NASA POWER, full 22 months

Once NASA POWER's two defects were fixed (50 m reference height, 1-hour timestamp
offset — both in §10), the whole 22-month window became available immediately, giving an
independent check on a **different truth source, a longer window, and a properly-trained
baseline**. All four gates pass there too.

| | ERA5 truth, 12 months | NASA POWER truth, 22 months |
|:---|---:|---:|
| rows | 8,568 | 15,911 |
| without NWP | 1.2695 m/s · R² 0.2415 | 0.8827 m/s · R² 0.4846 |
| **with NWP** | **0.7621 m/s · R² 0.7173** | **0.6780 m/s · R² 0.7024** |
| improvement | +40.0% | **+23.2%** |

**The two runs agree on the thing that matters.** With NWP, R² lands at **0.7173** and
**0.7024** — two different truth sources, two different window lengths, essentially the
same endpoint, and both far above the **0.326** ceiling for any predictor built on the
state at time *t*.

**They also confirm the 40% was inflated.** The ERA5 run's baseline was overfitting on
0.78 yr (train/test 1.50); with 22 months the baseline strengthens markedly (R² 0.2415 →
0.4846) and the gain settles at **23.2%**. That is very close to the ~25% predicted from
the ERA5 run when compared against its strongest baseline rather than its handicapped one.

**Headline to quote: ~23% MAE reduction, R² from ~0.48 to ~0.70.**

⚠️ Do **not** compare the two absolute MAE columns. NASA POWER's is inflated ~12% by the
50 m → 100 m extrapolation but simultaneously *deflated* by being coarser (~55 km) and
hour-averaged, which makes it inherently smoother and easier to predict. The two
confounds run in opposite directions. Only R² and relative improvement transfer.

### End-to-end power — the project's headline metric, with NWP

Full two-stage pipeline run (`python -m src.pipeline --config <cfg> --mode forecast`):

| Variant | ERA5 12 mo | NASA POWER 22 mo |
|:---|---:|---:|
| physics_only (no ML) | 11.21% nMAE / 15.03% nRMSE | 7.53% / 11.03% |
| **two_stage** | **7.66% / 10.55%** | 4.43% / 6.38% |
| direct | 7.49% / 9.94% | 5.29% / 7.86% |

For scale: **v4 was 11.25% nMAE / 15.50% nRMSE** on 2020–2023 without NWP, and the
Energies 2021 reference is ~8.8% / ~13%.

⚠️ **Do not claim the benchmark has been beaten.** 7.66% sits below 8.8%, but on a
different site, a different and demonstrably calmer window (persistence 1.09 vs 1.399 m/s),
12 months of data and a single seed. The defensible phrasing is *"comparable to or better
than the published reference, though not under like-for-like conditions"*.

Note also that **`two_stage` and `direct` swap order** between truth sources — direct wins
on ERA5 (7.49% vs 7.66%), two_stage on NASA POWER (4.43% vs 5.29%). Neither margin is
established, and the architectural argument for two-stage rests on interpretability and
modularity rather than accuracy ([01](01_problem_and_design.md)).

### Five-model bakeoff on NWP features — the v4 near-tie does **not** survive

> **What:** all five Stage 1 architectures retrained on the NWP feature set, on NASA POWER
> truth over the full 22 months. Completed 2026-08-11 18:10 (5 h 54 m wall clock).
> **Why it matters:** v4 concluded that architecture choice is not a material lever. That
> conclusion was measured on **state-only** features. Adding a strong forecast feature
> changes the information landscape, so the conclusion had to be re-tested rather than
> assumed to carry over.

⚠️ **NASA POWER truth — a cross-source check, not a v5 result.** These MAEs are not
comparable to any ERA5 figure and must not be placed in a table beside one.

| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | Beats persistence? |
|---:|:---|---:|---:|---:|:---:|
| 1 | **xgboost** | **0.678** | 0.869 | 0.699 | yes |
| 2 | tft | 0.827 | 1.024 | 0.582 | yes |
| 3 | lstm | 0.903 | 1.089 | 0.527 | yes |
| 4 | nbeats | 0.942 | 1.141 | 0.481 | yes |
| 5 | dlinear | 0.969 | 1.158 | 0.466 | yes |
| — | *persistence* | *1.110* | *1.442* | *0.171* | — |

**All five still beat persistence, but the field is no longer flat.** The spread across
architectures is now **0.291 m/s** — nearly **3× the 0.101 m/s spread seen in v4**, and
larger than the 0.141 m/s gap from the *weakest* model to persistence. In v4 the ordering
was a tie to be reported as such; here XGBoost leads the best deep model by **0.149 m/s
(18%)**, which is not a tie under any reasonable reading.

**This was the opposite of the expected direction.** [Doc 12 item 2](12_next_steps_v2.md)
anticipated that sequence models might *exploit a forecast trajectory* better than trees.
They did not — every deep model lost ground relative to the tree.

**Three artefact explanations were tested and all three fail**, which is why this is
reported as a result rather than a discrepancy:

| Suspicion | Test | Verdict |
|:---|:---|:---|
| Different train/test split | `validate_nwp.py:98` and `stage1_windspeed.py:310` both compute `int(n * (1 - test_split))` on the same frame | **No** — 15911 rows, split 12728, test **3183 points** for both paths |
| Different alignment | persistence baseline computed on each path | **No** — both give exactly **1.1100 m/s** |
| Hyperparameters flattering XGBoost | refit under the v3 Optuna `XGB_PARAMS` *and* the untuned `_make_regressor` defaults | **No** — **0.6780 either way**; the figure is insensitive to tuning |

**Mechanism: hypothesis only, untested.** A plausible reading is that
`nwp_ecmwf_ifs025_wind_speed_100m_d1` is a *pointwise* predictor — its value at feature
time *t* is close to the answer at *t+24 h* — which is precisely the structure a tree
splits on directly, whereas the darts models must carry it through an encoded
72-dimensional input window. The SHAP ranking is consistent with this (that column tops the
table at 0.3640 mean|SHAP|, roughly 5× the best non-NWP feature), **but consistency is not
evidence of mechanism.** Do not state this as the explanation without testing it.

### Carried forward to the full run

- Absolute MAE here is **not** comparable to v1–v4: different window, and this one is
  calmer (persistence 1.0928 vs 1.399 m/s on 2020–2023).
- **The bakeoff needs repeating on ERA5 truth before the reversal is claimed.** It is one
  source, one seed, one split. The direction is large enough to be unlikely to be noise,
  but "unlikely" is not "measured".
- The ERA5 arm is handicapped by the short window; the full ERA5 run is the fair test and
  should land near the NASA POWER figures for R² and relative gain.
- Pre-registered response for the LARGE band (§7b): bank it, then re-run the two-stage
  evaluation and the five-model bakeoff, since the feature set has changed materially.

## 7b. What each outcome means — decided 2026-08-10, before seeing the result

> **What:** three possible results, and what each one commits the project to doing next.
> **Why decide before measuring:** once you see a number it is remarkably easy to
> construct a reason why it justifies whatever you already wanted to do. Fixing the
> responses in advance removes that freedom. This is the same discipline as
> pre-registering a study.
> **Why it is especially warranted here:** this project has already been caught believing
> a plausible wrong number for two full iterations.

Bands are measured against **XGBoost-without-NWP on the same window** (gate 3), not
against the v1–v4 tables.

| Outcome | MAE reduction | Action |
|:---|:---|:---|
| **Large** | ≥ 20% | Bank it. Re-run the two-stage evaluation and the five-model bakeoff — the feature set has changed materially, so the v4 architecture ranking no longer transfers. |
| **Modest** | 5–20% | **Bank it, write it up honestly, then multi-horizon** using the `d2`/`d3` columns already fetched (doc 08 item 5). Explicitly **not** escalating to the GFS-GRIB route. |
| **None** | < 5% | Do not conclude "NWP doesn't help" yet. Re-check gates 1–2, then test **ECMWF alone** — GFS's 47% wind bias (§6) may be swamping the signal in a pooled feature set. |

**Why this is written down before the measurement exists:** this project's most valuable
finding is methodological — v3's alignment bug survived two review cycles because its
output was plausible and agreed with a prior everyone held (doc 10, doc 07). Choosing the
response to each outcome *in advance* is the cheapest available guard against the same
mechanism operating on the author. If the modest band is hit, the follow-on was decided
when it was still a hypothetical, not after seeing a number worth rationalising.

The GFS-GRIB fallback is reserved for a **null** result traced to input poverty — not for
a modest one. A modest gain does not justify a week of GRIB infrastructure for inputs
whose marginal value would still be unmeasured.

## 8. What is running right now, and how to resume

> **What:** the live state of the long-running download, and the commands to check on it
> or restart it.
> **Why this section exists at all:** the work is blocked on a multi-day download. Anyone
> returning to this project needs to know what is in flight before starting anything,
> because launching a second downloader corrupts the shared cache.
> **The one rule:** check for a running process before starting an ingest. It has already
> gone wrong once.

**ERA5 truth download for the new window.**

```
process : PID 42956 — python -u -m src.data_ingest_extend --config config_nwp.yaml
started : 2026-08-10 20:03
target  : 84 files (28 months × 3 request families)
logs    : outputs/logs/era5_2024_2026.log   (stderr — Python logging writes there)
          outputs/logs/era5_2024_2026.out.log
```

### ❌ Parallel downloading does not work — tested and rejected 2026-08-11

Running a second `data_ingest_extend` on a disjoint month range looked like an easy ~2×
speed-up. It was tried and **abandoned after 75 minutes**: CDS appears to grant **one
running job per user**, so the second client's request sat in `accepted` and never reached
`running` while the first continued normally.

The timings make the bottleneck unambiguous — this is one request from the main process:

```
09:39  accepted     <- queued
11:04  running      <- 85 minutes of waiting
11:08  successful   <- 4 minutes of actual work
```

**~4 minutes of work, ~85 minutes of queueing.** Combined throughput over the trial was
1 file in 74 min, versus 42.7 min/file for the single process beforehand — no better, and
possibly worse.

The corollary matters: **downloading manually through the CDS web form would not help
either**, for exactly the same reason. It joins the same per-user queue. There is no way
to parallelise around this; the only levers are a shorter window (§10) or patience.

**⚠️ Much slower than the 2010–2019 backfill, and getting slower.** The earlier backfill
sustained ~6 min per request. This one measured **23.3 min per request** over its first
11 files, and the trend is worsening — the five most recent gaps were 14, 33, 47, 53 and
49 minutes. At ~50 min/request the remaining 73 files take **2–3 days**, not the ~5 hours
first estimated from the backfill's rate.

Likely cause: this window is recent (2024–2026) and partly served from ERA5T, the
preliminary near-real-time stream, which queues differently from the fully consolidated
archive the 2010–2019 backfill was reading. Not confirmed.

**Consequence:** gates 1–4 (§7) are blocked for days rather than hours. If that is too
long, the window can be shortened — §1's learning curve flattens after ~1.6 years, so
trimming `config_nwp.yaml` to end 2025-12 (22 months) would still sit above the plateau
while cutting roughly a quarter of the remaining downloads. The Open-Meteo forecast data
is already fully fetched and would not need re-pulling.

**Do not trust a file count written in a document — the cache grows while you read it.**
(An earlier version of `00_data_card.md` quoted a snapshot of an actively-growing cache as
a static fact and was wrong within minutes.) Check live:

```powershell
$raw = "data\raw"
$have = @(Get-ChildItem $raw -Filter "*.nc" | Where-Object { $_.BaseName -match '_(2024|2025|2026)-\d{2}$' })
"{0} / 84 files" -f $have.Count
Get-Content outputs\logs\era5_2024_2026.log -Tail 3
```

It is **resumable and idempotent** — every month already on disk is skipped. If it dies,
just re-run it:

```powershell
python -u -m src.data_ingest_extend --config config_nwp.yaml
```

**⚠️ Never run two ingest processes at once.** `target.exists()` is the only re-download
guard, so concurrent runs pick the same missing month, issue duplicate CDS requests, and
can write the same path simultaneously. This happened on 2026-08-10 and was caught before
damage. Always check first:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
```

**The 10-year backfill was stopped** (PID 33992, ran 2026-08-08 23:11 → 2026-08-10 19:5x,
44.8 h). It halted on a complete month: **2010-01 → 2016-07 is intact across all three
families**, verified no file under 50 KB. `2016-08 → 2019-12` (41 months, 123 requests)
remains missing, so `config_10yr.yaml` still aborts in `src.pipeline`. Resume any time
with `python -m src.data_ingest_extend --config config_10yr.yaml`.

## 9. What may and may not be claimed today

> **What:** an explicit boundary between what this iteration has established and what it
> has not.
> **Why it is stated so bluntly:** the plumbing works, the data is downloaded, and the
> tests pass — which makes it tempting to describe the iteration as a success. It is not
> one yet. **No accuracy result exists.** Whether adding forecasts helps is still an
> open question, and the honest answer today is "we don't know".

**May:**
- The residual gap is information-limited, not sample-limited — with the learning curve,
  the 1.16 train/test ratio, the ablation and the 0.326 autocorrelation ceiling as
  evidence.
- The 10-year backfill was stopped on measured grounds, not abandoned.
- A three-model archived NWP feed is ingested, aligned under test, and wired end-to-end.

**May not:**
- Any statement that NWP features improve accuracy. **Not yet measured.**
- Any v5 metric. None exists.
- Any comparison between a future v5 number and the v1–v4 tables without re-running
  baselines on 2024-03 → 2025-12 first.

## 10. Decision log

> **What:** every decision taken in this iteration, dated, with the evidence behind it and
> whether it can be undone.
> **Why keep one:** months later "why is the download stopped at 2016-07?" is impossible
> to answer from the code alone. It also separates decisions forced by circumstance from
> ones that were genuinely chosen — useful when revisiting them.
> **The two near-misses at the end** are included deliberately: both would have been
> reasonable-looking mistakes, and recording avoided errors is as useful as recording
> made ones.

Every decision taken in this iteration, with the evidence behind it. Dates are when the
call was made, not when it was written up.

| Date | Decision | Evidence / rationale | Reversible? |
|:---|:---|:---|:---|
| 2026-08-10 | **Stop the 10-year ERA5 backfill at 2016-07** | Learning curve flat after ~1.6 yr; train/test ratio 1.16 (§1). More history was worth ~1% MAE while occupying the CDS queue. | Yes — resume with `config_10yr.yaml`, cache intact |
| 2026-08-10 | **Redirect to NWP forecast features** | 24 h autocorrelation 0.571 caps R² at 0.326; nowcast MAE 0.015 localises 100% of the error to the atmospheric gap (§1). | — |
| 2026-08-10 | **Open-Meteo Previous Runs, not GFS GRIB** | 2.6 MB vs 30–400 GB for the same window; adds ECMWF IFS and ICON. GFS retained as a documented fallback (§2). | Yes — GFS route is scoped in §2 |
| 2026-08-10 | **Window moves to 2024-03 → 2026-06** | Open-Meteo archive is all-null before 2024-03 (verified). Accepted because the learning curve says 2.3 yr (28 months) is past the 1.6 yr plateau. | No — forced by the archive |
| 2026-08-10 | **Fetch `previous_day1..3`, use only `d1`** | Free on the same request; enables multi-horizon (doc 08 item 5) without a re-fetch. | Yes |
| 2026-08-10 | **ERA5 stays the truth series** | Open-Meteo supplies forecasts only. Keeps the target definition identical to v1–v4 so only the *features* change. | Yes |
| 2026-08-10 | **Modest-gain branch chosen before measuring** | Pre-registration guard — see §7b. | Deliberately not |
| 2026-08-11 | **Study window shortened to 2024-03 → 2025-12** (22 months, was 28) | ERA5 for this period downloads at ~27 min/request vs ~6 min for the 2010–2019 backfill — 32 h of queueing. 22 months leaves ~1.47 yr of training after the 80/20 split, just under the ~1.6 yr plateau in §1, costing ~0.004 m/s to save ~8 h. Cutting further to 2025-06 would have dropped training to ~1.07 yr, on the steep part of the curve (~0.04 m/s) — rejected. | Yes — raise `data.end` and re-run; cached months stay valid |
| 2026-08-11 | **NASA POWER rejected as a truth source; used only to surface bugs** | Its feature deficit is small (0.0208 m/s measured on 2020–2023), but it tops out at **50 m** (no WS80M/WS100M) and carries a **1-hour timestamp offset** vs ERA5 — every model's cross-correlation lag is exactly 1 h more negative against NASA POWER, and the two truth sources differ by −1 h directly. MERRA-2 hourly values are hour-averaged, not instantaneous. Superseded by running on the 12 ERA5 months already cached. | n/a |
| 2026-08-11 | **Reference height made source-dependent** (`SOURCE_WIND_HEIGHT_M`) | `_parse_nasa_power` writes WS50M into `wind_speed_100m`, and `extrapolate_hub_wind` defaulted to a 100 m reference — so NASA POWER's 50 m wind was treated as hub height and the log law no-opped, understating hub wind by **10.7%**. **v1's published results carry this error.** ERA5 unaffected (100→100 is a no-op; baseline re-verified at exactly 1.1410 m/s). | Yes |
| 2026-08-11 | **Parallel downloading tried and rejected** | A second client on a disjoint month range never got a `running` slot in 75 min — CDS grants ~one running job per user. ~85 min of the ~89 min per file is *queueing*, not work. Manual web-form downloads hit the same queue and would not help either. See §8. | n/a — reverted, config and logs deleted |
| 2026-08-11 | **CDS retry budget widened to 5 attempts × 300 s** (was 3 × 60 s) | An SSL/DNS outage lasting ~6 min on 2026-08-11 exhausted the old ~3 min budget and silently dropped **all three families of 2024-07**. A missing core month makes `_fetch_era5` raise at the *end* of a multi-day download. Repaired on restart. | Yes |

**Two calls that were nearly made and should not be:**

- *Using Open-Meteo's Historical Forecast API.* It is the more discoverable endpoint and
  would have produced excellent, invalid numbers (§3).
- *Treating the ECMWF +1 h cross-correlation peak as a join bug.* Five separate checks
  (§6) established it as genuine model disagreement. Had it been "fixed" with a shift, a
  real physical signal would have been destroyed and the gate-1 criterion corrupted.

## 11. Repository hygiene note

> **What:** a warning that almost none of this work is committed to version control.
> **Why it is urgent:** everything from v2 onward — all the documentation, the deep-model
> code, the configs — exists only as files on one machine. There is no backup and no
> history to recover from a mistake, and the next step is a multi-day experiment.

As of 2026-08-10 the entire `docs/` tree, `RESULTS.md`, `config_10yr.yaml`,
`config_nwp.yaml`, `src/stage1_deep.py` and `src/data_ingest_extend.py` are **untracked**,
and `src/{data_ingest,features,pipeline,evaluate,utils,report,stage1_windspeed}.py` have
uncommitted modifications. `git log` shows two commits, the most recent being the v1
day-ahead work. **Everything from v2 onward exists only in the working tree.** Worth
committing before the next long-running experiment.
