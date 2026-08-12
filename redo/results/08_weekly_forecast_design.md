# 08 — Weekly forecast: design

> ✅ **BUILT AND MEASURED — results in [09](09_weekly_forecast_results.md).**
> A 7-day forecast beats climatology by **20.7%**, and weekly energy is predicted to within
> **~4% of rated capacity**. Two things in this design turned out wrong and are corrected in
> 09: the 168 h skill **projection of ~0.25 measured at 0.207** (decay is slightly faster
> than geometric), and the **ICON prior was refuted** — dropping a third ensemble member
> costs nothing, so 168 h is the better choice, not 144 h. The quantile arm **fails
> calibration** (coverage 0.65–0.74 vs 0.80 nominal).

> 🔮 **This document is the design, written before the build.** Documents 01–07 and 09 are
> measurements; this one is forward-looking, and its projections must not be read as results.
> Every number below either **cites [07](07_horizon_sweep.md)** or is explicitly labelled a
> **projection**. It is left as written — including the projection and the prior that the
> measurement went on to refute — because a design quietly edited after the answer is known
> teaches nothing.

> **What:** how to build an hourly wind-speed and power forecast covering the coming week,
> with calibrated uncertainty and a weekly-energy aggregate.
> **Why now:** [07](07_horizon_sweep.md) established that forecast skill decays but
> *decelerates*, which is what makes a week-ahead model worth building rather than a
> foregone failure. Had skill converged on climatology by 72 h, the honest answer would have
> been to stop — and that negative result would have been equally publishable.

## 1. The evidence that justifies building it

From [07](07_horizon_sweep.md), measured at 24/48/72 h on 4 rolling folds:

| Horizon | XGBoost MAE | Climatology | Skill vs climatology |
|---:|---:|---:|---:|
| 24 h | 0.8009 | 1.2725 | 0.371 |
| 48 h | 0.8739 | 1.2844 | 0.320 |
| 72 h | 0.9280 | 1.3050 | 0.289 |

Loss per extra day is shrinking: **−0.051, then −0.031** (ratio ~0.6).

> **Projection, not a result:** continuing that geometric decay puts skill at ~**0.25**
> against climatology at 168 h. This is an extrapolation from **three points** and assumes
> the decay stays geometric. Treat it as "worth measuring", never as a finding.

## 2. Three measured facts that shape the design

**Climatology replaces persistence as the reference baseline.** The crossover happens
between 24 h and 48 h — damped persistence beats climatology at 24 h (1.1861 vs 1.2725) and
loses to it at 48 h (1.3662 vs 1.2844). Climatology moves only +2.6% across the sweep while
persistence degrades +20.4%. **Quoting persistence skill at weekly range would flatter the
model by ~20%.**

**This is not a problem that needs a large model.** A 21-parameter linear MOS stays within
~5% of gradient boosting at every horizon tested (5.0% / 5.3% / 4.1%), and the gap does not
widen with lead time.

**The operational feature set is the only sensible one.** Reanalysis features cost nothing
to drop even at 24 h ([03](03_operational_vs_study.md)), and observation-side lags are dead
by day 3 — `hub(t−1)` says nothing useful about `hub(t+168)`.

## 3. The valid-time reframing

The rest of the pipeline is indexed by **feature time** *t* with target `hub(t + h)`, because
it needed those lag features. At weekly range they are dead, which removes the reason for
that framing and permits a simpler one:

> Index by **valid time τ**. For lead day *N*, the row is
> `(nwp_*_d{N}[τ], calendar(τ), spread(τ)) → hub(τ)`
> — exactly a lead-*N* Model Output Statistics problem, trained on every τ in the window.

Four consequences, all wanted:

| | |
|:---|:---|
| **Seven models, not 168** | one per lead day, ~15.9k rows each |
| **No horizon shift anywhere** | forecast and truth share an index, so the alignment bug that cost two iterations ([register A3](../00_mistake_register.md)) **cannot occur by construction** |
| **The lead guard becomes structural** | a lead-*N* model can only see `_d{N}` columns; no filter to get wrong |
| **The ICON gap needs no special-casing** | the day-7 model simply has two members instead of three |

Serving is trivial: to forecast the week from issue time *T*, take τ = T+1…T+168 and apply
the model for `N = ceil((τ − T)/24)`.

## 4. The ensemble-availability constraint

✅ **Confirmed across the full 22-month window** (16,104 hourly rows, 126 forecast columns,
all seven leads fetched):

| Model | d1 | d2 | d3 | d4 | d5 | d6 | **d7** | Max horizon |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ecmwf_ifs025` | 15966 | 15942 | 15912 | 15888 | 15864 | 15840 | **15816** | 168 h |
| `gfs_global` | 16104 | 16104 | 16104 | 16104 | 16104 | 16104 | **16104** | 168 h |
| **`icon_global`** | 16104 | 16104 | 16104 | 16104 | 16104 | 16104 | **0** | **144 h** |

*Non-null hours per series.* ICON's d7 is **exactly zero across all 16,104 hours** — not
sparse, absent. ECMWF thins by 24 rows per lead day, which is the expected tail effect: a d7
forecast for the final day of the window would have had to be issued a week before the
archive ends. `previous_day8` is null for all three models.

This upgrades an earlier three-day probe (2025-06-01→03) to a full-window measurement, and
satisfies verification gate 4 in §9.

So there is a real trade, and **inter-model spread is the uncertainty signal a weekly product
most needs**:

| Option | Horizon | Ensemble |
|:---|---:|---:|
| A | 168 h (7 days) | 2 members |
| B | 144 h (6 days) | 3 members |

**Resolve by measurement, not assertion.** Three scores settle it:

1. day-7 with ECMWF + GFS
2. day-6 with all three
3. day-6 with ECMWF + GFS only

Comparing 2 against 3 isolates *the cost of losing ICON*; comparing 1 against 2 mixes it with
*the cost of one more day*. Without run 3 the two effects are confounded.

*Prior: B looks better, because a third member matters more to the uncertainty band than a
seventh day does to the point forecast. That is reasoning, not evidence.*

## 5. Probabilistic output — mandatory, not optional

At day 7 a deterministic point forecast is not defensible. The signal is free and already
fetched: **disagreement between ECMWF, GFS and ICON**, which widens with lead time.

- Quantile gradient boosting (pinball objective) for **P10/P50/P90**, per lead day.
- **Trained separately from the point model.** A shared quantile objective cost TFT
  **0.380 m/s** in v4 ([register C1](../00_mistake_register.md)).
- **Calibration is the acceptance test**: ~10% of observations below P10 and above P90.
- **Report sharpness alongside** — a band spanning the whole turbine range is perfectly
  calibrated and perfectly useless.
- Bands should widen with lead day. If they do not, suspect the spread feature before
  believing it.

## 6. Weekly energy aggregate

**Derived from the hourly forecasts, not a separate model.** Push the 168 hourly wind speeds
through the existing two-stage power conversion, sum to MWh, report percent error on the
weekly total.

Hourly errors partially cancel in the sum, so this should be **materially more accurate than
any individual hour** — and it is the number a resource assessment actually wants, which
suits a site with no SCADA.

**Evaluate on ~95 non-overlapping weeks**, not a rolling hourly window. Quoting the hour
count would be fiction: adjacent weeks share weather.

## 7. Protocol changes forced by the horizon

Otherwise as [01](01_protocol.md) — rolling folds, 5 seeds, mean ± sd, Diebold–Mariano.

| | 24 h work | Weekly |
|:---|:---|:---|
| Reference baseline | damped persistence | **climatology** |
| DM HAC lags | 23 | **167** — week-ahead errors are far more serially correlated |
| Aggregate sample size | 3,183 test hours | **~95 independent weeks** |

## 8. Status

| Component | State |
|:---|:---|
| `config_weekly.yaml` — leads 1–7 | ✅ written |
| `src/weekly.py` — valid-time frame, per-lead evaluation | ✅ written |
| `tests/test_weekly.py` | ✅ **9 passing** |
| Cache-invalidation guard for widened `lead_days` | ✅ fixed, **confirmed firing** ([06 F7](06_what_the_protocol_caught.md)) |
| Re-fetch of leads d4–d7 | ✅ **complete** — 16,104 rows × 126 columns |
| Full-window lead coverage confirmed | ✅ **done** — ICON d7 empty across all 16,104 hours (§4) |
| Day-1 cross-check vs the known 24 h result | ✅ **pass** — 0.7322 vs 0.7299, same split |
| Per-lead decay curve (d1…d7) | ✅ **measured** ([09](09_weekly_forecast_results.md)) |
| 144 h vs 168 h measurement | ✅ **done — prior refuted, take 168 h** |
| Quantiles + calibration | ⚠️ **built, calibration FAILS** (0.65–0.74 vs 0.80) |
| Weekly energy aggregate | ✅ **~4% of rated at 168 h**, ~2.6× cancellation |

## 9. Verification gates

Each stated so it can fail:

1. **Sanity floor per lead day** — every model beats **climatology** at its own lead.
   Anything that doesn't is a bug until proven otherwise. This rule has already caught three
   defects ([06](06_what_the_protocol_caught.md)).
2. **Monotonic decay** with lead day. If day 5 beats day 3, suspect the lead wiring before
   believing the result.
3. **Day-1 cross-check** — the valid-time lead-1 model must land near the feature-time 24 h
   figure (XGBoost ~0.80 on `operational`). A large gap means the reframing changed something
   it should not have.
4. ~~ICON's d6 ceiling confirmed across the full window~~ — ✅ **done**, see §4.
5. **Quantile calibration** within a few points of nominal, with sharpness reported.
6. Every results table labelled with its lead day and ensemble membership — a day-7 number
   from a two-member ensemble must never sit unlabelled beside a three-member one.

## What this design does not claim

- **It is not an operational system.** Still no wind farm, no SCADA, no serving path, no
  monitoring, no retraining schedule.
- **The 168 h projection is not a result.** See the banner.
- **Deep models are excluded deliberately** — trees beat them by 5.1% at 24 h, they are
  3–10× more seed-sensitive ([04](04_deep_models.md)), and TFT costs ~4.3 h per fit. Revisit
  only if the per-lead curve shows headroom trees are not reaching.
