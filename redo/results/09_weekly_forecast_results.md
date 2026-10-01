# 09 — Weekly forecast: measured results

> **What:** the week-ahead model designed in [08](08_weekly_forecast_design.md), built and
> trained. Seven models, one per forecast lead day, covering **24 h to 168 h**.
> **Protocol:** valid-time framing, single chronological 80/20 split, seed 42, ~12.7k training
> rows and ~3.2k test rows per lead. **Not** the 4-fold × 5-seed protocol of
> [01](01_protocol.md) — see *Limitations*.
> **Source:** `outputs/reports/weekly_leads.csv`, produced by
> `python -m src.weekly --config config_weekly.yaml`.

## Point forecast — hub wind speed, MAE in m/s

| Lead | Horizon | Members | Climatology | Raw ens. mean | `mos_single` | `mos_multi` | **`xgboost`** | Skill vs clim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d1 | 24 h | 3 | 1.2065 | 1.0645 | 0.8890 | 0.7733 | **0.7322** | **0.393** |
| d2 | 48 h | 3 | 1.2075 | 1.0579 | 0.9236 | 0.8606 | **0.8251** | 0.317 |
| d3 | 72 h | 3 | 1.2087 | 1.0461 | 0.9251 | 0.9098 | **0.8729** | 0.278 |
| d4 | 96 h | 3 | 1.2085 | 1.0427 | 0.9334 | 0.9355 | **0.9166** | 0.242 |
| d5 | 120 h | 3 | 1.2088 | 1.0583 | 0.9321 | 0.9289 | **0.9147** | 0.243 |
| d6 | 144 h | 3 | 1.2085 | 1.0708 | 0.9844 | 0.9609 | **0.9354** | 0.226 |
| **d7** | **168 h** | **2** | 1.2075 | 1.0418 | 1.0578 | 0.9802 | **0.9580** | **0.207** |
| d6 | 144 h | **2** *(no ICON)* | 1.2085 | 1.0453 | 0.9907 | 0.9846 | **0.9343** | 0.227 |

**A 7-day forecast beats climatology by 20.7%.** The model works at weekly range.

> ⚠️ **Two later findings qualify this table — neither invalidates it, but both change what it
> may be compared against.** ([12](12_monthly_outlook_power.md))
>
> 1. **The climatology column is a *training-window* climatology** — (month, hour) means from
>    the ≤22-month window, via `weekly.climatology_predict`. Against the admissible
>    leave-one-year-out archive climatology (~14 years, which is operationally available since
>    ERA5 latency does not affect a climatological mean) the window baseline is **14.8% worse
>    hourly and 52.5% worse on weekly aggregates**. **The skill column is therefore measured
>    against a weak reference and the like-for-like values would be lower.**
> 2. **This table is hub wind speed in m/s; the deliverable is energy.** Re-scored in capacity
>    factor after the power curve, `mos_multi` beats `xgboost` at **5 of 7 leads** — reversing
>    the 7-of-7 ordering here. The power curve is nonlinear, so the model minimising
>    wind-speed error is not the one minimising energy error.
>
> Both are flagged rather than silently restated: the numbers below are correct for what they
> measure.

## Finding 1 — the projection was optimistic, and by how much

[08 §1](08_weekly_forecast_design.md) projected skill ~**0.25** at 168 h by extrapolating the
24/48/72 h decay geometrically. The measurement is **0.207**.

| | 96 h | 120 h | 144 h | 168 h |
|:---|---:|---:|---:|---:|
| Projected skill | ~0.27 | ~0.26 | ~0.25 | **~0.25** |
| **Measured** | **0.242** | **0.243** | **0.226** | **0.207** |

The projection was optimistic at every point, and the error grows with distance — decay is
**slightly faster than geometric**, not slower. The banner on 08 called this "an
extrapolation from three points… treat it as *worth measuring*, never as a finding", and that
caution was warranted: the direction was right, the magnitude drifted ~17% by day 7.

## Finding 2 — the ICON trade costs nothing, refuting the stated prior

[08 §4](08_weekly_forecast_design.md) recorded a prior: *"B looks better, because a third
member matters more to the uncertainty band than a seventh day does to the point forecast."*

**The measurement refutes it.** Scoring d6 with and without ICON isolates the cost of losing
a member from the cost of an extra day:

| Arm | Members | XGBoost MAE | P10–P90 coverage | Interval width |
|:---|---:|---:|---:|---:|
| d6, all three | 3 | 0.9354 | 0.705 | 2.411 |
| **d6, no ICON** | **2** | **0.9343** | **0.719** | 2.528 |
| d7, no ICON | 2 | 0.9580 | 0.743 | 2.699 |

Dropping ICON at d6 moves the point forecast by **−0.0011 m/s — in the *wrong* direction for
the prior**, i.e. marginally better — and coverage improves slightly. There is **no
measurable cost to running a two-member ensemble** at this site.

**So the 144 h vs 168 h choice is settled the other way: take 168 h.** The real price of a
seventh day is the 2.4% point-forecast degradation (0.9354 → 0.9580), not the ensemble size.
Recording the failed prior is the point of having written it down.

## Finding 3 — the classical/ML gap *narrows* with lead time

| Lead | `mos_multi` | `xgboost` | Gap |
|---:|---:|---:|---:|
| d1 | 0.7733 | 0.7322 | **5.6%** |
| d2 | 0.8606 | 0.8251 | 4.3% |
| d3 | 0.9098 | 0.8729 | 4.2% |
| d4 | 0.9355 | 0.9166 | 2.1% |
| d5 | 0.9289 | 0.9147 | 1.6% |
| d6 | 0.9609 | 0.9354 | 2.7% |
| d7 | 0.9802 | 0.9580 | **2.3%** |

At 24 h a ~21-parameter linear MOS trails gradient boosting by 5.6%; by day 7 it trails by
**2.3%**. This *extends* the finding in [02](02_classical_vs_ml.md): the further ahead you
forecast, the less machine learning adds — there is less exploitable nonlinear structure
left in the signal. **At weekly range this is very nearly a linear-regression problem.**

## Finding 4 — at long leads, averaging beats single-model post-processing

`mos_single` picks the single best NWP model on the training fold and rescales it.
`ensemble_mean_raw` is the plain mean of all members with **no fitting at all**.

| Lead | `ensemble_mean_raw` | `mos_single` | Winner |
|---:|---:|---:|:---|
| d1 | 1.0645 | **0.8890** | MOS, by 16% |
| d4 | 1.0427 | **0.9334** | MOS, by 10% |
| d6 | 1.0708 | **0.9844** | MOS, by 8% |
| **d7** | **1.0418** | 1.0578 | **raw average, by 1.5%** |

By day 7 a two-parameter fit on the best single model is **worse than simply averaging the
members untouched**. Multi-model combination is doing more work than statistical correction
at that range — which is the standard operational finding for long-lead NWP, reproduced here.

## Finding 5 — weekly aggregation cancels ~2.6× of the hourly error

Hourly wind speeds pushed through the two-stage power conversion, summed over
**non-overlapping 168 h blocks**:

| Lead | Hourly nMAE (% rated) | **Weekly nMAE (% rated)** | Cancellation | Weekly MAPE (% of actual) |
|---:|---:|---:|---:|---:|
| d1 | 7.73% | **2.24%** | 3.5× | 9.83% |
| d2 | 8.68% | 3.32% | 2.6× | 14.53% |
| d3 | 9.32% | 3.40% | 2.7× | 14.92% |
| d4 | 9.89% | 3.43% | 2.9× | 15.57% |
| d5 | 9.85% | 4.28% | 2.3× | 19.25% |
| d6 | 9.99% | 4.09% | 2.4× | 18.76% |
| **d7** | **10.32%** | **3.95%** | **2.6×** | 18.22% |

**Two normalisations, deliberately, because mixing them inverts the conclusion.**
`% of rated` divides by rated energy and **is** comparable to the hourly nMAE — it shows the
cancellation. `% of actual` divides by the energy actually produced, whose denominator is
~a quarter of rated (measured capacity factor **0.236–0.240**), and is the number an operator
would feel. The first version of this table reported only `% of actual`, which made the
aggregate look *worse* than the hourly forecast when it is in fact ~2.6× better.

**The weekly energy total is the strongest product here.** At a full seven days ahead the
pipeline predicts weekly energy to within **~4% of rated capacity** — and weekly energy is
what a site with no metered output can actually be assessed on.

## Finding 6 — ⚠️ the quantiles are NOT calibrated

Nominal P10–P90 coverage is 0.80. Measured:

| Lead | below P10 *(want 0.10)* | below P90 *(want 0.90)* | **Coverage** *(want 0.80)* | Width |
|---:|---:|---:|---:|---:|
| d1 | 0.159 | 0.832 | **0.673** | 1.811 |
| d2 | 0.162 | 0.828 | **0.666** | 2.020 |
| d3 | 0.172 | 0.853 | **0.681** | 2.145 |
| d4 | 0.187 | 0.838 | **0.651** | 2.147 |
| d5 | 0.161 | 0.843 | **0.682** | 2.254 |
| d6 | 0.152 | 0.857 | **0.705** | 2.411 |
| d7 | 0.149 | 0.891 | **0.743** | 2.699 |

**This fails acceptance gate 5 in [08 §9](08_weekly_forecast_design.md).** The bands are
systematically **too narrow** — real coverage 65–74% against 80% nominal, at every lead. The
miss is roughly symmetric (too many observations both below P10 and above P90), so it is
width, not bias.

**Reported as a failure rather than fixed by widening.** An interval tuned until it covers is
not calibrated, it is fitted to the test set. The honest options are: recalibrate on a
validation fold, or use a conformal wrapper that guarantees coverage by construction.

**One thing does behave correctly:** interval width grows monotonically with lead
(1.81 → 2.70) and coverage improves with it, so the spread feature is tracking real
uncertainty — the model is simply under-dispersed throughout.

## Verification gates from [08 §9](08_weekly_forecast_design.md)

| # | Gate | Result |
|---:|:---|:---|
| 1 | Every model beats climatology at its lead | ✅ **pass** — worst is `mos_single` at d7, 1.0578 vs 1.2075 |
| 2 | Monotonic decay with lead | ⚠️ **one inversion** — d5 (0.9147) marginally beats d4 (0.9166) by 0.002. Within single-seed noise; not investigated further |
| 3 | Day-1 cross-check vs the 24 h result | ✅ **pass** — 0.7322 vs 0.7299 on the same split, 0.3% apart |
| 4 | ICON ceiling confirmed full-window | ✅ **pass** ([08 §4](08_weekly_forecast_design.md)) |
| 5 | Quantile calibration | ❌ **FAIL** — coverage 0.65–0.74 vs 0.80 nominal |
| 6 | Every table labelled with lead and membership | ✅ **pass** — and the label bug that motivated this gate was caught by it ([06 F8](06_what_the_protocol_caught.md)) |

## Limitations

- **Single split, single seed.** [01](01_protocol.md)'s 4-fold × 5-seed protocol was not
  applied here. Every number above is one chronological 80/20 split at seed 42, so treat
  differences under ~2% as unresolved — including the d4/d5 inversion and the ICON result.
  **The ICON conclusion in particular deserves a multi-seed repeat before it is leaned on.**
- **18 weeks, not 95.** The energy aggregate is evaluated on the test fold, which holds
  **18–19 non-overlapping weeks**. The ~95 figure in [08 §6](08_weekly_forecast_design.md)
  refers to the full window; only the test portion is available for scoring.
- **No Diebold–Mariano.** Differences between models at a given lead are not significance
  tested here.
- **Physics-derived power truth**, as everywhere in this project — these figures validate the
  forecasting step, not physical yield.
- **Deep models not run.** Excluded by design ([08](08_weekly_forecast_design.md)); nothing
  here revisits that.
