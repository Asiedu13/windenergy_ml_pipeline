# 10 — Monthly scale: what is predictable, and what is not

> ⏱️ **Horizon: monthly.** Every figure here is a **calendar-month aggregate**. Documents
> [01](01_protocol.md)–[06](06_what_the_protocol_caught.md) are 24 h,
> [07](07_horizon_sweep.md) sweeps 24/48/72 h, [09](09_weekly_forecast_results.md) covers
> 24 h→168 h. This one operates a scale further out, and its errors are in **capacity-factor
> points over month-long aggregates**, not m/s over hours. **The two are not comparable.**
>
> **What:** the ceiling analysis that had to come before any monthly model was trained.
> **Why it exists:** "extend the model to a month" is not a config change — it is a request
> for something the data cannot supply. This measures what *can* be built.
> **Source:** `outputs/reports/monthly_*.csv`, from
> `python -m src.monthly --config config_monthly.yaml`.
> **Sample:** **156 complete months**, 2010-01 → 2026-07, density-corrected E-101/3050.

## The finding that reframed the task

A monthly model was requested as the next rung after 24 h → 168 h. Before training anything,
the monthly-scale signal was measured. **There is very little of it.**

| Quantity | Value | What it means |
|:---|---:|:---|
| Seasonal cycle share of monthly-CF variance | **81.6%** | the calendar is the model |
| Interannual anomaly share | **18.4%** | all a model can compete for |
| Anomaly standard deviation | 0.0454 CF | 17.4% of mean CF |
| Lag-1 anomaly autocorrelation | **0.229** | R² ceiling 0.052 |
| Lag-2 / lag-3 | 0.101 / 0.040 | signal is gone by month 2 |
| Lag-6 / lag-12 | −0.049 / 0.000 | nothing beyond the cycle itself |

Four-fifths of what happens month to month at Anloga is *which month it is*. The residual is
weather, and weather at monthly scale barely correlates with the month before it.

## The baselines, and one that must be ruled out

| Baseline | MAE (CF) | pp of rated | n |
|:---|---:|---:|---:|
| **Climatology (leave-one-year-out)** | **0.0402** | **4.02** | 156 |
| Climatology (in-sample) | 0.0371 | 3.71 | 156 |
| Anomaly persistence | 0.0447 | 4.47 | 153 |

Two things follow, and both invert a habit carried over from the hourly work:

**Persistence is not a weak baseline here — it is worse than doing nothing.** At 24 h damped
persistence beats climatology; by 48 h it loses ([07](07_horizon_sweep.md)); at monthly scale
it is beaten outright, 0.0447 against 0.0402. Last month's anomaly is not evidence about this
month's.

**In-sample climatology flatters by 7.6%.** Estimating the month-of-year mean from a set that
includes the target month gives 0.0371 instead of 0.0402. Any model scored against the
in-sample figure inherits that flattery, so `src.monthly` exposes only
[`loyo_climatology`](../../src/monthly.py) as a baseline.

## The ceiling on a monthly forecast

A monthly total issued on day 0 can use NWP for its first ~7 days — Open-Meteo previous-runs
stop at lead day 7, probed live 2026-08-12 — and must fall back to climatology for the other
~23. Granting the head a **perfect** forecast:

| Perfect days known | Monthly MAE (pp of rated) | Skill vs LOYO climatology |
|---:|---:|---:|
| 0 | 4.02 | 0.0% |
| 3 | 3.66 | 9.0% |
| 5 | 3.63 | 9.7% |
| **7** | **3.49** | **13.1%** |
| 10 | 3.16 | 21.5% |
| 14 | 2.78 | 30.7% |
| 21 | 1.95 | 51.5% |
| 30 | 0.22 | 94.5% |

**13.1% is unreachable.** It grants the first week zero forecast error, where the real day-7
model scores 20.7% skill rather than 100% ([09](09_weekly_forecast_results.md)). What was
actually achieved is in [11](11_monthly_results.md) — and it is a good deal less than this.

The `30` row is a sanity check rather than an option: perfect knowledge of the whole month
still leaves 5.5% error, from months clipped at the archive's edges.

Note the shape. Going from 0 to 7 perfect days buys 13.1 points; going from 7 to 14 buys
another 17.6. **The first week is the least valuable week to know**, because a month's total
is dominated by the three weeks the forecast cannot reach.

## Why there is no 30-day hourly model here

Not a resourcing decision. NWP previous-runs stop at d7, and deterministic point skill at 30
days is zero for physical reasons rather than data ones. A model trained to that target would
return a confident artifact with no skill — the failure this folder's
[mistake register](../00_mistake_register.md) exists to prevent. So "monthly" was split:

* **M2 — resource climatology.** P50/P75/P90 exceedance, diurnal profile, interannual spread.
  Well-posed at n=156 and does not depend on beating a baseline. → [11](11_monthly_results.md)
* **M1 — 30-day outlook.** NWP head + climatological tail, ceiling 13.1%.
  → [11](11_monthly_results.md)
* **M3 — interannual anomaly from large-scale indices.** Not attempted, by decision. Lag-1 is
  0.229 and gone by lag-2, so the only plausible predictor is ENSO/MJO state; at n=156 with
  leave-one-year-out folds, anything past ~3 parameters overfits. If picked up later it must
  be pre-registered, and a null result is the expected outcome.

## The seasonal cycle itself

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean CF | 0.188 | 0.287 | 0.335 | 0.248 | 0.178 | 0.244 | 0.411 | **0.429** | 0.337 | 0.174 | 0.141 | **0.140** |
| Interannual sd | 0.047 | 0.055 | 0.048 | 0.043 | 0.043 | **0.062** | 0.054 | 0.046 | 0.046 | 0.037 | **0.026** | 0.045 |
| Years | 13 | 13 | 14 | 14 | 14 | 14 | 14 | 12 | 12 | 12 | 12 | 12 |

August runs **3.07× December**. The monsoon peak (Jul–Sep) and the Harmattan trough (Oct–Dec)
are the whole story. Interannual spread is roughly flat across months at 0.026–0.062, so **no
month is meaningfully more predictable than another — only windier.**

## Protocol for monthly work

Forced by n=156, not inherited from the hourly protocol in [01](01_protocol.md):

* **Leave-one-year-out folds, never a random split.** Months within a year share weather.
* **Leave-one-year-out climatology is the only admissible baseline** (see above).
* **Diebold–Mariano with HAC lag 1** — anomaly autocorrelation is 0.229 at lag 1, ~0 after.
* **Report n on every table.** ~156 for climatology, ~22 for anything NWP-dependent. Mixing
  the two would be the most misleading thing this document could do.
* **Bootstrap intervals, not point estimates**, for anything on the 22-month sample.
* **No deep models.** At 156 samples the question does not arise.

## Data constraints, measured rather than assumed

| Constraint | Status |
|:---|:---|
| ERA5 archive | **156 complete months**, with a deliberate **41-month hole** (2016-08→2019-12) |
| NWP history | **all-null before 2024-03** — probed live, not inferred from the cache |
| NWP leads | d1–d7 (ECMWF, GFS); **ICON stops at d6**, confirmed again here |
| Months usable for M1 | **22**, after the expanding-window warm-up |

The 41-month hole is why [`anomaly_autocorrelation`](../../src/monthly.py) pairs only
genuinely consecutive months: a naive `shift(1)` would pair 2016-07 with 2020-01 and report
the result as a lag-1 correlation between adjacent months.

**The forward extension was worth more than the backward one.** The archive was extended
2026-01→2026-07 (7 ERA5 months, plus a re-fetch of the 2026 NWP caches at leads 1–7), which
took M1's evaluable sample from 16 to 22 months. Filling the 41-month backward hole would have
added ~2 years per calendar month to the climatology but **nothing at all** to M1, because
NWP does not exist before 2024-03. It was not run.

## Related

* [07](07_horizon_sweep.md) — the 24/48/72 h decay that motivated going further out
* [09](09_weekly_forecast_results.md) — the 168 h model that supplies this outlook's head
* [11](11_monthly_results.md) — the measured M2 and M1 results
