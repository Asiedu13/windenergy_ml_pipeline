# 11 — Monthly: measured results

> ⏱️ **Horizon: monthly.** Errors are in **capacity-factor points over calendar-month
> aggregates**. Nothing here is comparable to the m/s figures in [01](01_protocol.md)–[09](09_weekly_forecast_results.md).
>
> **What:** the two monthly products built after [10](10_monthly_ceiling.md) established what
> was possible — a resource climatology (M2) and a 30-day outlook (M1).
> **Protocol:** [10 §Protocol](10_monthly_ceiling.md). Leave-one-year-out throughout;
> bootstrap intervals on anything NWP-dependent.
> **Sources:** `outputs/reports/monthly_climatology.csv`, `monthly_exceedance_validation.csv`,
> `monthly_annual_energy.csv`, `monthly_diurnal.csv`, `monthly_outlook*.csv`.
> **Samples:** **156 months** for M2 (2010-01→2026-07); **22 months** for M1 (2024-10→2026-07).

## Summary

| | Verdict |
|:---|:---|
| **M2 — resource climatology** | ✅ **Delivered.** P50/P75/P90 per calendar month, validated out-of-sample |
| **M2 — P90 calibration** | ⚠️ **Fixed by measurement.** The obvious estimator was wrong in the dangerous direction |
| **M1 — 30-day outlook** | ❌ **No demonstrable skill.** +1.1%, interval [−19.7%, +17.7%] |

> 🔁 **M1's test was later found to be structurally underpowered, and was rebuilt.** Its
> minimum detectable effect was **26.1%** against a **13.1%** ceiling — it could not have
> resolved even a perfect forecast, so the null below was guaranteed by the design rather than
> by the physics. Re-scored on 7-day blocks the same forecast is worth **+39.4%
> (p = 1.1×10⁻⁵)**, and the monthly composite becomes **+4.4% [−10.5%, +19.1%]** — still not
> significant, because the 13.1% ceiling never moved. **This document is left as written**, as
> the record of what the first design returned. → [12](12_monthly_outlook_power.md)

---

# M2 — Monthly resource climatology

## The product — capacity factor by calendar month

**PXX is the value *exceeded* XX% of the time**, so P90 is a *low*, conservative figure (the
10th percentile), not a high one. Reading it the other way inverts the entire product, so the
convention is encoded once in `EXCEEDANCE_QUANTILE` and tested.

| Month | Years | Mean CF | sd | **P50** | **P75** | **P90** | P50 energy (MWh) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Jan | 13 | 0.1883 | 0.0473 | 0.1883 | 0.1564 | 0.1277 | 427 |
| Feb | 13 | 0.2868 | 0.0550 | 0.2868 | 0.2497 | 0.2163 | 588 |
| Mar | 14 | 0.3351 | 0.0484 | 0.3351 | 0.3025 | 0.2731 | 760 |
| Apr | 14 | 0.2483 | 0.0425 | 0.2483 | 0.2197 | 0.1939 | 545 |
| May | 14 | 0.1782 | 0.0429 | 0.1782 | 0.1493 | 0.1233 | 404 |
| Jun | 14 | 0.2437 | 0.0615 | 0.2437 | 0.2023 | 0.1649 | 535 |
| Jul | 14 | 0.4114 | 0.0540 | 0.4114 | 0.3750 | 0.3422 | 934 |
| **Aug** | 12 | **0.4292** | 0.0464 | **0.4292** | 0.3979 | 0.3698 | **974** |
| Sep | 12 | 0.3365 | 0.0456 | 0.3365 | 0.3057 | 0.2780 | 739 |
| Oct | 12 | 0.1742 | 0.0370 | 0.1742 | 0.1493 | 0.1268 | 395 |
| Nov | 12 | 0.1408 | 0.0262 | 0.1408 | 0.1231 | 0.1072 | 309 |
| **Dec** | 12 | **0.1400** | 0.0450 | **0.1400** | 0.1096 | 0.0823 | 318 |

Energy is for one E-101/3050 (3.05 MW). **August yields 3.07× December.**

## Finding 1 — the obvious exceedance estimator fails, in the dangerous direction

The natural way to compute a P90 from 12 years is to take the empirical 10th percentile. Under
leave-one-year-out validation it does not hold:

| Level | Nominal | **Empirical** | **Normal fit** | n |
|:---|---:|---:|---:|---:|
| P50 | 0.50 | 0.4936 | 0.4808 | 156 |
| P75 | 0.75 | 0.7115 | 0.7179 | 156 |
| **P90** | **0.90** | **0.8462** ❌ | **0.8974** ✅ | 156 |

The empirical P90 is exceeded **84.6%** of the time against a nominal 90% — meaning the
published P90 was **too high**, i.e. too optimistic. For a figure used to size a PPA or a debt
service cover ratio, optimistic is the failure direction that matters.

**The cause is sample size, not the concept.** With 12–14 years per calendar month, the
empirical 10th percentile sits between the first and second order statistics, so holding one
year out moves it by an entire rank. Fitting a normal and reading the tail off it uses all 12
points to place that tail, and lands within **0.3 points of nominal**.

`exceedance_table` therefore defaults to `method="normal"`, and both estimators are scored on
every run so the choice stays a measurement rather than a preference. Note the trade: the
normal fit is slightly *worse* at P50 (0.4808 vs 0.4936). It is chosen because the tail is
what an exceedance product exists to state.

## Finding 2 — the diurnal cycle rivals the seasonal one

| | Range in mean CF |
|:---|---:|
| Seasonal (Aug − Dec monthly means) | 0.289 |
| Diurnal (Feb, best hour − worst hour) | **0.207** |
| Diurnal (Nov, best hour − worst hour) | 0.107 |

Wind peaks **16:00–21:00 UTC** and troughs **08:00–11:00 UTC** in every month (Anloga is
UTC+0, so these are local hours). The evening peak is a sea-breeze/monsoon signature and it is
consistent year-round.

**This matters for a solar hybrid.** The wind trough (08:00–11:00) sits inside the solar
peak, and the wind peak (16:00–21:00) sits where solar is falling away. The two resources are
close to anti-correlated on the daily cycle at this site — the most useful operational fact in
this document, and it comes out of the climatology rather than any forecast.

## Finding 3 — interannual spread is large, and does not shrink

| Complete years | Mean CF range | Annual energy range |
|---:|:---|:---|
| 11 | 0.2342 – 0.2877 | **6,246 – 7,691 MWh** |

The best complete year produces **23% more energy** than the worst. A single-year resource
assessment at this site could be wrong by that much in either direction — which is the
argument for the 156-month archive over the 22-month NWP window, and the reason P90 exists
as a product at all.

---

# M1 — 30-day energy outlook

## Design

Issued on day 0: the weekly per-lead models cover days 1–7, climatology covers the rest. The
outlook and its baseline are **identical except for the head**, so any difference is
attributable to the forecast and nothing else:

```
baseline = clim(M)
outlook  = clim(M) + (head_hours / month_hours) x (head_forecast - head_clim)
```

The head enters as an **anomaly**, not a level. The first seven days of a month are not
climatologically identical to the month as a whole, and comparing a 7-day forecast level
against a 30-day climatological level would score that seasonal drift as forecast error.

Training is an **expanding window**: for target month *M*, every lead model sees only valid
times before *M* begins. That is a deployment rehearsal rather than a study — on the first of
the month you have the past and nothing else — and it costs the first several months of the
NWP record, which is why n is 22 and not 28.

## Result

| Arm | n | MAE (CF) | vs clim 0.0346 | Skill | 95% CI | P(better) |
|:---|---:|---:|---:|---:|:---|---:|
| **Outlook (real 7-day forecast)** | 22 | 0.0342 | −0.0004 | **+1.1%** | **[−19.7%, +17.7%]** | 0.53 |
| Perfect head (ceiling on these months) | 22 | 0.0262 | −0.0084 | +24.2% | [−0.0%, +44.1%] | 0.97 |

### ❌ There is no demonstrable skill.

The interval spans zero, the point estimate is +1.1%, and the outlook beat climatology in
**12 of 22 months** — indistinguishable from a coin flip. This is reported as a null result,
not rounded up into a claim.

**This was the predicted outcome, and predicting it is the point.** [10](10_monthly_ceiling.md)
established the ceiling at 13.1% *with a perfect head*, and 22 months has nowhere near the
power to resolve a real effect smaller than that. The experiment was run to measure it, not to
hope.

## Finding 4 — how unstable a 22-month estimate is

During this work the archive grew from 20 to 22 evaluable months. The skill estimate moved:

| Sample | Outlook skill | Perfect-head skill | Perfect-head verdict |
|:---|---:|---:|:---|
| 20 months | +6.9% | +32.5% | significant, [+7.1%, +49.2%] |
| 21 months | +4.7% | +26.0% | significant, [+0.3%, +45.3%] |
| **22 months** | **+1.1%** | **+24.2%** | **not significant, [−0.0%, +44.1%]** |

**Two extra months moved the outlook's headline by 5.8 points and flipped the ceiling arm's
verdict from significant to not.** Had this been written up two months earlier it would have
reported a +6.9% outlook and a significant ceiling. Nothing was wrong with those numbers — the
sample was simply too small to be stable, and only reporting the interval makes that visible.

## Finding 5 — the ceiling on 22 months is not the ceiling

The perfect-head arm scores **+24.2%** on these 22 months against the **13.1%** measured
across all 156. The gap is **sample, not method**:

| Sample | Head referenced to month mean | Head referenced to first-7-days clim |
|:---|---:|---:|
| All 156 months | 13.3% | 14.9% |
| The recent ~21 months | 22.1% | 24.0% |

Switching the head reference is worth ~1.6 points; switching the sample is worth ~10. The
22-month interval [−0.0%, +44.1%] comfortably contains the archive-wide 14.9%, so the two are
consistent — the recent window simply happened to be one where a known first week helps more.

**Quoting +24.2% as "the ceiling" would be wrong.** It is the ceiling on 22 particular months.

---

## What would change the M1 verdict

Not a better model — the ceiling is 13.1% and the classical/ML gap is already small at weekly
range ([09](09_weekly_forecast_results.md)). **The binding constraint is n.** Open-Meteo
previous-runs begin 2024-03 and that cannot be undone; the sample grows one month per month.
Detecting a true 5% effect against this month-to-month variance needs on the order of 100+
months, i.e. **another six years of waiting**, or a site with metered output and a longer
forecast archive.

That is the honest conclusion: **the 30-day outlook is not a modelling problem, it is a
sample-size problem**, and no amount of architecture fixes it.

## Limitations

* **M1's 22 months are not independent of each other's training data** in the usual sense —
  the expanding window means later months train on more data than earlier ones. That is
  operationally correct but makes the months non-exchangeable, so the bootstrap interval is
  approximate.
* **M2's exceedance rests on 12–14 years per calendar month.** The normal fit calibrates well
  (0.8974 vs 0.900) but assumes monthly CF is roughly Gaussian; a Weibull or skew-normal fit
  was not tested.
* **The 41-month archive hole (2016-08→2019-12) is not filled.** It costs ~2 years per
  calendar month in the climatology and nothing in M1.
* **Single seed.** M1 uses linear MOS heads, which are deterministic, so seed variance does
  not arise — but the underlying weekly models were selected on a single split
  ([09](09_weekly_forecast_results.md)).

## Related

* [10](10_monthly_ceiling.md) — the ceiling analysis that set these expectations
* [09](09_weekly_forecast_results.md) — the 7-day models supplying M1's head
* [07](07_horizon_sweep.md) — the decay curve that led here
