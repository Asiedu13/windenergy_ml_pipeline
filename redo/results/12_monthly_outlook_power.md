# 12 — Re-powering the 30-day outlook

> ⏱️ **Two scales in this document, and they must not be mixed.** *Block level* = a
> non-overlapping 7-day energy total, where the forecast is 100% of the prediction.
> *Monthly level* = a calendar-month total, where the forecast is ~23% of it. Skill figures
> at the two scales are not comparable and every table below says which it is.
>
> **What:** a fix for the sample problem that made [11](11_monthly_results.md)'s outlook test
> undecidable, and the model ladder that fix makes possible.
> **Why it exists:** [11](11_monthly_results.md) reported +1.1% skill, CI [−19.7%, +17.7%],
> and called it a sample-size problem. That was true but incomplete — the design was also
> *diluting* the signal, and dilution is fixable where waiting is not.

## The diagnosis

The 22 paired differences (climatology error − outlook error) sum to **+0.0080 CF in total**:
mean +0.00036 against a per-month spread of ±0.0151, i.e. **t ≈ 0.1**.

| Component | MAE (CF) | Share of monthly error |
|:---|---:|---:|
| Climatology baseline | 0.0346 | 100% |
| **Perfect-head floor** — the 23-day climatological tail | **0.0262** | **76%** |
| Most the head can ever move | 0.0084 | 24% |

A 7-day forecast is **23% of a 30-day month**. Three-quarters of the monthly error is tail
noise **common to both arms**, and because the metric is absolute error it does not cancel —
it only inflates the variance the comparison is measured against.

### The design could never have worked

| Design | n | MDE (80% power, two-sided 5%) | Target effect | Verdict |
|:---|---:|---:|---:|:---|
| **Monthly totals** (as built in [11](11_monthly_results.md)) | 22 | **26.1%** | 13.1% ceiling | ❌ **2× too coarse** |
| **7-day blocks** (this document) | **117** | **8.2%** *(17.8% worst-case)* | 20.7–39.3% head skill | ✅ |

**The original test's minimum detectable effect was twice the ceiling it was trying to
resolve.** No result it produced could have been informative — a null was guaranteed by the
design, not by the physics. That is the finding, and it is worth more than the number it
replaces.

---

## ✅ Pre-registration — recorded 2026-08-12, before any ladder result was computed

Fixed in advance so nothing below can be tuned toward a preferred answer.

**Sample.** NWP era 2024-03-01 → 2026-07-31; warm-up 2 months; evaluation from 2024-05-01.
**821 evaluable days → 117 non-overlapping 7-day blocks**, and 117 days (2,808 hours) per
lead day. Against 22 monthly totals: a **5.3×** larger sample carrying a **4.3×** larger
effect.

**Minimum detectable effect.** Block-level climatology MAE 0.0709 CF, arm sd 0.0487. The
paired-difference sd is far below the arm sd because both arms share the weather; the ratio
measured on the *existing* monthly design is **0.46**. Hence:

* **MDE 8.2%** using that ratio — the registered figure.
* **MDE 17.8%** assuming paired sd = arm sd, a bound that cannot be beaten.

**Gate 3, stated at the correct scale.** In a 7-day block the head *is* the prediction, so the
effect to detect is the head's own skill, which [09](09_weekly_forecast_results.md) measured
at **0.393 (d1) → 0.207 (d7)**. MDE 8.2% clears 20.7% comfortably and 17.8% clears it
marginally. **The 13.1% figure is a *monthly*-scale ceiling and this test does not address it
directly** — the monthly composite's interval comes from propagation instead.

> ⚠️ **A correction made during pre-registration.** The first draft of gate 3 compared the
> block-level MDE against the 13.1% *monthly* ceiling and "failed". That was a category
> error — two different scales — caught only because the MDE was computed before the result.
> Had it been computed afterwards, the mismatch would have been read as a finding about the
> models rather than a mistake in the gate.

**Registered expectations:**

1. The head **should** beat climatology clearly at block level. A null would indicate a bug in
   the tiling or aggregation, not a discovery.
2. The **monthly composite stays capped at 13.1%.** The aim is to replace an uninformative
   interval with a bounded one — *not* to convert the null in [11](11_monthly_results.md) into
   a win.
3. The ladder winner is **genuinely unknown**: [09](09_weekly_forecast_results.md) found the
   classical/ML gap narrowing to 2.3% by day 7, with raw ensemble averaging already beating
   single-model MOS there.

---

## Results

**Source:** `outputs/reports/outlook_*.csv`, from
`python -m src.outlook_power --config config_monthly.yaml`.
**Achieved sample:** 116 issues tiled the era; **114 blocks** scored after two were dropped
below 95% coverage (Open-Meteo holes in 2026-04, not a code fault — see *Gate 1*).

### Finding 1 — the baseline was the biggest single effect in the experiment

The first run scored XGBoost at **+60.3%**. That was against
`weekly.climatology_predict`, which builds its (month, hour) means from the **training
window** — at most two years inside the NWP era. [10](10_monthly_ceiling.md) requires
leave-one-year-out climatology from the full archive, and the archive *is* operationally
available: ERA5's ~5-day latency is irrelevant to a climatological mean.

| Baseline | Block MAE (CF) | Hourly MAE (CF) |
|:---|---:|---:|
| **Archive climatology** (~14 y, expanding) — admissible | **0.0728** | **0.1446** |
| Training-window climatology (≤2 y) — what 09 used | 0.1110 | 0.1659 |
| **Penalty for the weak baseline** | **+52.5%** | **+14.8%** |

**Choosing the weak baseline was worth more than every modelling decision combined**: it
moved XGBoost's headline from +39.4% to +60.3%, a 21-point swing, larger than the entire gap
between the best and worst candidate. The two climatologies are separated with
**DM p = 1.4×10⁻⁵** — this is not a rounding difference.

> ⚠️ **This reaches back into [09](09_weekly_forecast_results.md).** Its skill figures
> (0.393 at d1 → 0.207 at d7) are computed against the training-window climatology. They are
> not wrong for what they measure, but they are **not comparable** to the numbers here, and
> the like-for-like versions would be lower. Flagged rather than silently restated.

### Finding 2 — the re-powering worked

| Arm | Block MAE (CF) | Skill vs archive clim | DM p |
|:---|---:|---:|---:|
| `climatology_archive` *(reference)* | 0.0728 | — | — |
| `climatology_window` | 0.1110 | **−52.5%** | 1.4×10⁻⁵ |
| `ensemble_mean` | 0.0918 | **−26.1%** | 1.9×10⁻³ |
| `mos_single` | 0.0637 | +12.5% | 0.110 *(ns)* |
| `mos_multi` | 0.0505 | +30.7% | 2.8×10⁻⁴ |
| **`xgboost`** | **0.0441** | **+39.4%** | **1.1×10⁻⁵** |

The pre-registered MDE was 8.2%. XGBoost lands at 39.4% with **p = 1.1×10⁻⁵**, against the
old design's p ≈ 0.5. **The same forecast, the same models — only the scale it is scored at
changed.** And `mos_single` at +12.5% is *not* significant (p = 0.110), which is the MDE doing
its job: the test now distinguishes "small" from "absent" instead of calling everything absent.

### Finding 3 — the raw ensemble mean is almost pure bias

`ensemble_mean` beats climatology at **every lead** on hourly error (+0.073 to +0.141) yet
**loses by 26.1% at block level**. Decomposing:

| Arm | Block bias (CF) | Block MAE | \|bias\| / MAE |
|:---|---:|---:|---:|
| **`ensemble_mean`** | **−0.0902** | 0.0918 | **98%** |
| `mos_single` | −0.0428 | 0.0637 | 67% |
| `climatology_archive` | −0.0301 | 0.0728 | 41% |
| `mos_multi` | −0.0225 | 0.0505 | 45% |
| `xgboost` | −0.0215 | 0.0441 | 49% |

**The raw multi-model mean is 98% bias** — it under-predicts hub wind almost everywhere.
Hourly MAE partly forgives that because scatter dominates the metric; a 168-hour sum does not,
because bias is the one error that never cancels. **This is the argument for MOS stated as a
measurement**: the correction is not buying sharpness, it is removing a systematic offset that
an energy product would otherwise carry straight to the bottom line.

*(Every arm is negatively biased because 2024-05→2026-07 was windier than the 2010–2024
archive — actual block CF averaged 0.2901 against a climatological ~0.26. That is a property
of the period, not of the models.)*

### Finding 4 — the winner depends on the scale of the product

| | Hourly CF MAE | Block CF MAE |
|:---|:---|:---|
| Winner | **`mos_multi`** (5 of 7 leads) | **`xgboost`** (12.6% ahead) |

| Lead | `mos_multi` | `xgboost` | XGBoost vs MOS |
|---:|---:|---:|---:|
| d1 | **0.0921** | 0.0935 | +1.5% |
| d2 | 0.1097 | **0.1088** | −0.8% |
| d3 | 0.1195 | **0.1190** | −0.4% |
| d4 | 0.1202 | 0.1204 | +0.1% |
| d5 | **0.1098** | 0.1129 | +2.8% |
| d6 | **0.1115** | 0.1180 | +5.8% |
| d7 | **0.1232** | 0.1257 | +2.0% |

XGBoost wins on aggregates because it has both lower bias (−0.0215 vs −0.0225) and lower
block-error scatter (sd 0.0595 vs 0.0629), even while losing most hourly leads.

**And it disagrees with [09](09_weekly_forecast_results.md), which had XGBoost ahead at 7 of
7 leads.** The mechanism is not subtle: **09 scored hub wind speed in m/s; this scores
capacity factor after the power curve.** The curve is strongly nonlinear, so the model that
minimises wind-speed error is not necessarily the one that minimises energy error.

> **The project has been selecting models on m/s while the deliverable is energy.** That is
> worth more than the ranking itself. Neither result is wrong — they optimise different
> objectives — but any claim of the form "XGBoost is the best model here" needs to say *best
> at what*.

### Finding 5 — the monthly product improves, and still cannot be called significant

Composed from **month-start** issues of the selected head (leads 1–7), not from the tiled
grid — on a weekly grid the forecast covering day 3 of a month may have been issued five days
earlier, which would score leads the real product never uses.

| | n | MAE (CF) | Skill | 95% CI | Width |
|:---|---:|---:|---:|:---|---:|
| Original ([11](11_monthly_results.md)) | 22 | 0.0342 | +1.1% | [−19.7%, +17.7%] | 37.4 pp |
| **Re-powered, XGBoost head** | **26** | **0.0378** | **+4.4%** | **[−10.5%, +19.1%]** | **29.5 pp** |

The interval is **21% narrower** and the point estimate moved up, but it **still spans zero**
— exactly as pre-registered. The 13.1% ceiling did not move and never could: the forecast
still covers ~23% of a month (head weight 0.226–0.250), and no amount of head skill changes
that arithmetic. The outlook won 11 of 26 months.

**This is the honest end state.** A 39.4% improvement on the part that is forecastable becomes
a ~4% improvement on a monthly total, because three-quarters of a month is not forecastable at
all. The re-powering did not rescue the monthly product; **it explained it.**

---

## Verification gates

| # | Gate | Result |
|---:|:---|:---|
| 1 | **Tiling** — no hour predicted twice | ✅ 0 overlaps. **Fired on the first run**: 86 of 19,488 hours never predicted, traced to `build_valid_time_frame`'s `dropna` on Open-Meteo holes, not an off-by-one. Gaps became a per-block coverage guard; 2 blocks dropped |
| 2 | **No leakage** — training ends before the block | ✅ asserted every fit, never fired |
| 3 | **Power** — MDE below the effect | ✅ MDE 8.2%; achieved p = 1.1×10⁻⁵ |
| 4 | **Consistency** with the direct n=22 result | ✅ mutual containment: +4.4% inside [−19.7%, +17.7%], and +1.1% inside [−10.5%, +19.1%] |
| 5 | **Sanity floor** — every candidate beats climatology at d1 | ✅ +0.089 to +0.353 |
| 6 | **Cross-check** against [09](09_weekly_forecast_results.md) | ⚠️ **Diverges, and the divergence is the finding** — see Finding 4 (m/s vs capacity factor) |
| 7 | Tests | ✅ 28 new; 146 total |

## Limitations

* **One seed.** The three linear candidates are deterministic; XGBoost is not. Given
  [04](04_deep_models.md) measured XGBoost's seed sd at 0.0035 m/s the ranking is unlikely to
  move, but this is a single-seed result.
* **Refit monthly, issue weekly.** A mid-month issue uses a model up to four weeks stale.
  Conservative and leak-free, but it slightly understates a system that retrains on every issue.
* **114 blocks are not 114 independent samples.** Adjacent blocks share weather; the DM test
  uses HAC lag 1 at block level and lag 23 hourly, which is why p-values here are far from the
  naive ones.
* **The composite's n = 26 has not grown much.** Only the *head* was re-powered. The monthly
  interval is still a 26-sample bootstrap and will stay wide.

## What would actually make a monthly forecast viable

Not a better model — the 13.1% ceiling is arithmetic, not modelling. The one change that moves
it is **more forecastable days**, which means a longer-lead archive:

**NOAA GEFSv12 reforecast** (`noaa-gefs-retrospective` on AWS, free, no auth) covers
**2000–2019 with 16-day leads**. That would take the monthly sample from 26 to ~240 and lift
the ceiling from **13.1% to ~31%** ([10](10_monthly_ceiling.md)'s 14-day row is 30.7%), while
supplying 20 years of training data instead of two. It is GRIB2 work and a cross-source mix
that would need its own arm, and it was **not attempted here** — but it is the only lever that
changes the answer rather than the error bar.

## Related

* [10](10_monthly_ceiling.md) — the ceiling analysis and the monthly protocol
* [11](11_monthly_results.md) — the original underpowered result this re-powers
* [09](09_weekly_forecast_results.md) — the weekly models, scored in m/s against a
  training-window climatology (see Findings 1 and 4)
