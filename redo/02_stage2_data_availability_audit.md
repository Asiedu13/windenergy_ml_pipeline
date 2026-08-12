# Stage 2 — Data availability audit

> **What this stage is:** characterising every candidate data source *against the data
> itself* before designing anything around it.
> **Why it comes second:** four separate silent defects in the original project
> ([register group B](00_mistake_register.md)) were all assumptions about someone else's
> data that were never checked against a downloaded file. One table, built early, prevents
> all four — and prevents the late study-window move that invalidated three iterations of
> baselines.
> **Effort:** 2–3 days, most of it small downloads and inspection.

## 2.1 The rule

> **Documentation describes intent. The file describes reality. Assert against the file.**

Every row of the table below must be filled in from a **downloaded sample**, not from the
provider's docs page. Where the two disagree, the file wins and the disagreement gets
recorded — that disagreement is usually where the bug lives.

## 2.2 The source characterisation table

For each candidate source, establish all nine columns before writing any ingest code:

| Property | Why it matters | What went wrong originally |
|:---|:---|:---|
| **Archive start / end** | Determines the study window | Open-Meteo's archive begins 2024-03; discovered late, forcing the window to move and invalidating v1–v4 baselines |
| **Publication latency** | Determines deployability | Never recorded; ERA5's ~5-day latency made 53 of 104 features offline-only ([register A4](00_mistake_register.md)) |
| **Measurement height** | Wind scales with height | NASA POWER's 50 m wind was treated as 100 m — hub wind understated **10.7%**, and v1's results carry it |
| **Units** | Silent 100× errors | Open-Meteo reports hPa, ERA5 Pa — both landed in the same feature frame |
| **Timestamp convention** | Instantaneous vs interval-averaged, start- vs end-labelled | NASA POWER is hour-averaged and end-labelled: ERA5(t) = NASA(t+1) |
| **Time zone** | Off-by-hours errors | UTC throughout, asserted |
| **Spatial resolution** | Determines what "a point" means | NASA POWER's ~55 km cell makes its target smoother and easier to predict — a confound when comparing across sources |
| **Actual grid shape returned** | Requested ≠ received | `pad = 0.2°` was written for a 3×3 patch and returns **1×2**; ten spatial features are mathematically dead ([register B4](00_mistake_register.md)) |
| **Null pattern** | Distinguishes "no data" from "zero" | ECMWF serves no gusts here — three columns 100% null |

## 2.3 Verify each one this way

Concrete checks, each cheap:

- **Height and units** — compare a month of the candidate against a reference source. If
  one is systematically 10–15% lower, suspect height. If it is ~100× off, suspect units.
- **Timestamp convention** — cross-correlate the two sources over lags −6…+6 h. A clean peak
  at any lag other than 0 is a convention difference, and must be corrected explicitly at
  ingest rather than absorbed silently downstream.
- **Grid shape** — open the file and assert the dimension sizes. `assert ds.latitude.size == 3`
  would have caught B4 on the first download.
- **Latency** — request today's data. If it fails, request yesterday's. Binary-search back
  until it succeeds. That is the latency, measured rather than assumed.

## 2.4 Turn the audit into code, not just a document

Every property above becomes an assertion that runs at ingest:

```
SOURCE_METADATA = {
    "era5":       {"wind_height_m": 100.0, "pressure_unit": "Pa",  "time_offset_h": 0,  "latency_days": 5},
    "nasa_power": {"wind_height_m":  50.0, "pressure_unit": "Pa",  "time_offset_h": -1, "latency_days": 1},
    "open_meteo": {"wind_height_m": 100.0, "pressure_unit": "hPa", "time_offset_h": 0,  "latency_days": 0},
}
```

The original project eventually built exactly this (`SOURCE_WIND_HEIGHT_M`,
`NASA_POWER_TIME_OFFSET_H`) — but **as a fix, in iteration five, after the defects had
already contaminated published results.** The same twenty lines written in stage 2 cost
nothing and prevent both.

Add range assertions too: surface pressure in Pa should be ~90,000–105,000. A value of 1013
is hPa wearing a Pa label, and an assertion catches it in the second it happens.

## 2.5 Choose the study window *from* the table

Only now, with archive coverage known for every source, pick the window. The rule:

> **The study window is the intersection of every source's coverage, evaluated against the
> training-length requirement from [stage 4](04_stage4_ceiling_analysis.md).**

The original chose a window from ERA5 alone, built four iterations on it, then had to move
it to 2024-03 → 2025-12 when the forecast archive turned out not to reach back. Every
baseline had to be recomputed and no v1–v4 number remained comparable.

Also record the **download cost** honestly before committing. ERA5 for a recent window runs
at roughly 25–90 minutes per request, almost all of it queueing, and it cannot be
parallelised — CDS grants about one running job per user, which the original project spent
75 minutes proving ([register D2](00_mistake_register.md)).

## 2.6 Decide the truth source deliberately

Write down which source defines the **target**, and why, and what its known confounds are.
The original arrived at this correctly but late: ERA5 is the truth series, NASA POWER is a
cross-check only, and their absolute MAEs are **not comparable** — NASA POWER's is inflated
~12% by the height extrapolation and simultaneously deflated by being coarser and
hour-averaged. Two confounds running in opposite directions means only R² and relative
improvements transfer between them.

## Exit gate

You may not proceed to stage 3 until:

1. The nine-column table is complete for every candidate source, each entry verified against
   a downloaded file.
2. `SOURCE_METADATA` exists in code with assertions that run at ingest.
3. The study window is chosen, with the reasoning and the download-cost estimate recorded.
4. The set of features surviving the [stage 1](01_stage1_deployment_contract.md) latency
   budget is written down explicitly — **this is the operational feature set**, and
   [stage 4](04_stage4_ceiling_analysis.md) analyses it, not the offline one.

**How this gate can fail:** any blank cell; any property sourced from documentation rather
than a file; or a window chosen before the table was complete.

> **What the latency column turned out to be worth.** Running the original project's models
> on the operational feature set only — NWP + calendar, no reanalysis — scores **0.8009 m/s**
> against 0.8037 for the full set, at a 24 h horizon
> ([results/03](results/03_operational_vs_study.md)). Here, excluding every latency-blocked
> feature cost *nothing*.
>
> **That result is site-specific and must not be generalised.** At a location with weaker NWP
> skill, or a shorter horizon where persistence is strong, reanalysis features could matter a
> great deal. The audit's job is to tell you *which* features are in question; the rehearsal
> in [stage 8](08_stage8_deployment_rehearsal.md) tells you what they cost.

---

Previous: [Stage 1](01_stage1_deployment_contract.md) ·
Next: [Stage 3 — Alignment harness](03_stage3_alignment_harness.md)
