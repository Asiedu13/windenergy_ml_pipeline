# 07 — Horizon sweep: 24 h → 48 h → 72 h

> **What:** the same eleven predictors run at three forecast horizons, to measure how skill
> decays with lead time.
> **Why it exists:** every other document in this folder is a **24 h** result. A weekly
> forecast was proposed, and the honest first question is not "how do we build it" but
> "does forecast skill survive that far out". This measures the slope.
> **Protocol:** [01](01_protocol.md), unchanged apart from the horizon.
> **Sources:** `outputs/reports/bakeoff.csv`, `bakeoff_h48.csv`, `bakeoff_h72.csv`.
> **Feature set:** `operational` (NWP + calendar) for all three — the reanalysis features are
> near-dead beyond 24 h anyway, and [03](03_operational_vs_study.md) showed they cost nothing.

## ⚠️ The correctness problem this sweep had to fix first

A longer horizon is **not** a config change. Column `nwp_..._d{N}` at valid time τ is the
forecast for τ **issued 24·N hours before τ**. Row *t* receives the value at `t + h`, so it
was issued at `(t + h) − 24N` — knowable at *t* only when:

```
24·N ≥ h
```

| Horizon | Legitimate leads | Would have been used before the fix |
|---:|:---|:---|
| 24 h | N ≥ 1 | d1, d2, d3 ✅ |
| 48 h | N ≥ 2 | d1 ❌ *(issued at t+24 — one day in the future)* |
| 72 h | N ≥ 3 | d1, d2 ❌ |
| 168 h | N ≥ 7 | d1, d2, d3 ❌ *(d1 issued at t+144 — six days ahead)* |

Nothing checked this. `add_nwp_features` now drops every too-short lead and **raises** if none
survives. Two defects surfaced getting there — see [06](06_what_the_protocol_caught.md),
F3 and F4.

## Results — `operational` features, 4 rolling folds, MAE in m/s

| Model | **24 h** | **48 h** | **72 h** | 24→72 change |
|:---|---:|---:|---:|---:|
| `xgboost` | **0.8009** | **0.8739** | **0.9280** | +15.9% |
| `lightgbm` | 0.8032 | 0.8824 | 0.9431 | +17.4% |
| `ridge` | 0.8383 | 0.9093 | 0.9645 | +15.0% |
| **`mos_multi`** | **0.8412** | **0.9202** | **0.9658** | +14.8% |
| `kalman_mos` | 0.8572 | 0.9255 | 0.9829 | +14.7% |
| `analog_ensemble` | 0.8662 | 0.9543 | 0.9998 | +15.4% |
| `mos_single` | 0.9904 | 1.0293 | 1.0422 | +5.2% |
| `sarimax_nwp` | 1.2554 | 1.2970 | 1.3043 | +3.9% |
| **`climatology`** | **1.2725** | **1.2844** | **1.3050** | **+2.6%** |
| `damped_persistence` | 1.1861 | 1.3662 | 1.3439 | +13.3% |
| `persistence` | 1.3183 | 1.6162 | 1.5873 | +20.4% |

## Finding 1 — the baseline crossover happens between 24 h and 48 h

| Horizon | damped persistence | climatology | Winner |
|---:|---:|---:|:---|
| 24 h | **1.1861** | 1.2725 | persistence-family |
| 48 h | 1.3662 | **1.2844** | **climatology** |
| 72 h | 1.3439 | **1.3050** | **climatology** |

**Climatology is nearly horizon-independent** (+2.6% over 48 hours) because it never used the
recent past in the first place. Persistence degrades fast (+20.4%) because the
autocorrelation it depends on decays.

> **For any horizon beyond ~36 h, climatology is the baseline to beat — not persistence.**
> Reporting skill against persistence at weekly range would flatter the model by ~20% and is
> the long-lead version of the error catalogued in
> [register C3](../00_mistake_register.md).

## Finding 2 — decay is **decelerating**, which is what makes a weekly model viable

Skill against **climatology** (the correct long-lead reference), for XGBoost:

| Horizon | XGBoost MAE | Climatology | Skill vs climatology | Δ per 24 h |
|---:|---:|---:|---:|---:|
| 24 h | 0.8009 | 1.2725 | **0.371** | — |
| 48 h | 0.8739 | 1.2844 | **0.320** | −0.051 |
| 72 h | 0.9280 | 1.3050 | **0.289** | −0.031 |

The loss per extra day is shrinking (−0.051, then −0.031, a ratio of ~0.6). Continuing that
geometric decay:

| Horizon | Extrapolated skill vs climatology |
|---:|---:|
| 96 h | ~0.27 |
| 120 h | ~0.26 |
| 144 h | ~0.25 |
| **168 h (7 days)** | **~0.25** |

> ⚠️ **This is an extrapolation from three points, not a measurement.** It assumes the decay
> stays geometric, which is exactly the kind of assumption this project has been burned by
> before. Treat it as "worth building and measuring", not as a result.

**But the direction is clear enough to decide on:** a 7-day model would plausibly still beat
climatology by ~25%, and it does not collapse. **The weekly product is worth building.** Had
skill converged on climatology by 72 h, the honest answer would have been to stop — and that
negative result would have been equally publishable.

## Finding 3 — the classical/ML gap is stable across horizon

| Horizon | `mos_multi` | `xgboost` | Gap |
|---:|---:|---:|---:|
| 24 h | 0.8412 | 0.8009 | 5.0% |
| 48 h | 0.9202 | 0.8739 | 5.3% |
| 72 h | 0.9658 | 0.9280 | 4.1% |

A 21-parameter linear model stays within ~5% of gradient boosting at every horizon tested.
**The "most of the skill is information, not machine learning" finding
([02](02_classical_vs_ml.md)) does not erode with lead time.** At 72 h `ridge` (0.9645) and
`mos_multi` (0.9658) are separated by 0.0013 m/s — a tie.

## Finding 4 — skill *against persistence* rises with horizon, and that is a trap

XGBoost's skill vs persistence goes **+0.388 → +0.457 → +0.414** across the three horizons —
it looks like the model improves with lead time. It does not. Persistence is collapsing
faster than the model is. **A skill score is only as meaningful as its denominator**, which
is why the climatology-referenced table in finding 2 is the one to quote.

## Data constraint on any weekly build

Probed directly against the Open-Meteo Previous Runs API (2025-06-01→03, this site):

| Model | Leads available | Max horizon |
|:---|:---|---:|
| `ecmwf_ifs025` | d1–d7 | **168 h** |
| `gfs_global` | d1–d7 | **168 h** |
| `icon_global` | **d1–d6** | **144 h** |

`previous_day8` returns nulls for all three.

**So there is a genuine design choice**, and it is not free either way:

| Option | Horizon | Ensemble | Cost |
|:---|---:|---:|:---|
| **A** | 168 h (7 days) | 2 models | loses a third of the inter-model spread — the uncertainty signal a weekly product most needs |
| **B** | 144 h (6 days) | 3 models | a day short of a calendar week |

Given that a weekly forecast must be probabilistic ([07 stage](../07_stage7_power_and_uncertainty.md))
and inter-model spread is the only uncertainty signal on hand, **B is the better engineering
choice unless a full seven days is a hard requirement.** Neither has been measured; this is
reasoning, not evidence.

## Caveats

- **Three horizons is a short curve.** 24/48/72 h are what the cached `d1`–`d3` columns
  support. Extending to 96/120/144/168 h requires re-fetching with `lead_days: [1..7]` —
  cheap (~6 MB, no CDS queue) but not yet done.
- **`operational` features only.** The `study` set was not swept; at these horizons the
  reanalysis features are expected to be near-dead, but that is an expectation, not a
  measurement.
- **Deep models were not swept.** Phase B is 24 h only.
- Each horizon re-derives its own baselines. **Never compare an MAE across horizons without
  both labelled** — the error [register C4](../00_mistake_register.md) records.
