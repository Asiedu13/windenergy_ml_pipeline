# 03 — Operational vs study: the deployment rehearsal

> **What:** the same eleven predictors run twice — once on every feature, once on only the
> features that would actually exist when a forecast is issued.
> **Why it matters:** [stage 8](../08_stage8_deployment_rehearsal.md) identified this as the
> measurement the original project never made, and called it the folder's most important
> stage. This is that measurement.
> **Protocol:** [01](01_protocol.md). **Source:** `outputs/reports/bakeoff.csv`.

## The problem being tested

The v5 model uses 104 features. **Roughly half come from ERA5 reanalysis, which publishes
with about five days of latency.** At the moment a forecast for t+24 h would actually be
issued, the atmospheric state at *t* does not yet exist.

So the deployed system was never a slightly degraded version of the evaluated one — it was
a *different model on half the inputs*, and its accuracy was unknown.

| Feature set | Columns | Contents | Available at issue time? |
|:---|---:|:---|:---|
| `study` | 94 | Everything live | No — 39 columns are reanalysis-derived |
| `operational` | 55 | NWP forecasts + calendar only | **Yes** |

*Both sets exclude the ten mathematically dead spatial columns; see [01](01_protocol.md).*

## Result — 24 h horizon, 4 rolling folds

| Model | `study` MAE | `operational` MAE | Difference |
|:---|---:|---:|---:|
| **`xgboost`** | 0.8037 | **0.8009** | **−0.0028** *(operational better)* |
| `lightgbm` | 0.8086 | **0.8032** | −0.0054 *(operational better)* |
| `ridge` | 0.8422 | **0.8383** | −0.0039 *(operational better)* |
| `mos_multi` | 0.8412 | 0.8412 | 0.0000 |
| `kalman_mos` | 0.8572 | 0.8572 | 0.0000 |
| `analog_ensemble` | 0.8662 | 0.8662 | 0.0000 |
| `mos_single` | 0.9904 | 0.9904 | 0.0000 |
| `sarimax_nwp` | 1.2554 | 1.2554 | 0.0000 |
| `persistence` | 1.3183 | 1.3183 | 0.0000 |

The classical models are unchanged by construction — they only ever read NWP and calendar
columns, so the mask does nothing to them. **The three models that *can* see reanalysis all
get very slightly better without it.**

## What this establishes

> **Removing every latency-blocked feature costs nothing. The degradation is negative.**

The deployability concern that motivated [stage 8](../08_stage8_deployment_rehearsal.md) is
**real in principle and free in practice at this site.** An operational system running on
NWP forecasts alone loses no accuracy — it gains a hair.

That converts the project's position materially. Before this measurement, the honest
statement was "the headline is an offline number and the operational score is unknown."
Now it is: **the operational score is known, and it is the same.**

## Why the reanalysis features add nothing

Consistent with the SHAP ranking on the authoritative v5 run, where NWP forecast columns
occupied **all of the top six** and the best reanalysis feature sat at rank 7 with roughly
one-sixth the contribution of the top column
([`docs/11`](../../docs/11_nwp_forecast_features.md)).

The mechanism is straightforward and worth stating plainly: **a day-ahead forecast of the
target hour is simply better information than an observation from 24 hours earlier.** Once
the forecast is present, the analysed state at *t* is largely redundant — and 39 redundant,
partly-collinear columns are a small liability rather than an asset, which is why the
scores tick *down* when they are removed.

**A prediction that was recorded before measuring, and was right.**
[Stage 8](../08_stage8_deployment_rehearsal.md) wrote: *"the degradation may be small… but
that is a hypothesis, and the entire point of this stage is that it is currently
untested."* It was small. Recording the prediction before the measurement is what makes it
worth anything.

## What this does *not* establish

- **It does not make the system deployable.** There is still no wind farm at Anloga, no
  SCADA to validate against, no serving path, no monitoring, and no retraining schedule.
  This removes one blocker, not all of them.
- **It does not verify the lead-time semantics.** `previous_day1` is a fixed *day* offset,
  so the effective lead varies through the target day (~24–47 h) and has not been checked
  hour by hour. A true operational setup would fix the issue cycle.
- **It is one site and one horizon (24 h).** At a site with weaker NWP skill, or a shorter
  horizon where persistence is strong, reanalysis features could matter considerably more.
- **Deep models were not run on the operational set.** Phase B covers `study` only; see
  [04](04_deep_models.md).
