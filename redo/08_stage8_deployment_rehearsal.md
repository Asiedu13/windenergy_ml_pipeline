# Stage 8 — Deployment rehearsal

> **What this stage is:** re-running the finished model using **only information that would
> genuinely have existed at issue time**, and reporting the degradation.
> **Why it matters more than any other stage in this folder:** the original project has no
> equivalent. It produced a model using 104 features, of which **53 come from ERA5
> reanalysis — data published with roughly five days of latency.** At the moment a forecast
> for tomorrow would actually be issued, those 53 features do not exist. Nobody has ever
> measured what the model scores without them.
> **Effort:** 1–2 days. It produces the number an operator would actually see.

## 8.1 The gap this closes

The original's headline is **0.7295 m/s MAE, R² 0.665, 7.48% nMAE**. That is an *offline*
number: it assumes access to the analysed atmospheric state at the moment of forecasting.

| Feature group | Count | Available at issue time? |
|:---|---:|:---|
| NWP forecast columns | 51 | **Yes** — a forecast is available before its valid time by construction |
| ERA5-derived (surface, upper-air, spatial gradients, target lags) | 53 | **No** — ~5-day reanalysis latency |

So the deployed system is not a slightly degraded version of the evaluated one. **It is a
different model on half the inputs**, and its accuracy is unknown.

> This is not a criticism of using reanalysis. Reanalysis is the right choice for a *study*
> of forecast skill — it is consistent, gap-free and physically coherent. The error was
> never measuring the operational variant, and never saying which one the headline number
> described.

## 8.2 Build the operational feature set

From the [stage 1](01_stage1_deployment_contract.md) latency budget, partition every feature:

- **Tier 1 — always available.** NWP forecasts, calendar features, static site properties.
- **Tier 2 — available with real hardware.** On-site anemometry, live SCADA. Include only if
  the hardware exists or is genuinely planned; a met mast is a capital decision, not an
  assumption.
- **Tier 3 — offline only.** Reanalysis-derived anything.

Then define two models explicitly and name them so they can never be confused in a table:

| Model | Features | What it is for |
|:---|:---|:---|
| `study` | all tiers | Answers "does forecast information help?" — the scientific question |
| `operational` | tier 1 (+ tier 2 if real) | Answers "what would we actually get?" — the engineering question |

## 8.3 Run the rehearsal

1. Train and evaluate `operational` under the **same** protocol as `study`: same window,
   same split, same seeds, same baselines.
2. Report both, side by side, always. Never report one alone.
3. Compute the **degradation** explicitly — that number is the honest cost of deployability
   and is itself a finding worth reporting.

**A prediction worth recording before the measurement, in the spirit of
[stage 1](01_stage1_deployment_contract.md)'s pre-registration:** the degradation may be
small. On the original's authoritative run, SHAP put NWP forecast columns in **all of the
top six**, with the top one (`nwp_icon_global_wind_speed_100m_d1`, 0.4780) roughly six times
the best reanalysis feature below them (`wind_dir_850hpa_cos`, 0.0804 at rank 7);
`wind_speed_100m` itself sat at rank 13. If most of the signal is in the forecasts,
`operational` may lose little. **But SHAP importance is not the same as ablation loss, so
this is a hypothesis — and the entire point of this stage is that it is currently
untested.** Write the prediction down, then measure it, then report whether you were right.

### ✅ Measured 2026-08-12 — the prediction was right, and then some

24 h horizon, 4 rolling-origin folds ([results/03](results/03_operational_vs_study.md)):

| Model | `study` (94 feats) | `operational` (55 feats) | Difference |
|:---|---:|---:|---:|
| `xgboost` | 0.8037 | **0.8009** | **−0.0028** |
| `lightgbm` | 0.8086 | **0.8032** | −0.0054 |
| `ridge` | 0.8422 | **0.8383** | −0.0039 |

> **The degradation is negative.** Removing every one of the 39 latency-blocked reanalysis
> features makes the model marginally *better*, not worse.

The classical models are unchanged by construction — they only ever read NWP and calendar
columns. All three models that *can* see reanalysis improve slightly without it: once a
day-ahead forecast of the target hour is present, an observation from 24 hours earlier is
largely redundant, and 39 redundant partly-collinear columns are a small liability.

**What this settles and what it does not.** The deployability blocker this stage was written
around is **real in principle and free in practice at this site** — an operational system
running on forecasts alone loses no accuracy. It does not make the system deployable: there
is still no wind farm, no SCADA, no serving path and no monitoring, and §8.4's lead-time
question below remains open. **Deep models were not run on the operational set** — phase B
covers `study` only.

The prediction is left above **verbatim**, including its hedge, because a pre-registration
that gets quietly rewritten once the answer is known is worth nothing.

## 8.4 Verify the lead time empirically

Related, and also never resolved in the original: the forecast columns came from a provider
API using a fixed **day** offset (`previous_day1`), so the effective lead varies through the
target day rather than being a constant 24 hours. It was documented as "~24 h" and behaves
consistently with that — skill decays d1 → d2 → d3 — but was never verified hour by hour
([`12` item 9](../docs/12_next_steps_v2.md)).

For a deployment rehearsal this must be pinned down:

- Measure the actual issue-to-valid interval for every hour of the target day.
- Decide whether to restrict to a fixed issue cycle (e.g. always the 00Z run), which is how
  an operator would work.
- Report the lead as a **range**, not a single number, if it varies.

## 8.5 The rest of operational readiness

Even with a good `operational` score, a deployable system needs:

- **A serving path.** A scheduled job that fetches forecasts, runs the model, and writes
  output — not a notebook.
- **Input monitoring.** What happens when the provider is down, returns nulls, or changes
  units? The original had an SSL outage silently drop three months of data because the retry
  budget was ~3 minutes ([`11 §10`](../docs/11_nwp_forecast_features.md)).
- **Drift monitoring.** Rolling error against the baselines; alert when the model stops
  beating persistence.
- **A retraining schedule**, decided in advance.
- **A fallback.** When inputs are missing, degrade to climatology explicitly rather than
  producing an unmarked bad forecast.

None of these are research. All of them are the difference between a result and a system.

## Exit gate

You may not proceed to stage 9 until:

1. `study` and `operational` are both trained, evaluated under the identical protocol, and
   reported **side by side**.
2. The degradation is quantified and stated in one sentence.
3. The effective lead time is measured hour-by-hour, not assumed.
4. Every headline number in every document is labelled `study` or `operational`.

**How this gate can fail:** a single unlabelled headline number. If a reader cannot tell
which model produced it, the label is the finding that is missing.

---

Previous: [Stage 7](07_stage7_power_and_uncertainty.md) ·
Next: [Stage 9 — Write-up and hygiene](09_stage9_writeup_and_hygiene.md)
