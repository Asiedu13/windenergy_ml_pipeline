# Stage 7 — Power conversion and uncertainty

> **What this stage is:** turning forecast wind speed into forecast power, and — the part
> the original never did — attaching honest uncertainty to it.
> **Why it comes seventh:** power is a *derived* quantity here. Without SCADA, every power
> metric measures the pipeline against its own physics model, so it must be reported with
> that boundary stated rather than presented as physical yield.
> **Effort:** 2–3 days, most of it on the uncertainty work.

## 7.1 State the truth boundary before the first power number

> **What:** what the power metrics actually measure.
> **Why it must come first:** the original reported `nMAE 7.48%` against a published
> benchmark of ~8.8%, which reads like "we beat the literature". It is defensible only with
> its boundary attached.

Reference "truth" for power is **observed hub wind → manufacturer power curve →
density-corrected**. There is no metered generation in Ghana to compare against. So:

- ✅ These metrics validate the **forecasting step**: given that the physics is what it is,
  how well is tomorrow predicted?
- ❌ They do **not** validate physical yield. A wrong power curve, a wrong turbine choice, or
  wake and availability losses are all invisible to this evaluation.

Write that sentence into the report generator itself, so it appears on every report whether
or not anyone remembers to add it. The original did exactly this, and it is one of the
practices worth keeping unchanged.

## 7.2 Keep the two stages decoupled

The two-stage design — ML predicts wind speed, a simulator converts it to power — was a good
call and traces to a real constraint. Its advantages are **interpretability and
modularity**, not accuracy: in the original, `two_stage` (7.48% nMAE) and `direct`
(7.43%) were within 0.05 percentage points and **swapped order** between truth sources.

**So do not argue for two-stage on accuracy grounds.** Argue for it on the grounds that you
can inspect the wind forecast separately, swap the turbine without retraining, and explain
the failure mode. Those are real and they hold.

Enforce the decoupling in code: the power module must run standalone on any wind-speed
series, with no import from the ML side, and its tests must never import from Stage 1.

## 7.3 Report error propagation, and re-run it as accuracy improves

Wind error passes through a roughly **cubic** mid-section of the power curve, so it is
amplified nonlinearly. This is the central weakness of the two-stage architecture and it
should be measured, not hidden — bin absolute power error against absolute wind-speed error
and plot it.

**The part the original missed:** as Stage 1 error shrinks, the *relative* contribution of
that amplification grows, so the propagation analysis measures a different regime and needs
regenerating. The original's figure was computed when Stage 1 error was ~1.1 m/s and never
re-run at 0.73 ([`12` item 8](../docs/12_next_steps_v2.md)).

## 7.4 Uncertainty from the start, not as a follow-up

> **What:** predicting a distribution rather than a single number.
> **Why it belongs here rather than in a "future work" list:** a grid operator cannot use a
> point forecast. "Tomorrow at 14:00 you will generate 1.8 MW" is unusable; "1.2–2.4 MW with
> 80% confidence" supports a decision. The original left this in a next-steps document
> through five iterations.

**The signal is free and already on disk.** Multiple NWP models are fetched — in the
original, ECMWF, GFS and ICON. Their **disagreement on a given hour is a physically
meaningful confidence indicator**: when three independent models agree, the atmosphere is
predictable; when they diverge, it isn't.

Concretely:

1. Add per-hour ensemble **spread** (sd across models) as a feature.
2. Train quantile regression for P10 / P50 / P90 — gradient boosting supports a pinball
   objective directly.
3. **Check calibration**, which is the step that makes it meaningful: over the test set,
   ~10% of observations should fall below P10 and ~10% above P90. An uncalibrated interval
   is worse than no interval, because it invites false confidence.
4. Report sharpness alongside calibration — a P10–P90 band spanning the full turbine range
   is perfectly calibrated and perfectly useless.

**One caution carried from the original.** Quantile objectives are not a free addition: when
TFT was trained on a 17-quantile loss while the other models used MSE, it lost **0.380 m/s**
of point-forecast accuracy and was ranked last for it ([register C1](00_mistake_register.md)).
If you want both a point forecast and quantiles, train them as **separate models with
separate objectives**, and never compare a quantile-trained model against MSE-trained ones
on a point metric.

## 7.5 Comparing against published benchmarks

Only with the differences stated. The original's final comparison — 7.48% nMAE against a
~8.8% reference — is reasonable, but only because it eventually established that its window
was of comparable difficulty (persistence 1.31 vs 1.399 m/s). An earlier version of the same
comparison used a 12-month subset where persistence was 1.09, i.e. a markedly easier
problem, and the claim would have been inflated.

> ⏱️ **Every figure quoted on this page is a 24 h horizon result** — `7.48% nMAE`, the
> `~8.8%` reference, and the `0.380 m/s` quantile-objective cost alike. The project has never
> run any other horizon, so none of these transfer to a different lead time.

**The checklist before any benchmark comparison:**

- Same horizon? Same metric definition and same rated power for normalisation?
- Comparable window difficulty — quote the persistence baseline for both.
- Same kind of truth: metered SCADA, or physics-derived like yours?
- Multiple seeds on your side?

If any answer is no, the phrasing is *"comparable to"*, not *"better than"*.

## Exit gate

You may not proceed to stage 8 until:

1. The truth boundary is stated in the report generator, not just in a document.
2. Error propagation is measured **at the current Stage 1 accuracy**, not an older one.
3. P10/P50/P90 exist, with a calibration check and a sharpness number.
4. Any benchmark comparison passes the four-point checklist above.

**How this gate can fail:** quantiles that are not calibration-checked, or a benchmark
comparison whose window difficulty was never quoted.

---

Previous: [Stage 6](06_stage6_modelling.md) ·
Next: [Stage 8 — Deployment rehearsal](08_stage8_deployment_rehearsal.md)
