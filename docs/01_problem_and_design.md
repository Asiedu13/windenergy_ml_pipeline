# 01 — Problem & Design

> **What this document is:** the *why* behind every later decision — what problem is being
> solved, which architecture was chosen, and what was rejected.
> **Why read it first:** every other doc assumes these choices. Unfamiliar terms are
> defined in [glossary.md](glossary.md).

## The problem

> **What:** Ghana has good wind resources but no operating wind farm publishing
> operational data.
> **Why it matters:** you normally train a power-forecasting model on a real turbine's
> recorded output. With none available, the whole project has to be built a different
> way — this single constraint explains most of what follows.

Ghana has significant untapped wind resources, particularly along the south-eastern
coast where prevailing south-westerlies and the diurnal sea-breeze cycle produce sustained
moderate wind speeds. Yet Ghana has **no operating utility-scale wind farm publishing
SCADA data** — no turbine rotor-speed telemetry, no pitch angles, no real-time power
output records. This is a fundamental constraint on the modelling approach.

The goal of this project is to build a reproducible ML pipeline that forecasts what a
day-ahead wind power output would look like for a chosen Ghanaian site, using only
publicly available reanalysis weather data plus a manufacturer's turbine specification.
The output supports resource assessment and grid-integration planning studies where
observed SCADA is unavailable.

## Why the two-stage physical/indirect architecture

> **What:** the core design — first predict *wind speed* with machine learning, then
> convert that wind speed to *power* using the turbine manufacturer's published curve.
> **Why it matters:** predicting power directly needs recorded power to learn from, and
> none exists. Splitting the job means the learned half only needs weather data, and the
> physics half needs no data at all.

The Polish Frontiers 2025 paper (which inspired this work) uses turbine rotor speed as
a Stage-1 feature to predict power output. **We cannot do this** — no rotor speed
data exists for any Ghanaian turbine. So we adopted the alternative architecture from
**Donadio, Fang & Porté-Agel (Energies 2021) Model 2** and Xu, Pan et al. (IEEE TSTE 2015):
the physical/indirect two-stage approach.

**Stage 1 — ML:** predict *hub-height wind speed* at time t + h from historical
reanalysis weather variables (surface winds, temperature, pressure, upper-air fields,
calendar features). The model is trained on labels derived from the same reanalysis, so
there is no dependence on missing SCADA.

**Stage 2 — Simulator:** convert predicted wind speed to power using
[`windpowerlib`](https://github.com/wind-python/windpowerlib)'s manufacturer power curve
for a chosen turbine, with an air-density correction from the ideal gas law. This step
is deterministic — no training, no learned weights.

### Why decouple the stages

> **What:** the two stages share no code and can each run on their own.
> **Why it matters:** you can swap the turbine or the site with a one-line config change,
> and when real measurements eventually exist they can feed Stage 2 directly without
> rewriting anything.

The two stages are fully decoupled in code. Stage 2 can run standalone on any wind-speed
series (e.g., a met-mast reading, an operational NWP forecast, a scenario input). Stage
1 can be replaced with any better wind-speed forecaster in the future without touching
the power conversion.

This matters because:
- **Interpretability:** a wind-speed error is easier to reason about than a raw power
  error; you can inspect *where* Stage 1 went wrong before blaming the whole pipeline.
- **Modularity:** swapping the turbine (E-101/3050 → V112/3000) is a one-line config
  change with zero retraining. Swapping the site works the same way.
- **Portability:** the same Stage 2 code will accept live met-mast wind speed once
  observational infrastructure comes online — no re-architecting required.

### Reference "truth" and its limits

> **What:** with no real turbine to check against, "correct" power is manufactured by
> pushing the *observed* wind speed through the *same* power curve the prediction uses.
> **Why it matters:** this is the project's single biggest caveat. It fairly tests
> "did we predict the wind well?" but cannot test "would a real turbine have produced
> this much?" — the same curve sits on both sides of the comparison.

The pipeline has no real SCADA to validate against, so it constructs a **reference
truth** by taking the *actual observed* ERA5 hub-height wind speed and running it
through the same Stage 2 power curve used for prediction. All metrics (nMAE, nRMSE, R²
on power) are computed against this derived reference.

This means the pipeline validates the **forecasting step** — does the ML model predict
wind speed well enough that the derived power series is close to the physics-derived
reference — but it does **not** validate absolute physical yield. A future met-mast
installation or first-of-a-kind SCADA feed at Anloga would tighten this validation.

The auto-generated report is explicit about this caveat in its opening line.

## Alternatives considered and rejected

> **What:** the other five designs that were weighed, and why each lost.
> **Why it matters:** an examiner's first question is "why not do it the obvious way?"
> Two of these (`direct` and `physics_only`) are still built and reported every run, so
> the trade-off is measured rather than asserted.

**Direct-model architecture** (`weather → power` in one ML step). This is the
control variant in the pipeline (`variant: direct`), and is reported alongside the
two-stage variant so the trade-off is visible. It's competitive on point metrics
(within 0.4 percentage points nMAE) but sacrifices interpretability, modularity, and
the ability to reuse Stage 2 for any non-forecasting task. The two-stage approach is
preferred because those three properties are worth more than a fraction of a percentage
point in accuracy.

**NWP-only physics baseline** (`variant: physics_only`) — take the raw ERA5 hub wind
speed and push it through the curve, no ML at all. This is included as the "does the
ML actually add anything?" baseline. Under our v4 result it sits at 14.65% nMAE vs
11.25% for two-stage, confirming Stage 1 ML adds value.

**Turbine-SCADA-based supervised regression** (the Polish paper's approach). Impossible
for Ghana — no SCADA. Explicitly rejected up front.

**Multi-site pooled model** — train a single model on data from several Ghanaian
sites at once, sharing weight for common patterns. Deferred: only one site is being
modelled in this study, and pooling would require identifying and downloading data for
each candidate site (multi-week effort).

**Numerical weather prediction (NWP) forecast as input.** ECMWF, GFS or ICON
short-range forecasts could serve as Stage 1 input directly (bypassing the ML entirely).
Deferred because operational NWP fetch is a separate infrastructure problem, and the
whole point of ML here is to add value on top of reanalysis where forecast skill is
already exhausted.

## Non-negotiable design principles

> **What:** six rules the code is not allowed to break.
> **Why it matters:** most are guards against fooling yourself — never shuffling time
> series, always beating the trivial baselines, seeding all randomness. Time-series
> machine learning makes it very easy to produce impressive numbers that are simply
> wrong, and rules 4 and 5 exist specifically to catch that.

These constraints appear in the plan and shape every decision downstream:

1. **Site-specific model.** A model trained at Anloga will not generalise to Ada Foah
   or Takoradi. Retraining on a new lat/lon must be a **one-line config change**.
   Reflected in `config.yaml`'s `site:` block and in the caching filename convention
   (`era5_{site.name}_...`).

2. **Stage 1 and Stage 2 fully decoupled.** Stage 2 must run standalone on any
   wind-speed series with no dependency on Stage 1 internals. Enforced by module
   boundaries and by [tests/test_stage2.py](../tests/test_stage2.py), which never
   imports from Stage 1.

3. **Error propagation reported, not hidden.** Wind-speed error goes through the
   ~cubic power curve nonlinearly. This is the central weakness of the two-stage
   architecture and the pipeline explicitly measures it (binned |power error| vs
   |wind-speed error|, saved to `outputs/figures/error_propagation.png`).

4. **Never shuffle time series in CV.** Use `sklearn.model_selection.TimeSeriesSplit`
   (expanding-window splits) exclusively. Random-shuffle CV leaks future information
   into training folds and inflates metrics.

5. **Always beat persistence and climatology before reporting any ML metric.**
   Persistence (predict `ŷ_{t+h} = y_t`) and climatology (predict the training-set
   mean by month × hour-of-day) are the two mandatory baselines. If the ML model
   doesn't beat both, the report flags it with **NO — flag** rather than hiding the
   result.

6. **All randomness seeded; all outputs reproducible from `config.yaml`.**
   `seed: 42` propagates through numpy, XGBoost, Optuna's TPE sampler, and darts's
   PyTorch backend. Same config in → same numbers out.

## Site choice: Anloga

> **What:** why this particular spot on Ghana's south-eastern coast (5.79 °N, 0.90 °E).
> **Why it matters:** the model is site-specific by design — it learns Anloga's daily
> sea-breeze rhythm and will not transfer to another location without retraining.

**Anloga**, on Ghana's south-eastern coast in Volta Region (5.79 °N, 0.90 °E), was
chosen because:

- Recent Ghanaian wind-resource assessments identify it as one of the most viable
  onshore locations (GEDAP report, various academic surveys).
- Coastal south-easterlies + sea-breeze cycle produce a strong diurnal signal that
  ML should be able to learn.
- Prime meridian passes through neighbouring Tema — the longitude is nearly 0°,
  making coordinate arithmetic clean and reducing the chance of sign errors in
  bbox construction.

The site coordinates and hub height (100 m) are set in
[`config.yaml`](../config.yaml)'s `site:` block. Changing them re-triggers a fresh
ERA5 fetch under a new cache filename.

## Turbine choice: Enercon E-101/3050

> **What:** the specific 3.05 MW turbine model whose power curve Stage 2 uses.
> **Why it matters:** all the headline percentage errors (nMAE, nRMSE) are expressed as a
> share of this machine's 3.05 MW rating, so the turbine choice sets the scale of every
> reported number.

The Enercon E-101/3050 (**3.05 MW** nameplate, 101 m rotor, 100 m hub) was chosen
because:

- It's a widely-used utility-class turbine appropriate to the moderate wind regime
  along Ghana's coast (annual mean ≈ 6–7 m/s at 100 m).
- Its power curve is available in the OpenEnergy DataBase (`oedb`) turbine library
  that `windpowerlib` uses out-of-the-box.
- Cut-in 2.0 m/s, rated 12.0 m/s, cut-out 25.5 m/s (extracted programmatically from
  the curve) are reasonable for the site's wind climatology.

The turbine is selected via `config.yaml`'s `turbine:` block; switching to a different
OEM/model is a one-line change.

## Summary

> **What:** the argument of this document in one paragraph.
> **Why it matters:** it is the paragraph to reuse when someone asks "so what did you
> actually build, and why that way?"

The design is a deliberate response to a hard constraint (no SCADA in Ghana) using an
established academic architecture (Donadio et al. 2021). Every property listed under
"non-negotiable design principles" traces back to either the constraint or the
literature. The trade-offs versus a direct model or an NWP-input model have been
considered explicitly. The rest of these docs describe how the design was implemented.
