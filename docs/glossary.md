# Glossary — plain-English definitions

Every term the other documents use without explaining. If a doc says something like
"nMAE 11.25% against a 24 h persistence baseline", this page tells you what that means.

Ordered by topic, not alphabetically, so you can read it top-to-bottom as an introduction.

---

## How to read the numbers

**Read this before any other document.** These docs are full of bare figures like
"1.475 → 1.20" or "test MAE 1.114". This section explains, once, what they mean.

### Stage 1 figures — wind speed

Unless a document says otherwise, a Stage 1 number is a **test MAE in metres per second
(m/s)**. It answers:

> *When the model predicts hub-height wind speed 24 hours in advance, how far off is it
> on a typical hour?*

**Lower is better.** "1.114" means the prediction is typically wrong by about 1.1 metres
per second — roughly the difference between a light breeze and a moderate one.
"Test" means it was scored on data the model never saw during training.

Anchor values, so any figure can be judged on sight (all m/s, 2020–2023 study period):

| Value (m/s) | What it is | Read it as |
|---:|:---|:---|
| **1.114** | Best model (LSTM) | the best achieved so far |
| **1.143** | Tuned XGBoost | statistically tied with the best |
| **1.282** | Climatology baseline — "the usual wind for this month and hour" | the bar a real model must clear |
| **1.399** | 24 h persistence baseline — "tomorrow = right now" | the standard naive comparison |
| **≈1.466** | Guessing the overall average every time | the floor; anything worse is broken |

So a figure near 1.1 is good, near 1.4 means no better than a naive rule, and above 1.47
means something is wrong. The whole span between best and useless is only about 0.35 m/s
— which is itself a finding about how hard day-ahead wind forecasting is.

*Provenance:* the first four values are from the v4 table in
[`../RESULTS.md`](../RESULTS.md). The ≈1.466 floor was measured separately on the
2020–2023 test set (1.4658 m/s) and is the point where R² = 0 by definition; it is not a
baseline the pipeline reports. Note it is a *different, weaker* baseline from the 1.282
climatology, which uses the average for each month and hour rather than one overall mean.

### Stage 2 figures — electrical power

A Stage 2 number is usually **nMAE** or **nRMSE**, a **percentage of the turbine's
3.05 MW maximum output**. **Lower is better.**

| Value | What it is |
|---:|:---|
| **~8.8%** | Published benchmark from the paper this design follows |
| **11.25%** | Best achieved here |
| **14.65%** | No machine learning at all — raw weather through the power curve |

Percentages are used rather than watts so results can be compared against studies using
different turbine sizes.

### R² (R-squared)

The share of the variation the model explains. **Higher is better**, unlike everything
above.

- **1.0** = perfect · **0.0** = no better than always guessing the average ·
  **negative** = worse than that

Best here is **0.390**, meaning about 39% of the hour-to-hour variation is captured.
Negative values appear in early results and are not typos.

### Numbers with no unit

Some quantities are ratios and correctly have no unit:

- **Train/test ratio** (this project: **1.16**) — training error vs test error. **1.0**
  means the model performs equally on data it has and has not seen. Much larger means it
  is memorising rather than learning. 1.16 is healthy.
- **Correlation** (e.g. **0.571**) — how strongly two things move together, from 1.0
  (perfectly in step) through 0 (unrelated) to −1.0 (perfectly opposed). Hub wind now vs
  hub wind in 24 hours is 0.571: related, but with much left unexplained.
- **Training loss** (e.g. 0.42 vs 0.008) — an internal score used *during* training only.
  It is **not comparable between models** unless they optimise the same objective, and a
  case where they did not is discussed in [doc 10](10_v4_deep_model_retraining.md).

### Two warnings

1. **Superseded figures.** All deep-model numbers from v2 and v3 were measured with a bug
   and are invalid. They are kept for provenance and marked where they appear. Do not
   quote them.
2. **v5 uses a different date range.** The in-progress work covers 2024-03 → 2025-12,
   while everything above is 2020–2023. Numbers from the two periods are **not**
   comparable until baselines are re-run on the newer one.

---

## The problem being solved

**Wind power forecasting** — predicting how much electricity a wind turbine will generate
at some point in the future. Grid operators need this because electricity supply must
match demand minute by minute, and wind is variable.

**Day-ahead forecast** — a prediction for 24 hours from now. That is the horizon this
project targets, because day-ahead is when electricity markets and grid schedules are set.

**Forecast horizon** (also **lead time**) — how far ahead you are predicting. Written as
`t + h`: if *t* is now and *h* is 24 hours, you are predicting tomorrow at this time.
Longer horizons are harder.

**Valid time vs issue time** — a forecast has two clocks. The **issue time** is when the
forecast was made; the **valid time** is the moment it describes. A forecast issued
Monday 00:00 for Tuesday 00:00 has a 24 h lead. Mixing these two up is the single most
common way to accidentally cheat — see *leakage*.

**Nowcast** — predicting the present rather than the future. In this project it means
"convert the weather at time *t* into wind speed at time *t*", which is nearly trivial and
is used only as a diagnostic.

**SCADA** — Supervisory Control And Data Acquisition. The telemetry a real wind farm
records: rotor speed, blade pitch, actual power output. **Ghana has none published**,
which is the constraint that shaped this entire project.

---

## Where the weather data comes from

**Reanalysis** — a reconstruction of past weather. Scientists take historical observations
(weather stations, satellites, balloons, aircraft) and feed them into a weather model to
produce a complete, gap-free record of what the atmosphere *was* doing, on a regular grid,
every hour. It is the best available answer to "what was the weather at this spot in
2021?" — but it is a reconstruction, not a direct measurement.

**ERA5** — the specific reanalysis this project uses, produced by ECMWF (the European
Centre for Medium-Range Weather Forecasts). Hourly, on a 0.25° grid (~25 km squares),
covering 1940–present. This is the project's primary data source and its **truth**.

**NASA POWER** — a second, coarser reanalysis-derived dataset. Used only as a fallback and
a cross-check.

**NWP — Numerical Weather Prediction** — an actual *forecast* model, the thing that
produces the weather forecast on your phone. Key distinction from reanalysis: NWP looks
**forward** from now; reanalysis looks **backward** using observations that had not yet
happened at forecast time. **This project had no NWP input until v5**, which turned out to
be its main limitation.

**ECMWF IFS / GFS / ICON** — three of the world's major global NWP models (European,
American, German respectively). v5 uses forecasts from all three.

**Grid cell** — reanalysis data is not continuous; it comes as squares. At 0.25°, one cell
is roughly 25 km across. Everything inside a cell gets the same value, so fine local
terrain effects are averaged away.

**Pressure levels / 850 hPa / 500 hPa** — ways of describing altitude by air pressure
instead of metres. 850 hPa is roughly 1.5 km up; 500 hPa roughly 5.9 km. Winds high up
drive winds at the surface, so they are useful predictors.

**Geopotential** — a measure of the energy needed to lift air to a given height. Units are
m²/s², **not** metres — divide by 9.80665 to get height in metres. A frequent source of
confusion.

**Boundary layer height** — how deep the "churned up" layer of atmosphere near the ground
is. A deeper layer mixes fast air from above down to turbine height.

**Roughness length** — how rough the ground is (grass vs forest vs city), which controls
how quickly wind speed increases with height.

**Diurnal cycle** — a daily repeating pattern. At a coastal site like Anloga, land and sea
heat at different rates, producing a **sea breeze** that makes wind speed rise and fall on
a predictable daily rhythm.

**CDS / GRIB / NetCDF / parquet** — plumbing. CDS is the Copernicus portal you download
ERA5 from. GRIB and NetCDF are weather-data file formats. Parquet is a compact table
format used for the cleaned-up result.

---

## Turbines and turning wind into power

**Hub height** — how high the centre of the rotor sits. 100 m here. Wind speed matters *at
that height*, not at ground level.

**Wind shear** — wind gets faster the higher you go. Converting a 10 m wind speed to a
100 m one uses a **log law** or **Hellman** formula, both of which need the roughness
length.

**Power curve** — the manufacturer's table converting wind speed into electrical output
for a specific turbine. It has three landmarks:
- **Cut-in** (2.0 m/s here) — below this, too slow to generate anything.
- **Rated** (12.0 m/s) — at and above this, the turbine produces its maximum.
- **Cut-out** (25.5 m/s) — above this the turbine shuts down to avoid damage.

Between cut-in and rated, power rises roughly with the **cube** of wind speed — so a 10%
wind-speed error becomes roughly a 30% power error. This is why the project measures
*error propagation* explicitly.

**Nameplate / rated capacity** — the turbine's maximum output. 3.05 MW for the Enercon
E-101/3050 used here.

**Air-density correction** — hot air is thinner and carries less energy, so the same wind
speed yields less power. Ghana is hot, so the pipeline corrects for this using the ideal
gas law.

**windpowerlib** — the open-source Python library that holds the power curves and does
this physics.

**oedb** — the OpenEnergy DataBase, where the turbine power curve comes from.

---

## Machine learning terms

**Feature** — an input to the model (e.g. wind speed at 100 m, hour of day). **Target** or
**label** — what it is trying to predict (here: hub-height wind speed 24 h ahead).

**Lag feature** — the value of something a few hours ago. `wind_speed_hub_lag12` is hub
wind speed 12 hours before now. **Rolling feature** — a running average or spread over a
recent window.

**Training / test split** — the model learns on one slice of history (**train**) and is
scored on a slice it never saw (**test**). Here the split is chronological — the last 20%
of time is the test set — because shuffling time series randomly lets the model peek at
the future.

**Leakage** — accidentally giving the model information it would not have had in reality.
Produces excellent scores that collapse in the real world. This project hit a leakage-
adjacent bug twice; see doc 10.

**Cross-validation (CV) / TimeSeriesSplit** — repeating the train/test exercise on several
successive slices to check the result was not luck. `TimeSeriesSplit` always keeps
training data *before* test data in time.

**Overfitting** — memorising the training data instead of learning the pattern. The
telltale sign is a large gap between training error and test error. A **train/test ratio**
near 1.0 means no overfitting; this project measures 1.16.

**Baseline** — a deliberately simple predictor that a real model must beat to justify
itself:
- **Persistence** — "tomorrow will be like right now." Trivial, and surprisingly hard to
  beat at short horizons.
- **Climatology** — "tomorrow will be like the historical average for this month and hour."

**Hyperparameter** — a setting you choose before training (tree depth, learning rate), as
opposed to something the model learns. **Optuna** is the library used here to search for
good ones automatically.

**SHAP** — a method for attributing a prediction to individual features, used to rank
which inputs matter. ⚠️ SHAP rank is *not* the same as predictive contribution — this
project found features ranked highly by SHAP that changed accuracy by almost nothing.

**Autocorrelation** — how similar a series is to itself some hours later. Hub wind speed
has a 24 h autocorrelation of 0.571 here, which mathematically caps how well *any* model
can do using only present-time information.

**Bias correction / MOS (Model Output Statistics)** — learning the systematic errors of a
weather model and cancelling them out. Adding NWP forecasts as ML features (v5) is
essentially this.

---

## The model types compared

**XGBoost** — gradient-boosted decision trees. Builds many small "if/then" trees, each
correcting the previous one's mistakes. Strong on tabular data and the project's long-time
baseline.

**LSTM** — a recurrent neural network that processes a sequence step by step, carrying a
memory. **BlockRNN** is the darts implementation used here.

**TFT (Temporal Fusion Transformer)** — a large attention-based neural network designed
for forecasting.

**N-BEATS** — a deep network that decomposes a series into trend and seasonal building
blocks.

**DLinear** — a deliberately simple linear model, included because recent research showed
simple baselines often match complex ones.

**darts** — the Python library providing all four sequence models above.

**Backtest / historical forecasts** — replaying the model across history to see how it
would have performed. The function that does this returns results indexed by *valid time*,
which is what caused the v3 bug.

---

## Accuracy metrics

All measure the gap between prediction and truth. Lower is better except R².

**MAE — Mean Absolute Error** — the average size of the miss, ignoring direction. In the
same units as the thing predicted, so MAE 1.114 m/s means "typically wrong by about 1.1
metres per second". The easiest metric to interpret.

**RMSE — Root Mean Squared Error** — like MAE but squares errors first, so it punishes
large misses harder. Always ≥ MAE.

**R² (R-squared)** — the fraction of the variation the model explains, from 1.0 (perfect)
down through 0 (no better than always guessing the average) to negative (worse than that).
R² 0.390 means about 39% of the variation is captured.

**MAPE** — Mean Absolute Percentage Error. Error as a percentage of the true value.
Misleading when true values are near zero, which happens often with wind.

**nMAE / nRMSE (normalised)** — the error expressed as a percentage of the turbine's rated
capacity. nMAE 11.25% means the average power miss is about 11% of 3.05 MW. This is the
standard in wind-power literature because it lets different turbine sizes be compared.

---

## This project's own vocabulary

**Stage 1** — the machine-learning step: weather → hub-height wind speed at t + 24 h.

**Stage 2** — the physics step: wind speed → electrical power via the power curve. No
learning involved; entirely deterministic.

**Two-stage** — the main architecture: Stage 1 then Stage 2.

**physics_only** — control variant: skip the ML, push *observed* wind through the curve.
Answers "does the ML add anything?"

**direct** — control variant: one ML model straight from weather to power, skipping the
intermediate wind-speed step.

**Reference truth** — because no real SCADA exists, "correct" power is computed by pushing
the *observed* ERA5 wind through the *same* power curve. This validates the forecasting
step, **not** whether the pipeline predicts a real turbine's actual output.

**v1 → v5** — the project's iterations. v1 was the initial build; v4 fixed a major
measurement bug; v5 (current) adds NWP forecast features. See
[06_iteration_history.md](06_iteration_history.md).
