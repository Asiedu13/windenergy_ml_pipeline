# 07 — Findings & Analysis

This document steps back from the mechanics and interprets what the numbers
actually mean: what worked, what didn't, and how this project sits relative to
published benchmarks. Everything here is intended to be quotable in a thesis
chapter or a technical report.

> **Rewritten for v4.** An earlier version of this document argued that the deep
> models underperformed, and gave five reasons. That analysis was explaining a
> **measurement artefact** — the sequence models were being scored 24 hours out
> of phase. The corrected results invert the conclusion. The bug, the fix and the
> full re-run are in [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md);
> the original argument is preserved verbatim in
> [09_why_deep_learning_underperformed.md](09_why_deep_learning_underperformed.md)
> and the iteration-by-iteration record in
> [06_iteration_history.md](06_iteration_history.md).

> 📐 **Reading the numbers.** Unless stated otherwise, a figure like `1.114` is a **test
> MAE in m/s** — how far off the model typically is when predicting hub-height wind speed
> 24 hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399, guessing the average 1.466. R² runs the other way (higher is
> better; 1.0 perfect, 0 = no better than guessing the average, negative = worse).
> Percentages like `11.25%` are power error as a share of the turbine's 3.05 MW maximum.
> Full explanation: [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** what the numbers *mean* — the interpretation layer, written
> to be quoted in a thesis.
> **Why it is the most important doc here:** the others describe what was built; this one
> states what may honestly be claimed about it, and what may not. If you read only two
> files, make them this and [glossary.md](glossary.md).

## Summary of findings

> **What:** the five headline conclusions, each in a sentence.
> **Why start here:** everything below is the evidence for these five. If you are short of
> time, read this section and the "Defensible conclusions" section at the end.

1. **The two-stage architecture is validated.** In every iteration where Stage 1
   beats persistence, the two-stage variant also beats physics-only, closing the
   loop from "ML wind-speed prediction → curve → power" as a viable design.
2. **All five architectures beat persistence, and they are near-equivalent.**
   LSTM 1.114, XGBoost 1.143, N-BEATS 1.157, TFT 1.171, DLinear 1.215 m/s against
   a 24 h persistence baseline of 1.399. The spread across architectures
   (0.101 m/s) is *smaller* than the gap from the weakest of them to persistence
   (0.184 m/s).
3. **Architecture choice was not the lever; measurement correctness was.** Two
   configuration details — a 24-hour index offset in the deep-model scorer and one
   library default (`darts.TFTModel`'s 17-quantile objective) — moved the models
   further than any architectural difference between them.
4. **We sit ~2.5 percentage points above the Energies 2021 benchmark** (11.25% vs
   ~8.8% nMAE). The ceiling is **information-limited, not sample-limited** — see
   the measurements in the next section, which refute the "more data" reading
   this document previously gave.
5. **Feature engineering still matters, but not as the deciding factor.** The
   biggest metric jump from a *feature* change came from adding upper-air winds
   and spatial derivatives (v2). Handing the sequence models the same engineered
   lag features (v4, rec #2) was measurably neutral — see the ablation below.

## What the ceiling actually is (measured 2026-08-10)

> **What:** four experiments establishing *why* the model stopped improving.
> **Why this section exists:** for four iterations these docs asserted the limit was "not
> enough data" — a plausible claim that had never been tested. When it finally was, it
> turned out to be wrong. The limit is not the *amount* of data but the *kind*: the model
> was only ever shown weather that had already happened, never a forecast of the weather
> to come.
> **Why that distinction matters practically:** it redirected the project entirely.
> Downloading more history was abandoned mid-run; obtaining forecasts became the work.

Through v4 this document asserted the residual gap was "data-limited" and implied more
years of ERA5 would close it. That was never measured. It has now been, and it is wrong.

### The learning curve is flat

> **What:** train the same model on increasing amounts of history and watch whether it
> keeps improving. It stops improving after about 1.6 years.
> **Why this settles the "more data" question:** if a model were data-starved, feeding it
> more would help. Doubling the history from 1.6 to 3.2 years improved test MAE by only
> 1.1% (0.012 m/s). The **train/test ratio of 1.16** confirms it from the other direction:
> that ratio compares error on data the model trained on against error on data it never
> saw, so 1.0 would mean no memorising at all and a large value would mean heavy
> memorising. At 1.16 there is almost nothing for extra data to fix.
> **How to read the table:** "Train rows" is how many hours of history the model saw;
> "Test MAE" is its typical error in m/s on data it never saw. Lower is better.

XGBoost with the v3 Optuna hyperparameters, fixed test set, trained on the most-recent
fraction of the training slice:

| Train rows | Years | Test MAE | Δ | Train/test gap |
|---:|---:|---:|---:|---:|
| 3,502 | 0.4 | 1.5017 | — | 2.36 |
| 7,005 | 0.8 | 1.1933 | −0.3085 | 1.52 |
| 14,011 | 1.6 | 1.1534 | −0.0399 | 1.25 |
| 21,016 | 2.4 | 1.1479 | −0.0055 | 1.19 |
| 28,022 | 3.2 | **1.1410** | −0.0069 | **1.16** |

It flattens after ~1.6 years. The final doubling of data bought **0.012 m/s (1.1%)**.
Extrapolating that rate, 3.2 → 12.8 years (two more doublings) is worth ~0.02 m/s, i.e.
~0.2 pp of nMAE against a 2.45 pp gap. **The train/test ratio of 1.16 is the decisive
number**: more data cures variance, and there is essentially no variance left to cure.

*(This is why the 10-year ERA5 backfill was halted at 2016-07 — see
[00_data_card.md §2](00_data_card.md).)*

### The features are saturated

> **What:** add feature groups one at a time and see how much each improves accuracy.
> **Why it matters:** the gains shrink to nothing. Calendar features help most; the twenty
> spatial columns added in v2 are worth 0.003 m/s — essentially nothing, despite one of
> them ranking second in the model's own importance scores.
> **The lesson worth carrying:** measure a feature's value by removing it, not by asking
> the model how much it liked it.

Adding feature groups to XGBoost, same split:

| Feature set | n | Test MAE |
|:---|---:|---:|
| lag/rolling only | 9 | 1.2564 |
| + calendar | 13 | 1.1760 |
| + surface weather | 25 | 1.1615 |
| + upper-air (850/500 hPa) | 33 | 1.1444 |
| + spatial derivatives | 53 | **1.1410** |

The 20 spatial columns are worth **0.003 m/s** — and half of them are degenerate anyway
(§[03_data_sources.md](03_data_sources.md)). Earlier claims that spatial derivatives
"pay off" rest on SHAP ranking, not on predictive contribution.

### Against a one-parameter baseline

> **What:** comparing the full pipeline against the simplest possible statistical model —
> a straight line fitted to a single input, current wind speed.
> **Why it is a humbling comparison:** the one-parameter line scores test MAE
> **1.2388 m/s**; the full system with 53 features, automated tuning and five
> architectures scores **1.114 m/s**. That is a real improvement of 0.125 m/s (about 10%,
> lower being better) — but modest against the effort.
> **What it reveals:** wind speed 24 hours from now is only **0.571** correlated with wind
> speed now. Correlation runs from 1.0 (moving perfectly in step) through 0 (unrelated),
> so 0.571 means the two are related but with a great deal left unexplained. That figure
> alone caps how well *any* model can do from present-time information — a mathematical
> ceiling, not a modelling failure.
> ⚠️ Note also that naive persistence (test MAE **1.3990 m/s**) is a weak opponent because
> it never shrinks toward the average. The fitted line is the fairer baseline.

| Predictor | Test MAE | R² |
|:---|---:|---:|
| Naive persistence `hub(t)` | 1.3990 | 0.043 |
| **OLS on `hub(t)` alone — 1 parameter** | **1.2388** | **0.267** |
| OLS on lag1 + calendar (5 params) | 1.2216 | 0.284 |
| OLS on all 53 features | 1.1591 | 0.348 |
| XGBoost, Optuna-tuned, 53 features | 1.1410 | 0.365 |
| LSTM (v4 best) | 1.114 | 0.390 |

The 24 h autocorrelation of hub wind speed is **0.571**, capping R² at **0.326** for
anything built on the state at time *t*. The full apparatus — 53 features, upper-air,
spatial derivatives, Optuna, five architectures — buys ~0.06 R² over a single-parameter
regression. Note also that naive persistence is a *weak* baseline because it does not
shrink toward the mean; properly scaled persistence (the OLS row) is much stronger, and
is the fairer comparator.

### The diagnosis

> **What:** the experiment that pinpoints exactly where the error comes from.
> **How it works:** ask the model to predict wind *now* instead of in 24 hours. It scores
> almost perfectly. Since the only thing that changed is the 24-hour gap, that gap must
> account for essentially all of the error.
> **Why it is decisive:** it rules out the alternatives. It is not the height conversion,
> not the feature engineering, not the model architecture — all of those are shared
> between the two runs. It is the passage of time and the atmosphere's tendency to forget.

Running the pipeline in `mode="nowcast"` gives MAE **0.015**, R² **0.9998**. That number
is near-tautological — hub wind speed is a deterministic function of `wind_speed_100m`,
which is a feature — but it localises the error precisely: **converting weather to hub
wind is free; 100% of the error is the 24-hour atmospheric gap.**

Every exogenous feature in `config.yaml` is the atmospheric state at time *t*. There is
no forecast anywhere in the input. In operational forecasting you would have an NWP
forecast *valid at t+24* available at time *t*; this pipeline never had one. That, not
sample count and not architecture, is the binding constraint — and it is what
[`config_nwp.yaml`](../config_nwp.yaml) and the Open-Meteo ingest now address.

**What more years of ERA5 *is* worth:** statistical defensibility, not accuracy. Threats
3 and 4 below (single contiguous test window, single seed/split) are real and a longer
record would let rolling-origin backtesting settle them. That is a different benefit from
the one previously claimed.

## Why the deep models *appeared* to underperform

> **What:** for two iterations the neural networks lost to a simple tree model, and an
> elaborate explanation was written for why. The explanation was wrong; the results were
> a measurement error.
> **Why this is the project's most valuable finding:** the wrong numbers were *believable*.
> They matched a well-known rule of thumb, they were internally consistent, and they were
> ordered sensibly by model size. Nothing looked broken. The subsections below separate
> what actually caused it from what merely sounded convincing.
> **The transferable warning:** a bug that nudges every result toward a plausible value is
> far more dangerous than one that produces obvious nonsense — the first survives review.

Through v3 this looked like the project's most interesting finding: four modern
sequence architectures, all losing to a gradient-boosted tree, consistent with a
well-known prior about small tabular datasets. It was not a finding. It was two
bugs and a library default.

### 1. The 24-hour alignment bug (the dominant cause)

> **What:** each neural network's predictions were compared against the wrong hour — 24
> hours out of step.
> **Why it produced such convincing wrong answers:** comparing wind now against wind
> tomorrow is *precisely the definition* of the persistence baseline. So every model was
> unknowingly re-measuring persistence, with its own noise added on top. That is why all
> four clustered just above the persistence score — not four independent results, but the
> same baseline measured four times.
> **The tell that was visible all along:** four completely different architectures landing
> within 0.2 m/s of persistence is not a finding, it is a fingerprint.

`historical_forecasts(..., last_points_only=True)` returns a series indexed by
*target* time τ — `hist[τ]` is the prediction of the wind speed **at** τ, made
from data up to `τ − 24`. The feature frame is indexed by *feature* time t, where
`y[t] = hub(t + 24)`. The v2/v3 code compared them directly, so it scored the
model's estimate of `hub(t)` against `hub(t + 24)`.

That error is approximately `|hub(t) − hub(t+24)|` — which *is the definition of
24-hour persistence error*. Every deep model was therefore pinned to the
persistence floor (1.440 m/s), with its own error stacked on top:

| Model | v3 MAE (misaligned) | v4 MAE (aligned) |
|:---|---:|---:|
| LSTM | 1.475 | **1.114** |
| N-BEATS | 1.425 | **1.157** |
| DLinear | 1.519 | **1.215** |
| TFT | 1.643 | **1.171** (also needed #2) |

XGBoost was unaffected — the tree path never went through darts — which is why
it "won" and why it acts as the control confirming the movement is real.

**This is the instructive part, and it is worth a paragraph in the write-up:** a
bug that pushes every model toward the same plausible value is far more dangerous
than one that produces obvious garbage. The v3 table was internally consistent,
correctly ordered by model capacity, and matched a prior that any reviewer would
have nodded along with. The tell was visible the whole time — four unrelated
architectures clustering within 0.2 m/s of persistence is not a result, it is a
fingerprint of measuring persistence four times.

### 2. TFT's objective mismatch (the secondary cause)

> **What:** the largest model was quietly solving a harder problem than its competitors,
> because of a library default nobody set.
> **Why it crippled the score:** the others predicted one number per hour. TFT predicted a
> whole distribution — 17 percentiles — from the same data, and was then graded on only
> the middle one. Same data, much harder task, unfair comparison.
> **Why it was hard to spot:** its training loss read ~0.42 against the others' ~0.008.
> Training loss is an internal score used only *during* training, and it is not comparable
> between models optimising different objectives — but a fifty-fold difference simply
> looks like a broken model rather than a different question being asked. Naming the
> objective explicitly fixed it: test MAE **1.551 → 1.171 m/s**, lower being better.

`darts.models.TFTModel` defaults to `likelihood=QuantileRegression()` over **17
quantiles**, while `BlockRNNModel`, `NBEATSModel` and `DLinearModel` all default
to plain MSE. TFT was therefore fitting a full predictive distribution on the same
28k samples while its competitors fitted a point estimate, and only the median
quantile was being scored.

| TFT objective | Test MAE | R² |
|:---|---:|---:|
| `QuantileRegression` (darts default) | 1.551 | −0.163 |
| `MSELoss` (`model.deep.loss_fn: mse`) | **1.171** | **0.330** |

Both runs early-stopped at effectively the same point (28 vs 29 of 60 epochs), so
this is not a training-budget effect. The default accounts for the entire deficit.

So the v3 claim "TFT has an architecture mismatch" reached the right conclusion by
the wrong route: the problem was the objective, not the absence of static
covariates.

### 3. What did *not* turn out to be the cause

> **What:** three confident explanations from the earlier analysis, each tested and each
> found not to hold.
> **Why publishing the failures matters:** all three are standard, sensible-sounding
> arguments — too little data for deep models, too short a training budget, engineered
> features favouring the tree. Each was checked directly and none survived. The
> "too little data" argument fails most dramatically: the model with by far the worst
> ratio of examples to parameters placed *third*, not last.

Three explanations that seemed strong in v3 and did not survive contact with the
corrected results:

**Data volume / samples-per-parameter.** The heuristic below was the headline
argument in v3. It has no predictive power here — the model with the *worst*
ratio-adjusted standing (LSTM, ~0.9 samples/param) beat the model with the best
(XGBoost, ~4×), and N-BEATS at 7.4 M parameters on 28k samples lands mid-field.

| Model | ~Parameters | Samples / param | Final rank |
|:---|---:|---:|---:|
| XGBoost (450 trees × ~16 leaves) | ~7,000 effective leaves | ~4× | 2 |
| LSTM (hidden=64, 1 layer) | ~30,000 | ~0.9× | **1** |
| TFT (hidden=64 + attention) | ~100,000+ | ~0.3× | 4 |
| N-BEATS (3×3×256, 48 covariates) | ~7,400,000 (measured) | ~0.004× | 3 |
| DLinear (kernel=25) | ~113,000 (measured) | ~0.25× | 5 |

Note the two parameter counts that v3 estimated badly: N-BEATS is 7.4 M, not
~500k (the 48 covariates widen the first layer enormously), and DLinear is 113k,
not "a few thousand". If the sample-efficiency argument held, N-BEATS should have
been catastrophic. It placed third.

**Training budget.** v3 argued all four were undertrained at 30 epochs. With a
200-epoch cap and early stopping on validation loss, every model converged well
below the old budget: DLinear 23 epochs, LSTM 24, N-BEATS 19, TFT 29. Thirty
epochs was already *past* convergence. What v3 actually lacked was validation-based
stopping and best-checkpoint restore — without those, training for 30 epochs
returns whatever weights epoch 30 happened to hold.

**Feature engineering asymmetry.** v3 argued that XGBoost's labelled
`wind_speed_hub_lag12` column is more usable than the same information as raw
floats in a sequence. Plausible, and false here. Ablated directly on DLinear:

| DLinear | Test MAE | R² |
|:---|---:|---:|
| lag/rolling columns included (48 covariates) | 1.215 | 0.302 |
| lag/rolling columns withheld (39 covariates) | **1.196** | 0.295 |

Withholding them is marginally *better*. The sequence models already have the
target history in the input chunk; the lag columns are redundant re-encodings that
widen the covariate tensor for nothing.

### On N-BEATS and inductive bias

> **What:** an earlier claim that one architecture's design was especially suited to wind
> data — withdrawn, because the corrected results do not support it.
> **Why it is worth recording the retraction:** the story was appealing (a model built for
> trends and cycles, applied to data with a strong daily cycle). Once the scoring was
> fixed it landed mid-field, and all five architectures fell within 0.101 m/s. When five
> very different designs agree that closely, none of them is capturing something the
> others miss — the limit is in the data, not the design.

v3 attributed N-BEATS's (apparent) edge to hierarchical basis decomposition
matching wind's trend + seasonality structure. That story is no longer supported:
N-BEATS is now mid-field at 1.157, between XGBoost and TFT, and every architecture
in the zoo lands within 0.101 m/s. When five very different inductive biases
converge that tightly, the honest reading is that none of them is capturing
something the others miss — the signal available in these features is being
extracted about as well as any of them can manage, and the residual is weather
chaos plus reanalysis resolution.

## Feature-ranking interpretation

> **What:** reading the model's own account of which inputs it relied on, and checking it
> against meteorology.
> **Why it is reassuring:** the ordering makes physical sense — current wind speed first,
> then time of day and season, then winds high above the site. A model that predicted well
> for physically absurd reasons would be far harder to defend.
> ⚠️ **Read with the caveat attached:** relying on a feature is not the same as needing it.
> The ablation earlier in this document is the authority on value; this section is about
> plausibility.

The v3 top-15 features (from
[04_stage1_wind_speed.md](04_stage1_wind_speed.md)) tell a coherent physical
story:

- **The centre-cell 100 m wind speed dominates by a wide margin** (SHAP 0.44).
  Not surprising — it's the closest available proxy to the actual hub wind speed
  we're predicting.
- **Its patch mean is the #2 predictor** (SHAP 0.29). Spatial averaging carries
  meaningful information beyond the point value — reduces noise, captures the
  regional-scale wind field, and helps at grid-cell edges where point sampling
  is aliased.
- **Calendar features are the #3–5 predictors** (`hour_sin`, `month_cos`,
  `month_sin`). Anloga's diurnal sea-breeze cycle and seasonal (harmattan-adjacent)
  patterns are strong. The tree can lean heavily on the calendar for base
  predictions.
- **Upper-air features earn their keep** — 850 hPa wind direction (both cos and
  sin) at #6 and #15. In meteorology, mid-troposphere flow drives surface
  winds via pressure-gradient dynamics. Adding these features in v2 was
  physically motivated and shows up in the ranking exactly as expected.
- **Boundary-layer height** (#11) and **dewpoint** (#8) contribute meaningfully.
  BL height governs vertical mixing between surface and upper-air; dewpoint
  is a proxy for humidity and stability.
- **`temperature_c_grad_ew`** (#13) — the east-west temperature difference across
  the grid patch (1 lat × 2 lon, so this is a two-cell difference, not a 3×3 gradient).
  Temperature gradients drive pressure gradients drive wind, so it is physically expected
  to matter — but see the ablation caveat under "Defensible conclusions": the whole
  spatial block is worth 0.003 m/s. Note also that `_grad_ns` cannot matter, being
  identically zero on a single-latitude patch.
- **`wind_speed_hub_lag12`** is the highest-ranked lag feature (#12). Interesting
  that lag1, lag2, lag3 don't appear in the top 15 — for a 24 h forecast, they're
  correlated with each other and with the current observation. XGBoost picks
  lag12 as the most useful *representative* of the recent-past information.
- **`temperature_c` itself** is at #10 — captures the air-density effect (hotter
  → less dense air → lower power, indirectly relevant to wind-power modelling).

**What's absent from the top 15 that we might expect:**
- `wind_speed_10m` — not top-ranked because once the 100 m wind is in the model,
  the 10 m reading adds little independent signal.
- `wind_gust_10m` — the v2 gust variable didn't crack the top 15. Gusts are
  intermittent, and 24 h forecasting isn't gust-sensitive.
- Pressure-level winds at 500 hPa — 850 hPa is closer to the surface and
  more informative. 500 hPa is likely dropped because 850 hPa is already there.

## Positioning relative to Energies 2021 benchmark

> **What:** an honest comparison against the published study this architecture came from.
> **Where it stands:** about 2.5 percentage points behind (11.25% vs ~8.8%).
> **Why the gap is expected rather than embarrassing:** they had real turbine
> measurements from an operating Polish wind farm and an operational forecast as input;
> this project has neither, by necessity. The section separates the factors that genuinely
> explain the gap from the one that was assumed to and did not — data volume.

The published two-stage benchmark for this architecture is **8.8% nMAE / 13%
nRMSE** (Donadio, Fang & Porté-Agel, Energies 2021, Model 2). Our v4 result is
**11.25% nMAE / 15.50% nRMSE** — a gap of ~2.5 pp on nMAE and ~2.5 pp on nRMSE.
(v3 was 11.66 / 15.67; the alignment fix and the resulting switch to an LSTM
Stage 1 closed about 0.4 pp of it.)

**Explaining the gap** — factors that plausibly account for the difference:

- **No forecast in the input (the dominant factor).** Their setup is an operational
  forecasting problem; ours predicts t+24 from the analysed state at t alone, which caps
  R² at 0.326 (measured above). This is the one that is being fixed.
- **~~Data volume~~:** they used 5+ years of hourly SCADA at a real Polish farm; we have
  4 years of reanalysis. **Measured and largely ruled out** — the learning curve flattens
  after 1.6 years and the train/test ratio is 1.16. Retained only as a minor contributor.
- **Reference truth:** they validated against actual turbine power output. We
  validate against physics-derived reference (curve applied to actual observed
  wind speed). Our "truth" has zero measurement error but also zero
  turbine-specific artefacts. Real SCADA is noisier, but a model trained on real
  SCADA effectively learns to correct for turbine idiosyncrasies.
- **Spatial resolution:** ERA5 at 25 km smooths out coastal terrain effects that
  matter for a real turbine's inflow. A downscaled dataset (WRF at 3 km, or
  ICON-D2 at 2 km) would likely close some of the gap.
- **Data sparsity in region:** Ghana is under-represented in ERA5's data
  assimilation (fewer stations, fewer aircraft, sparser radiosondes). This
  reduces reanalysis skill locally compared to Europe.

**What the gap does not mean:** it does *not* mean our architecture is broken.
The two-stage design, the modelling approach, and the code all work correctly.
The gap is an **information** ceiling, not a sample-count one: the next improvement has
to add information the model does not currently have — an operational NWP forecast valid
at the target time, or real observations — rather than more rows of the same features or
another architecture.

## Threats to validity

> **What:** eight reasons the results might be less solid than they appear, written out
> deliberately.
> **Why volunteer them:** an examiner or reviewer will look for exactly this list. Stating
> the weaknesses yourself, with their size and direction, is far stronger than having them
> discovered — and several here are genuinely important. In particular, the results come
> from a single random seed on a single split, so small differences between models are
> not statistically separated and must not be reported as if they were.

**1. No real SCADA validation.** The whole pipeline is validated against a
physics-derived reference that uses the *same power curve* both sides of the
comparison. This means the pipeline validates the **forecasting step** (does ML
predict wind speed well enough that the derived power series is close to the
physics-derived reference), not **absolute physical yield** (would this pipeline
predict a real turbine's actual generation accurately). Bridging this gap
requires a met-mast installation or first-of-a-kind SCADA feed at Anloga —
noted as an explicit next step.

**2. Single-site generalisation.** Everything here is Anloga-specific. A model
trained on Anloga will not generalise to Ada Foah or Takoradi — this is a
documented property of the architecture, not a bug (retraining on a new lat/lon
is a one-line config change). But we don't know whether the *conclusions*
generalise. If we ran the same pipeline at Takoradi or Ada Foah, would
XGBoost still win by the same margin? Would the top features be similar?
Deferred to future work.

**3. Test window is one contiguous period.** The 20% hold-out is the last 20% of
the timeline (roughly late-2023). If late-2023 was atypical (unusual weather
regime, El Niño/La Niña state, exceptional storms), the metrics could be
positively or negatively skewed. Cross-validated metrics on the training slice
are similar to the test metric (CV mean MAE ≈ 1.11, test MAE 1.14), so this
risk is bounded — but it's not zero.

**4. Single seed, single split — the model ranking is not statistically
separated.** Every v4 number is one run at `seed=42` on one chronological 80/20
split. The LSTM's 0.029 m/s margin over XGBoost is comfortably inside what seed
variance could produce, and the five models span only 0.101 m/s in total. **Do
not claim "LSTM beats XGBoost" on this evidence.** The supported claim is that
these architectures are near-equivalent on this problem. A multi-seed repeat
(≥5 seeds, reporting mean ± sd) is the cheapest way to make the ordering
defensible and is listed in [08_next_steps.md](08_next_steps.md).

**5. The persistence baseline is a 25-hour persistence.** `PersistenceForecaster`
predicts from the `wind_speed_hub_lag1` column — `hub(t − 1)` — because
`build_feature_frame` excludes the raw `wind_speed_hub` column from `X`. Scored
against `y[t] = hub(t + 24)` that is a 25 h naive forecast. True 24 h persistence
is **MAE 1.399 / RMSE 1.791 / R² 0.043**, versus the 1.440 / 1.832 / −0.002
quoted throughout v1–v4 — so the bar has been ~2.9% too low in every iteration.
It was left unchanged so that v1–v4 stay comparable; all v4 tables quote both.
Any headline "beats persistence by X%" should use 1.399 (XGBoost's improvement is
18.3%, not 20.6%). See [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md) §4.

**6. 24-hour horizon only.** All results are at `horizon_hours = 24`. Different
horizons will produce different rankings — e.g., persistence at 6 h is very
hard to beat. Deep models might do relatively better at longer horizons where
persistence collapses further and sequence structure matters more. Deferred.

**7. Fixed feature set.** The feature ranking says the top ~10 features carry
most of the signal. We didn't try feature selection to see if dropping the
bottom half helps or hurts. Marginal expected gains.

**8. The LR sweep is biased toward small learning rates**, and the cosine
schedule never completes. Candidates are scored over a short budget with the
scheduler disabled, which favours whatever descends fastest early; and with a
200-epoch cosine cap and early stopping at ~23, the LR barely decays. Both are
mild confounds on the deep-model numbers, in the direction of *understating*
them.

## What worked, what didn't, and what surprised us

> **What:** a candid retrospective in four lists — successes, failures, pleasant surprises
> and unpleasant ones.
> **Why it earns its place in a technical document:** the surprises are where the real
> learning is. Two library defaults and one index offset moved results further than any
> deliberate design decision in the entire project — which is a finding about how this
> kind of work goes wrong, not just about wind forecasting.

**Worked:**
- The overall two-stage architecture (Donadio et al. 2021 Model 2) reproduces
  cleanly in code and behaves as expected.
- Regularisation of XGBoost collapsed the v1 overfit gap from 8× to 1.2×
  without hurting test performance.
- The pipeline's "beat persistence" guardrail caught the v1 failure exactly as
  designed — the design principles paid off in practice.
- Feature engineering (spatial derivatives, upper-air, BL height) improved the
  metric more than *switching architectures* did — though v4 shows both are
  dwarfed by fixing the evaluation.
- Optuna tuning delivered a small but real ~1% improvement.
- The chunked ERA5 loader survived multiple CDS failures (cost limits, licence,
  ZIP quirk, transient job failures) with retry logic and continued to progress.
- Early stopping on validation loss with **best-checkpoint restore** — the piece
  v3 was actually missing. Without the restore, early stopping hands back weights
  `patience` epochs worse than the best and is net harmful.
- The per-model LR sweep, which picked a different learning rate for each
  architecture (3e-4 / 1e-3 / 3e-3 / 1e-3). A single shared default would have
  been wrong for two of the four.

**Didn't work:**
- Two of doc 09's three implementable recommendations. "Train 200 epochs" was
  moot (models converge in ~20–25), and "feed the lag features as past
  covariates" was neutral-to-slightly-harmful.
- Getting substantially below 11% nMAE — architectural improvements alone are
  not enough; the ceiling is likely data.
- Multi-site pooling (doc 09 rec #4) is not testable here: Anloga is the only
  site, and ERA5 grid neighbours at 0.25° are the same synoptic cell, so they
  would add correlated rows rather than new information.

**Surprised us (in a good way):**
- Optuna finished 50 trials in 4 minutes, not the 30 minutes I estimated.
- A single-layer LSTM with 30k parameters beat a 50-trial Optuna-tuned XGBoost
  on 28k samples — the opposite of what the sample-efficiency argument predicted.
- One library default (`loss_fn: mse`) was worth 0.380 m/s on TFT, more than any
  architectural change in the project.
- Spatial derivatives (a v2 addition) actually appear in the top-15 SHAP
  ranking — the theory (pressure gradients drive wind) shows up empirically.

**Surprised us (in a bad way):**
- **The alignment bug survived two full iterations** because it produced
  believable numbers that confirmed a well-known prior. It was never caught by a
  test, because the deep-model tests asserted shape and finiteness, not quality.
  The v4 fix ships with a regression test that asserts a *quality floor* on a
  37-hour periodic signal, chosen so that a horizon-shifted model cannot pass.
- `darts.TFTModel` swaps in a 17-quantile likelihood when you leave `loss_fn`
  unset, and nothing in the training logs makes this visible — its `val_loss`
  simply reads ~0.42 against the other models' ~0.008, which looks like a broken
  model rather than a different objective.
- `joblib.dump` on a darts model silently writes a weightless shell — darts'
  `__getstate__` drops the LightningModule by design. This never bit before v4
  only because XGBoost always won and the code path was never taken.
- TFT costs hours per run on CPU (2h45 in v3; 4h33 and then 9h56 in v4, the last
  inflated by thermal throttling on a 15 W laptop part).
- CDS's cost limit is much more conservative than the documentation suggests —
  even 1-year requests failed until we went to per-month chunking.
- The ZIP-instead-of-NetCDF gotcha for accumulated variables cost ~8 hours of
  runtime — the downloads succeeded but the loader crashed.
- Trailing bugs (the `DEEP_ALGOS` whitelist not including `nbeats`/`dlinear`)
  survived tests because the smoke tests use them individually — a full
  end-to-end `--compare` test would have caught it.

## Defensible conclusions for a write-up

> **What:** six statements that the evidence supports, phrased so they can be quoted
> nearly verbatim in a thesis.
> **Why they are worded so carefully:** each is deliberately limited to what was actually
> measured. "These architectures are near-equivalent" is supportable; "the LSTM is better"
> is not, on one seed and one split.
> ⚠️ **Check the numbers before quoting.** Conclusion 5 carries a correction, and all six
> describe the 2020–2023 study period — the in-progress v5 work uses a different window.

You can honestly state the following in a thesis chapter or technical report.
Every number is from the v4 run; quote 1.399 (true 24 h persistence) as the
baseline, not 1.440 — see threat 5 above.

1. "For day-ahead wind-speed forecasting at Anloga using 4 years of ERA5
   reanalysis, a single-layer LSTM achieves test MAE 1.114 m/s and R² 0.390,
   a 20.4% MAE reduction over 24-hour persistence (1.399 m/s)."

2. "The two-stage (ML wind → curve) architecture achieves 11.25% nMAE and
   15.50% nRMSE on power output — comparable to Donadio et al. (2021)'s
   published benchmark of 8.8% / 13% nRMSE, within the expected gap given
   our single-site reanalysis-only constraints."

3. "Among a 50-trial Optuna-tuned XGBoost baseline and four modern sequence
   architectures (LSTM, TFT, N-BEATS, DLinear via `darts`), all five beat
   24-hour persistence once given identical information sets, a common MSE
   objective and validation-based early stopping. Test MAE ranged from 1.114
   (LSTM) to 1.215 m/s (DLinear), with the tuned tree at 1.143 — a spread
   (0.101 m/s) smaller than the gap from the weakest model to persistence
   (0.184 m/s). We therefore find no evidence that architecture choice is a
   material lever for single-site medium-horizon wind forecasting at this data
   volume."

4. *(the methodological finding — arguably the most valuable one here)*
   "Two configuration defects moved results further than any architectural
   choice: a 24-hour index misalignment between the backtest output (indexed at
   target time) and the supervised target (indexed at feature time), which
   pinned all four sequence models to the persistence error floor; and an
   undeclared library default (`darts.TFTModel`'s 17-quantile
   `QuantileRegression` likelihood) that had TFT optimising a distributional
   objective while its competitors optimised MSE. Correcting the first improved
   the sequence models by 0.27–0.36 m/s; correcting the second improved TFT by
   0.38 m/s. Both produced internally consistent, plausible results while wrong,
   and the first survived two review iterations because it agreed with the prior
   that gradient-boosted trees dominate small tabular problems."

5. "Feature-influence analysis (SHAP) confirmed that the strongest predictors
   are hub-height wind speed itself, its spatial mean over the ERA5 grid patch, and
   calendar features encoding diurnal + seasonal patterns. Upper-air winds (850 hPa),
   boundary-layer height, dewpoint, and horizontal temperature gradients all
   appear in the top 15, validating the meteorologically-motivated feature
   engineering."

   ⚠️ **Two corrections to make before quoting this.** (a) The patch is **1 lat × 2 lon**,
   not 3×3 — say "grid patch", never "3×3". (b) SHAP rank is not predictive contribution:
   ablating all 20 spatial columns changes test MAE by **0.003 m/s**, and half of them are
   structurally degenerate (`_grad_ns` identically zero, `_std` exactly `|_grad_ew|/2`).
   The upper-air claim survives — those columns are worth ~0.017 m/s — but the spatial
   derivatives should be described as physically motivated and empirically negligible.

6. "The remaining gap to published benchmarks is information-limited rather than
   sample-limited. A learning curve over 0.4–3.2 years of training data flattens after
   ~1.6 years (final doubling: −0.012 m/s) at a train/test MAE ratio of 1.16, and a
   feature-group ablation shows the 53-feature set saturating. With a 24-hour
   autocorrelation of 0.571 bounding R² at 0.326 for any predictor built on the analysed
   state at time t, and a measured R² of 0.390, the model is already extracting close to
   what these features permit. Closing the gap requires adding information the input does
   not contain — an operational NWP forecast valid at the target time, or real
   observations — not more history and not another architecture."

### What you should *not* claim

> **What:** three tempting statements the evidence does not support.
> **Why an explicit list:** these are the overclaims most likely to be made by accident,
> and each has a specific reason it fails — a margin inside seed noise, a single site at a
> single horizon, and figures from the superseded runs. Easier to check against a list
> than to re-derive the reasoning each time.

- ❌ "The LSTM outperforms XGBoost." One seed, one split, 0.029 m/s. Say the
  architectures are near-equivalent, or run ≥5 seeds first.
- ❌ "Deep models beat gradient-boosted trees on wind forecasting." The result
  here is a tie, on one site, at one horizon.
- ❌ Any figure from the v2/v3 deep-model tables (1.425 / 1.475 / 1.519 / 1.643).
  Those are misaligned and are retained only as history in
  [06_iteration_history.md](06_iteration_history.md).
