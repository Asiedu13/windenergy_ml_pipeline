# 08 — Next Steps

Ranked catalogue of realistic improvements to the pipeline, with estimated effort
and expected return-on-investment. Written for the future maintainer / researcher
who wants to know "what would I do next, and would it be worth it?"

The main constraint at this point is **data availability**. Architectural
experiments have shown diminishing returns (v1.5 → v2 → v3 → v4 nMAE 12.02% →
11.77% → 11.66% → 11.25%), and v4 established that the five architectures tested
are near-equivalent once measured correctly. The highest-ROI improvements from
here on need better data or better *evaluation*, not better models.

---

> ## ⚠️ Largely superseded — see [12_next_steps_v2.md](12_next_steps_v2.md)
>
> This document's tier structure assumes the binding constraint is **data volume**. That
> assumption has since been measured and refuted: the learning curve flattens after ~1.6
> years, and the real constraint was the *absence of a forecast* in the inputs. Adding
> archived NWP forecasts cut test MAE by ~23% and lifted R² from ~0.48 to ~0.70.
>
> **Use [12_next_steps_v2.md](12_next_steps_v2.md) for current priorities.** This file is
> kept for provenance and because several individual items (multi-seed, multi-horizon,
> probabilistic forecasting) carry over — but their ranking here is out of date, and the
> "extend to 10+ years" item is retired outright.

> 📐 **Reading the numbers.** Unless stated otherwise, a figure like `1.475` is a **test
> MAE in m/s** — how far off the model typically is when predicting hub-height wind speed
> 24 hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399. Percentages such as `11.25%` are power error as a share of the
> turbine's 3.05 MW maximum, also lower-is-better. Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** a ranked catalogue of everything that could be done next,
> with an effort estimate and an expected payoff for each.
> **Why rank rather than list:** effort and payoff are only loosely related. Some of the
> most valuable items here are cheap (re-running with several random seeds), while some
> expensive ones would barely move the result. The tiers exist so limited time goes to the
> right places.
> **How each item is structured:** *Why* it is worth doing, *What to change* in the code,
> *Effort*, and *Expected ROI*. Jargon: [glossary.md](glossary.md).

## Done since this list was written

> **What:** items from earlier versions of this list that have since been built, marked
> with what actually happened.
> **Why keep them rather than delete:** several delivered something quite different from
> what was predicted — one was abandoned once measurement showed its premise was wrong.
> Keeping the original estimate next to the real outcome is how the estimates in the rest
> of this document earn or lose credibility.

### ✅ LSTM training fixes *(was item 1)* — done in v4, and it overshot the estimate

Shipped in [`src/stage1_deep.py`](../src/stage1_deep.py): validation tail,
`EarlyStopping(monitor="val_loss", patience=20)`, best-checkpoint restore,
`n_epochs: 200`, `WarmupCosineLR` (used instead of `ReduceLROnPlateau`),
gradient clipping, an LR sweep, and lag features re-added as past covariates.

**Outcome vs the prediction above.** This item predicted the LSTM's test MAE would improve
from **1.475 m/s to 1.20–1.30 m/s** — "competitive with XGBoost but probably not
surpassing it". It landed at **1.114 m/s** and *did* surpass XGBoost (1.143 m/s). But
almost none of that came from the training fixes — it came from the 24-hour alignment bug
this work uncovered. (Lower MAE is better; see the reading note at the top.)
Measured separately:

- Early stopping fires at **24 epochs**, so "increase `n_epochs` to 200" was moot;
  30 epochs was already past convergence.
- Re-adding lag features was **neutral-to-harmful** — the DLinear ablation scored
  **1.215 m/s** with them and **1.196 m/s** without, i.e. slightly *worse* with.
- The LR sweep was the one piece that helped.

See [10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md).

### ✅ TFT with proper training *(was item 10)* — done in v4b, and the cause was elsewhere

Not undertraining. `darts.TFTModel` defaults to a 17-quantile
`QuantileRegression` likelihood while the other models use MSE. With
`model.deep.loss_fn: mse`, TFT's test MAE goes **1.551 → 1.171 m/s** (R² −0.163 → 0.330,
where higher R² is better) and
clears persistence. Both runs early-stopped at ~28 epochs of 60.

The `use_static_covariates=True` idea remains untested and is still gated on
multi-site pooling (item 9).

### 🔄 Extend the training window to 10+ years *(was item 4)* — in progress

`config_10yr.yaml` + [`src/data_ingest_extend.py`](../src/data_ingest_extend.py)
(a resumable, download-only CLI — `load_weather` aborts if any core month is
missing, which is unhelpful for a multi-hour job):

```bash
python -m src.data_ingest_extend --config config_10yr.yaml
```

**Revised effort:** the original ~15 h estimate was optimistic. Observed CDS
throughput is roughly one month-family per 2–4 min including queue, so 120 months
× 3 families lands nearer **30–40 h** of mostly-unattended wall time. It survives
network drops (cdsapi retries internally) and re-runs resume from cache.

**Expected ROI, revised:** v4 removed the "deep models are starved of data"
premise — the models converge in ~20–25 epochs on 28k samples and sit within
0.101 m/s of each other. More data should still help, but expect it to lift the
whole field rather than reorder it.

---

## Tier 1 — Highest expected ROI

> **What:** the six items worth doing first.
> **Why they rank highest:** most are about making existing results *trustworthy* rather
> than better — repeating runs with several random seeds, correcting a baseline that has
> been slightly too easy, regenerating a report that shows superseded numbers. None of
> them will improve the headline accuracy.
> **Why that is still the right priority:** a defensible modest result is worth far more
> than an impressive one that cannot survive scrutiny — and the current model rankings are
> not yet statistically separated.

### 1. Multi-seed repeat to settle the model ranking *(new, v4)*

**Why:** every v4 number is a single run at `seed=42` on one chronological split.
The LSTM's 0.029 m/s margin over XGBoost is well inside plausible seed variance,
and the whole five-model field spans 0.101 m/s. As it stands the ranking cannot
carry any weight in a write-up.

**What to change:** loop `config.model.seed` over ≥5 values in `--compare`,
report mean ± sd per model, and have the report rank on the mean. Cheap for
XGBoost/DLinear/LSTM; N-BEATS and TFT dominate the cost, so consider 5 seeds for
the cheap three and 3 for the heavy two.

**Effort:** ~2 hours of code; wall time dominated by TFT (~5 h × n_seeds — or
skip TFT and state that).

**Expected ROI:** converts "the LSTM beat XGBoost" from an unsupportable claim
into either a real finding or a properly-evidenced tie. Highest value per hour of
anything on this list for the write-up specifically.

### 2. Fix the persistence baseline and restate v1–v3 *(new, v4)*

**Why:** `PersistenceForecaster` predicts from `wind_speed_hub_lag1` = `hub(t−1)`,
scored against `hub(t+24)` — a **25-hour** persistence. True 24 h persistence is
MAE 1.399 vs the 1.440 reported in every iteration, so the bar has been 2.9% too
low throughout. Every "beats persistence by X%" claim is inflated.

**What to change** in [`src/stage1_windspeed.py`](../src/stage1_windspeed.py):
give `PersistenceForecaster` the unshifted hub series (or add `wind_speed_hub`
back to `X` purely for baseline use and exclude it from the model's feature list).
Then restate the v1–v3 tables against the corrected baseline.

**Effort:** ~1 hour of code, plus care in updating the historical tables.

**Expected ROI:** no metric moves, but every comparative claim in the write-up
becomes defensible. Deliberately deferred during v4 so that run stayed comparable
with v1–v3.

### 3. Regenerate the report with the MSE TFT *(new, v4b)*

**Why:** `outputs/reports/report.md` still shows the quantile TFT at rank 5
(1.551) because v4b was run standalone. The consolidated table lives only in
`RESULTS.md` and `outputs/reports/report_v4_bakeoff.md`.

**What to change:** nothing — just re-run
`python -m src.pipeline --config config.yaml --mode forecast --compare` now that
`loss_fn: mse` is the default.

**Effort:** zero code, ~7 h wall time. Natural to combine with item 1.

### 4. Match each cosine cap to observed convergence *(new, v4)*

**Why:** with a 200-epoch cap and early stopping at ~23, `WarmupCosineLR` barely
decays — the anneal never completes, so rec #3 was only half-tested. The LR sweep
has a related bias: candidates are scored over a short budget with the scheduler
off, favouring whatever descends fastest early.

**What to change:** set `n_epochs` per model near observed convergence (~60), so
the schedule completes within the run and early stopping acts as a safety net
rather than the primary terminator. Optionally lengthen `lr_sweep.epochs`.

**Effort:** config-only; one bakeoff to confirm.

**Expected ROI:** unknown, plausibly 0.01–0.03 m/s. Cheap, and it closes the last
untested part of doc 09's recommendations.

### 5. Multi-horizon ML training — 📌 **designated follow-on to v5**

**Why:** The current pipeline trains and evaluates at a single horizon (24 h). A
model trained to predict horizons simultaneously (e.g., 6 h, 12 h, 24 h, 48 h)
would demonstrate how the ML advantage over persistence scales with lead time —
a useful chart for the write-up.

**Status change (2026-08-10):** this is now the **pre-registered next step if v5 lands a
modest 5–20% gain** ([11_nwp_forecast_features.md §7b](11_nwp_forecast_features.md)). The
input data already exists — the Open-Meteo fetch pulled `_previous_day1..3`, so forecasts
at 24/48/72 h lead are on disk and need no re-fetch. That makes this the cheapest
remaining way to broaden the contribution, and it pairs naturally with the NWP work:
forecast skill decaying with lead time is exactly what a horizon-vs-error curve shows.

**What to change:**
- Modify [`src/features.py`](../src/features.py)'s `build_feature_frame` to
  optionally build multiple targets (one per horizon). Note `add_nwp_features` already
  takes `offset_hours`, so the NWP join generalises to other horizons for free — pair
  horizon *h* with the `d{ceil(h/24)}` columns.
- Modify [`src/stage1_windspeed.py`](../src/stage1_windspeed.py) to train one
  XGBoost per horizon.
- Modify [`src/report.py`](../src/report.py) to include a horizon-vs-error curve.

**Effort:** half day.

**Expected ROI:** doesn't move the headline metric, but produces a much richer
report and shows clearly *where* ML earns its keep (24 h+) vs. where persistence
dominates (1–3 h). Great chart for a thesis.

### 6. Probabilistic forecasting (quantile regression)

**Why:** Grid operations care about the *distribution* of possible wind speeds,
not just the point forecast. A P10/P50/P90 forecast is much more actionable for
dispatch and reserve sizing decisions.

**What to change:**
- Add `stage1_algo: xgboost_quantile` option using `XGBRegressor(objective=
  "reg:quantileerror", quantile_alpha=[0.1, 0.5, 0.9])`.
- Add CRPS (Continuous Ranked Probability Score) and coverage (fraction of
  observations in [P10, P90]) to `compute_metrics`.
- Extend the report table to show P10, P50, P90 rows per horizon.

**Effort:** half day.

**Expected ROI:** A different type of output than what we've built so far, but
arguably more useful for real deployment. Not directly comparable to the current
MAE-based rankings, but a strong addition to the pipeline's capability.

## Tier 2 — Data-side improvements

> **What:** giving the model better *inputs* rather than a better algorithm.
> **Why this tier now matters most for accuracy:** the measurements in
> [07](07_findings_and_analysis.md) showed the model is already extracting close to
> everything its current inputs allow. Five very different architectures land within
> 0.101 m/s of each other, which is the signature of an information limit rather than a
> modelling one. Further architecture work has little left to win; new information does.
> **The flagship item here** — adding real weather forecasts — is already underway as v5.

> Extending the training window to 10+ years was item 4 and is now in progress —
> see [Done since this list was written](#done-since-this-list-was-written).

### 6b. Widen the ERA5 bbox to recover a real 3×3 grid patch *(new, 2026-08)*

**Why:** the `pad = 0.2°` bbox in `src/data_ingest.py` was written for a 3×3 patch but
actually snaps to **1 latitude × 2 longitudes** (`latitude = [5.75]`,
`longitude = [0.75, 1.00]`). Two of the four spatial statistics are therefore
degenerate across all five driver variables: every `_grad_ns` column is **identically
zero** (one latitude row means the north and south edges are the same row), and every
`_std` column is **exactly `|_grad_ew| / 2`**. That is 10 of the 36 tidy columns
carrying no independent information. Measured over all 35,064 rows of the 2020–2023
parquet; details in [00_data_card.md §3.4](00_data_card.md).

**What to change:**
- `pad = 0.35` in both `_fetch_era5` and `download_era5_range` (they must stay
  identical). That gives latitude 5.50/5.75/6.00 and longitude 0.75/1.00/1.25.
- Re-download all months into a new cache family (`era5b_` / etc.) so the existing
  1×2 cache and the results built on it stay reproducible.
- Rebuild the parquet and re-run the bakeoff.

**Effort:** low code change, high wall time — a full re-ingest is ~370 CDS requests
plus a re-run of the 5-model comparison.

**Expected ROI:** modest and uncertain. Real north–south gradients would replace five
dead columns and five collinear ones, but Anloga is coastal-flat and the current
`_grad_ew` already captures the dominant zonal signal. Worth doing opportunistically
alongside the 10-year backfill (both are re-ingest work), not on its own.

### 7. Add operational NWP forecasts as input features — 🔄 **in progress (2026-08-10)**

**Why:** we predict *from reanalysis alone*. Every exogenous feature is the analysed
state at time t, and hub wind at t+24 is only 0.571 correlated with it — an R² ceiling of
0.326 against a measured 0.390. This was promoted from "biggest potential lever" to the
active line of work once the learning curve showed data volume is not the constraint
(see [07_findings_and_analysis.md](07_findings_and_analysis.md)).

**Source chosen: Open-Meteo Previous Runs API**, not ECMWF's own archive. It serves
`{var}_previous_day{N}` — genuine fixed-lead archived forecasts — as point JSON for
`ecmwf_ifs025`, `gfs_global` and `icon_global`. The whole 2024-03 → 2026-06 fetch is
~2.6 MB across 9 requests. The GRIB alternative (`noaa-gfs-bdp-pds` on S3, 2021-02
onward) was rejected: every GRIB message is a *global* 0.25° field (~1 MB), so an hourly
multi-year archive is 100s of GB. Keep it as the fallback if the 10–100 m wind profile,
`HPBL` or `FRICV` turn out to matter — GFS carries all of those and Open-Meteo does not.

**Shipped so far:**
- `data_ingest._fetch_open_meteo` / `load_nwp_forecasts`, cached per (model, year)
- `features.add_nwp_features` with the valid-time → feature-time alignment contract
- `NWPConfig` in `utils.py`; new `config_nwp.yaml`
- 9 new tests, including a perfect-forecast alignment assertion on a 37 h signal period

**Two constraints to carry into the write-up:**
1. The archive starts **2024-03**, so the window moves and v1–v4 numbers are not
   comparable. Baselines must be re-run on the new window.
2. Do **not** use Open-Meteo's *Historical Forecast API* — it stitches the first hours of
   each run and is analysis-like. As a t+24 feature it is leakage.

**Expected ROI:** 2–4 pp nMAE in published NWP+ML comparisons. Unverified here until the
ERA5 truth for 2024-03 → 2025-12 finishes downloading and the with/without experiment
runs — report whatever it gives, including no improvement.

**When the GFS-GRIB fallback gets revisited — and when it does not.** Reserved for a
**null** result (<5% gain) that survives re-checking the alignment gates and testing
ECMWF alone, i.e. a null traced to *input poverty*: Open-Meteo exposes 5 usable variables
at one height, while GFS f024 carries the full 10/20/30/40/50/80/100 m wind profile plus
`HPBL` and `FRICV`, which would let the model see shear and stability directly instead of
inferring them. It is **not** triggered by a modest gain — that would be a week of GRIB
infrastructure for inputs whose marginal value is still unmeasured. Outcome bands and the
pre-registered response are in
[11_nwp_forecast_features.md §7b](11_nwp_forecast_features.md).

### 8. Downscaled meteorological fields (WRF / ICON-D2)

**Why:** ERA5 at 25 km averages out coastal terrain effects that matter for a
real turbine's inflow. WRF at 3 km or ICON-D2 at 2 km would resolve local wind
patterns much better.

**What to change:** a separate ingestion pipeline for downscaled outputs (WRF
requires user-run simulations; ICON-D2 outputs are available for Central Europe
but not Ghana as far as I know). This is *substantial* infrastructure.

**Effort:** multi-week — running WRF for Anloga would require compute resources
and configuration expertise.

**Expected ROI:** 1–2 percentage points nMAE improvement. Realistic only for a
serious multi-month research effort, not a project extension.

### 9. Multi-site pooling

**Why:** Train one model on data from several Ghanaian sites at once. Shared
weights amortise the sample-efficiency problem for deep models.

**What to change:**
- Extend `config.yaml` to accept a list of sites.
- Modify the data flow to concatenate site data with a site-identifier column.
- Modify the report to break down metrics per site.
- Optional: use `darts`'s multi-series API for the deep models.

**Effort:** 1–2 weeks. Requires selecting and downloading data for several
sites, which is largely orthogonal to the current codebase.

**Expected ROI:** Likely 1–2 percentage points nMAE improvement on the pooled
metric, and better generalisation to sites with less local training data. Deep
models benefit disproportionately.

## Tier 3 — Model / evaluation refinements

> **What:** improvements to how models are chosen and combined, rather than to the data.
> **Why they sit below the data work:** each is sound and each is worth a small amount.
> Selecting models by the metric that actually matters, or blending two models together,
> are standard techniques with modest reliable gains.
> **Why "modest" is the honest word:** with all five architectures within 0.101 m/s,
> there is simply not much room for a smarter choice among them to matter.

### 10. Model selection by downstream nRMSE, not intermediate MAE

**Why:** The current `--compare` flow picks the Stage 1 winner by test MAE on
wind-speed. But the actual goal is low nRMSE on power, and the two are only
loosely correlated (see [05_stage2_and_evaluation.md](05_stage2_and_evaluation.md)
§"Why MAE vs RMSE matters here"). We should select on the metric we actually care
about.

**What to change** in [`src/pipeline.py`](../src/pipeline.py):
- For each Stage 1 candidate, evaluate the full three-variant chain
  (`evaluate_variants` is cheap since Stage 2 is a lookup).
- Pick the model whose *two-stage nRMSE* on the test window is lowest.

**Effort:** ~1 hour.

**Expected ROI:** Might flip the ranking in edge cases. In v3 `direct` and
`two_stage` were within 0.04 pp on nRMSE while two_stage led nMAE by 0.5 pp,
which is exactly the regime where the choice of selection metric decides the
winner. (In v4 two_stage leads on both, so the issue is latent rather than
active.) More defensible model-selection criterion either way.

### 11. XGBoost + best-deep ensemble

**Why:** A simple average of the top two models often improves the ensemble
metric by 2–5% for free.

**What to change** in [`src/pipeline.py`](../src/pipeline.py):
- After training, average XGBoost's and the LSTM's test predictions (the two
  best models in v4, and different enough in inductive bias that their errors
  should decorrelate).
- Add "ensemble" as a row in the comparison table.

**Effort:** ~2 hours.

**Expected ROI:** small but essentially free. Also opens the door to more
sophisticated stacking with a meta-learner.

> TFT with proper training was item 10 and is now done — see
> [Done since this list was written](#done-since-this-list-was-written). The
> `use_static_covariates=True` variant remains untested and is still gated on
> multi-site pooling (item 9).

### 12. Feature selection for the deep track

**Why:** The deep models are being fed 48 past-covariate columns (39 with
`include_target_lags: false`). Some — the bottom half of the SHAP ranking — are
probably noise. There is direct evidence this matters: dropping the 9
lag/rolling columns *improved* DLinear's test MAE from **1.215 m/s to 1.196 m/s** (lower
is better), so the covariate set is already past the point of diminishing returns.

**What to change:** in `train_stage1_deep`, filter `X` to only the top-K
SHAP-ranked features before passing to `darts`. Sweep K ∈ {10, 15, 20, 30}.

**Effort:** ~2 hours.

**Expected ROI:** possibly 5–10% improvement on the deep-model MAEs. Worth
trying as part of the "give deep models a fair chance" story.

## Tier 4 — Infrastructure / validation

> **What:** the items that would address the project's deepest limitation — that it has
> never been checked against a real measurement.
> **Why they are ranked last despite being the most valuable:** they are not software
> tasks. Installing a meteorological mast at Anloga, or obtaining the first operational
> data from a Ghanaian wind farm, requires equipment, money and time far beyond this
> project's scope.
> **Why list them anyway:** they define what "finished" would actually look like, and any
> honest account of this work should say what it would take to validate it properly.

### 13. Real met-mast validation

**Why:** The whole pipeline is validated against a physics-derived reference, not
real turbine SCADA. A met-mast at Anloga would enable true validation.

**What to change:** none in the pipeline — this is a physical infrastructure
project (~$200k for a 100 m mast). Once mast data exists, drop it in as an
additional data source; everything else works.

**Effort:** multi-month to years, includes procurement, installation,
commissioning, quality control.

**Expected ROI:** the definitive answer to "does this pipeline actually work
for real operational deployment." Without it, all our metrics are "close to
physics-derived reference," not "close to real power output."

### 14. GMet ground-station bias correction

**Why:** Ghana Meteorological Agency (GMet) operates surface weather stations at
Ada Foah and other coastal sites. These stations record 10 m wind speed at
sub-hourly resolution — much noisier than reanalysis but *real observations*.
Using them as bias-correction targets would tighten the reanalysis's local
accuracy without waiting for a met-mast.

**What to change:**
- Add a fetcher for GMet data (typically a weekly CSV download, format varies).
- Add a bias-correction step to `data_ingest.py`: fit a per-hour offset between
  ERA5 and GMet on a training window, apply to production data.

**Effort:** 1–2 weeks — GMet data access is variable; requires contact with the
agency, format cleaning, quality-control filters.

**Expected ROI:** 0.5–1 percentage point nMAE improvement. Modest but real, and
the only way to inject actual observations into the training signal without
turbine SCADA.

### 15. Full end-to-end `--compare` integration test

**Why (strengthened by v4):** the v3 `DEEP_ALGOS` whitelist bug survived the test
suite because `test_train_stage1_deep_lstm_runs_end_to_end` and the parametrised
`test_darts_new_models_smoke` each test one algo at a time via `stage1_deep`
directly; `pipeline.py`'s dispatcher was never exercised.

The v4 alignment bug makes the same point more sharply. It survived *two*
iterations of tests that all asserted the same weak property — that predictions
have the right shape and are finite. No shape-based assertion can detect a
24-hour phase error. The lesson is not just "add an integration test" but **"make
model tests assert a quality floor against a baseline"**, which
`test_deep_forecast_predictions_are_horizon_aligned` now does.

**What to change:** add `test_pipeline_compare_mode` running
`pipeline.run(config_path, mode="forecast", compare=True)` with
`compare_algos: [xgboost, dlinear]` on synthetic data. Assert both appear in the
report, and assert the winner beats persistence on the synthetic signal.

**Effort:** ~1 hour.

**Expected ROI:** small in metric terms, high in confidence. Should have been
there from v3.

## Not recommended (deliberately deferred)

> **What:** plausible-sounding ideas that were considered and consciously rejected.
> **Why write down what you are *not* doing:** these are the suggestions most likely to be
> raised by a reviewer, and "we considered it and here is why not" is a much stronger
> answer than encountering the question for the first time in a viva.
> **The common thread:** most would add complexity without addressing the actual
> constraint, which is the information available to the model rather than the model itself.

- **Switching to LightGBM as primary.** Similar accuracy to XGBoost in this
  problem class, no strong reason to swap. Available via `stage1_algo: lightgbm`
  if desired.

- **~~Deep learning without richer data.~~** *(Withdrawn after v4.)* This item
  claimed deep models could not beat XGBoost on the current dataset and that the
  data-volume ceiling was binding. Both halves were wrong: the LSTM beats the
  tuned XGBoost, and all five architectures land within 0.101 m/s of each other.

  The *revised* advice points the same way for a different reason. Adding more
  architectures (Informer, PatchTST, custom Transformers) is low-value not because
  they would lose, but because they would almost certainly **tie** — five very
  different inductive biases already converge to within 0.1 m/s, which says the
  extractable signal in these features is close to exhausted. Spend the effort on
  items 1–3 (making the existing comparison statistically sound) and on Tier 2
  data work instead.

- **Cross-region generalisation studies.** Interesting scientifically but adds
  substantial complexity (multiple sites, harmonising data quality) and isn't
  what the pipeline was built for. Leave for a follow-on research effort.

## Suggested prioritisation

> **What:** a recommended running order across all four tiers.
> **Why it does not simply follow the tier numbers:** the ordering weighs effort against
> payoff *and* against what a write-up needs. Cheap credibility work comes first, the
> in-flight forecast experiment next, and anything requiring new physical infrastructure
> is acknowledged as out of scope.
> **How to use it:** treat it as a default, not a rule — if the v5 forecast experiment
> lands well, its follow-on work legitimately jumps the queue.

If a follow-on developer has:

- **One day:** items 2 (fix the persistence baseline) + 3 (regenerate the report)
  + 4 (cosine cap). All cheap, and together they make every number in the
  write-up defensible. Start item 3's 7 h run in the background first.

- **One week:** add item 1 (multi-seed repeat — the single highest-value thing
  for the write-up), then 5 (multi-horizon), 10 (selection by nRMSE), 11
  (ensemble), 15 (integration test). Keep the 10-year ERA5 download running
  throughout; it is nearly all queue time.

- **One month:** all of the above plus item 6 (probabilistic forecasting) and
  meaningful progress on item 7 (NWP forecast integration) — item 7 remains the
  single biggest potential lever on the metric.

- **Publication-ready study:** all Tier 1–3 items, plus item 14 (GMet
  bias-correction). Item 13 (met-mast) is the only path to truly definitive
  validation and is a much larger commitment.

> **Priority note after v4.** Items 1–3 are ranked above the modelling work
> deliberately. The project's current bottleneck is not model quality — it is
> that the headline comparison rests on one seed, one split, and a baseline that
> is 2.9% too easy. Fixing measurement is worth more than another architecture.
