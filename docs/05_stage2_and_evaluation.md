# 05 — Stage 2 (Power Curve) & Evaluation

Stage 2 is the deterministic power-conversion step. This document describes the
turbine, the manufacturer power curve, the air-density correction, the three-variant
evaluation framework, the error-propagation analysis, and the definitions of nMAE /
nRMSE used to normalise power metrics.

> 📐 **Reading the numbers.** Percentages like `11.25%` are **nMAE** — the typical power
> error as a share of the turbine's 3.05 MW maximum output, where **lower is better**. For
> scale: published benchmark ~8.8%, best here 11.25%, no-ML physics baseline 14.65%.
> Figures in m/s such as `1.114` are wind-speed errors from Stage 1. Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** Stage 2 — turning a predicted wind speed into predicted
> electricity — plus how the whole pipeline is scored.
> **Why read it:** Stage 1 ([04](04_stage1_wind_speed.md)) is the machine learning; this is
> the physics and the marking scheme. No model is trained here at all. Terms:
> [glossary.md](glossary.md).

## Turbine: Enercon E-101/3050

> **What:** the specific turbine being simulated, and the three wind speeds that define
> its behaviour (start generating, reach maximum, shut down for safety).
> **Why it matters:** these thresholds make the wind→power relationship very non-linear.
> Below 2 m/s output is zero no matter how accurate the forecast; above 12 m/s output is
> capped. Forecast accuracy therefore only *matters* in the band between.

Loaded via [`src/stage2_powercurve.py`](../src/stage2_powercurve.py)'s
`load_turbine(name, hub_height_m)`, which wraps `windpowerlib.WindTurbine`:

```python
WindTurbine(hub_height=100, turbine_type="E-101/3050")
```

**Nameplate specifications** (from OpenEnergy DataBase via `windpowerlib`):

- **Nominal power:** 3,050,000 W (3.05 MW)
- **Rotor diameter:** 101.0 m
- **Hub height:** 100 m (configurable)

**Power curve characteristics** (extracted programmatically by `turbine_spec`):

- **Cut-in wind speed:** 2.0 m/s — turbine starts generating
- **Rated wind speed:** 12.0 m/s — power reaches nameplate maximum
- **Cut-out wind speed:** 25.5 m/s — turbine shuts down to protect blades

Between cut-in and rated, power output rises roughly as `wind_speed³` (fluid
dynamics — kinetic energy flux is `½ ρ A v³`, and Betz's limit caps efficiency at
16/27 ≈ 59.3%). Between rated and cut-out, pitch control holds output at nameplate.
Above cut-out, power drops to zero instantly. This creates a strongly nonlinear
mapping from wind-speed error to power error, which is central to the pipeline's
error-propagation analysis (see below).

The curve is a discrete lookup table of `(wind_speed, power)` pairs; the actual
`windpowerlib` implementation interpolates linearly between them.

## Air-density correction

> **What:** adjusting the power calculation for how thin or thick the air is.
> **Why it matters here specifically:** a turbine harvests the *kinetic energy* of moving
> air, which depends on air mass, not just speed. Hot air is less dense and carries less
> energy at the same wind speed. Ghana is hot — around 28 °C on average at this site — so
> skipping this correction would systematically overstate output.

Standard rated power curves assume sea-level ISA density (1.225 kg/m³ at 15°C).
Ghana's coastal climate is hotter (annual mean ~27°C at Anloga), which reduces air
density by ~3–4% and correspondingly reduces power output at any given wind speed.

`air_density(temperature_c, pressure_pa)` computes the ideal-gas density of dry
air:

```
ρ = p / (R_dry × T_K)      # R_dry = 287.058 J/(kg·K)
```

Applied when `windspeed_to_power` is called with a `density` argument. Internally,
`windpowerlib.power_output.power_curve` uses cube-root density correction:

```
u_effective = u × (ρ / ρ_standard)^(1/3)
```

then interpolates the standard curve at `u_effective`. This propagates the
density adjustment through the curve's nonlinear region correctly.

**Verification** (test_stage2.py):
- `air_density(15, 101_325)` returns ~1.225 kg/m³ (ISA standard, within 0.01 kg/m³).
- `air_density(35, 101_325)` returns < 1.225 (hotter → less dense).
- At mid-curve wind speeds, `windspeed_to_power(ws, turbine, density=1.10)` returns
  strictly lower power than `density=1.225` (correction goes the right direction).

Every production run applies density correction using the observed ERA5
temperature and pressure on the test window.

## `model_chain_from_weather` — physics-only path

> **What:** a route straight from raw weather to power, with no machine learning involved
> at any point.
> **Why it exists:** it answers the question every reader should ask — "does the ML
> actually add anything, or would plain physics do?" Having it built and reported on every
> run means the answer is measured rather than assumed.

For the "no ML" baseline variant, `model_chain_from_weather(weather_df, turbine,
...)` is a thin wrapper around `windpowerlib.ModelChain` that takes a raw weather
DataFrame straight to power in one call. It:

1. Extracts wind_speed at 100 m, temperature (K), pressure (Pa), roughness (falls
   back to 0.15 m if all-NaN).
2. Builds the MultiIndex DataFrame that `ModelChain` expects (variable name +
   measurement height).
3. Calls `ModelChain(turbine).run_model(frame)` and returns `power_output` as a
   `pd.Series` in W.

This is used only by the `physics_only` variant in the three-way comparison.

## The three-variant comparison

> **What:** three different ways of getting from weather to a power prediction, all run
> and scored side by side on identical data.
> **Why it matters:** a single number in isolation means nothing. Running a no-ML version
> and a shortcut version alongside the real architecture shows whether the extra
> complexity earns its place. The four subsections below define the yardstick and the
> three contenders.

Implemented in [`src/evaluate.py`](../src/evaluate.py)'s `evaluate_variants`.
Purpose: quantify what the ML in Stage 1 adds versus alternatives.

All three variants are evaluated on the **same test index** — the hold-out set
from `train_stage1`'s 80/20 split — and against the **same reference truth**.

### Reference truth

> **What:** what counts as the "right answer" the three variants are marked against.
> **Why it is unusual:** with no real turbine to measure, the correct power is *computed*
> by pushing the observed wind speed through the same power curve. The same physics
> therefore sits on both sides of the comparison, which cancels out any error in the
> curve itself. That makes this a fair test of forecasting skill and **not** a test of
> whether a real turbine would produce these numbers.

For every test timestamp `t + h`:

1. Take the *actual observed* hub wind speed from ERA5 at time `t + h`.
2. Compute density from actual observed temperature and pressure at `t + h`.
3. Push through `windspeed_to_power(ws, turbine, density)` → `y_true_power(t + h)`.

This is a physics-derived reference, not real SCADA. The pipeline is explicit
about this limitation in the auto-generated report's opening line.

### Variant A — physics_only

> **What:** no forecasting at all — take the weather as it was and convert it to power.
> **Why:** the floor. Any ML that cannot beat this is not worth running.

Take the *persistence baseline* wind-speed forecast (in forecast mode, this is the
ERA5 wind speed at time `t`, i.e., what you would know at prediction time), push
through the same power curve. No ML.

This measures how well a naive "current wind → curve" mapping would do at
day-ahead — a strawman baseline that tests whether the ML is even necessary.

### Variant B — two_stage (the target architecture)

> **What:** the real design — ML predicts wind speed, then physics converts it to power.
> **Why it is preferred:** wind-speed errors are far easier to diagnose than power errors,
> the turbine can be swapped without retraining, and Stage 2 will accept real
> measurements once any exist.

Take the Stage 1 ML-predicted wind speed for time `t + h`, push through the same
power curve.

This is what we care about — does an ML wind-speed forecast, converted via a
physical power curve, beat the two alternatives?

### Variant C — direct

> **What:** skip the middle step — one ML model straight from weather to power.
> **Why keep it:** it is the obvious simpler alternative, and it scores within about 0.4
> percentage points of the two-stage design. Reporting that honestly is what justifies
> choosing two-stage on interpretability and modularity rather than on accuracy.

Retrain an XGBoost model (same hyperparameters as Stage 1) but with the
physics-derived power as the direct target — i.e., skip the curve entirely and
predict power from features directly.

This is the "ML end-to-end" control. If direct beats two_stage substantially, the
curve is adding noise. If two_stage is at least competitive with direct, the
architectural benefits (interpretability, modular Stage 2) are earned honestly.

### v4 result summary (LSTM as Stage 1)

> **What:** the actual numbers the three variants scored in the latest complete run.
> **Why it matters:** two-stage wins, and physics-only is clearly worst — so the ML earns
> its keep. Note these are on the 2020–2023 period; the in-progress v5 work uses a
> different date range and is not comparable to them.

| Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|:---|---:|---:|---:|---:|---:|
| physics_only | 446,962 | 607,452 | −0.071 | 14.65% | 19.92% |
| **two_stage** | **343,197** | **472,715** | **0.351** | **11.25%** | **15.50%** |
| direct | 371,247 | 479,103 | 0.334 | 12.17% | 15.71% |

Two-stage wins on **both** nMAE (11.25% vs 12.17%) and nRMSE (15.50% vs 15.71%),
and beats physics-only by ~3.4 percentage points. The architecture validates: ML
wind-speed prediction → curve is competitive with end-to-end regression while
preserving modularity.

> In v3 the two were much closer (11.66% / 15.67% two-stage vs 12.17% / 15.71%
> direct) and the margin on nRMSE was inside noise. The v4 gain comes from the
> deep-model alignment fix, which promoted an LSTM (Stage 1 test MAE 1.114) over
> the tuned XGBoost (1.143) as the Stage 1 winner. `direct` is unchanged because
> it always trains an XGBoost on power and never went through the deep path.

## Normalised metrics: nMAE, nRMSE

> **What:** expressing the error as a percentage of the turbine's maximum output instead
> of in watts.
> **Why it matters:** "wrong by 343,000 W" means nothing on its own — it depends entirely
> on turbine size. Dividing by the 3.05 MW rating gives 11.25%, which can be compared
> against any published study using any turbine. This is the standard in the wind-power
> literature and the form the headline results are quoted in.

Absolute power errors in watts are hard to compare across turbines or sites.
Normalising by rated power gives a scale-invariant metric:

```
nMAE  = MAE  / rated_power × 100%       # rated_power = 3,050,000 W
nRMSE = RMSE / rated_power × 100%
```

**Literature reference:** Donadio, Fang & Porté-Agel (Energies 2021, Model 2)
report two-stage nMAE ≈ 8.8% and nRMSE ≈ 13% on a Polish wind farm. Our v4
two-stage sits at 11.25% / 15.50% — ~2.5 percentage points above the benchmark,
which is consistent with our tighter constraints (single 25 km grid cell, no
real SCADA validation, 4 years of data vs their ~5+ years).

**Interpretation guidance:** nMAE 10% means the model's typical error is 10% of
turbine capacity. For a 3.05 MW turbine that's ~300 kW — noticeable but tolerable
for grid-integration studies. Modern operational NWP+ML systems reach 4–7% at
this horizon, but only with substantially better inputs (ensemble NWP forecasts,
downscaled fields, met-mast bias-correction).

## Why MAE vs RMSE matters here

> **What:** the two headline error measures answer different questions — MAE is the
> typical miss, RMSE punishes big misses much harder.
> **Why the distinction is sharp in wind:** because power rises with roughly the cube of
> wind speed, a handful of badly-missed windy hours dominate the total energy error. RMSE
> notices those; MAE largely does not. Quoting only one of them hides half the story, so
> the pipeline always reports both.

MAE and RMSE tell subtly different stories:

- **MAE** penalises errors linearly. A 1 m/s wind-speed error at any speed
  contributes 1 m/s to MAE. Robust to outliers.
- **RMSE** penalises errors quadratically. A single 5 m/s miss contributes 25
  units to RMSE's inner sum; five 1 m/s misses contribute only 5 units total.

For a two-stage pipeline pushed through a ~cubic curve, **large wind-speed errors
propagate to disproportionately large power errors** — this is exactly what RMSE
captures. From the v2 report:

| Variant | Rank by nMAE | Rank by nRMSE |
|---|:---:|:---:|
| two_stage | 1st | 2nd |
| direct | 2nd | 1st |
| physics_only | 3rd | 3rd |

**The rankings can flip between metrics** — the two_stage model has fewer typical
errors (better MAE) but occasional larger misses (worse RMSE) than the direct
model. Neither metric alone is the whole story.

**Current pipeline choice:** Stage 1 model selection uses **MAE** as the tiebreaker
(via `min(comparison, key=lambda t: t[1].metrics_test.mae)` in
[`src/pipeline.py`](../src/pipeline.py)). This is a defensible convention because:
1. MAE is the metric in the plan's "beat persistence" guardrail.
2. At the wind-speed level, MAE and RMSE are usually highly correlated across
   models — picking one rarely flips the ranking.
3. Simpler for readers to interpret.

A future improvement (noted in [08_next_steps.md](08_next_steps.md)) would be to
select by downstream two-stage nRMSE — that would directly optimise the grid-relevant
metric rather than the intermediate wind-speed metric.

## Error propagation analysis

> **What:** measuring how a wind-speed mistake grows as it passes through the power curve.
> **Why it is required, not optional:** this is the central weakness of the two-stage
> design. Because power scales with roughly the cube of speed, a small wind error can
> become a large power error — and how large depends on *where* on the curve you are. A
> 1 m/s miss near cut-in barely matters; the same miss mid-curve is expensive. Design
> principle 3 in [doc 01](01_problem_and_design.md) requires this be measured and shown,
> not glossed over.

`error_propagation_plot` in [`src/evaluate.py`](../src/evaluate.py) explicitly
measures how wind-speed errors turn into power errors through the curve.

**Procedure:**
1. Compute per-hour |wind-speed error| = |ŷ_ws − y_ws| across the test window.
2. Compute per-hour |power error| = |ŷ_power − y_power| for the two_stage variant.
3. Bin samples by |ws error| into 10 quantile buckets (`pd.qcut(..., q=10)`).
4. For each bucket, plot mean |power error| in kW.

Saved to `outputs/figures/error_propagation.png`. Expected shape:
- Near-zero ws error → near-zero power error (obvious).
- Moderate ws error at mid-slope wind speeds (6–10 m/s) → disproportionately large
  power error (curve is steepest here).
- Large ws error at flat parts of the curve (above rated, below cut-in) →
  smaller power error than mid-slope, because the curve is flat there.

This plot is diagnostic — it tells the reader *where* the two-stage architecture
is most sensitive, and by implication where improvements to Stage 1 would pay
off most in power terms.

## Timeseries overlay plots

> **What:** charts showing predicted and actual power drawn on the same axes over a week.
> **Why they earn their place:** summary numbers hide the shape of failure. A plot reveals
> at a glance whether the model lags reality, flattens the peaks, or misses particular
> times of day — none of which a single MAE figure would tell you.

For each of the three variants, `timeseries_plot` produces a one-week overlay of
predicted vs reference power (in kW) at hourly resolution. Saved to
`outputs/figures/timeseries_{variant}.png`. Useful for spot-checking the model's
behaviour visually — does it track diurnal patterns, catch major weather events,
avoid systematic bias?

The `window_hours = 24 × 7 = 168` slice is the first week of the test window.

## Metric implementation details

> **What:** exactly how each metric is computed in code, including the edge cases.
> **Why it matters:** metrics with the same name are computed differently in different
> papers — R² in particular has several definitions, and MAPE misbehaves when true values
> approach zero, which happens constantly with wind. Spelling out the formulas makes the
> numbers reproducible and comparable.

`compute_metrics(y_true, y_pred)` in
[`src/stage1_windspeed.py`](../src/stage1_windspeed.py):

```python
mae   = mean_absolute_error(y_true, y_pred)
rmse  = sqrt(mean_squared_error(y_true, y_pred))
mape  = mean(|y - ŷ| / y) × 100 where y ≠ 0     # unstable near zero
r2    = r2_score(y_true, y_pred)
```

MAPE is included for completeness but is **unstable for power metrics** because
power series contain many near-zero values (below cut-in, above cut-out). This
is why the reported MAPE values are hundreds or thousands of percent for the
power variants — small denominators explode. Do not read MAPE as a primary
metric for power; use nMAE + nRMSE instead. The Stage 1 (wind-speed) MAPE
values are informative because wind-speed rarely goes to zero at Anloga.

Metrics are returned as a `RegressionMetrics` dataclass and rendered into the
report via `write_report`.

## Sanity tests

> **What:** automated checks that the physics behaves the way physics must.
> **Why:** these assert things that cannot be wrong in a correct implementation — no
> output below cut-in, no output above cut-out, never more than the nameplate rating,
> colder air yielding more power than warmer air at the same wind speed. Cheap to run and
> they catch whole categories of mistake instantly.

[`tests/test_stage2.py`](../tests/test_stage2.py) verifies the turbine + curve
behaviour without needing any weather data:

- **Below cut-in → 0 kW**: `windspeed_to_power([0, 0.5, 1.0, 1.5])` all zero.
- **Above cut-out → 0 kW**: `windspeed_to_power([26.5, 30.0])` all zero.
- **Between rated and cut-out → ~nameplate**: within 5% of `spec.nominal_power_w`.
- **Monotonic non-decreasing between cut-in and rated**: `np.diff(power) ≥ 0`.
- **Density correction lowers power at mid-range**: `density=1.10 < density=1.225`
  strictly reduces power output at any mid-curve wind speed.
- **`air_density(15, 101325) ≈ 1.225`**: ISA standard reproduces.
- **Turbine curve non-empty**: `turbine.power_curve` has both `wind_speed` and
  `value` columns, all values non-negative.

All 8 tests green in v3.
