
# 09 — Why the Deep Learning Methods Are Not Excelling

> ## ⚠️ Superseded — the premise of this document is false
>
> The deep-model results this analysis sets out to explain were a **measurement
> artefact**: every sequence model was scored 24 hours out of phase, which
> pinned all four to the persistence error floor and made them look uniformly
> bad. Once the alignment is fixed, an LSTM (test MAE **1.114 m/s**) *beats* the
> Optuna-tuned XGBoost (1.143), and N-BEATS and DLinear also clear persistence.
>
> Of the five causes proposed below, one is right for the wrong reason (TFT —
> it is darts' default 17-quantile objective, not static covariates; on MSE it
> goes 1.551 → **1.171** and clears persistence) and one is unchanged (the
> signal-to-noise ceiling). The rest are not supported. Of the five remedies
> proposed at the end, #1 and #2 measurably did **not** help; only #3 did.
>
> With the alignment fixed and a uniform objective, **all four sequence models
> beat persistence**, and the spread across all five architectures (0.101 m/s)
> is smaller than the gap from the weakest to persistence (0.184 m/s).
>
> See **[10_v4_deep_model_retraining.md](10_v4_deep_model_retraining.md)** for
> the bug, the fix, the implemented recommendations and the corrected bakeoff.
> This file is kept as written, for provenance.

> **What this document is:** a careful, well-argued explanation of something that turned
> out not to be true. It was written when the deep-learning models appeared to be losing
> to a simpler tree-based model, and it proposes five reasons why.
> **Why it is still here, unedited:** the results it explains were wrong — a scoring bug
> shifted every deep model by 24 hours. Rather than quietly delete the analysis, it is
> preserved exactly as written so the reasoning can be compared against what was actually
> happening.
> **Why that is worth your time:** the argument below is *convincing*. It matches a
> well-known rule of thumb (simple models beat deep ones on small data), it is internally
> consistent, and it survived review. It was still wrong. That is the most transferable
> lesson in this entire project: a plausible explanation for a plausible-looking number is
> not evidence that either is correct.
> **How to read it:** treat every section below as a hypothesis that was tested and
> mostly rejected. The verdict on each is in the banner above; the evidence is in
> [doc 10](10_v4_deep_model_retraining.md). **Do not quote any figure from this file.**
>
> *(Section-level What/Why notes are deliberately not added below — the body is kept
> verbatim.)*

> 📐 **Reading the numbers.** Figures like `1.475` are **test MAE in m/s** — how far off
> the model typically is when predicting hub-height wind speed 24 hours ahead, **lower is
> better**. For scale: best model 1.114, naive "tomorrow = today" 1.399, guessing the
> average 1.466. Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).
> ⚠️ **The figures in this document are invalid measurements.** They are the mis-scored
> ones this whole file was written to explain.

> This document reproduces, word-for-word, the original explanation given in
> response to the user's question: *"Explain to me why the Deep learning methods
> are not excelling in this case. What is the problem"*.
>
> For the extended, restructured version of this analysis (with additional
> feature-ranking interpretation and benchmark positioning), see
> [07_findings_and_analysis.md](07_findings_and_analysis.md).

---

Real question — and honestly the answer isn't a single bug. It's a stack of five things all pushing in the same direction. In rough order of importance:

## 1. Data volume is far below what deep models want

We have **~28,000 hourly training samples** (~3.2 years). Deep sequence models are usually validated on datasets 5–50× larger.

The rough "10 samples per parameter" rule illustrates the mismatch:

| Model | ~Parameters | Samples per param |
|---|---|---:|
| XGBoost (max_depth=4, 450 trees) | ~7,000 effective leaves | 4× |
| LSTM (hidden=64, 1 layer) | ~30,000 | **0.9×** |
| N-BEATS (3 stacks × 3 blocks × 256 widths) | ~500,000 | **0.06×** |
| TFT (hidden=64 + attention) | ~100,000+ | **0.3×** |

Trees are dramatically more sample-efficient because a split is a single boolean decision, not a continuous weight to be learned. This is the single biggest factor.

## 2. The training budget was tiny

- **30 epochs, no early stopping, no LR scheduling, no validation loss monitoring**
- Deep sequence models typically converge over 100–300 epochs with LR decay
- TFT in particular needs warmup + slow LR + gradient clipping to train well

So all four deep models are undertrained. TFT is the worst-off because it has the most parameters to fit with the fewest updates.

## 3. Feature engineering gave XGBoost the easy wins

XGBoost gets these hand-served features:
- `wind_speed_hub_lag12` (wind 12 hours ago)
- `wind_speed_hub_lag6` (wind 6 hours ago)
- `wind_speed_hub_rollmean24` (24h rolling mean)
- `hour_sin`, `month_cos` (diurnal + seasonal encoding)

For deep models I **deliberately excluded** these lag features because "the LSTM sees past targets in the sequence." That was correct in principle but wrong in practice — the RNN sees past values as **raw floats**, while XGBoost sees `lag12` as a labelled variable it can put an explicit splitting rule on ("if the wind 12 hours ago was between 6 and 8 m/s, predict X"). The deep model has to learn that same operation from scratch through its recurrent weights — and with 30 epochs it just doesn't.

## 4. This problem shape genuinely favours tabular models

Wind forecasting at 24h is dominated by:
- **Autocorrelation** (recent wind predicts future wind) — trivially encoded as lags
- **Diurnal + seasonal cycles** — trivially encoded as sin/cos calendar features
- **Simple pointwise weather variables** (temperature, pressure, 850 hPa wind, BL height)

None of these are the kind of signal deep learning shines at — there's no image, no long-range dependency requiring attention, no multi-modal input. It's structured tabular time series with strong exogenous variables. This is exactly the setting where gradient-boosted trees win in every major forecasting competition (M5, Kaggle, etc.).

## 5. TFT specifically had architecture mismatch

TFT is designed to shine when you have:
- **Static covariates** — site metadata (we have one site → none)
- **Future known covariates** — holidays, calendars (we set `add_relative_index=True` as a workaround, but that's a poor substitute)
- **Multiple related series** — many turbines with shared patterns (we have one)

Its attention mechanism and variable-selection networks are designed for a richer input structure than we're providing. Feeding it a single-site, no-static-covariate problem is asking it to use a fraction of its architecture.

## Why N-BEATS *did* work (barely)

N-BEATS uses **hierarchical basis decomposition** (trend + seasonality), which happens to match wind's structure exceptionally well. It's the deep model with the strongest inductive bias for this exact problem shape. But it still needs more training + tuning to move beyond "just past persistence."

## The signal-to-noise ceiling

Persistence MAE at 24h = 1.44 m/s. That's a hard floor set by weather chaos + our data resolution. XGBoost got 1.14 m/s — a 21% improvement. The remaining ~1 m/s error is largely irreducible with reanalysis data at a single 25 km grid cell. **No architecture will get you dramatic further gains without different data** (met-mast observations, downscaled fields, or ensemble forecast inputs).

## What would actually help deep models

If you wanted to give them a fair chance:
1. **Train longer** — 200 epochs with early stopping on validation loss
2. **Feed the lag features** as past covariates (undo my exclusion)
3. **Larger LR sweep + warmup + cosine decay**
4. **Multi-site training** if you had more Ghanaian sites — pooling amortises the sample-efficiency problem
5. **10+ years of data** — extend to 2010–2023 to get ~87k samples

None of these change the underlying reality: **for structured tabular time series with modest data and strong hand-engineered features, gradient-boosted trees are the right tool.** Deep models are the right tool for large-scale multi-modal problems, which this isn't. The v3 result is a defensible finding for your write-up: "an Optuna-tuned XGBoost with rich exogenous features remains competitive with modern deep sequence architectures for single-site medium-horizon wind forecasting."
