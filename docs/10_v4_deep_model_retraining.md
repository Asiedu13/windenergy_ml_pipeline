# 10 — v4: Retraining the Deep Models on doc 09's Recommendations

This document records what happened when the "What would actually help deep
models" list at the end of
[09_why_deep_learning_underperformed.md](09_why_deep_learning_underperformed.md)
was actually implemented and run.

The short version: **doc 09's diagnosis was mostly wrong, because it was
explaining an artefact.** The deep models were being scored 24 hours out of
phase. Once that is fixed, the cheapest deep model in the zoo goes from losing
to persistence to beating it comfortably — before any of the five recommended
remedies are applied. The remedies themselves turn out to range from
"marginal" to "actively unhelpful".

---

> 📐 **Reading the numbers.** Unless stated otherwise, a figure like `1.551` is a **test
> MAE in m/s** — how far off the model typically is when predicting hub-height wind speed
> 24 hours ahead, where **lower is better**. For scale: best model 1.114, naive
> "tomorrow = today" 1.399, guessing the average 1.466. R² runs the other way (higher is
> better). Full explanation:
> [glossary — How to read the numbers](glossary.md#how-to-read-the-numbers).

> **What this document is:** the discovery, diagnosis and repair of the project's most
> serious bug — plus everything that changed once results could be trusted.
> **Why it matters beyond this project:** the bug did not crash anything or produce
> obvious nonsense. It produced *believable* numbers that supported a tidy story, and it
> survived two full rounds of review. That is the failure mode worth studying.
> Unfamiliar terms: [glossary.md](glossary.md).

## 1. The alignment bug

> **What:** every neural network was being scored against the wrong hour — its predictions
> were compared to reality 24 hours out of step.
> **Why nobody noticed for two iterations:** the mistake did not make the models look
> broken. It made them look *mediocre in a plausible way* — clustered just behind a simple
> baseline, which matched what everyone already expected of deep models on modest data.
> **The scale of it:** fixing this one line moved the sequence models by 0.27–0.36 m/s,
> more than any deliberate design change made in the whole project, and reversed its
> headline conclusion.

### What was wrong

> **What:** two tables of numbers, each stamped with times measured on a different clock,
> were lined up as though the clocks matched.
> **The two clocks, plainly:** the forecasting library labels each prediction with *the
> hour it describes*. The training table labels each row with *the hour the prediction is
> made from*. For a 24-hour forecast those labels differ by exactly 24 hours. Joining them
> directly compares a prediction of Monday against the truth for Tuesday.

`src/stage1_deep.py` compared darts' backtest output against the feature
frame's target column without reconciling their two different time indices.

- **The feature frame** is indexed by *feature time* `t`. In forecast mode
  `build_feature_frame` sets `target = hub.shift(-horizon)`, so
  `y[t] = hub(t + 24)`. Row `t` means "given everything known at `t`, what is
  the wind speed at `t + 24`?"
- **darts `historical_forecasts(..., last_points_only=True)`** returns a series
  indexed by *target time* `τ`. `hist[τ]` is the prediction of the wind speed
  **at** `τ`, produced from data up to `τ - 24`.

The old code did:

```python
pred_test = hist_series.reindex(y_test.index)   # hist[t]  vs  y[t] = hub(t+24)
```

So `pred_test[t]` was the model's estimate of `hub(t)`, scored against
`hub(t + 24)`. Every deep-model number in v2 and v3 was a 24-hour-lagged
comparison.

### Why it produced such plausible-looking numbers

> **What:** the mathematical reason the wrong answers looked so reasonable.
> **The key insight:** comparing wind now against wind 24 hours later is *the definition
> of* the persistence baseline. So the broken comparison was not measuring the models at
> all — it was measuring persistence, four times over, with each model's own noise layered
> on top. That is exactly why all four landed just behind persistence in a neat row.
> **Why that was so convincing:** the ordering even looked sensible, with bigger models
> apparently doing worse — consistent with the "deep models need more data" folklore
> everyone already believed.

This is what made the bug survive two iterations. A good model's estimate of
`hub(t)` is close to `hub(t)`, so the error being measured was approximately

```
|hub(t) − hub(t+24)|
```

which *is the definition of 24-hour persistence error* (MAE 1.440 m/s). Every
deep model therefore landed in a tight band just around persistence — LSTM
1.475, N-BEATS 1.425, DLinear 1.519, TFT 1.643 — with the spread coming from
each model's own error stacked on top of the persistence floor. The result was
a table that looked like a meaningful ranking and told a tidy story ("deep
models can't beat persistence here"), when it was really measuring the same
persistence baseline four times with different noise added.

Doc 09 then explained that artefact with five plausible mechanisms. The
mechanisms are all real phenomena; none of them was what produced those
numbers.

### Verification

> **What:** the checks that confirmed the diagnosis rather than assuming it.
> **Why they were necessary:** "the numbers moved in the direction I hoped" is not
> evidence. The tests here demonstrate the shift is exactly 24 hours and that correcting
> it recovers the right answer on data where the right answer is known in advance.

Before changing anything, darts' index semantics were checked directly rather
than assumed:

```
requested start        : 2023-01-17 16:00
hist first index       : 2023-01-18 15:00     # start + horizon − 1
predict() from series ending 2023-01-17 15:00 → 16:00 … next-day 15:00
```

`start` is the first point of the first forecast *window*; with
`last_points_only=True` the first value returned lands at
`start + horizon − 1`, and each value is an honest `horizon`-ahead forecast.

### The fix

> **What:** convert predictions from "hour described" back to "hour predicted from" before
> scoring, and guard it with tests that a shifted result cannot pass.
> **Why the test design matters:** the original tests checked that outputs had the right
> shape and contained no missing values — both true of the broken version. The replacement
> asserts a *quality floor* on a signal with a 37-hour cycle, chosen deliberately so it
> does not divide evenly into 24. A model off by a day cannot fake its way past that.

Two changes in `src/stage1_deep.py`:

1. `_to_feature_time()` converts a backtest result from target time to feature
   time by subtracting the horizon, so it can be compared with `y` directly.
2. The backtest `start` is set to `split_ts + offset − horizon + 1`, so the
   first returned point lands exactly on `y_test.index[0]`'s target.

The information sets now match the tree model exactly: at feature time `t`
both XGBoost and the sequence models see data up to and including `t`, and
predict `hub(t + 24)`.

Regression coverage in `tests/test_stage1.py`:

- `test_to_feature_time_shifts_index_back_by_horizon` — unit test on the
  conversion, including the nowcast (`offset = 0`) case.
- `test_deep_forecast_predictions_are_horizon_aligned` — end-to-end guard on a
  37-hour periodic signal, chosen because it is not commensurate with the 24 h
  horizon, so persistence is weak and a correctly aligned model must beat it by
  a wide margin. A `horizon`-shifted model cannot pass.

---

## 2. What was implemented from doc 09's list

> **What:** the earlier analysis ended with five suggested fixes. Three were implementable
> and all three were built and measured.
> **Why the outcome is instructive:** two of the three made no difference or made things
> slightly worse. Only one helped. Since that analysis was explaining a bug rather than a
> real effect, its remedies were aimed at the wrong target — but building them anyway is
> what proved that, rather than leaving it as an assumption.

| # | Recommendation | Status |
|---:|:---|:---|
| 1 | Train longer — 200 epochs with early stopping on validation loss | Implemented |
| 2 | Feed the lag features as past covariates | Implemented (default on) |
| 3 | Larger LR sweep + warmup + cosine decay | Implemented |
| 4 | Multi-site training | Not possible — Anloga is the only site |
| 5 | 10+ years of data (2010–2023) | Download started; see §5 |

### Rec #1 — long budget with early stopping

> **What:** train far longer, but stop automatically when performance stops improving.
> **Result — the "train longer" part was pointless:** every model settled after about
> 20–25 passes, well under the old 30-pass limit. They were never undertrained.
> **What actually helped was the other half:** keeping the *best* version seen rather than
> whatever the last pass produced. Without that, stopping early hands back weights that
> are measurably worse than the best already found.

- The last `val_fraction` (0.15) of the train slice is held out as a validation
  tail. The validation series carries `input_chunk_length` hours of its own
  context so its first scored sample sits exactly at the boundary.
- `EarlyStopping(monitor="val_loss", patience=20, min_delta=1e-4)`.
- **Best-checkpoint restore.** Early stopping leaves the *last* weights loaded,
  which are by construction `patience` epochs worse than the best. darts'
  `save_checkpoints=True` writes a `best-{epoch}-{val_loss}` checkpoint, and
  the model is reloaded from it after fitting. Without this, early stopping
  actively hurts.
- Scalers are fit on the fitting slice only — never on validation or test.

### Rec #2 — lag features as past covariates

> **What:** hand the sequence models the same ready-made "wind N hours ago" columns the
> tree model uses.
> **Result — slightly harmful.** Tested directly on DLinear: test MAE **1.215 m/s** with
> the lag columns, **1.196 m/s** without. Lower is better, so removing them was marginally
> the better choice.
> **Why, in hindsight:** a sequence model already reads the last 48 hours directly, so
> these columns re-state information it can see, widening the input for no gain. The
> reasoning behind the recommendation was sound and the measurement still contradicted it.

`_exogenous_cols(..., include_target_lags=True)` now keeps the
`wind_speed_hub_lag*` and `wind_speed_hub_roll*` columns instead of dropping
them. These are strictly backward-looking (`add_rolling_features` shifts by 1
before rolling) and sit inside the input chunk, so they cannot leak. Covariate
count goes from 39 to 48. Set `include_target_lags: false` to reproduce v3.

### Rec #3 — LR sweep, warmup, cosine decay

> **What:** tune how fast each model learns, individually, instead of sharing one setting.
> **Why "learning rate" matters:** too high and training overshoots and never settles; too
> low and it crawls. It is usually the single most important training setting.
> **Result — the one recommendation that paid off.** The search picked a different rate
> for each of the four models, spanning a tenfold range. Any single shared default would
> have been wrong for at least two of them.

- `WarmupCosineLR` — linear warmup for `warmup_epochs`, then cosine decay to
  `min_lr_factor × lr`, stepped once per epoch. Implemented as a module-level
  class over a `functools.partial` of a module-level function so the fitted
  model stays picklable.
- `gradient_clip_val: 1.0` on every model (doc 09 flags this specifically for
  TFT).
- `lr_sweep` trains each candidate LR for a short budget with the scheduler
  disabled (a decaying LR would confound the comparison) and keeps the one with
  the lowest `val_loss`.

All of it is config-driven under `model.deep`, with `model.deep.overrides.<algo>`
deep-merging per-model settings on top.

---

## 3. Results

> **What:** the corrected scores, and the complete reversal of the previous conclusion.
> **The headline:** with the scoring fixed and every model given the same objective, **all
> five beat both naive baselines**, and they land within 0.101 m/s of one another. In the
> broken version, four of five *lost* to persistence and the spread was five times wider.
> **How to read this fairly:** the near-tie is the finding. The exact ordering is not —
> it rests on one random seed and one split, and the gaps are smaller than that noise.

Run: `python -m src.pipeline --config config.yaml --mode forecast --compare`,
2026-08-08 23:41 → 2026-08-09 04:58 (5 h 17 min wall, CPU-only, 12 threads).
Log: `outputs/logs_v4_bakeoff.log`.

### Stage 1 — hub wind speed at t+24 h

> **What:** each model's typical error when predicting wind speed a day ahead, before any
> power conversion.
> **Why judge here rather than on power:** wind-speed error is the honest measure of the
> machine learning. Converting to power afterwards adds the power curve's own non-linear
> distortion, which would obscure what the model itself is doing.
> **The control that proves the movement is real:** the tree model's score is unchanged to
> three decimals, because its code path never touched the buggy comparison. Everything
> that moved, moved for the expected reason.

| Rank | Model | Test MAE (m/s) | RMSE | R² | v3 MAE |
|---:|:---|---:|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** | 1.475 |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 | 1.143 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 | 1.425 |
| 4 | DLinear | 1.215 | 1.530 | 0.302 | 1.519 |
| 5 | TFT | 1.551 | 1.974 | −0.163 | 1.643 |
| — | Climatology | 1.282 | 1.617 | 0.220 | |
| — | Persistence, 25 h (as reported v1–v4) | 1.440 | 1.832 | −0.002 | |
| — | Persistence, 24 h (correct — see §4) | 1.399 | 1.791 | 0.043 | |

**Three of the four deep models now beat both persistence baselines, and the
LSTM beats the Optuna-tuned XGBoost.** XGBoost's 1.143 is byte-identical to v3,
which is the control: the tree path was untouched, so the entire deep-model
movement comes from the alignment fix plus the training changes.

### Stage 1+2 — end-to-end power (LSTM as Stage 1)

> **What:** the full pipeline's numbers — predicted wind carried through the power curve
> into predicted electricity.
> **Why they matter:** this is the form the wind-power literature reports, so it is what
> makes the project comparable to published work. The improvement is real but modest: a
> better wind forecast does not translate one-for-one into a better power forecast,
> because the power curve flattens above rated speed and is zero below cut-in.

| Variant | MAE (W) | RMSE (W) | R² | nMAE | nRMSE |
|:---|---:|---:|---:|---:|---:|
| physics_only | 446,962 | 607,452 | −0.071 | 14.65 % | 19.92 % |
| **two_stage** | **343,197** | **472,715** | **0.351** | **11.25 %** | **15.50 %** |
| direct | 371,247 | 479,103 | 0.334 | 12.17 % | 15.71 % |

Best two-stage result so far: nMAE 11.66 % (v3) → **11.25 %**, R² 0.337 →
0.351. The gap to the Energies 2021 ~8.8 % benchmark narrows from ~2.9 to
~2.5 points.

### Which recommendation actually did the work?

> **What:** separating the improvement caused by fixing the bug from the improvement
> caused by the three training changes.
> **Why the separation is essential:** all four changes shipped together, so without
> isolating them you could easily credit the wrong one — and then carry a false lesson
> into the next project. The answer is that the bug fix dominates; of the three
> recommendations only the learning-rate work contributed.

Almost none of them. The alignment fix did.

**Rec #1 (train 200 epochs) — moot.** Every model early-stopped far below its
cap: DLinear at 23, LSTM at 24, N-BEATS at 19 (of 100), TFT at 28 (of 60). The
premise that the v3 models were "undertrained at 30 epochs" is wrong; 30 epochs
was already past convergence. What v3 lacked was not epochs but *validation-based
stopping and best-checkpoint restore* — without those, 30 epochs of training
returns whatever weights epoch 30 happened to hold.

This also means the cosine schedule never completes: a 200-epoch anneal that
stops at epoch 23 has barely decayed. Matching the cap to observed convergence
(~60 epochs) would make the anneal meaningful and is worth a follow-up.

**Rec #2 (feed the lag features) — neutral to slightly harmful.** Ablated
directly on DLinear:

| DLinear | Test MAE | RMSE | R² |
|:---|---:|---:|---:|
| lags included (48 covariates) | 1.215 | 1.530 | 0.302 |
| lags withheld (39 covariates) | **1.196** | 1.537 | 0.295 |

Withholding them is marginally *better* on MAE. Doc 09's reasoning — that a
tree can put an explicit split on `lag12` while an RNN must learn the same
operation through its recurrent weights — is sound in the abstract, but the
sequence models already have the target history in the input chunk, so the lag
columns are redundant re-encodings that widen the covariate tensor for nothing.

**Rec #3 (LR sweep + warmup + cosine) — real but small, and it matters most
where it was cheapest to skip.** The sweep picked a different LR for every
model, and the spread in validation loss was material:

| Model | 3e-4 | 1e-3 | 3e-3 | chosen |
|:---|---:|---:|---:|---:|
| DLinear | **0.00766** | 0.00864 | 0.01457 | 3e-4 |
| LSTM | 0.01006 | **0.00955** | 0.00998 | 1e-3 |
| N-BEATS | 0.01201 | 0.00837 | **0.00791** | 3e-3 |
| TFT | 0.42856 | **0.41623** | — | 1e-3 |

A single shared default would have been wrong for two of the four. Note the
sweep is biased toward small LRs — it evaluates candidates over a short budget
with the scheduler off, which favours whatever descends fastest early.

### Why TFT is still the only model that loses

> **What:** after the alignment fix, four of five models cleared the baselines and one
> still did not.
> **Why that residue was worth chasing:** having just been caught believing a plausible
> wrong number, taking "TFT is simply unsuited to this problem" on faith would have
> repeated the mistake. The real cause turned out to be a library default nobody had set
> — described in the next section.

Not undertraining — it early-stopped at 28 of 60 epochs, and the run took
4 h 33 min. The concrete cause is an objective mismatch that doc 09 did not
identify:

**`darts.models.TFTModel` defaults to `likelihood=QuantileRegression()` with 17
quantiles**, while BlockRNN, N-BEATS and DLinear all train on plain MSE. TFT is
therefore solving a substantially harder problem — fitting 17 quantile heads of
a predictive distribution — on the same 28k samples, and its point forecast is
just the median. This is also why its `val_loss` (0.416) is two orders of
magnitude above the others (~0.008): it is quantile loss, not MSE, and the two
are not comparable.

**This was tested, and it accounts for essentially all of the gap** — see §3b.

## 3b. v4b — TFT on the same objective as everyone else

> **What:** the largest model was answering a harder question than its rivals, purely
> because of an unset default, and was then graded on only part of its answer.
> **In plain terms:** the others were asked "what will the wind speed be?" TFT was asked
> "what is the full range of wind speeds this could be?" — 17 separate percentiles — from
> the same data and the same time budget. Only the middle answer was scored.
> **The fix and its size:** naming the objective explicitly took TFT's test MAE from
> **1.551 m/s to 1.171 m/s** — a 0.380 m/s improvement, where lower is better. That single
> configuration line was worth more than any architectural decision in the project, and
> nothing in the training output made the problem visible.

Run: `outputs/logs_v4b_tft_mse.log`, 2026-08-09 09:00 → 18:57.
Config: `model.deep.loss_fn: mse` (see §2, rec #3 addendum). darts only
substitutes `QuantileRegression()` when **both** `likelihood` and `loss_fn` are
`None`, so naming the loss makes TFT deterministic. It is a no-op for the other
three models, which were already on MSE — the whole zoo now shares one
objective.

| TFT objective | Test MAE | RMSE | R² | Beats persistence? |
|:---|---:|---:|---:|:---:|
| `QuantileRegression` (darts default, 17 quantiles) | 1.551 | 1.974 | −0.163 | No |
| **`MSELoss` (deterministic)** | **1.171** | **1.498** | **0.330** | **Yes** |

The objective mismatch accounts for essentially the entire deficit. TFT goes
from the worst model in the zoo — the only one losing to persistence — to
mid-field, 0.057 m/s off the winner. The v4 conclusion "TFT is badly suited to
this problem" was wrong; what was actually measured is that
*TFT-fitted-to-17-quantiles is mediocre at point forecasting*, which is close to
a tautology.

Both runs early-stopped at essentially the same point (28 vs 29 epochs of 60),
so this is not a training-budget effect.

> **Timing is not comparable between the two runs.** The MSE run took 596 min
> against 273 min for the quantile run, despite having one output head instead
> of 17. The machine was in interactive use during the MSE run and this is a
> 15 W laptop part, so the difference is almost certainly thermal throttling
> (~8 min/epoch vs ~18.6 min/epoch), not anything architectural. Do not quote
> these two wall times against each other.

### Final v4 standings (all five models, aligned, uniform MSE objective)

> **What:** the definitive table — all five models, correctly scored, all answering the
> same question.
> **What it supports:** every model beats both naive baselines, and the whole field spans
> 0.101 m/s — less than the distance from the *worst* of them to persistence. Architecture
> choice is not the lever here.
> **What it does not support:** any claim that one model beats another. Use this table for
> the near-equivalence finding, not for a ranking.

| Rank | Model | Test MAE (m/s) | RMSE | R² |
|---:|:---|---:|---:|---:|
| 1 | **LSTM** | **1.114** | **1.429** | **0.390** |
| 2 | XGBoost (Optuna) | 1.143 | 1.461 | 0.364 |
| 3 | N-BEATS | 1.157 | 1.483 | 0.343 |
| 4 | TFT (MSE) | 1.171 | 1.498 | 0.330 |
| 5 | DLinear | 1.215 | 1.530 | 0.302 |
| — | Climatology | 1.282 | 1.617 | 0.220 |
| — | Persistence, 24 h (correct) | 1.399 | 1.791 | 0.043 |
| — | Persistence, 25 h (as reported v1–v4) | 1.440 | 1.832 | −0.002 |

**Every model now beats both persistence baselines**, and all five sit within
0.10 m/s of each other. Against v3 — where four of five lost to persistence and
the spread was 0.50 m/s — the picture is completely inverted. The spread between
best and worst model (0.101 m/s) is smaller than the gap from the worst model to
persistence (0.184 m/s), which is the real headline: **on this problem the
architecture is close to irrelevant compared with getting the evaluation and the
objective right.**

### Revised verdict on doc 09

> **What:** going back through the earlier analysis point by point and marking each claim
> as supported, wrong, or right-for-the-wrong-reason.
> **Why do it explicitly:** that document is kept unedited for provenance, so without a
> verdict attached a reader could easily take its confident, well-argued claims at face
> value. Of five proposed causes, one holds, one was right by accident, and three do not
> survive.

| Doc 09 claim | Verdict |
|:---|:---|
| Data volume too small for deep models | Overstated — LSTM wins at 28k samples |
| Training budget too small (30 epochs) | Wrong — models converge in ~20–25 epochs; the missing piece was early stopping + best-checkpoint restore, not epochs |
| Feature engineering handed XGBoost the win | Wrong — giving the deep models the same lags is neutral-to-harmful |
| Problem shape favours tabular models | Not supported — the LSTM beats the tuned tree and all four sequence models beat persistence |
| TFT has architecture mismatch | Right conclusion, wrong mechanism — the default 17-quantile objective, not static covariates. Fixing it recovers 0.380 m/s (§3b) |
| Signal-to-noise ceiling ~1 m/s | Still plausible and unchanged; LSTM at 1.114 is closer to it |

The one claim that survives intact is the last one. The closing line of doc 09 —
"for structured tabular time series with modest data and strong hand-engineered
features, gradient-boosted trees are the right tool" — is not supported by these
numbers. A defensible restatement for the write-up:

> With identical information sets, a uniform MSE objective and equivalent
> training care, a single-layer LSTM (test MAE 1.114 m/s, R² 0.390) modestly
> outperforms a 50-trial Optuna-tuned XGBoost (1.143, 0.364) for day-ahead
> single-site wind-speed forecasting, with N-BEATS (1.157), TFT (1.171) and
> DLinear (1.215) close behind. All five beat 24-hour persistence (1.399). The
> spread across architectures (0.101 m/s) is smaller than the gap from the
> weakest of them to persistence (0.184 m/s), so the honest headline is that
> architecture choice matters far less on this problem than correct temporal
> alignment, a comparable training objective, and validation-based stopping.

### Caveats on the ranking

> **What:** why the model ordering in the table above should not be treated as a result.
> **The core issue:** every number comes from one random seed on one chronological split.
> Re-running with a different seed would reshuffle models separated by hundredths of a
> m/s. Running five seeds and reporting a mean and spread is the cheap fix, and it is
> listed as a next step rather than done.

- **Single seed, single split.** Every number is one run at `seed=42` on one
  chronological 80/20 split. The LSTM's 0.029 m/s margin over XGBoost is well
  inside what seed variance could plausibly produce. "The LSTM beats XGBoost"
  should not be claimed without a multi-seed repeat; "they are near-equivalent,
  and the v3 gap was measurement error" is fully supported.
- **The LR sweep is biased toward small learning rates.** Candidates are scored
  over a short budget with the scheduler off, which favours whatever descends
  fastest early rather than what trains best over a full run.
- **The cosine schedule never completes.** With a 200-epoch cap and early
  stopping at ~23, the LR barely decays. Matching each cap to observed
  convergence (~60 epochs) would make the anneal meaningful and is untested.

---

## 4. Two adjacent defects found while doing this

> **What:** two further bugs uncovered while fixing the main one.
> **Why they are grouped here:** both are the same species — silent failures that produce
> usable-looking output. One saved model files that contained no actual model; the other
> made the benchmark slightly easier to beat than advertised. Neither raised an error.

**`joblib.dump` silently discards a darts model's weights.** darts'
`TorchForecastingModel.__getstate__` deliberately drops the LightningModule and
Trainer, so `save_stage1` was writing a `stage1.pkl` containing a model shell
with no weights. It never mattered before because XGBoost always won the
bakeoff and this path was never taken with a deep model. Now that a deep model
can win, `save_stage1`/`load_stage1` detect darts models and use darts' own
serializer, writing a `stage1.darts.pt` sidecar.
Covered by `test_save_load_round_trip_darts_model`.

**The report hardcoded the config's algorithm name.** In `--compare` mode the
reported Stage 1 is the *winner*, which need not be `config.model.stage1_algo`.
`write_report` now takes an explicit `stage1_algo_label`.

### The persistence baseline is a 25-hour persistence, not 24

> **What:** the "predict tomorrow will be like now" benchmark actually looked back 25
> hours, not 24.
> **Why it happened:** the raw current wind-speed column is excluded from the model's
> inputs, so the baseline reached for the nearest available substitute — the value from an
> hour ago. Against a target 24 hours ahead, that is a 25-hour gap.
> **Why it matters:** an older reading is a worse predictor, so the bar was about 2.9% too
> low in every iteration. The corrected baseline is test MAE **1.399 m/s** rather than
> **1.440 m/s** — a lower number means a *harder* bar to beat, so the real baseline is
> tougher than the one quoted throughout v1–v4. **1.399 m/s is the number to quote.** The
> original was left in place so v1–v4 remain comparable to each other.

Found but **deliberately not changed** — see below.

`PersistenceForecaster` predicts from the `wind_speed_hub_lag1` column, because
`build_feature_frame` excludes the raw `wind_speed_hub` column from `X`. At
feature time `t` that column holds `hub(t − 1)`, and it is scored against
`y[t] = hub(t + 24)`. So the baseline every iteration has reported is a
**25-hour** persistence. True 24-hour persistence — `hub(t)` → `hub(t + 24)` —
is measurably harder:

| Baseline | MAE (m/s) | RMSE | R² |
|:---|---:|---:|---:|
| 25 h — `hub(t−1)`, as reported in v1–v4 | 1.440 | 1.832 | −0.002 |
| 24 h — `hub(t)`, the correct comparison | **1.399** | **1.791** | **0.043** |

The bar has been 0.041 m/s (2.9 %) too low throughout. Consequences for the
existing write-up:

- XGBoost's improvement over persistence is 18.3 %, not 20.6 %.
- v3's headline "**N-BEATS** narrowly beats persistence (1.425 vs 1.440)" is
  wrong twice over — that number came from the misaligned scorer, and 1.425
  loses to a correct 24 h persistence anyway.

This was **not** changed during the v4 run, on purpose: the 25 h baseline is
what v1, v1.5, v2 and v3 were all measured against, and swapping it mid-stream
would silently break comparability across the whole of `RESULTS.md`. The v4
tables below therefore quote both numbers. Switching `PersistenceForecaster` to
the true 24 h baseline — and restating the earlier iterations against it — is a
clean follow-up, not something to smuggle into this run.

---

## 5. Rec #5 — the 10-year ERA5 extension

> **What:** the fifth recommendation was "get 10+ years of data", and a long download was
> started to do exactly that.
> **How it ended:** abandoned partway. In August 2026 the assumption behind it was finally
> measured and did not hold — performance stops improving after about 1.6 years of
> history, so the extra decade was worth roughly 1%.
> **Where the effort went instead:** obtaining actual weather *forecasts*, which the model
> had never had. See [11_nwp_forecast_features.md](11_nwp_forecast_features.md).

`config_10yr.yaml` plus `src/data_ingest_extend.py` extend the window to
2010–2023 (~123k hourly rows, up from 35k). The downloader is separate from
`load_weather` because that function assembles a parquet at the end and aborts
if any core month is missing — unhelpful for a resumable multi-hour job that
has to clear the CDS queue.

    python -m src.data_ingest_extend --config config_10yr.yaml

Downloads are cached per (year, month) per request family, so the 48 months
already on disk are skipped and re-runs resume. Re-run until it reports zero
missing months, then pass `config_10yr.yaml` to the pipeline.

## 6. Rec #4 — multi-site training

> **What:** the suggestion to train on several Ghanaian sites at once so the model sees
> more examples.
> **Why it does not work here:** neighbouring points on the ERA5 grid sit inside the same
> weather system, so they would supply near-duplicate rows rather than new information —
> more data in volume, none in substance. Genuinely independent sites would need to be far
> enough apart to experience different weather, which is a separate multi-week data
> exercise.

Not actionable. Pooling across sites amortises the sample-efficiency problem,
but Anloga is the only site in this study and ERA5 grid neighbours are not
independent sites — at 0.25° they are the same synoptic cell and would add
correlated rows rather than new information. This would need a genuine
multi-site Ghanaian campaign to test.
