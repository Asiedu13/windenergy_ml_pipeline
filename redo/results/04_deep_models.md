# 04 — Deep models

> ⚠️ **PARTIAL — this run is still in progress.** 12 of 15 fits complete as of
> 2026-08-12 07:33. **Only TFT remains** (×3 seeds, ~6 h each; ETA ~2026-08-13 01:00).
> Every other model has all three seeds. The seed-sensitivity finding is solid.

> **What:** the four deep architectures on the same window and target, over three seeds.
> **Why separate from [02](02_classical_vs_ml.md):** they are on a **different evaluation
> basis** and mixing them into that table would be exactly the unlabelled cross-basis
> comparison that produced a false conclusion in this project once already.
> **Source:** `outputs/reports/bakeoff_deep.csv`.

## ⚠️ Different evaluation basis — do not merge these tables

| | [02 classical vs ML](02_classical_vs_ml.md) | This page |
|:---|:---|:---|
| Evaluation | **4 rolling-origin folds** | **1 chronological split** (last 20%) |
| Seeds | 5 (stochastic models) | 3 |
| Feature set | `study` and `operational` | `study` only |
| Horizon | **24 h** | **24 h** |

The reason is cost: TFT alone took 5 h 54 m for a single fit, so four rolling folds would be
~24 h per seed for that one model. **XGBoost is re-run here on the same single split** so
the deep scores have a like-for-like anchor rather than being silently compared against
phase A's fold means.

## Results — 24 h horizon, single split, `study` features

| Model | Seeds done | Mean MAE (m/s) | Seed sd | Mean R² | Minutes/fit |
|:---|---:|---:|---:|---:|---:|
| **`xgboost`** *(anchor)* | 3/3 | **0.7329** | **0.0035** | 0.6615 | 0.1 |
| `lstm` | 3/3 | 0.7701 | 0.0276 | 0.6217 | 4.8 |
| `nbeats` | 3/3 | 0.8234 | **0.0361** | 0.5646 | 20.7 |
| `dlinear` | 3/3 | 0.8244 | 0.0115 | 0.5837 | 1.1 |
| `tft` | **0/3** | *pending* | — | — | ~350 |

Per-seed detail:

| Model | seed 42 | seed 7 | seed 1234 | range |
|:---|---:|---:|---:|---:|
| `xgboost` | 0.7299 | 0.7367 | 0.7321 | 0.0068 |
| `lstm` | 0.7572 | **0.8019** | 0.7514 | **0.0505** |
| `nbeats` | 0.8145 | **0.8630** | 0.7925 | **0.0705** |
| `dlinear` | 0.8113 | 0.8292 | 0.8328 | 0.0215 |

**N-BEATS and DLinear are separated by 0.0010 m/s on the mean and overlap heavily across
seeds** — on this evidence they are tied, and any ranking between them is noise.

## Finding 1 — deep models are ~8× more seed-sensitive than trees

| Model | Seed sd | Range across seeds | vs XGBoost |
|:---|---:|---:|---:|
| `xgboost` | **0.0035** | 0.0068 | 1× |
| `dlinear` | 0.0115 | 0.0215 | 3× |
| `lstm` | 0.0276 | **0.0505** | **8×** |
| `nbeats` | **0.0361** | **0.0705** | **10×** |

LSTM's spread across three seeds (0.7514 → 0.8019) is **0.0505 m/s** — larger than its
entire mean gap to XGBoost (0.0372). **Change the seed and the ranking between LSTM and
XGBoost could plausibly flip in a single run.** N-BEATS is worse still: a 0.0705 range, wide
enough to overlap both LSTM above it and DLinear below it.

**This is a finding no single-seed experiment could produce**, and every prior architecture
conclusion in this project was single-seed. It is direct evidence for
[stage 5](../05_stage5_baselines_and_protocol.md)'s insistence that multi-seed belongs in
the protocol rather than in a follow-up list.

## Finding 2 — the earlier 18% claim does not transfer

[`docs/11`](../../docs/11_nwp_forecast_features.md) records a five-model bakeoff on **NASA
POWER** truth where XGBoost led the best deep model by **18%** (0.678 vs 0.827). On **ERA5**
truth the same comparison gives:

| Truth source | XGBoost | Best deep | Gap |
|:---|---:|---:|---:|
| NASA POWER, 22 mo | 0.678 | 0.827 (tft) | **18%** |
| **ERA5, 22 mo** | **0.7329** | **0.7701** (lstm) | **5.1%** |

**The direction holds; the magnitude does not.** Trees still win, but by roughly a third of
what the NASA POWER run suggested. Anywhere the 18% figure is quoted, this belongs beside
it — and with TFT still pending, even the 5.1% is provisional.

Why the two disagree is not established. NASA POWER is coarser (~55 km) and hour-averaged,
making its target smoother; whether that favours one model class is a hypothesis, not a
finding.

## Finding 3 — cost per unit of accuracy is stark

| Model | Minutes/fit | MAE | Relative cost vs XGBoost |
|:---|---:|---:|---:|
| `xgboost` | 0.1 | 0.7329 | 1× |
| `dlinear` | 1.1 | 0.8244 | ~10× |
| `lstm` | 4.8 | 0.7701 | ~42× |
| `nbeats` | 20.7 | 0.8234 | ~185× |
| `tft` | ~350 | pending | ~3,000× |

The best deep model costs **~42× the training time to finish 5% behind**. For a single-site
day-ahead forecast this is not a close decision, and it supports
[stage 6](../06_stage6_modelling.md)'s rule: deep learning is the option to justify, not the
default.

## Caveats

- **Partial run — TFT only.** It is the architecture with the strongest prior for this
  problem: it handles known-future covariates natively, which is exactly what an NWP forecast
  is, and it won the deep bracket on NASA POWER truth. Its absence is the biggest gap here,
  and it is the one model that could narrow the 5.1% figure.
- **Single split.** Section [02](02_classical_vs_ml.md) shows fold difficulty varies 46%
  across the year; these numbers come from one quarter-and-a-bit and carry that exposure.
- **`study` features only.** The operational arm ([03](03_operational_vs_study.md)) has not
  been run for deep models.
- **Three seeds is few.** Enough to reveal that variance differs by ~8× between model
  classes; not enough for a tight interval on any single model.
