# 13 — Does it generalise? The 24 h ladder at 17 Ghanaian sites

> **What:** the published 24 h bakeoff, unchanged, run at Ghana's 16 regional capitals plus
> Anloga — same folds, same seeds, same Diebold–Mariano test, same report formatter.
> **Why it matters:** three documents recorded the generalisation question as deferred —
> [`docs/07`](../../docs/07_findings_and_analysis.md) threat 2 and thesis §4.14.4 — and thesis
> §4.9.3 made it *falsifiable*. This is the verdict.
> **Protocol:** [01](01_protocol.md). **Pre-registration:**
> [`docs/13`](../../docs/13_multi_site_extension.md), written before any of this ran.
> **Source:** `outputs/sites/*/reports/bakeoff.csv` (17 files).

⚠️ **Window: 2024-03-01 → 2025-12-31.** Fixed by data availability, not choice — the
Open-Meteo previous-runs archive is all-null before 2024-03. **No figure here is comparable to
the v1–v4 tables in [`../../RESULTS.md`](../../RESULTS.md), which are on 2020–2023.** It *is*
comparable to [02](02_classical_vs_ml.md) and [03](03_operational_vs_study.md), which share
this window.

**The folds are identical at all 17 sites — verified, not assumed.** Every site produced
15,912 rows and the same four test blocks, checked by extracting every fold line from the batch
log and confirming it reduces to exactly four distinct values:

| Fold | Train rows | Test rows | Test period |
|:---|---:|---:|:---|
| fold1 | 7,004 | 2,227 | 2024-12-24 → 2025-03-27 |
| fold2 | 9,231 | 2,227 | 2025-03-27 → 2025-06-28 |
| fold3 | 11,458 | 2,227 | 2025-06-28 → 2025-09-29 |
| fold4 | 13,685 | 2,227 | 2025-09-29 → 2025-12-30 |

Definitions and rationale are in [01](01_protocol.md) and not restated here. **This table is
what licenses the cross-site comparison**: a per-site difference in fold boundaries would make
every column below incomparable, and nothing in the per-site reports would have revealed it.
Each block is roughly a quarter, so fold numbers read as seasons — which is why the per-fold
spread in §"Seasonal variation" below is a seasonal statement rather than noise.

## The question

Every headline number in this project came from one coastal site. The thesis said so, and said
what might break:

> *"It is one site and one horizon. At a location with weaker NWP skill, or at a shorter
> horizon where persistence is strong, reanalysis features could matter considerably more."*
> — thesis §4.9.3

Twelve of the sixteen capitals are low-wind interior locations, four of them in the northern
savannah. If the Anloga conclusions were an artefact of a windy coastal site, this is where it
shows.

## Result — 24 h horizon, 4 rolling folds, 5 seeds, `study` arm

| site | region | mean wind | best | MAE | sd | **MAE / wind** | R² | skill vs pers. |
|:---|:---|---:|:---|---:|---:|---:|---:|---:|
| accra | Greater Accra ᶜ | 4.09 | xgboost | 0.5453 | 0.0617 | 0.133 | 0.588 | +0.379 |
| koforidua | Eastern | 3.58 | xgboost | 0.5467 | 0.0616 | 0.153 | 0.604 | +0.376 |
| ho | Volta | 3.32 | **ridge** | 0.5663 | 0.0657 | 0.170 | 0.598 | +0.346 |
| cape_coast | Central ᶜ | 3.99 | **ridge** | 0.5694 | 0.0733 | 0.143 | 0.565 | +0.355 |
| sekondi_takoradi | Western ᶜ | 4.04 | xgboost | 0.5952 | 0.0669 | 0.147 | 0.544 | +0.354 |
| kumasi | Ashanti | 3.59 | xgboost | 0.5983 | 0.0573 | 0.167 | 0.614 | +0.338 |
| sefwi_wiawso | Western North | 3.38 | xgboost | 0.6103 | 0.0784 | 0.180 | 0.577 | +0.315 |
| goaso | Ahafo | 3.48 | xgboost | 0.6377 | 0.0640 | 0.183 | 0.562 | +0.307 |
| sunyani | Bono | 3.65 | xgboost | 0.6663 | 0.0626 | 0.183 | 0.601 | +0.314 |
| dambai | Oti | 3.31 | xgboost | 0.6879 | 0.0620 | 0.208 | 0.537 | +0.312 |
| techiman | Bono East | 3.92 | xgboost | 0.6992 | 0.0748 | 0.178 | 0.634 | +0.306 |
| damongo | Savannah | 3.66 | **ridge** | 0.7712 | 0.0569 | 0.211 | 0.556 | +0.271 |
| *anloga* | *Volta — reference, not a capital* ᶜ | *6.24* | *xgboost* | *0.8006* | *0.0986* | ***0.128*** | *0.589* | *+0.388* |
| tamale | Northern | 4.16 | **ridge** | 0.8479 | 0.0536 | 0.204 | 0.587 | +0.304 |
| nalerigu | North East | 4.30 | **ridge** | 0.8586 | 0.0807 | 0.200 | 0.547 | +0.307 |
| bolgatanga | Upper East | 4.39 | **ridge** | 0.8601 | 0.0923 | 0.196 | 0.566 | +0.313 |
| wa | Upper West | 4.29 | xgboost | 0.9031 | 0.0457 | 0.211 | 0.567 | +0.307 |

ᶜ coastal. Mean wind is ERA5 100 m, m/s, over the same window.

## What this establishes

> **The Anloga conclusions generalise across Ghana's climatic zones.**

**1. The sanity floor holds everywhere — 34/34 site-arms** (17 sites × 2 feature sets)**.**
Every site's best model beats **both** persistence and climatology on its own folds, and nowhere
narrowly: the weakest margin is damongo on the `operational` arm, still **+24.6%** over
persistence (**+27.1%** on `study`). Zero site failures.

**2. Skill is far more stable than error.** Across coastal rainforest to Sahel:

All on the `study` arm, matching the table above.

| | min | max | median | spread |
|:---|---:|---:|---:|---:|
| MAE (m/s) | 0.5453 | 0.9031 | 0.666 | **1.66×** |
| MAE / mean wind | 0.128 | 0.211 | 0.180 | 1.65× |
| skill vs persistence | 0.271 | 0.388 | 0.314 | 1.43× |
| **R²** | **0.537** | **0.634** | **0.577** | **1.18×** |

*Skill is `mean(1 − MAE/MAE_pers)` across folds — the definition `bakeoff.md` prints. Taking the
ratio of fold-averaged MAEs instead gives 0.274–0.393; the two differ in the third decimal and
must not be mixed in one table.*

R² varies by less than 0.1 across seventeen sites spanning ~800 km and three climatic zones.
That stability, not any individual MAE, is the finding.

**3. The ceiling argument replicates 17 times, in both directions.** This is the part that
transfers most cleanly from [02](02_classical_vs_ml.md) and [05](05_ceiling_and_information.md).

| Comparison | Range across 17 sites | Reading |
|:---|:---|:---|
| `mos_multi` (21-param OLS) vs `xgboost` | **+2.17% … +7.46%**, median 5.16% | the problem is **not model-limited** anywhere |
| `mos_single` (one calibrated NWP model) vs `xgboost` | **+14.56% … +30.58%**, median ~18% | but the pipeline is **not redundant** either |

The two together are more informative than either alone. A 21-parameter linear fit reaches
within a few percent of gradient boosting at **every** site, so [05](05_ceiling_and_information.md)'s
conclusion — the binding constraint is information, not model class — is not an Anloga artefact.
Yet multi-model combination plus ML still beats bias-correcting a single NWP model by a median
~18%, so the combination earns its place.

**Practical consequence:** further model sophistication has almost nothing left to win here;
better *inputs* (higher-resolution NWP, more ensemble members, on-site observation) are where
the remaining error lives. That is [stage 4](../04_stage4_ceiling_analysis.md)'s
ceiling-before-model-selection rule, now evidenced across 17 sites rather than one.

**4. The operational result survives, with a caveat.** Dropping every latency-blocked
reanalysis feature costs **under 1.3% at 15 of 17 sites**. [03](03_operational_vs_study.md)'s
"free in practice" holds broadly. But it is no longer *free* — see below.

## ⚠️ Raw MAE is not comparable across these sites

MAE spans a factor of 1.66, and most of that is how windy a site is, not how predictable.
Normalised, the ranking **inverts at the top**: Anloga has the second-worst raw MAE in the
study (0.8006) and the **best** normalised error (0.128), because its mean wind is 6.24 m/s
against 3.3–4.4 everywhere else — and its cell is 88% water
([`docs/13` §3a](../../docs/13_multi_site_extension.md)), which smooths the series further.

**A bare MAE ranking would read "the model is worst at Anloga", which is the opposite of what
the data says.** Every cross-site table must carry mean wind or the normalised column.

## Seasonal variation — the headline MAE hides a third of the error

Every MAE in the table above is a mean over four folds, and the folds are not equally hard.
`xgboost` on the `study` arm, per fold, mean over the five seeds:

| site | fold1 (Jan–Mar) | fold2 (Apr–Jun) | fold3 (Jul–Sep) | fold4 (Oct–Dec) | spread |
|:---|---:|---:|---:|---:|---:|
| accra | 0.5130 | **0.6458** | 0.4891 | 0.5331 | 32.0% |
| anloga | 0.7814 | **0.9570** | 0.6956 | 0.7684 | **37.6%** |
| sefwi_wiawso | 0.6762 | 0.6869 | 0.5006 | 0.5773 | 37.2% |
| ho | 0.5698 | 0.6695 | 0.4893 | 0.5416 | 36.8% |
| sekondi_takoradi | 0.5923 | 0.6937 | 0.5107 | 0.5840 | 35.8% |
| techiman | 0.8043 | 0.7072 | 0.5994 | 0.6859 | 34.2% |
| koforidua | 0.5561 | 0.6385 | 0.4763 | 0.5159 | 34.1% |
| sunyani | 0.7409 | 0.6916 | 0.5734 | 0.6595 | 29.2% |
| goaso | 0.7156 | 0.6763 | 0.5550 | 0.6041 | 28.9% |
| bolgatanga | 0.7493 | 0.9058 | 0.8492 | 0.9550 | 27.4% |
| kumasi | 0.6428 | 0.6511 | 0.5114 | 0.5878 | 27.3% |
| nalerigu | 0.7803 | 0.8625 | 0.8429 | 0.9820 | 25.8% |
| cape_coast | 0.5374 | 0.6696 | 0.5347 | 0.5450 | 25.2% |
| dambai | 0.7897 | 0.6446 | 0.6410 | 0.6763 | 23.2% |
| damongo | 0.8358 | 0.7281 | 0.7592 | 0.7811 | 14.8% |
| wa | 0.9348 | 0.8442 | 0.8776 | 0.9558 | 13.2% |
| tamale | 0.8724 | 0.8442 | 0.7877 | 0.8884 | **12.8%** |

**A single-split evaluation would have been misleading by up to a third.** At Accra the quarterly
MAE runs 0.4891 to 0.6458; a split landing on fold 3 reports 0.489 and is 24% optimistic against
the four-fold mean of 0.545. This is the fold-difficulty argument of [01](01_protocol.md)
reproduced independently at 17 sites.

**An inversion worth noting — and not predicted.** The spread is *smallest* at the sites with
the largest errors: tamale 12.8%, wa 13.2%, damongo 14.8% are the three northern sites, while
the widest spreads are southern (anloga 37.6%, sefwi_wiawso 37.2%, ho 36.8%). **The north is
harder but steadier; the south is easier but far more seasonal.** Exploratory — not
pre-registered, and a mechanism is not established here.

Two consequences:

- **Any deployed interface must express a season-aware error, not one figure.** A constant
  "±0.55 m/s" would be wrong for a quarter of the year at most southern sites.
- Note this interacts with thesis §4.14.4's record that the **quantile intervals are
  uncalibrated**. Seasonal error variation of this size is precisely what an uncalibrated
  interval will misstate, and recalibration is the prerequisite for quoting uncertainty at all.

## Thesis §4.9.3 gets a verdict, and it is a split one

The prediction was that reanalysis features would matter more where NWP skill is weaker. The
pre-registered criterion ([`docs/13` §6b](../../docs/13_multi_site_extension.md)) was *refuted
if degradation stays ≤ 0 at all 16*.

| | |
|:---|:---|
| Sites with a positive penalty | **15 of 17** (techiman 0.00%, sunyani −0.06%) |
| Anloga reference | **+0.03%** |
| Largest | **damongo +3.45%**, **tamale +2.55%**, cape_coast +1.61% |
| Sign test, 16 non-zero sites | **p ≈ 0.0003** |
| Penalties exceeding that site's own seed sd | **0 of 17** |

**The direction is solid and lands exactly where the thesis pointed — the northern savannah.
No individual site's magnitude clears its own noise floor.**

⚠️ **The pre-registration never quantified "materially more".** That is a defect in the
criterion's wording, recorded rather than resolved in whichever direction suits the result.
Under a sign-test reading §4.9.3 is supported; under a per-site significance reading it is not
reached. The defensible claim is the narrow one: **dropping reanalysis features is consistently
but slightly costly away from Anloga, with the cost concentrated in the north.**

## Two exploratory findings — not pre-registered

Hypotheses for a future study, not results this one tested.

**1. `ridge` beats both gradient-boosting models at 6 of 17 sites** — ho, cape_coast, and **4
of the 5 far-northern** sites (damongo, tamale, nalerigu, bolgatanga; `wa` goes to xgboost).
Three independent signals point the same way in that region: ridge wins, the ERA5 penalty is
largest, and the MOS gap is *narrowest* (nalerigu 2.17%, bolgatanga 2.44%). **The north appears
to favour simpler, more linear models.** Nothing in the pre-registration anticipated a
model-family split by climatic zone.

**2. Climatology beats persistence at 13 of 17 sites, and the four exceptions are exactly the
four northernmost** (tamale, nalerigu, bolgatanga, wa). Even the reference ladder reorders by
zone.

Both would need their own pre-registration to be claims. They are recorded because suppressing
an unpredicted pattern is worse than labelling it.

## What this does *not* establish

- **Nothing about resource or viability.** Prediction 4 (the capacity-factor floor) is
  **unmeasured** — the long archive needed for the monthly climatology is at 144 of 576 files
  ([`docs/13` §10](../../docs/13_multi_site_extension.md)). No site in this document has a
  capacity factor, and none should be quoted from it.
- **It is still one horizon.** 24 h only. §4.9.3's other half — *"or at a shorter horizon where
  persistence is strong"* — is untested here; [07](07_horizon_sweep.md) sweeps horizons at
  Anloga alone.
- **It is still reanalysis truth, not observations.** Every site is scored against ERA5, so
  this measures agreement with a model, not with a mast. No Ghanaian site in this study has
  SCADA or met-mast data.
- **Anloga's cell is 88% water and was deliberately not moved.** Its *forecast* metrics stay
  valid — truth and features come from the same cell — but it is not a land site, and the
  comparison to a land farm is flattered. See [`docs/13` §3a](../../docs/13_multi_site_extension.md).
- **Deep models were not run.** Excluded by pre-registration: 12.9 h × 17 ≈ 206 h against a
  measured 5.1% deficit ([04](04_deep_models.md)).
- **The 3×3 spatial patch bought nothing.** The regional grid revives ten previously degenerate
  columns, and the feature mask discards them by name anyway; measured, reviving them moves
  every score less than its own seed sd. See [`docs/13` §4d](../../docs/13_multi_site_extension.md).
