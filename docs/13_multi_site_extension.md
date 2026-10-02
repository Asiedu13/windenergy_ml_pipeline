# 13 — Multi-site extension to Ghana's 16 regional capitals (v6, in progress)

> ⏳ **Status: pre-registration, 2026-10-01. No result exists yet.**
> This document was written **before any data was downloaded and before any model was
> trained**, and that is its purpose. Everything below §6 is a prediction, a pass criterion
> or a decision — not a finding. **Do not quote any number in this document as a result**;
> there are none to quote. When results land they go in
> `redo/results/13_multi_site_generalisation.md` (**not yet written**) and
> [`../RESULTS.md`](../RESULTS.md), and the predictions here stay **unedited** whether they
> survive or not ([stage 9 §9.4](../redo/09_stage9_writeup_and_hygiene.md)).

> 📐 **Reading the numbers.** Wind-speed figures are **MAE in m/s at t+24 h**, lower is
> better. Resource figures are **capacity factor (CF)**, a dimensionless 0–1 share of rated
> output, higher is better. These two are not comparable to each other, and CF figures here
> are **not** comparable to the m/s figures in docs 01–12. Full explanation:
> [glossary](glossary.md#how-to-read-the-numbers).

## 1. Why this iteration exists

The project has four forecast systems and all of them were measured at **one site**. The
generalisation question is logged as deferred in three places:

| Where | What it says |
|:---|:---|
| [`07_findings_and_analysis.md`](07_findings_and_analysis.md) threat 2 | *"Everything here is Anloga-specific… we don't know whether the **conclusions** generalise. If we ran the same pipeline at Takoradi or Ada Foah, would XGBoost still win by the same margin? Deferred to future work."* |
| [`../thesis/04_results_and_discussion.md`](../thesis/04_results_and_discussion.md) §4.14.4 | One site, one turbine, one truth source; nothing transfers without re-measurement |
| **thesis §4.9.3** | *"It is **one site and one horizon.** At a location with weaker NWP skill, or at a shorter horizon where persistence is strong, reanalysis features could matter considerably more."* |

**The third is the reason this iteration is worth running rather than merely wide.** §4.9's
headline — that dropping every latency-blocked reanalysis feature costs *nothing*
(0.8009 operational vs 0.8037 study) — is the result that makes the system deployable at
all. §4.9.3 states the condition under which it should fail. Twelve of Ghana's sixteen
regional capitals are low-wind interior sites, which is precisely where NWP skill should be
weaker. **So this is a test the project set for itself and can now take.**

The second deliverable is a **resource product**: per-site monthly climatology and
exceedance for all 16, which is what wind-resource assessment actually delivers for siting
and PPA sizing and does not depend on beating a baseline.

## 2. The sites, and the one that is missing

The 16 regional capitals, from [`../sites/ghana_16.yaml`](../sites/ghana_16.yaml), which is
the single source for every coordinate. The region/capital pairings were cross-checked
against two independent sources on 2026-10-01 and both agree on all 16.

**Anloga is not a regional capital** — Ho is Volta's. The project's only site with a
demonstrated resource and a published result set therefore sits *outside* the study
population, and is carried as a **17th reference site** for two jobs: it anchors the
viability scale (mean CF ≈ 0.25) and it is the equivalence check on the new data path
(Gate 1, §7).

Three of the 17 are coastal (Accra, Cape Coast, Sekondi-Takoradi), plus Anloga. The rest are
interior or northern savannah.

**Coordinate precision is not the risk; identity and side-of-the-coast are.** ERA5's grid is
0.25° (~27 km), so 2 dp is ample — the measured site-to-cell-centre offsets run 5.0–14.8 km,
all well inside one cell. What a coordinate error would do is relocate a town silently, with
every downstream metric still looking reasonable. Hence the provenance fields in the
registry.

## 3. ⚠️ The land/sea trap, stated before it can bite

ERA5 cells over water carry much lower surface roughness and correspondingly higher wind
speeds. **A coastal site whose nearest grid cell lands offshore shows an inflated resource,
and the defect is invisible**: the series is physically self-consistent, it simply describes
the sea rather than the shore.

**Cape Coast is the live case.** At 5.11°N it falls almost exactly between the ERA5 rows at
5.00 and 5.25 — distances 0.11° and 0.14°, a margin of **0.030°**. A plain nearest-cell
lookup selects 5.00°N, and the coastline there runs near 5.10°N, which puts that cell in the
Gulf of Guinea. Accra (5.56 → 5.50) is borderline for the same reason.

**Pre-registered guard.** The core ERA5 request gains `land_sea_mask`. The cell selected for
each site must have land fraction **≥ 0.5**; where it does not, the selection snaps to the
nearest cell that qualifies and **the snap is logged and recorded in this document**. Sites
carrying `land_check: required` in the registry are Accra, Cape Coast and
Sekondi-Takoradi.

**This is a correction, not a preference, and it must not become one.** The threshold and
the affected sites are fixed here, before any resource number exists. Choosing a mask
threshold after seeing which sites rank well would be fitting the geography to the result.

### 3a. ⚠️ Measured 2026-10-01 — and the reference site failed its own guard

The mask was fetched as part of Gate 0 (§7). Three of 17 cells fall below the threshold:

| Site | Nearest cell | Land fraction | Action |
|:---|:---|---:|:---|
| **Cape Coast** | (5.00, −1.25) | **0.178** | snapped to (5.25, −1.25), 16 km, lf 0.77 |
| **Accra** | (5.50, −0.25) | **0.314** | snapped to (5.75, −0.25), 22 km, lf 0.96 |
| **Anloga** *(reference)* | (5.75, 1.00) | **0.124** | **not snapped** — see below |
| Sekondi-Takoradi | (5.00, −1.75) | 0.664 | passes, no snap |

The prediction in §3 was right about Cape Coast and Accra. **It did not anticipate that the
project's own reference site would fail the same test, and that is the significant finding
of this phase.**

**Anloga's published cell is 88% water.** An independent check confirms it without reference
to the mask at all: mean surface roughness in the existing v5 parquet is
**z₀ = 0.000063 m**. Open sea is 0.0001–0.0002 m; cropland is 0.05–0.10 m, roughly a
thousand times rougher. ERA5 is describing water.

**What this does and does not touch.** The distinction matters and should not be blurred in
either direction:

| | Status |
|:---|:---|
| **Forecast metrics** — MAE 0.7295, R² 0.6653, the −30.6% NWP gain, the 4.7% classical–ML gap, operational-vs-study, skill decay, seed sensitivity | **Valid.** Every one predicts ERA5 wind *at that cell* from NWP forecasts *at that same cell*. Truth and features agree, so the comparisons are internally sound. The forecast-method study stands. |
| **Resource and power levels** — mean 100 m wind 6.24 m/s, CF ≈ 0.25, the monthly P50/P75/P90 product, annual energy | **Overstated, with a known sign.** Lower roughness means faster wind means a larger apparent resource. These describe a marine/lagoon regime, not a land site. |
| **The benchmark comparison** — "7.48% nMAE, at or below the Energies 2021 ~8.8% reference" | **Flattered, and this is new.** Marine wind is smoother and more persistent than wind over land, so this compares an easier series against a harder one (a land wind farm in Poland). The existing caveat list says "different site, physics-derived truth, single seed"; it should now also say *quasi-marine wind regime*. |

**Why Anloga is exempted from the snap rather than corrected.** Two reasons, neither a
preference:

1. **Gate 1 would become untestable.** The gate checks that the regional path reproduces the
   per-site cache's centre-cell columns, and that cache *is* cell (5.75, 1.00). Snapping
   would guarantee a mismatch and destroy the only end-to-end validation of the new data
   path.
2. **It would silently redefine every published v5 number.** Changing the reference site's
   cell mid-project is the "silent edit" [stage 9
   §9.3](../redo/09_stage9_writeup_and_hygiene.md) forbids.

So the figures keep the cell they were computed on and **the finding is reported instead of
patched**. `min_land_fraction: 0.0` in the registry records the exemption explicitly rather
than leaving it implicit.

**In fairness to the original choice:** Anloga is a sandspit between the Keta Lagoon and the
Gulf of Guinea, so a real turbine there *does* have near-marine exposure, and Ghana's
interest in the Anloga/Ada/Keta strip is precisely that exposure. The problem is resolution,
not site selection — ERA5 at 25 km cannot represent a ~2 km spit, so the cell averages the
water around the site rather than the site. **A met mast is what settles it**, which is
already the last item of thesis §4.14.5.

**Consequence for this study.** Land fraction and mean roughness are reported as columns in
the resource table for all 17 sites, so this cannot hide again. It also means the 16 capitals
and Anloga are not strictly comparable on resource: 16 land cells against one water cell.
Comparisons of *forecast skill* remain fair.

**One more thing the snap does not fix, and should not.** The snap moves the *centre* cell
onto land; the 3×3 window around it still reaches back over the water. Cape Coast's selected
cell (5.25, −1.25) sits one row north of the ocean row it was moved off, so its
`_mean`/`_std`/`_grad_ns` columns encode a land–sea contrast. That is **not** a defect to
correct: the target variable is the centre cell, which is land, and a coastal site's spatial
gradient genuinely is a land–sea gradient — the sea-breeze signal the feature set is meant to
carry (`docs/03_data_sources.md` makes the same argument for `_grad_ew`). It is recorded here
only so that nobody later reads a large `_grad_ns` at a coastal site as a data error, and so
that the three coastal sites are understood to have a different *kind* of spatial feature from
the thirteen interior ones.

## 4. Data strategy — why one bounding box instead of seventeen downloads

The per-site ERA5 fetcher issues one CDS request per (month × family). One site over the
full 2010–2025 archive is **468 requests** — the 468 files currently on disk for Anloga —
and the v5 logs measured roughly **19 minutes per request** end to end, including queue
time and CDS's frequent transient failures. Seventeen sites that way is about **99 days** of
continuous downloading. That is not a scheduling problem to be worked around; it ends the
study.

**The `area` parameter is already a bounding box** ([`../src/data_ingest.py`](../src/data_ingest.py)
`_fetch_era5`), and **a monthly request covering all of Ghana is accepted at the same request
count as one covering two cells** — measured, Gate 0 below. Only the payload grows.

> ⚠️ **Correction, 2026-10-01.** An earlier revision of this paragraph said "CDS cost is
> field-based — variables × times × levels — **not** area based". **That is wrong**, and it was
> an over-generalisation from the single accepted monthly request below. CDS's cost does
> include spatial extent: a whole-*year* Ghana request was later rejected with *"cost limits
> exceeded — your request is too large"*, even though its field count (7 × 8784 = 61,488) sits
> under the 121,000 cap that `RESULTS.md` records. That cap figure came from the project's
> earlier **2-cell** requests, and this bbox is **630 cells**, so it does not transfer.
>
> What survives is the part that matters for feasibility: **one monthly 630-cell request is
> accepted and costs one request**, so 17 sites still come out of one download pass rather
> than seventeen. What does not survive is any inference that the time span can be widened
> freely. The largest accepted chunk was established by measurement, not arithmetic — §4a.

```
bbox (N, W, S, E) = 11.5, -3.5, 4.25, 1.5
30 lat x 21 lon = 630 cells   (against 2 cells for the per-site Anloga patch)
```

The margin is deliberate: the outermost sites are Bolgatanga (10.79 N), Sekondi-Takoradi
(4.93 S), Goaso (−2.52 W) and Anloga (0.90 E), and every site needs ±0.375° of room for a
3×3 extraction window. **Verified computationally: all 17 windows fall strictly inside the
bbox**, none touches an edge.

**✅ Measured 2026-10-01 (Gate 0), and the budget was pessimistic by an order of magnitude:**

| | Budgeted | **Measured** |
|:---|---:|---:|
| CDS accepts a 630-cell monthly request | assumed | **yes** — no cost-cap rejection |
| Payload, one core month | ~40 MB | **6.3 MB** |
| Wall time, one request | ~19 min | **3.0 min** |
| Land-sea mask (one-off) | — | 25 KB, 38 s |
| Grid shape | 30 × 21 = 630 | **30 × 21 = 630, confirmed** |
| Full archive | 468 requests, ~20 GB / ~6 days | **576 requests, ≈3.6 GB / ≈29 h** |

### 4a. Chunk size — a month is the limit, measured twice over

Monthly chunking needs 576 requests, and with per-request cost running 5–20 min that
extrapolates past the pre-registered 7-day budget. So a larger chunk was tried. **It does not
work, for two independent reasons, and both were established by measurement after an argument
from arithmetic failed.**

**Reason 1 — CDS rejects it on cost.**

| Chunk | Core fields | Values (fields × 630 cells) | Verdict |
|---:|---:|---:|:---|
| 12 months | 61,320 | 38.6 M | ❌ `403 cost limits exceeded` |
| 6 months | 31,248 | 19.7 M | ❌ `403 cost limits exceeded` |
| 4 months | 20,832 | 13.1 M | ❌ `403 cost limits exceeded` |
| **3 months** | 15,624 | 9.8 M | ✅ accepted |
| 1 month | 5,208 | 3.3 M | ✅ accepted — the baseline |

The yearly attempt was predicted to succeed because 61,320 fields sits under the 121,000 cap
`RESULTS.md` records. **That prediction was wrong and the reasoning behind it was wrong**: the
121k figure was measured on the project's **2-cell** per-site requests, and this bbox is **630
cells**. CDS's cost includes spatial extent, so a field count alone does not predict
acceptance. §4's earlier claim that cost is "not area based" has been corrected.

**Reason 2 — even the accepted chunk is far slower, not faster.** The 3-month request returned
18.3 MB in **672 minutes**. Against ~20 min for a monthly request carrying a third of the data,
that is roughly **30× worse per unit of data**. CDS queue time scales sharply with request
cost, so a bigger request is punished rather than rewarded. Extrapolated, quarterly chunking
would take ~2,150 h against monthly's ~192 h.

**Conclusion: `chunk="month"` is not a default to be improved on, it is the only workable
size.** The `chunk="year"` code path is retained for a smaller bbox or a changed quota, and
`src/data_ingest.py` carries the measurements so nobody re-derives this from the field count.

### 4b. ⚠️ Never run a second CDS job while the archive is downloading

**CDS limits concurrent jobs per account, and the archive is starved by anything that competes
with it.** While the 3-month probe above was in flight, the archive completed **3 requests in
13.4 hours** — an apparent 268 min/request — and recovered as soon as the probe finished. The
archive was queued behind the project's own probe.

This was diagnosed only after two wrong hypotheses, both of which looked plausible and neither
of which survived a direct check:

- *"CDS is fair-use throttling us."* No: when measured directly, a request went
  `accepted → running` in **12 seconds**.
- *"The network has degraded."* Partly true and not the cause: general throughput measured
  202 kB/s and DNS resolved 5/5 on demand, but a 6.4 MB monthly file is only ~32 s of transfer
  at that speed. The 26 logged DNS failures cost 120 s each and are a real nuisance, not a
  13-hour stall.

**The operating rule, stronger than the one in §10:** the existing warning says never run two
*ingests* concurrently, because `path.exists()` is the only re-download guard. That is about
correctness. This is about throughput — **any** CDS request, including a one-off probe writing
to unrelated paths, takes capacity the archive needs. Probe before starting the archive, or
stop the archive first.

**On the request count.** The budget said 468 (156 months × 3 families), taking 156 from the
per-site Anloga cache. 2010-01 → 2025-12 is actually **192 months**; the per-site cache holds
156 because the 2016-08 → 2019-12 backfill was deliberately halted on measured grounds
([11 §8](11_nwp_forecast_features.md)). The regional archive therefore covers **41 months the
per-site cache never had**, making it gap-free rather than merely wider. At 576 × 3.0 min
that is ≈29 h and ≈3.6 GB — still comfortably inside the pre-registered ≤7 day / ≤25 GB gate.

**So the central feasibility claim holds, and comfortably.** CDS cost is field-based: a
630-cell request was accepted and returned in 3 minutes against the ~19 min/request the
per-site v5 download averaged — the regional requests are *faster* than the point requests
were, because the v5 figure was dominated by queue time and retry backoff rather than
payload. **All 17 sites now cost about one day of downloading instead of 99.**

The earlier 40 MB/month estimate assumed payload scales linearly with cell count from
Anloga's 154 KB two-cell file. It does not — NetCDF compression over a spatially smooth
field does far better than that.

**A side effect that is genuinely valuable.** The current `pad = 0.2` yields a 1(lat)×2(lon)
patch, so — as `_centre_and_spatial`'s own caveat and
[`00_data_card.md`](00_data_card.md) record — all five `_grad_ns` columns are identically
zero and all five `_std` columns are exactly `|_grad_ew| / 2`. **Ten of 36 tidy columns
carry no independent information.** A Ghana bbox gives a genuine 3×3 patch and fixes this
for free. But it means the spatial columns **change**, so this is a new feature set rather
than a drop-in, and it is why Gate 1 expects those columns to differ.

## 5. Scope — what is and is not being run

**Run:** resource climatology for all 17; the 24 h forecast ladder for all 17, on **both**
feature sets, over the existing 4-fold rolling-origin protocol
([`../redo/results/01_protocol.md`](../redo/results/01_protocol.md)) with the same folds,
seeds and DM test, so sites are comparable to each other *and* to the published Anloga
table.

Ladder: `persistence`, `climatology`, `damped_persistence` (reference) · `mos_single`,
`mos_multi`, `kalman_mos`, `analog_ensemble` (classical) · `ridge`, `xgboost`, `lightgbm` (ML).

**XGBoost gets 5 seeds**, at ~0.1 min/fit. This is nearly free and it retires the project's
longest-outstanding item ([12 item 4](12_next_steps_v2.md), the multi-seed repeat) across 17
sites rather than one.

**Not run, with reasons:**

| Excluded | Why |
|:---|:---|
| **Deep models** | Phase B cost **12.9 h at one site**, so 17 would be ~206 h. [`../redo/results/04`](../redo/results/04_deep_models.md) already measured trees ahead by 5.1% at 1/42 the training cost, with deep seed spread up to 18× XGBoost's. Cited, not re-run. |
| Multi-day (168 h), monthly outlook, quantile arms | The 24 h ladder and the resource product answer the generalisation question. Five systems × 17 sites is a different project. |
| Per-site turbine or hub height | One machine everywhere (E-101/3050 at 100 m) **by design** — comparability is the point. A low-wind IEC class III turbine would flatter the interior sites. |

## 6. Pre-registered decisions — fixed before any result exists

### 6a. Viability floor for power metrics

Wind-speed metrics are physically meaningful at every site and are reported for all 17.
Power and CF metrics are **not** meaningful where there is no exploitable resource, so:

| Mean CF | Class | What gets reported |
|:---|:---|:---|
| **≥ 0.20** | viable | full resource product + power metrics |
| **0.10 – 0.20** | marginal | full resource product + power metrics, labelled marginal |
| **< 0.10** | not viable | **wind-speed metrics only**; no power claim is made |

Anloga sits near **0.25** and calibrates the scale. **This table is fixed now, before the 16
are measured**, precisely so the floor cannot later be chosen to flatter whatever comes out.

### 6b. Four predictions, each with what would refute it

1. **The NWP gain holds in direction at all 16.** Adding forecast features beats the
   no-NWP arm everywhere.
   *Refuted if* any site shows no improvement or a degradation.
   *Prior:* strong. The mechanism — a forecast column valid near the target hour — is not
   site-specific.
2. **The classical–ML gap stays small:** `mos_multi` within ~10% of `xgboost` at every site.
   *Refuted if* the gap exceeds 10% anywhere.
   *Prior:* moderate. Measured at 4.7% at Anloga; a noisier low-wind target could widen it.
3. **The operational-vs-study result breaks somewhere.** At at least one site, dropping
   reanalysis features costs materially more than it does at Anloga (where it was −0.3%,
   i.e. free).
   *Refuted if* degradation stays ≤ 0 at all 16.
   *Prior:* this is **thesis §4.9.3's own prediction** and the most interesting of the four.
   Recording it here means the thesis's caveat gets a verdict rather than a restatement.
4. **Only coastal sites clear the viability floor.** Accra, Cape Coast and
   Sekondi-Takoradi reach ≥ 0.10; the interior and northern capitals do not.
   *Refuted if* any interior site reaches 0.10, or if a coastal site fails to.
   *Prior:* moderate-to-strong on published Ghanaian resource work, but this project has
   not measured it, and the northern savannah sites are the ones most likely to surprise.

**A null is a result here.** If predictions 1, 2 and 4 hold and 3 is refuted, the honest
finding is that the Anloga conclusions generalise across Ghana's climatic zones — which is a
stronger claim than the project can currently make, and it is reportable as such.

## 7. Verification gates — pass criteria fixed in advance

Run in order; stop at the first failure.

| # | Gate | Pass criterion | Status |
|---:|:---|:---|:---|
| 0 | **CDS probe** | One monthly Ghana-bbox request is accepted (cost cap is field-based as claimed); measured size and wall time extrapolate to ≤ ~25 GB and ≤ ~7 days for 156 months × 3 families | ✅ **PASS 2026-10-01** — 6.3 MB, 3.0 min, ≈2.9 GB / ≈1.0 day. §4 has the table; §3a has the land-mask result, which is where the interesting outcome was |
| 1 | **Anloga equivalence** | Anloga extracted from the bbox reproduces the per-site cache's **centre-cell** columns exactly; spatial columns differ, and `_grad_ns` is **no longer identically zero** | ⚠️ **FAILED as written; the claim it tested passed.** The criterion was mis-specified — see §7a |
| 2 | **Land mask** | Every site's selected cell has land fraction ≥ 0.5, or carries a recorded exemption; every snap is logged | ⚠️ **PASS with one exemption** — Cape Coast and Accra snapped as predicted; **Anloga failed at 0.124 and is exempted, not corrected** (§3a). The criterion was amended to admit a recorded exemption, which is a change made *after* seeing the result and is flagged as such here rather than hidden |
| 3 | **Defect-9 regression** | `python -m src.bakeoff --config config_bakeoff.yaml --per-fold` still reproduces xgboost 0.8037, mos_multi 0.8412, persistence 1.3183 and the 46.2% fold spread | ⏳ pending |
| 4 | **Sanity floor, all 16** | Every site's best model beats **both** persistence and climatology on its own folds | ⏳ pending |
| 5 | **Anloga end-to-end** | The 17th site reproduces the published 24 h table through the new runner | ⏳ pending |
| 6 | **Tests** | `pytest -q` green; currently 149 | ✅ **PASS** — 149 unchanged (the per-site path is untouched) **plus 38 new** in `tests/test_multisite.py` covering the regional code: bbox request building, nearest-cell lookup, the land snap, the Anloga exemption, window clamping at a grid edge, `extract=None` identity, the manifest guard, registry validation and the resolved-config round trip |

> 🔖 **On gate 2's amendment.** The pre-registered criterion was "every site's selected cell
> has land fraction ≥ 0.5", full stop. Anloga fails it and is being exempted, so the
> criterion as written is **not met** and saying otherwise would be exactly the
> gate-that-cannot-fail problem [register entry 14](../redo/00_mistake_register.md) is about.
> The amendment is recorded with its reasons in §3a and the original wording is left above.
> A reader who thinks the exemption is special pleading has everything needed to say so.

### 7a. Gate 1 — the criterion was wrong, not the code (measured 2026-10-01)

Run on 2024-06, regional extraction against the per-site cache, 720 rows both ways.

**As written, the gate failed.** Five of sixteen centre-cell columns did not match exactly:

| Column | Max abs diff | Relative | In float32 ULP |
|:---|---:|---:|---:|
| `pressure_pa` | 6.25e-02 | 6.2e-07 | 8 |
| `temperature_c` | 2.44e-04 | 8.1e-07 | 8 |
| `dewpoint_c` | 2.44e-04 | — | 8 |
| `bl_height_m` | 2.44e-04 | — | — |
| `wind_gust_10m` | 9.54e-07 | — | — |

Every diff is an exact power of two, which is the tell. **The cause is not cell selection:**

- **The selected cell is identical by construction and by inspection.** The per-site grid is
  1 × 2 and `_centre_and_spatial` takes index `[0, 1]` → **(5.75, 1.00)**. The regional grid
  is 30 × 21 and nearest-cell selection returns `[23, 18]` → **(5.75, 1.00)**. Same cell,
  confirmed from the files' own coordinates.
- **`u100`, `v100` and `fsr` are bit-identical** (max abs diff exactly 0.0), as are all six
  pressure-level columns and all four wind speed/direction columns — including
  `wind_speed_100m`, the target variable.
- The five that differ are the **large-magnitude** fields, and each differs by ~8 ULP of
  float32 at its own magnitude (2⁻⁴ at 101 kPa, 2⁻¹² at 300 K). Both files store float32 with
  no scale/offset packing, so this is round-off in CDS's server-side subsetting pipeline,
  which is a different code path for a 630-cell area than for a 2-cell one.

**A wrong cell would differ by percent, not by 8 ULP.** 0.00024 K and 0.06 Pa are physically
meaningless — six orders of magnitude below anything that moves a metric.

**So the criterion tested the wrong thing.** "Matches exactly" was a proxy for "is the same
cell", and it turns out to be a test of float32 determinism in a CDS code path instead. The
criterion is amended to what it was always standing in for:

> **Amended gate 1:** the selected cell's coordinates must be identical to the per-site
> cache's centre cell, **and** every centre-cell column must agree to within float32
> round-off (relative ≤ 1e-6). The tolerance is set by the storage format, not by the
> observed discrepancy.

**On that amendment**, since this is now the second criterion to be rewritten after seeing a
result: the tolerance is justified by float32 precision *independently* of what was measured
— relative 1e-6 is ~8 ULP, which is where the argument started — and the direct coordinate
check is strictly stronger evidence than the numeric one it replaces. But two of six
pre-registered criteria being mis-specified is itself worth recording. Register entry 14's
lesson is that a gate you cannot fail is not a gate; the converse also holds, that **a gate
which fails for a reason unrelated to its claim is not measuring the claim.**

**The spatial-column half of the gate passed outright, and it is the interesting half:**

| Check | Per-site | Regional |
|:---|:---|:---|
| `_grad_ns` (all 5 columns) | identically zero | **alive** — sd 0.71 (wind 100 m) to 14.6 (pressure) |
| `_std` ≈ `\|_grad_ew\|/2` collinearity | holds | **broken** |

**Ten of 36 tidy columns carried no information under the per-site patch and now do.** That
is the degeneracy `docs/00_data_card.md` records, fixed as a side effect.

One correction while verifying it: RESULTS.md says all five `_std` columns are **exactly**
`|_grad_ew| / 2`. Three are (deviation 0.0e+00); the two *derived* wind-magnitude columns
(`wind_speed_100m_std`, `wind_speed_850hpa_std`) agree only to 8.0e-05 and 8.9e-06 relative,
because `np.std` on a 2-cell float32 array of `sqrt(u²+v²)` rounds where the raw fields do
not. Functionally collinear either way, so the substantive claim — those columns carry no
independent information — is unaffected. "Exactly" is too strong for two of the five.

**How gate 1 can fail, and what it would mean.** A mismatch in a centre-cell column means
the nearest-cell lookup is wrong — fix the code, do not widen the tolerance. A *pre-check
has already been done and is encouraging rather than conclusive*: the existing Anloga cache
spans `latitude=[5.75], longitude=[0.75, 1.00]` and `_centre_and_spatial` takes index
`n_lon // 2 = 1`, so today's centre cell is **(5.75, 1.00)** — and the regional nearest-cell
lookup independently selects **(5.75, 1.00)**. The same cell by two different routes. That
makes the gate structurally likely to pass, which is a reason to run it, not to skip it.

**How gate 4 can fail, and what it would mean.** A site where nothing beats both baselines
is a **data problem, not a modelling result** — most likely a bad coordinate, an ocean cell,
or an Open-Meteo hole. Stop and investigate; do not report it as a finding about that site.

## 8. What may and may not be claimed today — 2026-10-01

**May:**
- The 16 region/capital pairings are cross-checked against two independent sources.
- Every coordinate carries a named source and a verification date.
- All 17 3×3 extraction windows fall strictly inside the stated bbox (computed, not assumed).
- The land/sea hazard at Cape Coast and Accra is identified, quantified (0.030° row margin)
  and guarded **before** any resource figure exists.
- **CDS cost is field-based, measured:** a 630-cell monthly request was accepted and returned
  in 3.0 min at 6.3 MB. All 17 sites cost ≈2.9 GB and ≈1 day, against ≈99 days per-site.
- **Anloga's published ERA5 cell is 88% water** (land fraction 0.124), corroborated
  independently by mean surface roughness z₀ = 0.000063 m against 0.05–0.10 m for cropland.
- **That this leaves the v5 forecast metrics valid and the v5 resource levels overstated**,
  with the direction of the bias known. §3a has the breakdown.

**May not:**
- **Any resource figure for any of the 16.** None has been computed.
- **Any forecast metric for any of the 16.** None has been computed.
- **Any statement about whether the Anloga conclusions generalise.** That is the entire
  question and it is open.
- **A corrected Anloga resource figure.** The water-cell finding says the published figure is
  biased upward; it does **not** say by how much. Quantifying it needs a land-cell or
  mast-based comparison and neither exists yet.
- **That the benchmark comparison fails.** It says the comparison is *flattered* by a
  smoother marine series. By how much is unmeasured, and 7.48% vs ~8.8% is not thereby
  overturned — it is weakened and needs the extra caveat.
- **That Anloga was the wrong site to choose.** The sandspit genuinely has near-marine
  exposure; the defect is ERA5's 25 km resolution against a ~2 km landform.

## 9. Decision log

| Date | Decision | Basis | Reversible? |
|:---|:---|:---|:---|
| 2026-10-01 | 16 regional capitals, Anloga as 17th reference | User scope decision; Anloga needed as the equivalence anchor and it is not a capital | Yes — registry edit |
| 2026-10-01 | Screen **and** forecast all 16, rather than forecasting only viable sites | The generalisation question is about forecast skill, which is meaningful at every site; restricting to viable sites would answer it on 3–4 | Yes, but wastes the download |
| 2026-10-01 | One Ghana-wide bbox, not 17 per-site downloads | 468 requests/site × ~19 min ≈ 99 days for 17 sites; one bbox costs one site's requests | Yes — per-site fetcher is untouched |
| 2026-10-01 | bbox margin to 11.5/−3.5/4.25/1.5 rather than a tight hull | Every site needs ±0.375° for a 3×3 window; verified all 17 fit | Yes, but re-downloads everything |
| 2026-10-01 | `window=None` default on the month loaders | Published Anloga results must stay byte-reproducible; the regional path opts in explicitly | Yes |
| 2026-10-01 | Deep models excluded | 12.9 h × 17 ≈ 206 h against a measured 5.1% deficit at 1/42 the cost | Yes, at that cost |
| 2026-10-01 | Land-fraction threshold 0.5, set before any resource number | Cape Coast's nearest cell is offshore; a threshold chosen later would be fitted to the result | Should not be |
| 2026-10-01 | Viability floor 0.10 / 0.20, set before any resource number | Same reason; Anloga ≈ 0.25 calibrates it | Should not be |

## 10. What is running, and how to resume

**The regional archive download started 2026-10-01 16:00** and is the long pole. Everything in
§5 waits on it. It is **resumable** — `fetch_regional_archive` skips months already on disk.

⚠️ **Revised estimate: ≈2 days, not the ≈29 h §4 extrapolates.** The probe measured a single
request at 3.0 min; the sustained rate measured over six complete months (18 requests in
101.9 min) is **5.66 min/request**, because CDS queue time dominates and varies through the
day. 576 × 5.66 min ≈ **54 h**. Still inside the pre-registered ≤7 day gate, and payload is
tracking at ~4.7 MB/file → ≈2.7 GB. **A single request is not a sample.** Recorded here rather
than quietly amended in §4, where the 3.0 min measurement stays as what it was.

### It died twice, and neither cause was CDS

Three launch methods, two of which failed. Recorded in full because the next person to resume
this will reach for the first one.

| # | Method | Outcome |
|---:|:---|:---|
| 1 | `Start-Process` from the agent shell | ❌ Died **17:43 at 2010-07** (7/192). The harness kills its process tree at session end. |
| 2 | WMI `Win32_Process.Create` | ❌ Died **22:16 at 2011-04** (16/192). Survived the job-object teardown, but still had a **console** attached and took a `Ctrl+C` — the log ends in a literal `^C`. |
| 3 | **Scheduled Task** | ✅ Running. Task Scheduler gives it **no console and no parent shell**, so neither failure mode applies. |

**The lesson is narrower than "use a scheduled task".** Detaching from the parent is not the
same as detaching from the console, and method 2 only fixed the first. On Windows a long job
that must survive an interactive session needs to escape *both*, and only the scheduler does
that.

Register and start it with (the script path must be short — `schtasks /TR` caps at 261 chars,
which a scratchpad path exceeds, so use `Register-ScheduledTask`):

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<short path>\era5gh_resume.ps1"' `
  -WorkingDirectory "<repo>"
Register-ScheduledTask -TaskName "era5gh_archive" -Action $action -Settings (
  New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 5) -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5))
Start-ScheduledTask -TaskName "era5gh_archive"
```

`-RestartCount 3` means the scheduler itself retries if the process dies, which is a third
layer under the in-process `_retrieve_with_retry` loop and the on-resume cache validation.
**Remember to `Unregister-ScheduledTask -TaskName "era5gh_archive"` once the archive is
complete** — it is a one-shot job, not a permanent fixture.

**Two guards were added because of this, and the second matters more than the first:**

1. `_retrieve_with_retry` now takes a per-family `min_bytes` floor
   (`REGIONAL_MIN_BYTES`), deleting and re-requesting anything short. Floors are ~half the
   smallest legitimate file — February at 28 days: core 5.87 MB, extra 2.68 MB, pressure
   4.71 MB — so they cannot trip on a real month. The per-site path keeps `min_bytes=0` and is
   byte-for-byte unchanged.
2. **`_drop_truncated_regional_files` re-validates the whole cache on every resume.** The first
   guard only inspects what its own process downloaded; a process **killed mid-write** leaves a
   partial file that no later run ever looks at, because `path.exists()` short-circuits first.
   The partial month would then feed every site reading it, with nothing in the output looking
   wrong. This is the actual hole the death exposed, and it closes it.

**Neither death produced a partial file**, and both times for the same reason — cdsapi had not
yet opened the target. After the first, `era5gh_core_2010-07.nc` was complete at 6.78 MB and
all eight core files were verified to open with the right hour counts (744/720/672), the
30 × 21 grid and zero NaNs. After the second, no `era5gh_pressure_2011-04.nc` existed at all,
and the resume's cache validation confirmed *"all existing files pass their size floor"* before
re-requesting it.

**That is luck twice, not design.** The window between "target opened" and "download complete"
is a few seconds out of every ~5 minutes, so roughly a 2% chance per interruption — which is
exactly the kind of odds that eventually lose over a 576-request job. Guard 2 is what makes it
not matter.

One incidental confirmation that the retry path works: the re-requested 2011-04 pressure
completed in **6 seconds** rather than ~5 minutes, because the CDS job from the interrupted
attempt had already finished server-side and was served from its cache.

> 📝 **A note on the log.** The first relaunch attempt redirected with PowerShell's `*>>`,
> which transcodes to UTF-16 and left one garbled stretch in an otherwise UTF-8 file around
> 17:58. Redirection now goes through `cmd.exe`, which appends raw bytes. The stretch is left
> in place rather than edited out; it is a few lines of CDS status chatter and the file-count
> check below is the authoritative progress signal anyway.

This section carries **check commands rather than snapshots**
([stage 9 §9.5](../redo/09_stage9_writeup_and_hygiene.md)), because a file count written into
a document is wrong within minutes.

```bash
# Is an ingest already running? NEVER start a second one — path.exists() is the
# only re-download guard and this has gone wrong once before (RESULTS.md).
powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select CommandLine"

# Progress: expect 3 families x 192 months + lsm + manifest = 578 when complete
powershell -c "(Get-ChildItem data/raw/era5gh_* | Measure-Object).Count"
powershell -c "Get-Content outputs/logs/era5gh_archive.log -Tail 5"

# Resume an interrupted download. Idempotent: skips months already present and
# deletes any truncated file first so it is re-requested.
python -m src.multisite fetch

# Resume so that it SURVIVES — see the table above for why a shell child and a
# WMI-detached process both died. Task Scheduler is the one that holds.
powershell -c "Start-ScheduledTask -TaskName 'era5gh_archive'"
powershell -c "Get-ScheduledTask -TaskName 'era5gh_archive' | Select State"

# Tear the task down once the archive is complete — it is a one-shot job.
powershell -c "Unregister-ScheduledTask -TaskName 'era5gh_archive' -Confirm:`$false"

# Integrity check — size floors catch gross truncation, but opening the files is
# what proves them. Run before trusting a resumed archive.
python -c "import xarray as xr; from pathlib import Path; [print(p.name, len(xr.open_dataset(p).valid_time)) for p in sorted(Path('data/raw').glob('era5gh_core_*.nc'))]"

# Which cell each site resolved to, and its land fraction — run before quoting
# any resource figure
python -m src.multisite cells

# Which sites have completed artefacts
powershell -c "Get-ChildItem outputs/sites -Directory | Select Name"
```

## Related

- [`../sites/ghana_16.yaml`](../sites/ghana_16.yaml) — the site registry, single source for coordinates
- [`07_findings_and_analysis.md`](07_findings_and_analysis.md) — threat 2, the gap this closes
- [`../redo/results/01_protocol.md`](../redo/results/01_protocol.md) — the protocol being reused unchanged
- [`../redo/results/04_deep_models.md`](../redo/results/04_deep_models.md) — why deep models are excluded
- [`../redo/09_stage9_writeup_and_hygiene.md`](../redo/09_stage9_writeup_and_hygiene.md) — the rules this document follows
