# Stage 3 — The alignment harness

> **What this stage is:** building the tests that catch time-alignment and leakage errors,
> **before** the first model exists.
> **Why it comes third:** the single most expensive mistake in this project's history was a
> 24-hour misalignment that produced plausible numbers, agreed with everyone's prior, and
> survived two review cycles ([register A3](00_mistake_register.md)). A test written after
> seeing a result is not a test. This stage is roughly two days and would have saved two
> iterations.
> **Effort:** 2 days.

## 3.1 Why alignment is where time-series projects die

Two tables, two different clocks:

- **Forecasts** are stamped with the hour they *describe* — the valid time τ.
- **Training rows** are stamped with the hour the prediction is *made from* — the feature
  time t.

They differ by the horizon. Get it wrong in one direction and the model sees the future:
free marks, worthless model. Get it wrong in the other and the model is graded against the
wrong hour: every architecture collapses onto the persistence floor, which is exactly what
happened and exactly what got written up as a finding about deep learning.

**The contract, stated once:**

> Row *t* receives the forecast **valid at t + horizon**, which was issued around *t* and
> was therefore legitimately knowable then.

Implement it with an explicit reindex on `index + horizon`, never a bare `.shift()`. The
reindex makes the intent visible and turns gaps into NaN instead of a silently displaced
value.

## 3.2 The four guards

### Guard 1 — the perfect-forecast test *(the important one)*

Construct a forecast column that is **exactly** the future target. After alignment it must
reproduce the target exactly. If it doesn't, the join is wrong.

The critical detail, and the reason this test is sharp: build the underlying signal on a
period **deliberately not commensurate with the horizon** — the original used a 37-hour
period against a 24-hour horizon. On a 24-hour signal, a frame shifted by exactly 24 hours
passes by coincidence. On 37 hours, nothing passes but the truth.

### Guard 2 — lag-peak against truth

Cross-correlate each forecast series against the observed truth over lags −48…+48 h. The
peak must sit at |lag| ≤ 1 h.

Distinguish the failure modes rather than lumping them together, because they have different
fixes:

| Symptom | Diagnosis | Response |
|:---|:---|:---|
| Peak at ±24 h | The horizon shift is wrong | **Fix the code, never the criterion** |
| Constant 2–3 h offset, one source only | A real convention difference | Document it, apply an explicit named shift |
| No clear peak at all | Units, coordinates, or the wrong variable | Stop; nothing downstream is meaningful |

The 1-hour tolerance is not slack — it exists because genuine physical disagreement between
forecast models about sea-breeze phase produces ~1 h offsets, and a zero-tolerance rule
would fire on correct behaviour. **Set tolerances from measured physics, not from taste.**

### Guard 3 — unit and range assertions

From [stage 2](02_stage2_data_availability_audit.md)'s metadata: assert every column's unit
by asserting its plausible range at ingest. Surface pressure in Pa is ~90,000–105,000; a
1013 is hPa wearing the wrong label.

### Guard 4 — the leakage test

Genuine forecast skill must **decay with lead time**. Score the same pipeline using
day-1, day-2 and day-3 forecasts. If skill is flat across leads, the feature is not a
forecast — it is analysis wearing a forecast's name.

This is the guard that catches the trap the original nearly walked into: the provider's
Historical Forecast API stitches together the *opening hours* of each run, which is
analysis-like, and is the more prominent and obvious-sounding of the two endpoints. It would
have produced superb, invalid numbers ([`11 §3`](../docs/11_nwp_forecast_features.md)).

A useful secondary signal: a multi-model ensemble should beat each of its members. If it
doesn't, something is wrong with the members.

## 3.3 The rule the original project got wrong

> **A gate that cannot fail is not a gate.**

The original built a four-gate harness. It was a good idea and it caught two real data
defects. But two of the four gates **hardcode `passed=True`** — gate 3's sanity band only
changes the wording of its own output, and gate 4 passes as long as any forecast column
exists ([register E1](00_mistake_register.md)). "All four gates pass" appears in several
documents and actually means "two passed and two reported".

So, for every guard in this stage:

1. Feed it deliberately broken input — a frame shifted by 24 h, a pressure column in the
   wrong unit, a leaked analysis column.
2. **Watch it go red.**
3. Only then trust it when it is green.

Keep the broken-input cases as permanent tests. They are the only proof the guard works.

**This principle has since paid for itself four times.** The bakeoff's guards surfaced four
defects in a single day: a SARIMAX configuration producing numerically explosive forecasts,
a significance test comparing different time periods, a 25-hour persistence baseline quoted
as 24-hour, and ten mathematically dead feature columns. **None crashed**, and three
produced numbers that agreed with a reasonable prior — which is precisely why explicit
falsifiable checks beat careful reading. Worked examples:
[results/06](results/06_what_the_protocol_caught.md).

Two guards generalise beyond alignment and are worth stating here:

- **A sanity floor.** Every predictor above the trivial baselines must beat naive
  persistence; anything that doesn't is a bug until proven otherwise.
- **Assert, never reconcile.** When two things should be the same length and are not, the
  mismatch *is* the signal. Silently truncating to the shorter one is worse than crashing.

## 3.4 What to build here, concretely

- A `tests/` module holding all four guards, runnable in seconds against synthetic data.
- A separate CLI harness that runs them against **real** data for a given config, exits
  non-zero on failure, and refuses to run later checks once an early one fails — because
  results downstream of a broken alignment are not merely wrong, they are misleading.
- Each guard paired with its red-case test.

## Exit gate

You may not proceed to stage 4 until:

1. All four guards exist and pass on real data.
2. **Each has been demonstrated failing** on deliberately broken input, and the red case is
   retained as a test.
3. No guard in the suite returns a hardcoded pass.
4. The alignment contract is written down in prose next to the code that implements it.

**How this gate can fail:** any guard that has never been seen red. That is not a passing
test — it is an untested one.

---

Previous: [Stage 2](02_stage2_data_availability_audit.md) ·
Next: [Stage 4 — Ceiling analysis](04_stage4_ceiling_analysis.md)
