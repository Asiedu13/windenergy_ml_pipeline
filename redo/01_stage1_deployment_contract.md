# Stage 1 — The deployment contract

> **What this stage is:** writing down, before touching any data, what the system must do,
> when it must do it, and what information it is allowed to use at that moment.
> **Why it comes first:** the original project answered "how accurate can we be?" for
> ninety days and only then asked "with what, and when?". The answer turned out to
> invalidate half the feature set ([register A4](00_mistake_register.md)). This stage is
> two days of writing that would have reshaped everything downstream.
> **Effort:** 1–2 days, entirely on paper.

## 1.1 Decide what is actually being built

There are three defensible projects here and they need different metrics. Pick one **in
writing**, because the choice propagates into every table you will ever produce.

| Project | The question it answers | Primary metric | Does the no-SCADA problem block it? |
|:---|:---|:---|:---|
| **Operational forecaster** | "How much power will this farm produce tomorrow?" | nMAE/nRMSE vs metered output | **Yes, fatally** — there is no metered output |
| **Resource assessment** | "Is this site worth building on, and how variable is it?" | Capacity factor, distribution fit, inter-annual variability | No |
| **Forecast-method study** | "Does adding NWP forecasts improve day-ahead wind prediction at a tropical coastal site?" | Wind-speed MAE/R² vs baselines | No — wind speed is directly observable in the reanalysis |

The original project was written as the first, evaluated as the third, and honestly
defensible only as the second and third. **Say which one it is on page one.** The third is
the strongest available framing: the wind-speed target is real, the baselines are real, and
the ~30% improvement is a genuine finding that does not depend on the synthetic power curve.

> **The no-SCADA constraint is not a flaw to apologise for — it is a scope boundary.** State
> it once, clearly, and then every downstream claim can be precise instead of hedged. What
> went wrong originally was not the missing data; it was that power metrics were reported as
> if they measured physical yield.

## 1.2 Write the operational timeline

> **What:** the clock. When is the forecast issued, for what period, and what exists at that
> instant.
> **Why it is the whole ballgame:** a feature is only a feature if it exists when you press
> the button.

Fill this in **before** choosing sources:

```
Issue time         : e.g. 06:00 UTC daily
Target period      : e.g. 00:00–23:00 UTC the following day
Effective lead     : 18–42 h (varies through the target day — state the range, not "24 h")
Decision it serves : e.g. day-ahead market bid, unit commitment
```

**Note the lead-time subtlety the original project never resolved.** A fixed *issue* time
gives a *varying* lead across the target day. A fixed lead gives a varying issue time, which
is not how any operator works. The original used Open-Meteo's `previous_day1`, a fixed **day**
offset, and never verified the per-hour lead — it remains an open caveat
([`12` item 9](../docs/12_next_steps_v2.md)). Decide here, verify in
[stage 3](03_stage3_alignment_harness.md).

## 1.3 The latency budget — the table that matters most

> **What:** for every candidate input, how long after the fact it becomes available.
> **Why:** this is the single artefact whose absence cost the original project its
> deployability.

| Source | Publication latency | Available at issue time? |
|:---|:---|:---|
| ERA5 reanalysis | ~5 days (ERA5T preliminary, then consolidated) | **No** |
| NWP forecasts (Open-Meteo previous runs, GFS, ICON) | Available before valid time by construction | **Yes** |
| On-site observations | Real time — **if hardware exists** | Only with a met mast |
| Satellite / scatterometer | Hours | Case by case |

Then state the rule and hold to it:

> **Any feature whose source is unavailable at issue time is excluded from the operational
> model.** It may appear in an offline study model, but the two must be reported as separate
> systems with separate numbers.

Had this table existed on day one, the project would have discovered immediately that its
reanalysis features were offline-only, and would either have built the NWP-only model from
the start or scoped itself explicitly as a method study.

## 1.4 Pre-register the evaluation

> **What:** deciding how you will be judged, before you can see which answer flatters you.
> **Why:** this project's most valuable methodological lesson is that a plausible wrong
> number *that agrees with your prior* is nearly undetectable
> ([register A3](00_mistake_register.md)). Pre-registration is the cheapest defence.

Commit in writing to:

- **The split.** Chronological, never shuffled. Exact train/validation/test boundaries.
- **The seeds.** At least five, fixed in advance. Report mean ± sd, never best-of-N.
- **The baselines.** Persistence at the *true* horizon, climatology, and a one-parameter OLS
  on the current state. Defined once, in one function, tested. (See
  [register C3](00_mistake_register.md) — the original quoted a 25-hour persistence as 24.)
- **The metrics**, with units, and what counts as a material difference.
- **The falsifier.** What result would mean the thesis is wrong? Write it down. "NWP features
  improve MAE by less than 5% against the strongest baseline" is a falsifier; "the results
  are disappointing" is not.

The original project did do this once, for the v5 experiment, and it worked — the
pre-registered outcome bands in [`11 §7b`](../docs/11_nwp_forecast_features.md) meant the
response to the result was decided before the result existed. **Do it from the start, not
in iteration five.**

## 1.5 Set up the repository first

Non-negotiable, and it is here because the original left it until it had accumulated months
of untracked work across multiple multi-day experiments
([register D4](00_mistake_register.md)):

- `git init`, a `.gitignore` that excludes `data/` and large outputs but **not** `docs/` or
  results files, and a first commit containing this contract.
- A rule: **commit before every job that runs longer than an hour.**

## Exit gate

You may not proceed to stage 2 until:

1. A one-page contract exists that a supervisor could read and challenge, naming which of
   the three projects this is.
2. The latency table is filled in, and the set of features that survive it is written down.
3. The evaluation protocol — split, seeds, baselines, metrics, falsifier — is committed.
4. `git log` shows at least one commit.

**How this gate can fail:** if you cannot name the falsifier, or if the latency column has a
blank in it, you are not ready. Both were blank in the original project for its entire
duration.

---

Next: [Stage 2 — Data availability audit](02_stage2_data_availability_audit.md)
