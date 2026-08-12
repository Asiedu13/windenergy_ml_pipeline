# Stage 9 — Write-up and project hygiene

> **What this stage is:** the rules for recording results so the documentation does not
> quietly become false.
> **Why it needs a stage of its own:** the original project's documentation was one of its
> genuine strengths — thirteen cross-linked documents that made it possible to diagnose its
> own ceiling. It is *because* the docs were good that their failure modes are worth
> studying. One entire document explains a measurement artefact. Another, explicitly "the
> doc written to be quoted", has been rewritten once and is stale again today.
> **Effort:** continuous, not a phase.

## 9.1 One source of truth for numbers

> **What:** a single results file that every other document cites.
> **Why:** the original had numbers in an auto-generated report, a hand-maintained results
> log, and thirteen documents. The auto-generated report is **overwritten by whichever run
> finished last** — so at one point it silently stopped being a v4 artefact and became a
> pilot-run artefact, while a document still described it as the former.

Rules:

1. **One results file is authoritative.** Every other document cites it rather than
   restating figures.
2. **Auto-generated reports are disposable.** Never cite them as a record; archive a copy
   under a distinct name if a particular run matters.
3. **Every number carries its provenance**: window, horizon, truth source, seed count.
4. **No document may state a number it did not compute or cite.**

## 9.2 Write interpretation last

> **The failure mode:** a well-argued essay explaining a result that turns out not to exist.

`09_why_deep_learning_underperformed.md` is a careful, technically literate document about
why four deep architectures lost to gradient boosting. The premise was a 24-hour alignment
bug. It was persuasive precisely because it explained something everyone already believed.

So:

- **Descriptive records** (what was run, what came out) can be written immediately.
- **Interpretive documents** (why it happened, what it means) wait until the numbers have
  survived at least one attempt to break them — a different seed, a different split, a
  different truth source.
- Any interpretation written early carries a visible provisional marker.

## 9.3 Supersede; never silently edit

When a result is overturned, the temptation is to fix the old document. Resist it — the
reasoning that led to the wrong conclusion is the most instructive thing in the project.

The original handled this well and the pattern is worth copying exactly:

| Document | Treatment | Why |
|:---|:---|:---|
| Doc 09 (refuted) | Kept **verbatim** behind a supersede banner | Preserves the original reasoning |
| Doc 07 (quotable) | **Rewritten** against corrected numbers | Leaving refuted figures in the doc people quote was too dangerous |
| Doc 06 (history) | v2/v3 tables kept **struck through** | Chronology stays intact |

The rule that decides which: **rewrite what will be quoted, preserve what will be learned
from.**

## 9.4 Record failed predictions

When something is pre-registered and the outcome contradicts it, **leave the prediction
visible** and note that it failed. The original did this once — its next-steps document
predicted sequence models would exploit forecast trajectories better than trees, and every
deep model instead lost ground. Keeping the failed prediction is what makes the
pre-registration worth anything; quietly rewriting it converts a real test into a
post-hoc rationalisation.

## 9.5 Never write moving state as a fact

A cache count written into a document was wrong within minutes because the cache was still
growing ([register D3](00_mistake_register.md)).

**Documents contain check commands, not snapshots.** Anything that changes — file counts,
job status, progress — is represented by the command that reveals it.

## 9.6 Version control

From [stage 1](01_stage1_deployment_contract.md), restated because the original violated it
for months:

- Commit **before** every job longer than an hour.
- Commit after every result worth keeping.
- Documentation and results files are tracked; large data and outputs are not.
- Never let a branch accumulate a whole iteration of work uncommitted. The original reached
  **two commits** with all of v2–v5 living in one machine's working tree, through several
  multi-day experiments.

## 9.7 What may and may not be claimed

Maintain an explicit boundary list, and update it whenever a result lands. The original's
version of this is one of its best artefacts — a plain statement of what the evidence did
and did not support, written while it was tempting to overclaim.

For a project of this shape the boundaries will include:

- **May** claim measured relative improvements against baselines computed on the same window.
- **May** claim the ceiling analysis and what it implies about where effort should go.
- **May not** claim a benchmark is beaten without stating site, window difficulty, truth
  type, and seed count.
- **May not** present physics-derived power metrics as validated physical yield.
- **May not** quote an offline number as an operational one — see
  [stage 8](08_stage8_deployment_rehearsal.md).

## 9.8 A note on documentation style

Keep the practice the original adopted late and should have adopted early: **every section
opens with what it covers and why it exists**, written so a reader outside the project can
follow it. A glossary defining every term the documents use without explaining.

The test: hand it to someone who has never seen the project. If they can say what the
project does, what it found, and what it does not claim, the documentation works.

## Exit gate

The project is finished when:

1. One authoritative results file exists and every document cites it.
2. No document states a number it did not compute or cite.
3. Refuted material is superseded, not deleted; failed predictions remain visible.
4. Every headline number carries window, horizon, truth source, seed count, and its
   `study`/`operational` label.
5. The "may / may not claim" list is current.
6. Everything is committed.

**How this gate can fail:** any number in any document that cannot be traced to the run that
produced it.

---

Previous: [Stage 8](08_stage8_deployment_rehearsal.md) ·
Back to: [redo index](README.md)
