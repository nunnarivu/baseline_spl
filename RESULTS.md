# Symbolic baseline runs (B3-a DreamCoder, B3-b LILO)

What has been run, what it measured, and which numbers are still trustworthy. Kept here rather
than under `runs/`, which is gitignored, so deleting a run directory does not delete its record.

**Read the caveat in every row.** Two defects found on 2026-09-16 invalidate parts of what is
below, and the runs predate the 99-concept dataset.

---

## Standing caveats

| Caveat | Effect |
|---|---|
| **Request-type bug** (fixed 09-16) | Demo-level runs scored every closed program at the concept-level type, so its likelihood was −∞. `corpus_mdl` returned `inf` for the baseline *and* every candidate, making `score < best_score` a comparison of `inf < inf`, so **no abstraction could ever be adopted**. Every demo-level `abstractions = 0` below is this bug, not a result. Concept-level MDL was always finite (15.1 → 14.4), so concept-level `abstractions = 0` IS a result. |
| **16-concept dataset** | All runs below used `DATA/structures` with 16 concepts. `ALL_CONCEPTS` is now 99 and the dataset is `DATA/new_data`. Solve counts are not comparable to anything run after. |
| **Acceptance changed 09-16** | These runs used `srn_likelihood` + `accept_criterion=mahalanobis`, τ=2.0, `saved_resync=True`. The evaluator is now `mahalanobis` (same maths, renamed) or `distance`, and `saved_resync` defaults to False. |
| **Fitted poses** | 317/385 demos carry a pose fitted from images that is off by >2 cm, usually a block-height z error. Refitting is deferred. |

---

## 2026-09-08 — four 12-hour runs, 16 concepts

Launched with the gated launcher after an earlier ungated attempt exhausted memory (see
*Operational notes*). Each: 3 iterations × 14,400 s enumeration, `DATA/structures`, continuous
observations, τ = 2.0.

| Run | Config | Granularity | Solved | Abstractions |
|---|---|---|---|---|
| B3-a concept | `r2_full16_rfix` | concept | **9 / 16** | 0 — a real result |
| B3-a concept, no Sleep-R | `r2_full16_norecog` | concept | **9 / 16** | 0 |
| B3-a demo | `r2_demo16` | demo | **18 / 31** | 0 — **the bug**, not a result |
| B3-a demo, no Sleep-R | `r2_demo16_norecog` | demo | **18 / 31** | 0 — **the bug** |

Solved concepts (identical in all four): `row, column, tower, inverted_row, inverted_column,
diagonal_45, diagonal_135, diagonal_225, diagonal_315`. The 18 demo-level solves are exactly
those 9 concepts × 2 demos, so **the parameterised framing was not handicapping the search** —
concept-level and demo-level reach the same concepts.

**Recognition changed nothing.** Both pairs are identical, and every solve landed in
**iteration 0**, before any recognizer existed; iterations 1–2 added zero. Diagnosed rather than
assumed: the model trained fully (9,990 gradient steps, per-task grammars genuinely used), but
had almost nothing to learn from — concept-level dreaming yielded 10–20 valid samples per 500
(2–4%) against demo-level's 283–310 (57–62%), so it saw ~25 training examples. With 0
abstractions a per-task grammar can only *reweight* 26 fixed productions; it cannot invent the
structure the 7 unsolved concepts need.

The 7 never solved: `staircase, inverted_staircase, arch_bridge, pyramid, pins, x,
isosceles_right_triangle`.

## 2026-09-09 — generalisation to unseen sizes

Post-processing over the finished demo-level run; no re-search. Two arms on one held-out set
(ground truth from the oracles at n = 8, 10, 12).

- **Arm B, recovered classes**: 9/9 concepts with solved demos recovered a parameterised class
  (8 by anti-unification, `diagonal_135` by the tagged single-demo fallback, since both its
  demos share n=4). All nine registered and scored **`program_accuracy = 1.0`,
  `plan_accuracy = 1.0`** — metrics structurally undefined for a demo-level run before this.
- **Held-out**: arm A (enumerate the task afresh) **19/48**; arm B (instantiate the class)
  **27/48**. Arm A degrades with size — 9 at n=8, 5 at n=10, 5 at n=12 — because a diagonal's
  program is deeper than a row's and crosses the enumeration budget as the literal grows. Arm B
  is flat at 9 per size. Neither rescues a concept that was never solved.

**Caveat**: arm B is *not* published DreamCoder. It consumes the concept grouping and the demo
size labels, which DreamCoder's task formulation never receives. Report it as a disclosed
variant. The run also inherits the request-type bug, so the library it worked from was empty.

## Earlier — `symbolic_sweep8`

8 concepts, τ = 3.0, concept-level: 6 solved, 2 approximate, **1 abstraction** (the only
abstraction this project has ever learned):
`#(lambda (lambda (lambda (loop $1 (lambda (lambda (shift $4 (place $0)))) $0))))` — the shared
line-drawing skeleton of row/column/tower. Kept because `configs/r2_infer8.py` pins this run
directory by name.

---

## Operational notes

- **Memory is the binding constraint, not CPU.** One run peaks at ~99 GB during demo loading
  (a PyTorch `DataLoader`, `num_workers=4`, each worker holding geometry; PSS ≈ RSS, so none of
  it is shared) and drops to ~1 GB for the remaining 12 hours. Four concurrent loads want
  ~400 GB on a 251 GB machine: the first launch attempt died with two runs reporting
  `DataLoader worker exited unexpectedly` and three vanishing with no traceback at all.
- **The fix is to serialise the loads**, not to stagger them by a guess. `runs/_launch/` holds a
  gated launcher that waits for the previous run to be *past* loading and back under 5 GB before
  starting the next, plus a memory watchdog that pauses rather than kills. Measured with the
  gate: peak 89 GB, zero watchdog interventions.
- Enumeration itself is cheap: it uses a `fork` pool against a ~1 GB parent.
- Without a recognizer every task shares one grammar, so `_units` returns a single unit and
  `cpus > 1` does nothing. `cpus = 1` in the no-recognition ablations is correct, not starved.

## What was deleted, and why it was safe

Cleared on 2026-09-16: 11 `*_probe` and 5 `check_*` directories (created by config-building in
the tests, not by runs — now built into a temp root so they stop reappearing), `test_leakage`
and `test_serialize` scratch dirs, and the regenerable `symbolic_smoke` / `demo_smoke` /
`determinism_pin` run directories, which every golden check rebuilds.

The `demo_smoke` directory had been silently **resuming a stale `concept_library.pt`** left by
the 09-09 recovery run, so a fresh smoke run skipped row/tower/column as "already learned" and
tested only `staircase`. Stale run directories change test outcomes; that is why they go.

## 09-17 mixed-arity smoke (before the first real 24h launch)

Four ~45-min runs, `configs/mixed_smoke.py` / `mixed_smoke_demo.py` (now under
`tests/configs/`, `concepts=[row, rectangle, cuboid, podium, wall]` -- one per arity 1/2/3/0,
plus `wall` for the inexpressible path), against the N-argument code (Stage 2) after its own
mixed-arity smoke had already landed once. This run is what surfaced two further real bugs,
both fixed the same session, both with regression tests:

- **Search readback used one arity for the whole run** (`not tasks[0].closed`, i.e. "arity 1
  unless demo-level") instead of each task's own -- `rectangle`/`cuboid`/`podium` were found by
  enumeration and then discarded as untranslatable while only `row` survived at concept level.
  Fixed in `search.py`, `driver.py`, `harness.py` (each solution now reads back at its own
  task's arity); `test_every_arity_is_read_back_at_its_own_arity` guards it.
- **`stitch_bridge.extend()` rebuilt the grammar from `primitives_for(level)` (the default
  `{1,2}` literal set) instead of the grammar's own** after every successful compression,
  silently dropping any wider literal `_literal_ceiling` had registered. Since `_literal_ceiling`
  floors at 12 for EVERY run (concept or demo level), this meant no abstraction could ever
  score as an improvement once one was extracted -- `corpus_mdl` returned `inf` for every
  candidate against the literal-stripped grammar, so `best_compression` silently kept
  `abstractions=[]` every time, logged as the unremarkable "no abstraction lowered the corpus
  description length." Fixed by rebuilding from the grammar's own non-invented productions;
  `test_extend_preserves_the_grammars_own_literals` guards it.
- Also fixed, lower-severity: `saved_is_exact` (the focus-restore lowerability check) defaulted
  to arity 1, so `cuboid` (arity 3) was misreported "may fail on the live executor" from an
  `IndexError` in the check itself, not from an actual executor failure.

Result after both fixes, all four combinations clean (zero tracebacks, zero silent scoring
failures): `dc-concept`/`lilo-concept` each solved `row`, registered 3 approximate concepts,
`wall` recorded inexpressible, inference `program_accuracy=1.0` on all 6 test instances.
`dc-demo`/`lilo-demo` each solved 3/11 closed demos and **recovered `row`'s general class via
anti-unification**, confirming the fix live (grammar grew 26->27 productions on the first
kept abstraction, finite MDL, not `inf`); inference again `program_accuracy=1.0`, 6/6 `ok`.

Full suite: 110 passed / 1 known failure (`test_keyframe_cells_match_ground_truth`, data drift,
unrelated). Run directories deleted after this entry was written -- regenerable by rerunning
`tests/configs/mixed_smoke.py` / `mixed_smoke_demo.py`.

## 09-20 the stale-term defect, found in the finished 99-concept LILO run

`runs/lilo_gptluna_concept_level_depthimages_run1` (99 concepts, concept level, `demo_modality
= both`, `gpt-5.6-luna`) reported **11/99** at `program_accuracy == 1.0`. That number was an
artifact. The real figure is **~36/99**.

**The defect.** `Solution.program` was a derived property over a mutable `frontier`;
`Solution.term` was a stored field. `add_exact`/`add_approximate` changed the program without
invalidating the term, and both writers were guarded `if solution.term is None`
(`search.py:455`, `driver.py:474`), so they refused to refresh a stale one. Iteration 0's
enumeration left an approximate program's term in place; iteration 1/2's accepted LLM
proposal replaced the program under it. The concept was then lowered, registered and scored
from a term belonging to a program the run had already discarded.

Measured by re-translating every recorded program: `match 70 / mismatch 28 / untranslatable 0`.
**All 28 mismatches were `solved_by: llm`; zero enumeration solves mismatched.** The run log
gives the same 28 independently -- `[iter 0] proposer solved 0`, `[iter 1] 21`, `[iter 2] 7`.
Every one of the 28 scored `program_accuracy` 0.0 with the correct program recorded beside it,
and the failure was logged as the unremarkable "accepted by the distance evaluator but NOT
equivalent to ground truth", which reads as an evaluator problem and is not one.

**Nothing expensive was damaged.** The wake/sleep loop (`_corpus` -> STITCH -> re-weight) reads
`solution.programs`, never `solution.term`, so the search, the library and the LLM solves were
unaffected. Only term -> lower -> register -> score consumed the bad input, which is why the
run was recoverable from its own `search_programs.yml` with no enumeration and no LLM call
(`tests/tools/rescore_run.py`, output in `runs/lilo_run0_corrected`).

**Fix.** `Solution.term` is now derived from `Solution.program` and re-derives whenever the
program changes (`search.py`), with `Solution.arity` carried from the task so it needs no
caller. Both external writers were deleted -- there is no guard left to get wrong.

**Two further defects found in the same pass:**

- `add_exact` appended, sorted and TRIMMED the frontier, then assigned `self.distance`
  unconditionally, so the 6th-best accepted program (deleted by the trim at
  `maximum_frontier = 5`) still set the distance the solution reports. Invisible under
  `exact`; real under `distance`/`mahalanobis`.
- `test_keyframe_cells_match_ground_truth` had been pinned to `DATA/structures`, which no
  longer exists, so it had been failing with `FileNotFoundError` -- checking nothing -- for the
  whole period this defect went unnoticed. Repointed to `DATA/new_data` and all 99 concepts.

**The acceptance threshold is not the problem, and cannot be made into the solution.** 25 of the
28 recovered programs are exactly correct against ground truth at sizes the run never tested
(n = 7,8,9), so the evaluator was right to accept them. Only 3 accepted programs
(`arch_bridge`, `battlement`, `podium`) are genuinely wrong. Their accept margins are 0.0488 /
0.0513 / 0.0532 -- while CORRECT programs score as high as 0.060 (`filled_triangle`), 0.0597
(`rocket`), 0.0573 (`koodai`). **The two populations overlap, so no threshold separates them**:
0.06 m is wider than one lattice cell (~0.05 m), so the metric cannot distinguish a structure
from one with a block a cell out.

What does separate them exactly is a structural invariant: all three place two blocks into one
cell (`arch_bridge` 20 blocks into 14 cells, `battlement` 11 into 7, `podium` 6 into 4), which
ground truth never does. Over every solved concept: **3 caught, 0 false positives, 0 missed.**
Recorded as a `duplicate_cells` warning in `training_metrics.json` rather than a rejection, so
acceptance semantics stay comparable across runs -- the flag says which results not to trust.

**A coordinate-frame trap worth knowing, since it looks exactly like data corruption.**
`demo_cells` measures from the first block PLACED; `run_gt_program` measures from where the
focus STARTED. They differ by whatever a program shifts before its first placement, so `rocket`
and `tree` -- whose `PROGRAM_LIB` definitions open with `for i in range(2):
shift_focus('right')` -- come out uniformly translated by `(0, 2, 0)` against ground truth while
being the same shape. `rocket` was initially misdiagnosed here as a corrupt demo; it is
correct, and it counts toward the 36. Every comparison of placements against ground truth now
normalises both sides to their own first placement (`tests/test_term_fidelity.normalise`,
`tests/test_tasks.normalise`). A translation is not an error; a different shape is.

**New coverage.** `tests/test_term_fidelity.py` validates term -> ground truth, lowered class ->
term, and the bridge round trip, with ground truth derived mechanically from `PROGRAM_LIB`
(98/99 concepts; `wall` reported `inexpressible`, not failed) rather than from the 16
hand-written oracles. Concepts added to the dataset are covered with no oracle to write. It
also carries the regression test for this defect: every `term` a run records must equal
`to_term` of the `program` recorded beside it.

**LILO vs SPL** (`SPL/runs/spl_100struct_gptluna_run1`, 56/99): SPL leads 56 to ~36, and the
sets are complementary rather than nested -- 26 both, 30 SPL only, **10 LILO only**
(`corner_ramp`, `diagonal_ramp`, `filled_triangle`, `stepped_pyramid`, `triangle_prism`,
`valley`, `picket_fence`, `rocket`, `chair`, `smiley`).
