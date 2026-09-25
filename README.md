# SPL baselines

LLM/VLM baselines for SPL, scored by SPL's own metrics so every number lands in one table.

## What is here

All settings live in a file under `configs/` — there are no command-line arguments, so a
run is reproducible from one file. Copy `configs/default.py`, change the values, and pick
it with `BASELINE_CONFIG`. That file documents every setting inline and is the reference;
this README covers only the ones whose consequences are not obvious.

Run everything from \<ROOT\> which has both `SPL` and `baseline_spl` repos inside it.
```bash
cd <ROOT>
BASELINE_CONFIG=my_experiment python -m baseline_spl.VLM.cap.run                    # Code-as-Policies
BASELINE_CONFIG=my_experiment python -m baseline_spl.VLM.demo2code.run              # Demo2Code
```

Demo2Code's released code is a git submodule, so clone with it:
```bash
git clone --recurse-submodules <url> baseline_spl
# already cloned without it:
git submodule update --init --recursive
```

`BASELINE_CONFIG` defaults to `default`; an unknown name fails listing what exists.

Output goes to `runs/<config>_<baseline>/`, so two configs never overwrite each other. The
baseline part spells out the condition — `simple_exp_cap_images_nosketch`,
`default_demo2code_text` — so runs of different conditions never collide either. Setting
`run_name` in the config overrides the whole thing.

| Variant | Evidence it receives | Selected by |
|---|---|---|
| **CaP (no demo)** | instruction only — the world-knowledge control | `CapConfig.use_demo = False` |
| **CaP (text)** | instruction + serialized demonstration | `use_demo = True`, `demo_modality = "text"` |
| **CaP (images)** | instruction + keyframe **images** | `use_demo = True`, `demo_modality = "images"` |
| **Demo2Code-text** | instruction + symbolic state, staged summarization (upstream-faithful) | `Demo2CodeConfig.variant = "text"` |
| **Demo2Code-VLM** | instruction + keyframe **images**, summarized by a vision model | `Demo2CodeConfig.variant = "vlm"` |
| **SayCan** | demonstrations + one primitive at a time, **no program** | `python -m baseline_spl.VLM.saycan.run` |
| **DreamCoder (B3-a)** | demonstrations only — **no LLM**; programs come from typed enumeration | `python -m baseline_spl.symbolic.dreamcoder.run` |
| **LILO (B3-b)** | the same search, plus an LLM proposer and library auto-documentation | `python -m baseline_spl.symbolic.lilo.run` |

SayCan is the control for whether a program is needed at all: it scores the available
primitives at each step and executes the best, so `row(5)` needs five placements chosen
separately and `row(11)` eleven. It reports every metric the others do except
`program_accuracy`, which is recorded as `null` with verdict `"no_program"` — the blank
column is the finding, not a failure. `recursive_learn` / `recursive_infer` swap the
per-step loop for a single call that emits the whole plan (see Cost, below).

Crossed with that, `sketch_mode` controls what signature information reaches program
generation:

- `"corrected"` — the sketch, after SPL's `validate_and_correct_sketch`.
- `"none"` — nothing. The model picks the class name and arguments itself, told only the
  structural requirement (one numeric argument plus `objects: list`) without which the
  class cannot be executed at all. The sketch still runs *after* generation, against the
  now-registered class, to produce the instantiations.

Inference uses the sketch either way.

They form an evidence ladder, so each adjacent pair is a single-variable contrast:
CaP-text vs Demo2Code-text isolates **staged recursive summarization** (Demo2Code's actual
contribution); text vs images isolates **privileged 3-D state**, now available in both
methods.

`CommonConfig.learn` and `CommonConfig.inference` choose the phases. The concept library is
loaded and saved through the same knobs `SPLConfig` uses, with this package's values:
`load_concept_checkpoint` names the library to load (default `None`, which loads nothing),
`skip_loading_concepts` leaves concepts out of it, `ignore_learnt_concepts` (default True)
skips learning what that library already holds, and `concept_save_path` is where the run
writes. Load and save are independent, so nothing is inferred from the other — to continue
an interrupted run, to add concepts to a finished one, or to run inference in a separate
invocation, set `load_concept_checkpoint` to `runs/<run_name>/concept_library.pt`. The
metric files are then appended to; with `None` they are rewritten and the harness warns
first. Neither path may point inside SPL's tree (`assert_not_spl_library`).

## Layout

```
configs/           THE files you edit: one per experiment, all the same structure
  default.py       copy this to start a new one
config.py          loader: picks a configs/ file via BASELINE_CONFIG
common/            shared spine, reused by the VLA/ and neurosymbolic/ baselines later
  config.py        machinery only: applies the config, owns PARITY_CRITICAL and the
                   fairness invariants
  factory.py       builds a harness from a run config; derives the run directory
  harness.py       owns an SPL instance; drives learn/infer and writes the metric JSONs
  llm_backend.py   drop-in replacement for Demo2Code's call_openai_api
  sketch.py        shared SketchAgent: signatures, SPL's correction, post-hoc grounding
  codegen.py       validation + retry loop (identical budget for all baselines) and
                   class_signature, which reads a model-invented signature back
  dsl_prompt.py    DSL docs, library context, required class shape, worked example
  primitive_stats.py   SRN per-direction delta mean/std for the prompt
  serialize_text.py    demonstration -> Demo2Code's [Scenario i] text format
  serialize_visual.py  keyframes -> downscaled PNGs for the VLM
VLM/cap/           Code-as-Policies
VLM/demo2code/     Demo2Code (our adapter)
VLM/saycan/        SayCan: step-by-step primitives, no program
symbolic/          the search baselines: DreamCoder (B3-a) and LILO (B3-b)
                   read in this order -- each file depends only on the ones above it
  lattice.py       the DSL's semantics as pure functions over an immutable state
  gaussian.py      the same semantics over SPL's probabilistic focus (mean + covariance)
  ir.py            the Term ADT that is searched, and the grammar levels
  oracles.py       a hand-written reference term per concept -- a test fixture, not pipeline
  bridge.py        DreamCoder types/primitives/grammar, and Term <-> Program
  lower.py         a found Term -> the Python concept class SPL's metrics score
  srn.py           the placement primitive's per-direction Gaussian, as a lookup table
  tasks.py         demonstrations -> synthesis tasks; demo selection
  evaluate.py      how a candidate is judged correct (exact / tolerance / SRN likelihood)
  search.py        the wake phase: enumerate, score, keep frontiers
  stitch_bridge.py STITCH compression and grammar re-weighting (Sleep-G)
  recognition.py   the neural recognition model and dreaming (Sleep-R)
  driver.py        the wake/sleep loop that drives all of the above
  harness.py       plugs the loop into SPL's metrics, resume and saving
  lilo/            B3-b's three additions: proposer, namer, prompts, harness
  _dreamcoder.py   import shim; keeps Primitive.GLOBALS clean (scaffolding, not pipeline)
third_party/
  demo2code/       the authors' clone, UNTOUCHED, pinned as a git submodule
  lilo/            the LILO/DreamCoder clone, UNTOUCHED, pinned as a git submodule
tests/             regression tests (see "Tests" below)
  goldens/         recorded artifacts a refactor must reproduce
runs/<config>_<baseline>/   outputs: concept_library.pt, training_metrics.json,
                   inference_metrics.json, learning_times.json, llm_cache/,
                   sketch_cache/, demo2code_artifacts/
```

## Design

**Only the concept-generation step is swapped.** SPL already owns the executor, the
concept library, the metrics, stability and the inference loop, so each baseline holds an
`SPL` instance and replaces exactly one step. Nothing in `SPL/` is modified, and nothing
in the upstream Demo2Code clone is modified.

**Demo2Code runs the authors' released code.** `common/llm_backend.py` reimplements their
`call_openai_api` message assembly exactly (system split on `<end_of_system_message>`,
few-shot as alternating turns) but dispatches through SPL's `openaiClient`, then
monkeypatches it in. So the pipeline is theirs; only the LLM backend is substituted, which
is what gives model parity, a response cache and token accounting. Their `lmp.py` is
shimmed out rather than imported — it star-imports `shapely`/`astunparse` purely for the
CaP machinery we reimplement in `VLM/cap/fgen.py`.

**Fairness invariants**, all enforced by `tests/test_leakage.py`:
- A baseline may only load from, and save to, its own library; `assert_not_spl_library`
  refuses either path inside SPL's tree, so no baseline inherits SPL's learned concepts
  and none can write over them. `BaselineConfig` declares both rather than inheriting
  them, because `SPLConfig`'s defaults name SPL's own run directory.
- Each variant gets a private `SketchAgent` cache; `BaselineConfig` also deep-copies
  `evaluation_config` and `generalize_config`, because `SPLConfig` otherwise shares one
  instance of each and an override would silently reconfigure SPL itself.
- `CapConfig.use_demo = False` never reads `meshes`/`rgbs`/`depths`/`masks`.
- Prompt worked-examples use the invented concept `zigzag_lane`, never an evaluated one.
  **Never put `row`/`tower`/`pyramid`/... in a prompt example** — it hands over the answer.

What baselines are given that SPL's Generalize stage is not, all deliberate and all in the
generous direction, since they have no Plan stage to compensate them: the SRN's
per-direction delta table, and the whole concept library rather than the subset a plan
called.

**Shared Sketch.** All methods parse instructions with SPL's `SketchAgent` and are given
the same initialized sketch, so a baseline failure is a *learning* failure, and integer
argument arity matches ground truth (otherwise `check_program_equivalence` returns
`program_accuracy = None`, "undecided", and the headline column is blank).

**Library context.** Every generation prompt carries the concepts learned so far, with
their source, so a new concept can be built out of them — the same thing SPL's Generalize
stage receives. It is rendered by SPL's own `GeneralizeAgent._render_library_context`
(borrowed in `dsl_prompt.build_library_block`) so the two cannot drift. SPL filters it to
the concepts its plans actually called; a baseline has no plans, so it gets the whole
library, which is the generous direction.

**Execution-grounded verification is off by default.** Baselines retry on parse/structure
failure only; SPL's `_build_concept_class_evaluator` loop stays an SPL contribution.
`use_evaluator_feedback = True` is the ablation (`common/evaluator.py`). Each class runs on
the demonstrations and the retry gets SPL's report: crash, block count, per-block distance
to the demo's final state, bookkeeping. A class passes when every block is within
`sketch_val_state_error_threshold`. SPL's reward and its MCTS-plan reference are not given,
because they come from the Plan stage. Image-only variants get the report without distances.
It needs `sketch_mode="corrected"` and `use_demo=True`, and SayCan ignores it. The outcome is
recorded as `evaluator` in `training_metrics.json`.

## Model parity

Pinned in each config file as `CODEGEN_MODEL` / `VLM_MODEL`. `BaselineConfig.from_run_config`
compares `codegen_model` and `vlm_model` with SPL's `GeneralizeConfig.llm_model`. On a mismatch
it warns and asks you to type `yes`; any other answer, or no terminal to answer from
(background run, pytest), stops the run.
`PARITY_CRITICAL` in `common/config.py` — `concepts`, `num_demos_per_concept`,
`codegen_model`, `vlm_model`, `max_code_retries`, `use_evaluator_feedback` — must be identical across baselines;
`assert_parity` refuses to build a config whose baseline class overrides one. That list
lives in `common/` rather than the config files so it cannot drift between experiments.

> SPL's `GeneralizeConfig.llm_model` is `gpt-5.1-codex-max`, which OpenAI has
> **deprecated** — it 404s on both endpoints, as does `gpt-5.2-codex`. SPL's Generalize
> stage cannot run as configured. When you update `SPL/config/spl_config.py`, set it to
> `CODEGEN_MODEL` so the comparison stays valid. `gpt-5.3-codex` also works but is
> Responses-endpoint only; the backend switches automatically if you pick it.

`service_tier` applies to every LLM and VLM call. `"flex"` is cheaper but requests queue,
so a run can sit idle for a long time before anything happens; `"default"` is standard
processing. If a run seems hung with no output, check this first.

## Concept anonymisation

`SPLConfig.concept_name` (`"normal"` | `"reversed"` | `"randomized"`) renames the concept
inside every language instruction, to ask whether a program came from the demonstration or
from the model's prior on the word "tower". See `overview.md` for the mechanism.

Set it in `SPL/config/spl_config.py` and nowhere else: `BaselineConfig` subclasses `SPLConfig`,
and the sketch agent scopes its cache directory from the same global, so every baseline shares
one value by construction. That is why it is **not** in `PARITY_CRITICAL` — that list is
compared against `CommonConfig`, which is standalone and does not declare it.
A run whose mode is not `"normal"` gets an `_anon<mode>` suffix on its run directory, so a
natural and an anonymised run never share a `plan_library.json`, a sketch cache or an
`llm_cache`.

What each baseline actually sees:

- **CaP, Demo2Code, SayCan** — anonymised end to end, with no baseline-side code. They read
  `language_instruction` from the same dataloader the rewrite hooks into, and their records are
  already keyed by the dataset concept with `pred_concept` alongside.
- **B3-b (LILO)** — the proposer, the LLM that writes programs, reads `task.instruction` and is
  anonymised with them. The **namer** needed fixing: it renders each usage as
  `building a <task name>`, and task names are the dataset's concept however the instruction was
  rewritten. Its names become the library documentation that later proposer prompts carry, so
  the leak propagated. `search.alias_task_name` now aliases what is rendered; `SearchTask.name`,
  `_record_program`, `_score_window` and `heldout`'s `PROGRAM_LIB` lookups keep natural names.
- **B3-a (DreamCoder)** — **the ablation is vacuous for it, by construction.** It reads no
  language at all; the only model anywhere near it is the shared sketch agent, which supplies
  the concept and argument name and does receive the anonymised instruction. Report B3-a's
  anonymised numbers as unchanged-by-design rather than as a null result.

## Known behaviour worth knowing

- **The domain is anisotropic**: horizontal block pitch ≈0.109 m, vertical ≈0.050 m. The
  serializer estimates pitch per concept and falls back to those constants.
- **Direction axes are not the obvious ones**: `RIGHT` is −y and `FRONT` is −x. The naive
  guess makes every baseline build a rotated structure. The vectors are imported from the
  simulator's `ParameterSettings.DIRECTIONS`, and `tests/test_serialize.py` additionally
  checks the SRN's learned mean delta agrees with them on every direction.
- **The serializer does not name directions.** A step can be diagonal, or have no single
  LEFT/RIGHT/... name, so deciding that is the model's job. It gets the primitive's
  per-direction delta statistics instead (`include_primitive_stats`, on by default) —
  turning that off is the ablation for "did the model infer the directions, or did we
  hand them over?".
- `coordinate_mode` picks how positions are written: `"lattice"` (integer grid cell) or
  `"raw"` (metres to 2 decimals). Both are relative to the first placed block, so scenes
  at different table positions stay comparable.
- Each demonstration is a separate scene, so the header lists objects per scenario
  (`scenario_1_objects=...`). It has to live there: upstream discards anything between
  `[Scenario i]` and `State 2:` for scenarios after the first.
- With `sketch_mode="none"` the model names the class itself, so the registered name can
  differ from the dataset's. Records are keyed by the **dataset** concept with the
  registered one alongside as `pred_concept` — that lets the metric files still diff
  against SPL's, and lets a record survive pruning through the name actually registered.
  Whether a concept is learned again is decided from the library alone, exactly as
  `learn_spl_concept` decides it, so a concept registered under an invented name is not
  recognised and is relearned. A demonstration whose instruction will not
  ground onto the registered class is dropped with a warning (`partially_grounded`), never
  by failing the run.
- `validate_and_correct_sketch` returns unchanged, with no LLM call, when the demos already
  agree on concept and argument names. On a single fresh concept `"corrected"` is therefore
  often identical to no correction; it bites once the library grows and names start
  colliding.
- Long demos (`arch_bridge` has 31 keyframes) cost a lot of image tokens in the VLM
  variant. `vlm_max_keyframes` subsamples as a last resort and warns loudly, because every
  keyframe here is a placement.
- **SayCan's `stop_reason` needs reading alongside its scores.** The loop ends on `done`,
  `all_placed` or `max_steps`. Only `done` is the agent judging the structure finished;
  `all_placed` means `objects.pop(0)` ran out, and when a scene holds exactly as many
  blocks as the structure needs (`row(5)` with 5 objects) that hands it the count for free.
  It is recorded per instruction so a run cannot look like counting when it was not.
- SayCan costs one LLM call **per action**, so ~8-10 per instruction against 1-4 per
  *concept* for the program baselines. Check the token ledger after one concept before a
  sweep. `max_steps` caps the worst case.
- **`recursive_learn` / `recursive_infer = False`** drops that to one call per instruction:
  the model emits the whole plan at once and it is executed in order (`stop_reason`
  `plan_complete`, or `no_plan` if nothing parseable came back). Still program-free, so it
  answers the same research question — but the model never sees the state its own actions
  produced, so a drifting plan is never corrected. The knobs are separate because learning
  touches `num_demos_per_concept` demos while inference touches the whole test set, so
  inference is where the per-action cost lands; `recursive_learn=True` with
  `recursive_infer=False` buys most of the saving. Say which mode each reported number came
  from, and note the asymmetry when they differ — plans cached per-step are then reused as
  worked examples by a one-shot call. Do not mix the two in one column.

## Tests

```
python -m pytest baseline_spl/tests/ -q       # everything, ~7 minutes
```

Standalone entry points, for the ones that also print a report:

```
python -m baseline_spl.tests.test_leakage     # fairness invariants, per-demo prompt content
python -m baseline_spl.tests.test_serialize   # direction mapping, SRN agreement, both coordinate modes
python -m baseline_spl.tests.test_resume      # which library gets loaded; metric merging and pruning
python -m baseline_spl.tests.test_saycan      # flat-plan scoring and all three stop conditions
python -m baseline_spl.tests.test_ir_fidelity # every oracle reproduces run_gt_program, n = 1..12
python -m baseline_spl.tests.test_bridge      # Term <-> Program round trip; no foreign primitives
```

The symbolic baselines add: `test_ir_fidelity`, `test_lowering`, `test_bridge`, `test_tasks`,
`test_stitch_bridge`, `test_search_parallel`, `test_recognition`, `test_gaussian`, `test_lilo`.

`test_serialize` is the important one: it guards constants that, if wrong, silently make
every baseline look worse than it is.

**A passing suite is not sufficient evidence that a change is safe.** Twice now a regression
has exited 0 and passed every test: a metric silently became `None`, and the recognition model
trained for 45 gradient steps instead of 10,000. For anything touching `symbolic/`, also diff
the run artifacts against the recorded goldens:

```
BASELINE_CONFIG=determinism_pin python -m baseline_spl.symbolic.dreamcoder.run
python -m baseline_spl.tests.compare_artifacts \
    runs/determinism_pin_dreamcoder_standard_maha2 tests/goldens/determinism_pin
```

`determinism_pin` ends its search on the MDL band rather than the clock, so it is reproducible
run to run; the smoke configs are clock-bound, so compare those with `--clock-bound`.
