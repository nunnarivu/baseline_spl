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
BASELINE_CONFIG=my_experiment python -m baseline_spl.VLM.demo2code.spl_baseline.run # Demo2Code
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

`CommonConfig.learn` and `CommonConfig.inference` choose the phases. `resume` (default
True) loads the concepts this baseline learned before, so an interrupted run continues,
new concepts can be added to an existing run, and inference can run in a separate
invocation. Already-learned concepts are skipped and the metric files are appended to.
Set `resume_from` to continue from a different run's library.

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
VLM/demo2code/     the authors' clone, UNTOUCHED, plus spl_baseline/ (our adapter)
tests/             regression tests (see "Tests" below)
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
- A baseline may only resume from its own library; `assert_not_spl_library` refuses any
  checkpoint inside SPL's tree, so no baseline inherits SPL's learned concepts.
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

**No execution-grounded verification for baselines.** They retry on parse/structure
failure only. SPL's `_build_concept_class_evaluator` loop stays an SPL contribution.

## Model parity

Pinned in each config file as `CODEGEN_MODEL` / `VLM_MODEL`, currently `gpt-5.4`.
`PARITY_CRITICAL` in `common/config.py` — `concepts`, `num_demos_per_concept`,
`codegen_model`, `vlm_model`, `max_code_retries` — must be identical across baselines;
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
  registered one alongside as `pred_concept` — that keeps resume's skip working and lets
  the metric files still diff against SPL's. A demonstration whose instruction will not
  ground onto the registered class is dropped with a warning (`partially_grounded`), never
  by failing the run.
- `validate_and_correct_sketch` returns unchanged, with no LLM call, when the demos already
  agree on concept and argument names. On a single fresh concept `"corrected"` is therefore
  often identical to no correction; it bites once the library grows and names start
  colliding.
- Long demos (`arch_bridge` has 31 keyframes) cost a lot of image tokens in the VLM
  variant. `vlm_max_keyframes` subsamples as a last resort and warns loudly, because every
  keyframe here is a placement.

## Tests

```
python -m baseline_spl.tests.test_leakage     # fairness invariants, per-demo prompt content
python -m baseline_spl.tests.test_serialize   # direction mapping, SRN agreement, both coordinate modes
python -m baseline_spl.tests.test_resume      # which library gets loaded; metric merging
```

`test_serialize` is the important one: it guards constants that, if wrong, silently make
every baseline look worse than it is.

Note both `common/harness.py` and `VLM/cap/run.py` currently contain a `breakpoint()`.
Run with `PYTHONBREAKPOINT=0` for anything unattended.
