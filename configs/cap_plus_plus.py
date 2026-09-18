
from __future__ import annotations

from SPL.config.spl_config import ALL_CONCEPTS  # noqa: F401  — for setting `concepts`


# All methods must generate code with the same model, or the results compare models
# instead of methods. SPL's configured 'gpt-5.1-codex-max' is deprecated and 404s (so is
# 'gpt-5.2-codex') — set SPL's GeneralizeConfig to CODEGEN_MODEL when you sync it.
# 'gpt-5.3-codex' also works but is Responses-endpoint only; the backend handles that.
CODEGEN_MODEL = "gpt-5.6-luna"
VLM_MODEL = "gpt-5.6-luna"


class CommonConfig:
    '''Shared by every baseline. Change these here, not in a subclass.'''

    # What to run.
    learn = True
    inference = False

    # Concept library to load (None loads nothing). Concepts it holds are not learned again
    # and the metric files are appended to. See configs/default.py for the full note.
    load_concept_checkpoint = None
    skip_loading_concepts = ()
    ignore_learnt_concepts = False  # True: ignore the learnt concepts and learn remaining

    # Data. Use ALL_CONCEPTS for the full sweep. row/tower are smoke-test concepts —
    # an LLM already knows them, so they test plumbing, not capability; pins, psi,
    # arch_bridge and x are the ones that discriminate.
    concepts = ALL_CONCEPTS
    num_demos_per_concept = 3
    num_workers = 4

    # Models (see the parity note above).
    codegen_model = CODEGEN_MODEL
    vlm_model = VLM_MODEL

    # Which provider serves codegen_model/vlm_model ('openai' | 'qwen' | 'vertexai' | 'google').
    llm_provider = "openai"

    # OpenAI processing tier for every LLM/VLM call.
    #   'flex'    : cheaper, but requests queue and can take much longer
    #   'default' : standard processing
    service_tier = "default"

    # Retries per concept class, shared by parse failures and (below) evaluation failures.
    max_code_retries = 3

    # Completion budget for codegen.generate_with_retries and call_vlm. See default.py's
    # copy of this field for the full rationale (reasoning models can spend the whole
    # budget on hidden reasoning and return empty content if cut off before answering).
    codegen_max_tokens = 100000

    # Run each generated class on the demonstrations and retry with a report (crash, block
    # count, per-block distance to the demo's final state, bookkeeping), as SPL's Generalize
    # evaluator does. Passes when every block is within sketch_val_state_error_threshold.
    # Not given: SPL's reward and its MCTS-plan reference. Image-only variants get the report
    # without distances. Needs sketch_mode='corrected' and use_demo=True; SayCan ignores it.
    # False: parse retries only, the loop stays an SPL contribution.
    use_evaluator_feedback = True

    # What signature information reaches program generation.
    #   'corrected' : the sketch, after SPL's validate_and_correct_sketch
    #   'none'      : nothing — the model picks the class name and arguments itself.
    #                 The sketch still runs afterwards, against the now-registered class,
    #                 to produce the instantiations; inference is unchanged either way.
    sketch_mode = "corrected"

    # How a placement's position is written in the serialized demonstration.
    #   'lattice' : integer grid cell, e.g. (0, -1, 0)
    #   'raw'     : centroid relative to the first placed block, 2 decimals, e.g. (0.00, -0.11, 0.00)
    coordinate_mode = "raw"

    # Include a table of the placement primitive's per-direction mean/std delta, so the
    # model can work out which shift_focus direction a movement corresponds to.
    # False is the ablation: did the model infer the directions, or did we hand them over?
    include_primitive_stats = True

    # Images sent to a vision model. Source frames are 720x1280; demos run 4-37 keyframes.
    vlm_max_image_px = 512
    # Every keyframe here is a placement, so subsampling deletes construction steps.
    # Last resort only; the serializer warns when it fires.
    vlm_max_keyframes = 100

    # Output directory under runs/. None means use the config file's own name, so two
    # configs can never overwrite each other's results.
    run_name = "cap_plus_plus_gptluna_run3"


class CapConfig(CommonConfig):
    '''Code-as-Policies (Liang et al., ICRA 2023).'''

    # True: instruction + demonstration, matching Demo2Code's input so the only
    # difference is Demo2Code's staged summarization.
    # False: instruction only — the world-knowledge control, which never reads
    # meshes or rgbs (enforced by tests/test_leakage.py).
    use_demo = True

    # How the demonstration is shown when use_demo is True.
    #   'text'   : the serialized [Scenario i] block, as Demo2Code receives it
    #   'images' : the keyframe images, passed into the single code-generation call
    demo_modality = "images"

    # Depth limit for CaP's recursive generation of helpers the code calls but
    # never defines.
    max_expansion_depth = 3


class Demo2CodeConfig(CommonConfig):
    '''Demo2Code (Wang et al., NeurIPS 2023), running the authors' released pipeline.'''

    # 'text': symbolic state + the authors' staged recursive summarization.
    # 'vlm':  keyframe images summarized by a vision model, with no 3-D state.
    variant = "text"


class SayCanConfig(CommonConfig):
    '''SayCan (Ahn et al., 2022): one primitive at a time, no program. The control for
    whether a program is needed at all.

    sketch_mode and use_evaluator_feedback are ignored — SayCan emits actions, so there is
    no signature to give it and no class to evaluate.
    '''

    # How the demonstrations are shown: "text" : the serialized [Scenario i] block, or the "images".
    demo_modality = "images"

    # Per-action scoring, or one call for the whole plan.
    #   True  : SayCan proper — score every action from the current state, execute the
    #           best, re-ask. One LLM call per action, so cost scales with structure size.
    #   False : one call emits the whole plan. The model never sees the state its own
    #           actions produced, so nothing corrects a plan that drifts.
    # Set separately per phase: learning touches num_demos_per_concept demos, inference
    # touches the whole test set, so inference is where the per-action cost lands. When
    # these differ, say so in the results — plans cached per-step are then reused as
    # worked examples by a one-shot call, and the two modes are not interchangeable.
    recursive_learn = True
    recursive_infer = False

    # Cap on actions per instruction. Nothing stops the loop on its own — the executor's
    # _is_terminal is always False at inference — so this and the done() action are what
    # end it. Generous enough for the longest structures; a run that hits it says so.
    max_steps = 40

    # Cached plans from earlier instructions, shown as worked examples.
    plan_library_top_k = 3
