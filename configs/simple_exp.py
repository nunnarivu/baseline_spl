
from __future__ import annotations

from SPL.config.spl_config import ALL_CONCEPTS  # noqa: F401  — for setting `concepts`


# All methods must generate code with the same model, or the results compare models
# instead of methods. SPL's configured 'gpt-5.1-codex-max' is deprecated and 404s (so is
# 'gpt-5.2-codex') — set SPL's GeneralizeConfig to CODEGEN_MODEL when you sync it.
# 'gpt-5.3-codex' also works but is Responses-endpoint only; the backend handles that.
CODEGEN_MODEL = "gpt-5.1"
VLM_MODEL = "gpt-5.1"


class CommonConfig:
    '''Shared by every baseline. Change these here, not in a subclass.'''

    # What to run.
    learn = True
    inference = True

    # Load concepts learned earlier: resumes an interrupted run, adds concepts to an
    # existing one, and lets inference run separately from learning. Already-learned
    # concepts are skipped and metric files are appended to, not overwritten.
    resume = True
    # Library to resume from. None = this run's own concept_library.pt.
    resume_from = None

    # Data. Use ALL_CONCEPTS for the full sweep. row/tower are smoke-test concepts —
    # an LLM already knows them, so they test plumbing, not capability; pins, psi,
    # arch_bridge and x are the ones that discriminate.
    concepts = ["row", "tower", "staircase", "pyramid", "arch_bridge"]
    num_demos_per_concept = 2
    num_workers = 4

    # Models (see the parity note above).
    codegen_model = CODEGEN_MODEL
    vlm_model = VLM_MODEL

    # OpenAI processing tier for every LLM/VLM call.
    #   'flex'    : cheaper, but requests queue and can take much longer
    #   'default' : standard processing
    service_tier = "flex"

    # Retries on invalid/unparseable code only. Baselines deliberately get no
    # execution-grounded verification — that loop is an SPL contribution.
    max_code_retries = 3

    # What signature information reaches program generation.
    #   'corrected' : the sketch, after SPL's validate_and_correct_sketch
    #   'none'      : nothing — the model picks the class name and arguments itself.
    #                 The sketch still runs afterwards, against the now-registered class,
    #                 to produce the instantiations; inference is unchanged either way.
    sketch_mode = "none"

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
    vlm_max_keyframes = 40

    # Output directory under runs/. None means use the config file's own name, so two
    # configs can never overwrite each other's results.
    run_name = "test_dry_run"


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
    max_expansion_depth = 2


class Demo2CodeConfig(CommonConfig):
    '''Demo2Code (Wang et al., NeurIPS 2023), running the authors' released pipeline.'''

    # 'text': symbolic state + the authors' staged recursive summarization.
    # 'vlm':  keyframe images summarized by a vision model, with no 3-D state.
    variant = "text"
