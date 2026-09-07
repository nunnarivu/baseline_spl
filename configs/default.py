'''
default.py

A baseline run configuration. Copy this file to configs/<name>.py, change the values,
and select it with the BASELINE_CONFIG environment variable:

    BASELINE_CONFIG=<name> python -m baseline_spl.VLM.cap.run

Every config file has this same structure. Which settings must stay identical across
baselines is machinery, not a knob: see PARITY_CRITICAL in common/config.py.

Run from the directory holding both SPL and baseline_spl.
'''

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
    sketch_mode = "corrected"

    # How a placement's position is written in the serialized demonstration.
    #   'lattice' : integer grid cell, e.g. (0, -1, 0)
    #   'raw'     : centroid relative to the first placed block, 2 decimals, e.g. (0.00, -0.11, 0.00)
    coordinate_mode = "lattice"

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
    run_name = None


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
    demo_modality = "text"

    # Depth limit for CaP's recursive generation of helpers the code calls but
    # never defines.
    max_expansion_depth = 2


class Demo2CodeConfig(CommonConfig):
    '''Demo2Code (Wang et al., NeurIPS 2023), running the authors' released pipeline.'''

    # 'text': symbolic state + the authors' staged recursive summarization.
    # 'vlm':  keyframe images summarized by a vision model, with no 3-D state.
    variant = "text"


class SayCanConfig(CommonConfig):
    '''SayCan (Ahn et al., 2022): one primitive at a time, no program. The control for
    whether a program is needed at all.

    sketch_mode is ignored — SayCan emits actions, so there is no signature to give it.
    '''

    # How the demonstrations are shown: the serialized [Scenario i] block, or the images.
    demo_modality = "text"

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
    recursive_infer = True

    # Cap on actions per instruction. Nothing stops the loop on its own — the executor's
    # _is_terminal is always False at inference — so this and the done() action are what
    # end it. Generous enough for the longest structures; a run that hits it says so.
    max_steps = 40

    # Cached plans from earlier instructions, shown as worked examples.
    plan_library_top_k = 3


class DreamCoderConfig(CommonConfig):
    '''B3-a: DreamCoder-style program search over SPL's DSL. No LLM except the shared
    sketch, so this run costs no API budget and is reproducible forever.

    sketch_mode is used only for the class name and argument name; the program itself
    comes from search, never from a model.
    '''

    # Which primitives the search may use. Bigger is more expressive and exponentially
    # harder to search, which is the ablation this exposes.
    #   'minimal'  : shift/place/loop only — the 10 concepts needing no focus save/restore
    #   'standard' : + saved and integer arithmetic — all 16 concepts expressible
    #   'extended' : + more literals and a conditional — for probing search degradation
    grammar_level = "standard"

    # Largest integer literal in the grammar. None = derive it from the task parameters
    # (see SearchHarness._literal_ceiling), which is what keeps the grammar right when the
    # dataset grows past these concepts. DreamCoder's tower domain hard-codes 1..49 because
    # its towers are that big; copying the number rather than the reasoning would add
    # productions we can never use, and every extra production dilutes the rest.
    # Set an integer to pin it for an ablation.
    int_literals_upto = None

    # Concepts whose demonstrations are reloaded together for scoring. Scoring needs the
    # meshes back, so this caps that peak: the search itself never holds them (see
    # SearchHarness._load_demos). Lower it if the machine is tight, raise it to reload less
    # often on a machine with room.
    scoring_chunk = 8

    # continuationType, as tower, LOGO, regex and LILO's own `structures` domain all set it
    # (`LAPSGrammar.uniform(primitives, continuationType=ttower)`). Makes state threading
    # implicit, so programs are shorter at equal semantics.
    use_continuation_type = True

    # Wake/sleep rounds. Each one searches the still-unsolved tasks, compresses the
    # solutions into abstractions, and re-weights the grammar on what was used.
    search_iterations = 3

    # Seconds of enumeration per iteration, across all unsolved tasks.
    # Cost grows as ~e^(0.79 x MDL), so this buys description length logarithmically:
    # the shallow concepts land around MDL 12-15 and the composites at 35+, which no
    # feasible budget reaches. Raising this does not change which band is reachable.
    enumeration_timeout = 300.0
    max_mdl = 100.0

    # Feed STITCH abstractions back into the grammar. False is the "no library growth"
    # control: it isolates how much comes from compression rather than raw enumeration.
    use_library = True
    stitch_max_arity = 3
    # Smoothing for the insideOutside re-weighting of production probabilities.
    # 30, matching LILO's own re-weighting call:
    #   new_grammar.insideOutside(frontiers_rewritten, pseudoCounts=30, iterations=1)
    #   (src/models/stitch_proposer.py:176-178)
    # We had 1.0, which concentrates the grammar far more aggressively on the handful of
    # productions the solved programs happen to use. With few solved tasks that is close to
    # overfitting the prior to the easy concepts.
    pseudo_counts = 30.0

    # Programs kept per solved task (DreamCoder's maximumFrontier). More than one gives
    # STITCH more shared structure to anti-unify and the recognition model more to train on;
    # the search keeps enumerating until every frontier is full, as enumerateForTasks does.
    maximum_frontier = 5

    # Worker processes for the wake phase. Parallelism is across *tasks*, so it only helps
    # once the recognition model gives each task its own grammar -- with one shared grammar
    # a single enumeration scored against every task is strictly better. Splitting MDL bands
    # instead does NOT work: lowerBound filters without pruning, so each slice re-walks the
    # whole tree (measured: a 16-way split ran 3x slower than serial).
    cpus = 16

    # Sleep-R: train a recognition model each iteration and enumerate the next wake under
    # its per-task grammars. False is the ablation -- report the comparison either way, a
    # null result is a result. With 16 concepts the model trains mostly on fantasies, which
    # is normal for DreamCoder but worth stating in the paper.
    use_recognition = True
    # How long Sleep-R actually trains. This is not a tuning knob to be set casually — it was
    # the difference between a working recognition model and a random one.
    #
    # Upstream's contract (`recognition.py:1269-1272`) is that `steps` and `epochs` both
    # default to 9,999,999 and the *timeout* stops training. An earlier version here passed
    # `recognition_epochs = 5` with no timeout, which caps the loop at
    # `epochs x len(frontiers)` gradient steps — with 9 solved tasks, **45 steps**. LILO's own
    # config asks for 10,000 (`template_lilo.json: "recognition_train_steps": 10000`).
    #
    # A network trained for 45 steps still emits a different grammar per task, because random
    # weights do that. So the wake phase looks conditioned, throughput changes, and nothing
    # appears broken — while the model carries no learned signal at all.
    recognition_steps = 10000
    recognition_epochs = None      # None = unbounded; `recognition_steps` is the budget
    recognition_timeout = 1800.0   # seconds, a safety cap on top of the step budget
    # Fraction of training data drawn from the generative model rather than solved tasks.
    helmholtz_ratio = 0.5

    # One task per concept with one example per demo (the scored path), or one task per
    # demo with the parameter baked in as a literal (DreamCoder as published).
    task_granularity = "concept"

    # ---------------------------------------------------------------------------------- #
    # What the search observes, and how it decides a program is correct.
    #
    # Round 1 gave the search integer lattice cells and accepted on exact equality. SPL is
    # never given a lattice -- it receives meshes and recovers actions by MCTS over a learned
    # stochastic primitive -- so that was a representation handed over rather than earned.
    # Measured: the rounding was lossless here (worst residual 0.209 of a cell against the
    # 0.5 that would misassign one), so it removed a *decision*, not a *difficulty*.
    #
    #   'lattice'    : integer cells                 (Round 1; the ablation row)
    #   'continuous' : centroids in metres relative to the first placed block -- the same
    #                  quantity CaP/Demo2Code get at coordinate_mode='raw'
    observation_mode = "continuous"

    #   'exact'          : integer cells must match      (requires observation_mode='lattice')
    #   'tolerance'      : every placement within accept_epsilon metres; deterministic, no
    #                      SRN in the loop, but charges a placement after ten compounding
    #                      shifts as strictly as the first
    #   'srn_likelihood' : SPL's own probabilistic focus -- executor._predict_focus's Kalman
    #                      step and executor._penalised_score. Identical machinery to SPL's
    #                      MCTS reward rather than an analogy to it.
    # Only three of the six pairs are meaningful; the rest are rejected at startup.
    evaluator = "srn_likelihood"

    # evaluator='srn_likelihood' only.
    #   'mahalanobis'      : accept iff every placement is within accept_tau sigma. Unit-free,
    #                        no per-demo calibration, degrades naturally as variance compounds.
    #   'penalised_loglik' : SPL's _penalised_score verbatim, as a mean shortfall in nats
    #                        below the best score achievable at that covariance.
    # The two are calibrated to each other: a Gaussian's log-density falls by d**2/2 at
    # Mahalanobis distance d, so accept_margin=4.5 IS accept_tau=3.0. Switching between them
    # therefore compares aggregation (worst placement vs mean, and whether the variance
    # penalty participates), not scale.
    accept_criterion = "mahalanobis"
    # 2.0, not the 3.0 the synthetic pre-flight chose. Measured on the real 8-concept sweep:
    # correct programs topped out at 0.62 sigma while the nearest WRONG program (staircase's
    # best near-miss) sat at 3.11, so 3.0 was running on a 0.11-sigma margin and 5.0 -- which
    # that pre-flight declared safe -- would have accepted a wrong program as solved.
    # 2.0 sits in the empty gap with 3x headroom below and 1.5x above, and changes no result
    # measured so far. See Finding M: a threshold calibrated on synthetic negatives is only
    # trustworthy at its strict end, because real near-misses are the hard cases by
    # construction while invented ones are wrong in obvious ways.
    accept_tau = 2.0
    # The Mahalanobis-equivalent margin, kept calibrated: a Gaussian's log-density falls by
    # d**2/2 at distance d, so tau=2.0 is 2.0 nats.
    accept_margin = 2.0
    # SPL's own lambda in log p(x|focus) - lambda*tr(Sigma) (executor._penalised_score).
    variance_penalty_weight = 1.2

    # evaluator='tolerance' only. ~1 sigma of the SRN's along-axis horizontal prediction.
    accept_epsilon = 0.03

    # `saved` lowers to assign_focus(object_id=...), which SPL resolves to the *observed*
    # centroid of the placed block. Mirroring that is faithful, but it also lets a candidate
    # re-synchronise to the demonstration at every `saved`, which makes acceptance easier for
    # exactly the composite concepts the search struggles with. False restores the predicted
    # mean instead; tests/test_gaussian.py measures the difference.
    saved_resync = True

    # Which demos a concept-level task learns from.
    #   'distinct_params' : prefer demos whose parameter differs, so the loop is forced
    #   'parity'          : the first N, exactly what SPL sees
    # Only `row` differs between these — its first two demos are both length 5. This is
    # an input asymmetry against SPL, so the chosen ids are written to demo_selection.json.
    demo_selection = "distinct_params"


class LiloConfig(DreamCoderConfig):
    '''B3-b: LILO — an LLM synthesizer, STITCH compression, and library auto-documentation,
    on the same wake/sleep driver as B3-a.

    B3-a measured a hard ceiling: enumeration cost grows as ~e^(0.79 x MDL), so the lines
    land at 12-15 nats and the composites at 23-79, out of reach at any feasible budget.
    An LLM does not sample by description length, so it is not bound by that exponential.
    Whether it actually clears the ceiling is what this run measures.

    Inherits every DreamCoderConfig knob, so the two baselines can be run at matched search
    budgets and differ by exactly LILO's contributions.
    '''

    # Sampling, taken from LILO's own experiment config
    # (third_party/lilo/experiments_iterative/templates/template_lilo.json):
    #   "n_queries_per_task": 4, "n_samples_per_query": 4, "temperature": 0.7
    # Each query re-randomises the order of the few-shot body tasks, and each asks the API
    # for n completions — that is where sample diversity comes from, not from asking the
    # model for several programs in one reply.
    llm_queries_per_task = 4
    llm_samples_per_query = 4
    llm_max_tokens = 3000
    # Ignored by the reasoning-model families (llm_backend strips it), kept for models that
    # do honour it.
    llm_temperature = 0.7

    # LILO's third contribution: name and describe each new abstraction so the next prompt
    # can refer to the library in words. False is the ablation that isolates its value.
    auto_document = True

    # The one deliberate deviation from published LILO, and it must be disclosed.
    # Measured justification: with strict LILO every candidate died BEFORE execution
    # (12 of 12, 10 at typecheck, 11 involving `saved`) — the model never produced a
    # program that typechecked, so it was never judged on whether it solved the task. The
    # bottleneck was de Bruijn notation, not program structure.
    # With this on, the model may write `(lambda (n) ...)` and we translate to indices
    # before parsing — an extension of the surface-form translation LILO already does for
    # function names (`show_program`). False reproduces the paper exactly; keep a False run
    # as the fidelity baseline.
    allow_named_variables = True

    # How the demonstration reaches the model. Round 2's fairness knob for B3-b, mirroring
    # what CaP and Demo2Code already expose.
    #   'coords' : raw centroids in the task language + the shift_focus delta table —
    #              parity with CaP / Demo2Code-text at coordinate_mode='raw'
    #   'images' : keyframe PNGs, geometry removed from the text — parity with CaP-images
    #              and Demo2Code-vlm
    #   'both'   : both channels
    # 'images' and 'both' are deviations from published LILO, which is text-only, and must
    # be disclosed as such. 'images' also forces the dataloader to load pixels.
    demo_modality = "coords"

    # LILO itself sets enumeration_timeout=10 — the LLM is the primary solver there. Search
    # is free where API calls are not, so this runs the enumerator far longer than LILO does
    # while keeping LLM usage exactly at LILO's level. Set to 0 for the LLM-only ablation.
    enumeration_timeout = 600.0
    search_iterations = 3
