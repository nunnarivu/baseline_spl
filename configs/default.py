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
from SPL.config.spl_config import GeneralizeConfig

CODEGEN_MODEL = "gpt-5.1"
VLM_MODEL = "gpt-5.1"


class CommonConfig:
    '''Shared by every baseline. Change these here, not in a subclass.'''

    # What to run.
    learn = True
    inference = True

    # Concept library to load, named exactly as SPLConfig names it. None loads nothing and
    # the run starts from an empty library — so to resume an interrupted run, to add concepts
    # to an existing one, or to run inference separately from learning, put that run's library
    # here: runs/<run_name>/concept_library.pt. Concepts it holds are not learned again and
    # the metric files are appended to; with None they are rewritten, and the harness warns
    # before it does. It may not point inside SPL's tree — see assert_not_spl_library.
    load_concept_checkpoint = None
    # Concepts to leave out when that library is loaded, so they are learned again.
    skip_loading_concepts = ()
    # True skips learning a concept the loaded library already holds. False relearns it, and
    # register_inductive_concepts then raises on the duplicate name (as in SPL), so pair it
    # with skip_loading_concepts for the concepts you actually want redone.
    ignore_learnt_concepts = True

    # Data. Use ALL_CONCEPTS for the full sweep. row/tower are smoke-test concepts —
    # an LLM already knows them, so they test plumbing, not capability; pins, psi,
    # arch_bridge and x are the ones that discriminate.
    concepts = list(ALL_CONCEPTS)
    # 3, not 2: 14 of 99 concepts take 2-3 integer arguments, and with only 2 demos a
    # recovery hole is under-determined (an affine fit of one argument can reproduce another
    # exactly -- measured: `3*length - 7` equals `breadth` on [(3,2), (4,5)]). At 2 demos
    # those 14 concepts report `ambiguous` and register nothing; 3 distinct sizes resolves it.
    num_demos_per_concept = 3
    num_workers = 4

    # Models (see the parity note above).
    codegen_model = CODEGEN_MODEL
    vlm_model = VLM_MODEL

    # Which provider serves codegen_model/vlm_model ('openai' | 'qwen' | 'vertexai' | 'google').
    # Independent of SPL's own GeneralizeConfig.llm_provider — a baseline can point at a
    # different provider than SPL is currently using, while still reaching it through SPL's
    # stored connection details (SPL/config/spl_config.py), not a second copy of them.
    llm_provider = "openai"

    # OpenAI processing tier for every LLM/VLM call.
    #   'flex'    : cheaper, but requests queue and can take much longer
    #   'default' : standard processing
    service_tier = "flex"

    # Retries per concept class, shared by parse failures and (below) evaluation failures.
    max_code_retries = 3

    # Completion budget for codegen.generate_with_retries and call_vlm. This used to be a
    # hardcoded 6000, tied to no real limit. Reasoning models (e.g. Qwen3's "thinking" mode)
    # can spend the whole budget on hidden reasoning and return empty content if cut off
    # before ever answering -- a large budget makes that far less likely. Qwen's own vLLM
    # deployment reports a 262144-token max_model_len, which is prompt + completion
    # combined -- the server hard-rejects (400) a request whose max_tokens leaves no room
    # for the prompt, and that particular error isn't one LLMBackend's parameter-retry loop
    # recognizes, so this is deliberately short of the ceiling: 62144 tokens of headroom is
    # generous next to these calls' actual prompts (a few thousand tokens even with a few
    # dozen keyframe images).
    codegen_max_tokens = 200000

    # Run each generated class on the demonstrations and retry with a report (crash, block
    # count, per-block distance to the demo's final state, bookkeeping), as SPL's Generalize
    # evaluator does. Passes when every block is within sketch_val_state_error_threshold.
    # Not given: SPL's reward and its MCTS-plan reference. Image-only variants get the report
    # without distances. Needs sketch_mode='corrected' and use_demo=True; SayCan ignores it.
    # False: parse retries only, the loop stays an SPL contribution.
    use_evaluator_feedback = False

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

    sketch_mode and use_evaluator_feedback are ignored — SayCan emits actions, so there is
    no signature to give it and no class to evaluate.
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

    # Attempts (1 initial + retries) to get a usable JSON score reply per step, before giving
    # up with "scoring_failed". Some models/deployments are measurably less reliable here than
    # others (e.g. a self-hosted Qwen deployment was measured at ~50% per-call failure on a
    # dynamic JSON key containing an embedded quote, such as shift_focus("RIGHT") — both in
    # schema-guided and loose JSON mode); raise this per-config for such a provider rather than
    # changing the shared action-string representation.
    score_reply_attempts = 2


class DreamCoderConfig(CommonConfig):
    '''B3-a: DreamCoder-style program search over SPL's DSL. No LLM except the shared
    sketch, so this run costs no API budget and is reproducible forever.

    sketch_mode is used only for the class name and argument name; the program itself
    comes from search, never from a model.
    '''

    # None means "calls no model", which is the literal truth for B3-a: enumeration writes the
    # program, and the only LLM anywhere near this run is the SHARED sketch agent, which uses
    # SPL's own SketchConfig.llm_model rather than either of these.
    #
    # Inheriting CommonConfig's model instead made this baseline declare one it never calls, so
    # `from_run_config` compared it against SPL's and stopped every run at "Type yes to continue
    # with these models" -- including background runs and pytest, which have no stdin.
    codegen_model = None
    vlm_model = None

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

    # --- generalising demo-level solutions to unseen sizes ------------------------------ #
    # A demo-level run solves each demonstration separately, so its programs are closed and
    # the size is a literal. Anti-unifying a concept's solved demos recovers the parameterised
    # class (`symbolic/generalise.py`), which is more than published DreamCoder does -- so it
    # is reported as a separate, disclosed variant, never as the headline B3-a number.
    recover_concept_classes = True

    # DreamCoder's OWN route to an unseen size: enumerate the held-out task again under the
    # learnt library (`dreamcoder.py:567`). The two arms answer "which generalises better".
    heldout_enumeration = True

    # Sizes to test generalisation on, with ground truth from `oracles.trace`. Keep inside the
    # literal ceiling (`_literal_ceiling`, currently 13) or the SEARCH arm cannot express the
    # target at all and the comparison measures the grammar rather than the mechanism.
    heldout_sizes = (8, 10, 12)

    # ONE budget for the whole held-out phase, not per task -- upstream passes a single
    # `enumerationTimeout` for the entire test set. With a global grammar every held-out task
    # shares one enumeration, so each candidate program is tested against all of them at once.
    heldout_timeout = 300.0

    # Fallback for a concept with one demo, or several demos that share a size. Temporary:
    # every concept is expected to carry 2-3 DISTINCT sizes once the dataset is updated, and
    # recoveries made this way are tagged so they cannot be mistaken for the real thing.
    recover_allow_single_demo = True

    # At k>=3 demos, rebuild the class from k-1 and require it to reproduce the excluded one.
    # A class that only fits the demos it was built from has not generalised.
    recover_holdout_validate = True

    # continuationType, as tower, LOGO, regex and LILO's own `structures` domain all set it
    # (`LAPSGrammar.uniform(primitives, continuationType=ttower)`). Makes state threading
    # implicit, so programs are shorter at equal semantics.
    use_continuation_type = True

    # Wake/sleep rounds. Each one searches the still-unsolved tasks, compresses the
    # solutions into abstractions, and re-weights the grammar on what was used.
    #
    # 6, not 3: at 99 concepts the library has far more room to grow across rounds than the
    # 5-16 concept smoke/probe runs this file used to size for, and each round is what lets
    # a freshly-learned abstraction feed back into the NEXT round's enumeration and
    # recognition training. Sized jointly with enumeration_timeout below for a ~22-23h total
    # (recognition_timeout=1800s below is unchanged) -- see enumeration_timeout's comment.
    search_iterations = 6

    # Seconds of enumeration per iteration, across all unsolved tasks.
    # Cost grows as ~e^(0.79 x MDL), so this buys description length logarithmically:
    # the shallow concepts land around MDL 12-15 and the composites at 35+, which no
    # feasible budget reaches. Raising this does not change which band is reachable.
    #
    # 10800 (3h), not 300 (5min): 300s was sized for a smoke/probe run, not a real budget.
    # 6 iterations x (10800s enumeration + up to 1800s recognition) = ~21h core loop, leaving
    # ~1-2h for demo loading/sketching and streamed scoring across 99 concepts -- targets a
    # single ~24h job slot. Re-check against the actual per-iteration timing this run logs in
    # its first two iterations and adjust search_iterations if it is tracking short or long.
    enumeration_timeout = 10800.0
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

    # The recognition model's architecture, matching LILO's own loader
    # (src/models/laps_dreamcoder_recognition.py) rather than upstream's library defaults,
    # which are weaker (contextual=False, biasOptimal=None, auxLoss=False). These lived only
    # as inline fallbacks in the harness, which meant the production value for every run was
    # written in a file nobody would think to look in.
    #   contextual : predict a bigram transition matrix over productions rather than
    #                marginal unigram weights, so a per-task grammar can say "inside a loop
    #                body, prefer place"
    #   hidden     : must equal the feature extractor's output width -- frontierKL computes
    #                _MLP(features).expand(1, features.size(-1)) and assumes it is preserved
    recognition_hidden = 64
    recognition_contextual = True
    recognition_bias_optimal = True
    recognition_auxiliary_loss = True
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

    #   'exact'       : integer cells must match      (requires observation_mode='lattice')
    #   'distance'    : every placement within accept_epsilon metres -- the SAME bar the LLM
    #                   baselines apply (common/evaluator.py checks every block against
    #                   sketch_val_state_error_threshold), so the numbers share one table
    #   'mahalanobis' : every placement within accept_tau sigma of SPL's predicted focus.
    #                   Same machinery as SPL's own focus belief, but still a 0/1 check --
    #                   never SPL's graded reward, which is SPL's contribution to claim.
    # Only three of the six pairs are meaningful; the rest are rejected at startup.
    evaluator = "mahalanobis"

    # evaluator='mahalanobis' only.
    # 2.0, not the 3.0 the synthetic pre-flight chose. Measured on the real 8-concept sweep:
    # correct programs topped out at 0.62 sigma while the nearest WRONG program (staircase's
    # best near-miss) sat at 3.11, so 3.0 was running on a 0.11-sigma margin and 5.0 -- which
    # that pre-flight declared safe -- would have accepted a wrong program as solved.
    # 2.0 sits in the empty gap with 3x headroom below and 1.5x above, and changes no result
    # measured so far. See Finding M: a threshold calibrated on synthetic negatives is only
    # trustworthy at its strict end, because real near-misses are the hard cases by
    # construction while invented ones are wrong in obvious ways.
    accept_tau = 2.0

    # evaluator='distance' only. None means "use SPL's own sketch_val_state_error_threshold",
    # which is what common/evaluator.py applies per block for the LLM baselines -- so the bar
    # tracks SPL instead of being a second copy that can drift from it.
    accept_epsilon = None

    # `saved` lowers to assign_focus(object_id=...). False restores the PREDICTED mean; True
    # snaps to the demonstration's observed centroid, which lets a candidate re-synchronise to
    # the demo at every `saved` and makes acceptance easier for exactly the composite concepts
    # the search struggles with.
    #
    # False now mirrors SPL: `assign_focus_to_placed_position = True` sends the focus where the
    # PLAN placed the block, not where the demo has it, precisely so a wrong plan cannot jump
    # back onto the correct structure. True was faithful to the older SPL and is kept as a knob;
    # tests/test_gaussian.py measures the difference.
    saved_resync = False

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

    # LILO does call a model (the proposer and the library namer), unlike B3-a, so it
    # takes the models named at the top of this file instead of inheriting DreamCoderConfig's
    # None. With None, LLMBackend fell back to SPL's own GeneralizeConfig.llm_model.
    codegen_model = CODEGEN_MODEL
    vlm_model = VLM_MODEL

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
    #
    # 21600 (6h) = 2x DreamCoderConfig.enumeration_timeout above, keeping the ratio you chose
    # at smoke scale. search_iterations stays matched at 6 for comparable wake/sleep depth.
    # Flagged, not silently absorbed: 6 x (21600s + up to 1800s recognition) = ~39h core loop
    # BEFORE the LLM proposer's own latency (measured 400-560s per iteration on the 5-concept
    # smoke test; unpredictable and likely larger across 99 concepts' worth of still-unsolved
    # tasks). This will not fit one 24h job slot. Plan on it spanning 2+ segments via
    # load_concept_checkpoint (see lilo_resume.md) -- true iteration-exact resume isn't built
    # yet, so a segment boundary loses only the search work of whichever concept was mid-wake
    # when it was cut, same as DreamCoder above.
    enumeration_timeout = 21600.0
    search_iterations = 6
