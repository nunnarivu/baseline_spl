'''
run.py

Runs the DreamCoder baseline (B3-a). Settings come from the DreamCoderConfig of the config
file named by BASELINE_CONFIG (default: configs/default.py). No command-line arguments.

    BASELINE_CONFIG=my_experiment python -m baseline_spl.symbolic.dreamcoder.run

Unlike the LLM baselines this makes no generation calls -- the only model use is the shared
SketchAgent, which is cached -- so a re-run costs nothing.
'''

from __future__ import annotations

from baseline_spl.common.config import BaselineConfig, assert_parity
from baseline_spl.common.factory import resolve_run_name
from baseline_spl.config import DreamCoderConfig
from baseline_spl.symbolic.harness import SearchHarness


def share_sketch_cache(configs, run_cfg) -> str:
    '''Repoint the sketch cache at a directory shared by the symbolic runs.

    Every run otherwise starts with an empty `<run_dir>/sketch_cache`, so a fresh 16-concept
    run re-sketches ~35 instructions through the API before it has searched anything. Those
    calls are pure waste: the same instruction yields the same signature in every symbolic
    run, for a reason specific to this baseline.

    Why it is safe HERE and not for the LLM baselines
    -------------------------------------------------
    `_cache_key` is the bare instruction (SPL/model/sketch.py) -- no sketch_mode, no baseline,
    no library state -- while `extract_function_signature` consults `_similar_concepts`, which
    ranks against the LIVE concept library. So a sketch is not a pure function of its
    instruction in general, and a shared cache is only sound where the context is fixed.

    For the symbolic baselines it is fixed, twice over:

      * `SearchHarness.learn_all` computes every sketch in one batch BEFORE any concept is
        registered, so all of them see an empty library;
      * the sketch supplies only the concept name, the argument name and the integer
        parameter. The program comes from search, so a sketch can change what a concept is
        called but never what it does.

    The LLM baselines have neither property. Under `sketch_mode="none"` they call
    `SharedSketch.ground` AFTER registering a model-invented class, and the invented name is
    then cached under the bare instruction key -- which is exactly the contamination this
    directory must not inherit. Hence the scoping below, and hence LLM baselines are left on
    their per-run caches.

    The directory is therefore keyed by sketch_mode, so a symbolic run in some future
    no-sketch ablation cannot poison the corrected-mode entries. Returns the path used.
    '''
    import os

    from baseline_spl.common.config import RUNS_ROOT

    if not getattr(run_cfg, "share_sketch_cache", True):
        return configs.sketch_config.cache_dir

    mode = getattr(run_cfg, "sketch_mode", "corrected")
    shared = RUNS_ROOT / "_sketch_cache_symbolic" / str(mode)
    os.makedirs(shared, exist_ok=True)
    configs.sketch_config.cache_dir = str(shared)
    return str(shared)


def evaluator_suffix(run_cfg) -> str:
    """A run-name fragment naming the acceptance rule.

    Round 1 (lattice + exact) keeps its bare name, so its existing run directories and the
    numbers already reported stay exactly where they are. Every Round 2 acceptance rule adds
    a fragment, which is what lets the headline and the ablation live side by side instead of
    overwriting one another -- these are the two rows of the fairness comparison.
    """
    name = getattr(run_cfg, "evaluator", "exact")
    if name == "exact":
        return ""
    if name == "distance":
        from baseline_spl.symbolic.evaluate import default_distance_threshold

        epsilon = getattr(run_cfg, "accept_epsilon", None)
        return f"_dist{(default_distance_threshold() if epsilon is None else epsilon):g}"
    return f"_maha{getattr(run_cfg, 'accept_tau', 3.0):g}"


def build(run_cfg) -> SearchHarness:
    '''Mirrors common/factory.build, minus the LLM backend this baseline does not use.'''
    assert_parity(run_cfg)
    baseline = f"dreamcoder_{run_cfg.grammar_level}" + evaluator_suffix(run_cfg)
    if not run_cfg.use_library:
        baseline += "_nolib"
    if run_cfg.demo_selection != "distinct_params":
        baseline += f"_{run_cfg.demo_selection}"
    configs = BaselineConfig.from_run_config(run_cfg, resolve_run_name(run_cfg, baseline))
    # Before the harness, which is what constructs SPL and therefore the SketchAgent that
    # reads this path.
    share_sketch_cache(configs, run_cfg)
    return SearchHarness(configs)


def main() -> None:
    if DreamCoderConfig is None:
        raise SystemExit(
            "The active config file defines no DreamCoderConfig. Copy the one in "
            "baseline_spl/configs/default.py into it.")
    if not (DreamCoderConfig.learn or DreamCoderConfig.inference):
        raise SystemExit("Set learn and/or inference to True in the active config file")

    harness = build(DreamCoderConfig)
    if DreamCoderConfig.learn:
        harness.learn_all()
    if DreamCoderConfig.inference:
        # A finished demo-level run registers no concept, and `infer_all` raises on an empty
        # library. Generalisation to unseen sizes is what "inference" means for that run, so
        # route it there instead of failing.
        if _is_finished_demo_run(harness):
            harness.generalise_all()
        else:
            harness.infer_all()


def _is_finished_demo_run(harness) -> bool:
    '''A run whose artifacts say demo-level, with nothing registered to infer from.'''
    import json
    import os

    stats = os.path.join(harness.configs.run_dir, "search_stats.json")
    if not os.path.exists(stats):
        return False
    try:
        granularity = json.loads(open(stats).read()).get("task_granularity")
    except Exception:  # noqa: BLE001 - a malformed artifact is not a reason to crash here
        return False
    return granularity == "demo" and not harness.spl.concept_library.inductive_concepts


if __name__ == "__main__":
    main()
