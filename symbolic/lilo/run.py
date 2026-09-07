'''
run.py

Runs the LILO baseline (B3-b). Settings come from the LiloConfig of the config file named by
BASELINE_CONFIG (default: configs/default.py). No command-line arguments.

    BASELINE_CONFIG=lilo_smoke python -m baseline_spl.symbolic.lilo.run

Unlike B3-a this does make generation calls, so it draws on the API budget -- though the
shared response cache means a re-run of an unchanged config costs nothing.
'''

from __future__ import annotations

from baseline_spl.common.config import BaselineConfig, assert_parity
from baseline_spl.common.factory import resolve_run_name
from baseline_spl.common.llm_backend import LLMBackend
from baseline_spl.config import LiloConfig
from baseline_spl.symbolic.dreamcoder.run import evaluator_suffix, share_sketch_cache
from baseline_spl.symbolic.lilo.harness import LiloHarness


def build(run_cfg) -> LiloHarness:
    assert_parity(run_cfg)
    baseline = f"lilo_{run_cfg.grammar_level}" + evaluator_suffix(run_cfg)
    # The demonstration modality is a fairness knob, so it must not be silently mixed into
    # one directory either.
    modality = getattr(run_cfg, "demo_modality", "coords")
    if modality != "coords":
        baseline += f"_{modality}"
    if not run_cfg.use_library:
        baseline += "_nolib"
    if run_cfg.enumeration_timeout <= 0:
        baseline += "_llmonly"
    if not getattr(run_cfg, "auto_document", True):
        baseline += "_nodoc"
    # Strict LILO and the named-variable variant must never share a run directory: they are
    # the two rows of the fidelity comparison.
    if getattr(run_cfg, "allow_named_variables", False):
        baseline += "_named"
    if run_cfg.demo_selection != "distinct_params":
        baseline += f"_{run_cfg.demo_selection}"
    configs = BaselineConfig.from_run_config(run_cfg, resolve_run_name(run_cfg, baseline))
    # Shared with B3-a: both symbolic baselines sketch the same instructions to the same
    # answers, and the sketch cannot influence the program either way. See the helper for why
    # this is sound here but not for the LLM baselines, which keep their per-run caches.
    share_sketch_cache(configs, run_cfg)
    return LiloHarness(configs, LLMBackend(configs))


def main() -> None:
    if LiloConfig is None:
        raise SystemExit(
            "The active config file defines no LiloConfig. Copy the one in "
            "baseline_spl/configs/default.py into it.")
    if not (LiloConfig.learn or LiloConfig.inference):
        raise SystemExit("Set learn and/or inference to True in the active config file")

    harness = build(LiloConfig)
    if LiloConfig.learn:
        harness.learn_all()
    if LiloConfig.inference:
        harness.infer_all()


if __name__ == "__main__":
    main()
