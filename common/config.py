'''
config.py

``BaselineConfig`` — the SPL-compatible config object every baseline runs on.

This file holds no settings of its own; all knobs live in baseline_spl/config.py and are
copied in by ``from_run_config``. What it does own are the fairness invariants, which are
correctness guarantees rather than knobs:

  * ``load_concept_checkpoint`` and ``concept_save_path`` may only point outside SPL's
    tree, so a baseline never inherits the concepts SPL already learned and never writes
    over SPL's library. Pointing the first at a baseline's own library is what resumes a run.
  * Both are declared on ``BaselineConfig`` rather than inherited: SPLConfig's defaults
    name SPL's own run directory, so inheriting either is exactly the leak above.
  * The shared SketchAgent gets a per-run cache directory, so no baseline reads a
    signature that was computed with SPL's concept library in context.
  * SPLConfig shares one instance of each sub-config across all config objects, so they
    are deep-copied here; otherwise a baseline setting would change SPL's own behaviour.
'''

from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path

from SPL.config.spl_config import GeneralizeConfig, SPLConfig, SketchConfig

BASELINE_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = BASELINE_ROOT / "runs"


def _public_attrs(cls) -> dict:
    '''Settings declared on a run-config class and its bases.

    Walks the MRO rather than using dir(), which would also pull in metaclass
    attributes such as `mro`.
    '''
    attrs = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        attrs.update({k: v for k, v in vars(klass).items()
                      if not k.startswith("_") and not callable(v)})
    return attrs


def assert_not_spl_library(path) -> None:
    '''Refuse a checkpoint that lives inside the SPL package: loading it would hand the
    baseline the concepts SPL itself learned, and saving into it would overwrite them.
    Any path outside SPL is allowed.'''
    import SPL

    spl_root = Path(SPL.__file__).resolve().parent
    resolved = Path(path).resolve()
    if spl_root == resolved or spl_root in resolved.parents:
        raise ValueError(
            f"{path} is inside SPL's own tree. A baseline that loads SPL's concept library "
            f"is not a baseline, and one that saves into it destroys SPL's results. Point "
            f"load_concept_checkpoint / concept_save_path at a baseline run under {RUNS_ROOT}, "
            f"or set load_concept_checkpoint = None to start from an empty library."
        )


# Settings that must be identical across baselines for the comparison to mean anything.
# Kept here rather than in the config files: *which* fields must match is machinery, and
# duplicating the list per config would let it drift between experiments.
PARITY_CRITICAL = ("concepts", "num_demos_per_concept", "codegen_model",
                   "vlm_model", "llm_provider", "max_code_retries", "use_evaluator_feedback")


#: Parity-critical settings a baseline may opt OUT of by setting them to None, meaning "this
#: baseline never calls a model". Only the model fields: a baseline that calls no LLM cannot
#: meaningfully share one, and forcing it to name a model it never uses is what made every
#: symbolic run stop at the mismatch prompt. Every other field in PARITY_CRITICAL stays
#: mandatory, so `concepts` or `num_demos_per_concept` can never be opted out of.
OPTIONAL_WHEN_NONE = ("codegen_model", "vlm_model")


def assert_parity(run_cfg) -> None:
    '''Fail if a baseline subclass overrode a setting that must be shared.'''
    from baseline_spl.config import CommonConfig

    violations = [name for name in PARITY_CRITICAL
                  if not (name in OPTIONAL_WHEN_NONE and getattr(run_cfg, name, None) is None)
                  and getattr(run_cfg, name, None) != getattr(CommonConfig, name, None)]
    if violations:
        raise ValueError(
            f"{run_cfg.__name__} overrides parity-critical setting(s) {violations}. "
            f"These must match CommonConfig or baselines are not comparable — change "
            f"them on CommonConfig in baseline_spl/config.py instead."
        )


class BaselineConfig(SPLConfig):
    '''Instance attributes shadow SPLConfig's class attributes, so nothing here mutates
    the configuration SPL itself uses.'''

    # The concept-library knobs, with SPLConfig's names and meanings but this package's
    # values. Declared rather than inherited: SPLConfig's defaults point at SPL's own run
    # directory, so an inherited value would load SPL's concepts and save over its library.
    load_concept_checkpoint = None   # library to load; None = start from an empty one
    skip_loading_concepts = ()       # concepts to leave out of that library
    ignore_learnt_concepts = True    # skip learning concepts the loaded library already has

    def __init__(self, run_name: str, **overrides):
        self.run_name = run_name
        run_dir = RUNS_ROOT / run_name
        self.run_dir = str(run_dir)

        self.sketch_config = SketchConfig()
        self.sketch_config.cache_dir = str(run_dir / "sketch_cache")

        self.evaluation_config = copy.deepcopy(SPLConfig.evaluation_config)
        self.generalize_config = copy.deepcopy(SPLConfig.generalize_config)

        self.concept_save_path = str(run_dir / "concept_library.pt")
        self.inference_record_path = str(run_dir / "inference_records")
        self.llm_cache_dir = str(run_dir / "llm_cache")

        for key, value in overrides.items():
            setattr(self, key, value)

        # Checked after overrides, since the run config is what names a checkpoint. Both
        # directions: loading SPL's library leaks its concepts in, saving into it destroys them.
        if self.load_concept_checkpoint:
            assert_not_spl_library(self.load_concept_checkpoint)
            if not os.path.exists(self.load_concept_checkpoint):
                raise FileNotFoundError(
                    f"load_concept_checkpoint = {self.load_concept_checkpoint} does not exist. "
                    f"Set it to None to start from an empty library.")
            self.load_concept_checkpoint = str(self.load_concept_checkpoint)
        else:
            self.load_concept_checkpoint = None
        assert_not_spl_library(self.concept_save_path)

        os.makedirs(self.sketch_config.cache_dir, exist_ok=True)
        os.makedirs(self.llm_cache_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)

    @classmethod
    def from_run_config(cls, run_cfg, run_name: str) -> "BaselineConfig":
        '''Build from a run-config class in baseline_spl/config.py.

        Every public attribute is copied across, then the few that the SPL machinery
        expects under a different name are mapped onto it.
        '''
        assert_parity(run_cfg)
        # Baselines and SPL's Generalize stage must write code with the same model, or the
        # results compare models instead of methods. A mismatch can be deliberate, so ask.
        spl_model = GeneralizeConfig.llm_model
        # None means "calls no model" (B3-a enumerates; it never prompts an LLM), so it is not a
        # mismatch and must not raise the question. Without this the baseline that uses no model
        # at all was the one most reliably stopped by the model prompt.
        mismatched = {name: getattr(run_cfg, name) for name in ("codegen_model", "vlm_model")
                      if getattr(run_cfg, name) is not None
                      and getattr(run_cfg, name) != spl_model}
        if mismatched:
            warnings.warn(f"{run_cfg.__name__} uses {mismatched}, but SPL's GeneralizeConfig.llm_model "
                          f"is {spl_model!r}: the results would compare models, not methods.")
            try:
                answer = input("Type yes to continue with these models: ")
            except (EOFError, OSError):   # no terminal to answer from (background run, pytest)
                answer = ""
            if answer.strip().lower() != "yes":
                raise RuntimeError(f"Stopped: set codegen_model / vlm_model to {spl_model!r}, "
                                   f"or type yes to run with {mismatched}.")
        settings = _public_attrs(run_cfg)
        settings.pop("run_name", None)   # resolved by the caller and passed explicitly
        # The anonymisation ablation gets its own run directory. A natural and an anonymised
        # run must not share artifacts: SayCan's plan_library.json stores instructions verbatim
        # and replays them as worked examples, which would show one mode's wording to the other.
        # Read from SPLConfig, not from `settings`: the baselines' CommonConfig is standalone
        # (it does not subclass SPLConfig), so the knob reaches BaselineConfig by class
        # inheritance rather than through the overrides. SPLConfig is also the ONLY place to
        # set it -- the sketch agent scopes its cache from the same global, so a per-baseline
        # override would put the run in one mode's directory with the other mode's cache.
        naming = getattr(SPLConfig, "concept_name", "normal")
        if naming != "normal":
            run_name = f"{run_name}_anon{naming}"
        configs = cls(run_name, **settings)

        configs.concepts_to_learn = list(run_cfg.concepts)
        configs.concepts_to_infer = list(run_cfg.concepts)
        # Only when the baseline names one: assigning None here would blank a working model on
        # the deep-copied Generalize config, and the sketch agent would fail later, far from here.
        if run_cfg.codegen_model is not None:
            configs.generalize_config.llm_model = run_cfg.codegen_model
            # The sketch agent reads its provider and model from sketch_config, which would
            # otherwise be SPL's own. Set as a pair, never one alone: a baseline provider with
            # SPL's model name (or the reverse) is rejected by the server. Credentials and
            # base_url stay SPL's.
            configs.sketch_config.llm_provider = run_cfg.llm_provider
            configs.sketch_config.llm_model = run_cfg.codegen_model

        import inspect
        from SPL.utils.config_snapshot import save_config_snapshot
        try:
            save_config_snapshot(inspect.getfile(run_cfg), configs.run_dir,
                                "baseline_config_used.py")
        except OSError as exc:
            warnings.warn(f"Could not save a config snapshot for this run: {exc}")
        return configs

    def __repr__(self):
        return f"BaselineConfig(run_name={self.run_name!r}, run_dir={self.run_dir!r})"
