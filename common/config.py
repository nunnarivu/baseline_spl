'''
config.py

``BaselineConfig`` — the SPL-compatible config object every baseline runs on.

This file holds no settings of its own; all knobs live in baseline_spl/config.py and are
copied in by ``from_run_config``. What it does own are the fairness invariants, which are
correctness guarantees rather than knobs:

  * ``load_concept_checkpoint`` may only point inside this run's own directory, so a
    baseline never inherits the concepts SPL already learned. Pointing it at the
    baseline's own library is what lets a run resume.
  * The shared SketchAgent gets a per-run cache directory, so no baseline reads a
    signature that was computed with SPL's concept library in context.
  * SPLConfig shares one instance of each sub-config across all config objects, so they
    are deep-copied here; otherwise a baseline setting would change SPL's own behaviour.
'''

from __future__ import annotations

import copy
import os
from pathlib import Path

from SPL.config.spl_config import SPLConfig, SketchConfig

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
    baseline the concepts SPL itself learned. Any path outside SPL is allowed.'''
    import SPL

    spl_root = Path(SPL.__file__).resolve().parent
    resolved = Path(path).resolve()
    if spl_root == resolved or spl_root in resolved.parents:
        raise ValueError(
            f"resume_from points inside SPL's own tree ({path}). A baseline that loads "
            f"SPL's concept library is not a baseline. Point it at a baseline run under "
            f"{RUNS_ROOT}, or set resume=False."
        )


# Settings that must be identical across baselines for the comparison to mean anything.
# Kept here rather than in the config files: *which* fields must match is machinery, and
# duplicating the list per config would let it drift between experiments.
PARITY_CRITICAL = ("concepts", "num_demos_per_concept", "codegen_model",
                   "vlm_model", "max_code_retries")


def assert_parity(run_cfg) -> None:
    '''Fail if a baseline subclass overrode a setting that must be shared.'''
    from baseline_spl.config import CommonConfig

    violations = [name for name in PARITY_CRITICAL
                  if getattr(run_cfg, name, None) != getattr(CommonConfig, name, None)]
    if violations:
        raise ValueError(
            f"{run_cfg.__name__} overrides parity-critical setting(s) {violations}. "
            f"These must match CommonConfig or baselines are not comparable — change "
            f"them on CommonConfig in baseline_spl/config.py instead."
        )


class BaselineConfig(SPLConfig):
    '''Instance attributes shadow SPLConfig's class attributes, so nothing here mutates
    the configuration SPL itself uses.'''

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

        self.resume = True
        self.resume_from = None

        for key, value in overrides.items():
            setattr(self, key, value)

        # Resolved after overrides, since it depends on resume / resume_from.
        self.load_concept_checkpoint = self._resolve_checkpoint()

        os.makedirs(self.sketch_config.cache_dir, exist_ok=True)
        os.makedirs(self.llm_cache_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)

    def _resolve_checkpoint(self):
        '''Which concept library to load, or None to start empty.'''
        if not self.resume:
            return None
        path = self.resume_from or self.concept_save_path
        if not os.path.exists(path):
            return None
        assert_not_spl_library(path)
        return str(path)

    @classmethod
    def from_run_config(cls, run_cfg, run_name: str) -> "BaselineConfig":
        '''Build from a run-config class in baseline_spl/config.py.

        Every public attribute is copied across, then the few that the SPL machinery
        expects under a different name are mapped onto it.
        '''
        assert_parity(run_cfg)
        settings = _public_attrs(run_cfg)
        settings.pop("run_name", None)   # resolved by the caller and passed explicitly
        configs = cls(run_name, **settings)

        configs.concepts_to_learn = list(run_cfg.concepts)
        configs.concepts_to_infer = list(run_cfg.concepts)
        configs.generalize_config.llm_model = run_cfg.codegen_model
        return configs

    def __repr__(self):
        return f"BaselineConfig(run_name={self.run_name!r}, run_dir={self.run_dir!r})"
