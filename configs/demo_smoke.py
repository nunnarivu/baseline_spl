'''
demo_smoke.py

Fast end-to-end check of `task_granularity="demo"`. This path had been unit-tested but never
run through the harness, and the first real attempt died at a log line that assumed a task is
named after its concept (demo-level names it "<concept>_<demo_id>"). Cheap insurance before
committing 12 hours to it.
'''
from __future__ import annotations

from baseline_spl.configs.default import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.default import CapConfig as _CapConfig
from baseline_spl.configs.default import CommonConfig as _CommonConfig
from baseline_spl.configs.default import Demo2CodeConfig as _Demo2CodeConfig
from baseline_spl.configs.default import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.default import LiloConfig as _LiloConfig
from baseline_spl.configs.default import SayCanConfig as _SayCanConfig


class CommonConfig(_CommonConfig):
    concepts = ["row", "tower", "column", "staircase"]
    inference = False


class CapConfig(_CapConfig, CommonConfig): pass
class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig): pass
class SayCanConfig(_SayCanConfig, CommonConfig): pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    task_granularity = "demo"
    enumeration_timeout = 45.0
    search_iterations = 2
    cpus = 8
    recognition_steps = 200          # keep the smoke test quick
    recognition_timeout = 120.0
    helmholtz_ratio = 0.0


class LiloConfig(_LiloConfig, DreamCoderConfig):
    enumeration_timeout = 45.0
