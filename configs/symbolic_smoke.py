'''
symbolic_smoke.py

A fast end-to-end check of the symbolic (search) baseline:

    BASELINE_CONFIG=symbolic_smoke python -m baseline_spl.symbolic.dreamcoder.run

Three concepts and a short enumeration budget, chosen to exercise both outcomes: `row`,
`tower` and `column` are within reach of enumeration, while `staircase` is not and should be
reported unsolved or approximate rather than silently skipped. Results land in
runs/symbolic_smoke_dreamcoder_standard/ and are not comparable to a full sweep.

Everything except the DreamCoder settings is inherited from configs/default.py, so this file
stays a diff rather than a copy.
'''

from __future__ import annotations

from baseline_spl.configs.default import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.default import CapConfig as _CapConfig
from baseline_spl.configs.default import CommonConfig as _CommonConfig
from baseline_spl.configs.default import Demo2CodeConfig as _Demo2CodeConfig
from baseline_spl.configs.default import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.default import SayCanConfig as _SayCanConfig


class CommonConfig(_CommonConfig):
    concepts = ["row", "tower", "column", "staircase"]
    inference = False          # learning is what this smoke test exercises


class CapConfig(_CapConfig, CommonConfig):
    pass


class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig):
    pass


class SayCanConfig(_SayCanConfig, CommonConfig):
    pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    # Short enough to finish in a couple of minutes. Cost grows as ~e^(0.79 x MDL), so this
    # reaches the shallow band and nothing deeper -- which is the point: `staircase` should
    # come back unsolved, and that must be recorded rather than hidden.
    enumeration_timeout = 45.0
    search_iterations = 2
