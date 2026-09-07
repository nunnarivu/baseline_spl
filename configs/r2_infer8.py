'''
r2_infer8.py

The inference pass, run against the library the 8-concept continuous B3-a sweep already left
behind.

    BASELINE_CONFIG=r2_infer8 python -m baseline_spl.symbolic.dreamcoder.run

Why this exists as its own config: `infer_all` has **never executed** for either symbolic
baseline. Every config so far sets `inference = False`, so the inference column of the results
table is blank and the code path is untested. That makes this both a missing result and the
cheapest place a latent bug could still be hiding -- worth clearing before ~12-hour runs are
scored through the same path.

`run_name` is pinned rather than derived, so this resumes the existing run directory instead of
starting an empty one. `learn = False` means nothing is re-searched; the concepts are loaded
from that run's concept_library.pt and only evaluated.
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
    concepts = ["row", "column", "tower", "inverted_row",
                "diagonal_45", "diagonal_135", "staircase", "inverted_staircase"]
    learn = False
    inference = True
    # The directory the continuous sweep wrote. Pinned so this scores that library rather
    # than creating a new empty run.
    run_name = "symbolic_sweep8_dreamcoder_standard_maha3"


class CapConfig(_CapConfig, CommonConfig):
    pass


class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig):
    pass


class SayCanConfig(_SayCanConfig, CommonConfig):
    pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    pass


class LiloConfig(_LiloConfig, CommonConfig):
    pass
