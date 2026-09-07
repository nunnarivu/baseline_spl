'''
r2_full16.py

The B3-a control table: DreamCoder-style search over all 16 concepts, under Round 2's fair
setting (continuous observations, accepted by SPL's own probabilistic focus).

    BASELINE_CONFIG=r2_full16 python -m baseline_spl.symbolic.dreamcoder.run

Budget. Earlier sweeps ran 1,200 s per iteration and `max_mdl_reached` plateaued at 19.5 no
matter how much the grammar was re-weighted (Finding I: concentrating the grammar buys density,
not reach). The only lever left is wall clock, and it is a brutal one -- cost grows as
`~e^(0.79 x MDL)`, so each extra nat costs 2.2x more time.

At 4 hours per iteration, i.e. 12x the previous budget:

    ln(12) / 0.79 = 3.1 extra nats

which takes the reachable depth from ~19.5 to ~22.6. `staircase` in library form sits at 23.2.
So this run lands just short of it on paper -- close enough that whether it actually solves is
a real question rather than a foregone conclusion, which is the point of spending the time.

The concepts at MDL 50+ (pyramid, x, arch_bridge, isosceles_right_triangle) are included and
expected to stay unsolved. That is the reportable control result, not a gap to apologise for:
`baseline.md` predicts exactly this, and reporting them as unsolved is what makes the
prediction a measurement.

Early stopping matters here. The driver halts when neither the solutions nor the library
changed, because each iteration restarts the MDL band loop from zero -- with an unchanged
grammar the next iteration re-enumerates precisely the same programs. So the wall clock is
3 x 4 h = 12 h at most, and 8 h if it converges after two iterations.
'''

from __future__ import annotations

from SPL.config.spl_config import ALL_CONCEPTS

from baseline_spl.configs.default import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.default import CapConfig as _CapConfig
from baseline_spl.configs.default import CommonConfig as _CommonConfig
from baseline_spl.configs.default import Demo2CodeConfig as _Demo2CodeConfig
from baseline_spl.configs.default import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.default import LiloConfig as _LiloConfig
from baseline_spl.configs.default import SayCanConfig as _SayCanConfig


class CommonConfig(_CommonConfig):
    concepts = list(ALL_CONCEPTS)
    # Learning is what this run measures; inference is a separate pass over the library it
    # leaves behind (configs/r2_infer.py).
    inference = False


class CapConfig(_CapConfig, CommonConfig):
    pass


class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig):
    pass


class SayCanConfig(_SayCanConfig, CommonConfig):
    pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    # 4 hours per iteration; see the module docstring for why this specific number.
    enumeration_timeout = 14400.0
    search_iterations = 3
    # Parallelism is across tasks, so it only pays once the recognition model gives each task
    # its own grammar -- which it does here. 16 of 112 cores, leaving room for the sibling
    # runs launched alongside this one.
    cpus = 16


class LiloConfig(_LiloConfig, DreamCoderConfig):
    # Restated, not inherited. `configs/default.py`'s LiloConfig assigns
    # `enumeration_timeout = 600.0` -- LILO's own setting, where the LLM is the primary
    # solver -- and that assignment sits earlier in this MRO than DreamCoderConfig above, so
    # inheriting would silently hand B3-b a 24x smaller budget than B3-a in a comparison
    # whose whole point is to be matched. Pinned by
    # tests/test_lilo.py::test_matched_comparison_configs_really_are_matched.
    enumeration_timeout = 14400.0
    search_iterations = 3
    cpus = 16
