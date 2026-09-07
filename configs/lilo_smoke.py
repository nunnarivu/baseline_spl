'''
lilo_smoke.py

The first live B3-b run: four concepts chosen so the LLM's contribution is visible rather
than assumed.

  row, tower        MDL ~12 — enumeration solves these in seconds. They are the control:
                    if the LLM cannot get these right, the prompt is broken, not the method.
  staircase         MDL 35.5 inlined / 23.2 in library form. B3-a reached MDL 19.5 in
                    1,200 s and did not solve it.
  pyramid           MDL 50.2 — unreachable by enumeration at any budget this machine can
                    spend. Needs `length = height*2-1`, so it also exercises the arithmetic
                    primitives the tower DSL did not need.

The short enumeration budget is deliberate: at 60 s the enumerator gets the two lines and
nothing else, so anything deeper that lands is attributable to the proposer.

    BASELINE_CONFIG=lilo_smoke python -m baseline_spl.symbolic.lilo.run

Results land in runs/lilo_smoke_lilo_standard/, with the gate histogram and per-task token
spend in lilo_stats.json.
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
    concepts = ["row", "tower", "staircase", "pyramid"]
    inference = False          # learning is what this run measures

    # The default 'flex' tier returns 429 flex_unavailable when OpenAI has no spare
    # capacity, which fails every proposal and silently reduces B3-b to B3-a. Not a
    # parity-critical setting (it is cost and latency, not what the model sees), and
    # scoped to this smoke config rather than changed on the shared CommonConfig.
    service_tier = "default"


class CapConfig(_CapConfig, CommonConfig):
    pass


class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig):
    pass


class SayCanConfig(_SayCanConfig, CommonConfig):
    pass


# Search is free where API calls are not, so both baselines get a long enumeration budget and
# the SAME one -- 600 s, 10x what the earlier 60 s runs used. Matching them is what makes the
# head-to-head answer "does the LLM proposer reach what enumeration cannot" rather than
# "which one was given more compute".
ENUMERATION_TIMEOUT = 600.0
SEARCH_ITERATIONS = 2


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    enumeration_timeout = ENUMERATION_TIMEOUT
    search_iterations = SEARCH_ITERATIONS
    cpus = 4


class LiloConfig(_LiloConfig, CommonConfig):
    enumeration_timeout = ENUMERATION_TIMEOUT
    search_iterations = SEARCH_ITERATIONS
    cpus = 4
    # LILO's own sampling numbers, inherited from LiloConfig: 4 queries x 4 samples per task
    # per iteration, temperature 0.7.
