'''
r2_demo16.py

DreamCoder as published: one task per demonstration, with the size baked in as a literal.

    BASELINE_CONFIG=r2_demo16 python -m baseline_spl.symbolic.dreamcoder.run

Why this run exists
-------------------
Our concept-level framing asks for a *recipe*: one task per concept, whose program must work
for every size. DreamCoder's own benchmarks never do this. Its tower domain is 107 tasks whose
request type is `ttower -> ttower` -- no argument at all -- and the sizes appear as separate
tasks: "arch leg 1", "arch leg 2", ... "arch leg 8" are eight distinct problems, each a closed
program. LILO's re2 domain is the same shape at larger scale: ~500 tasks that are combinatorial
variants of about four templates.

So we have been running a harder problem than the one these methods were built for, and
reporting the result as though it were theirs. `task_granularity="demo"` gives them their own
framing: one task per demonstration, the parameter baked in, no induction required. Every
concept contributes `num_demos_per_concept` tasks instead of one.

What this can and cannot show
-----------------------------
It removes one of the three causes of our 9/16 result -- the parameterised request. It does NOT
create the other thing their domains have: many tasks per shape. Two demos per concept gives 32
tasks over 16 shapes, against tower's 107 tasks over ~17 shapes. So a partial recovery is the
honest expectation, not parity.

Read the result against `r2_full16` (same concepts, same budget, concept-level). The difference
between the two is the price of asking for a recipe instead of an instance.
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
    inference = False


class CapConfig(_CapConfig, CommonConfig):
    pass


class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig):
    pass


class SayCanConfig(_SayCanConfig, CommonConfig):
    pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    # The point of this config.
    task_granularity = "demo"

    # Matched to r2_full16 so the only difference is the task framing.
    enumeration_timeout = 14400.0
    search_iterations = 3

    # 32, not 16, and this is a correctness requirement rather than a speed choice.
    # Demo-level gives 16 concepts x num_demos_per_concept = 32 tasks, and with a recognizer
    # each task is its own work unit. `_enumerate_unit` shares one ABSOLUTE deadline, so if
    # units outnumber workers the first batch consumes the whole budget and every later unit
    # breaks immediately with nothing searched. At cpus=16 that would silently starve half
    # the tasks -- they would report unsolved having never been looked at.
    cpus = 32

    # Sleep-R at the budget upstream actually uses. Restated here rather than inherited so a
    # reader of this file can see what the recognition ablation is comparing against.
    recognition_steps = 10000
    recognition_epochs = None
    recognition_timeout = 1800.0

    # DEVIATION, and it must be disclosed. Helmholtz dreaming produces nothing for the closed
    # request: the run logs "Got 0/500 valid samples ... 0 Helmholtz entries" and upstream
    # then retries forever rather than giving up, so the run hangs instead of failing. The
    # same sampler works for the concept-level request (369 entries observed), so this is
    # specific to `tstate -> tstate`.
    #
    # Training on the real frontiers alone sidesteps it deterministically. That is a real
    # departure from DreamCoder, where fantasies dominate the training signal, and it makes
    # this a weaker test of Sleep-R than the concept-level one -- 32 real frontiers is not
    # much to learn from. Report it as such rather than as "recognition does not help".
    helmholtz_ratio = 0.0


class LiloConfig(_LiloConfig, DreamCoderConfig):
    # Restated: default.py's LiloConfig assigns enumeration_timeout=600 and sits earlier in
    # the MRO, which would silently unmatch the comparison.
    enumeration_timeout = 14400.0
    search_iterations = 3
    cpus = 16
