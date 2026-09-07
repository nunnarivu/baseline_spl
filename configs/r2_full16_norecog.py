'''
r2_full16_norecog.py

The recognition ablation: r2_full16 with Sleep-R switched off, everything else identical.

    BASELINE_CONFIG=r2_full16_norecog python -m baseline_spl.symbolic.dreamcoder.run

The question deferred since Stage 2: does DreamCoder's recognition model actually help in this
domain? It has been built and training since then, but never measured against its own absence
at a matched budget.

**A null result is a result. Do not tune until it looks good.** With 16 concepts the recognizer
trains almost entirely on Helmholtz dreams, which is normal for DreamCoder -- dreams dominate
its training signal in every published domain -- but it does mean there is little real signal
to learn from, and "no measurable difference" is a perfectly publishable outcome.

One structural effect to expect and to report separately from any capability difference: with
no recognition model every task shares one grammar, so the wake phase runs a single enumeration
scored against all 16 tasks instead of one enumeration per task. That is strictly less work,
so this run should be *faster* per iteration at equal depth. Compare `max_mdl_reached` and the
solved set, not wall clock.

Matched to r2_full16 in every other respect: same concepts, same 4 h x 3 budget, same seed,
same demos, same acceptance rule.
'''

from __future__ import annotations

from baseline_spl.configs.r2_full16 import CapConfig, CommonConfig  # noqa: F401
from baseline_spl.configs.r2_full16 import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.r2_full16 import Demo2CodeConfig, SayCanConfig  # noqa: F401
from baseline_spl.configs.r2_full16 import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.r2_full16 import LiloConfig as _LiloConfig


class DreamCoderConfig(_DreamCoderConfig):
    use_recognition = False
    # One shared grammar means one enumeration scored against every task, so extra workers
    # would only re-enumerate the same programs. Task parallelism pays off only alongside
    # per-task grammars (Finding F).
    cpus = 1


class LiloConfig(_LiloConfig):
    use_recognition = False
    cpus = 1
