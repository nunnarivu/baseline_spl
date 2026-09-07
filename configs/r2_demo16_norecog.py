'''
r2_demo16_norecog.py

The recognition ablation, demo-level: `r2_demo16` with Sleep-R switched off and everything
else identical.

    BASELINE_CONFIG=r2_demo16_norecog python -m baseline_spl.symbolic.dreamcoder.run

This pair is the first honest test of whether DreamCoder's recognition model helps here,
because the two earlier attempts were both invalid:

  1. **The model was barely trained.** Upstream stops training on a *timeout*, with `steps`
     and `epochs` defaulting to 9,999,999 (`recognition.py:1269-1272`). We passed
     `epochs=5, timeout=None`, which caps the loop at `epochs x len(frontiers)` gradient
     steps -- 45 with 9 solved tasks, against the 10,000 LILO's own config specifies. A
     network trained for 45 steps is its random initialisation with a nudge.
  2. **Four tasks were invisible to it.** The tokenizer named only the dominant axis of a
     step, so `diagonal_45`/`diagonal_135` and `diagonal_225`/`diagonal_315` encoded
     identically. Sixteen concepts presented as fourteen.

Both are fixed. Note what the first bug did to the *appearance* of the earlier runs: random
per-task grammars still differ from each other, so the wake phase looked conditioned and
throughput moved (309/s with recognition vs 200/s without) while capability did not. Any
result that reads "recognition helps speed but not reach" has to be re-established now, not
carried over.

Matched to r2_demo16 in every other respect. A null result is still a result -- do not tune
until it looks good.
'''

from __future__ import annotations

from baseline_spl.configs.r2_demo16 import CapConfig, CommonConfig  # noqa: F401
from baseline_spl.configs.r2_demo16 import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.r2_demo16 import Demo2CodeConfig, SayCanConfig  # noqa: F401
from baseline_spl.configs.r2_demo16 import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.r2_demo16 import LiloConfig as _LiloConfig


class DreamCoderConfig(_DreamCoderConfig):
    use_recognition = False
    # With no recognizer every task shares one grammar, so the wake phase runs a single
    # enumeration scored against all tasks -- one work unit. Extra workers would only
    # re-enumerate the same programs (Finding F), and `_enumerate_unit` shares an absolute
    # deadline, so splitting would starve nothing but also gain nothing.
    cpus = 1


class LiloConfig(_LiloConfig):
    use_recognition = False
    cpus = 1
