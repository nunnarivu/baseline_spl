'''
r2_lilo16.py

B3-b (LILO) over all 16 concepts, at a search budget matched to r2_full16.

    BASELINE_CONFIG=r2_lilo16 python -m baseline_spl.symbolic.lilo.run

This is the first setting in which LILO gets a fair test, and the reason is Finding K: LILO's
solver *cannot bootstrap*. Upstream raises "At least 2 tasks must have non-empty frontiers to
construct a prompt" (`gpt_solver.py:366`), so enumeration must solve tasks before the LLM has
anything to generalise from. With 4 concepts that left exactly two few-shot examples -- both
easy lines -- and the LLM was then asked to do only the hard half from them.

With 16 concepts enumeration should supply roughly ten solved tasks, so the prompt body is five
times richer. That is also the LILO-native fix for the residual failure seen at 4 concepts: a
structurally perfect staircase built along +y instead of -y, because the DSL description says
only `RIGHT :: tdir` and never which axis that is. More solved examples means more traces from
which to infer the mapping.

Expectations, set from measurement rather than hope: the LLM has solved **zero** concepts in
every run so far, and Findings J and N both showed that adding material to this prompt made
results worse rather than better. The row is needed for the table; it is not expected to be
the headline.

Search budget matches r2_full16 exactly (4 h x 3), because a head-to-head where B3-b searched
less than B3-a would confound the LLM's contribution with the search budget. LLM usage stays at
LILO's own level -- 4 queries per task, 4 samples per query, temperature 0.7 -- so the extra
compute costs no extra API budget.
'''

from __future__ import annotations

from baseline_spl.configs.r2_full16 import CapConfig, CommonConfig as _CommonConfig  # noqa: F401
from baseline_spl.configs.r2_full16 import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.r2_full16 import Demo2CodeConfig, SayCanConfig  # noqa: F401
from baseline_spl.configs.r2_full16 import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.r2_full16 import LiloConfig as _LiloConfig


class CommonConfig(_CommonConfig):
    # The shared 'flex' tier returns 429 flex_unavailable whenever OpenAI has no spare
    # capacity, which fails every proposal and silently reduces B3-b to B3-a. Not a
    # parity-critical setting -- it is cost and latency, not what the model sees.
    service_tier = "default"


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    pass


class LiloConfig(_LiloConfig, CommonConfig):
    # These MUST be restated, not inherited. `configs/default.py`'s LiloConfig sets
    # `enumeration_timeout = 600.0` explicitly, and it sits earlier in this class's MRO than
    # r2_full16's DreamCoderConfig, so inheriting silently gave B3-b a 600 s budget while
    # B3-a got 14,400 s -- a 24x mismatch in a comparison whose entire purpose is to be
    # matched. The first run of this config hit exactly that and had to be discarded.
    enumeration_timeout = 14400.0
    search_iterations = 3
    cpus = 16
