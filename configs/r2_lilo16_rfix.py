'''
r2_lilo16_rfix.py

`r2_lilo16` (B3-b, LILO) re-run with the recognition model actually working.

    BASELINE_CONFIG=r2_lilo16_rfix python -m baseline_spl.symbolic.lilo.run

Same two recognition bugs as `r2_full16_rfix` describes -- 45 gradient steps instead of
10,000, and four concepts that were indistinguishable to the encoder. LILO's loop trains the
same Sleep-R model, so its earlier run carries the same defect.

Kept matched to `r2_full16_rfix` so the head-to-head still isolates LILO's contributions: same
concepts, same 4 h x 3 search budget, same acceptance rule, same recognition budget. LLM usage
stays at LILO's own level (4 queries per task, 4 samples per query, temperature 0.7).

Expectation, from measurement rather than hope: the LLM has solved **zero** concepts in every
run so far, and the reason is structural (Finding K -- LILO cannot bootstrap, so enumeration
must hand it solved tasks first, and here enumeration solves only the easy band). A better
recognizer changes what *enumeration* reaches, which changes what the LLM is shown. That is
the only route by which this run could differ, and it is worth measuring precisely because it
is indirect.

The API cost is unchanged from the earlier run and mostly cached.
'''

from __future__ import annotations

from baseline_spl.configs.r2_lilo16 import CapConfig, CommonConfig  # noqa: F401
from baseline_spl.configs.r2_lilo16 import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.r2_lilo16 import Demo2CodeConfig, SayCanConfig  # noqa: F401
from baseline_spl.configs.r2_lilo16 import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.r2_lilo16 import LiloConfig as _LiloConfig


class DreamCoderConfig(_DreamCoderConfig):
    recognition_steps = 10000
    recognition_epochs = None
    recognition_timeout = 1800.0
    helmholtz_ratio = 0.5


class LiloConfig(_LiloConfig):
    recognition_steps = 10000
    recognition_epochs = None
    recognition_timeout = 1800.0
    helmholtz_ratio = 0.5
