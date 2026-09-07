'''
r2_full16_rfix.py

`r2_full16` re-run with the recognition model actually working.

    BASELINE_CONFIG=r2_full16_rfix python -m baseline_spl.symbolic.dreamcoder.run

The original run is kept at `runs/r2_full16_dreamcoder_standard_maha2` and is NOT superseded
for its solved count -- enumeration finds the lines and diagonals whether or not Sleep-R
works. What it cannot support is any claim about the recognition model, because two bugs made
that model meaningless:

  * it trained for `epochs x len(frontiers)` = **45 gradient steps**, against the 10,000
    LILO's own config specifies. Upstream stops on a timeout with steps/epochs unbounded
    (`recognition.py:1269-1272`); passing `epochs=5, timeout=None` capped it instead.
  * the encoder named only the dominant axis of each step, so `diagonal_45`/`diagonal_135`
    and `diagonal_225`/`diagonal_315` were byte-identical inputs. Sixteen concepts presented
    as fourteen -- visible in upstream's own "14-way auxiliary classification loss".

Neither crashed. A 45-step network still emits a different grammar per task, because random
weights do that, so the wake phase looked conditioned throughout.

The open question this settles
------------------------------
The matched no-recognition run reached **MDL 22.5** while recognition-on reached **19.5** --
i.e. conditioning appeared to *cost* reach. That is what you would expect from noise: with one
shared grammar the wake phase runs a single enumeration scored against all 16 tasks, while
per-task grammars split the same budget 16 ways. Splitting only pays if the grammars carry
signal. Now they might.

Compare against:
  runs/r2_full16_dreamcoder_standard_maha2          same config, broken recognition
  runs/r2_full16_norecog_dreamcoder_standard_maha2  recognition off (its search is valid;
                                                    only its scoring was lost to an OOM)
'''

from __future__ import annotations

from baseline_spl.configs.r2_full16 import CapConfig, CommonConfig  # noqa: F401
from baseline_spl.configs.r2_full16 import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.r2_full16 import Demo2CodeConfig, SayCanConfig  # noqa: F401
from baseline_spl.configs.r2_full16 import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.r2_full16 import LiloConfig as _LiloConfig


class DreamCoderConfig(_DreamCoderConfig):
    # Everything is inherited; the fixes live in the code, not in config. Restated only so
    # this file records what the run is actually using.
    recognition_steps = 10000
    recognition_epochs = None
    recognition_timeout = 1800.0
    # Concept-level dreaming works (369 Helmholtz entries observed), unlike the closed
    # request, so the faithful ratio stands here.
    helmholtz_ratio = 0.5


class LiloConfig(_LiloConfig):
    recognition_steps = 10000
    recognition_epochs = None
    recognition_timeout = 1800.0
    helmholtz_ratio = 0.5
