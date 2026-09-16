'''
mixed_smoke_demo.py

The demo-level sibling of mixed_smoke.py -- same five concepts, same ~45-minute budget, but
`task_granularity = "demo"`: every program is CLOSED (the size is a literal, not an argument),
which is what exercises the request-type fix (stitch_bridge.py / driver.py / lilo/proposer.py
scoring each task at ITS OWN request type instead of a hardcoded CONCEPT_REQUEST). Without that
fix every demo-level run logged `mean solved MDL now inf` and no abstraction was ever adopted.

    BASELINE_CONFIG=mixed_smoke_demo python -m baseline_spl.symbolic.dreamcoder.run
    BASELINE_CONFIG=mixed_smoke_demo python -m baseline_spl.symbolic.lilo.run

With `learn = True, inference = True`, one invocation both learns and, per
`SearchHarness.learn_all` (harness.py:555-595), recovers a parameterised class from each
concept's solved demos inline, registers it, and scores it against the real demos -- the same
`training_metrics.json` / `learning_times.json` shape as every other baseline. If recovery
finds nothing to register, `dreamcoder/run.py:_is_finished_demo_run` routes to
`generalise_all()` instead, which writes `heldout_metrics.json` (arms A/B at synthetic sizes
7/8) rather than leaving the run silently short of output either way.

See mixed_smoke.py for the concepts, the arity-readback bug this pair of configs caught, and
the disclosed LILO smoke-only trim (2x2 queries/samples, `default` tier).
'''

from __future__ import annotations

from baseline_spl.tests.configs.mixed_smoke import CapConfig  # noqa: F401
from baseline_spl.tests.configs.mixed_smoke import CODEGEN_MODEL  # noqa: F401
from baseline_spl.tests.configs.mixed_smoke import CommonConfig as _CommonConfig
from baseline_spl.tests.configs.mixed_smoke import Demo2CodeConfig  # noqa: F401
from baseline_spl.tests.configs.mixed_smoke import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.tests.configs.mixed_smoke import LiloConfig as _LiloConfig
from baseline_spl.tests.configs.mixed_smoke import SayCanConfig  # noqa: F401
from baseline_spl.tests.configs.mixed_smoke import VLM_MODEL  # noqa: F401


class CommonConfig(_CommonConfig):
    pass


class DreamCoderConfig(_DreamCoderConfig):
    task_granularity = "demo"


class LiloConfig(_LiloConfig):
    task_granularity = "demo"
