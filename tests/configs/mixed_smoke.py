'''
mixed_smoke.py

End-to-end smoke across every argument shape the 99-concept space contains, sized for a
~45-minute wall-clock check before a long run (concept-level; see mixed_smoke_demo.py for the
demo-level sibling).

    BASELINE_CONFIG=mixed_smoke python -m baseline_spl.symbolic.dreamcoder.run
    BASELINE_CONFIG=mixed_smoke python -m baseline_spl.symbolic.lilo.run

Five concepts, chosen to cover each arity exactly once:

    row        1 integer   the 69-concept common case
    rectangle  2 integers  12 concepts
    cuboid     3 integers  2 concepts (room, cuboid)
    podium     0 integers  15 fixed-size concepts -- a CLOSED program at concept level too
    wall       a concept   inexpressible; must be RECORDED, not crash the run

What this is for: the N-argument change touches eleven modules, and an int-or-tuple value
reaching int-only code fails only at runtime. Every arity passing through every module is the
gate. Run it at both granularities -- demo-level additionally exercises the request-type fix,
whose absence made closed-program MDL infinite.

This caught a real one, 09-16: the wake phase read every solved program back at ONE arity for
the whole run, so `rectangle`, `cuboid` and `podium` were found and then discarded as
untranslatable while only `row` survived. Concept-level alone was not enough to see it: at
demo level every task is closed, so the single arity was right by accident. Fixed in
search.py/driver.py/harness.py (each task now reads back at its OWN arity); this config is
what caught it, so it keeps running both directions.

`num_demos_per_concept = 3` deliberately. A 2-argument concept shown twice is
under-determined: a hole has two unknowns, so an affine function of one argument reproduces
another exactly (measured: `3*length - 7` equals `breadth` on [(3,2), (4,5)]). Recovery reports
`ambiguous` rather than guessing. `cuboid` has only 2 demos in DATA/new_data, so it is expected
to stay ambiguous -- that is the dataset's shape, not a defect.

`inference = True`: a smoke test should exercise the path a real run takes after learning, not
stop at search. For concept-level that is `infer_all` (score the registered concepts against
held-out demos); for demo-level, see mixed_smoke_demo.py.

LILO's LLM knobs are trimmed from the real experiment's (4 queries x 4 samples, `flex` tier) to
2x2 on `default` tier. That is a SMOKE-ONLY deviation, disclosed here rather than silently: the
point is bounding wall-clock time to verify the code path runs end to end, not measuring
proposer quality. Do not read numbers from this run as a LILO result.
'''

from __future__ import annotations

from baseline_spl.configs.default import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.default import CapConfig as _CapConfig
from baseline_spl.configs.default import CommonConfig as _CommonConfig
from baseline_spl.configs.default import Demo2CodeConfig as _Demo2CodeConfig
from baseline_spl.configs.default import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.default import LiloConfig as _LiloConfig
from baseline_spl.configs.default import SayCanConfig as _SayCanConfig

MIXED_ARITY_CONCEPTS = ["row", "rectangle", "cuboid", "podium", "wall"]


class CommonConfig(_CommonConfig):
    concepts = list(MIXED_ARITY_CONCEPTS)
    num_demos_per_concept = 3
    inference = True


class CapConfig(_CapConfig, CommonConfig): pass
class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig): pass
class SayCanConfig(_SayCanConfig, CommonConfig): pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    # 600s x 3 iterations = 30 min ceiling on search; the loop already stops early once a
    # wake produces nothing new (see search.py), so this is a ceiling, not a promise to spend
    # the whole budget. Plus demo load/sketch/score overhead, targets ~40-45 min total.
    enumeration_timeout = 600.0
    search_iterations = 3
    cpus = 16

    recognition_steps = 300
    recognition_timeout = 180.0

    # Both arms of the generalisation measurement, over the same held-out set.
    recover_concept_classes = True
    heldout_enumeration = True
    heldout_sizes = (7, 8)
    heldout_timeout = 300.0


class LiloConfig(_LiloConfig, DreamCoderConfig):
    # Restated: default.py's LiloConfig assigns enumeration_timeout=600 and sits earlier in
    # the MRO, which would silently unmatch the comparison.
    enumeration_timeout = 600.0
    search_iterations = 3
    cpus = 16

    # Smoke-only trim -- see module docstring.
    llm_queries_per_task = 2
    llm_samples_per_query = 2
    service_tier = "default"
