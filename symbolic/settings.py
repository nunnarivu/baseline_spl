'''
settings.py

Every knob the wake/sleep loop takes, in one place.

`driver.run` used to accept 25 keyword arguments, and the harness restated all 25 defaults as
`getattr(cfg, "name", default)` at the call site -- so each default existed twice, in two files,
with nothing keeping the copies in step. Four of them had already drifted apart. Reading the
loop meant scrolling past a 26-line call; changing a default meant remembering to change it
twice.

Grouping them makes the call site legible and gives each default exactly one home. The
dataclasses also document what the loop actually takes, which a 25-parameter signature does not.

`configs/default.py` remains where a *user* sets these. `from_config` is the single place that
reads a config object, so a missing knob falls back here and nowhere else.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecognitionSettings:
    '''Sleep-R: the neural model that conditions the grammar on the task.

    Defaults follow LILO's own recognition loader
    (`src/models/laps_dreamcoder_recognition.py`), not upstream's library defaults, which are
    weaker: `contextual=False`, `biasOptimal=None`, `auxLoss=False`.
    '''

    enabled: bool = True

    #: Gradient steps. LILO's config asks for 10,000 (`template_lilo.json`). Upstream defaults
    #: `steps` and `epochs` to 9,999,999 and stops on `timeout`, so passing a small `epochs`
    #: silently caps training -- and an undertrained network still emits a different grammar
    #: per task, so nothing looks wrong. `steps` is the budget; leave `epochs` None.
    steps: int = 10000
    epochs: Optional[int] = None
    timeout: Optional[float] = 1800.0

    #: Must equal the feature extractor's output width: `frontierKL` computes
    #: `self._MLP(features).expand(1, features.size(-1))`, which assumes the MLP preserves it.
    hidden: int = 64

    #: True predicts a bigram transition matrix over productions (conditioned on the parent
    #: production and argument index) rather than marginal unigram weights, so `grammarOfTask`
    #: can express "inside a loop body, prefer place".
    contextual: bool = True
    bias_optimal: bool = True
    auxiliary_loss: bool = True

    #: Fraction of training data drawn from fantasies rather than solved tasks.
    helmholtz_ratio: float = 0.5


@dataclass
class SearchSettings:
    '''Everything `driver.run` needs beyond the tasks themselves.'''

    # --- the grammar being searched ----------------------------------------------------- #
    level: str = "standard"
    #: 0 leaves the grammar's own literals ({1, 2}) alone; the harness derives a ceiling from
    #: the task parameters so a larger dataset widens it automatically.
    int_literals_upto: int = 0
    #: continuationType, as tower, LOGO, regex and LILO's `structures` domain all set it.
    #: Makes state threading implicit, so programs are shorter at equal semantics.
    continuation: bool = True

    # --- the wake phase ------------------------------------------------------------------ #
    iterations: int = 3
    timeout: float = 60.0
    max_mdl: float = 100.0
    maximum_frontier: int = 5
    cpus: int = 1

    # --- Sleep-G: compression and re-weighting ------------------------------------------- #
    use_library: bool = True
    #: 30, matching LILO's `insideOutside(frontiers_rewritten, pseudoCounts=30, iterations=1)`.
    pseudo_counts: float = 30.0
    max_arity: int = 3

    # --- Sleep-R -------------------------------------------------------------------------- #
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)

    @classmethod
    def from_config(cls, cfg) -> "SearchSettings":
        '''Read a run config. The only place a config object is inspected.'''
        get = lambda name, default: getattr(cfg, name, default)  # noqa: E731
        return cls(
            level=get("grammar_level", "standard"),
            continuation=get("use_continuation_type", True),
            iterations=get("search_iterations", 3),
            timeout=get("enumeration_timeout", 60.0),
            max_mdl=get("max_mdl", 100.0),
            maximum_frontier=get("maximum_frontier", 5),
            cpus=get("cpus", 1),
            use_library=get("use_library", True),
            pseudo_counts=get("pseudo_counts", 30.0),
            max_arity=get("stitch_max_arity", 3),
            recognition=RecognitionSettings(
                enabled=get("use_recognition", True),
                steps=get("recognition_steps", 10000),
                epochs=get("recognition_epochs", None),
                timeout=get("recognition_timeout", 1800.0),
                hidden=get("recognition_hidden", 64),
                contextual=get("recognition_contextual", True),
                bias_optimal=get("recognition_bias_optimal", True),
                auxiliary_loss=get("recognition_auxiliary_loss", True),
                helmholtz_ratio=get("helmholtz_ratio", 0.5),
            ),
        )
