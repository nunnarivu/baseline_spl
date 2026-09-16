'''
evaluate.py

How the search decides a candidate program is correct.

Round 1 had one answer: the program's integer placement cells equal the demonstration's,
exactly. That is DreamCoder's own all-or-nothing acceptance (`task.py:126`) and it makes
acceptance a proof. It also only exists because the demonstration was discretized first --
a step SPL never takes, and therefore an advantage B3 was being handed rather than earning.

This module provides three evaluators behind one interface, so the choice is a config knob
and a reportable comparison rather than a decision baked into the search:

    exact        integer cells must match            (observation_mode='lattice')
    distance     every placement within eps metres   (observation_mode='continuous')
    mahalanobis  every placement within tau sigma    (observation_mode='continuous')

The interface
-------------
`score(program, task)` returns a single non-negative *badness*, lower being better, and
`accepts(value)` tests it against the evaluator's threshold. One number serves both ranking
and acceptance, so `Solution.add_approximate` keeps working unchanged and an unsolved concept
still reports its best attempt.

Wrong-length programs
---------------------
Every evaluator rejects a program that places the wrong number of blocks. Under `exact` the
edit distance did this implicitly; the continuous evaluators must do it explicitly, or a
program placing three blocks would score well against a five-block demonstration by matching
a prefix. They return `miss_value` -- a finite sentinel far above any sane threshold, so such
a program can never be accepted but still orders sensibly against other wrong-length programs
when nothing better was found.

Acceptance is 0/1, and never SPL's reward
-----------------------------------------
DreamCoder and LILO accept a program when it reproduces the task, full stop. Borrowing SPL's
graded reward would hand the baseline part of what SPL is being credited for, so both
continuous evaluators are threshold tests on a single badness number.

`distance` uses the SAME bar as the LLM baselines: `common/evaluator.py` accepts a class when
every block is within `sketch_val_state_error_threshold`, so this compares the worst placement
against that same value. A mean or RMSE would be looser -- one bad block averages away -- and
would put an asterisk on a results table the baselines share.

A third criterion, `penalised_loglik`, was removed: it mirrored SPL's `_penalised_score` at
`focus_score_penalty="trace"`, a formula SPL has since retired in favour of `"reward"`.
'''

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from baseline_spl.symbolic import gaussian
from baseline_spl.symbolic.lattice import EMPTY, LatticeBudgetExceeded

# Exceptions an enumerated program can raise that mean "this program does not solve the
# task", never "the run is broken". These guards fire millions of times a second inside the
# wake phase and are deliberately silent: a failed candidate is the normal case, not a
# diagnostic. Handlers that hide a real problem -- a lost solution, a rejected library form --
# do log, in search.py and driver.py.
_RUN_ERRORS = (LatticeBudgetExceeded, RecursionError, IndexError, ValueError, TypeError,
               ZeroDivisionError, KeyError, OverflowError, AttributeError)

VALID_COMBINATIONS = {
    ("lattice", "exact"),
    ("continuous", "distance"),
    ("continuous", "mahalanobis"),
}


def validate(observation_mode: str, evaluator: str) -> None:
    '''Reject a meaningless (observation, evaluator) pair at startup.

    Integers admit no tolerance and floats are never exactly equal, so three of the six
    pairs are nonsense. They must fail loudly: the recurring failure in this codebase is a
    correct-looking configuration that silently produces nothing (the plan's Traps 2 and 3),
    and "every frontier was empty" is exactly what a bad pair would look like.
    '''
    pair = (observation_mode, evaluator)
    if pair not in VALID_COMBINATIONS:
        valid = ", ".join(f"{o}+{e}" for o, e in sorted(VALID_COMBINATIONS))
        raise ValueError(
            f"observation_mode={observation_mode!r} cannot be combined with "
            f"evaluator={evaluator!r}. Valid combinations: {valid}.")


# --------------------------------------------------------------------------------------- #
# Ranking helper, shared by the lattice evaluator and by reporting.
# --------------------------------------------------------------------------------------- #

def trace_distance(predicted: Sequence, target: Sequence) -> float:
    '''Normalised edit distance between two placement sequences, in [0, 1].

    Levenshtein over cells, so an inserted or dropped placement costs one rather than
    shifting everything after it.
    '''
    if not predicted and not target:
        return 0.0
    previous = list(range(len(target) + 1))
    for i, p in enumerate(predicted, start=1):
        current = [i]
        for j, t in enumerate(target, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (p != t)))
        previous = current
    return previous[-1] / max(len(predicted), len(target))


# --------------------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------------------- #

class Evaluator:
    '''Base class. Subclasses implement `_example_statistic`; aggregation is shared.'''

    name = "base"
    threshold = 0.0
    #: Returned when a program places the wrong number of blocks. Finite so wrong-length
    #: programs still order against each other, far above any threshold so none is accepted.
    miss_value = 1e3

    def score(self, program, task) -> Optional[float]:
        '''Mean statistic across the task's examples, or None if the program cannot run.'''
        try:
            fn = program.evaluate([])
        except Exception:  # noqa: BLE001
            return None

        total = 0.0
        for index, (param, target) in enumerate(task.examples):
            try:
                value = self._example_statistic(fn, param, target, task, index)
            except _RUN_ERRORS:
                return None
            except Exception:  # noqa: BLE001 - an enumerated program may fail in any way
                return None
            if value is None:
                return None
            total += value
        return total / len(task.examples)

    def accepts(self, value: Optional[float]) -> bool:
        return value is not None and value < self.threshold

    def describe(self) -> dict:
        return {"evaluator": self.name, "accept_threshold": self.threshold}

    def _example_statistic(self, fn, param, target, task, index) -> Optional[float]:
        raise NotImplementedError


class ExactEvaluator(Evaluator):
    '''Round 1: integer placement cells must match exactly.

    Kept as the ablation that quantifies what the lattice representation was worth. Also the
    strictest available check, so it stays useful as a rigour cross-reference even when a
    continuous evaluator is the headline.
    '''

    name = "exact"
    threshold = 0.0
    miss_value = 1.0        # trace_distance is already normalised to [0, 1]

    def accepts(self, value: Optional[float]) -> bool:
        return value is not None and value <= 0.0

    def _example_statistic(self, fn, param, target, task, index) -> Optional[float]:
        # A closed program IS the state transformer; a parameterised one must be applied to
        # the size first. Getting this wrong does not crash -- it raises inside the scorer's
        # own guard and silently reports "this program does not run".
        from baseline_spl.symbolic.search import apply_args

        produced = (fn(EMPTY) if task.closed else apply_args(fn, param)(EMPTY)).positions
        return trace_distance(produced, target)


def default_distance_threshold() -> float:
    '''SPL's `sketch_val_state_error_threshold`, the tolerance the LLM baselines already use.'''
    from SPL.config.spl_config import SPLConfig

    return float(getattr(SPLConfig, "sketch_val_state_error_threshold", 0.06))


class DistanceEvaluator(Evaluator):
    '''Continuous observations: every placement must land within a fixed radius, in metres.

    **The same bar the LLM baselines apply.** `common/evaluator.py` accepts a generated class
    when EVERY block is within `sketch_val_state_error_threshold` of the demo's last keyframe
    (`far = [... if d > self.tolerance]`), so this uses the worst per-placement distance and the
    same threshold. An RMSE or mean would be a looser bar -- one badly-placed block averages
    away -- and would judge the symbolic baselines more leniently than CaP/Demo2Code while their
    numbers sit in one table.

    Uses the SRN means and ignores its variances, so the executor is the Gaussian one with the
    uncertainty channel unused. It charges a placement reached after ten compounding shifts
    exactly as strictly as the first; whether that matters is what the mahalanobis knob answers.
    '''

    name = "distance"

    def __init__(self, epsilon: Optional[float] = None, resync: bool = True,
                 per_step_variance: bool = False):
        self.threshold = default_distance_threshold() if epsilon is None else float(epsilon)
        self.resync = resync
        self.per_step_variance = per_step_variance

    def accepts(self, value: Optional[float]) -> bool:
        # Inclusive: "within 0.06" has to include 0.06. The base class is a strict `<`, which
        # would reject a placement exactly on the tolerance the LLM baselines accept.
        return value is not None and value <= self.threshold

    def describe(self) -> dict:
        return {"evaluator": self.name, "accept_threshold": self.threshold,
                "saved_resync": self.resync}

    def _example_statistic(self, fn, param, target, task, index) -> Optional[float]:
        table = task.table_for(index)
        if table is None:
            return None
        state = gaussian.run(fn, None if task.closed else param, table, target, self.resync,
                             self.per_step_variance)
        produced = state.positions
        if len(produced) != len(target):
            return self.miss_value + abs(len(produced) - len(target))
        worst = 0.0
        for mu, obs in zip(produced, target):
            d = math.sqrt(sum((mu[i] - obs[i]) ** 2 for i in range(3)))
            worst = max(worst, d)
        return worst


class MahalanobisEvaluator(Evaluator):
    '''Continuous observations scored against SPL's probabilistic focus, in sigma.

    The executor is `gaussian.GaussianState`, which mirrors `executor._predict_focus`
    (mu += mu_d, var += var_d) and `assign_focus` (mean snaps to the placed block, variance
    resets). Acceptance is "every placement within tau sigma": it needs no per-demo
    calibration and degrades naturally as the covariance compounds, unlike `distance`, which
    charges the tenth placement as strictly as the first.

    A second criterion, `penalised_loglik`, used to live here. It reproduced SPL's
    `_penalised_score` at `focus_score_penalty="trace"` -- a reward formula SPL has since
    retired (`spl_config.py` now defaults to `"reward"`). Mirroring SPL's *reward* is also the
    wrong shape for a baseline: DreamCoder and LILO accept a program on a 0/1 check, and
    borrowing SPL's reward would hand the baseline part of what SPL is being credited for.
    '''

    name = "mahalanobis"

    def __init__(self, tau: float = 2.0, resync: bool = True,
                 per_step_variance: bool = False):
        self.threshold = float(tau)
        self.resync = resync
        self.per_step_variance = per_step_variance

    def describe(self) -> dict:
        return {"evaluator": self.name, "accept_threshold": self.threshold,
                "saved_resync": self.resync}

    def _example_statistic(self, fn, param, target, task, index) -> Optional[float]:
        table = task.table_for(index)
        if table is None:
            return None
        state = gaussian.run(fn, None if task.closed else param, table, target, self.resync,
                             self.per_step_variance)
        placements = state.gaussians
        if len(placements) != len(target):
            return self.miss_value + abs(len(placements) - len(target))

        worst = 0.0
        for (mu, var), obs in zip(placements, target):
            worst = max(worst, gaussian.mahalanobis(obs, mu, var))
        return worst


# --------------------------------------------------------------------------------------- #

def build(configs) -> Evaluator:
    '''The evaluator named by a run config. Validates the combination first.'''
    observation_mode = getattr(configs, "observation_mode", "lattice")
    name = getattr(configs, "evaluator", "exact")
    validate(observation_mode, name)

    if name == "exact":
        return ExactEvaluator()
    resync = getattr(configs, "saved_resync", True)
    # Inherited from SPLConfig, so `move` is scored exactly as SPL's shift_focus(d, num_steps=n).
    per_step_variance = getattr(configs, "accumulate_shift_variance_per_step", False)
    if name == "distance":
        # None means "SPL's own tolerance", so the bar tracks SPL rather than a copy of it.
        return DistanceEvaluator(getattr(configs, "accept_epsilon", None), resync,
                                 per_step_variance)
    return MahalanobisEvaluator(tau=getattr(configs, "accept_tau", 3.0),
                                resync=resync, per_step_variance=per_step_variance)
