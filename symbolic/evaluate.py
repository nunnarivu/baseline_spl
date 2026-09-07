'''
evaluate.py

How the search decides a candidate program is correct.

Round 1 had one answer: the program's integer placement cells equal the demonstration's,
exactly. That is DreamCoder's own all-or-nothing acceptance (`task.py:126`) and it makes
acceptance a proof. It also only exists because the demonstration was discretized first --
a step SPL never takes, and therefore an advantage B3 was being handed rather than earning.

This module provides three evaluators behind one interface, so the choice is a config knob
and a reportable comparison rather than a decision baked into the search:

    exact           integer cells must match       (observation_mode='lattice')
    tolerance       every placement within eps m   (observation_mode='continuous')
    srn_likelihood  SPL's own probabilistic score  (observation_mode='continuous')

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

Calibration
-----------
The two `srn_likelihood` criteria are deliberately calibrated to each other. For a Gaussian
the log-density falls by exactly d**2/2 at Mahalanobis distance d, so `accept_margin = 4.5`
nats *is* `accept_tau = 3.0`. Choosing between them therefore compares how placements are
aggregated -- worst-case versus mean, and whether SPL's -lambda*tr(Sigma) variance penalty
participates -- rather than comparing two arbitrary scales.
'''

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from baseline_spl.symbolic import gaussian
from baseline_spl.symbolic.lattice import EMPTY, LatticeBudgetExceeded

# Exceptions an enumerated program can raise that mean "this program does not solve the
# task", never "the run is broken".
_RUN_ERRORS = (LatticeBudgetExceeded, RecursionError, IndexError, ValueError, TypeError,
               ZeroDivisionError, KeyError, OverflowError, AttributeError)

VALID_COMBINATIONS = {
    ("lattice", "exact"),
    ("continuous", "tolerance"),
    ("continuous", "srn_likelihood"),
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
        produced = (fn(EMPTY) if task.closed else fn(param)(EMPTY)).positions
        return trace_distance(produced, target)


class ToleranceEvaluator(Evaluator):
    '''Continuous observations, deterministic execution, a fixed radius in metres.

    Uses the SRN means and ignores its variances, so the executor is the Gaussian one with
    the uncertainty channel unused. Simple to explain and independent of any threshold
    calibration, but it charges a placement reached after ten compounding shifts exactly as
    strictly as the first -- which is the asymmetry the probabilistic evaluator exists to
    remove. Whether that matters here is the empirical question the two knobs answer.
    '''

    name = "tolerance"

    def __init__(self, epsilon: float = 0.03, resync: bool = True):
        self.threshold = float(epsilon)
        self.resync = resync

    def describe(self) -> dict:
        return {"evaluator": self.name, "accept_threshold": self.threshold,
                "saved_resync": self.resync}

    def _example_statistic(self, fn, param, target, task, index) -> Optional[float]:
        table = task.table_for(index)
        if table is None:
            return None
        state = gaussian.run(fn, None if task.closed else param, table, target, self.resync)
        produced = state.positions
        if len(produced) != len(target):
            return self.miss_value + abs(len(produced) - len(target))
        worst = 0.0
        for mu, obs in zip(produced, target):
            d = math.sqrt(sum((mu[i] - obs[i]) ** 2 for i in range(3)))
            worst = max(worst, d)
        return worst


class LikelihoodEvaluator(Evaluator):
    '''Continuous observations scored with SPL's own probabilistic focus.

    The executor is `gaussian.GaussianState`, which mirrors `executor._predict_focus`
    (mu += mu_d, var += var_d) and `assign_focus` (mean snaps to the placed block, variance
    resets). Two acceptance criteria, selected by `criterion`:

      'mahalanobis'       worst placement, in sigma.  Interpretable as "within tau sigma
                          everywhere", needs no per-demo calibration, and degrades naturally
                          as the covariance compounds.
      'penalised_loglik'  SPL's `_penalised_score` verbatim, including the
                          -lambda*tr(Sigma) variance penalty, expressed as a mean shortfall
                          in nats below the best score achievable at that covariance.
                          Subtracting the ceiling is what makes placements with very
                          different variances comparable.
    '''

    name = "srn_likelihood"

    def __init__(self, criterion: str = "mahalanobis", tau: float = 2.0,
                 margin: float = 2.0, lam: float = 1.2, resync: bool = True):
        if criterion not in ("mahalanobis", "penalised_loglik"):
            raise ValueError(f"unknown accept_criterion {criterion!r}")
        self.criterion = criterion
        self.lam = float(lam)
        self.resync = resync
        self.threshold = float(tau) if criterion == "mahalanobis" else float(margin)

    def describe(self) -> dict:
        return {"evaluator": self.name, "accept_criterion": self.criterion,
                "accept_threshold": self.threshold,
                "variance_penalty_weight": self.lam, "saved_resync": self.resync}

    def _example_statistic(self, fn, param, target, task, index) -> Optional[float]:
        table = task.table_for(index)
        if table is None:
            return None
        state = gaussian.run(fn, None if task.closed else param, table, target, self.resync)
        placements = state.gaussians
        if len(placements) != len(target):
            return self.miss_value + abs(len(placements) - len(target))

        if self.criterion == "mahalanobis":
            worst = 0.0
            for (mu, var), obs in zip(placements, target):
                worst = max(worst, gaussian.mahalanobis(obs, mu, var))
            return worst

        total = 0.0
        for (mu, var), obs in zip(placements, target):
            shortfall = (gaussian.penalised_ceiling(var, self.lam)
                         - gaussian.penalised_score(obs, mu, var, self.lam))
            total += max(0.0, shortfall)
        return total / max(1, len(placements))


# --------------------------------------------------------------------------------------- #

def build(configs) -> Evaluator:
    '''The evaluator named by a run config. Validates the combination first.'''
    observation_mode = getattr(configs, "observation_mode", "lattice")
    name = getattr(configs, "evaluator", "exact")
    validate(observation_mode, name)

    if name == "exact":
        return ExactEvaluator()
    resync = getattr(configs, "saved_resync", True)
    if name == "tolerance":
        return ToleranceEvaluator(getattr(configs, "accept_epsilon", 0.03), resync)
    return LikelihoodEvaluator(
        criterion=getattr(configs, "accept_criterion", "mahalanobis"),
        tau=getattr(configs, "accept_tau", 3.0),
        margin=getattr(configs, "accept_margin", 4.5),
        lam=getattr(configs, "variance_penalty_weight", 1.2),
        resync=resync)
