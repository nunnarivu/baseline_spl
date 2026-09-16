'''
gaussian.py

The same DSL semantics as `lattice.py`, but threading SPL's *probabilistic* focus instead of
an integer cell.

`lattice.LatticeState` answers "which cell does this program place blocks on". That question
only exists because the demonstrations were discretized first, which is a step SPL never
takes: `SPL.learn` receives meshes, and `plan_agent` recovers actions by searching over a
learned stochastic primitive. This module answers the question SPL actually faces -- "where,
and with what confidence, does this program predict each block lands" -- so the search can be
scored on the same terms.

It mirrors `SPL.model.executor.InductiveStructureExecutor` primitive for primitive:

    shift(d)   _predict_focus  : mu += mu_d ;  var += var_d      (Kalman prediction)
    place      place at focus  : record (mu, var)
    saved f    assign_focus    : mu <- the block the body placed first ;  var <- 1e-4

Two implementation notes that matter more than they look.

**The covariance is diagonal everywhere, so it is a 3-vector.** The SRN emits a diagonal
Gaussian, the initial focus is isotropic, sums of diagonals stay diagonal, and `assign_focus`
writes an isotropic reset. Nothing in the DSL can introduce a correlation. So Mahalanobis
distance needs no matrix inverse, and this executor costs a handful of float operations per
primitive rather than a linear solve -- which is what lets it sit inside an enumeration
running at ~1,800 programs/s.

**Plain tuples, not numpy.** These are 3-vectors evaluated millions of times; numpy's
per-call overhead (~1 us) would dominate, while tuple arithmetic is ~0.3 us. The arrays only
appear at the boundary, in the evaluators.

The state deliberately implements the same duck-typed protocol as `LatticeState`
(`shifted`, `placed`, `restored`, `focus`, `positions`), so `bridge.PRIMITIVES` -- which binds
`lattice.shift`, `lattice.place`, `lattice.loop`, `lattice.saved` -- drives either executor
with no change at all. The choice of executor is made by the initial state handed to the
program, and nowhere else.
'''

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.symbolic.lattice import (MAX_PLACEMENTS, MAX_STEPS,
                                           LatticeBudgetExceeded)

Vec = Tuple[float, float, float]

# SPL's assign_focus sets cov_scale=0.0001 (executor.py, `make_focus(position,
# cov_scale=0.0001)`), i.e. sigma = 0.01 m. Used both for the initial focus and for the reset
# `saved` performs, because both are the same operation in SPL.
RESET_VAR: Vec = (1e-4, 1e-4, 1e-4)

ZERO: Vec = (0.0, 0.0, 0.0)


class GaussianState:
    '''focus        - (mean, variance) of the current focus, both 3-vectors in metres
    placements      - ((object_id, mean, variance), ...) captured at each `place`
    cursor          - index of the next object to place
    steps           - work done so far, for the budget guard
    table           - {direction: (mean, variance)} from srn.py; constant for a demo
    observations    - the demo's observed centroids, used by `saved` (see `restored`)
    resync          - whether `saved` snaps to the observation or to the prediction
    per_step_variance - whether `moved` adds the variance once per step or once per call
                      (SPL's `accumulate_shift_variance_per_step`)

    Immutable: every primitive returns a new state.
    '''

    __slots__ = ("mu", "var", "placements", "cursor", "steps", "table",
                 "observations", "resync", "per_step_variance")

    def __init__(self, table: Dict[str, Tuple[Vec, Vec]], mu: Vec = ZERO,
                 var: Vec = RESET_VAR, placements: Tuple = (), cursor: int = 0,
                 steps: int = 0, observations: Sequence[Vec] = (), resync: bool = True,
                 per_step_variance: bool = False):
        self.table = table
        self.mu = mu
        self.var = var
        self.placements = placements
        self.cursor = cursor
        self.steps = steps
        self.observations = observations
        self.resync = resync
        self.per_step_variance = per_step_variance

    def _child(self, mu: Vec, var: Vec, placements: Tuple, cursor: int, steps: int):
        return GaussianState(self.table, mu, var, placements, cursor, steps,
                             self.observations, self.resync, self.per_step_variance)

    def _tick(self, n: int = 1) -> int:
        steps = self.steps + n
        if steps > MAX_STEPS:
            raise LatticeBudgetExceeded(f"exceeded {MAX_STEPS} steps")
        return steps

    @property
    def focus(self) -> Tuple[Vec, Vec]:
        return (self.mu, self.var)

    def shifted(self, direction: str) -> "GaussianState":
        '''Kalman prediction: the mean translates and the variance accumulates.'''
        entry = self.table.get(direction)
        if entry is None:
            raise LatticeBudgetExceeded(f"no SRN entry for direction {direction!r}")
        d_mu, d_var = entry
        mu = (self.mu[0] + d_mu[0], self.mu[1] + d_mu[1], self.mu[2] + d_mu[2])
        var = (self.var[0] + d_var[0], self.var[1] + d_var[1], self.var[2] + d_var[2])
        return self._child(mu, var, self.placements, self.cursor, self._tick())

    def moved(self, direction: str, n: int) -> "GaussianState":
        '''SPL's shift_focus(direction, num_steps=n): the mean translates n steps; the
        variance is added once, or n times when `per_step_variance`.'''
        entry = self.table.get(direction)
        if entry is None:
            raise LatticeBudgetExceeded(f"no SRN entry for direction {direction!r}")
        d_mu, d_var = entry
        k = n if self.per_step_variance else 1
        mu = (self.mu[0] + n * d_mu[0], self.mu[1] + n * d_mu[1], self.mu[2] + n * d_mu[2])
        var = (self.var[0] + k * d_var[0], self.var[1] + k * d_var[1], self.var[2] + k * d_var[2])
        return self._child(mu, var, self.placements, self.cursor, self._tick(n))

    def placed(self) -> "GaussianState":
        if len(self.placements) >= MAX_PLACEMENTS:
            raise LatticeBudgetExceeded(f"exceeded {MAX_PLACEMENTS} placements")
        placements = self.placements + ((self.cursor, self.mu, self.var),)
        return self._child(self.mu, self.var, placements, self.cursor + 1, self._tick())

    def restored(self, saved: "GaussianState") -> "GaussianState":
        '''`saved`'s restore step, mirroring SPL's `assign_focus(object_id=...)`.

        The lowering emits `assign_focus(object_id=self._placed[_a])` where `_a` is the index
        of the first block the body placed, and SPL resolves that to
        `mesh_centroid(state.state[object_id])` -- the block's *observed* position, with the
        covariance reset to 1e-4. So the faithful restore snaps to reality, not to what the
        program predicted, and that is what `resync=True` does.

        This is load-bearing and cuts both ways. It is what SPL's live executor does, so it
        belongs in a faithful port; but it also lets a candidate re-synchronise to the
        demonstration at every `saved`, which makes acceptance easier for exactly the
        composite concepts the search struggles with. `resync=False` restores the predicted
        mean instead, and the two are compared in tests/test_gaussian.py.
        '''
        index = len(saved.placements)
        mu = saved.mu
        # The body must actually have placed a block. The lowering emits
        # `assign_focus(object_id=self._placed[_a])`, which raises IndexError when the body
        # placed nothing -- so an empty body cannot snap to an observation it never produced.
        # Guarding on `index < len(observations)` alone is not enough: that asks whether the
        # *demonstration* has a k-th block, not whether this program placed one.
        body_placed = len(self.placements) > index
        if self.resync and body_placed and index < len(self.observations):
            mu = self.observations[index]
        return self._child(mu, RESET_VAR, self.placements, self.cursor, self._tick(0))

    # `LatticeState` restores through `with_focus`; keep the same entry point so anything
    # written against either state keeps working.
    def with_focus(self, focus: Tuple[Vec, Vec]) -> "GaussianState":
        mu, var = focus
        return self._child(mu, var, self.placements, self.cursor, self._tick(0))

    @property
    def positions(self) -> List[Vec]:
        '''Predicted means, in placement order. Same shape as `LatticeState.positions`, so
        the recognition model's tokenizer and the near-miss ranking read either state.'''
        return [mu for _obj, mu, _var in self.placements]

    @property
    def variances(self) -> List[Vec]:
        return [var for _obj, _mu, var in self.placements]

    @property
    def gaussians(self) -> List[Tuple[Vec, Vec]]:
        return [(mu, var) for _obj, mu, var in self.placements]

    def __repr__(self) -> str:
        return (f"GaussianState(mu={tuple(round(v, 4) for v in self.mu)}, "
                f"n_placed={len(self.placements)})")


def initial(table: Dict[str, Tuple[Vec, Vec]], observations: Sequence[Vec] = (),
            resync: bool = True, per_step_variance: bool = False) -> GaussianState:
    '''The empty state for one (demonstration, program) pair.

    The origin is the first placed block, matching `tasks.demo_centroids`, so the first
    placement is at (0,0,0) by construction and the comparison is translation-invariant.
    '''
    return GaussianState(table, ZERO, RESET_VAR, (), 0, 0, tuple(observations), resync,
                         per_step_variance)


def mahalanobis(observed: Vec, mu: Vec, var: Vec) -> float:
    '''Distance in sigma. Diagonal covariance, so this is a sum of three ratios.'''
    total = 0.0
    for i in range(3):
        v = var[i]
        if v <= 0.0:
            v = 1e-12
        delta = observed[i] - mu[i]
        total += delta * delta / v
    return total ** 0.5


# log(2*pi), for the Gaussian normaliser.
_LOG_2PI = 1.8378770664093453


def log_pdf(observed: Vec, mu: Vec, var: Vec) -> float:
    '''log N(observed | mu, diag(var)) for a 3-D diagonal Gaussian.'''
    import math

    total = 0.0
    for i in range(3):
        v = var[i]
        if v <= 0.0:
            v = 1e-12
        delta = observed[i] - mu[i]
        total += -0.5 * (_LOG_2PI + math.log(v) + delta * delta / v)
    return total


# `penalised_score` / `penalised_ceiling` lived here: SPL's `_penalised_score` at
# focus_score_penalty="trace", used by the `penalised_loglik` acceptance criterion. Both are
# gone. SPL retired that formula (its default is now "reward"), and mirroring SPL's *reward* is
# the wrong shape for a baseline anyway -- DreamCoder and LILO accept on a 0/1 check, so
# borrowing the reward would hand the baseline part of what SPL is credited for.


def run(fn, param, table, observations=(), resync: bool = True,
        per_step_variance: bool = False):
    '''Apply an evaluated IR term to its arguments under the Gaussian executor.

    `param` is None for a closed program, a bare int for a 1-argument concept, or a tuple in
    sketch order. A program of arity k is k nested closures, so each argument is applied in
    turn.
    '''
    from baseline_spl.symbolic.ir import args

    state = initial(table, observations, resync, per_step_variance)
    for value in args(param):
        fn = fn(value)
    return fn(state)
