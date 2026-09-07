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

    Immutable: every primitive returns a new state.
    '''

    __slots__ = ("mu", "var", "placements", "cursor", "steps", "table",
                 "observations", "resync")

    def __init__(self, table: Dict[str, Tuple[Vec, Vec]], mu: Vec = ZERO,
                 var: Vec = RESET_VAR, placements: Tuple = (), cursor: int = 0,
                 steps: int = 0, observations: Sequence[Vec] = (), resync: bool = True):
        self.table = table
        self.mu = mu
        self.var = var
        self.placements = placements
        self.cursor = cursor
        self.steps = steps
        self.observations = observations
        self.resync = resync

    def _child(self, mu: Vec, var: Vec, placements: Tuple, cursor: int, steps: int):
        return GaussianState(self.table, mu, var, placements, cursor, steps,
                             self.observations, self.resync)

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
            resync: bool = True) -> GaussianState:
    '''The empty state for one (demonstration, program) pair.

    The origin is the first placed block, matching `tasks.demo_centroids`, so the first
    placement is at (0,0,0) by construction and the comparison is translation-invariant.
    '''
    return GaussianState(table, ZERO, RESET_VAR, (), 0, 0, tuple(observations), resync)


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


def penalised_score(observed: Vec, mu: Vec, var: Vec, lam: float = 1.2) -> float:
    '''SPL's `executor._penalised_score`: log p(x | focus) - lambda * tr(Sigma).

    Higher is better. `lam` is SPL's own `variance_penalty_weight`, 1.2 by default, and the
    trace of a diagonal covariance is the sum of the variance vector.
    '''
    return log_pdf(observed, mu, var) - lam * (var[0] + var[1] + var[2])


def penalised_ceiling(var: Vec, lam: float = 1.2) -> float:
    '''The best `penalised_score` achievable at this covariance -- the value when the
    prediction lands exactly on the observation. Subtracting it turns an unbounded
    log-density into a per-placement shortfall in nats, which is comparable across
    placements whose variance differs by orders of magnitude.
    '''
    return penalised_score(ZERO, ZERO, var, lam)


def run(fn, param: Optional[int], table, observations=(), resync: bool = True):
    '''Apply an evaluated IR term to its argument under the Gaussian executor.'''
    state = initial(table, observations, resync)
    return (fn if param is None else fn(param))(state)
