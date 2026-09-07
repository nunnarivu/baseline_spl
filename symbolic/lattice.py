'''
lattice.py

The semantics of SPL's DSL as pure functions over an immutable state.

This is a functional restatement of `SPL.utils.metrics.IdealExecutor`: focus starts at the
origin, each shift moves exactly one unit cell along `ParameterSettings.DIRECTIONS`, and every
placement lands on an integer cell. Both must agree exactly -- `tests/test_ir_fidelity.py`
checks that against `run_gt_program` for every ground-truth concept.

Why a second implementation rather than reusing IdealExecutor: DreamCoder evaluates a program
by applying curried closures (`dreamcoder/program.py:406`), so each primitive has to be a pure
function of the state rather than a method mutating an executor. Purity also makes `saved`
(restore the focus after running a sub-program) a one-liner instead of a save/restore protocol.
'''

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

Cell = Tuple[int, int, int]

# Integer direction vectors, taken from the simulator so the two cannot drift.
# Note DEFAULT_ACTIONS exposes only five of these six to the search; "below" is unused by
# every ground-truth program and by SPL's own action space (config/primitive_config.py:54).
def _load_directions() -> Dict[str, Cell]:
    from nsei_simulator.dataset.spg.configs import ParameterSettings
    return {k.lower(): tuple(int(round(v)) for v in d)
            for k, d in ParameterSettings.DIRECTIONS.items()}


DIRECTIONS: Dict[str, Cell] = _load_directions()

ORIGIN: Cell = (0, 0, 0)

# Guards against a runaway program during enumeration. A term like
# `loop (12*12) (loop (12*12) place)` is well-typed and cheap to write but expensive to run,
# and the enumerator will produce many of them.
MAX_PLACEMENTS = 20000
MAX_STEPS = 200000


class LatticeBudgetExceeded(Exception):
    '''Raised when a program exceeds MAX_PLACEMENTS / MAX_STEPS. Caught by the task scorer and
    treated as "this program does not solve the task", never as a crash.'''


class LatticeState:
    '''focus     - the current cell
    placements   - ((object_id, cell), ...) in placement order
    cursor       - index of the next object to place
    steps        - work done so far, for the budget guard

    Immutable: every primitive returns a new state. `positions` is the quantity every metric
    compares (see metrics.py:495, which diffs exactly this list).
    '''

    __slots__ = ("focus", "placements", "cursor", "steps")

    def __init__(self, focus: Cell = ORIGIN, placements: Tuple = (), cursor: int = 0,
                 steps: int = 0):
        self.focus = focus
        self.placements = placements
        self.cursor = cursor
        self.steps = steps

    def _tick(self, n: int = 1) -> int:
        steps = self.steps + n
        if steps > MAX_STEPS:
            raise LatticeBudgetExceeded(f"exceeded {MAX_STEPS} steps")
        return steps

    def shifted(self, direction: str) -> "LatticeState":
        d = DIRECTIONS[direction]
        focus = (self.focus[0] + d[0], self.focus[1] + d[1], self.focus[2] + d[2])
        return LatticeState(focus, self.placements, self.cursor, self._tick())

    def placed(self) -> "LatticeState":
        if len(self.placements) >= MAX_PLACEMENTS:
            raise LatticeBudgetExceeded(f"exceeded {MAX_PLACEMENTS} placements")
        return LatticeState(self.focus, self.placements + ((self.cursor, self.focus),),
                            self.cursor + 1, self._tick())

    def with_focus(self, focus: Cell) -> "LatticeState":
        return LatticeState(focus, self.placements, self.cursor, self._tick(0))

    def restored(self, saved: "LatticeState") -> "LatticeState":
        '''`saved`'s restore step. On a lattice this is just "put the focus back".

        It exists as a method rather than as `with_focus(s.focus)` inside the primitive
        because the probabilistic executor cannot express its restore that way: SPL's
        `assign_focus` also resets the covariance, and optionally snaps the mean to the
        observed block. Dispatching on the state keeps one set of primitives driving both.
        '''
        return self.with_focus(saved.focus)

    @property
    def positions(self) -> List[Cell]:
        return [cell for _obj, cell in self.placements]

    def __repr__(self) -> str:
        return f"LatticeState(focus={self.focus}, n_placed={len(self.placements)})"


EMPTY = LatticeState()


# --------------------------------------------------------------------------------------- #
# Primitive semantics. Curried, because DreamCoder applies one argument at a time.
# --------------------------------------------------------------------------------------- #

def shift(direction: str) -> Callable[[LatticeState], LatticeState]:
    return lambda s: s.shifted(direction)


def move(direction: str):
    """move :: tdir -> tint -> tstate -> tstate

    Shift `n` cells in one application, mirroring DreamCoder's `left, right :: tint ->
    ttower -> ttower` (towerPrimitives.py). `shift` is kept for the unit case, so the
    grammar is strictly more expressive than either theirs or our previous one.
    """
    def with_count(n):
        def run(s):
            count = int(n)
            if count < 0:
                return s
            if count > MAX_PLACEMENTS:
                raise LatticeBudgetExceeded(f"move distance {count} too large")
            for _ in range(count):
                s = s.shifted(direction)
            return s
        return run
    return with_count


def place(s: LatticeState) -> LatticeState:
    return s.placed()


def loop(n):
    '''loop :: tint -> (tint -> tstate -> tstate) -> tstate -> tstate

    Bounded iteration exposing the index, mirroring DreamCoder's `tower_loopM`
    (domains/tower/towerPrimitives.py:149). The index is what lets a body depend on the
    iteration -- `tower(i+1)` in staircase, `length-i` in pins, `height*2-1-2i` in pyramid.
    '''
    def with_body(f):
        def run(s: LatticeState) -> LatticeState:
            count = int(n)
            if count < 0:
                return s
            if count > MAX_PLACEMENTS:
                raise LatticeBudgetExceeded(f"loop bound {count} too large")
            for i in range(count):
                s = f(i)(s)
            return s
        return run
    return with_body


def saved(f):
    '''saved :: (tstate -> tstate) -> tstate -> tstate

    Run f, then restore the focus to where it was. Placements made by f are kept. This is the
    ground truth's `save_focus = get_focus() ... assign_focus(save_focus)` idiom, and it is
    the same primitive as DreamCoder's `tower_embed`.
    '''
    def run(s: LatticeState) -> LatticeState:
        return f(s).restored(s)
    return run


def add(a):
    return lambda b: a + b


def sub(a):
    return lambda b: a - b


def mul(a):
    return lambda b: a * b


def run(term_value, argument=None, state: LatticeState = None) -> LatticeState:
    '''Apply an evaluated IR term to its argument and an initial state.

    `term_value` is what `Program.evaluate([])` returns: a curried closure of type
    `tint -> tstate -> tstate` for a concept-level task, or `tstate -> tstate` for a
    demo-level one (argument=None).
    '''
    s = EMPTY if state is None else state
    fn = term_value if argument is None else term_value(argument)
    return fn(s)
