'''
ir.py

The intermediate representation the search works in, as a small algebraic datatype.

Why a local ADT rather than using `dreamcoder.program.Program` directly: `Term` is the pivot
between the two halves of this baseline. DreamCoder's `Program` is a lambda calculus with de
Bruijn indices -- the right shape for enumeration and for STITCH, and the wrong shape for
printing Python, where a loop index wants a name and a sequence wants to be a block of
statements. Translating once, here, keeps `lower.py` free of de Bruijn arithmetic and lets
everything downstream be built and tested without DreamCoder installed.

    dreamcoder Program  --(bridge.py)-->  Term  --(lower.py)-->  Python concept class
                                           |
                                           +--(evaluate)--> LatticeState

The grammar this represents is deliberately small: bounded iteration, sequencing, focus
save/restore, and integer arithmetic over the concept parameter and the loop indices. No
conditionals, no recursion, no assignment. See `grammar_level` for the three sizes.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from baseline_spl.symbolic.lattice import EMPTY, LatticeState

DIRECTION_NAMES: Tuple[str, ...] = ("left", "right", "front", "behind", "top")


# --------------------------------------------------------------------------------------- #
# Integer expressions: the concept parameter, loop indices, literals, arithmetic.
# --------------------------------------------------------------------------------------- #

class IntExpr:
    def eval(self, param: int, loops: Sequence[int]) -> int:
        raise NotImplementedError

    def render(self, param_name: str, loop_names: Sequence[str]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(IntExpr):
    value: int

    def eval(self, param, loops):
        return self.value

    def render(self, param_name, loop_names):
        return str(self.value)


@dataclass(frozen=True)
class Param(IntExpr):
    '''The concept's numeric argument -- `length`, `height`, `steps`.'''

    def eval(self, param, loops):
        return param

    def render(self, param_name, loop_names):
        return param_name


@dataclass(frozen=True)
class LoopVar(IntExpr):
    '''Index of an enclosing loop. depth 0 is the innermost.'''
    depth: int = 0

    def eval(self, param, loops):
        return loops[len(loops) - 1 - self.depth]

    def render(self, param_name, loop_names):
        return loop_names[len(loop_names) - 1 - self.depth]


@dataclass(frozen=True)
class BinOp(IntExpr):
    op: str          # "+", "-", "*"
    left: IntExpr
    right: IntExpr

    def eval(self, param, loops):
        a, b = self.left.eval(param, loops), self.right.eval(param, loops)
        return {"+": a + b, "-": a - b, "*": a * b}[self.op]

    def render(self, param_name, loop_names):
        a = self.left.render(param_name, loop_names)
        b = self.right.render(param_name, loop_names)
        # Parenthesise the lower-precedence operands rather than tracking precedence: the
        # expressions are two or three terms deep, so the extra parentheses cost nothing.
        if self.op == "*":
            a = f"({a})" if isinstance(self.left, BinOp) else a
            b = f"({b})" if isinstance(self.right, BinOp) else b
        else:
            b = f"({b})" if isinstance(self.right, BinOp) else b
        return f"{a} {self.op} {b}"


# --------------------------------------------------------------------------------------- #
# Terms: state -> state.
# --------------------------------------------------------------------------------------- #

class Term:
    def evaluate(self, state: LatticeState, param: int, loops: Tuple[int, ...] = ()) -> LatticeState:
        raise NotImplementedError


@dataclass(frozen=True)
class Shift(Term):
    direction: str

    def evaluate(self, state, param, loops=()):
        return state.shifted(self.direction)


@dataclass(frozen=True)
class Move(Term):
    """Shift `distance` cells in one direction, as one primitive application.

    DreamCoder's movement primitives take a distance: `left, right :: tint -> ttower ->
    ttower` (towerPrimitives.py), so moving four cells costs them ONE application plus a
    literal. Our `shift` moves exactly one cell, so the same move cost four applications
    nested inside every loop iteration -- pure overhead against their grammar, and it fell
    hardest on the concepts that step more than one cell at a time (`pins` moves two per
    placement; `x` and `arch_bridge` jump further).

    `shift` is KEPT alongside this rather than replaced. Replacing it would mirror them
    exactly but make our common case -- a single-cell step -- dearer than it is today, since
    every unit move would have to carry a literal `1`. Having both is strictly more
    expressive than either grammar alone, which is the intended direction.
    """
    direction: str
    distance: "IntExpr"

    def evaluate(self, state, param, loops=()):
        n = int(self.distance.value(param, loops))
        if n < 0:
            return state
        for _ in range(n):
            state = state.shifted(self.direction)
        return state


@dataclass(frozen=True)
class Place(Term):
    def evaluate(self, state, param, loops=()):
        return state.placed()


@dataclass(frozen=True)
class Seq(Term):
    steps: Tuple[Term, ...]

    def __init__(self, *steps: Term):
        flat: List[Term] = []
        for s in steps:
            # Flatten nested Seqs so the printed body has no redundant structure and STITCH
            # sees a canonical form.
            flat.extend(s.steps if isinstance(s, Seq) else [s])
        object.__setattr__(self, "steps", tuple(flat))

    def evaluate(self, state, param, loops=()):
        for step in self.steps:
            state = step.evaluate(state, param, loops)
        return state


@dataclass(frozen=True)
class Loop(Term):
    count: IntExpr
    body: Term

    def evaluate(self, state, param, loops=()):
        from baseline_spl.symbolic.lattice import LatticeBudgetExceeded, MAX_PLACEMENTS
        n = self.count.eval(param, loops)
        if n <= 0:
            return state
        if n > MAX_PLACEMENTS:
            raise LatticeBudgetExceeded(f"loop bound {n} too large")
        for i in range(n):
            state = self.body.evaluate(state, param, loops + (i,))
        return state


@dataclass(frozen=True)
class Saved(Term):
    '''Run the body, then restore the focus. Placements are kept.

    The ground truth's `save_focus = get_focus() ... assign_focus(save_focus)` pair, and the
    same primitive as DreamCoder's `tower_embed`.
    '''
    body: Term

    def evaluate(self, state, param, loops=()):
        return self.body.evaluate(state, param, loops).with_focus(state.focus)


# --------------------------------------------------------------------------------------- #

def evaluate(term: Term, param: int, state: LatticeState = None) -> LatticeState:
    return term.evaluate(EMPTY if state is None else state, param)


def positions(term: Term, param: int) -> List[Tuple[int, int, int]]:
    '''The placement cells a term produces -- the quantity every SPL metric compares.'''
    return evaluate(term, param).positions


def simplify(expr: IntExpr) -> IntExpr:
    '''Fold the arithmetic an IR term carries incidentally.

    A term built by composition often reads `i + 1 - 1` where it means `i` -- correct, but
    the emitted class is compared against SPL's on readability, so leaving it unfolded would
    handicap the baseline for no reason. Semantics-preserving; test_ir_fidelity and
    test_lowering both re-check the traces after folding.
    '''
    if not isinstance(expr, BinOp):
        return expr

    left, right = simplify(expr.left), simplify(expr.right)
    op = expr.op

    if isinstance(left, Const) and isinstance(right, Const):
        return Const(BinOp(op, left, right).eval(0, ()))

    # Identities.
    if op == "+" and isinstance(right, Const) and right.value == 0:
        return left
    if op == "+" and isinstance(left, Const) and left.value == 0:
        return right
    if op == "-" and isinstance(right, Const) and right.value == 0:
        return left
    if op == "*" and isinstance(right, Const) and right.value == 1:
        return left
    if op == "*" and isinstance(left, Const) and left.value == 1:
        return right
    if op == "*" and ((isinstance(right, Const) and right.value == 0)
                      or (isinstance(left, Const) and left.value == 0)):
        return Const(0)

    # (x + c1) +/- c2  and  (x - c1) +/- c2  ->  x +/- c
    if op in ("+", "-") and isinstance(right, Const) and isinstance(left, BinOp) \
            and left.op in ("+", "-") and isinstance(left.right, Const):
        inner = left.right.value if left.op == "+" else -left.right.value
        delta = inner + (right.value if op == "+" else -right.value)
        if delta == 0:
            return left.left
        return BinOp("+" if delta > 0 else "-", left.left, Const(abs(delta)))

    return BinOp(op, left, right)


def uses(term: Term, kind) -> bool:
    '''Does the term contain a node of this class? Used for the grammar_level contract.'''
    if isinstance(term, kind):
        return True
    if isinstance(term, Seq):
        return any(uses(s, kind) for s in term.steps)
    if isinstance(term, (Loop, Saved)):
        return uses(term.body, kind)
    return False


def size(term: Term) -> int:
    '''Node count -- a rough stand-in for description length when reporting search results.'''
    if isinstance(term, Seq):
        return 1 + sum(size(s) for s in term.steps)
    if isinstance(term, (Loop, Saved)):
        return 1 + size(term.body)
    return 1


# --------------------------------------------------------------------------------------- #
# Grammar levels -- which primitives the search may use. See the plan.
# --------------------------------------------------------------------------------------- #

GRAMMAR_LEVELS = {
    "minimal": {"shift", "place", "loop", "const", "param", "loopvar", "sub"},
    "standard": {"shift", "move", "place", "loop", "saved", "const", "param", "loopvar",
                 "add", "sub", "mul"},
    "extended": {"shift", "move", "place", "loop", "saved", "const", "param", "loopvar",
                 "add", "sub", "mul", "if_positive"},
}


def level_features(level: str) -> set:
    if level not in GRAMMAR_LEVELS:
        raise ValueError(f"unknown grammar_level {level!r}; "
                         f"choose from {sorted(GRAMMAR_LEVELS)}")
    return GRAMMAR_LEVELS[level]
