'''
generalise.py

Recover a size-parameterised concept from a concept's CLOSED demo-level solutions.

A demo-level run solves each demonstration separately, so `row` at length 5 and `row` at
length 3 are two unrelated programs with the size frozen as a literal:

    row_0000   (lambda (loop 5 (lambda (lambda (shift RIGHT (place $0)))) $0))
    row_0032   (lambda (loop 3 (lambda (lambda (shift RIGHT (place $0)))) $0))

They differ in exactly one place. Anti-unifying them -- replacing the one difference with a
variable -- yields the parameterised program a concept-level search would have found:

    row        (lambda (lambda (loop $1 (lambda (lambda (shift RIGHT (place $0)))) $0)))

That is the whole idea. It matters because DreamCoder's own generalisation is a *second
search* over held-out tasks (`dreamcoder.py:567`), which needs the target size to exist as a
literal in the grammar; a recovered class takes the size as an argument and so works at any n.

Scope and honesty
-----------------
This gives the baseline something published DreamCoder does not have. It is reported as a
separate, disclosed variant, never folded into the headline B3-a number.

Only SOLVED demos may be used. An `approximate` solution is a near miss, and anti-unifying two
near misses produces a confident-looking program that is simply wrong -- measured: `staircase`'s
two approximate solutions differ in both the size AND the direction, and generalising them
invents a spurious direction argument.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.symbolic import ir
from baseline_spl.symbolic.ir import (BinOp, Const, IntExpr, Loop, Move, Param, Place,
                                      Saved, Seq, Shift, Term)


@dataclass(frozen=True)
class Hole:
    '''One position where the demos disagree, and what each of them had there.

    `numeric` separates "these programs differ in a size" -- the difference we are looking
    for -- from "these programs have different shapes", which is a disagreement no amount of
    parameterisation can repair.
    '''

    index: int
    values: Tuple[object, ...]
    numeric: bool

    def constants(self) -> Optional[Tuple[int, ...]]:
        '''The witnessed values as plain ints, or None if any is not a literal.'''
        if not self.numeric or not all(isinstance(v, Const) for v in self.values):
            return None
        return tuple(v.value for v in self.values)


class _HoleExpr(IntExpr):
    '''Placeholder standing in for a hole while the generalisation is being built.'''

    def __init__(self, index: int):
        self.index = index

    def eval(self, param, loops):                      # pragma: no cover - never executed
        raise RuntimeError("a hole has no value; substitute it before evaluating")

    def render(self, param_name, loop_names):          # pragma: no cover
        return f"?{self.index}"

    def __eq__(self, other):
        return isinstance(other, _HoleExpr) and other.index == self.index

    def __hash__(self):
        return hash(("hole", self.index))


# --------------------------------------------------------------------------------------- #
# Anti-unification
# --------------------------------------------------------------------------------------- #

def antiunify(terms: Sequence[Term]) -> Tuple[Term, List[Hole]]:
    '''Least general generalisation of `k` terms.

    Folds over all `k` demos at once rather than pairwise, so every witnessed value has to line
    up. That makes the test STRICTER as the dataset grows -- three demos at three distinct
    parameters is far stronger evidence for "this hole is the size" than two.
    '''
    if not terms:
        raise ValueError("nothing to anti-unify")
    holes: List[Hole] = []
    generalised = _au_term(list(terms), holes)
    return generalised, holes


def _fresh(values: Sequence[object], numeric: bool, holes: List[Hole]):
    hole = Hole(index=len(holes), values=tuple(values), numeric=numeric)
    holes.append(hole)
    return _HoleExpr(hole.index) if numeric else hole


def _au_term(terms: List[Term], holes: List[Hole]):
    '''Generalise `k` Terms. A structural disagreement is recorded as a non-numeric hole.'''
    head = terms[0]

    if all(t == head for t in terms[1:]):
        return head

    kinds = {type(t) for t in terms}
    if len(kinds) != 1:
        return _fresh(terms, numeric=False, holes=holes)

    if isinstance(head, Shift):
        # Same node type but unequal, so the directions differ: a shape disagreement.
        return _fresh(terms, numeric=False, holes=holes)

    if isinstance(head, Move):
        if len({t.direction for t in terms}) != 1:
            return _fresh(terms, numeric=False, holes=holes)
        return Move(head.direction, _au_int([t.distance for t in terms], holes))

    if isinstance(head, Place):
        return head

    if isinstance(head, Seq):
        if len({len(t.steps) for t in terms}) != 1:
            return _fresh(terms, numeric=False, holes=holes)
        return Seq(*[_au_term(list(group), holes) for group in zip(*[t.steps for t in terms])])

    if isinstance(head, Loop):
        return Loop(_au_int([t.count for t in terms], holes),
                    _au_term([t.body for t in terms], holes))

    if isinstance(head, Saved):
        return Saved(_au_term([t.body for t in terms], holes))

    return _fresh(terms, numeric=False, holes=holes)


def _au_int(exprs: List[IntExpr], holes: List[Hole]) -> IntExpr:
    '''Generalise `k` integer expressions. Differences here are the ones we want.'''
    head = exprs[0]
    if all(e == head for e in exprs[1:]):
        return head

    kinds = {type(e) for e in exprs}
    if kinds == {Const}:
        return _fresh(exprs, numeric=True, holes=holes)

    if len(kinds) == 1 and isinstance(head, BinOp):
        if len({e.op for e in exprs}) == 1:
            return BinOp(head.op,
                         _au_int([e.left for e in exprs], holes),
                         _au_int([e.right for e in exprs], holes))

    # Mixed kinds (a Const in one demo, a LoopVar in another) is a shape disagreement dressed
    # up as arithmetic; treat it as numeric so the gate can still reject it on its values.
    return _fresh(exprs, numeric=True, holes=holes)


# --------------------------------------------------------------------------------------- #
# Substitution
# --------------------------------------------------------------------------------------- #

def substitute(term: Term, replacements: Dict[int, IntExpr]) -> Term:
    '''Replace hole placeholders by index. Unlisted holes are left in place.'''
    return _sub_term(term, replacements)


def _sub_term(term: Term, rep: Dict[int, IntExpr]) -> Term:
    if isinstance(term, Move):
        return Move(term.direction, _sub_int(term.distance, rep))
    if isinstance(term, Seq):
        return Seq(*[_sub_term(s, rep) for s in term.steps])
    if isinstance(term, Loop):
        return Loop(_sub_int(term.count, rep), _sub_term(term.body, rep))
    if isinstance(term, Saved):
        return Saved(_sub_term(term.body, rep))
    return term


def _sub_int(expr: IntExpr, rep: Dict[int, IntExpr]) -> IntExpr:
    if isinstance(expr, _HoleExpr):
        return rep.get(expr.index, expr)
    if isinstance(expr, BinOp):
        return ir.simplify(BinOp(expr.op, _sub_int(expr.left, rep), _sub_int(expr.right, rep)))
    return expr


def _constants(expr: IntExpr) -> List[int]:
    '''Every integer literal in an expression, for the single-demo fallback.'''
    if isinstance(expr, Const):
        return [expr.value]
    if isinstance(expr, BinOp):
        return _constants(expr.left) + _constants(expr.right)
    return []


def replace_constant(term: Term, target: int) -> Term:
    '''Replace every `Const(target)` with `Param()` -- the single-demo fallback.

    Unsound on its own: a literal that merely happens to equal the parameter is replaced too.
    Only the validation gate makes it safe, which is why this route is tagged in the report.
    '''
    return _rc_term(term, target)


def _rc_term(term: Term, target: int) -> Term:
    if isinstance(term, Move):
        return Move(term.direction, _rc_int(term.distance, target))
    if isinstance(term, Seq):
        return Seq(*[_rc_term(s, target) for s in term.steps])
    if isinstance(term, Loop):
        return Loop(_rc_int(term.count, target), _rc_term(term.body, target))
    if isinstance(term, Saved):
        return Saved(_rc_term(term.body, target))
    return term


def _rc_int(expr: IntExpr, target: int) -> IntExpr:
    if isinstance(expr, Const) and expr.value == target:
        return Param()
    if isinstance(expr, BinOp):
        return BinOp(expr.op, _rc_int(expr.left, target), _rc_int(expr.right, target))
    return expr


# --------------------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------------------- #

#: Why a concept produced no parameterised class. Reported rather than silently dropped, so a
#: run's recovery rate can be read against a reason instead of a shrug.
REASONS = {
    "no_solved_demos": "no demonstration of this concept was solved",
    "structural_disagreement": "the solved demos have different shapes, not just different sizes",
    "no_hole": "the solved demos are identical, so no size could be identified",
    "many_holes": "the solved demos differ in more than one place",
    "hole_not_param": "the differing values do not match the demonstrations' parameters",
    "single_demo_disabled": "only one usable demo and the fallback route is disabled",
    "no_candidate": "no substitution reproduced the demonstrations",
    "failed_holdout": "fits the demos it was built from but not one held out from it",
    "ambiguous": ("several readings reproduce every demo -- the demos cannot say which "
                  "argument each hole tracks, so none is registered"),
}


@dataclass
class Recovery:
    '''What recovery produced for one concept, and why.'''

    concept: str
    term: Optional[Term] = None
    route: str = "none"                 # "antiunify" | "single_demo" | "none"
    reason: Optional[str] = None
    holes: int = 0
    parameters: Tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return self.term is not None


def candidates(solutions: Sequence[Term], parameters: Sequence[int], *,
               allow_single_demo: bool = True) -> Tuple[List[Term], str, Optional[str]]:
    '''Parameterised programs worth validating, best route first.

    Returns (candidates, route, reason-if-none). Never decides correctness -- that is the
    validation gate's job, because only the gate can run the program against the demos.
    '''
    if not solutions:
        return [], "none", "no_solved_demos"

    distinct = len(set(parameters)) > 1
    if len(solutions) >= 2 and distinct:
        generalised, holes = antiunify(solutions)
        if any(not h.numeric for h in holes):
            return [], "none", "structural_disagreement"
        if not holes:
            return [], "none", "no_hole"

        # Each hole must be an affine function of the size: w = a*n + b. Two things make this
        # necessary rather than fancy.
        #
        # Several holes is normal -- a concept that mentions its parameter more than once
        # produces one hole per mention. Requiring a single hole rejected 5 of 16 oracles.
        #
        # And the coefficients are rarely 1 and 0, because the SEARCH FOLDS THE ARITHMETIC.
        # `pyramid` loops over a row of 2n-1, so at n=3 the solver finds `loop 5`, not
        # `loop (2*3-1)`. Anti-unifying gives a hole witnessing (5, 9) against parameters
        # (3, 5); demanding w == n would throw the concept away. Solving w = 2n-1 keeps it.
        # With more than one argument each hole is solved against EACH of them, and a hole that
        # fits several is genuinely ambiguous -- `rectangle(4, 4)` and `rectangle(6, 6)` cannot
        # say whether a hole tracks length or breadth. Picking one would be a coin flip that
        # produces a confident wrong class, so every consistent reading becomes its own
        # candidate and the validation gate decides. If more than one survives the gate the
        # caller records `ambiguous` rather than registering either.
        from baseline_spl.symbolic.ir import args as _args

        arity = max((len(_args(p)) for p in parameters), default=1)
        readings: List[Dict[int, IntExpr]] = [{}]
        for hole in holes:
            extended: List[Dict[int, IntExpr]] = []
            for index in range(arity):
                expr = _affine_against(hole.constants(), parameters, index)
                if expr is None:
                    continue
                for reading in readings:
                    extended.append({**reading, hole.index: expr})
            if not extended:
                return [], "none", "hole_not_param"
            readings = extended

        return [substitute(generalised, r) for r in readings], "antiunify", None

    # Fallback: one demo, or several demos that all share a parameter. Temporary -- every
    # concept is expected to have 2-3 distinct parameters once the dataset is updated.
    if not allow_single_demo:
        return [], "none", "single_demo_disabled"
    term, param = solutions[0], parameters[0]
    if param not in _all_constants(term):
        return [], "none", "hole_not_param"
    return [replace_constant(term, param)], "single_demo", None


def _affine_against(witnessed: Optional[Tuple[int, ...]], parameters: Sequence[int],
                    index: int = 0) -> Optional[IntExpr]:
    '''Express a hole as `a * arg[index] + b`, or None if no integer relation fits every demo.

    Two demos determine `a` and `b` exactly; three or more over-determine them, so the extra
    demos become a consistency check rather than more unknowns -- which is why recovery gets
    *stricter*, not looser, as the dataset grows.
    '''
    from baseline_spl.symbolic.ir import args as _args

    axis = [_args(p)[index] for p in parameters if len(_args(p)) > index]
    if len(axis) != len(parameters):
        return None
    expr = _affine(witnessed, axis)
    if expr is None or index == 0:
        return expr
    # Re-point the recovered expression at argument `index`.
    return _reindex(expr, index)


def _reindex(expr: IntExpr, index: int) -> IntExpr:
    if isinstance(expr, Param):
        return Param(index)
    if isinstance(expr, BinOp):
        return BinOp(expr.op, _reindex(expr.left, index), _reindex(expr.right, index))
    return expr


def _affine(witnessed: Optional[Tuple[int, ...]],
            parameters: Sequence[int]) -> Optional[IntExpr]:
    '''Express a hole as `a*n + b` against a single sequence of integers.'''
    if witnessed is None or len(witnessed) != len(parameters) or len(parameters) < 2:
        return None

    # Any pair with distinct sizes determines the line -- not necessarily the first two. With
    # three demos at sizes (5, 5, 7) the first pair is degenerate while the concept is still
    # perfectly recoverable, and keying off index 0 and 1 threw those away.
    pair = next((((parameters[i], witnessed[i]), (parameters[j], witnessed[j]))
                 for i in range(len(parameters))
                 for j in range(i + 1, len(parameters))
                 if parameters[j] != parameters[i]), None)
    if pair is None:
        return None
    (n0, w0), (n1, w1) = pair
    numerator, denominator = w1 - w0, n1 - n0
    if numerator % denominator:
        return None                       # not an integer slope
    a = numerator // denominator
    b = w0 - a * n0
    if any(w != a * n + b for n, w in zip(parameters, witnessed)):
        return None                       # a later demo contradicts the relation

    if a == 0:
        return Const(b)                   # constant across demos; not the parameter at all
    expr: IntExpr = Param() if a == 1 else BinOp("*", Const(a), Param())
    if b > 0:
        expr = BinOp("+", expr, Const(b))
    elif b < 0:
        expr = BinOp("-", expr, Const(-b))
    return ir.simplify(expr)


def _all_constants(term: Term) -> List[int]:
    if isinstance(term, Move):
        return _constants(term.distance)
    if isinstance(term, Seq):
        return [c for s in term.steps for c in _all_constants(s)]
    if isinstance(term, Loop):
        return _constants(term.count) + _all_constants(term.body)
    if isinstance(term, Saved):
        return _all_constants(term.body)
    return []


def reproduces(candidate: Term, demos: Sequence[Term], parameters: Sequence[int]) -> bool:
    '''Does the parameterised program rebuild every demo when given that demo's size?

    The default gate. `ir.positions` is the same quantity every SPL metric compares, so this
    asks exactly the right question. The harness passes its configured evaluator instead, which
    additionally honours the continuous observation modes.
    '''
    try:
        return all(ir.positions(candidate, p) == ir.positions(demo, 0)
                   for demo, p in zip(demos, parameters))
    except Exception:  # noqa: BLE001 - a wrong generalisation can blow any budget
        return False


def recover(concept: str, solutions: Sequence[Term], parameters: Sequence[int], *,
            allow_single_demo: bool = True, accepts=None,
            holdout_validate: bool = True) -> Recovery:
    '''Recover one concept's parameterised class, or say why not.

    `accepts(term) -> bool` is the correctness gate; it defaults to `reproduces`, and the
    harness supplies one backed by its configured evaluator.

    At `k >= 3` the class is additionally rebuilt from `k-1` demos and required to reproduce
    the excluded one. A class that only fits the demos it was built from has not generalised,
    and this catches it without needing a single held-out size.
    '''
    solutions, parameters = list(solutions), list(parameters)
    gate = accepts or (lambda t: reproduces(t, solutions, parameters))

    cands, route, reason = candidates(solutions, parameters,
                                      allow_single_demo=allow_single_demo)
    if not cands:
        return Recovery(concept, reason=reason, parameters=tuple(parameters))

    survivors = [c for c in cands if gate(c)]
    if not survivors:
        return Recovery(concept, route=route, reason="no_candidate",
                        parameters=tuple(parameters))

    # Several readings reproducing every demo means the demos cannot distinguish them -- a
    # multi-argument concept demonstrated only at square sizes cannot say which argument a hole
    # tracks. Registering either would be a coin flip presented as a result, and the class would
    # be wrong on the first non-square instance. `distinct_params` demo selection is what
    # normally prevents this; when it cannot, say so.
    distinct = {repr(c) for c in survivors}
    if len(distinct) > 1:
        return Recovery(concept, route=route, reason="ambiguous",
                        parameters=tuple(parameters))
    winner = survivors[0]

    if holdout_validate and len(solutions) >= 3:
        for i in range(len(solutions)):
            kept_p = [p for j, p in enumerate(parameters) if j != i]
            # Skip folds the retained demos cannot possibly answer. Leaving (5, 5) behind
            # determines no relation at all, so a failure there says nothing about the
            # concept -- counting it rejected 9 recoverable concepts at scale.
            if len(set(kept_p)) < 2:
                continue
            kept = [s for j, s in enumerate(solutions) if j != i]
            sub, _, _ = candidates(kept, kept_p, allow_single_demo=False)
            if not sub:
                return Recovery(concept, route=route, reason="failed_holdout",
                                parameters=tuple(parameters))
            # More than one reading means the RETAINED demos are under-determined, so this fold
            # cannot answer anything. That is routine with several arguments: two demos give two
            # equations for a hole's two unknowns, so an affine function of one argument mimics
            # another exactly -- measured, `3*length - 7` reproduces `breadth` on
            # [(3,2), (4,5)]. Failing the concept for that would reject every multi-argument
            # concept demonstrated three times.
            if len(sub) > 1:
                continue
            if not reproduces(sub[0], [solutions[i]], [parameters[i]]):
                return Recovery(concept, route=route, reason="failed_holdout",
                                parameters=tuple(parameters))

    return Recovery(concept, term=winner, route=route, parameters=tuple(parameters))
