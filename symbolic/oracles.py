'''
oracles.py

Hand-written reference IR terms for every concept in SPLConfig.ORIGINAL_CONCEPTS.

Their purpose is to answer one question before any search is written: *is the IR expressive
enough?* `tests/test_ir_fidelity.py` runs each against `run_gt_program` on the same parameter
and requires identical placement sequences. A concept that cannot be written here means the
grammar is under-expressive and the primitives need fixing -- not the search.

They are also the upper bound the search is measured against, and the input to
`tests/test_lowering.py`, which lowers each one to a concept class and requires
`program_accuracy == 1.0`.

Note these are *symbolic* terms, not closures: the concept parameter is `Param()` and loop
indices are `LoopVar()`, so a single term covers every value of the parameter. That is what
makes them lowerable to a parametric class, and what the search is trying to rediscover.

Everything here uses the `standard` grammar level -- no conditionals. `pyramid` is the
interesting case: its ground truth guards the trailing focus moves with `if i != height - 1`,
but those moves happen after the last placement and every metric compares placement positions
only (metrics.py:495), so dropping the guard is observationally equivalent.
'''

from __future__ import annotations

from typing import Dict

from baseline_spl.symbolic.ir import (BinOp, Const, IntExpr, Loop, LoopVar, Param, Place,
                                      Saved, Seq, Shift, Term, positions, uses)

# Shorthands. `N` is the concept's parameter, `i` the innermost loop index.
N = Param()
i = LoopVar(0)


def add(a: IntExpr, b: IntExpr) -> IntExpr:
    return BinOp("+", a, b)


def sub(a: IntExpr, b: IntExpr) -> IntExpr:
    return BinOp("-", a, b)


def mul(a: IntExpr, b: IntExpr) -> IntExpr:
    return BinOp("*", a, b)


# --------------------------------------------------------------------------------------- #
# Building blocks, parameterised by an integer *expression* so they can be reused inside a
# loop with a count that depends on the index -- tower(i+1) in staircase, row(2n-1-2i) in
# pyramid, and so on.
# --------------------------------------------------------------------------------------- #

def line(direction: str, count: IntExpr) -> Term:
    '''place, then count-1 times (shift, place).'''
    return Seq(Place(), Loop(sub(count, Const(1)), Seq(Shift(direction), Place())))


def diagonal(first: str, second: str, count: IntExpr) -> Term:
    '''count times (place, shift, shift).'''
    return Loop(count, Seq(Place(), Shift(first), Shift(second)))


def staircase_of(direction: str, steps: IntExpr) -> Term:
    '''Towers of increasing height, stepping sideways between them.'''
    return Loop(steps, Seq(Saved(line("top", add(LoopVar(0), Const(1)))), Shift(direction)))


# --------------------------------------------------------------------------------------- #

def _build() -> Dict[str, Term]:
    row = line("right", N)
    inverted_row = line("left", N)
    column = line("front", N)
    inverted_column = line("behind", N)
    tower = line("top", N)

    diagonal_45 = diagonal("right", "behind", N)
    diagonal_135 = diagonal("left", "behind", N)
    diagonal_225 = diagonal("left", "front", N)
    diagonal_315 = diagonal("right", "front", N)

    staircase = staircase_of("right", N)
    inverted_staircase = staircase_of("left", N)

    # Rows of decreasing width stacked upward: width at iteration i is 2N-1-2i.
    pyramid = Loop(N, Seq(
        Saved(line("right", sub(sub(mul(Const(2), N), Const(1)), mul(Const(2), i)))),
        Shift("top"), Shift("right")))

    # Four diagonals from a shared centre, each returning to it before the next.
    x = Seq(Saved(diagonal("right", "behind", N)), Shift("left"),
            Saved(diagonal("left", "behind", N)), Shift("front"),
            Saved(diagonal("left", "front", N)), Shift("right"),
            diagonal("right", "front", N))

    arch_bridge = Seq(Saved(staircase_of("right", N)), Shift("left"),
                      staircase_of("left", N))

    # Straight-line composition -- no focus restore, so this one is reachable at `minimal`.
    # GT: column(N), right, row(N-1), behind, diagonal_135(N-2). line() places its first block
    # even for a count <= 0, as the GT row does, and diagonal() places nothing then, as GT does.
    isosceles_right_triangle = Seq(line("front", N), Shift("right"),
                                   line("right", sub(N, Const(1))), Shift("behind"),
                                   diagonal("left", "behind", sub(N, Const(2))))

    # Rows of decreasing length laid out back-to-front, with a trailing single block.
    pins = Seq(
        Loop(N, Seq(
            Saved(Seq(Place(),
                      Loop(sub(N, i), Seq(Shift("right"), Shift("right"), Place())))),
            Shift("behind"), Shift("right"))),
        Place())

    return {
        "row": row,
        "inverted_row": inverted_row,
        "column": column,
        "inverted_column": inverted_column,
        "tower": tower,
        "diagonal_45": diagonal_45,
        "diagonal_135": diagonal_135,
        "diagonal_225": diagonal_225,
        "diagonal_315": diagonal_315,
        "staircase": staircase,
        "inverted_staircase": inverted_staircase,
        "pyramid": pyramid,
        "x": x,
        "arch_bridge": arch_bridge,
        "isosceles_right_triangle": isosceles_right_triangle,
        "pins": pins,
    }


ORACLES: Dict[str, Term] = _build()

# Concepts expressible without `saved` -- the `minimal` grammar level. Derived from the terms
# rather than hand-listed so the two cannot drift apart.
MINIMAL_CONCEPTS = frozenset(name for name, term in ORACLES.items()
                             if not uses(term, Saved))


def trace(concept: str, n: int) -> list:
    '''Placement cells produced by the oracle for `concept` at parameter `n`.'''
    return positions(ORACLES[concept], n)
