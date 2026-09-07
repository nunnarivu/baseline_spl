'''
lower.py

Print an IR term as the Python concept class SPL's metrics score.

This is a deterministic printer, not a search and not an LLM call: every IR node has exactly
one Python rendering, so one term maps to exactly one class. Search happens in the small typed
IR (`ir.py`); this runs once, on the winning term.

The emitted class has to satisfy `GeneralizeAgent._validate_class_code` (SPL/model/generalize.py:375)
-- __init__, construct, actions, blocks, substructures, key_blocks, argument_sampler -- and to
accept an `objects` keyword, because `check_program_equivalence` calls
`<cls>(<int kwargs>, objects=objects)` (metrics.py:401). The shape follows SPL's own learned
classes in runs/cpt_learning/concept_library_program_lib.yml.

Focus save/restore is the one lossy step. The live executor (SPL/model/executor.py) has no
`get_focus`, so `Saved` is rendered as `assign_focus(object_id=<first block placed inside>)`.
That is exact whenever the body's first placement lands on the saved cell, which holds for
every ground-truth program; `saved_is_exact` checks it by execution rather than assuming it,
and the caller records `live_execution: unsupported` when it does not hold.
'''

from __future__ import annotations

import textwrap
from typing import List, Optional, Tuple

from baseline_spl.symbolic.ir import (Loop, Move, Place, Saved, Seq, Shift, Term,
                                      evaluate, simplify)
from baseline_spl.symbolic.lattice import EMPTY

INDENT = "    "

CLASS_TEMPLATE = '''class {name}:
    def __init__(self, {param}: int, objects: list):
        self.{param} = {param}
        self.objects = list(objects) if objects is not None else []
        self.constructed = False

        # Book keeping
        self._plan = []
        self._placed = []
        self._substructures = []

    def construct(self):
        if self.constructed:
            raise Exception("Construct method has already been called for this instance.")
        self.constructed = True

        {param} = self.{param}
{body}

    @staticmethod
    def argument_sampler():
        for _n in range(1, 1001):
            yield (_n, None)

    @property
    def blocks(self):
        return list(self._placed)

    @property
    def key_blocks(self):
        if not self._placed:
            return []
        return [self._placed[0], self._placed[-1]]

    @property
    def substructures(self):
        return list(self._substructures)

    @property
    def actions(self):
        return [f"assign_focus(object_id = {{bid}})" for bid in self.key_blocks]
'''


def _loop_name(depth: int) -> str:
    return f"_i{depth}"


def _anchor_name(depth: int) -> str:
    return f"_anchor{depth}"


def _emit(term: Term, param: str, loops: List[str], saved_depth: int) -> List[str]:
    '''Statement lines for a term, unindented. `loops` is outermost-first, matching the order
    IntExpr.render expects.'''
    if isinstance(term, Place):
        return ["_obj = self.objects.pop(0)",
                "place_object_at_focus(_obj)",
                "self._plan.append(f\"place_object_at_focus(object_id = {_obj})\")",
                "self._placed.append(_obj)"]

    if isinstance(term, Shift):
        d = term.direction.upper()
        return [f'shift_focus("{d}")',
                f"self._plan.append('shift_focus(\"{d}\")')"]

    if isinstance(term, Move):
        # SPL's executor has no multi-cell move, so `move` lowers to a loop of unit shifts.
        # The emitted plan is byte-identical to the equivalent chain of `shift`s: `move` is a
        # search-space convenience that lets one application cover a k-cell step, exactly as
        # DreamCoder's `left`/`right` do, not a new action in the robot's vocabulary.
        d = term.direction.upper()
        count = simplify(term.distance).render(param, loops)
        var = _loop_name(len(loops))
        return [f"for {var} in range({count}):",
                INDENT + f'shift_focus("{d}")',
                INDENT + f"self._plan.append('shift_focus(\"{d}\")')"]

    if isinstance(term, Seq):
        lines: List[str] = []
        for step in term.steps:
            lines.extend(_emit(step, param, loops, saved_depth))
        return lines or ["pass"]

    if isinstance(term, Loop):
        var = _loop_name(len(loops))
        count = simplify(term.count).render(param, loops)
        body = _emit(term.body, param, loops + [var], saved_depth)
        return [f"for {var} in range({count}):"] + [INDENT + line for line in body]

    if isinstance(term, Saved):
        anchor = _anchor_name(saved_depth)
        body = _emit(term.body, param, loops, saved_depth + 1)
        # The guard matters: a body that places nothing leaves no block to anchor on, so the
        # focus cannot be restored. Skipping is wrong but does not crash, and saved_is_exact
        # reports the term as not exactly lowerable.
        return ([f"{anchor} = len(self._placed)"] + body +
                [f"if len(self._placed) > {anchor}:",
                 INDENT + f"assign_focus(object_id = self._placed[{anchor}])",
                 INDENT + "self._plan.append("
                          f"f\"assign_focus(object_id = {{self._placed[{anchor}]}})\")"])

    raise TypeError(f"cannot lower {type(term).__name__}")


def lower(term: Term, name: str, param: str = "length") -> str:
    '''IR term -> concept class source.'''
    body = _emit(term, param, [], 0)
    indented = textwrap.indent("\n".join(body), INDENT * 2)
    return CLASS_TEMPLATE.format(name=name, param=param, body=indented)


def saved_is_exact(term: Term, params=range(1, 9)) -> Tuple[bool, Optional[str]]:
    '''Is every `Saved` in this term restorable via a placed block?

    Checks by execution rather than by assumption: for each parameter value, every Saved body
    must place at least one block, and its first placement must land on the cell the focus
    held when the body started. Returns (True, None) or (False, reason).
    '''
    def check(t: Term, state, param, loops):
        '''Evaluate t from `state`, verifying Saved nodes as we go. Returns the new state.'''
        if isinstance(t, Seq):
            for step in t.steps:
                state = check(step, state, param, loops)
            return state
        if isinstance(t, Loop):
            n = t.count.eval(param, loops)
            for k in range(max(0, n)):
                state = check(t.body, state, param, loops + (k,))
            return state
        if isinstance(t, Saved):
            before = state
            after = check(t.body, state, param, loops)
            placed = after.placements[len(before.placements):]
            if not placed:
                raise _Inexact("a saved body places no block, so the focus cannot be restored")
            if placed[0][1] != before.focus:
                raise _Inexact(f"a saved body's first placement is at {placed[0][1]} "
                               f"but the focus was at {before.focus}")
            return after.with_focus(before.focus)
        return t.evaluate(state, param, loops)

    for n in params:
        try:
            check(term, EMPTY, n, ())
        except _Inexact as exc:
            return False, f"{exc} (at parameter {n})"
        except Exception as exc:  # noqa: BLE001 - execution failure is a separate concern
            return False, f"term failed to execute at parameter {n}: {exc}"
    return True, None


class _Inexact(Exception):
    pass


def lowered_positions(term: Term, name: str, param_name: str, n: int) -> Optional[list]:
    '''Round-trip helper: lower the term, run the class on the ideal executor, return its
    placement cells. Mirrors exactly what check_program_equivalence does.'''
    from SPL.utils.metrics import run_predicted_program

    code = lower(term, name, param_name)
    definitions = {name: {"arguments": {param_name: int, "objects": list}, "code": code}}
    init = f"{name}_1 = {name}({param_name}={n}, objects=objects)\n{name}_1.construct()"
    pool = len(evaluate(term, n).positions) + 8
    result = run_predicted_program(definitions, init, pool)
    return None if result is None else result["positions"]
