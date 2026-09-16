'''
heldout.py

Generalisation to sizes never seen in training, measured two ways on one set of tasks.

DreamCoder generalises by *searching again*. A held-out task is a new problem, enumerated
under the library learning left behind (`dreamcoder.py:567`):

    testFrontiers, _, testingTimes = enumerator(testingTasks,
                                                enumerationTimeout=testingTimeout, testing=True)

Nothing is instantiated from a stored concept; the abstractions simply make the new program
short enough to find. That is arm A, and it is the faithful baseline.

Arm B is the recovered class (`generalise.py`) applied at the new size. It costs no search at
all, but it only exists for concepts whose demos were solved, and it is more than published
DreamCoder does -- so it is reported separately and never folded into the B3-a number.

Ground truth comes from SPL's OWN program library rather than from data, so the held-out sizes
need no new demonstrations: `run_gt_program(f"{concept}(args, blocks)")` is the placement
sequence the concept means at those arguments, for every concept and every size. That is the
same function `plan_accuracy` and `program_accuracy` are computed against, so the target is the
one the metrics already agree on rather than a second opinion -- and it covers all 99 concepts,
where the 16 hand-written oracles covered a sixth of them.

A concept of k integers is tested at (n, n-1, n-2)[:k]: DISTINCT arguments, so a class that
swaps two of them cannot pass by accident, which a square (n, n) would allow. Fixed-size
concepts (15 of the 99) have no unseen size and are skipped, as is `wall`.

The one asymmetry to keep in view when reading results: arm A can only express a size that
exists as a literal in the grammar (ours reaches `_literal_ceiling`, DreamCoder's tower
hardcodes 1..49), while arm B takes the size as an argument and is free at any n. Keep the
held-out sizes inside the ceiling or the comparison measures the grammar, not the mechanism.
'''

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.symbolic import ir
from baseline_spl.symbolic.search import SearchTask


@dataclass
class HeldoutResult:
    '''One concept at one unseen size, under both arms.'''

    concept: str
    size: int
    search_solved: Optional[bool] = None      # None = arm not run
    search_program: Optional[str] = None
    search_seconds: float = 0.0
    class_solved: Optional[bool] = None
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {"concept": self.concept, "size": self.size,
                "search_solved": self.search_solved, "search_program": self.search_program,
                "search_seconds": round(self.search_seconds, 3),
                "class_solved": self.class_solved, "reason": self.reason}


def concept_arity(concept: str) -> Optional[int]:
    '''How many integers SPL's own ground-truth program takes, or None if it is not a concept
    the search can express (`wall`, whose argument is another concept).'''
    from nsei_simulator.dataset.spg.construct.templates.program_lib import PROGRAM_LIB

    entry = PROGRAM_LIB.get(concept)
    if entry is None:
        return None
    signature = entry[0]
    if any(t is not int and name != "objects" for name, t in signature):
        return None
    return sum(1 for _name, t in signature if t is int)


def ground_truth(concept: str, values: Sequence[int]) -> Optional[List]:
    '''The placement cells `concept(*values)` means, from SPL's OWN program library.

    Replaces the 16 hand-written oracles: those covered a sixth of the 99 concepts, so a
    held-out set built from them could never test the rest. `run_gt_program` is the same
    function SPL's `plan_accuracy` and `program_accuracy` are computed against, so the target
    is the one the metrics already agree on rather than a second opinion.
    '''
    from SPL.utils.metrics import run_gt_program

    call = ", ".join(str(int(v)) for v in values)
    program = f"blocks = filter()\n{concept}({call + ', ' if call else ''}blocks)"
    # Generous object pool: the ideal executor pops from it, and a structure at a held-out
    # size needs more blocks than any training demo did.
    result = run_gt_program(program, 4096)
    return None if result is None else result["positions"]


def tasks_for(concepts: Sequence[str], sizes: Sequence[int],
              param_names: Optional[Dict[str, str]] = None) -> List[SearchTask]:
    '''One CLOSED task per (concept, size) -- DreamCoder's own framing for a held-out problem.

    Closed, not parameterised, because that is what arm A is meant to reproduce: a held-out
    tower of 10 is its own problem, exactly as "arch leg 7" is separate from "arch leg 6".
    '''
    param_names = param_names or {}
    out: List[SearchTask] = []
    for concept in concepts:
        arity = concept_arity(concept)
        # None: inexpressible (`wall`). 0: a fixed-size concept has no unseen size to test, so
        # it is not part of a generalisation experiment at all.
        if not arity:
            continue
        for values in held_out_arguments(arity, sizes):
            target = ground_truth(concept, values)
            if not target:
                continue
            label = "x".join(str(v) for v in values)
            out.append(SearchTask(
                name=f"{concept}@{label}",
                # The value is carried for bookkeeping; a closed program never reads it.
                examples=[(values[0] if len(values) == 1 else values, target)],
                param_name=param_names.get(concept, "length"),
                observation_mode="lattice",
                closed=True))
    return out


def held_out_arguments(arity: int, sizes: Sequence[int]) -> List[Tuple[int, ...]]:
    '''Argument tuples to test a concept of `arity` integers at.

    The arguments within a tuple are made DISTINCT -- (n, n-1, n-2) -- so a class that swaps
    two of them cannot pass by accident, which a square (n, n) would let it do. They stay at or
    below `n` so every value remains inside the literal ceiling the search was given.
    '''
    out = []
    for n in sizes:
        values = tuple(max(1, n - i) for i in range(arity))
        if len(set(values)) == arity or arity == 1:
            out.append(values)
    return out


def evaluate_classes(recovered: Dict[str, "ir.Term"], concepts: Sequence[str],
                     sizes: Sequence[int]) -> Dict[str, bool]:
    '''Arm B: instantiate each recovered class at each unseen size, check against the oracle.

    No search, no timeout -- this is a function call. A concept with no recovered class simply
    has no entry, which is the honest record: arm B cannot answer for it.
    '''
    out: Dict[str, bool] = {}
    for concept in concepts:
        term = recovered.get(concept)
        arity = concept_arity(concept)
        if term is None or not arity:
            continue
        for values in held_out_arguments(arity, sizes):
            label = "x".join(str(v) for v in values)
            target = ground_truth(concept, values)
            if not target:
                continue
            argument = values[0] if len(values) == 1 else values
            try:
                out[f"{concept}@{label}"] = ir.positions(term, argument) == target
            except Exception:  # noqa: BLE001 - a wrong class can exceed any budget
                out[f"{concept}@{label}"] = False
    return out


def run(grammar, concepts: Sequence[str], sizes: Sequence[int], *,
        recovered: Optional[Dict[str, "ir.Term"]] = None,
        search: bool = True, timeout: float = 300.0, cpus: int = 1,
        evaluator=None, log=print) -> List[HeldoutResult]:
    '''Both arms over the same held-out set. Either can be switched off by its config knob.'''
    from baseline_spl.symbolic.evaluate import ExactEvaluator
    from baseline_spl.symbolic.search import wake

    # `tasks_for` targets are oracle-generated INTEGER CELLS, so exact matching is both the
    # right question and the strictest one. A continuous evaluator would need SRN tables these
    # tasks cannot have and would score every one as None -- silently reporting 0 solved.
    evaluator = evaluator or ExactEvaluator()

    tasks = tasks_for(concepts, sizes)
    if not tasks:
        log("  no held-out tasks could be built; are the oracles defined for these concepts?")
        return []

    by_name: Dict[str, HeldoutResult] = {
        t.name: HeldoutResult(concept=t.name.split("@")[0], size=int(t.name.split("@")[1]))
        for t in tasks}

    if search:
        # One PHASE budget for the whole held-out set, not a budget per task. That is
        # upstream's shape -- `enumerator(testingTasks, enumerationTimeout=testingTimeout)`
        # passes the same single figure it uses for training -- and it matters here because
        # with a global grammar `_units` yields ONE unit covering every task, so enumeration
        # is shared: each candidate program is tested against all of them at once.
        log(f"  arm A: enumerating {len(tasks)} held-out task(s), {timeout:g}s for the phase")
        started = time.time()
        stats = wake(grammar, tasks, timeout=timeout, cpus=cpus,
                     evaluator=evaluator, max_mdl=100.0)
        elapsed = time.time() - started
        for name, solution in stats.per_task.items():
            entry = by_name.get(name)
            if entry is None:
                continue
            entry.search_solved = solution.status == "solved"
            entry.search_program = solution.program and str(solution.program)
            entry.search_seconds = getattr(solution, "seconds", 0.0) or 0.0
        log(f"  arm A: {sum(1 for e in by_name.values() if e.search_solved)}"
            f"/{len(tasks)} solved in {elapsed:.0f}s")

    if recovered is not None:
        outcome = evaluate_classes(recovered, concepts, sizes)
        for name, ok in outcome.items():
            if name in by_name:
                by_name[name].class_solved = ok
        for name, entry in by_name.items():
            if entry.class_solved is None:
                entry.reason = "no recovered class for this concept"
        log(f"  arm B: {sum(1 for e in by_name.values() if e.class_solved)}"
            f"/{len(tasks)} reproduced by an instantiated class")

    return [by_name[t.name] for t in tasks]


def summarise(results: Sequence[HeldoutResult]) -> dict:
    '''Aggregate for `heldout_metrics.json`, per size and overall.'''
    per_size: Dict[int, Dict[str, int]] = {}
    for r in results:
        bucket = per_size.setdefault(r.size, {"tasks": 0, "search": 0, "klass": 0})
        bucket["tasks"] += 1
        bucket["search"] += 1 if r.search_solved else 0
        bucket["klass"] += 1 if r.class_solved else 0
    return {
        "per_size": {str(k): v for k, v in sorted(per_size.items())},
        "overall": {"tasks": len(results),
                    "search_solved": sum(1 for r in results if r.search_solved),
                    "class_solved": sum(1 for r in results if r.class_solved)},
        "per_task": [r.as_dict() for r in results],
    }
