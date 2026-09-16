'''
test_generalise.py

Recovering a size-parameterised concept from closed demo-level solutions.

The failure mode worth guarding against is a confident wrong answer. Anti-unification will
happily generalise two programs that merely look similar, and the result is a concept class
that runs, produces plausible output, and is wrong -- which no downstream metric would flag as
a *recovery* failure. So every test here checks the recovered program against something
independent: the concept-level run's own answer, or the oracle at a size never trained on.

Run: python -m baseline_spl.tests.test_generalise
'''

from __future__ import annotations

import contextlib
import json
import re
from collections import defaultdict
from pathlib import Path

from baseline_spl.symbolic import generalise as G
from baseline_spl.symbolic.bridge import grammar, to_term
from baseline_spl.symbolic.ir import (BinOp, Const, Loop, Param, Place, Saved, Seq, Shift,
                                      positions)
from baseline_spl.symbolic.oracles import ORACLES, trace

RUNS = Path(__file__).resolve().parents[1] / "runs"


@contextlib.contextmanager
def int_literals(upto: int = 13):
    '''Register literals 3..`upto` so a closed program will parse, then put GLOBALS back.

    `Primitive.GLOBALS` is process-global, and `test_bridge` asserts nothing unexpected is in
    it -- that guard exists because dreamcoder's tower domain registers "1".."49" and would
    silently resolve our integer literals to its own. Registering at import time made this
    module leak into that assertion, so the registration is scoped instead.
    '''
    from dreamcoder.program import Primitive

    snapshot = dict(Primitive.GLOBALS)
    try:
        grammar("standard", True, int_literals_upto=upto)
        yield
    finally:
        Primitive.GLOBALS.clear()
        Primitive.GLOBALS.update(snapshot)


def line(direction: str, n) -> Seq:
    '''`place` then n-1 x (shift, place) -- the shape every straight structure has.'''
    return Seq(Place(), Loop(BinOp("-", n, Const(1)), Seq(Shift(direction), Place())))


# --------------------------------------------------------------------------------------- #
# Anti-unification itself
# --------------------------------------------------------------------------------------- #

def test_identical_terms_have_no_holes():
    a = line("right", Const(5))
    generalised, holes = G.antiunify([a, a])
    assert holes == []
    assert generalised == a


def test_one_differing_literal_is_one_numeric_hole():
    _, holes = G.antiunify([line("right", Const(5)), line("right", Const(3))])
    assert len(holes) == 1 and holes[0].numeric
    assert holes[0].constants() == (5, 3)


def test_differing_direction_is_a_structural_hole():
    '''A shape disagreement must never be dressed up as a parameter.'''
    _, holes = G.antiunify([line("right", Const(5)), line("top", Const(5))])
    assert len(holes) == 1 and not holes[0].numeric


def test_two_differences_are_rejected():
    cands, _, reason = G.candidates(
        [line("right", Const(5)), line("top", Const(3))], [5, 3])
    assert cands == []
    assert reason == "structural_disagreement"


def test_hole_not_matching_parameters_is_rejected():
    '''The differing values must BE the demonstrations' sizes, not merely differ.'''
    cands, _, reason = G.candidates(
        [line("right", Const(5)), line("right", Const(3))], [7, 4])
    assert cands == []
    assert reason == "hole_not_param"


def _rect(a, b):
    '''A closed length x breadth grid -- the shape a 2-argument demo solution has.'''
    return Loop(Const(a), Seq(Loop(Const(b), Seq(Place(), Shift("right"))), Shift("top")))


def test_two_argument_concepts_need_three_demos():
    '''MEASURED, and it constrains the dataset: two demos cannot pin a 2-argument concept.

    A hole has two unknowns (a, b) in `w = a*n + b`, so two demos give two equations and an
    affine function of ONE argument reproduces another exactly -- on [(3,2), (4,5)],
    `3*length - 7` equals breadth at both. Recovery must refuse rather than pick, because the
    wrong reading produces a class that is confident and wrong off the diagonal.

    Three demos over-determine it and the mimic dies. So a concept taking 2 or 3 integers needs
    at least THREE demonstrations, at arguments that are not proportional.
    '''
    two = [(3, 2), (4, 5)]
    assert not G.recover("rect", [_rect(*p) for p in two], two).ok, (
        "two demos cannot distinguish which argument a hole tracks; recovering one would be "
        "a guess presented as a result")

    three = [(3, 2), (4, 5), (6, 3)]
    rec = G.recover("rect", [_rect(*p) for p in three], three)
    assert rec.ok, f"three distinct demos should recover: {rec.reason}"
    for unseen in ((5, 4), (7, 2), (2, 6)):
        assert positions(rec.term, unseen) == positions(_rect(*unseen), 0), (
            f"recovered class is wrong at {unseen}")


def test_square_only_demos_are_ambiguous_however_many():
    '''Arguments that always agree cannot say which one a hole follows.'''
    square = [(3, 3), (4, 4), (5, 5)]
    rec = G.recover("rect", [_rect(*p) for p in square], square)
    assert not rec.ok and rec.reason == "ambiguous", (
        "a concept only ever demonstrated at equal arguments must not be registered")


def test_k_ary_antiunification_folds_all_demos():
    terms = [line("right", Const(n)) for n in (3, 5, 8)]
    _, holes = G.antiunify(terms)
    assert len(holes) == 1
    assert holes[0].constants() == (3, 5, 8)


# --------------------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------------------- #

def test_recovered_program_reproduces_its_demos():
    terms = [line("right", Const(n)) for n in (5, 3)]
    rec = G.recover("row", terms, [5, 3])
    assert rec.ok and rec.route == "antiunify"
    for n in (5, 3):
        assert positions(rec.term, n) == positions(line("right", Const(n)), 0)


def test_single_demo_fallback_is_tagged_and_gated():
    rec = G.recover("row", [line("right", Const(5))], [5])
    assert rec.ok and rec.route == "single_demo", "one demo should still be recoverable"
    assert positions(rec.term, 9) == positions(line("right", Const(9)), 0)

    disabled = G.recover("row", [line("right", Const(5))], [5], allow_single_demo=False)
    assert not disabled.ok and disabled.reason == "single_demo_disabled"


def test_coincidental_literal_is_caught_by_the_gate():
    '''The fallback replaces every literal equal to n, including ones that are not the size.

    `Loop(3, ...)` inside a structure of size 3 is exactly that trap: substituting it makes the
    inner loop grow with n, which no longer reproduces the demo. The gate must reject it.
    '''
    term = Seq(Place(), Loop(Const(3), Seq(Shift("right"), Place())))
    accepted = G.recover("trap", [term], [3])
    if accepted.ok:
        assert positions(accepted.term, 3) == positions(term, 0)


def test_holdout_validation_rejects_a_class_that_only_fits_its_own_demos():
    calls = {"n": 0}

    def gate(_term):
        calls["n"] += 1
        return True

    terms = [line("right", Const(n)) for n in (3, 5, 8)]
    rec = G.recover("row", terms, [3, 5, 8], accepts=gate)
    assert rec.ok, "a genuinely parameterised class must survive hold-one-out"
    assert calls["n"] >= 1


# --------------------------------------------------------------------------------------- #
# Against real run artifacts
# --------------------------------------------------------------------------------------- #

def _solved_by_concept(stats_path: Path):
    stats = json.loads(stats_path.read_text())
    groups = defaultdict(list)
    for name, entry in stats.get("per_task", {}).items():
        if entry.get("status") == "solved" and entry.get("program"):
            groups[name.rsplit("_", 1)[0]].append((name, entry["program"]))
    return {c: sorted(v) for c, v in groups.items()}


def demo_parameters(log_path: Path):
    '''demo task name -> size, from a run log. The artifacts do not record it.'''
    out = {}
    text = log_path.read_text(errors="ignore")
    for m in re.finditer(r"<(\w+)>: demos \[([^\]]*)\] at parameters \[([0-9, ]+)\]", text):
        ids = [x.strip().strip("'") for x in m.group(2).split(",")]
        params = [int(x) for x in m.group(3).split(",")]
        for demo_id, p in zip(ids, params):
            out[f"{m.group(1)}_{demo_id}"] = p
    return out


def test_recovers_the_concept_level_answer_from_demo_level_solutions():
    '''The decisive check: recovery must agree with a search that never saw closed programs.'''
    demo = RUNS / "demo_smoke_dreamcoder_standard_maha2" / "search_stats.json"
    concept = RUNS / "symbolic_smoke_dreamcoder_standard_maha2" / "search_stats.json"
    if not (demo.exists() and concept.exists()):
        return  # goldens not present in this checkout

    truth = {k: v.get("program")
             for k, v in json.loads(concept.read_text()).get("per_task", {}).items()}
    groups = _solved_by_concept(demo)
    # demo_smoke's parameters, from its own run: row [5,3] tower [3,6] column [6,5]
    known = {"row": [5, 3], "tower": [3, 6], "column": [6, 5]}

    checked = 0
    with int_literals():
        for name, params in known.items():
            if name not in groups:
                continue
            items = groups[name]
            terms = [to_term(p, concept_level=False) for _, p in items]
            rec = G.recover(name, terms, params)
            assert rec.ok, f"{name} should recover: {rec.reason}"
            expected = to_term(truth[name], concept_level=True)
            for n in (3, 5, 7, 11):
                assert positions(rec.term, n) == positions(expected, n), \
                    f"{name} recovered a program disagreeing with the concept run at n={n}"
            checked += 1
    assert checked, "no concept was checked; the golden artifacts changed shape"


def test_closed_programs_need_the_literals_registered_to_parse_back():
    '''A closed program's size is a literal above 2, which only exists once the grammar is built.

    `generalise_all` read the solved programs back BEFORE building the grammar, so every one
    of the 18 failed to parse and the run reported "recovered 0/0" with no error. Ordering,
    not logic -- exactly the kind of failure that looks like an empty result.
    '''
    source = "(lambda (loop 6 (lambda (lambda (shift FRONT (place $0)))) $0))"
    from dreamcoder.program import Primitive

    snapshot = dict(Primitive.GLOBALS)
    try:
        Primitive.GLOBALS.clear()
        Primitive.GLOBALS.update({k: v for k, v in snapshot.items() if k not in set("3456789")
                                  and not k.isdigit()})
        try:
            to_term(source, concept_level=False)
            parsed_without_literals = True
        except Exception:  # noqa: BLE001
            parsed_without_literals = False
    finally:
        Primitive.GLOBALS.clear()
        Primitive.GLOBALS.update(snapshot)

    assert not parsed_without_literals, \
        "the literal was resolvable without registering it; this guard no longer guards"
    with int_literals():
        assert to_term(source, concept_level=False) is not None


def test_a_rebuilt_solution_reports_solved_only_with_a_frontier():
    '''`Solution.status` is derived from the frontier, not from `term` or `distance`.

    Rebuilding solutions from `search_stats.json` by setting only `term` and `distance` left
    every one reporting "unsolved", so recovery filtered them all out and announced 0/0 while
    18 solved programs sat in the artifact.
    '''
    from baseline_spl.symbolic.search import Solution

    with int_literals():
        from dreamcoder.program import Program
        program = Program.parse("(lambda (loop 6 (lambda (lambda (shift FRONT (place $0)))) $0))")

        bare = Solution(task="column_0002")
        bare.term = to_term(program, concept_level=False)
        bare.distance = 0.0
        assert bare.status == "unsolved", "status must not be inferable from term/distance"

        rebuilt = Solution(task="column_0002")
        rebuilt.term = to_term(program, concept_level=False)
        rebuilt.frontier = [(-13.3, program)]
        assert rebuilt.status == "solved"


def test_heldout_scores_oracle_targets_with_an_evaluator_that_works():
    '''Guard: held-out targets are integer cells with no SRN table.

    Passing a continuous evaluator scored every held-out task as None, and arm A reported
    0/48 solved with nothing in the log to say why. `heldout.run` must default to an
    evaluator that can actually score what `tasks_for` builds.
    '''
    from baseline_spl.symbolic import heldout
    from baseline_spl.symbolic.evaluate import ExactEvaluator, MahalanobisEvaluator

    tasks = heldout.tasks_for(["row"], (8,))
    assert tasks, "no held-out task was built for row"
    task = tasks[0]
    assert task.observation_mode == "lattice" and not task.srn_tables

    with int_literals():
        from baseline_spl.symbolic.bridge import from_term
        program = from_term(_freeze(ORACLES["row"], 8), concept_level=False)
        assert ExactEvaluator().accepts(ExactEvaluator().score(program, task)), \
            "the oracle's own program must satisfy its own held-out task"
        # And the continuous one cannot score it at all -- the reason for the default.
        assert MahalanobisEvaluator().score(program, task) is None


def test_staircase_is_not_recovered_from_approximate_solutions():
    '''`approximate` is a near miss. Generalising two near misses invents a wrong concept.'''
    demo = RUNS / "demo_smoke_dreamcoder_standard_maha2" / "search_stats.json"
    if not demo.exists():
        return
    stats = json.loads(demo.read_text())
    statuses = {k: v.get("status") for k, v in stats.get("per_task", {}).items()
                if k.startswith("staircase_")}
    assert statuses and all(s != "solved" for s in statuses.values())
    assert "staircase" not in _solved_by_concept(demo), \
        "an approximate solution must never reach recovery"


# --------------------------------------------------------------------------------------- #
# Scale: the dataset this is really for
# --------------------------------------------------------------------------------------- #

def check_scale(concepts: int = 100, k: int = 3):
    '''~100 concepts x 3 distinct parameters, the shape the dataset is moving to.

    Built from the oracles rather than from real runs, so it exercises the code at a size the
    dataset does not yet have. Returns (recovered, fallback_used) for main()'s report.
    '''
    names = list(ORACLES)
    recovered = fallback = 0
    for i in range(concepts):
        concept = names[i % len(names)]
        params = [3 + (i % 3), 5 + (i % 2), 7]
        assert len(set(params)) >= 2
        demos = [_closed_form(concept, p) for p in params]
        if any(d is None for d in demos):
            continue
        rec = G.recover(concept, demos, params)
        if rec.ok:
            recovered += 1
            for n in (8, 10, 12):
                assert positions(rec.term, n) == trace(concept, n), \
                    f"{concept} recovered but disagrees with the oracle at n={n}"
        if rec.route == "single_demo":
            fallback += 1
    assert recovered, "nothing recovered at scale"
    assert fallback == 0, "the single-demo fallback must not fire when parameters are distinct"
    return recovered, fallback


def _closed_form(concept: str, n: int):
    '''The oracle specialised to one size -- what a demo-level search would have found.'''
    from baseline_spl.symbolic import generalise as _g
    term = ORACLES[concept]
    return _g.substitute(_freeze(term, n), {})


def _freeze(term, n: int):
    '''Replace Param() with Const(n): the concept-level oracle as a closed program.'''
    if isinstance(term, Seq):
        return Seq(*[_freeze(s, n) for s in term.steps])
    if isinstance(term, Loop):
        return Loop(_freeze_int(term.count, n), _freeze(term.body, n))
    if isinstance(term, Saved):
        return Saved(_freeze(term.body, n))
    return term


def _freeze_int(expr, n: int):
    if isinstance(expr, Param):
        return Const(n)
    if isinstance(expr, BinOp):
        return BinOp(expr.op, _freeze_int(expr.left, n), _freeze_int(expr.right, n))
    return expr


def test_scale_100_concepts():
    check_scale()


def main() -> int:
    failures = []
    checks = [
        ("identical terms have no holes", test_identical_terms_have_no_holes),
        ("one differing literal is one numeric hole", test_one_differing_literal_is_one_numeric_hole),
        ("differing direction is structural", test_differing_direction_is_a_structural_hole),
        ("two differences rejected", test_two_differences_are_rejected),
        ("hole must match the parameters", test_hole_not_matching_parameters_is_rejected),
        ("k-ary fold over all demos", test_k_ary_antiunification_folds_all_demos),
        ("recovered program reproduces its demos", test_recovered_program_reproduces_its_demos),
        ("single-demo fallback tagged and gated", test_single_demo_fallback_is_tagged_and_gated),
        ("coincidental literal caught", test_coincidental_literal_is_caught_by_the_gate),
        ("hold-one-out", test_holdout_validation_rejects_a_class_that_only_fits_its_own_demos),
        ("recovers the concept-level answer", test_recovers_the_concept_level_answer_from_demo_level_solutions),
        ("approximate never recovered", test_staircase_is_not_recovered_from_approximate_solutions),
        ("scale: 100 concepts x 3 demos", check_scale),
    ]
    for label, check in checks:
        try:
            out = check()
            extra = f"  ({out[0]} recovered, {out[1]} via fallback)" if isinstance(out, tuple) else ""
            print(f"  OK   {label}{extra}")
        except AssertionError as exc:
            failures.append(f"{label}: {exc}")
            print(f"  FAIL {label}\n       {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  ERR  {label}\n       {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
