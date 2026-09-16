'''
test_search_parallel.py

Two things about the wake phase that are easy to get wrong and hard to notice.

**Why parallelism is across tasks, not MDL sub-bands.** Sub-bands are tempting:
`Grammar.enumeration(lowerBound=l, upperBound=u)` yields exactly the programs whose MDL lies
in (l, u], so contiguous slices partition a band's *output* perfectly -- the first test below
confirms that to the program. But partitioned output is not partitioned work: `lowerBound`
only filters, it does not prune the traversal, so every slice re-walks the whole tree. The
second test measures that directly. A 16-way sub-band split measured 3x *slower* than serial.

This is exactly the failure mode worth a regression test, because a plausible-looking
"optimisation" here still returns correct answers -- just far more slowly -- so nothing fails
and the cost hides in the run time.

**Frontiers.** A solved task should carry several programs (DreamCoder's `maximumFrontier`),
every one of them reproducing its examples.

Run: python -m baseline_spl.tests.test_search_parallel
'''

from __future__ import annotations

import time
from collections import Counter

from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, grammar
from baseline_spl.symbolic.ir import positions
from baseline_spl.symbolic.lattice import EMPTY
from baseline_spl.symbolic.oracles import ORACLES
from baseline_spl.symbolic.search import SearchTask, slices, wake

CONCEPTS = ["row", "column", "tower"]
BAND = (0.0, 13.5)


def make_tasks():
    return [SearchTask(name=c, examples=[(3, positions(ORACLES[c], 3)),
                                         (5, positions(ORACLES[c], 5))])
            for c in CONCEPTS]


def enumerate_band(g, lower, upper):
    from dreamcoder.type import Context
    return Counter(str(p) for _prior, _c, p in g.enumeration(
        Context.EMPTY, [], CONCEPT_REQUEST, maximumDepth=99,
        upperBound=upper, lowerBound=lower))


# ----------------------------------------------------------------------------------- #

def test_slices_partition_the_band_exactly():
    '''The output property that makes sub-bands look attractive. It is true.'''
    g = grammar("standard")
    whole = enumerate_band(g, *BAND)
    for n in (2, 4, 7):
        split = Counter()
        for lo, hi in slices(BAND[0], BAND[1], n):
            split += enumerate_band(g, lo, hi)
        assert split == whole, (
            f"{n} slices gave {sum(split.values())} programs vs {sum(whole.values())}; "
            f"missing {sum((whole - split).values())}, "
            f"duplicated {sum((split - whole).values())}")


def test_slices_abut_exactly():
    parts = slices(2.0, 8.0, 4)
    assert parts[0][0] == 2.0 and parts[-1][1] == 8.0
    for (_, end), (start, _) in zip(parts, parts[1:]):
        assert end == start, "slices must abut, or programs fall between them"


def test_lower_bound_does_not_prune():
    '''The work property that makes sub-bands useless. Guards against re-introducing them.'''
    g = grammar("standard")

    def timed(lo, hi):
        t0 = time.time()
        n = sum(1 for _ in enumerate_band(g, lo, hi).elements())
        return n, time.time() - t0

    whole_n, whole_t = timed(0.0, 13.5)
    top_n, top_t = timed(12.0, 13.5)
    assert top_n < whole_n, "the top slice should hold fewer programs than the whole band"
    assert top_t > 0.5 * whole_t, (
        f"the top slice holds {top_n / whole_n:.0%} of the programs but took only "
        f"{top_t / whole_t:.0%} of the time -- lowerBound now prunes, so sub-band "
        f"parallelism may be worth revisiting")


def test_frontier_holds_several_verified_programs():
    tasks = make_tasks()
    # The alternates for these concepts sit at MDL ~15.4, so the budget has to reach past
    # the first solution at 12.1 or every frontier trivially holds one program.
    stats = wake(grammar("standard"), tasks, timeout=100.0, cpus=1, maximum_frontier=5)
    by_name = {t.name: t for t in tasks}
    kept = 0
    for name, solution in stats.per_task.items():
        if not solution.exact:
            continue
        kept = max(kept, len(solution.frontier))
        assert len(solution.frontier) <= 5
        for program in solution.programs:
            fn = program.evaluate([])
            for param, target in by_name[name].examples:
                assert fn(param)(EMPTY).positions == target, \
                    f"{name}: a frontier program does not reproduce its examples"
        priors = [pr for pr, _p in solution.frontier]
        assert priors == sorted(priors, reverse=True), "frontier must be best-first"
    assert kept > 1, f"no task kept more than one program (best was {kept})"


def test_per_task_grammars_are_grouped():
    '''With a {task: grammar} map the wake splits into one unit per distinct grammar --
    the structure the recognition model needs.'''
    from baseline_spl.symbolic.search import _units
    tasks = make_tasks()
    shared = grammar("standard")
    assert len(_units(shared, tasks)) == 1, "one grammar should give one unit"

    per_task = {t.name: grammar("standard") for t in tasks}
    assert len(_units(per_task, tasks)) == len(tasks), \
        "distinct per-task grammars should give one unit each"


def main() -> int:
    failures = []
    checks = [
        ("slices partition the band's output exactly", test_slices_partition_the_band_exactly),
        ("slices abut", test_slices_abut_exactly),
        ("lowerBound does not prune (so sub-bands cannot win)", test_lower_bound_does_not_prune),
        ("per-task grammars group into one unit each", test_per_task_grammars_are_grouped),
        ("frontiers hold several verified programs", test_frontier_holds_several_verified_programs),
    ]
    for label, check in checks:
        try:
            check()
            print(f"  OK   {label}")
        except AssertionError as exc:
            failures.append(f"{label}: {exc}")
            print(f"  FAIL {label}\n       {exc}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nwake phase: partitioning understood, frontiers collected, units grouped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_every_unit_is_searched_when_units_outnumber_workers():
    '''With per-task grammars each task is its own work unit, and a pool of W workers runs
    ceil(N/W) waves. Each unit must get a share of the clock measured from when IT starts.

    This used to be wrong in a way that scaled invisibly: every unit carried the same
    ABSOLUTE deadline, so the first wave consumed the whole budget and every later unit broke
    out immediately, reporting "unsolved" without enumerating a single program. Correct while
    units <= workers -- which is why 16 concepts on 16 workers never showed it -- and silently
    catastrophic at 100 concepts, where most of the task set would never be looked at.
    '''
    import time

    from baseline_spl.symbolic import ir, oracles
    from baseline_spl.symbolic.bridge import grammar
    from baseline_spl.symbolic.search import SearchTask, wake

    names = ["row", "tower", "column", "inverted_row", "inverted_column", "diagonal_45"]
    tasks = [SearchTask(name=n,
                        examples=[(k, ir.positions(oracles.ORACLES[n], k)) for k in (3, 5)])
             for n in names]
    base = grammar("standard")
    # One grammar object per task => one work unit per task, as the recognizer produces.
    per_task = {t.name: grammar("standard") for t in tasks}

    started = time.time()
    stats = wake(per_task, tasks, timeout=12.0, max_mdl=14.0, cpus=2, log=None)
    assert time.time() - started < 60, "wake overran its timeout"

    # Every unit must have been given real work, whether or not it found a solution.
    assert stats.programs_enumerated > 0
    searched = [name for name, sol in stats.per_task.items()
                if sol.exact or sol.program is not None]
    assert len(searched) >= len(names) // 2, (
        f"only {len(searched)}/{len(names)} tasks produced any result with 2 workers; "
        f"later units were starved of time")


def test_enumerate_unit_finds_a_known_program():
    '''`_enumerate_unit` is what every worker actually runs, and until now no test called it.

    That gap is why its argument list was a hazard: it took a bare 9-tuple whose fields
    included both a duration (`slice_seconds`) and an absolute timestamp (`hard_deadline`).
    Swapping those two positionally makes every unit expire the moment it starts and report
    its tasks unsolved with nothing enumerated -- no exception, no failing test, just a run
    that quietly finds nothing.
    '''
    import time

    from baseline_spl.symbolic import ir, oracles
    from baseline_spl.symbolic.bridge import grammar
    from baseline_spl.symbolic.search import EnumerationJob, SearchTask, _enumerate_unit

    task = SearchTask(name="row",
                      examples=[(n, ir.positions(oracles.ORACLES["row"], n)) for n in (3, 5)])
    job = EnumerationJob(grammar=grammar("standard"), tasks=[task],
                         lower=0.0, upper=14.0,
                         slice_seconds=60.0, hard_deadline=time.time() + 60.0,
                         soft_frontier=True, evaluator=None, keep=5)

    count, hits, misses = _enumerate_unit(job)

    assert count > 0, "enumerated nothing at all"
    assert "row" in hits, f"row not solved in {count} programs; hits={list(hits)}"
    prior, source, distance = hits["row"][0]
    assert distance == 0.0 and "loop" in source


def test_enumerate_unit_respects_its_own_slice_not_the_hard_deadline():
    '''The slice is measured from when the unit starts; the hard deadline only caps it.

    With a generous hard deadline and a tiny slice the unit must stop almost immediately --
    that is what stops later waves being starved when units outnumber workers.
    '''
    import time

    from baseline_spl.symbolic import ir, oracles
    from baseline_spl.symbolic.bridge import grammar
    from baseline_spl.symbolic.search import EnumerationJob, SearchTask, _enumerate_unit

    task = SearchTask(name="staircase",
                      examples=[(n, ir.positions(oracles.ORACLES["staircase"], n))
                                for n in (3, 4)])
    job = EnumerationJob(grammar=grammar("standard"), tasks=[task],
                         lower=0.0, upper=99.0,          # a band it could never finish
                         slice_seconds=1.0, hard_deadline=time.time() + 600.0,
                         soft_frontier=True, evaluator=None, keep=5)

    started = time.time()
    _enumerate_unit(job)
    elapsed = time.time() - started
    assert elapsed < 30, (
        f"unit ran {elapsed:.0f}s on a 1s slice -- it is following the hard deadline instead, "
        f"which starves every later wave")
