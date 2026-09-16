'''
test_stitch_bridge.py

Does STITCH compression actually buy what the whole baseline depends on?

Enumeration cost grows as ~e^(0.79 x MDL), so an abstraction is the only lever that moves a
concept between bands. Three checks:

  1. compression returns abstractions that parse and type-check,
  2. rewritten programs still produce identical traces (compression is meaning-preserving),
  3. the MDL of a *deeper* concept falls once the shallower ones are abstracted -- the
     library-growth result in miniature.

Run: python -m baseline_spl.tests.test_stitch_bridge
'''

from __future__ import annotations

from baseline_spl.symbolic._dreamcoder import Program
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, from_term, grammar, to_term
from baseline_spl.symbolic.ir import positions
from baseline_spl.symbolic.lattice import EMPTY
from baseline_spl.symbolic.oracles import ORACLES
from baseline_spl.symbolic.stitch_bridge import (best_compression, compress, extend,
                                                 reweight)

# The concepts a real run would have solved first: all four lines, which share the
# "place, then n-1 times (shift d, place)" skeleton STITCH should lift out.
SOLVED = ["row", "column", "tower", "inverted_row", "inverted_column"]
# The deeper concept whose cost we want to see fall.
DEEPER = ["staircase", "inverted_staircase", "pyramid"]

PARAMS = (1, 3, 5, 7)


def solved_programs():
    return [(c, from_term(ORACLES[c])) for c in SOLVED]


def trace_of(program: Program, n: int):
    try:
        return program.evaluate([])(n)(EMPTY).positions
    except Exception:  # noqa: BLE001
        return None


def run():
    base = grammar("standard")
    solved = solved_programs()
    result = compress(base, solved, iterations=8, max_arity=3)

    problems = []
    if result.error:
        problems.append(f"compression failed: {result.error}")
        return problems, None, {}
    if not result.abstractions:
        problems.append("compression returned no abstractions")
        return problems, None, {}

    # 2. rewriting must preserve behaviour.
    for task, source in result.rewritten.items():
        try:
            rewritten = Program.parse(source)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{task}: rewritten program does not parse: {exc}")
            continue
        for n in PARAMS:
            expected = positions(ORACLES[task], n)
            actual = trace_of(rewritten, n)
            if actual != expected:
                problems.append(f"{task}: rewritten program changed the trace at n={n}")
                break

    # 3. keep only the abstractions that pay for themselves, then re-weight.
    result, extended = best_compression(base, solved)
    extended = reweight(extended, solved)

    deltas = {}
    for concept in DEEPER + SOLVED:
        program = from_term(ORACLES[concept])
        before = base.logLikelihood(CONCEPT_REQUEST, program)
        after = extended.logLikelihood(CONCEPT_REQUEST, program)
        deltas[concept] = (-before, -after)

    return problems, result, deltas


# ----------------------------------------------------------------------------------- #

def test_compression_produces_valid_abstractions():
    problems, result, _ = run()
    assert not problems, "\n".join(problems)
    assert result.abstractions
    for abstraction in result.abstractions:
        abstraction.infer()


def _closed_corpus():
    '''The line concepts as CLOSED programs, as a demo-level run solves them.'''
    from baseline_spl.symbolic.bridge import DEMO_REQUEST, to_source
    # `_freeze` already substitutes the parameter everywhere it appears, including inside
    # arithmetic -- `row` is `loop (n-1)`, so a version that only handled a bare `Param` missed it.
    from baseline_spl.tests.test_generalise import _freeze as frozen

    corpus, requests = [], {}
    for i, concept in enumerate(("row", "column", "tower")):
        n = 3 + i
        name = f"{concept}_{n}"
        source = to_source(frozen(ORACLES[concept], n), concept_level=False)
        corpus.append((name, Program.parse(source)))
        requests[name] = DEMO_REQUEST
    return corpus, requests


def test_closed_programs_have_finite_mdl():
    '''Demo-level programs must be scored at their OWN type.

    Scored at the concept-level request a closed program's likelihood is -inf, so `corpus_mdl`
    returned inf for the baseline AND every candidate; `best_compression`'s `score < best_score`
    became `inf < inf`, and no abstraction was ever adopted. Both 12-hour demo-level runs
    learned nothing for this reason while logging only "mean solved MDL now inf".
    '''
    from baseline_spl.symbolic.bridge import DEMO_REQUEST
    from baseline_spl.symbolic.stitch_bridge import corpus_mdl, program_mdl

    base = grammar("standard", int_literals_upto=13)
    corpus, requests = _closed_corpus()

    for name, program in corpus:
        wrong = program_mdl(base, program)                       # concept-level default
        right = program_mdl(base, program, DEMO_REQUEST)
        assert wrong == float("inf"), (
            f"{name}: the concept-level request no longer rejects a closed program, so this "
            f"test has stopped guarding anything")
        assert right != float("inf"), f"{name}: closed program still scores as unreachable"

    assert corpus_mdl(base, corpus, requests) != float("inf")
    assert corpus_mdl(base, corpus) == float("inf"), "the default must stay concept-level"


def test_reweighting_uses_each_task_own_request():
    '''With the wrong request every frontier is skipped and the grammar is returned unchanged.'''
    from baseline_spl.symbolic.stitch_bridge import reweight

    base = grammar("standard", int_literals_upto=13)
    corpus, requests = _closed_corpus()

    skipped = []
    unchanged = reweight(base, corpus, 1.0, log=skipped.append)
    assert unchanged is base, "without requests, closed programs cannot be re-weighted"

    rewritten = reweight(base, corpus, 1.0, log=None, requests=requests)
    assert rewritten is not base, "re-weighting a closed corpus produced no new grammar"


def main() -> int:
    problems, result, deltas = run()

    if result is not None:
        print(f"STITCH returned {len(result.abstractions)} abstraction(s) "
              f"from {len(SOLVED)} solved programs:\n")
        for abstraction in result.abstractions:
            name = result.names.get(str(abstraction), "?")
            print(f"  {name}  ::  {abstraction.infer()}")
            print(f"      {abstraction}")
        print()

    if deltas:
        print(f"{'concept':<22} {'MDL before':>11} {'MDL after':>11} {'change':>10}")
        print("-" * 58)
        for concept in DEEPER + SOLVED:
            before, after = deltas[concept]
            arrow = "cheaper" if after < before - 0.01 else (
                "dearer" if after > before + 0.01 else "same")
            print(f"{concept:<22} {before:>11.1f} {after:>11.1f} "
                  f"{before - after:>+9.1f}  {arrow}")
        print("-" * 58)
        improved = [c for c in DEEPER if deltas[c][1] < deltas[c][0] - 0.01]
        print(f"\ndeeper concepts made cheaper: {improved or 'NONE'}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\ncompression is meaning-preserving and the library is well-formed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
