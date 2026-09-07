'''
test_bridge.py

The Term <-> DreamCoder Program translation, checked four ways on every oracle:

  1. `to_source` produces a parseable program that type-checks to the concept request type,
  2. `to_term(from_term(t))` gives back a term with the same traces (round trip),
  3. DreamCoder's own evaluator agrees with the Term evaluator (so the primitive closures and
     the ADT semantics cannot drift apart),
  4. the grammar assigns every oracle a finite log-likelihood, i.e. the search *could* in
     principle enumerate it -- an expressiveness check on the grammar rather than the ADT.

Run: python -m baseline_spl.tests.test_bridge
'''

from __future__ import annotations

from SPL.config.spl_config import ALL_CONCEPTS
from baseline_spl.symbolic.bridge import (CONCEPT_REQUEST, from_term, grammar, to_source,
                                          to_term)
from baseline_spl.symbolic.ir import positions, size
from baseline_spl.symbolic.oracles import ORACLES

PARAMS = range(1, 9)


def check_concept(concept: str, g):
    term = ORACLES[concept]

    try:
        source = to_source(term)
        program = from_term(term)
    except Exception as exc:  # noqa: BLE001
        return f"could not render to DreamCoder syntax: {exc}", None

    try:
        inferred = program.infer()
    except Exception as exc:  # noqa: BLE001
        return f"type inference failed: {exc}", None
    if not str(inferred) == str(CONCEPT_REQUEST):
        return f"inferred type {inferred}, expected {CONCEPT_REQUEST}", None

    try:
        round_tripped = to_term(program)
    except Exception as exc:  # noqa: BLE001
        return f"round trip failed: {exc}", None
    for n in PARAMS:
        if positions(round_tripped, n) != positions(term, n):
            return f"round trip changed the trace at n={n}", None

    # DreamCoder's evaluator vs. the Term evaluator.
    try:
        fn = program.evaluate([])
    except Exception as exc:  # noqa: BLE001
        return f"DreamCoder evaluation failed: {exc}", None
    from baseline_spl.symbolic.lattice import EMPTY
    for n in PARAMS:
        try:
            native = fn(n)(EMPTY).positions
        except Exception as exc:  # noqa: BLE001
            return f"DreamCoder evaluation raised at n={n}: {exc}", None
        if native != positions(term, n):
            return f"DreamCoder evaluator disagrees with the Term evaluator at n={n}", None

    likelihood = g.logLikelihood(CONCEPT_REQUEST, program)
    if likelihood is None or likelihood == float("-inf"):
        return "grammar assigns the oracle zero probability (it is unreachable)", None

    return None, (len(source), size(term), likelihood)


# ----------------------------------------------------------------------------------- #

def test_oracles_round_trip_and_are_reachable():
    g = grammar("standard")
    problems = {c: why for c in sorted(set(ALL_CONCEPTS) & set(ORACLES))
                if (why := check_concept(c, g)[0])}
    assert not problems, "\n".join(f"{c}: {w}" for c, w in problems.items())


def test_no_other_domain_leaked_into_primitive_globals():
    '''Primitive.GLOBALS is process-global. If dreamcoder's package __init__ ever runs, the
    tower domain registers "1".."49" and Program.parse silently resolves our integer literals
    to its primitives instead. Guard the isolation _dreamcoder.py buys.'''
    from baseline_spl.symbolic._dreamcoder import registered_primitive_names
    from baseline_spl.symbolic.bridge import PRIMITIVES

    registered = registered_primitive_names()
    unexpected = registered - set(PRIMITIVES)
    assert not unexpected, (
        f"{len(unexpected)} foreign primitives are registered, so another domain was "
        f"imported: {sorted(unexpected)[:12]}")


def main() -> int:
    g = grammar("standard")
    print(f"{'concept':<28} {'status':<8} {'chars':>6} {'nodes':>6} {'log P':>10}")
    print("-" * 66)
    failures = []
    for concept in sorted(set(ALL_CONCEPTS) & set(ORACLES)):
        why, stats = check_concept(concept, g)
        if why:
            failures.append((concept, why))
            print(f"{concept:<28} {'FAIL':<8}")
        else:
            chars, nodes, likelihood = stats
            print(f"{concept:<28} {'ok':<8} {chars:>6} {nodes:>6} {likelihood:>10.2f}")
    print("-" * 66)
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for concept, why in failures:
            print(f"  {concept}: {why}")
        return 1
    covered = sorted(set(ALL_CONCEPTS) & set(ORACLES))
    print(f"\nall {len(covered)} oracles round-trip and are reachable under the grammar")
    print("\nexample -- row:")
    print("  " + to_source(ORACLES["row"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
