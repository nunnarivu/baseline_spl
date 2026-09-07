'''
test_ir_fidelity.py

Verification step 1: is the IR expressive enough to say what the ground truth says?

For every concept in SPLConfig.ALL_CONCEPTS and every parameter 1..12, the oracle IR program
must produce exactly the placement sequence `run_gt_program` produces from the ground-truth
PROGRAM_LIB definition. This is the gate for the whole baseline: if it fails, the grammar is
under-expressive and no amount of search will fix it.

Also checks the `minimal` grammar-level contract -- exactly the lines and diagonals are
expressible without `saved`.

Run: python -m baseline_spl.tests.test_ir_fidelity
'''

from __future__ import annotations

from SPL.config.spl_config import ALL_CONCEPTS
from SPL.utils.metrics import run_gt_program
from baseline_spl.symbolic.ir import Saved, uses
from baseline_spl.symbolic.oracles import MINIMAL_CONCEPTS, ORACLES, trace

MIN_PARAM = 1
MAX_PARAM = 12


def gt_positions(concept: str, n: int):
    '''Ground truth pops from a finite `objects` list, so size the pool from the oracle's own
    placement count plus headroom.'''
    pool = len(trace(concept, n)) + 8
    result = run_gt_program(f"{concept}({n}, objects)", pool)
    return None if result is None else result["positions"]


def check_concept(concept: str):
    '''Returns None if the oracle matches ground truth at every parameter, else why not.'''
    if concept not in ORACLES:
        return "no oracle IR program written"
    for n in range(MIN_PARAM, MAX_PARAM + 1):
        expected = gt_positions(concept, n)
        if expected is None:
            return f"ground truth failed to run at n={n}"
        actual = trace(concept, n)
        if actual != expected:
            head = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                        min(len(actual), len(expected)))
            return (f"differs at n={n}: {len(actual)} vs {len(expected)} placements, "
                    f"first mismatch at index {head}\n"
                    f"      gt   {expected[max(0, head - 2):head + 3]}\n"
                    f"      ours {actual[max(0, head - 2):head + 3]}")
    return None


def uses_saved(concept: str) -> bool:
    '''Structural: does this oracle term contain a Saved node?'''
    return uses(ORACLES[concept], Saved)


# ----------------------------------------------------------------------------------- #
# pytest entry points. The module also runs standalone -- see main().
# ----------------------------------------------------------------------------------- #

def test_oracle_coverage_is_reported(capsys):
    """Which concepts have a hand-written oracle, and which do not.

    An oracle is a *test fixture*, not part of the pipeline -- the search never reads one. Its
    job is to prove the IR can express a concept at all, which is why every oracle that exists
    is checked strictly below.

    This deliberately does NOT fail on an uncovered concept. The dataset grows (16 structures
    today, ~100 planned) and hand-writing an oracle per structure is not the intent; requiring
    one would turn every dataset addition into a broken test suite. What it does instead is
    report coverage, so nobody mistakes "the suite is green" for "the IR provably expresses
    every concept in the dataset".
    """
    covered = sorted(set(ALL_CONCEPTS) & set(ORACLES))
    missing = sorted(set(ALL_CONCEPTS) - set(ORACLES))
    with capsys.disabled():
        print(f"\n  oracle coverage: {len(covered)}/{len(ALL_CONCEPTS)} concepts")
        if missing:
            print(f"  NO oracle (expressiveness unproven): {missing}")
    assert covered, "no concept has an oracle; the IR fidelity proof is entirely absent"


def test_oracles_match_ground_truth():
    """Strict over every oracle that exists, whatever the dataset size."""
    checkable = sorted(set(ALL_CONCEPTS) & set(ORACLES))
    problems = {c: why for c in checkable if (why := check_concept(c))}
    assert not problems, "\n".join(f"{c}: {w}" for c, w in problems.items())


def test_minimal_level_contract():
    saved_free = {c for c in ALL_CONCEPTS if c in ORACLES and not uses_saved(c)}
    # Restricted to concepts we actually have an oracle for, so a dataset addition without
    # one cannot make this fail for a reason unrelated to the grammar-level contract.
    declared = set(MINIMAL_CONCEPTS) & set(ORACLES)
    assert saved_free == declared, (
        f"MINIMAL_CONCEPTS (restricted to concepts with an oracle) is {sorted(declared)} "
        f"but the saved-free concepts are {sorted(saved_free)}")


def main() -> int:
    print(f"{'concept':<28} {'n = 1..12':<10} {'placements @ n=5':>17}  minimal")
    print("-" * 72)

    failures = []
    for concept in sorted(set(ALL_CONCEPTS) & set(ORACLES)):
        why = check_concept(concept)
        if why:
            failures.append((concept, why))
            print(f"{concept:<28} {'FAIL':<10} {'-':>17}  -")
            continue
        minimal = "yes" if not uses_saved(concept) else "no"
        print(f"{concept:<28} {'ok':<10} {len(trace(concept, 5)):>17}  {minimal}")

    print("-" * 72)

    missing = sorted(set(ALL_CONCEPTS) - set(ORACLES))
    if missing:
        failures.append(("<coverage>", f"concepts with no oracle: {missing}"))

    # minimal-level contract: MINIMAL_CONCEPTS must be exactly the saved-free concepts.
    saved_free = {c for c in ALL_CONCEPTS if c in ORACLES and not uses_saved(c)}
    if saved_free != set(MINIMAL_CONCEPTS):
        failures.append(("<minimal level>",
                         f"MINIMAL_CONCEPTS is {sorted(MINIMAL_CONCEPTS)} "
                         f"but the saved-free concepts are {sorted(saved_free)}"))

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for concept, why in failures:
            print(f"  {concept}: {why}")
        return 1

    print(f"\nall {len(ALL_CONCEPTS)} concepts match ground truth for n = 1..12")
    print(f"minimal level covers {len(MINIMAL_CONCEPTS)}: {sorted(MINIMAL_CONCEPTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
