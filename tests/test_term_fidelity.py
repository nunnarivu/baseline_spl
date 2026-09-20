'''
test_term_fidelity.py

Is the IR term a run produces actually the term its own program means, and does it mean what
SPL's ground truth means?

`test_ir_fidelity.py` answers a narrower question -- it checks the 16 HAND-WRITTEN oracle
terms in `symbolic/oracles.py`. That leaves 83 concepts with no term-level check at all, and
it is where a real defect hid: in `lilo_gptluna_concept_level_depthimages_run1`, 28 of 99
concepts were lowered, registered and scored from a term belonging to a program the run had
already discarded, every one scored `program_accuracy` 0.0, and nothing noticed, because
nothing compared a run's own term against its own program.

Nothing here is hand-written per concept. Ground truth comes from `PROGRAM_LIB` via
`heldout.ground_truth`, which already defines all 99 -- so a concept added to the dataset
tomorrow is covered the day it lands, with no oracle to write.

Three checks, over whatever terms exist:

  term    == ground truth      the term means what SPL says the concept means
  lowered == term              the emitted Python class means what the term means
  to_term(program) == term     the term belongs to the program it was recorded next to

Run: python -m baseline_spl.tests.test_term_fidelity
'''

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import pytest
import yaml

from SPL.config.spl_config import ALL_CONCEPTS
from baseline_spl.symbolic import heldout
from baseline_spl.symbolic.bridge import from_term, grammar, to_term
from baseline_spl.symbolic.ir import args as ir_args
from baseline_spl.symbolic.ir import positions
from baseline_spl.symbolic.lower import lowered_positions
from baseline_spl.symbolic.oracles import ORACLES

# Small enough to stay fast, wide enough that a program which memorised one demo size fails.
SIZES = (3, 4, 5, 7)
# Literal ceiling for re-parsing recorded programs. `SearchHarness._literal_ceiling` derives
# this per run; here it just has to be at least as large as any literal a run recorded.
LITERAL_CEILING = 60


def normalise(cells: Sequence) -> List[tuple]:
    '''Re-express placements with the first one at the origin.

    Ground truth counts from wherever the focus STARTED; a demo (and the IR) counts from the
    first block actually PLACED. Those differ by exactly the shifts a program performs before
    its first placement, so `rocket` and `tree` -- whose PROGRAM_LIB definitions open with
    `for i in range(2): shift_focus('right')` -- come out uniformly translated by (0, 2, 0)
    while being the same shape. Comparing raw positions reports both as wrong.

    A translation is not an error; a different shape is. Normalising keeps only the second.
    '''
    cells = [tuple(c) for c in cells]
    if not cells:
        return cells
    first = cells[0]
    return [tuple(c[i] - first[i] for i in range(3)) for c in cells]


def param_names(concept: str) -> tuple:
    '''The integer argument names, in the order PROGRAM_LIB declares them -- the same order
    `check_program_equivalence` pairs arguments by.'''
    from nsei_simulator.dataset.spg.construct.templates.program_lib import PROGRAM_LIB

    entry = PROGRAM_LIB.get(concept)
    if entry is None:
        return ()
    return tuple(name for name, t in entry[0] if t is int)


def check_term(concept: str, term, arity: int, sizes: Sequence[int] = SIZES) -> Optional[str]:
    '''None if `term` is a faithful IR form of `concept`, else what is wrong with it.'''
    names = param_names(concept)
    for size in sizes:
        values = tuple([size] * arity)
        expected = heldout.ground_truth(concept, values)
        if expected is None:
            return f"ground truth would not run at {values or '()'}"

        try:
            actual = positions(term, values if arity else ())
        except Exception as exc:  # noqa: BLE001
            return f"term would not evaluate at {values or '()'}: {type(exc).__name__}: {exc}"

        if normalise(actual) != normalise(expected):
            head = next((i for i, (a, b) in enumerate(zip(normalise(actual),
                                                          normalise(expected))) if a != b),
                        min(len(actual), len(expected)))
            return (f"term differs from ground truth at {values or '()'}: "
                    f"{len(actual)} vs {len(expected)} placements, first at index {head}\n"
                    f"      gt   {normalise(expected)[max(0, head - 2):head + 3]}\n"
                    f"      ours {normalise(actual)[max(0, head - 2):head + 3]}")

        # The class is what actually gets registered and scored, so an IR that is right while
        # its lowering is wrong would still produce a wrong result.
        try:
            lowered = lowered_positions(term, concept, names or "length",
                                        values if arity else ())
        except Exception as exc:  # noqa: BLE001
            return f"lowered class would not run at {values or '()'}: {type(exc).__name__}: {exc}"
        if lowered is None:
            return f"lowered class produced nothing at {values or '()'}"
        if normalise(lowered) != normalise(actual):
            return (f"lowered class disagrees with its own term at {values or '()'}: "
                    f"{len(lowered)} vs {len(actual)} placements")

    # Round trip through the lambda-calculus form the search actually manipulates.
    try:
        back = to_term(from_term(term, concept_level=arity), concept_level=arity)
    except Exception as exc:  # noqa: BLE001
        return f"term does not survive a round trip through the program form: {exc}"
    for size in sizes:
        values = tuple([size] * arity)
        if positions(back, values if arity else ()) != positions(term, values if arity else ()):
            return f"round trip through the program form changed the term at {values or '()'}"
    return None


# ----------------------------------------------------------------------------------- #
# The infrastructure guarantee: every concept in the dataset can be checked at all.
# ----------------------------------------------------------------------------------- #

def survey_ground_truth(concepts: Sequence[str] = None) -> dict:
    '''concept -> "inexpressible" | "ok" | a failure reason.'''
    out = {}
    for concept in sorted(concepts if concepts is not None else ALL_CONCEPTS):
        arity = heldout.concept_arity(concept)
        if arity is None:
            # `wall` takes another CONCEPT as its argument, which the grammar has no type
            # for. Recorded, not failed: it is a property of the concept, and a dataset that
            # adds more of them must not turn this test red.
            out[concept] = "inexpressible"
            continue
        cells = heldout.ground_truth(concept, tuple([3] * arity))
        out[concept] = "ok" if cells else "ground truth produced nothing at 3"
    return out


def test_ground_truth_is_available_for_every_concept():
    '''The checking infrastructure must cover the whole dataset, including whatever is added
    to it next -- otherwise a new concept silently gets no term-level check, which is the
    hole the 28-concept bug lived in.'''
    survey = survey_ground_truth()
    broken = {c: why for c, why in survey.items() if why not in ("ok", "inexpressible")}
    assert not broken, "\n".join(f"{c}: {why}" for c, why in broken.items())
    assert any(why == "ok" for why in survey.values()), "no concept could be checked at all"


def test_inexpressible_concepts_are_named_not_hidden():
    '''A concept the grammar cannot type is a recorded outcome, not an omission.'''
    survey = survey_ground_truth()
    inexpressible = sorted(c for c, why in survey.items() if why == "inexpressible")
    # `wall` is the current one. This asserts the mechanism reports rather than crashes; it
    # deliberately does not pin the list, which grows with the dataset.
    assert "wall" in inexpressible, f"expected `wall` to be inexpressible, got {inexpressible}"


# ----------------------------------------------------------------------------------- #
# The hand-written oracles, through the full pipeline (IR -> ground truth -> lowering).
# ----------------------------------------------------------------------------------- #

def test_oracle_terms_survive_the_whole_pipeline():
    '''`test_ir_fidelity` checks oracle IR against ground truth. This additionally checks the
    LOWERED class and the round trip through the program form, so a break anywhere between
    "term" and "registered Python class" is attributed rather than showing up as a bad score.'''
    grammar("standard", int_literals_upto=LITERAL_CEILING)
    problems = {}
    for concept in sorted(set(ALL_CONCEPTS) & set(ORACLES)):
        arity = heldout.concept_arity(concept)
        if arity is None:
            continue
        why = check_term(concept, ORACLES[concept], arity)
        if why:
            problems[concept] = why
    assert not problems, "\n".join(f"{c}: {why}" for c, why in problems.items())


# ----------------------------------------------------------------------------------- #
# The regression test for the bug: a run's term must belong to that run's program.
# ----------------------------------------------------------------------------------- #

def term_program_mismatches(run_dir: str) -> dict:
    '''concept -> why, for every record whose `term` is not what its `program` translates to.

    This is the whole of the 28-concept defect, in one comparison. `Solution.term` used to be
    a stored field written under an `if term is None` guard while `Solution.program` was
    derived from a mutable frontier, so an accepted LLM proposal replaced the program and
    left the earlier program's term behind.
    '''
    path = os.path.join(run_dir, "search_programs.yml")
    records = yaml.safe_load(open(path, encoding="utf-8")) or {}
    grammar("standard", int_literals_upto=LITERAL_CEILING)

    problems = {}
    for concept, record in records.items():
        program, term = record.get("program"), record.get("term")
        if not program or not term:
            continue
        arity = heldout.concept_arity(concept)
        try:
            derived = repr(to_term(program, concept_level=0 if arity is None else arity))
        except Exception as exc:  # noqa: BLE001
            problems[concept] = f"recorded program will not translate: {exc}"
            continue
        if derived != term:
            problems[concept] = (f"recorded term is not this program's term\n"
                                 f"      program -> {derived}\n"
                                 f"      recorded   {term}")
    return problems


@pytest.mark.parametrize("run_dir", [
    d for d in [os.path.join("baseline_spl", "runs", "lilo_run0_corrected")]
    if os.path.exists(os.path.join(d, "search_programs.yml"))
])
def test_recorded_terms_belong_to_their_programs(run_dir):
    '''Every term a run wrote down must be the term its own recorded program translates to.'''
    problems = term_program_mismatches(run_dir)
    assert not problems, (f"{len(problems)} concept(s) in {run_dir}:\n"
                          + "\n".join(f"{c}: {why}" for c, why in problems.items()))


# ----------------------------------------------------------------------------------- #

def main() -> int:
    grammar("standard", int_literals_upto=LITERAL_CEILING)
    survey = survey_ground_truth()
    ok = [c for c, w in survey.items() if w == "ok"]
    inexpressible = [c for c, w in survey.items() if w == "inexpressible"]
    broken = {c: w for c, w in survey.items() if w not in ("ok", "inexpressible")}

    print(f"ground truth available : {len(ok)}/{len(survey)}")
    print(f"inexpressible          : {len(inexpressible)} ({', '.join(inexpressible) or 'none'})")
    for concept, why in broken.items():
        print(f"  BROKEN {concept}: {why}")

    print(f"\noracle terms through the full pipeline ({len(set(ALL_CONCEPTS) & set(ORACLES))}):")
    for concept in sorted(set(ALL_CONCEPTS) & set(ORACLES)):
        arity = heldout.concept_arity(concept)
        if arity is None:
            continue
        why = check_term(concept, ORACLES[concept], arity)
        print(f"  {concept:28s} {'ok' if why is None else why}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
