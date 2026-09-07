'''
test_lowering.py

Verification step 2: does a lowered IR term score as correct through SPL's own metric path?

For every concept, lower the oracle term to a concept class, then require:

  1. it passes GeneralizeAgent._validate_class_code (the structural contract),
  2. class_signature reads back one int argument plus `objects`,
  3. its placement cells match the oracle's at every parameter,
  4. check_program_equivalence returns program_accuracy == 1.0 against the ground truth.

Passing this proves the whole metric path end to end before a single program is enumerated:
anything the search finds that is genuinely correct will score as correct.

Run: python -m baseline_spl.tests.test_lowering
'''

from __future__ import annotations

from SPL.config.spl_config import ALL_CONCEPTS
from SPL.utils.metrics import check_program_equivalence
from baseline_spl.common.codegen import class_signature, validate
from baseline_spl.symbolic.ir import positions
from baseline_spl.symbolic.lower import lower, lowered_positions, saved_is_exact
from baseline_spl.symbolic.oracles import ORACLES

# The ground truth names its parameter differently per concept; the real pipeline takes this
# from the shared sketch. Here we only need it to be *a* single int argument.
PARAM_NAMES = {
    "tower": "height", "pyramid": "height", "arch_bridge": "height",
    "staircase": "steps", "inverted_staircase": "steps",
}
MIN_PARAM, MAX_PARAM = 1, 8


def param_name(concept: str) -> str:
    return PARAM_NAMES.get(concept, "length")


def check_concept(concept: str):
    '''Returns None if the lowered oracle is correct, else why not.'''
    term = ORACLES[concept]
    name, param = concept, param_name(concept)
    code = lower(term, name, param)

    try:
        validate(code)
    except Exception as exc:  # noqa: BLE001
        return f"structural validation failed: {exc}"

    try:
        cls_name, arguments = class_signature(code)
    except Exception as exc:  # noqa: BLE001
        return f"class_signature failed: {exc}"
    if cls_name != name:
        return f"class name is {cls_name!r}, expected {name!r}"
    ints = [n for n, t in arguments.items() if t is int]
    if ints != [param]:
        return f"int arguments are {ints}, expected [{param!r}]"

    exact, why = saved_is_exact(term)
    if not exact:
        return f"saved is not exactly lowerable: {why}"

    for n in range(MIN_PARAM, MAX_PARAM + 1):
        expected = positions(term, n)
        actual = lowered_positions(term, name, param, n)
        if actual is None:
            return f"lowered class failed to execute at {param}={n}"
        if actual != expected:
            return (f"lowered class diverges from its term at {param}={n}: "
                    f"{len(actual)} vs {len(expected)} placements")

    definitions = {name: {"arguments": {param: int, "objects": list}, "code": code}}
    verdict = check_program_equivalence(definitions, name, concept,
                                        gt_program=f"{concept}(3, objects)")
    if verdict.get("program_accuracy") != 1.0:
        return (f"program_accuracy is {verdict.get('program_accuracy')} "
                f"({verdict.get('program_verdict')}: {verdict.get('program_reason')}"
                f"{', counterexample ' + str(verdict['program_counterexample']) if verdict.get('program_counterexample') else ''})")
    return None


# ----------------------------------------------------------------------------------- #

def test_lowered_oracles_are_equivalent_to_ground_truth():
    problems = {c: why for c in sorted(ALL_CONCEPTS) if (why := check_concept(c))}
    assert not problems, "\n".join(f"{c}: {w}" for c, w in problems.items())


def test_emitted_class_is_valid_python_for_every_concept():
    for concept in ALL_CONCEPTS:
        validate(lower(ORACLES[concept], concept, param_name(concept)))


def main() -> int:
    print(f"{'concept':<28} {'verdict':<10} notes")
    print("-" * 72)
    failures = []
    for concept in sorted(ALL_CONCEPTS):
        why = check_concept(concept)
        if why:
            failures.append((concept, why))
            print(f"{concept:<28} {'FAIL':<10}")
        else:
            print(f"{concept:<28} {'ok':<10} program_accuracy = 1.0")
    print("-" * 72)
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for concept, why in failures:
            print(f"  {concept}: {why}")
        return 1
    print(f"\nall {len(ALL_CONCEPTS)} lowered oracles proven equivalent to ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
