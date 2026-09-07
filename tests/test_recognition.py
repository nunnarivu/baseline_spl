'''
test_recognition.py

Sleep-R: the recognition model and the dreaming that trains it.

The failure mode worth guarding against is silence. `train_recognizer` deliberately does not
let a training failure end a run, so a recognizer that never trains looks exactly like one
that trains and does not help. These checks make the difference observable:

  1. the task encoding is well-formed -- every token is in the lexicon, and the encoding is
     translation-invariant (the same structure built elsewhere encodes identically),
  2. tasks carry the `(parameter, state) -> state` example shape the inherited `taskOfProgram`
     needs to build dream tasks; get this wrong and dreaming silently yields nothing,
  3. dreaming actually produces non-degenerate tasks,
  4. training returns a model, and `grammarOfTask` gives a well-formed grammar whose
     productions match the library it was built from.

Run: python -m baseline_spl.tests.test_recognition
'''

from __future__ import annotations

from baseline_spl.symbolic import recognition
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, from_term, grammar
from baseline_spl.symbolic.ir import positions
from baseline_spl.symbolic.lattice import EMPTY, LatticeState
from baseline_spl.symbolic.oracles import ORACLES
from baseline_spl.symbolic.search import SearchTask, Solution

CONCEPTS = ["row", "column", "tower", "staircase"]


def make_tasks():
    return [SearchTask(name=c, examples=[(3, positions(ORACLES[c], 3)),
                                         (5, positions(ORACLES[c], 5))])
            for c in CONCEPTS]


def solved_solutions(tasks):
    '''Stand in for a wake phase: every task solved by its oracle.'''
    out = {}
    for task in tasks:
        solution = Solution(task=task.name)
        program = from_term(ORACLES[task.name])
        g = grammar("standard")
        solution.add_exact(g.logLikelihood(CONCEPT_REQUEST, program), program, 5)
        out[task.name] = solution
    return out


# ----------------------------------------------------------------------------------- #

def test_encoding_is_inside_the_lexicon():
    vocabulary = set(recognition.lexicon())
    for concept in CONCEPTS:
        for n in (1, 2, 3, 5, 8):
            cells = positions(ORACLES[concept], n)
            state = LatticeState(placements=tuple((i, tuple(c)) for i, c in enumerate(cells)),
                                 cursor=len(cells))
            tokens = recognition.encode(n, state)
            assert tokens, f"{concept}({n}) encoded to nothing"
            unknown = [t for t in tokens if t not in vocabulary]
            assert not unknown, f"{concept}({n}) emitted tokens outside the lexicon: {unknown}"


def test_encoding_is_translation_invariant():
    '''The same structure built somewhere else is the same task.'''
    cells = positions(ORACLES["row"], 5)
    shifted = [(x + 3, y - 2, z + 1) for x, y, z in cells]

    def state_of(cs):
        return LatticeState(placements=tuple((i, tuple(c)) for i, c in enumerate(cs)),
                            cursor=len(cs))

    assert recognition.encode(5, state_of(cells)) == recognition.encode(5, state_of(shifted))


def test_task_examples_have_the_shape_dreaming_needs():
    tasks = recognition.build_tasks(make_tasks())
    for name, task in tasks.items():
        assert str(task.request) == str(CONCEPT_REQUEST)
        assert task.examples, f"{name} has no examples"
        for xs, y in task.examples:
            assert isinstance(xs, tuple) and len(xs) == 2, \
                f"{name}: inputs must be (parameter, state), got {xs!r}"
            assert isinstance(xs[1], LatticeState) and isinstance(y, LatticeState)
        # This is what RecurrentFeatureExtractor indexes when sampling dream inputs.
        args = task.request.functionArguments()
        assert len(args) == 2, f"expected two argument types, got {args}"


def check_extractor_builds_and_dreams():
    '''Returns (dreams, distinct outputs) for main()'s report. The test wrapper below is what
    pytest collects -- a `test_` function that returns a value is a pytest warning, and the
    warning is worth avoiding so a real one is not lost in the noise.'''
    tasks = list(recognition.build_tasks(make_tasks()).values())
    extractor = recognition.make_extractor(tasks)

    assert extractor.outputDimensionality > 0
    for task in tasks:
        assert extractor.tokenize(task.examples) is not None, \
            f"{task.name} was rejected by tokenize"

    g = grammar("standard")
    dreamed = []
    for _ in range(60):
        program = g.sample(CONCEPT_REQUEST, maximumDepth=6)
        if program is None:
            continue
        task = extractor.taskOfProgram(program, CONCEPT_REQUEST)
        if task is not None and task.examples:
            dreamed.append(task)
    assert dreamed, "dreaming produced no tasks at all"
    outputs = {tuple(y.positions) for t in dreamed for _xs, y in t.examples}
    assert len(outputs) > 1, "every dream produced the same output; dreams are degenerate"
    return len(dreamed), len(outputs)


def test_extractor_builds_and_dreams():
    check_extractor_builds_and_dreams()


def check_training_yields_a_usable_per_task_grammar():
    tasks = make_tasks()
    dc_tasks = recognition.build_tasks(tasks)
    g = grammar("standard")
    recognizer = recognition.train_recognizer(
        g, dc_tasks, solved_solutions(tasks), epochs=1, helmholtz_ratio=0.5, log=print)
    assert recognizer is not None, "training returned nothing"

    grammars = recognition.grammars_for(recognizer, dc_tasks, g, log=print)
    assert set(grammars) == set(dc_tasks)
    base = {str(p) for _l, _t, p in g.productions}
    for name, conditioned in grammars.items():
        produced = {str(p) for _l, _t, p in conditioned.productions}
        assert produced == base, f"{name}: productions differ from the library"
    return grammars


def test_training_yields_a_usable_per_task_grammar():
    check_training_yields_a_usable_per_task_grammar()


def main() -> int:
    failures = []
    checks = [
        ("encoding stays inside the lexicon", test_encoding_is_inside_the_lexicon),
        ("encoding is translation-invariant", test_encoding_is_translation_invariant),
        ("tasks have the (parameter, state) shape dreaming needs",
         test_task_examples_have_the_shape_dreaming_needs),
        ("extractor builds and dreams non-degenerate tasks", check_extractor_builds_and_dreams),
        ("training yields usable per-task grammars",
         check_training_yields_a_usable_per_task_grammar),
    ]
    for label, check in checks:
        try:
            out = check()
            extra = ""
            if label.startswith("extractor") and isinstance(out, tuple):
                extra = f"  ({out[0]} dreams, {out[1]} distinct outputs)"
            print(f"  OK   {label}{extra}")
        except AssertionError as exc:
            failures.append(f"{label}: {exc}")
            print(f"  FAIL {label}\n       {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  ERROR {label}\n       {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nSleep-R: encoding, dreaming and per-task grammars all work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_every_concept_gets_a_distinct_encoding():
    '''The recognition model can only give a task its own grammar if it can tell that task
    apart from the others. It could not.

    `delta_token` named only the *dominant* axis of a step and dropped the rest, so a
    (+1,-1,0) move and a (+1,+1,0) move encoded identically -- `diagonal_45` and
    `diagonal_135` produced byte-identical token sequences, as did `diagonal_225` and
    `diagonal_315`. Sixteen concepts collapsed to fourteen distinct inputs, which upstream's
    own log reported as a "14-way auxiliary classification loss" without anyone noticing.
    '''
    from baseline_spl.symbolic import ir, oracles, recognition

    class _State:
        def __init__(self, positions): self.positions = positions

    encodings = {}
    for name, term in oracles.ORACLES.items():
        encodings[name] = tuple(recognition.encode(4, _State(ir.positions(term, 4))))

    collisions = {}
    for name, seq in encodings.items():
        collisions.setdefault(seq, []).append(name)
    clashing = [v for v in collisions.values() if len(v) > 1]
    assert not clashing, f"concepts share an encoding and cannot be told apart: {clashing}"
    assert len(set(encodings.values())) == len(oracles.ORACLES)


def test_encoding_stays_inside_the_fixed_lexicon():
    '''A token outside the lexicon makes a trained recognizer invalid, so the vocabulary has
    to cover every delta the domain can produce.'''
    from baseline_spl.symbolic import ir, oracles, recognition

    class _State:
        def __init__(self, positions): self.positions = positions

    lexicon = set(recognition.lexicon())
    for name, term in oracles.ORACLES.items():
        for n in (1, 3, 4, 7):
            for token in recognition.encode(n, _State(ir.positions(term, n))):
                assert token in lexicon, f"{name} at n={n} emitted {token!r}, not in lexicon"
