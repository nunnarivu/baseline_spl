'''
test_lilo.py

B3-b's offline tests. Every one of these runs with zero API calls, which is the point: the
prompt, the gates, the alias substitution and the whole driver path are proven before a
single token is spent.

The DSL description and message format are ports of LILO's own functions, so the tests here
pin them to upstream's shape rather than to anything I chose. `test_dsl_description_is_lilo_shaped`
is the load-bearing one: an earlier version of this file was my own prompt engineering, and
the one rule I added measurably made results worse (typecheck rejections 13 -> 22), so the
guard exists to stop it creeping back.
'''

from __future__ import annotations

import pytest

from baseline_spl.symbolic._dreamcoder import Program
from baseline_spl.symbolic import driver
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, from_term, grammar, to_term
from baseline_spl.symbolic.ir import positions
from baseline_spl.symbolic.lattice import EMPTY
from baseline_spl.symbolic.lilo import prompts
from baseline_spl.symbolic.lilo.namer import LibraryNamer, _parse_reply
from baseline_spl.symbolic.lilo.proposer import (ACCEPTED, ETA_LONG, FREE_VARS, INFER, PARSE,
                                                 REQUEST, WRONG_TRACE, LLMProposer,
                                                 anonymous_names, extract_candidates,
                                                 substitute_aliases, to_debruijn)
from baseline_spl.symbolic.oracles import ORACLES
from baseline_spl.symbolic.search import SearchTask


def _task(concept: str, params=(3, 5)) -> SearchTask:
    return SearchTask(name=concept,
                      examples=[(n, positions(ORACLES[concept], n)) for n in params],
                      param_name="length",
                      instruction=f"Construct a {concept} of length {params[0]}")


# --------------------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------------------- #

def test_dsl_description_documents_only_the_live_grammar():
    '''`minimal` has no `saved` or arithmetic; the description must not advertise them.'''
    minimal = prompts.dsl_description(grammar("minimal"))
    standard = prompts.dsl_description(grammar("standard"))
    for absent in ("saved ::", "add ::"):
        assert absent not in minimal, f"minimal description advertises {absent!r}"
    assert "saved ::" in standard and "add ::" in standard
    assert "shift ::" in standard and "place ::" in standard


def test_dsl_description_is_lilo_shaped_and_nothing_more():
    '''The regression guard against re-adding my own prompt engineering.

    An earlier version of this file hand-wrote a preamble, prose per primitive, de Bruijn
    index rules and two invented worked examples. None of that is in LILO, and the one rule
    I added measurably made things worse (typecheck rejections 13 -> 22). This pins the
    description to upstream's shape so it cannot creep back.
    '''
    text = prompts.dsl_description(grammar("standard"))

    # Upstream's exact opening (gpt_base.py:290-292).
    assert text.startswith(
        "You are an expert programmer working in a language based on lambda calculus.\n"
        "Your goal is to write programs that accomplish the tasks specified by the user.\n")
    assert "Write programs using the available functions:" in text

    # None of my own material may return to the strict description.
    for banned in ("de Bruijn", "$1", "$2", "shifts the outer indices",
                   "picket_fence", "flag_pole", "Worked example", "Rules that decide"):
        assert banned not in text, f"{banned!r} is my writing, not LILO's"

    # The named-variable note is a disclosed deviation and must be opt-in only.
    assert "(lambda (n)" not in text, "the deviation leaked into the strict description"
    assert "(lambda (n)" in prompts.dsl_description(grammar("standard"),
                                                    named_variables=True)

    # Base primitives are `name :: type` only -- upstream gives them no description,
    # because only the library namer ever writes one (laps_grammar.py:228).
    assert "description:" not in text


def test_message_list_matches_upstream_shape():
    '''Port of `Prompt.to_message_list` (gpt_base.py:158).'''
    body = [("Construct a row of length 5", "(lambda (lambda (place $0)))")]
    messages = prompts.message_list(body, "Construct a staircase of steps 4")

    assert messages[0] == {"role": "user", "content": "Here are some example programs:"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("-- ")       # Haskell-style comment prefix
    assert messages[2]["role"] == "assistant"             # the program is the model's turn
    assert messages[-1]["role"] == "user"
    assert "staircase" in messages[-1]["content"]


def test_task_language_carries_the_demonstration():
    '''Our one disclosed adaptation: the domain supplies the demonstration as the task's
    language, because the instruction alone does not determine the geometry.'''
    task = _task("row")
    language = prompts.task_language(task)
    for param, cells in task.examples:
        assert f"length={param}" in language
        assert f"({cells[0][0]},{cells[0][1]},{cells[0][2]})" in language
    assert "\n" not in prompts._one_line(language), "upstream strips line separators"


# --------------------------------------------------------------------------------------- #
# Candidate extraction and alias substitution
# --------------------------------------------------------------------------------------- #

def test_one_completion_is_one_program():
    '''Upstream takes `choice["text"]` whole (`sample_generator.py:566`); it never splits a
    completion into several candidates. Diversity comes from `n_samples_per_query`.'''
    assert extract_candidates("(lambda (lambda (place $0)))") == \
        ["(lambda (lambda (place $0)))"]
    # Trailing commentary is dropped at the balance point, not treated as a second program.
    assert extract_candidates("(lambda (lambda (place $0)))  -- one block") == \
        ["(lambda (lambda (place $0)))"]


def test_a_pretty_printed_program_survives_intact():
    '''The regression guard for a silently destructive bug.

    An earlier line-based extractor shredded a program printed across several lines into
    one "candidate" per line. Measured on the first exact-LILO run: 72 candidates, 68
    rejected as `wrong_request_type` because they were fragments like `(shift TOP (place
    $0))` inferring to bare `tstate` -- while the model had written well-formed programs.
    The run reported 0/72 accepted, and none of that was a fact about LILO.
    '''
    pretty = ("(lambda\n"
              "  (lambda\n"
              "    (loop $1\n"
              "      (lambda (lambda (shift RIGHT (place $0))))\n"
              "      $0)))\n")
    assert extract_candidates(pretty) == [
        "(lambda (lambda (loop $1 (lambda (lambda (shift RIGHT (place $0)))) $0)))"]
    assert Program.parse(extract_candidates(pretty)[0]).infer() == CONCEPT_REQUEST


def test_extract_candidates_drops_unbalanced():
    assert extract_candidates("(lambda (place\n") == []


# --------------------------------------------------------------------------------------- #
# Named variables -- the one disclosed deviation from published LILO
# --------------------------------------------------------------------------------------- #

def test_to_debruijn_leaves_existing_programs_untouched():
    '''Backward compatibility is what makes this safe: the model may write either form, and
    strict-LILO output must survive the translation unchanged.'''
    for concept, term in ORACLES.items():
        source = str(from_term(term))
        assert to_debruijn(source) == source, f"{concept} was altered"


def test_to_debruijn_resolves_the_shape_that_defeated_the_model():
    '''`staircase` is the concept every strict-LILO candidate failed on, always at
    typecheck, always inside `saved`. Written with names it must convert, typecheck and
    reproduce the demonstration exactly.'''
    named = ("(lambda (n) (lambda (s) "
             "(loop n (lambda (i) (lambda (s2) "
             "(shift RIGHT (saved (lambda (s3) (loop (add i 1) (lambda (j) (lambda (s4) "
             "(shift TOP (place s4)))) s3)) s2)))) s)))")
    program = Program.parse(to_debruijn(named))
    assert program.infer() == CONCEPT_REQUEST
    for n in (1, 3, 4, 6):
        assert program.evaluate([])(n)(EMPTY).positions == positions(ORACLES["staircase"], n)


def test_to_debruijn_handles_mixed_and_shadowed_names():
    '''A named binder may enclose an already-de-Bruijn lambda, and an inner name may shadow
    an outer one. Both must resolve to the innermost binder.'''
    # The placeholder for an anonymous binder must still occupy a slot, or `$0` inside it
    # would resolve past its own lambda.
    mixed = to_debruijn("(lambda (n) (lambda (place $0)))")
    assert mixed == "(lambda (lambda (place $0)))"
    # Shadowing: the inner `x` wins.
    shadowed = to_debruijn("(lambda (x) (lambda (x) (place x)))")
    assert shadowed == "(lambda (lambda (place $0)))"


def test_named_variables_are_gated_exactly_like_de_bruijn():
    '''The translation happens before parsing, so all seven gates still apply -- a named
    program that builds the wrong thing must still be rejected at the trace check.'''
    good = ("(lambda (n) (lambda (s) "
            "(loop n (lambda (i) (lambda (s2) (shift RIGHT (place s2)))) s)))")
    wrong = ("(lambda (n) (lambda (s) "
             "(loop (add n 1) (lambda (i) (lambda (s2) (shift RIGHT (place s2)))) s)))")

    proposer = LLMProposer(_StubBackend([]), allow_named_variables=True,
                           log=lambda _m: None)
    task = _task("row")
    assert proposer._gate(good, task, grammar("standard"), {}) is not None
    assert proposer._gate(wrong, task, grammar("standard"), {}) is None
    assert proposer.rejections[WRONG_TRACE] == 1

    # With the deviation off, the same named program is simply unparseable -- which is what
    # keeps the strict-LILO run strict.
    strict = LLMProposer(_StubBackend([]), allow_named_variables=False,
                         log=lambda _m: None)
    assert strict._gate(good, task, grammar("standard"), {}) is None
    assert strict.rejections[PARSE] == 1


def test_substitute_aliases_prefers_the_longest_name():
    '''A short name must not eat a prefix of a longer one.'''
    aliases = {"line": "#(SHORT)", "line_of_blocks": "#(LONG)"}
    assert substitute_aliases("(line_of_blocks RIGHT $0)", aliases) == "(#(LONG) RIGHT $0)"
    assert substitute_aliases("(line RIGHT $0)", aliases) == "(#(SHORT) RIGHT $0)"
    # A primitive whose name merely contains an alias is untouched.
    assert substitute_aliases("(shift RIGHT $0)", aliases) == "(shift RIGHT $0)"


# The abstraction STITCH actually extracts from the solved lines: line(direction, n).
LINE_ABSTRACTION = ("#(lambda (lambda (lambda (loop (sub $1 1) "
                    "(lambda (lambda (place (shift $4 $0)))) (place $0)))))")


def test_alias_round_trip_parses_back_to_the_same_program():
    abstraction = Program.parse(LINE_ABSTRACTION)
    names = anonymous_names([abstraction])
    aliases = {"line_of_blocks": str(abstraction), names[str(abstraction)]: str(abstraction)}

    written = "(lambda (lambda (line_of_blocks RIGHT $1 $0)))"
    parsed = Program.parse(substitute_aliases(written, aliases))
    direct = Program.parse(f"(lambda (lambda ({LINE_ABSTRACTION} RIGHT $1 $0)))")
    assert str(parsed) == str(direct)
    assert parsed.infer() == CONCEPT_REQUEST
    # The alias is not a nicety: written out, that name stands for 80 characters of
    # anonymous body, which is what makes a grown library usable by a model at all.
    assert parsed.evaluate([])(3)(EMPTY).positions == positions(ORACLES["row"], 3)


# --------------------------------------------------------------------------------------- #
# The seven gates
# --------------------------------------------------------------------------------------- #

class _StubBackend:
    '''Returns canned completions. No network, no key, no cost.

    Implements both backend entry points the LILO code uses: `call_messages_sampled` for the
    proposer (LILO's n-samples-per-query) and `call_text` for the namer.
    '''

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self._tokens = 0

    def call_messages_sampled(self, messages, *, n=1, temperature=None, stop=None,
                              images=None,
                              max_tokens=None):
        self.prompts.append((messages[0]["content"], messages[-1]["content"]))
        self._tokens += 100
        return [self.replies.pop(0)] if self.replies else []

    def call_text(self, system, query, max_tokens=None):
        self.prompts.append((system, query))
        self._tokens += 100
        return self.replies.pop(0) if self.replies else ""

    def ledger(self):
        return {"prompt_tokens": self._tokens, "completion_tokens": self._tokens // 2,
                "num_llm_calls": len(self.prompts), "num_cache_hits": 0}


def _gate_one(source, task=None):
    proposer = LLMProposer(_StubBackend([]), log=lambda _m: None)
    task = task or _task("row")
    program = proposer._gate(source, task, grammar("standard"), {})
    return program, proposer.rejections


def test_gates_reject_each_failure_mode():
    row = str(from_term(ORACLES["row"]))
    cases = [
        ("(lambda (lambda (place $0)",                       PARSE),      # unbalanced
        ("(lambda (lambda (place RIGHT)))",                  INFER),      # tdir where tstate
        ("(lambda (place $0))",                              REQUEST),    # tstate -> tstate
        ("(lambda (lambda (saved $2 $0)))",                  REQUEST),    # free variable
        # Runs, well typed, uses its argument -- and builds the wrong thing. This is the
        # only gate that needs execution, and the only one an enumerated program also faces.
        ("(lambda (lambda (loop (add $1 1) "
         "(lambda (lambda (shift RIGHT (place $0)))) $0)))",  WRONG_TRACE),
    ]
    for source, expected in cases:
        program, rejections = _gate_one(source)
        assert program is None, f"{source!r} should have been rejected at {expected}"
        assert rejections[expected] == 1, \
            f"{source!r} rejected at {dict(rejections)}, expected {expected}"

    program, rejections = _gate_one(row)
    assert program is not None and rejections[ACCEPTED] == 1


def test_the_request_type_forces_the_program_to_use_its_argument():
    '''A stronger guarantee than the examples give, and it costs nothing.

    A program that ignores its integer argument infers `t0 -> tstate -> tstate`, not
    `tint -> tstate -> tstate`, so gate 3 rejects it before it is ever executed. Hard-coding
    the size is therefore not merely penalised by the examples -- it does not typecheck.
    This is structural work the concept-level request does and the demo-level framing (which
    bakes the parameter in as a literal) cannot.
    '''
    for hardcoded in ("(lambda (lambda (loop 2 (lambda (lambda "
                      "(shift RIGHT (place $0)))) $0)))",
                      "(lambda (lambda (place (shift RIGHT (place $0)))))"):
        program, rejections = _gate_one(hardcoded)
        assert program is None
        assert rejections[REQUEST] == 1, \
            f"a parameter-ignoring program reached {dict(rejections)}"


def test_gate_requires_every_example_not_just_one():
    '''Right at one parameter, wrong at another, must still fail.'''
    row_only_at_three = ("(lambda (lambda (loop (mul $1 $1) "
                         "(lambda (lambda (shift RIGHT (place $0)))) $0)))")
    task = SearchTask(name="row", examples=[(1, positions(ORACLES["row"], 1)),
                                            (5, positions(ORACLES["row"], 5))])
    program, rejections = _gate_one(row_only_at_three, task)
    assert program is None and rejections[WRONG_TRACE] == 1


# --------------------------------------------------------------------------------------- #
# The proposer end to end, still offline
# --------------------------------------------------------------------------------------- #

def test_proposer_accepts_a_good_reply_and_records_tokens():
    row = str(from_term(ORACLES["row"]))
    wrong = "(lambda (lambda (loop (add $1 1) (lambda (lambda (shift RIGHT (place $0)))) $0)))"
    backend = _StubBackend([row, wrong])
    proposer = LLMProposer(backend, queries_per_task=2, samples_per_query=1,
                           log=lambda _m: None)

    seeded = [("Construct a column of length 4", "(lambda (lambda (place $0)))"),
              ("Construct a tower of height 3", "(lambda (lambda (place $0)))")]
    out = proposer(grammar("standard"), [_task("row")], 0, {}, seeded)

    assert [str(p) for p in out["row"]] == [row]
    assert proposer.rejections[ACCEPTED] == 1
    assert proposer.rejections[WRONG_TRACE] == 1
    ledger = proposer.ledger_by_task["row"]
    assert ledger["prompt_tokens"] > 0, "token spend must be attributed to the task"


def test_proposer_is_idle_below_two_solved_tasks():
    '''LILO refuses to build a prompt from fewer than two solved tasks
    (`gpt_solver.py:366`, `sample_generator.py:402`), so its solver cannot run from an empty
    state -- it generalises from what search already found. Without this guard the first
    iteration sends a prompt LILO would never have sent, and pays for it.
    '''
    backend = _StubBackend([str(from_term(ORACLES["row"]))] * 8)
    proposer = LLMProposer(backend, log=lambda _m: None)

    for solved in ([], [("Construct a row of length 5", "(lambda (lambda (place $0)))")]):
        assert proposer(grammar("standard"), [_task("row")], 0, {}, solved) == {}
    assert backend.prompts == [], "no API call may be made below the threshold"

    two = [("Construct a row of length 5", "(lambda (lambda (place $0)))"),
           ("Construct a tower of height 3", "(lambda (lambda (place $0)))")]
    proposer(grammar("standard"), [_task("row")], 0, {}, two)
    assert backend.prompts, "with two solved tasks the proposer must run"


def test_proposer_survives_a_backend_that_raises():
    class _Broken(_StubBackend):
        def call_messages_sampled(self, messages, *, n=1, temperature=None, stop=None,
                                  images=None,
                              max_tokens=None):
            raise RuntimeError("no api key")

    seeded = [("a", "(lambda (lambda (place $0)))"),
              ("b", "(lambda (lambda (place $0)))")]
    proposer = LLMProposer(_Broken([]), log=lambda _m: None)
    assert proposer(grammar("standard"), [_task("row")], 0, {}, seeded) == {}


def test_driver_solves_by_proposal_without_enumerating():
    '''The LLM-only ablation: timeout=0, so anything solved is the proposer's.'''
    tasks = [_task("row"), _task("tower")]
    sources = {t.name: from_term(ORACLES[t.name]) for t in tasks}

    def propose(g, pending, iteration, docs, solved):
        return {t.name: [sources[t.name]] for t in pending}

    result = driver.run(tasks, iterations=1, timeout=0.0, use_recognition=False,
                        use_library=False, propose=propose, log=lambda _m: None)

    assert result.solved == ["row", "tower"]
    assert result.solved_by == {"row": "llm", "tower": "llm"}
    assert result.reports[0].programs_enumerated == 0, "enumeration should not have run"
    assert set(result.reports[0].proposed_by_llm) == {"row", "tower"}
    for name in ("row", "tower"):
        assert result.solutions[name].term is not None, "must be lowerable to Python"


def test_driver_ignores_a_proposal_that_does_not_solve_the_task():
    '''Nothing is accepted because a model produced it.'''
    tasks = [_task("row")]
    wrong = Program.parse("(lambda (lambda (place $0)))")

    result = driver.run(tasks, iterations=1, timeout=0.0, use_recognition=False,
                        use_library=False,
                        propose=lambda g, p, i, d, s: {"row": [wrong]},
                        log=lambda _m: None)

    assert result.solved == [] and result.approximate == ["row"]
    assert result.solved_by == {}
    # A near-miss must still be lowerable, exactly as an enumerated near-miss is -- otherwise
    # the LLM's approximate results would be reported as untranslatable rather than scored.
    assert result.solutions["row"].term is not None


def test_b3a_is_unchanged_when_no_hooks_are_passed():
    '''The control must stay a control: with no hooks the driver never mentions an LLM.'''
    result = driver.run([_task("row")], iterations=1, timeout=20.0, use_recognition=False,
                        use_library=False, log=lambda _m: None)
    assert result.solved == ["row"]
    assert result.solved_by == {"row": "enumeration"}
    assert result.reports[0].proposed_by_llm == []
    assert result.reports[0].proposal_seconds == 0.0


# --------------------------------------------------------------------------------------- #
# Auto-documentation
# --------------------------------------------------------------------------------------- #

def test_namer_reply_parsing():
    assert _parse_reply('{"readable_name": "line_of_blocks"}')["readable_name"] == "line_of_blocks"
    fenced = '```json\n{"readable_name": "a_b", "description": "d"}\n```'
    assert _parse_reply(fenced)["description"] == "d"
    assert _parse_reply("no json here") is None
    assert _parse_reply("") is None


def test_namer_documents_and_names_uniquely():
    from baseline_spl.symbolic.lilo.namer import LibraryNamer

    body = LINE_ABSTRACTION
    abstraction = Program.parse(body)
    backend = _StubBackend(['{"readable_name": "line of blocks", "description": "a line"}'])
    namer = LibraryNamer(backend, log=lambda _m: None)

    docs = namer([abstraction], [("row", Program.parse(f"(lambda (lambda ({body} RIGHT $1 $0)))"))])
    assert docs[body]["readable_name"] == "line_of_blocks", "spaces must become underscores"
    assert docs[body]["description"] == "a line"
    # The usage example must reach the prompt, or naming is guesswork.
    assert "building a row" in backend.prompts[0][1]

    # A repeat of the same abstraction costs no second call.
    assert namer([abstraction], []) == {}
    assert namer.calls == 1


def test_namer_declining_is_not_fatal():
    from baseline_spl.symbolic.lilo.namer import LibraryNamer

    abstraction = Program.parse("#(lambda (place $0))")
    namer = LibraryNamer(_StubBackend(['{"readable_name": null}']), log=lambda _m: None)
    assert namer([abstraction], []) == {}
    assert namer.failures == 1


# --------------------------------------------------------------------------------------- #
# The three contributions, closing the loop
# --------------------------------------------------------------------------------------- #

def test_documentation_closes_the_loop():
    '''LILO's three contributions have to compose, not merely each work.

    Propose -> compress -> name -> and the name must come back round into the NEXT prompt,
    where a program written with it parses and runs. No unit test covers the seam, and the
    seam is where the bug was: once the library grows, `grammar.primitives` contains
    `Invented` objects with no `.name`, which threw inside prompt building. The driver
    catches proposer exceptions by design, so this silently disabled the proposer from
    iteration 1 onward -- visible only as a log line in a run that otherwise looked fine.

    Enumeration runs here rather than being switched off, because it has to: LILO's solver
    needs two solved tasks before it can build a prompt at all, so search seeds the loop and
    the LLM takes over from there. That ordering is the method, not a test artifact.
    '''
    solvable = ["row", "column", "tower"]
    # Held back so a task is still pending at iteration 1, forcing a second prompt.
    tasks = [_task(c) for c in solvable + ["staircase"]]
    answers = {c: str(from_term(ORACLES[c])) for c in solvable}

    class _Backend(_StubBackend):
        def call_messages_sampled(self, messages, *, n=1, temperature=None, stop=None,
                                  images=None,
                              max_tokens=None):
            system, target = messages[0]["content"], messages[-1]["content"]
            self.prompts.append((system, target))
            self._tokens += 100
            return [next((p for c, p in answers.items() if c in target), "")]

        def call_text(self, system, query, max_tokens=None):
            self.prompts.append((system, query))
            self._tokens += 100
            return ('{"readable_name": "line_of_blocks", '
                    '"description": "a straight line of n blocks"}')

    backend = _Backend([])
    proposer = LLMProposer(backend, queries_per_task=1, samples_per_query=3,
                           log=lambda _m: None)
    namer = LibraryNamer(backend, log=lambda _m: None)

    result = driver.run(tasks, iterations=2, timeout=30.0, use_recognition=False,
                        propose=proposer, document=namer, log=lambda _m: None)

    assert sorted(result.solved) == sorted(solvable)
    assert len(result.abstractions) == 1, "compression should have found line(dir, n)"

    body = str(result.abstractions[0])
    assert result.documentation[body]["readable_name"] == "line_of_blocks"

    # The name reached the next PROPOSAL prompt (naming calls are a different message).
    system = next(s for s, q in reversed(backend.prompts) if "readable_name" not in q)
    assert "line_of_blocks" in system, "the learned name never reached a prompt"
    assert "a straight line of n blocks" in system, "the description never reached a prompt"

    # ...and a program written with that name resolves and runs correctly.
    written = "(lambda (lambda (line_of_blocks RIGHT $1 $0)))"
    parsed = Program.parse(substitute_aliases(written, {"line_of_blocks": body}))
    assert parsed.evaluate([])(4)(EMPTY).positions == positions(ORACLES["row"], 4)


def test_prompt_building_survives_a_grown_library():
    '''The narrow regression guard for the bug above.'''
    from baseline_spl.symbolic.stitch_bridge import extend

    abstraction = Program.parse(LINE_ABSTRACTION)
    grown = extend(grammar("standard"), [abstraction])
    assert any(not hasattr(p, "name") for p in grown.primitives), \
        "vacuous unless the grammar really holds an Invented with no .name"

    text = prompts.dsl_description(grown, {}, {})
    assert "shift ::" in text, "the base primitives must still be documented"
    assert LINE_ABSTRACTION in text, "the abstraction must be offered to the model"


# --------------------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------------------- #

def test_ground_truth_never_reaches_a_prompt():
    '''The proposer may see programs the method itself found; it must never see
    `demo['program']`. Guarded because the two look alike at a glance.'''
    task = _task("staircase")
    messages = prompts.message_list(
        [("Construct a row of length 5", str(from_term(ORACLES["row"])))],
        prompts.task_language(task))
    text = prompts.dsl_description(grammar("standard")) + "".join(
        m["content"] for m in messages)

    for banned in ("get_focus", "save_focus", "def construct", "run_gt_program",
                   "place_object_at_focus"):
        assert banned not in text, f"{banned!r} leaked into the prompt"


# --------------------------------------------------------------------------------------- #
# Round 2: demonstration modality, and honest counting
# --------------------------------------------------------------------------------------- #

def _continuous_task(concept: str, params=(3, 5)) -> SearchTask:
    from baseline_spl.symbolic import srn

    table = srn.analytic_table((0.109, 0.05))
    scale = (0.109, 0.109, 0.05)
    examples = [(n, [tuple(c[i] * scale[i] for i in range(3))
                     for c in positions(ORACLES[concept], n)]) for n in params]
    return SearchTask(name=concept, examples=examples, param_name="length",
                      instruction=f"Construct a {concept} of length {params[0]}",
                      observation_mode="continuous", srn_tables=[table] * len(params))


def test_continuous_tasks_render_metres_the_way_the_llm_baselines_do():
    '''B3-b must read the demonstration in the same units CaP and Demo2Code read it at
    coordinate_mode='raw', or the comparison is between representations again.'''
    text = prompts.task_language(_continuous_task("row"))
    assert "(0.00,-0.11,0.00)" in text, text
    assert "(0,-1,0)" not in text, "integer cells must not appear in a continuous run"


def test_images_modality_removes_the_geometry_from_the_text():
    '''With keyframes attached, repeating the coordinates as text would hand B3-b strictly
    more than CaP-images or Demo2Code-vlm receives.'''
    task = _continuous_task("row")
    with_text = prompts.task_language(task, include_demonstration=True)
    without = prompts.task_language(task, include_demonstration=False)
    assert "->" in with_text and "0.11" in with_text
    assert "0.11" not in without and without == task.instruction


def test_images_reach_the_backend_only_in_an_image_modality():
    seen = {}

    class _Recording(_StubBackend):
        def call_messages_sampled(self, messages, *, n=1, temperature=None, stop=None,
                                  images=None, max_tokens=None):
            seen["images"] = images
            return super().call_messages_sampled(messages, n=n, temperature=temperature,
                                                 stop=stop, max_tokens=max_tokens)

    seeded = [("Construct a column of length 4", "(lambda (lambda (place $0)))"),
              ("Construct a tower of height 3", "(lambda (lambda (place $0)))")]
    row = str(from_term(ORACLES["row"]))

    for modality, expected in (("coords", None), ("images", [b"png"])):
        backend = _Recording([row] * 4)
        proposer = LLMProposer(backend, queries_per_task=1, samples_per_query=1,
                               demo_modality=modality, log=lambda _m: None)
        proposer.demo_frames["row"] = [b"png"]
        proposer(grammar("standard"), [_task("row")], 0, {}, seeded)
        assert seen["images"] == expected, modality


def test_the_delta_table_reaches_the_dsl_description():
    '''The stats block is what tells the model that RIGHT moves along -y. Without it the
    model must infer the axis mapping from the few-shot traces alone, which is what produced
    a structurally perfect staircase built in the wrong direction.'''
    block = "# shift_focus delta statistics\nRIGHT (0.0002, -0.1106, 0.0006)\n"
    text = prompts.dsl_description(grammar("standard"), stats_block=block)
    assert "-0.1106" in text
    assert "-0.1106" not in prompts.dsl_description(grammar("standard"))


def test_cached_queries_are_not_counted_as_api_calls_or_candidates():
    '''An earlier run reported 8 calls and 32 candidates when the truth was 3 calls and 12
    distinct candidates: a cached prompt returns identical completions, and each was re-gated
    and re-counted. Both populations must be reported apart.'''
    row = str(from_term(ORACLES["row"]))

    class _Cached(_StubBackend):
        '''Always returns the same completion and never reports an API call.'''

        def call_messages_sampled(self, messages, *, n=1, temperature=None, stop=None,
                                  images=None, max_tokens=None):
            self.prompts.append((messages[0]["content"], messages[-1]["content"]))
            return [row]

        def ledger(self):
            return {"prompt_tokens": 0, "completion_tokens": 0,
                    "num_llm_calls": 0, "num_cache_hits": len(self.prompts)}

    backend = _Cached([])
    proposer = LLMProposer(backend, queries_per_task=4, samples_per_query=1,
                           log=lambda _m: None)
    seeded = [("Construct a column of length 4", "(lambda (lambda (place $0)))"),
              ("Construct a tower of height 3", "(lambda (lambda (place $0)))")]
    proposer(grammar("standard"), [_task("row")], 0, {}, seeded)

    stats = proposer.stats()
    assert stats["queries_issued"] == 4
    assert stats["api_calls"] == 0 and stats["cache_hits"] == 4
    assert stats["llm_calls"] == 0, "llm_calls must report real API calls, not queries"
    # One distinct source string, gated once; the other three were duplicates.
    assert stats["candidates_examined"] == 1
    assert stats["duplicate_candidates_skipped"] == 3


def test_comments_are_stripped_before_lines_are_joined():
    '''Order matters and is easy to get wrong. A `;` comment ends at its newline, so if the
    completion's lines are joined first, the comment text is absorbed into the expression and
    swallows the rest of the program.

    Measured, not hypothetical: once the shift_focus delta table entered the prompt the model
    began annotating its output, and 12 of 12 completions carried `;;` comments -- every one
    rejected at `parse` with the comment inside the s-expression.
    '''
    completion = (
        "(lambda (steps)\n"
        "  (lambda (s)          ; the state\n"
        "    (loop steps        ;; iterate over the steps\n"
        "      (lambda (i)\n"
        "        (lambda (s2) (shift RIGHT (place s2)))) s)))\n")
    candidates = extract_candidates(completion)
    assert len(candidates) == 1
    assert ";" not in candidates[0]
    assert "the state" not in candidates[0]
    # And the result is a real program once the named variables are converted.
    Program.parse(to_debruijn(candidates[0]))


def test_matched_comparison_configs_really_are_matched():
    '''A head-to-head is only a comparison of methods if the search budgets agree.

    This exists because they silently did not. `configs/default.py`'s LiloConfig sets
    `enumeration_timeout = 600.0` explicitly -- LILO's own setting, where the LLM is the
    primary solver -- and in a config that inherits both, that assignment sits earlier in the
    MRO than the run-level DreamCoderConfig. The first r2_lilo16 run therefore gave B3-b a
    600 s budget while B3-a got 14,400 s, a 24x mismatch, and nothing complained: the run
    completed, wrote plausible numbers, and only the wall clock gave it away.

    Scoped to the r2_* configs, which are the matched-comparison ones. default.py is
    deliberately unmatched (300 s vs 600 s) because it is a template, not an experiment.
    '''
    import importlib

    for name in ("r2_full16", "r2_lilo16"):
        module = importlib.import_module(f"baseline_spl.configs.{name}")
        dreamcoder = module.DreamCoderConfig
        lilo = module.LiloConfig
        assert lilo.enumeration_timeout == dreamcoder.enumeration_timeout, (
            f"{name}: B3-b searches for {lilo.enumeration_timeout}s while B3-a searches for "
            f"{dreamcoder.enumeration_timeout}s; the comparison would confound the LLM's "
            f"contribution with the search budget")
        assert lilo.search_iterations == dreamcoder.search_iterations, name
        assert lilo.grammar_level == dreamcoder.grammar_level, name
        assert lilo.accept_tau == dreamcoder.accept_tau, name
        assert lilo.observation_mode == dreamcoder.observation_mode, name
