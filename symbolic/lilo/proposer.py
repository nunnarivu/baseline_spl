'''
proposer.py

LILO's first contribution: an LLM proposes programs, replacing (or augmenting) enumeration.

The proposer is deliberately *not* trusted. A sampled string passes through the same seven
checks LILO applies in `src/models/sample_generator.py:568-660` before it is even considered a
candidate, and the driver then requires it to reproduce every one of its task's examples --
the same all-or-nothing test an enumerated program faces. What the LLM changes is *which*
programs get tested, never the standard they are held to.

That is the whole scientific point. B3-a measured a hard ceiling: enumeration cost grows as
`~e^(0.79 x MDL)`, so the lines fall at 12-15 nats and the composites sit at 23-79, out of
reach at any feasible budget. A language model is not sampling by description length, so it is
not bound by that exponential. Whether it actually clears the ceiling is what this measures.

The rejection histogram is a reportable result rather than plumbing: "how often does a
language model emit well-typed terms in an unfamiliar DSL?" is a number the paper wants, and
it is free to collect.
'''

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from baseline_spl.symbolic._dreamcoder import Grammar, Invented, Program
from baseline_spl.symbolic.lilo import prompts
from baseline_spl.symbolic.search import SearchTask, score
from baseline_spl.symbolic.stitch_bridge import eta_long

# The seven gates, in LILO's order. Names double as the histogram keys.
PARSE = "parse"
INFER = "typecheck"
REQUEST = "wrong_request_type"
FREE_VARS = "free_variables"
ETA_LONG = "eta_long"
LIKELIHOOD = "likelihood"
WRONG_TRACE = "wrong_trace"
ACCEPTED = "accepted"


def _invented(grammar: Grammar) -> List[Invented]:
    return [p for _l, _t, p in grammar.productions if isinstance(p, Invented)]


def anonymous_names(abstractions: Sequence[Invented]) -> Dict[str, str]:
    '''str(abstraction) -> fn_N, in grammar order so it is stable within a run.'''
    return {str(a): f"fn_{i}" for i, a in enumerate(abstractions)}


def substitute_aliases(source: str, aliases: Dict[str, str]) -> str:
    '''Replace library names with the anonymous bodies `Program.parse` understands.

    Our inverse of LILO's `grammar.show_program(input_name_class=...)`. Without it,
    auto-documentation would be decorative: the model could be *shown* a readable name but
    could not *write* one, so naming could not affect what it produces.

    Longest name first, and anchored to token boundaries, so a short name cannot eat a
    prefix of a longer one (`line` must not match inside `line_of_blocks`).
    '''
    for name in sorted(aliases, key=len, reverse=True):
        source = re.sub(rf"(?<![\w#]){re.escape(name)}(?![\w])", aliases[name], source)
    return source


_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def to_debruijn(source: str) -> str:
    '''Rewrite `(lambda (x) body)` into `(lambda body)` with de Bruijn indices.

    The one deliberate deviation from published LILO, behind `allow_named_variables`, and
    made on measured grounds rather than taste: in the strict-LILO run **every** candidate
    died before execution -- 12 of 12, 10 at typecheck, 11 of them involving `saved` -- and
    the failures were de Bruijn bookkeeping, not program structure. The model was never
    wrong about staircase; it never produced a program that typechecked.

    This is an extension of a mechanism LILO already has rather than a new one. Upstream's
    `show_program` takes `name_classes` and a `lam` parameter precisely so the model can be
    shown, and can write, a different surface form that is translated back before parsing
    (`laps_grammar.py:240`). LILO applies that to function names; we apply it to variables
    too, at the same point in the pipeline as `substitute_aliases`.

    Already-de-Bruijn input passes through untouched, so the model may write either form and
    the strict-LILO behaviour is unchanged (`tests/test_lilo.py` pins this on all 16
    oracles).
    '''
    tokens = _TOKEN.findall(source)
    position = 0

    def parse():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("unexpected end of expression")
        if tokens[position] == "(":
            position += 1
            items = []
            while position < len(tokens) and tokens[position] != ")":
                items.append(parse())
            if position >= len(tokens):
                raise ValueError("unbalanced expression")
            position += 1
            return items
        token = tokens[position]
        position += 1
        return token

    tree = parse()

    def emit(node, scope):
        if isinstance(node, str):
            if node in scope:
                # Innermost binder is $0, counting outwards. Searching the REVERSED scope
                # is what makes shadowing correct: a model that reuses `s` for two state
                # binders must bind to the inner one. Taking the first match instead would
                # silently produce a well-typed program with the wrong semantics.
                return f"${scope[::-1].index(node)}"
            return node                                   # primitive, literal or direction
        # (lambda (x) body): a named binder.
        if (len(node) == 3 and node[0] == "lambda" and isinstance(node[1], list)
                and len(node[1]) == 1 and isinstance(node[1][0], str)):
            return f"(lambda {emit(node[2], scope + [node[1][0]])})"
        # (lambda body): already de Bruijn. The placeholder keeps the depth right so any
        # $n inside still resolves against the correct number of enclosing binders.
        if len(node) == 2 and node[0] == "lambda":
            return f"(lambda {emit(node[1], scope + ['_'])})"
        return "(" + " ".join(emit(child, scope) for child in node) + ")"

    return emit(tree, [])


def extract_candidates(text: str) -> List[str]:
    '''One completion is ONE program, as upstream treats it.

    `parse_completion` iterates over `completion["choices"]` and takes `choice["text"]`
    whole (`sample_generator.py:566-567`); it never splits a completion into several
    candidates. Diversity comes from `n_samples_per_query`, not from the model listing
    programs in one reply. Upstream also passes `stop="\\n"` (`gpt_base.py:361`), which
    keeps a completion to a single line in the first place.

    Do NOT split on lines: a model that pretty-prints across several lines would have each
    line taken as a separate candidate, shredding whole programs into sub-expressions that
    infer to bare `tstate` and are rejected as wrong-request-type.

    Returns a single-element list (or empty) so the caller's loop is unchanged.
    '''
    cleaned = re.sub(r"```[a-zA-Z]*", " ", text)
    # Strip Lisp comments BEFORE joining lines. Order is the whole point: a `;` comment ends
    # at its newline, so once lines are joined it would swallow the entire rest of the
    # program. Measured -- with the delta table in the prompt the model started annotating
    # its output, and 12 of 12 completions carried `;;` comments; every one died at `parse`
    # with the comment text absorbed into the expression. Comments are not part of the
    # program, so removing them reads what the model wrote rather than changing it.
    cleaned = re.sub(r";[^\n]*", " ", cleaned)
    cleaned = " ".join(cleaned.split())          # newlines are formatting, not structure
    start = cleaned.find("(")
    if start < 0:
        return []
    cleaned = cleaned[start:]

    depth = 0
    for i, ch in enumerate(cleaned):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return [cleaned[:i + 1]]
    return []                                     # unbalanced: the model's own error


class LLMProposer:
    '''Samples candidate programs for the unsolved tasks and gates them.

    Input: backend  - common.llm_backend.LLMBackend (model parity, cache, token ledger)
           samples  - candidate programs requested per task per iteration
    '''

    def __init__(self, backend, *, queries_per_task: int = 4, samples_per_query: int = 4,
                 max_tokens: int = 3000, temperature: float = 0.7, seed: int = 0,
                 allow_named_variables: bool = False, evaluator=None,
                 demo_modality: str = "coords", log=print):
        self.backend = backend
        # How the demonstration reaches the model.
        #   'coords' : raw centroids in the task language + the shift_focus delta table,
        #              i.e. parity with CaP / Demo2Code-text at coordinate_mode='raw'
        #   'images' : keyframe PNGs on the first message, the geometry removed from the
        #              text, i.e. parity with CaP-images / Demo2Code-vlm
        #   'both'   : both channels
        # 'images' and 'both' are deviations from published LILO, which is text-only.
        if demo_modality not in ("coords", "images", "both"):
            raise ValueError(f"unknown demo_modality {demo_modality!r}")
        self.demo_modality = demo_modality
        # {task name: [png bytes]} and the primitive delta table, both supplied by the
        # harness, which is what holds the demonstrations.
        self.demo_frames: Dict[str, list] = {}
        self.stats_block: str = ""
        # The run's acceptance rule. Gate 7 must use the same one the driver will, or the
        # histogram would report candidates as wrong-trace that the run then accepts.
        from baseline_spl.symbolic.search import DEFAULT_EVALUATOR
        self.evaluator = evaluator or DEFAULT_EVALUATOR
        # The single knob separating published LILO from our variant. False reproduces the
        # paper exactly; True lets the model name its lambda arguments (see `to_debruijn`).
        self.allow_named_variables = allow_named_variables
        # template_lilo.json: n_queries_per_task=4, n_samples_per_query=4, temperature=0.7.
        self.queries_per_task = queries_per_task
        self.samples_per_query = samples_per_query
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.log = log
        # Reportable in their own right.
        self.rejections: Counter = Counter()
        self.calls = 0
        # `calls` counts queries issued. A query whose prompt is already cached returns the
        # same completions without touching the API, and every candidate in it would then be
        # re-gated and re-counted -- which is how an earlier run reported 8 calls and 32
        # candidates when the truth was 3 calls and 12 distinct candidates. Count the two
        # populations apart, and gate each distinct source string once.
        self.api_calls = 0
        self.cache_hits = 0
        self.duplicate_candidates = 0
        self._seen: Dict[str, set] = {}
        # task -> token ledger delta. `learn_concept` resets the ledger per concept via
        # `agent.backend`, but this all happens before any `learn_concept` call, so without
        # this the LLM spend would be invisible in the cost table.
        self.ledger_by_task: Dict[str, dict] = {}

    # ------------------------------------------------------------------ #

    def __call__(self, grammar: Grammar, pending: Sequence[SearchTask], iteration: int,
                 documentation: Dict[str, dict],
                 solved: Sequence = ()) -> Dict[str, List[Program]]:
        '''The `propose` hook `driver.run` calls.'''
        # Upstream refuses to build a prompt from fewer than two solved tasks --
        # `gpt_solver.py:366` raises "At least 2 tasks must have non-empty frontiers to
        # construct a prompt", and `sample_generator.py:402` says the same. So LILO's
        # solver cannot run from an empty state at all: it generalises from what search has
        # already found rather than solving from scratch. Skipping here rather than raising
        # keeps the driver's contract, and it is also what stops the first iteration
        # spending API budget on a prompt LILO would never have sent.
        if len(list(solved)) < 2:
            self.log(f"[iter {iteration}] proposer idle: {len(list(solved))} solved task(s), "
                     f"LILO needs 2 to build a prompt")
            return {}

        abstractions = _invented(grammar)
        names = anonymous_names(abstractions)
        aliases = self._aliases(abstractions, names, documentation)
        # `prepend_dsl_description: true` in template_lilo.json.
        system = prompts.dsl_description(grammar, documentation, names,
                                         named_variables=self.allow_named_variables,
                                         stats_block=self.stats_block)

        out: Dict[str, List[Program]] = {}
        for task in pending:
            programs = self._propose_one(task, grammar, system, aliases, list(solved))
            if programs:
                out[task.name] = programs
        return out

    # ------------------------------------------------------------------ #

    @staticmethod
    def _aliases(abstractions, names, documentation) -> Dict[str, str]:
        '''Every name the model might write -> the body `Program.parse` accepts.'''
        aliases: Dict[str, str] = {}
        for abstraction in abstractions:
            body = str(abstraction)
            aliases[names[body]] = body                     # fn_0
            readable = (documentation.get(body) or {}).get("readable_name")
            if readable:
                aliases[readable] = body                    # line_of_blocks
        return aliases

    def _propose_one(self, task: SearchTask, grammar: Grammar, system: str,
                     aliases: Dict[str, str], solved: List) -> List[Program]:
        '''LILO's sampling loop: `n_queries_per_task` queries, each with a freshly
        randomised body-task ordering (`sample_generator.py:397`), and
        `n_samples_per_query` completions per query.'''
        before = self.backend.ledger()
        accepted: List[Program] = []
        # Under 'images' the geometry arrives as keyframes, so repeating it as text would
        # give B3-b strictly more than CaP-images or Demo2Code-vlm receives.
        target = prompts.task_language(
            task, include_demonstration=self.demo_modality != "images")
        images = (self.demo_frames.get(task.name) or None
                  if self.demo_modality in ("images", "both") else None)

        try:
            for _query in range(self.queries_per_task):
                body = list(solved)
                self.rng.shuffle(body)
                messages = [{"role": "system", "content": system}]
                messages += prompts.message_list(body, target)
                calls_before = self.backend.ledger().get("num_llm_calls", 0)
                try:
                    texts = self.backend.call_messages_sampled(
                        messages, n=self.samples_per_query,
                        temperature=self.temperature, stop="\n",
                        images=images, max_tokens=self.max_tokens)
                except Exception as exc:  # noqa: BLE001 - one query must not end the run
                    self.log(f"  proposal query for {task.name} failed: "
                             f"{type(exc).__name__}: {exc}")
                    continue
                self.calls += 1
                if self.backend.ledger().get("num_llm_calls", 0) > calls_before:
                    self.api_calls += 1
                else:
                    self.cache_hits += 1
                seen = self._seen.setdefault(task.name, set())
                for text in texts:
                    for source in extract_candidates(text):
                        # A cached query returns identical completions; gating them again
                        # would inflate the rejection histogram, which is a reported result.
                        if source in seen:
                            self.duplicate_candidates += 1
                            continue
                        seen.add(source)
                        program = self._gate(source, task, grammar, aliases)
                        if program is not None and not any(
                                str(program) == str(p) for p in accepted):
                            accepted.append(program)
        finally:
            self._record_ledger(task.name, before)
        return accepted

    def _record_ledger(self, task: str, before: dict) -> None:
        after = self.backend.ledger()
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        prior = self.ledger_by_task.setdefault(
            task, {k: 0 for k in ("prompt_tokens", "completion_tokens",
                                  "num_llm_calls", "num_cache_hits")})
        for key, value in delta.items():
            prior[key] = prior.get(key, 0) + value

    def _gate(self, source: str, task: SearchTask, grammar: Grammar,
              aliases: Dict[str, str]) -> Optional[Program]:
        '''LILO's CHECK 1-7. Returns the program, or None with the reason counted.'''
        # 1. parses (after resolving library names, then named binders, to the forms
        #    `Program.parse` understands). Both are surface-syntax translations applied
        #    before parsing, which is where upstream does its own (`show_program`).
        try:
            resolved = substitute_aliases(source, aliases)
            if self.allow_named_variables:
                resolved = to_debruijn(resolved)
            program = Program.parse(resolved)
        except Exception:  # noqa: BLE001
            self.rejections[PARSE] += 1
            return None
        # 2. typechecks
        try:
            inferred = program.infer()
        except Exception:  # noqa: BLE001
            self.rejections[INFER] += 1
            return None
        # 3. has the request type THIS task asks for. A demo-level task is closed
        # (`tstate -> tstate`); comparing against the concept-level request rejected every
        # proposal on such a run, so B3-b's proposer contributed nothing at demo level.
        expected = task.request
        if inferred != expected:
            self.rejections[REQUEST] += 1
            return None
        # 4. no free variables
        if not program.closed:
            self.rejections[FREE_VARS] += 1
            return None
        # 5. normalises to eta-long form, which logLikelihood requires
        normalised = eta_long(program, expected)
        if normalised is None:
            self.rejections[ETA_LONG] += 1
            return None
        # 6. the grammar can score it
        try:
            prior = grammar.logLikelihood(expected, normalised)
        except Exception:  # noqa: BLE001
            self.rejections[LIKELIHOOD] += 1
            return None
        if prior is None or prior == float("-inf"):
            self.rejections[LIKELIHOOD] += 1
            return None
        # 7. reproduces every example. The driver re-checks this before accepting; counted
        #    here so the histogram distinguishes "wrote nonsense" from "wrote a real program
        #    that builds the wrong thing", which are very different failures.
        if not self.evaluator.accepts(self.evaluator.score(program, task)):
            self.rejections[WRONG_TRACE] += 1
            return None

        self.rejections[ACCEPTED] += 1
        return program

    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        total = sum(self.rejections.values())
        return {
            # Queries issued, and how many of those actually reached the API. With only a
            # handful of solved tasks the randomised few-shot ordering yields few distinct
            # prompts, so these two differ a lot -- quote `api_calls` as the cost.
            "queries_issued": self.calls,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            # Retained under its old name so existing readers do not silently change
            # meaning; it is the honest figure now.
            "llm_calls": self.api_calls,
            "candidates_examined": total,
            "duplicate_candidates_skipped": self.duplicate_candidates,
            "accepted": self.rejections.get(ACCEPTED, 0),
            "acceptance_rate": round(self.rejections.get(ACCEPTED, 0) / total, 4) if total else 0.0,
            "rejected_by_gate": {k: v for k, v in sorted(self.rejections.items())
                                 if k != ACCEPTED},
            "tokens_by_task": self.ledger_by_task,
        }
