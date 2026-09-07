'''
stitch_bridge.py

The sleep/abstraction phase: hand solved programs to STITCH, get reusable abstractions back.

STITCH (Bowers et al., POPL 2023) is the compressor LILO itself uses, and `stitch_core` is the
authors' released package, so this is their algorithm rather than a reimplementation. The call
mirrors `third_party/lilo/src/models/stitch_proposer.py:200-232`.

Why this phase matters more here than it does in a typical domain: the measured cost of
enumeration grows as ~e^(0.79 x MDL), and an abstraction is the only lever that moves a
concept between MDL bands. Adding `tower` as an abstraction takes `staircase` from 35.5 to
23.8 nats -- roughly 10^5 times more probable, and the difference between "unreachable" and
"reachable".
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import stitch_core

from baseline_spl.symbolic._dreamcoder import Grammar, Invented, Program
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, primitives_for


@dataclass
class Compression:
    '''What one abstraction round produced.'''
    abstractions: List[Invented] = field(default_factory=list)
    rewritten: Dict[str, str] = field(default_factory=dict)   # task -> rewritten program
    names: Dict[str, str] = field(default_factory=dict)       # str(abstraction) -> fn_N
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.abstractions)


def frontier_json(grammar: Grammar, solved: Sequence[Tuple[str, Program]]) -> dict:
    '''The DreamCoder-format payload `stitch.from_dreamcoder` expects.

    Shape taken from `src/models/stitch_base.py:57-71`: a DSL plus one frontier per task,
    each carrying its programs as s-expression strings.
    '''
    grouped: Dict[str, List[str]] = {}
    for name, program in solved:
        grouped.setdefault(name, []).append(str(program))
    return {
        "DSL": grammar.json(),
        "frontiers": [{"task": name, "programs": [{"program": p} for p in programs]}
                      for name, programs in grouped.items()],
    }


def compress(grammar: Grammar, solved: Sequence[Tuple[str, Program]], *,
             iterations: int = 3, max_arity: int = 3, silent: bool = True) -> Compression:
    '''Extract abstractions from the solved programs.

    `solved` is [(task name, program), ...]. Returns the abstractions plus each task's program
    rewritten to use them. Compression is skipped for fewer than two solutions: there is no
    shared structure to anti-unify, and STITCH would invent something useless.
    '''
    if len(solved) < 2:
        return Compression(error=f"only {len(solved)} solved task(s); nothing to compress")

    try:
        kwargs = stitch_core.from_dreamcoder(frontier_json(grammar, solved))
        # LILO passes eta_long=True and utility_by_rewrite=True as well
        # (stitch_proposer.py:211-217). Both are DROPPED here, and not by choice:
        # `utility_by_rewrite=True` segfaults the Rust backend in stitch_core 0.1.29 --
        # reproducibly, and on stitch's own two-line toy example, so it is a bug in the
        # installed build rather than anything about our input. `eta_long` cannot be used
        # without it ("eta long form requires utility_by_rewrite"). `no_other_util=True` is
        # theirs and is kept.
        #
        # Effect: utility is estimated rather than measured by rewriting, so STITCH may rank
        # candidate abstractions slightly differently than LILO's configuration would. The
        # algorithm is the same. Worth one sentence in the paper's setup.
        result = stitch_core.compress(**kwargs, iterations=iterations, max_arity=max_arity,
                                      no_other_util=True, silent=silent)
    except Exception as exc:  # noqa: BLE001 - a failed round must not kill the run
        return Compression(error=f"stitch.compress failed: {exc}")

    payload = result.json
    abstractions: List[Invented] = []
    names: Dict[str, str] = {}
    for entry in payload.get("abstractions", []):
        source = entry.get("dreamcoder")
        if not source:
            continue
        try:
            abstraction = Invented.parse(source)
            abstraction.infer()          # reject anything that will not type-check later
        except Exception:  # noqa: BLE001
            continue
        abstractions.append(abstraction)
        names[str(abstraction)] = entry.get("name") or f"fn_{len(names)}"

    # Align rewritten programs using the task list `from_dreamcoder` produced, not our own:
    # frontiers are grouped by task and may hold several programs each, so the flattened
    # order is STITCH's to define (this is what stitch_proposer.py:227 does). Keep the first
    # rewrite per task -- frontiers are best-first, so that is the rewrite of the best program.
    rewritten: Dict[str, str] = {}
    for task, program in zip(kwargs.get("tasks", []),
                             payload.get("rewritten_dreamcoder", [])):
        rewritten.setdefault(task, program)

    return Compression(abstractions=abstractions, rewritten=rewritten, names=names)


def eta_long(program: Program, request=None) -> Optional[Program]:
    '''Normalise to eta-long form, which `Grammar.logLikelihood` requires.

    A program STITCH rewrote comes back partially applied -- `row` becomes `(fn_0 RIGHT)` --
    and `closedLikelihoodSummary` returns None for those, which trips an `assert False` in
    `grammar.py:526` rather than raising something catchable. LILO avoids this by asking
    STITCH for eta-long output; that flag segfaults here (see `compress`), so normalise on
    our side instead, exactly as `sample_generator.py:631` does.

    `request` defaults to the program's OWN inferred type rather than to CONCEPT_REQUEST.
    Assuming concept-level silently broke demo-level: a closed program is `tstate -> tstate`,
    the visitor rejected it against `tint -> tstate -> tstate`, and every caller read the
    resulting None as "this program cannot be normalised" -- so closed solutions produced no
    training frontiers and had an infinite MDL, with nothing raised anywhere.
    '''
    from dreamcoder.program import EtaLongVisitor
    try:
        if request is None:
            request = program.infer()
        return EtaLongVisitor(request=request).execute(program)
    except Exception:  # noqa: BLE001
        return None


def program_mdl(grammar: Grammar, program: Program) -> float:
    '''Description length of one program, or +inf if the grammar cannot express it.'''
    normalised = eta_long(program)
    if normalised is None:
        return float("inf")
    try:
        prior = grammar.logLikelihood(CONCEPT_REQUEST, normalised)
    except Exception:  # noqa: BLE001
        return float("inf")
    if prior is None or prior == float("-inf"):
        return float("inf")
    return -prior


def corpus_mdl(grammar: Grammar, solved: Sequence[Tuple[str, Program]]) -> float:
    '''Total description length of the solved programs under a grammar.

    Lower is better. Returns +inf if any program is unreachable, which is what makes this
    safe to use as an acceptance test: an abstraction that breaks a solution is rejected.
    '''
    total = 0.0
    for _name, program in solved:
        cost = program_mdl(grammar, program)
        if cost == float("inf"):
            return float("inf")
        total += cost
    return total


def best_compression(grammar: Grammar, solved: Sequence[Tuple[str, Program]], *,
                     candidate_iterations: Sequence[int] = (1, 2, 3, 5, 10),
                     max_arity: int = 3, level: str = "standard",
                     ) -> Tuple[Compression, Grammar]:
    '''Choose how many abstractions to keep by description length.

    STITCH's `iterations` says how many abstractions to *extract*, not how many are worth
    keeping, and the two differ sharply here. Every production added dilutes the probability
    mass of the others, so an abstraction that is not used enough makes the corpus more
    expensive: measured on this domain, extracting 1 improves the deep concepts by 11.5 nats
    in total while extracting 8 makes them 18.8 nats *worse*.

    DreamCoder's criterion is description length, so apply it directly. For each candidate
    count, compare

        original programs under the current grammar
        vs. STITCH's *rewritten* programs under the extended grammar

    and keep the best. Scoring the rewritten programs is essential: the originals do not call
    the abstraction, so under the extended grammar they only ever look more expensive, and a
    useful abstraction would always be rejected. (This is what LILO's `utility_by_rewrite`
    flag would have done had it not segfaulted -- see `compress`.)

    The sweep now reaches 10, which is STITCH's own default in LILO
    (`DEFAULT_STITCH_PARAMS = {"max_arity": 3, "iterations": 10, ...}`, src/config_builder.py).
    Stopping the sweep at 5 meant we could never even consider the library size they run with.
    The MDL criterion still decides what is kept, so raising the ceiling costs nothing when
    fewer abstractions are better -- it only removes an arbitrary cap.

    Returns (chosen compression, grammar to search with). An empty Compression means no
    abstraction paid for itself this round, which is a legitimate outcome.
    '''
    baseline = corpus_mdl(grammar, solved)
    best_result, best_grammar, best_score = Compression(), grammar, baseline

    for iterations in candidate_iterations:
        result = compress(grammar, solved, iterations=iterations, max_arity=max_arity)
        if result.error or not result.abstractions:
            continue

        candidate_grammar = extend(grammar, result.abstractions, level=level)
        rewritten = []
        for name, original in solved:
            source = result.rewritten.get(name)
            if not source:
                rewritten.append((name, original))
                continue
            try:
                rewritten.append((name, Program.parse(source)))
            except Exception:  # noqa: BLE001
                rewritten.append((name, original))

        score = corpus_mdl(candidate_grammar, rewritten)
        if score < best_score - 1e-9:
            best_result, best_grammar, best_score = result, candidate_grammar, score

    return best_result, best_grammar


def extend(grammar: Grammar, abstractions: Sequence[Invented], level: str = "standard",
           uniform: bool = True) -> Grammar:
    '''A grammar with the abstractions added as productions.

    New productions get log-probability 0.0 before renormalisation, matching
    `stitch_proposer.py:167`. Re-weighting from the frontiers is a separate step
    (`Grammar.insideOutside`), deliberately not folded in here.
    '''
    existing = {str(p) for _l, _t, p in grammar.productions}
    fresh = [a for a in abstractions if str(a) not in existing]
    if not fresh:
        return grammar
    if uniform:
        return Grammar.uniform(primitives_for(level) + list(_invented(grammar)) + fresh,
                               continuationType=grammar.continuationType)
    return Grammar(grammar.logVariable,
                   list(grammar.productions) + [(0.0, a.infer(), a) for a in fresh],
                   continuationType=grammar.continuationType)


def _invented(grammar: Grammar) -> List[Invented]:
    return [p for _l, _t, p in grammar.productions if isinstance(p, Invented)]


def reweight(grammar: Grammar, solved: Sequence[Tuple[str, Program]],
             pseudo_counts: float = 1.0, log=None) -> Grammar:
    '''Re-fit the production probabilities to the solutions found so far.

    DreamCoder's other sleep step: productions that appear in solutions get cheaper, unused
    ones dearer, lowering the MDL of the programs the search is likely to want next.

    Two things have to be right here, and both were wrong at first:

    1. The programs must be in *library form* -- i.e. STITCH's rewritten versions, which call
       the abstractions. Re-weighting on the original inlined programs gives every abstraction
       a usage count of zero, so its probability is driven DOWN: measured at -2.944 for an
       abstraction that had just been accepted. That is the exact opposite of the intent, and
       it makes the library actively harmful. The caller is responsible for passing rewritten
       programs; `driver` does.
    2. They must be eta-long, or `insideOutside` raises and the whole step is skipped. A
       rewritten program comes back partially applied (`(fn_0 RIGHT)`), which is not eta-long,
       so this silently no-opped until normalised.

    Failures are reported rather than swallowed: a re-weighting that quietly does nothing
    looks identical to one that worked, and that hid bug 2 for a full run.
    '''
    if not solved:
        return grammar

    from dreamcoder.frontier import Frontier, FrontierEntry

    frontiers = []
    skipped = []
    for name, program in solved:
        normalised = eta_long(program)
        if normalised is None:
            skipped.append(name)
            continue
        try:
            prior = grammar.logLikelihood(CONCEPT_REQUEST, normalised)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{name} ({exc})")
            continue
        frontiers.append(Frontier([FrontierEntry(program=normalised, logLikelihood=0.0,
                                                 logPrior=prior)],
                                  task=_StubTask(name)))

    if skipped and log:
        log(f"  re-weighting skipped {len(skipped)} program(s): {', '.join(skipped[:4])}")
    if not frontiers:
        if log:
            log("  re-weighting had nothing usable; grammar left unchanged")
        return grammar

    try:
        return grammar.insideOutside(frontiers, pseudo_counts)
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"  re-weighting failed ({exc}); grammar left unchanged")
        return grammar


class _StubTask:
    '''Frontier requires a task with a `request`; insideOutside only reads that.'''

    def __init__(self, name: str):
        self.name = name
        self.request = CONCEPT_REQUEST

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return getattr(other, "name", None) == self.name
