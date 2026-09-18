'''
driver.py

The wake/sleep loop: search, compress, re-weight, search again.

    for each iteration:
        wake       enumerate under the current grammar; accept exact trace matches
        sleep      STITCH-compress the solutions into reusable abstractions
        re-weight  refit production probabilities to what the solutions actually used

Solved tasks are not re-searched, but unsolved ones are retried after every grammar change --
which is the entire point of the loop: an abstraction lowers the description length of the
programs that use it, so a concept out of reach in one iteration may be in reach in the next.

Solutions are carried forward in *library form* -- STITCH's rewritten versions, which call the
abstractions. That is load-bearing rather than cosmetic: re-weighting on the inlined originals
gives each abstraction a usage count of zero and drives its probability down (measured: -2.94
for one just accepted), so the sleep phase ends up penalising the library it just built. With
it fixed, compressing the lines takes `staircase` from 35.5 to 23.9 nats.
'''

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.symbolic._dreamcoder import Grammar, Invented, Program
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, grammar as build_grammar, to_term
from baseline_spl.symbolic import recognition
from baseline_spl.symbolic.search import (DEFAULT_MAXIMUM_FRONTIER, SearchStats,
                                          SearchTask, Solution, wake)
from baseline_spl.symbolic.settings import SearchSettings
from baseline_spl.symbolic.stitch_bridge import best_compression, program_mdl, reweight


@dataclass
class IterationReport:
    index: int
    solved: List[str] = field(default_factory=list)
    newly_solved: List[str] = field(default_factory=list)
    abstractions: List[str] = field(default_factory=list)
    programs_enumerated: int = 0
    seconds: float = 0.0
    rate: float = 0.0
    max_mdl_reached: float = 0.0
    recognition: bool = False
    # Per-stage wall clock, so a slow run can be attributed rather than guessed at.
    wake_seconds: float = 0.0
    compress_seconds: float = 0.0
    reweight_seconds: float = 0.0
    recognition_seconds: float = 0.0
    conditioning_seconds: float = 0.0
    # B3-b only: what the LLM proposer contributed this iteration, kept apart from the
    # enumerator's numbers so the two sources of solutions can be reported separately.
    proposal_seconds: float = 0.0
    documentation_seconds: float = 0.0
    proposed_by_llm: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"iteration": self.index, "solved": self.solved,
                "newly_solved": self.newly_solved, "abstractions": self.abstractions,
                "programs_enumerated": self.programs_enumerated,
                "seconds": round(self.seconds, 2), "programs_per_second": round(self.rate, 1),
                "max_mdl_reached": round(self.max_mdl_reached, 2),
                "recognition_trained": self.recognition,
                "solved_by_llm": self.proposed_by_llm,
                "seconds_by_stage": {
                    "proposal": round(self.proposal_seconds, 2),
                    "wake": round(self.wake_seconds, 2),
                    "conditioning": round(self.conditioning_seconds, 2),
                    "compress": round(self.compress_seconds, 2),
                    "documentation": round(self.documentation_seconds, 2),
                    "reweight": round(self.reweight_seconds, 2),
                    "recognition": round(self.recognition_seconds, 2)}}


@dataclass
class RunResult:
    grammar: Optional[Grammar] = None
    solutions: Dict[str, Solution] = field(default_factory=dict)
    abstractions: List[Invented] = field(default_factory=list)
    reports: List[IterationReport] = field(default_factory=list)
    seconds: float = 0.0
    # task -> "llm" or "enumeration". B3-a only ever writes "enumeration"; for B3-b this is
    # the headline split, since the claim under test is that an LLM proposer reaches concepts
    # enumeration cannot.
    solved_by: Dict[str, str] = field(default_factory=dict)
    # str(abstraction) -> {"readable_name", "description"}, from B3-b's auto-documentation.
    documentation: Dict[str, dict] = field(default_factory=dict)
    # How this run decided a program was correct. Recorded because `search_score` is in the
    # evaluator's own units -- cells, metres, sigma or nats -- and is uninterpretable without
    # it. Also the disclosure that acceptance was probabilistic rather than exact.
    evaluator: Dict[str, object] = field(default_factory=dict)

    @property
    def solved(self) -> List[str]:
        return sorted(n for n, s in self.solutions.items() if s.exact)

    @property
    def approximate(self) -> List[str]:
        return sorted(n for n, s in self.solutions.items()
                      if not s.exact and s.program is not None)

    @property
    def unsolved(self) -> List[str]:
        return sorted(n for n, s in self.solutions.items() if s.program is None)

    def stats(self) -> dict:
        '''What lands in search_stats.json.'''
        per_task = {}
        for name, solution in self.solutions.items():
            per_task[name] = {
                "status": solution.status,
                "solved_by": self.solved_by.get(name),
                "mdl": round(-solution.log_prior, 2) if solution.program is not None else None,
                # The accepting evaluator's own statistic. Under 'exact' this is the
                # normalised trace distance Round 1 reported; under a continuous evaluator it
                # is metres, sigma or nats -- see the run-level "evaluator" block.
                "search_score": round(solution.distance, 4),
                # Kept populated only where it means what its name says.
                "trace_distance": (round(solution.distance, 4)
                                   if self.evaluator.get("evaluator", "exact") == "exact"
                                   else None),
                "programs_tried": solution.programs_tried or None,
                "seconds": round(solution.seconds, 2) or None,
                "program": str(solution.program) if solution.program is not None else None,
            }
        return {
            "solved": self.solved,
            "approximate": self.approximate,
            "unsolved": self.unsolved,
            "solved_by_llm": sorted(n for n, s in self.solved_by.items() if s == "llm"),
            "solved_by_enumeration": sorted(n for n, s in self.solved_by.items()
                                            if s == "enumeration"),
            "abstractions": [str(a) for a in self.abstractions],
            "documentation": self.documentation,
            "evaluator": self.evaluator,
            "total_seconds": round(self.seconds, 2),
            "iterations": [r.as_dict() for r in self.reports],
            "per_task": per_task,
        }


class _Loop:
    """One wake/sleep run.

    The phases share a lot of state -- the growing library, each task's best solution so far,
    the trained recognizer -- so they are methods over that state rather than functions with
    long parameter lists. `run()` below is the whole loop, and reads as the sequence of phases
    it is.
    """

    def __init__(self, tasks, settings, evaluator, propose, document, log, on_iteration=None):
        self.tasks = list(tasks)
        self.settings = settings
        self.evaluator = evaluator
        self.propose_hook = propose
        self.document_hook = document
        self.log = log
        # Fires once per completed iteration, so a concept solved in iteration 1 can be
        # registered and saved to disk immediately rather than waiting for every iteration
        # (`search_iterations` may be 10+) to finish -- see `on_iteration`'s call site below.
        self.on_iteration_hook = on_iteration

        self.result = RunResult(grammar=build_grammar(
            settings.level, continuation=settings.continuation,
            int_literals_upto=settings.int_literals_upto))
        self.result.evaluator = evaluator.describe()
        self.result.solutions = {t.name: Solution(task=t.name) for t in self.tasks}
        self.targets = {t.name: t for t in self.tasks}
        self.dc_tasks = recognition.build_tasks(self.tasks)
        # task -> its solution rewritten to call the library. Kept apart from
        # Solution.program so the originally-found program stays on record.
        self.library_form: Dict[str, Program] = {}
        self.recognizer = None

    # ---------------------------------------------------------------- phases -- #

    def _pending(self) -> List[SearchTask]:
        return [t for t in self.tasks if not self.result.solutions[t.name].exact]

    def _requests(self) -> Dict[str, object]:
        """task name -> the type its programs must be scored at.

        Compression and re-weighting receive `(task name, program)` pairs, so they cannot ask a
        task for its own type. Without this map they assumed the concept-level request, which
        scores a closed demo-level program at -inf and silently disables library learning.
        """
        return {name: task.request for name, task in self.targets.items()}

    def _propose(self, index: int, pending) -> Tuple[List[str], float]:
        """B3-b's wake phase, ahead of enumeration so an LLM-solved task does not then spend
        enumeration budget. With `timeout=0` this becomes the LLM-only ablation."""
        if self.propose_hook is None:
            return [], 0.0
        started = time.time()
        # Few-shot material: what this method's own search has already found, paired with the
        # instruction it was found for. Never `demo['program']` -- these are the method's
        # discoveries, exactly what LILO shows itself.
        found = [(self.targets[n].instruction or n, str(s.program))
                 for n, s in self.result.solutions.items() if s.exact and n in self.targets]
        try:
            candidates = self.propose_hook(self.result.grammar, pending, index,
                                           self.result.documentation, found)
        except Exception as exc:  # noqa: BLE001 - a failed proposal must not end the run
            self.log(f"[iter {index}] proposer raised ({type(exc).__name__}: {exc}); "
                     f"falling back to enumeration.")
            candidates = {}
        names = _accept_proposals(self.result, candidates, self.targets, self.evaluator,
                                  self.settings.maximum_frontier, log=self.log)
        seconds = time.time() - started
        self.log(f"[iter {index}] proposer solved {len(names)} task(s) "
                 f"({', '.join(names) or 'none'}) in {seconds:.0f}s")
        return names, seconds

    def _train_recognition(self, index: int) -> float:
        """Sleep-R, trained BEFORE this iteration's enumeration, as LILO orders it
        (llm_solver -> optimize_model_for_frontiers -> enumerate). Training here rather than
        at the end of the iteration is what lets enumeration benefit from the solutions the
        proposer just found, in the same iteration."""
        recognition_settings = self.settings.recognition
        if not recognition_settings.enabled:
            return 0.0
        started = time.time()
        # Reassigned unconditionally: if training fails this iteration searches under the
        # global grammar rather than a stale model from the previous one.
        self.recognizer = recognition.train_recognizer(
            self.result.grammar, self.dc_tasks, self.result.solutions,
            epochs=recognition_settings.epochs, steps=recognition_settings.steps,
            timeout=recognition_settings.timeout,
            helmholtz_ratio=recognition_settings.helmholtz_ratio,
            hidden=recognition_settings.hidden,
            contextual=recognition_settings.contextual,
            bias_optimal=recognition_settings.bias_optimal,
            auxiliary_loss=recognition_settings.auxiliary_loss,
            cpus=self.settings.cpus, log=self.log)
        return time.time() - started

    def _wake(self, index: int, pending) -> Tuple[SearchStats, float]:
        """Enumerate. Sleep-R's output is consumed here: with a recognizer trained, each task
        gets its own grammar rather than one global one."""
        settings = self.settings
        if not pending:
            return SearchStats(), 0.0
        if settings.timeout <= 0:
            # The LLM-only ablation: proposal is the entire wake phase.
            self.log(f"[iter {index}] enumeration disabled (timeout={settings.timeout}); "
                     f"{len(pending)} task(s) left to the proposer")
            return SearchStats(), 0.0

        started = time.time()
        searching_with = self.result.grammar
        if self.recognizer is not None:
            searching_with = recognition.grammars_for(
                self.recognizer, {t.name: self.dc_tasks[t.name] for t in pending},
                self.result.grammar, log=self.log)
        conditioning_seconds = time.time() - started
        conditioned = "per-task grammars" if self.recognizer is not None else "one grammar"
        self.log(f"[iter {index}] searching {len(pending)} task(s) under {conditioned} of "
                 f"{len(self.result.grammar.productions)} productions")
        stats = wake(searching_with, pending, timeout=settings.timeout,
                     max_mdl=settings.max_mdl, evaluator=self.evaluator,
                     maximum_frontier=settings.maximum_frontier, cpus=settings.cpus,
                     log=self.log)
        return stats, conditioning_seconds

    def _adopt(self, stats: SearchStats, report: IterationReport) -> None:
        """Fold the wake phase's findings into the run, keeping the better of the two: an
        exact hit always wins, otherwise the nearer miss."""
        for name, solution in stats.per_task.items():
            previous = self.result.solutions[name]
            if solution.exact or (previous.program is None) or (
                    solution.program is not None and solution.distance < previous.distance):
                self.result.solutions[name] = solution
            if solution.exact:
                report.newly_solved.append(name)
                self.result.solved_by[name] = "enumeration"
        report.solved = sorted(n for n, s in self.result.solutions.items() if s.exact)

    def _compress(self, index: int, report: IterationReport, solved_programs):
        """Sleep-G: extract abstractions, adopt the rewritten solutions, document them."""
        settings = self.settings
        if not (settings.use_library and len(solved_programs) >= 2):
            return solved_programs

        started = time.time()
        compression, grammar_with_library = best_compression(
            self.result.grammar, solved_programs, max_arity=settings.max_arity,
            level=settings.level, requests=self._requests())
        report.compress_seconds = time.time() - started
        if not compression.abstractions:
            self.log(f"[iter {index}] no abstraction lowered the corpus description length"
                     + (f" ({compression.error})" if compression.error else ""))
            return solved_programs

        self.result.grammar = grammar_with_library
        self.result.abstractions = _merge(self.result.abstractions, compression.abstractions)
        report.abstractions = [str(a) for a in compression.abstractions]
        adopted = _adopt_rewritten(compression, self.targets, self.library_form,
                                   self.evaluator)
        self.log(f"[iter {index}] kept {len(compression.abstractions)} abstraction(s); "
                 f"library now has {len(self.result.abstractions)}; "
                 f"{adopted} solution(s) rewritten against it")
        solved_programs = _corpus(self.result, self.library_form)
        self._document(index, report, compression.abstractions, solved_programs)
        return solved_programs

    def _document(self, index, report, abstractions, solved_programs) -> None:
        """LILO's third contribution: name and describe each new abstraction so the next
        proposal prompt can refer to the library in words. Failure is non-fatal -- an
        undocumented abstraction simply keeps its anonymous form."""
        if self.document_hook is None:
            return
        started = time.time()
        try:
            docs = self.document_hook(abstractions, solved_programs) or {}
        except Exception as exc:  # noqa: BLE001
            self.log(f"[iter {index}] auto-documentation raised "
                     f"({type(exc).__name__}: {exc}); abstractions stay anonymous.")
            docs = {}
        self.result.documentation.update(docs)
        report.documentation_seconds = time.time() - started
        named = ", ".join(d.get("readable_name", "?") for d in docs.values())
        self.log(f"[iter {index}] documented {len(docs)} abstraction(s)"
                 + (f": {named}" if named else ""))

    def _reweight(self, index: int, report: IterationReport, solved_programs) -> None:
        """Refit production probabilities to what the solutions actually used.

        On the LIBRARY FORM of each solution, not the inlined original: re-weighting on the
        originals gives every abstraction a usage count of zero and drives its probability
        DOWN (measured -2.944 for one just accepted), so the sleep phase ends up penalising
        the library it just built.
        """
        if not solved_programs:
            return
        started = time.time()
        requests = self._requests()
        self.result.grammar = reweight(self.result.grammar, solved_programs,
                                       self.settings.pseudo_counts, log=self.log,
                                       requests=requests)
        report.reweight_seconds = time.time() - started
        mean = sum(program_mdl(self.result.grammar, p, requests.get(n))
                   for n, p in solved_programs) / len(solved_programs)
        self.log(f"[iter {index}] re-weighted on {len(solved_programs)} solution(s); "
                 f"mean solved MDL now {mean:.1f}")

    # ------------------------------------------------------------------ loop -- #

    def run(self) -> RunResult:
        self.log(f"acceptance: {self.result.evaluator}")
        started = time.time()

        for index in range(self.settings.iterations):
            pending = self._pending()
            if not pending:
                self.log(f"[iter {index}] every task solved; stopping early.")
                break

            proposed_names, proposal_seconds = self._propose(index, pending)
            if proposed_names:
                pending = self._pending()
                if not pending:
                    self.log(f"[iter {index}] every task solved by proposal; "
                             f"skipping enumeration.")

            recognition_seconds = self._train_recognition(index)
            stats, conditioning_seconds = self._wake(index, pending)

            report = IterationReport(
                index=index, programs_enumerated=stats.programs_enumerated,
                seconds=stats.seconds, rate=stats.rate,
                max_mdl_reached=stats.max_mdl_reached, wake_seconds=stats.seconds,
                conditioning_seconds=conditioning_seconds,
                proposal_seconds=proposal_seconds, proposed_by_llm=proposed_names,
                recognition_seconds=recognition_seconds,
                recognition=self.recognizer is not None)
            report.newly_solved.extend(proposed_names)
            self._adopt(stats, report)
            self.log(f"[iter {index}] {len(report.newly_solved)} newly solved "
                     f"({', '.join(report.newly_solved) or 'none'}); "
                     f"{stats.programs_enumerated:,} programs at {stats.rate:,.0f}/s")

            solved_programs = _corpus(self.result, self.library_form)
            solved_programs = self._compress(index, report, solved_programs)
            self._reweight(index, report, solved_programs)

            self.result.reports.append(report)
            if self.on_iteration_hook is not None:
                try:
                    self.on_iteration_hook(self.result, index)
                except Exception as exc:  # noqa: BLE001 - registering early must never end the run
                    self.log(f"[iter {index}] on_iteration hook raised "
                             f"({type(exc).__name__}: {exc}); continuing.")
            if not report.newly_solved and not report.abstractions:
                self.log(f"[iter {index}] neither the solutions nor the library changed; "
                         f"stopping.")
                break

        self.result.seconds = time.time() - started
        self.log(f"done in {self.result.seconds:.0f}s: {len(self.result.solved)} solved, "
                 f"{len(self.result.approximate)} approximate, "
                 f"{len(self.result.unsolved)} unsolved")
        return self.result


def run(tasks: Sequence[SearchTask], settings: Optional[SearchSettings] = None, *,
        propose=None, document=None, evaluator=None, log=print, on_iteration=None) -> RunResult:
    '''Search for a program for every task, growing a library as it goes.

    `propose` and `document` are B3-b's (LILO's) two additions, and they are the *only*
    difference between the two baselines -- passing neither is B3-a exactly:

        propose(grammar, pending, iteration, docs, solved) -> {task: [candidate Program]}
            asked before enumeration, so a task the LLM solves does not then burn
            enumeration budget. Candidates are gated by the same all-or-nothing trace
            match the wake phase uses; nothing is trusted because a model proposed it.
            `solved` is (instruction, program source) for what has been found so far.
        document(new_abstractions, corpus) -> {str(abstraction): {readable_name, ...}}
            asked after compression accepts abstractions, so the next proposal prompt can
            refer to the library by name rather than by anonymous s-expression.

    `on_iteration(result, index)` runs once per completed iteration, mainly so a caller can
    register+save newly-solved concepts as they happen instead of waiting for every iteration
    to finish -- with `settings.iterations` in the double digits and each one hours long,
    "registration happens once at the very end" meant a kill anywhere before the last
    iteration lost everything, not just that iteration's work.
    '''
    from baseline_spl.symbolic.search import DEFAULT_EVALUATOR

    return _Loop(tasks, settings or SearchSettings(), evaluator or DEFAULT_EVALUATOR,
                 propose, document, log, on_iteration=on_iteration).run()


def _accept_proposals(result: RunResult, candidates: Dict[str, Sequence[Program]],
                      targets: Dict[str, SearchTask], evaluator, maximum_frontier: int,
                      log=print) -> List[str]:
    '''Score LLM-proposed programs and fold the good ones into the solutions.

    A proposed program earns its place exactly as an enumerated one does: it must reproduce
    every example of its task. Nothing is accepted because a model wrote it. Programs that
    run but produce the wrong trace are kept as near-misses, so an unsolved concept still
    reports a best effort -- the same treatment enumeration's misses get.

    Returns the names of tasks this call solved for the first time.
    '''
    from baseline_spl.symbolic.search import DEFAULT_EVALUATOR
    from baseline_spl.symbolic.stitch_bridge import program_mdl

    scorer = evaluator or DEFAULT_EVALUATOR
    solved: List[str] = []
    for name, programs in (candidates or {}).items():
        task = targets.get(name)
        solution = result.solutions.get(name)
        if task is None or solution is None or solution.exact:
            continue
        for program in programs:
            distance = scorer.score(program, task)
            if distance is None:            # did not run at all
                continue
            # MDL under the *current* grammar, so a proposal is directly comparable with an
            # enumerated solution and can be handed to compression on the same footing.
            prior = -program_mdl(result.grammar, program,
                                 getattr(targets.get(name), "request", None))
            if scorer.accepts(distance):
                if (solution.add_exact(prior, program, maximum_frontier, distance)
                        and name not in solved):
                    solved.append(name)
                    result.solved_by[name] = "llm"
            else:
                solution.add_approximate(prior, program, distance)

        if solution.program is not None and solution.term is None:
            # `lower` needs a Term. Done for near-misses too, exactly as the wake phase does
            # (`search.py:345`): otherwise an approximate LLM proposal would be recorded as
            # untranslatable where an approximate enumerated one is lowered and scored, which
            # would under-report this baseline for no reason but an inconsistency here.
            # A task's program has as many leading lambdas as its arity; hardcoding the
            # 1-argument shape here meant every LLM proposal on a closed or multi-argument
            # task failed translation and was recorded untranslatable. `search.py` gets this
            # right from `task.arity`; do the same rather than trust the default.
            arity = getattr(targets.get(name), "arity", 0)
            try:
                solution.term = to_term(solution.program, concept_level=arity)
            except Exception as exc:  # noqa: BLE001
                log(f"  proposal for {name} could not be read back as a term ({exc})")
    return solved


def _corpus(result, library_form: Dict[str, Program]) -> List[Tuple[str, Program]]:
    """Every solved task's programs, in library form where one exists.

    All of a task's frontier is included, not just its best program: STITCH anti-unifies
    across whatever it is given, so more verified solutions per task means more shared
    structure to find. The library form of a task replaces only its best program, since that
    is the one STITCH rewrote.
    """
    corpus: List[Tuple[str, Program]] = []
    for name, solution in result.solutions.items():
        if not solution.exact:
            continue
        programs = list(solution.programs)
        if name in library_form:
            programs[0] = library_form[name]
        corpus.extend((name, p) for p in programs)
    return corpus


def _merge(existing: Sequence[Invented], fresh: Sequence[Invented]) -> List[Invented]:
    seen = {str(a) for a in existing}
    return list(existing) + [a for a in fresh if str(a) not in seen]


def _adopt_rewritten(compression, targets: Dict[str, SearchTask],
                     library_form: Dict[str, Program], evaluator=None) -> int:
    '''Adopt STITCH's rewritten programs as the library form of each solution.

    Verified rather than trusted: a rewritten program is adopted only if it still reproduces
    every one of its task's examples. Compression is meant to be meaning-preserving, but this
    is what the whole loop is built on, so it is checked rather than assumed.
    '''
    from baseline_spl.symbolic.search import DEFAULT_EVALUATOR

    # Verified under the *same* evaluator the search accepted by. Re-checking a rewritten
    # program against exact lattice equality while the run accepts probabilistically would
    # reject library forms the run considers correct, silently keeping every solution in
    # inlined form -- which is Finding E's bug, in a new place.
    scorer = evaluator or DEFAULT_EVALUATOR
    adopted = 0
    for task_name, source in compression.rewritten.items():
        task = targets.get(task_name)
        if task is None or not source:
            continue
        try:
            candidate = Program.parse(source)
            if scorer.accepts(scorer.score(candidate, task)):
                library_form[task_name] = candidate
                adopted += 1
        except Exception as exc:  # noqa: BLE001 - keep the original form for this task
            # Not fatal, but never silent: a solution left inlined gives its abstraction a
            # usage count of zero, and re-weighting then drives that abstraction's
            # probability DOWN. That is how the library became actively harmful once before.
            print(f"  <{task_name}> could not adopt its library form "
                  f"({type(exc).__name__}: {exc}); keeping the inlined program")
            continue
    return adopted
