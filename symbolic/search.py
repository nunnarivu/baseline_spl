'''
search.py

The wake phase: enumerate programs in decreasing prior probability and keep those whose
execution reproduces every example.

This is DreamCoder's own search, reduced to what this domain needs. `Grammar.enumeration`
(grammar.py:626) yields well-typed programs inside an MDL band; we widen the band until the
budget runs out, mirroring `enumerateForTasks` (enumeration.py:517).

Acceptance is all-or-nothing, as in DreamCoder (`task.py:126`): a program either reproduces
every example's placement sequence exactly or it does not count. Alongside that we keep a
*soft frontier* of near misses ranked by trace distance, so a concept the search cannot solve
still reports its best attempt instead of a blank. The soft frontier ranks; it does not guide
-- enumeration order comes from the grammar prior alone.

Two things follow DreamCoder rather than being simplifications of it:

  frontiers   Up to `maximum_frontier` programs are kept per task, not just the first hit
              (DreamCoder's `maximumFrontier`). More programs per task gives STITCH more
              material to anti-unify and the recognition model more to train on.

  parallelism Across *tasks*, as `multicoreEnumeration` does -- not across MDL sub-bands.
              Sub-bands look tempting because `Grammar.enumeration(lowerBound=l,
              upperBound=u)` yields exactly the programs whose MDL lies in (l, u], so they
              partition a band's *output* exactly (test_search_parallel checks this). But
              `lowerBound` only filters; it does not prune the traversal. Measured: the slice
              (12.0, 13.5] holds 57% of the programs of (0, 13.5] and costs 100.2% of its
              time. Splitting a band therefore multiplies total work by the number of slices,
              and a 16-way split ran 3x slower than serial.

              Task-level parallelism is redundant while every task shares one grammar -- each
              worker would re-enumerate the same programs -- so with a shared grammar we keep
              a single enumeration scored against all tasks, which is strictly better than
              DreamCoder's. It becomes the right structure once the recognition model gives
              each task its own grammar, because then the enumerations genuinely differ.
'''

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.symbolic._dreamcoder import Program
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, to_term
from baseline_spl.symbolic.evaluate import ExactEvaluator, trace_distance  # noqa: F401
from baseline_spl.symbolic.ir import Term
from baseline_spl.symbolic.lattice import EMPTY, LatticeBudgetExceeded

Cell = Tuple[int, int, int]

DEFAULT_MAXIMUM_FRONTIER = 5

# Round 1's acceptance rule, used whenever a caller does not pass an evaluator. Keeping it as
# the default is what lets every existing call site and test stay unchanged.
DEFAULT_EVALUATOR = ExactEvaluator()


@dataclass
class SearchTask:
    '''One synthesis task: a concept, and one (parameter, placement cells) example per demo.

    A program with the parameter hard-coded satisfies one example and fails the others, which
    is what makes induction the search objective rather than a hope.
    '''
    name: str
    examples: List[Tuple[int, List[Cell]]]
    param_name: str = "length"
    demo_ids: Tuple[str, ...] = ()
    # The natural-language instruction, carried for B3-b's LLM proposer. B3-a never reads it:
    # enumeration is language-free, which is the whole point of the control.
    instruction: str = ""
    # 'lattice'    -> each example's target is a list of integer cells
    # 'continuous' -> each example's target is a list of centroids in metres
    observation_mode: str = "lattice"
    # One SRN table per example, aligned by index, for the probabilistic evaluator. Empty
    # under 'lattice', where nothing reads it.
    srn_tables: List[dict] = field(default_factory=list)
    # False: the program takes the structure's size as an argument and must work for every
    #        example (`tint -> tstate -> tstate`). This is our concept-level framing.
    # True:  the program is CLOSED (`tstate -> tstate`) and builds one specific structure,
    #        with the size appearing only as a literal. This is DreamCoder as published --
    #        its tower tasks are `ttower -> ttower`, and "arch leg 1" ... "arch leg 8" are
    #        eight separate problems rather than one parameterised one.
    closed: bool = False

    @property
    def request(self):
        '''The type enumeration must produce for this task.'''
        from baseline_spl.symbolic.bridge import CONCEPT_REQUEST, DEMO_REQUEST

        return DEMO_REQUEST if self.closed else CONCEPT_REQUEST

    def table_for(self, index: int) -> Optional[dict]:
        return self.srn_tables[index] if index < len(self.srn_tables) else None

    @property
    def parameters(self) -> List[int]:
        return [n for n, _ in self.examples]


@dataclass
class Solution:
    '''What the search found for one task.

    `frontier` holds up to `maximum_frontier` exact solutions, best (lowest MDL) first --
    DreamCoder's `maximumFrontier`. `program` and `log_prior` are the best of them, so callers
    that want a single program are unaffected.
    '''
    task: str
    frontier: List[Tuple[float, Program]] = field(default_factory=list)
    term: Optional[Term] = None
    distance: float = 1.0          # 0.0 == exact
    seconds: float = 0.0
    programs_tried: int = 0
    _approximate: Optional[Tuple[float, Program]] = None

    @property
    def exact(self) -> bool:
        return bool(self.frontier)

    @property
    def program(self) -> Optional[Program]:
        if self.frontier:
            return self.frontier[0][1]
        return self._approximate[1] if self._approximate else None

    @property
    def log_prior(self) -> float:
        if self.frontier:
            return self.frontier[0][0]
        return self._approximate[0] if self._approximate else float("-inf")

    @property
    def programs(self) -> List[Program]:
        '''Every exact solution, best first. This is what compression should be given.'''
        return [p for _prior, p in self.frontier]

    @property
    def status(self) -> str:
        if self.exact:
            return "solved"
        if self.program is not None:
            return "approximate"
        return "unsolved"

    def add_exact(self, prior: float, program: Program, maximum_frontier: int,
                  distance: float = 0.0) -> bool:
        '''Record an accepted solution. Returns True if it was new.

        `distance` is the accepting evaluator's statistic. Under exact match it is always
        0.0 and carries no information, but under a probabilistic evaluator it is the accept
        *margin* -- how far inside the threshold the program landed. That is what makes a
        near-threshold acceptance auditable rather than indistinguishable from a perfect one,
        which is the whole risk a soft evaluator introduces.
        '''
        source = str(program)
        if any(str(p) == source for _pr, p in self.frontier):
            return False
        self.frontier.append((prior, program))
        self.frontier.sort(key=lambda entry: -entry[0])     # highest prior = lowest MDL first
        del self.frontier[maximum_frontier:]
        self.distance = distance
        return True

    def add_approximate(self, prior: float, program: Program, distance: float) -> None:
        if self.exact:
            return
        if self._approximate is None or distance < self.distance:
            self._approximate = (prior, program)
            self.distance = distance


@dataclass
class SearchStats:
    programs_enumerated: int = 0
    seconds: float = 0.0
    max_mdl_reached: float = 0.0
    per_task: Dict[str, Solution] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.programs_enumerated / self.seconds if self.seconds else 0.0


def run_program(fn, param: int) -> Optional[List[Cell]]:
    '''Apply an evaluated program to one parameter under the lattice executor.
    None if it cannot run. Kept for callers that want cells regardless of the evaluator.'''
    try:
        return fn(param)(EMPTY).positions
    except (LatticeBudgetExceeded, RecursionError, IndexError, ValueError, TypeError,
            ZeroDivisionError, KeyError, OverflowError):
        return None
    except Exception:  # noqa: BLE001 - an enumerated program may fail in any way at all
        return None


def score(program: Program, task: SearchTask, evaluator=None) -> Optional[float]:
    '''The evaluator's badness statistic for a program on a task; lower is better, and None
    means the program could not run.

    Defaults to the exact lattice evaluator so every Round 1 caller keeps its behaviour: with
    that evaluator 0.0 still means "reproduced every example exactly".
    '''
    return (evaluator or DEFAULT_EVALUATOR).score(program, task)


def accepts(value: Optional[float], evaluator=None) -> bool:
    '''Whether a statistic clears the evaluator's acceptance threshold.'''
    return (evaluator or DEFAULT_EVALUATOR).accepts(value)


# --------------------------------------------------------------------------------------- #
# Band partitioning
# --------------------------------------------------------------------------------------- #

def slices(lower: float, upper: float, n: int) -> List[Tuple[float, float]]:
    '''Partition (lower, upper] into n contiguous half-open slices.

    Exactness matters: `Grammar.enumeration` yields programs with lowerBound < MDL <= upperBound,
    so abutting slices cover the band with no duplicates and no gaps. The final slice ends on
    `upper` exactly rather than on an accumulated float, which would drop programs at the edge.
    '''
    if n <= 1 or upper <= lower:
        return [(lower, upper)]
    step = (upper - lower) / n
    out = [(lower + i * step, lower + (i + 1) * step) for i in range(n)]
    out[-1] = (out[-1][0], upper)
    return out


# --------------------------------------------------------------------------------------- #
# Worker side. Programs cross the process boundary as source strings: `Primitive.GLOBALS` is
# inherited through fork, so re-parsing in the parent is exact and avoids pickling closures.
# --------------------------------------------------------------------------------------- #

def _enumerate_unit(args):
    '''Enumerate one (grammar, tasks) unit over one MDL band. Returns plain data only.

    A unit is a grammar plus the tasks that share it. With one global grammar there is a
    single unit covering every task; with per-task grammars from the recognition model there
    is one unit per task, and those run in parallel.
    '''
    grammar, tasks, lower, upper, slice_seconds, hard_deadline, soft, evaluator, keep = args
    from dreamcoder.type import Context

    # Each unit gets its own share of the clock, measured from when IT starts, capped by the
    # phase's hard deadline. Previously every unit carried the same ABSOLUTE deadline, which
    # is correct only while units <= workers: beyond that the first batch consumed the entire
    # budget and every later unit broke out immediately, reporting "unsolved" without having
    # enumerated a single program. Invisible at 16 concepts on 16 workers; at 100 concepts it
    # would silently starve most of the task set.
    deadline = min(time.time() + slice_seconds, hard_deadline)

    evaluator = evaluator or DEFAULT_EVALUATOR
    # Bound on how many accepted programs to carry back per task. The parent keeps only
    # `maximum_frontier` of them anyway (`Solution.add_exact` sorts by prior and truncates),
    # so returning more is pure waste -- and at a long budget it is ruinous waste. Every
    # enumerated program that satisfies an easy task used to be appended here with its full
    # source string, unbounded for the whole band. At 1,200 s iterations that was invisible;
    # at 14,400 s across 16 tasks it reached 51 GB RSS and got a sibling run OOM-killed.
    # `misses` was always bounded (it keeps only the best), which is why only `hits` grew.
    keep = max(1, int(keep))
    compact_at = keep * 8            # amortise the sort rather than running it per append
    count = 0
    hits: Dict[str, List[Tuple[float, str, float]]] = {}
    misses: Dict[str, Tuple[float, float, str]] = {}

    def _record(name: str, entry) -> None:
        bucket = hits.setdefault(name, [])
        bucket.append(entry)
        if len(bucket) >= compact_at:
            bucket.sort(key=lambda e: -e[0])     # highest prior == lowest MDL first
            del bucket[keep:]

    # Enumerate at the request type the tasks actually want. A unit's tasks share a grammar,
    # and in practice a run is either all concept-level or all demo-level, so the first task
    # settles it. Hard-coding CONCEPT_REQUEST here is what made `task_granularity="demo"`
    # a no-op: it produced one task per demonstration but still demanded a parameterised
    # program, so it never tested DreamCoder's closed-program framing at all.
    request = tasks[0].request if tasks else CONCEPT_REQUEST
    for prior, _context, program in grammar.enumeration(
            Context.EMPTY, [], request,
            maximumDepth=99, upperBound=upper, lowerBound=lower):
        count += 1
        if count % 512 == 0 and time.time() > deadline:
            break
        for task in tasks:
            distance = evaluator.score(program, task)
            if distance is None:
                continue
            if evaluator.accepts(distance):
                _record(task.name, (prior, str(program), distance))
            elif soft:
                best = misses.get(task.name)
                if best is None or distance < best[0]:
                    misses[task.name] = (distance, prior, str(program))
    for bucket in hits.values():                 # final compaction before crossing processes
        bucket.sort(key=lambda e: -e[0])
        del bucket[keep:]
    return count, hits, misses


# --------------------------------------------------------------------------------------- #

def _units(grammar, tasks: Sequence[SearchTask]):
    '''Group tasks by the grammar they will be searched under.

    A single Grammar gives one unit covering every task (no redundant re-enumeration). A
    {task name: Grammar} map -- what the recognition model produces -- gives one unit per
    distinct grammar, and those are what parallelise.
    '''
    if not isinstance(grammar, dict):
        return [(grammar, list(tasks))]
    grouped: Dict[int, Tuple[object, List[SearchTask]]] = {}
    for task in tasks:
        g = grammar[task.name]
        grouped.setdefault(id(g), (g, []))[1].append(task)
    return list(grouped.values())


def wake(grammar, tasks: Sequence[SearchTask], *, timeout: float = 60.0,
         max_mdl: float = 100.0, budget_increment: float = 1.5,
         soft_frontier: bool = True, maximum_frontier: int = DEFAULT_MAXIMUM_FRONTIER,
         cpus: int = 1, evaluator=None, log=None) -> SearchStats:
    '''Enumerate until every task's frontier is full or the budget is spent.

    `grammar` is either one Grammar shared by all tasks, or {task name: Grammar} from the
    recognition model. Parallelism is across those units; see the module docstring for why it
    is not across MDL sub-bands.
    '''
    stats = SearchStats()
    stats.per_task = {t.name: Solution(task=t.name) for t in tasks}
    started = time.time()
    deadline = started + timeout
    lower, budget = 0.0, budget_increment

    units = _units(grammar, tasks)
    pool = None
    if cpus > 1 and len(units) > 1:
        pool = _make_pool(min(cpus, len(units)), log)

    try:
        while time.time() < deadline and budget <= max_mdl:
            # Keep going while any task still has room in its frontier, exactly as
            # `enumerateForTasks` does (enumeration.py:559). Stopping at the first solution
            # per task would cap every frontier at one program, leaving STITCH nothing to
            # anti-unify beyond a single sample and the recognition model one example.
            if all(len(s.frontier) >= maximum_frontier for s in stats.per_task.values()):
                break

            # Fair share: with more units than workers the pool runs ceil(units/workers)
            # waves, so each unit may use at most that fraction of the time left.
            workers = min(cpus, len(units)) if pool is not None else 1
            waves = max(1, math.ceil(len(units) / max(1, workers)))
            remaining = max(0.0, deadline - time.time())
            slice_seconds = remaining / waves
            work = [(g, unit_tasks, lower, budget, slice_seconds, deadline, soft_frontier,
                     evaluator, maximum_frontier)
                    for g, unit_tasks in units]

            if pool is None:
                results = [_enumerate_unit(unit) for unit in work]
            else:
                results = pool.map(_enumerate_unit, work)

            for count, hits, misses in results:
                stats.programs_enumerated += count
                for name, found in hits.items():
                    solution = stats.per_task[name]
                    for prior, source, distance in found:
                        program = _parse(source)
                        if program is not None:
                            solution.add_exact(prior, program, maximum_frontier, distance)
                for name, (distance, prior, source) in misses.items():
                    program = _parse(source)
                    if program is not None:
                        stats.per_task[name].add_approximate(prior, program, distance)

            for name, solution in stats.per_task.items():
                if solution.exact and not solution.seconds:
                    solution.seconds = time.time() - started
                    solution.programs_tried = stats.programs_enumerated
                    if log:
                        log(f"  solved {name} at MDL {-solution.log_prior:.1f} "
                            f"after {solution.programs_tried:,} programs "
                            f"({solution.seconds:.1f}s, frontier {len(solution.frontier)})")

            stats.max_mdl_reached = budget
            lower, budget = budget, budget + budget_increment
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    stats.seconds = time.time() - started
    for solution in stats.per_task.values():
        if solution.term is None and solution.program is not None:
            try:
                solution.term = to_term(solution.program, concept_level=not _closed_run(tasks))
            except Exception:  # noqa: BLE001
                solution.term = None
    return stats


def _closed_run(tasks) -> bool:
    '''Whether this run's tasks are closed programs (demo-level).'''
    return bool(tasks) and bool(getattr(list(tasks)[0], "closed", False))


def _parse(source: str) -> Optional[Program]:
    try:
        return Program.parse(source)
    except Exception:  # noqa: BLE001
        return None


def _make_pool(cpus: int, log):
    '''A fork-context pool sharing the parent's Primitive.GLOBALS.

    fork rather than spawn so workers inherit the primitive registry and the invented
    abstractions parsed at runtime; the alternative is re-registering them per worker. Safe
    here because the search runs before any CUDA context exists (SPL builds its network on
    CPU). Falls back to serial if the pool cannot be created.
    '''
    import multiprocessing

    try:
        return multiprocessing.get_context("fork").Pool(processes=cpus)
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"  could not start a worker pool ({exc}); enumerating serially")
        return None


def default_cpus() -> int:
    '''Conservative default: this is often a shared machine, so do not take every core.'''
    return max(1, min(32, (os.cpu_count() or 1)))
