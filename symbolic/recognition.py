'''
recognition.py

Sleep-R: the neural recognition model that conditions the grammar on the task.

This is the half of DreamCoder's sleep phase that abstraction/compression is not. After each
wake it trains `Q(program | task)` on the solved frontiers plus *fantasies* -- programs
sampled from the current library and executed to produce synthetic tasks -- and at the next
wake each task is enumerated under its own `grammarOfTask(task)` rather than one global
grammar.

`dreamcoder/recognition.py` is pure Python + torch and is reused as-is; all this module
supplies is the two domain-specific pieces it cannot know:

  a lexicon    the symbols a task's I/O can contain
  tokenize     how to render a task's examples as a sequence over that lexicon

Two things worth knowing, both discovered by reading the upstream code rather than assumed:

1. **Dreaming is free.** `RecognitionModel.train(..., helmholtzRatio > 0)` samples Helmholtz
   programs *itself* when no `helmholtzFrontiers` are supplied (recognition.py:1264-1266),
   driving them through `featureExtractor.taskOfProgram` (recognition.py:1328). Writing a
   separate dream loop would duplicate that and diverge from DreamCoder.

2. **The example shape is load-bearing.** `RecurrentFeatureExtractor.__init__` builds
   `argumentsWithType` from `request.functionArguments()`, and the inherited `taskOfProgram`
   samples from it to build dream tasks. Our request is `tint -> tstate -> tstate`, so a task's
   examples must be `((parameter, state), state)` -- not `((parameter,), cells)`. Get that
   wrong and dreaming silently produces nothing.

Honest caveat for the paper: with 16 concepts the recognizer trains almost entirely on
fantasies. That is not a compromise -- Helmholtz samples dominate DreamCoder's training signal
in every published domain -- but it should be stated.
'''

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.symbolic._dreamcoder import Frontier, FrontierEntry, Program, Task
from baseline_spl.symbolic.bridge import CONCEPT_REQUEST
from baseline_spl.symbolic.lattice import EMPTY, LatticeState
from baseline_spl.symbolic.search import SearchTask

# Inter-placement deltas seen in this domain: the six unit steps, plus the doubled horizontal
# step `pins` takes and the diagonal pairs. Anything outside this is rendered as OTHER rather
# than growing the lexicon at runtime, which would break a trained model.
_UNIT_DELTAS = {
    (0, 0, 1): "UP", (0, 0, -1): "DOWN",
    (0, 1, 0): "LEFT", (0, -1, 0): "RIGHT",
    (-1, 0, 0): "FRONT", (1, 0, 0): "BEHIND",
}
MAX_PARAMETER_TOKEN = 24

# Cap on placements in a sampled "dream", mirroring tower's `len(pl) > 100` guard. A dream
# that builds thousands of blocks is not a useful training signal and costs real time to
# tokenize.
MAX_DREAM_PLACEMENTS = 100


def delta_tokens(previous, current) -> List[str]:
    '''Name the step between two placements, one token per axis that moved.

    Deltas rather than absolute cells: a structure built two cells to the left is the same
    structure, and encoding it as such keeps the lexicon small and the encoder
    translation-invariant.

    **One token per moving axis, not one per step.** Naming only the dominant axis makes
    diagonals collide -- (+1,-1,0) and (+1,+1,0) tie, so the second axis is lost and
    `diagonal_45` encodes identically to `diagonal_135`. The recognizer then cannot tell
    those tasks apart however long it trains. Per-axis costs nothing: every token emitted
    here is already in the lexicon.
    '''
    d = (current[0] - previous[0], current[1] - previous[1], current[2] - previous[2])
    if d in _UNIT_DELTAS:
        return [_UNIT_DELTAS[d]]
    out: List[str] = []
    for axis in range(3):
        if d[axis] == 0:
            continue
        unit = tuple(1 if i == axis and d[axis] > 0 else
                     -1 if i == axis else 0 for i in range(3))
        name = _UNIT_DELTAS.get(unit, "OTHER")
        magnitude = min(abs(d[axis]), 4)
        out.append(name if magnitude == 1 else f"{name}x{magnitude}")
    return out or ["SAME"]


def delta_token(previous, current) -> str:
    '''Single-token form, kept for callers that want one symbol. Prefer `delta_tokens`.'''
    return delta_tokens(previous, current)[0]


def lexicon() -> List[str]:
    '''Every symbol `tokenize` can emit. Fixed up front, so a trained model stays valid.'''
    symbols = ["START", "PLACE", "SAME", "OTHER"]
    symbols += list(_UNIT_DELTAS.values())
    # k starts at 1, not 2: a diagonal step such as (0,-1,1) has a dominant-axis
    # magnitude of 1 while not being a unit delta, so it renders as e.g. RIGHTx1.
    symbols += [f"{n}x{k}" for n in _UNIT_DELTAS.values() for k in range(1, 5)]
    symbols += [f"n={i}" for i in range(1, MAX_PARAMETER_TOKEN + 1)]
    # "n=big" for a value past the cap; "n=none" for a fixed-size concept, which takes no
    # integer at all. An unknown symbol is fatal to the extractor, not merely unhelpful.
    symbols += ["n=big", "n=none"]
    return sorted(set(symbols))


def _parameter_token(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "n=big"
    return f"n={n}" if 1 <= n <= MAX_PARAMETER_TOKEN else "n=big"


def _parameter_tokens(parameter) -> List[str]:
    '''One token per integer argument, positional.

    A concept with two arguments encoded as a single token would make `rectangle(5, 3)` and
    `rectangle(5, 4)` look identical to the model, so it could not learn that the second
    argument matters. Positional rather than named because the request is positional.
    '''
    from baseline_spl.symbolic.ir import args as _args

    return [_parameter_token(v) for v in _args(parameter)] or ["n=none"]


def encode(parameter, state) -> Optional[List[str]]:
    '''One example -> a token sequence. None rejects the example.'''
    positions = getattr(state, "positions", None)
    if positions is None:
        return None
    tokens = ["START"] + _parameter_tokens(parameter)
    previous = None
    for cell in positions:
        if previous is not None:
            tokens.extend(delta_tokens(previous, cell))
        tokens.append("PLACE")
        previous = cell
    return tokens


def _scale(table) -> Tuple[float, float, float]:
    '''The unit step implied by an SRN table: the mean displacement of one shift.'''
    horizontal = abs(table["right"][0][1]) or abs(table["left"][0][1]) or 0.109
    vertical = abs(table["top"][0][2]) or 0.05
    return (horizontal, horizontal, vertical)


def quantize(positions: Sequence, table) -> List[Tuple[int, int, int]]:
    '''Continuous centroids -> integer cells, for the tokenizer only.

    The recognition model conditions the grammar; it does not decide correctness. So
    discretizing here costs nothing that Round 2 was trying to protect -- acceptance stays
    continuous and probabilistic -- while keeping the lexicon a small fixed vocabulary. A
    lexicon that grew with float values would make a trained recognizer invalid the moment
    the data shifted.

    The unit comes from the SRN table itself rather than from a separately estimated pitch,
    so the tokenizer cannot drift from the executor.
    '''
    scale = _scale(table)
    return [tuple(int(round(p[i] / scale[i])) for i in range(3)) for p in positions]


def build_task(name: str, examples: Sequence[Tuple[int, Sequence]], task=None) -> Task:
    '''A DreamCoder Task whose examples match `tint -> tstate -> tstate`.

    The inputs are `(parameter, empty state)` and the output is the finished state, so that
    `argumentsWithType` gets both a tint and a tstate to sample from when dreaming.
    '''
    if task is not None and getattr(task, "observation_mode", "lattice") == "continuous":
        converted = []
        for index, (parameter, positions) in enumerate(examples):
            table = task.table_for(index)
            converted.append((parameter, quantize(positions, table) if table else positions))
        examples = converted

    closed = bool(task is not None and getattr(task, "closed", False))
    request = task.request if task is not None else CONCEPT_REQUEST

    pairs = []
    for parameter, cells in examples:
        final = LatticeState(focus=EMPTY.focus,
                             placements=tuple((i, tuple(c)) for i, c in enumerate(cells)),
                             cursor=len(cells))
        # The input tuple must match the request's argument list, because
        # `RecurrentFeatureExtractor` builds `argumentsWithType` from
        # `request.functionArguments()` and `taskOfProgram` samples from it when dreaming.
        # One entry per integer argument, then the state; a closed task takes only the state.
        from baseline_spl.symbolic.ir import args as _args

        inputs = () if closed else tuple(_args(parameter))
        pairs.append((inputs + (EMPTY,), final))
    return Task(name, request, pairs)


def build_tasks(tasks: Sequence[SearchTask]) -> Dict[str, Task]:
    return {t.name: build_task(t.name, t.examples, t) for t in tasks}


def make_extractor(tasks: Sequence[Task], hidden: int = 32, cuda: bool = False):
    '''The feature extractor, defined here so `tasks` and `lexicon` are bound together.'''
    from dreamcoder.recognition import RecurrentFeatureExtractor

    class SPLFeatureExtractor(RecurrentFeatureExtractor):
        '''Encodes a construction task as the parameter plus the sequence of steps taken.

        Only `tokenize` is domain-specific; `outputDimensionality`, `featuresOfTask`,
        `taskOfProgram` and the parallel dreaming machinery are all inherited.
        '''

        special = None
        recomputeTasks = True

        def __init__(self):
            # bidirectional=True is required, not a choice: `examplesEncoding` returns
            # `hidden[0] + hidden[1]` (recognition.py:1921), which needs two directions
            # to index. With a single direction it raises IndexError. The sum collapses
            # them back to H, so outputDimensionality == H stays correct.
            super().__init__(lexicon=lexicon(), tasks=list(tasks), H=hidden,
                             bidirectional=True, cuda=cuda)

        def tokenize(self, examples):
            out = []
            for xs, y in examples:
                if not isinstance(xs, (list, tuple)) or not xs:
                    return None
                encoded = encode(xs[0], y)
                if encoded is None:
                    return None
                # The base class embeds every symbol, so an unknown one is fatal rather than
                # merely unhelpful. Reject the example instead.
                if any(token not in self.symbolToIndex for token in encoded):
                    return None
                # `xs` is (arg_0, ..., arg_{k-1}, state), so everything but the state is an
                # argument. At k = 0 that is the empty tuple, which becomes ["n=none"].
                out.append(((_parameter_tokens(tuple(xs[:-1])),), encoded))
            return out

        def taskOfProgram(self, p, tp):
            '''Turn a sampled (Helmholtz / "dream") program into a task.

            This overrides `RecurrentFeatureExtractor.taskOfProgram`, and it has to, for a
            reason specific to construction domains. Upstream's generic version samples
            inputs of each argument type and then rejects the dream if every example gives
            the same output:

                def is_not_degenerate_outputs(examples):
                    outputs = [y for (xs, y) in examples]
                    return not (all(y == outputs[0] for y in outputs))

            That needs the inputs to vary. A CLOSED program is `tstate -> tstate`, its only
            argument is the start state, and in this domain every task starts from the same
            empty state -- so a deterministic program returns the same output every time and
            **every** closed dream is degenerate by construction. Measured: "Got 0/500 valid
            samples ... 0 Helmholtz entries", every round, after which upstream retries
            forever rather than giving up, so the run hangs instead of failing.

            DreamCoder hits exactly this in its own closed domain and solves it the same way:
            `TowerCNN.taskOfProgram` (domains/tower/main.py) ignores input sampling entirely,
            runs the program once, and judges the *output* --

                pl = executeTower(p, 0.05)
                if pl is None or (not lenient and len(pl) == 0): return None
                if len(pl) > 100 or towerLength(pl) > 360: return None
                return SupervisedTower("tower dream", p)

            -- rejecting empty and runaway traces. This mirrors that. For the parameterised
            request the degeneracy check is meaningful (the parameter genuinely varies, and a
            program ignoring it *should* be rejected), so it is kept there.
            '''
            import random as _random

            closed = not tp.functionArguments() or len(tp.functionArguments()) == 1
            try:
                fn = p.evaluate([])
            except Exception:  # noqa: BLE001
                return None

            if closed:
                # Tower's contract: one run, judge the trace.
                try:
                    state = fn(EMPTY)
                except Exception:  # noqa: BLE001
                    return None
                positions = getattr(state, "positions", None)
                if not positions or len(positions) > MAX_DREAM_PLACEMENTS:
                    return None
                examples = [((EMPTY,), state)]
            else:
                # One pool per integer argument, so a 2-argument dream varies BOTH. Sampling
                # only the first would teach the model that the second never matters.
                arg_types = tp.functionArguments()[:-1]
                pools = [list(self.argumentsWithType.get(t) or [3, 4, 5]) for t in arg_types]
                draws = [_random.sample(p, min(3, len(p))) for p in pools]
                examples = []
                for combination in zip(*draws) if len(draws) > 1 else [(v,) for v in draws[0]]:
                    try:
                        state = fn
                        for value in combination:
                            state = state(value)
                        state = state(EMPTY)
                    except Exception:  # noqa: BLE001
                        return None
                    positions = getattr(state, "positions", None)
                    if not positions or len(positions) > MAX_DREAM_PLACEMENTS:
                        return None
                    examples.append((tuple(combination) + (EMPTY,), state))
                # Upstream's check, kept where it means something: a program that ignores its
                # parameter produces the same structure at every size and teaches nothing.
                outputs = [y.positions for _xs, y in examples]
                if len(outputs) > 1 and all(o == outputs[0] for o in outputs):
                    return None

            if self.tokenize(examples) is None:
                return None
            return Task("dream", tp, examples)

    return SPLFeatureExtractor()


def frontiers_for(tasks: Dict[str, Task], solutions, grammar) -> List[Frontier]:
    '''DreamCoder Frontiers from the search's solutions, for the replay half of training.'''
    from baseline_spl.symbolic.stitch_bridge import eta_long

    out = []
    for name, solution in solutions.items():
        if not solution.exact or name not in tasks:
            continue
        entries = []
        for prior, program in solution.frontier:
            normalised = eta_long(program)
            if normalised is None:
                continue
            entries.append(FrontierEntry(program=normalised, logLikelihood=0.0,
                                         logPrior=prior))
        if entries:
            out.append(Frontier(entries, task=tasks[name]))
    return out


def train_recognizer(grammar, tasks: Dict[str, Task], solutions, *,
                     epochs: Optional[int] = None, steps: int = 10000,
                     helmholtz_ratio: float = 0.5, hidden: int = 64, cpus: int = 1,
                     timeout: Optional[float] = 1800.0, contextual: bool = True,
                     bias_optimal: bool = True, auxiliary_loss: bool = True, log=None):
    '''Train the recognition model. Returns it, or None if training could not run.

    A failure here must not end the run -- the loop simply continues with the global grammar
    -- but it is reported rather than swallowed, because a recognizer that silently never
    trains looks exactly like one that does not help.

    `steps` is the budget that matters; upstream defaults both `steps` and `epochs` to
    9,999,999 and stops on `timeout` (`recognition.py:1269-1272`). Passing a small `epochs`
    silently caps training at `epochs x len(frontiers)` gradient steps, and an undertrained
    network still emits a different grammar per task, so nothing looks wrong. See
    `configs/default.py:recognition_steps`.
    '''
    from dreamcoder.recognition import RecognitionModel

    try:
        extractor = make_extractor(list(tasks.values()), hidden=hidden)
        # `hidden` must match the extractor's output width. This is not a tuning choice:
        # `frontierKL` computes `self._MLP(features).expand(1, features.size(-1))`
        # (recognition.py:1124), which assumes the MLP preserves width. RecognitionModel's
        # default hidden=[64] against a 32-wide extractor therefore raises
        # "expanded size (32) must match the existing size (64)". Upstream never hits it
        # because their extractors happen to be 64 wide.
        # LILO's own recognition loader sets contextual=True, bias_optimal=True and
        # auxiliary_loss=True (src/models/laps_dreamcoder_recognition.py:118,152-165). We had
        # all three at the upstream *library* defaults -- contextual=False, biasOptimal=None,
        # auxLoss=False -- which is a strictly weaker model than the one they run.
        #
        # contextual=True is the substantive one: it predicts a bigram transition matrix over
        # productions (conditioned on the parent production and argument index) instead of
        # marginal unigram weights, so `grammarOfTask` can express "inside a loop body, prefer
        # place" rather than only "prefer place".
        recognizer = RecognitionModel(example_encoder=extractor, grammar=grammar,
                                      hidden=[hidden], contextual=contextual)
        replays = frontiers_for(tasks, solutions, grammar)
        # With nothing solved yet there are no replay frontiers, and upstream's loop then
        # takes `permutedFrontiers = list(frontiers)` -- an empty list -- so epoch 1 does zero
        # gradient steps and its logging line calls `mean([])`. That is an upstream
        # precedence bug: `if (i == 1 or i % 10 == 0) or (totalGradientSteps % 10 == 0) and
        # losses` parses as `A or B or (C and D)`, so the `and losses` guard never protects
        # the `i == 1` case (recognition.py:1609).
        #
        # Training purely on fantasies is the right behaviour here and is what DreamCoder
        # does with an empty frontier set: ratio >= 1.0 selects `permutedFrontiers = [None]`,
        # which drives the Helmholtz sampler instead.
        if not replays:
            # Nothing has been solved yet, so there is nothing to train on but fantasies --
            # and upstream cannot survive that path here. With no real frontiers its loop
            # takes `permutedFrontiers = list(frontiers)` (empty), completes zero gradient
            # steps, and then divides by `len(classificationLosses)` == 0
            # (recognition.py:1641). Its epoch-1 logging is unguarded for the same reason:
            # `if (i == 1 or i % 10 == 0) or (totalGradientSteps % 10 == 0) and losses`
            # parses as `A or B or (C and D)`, so `and losses` never protects `i == 1`.
            #
            # This only arises because our loop follows LILO's order -- propose, train
            # recognition, then enumerate -- so Sleep-R is reached before the first wake has
            # solved anything. DreamCoder's own order is wake -> Sleep-G -> Sleep-R, where
            # frontiers always exist by the time the recognizer trains. Skipping the
            # degenerate first call restores that guarantee without reordering the loop.
            if log:
                log("  no solved frontiers yet; skipping recognition this iteration "
                    "(nothing to condition on until the first wake succeeds)")
            return None
        ratio = helmholtz_ratio
        if log:
            log(f"  training recognizer on {len(replays)} replay frontier(s) "
                f"+ fantasies (helmholtzRatio={ratio})")
        # `steps` is the binding budget, as upstream intends; `epochs=None` lets it run.
        request = next(iter(tasks.values())).request if tasks else CONCEPT_REQUEST
        recognizer.train(replays, epochs=epochs, steps=steps, helmholtzRatio=ratio,
                         CPUs=cpus, timeout=timeout, defaultRequest=request,
                         biasOptimal=bias_optimal, auxLoss=auxiliary_loss)
        return recognizer
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"  recognition training failed ({type(exc).__name__}: {exc}); "
                f"continuing with the global grammar")
        return None


def grammars_for(recognizer, tasks: Dict[str, Task], fallback, log=None) -> Dict:
    '''{task name: grammar} from the recognizer, falling back per task on failure.'''
    out = {}
    failures = 0
    for name, task in tasks.items():
        try:
            out[name] = recognizer.grammarOfTask(task).untorch()
        except Exception:  # noqa: BLE001
            out[name] = fallback
            failures += 1
    if failures and log:
        log(f"  {failures}/{len(tasks)} task(s) fell back to the global grammar")
    return out
