'''
bridge.py

Translation between DreamCoder's `Program` (lambda calculus, de Bruijn indices) and this
domain's `Term` ADT, plus the typed grammar the enumerator searches.

    Grammar.enumeration  ->  Program  --to_term-->  Term  --lower-->  concept class
                             Program  <-from_term-  Term
                                |
                                +-- Grammar.json / stitch.from_dreamcoder / Invented.parse

Both directions are needed: `to_term` reads what the search found, and `from_term` puts the
oracle terms into DreamCoder syntax so STITCH can compress them and so the round trip can be
tested.

Why the translation is total: `tstate` appears only in the final argument position of every
primitive and no primitive consumes two states, so every well-typed term is a linear chain of
state transformers. There is no way to enumerate a program this cannot read.

The primitive set is deliberately DreamCoder's own tower DSL
(dreamcoder/domains/tower/towerPrimitives.py:145-170): `loop` is `tower_loopM` and `saved` is
`tower_embed`, with the same types. The one addition is integer arithmetic, which tower did
not need because its tasks bake the size in as a literal; ours make it a parameter.
'''

from __future__ import annotations

from typing import Dict, List, Tuple, Union

# Must precede any other dreamcoder import -- see _dreamcoder.py for why.
from baseline_spl.symbolic._dreamcoder import (Abstraction, Application, Grammar, Index,
                                               Invented, Primitive, Program, arrow, baseType)
from baseline_spl.symbolic import lattice
from baseline_spl.symbolic.ir import (BinOp, Const, IntExpr, Loop, LoopVar, Param,
                                      Move, Place, Saved, Seq, Shift, Term,
                                      level_features)

tstate = baseType("tstate")
tint = baseType("tint")
tdir = baseType("tdir")

DIRECTIONS = ("left", "right", "front", "behind", "top")
INT_LITERALS = (1, 2)
OPS = {"+": "add", "-": "sub", "*": "mul"}


class TranslationError(Exception):
    '''A program that cannot be read as a Term. Should be unreachable for well-typed terms;
    the search treats it as "this program does not solve the task" rather than crashing.'''


# --------------------------------------------------------------------------------------- #
# Primitives. `.value` is a real closure, so DreamCoder can evaluate a program directly --
# test_bridge cross-checks that against the Term evaluator on every oracle.
# --------------------------------------------------------------------------------------- #

def _make_primitives() -> Dict[str, Primitive]:
    prims: Dict[str, Primitive] = {}

    prims["shift"] = Primitive("shift", arrow(tdir, tstate, tstate), lattice.shift)
    prims["move"] = Primitive("move", arrow(tdir, tint, tstate, tstate), lattice.move)
    prims["place"] = Primitive("place", arrow(tstate, tstate), lattice.place)
    prims["loop"] = Primitive(
        "loop", arrow(tint, arrow(tint, tstate, tstate), tstate, tstate), lattice.loop)
    prims["saved"] = Primitive(
        "saved", arrow(arrow(tstate, tstate), tstate, tstate), lattice.saved)

    for d in DIRECTIONS:
        prims[d.upper()] = Primitive(d.upper(), tdir, d)
    for k in INT_LITERALS:
        prims[str(k)] = Primitive(str(k), tint, k)
    for symbol, name in OPS.items():
        prims[name] = Primitive(name, arrow(tint, tint, tint), getattr(lattice, name))
    return prims


PRIMITIVES: Dict[str, Primitive] = _make_primitives()

# Which primitives each grammar level exposes. Mirrors ir.GRAMMAR_LEVELS.
_FEATURE_OF = {"shift": "shift", "place": "place", "loop": "loop", "saved": "saved",
               "add": "add", "sub": "sub", "mul": "mul"}


def extra_int_literals(upto: int) -> List[Primitive]:
    """Integer literals 3..`upto`, for closed (demo-level) tasks.

    DreamCoder's tower domain registers `Primitive(str(j), tint, j)` for j in 1..49, and it
    does so *because* its tasks are closed: "arch leg 7" needs the literal 7, and building it
    from {1, 2} by arithmetic would make every closed program dramatically dearer than the
    domain author intended.

    Our default grammar carries only {1, 2}, which is right for concept-level tasks -- the
    size arrives as an argument, so literals are only needed for small offsets. Applying that
    same grammar to closed tasks would measure our literal shortage rather than the framing,
    and would make demo-level look worse than concept-level for a reason that has nothing to
    do with DreamCoder.
    """
    return [Primitive(str(k), tint, k) for k in range(3, int(upto) + 1)]


def primitives_for(level: str = "standard", int_literals_upto: int = 0) -> List[Primitive]:
    features = level_features(level)
    chosen = []
    for name, prim in PRIMITIVES.items():
        feature = _FEATURE_OF.get(name)
        if feature is None:            # direction and integer literals are always available
            chosen.append(prim)
        elif feature in features:
            chosen.append(prim)
    if int_literals_upto and int_literals_upto > max(INT_LITERALS):
        chosen.extend(extra_int_literals(int_literals_upto))
    return chosen


def grammar(level: str = "standard", continuation: bool = True,
            int_literals_upto: int = 0) -> Grammar:
    '''The uniform grammar the search starts from.

    `continuation=True` sets continuationType, which every imperative DreamCoder domain uses:
    tower (`main.py:270`), LOGO (`makeLogoTasks.py:215`), regex, and -- most directly
    comparable -- LILO's own block-construction domain, which builds its grammar as
    `LAPSGrammar.uniform(towerPrimitives.primitives, continuationType=towerPrimitives.ttower)`
    (`data/structures/grammar.py`).

    It is ON by default because leaving it off was a fidelity gap, not a design choice: state
    threading became explicit, so every program carried argument structure their equivalents
    do not. Measured on the oracles, turning it on lowers description length (the diagonals
    go 15.4 -> 14.7 nats) while all 16 remain scoreable and enumeration throughput is
    unchanged. Cheaper programs at equal depth means strictly more reach for the same budget.
    '''
    prims = primitives_for(level, int_literals_upto)
    return Grammar.uniform(prims, continuationType=tstate if continuation else None)


CONCEPT_REQUEST = arrow(tint, tstate, tstate)
DEMO_REQUEST = arrow(tstate, tstate)


# --------------------------------------------------------------------------------------- #
# Program -> Term
# --------------------------------------------------------------------------------------- #

STATE = "state"
PARAM = "param"


def _spine(expr) -> Tuple[object, List]:
    '''Uncurry an application: f applied to [x1, x2, ...].'''
    args = []
    while isinstance(expr, Application):
        args.append(expr.x)
        expr = expr.f
    return expr, list(reversed(args))


def _head_name(head) -> str:
    if isinstance(head, (Primitive, Invented)):
        return head.name if isinstance(head, Primitive) else str(head)
    raise TranslationError(f"unexpected head {head}")


def _loop_depth(binders: List, ordinal: int) -> int:
    '''ir.LoopVar counts from the innermost, binders record absolute nesting.'''
    ordinals = [b[1] for b in binders if isinstance(b, tuple) and b[0] == "loop"]
    return ordinals.index(ordinal)


def _to_int(expr, binders: List) -> IntExpr:
    if isinstance(expr, Index):
        if expr.i >= len(binders):
            raise TranslationError(f"free variable ${expr.i}")
        tag = binders[expr.i]
        if tag == PARAM:
            return Param()
        if isinstance(tag, tuple) and tag[0] == "loop":
            return LoopVar(_loop_depth(binders, tag[1]))
        raise TranslationError(f"${expr.i} is {tag}, not an integer")

    head, args = _spine(expr)
    name = _head_name(head)
    if not args:
        try:
            return Const(int(name))
        except ValueError as exc:
            raise TranslationError(f"{name} is not an integer literal") from exc
    for symbol, op_name in OPS.items():
        if name == op_name:
            if len(args) != 2:
                raise TranslationError(f"{name} applied to {len(args)} arguments")
            return BinOp(symbol, _to_int(args[0], binders), _to_int(args[1], binders))
    raise TranslationError(f"{name} is not an integer expression")


def _to_state(expr, binders: List, next_ordinal: int) -> Tuple[Term, int]:
    '''Translate an expression of type tstate. Returns (term, next unused loop ordinal).'''
    if isinstance(expr, Index):
        if expr.i >= len(binders) or binders[expr.i] != STATE:
            raise TranslationError(f"${expr.i} is not the threaded state")
        return Seq(), next_ordinal                       # identity: the chain ends here

    head, args = _spine(expr)
    name = _head_name(head)

    if name == "place":
        if len(args) != 1:
            raise TranslationError("place takes one argument")
        prefix, ordinal = _to_state(args[0], binders, next_ordinal)
        return Seq(prefix, Place()), ordinal

    if name == "shift":
        if len(args) != 2:
            raise TranslationError("shift takes two arguments")
        direction = _head_name(_spine(args[0])[0])
        if direction.lower() not in DIRECTIONS:
            raise TranslationError(f"{direction} is not a direction")
        prefix, ordinal = _to_state(args[1], binders, next_ordinal)
        return Seq(prefix, Shift(direction.lower())), ordinal

    if name == "move":
        if len(args) != 3:
            raise TranslationError("move takes three arguments")
        direction = _head_name(_spine(args[0])[0])
        if direction.lower() not in DIRECTIONS:
            raise TranslationError(f"{direction} is not a direction")
        distance = _to_int(args[1], binders)
        prefix, ordinal = _to_state(args[2], binders, next_ordinal)
        return Seq(prefix, Move(direction.lower(), distance)), ordinal

    if name == "loop":
        if len(args) != 3:
            raise TranslationError("loop takes three arguments")
        count = _to_int(args[0], binders)
        body_fn = args[1]
        if not (isinstance(body_fn, Abstraction)
                and isinstance(body_fn.body, Abstraction)):
            raise TranslationError("loop body must be (lambda (lambda ...))")
        ordinal = next_ordinal
        inner = [STATE, ("loop", ordinal)] + binders     # innermost first: $0 state, $1 index
        body, after = _to_state(body_fn.body.body, inner, ordinal + 1)
        prefix, after = _to_state(args[2], binders, after)
        return Seq(prefix, Loop(count, body)), after

    if name == "saved":
        if len(args) != 2:
            raise TranslationError("saved takes two arguments")
        body_fn = args[0]
        if not isinstance(body_fn, Abstraction):
            raise TranslationError("saved body must be (lambda ...)")
        body, after = _to_state(body_fn.body, [STATE] + binders, next_ordinal)
        prefix, after = _to_state(args[1], binders, after)
        return Seq(prefix, Saved(body)), after

    raise TranslationError(f"{name} does not produce a state")


def to_term(program: Union[Program, str], concept_level: bool = True) -> Term:
    '''Read a DreamCoder program as a Term. Raises TranslationError if it is not a state chain.

    Once the library grows, solutions call invented abstractions (`#(lambda ...)`), which are
    not primitives and so have no direct Term form. Beta-normalising inlines them, after which
    only primitives remain. The direct translation is tried first because it preserves the
    program's structure; normalisation is the fallback rather than the default.
    '''
    if isinstance(program, str):
        program = Program.parse(program)
    try:
        return _to_term(program, concept_level)
    except TranslationError:
        try:
            inlined = program.betaNormalForm()
        except Exception as exc:  # noqa: BLE001
            raise TranslationError(f"could not inline abstractions: {exc}") from exc
        return _to_term(inlined, concept_level)


def _to_term(program: Program, concept_level: bool) -> Term:
    if concept_level:
        if not (isinstance(program, Abstraction) and isinstance(program.body, Abstraction)):
            raise TranslationError("expected (lambda (lambda ...)) for a concept-level task")
        binders = [STATE, PARAM]                          # $0 state, $1 the parameter
        body = program.body.body
    else:
        if not isinstance(program, Abstraction):
            raise TranslationError("expected (lambda ...) for a demo-level task")
        binders = [STATE]
        body = program.body
    term, _ = _to_state(body, binders, 0)
    return term


# --------------------------------------------------------------------------------------- #
# Term -> Program
# --------------------------------------------------------------------------------------- #

def _const_source(k: int) -> str:
    '''Only 1 and 2 are primitives, so other literals are built from them. Every oracle needs
    just 1 and 2; the search can reach the rest through arithmetic.'''
    if k in INT_LITERALS:
        return str(k)
    if k == 0:
        return "(sub 1 1)"
    if k < 0:
        return f"(sub (sub 1 1) {_const_source(-k)})"
    if k % 2 == 0:
        return f"(mul 2 {_const_source(k // 2)})"
    return f"(add 1 {_const_source(k - 1)})"


def _int_source(expr: IntExpr, binders: List) -> str:
    if isinstance(expr, Const):
        return _const_source(expr.value)
    if isinstance(expr, Param):
        return f"${binders.index(PARAM)}"
    if isinstance(expr, LoopVar):
        ordinals = [b[1] for b in binders if isinstance(b, tuple) and b[0] == "loop"]
        if expr.depth >= len(ordinals):
            raise TranslationError(f"LoopVar(depth={expr.depth}) outside any loop")
        return f"${binders.index(('loop', ordinals[expr.depth]))}"
    if isinstance(expr, BinOp):
        return (f"({OPS[expr.op]} {_int_source(expr.left, binders)} "
                f"{_int_source(expr.right, binders)})")
    raise TranslationError(f"cannot render {expr!r}")


def _state_source(term: Term, incoming: str, binders: List, ordinal: List[int]) -> str:
    if isinstance(term, Seq):
        for step in term.steps:
            incoming = _state_source(step, incoming, binders, ordinal)
        return incoming
    if isinstance(term, Place):
        return f"(place {incoming})"
    if isinstance(term, Shift):
        return f"(shift {term.direction.upper()} {incoming})"
    if isinstance(term, Move):
        return (f"(move {term.direction.upper()} "
                f"{_int_source(term.distance, binders)} {incoming})")
    if isinstance(term, Loop):
        count = _int_source(term.count, binders)
        mine = ordinal[0]
        ordinal[0] += 1
        inner = [STATE, ("loop", mine)] + binders
        body = _state_source(term.body, "$0", inner, ordinal)
        return f"(loop {count} (lambda (lambda {body})) {incoming})"
    if isinstance(term, Saved):
        body = _state_source(term.body, "$0", [STATE] + binders, ordinal)
        return f"(saved (lambda {body}) {incoming})"
    raise TranslationError(f"cannot render {type(term).__name__}")


def to_source(term: Term, concept_level: bool = True) -> str:
    '''Term -> DreamCoder s-expression source.'''
    if concept_level:
        body = _state_source(term, "$0", [STATE, PARAM], [0])
        return f"(lambda (lambda {body}))"
    body = _state_source(term, "$0", [STATE], [0])
    return f"(lambda {body})"


def from_term(term: Term, concept_level: bool = True) -> Program:
    return Program.parse(to_source(term, concept_level))
