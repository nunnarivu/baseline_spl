'''
dsl_prompt.py

The shared prompt furniture every baseline receives: documentation of SPL's action
DSL, the required class structure, and the initialized-sketch block.

Two rules keep the comparison honest:

  1. Every method sees the *same* DSL documentation and the *same* required class
     shape (``CONCEPT_STRUCTURE_PROMPT``, imported from SPL rather than copied, so the
     two can never drift). Baselines emit the same class SPL does — that is deliberate
     and generous: it is the only way ``run_predicted_program``, ``compute_plan_metrics``
     and ``check_program_equivalence`` can score them at all.
  2. Worked examples never use an evaluated concept. SPL's own
     ``CONCEPT_GENERALIZATION_PROMPT`` uses the invented concept ``rewot`` for exactly
     this reason; handing a baseline a worked ``row`` or ``tower`` would leak the answer
     and break the comparison in the generous direction. The example below uses the
     invented concept ``zigzag_lane``, which is in no split.
'''

from __future__ import annotations

from SPL.config.primitive_config import DEFAULT_ACTIONS
from SPL.model.generalize import GeneralizeAgent
from SPL.model.prompts import CONCEPT_STRUCTURE_PROMPT


DIRECTIONS = ", ".join(f'"{d}"' for d in DEFAULT_ACTIONS)


DSL_DOC = f'''# Action DSL available inside construct()

The environment keeps a single global FOCUS: a 3-D location (with uncertainty) where the
next object will be placed. A construction program moves the focus around and drops
objects at it. These four calls are the only primitives:

    assign_focus(position: list[float] = None, object_id: int = None)
        Move the focus to an absolute position, or onto an existing object's location.
        Exactly one of the two arguments is given.

    shift_focus(direction: str)
        Move the focus along the direction. The focus position will move along the direction and the uncertainity will be updated. 
        direction is one of: {DIRECTIONS}

    place_object_at_focus(object_id: int)
        Place the given object at the current focus. The focus does not move.
        Each object may be placed once; take them from the `objects` list in order.

    filter(*attributes) -> list[int]
        Return the ids of the objects in the scene matching attributes such as
        ("blue", "cube"). Already resolved for you in the initialized sketch below.

Coordinate conventions used in this domain:
  - "TOP" is the vertical axis;
  - "LEFT"/"RIGHT" and "FRONT"/"BEHIND" are the two horizontal axes.
  - Physics is real: an object placed with nothing under it will fall. Build supports
    before the things they hold up.
'''


WORKED_EXAMPLE = '''# Worked example (a made-up concept, purely to show the expected output shape)

Instruction: "Construct a zigzag_lane of length 3 using red cube"
Initialized sketch:
```python
zigzag_lane_1 = zigzag_lane(length=3, objects=[4, 5, 6, 7, 8])
zigzag_lane_1.construct()
```

Expected output:
```python
class zigzag_lane:
    def __init__(self, length: int, objects: list):
        self.length = length
        self.objects = list(objects) if objects is not None else []
        self.constructed = False

        # Book keeping
        self._plan = []
        self._placed = []
        self._substructures = []

    def construct(self):
        if self.constructed:
            raise Exception("Construct method has already been called for this instance.")
        self.constructed = True

        if len(self.objects) < self.length:
            raise ValueError(f"Not enough objects for a zigzag_lane of length {self.length}")

        for i in range(self.length):
            obj = self.objects.pop(0)
            place_object_at_focus(obj)
            self._plan.append(f"place_object_at_focus(object_id = {obj})")
            self._placed.append(obj)
            if i < self.length - 1:
                direction = "RIGHT" if i % 2 == 0 else "FRONT"
                shift_focus(direction)
                self._plan.append(f'shift_focus("{direction}")')

    @staticmethod
    def argument_sampler():
        for length in range(1, 1001):
            yield (length, None)

    @property
    def plan(self):
        return list(self._plan)

    @property
    def blocks(self):
        return list(self._placed)

    @property
    def substructures(self):
        return list(self._substructures)

    @property
    def key_blocks(self):
        if not self._placed:
            return []
        return [self._placed[0]] if len(self._placed) == 1 else [self._placed[0], self._placed[-1]]

    @property
    def actions(self):
        return [f"assign_focus(object_id = {b})" for b in self.key_blocks]
```
'''


OUTPUT_RULES = '''# Output rules
- Output ONLY the class definition inside a single ```python ... ``` block.
- No explanation, no reasoning, no example usage outside the block.
- The class name and its __init__ arguments MUST match the initialized sketch exactly.
- The class must define: __init__, construct, argument_sampler, plan, blocks,
  substructures, key_blocks, actions.
- construct() must generalize: it takes the numeric argument and works for ANY value of
  it, not only the one shown. never hard-code a fixed number of placements.
'''


class _LibraryRenderer:
    '''Borrows SPL's own renderer so the library is described to a baseline exactly as it
    is to SPL's Generalize stage, and cannot drift from it.'''

    _render_library_context = GeneralizeAgent._render_library_context
    _concept_summary = staticmethod(GeneralizeAgent._concept_summary)

    def __init__(self, concept_library):
        self.concept_library = concept_library


def build_library_block(concept_library) -> str:
    '''The concepts learned so far, with their source, so a new concept can reuse them.

    Filtered to the learned concepts: the primitives are already documented by hand in
    DSL_DOC, and passing their names too would print them twice.

    SPL filters this to the concepts its plans actually called; a baseline has no plans,
    so it gets the whole library. That is the generous direction, and consistent with the
    rest of the setup compensating baselines for having no Plan stage.
    '''
    learned = set(getattr(concept_library, "inductive_concepts", []) or [])
    if not learned:
        return ""
    rendered = _LibraryRenderer(concept_library)._render_library_context(learned)
    if not rendered:
        return ""
    return (f"{rendered}\n\n"
            "You may call any of these concepts inside construct() to build part of this "
            "structure, exactly as shown above. Reuse one when the structure genuinely "
            "contains it; otherwise place blocks directly.\n")


def build_task_block(demo_specs) -> str:
    '''The per-concept portion of a prompt: what to build and with which signature.

    ``demo_specs`` is one (instruction, initialized_sketch) pair per demonstration.
    All of them are rendered, mirroring SPL's own generalization prompt: the
    demonstrations instantiate the concept at different parameter values, and seeing
    that variation is the main evidence that the argument should drive a loop. Showing
    only the first would both hide it and contradict the demonstrations that follow.
    '''
    specs = list(demo_specs)
    # No sketch: the model chooses the class name and arguments itself, so only the
    # instructions are given.
    if all(initialized is None for _instruction, initialized in specs):
        lines = "\n".join(f"  Demonstration {i}: {instruction}"
                          for i, (instruction, _) in enumerate(specs, 1))
        return ("# Task\n"
                "The same concept is demonstrated below, at different values of its "
                "argument.\nWrite one class that handles all of them, and any other value "
                "of that argument.\n\n"
                f"{lines}\n")

    if len(specs) == 1:
        instruction, initialized = specs[0]
        return (f"# Task\nInstruction: {instruction}\n\n"
                f"Initialized sketch (your class must match this signature exactly):\n"
                f"```python\n{initialized.strip()}\n```\n")

    blocks = [
        f"Demonstration {i}:\n"
        f"  Instruction: {instruction}\n"
        f"  Initialized sketch:\n```python\n{initialized.strip()}\n```\n"
        for i, (instruction, initialized) in enumerate(specs, 1)
    ]
    return ("# Task\n"
            "The same concept is demonstrated below at different argument values. Write one\n"
            "class that handles all of them, and any other value of the argument.\n"
            "Your class name and __init__ arguments must match the sketches exactly; only\n"
            "the argument VALUES differ between demonstrations.\n\n"
            + "\n".join(blocks))


NO_SKETCH_RULES = '''# Output rules
- Output ONLY the class definition inside a single ```python ... ``` block.
- No explanation, no reasoning, no example usage outside the block.
- YOU choose the class name and the argument names. Name them the way a person describing
  this structure would.
- __init__ must take exactly two arguments besides self: one numeric argument (the size of
  the structure, annotated `int`) and `objects: list`, the block ids to place. The robot
  supplies the object ids through `objects`, so a class without it cannot be run.
- The class must define: __init__, construct, argument_sampler, plan, blocks,
  substructures, key_blocks, actions.
- construct() must generalize: it takes the numeric argument and works for ANY value of
  it, not only the ones demonstrated. Never hard-code a fixed number of placements.
'''


def build_system_prompt(stats_block: str = "", require_signature: bool = True,
                        library_block: str = "") -> str:
    '''The shared system message: DSL + library + required class structure + output rules.

    ``stats_block`` is the placement primitive's per-direction delta table (see
    common/primitive_stats.py). ``library_block`` is the concepts learned so far. Both are
    per-concept, which is fine because this is built fresh inside each agent's generate();
    pass "" to omit either.

    ``require_signature=False`` is the no-sketch condition: the model is asked to choose
    the class name and arguments, and told only the structural requirement (a numeric
    argument plus `objects`) without which the class cannot be executed at all.
    '''
    stats = f"{stats_block}\n" if stats_block else ""
    library = f"{library_block}\n" if library_block else ""
    rules = OUTPUT_RULES if require_signature else NO_SKETCH_RULES
    return (
        "You write Python classes that construct spatial block structures for a robot.\n\n"
        f"{DSL_DOC}\n"
        f"{stats}"
        f"{library}"
        f"{CONCEPT_STRUCTURE_PROMPT}\n"
        f"{WORKED_EXAMPLE}\n"
        f"{rules}"
    )
