'''
agent.py

SayCan baseline (Ahn et al., 2022), adapted to SPL's primitives.

At each step the agent scores every available primitive and executes the best one. It
never builds an abstraction: `row(5)` needs five placements and `row(11)` needs eleven,
planned afresh each time. That is the point — it is the control for "is a program needed
at all?".

Two deliberate departures from the paper, both chosen up front:

  * **Say-only.** SayCan multiplies the LLM score by a learned value function ("Can").
    There is no such value function here, and with SPL's primitive set the action space is
    seven items with no wrong-object failure mode, so the affordance term has little to
    mask. This matches an ablation the paper itself reports.
  * **Ranking instead of log-probabilities.** The models this suite is pinned to do not
    expose logprobs, so candidates are scored by asking (see
    ``LLMBackend.score_candidates``).

The candidate set is SPL's own ``ActionSpace.get_primitives`` — five shifts plus
``place_object_at_focus(objects.pop(0))`` — with ``done()`` appended. The object order is
fixed, exactly as SPL's planner sees it, so the agent chooses only *when* to place.

``solve(recursive=False)`` replaces the per-step loop with a single call that emits the
whole plan. It costs one call instead of one per action, but the model never sees the
state its own actions produced — a weaker control, not a cheaper equivalent one. The
harness chooses per phase, from ``recursive_learn`` / ``recursive_infer``.
'''

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

from baseline_spl.common import dsl_prompt
from baseline_spl.common.harness import log
from baseline_spl.common.primitive_stats import build_stats_block
from baseline_spl.common.serialize_text import to_demo2code_text
from baseline_spl.common.serialize_visual import demo_frames

DONE = "done()"
ASSIGN_TEMPLATE = "assign_focus(object_id=<id of a block already placed>)"
_ASSIGN_RE = re.compile(r"^\s*assign_focus\(\s*object_id\s*=\s*(\d+)\s*\)\s*$")

# The same semantics the program baselines get from dsl_prompt.DSL_DOC, restated for a
# flat action sequence: DSL_DOC documents calls "inside construct()" and refers to the
# objects list and the initialized sketch, none of which exist here. Directions come from
# dsl_prompt so the two cannot drift apart.
ACTION_DOC = f'''# What the actions do

The environment keeps a single global FOCUS: a 3-D location where the next object will be
placed. You build a structure by moving the focus and dropping objects at it.

    shift_focus("DIRECTION")
        Move the focus one step along DIRECTION, one of: {dsl_prompt.DIRECTIONS}.
        Nothing is placed.

    place_object_at_focus(objects.pop(0))
        Place the next unplaced object at the current focus. The focus does NOT move
        afterwards, so two placements in a row would put two blocks in the same place.
        Objects are taken from a fixed list in a fixed order — you choose *when* to
        place, never which object.

    assign_focus(object_id=N)
        Jump the focus onto block N, which must already have been placed. Use it to start
        a new part of the structure from an earlier placed block, instead of shifting all the way
        back one step at a time.

    done()
        The structure is complete; stop.

Conventions: "TOP" is the vertical axis; "LEFT"/"RIGHT" and "FRONT"/"BEHIND" are the two
horizontal axes. Physics is real — a block with nothing under it falls, so place supports
before whatever rests on them.
'''

SYSTEM = '''You are directing a robot that builds block structures one action at a time.
You are given a instruction and demonstration of how to construct the structure. At each step,
you will be given the current state and set of actions possible. You should rank them.
You score the available actions; the highest
scoring one is executed, then you are asked again from the new state.

The instruction and demonstration are given once, at the start, and stay in context for the
rest of the conversation.

You are choosing ONE action at a time. There is no program and no loop — to place five
blocks you must choose a placement action five separate times.

Reply with a JSON object mapping each action to a score from 0 to 100, and nothing else:

  {"shift_focus(\\"RIGHT\\")": 80, "place_object_at_focus(objects.pop(0))": 95, "done()": 2}

Score every listed action. For assign_focus you choose the argument yourself: write the
concrete call, such as "assign_focus(object_id=3)", using the id of a block that has
already been placed. Include it only when it is useful; omit it otherwise.'''

ONESHOT_SYSTEM = '''You are directing a robot that builds block structures.

You are given an instruction, a demonstration of how the structure is built, the current
scene, and the actions the robot can take. Output the COMPLETE plan in one reply: the
ordered list of actions that builds the structure. You will not be asked again, so the
list must be complete and in order.

There is no program and no loop — to place five blocks, list the placement action five
separate times.

assign_focus(object_id=N) moves the focus onto a block placed earlier in this same plan.
Use it when the next part of the structure starts from an earlier block rather than from
where the focus has ended up; write the concrete call with the id.

Reply with a JSON object holding the ordered action list, and nothing else:

  {"plan": ["place_object_at_focus(objects.pop(0))", "shift_focus(\\"RIGHT\\")",
            "place_object_at_focus(objects.pop(0))"]}'''


class SayCanAgent:
    '''Reads demo_modality, max_steps, include_primitive_stats and plan_library_top_k
    from the active config file.'''

    def __init__(self, configs, backend):
        self.configs = configs
        self.backend = backend
        self.demo_modality = configs.demo_modality
        if self.demo_modality not in ("text", "images"):
            raise ValueError(f"demo_modality must be 'text' or 'images', "
                             f"got {self.demo_modality!r}")

    # ------------------------------------------------------------------ #
    # Prompt pieces
    # ------------------------------------------------------------------ #
    def _context(self, demos: Sequence[dict], cached_plans: Sequence[dict]):
        '''(text context, images) shared by every step of one instruction.'''
        images = None
        parts = []
        if not demos:
            parts.append("# Demonstrations\n(none available)")
        elif self.demo_modality == "text":
            parts.append("# Demonstrations\n"
                         + to_demo2code_text(demos,
                                             coordinate_mode=self.configs.coordinate_mode))
        else:
            images = [frame for demo in demos
                      for frame in demo_frames(demo,
                                               max_px=self.configs.vlm_max_image_px,
                                               max_keyframes=self.configs.vlm_max_keyframes)]
            parts.append("# Demonstrations\nThe images show the keyframes of each "
                         "demonstration, in order.")

        if cached_plans:
            lines = []
            for plan in cached_plans:
                actions = "\n  ".join(plan["actions"])
                lines.append(f"Instruction: {plan['instruction']}\n  {actions}")
            parts.append("# Action sequences you produced for earlier instructions\n"
                         + "\n\n".join(lines))
        return "\n\n".join(parts), images

    @staticmethod
    def _focus_text(state) -> str:
        focus_mean = state.focus[0] if state.focus is not None else None
        if focus_mean is None:
            return "Current focus: unknown"
        return "Current focus: " + str(tuple(round(float(v), 2) for v in focus_mean))

    @staticmethod
    def _state_text(state, placed: List[int], names: Sequence[str] = ()) -> str:
        '''The whole scene: every object's current position from the executor's state.

        Listed in full, not just what has been placed, because choosing
        assign_focus(object_id=...) requires knowing which blocks exist and where, and the
        unplaced ones are what `place_object_at_focus` will consume next.

        Sent on the FIRST turn only — see _state_delta for the rest.
        '''
        from SPL.model.executor import mesh_centroid

        order_of = {obj: i + 1 for i, obj in enumerate(placed) if obj is not None}
        lines = ["Objects (position is the current centroid):"]
        for obj in range(len(state.state)):
            position = tuple(round(float(v), 2) for v in mesh_centroid(state.state[obj]))
            name = names[obj] if obj < len(names) else f"object_{obj}"
            if obj in order_of:
                lines.append(f"  id={obj} {name} at {position} — PLACED "
                             f"(#{order_of[obj]} in build order)")
            else:
                lines.append(f"  id={obj} {name} at {position} — not yet placed")

        lines.append(SayCanAgent._focus_text(state))
        lines.append(f"{len(order_of)} of {len(state.state)} object(s) placed so far.")
        return "\n".join(lines)

    @staticmethod
    def _state_delta(state, placed: List[int], names: Sequence[str] = ()) -> str:
        '''What changed since the previous turn.

        The conversation already holds the full scene from turn one and every action since,
        so repeating all object positions each turn is redundant — and it is charged, since
        conversation history is billed as input on every turn. Only the newest placement
        and the focus are new information.
        '''
        from SPL.model.executor import mesh_centroid

        lines = []
        if placed:
            last = placed[-1]
            if last is not None and last < len(state.state):
                position = tuple(round(float(v), 2) for v in mesh_centroid(state.state[last]))
                name = names[last] if last < len(names) else f"object_{last}"
                lines.append(f"Placed id={last} {name} at {position}.")
        lines.append(SayCanAgent._focus_text(state))
        lines.append(f"{len(placed)} of {len(state.state)} object(s) placed so far.")
        return "\n".join(lines)

    @staticmethod
    def _parse_scores(reply: str, fixed: Sequence[str], placed: Sequence[int]):
        '''JSON reply -> {action: score}. Keys must be a listed action or a well-formed
        assign_focus onto an already-placed block; anything else is dropped rather than
        executed. Returns {} when nothing usable came back.'''
        text = reply.strip()
        if "{" in text:                      # tolerate prose or fences around the object
            text = text[text.index("{"): text.rindex("}") + 1] if "}" in text else text
        try:
            raw = json.loads(text)
        except (ValueError, TypeError):
            return {}
        if not isinstance(raw, dict):
            return {}

        allowed = set(fixed)
        placed_ids = {p for p in placed if p is not None}
        scores: Dict[str, float] = {}
        for action, score in raw.items():
            if not isinstance(action, str):
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            action = action.strip()
            if action in allowed:
                scores[action] = value
                continue
            match = _ASSIGN_RE.match(action)
            if match and int(match.group(1)) in placed_ids:
                # Normalised so the executed string is exactly what we validated.
                scores[f"assign_focus(object_id={int(match.group(1))})"] = value
        return scores

    @staticmethod
    def _parse_plan(reply: str, fixed: Sequence[str]) -> List[str]:
        '''JSON reply -> ordered action list. Entries that are neither a listed action nor
        a well-formed assign_focus are dropped rather than executed.'''
        text = reply.strip()
        start = min((text.index(c) for c in "[{" if c in text), default=-1)
        end = max((text.rindex(c) for c in "]}" if c in text), default=-1)
        if start < 0 or end < start:
            return []
        try:
            raw = json.loads(text[start:end + 1])
        except (ValueError, TypeError):
            return []
        if isinstance(raw, dict):        # {"plan": [...]}, under whatever key it used
            raw = next((v for v in raw.values() if isinstance(v, list)), [])
        if not isinstance(raw, list):
            return []

        allowed = set(fixed)
        plan: List[str] = []
        for action in raw:
            if not isinstance(action, str):
                continue
            action = action.strip()
            if action in allowed:
                plan.append(action)
                continue
            match = _ASSIGN_RE.match(action)
            if match:
                # Any id is accepted here: which blocks are placed depends on the plan's
                # own prefix, so the executor is what rejects one that is not there yet.
                plan.append(f"assign_focus(object_id={int(match.group(1))})")
        return plan

 
    def _solve_oneshot(self, instruction: str, demos: Sequence[dict], executor,
                       action_space, cached_plans: Sequence[dict], stats_block: str):
        '''The whole plan in one call — no scoring, no per-step turns.'''
        context, images = self._context(demos, cached_plans)
        names = list(demos[0].get("object_names") or []) if demos else []
        state = executor.query_current_state()
        fixed = list(action_space.get_primitives(state)) + [DONE]
        preamble = f"{stats_block}\n{context}" if stats_block else context
        breakpoint()
        conversation = self.backend.start_conversation(f"{ONESHOT_SYSTEM}\n\n{ACTION_DOC}",
                                                       images=images)
        reply = conversation.ask(
            f"{preamble}\n\n# Instruction\n{instruction}\n\n"
            f"# Current state\n{self._state_text(state, [], names)}\n\n"
            "# Available actions\n"
            + "\n".join(f"- {c}" for c in fixed + [ASSIGN_TEMPLATE]),
            response_format="json", max_tokens=4000)

        plan = self._parse_plan(reply, fixed)[: self.configs.max_steps]
        if not plan:
            log("[saycan] no usable plan in the reply")
            return [], "no_plan"

        actions: List[str] = []
        for action in plan:
            if action == DONE:
                break
            try:
                executor.execute(action, allow_exceptions=True)
            except Exception as exc:  # noqa: BLE001
                log(f"[saycan] action {action!r} failed ({exc}); stopping")
                return actions, "action_failed"
            actions.append(action)
        log(f"[saycan] one-shot plan of {len(actions)} action(s)")
        return actions, "plan_complete"

    def solve(self, instruction: str, demos: Sequence[dict], executor, action_space,
              cached_plans: Sequence[dict] = (), stats_block: str = "",
              recursive: bool = True):
        '''Plan and execute one instruction. Returns (actions, stop_reason).

        The executor must already hold the initial state and focus; each accepted action
        is executed through it, so the state advances as we go.

        `recursive` is passed in rather than read from the config, because learning and
        inference can use different modes; the harness supplies the one for its phase.

        The stop reason matters for reading the results. `all_placed` means the object
        list ran out, which for a scene holding exactly as many blocks as the structure
        needs hands the agent the count for free — it never had to decide when to stop.
        Only `done` is the agent actually judging the structure complete. With
        recursive=False the plan arrives whole, so `plan_complete` is that same judgement.
        '''
        log(f'[Saycan] Finding plan for Instruction: {instruction}')
        if not recursive:
            return self._solve_oneshot(instruction, demos, executor, action_space,
                                       cached_plans, stats_block)

        context, images = self._context(demos, cached_plans)
        names = list(demos[0].get("object_names") or []) if demos else []
        actions: List[str] = []
        breakpoint()
        # One conversation per instruction: the demonstrations (and, in the image
        # modality, every keyframe) are sent on the first turn only; each later turn
        # carries just the new state and is continued via previous_response_id.
        preamble = f"{stats_block}\n{context}" if stats_block else context
        conversation = self.backend.start_conversation(f"{SYSTEM}\n\n{ACTION_DOC}",
                                                       images=images)
        opening = f"{preamble}\n\n# Instruction\n{instruction}\n"

        for step in range(self.configs.max_steps):
            state = executor.query_current_state()
            placed = list(getattr(state, "objects_moved", []) or [])
            if len(placed) >= state.num_objects:
                log(f"[saycan] every object placed; stopping at step {step}")
                return actions, "all_placed"

            fixed = list(action_space.get_primitives(state)) + [DONE]
            # assign_focus is offered as a template: SPL's own assign actions are pruned
            # by distance to the next demonstrated block, which is teacher forcing and
            # yields nothing at inference, so the argument is the model's to choose.
            listed = fixed + ([ASSIGN_TEMPLATE] if placed else [])

            # Full scene once; after that only what changed, since the conversation still
            # holds the scene and every action taken, and that history is billed each turn.
            scene = (self._state_text(state, placed, names) if step == 0
                     else self._state_delta(state, placed, names))
            header = "# Current state" if step == 0 else "# Since the last action"
            turn = (f"{header}\n{scene}\n\n"
                    f"# Available actions\n" + "\n".join(f"- {c}" for c in listed))
            if step == 0:
                turn = f"{opening}\n{turn}"

            scores = {}
            for _attempt in range(2):       # one retry, then give up rather than guess
                reply = conversation.ask(turn, response_format="json", max_tokens=2000)
                scores = self._parse_scores(reply, fixed, placed)
                if scores:
                    break
                log(f"[saycan] step {step}: unusable score reply; re-asking")
                turn = ("Your reply was not a usable JSON object. Reply with ONLY a JSON "
                        "object mapping each listed action to a number from 0 to 100.")
            if not scores:
                log(f"[saycan] step {step}: no valid scores after a retry; stopping")
                return actions, "scoring_failed"

            best = max(scores, key=scores.get)
            log(f"[saycan] Action at step {step}: {best}")
            if best == DONE:
                log(f"[saycan] chose done() after {len(actions)} action(s)")
                return actions, "done"
            try:
                executor.execute(best, allow_exceptions=True)
            except Exception as exc:  # noqa: BLE001
                log(f"[saycan] action {best!r} failed ({exc}); stopping")
                return actions, "action_failed"
            actions.append(best)

        log(f"[saycan] hit max_steps={self.configs.max_steps} without done(); "
            f"the plan may be truncated")
        return actions, "max_steps"
