'''
test_saycan.py

SayCan has no program, so its whole evaluation rests on two things working:

  1. A flat action list replays on the ideal executor and scores against ground truth,
     giving plan_accuracy without a program.
  2. The loop actually stops. Nothing halts it on its own — the executor's _is_terminal
     is always False at inference — so done(), all-objects-placed and max_steps are
     load-bearing, and all three are exercised here with a stub backend (no network).

Run: python -m baseline_spl.tests.test_saycan
'''

from __future__ import annotations

import json
import sys
import types

from baseline_spl.VLM.saycan.agent import DONE, SayCanAgent
from baseline_spl.VLM.saycan.harness import SayCanHarness


class StubConversation:
    '''Records every turn, and whether the bulky context was re-sent after the first.'''

    def __init__(self, backend, system, images=None):
        self.backend = backend
        self.system = system
        self.images = images
        self.turns = []

    def ask(self, user_query, response_format=None, max_tokens=None):
        self.backend.calls += 1
        self.turns.append(user_query)
        reply = self.backend.reply(self.backend.calls, user_query)
        return reply


class StubBackend:
    '''Returns JSON scores from a fixed preference order; records the conversation.'''

    def __init__(self, prefer, raw_reply=None):
        self.prefer = prefer
        self.raw_reply = raw_reply
        self.calls = 0
        self.conversations = []

    def start_conversation(self, system, images=None, model=None):
        convo = StubConversation(self, system, images)
        self.conversations.append(convo)
        return convo

    def reply(self, call, user_query):
        if self.raw_reply is not None:
            return self.raw_reply
        listed = [line[2:] for line in user_query.splitlines() if line.startswith("- ")]
        choice = self.prefer(call, listed)
        return json.dumps({c: (100 if c == choice else 1) for c in listed})


class StubState:
    def __init__(self, num_objects, placed=()):
        self.num_objects = num_objects
        self.objects_moved = list(placed)
        self.focus = ([0.0, 0.0, 0.0], None)
        self.state = []


class StubExecutor:
    '''Records executed actions and marks an object placed for each placement.'''

    def __init__(self, num_objects=5):
        self.state = StubState(num_objects)
        self.executed = []

    def query_current_state(self):
        return self.state

    def execute(self, action, allow_exceptions=False):
        self.executed.append(action)
        if "place_object_at_focus" in action:
            self.state.objects_moved.append(len(self.state.objects_moved))


class StubActionSpace:
    PRIMITIVES = ['shift_focus("LEFT")', 'shift_focus("RIGHT")', 'shift_focus("FRONT")',
                  'shift_focus("BEHIND")', 'shift_focus("TOP")',
                  'place_object_at_focus(objects.pop(0))']

    def get_primitives(self, state):
        return list(self.PRIMITIVES)


def _agent(max_steps=40, prefer=None, raw_reply=None):
    configs = types.SimpleNamespace(
        demo_modality="text", max_steps=max_steps, coordinate_mode="lattice",
        include_primitive_stats=False, vlm_max_image_px=512, vlm_max_keyframes=40,
        plan_library_top_k=3)
    return SayCanAgent(configs, StubBackend(prefer, raw_reply))


def check_context_sent_once(failures):
    '''The demonstration goes out on turn 1 and never again; later turns carry only the
    new state. This is the whole point of using a conversation.'''
    place = StubActionSpace.PRIMITIVES[-1]
    agent = _agent(prefer=lambda call, cands: place)
    executor = StubExecutor(num_objects=4)
    agent.solve("build a row", [], executor, StubActionSpace())

    convos = agent.backend.conversations
    if len(convos) != 1:
        failures.append(f"expected one conversation per instruction, got {len(convos)}")
        return
    turns = convos[0].turns
    if "# Instruction" not in turns[0]:
        failures.append("first turn is missing the instruction/demonstration block")
    elif any("# Instruction" in t for t in turns[1:]):
        failures.append("a later turn re-sent the instruction block")
    elif "# Current state" not in turns[0] or not all(("# Current state" in t or "# Since the last action" in t) for t in turns):
        failures.append("every turn should carry the current state")
    else:
        print(f"  OK   context sent once, {len(turns) - 1} follow-up turn(s) state-only")


def check_assign_offered_and_validated(failures):
    '''assign_focus is offered as a template and the model fills the id; only ids of
    already-placed blocks are accepted.'''
    agent = _agent()
    fixed = list(StubActionSpace.PRIMITIVES) + [DONE]

    good = agent._parse_scores('{"assign_focus(object_id=2)": 90, "done()": 5}',
                               fixed, placed=[0, 1, 2])
    if good.get("assign_focus(object_id=2)") != 90:
        failures.append(f"a valid assign onto a placed block was dropped: {good}")
    else:
        print("  OK   assign_focus onto a placed block is accepted")

    bad = agent._parse_scores('{"assign_focus(object_id=7)": 90, "done()": 5}',
                              fixed, placed=[0, 1, 2])
    if any("assign_focus" in k for k in bad):
        failures.append(f"assign onto an unplaced block should be dropped: {bad}")
    else:
        print("  OK   assign_focus onto an unplaced block is dropped")

    junk = agent._parse_scores('{"delete_everything()": 99, "done()": 5}', fixed, placed=[0])
    if any(k not in fixed for k in junk):
        failures.append(f"an invented action should be dropped: {junk}")
    else:
        print("  OK   an action outside the candidate set is dropped")


def check_scoring_failure_stops(failures):
    '''Unusable JSON is retried once, then the instruction stops — no invented action.'''
    agent = _agent(raw_reply="I think you should place a block, probably.")
    executor = StubExecutor(num_objects=5)
    actions, stop = agent.solve("build a row", [], executor, StubActionSpace())
    if stop != "scoring_failed":
        failures.append(f"unusable JSON should stop with scoring_failed, got {stop}")
    elif actions:
        failures.append(f"no action should be executed on unusable scores, got {actions}")
    elif agent.backend.calls != 2:
        failures.append(f"expected one retry (2 calls), got {agent.backend.calls}")
    else:
        print("  OK   unusable JSON retries once then stops with 'scoring_failed'")


def check_state_lists_every_object(failures):
    from SPL.utils.metrics import IdealExecutor  # noqa: F401  (kept import cheap)

    class FakeMesh:
        def __init__(self, xyz):
            self.vertices = __import__("numpy").array([xyz], dtype=float)

    state = types.SimpleNamespace(
        state=[FakeMesh((0.1, 0.2, 0.05)), FakeMesh((0.3, 0.4, 0.05))],
        focus=([0.5, 0.6, 0.05], None))
    text = SayCanAgent._state_text(state, placed=[0], names=["cube_a", "cube_b"])
    if "id=0" not in text or "id=1" not in text:
        failures.append(f"state text should list every object: {text}")
    elif "PLACED" not in text or "not yet placed" not in text:
        failures.append(f"state text should mark placed vs unplaced: {text}")
    elif "Current focus" not in text:
        failures.append("state text should include the focus")
    else:
        print("  OK   state text lists every object, its position and placed status")


def check_replay(failures):
    '''A flat action list must score against ground truth with no program involved.'''
    from SPL.utils.metrics import compute_plan_metrics, run_gt_program

    actions = ["assign_focus([0.75, 0, 0.05])"]
    for i in range(5):
        actions.append("place_object_at_focus(objects.pop(0))")
        if i < 4:
            actions.append('shift_focus("RIGHT")')

    trace = SayCanHarness._replay(actions, 30)
    gt = run_gt_program("blocks = filter(('blue','cube'))\n"
                        "assign_focus([0.75, 0, 0.05])\nrow(5, blocks)", 30)
    metrics = compute_plan_metrics(trace, gt)
    if metrics.get("plan_accuracy") != 1.0:
        failures.append(f"flat action list should score 1.0, got {metrics}")
    else:
        print("  OK   a flat action list scores plan_accuracy 1.0 against ground truth")

    # Uppercase directions must work: SPL's candidates are upper, the GT program is lower.
    if trace["positions"] != gt["positions"]:
        failures.append("uppercase shift directions did not replay identically")
    else:
        print("  OK   uppercase shift directions replay identically to ground truth")


def check_stops_on_done(failures):
    # cands[-1] is done() itself, so name the placement action explicitly.
    place = StubActionSpace.PRIMITIVES[-1]
    agent = _agent(prefer=lambda call, cands: DONE if call > 3 else place)
    executor = StubExecutor(num_objects=10)
    actions, stop = agent.solve("build something", [], executor, StubActionSpace())
    if len(actions) != 3 or stop != "done":
        failures.append(f"done() should stop after 3 actions, got {len(actions)}/{stop}")
    else:
        print("  OK   loop stops when done() is selected (stop_reason='done')")


def check_stops_when_all_placed(failures):
    place = StubActionSpace.PRIMITIVES[-1]
    agent = _agent(prefer=lambda call, cands: place)
    executor = StubExecutor(num_objects=4)
    actions, stop = agent.solve("build something", [], executor, StubActionSpace())
    if len(actions) != 4 or stop != "all_placed":
        failures.append(f"should stop once all 4 placed, got {len(actions)}/{stop}")
    else:
        print("  OK   loop stops when every object placed (stop_reason='all_placed')")


def check_stops_on_max_steps(failures):
    shift = StubActionSpace.PRIMITIVES[0]
    agent = _agent(max_steps=7, prefer=lambda call, cands: shift)
    executor = StubExecutor(num_objects=99)
    actions, stop = agent.solve("build something", [], executor, StubActionSpace())
    if len(actions) != 7 or stop != "max_steps":
        failures.append(f"max_steps=7 should cap at 7, got {len(actions)}/{stop}")
    else:
        print("  OK   loop stops at max_steps (stop_reason='max_steps')")


def check_candidate_set(failures):
    agent = _agent(prefer=lambda call, cands: DONE)
    executor = StubExecutor()
    agent.solve("build something", [], executor, StubActionSpace())
    asked = agent.backend.calls
    if asked != 1:
        failures.append(f"expected one scoring call before done(), got {asked}")
    else:
        print("  OK   one scoring call per step")


def check_oneshot(failures):
    '''recursive=False costs one call for the whole plan, drops invented actions, and
    stops at done(). The cheap mode is only worth having if it is still validated.'''
    reply = ('{"plan": ["place_object_at_focus(objects.pop(0))", "shift_focus(\\"RIGHT\\")",'
             ' "place_object_at_focus(objects.pop(0))", "assign_focus(object_id=0)",'
             ' "fly_away()", "done()", "shift_focus(\\"TOP\\")"]}')
    agent = _agent(raw_reply=reply)
    executor = StubExecutor(num_objects=5)
    actions, stop = agent.solve("build a row", [], executor, StubActionSpace(),
                                recursive=False)

    expected = ["place_object_at_focus(objects.pop(0))", 'shift_focus("RIGHT")',
                "place_object_at_focus(objects.pop(0))", "assign_focus(object_id=0)"]
    if agent.backend.calls != 1:
        failures.append(f"one-shot should cost exactly 1 call, got {agent.backend.calls}")
    elif stop != "plan_complete":
        failures.append(f"expected stop_reason 'plan_complete', got {stop}")
    elif actions != expected:
        failures.append(f"one-shot plan mis-parsed or mis-executed: {actions}")
    else:
        print("  OK   recursive=False: 1 call, invented action dropped, done() ends it")

    if "assign_focus" not in agent.backend.conversations[0].turns[0]:
        failures.append("one-shot prompt never mentions assign_focus")
    else:
        print("  OK   one-shot prompt offers assign_focus")

    unusable = _agent(raw_reply="sure, place some blocks")
    actions, stop = unusable.solve("build a row", [], StubExecutor(), StubActionSpace(),
                                   recursive=False)
    if stop != "no_plan" or actions:
        failures.append(f"unparseable one-shot reply should give no_plan/[], got {stop}/{actions}")
    else:
        print("  OK   an unparseable one-shot reply stops with 'no_plan'")


def check_action_descriptions(failures):
    '''Both modes must document what each primitive does. Scoring bare call strings with
    no semantics would handicap SayCan against the program baselines, which get the same
    information from dsl_prompt.DSL_DOC.'''
    from SPL.config.primitive_config import DEFAULT_ACTIONS

    wanted = ["shift_focus", "place_object_at_focus", "assign_focus", "done()",
              "FOCUS"] + [f'"{d}"' for d in DEFAULT_ACTIONS]
    for recursive, label in ((True, "recursive"), (False, "one-shot")):
        agent = _agent(prefer=lambda call, cands: DONE,
                       raw_reply=None if recursive else '{"plan": ["done()"]}')
        agent.solve("build something", [], StubExecutor(), StubActionSpace(),
                    recursive=recursive)
        system = agent.backend.conversations[0].system
        missing = [w for w in wanted if w not in system]
        if missing:
            failures.append(f"{label} prompt omits action documentation for {missing}")
        else:
            print(f"  OK   {label} prompt documents every primitive and direction")


def check_phase_knobs(failures):
    '''Learning reads recursive_learn, not a constant and not the inference knob — the
    whole point of splitting them is that a run can use different modes per phase.'''
    import os
    import tempfile

    h = object.__new__(SayCanHarness)          # skip __init__: no SPL instance needed
    h.configs = types.SimpleNamespace(recursive_learn=True, recursive_infer=False)
    h._plans, h._plans_path = {}, os.path.join(tempfile.mkdtemp(), "plans.json")
    h._metric_records, h._time_records = {}, {}
    h._flush = lambda: None
    h._plan_accuracy = lambda actions, demo: 1.0
    h.agent = types.SimpleNamespace()          # no backend -> no ledger calls

    seen = []

    def fake_solve(demo, cached_plans, recursive):
        seen.append(recursive)
        return [], "done", None

    h._solve = fake_solve
    demo = [{"demo_id": "0", "concept": "row", "language_instruction": "build a row"}]

    h.learn_concept(demo)
    if seen != [True]:
        failures.append(f"learn should have used recursive_learn=True, saw {seen}")
        return

    h.configs.recursive_learn = False           # flip it: the value must track the config
    h._plans.clear()
    seen.clear()
    h.learn_concept(demo)
    if seen != [False]:
        failures.append(f"learn ignored recursive_learn=False, saw {seen}")
    else:
        print("  OK   learning follows recursive_learn, independently of recursive_infer")


def check_no_program_verdict(failures):
    verdict = SayCanHarness.program_equivalence(None, {"concept": "row"})
    if verdict.get("program_accuracy") is not None:
        failures.append("SayCan must report program_accuracy as None")
    elif verdict.get("program_verdict") != "no_program":
        failures.append(f"expected verdict 'no_program', got {verdict.get('program_verdict')}")
    else:
        print("  OK   program accuracy is reported as 'no_program', not a failed check")


def main() -> int:
    failures = []
    check_replay(failures)
    check_stops_on_done(failures)
    check_stops_when_all_placed(failures)
    check_stops_on_max_steps(failures)
    check_candidate_set(failures)
    check_oneshot(failures)
    check_action_descriptions(failures)
    check_phase_knobs(failures)
    check_no_program_verdict(failures)
    check_context_sent_once(failures)
    check_assign_offered_and_validated(failures)
    check_scoring_failure_stops(failures)
    check_state_lists_every_object(failures)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all saycan checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
