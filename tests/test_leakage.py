'''
test_leakage.py

Guards the three things that would silently invalidate every baseline number.

  1. A baseline must not inherit SPL's learned concept library.
  2. A baseline must not share SketchAgent cache state with SPL.
  3. `CapConfig.use_demo = False` is the world-knowledge control, so it must never read a
     demonstration's perceptual state. Enforced by handing the agent demo dicts that
     raise on any access to meshes/rgbs/depths/masks, rather than by inspecting strings.

Run: python -m baseline_spl.tests.test_leakage
'''

from __future__ import annotations

import sys
import types

from baseline_spl.common.config import BaselineConfig, assert_parity, assert_not_spl_library
from baseline_spl.config import CapConfig, CommonConfig


GUARDED_KEYS = ("meshes", "rgbs", "depths", "masks", "objects_moved")

VALID_CLASS = '''
class row:
    def __init__(self, length: int, objects: list):
        self.length = length
        self.objects = list(objects) if objects is not None else []
        self.constructed = False
        self._plan = []
        self._placed = []
        self._substructures = []

    def construct(self):
        if self.constructed:
            raise Exception("already constructed")
        self.constructed = True
        for i in range(self.length):
            obj = self.objects.pop(0)
            place_object_at_focus(obj)
            self._plan.append(f"place_object_at_focus(object_id = {obj})")
            self._placed.append(obj)
            if i < self.length - 1:
                shift_focus("RIGHT")
                self._plan.append('shift_focus("RIGHT")')

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
        return [self._placed[0]] if self._placed else []

    @property
    def actions(self):
        return [f"assign_focus(object_id = {b})" for b in self.key_blocks]
'''


class GuardedDemo(dict):
    '''A demo whose perceptual fields explode on access.'''

    def __getitem__(self, key):
        if key in GUARDED_KEYS:
            raise AssertionError(f"no-demo run read demonstration field '{key}'")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in GUARDED_KEYS:
            raise AssertionError(f"no-demo run read demonstration field '{key}'")
        return super().get(key, default)


class FakeBackend:
    '''Returns a canned valid class; records prompts. No network.'''

    def __init__(self, code=VALID_CLASS):
        self.code = code
        self.prompts = []

    def call_text(self, system_message, user_query, max_tokens=None):
        self.prompts.append(user_query)
        return f"```python\n{self.code}\n```"


def _fake_shared():
    library = types.SimpleNamespace(operators={
        "filter": None, "assign_focus": None, "shift_focus": None,
        "place_object_at_focus": None})
    # executor is only reached for the SRN stats, which these tests disable.
    spl = types.SimpleNamespace(concept_library=library, executor=None)
    def initialized(si):
        args = ", ".join(f"{k}={v['value']!r}" for k, v in si["arguments"].items())
        name = si["concept"]
        return f"{name}_1 = {name}({args})\n{name}_1.construct()"

    shared = types.SimpleNamespace(
        spl=spl,
        initialized_sketch=initialized,
        attributes=lambda si: {"length": int, "objects": list},
    )
    return shared


def main() -> int:
    failures = []

    # 1 + 2: configuration isolation, no SPL construction needed.
    cfg = BaselineConfig.from_run_config(CapConfig, "test_leakage")
    ckpt = cfg.load_concept_checkpoint
    if ckpt is not None and cfg.run_dir not in ckpt:
        failures.append(f"load_concept_checkpoint is outside this run: {ckpt}")
    else:
        print(f"  OK   load_concept_checkpoint stays within the run ({ckpt})")

    # SPL's own library must be refused even when explicitly asked for.
    from SPL.config.spl_config import SPLConfig as _SPLConfig
    try:
        assert_not_spl_library(_SPLConfig.load_concept_checkpoint)
        failures.append("assert_not_spl_library accepted SPL's own concept library")
    except ValueError:
        print("  OK   resuming from SPL's own library is refused")

    if "SPL/model/cache" in cfg.sketch_config.cache_dir:
        failures.append(f"sketch cache points into SPL's tree: {cfg.sketch_config.cache_dir}")
    elif cfg.run_dir not in cfg.sketch_config.cache_dir:
        failures.append(f"sketch cache is outside the run dir: {cfg.sketch_config.cache_dir}")
    else:
        print("  OK   sketch cache is private to this variant")

    from SPL.config.spl_config import SPLConfig
    if SPLConfig().sketch_config.cache_dir == cfg.sketch_config.cache_dir:
        failures.append("BaselineConfig mutated SPL's shared SketchConfig instance")
    else:
        print("  OK   SPL's own SketchConfig is untouched")

    # 3: the no-demo control must not read perceptual state.
    from baseline_spl.VLM.cap.agent import CodeAsPoliciesAgent

    demos = [GuardedDemo(language_instruction="Construct a row of length 5 using blue cube",
                         object_names=["cube_blue_0"], demo_id="0000", concept="row")]
    sketch_infos = [{"concept": "row",
                     "arguments": {"length": {"type": "int", "value": 5},
                                   "objects": {"type": "list", "value": [0, 1, 2, 3, 4]}}}]

    cfg.use_demo = False
    cfg.include_primitive_stats = False   # would need the SRN and the demo's meshes
    backend = FakeBackend()
    agent = CodeAsPoliciesAgent(cfg, backend)
    try:
        result = agent.generate(demos, sketch_infos, _fake_shared())
        if result is None or not result.code:
            failures.append("no-demo agent produced no code")
        else:
            print("  OK   use_demo=False never touched meshes/rgbs/depths/masks")
    except AssertionError as exc:
        failures.append(f"no-demo leakage: {exc}")

    # ... and the with-demo path must genuinely read them (guard fires).
    cfg.use_demo = True
    agent_with_demo = CodeAsPoliciesAgent(cfg, FakeBackend())
    try:
        agent_with_demo.generate(demos, sketch_infos, _fake_shared())
        failures.append("use_demo=True did not read the demonstration at all")
    except AssertionError:
        print("  OK   use_demo=True does read the demonstration (guard fired as expected)")

    # 4: every demonstration's binding must reach the prompt, not just the first.
    # The demos instantiate the concept at different argument values, and that variation
    # is the main evidence for a loop; showing one would handicap the baseline against
    # SPL, whose Generalize stage receives a per-demonstration listing.
    two_demos = [
        dict(language_instruction="Construct a tower of height 3 using blue cube",
             object_names=["cube_blue_0"], demo_id="0000", concept="tower"),
        dict(language_instruction="Construct a tower of height 6 using red cube",
             object_names=["cube_red_0"], demo_id="0001", concept="tower"),
    ]
    two_sketches = [
        {"concept": "tower", "arguments": {"height": {"type": "int", "value": 3},
                                           "objects": {"type": "list", "value": [0, 1, 2]}}},
        {"concept": "tower", "arguments": {"height": {"type": "int", "value": 6},
                                           "objects": {"type": "list", "value": [0, 1, 2]}}},
    ]
    cfg.use_demo = False          # keep the check on the prompt, not the serializer
    prompt_backend = FakeBackend()
    CodeAsPoliciesAgent(cfg, prompt_backend).generate(
        two_demos, two_sketches, _fake_shared())
    prompt = prompt_backend.prompts[0] if prompt_backend.prompts else ""
    missing = [v for v in ("height=3", "height=6") if v not in prompt]
    if missing:
        failures.append(f"prompt omits demonstration binding(s) {missing}")
    elif "height 3" not in prompt or "height 6" not in prompt:
        failures.append("prompt omits one of the per-demonstration instructions")
    else:
        print("  OK   all demonstrations' instructions and bindings reach the prompt")

    # 5: parity-critical settings must not be overridable per baseline.
    class BadConfig(CapConfig):
        concepts = ["only_this_one"]

    try:
        assert_parity(BadConfig)
        failures.append("assert_parity did not reject an overridden parity-critical field")
    except ValueError:
        print("  OK   assert_parity rejects a baseline that overrides shared settings")

    try:
        assert_parity(CapConfig)
        print("  OK   assert_parity accepts the real CapConfig")
    except ValueError as exc:
        failures.append(f"assert_parity rejected the real CapConfig: {exc}")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all leakage checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
