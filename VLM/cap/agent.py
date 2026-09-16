'''
agent.py

Code-as-Policies baseline (Liang et al., ICRA 2023).

Single-stage hierarchical code generation: one prompt carrying the instruction, the
shared-sketch signature, the action DSL and the required class structure, followed by
CaP's recursive expansion of any helper the generated code calls but never defines
(``fgen.py``).

Two configurations:

  ``use_demo=True``  (default) - the demonstration is included in the prompt, serialized
      exactly as the Demo2Code baseline receives it. This makes CaP an input-matched
      competitor, and the only remaining difference from demo2code-text is Demo2Code's
      staged recursive summarization, which is precisely that paper's contribution.

  ``use_demo=False`` - instruction only. The world-knowledge control: how much of
      `row`/`tower` does the model already know without ever seeing a demonstration?
      This run never touches ``meshes`` or ``rgbs``, and asserts as much.
'''

from __future__ import annotations

from typing import List

from baseline_spl.common import codegen, dsl_prompt
from baseline_spl.common.evaluator import ClassEvaluator
from baseline_spl.common.harness import GeneratedConcept, log
from baseline_spl.common.primitive_stats import build_stats_block
from baseline_spl.common.serialize_text import to_demo2code_text
from baseline_spl.common.serialize_visual import demo_frames


DEMO_PREAMBLE = '''# Demonstrations
Below are human demonstrations of this concept. Each [Scenario] is a separate scene with
its own objects, and lists the keyframes of one demonstration: which object moved at each
step and where it ended up.

Work out for yourself which shift_focus direction (or sequence of directions) and num_steps
account for the movement between consecutive placements. Your class must reproduce the construction
procedure these demonstrations show — the same placement order and the same relative
movement — generalized to any value of the numeric argument.

'''

IMAGE_PREAMBLE = '''# Demonstrations
The images below are the keyframes of the human demonstrations, in order. Each scenario is
a separate scene; the first image of a scenario is before anything was placed, and each
later image follows one placement.

Work out which object is added at each step and where it goes relative to the previous
one. Your class must reproduce the construction procedure shown — the same placement order
and the same relative movement — generalized to any value of the numeric argument.

'''


class CodeAsPoliciesAgent:
    '''Reads use_demo, demo_modality and max_expansion_depth from the active config.'''

    def __init__(self, configs, backend):
        self.configs = configs
        self.backend = backend
        self.use_demo = configs.use_demo
        self.demo_modality = configs.demo_modality
        if self.demo_modality not in ("text", "images"):
            raise ValueError(f"demo_modality must be 'text' or 'images', "
                             f"got {self.demo_modality!r}")

    def generate(self, demos: List[dict], sketch_infos, shared) -> GeneratedConcept:
        # sketch_infos is None in the no-sketch condition: the model picks the class name
        # and arguments, which are then read back off the generated code.
        concept = sketch_infos[0]["concept"] if sketch_infos else None
        demo_specs = [(d.get("language_instruction", ""),
                       shared.initialized_sketch(si) if si is not None else None)
                      for d, si in zip(demos, sketch_infos or [None] * len(demos))]

        # The primitive's per-direction deltas, so the model can name the movements
        # itself; the serializer no longer labels them.
        stats_block = build_stats_block(
            shared.spl.executor, demos if self.use_demo else [],
            enabled=self.use_demo and self.configs.include_primitive_stats)

        # Concepts learned so far, so this one can be built out of them.
        library_block = dsl_prompt.build_library_block(shared.spl.concept_library)

        system_prompt = dsl_prompt.build_system_prompt(
            stats_block, require_signature=bool(sketch_infos), library_block=library_block)
        user_prompt = dsl_prompt.build_task_block(demo_specs)

        # This is the ONLY place a demonstration's perceptual state is read. With
        # use_demo=False nothing below touches meshes/rgbs, which is what makes the run a
        # genuine world-knowledge control; tests/test_leakage.py enforces it by handing
        # the agent demos whose 'meshes' key raises on access.
        images = None
        if self.use_demo and self.demo_modality == "text":
            demo_text = to_demo2code_text(
                demos, coordinate_mode=self.configs.coordinate_mode)
            user_prompt = f"{user_prompt}\n{DEMO_PREAMBLE}{demo_text}\n"
        elif self.use_demo:
            images = [frame for demo in demos
                      for frame in demo_frames(demo,
                                               max_px=self.configs.vlm_max_image_px,
                                               max_keyframes=self.configs.vlm_max_keyframes)]
            log(f"[cap] {len(images)} keyframe images across {len(demos)} demonstration(s)")
            user_prompt = f"{user_prompt}\n{IMAGE_PREAMBLE}"

        # Scores the helper-expanded class, which is what gets registered. Image runs get no
        # distances. The harness refuses this with use_demo=False or no sketch.
        known = list(shared.spl.concept_library.operators.keys())
        evaluator = ClassEvaluator(
            shared, demos, sketch_infos, with_numbers=self.demo_modality == "text",
            transform=lambda c: fgen_expand(c, self.backend, known,
                                            max_depth=self.configs.max_expansion_depth),
        ) if self.configs.use_evaluator_feedback else None

        code = codegen.generate_with_retries(
            self.backend, system_prompt, user_prompt,
            wanted_name=concept,
            max_retries=self.configs.max_code_retries,
            max_tokens=self.configs.codegen_max_tokens,
            images=images,
            evaluator=evaluator,
            log=log,
        )

        if code is None:
            return None

        # Code-as-Policies' recursive expansion of undefined helpers. If expansion
        # somehow breaks the class, fall back to the version that already validated
        # rather than registering something broken.
        expanded = fgen_expand(code, self.backend, known,
                               max_depth=self.configs.max_expansion_depth)
        info = {"use_demo": self.use_demo, "demo_modality": self.demo_modality,
                "expanded": expanded != code}
        if evaluator is not None:
            info["evaluator"] = evaluator.record
        if expanded != code:
            try:
                codegen.validate(expanded)
                code = expanded
            except Exception as exc:  # noqa: BLE001
                log(f"[cap] helper expansion broke the class ({exc}); keeping unexpanded.")
                info["expansion"] = f"reverted: {exc}"

        # With a sketch the name and types come from it; without one they are whatever
        # the model wrote, recovered from the class's own __init__.
        if sketch_infos:
            name, attributes = concept, shared.attributes(sketch_infos[0])
        else:
            name, attributes = codegen.class_signature(code)
        return GeneratedConcept(name, attributes, code, info)


def fgen_expand(code: str, backend, known, *, max_depth: int) -> str:
    '''Wrapper kept separate so tests can exercise expansion without an agent.'''
    from baseline_spl.VLM.cap.fgen import expand_helpers

    return expand_helpers(code, backend=backend, known=known,
                          dsl_doc=dsl_prompt.DSL_DOC, max_depth=max_depth, log=log)
