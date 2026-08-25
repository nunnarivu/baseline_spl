'''
agent_vlm.py

Demo2Code baseline, VLM variant.

Same two-stage pipeline as ``agent_text.py``, but the *first* summarization round is done
by a vision model reading the demonstration's keyframe images instead of a language model
reading a symbolically serialized state. Everything after that is the authors' code:
the summaries are handed straight to upstream ``summarize()``, which keeps recursing if
they are not yet compact, merges them via ``summary_2_spec`` and runs stage 2 unchanged.

For that hand-off to work, the VLM's reply must carry upstream's own control tags —
``[[Is the new trajectory sufficiently summarized? (yes/no):]]`` and
``[[Summarized trajectory:]]`` — which ``is_summarized()`` and ``prepare_demo_query()``
parse. The prompt below demands exactly that format.

This variant receives NO privileged 3-D state: no centroids, no grid cells, no direction
labels. That is the point of the condition, and ``generate`` asserts it.
'''

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from baseline_spl.common import codegen, dsl_prompt
from baseline_spl.common.harness import GeneratedConcept, log
from baseline_spl.common.llm_backend import load_upstream_code_generator
from baseline_spl.common.primitive_stats import build_stats_block
from baseline_spl.common.serialize_visual import demo_frames
from baseline_spl.VLM.demo2code.agent_text import Demo2CodeTextAgent

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "spl_demo2code_prompt.yaml"


VLM_SUMMARY_SYSTEM = '''You are watching a robot build a structure out of blocks.'''

VLM_SUMMARY_USER = '''The images below are the keyframes of ONE demonstration, in order. The first
image is the scene before anything was placed; each later image follows one placement.

Goal stated by the human: "{instruction}"

Describe the CONSTRUCTION PROCEDURE you observe: which block is added at each step, where it
goes relative to the block placed before it, and what pattern is repeated. Use only these
direction words for relations: LEFT, RIGHT, FRONT, BEHIND, TOP ("TOP" means stacked
vertically on top of another block). Count the total number of placements.

Answer in exactly this format and nothing else:

[[Is the new trajectory sufficiently summarized? (yes/no):]]
yes
[[Summarized trajectory:]]
<your description of the procedure, including the number of placements and the repeated pattern>
'''


class Demo2CodeVLMAgent(Demo2CodeTextAgent):
    '''Input: configs : BaselineConfig
              backend : LLMBackend (its call_vlm is used for stage 1)
    '''

    def __init__(self, configs, backend, prompt_path: str = None):
        # Deliberately skip Demo2CodeTextAgent.__init__'s serializer settings; we share
        # only its upstream plumbing and stage-2 handling.
        self.configs = configs
        self.backend = backend
        self.prompt_path = Path(prompt_path) if prompt_path else PROMPT_PATH
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_dict = yaml.safe_load(f)
        self.upstream = load_upstream_code_generator(backend, configs)

    # ------------------------------------------------------------------ #
    def _summarize_visually(self, demos: List[dict]) -> List[str]:
        '''One VLM call per demonstration -> upstream-tagged summarized trajectories.'''
        summaries = []
        for demo in demos:
            frames = demo_frames(demo,
                                 max_px=self.configs.vlm_max_image_px,
                                 max_keyframes=self.configs.vlm_max_keyframes)
            log(f"[demo2code-vlm] demo {demo.get('demo_id')}: {len(frames)} keyframe images")
            reply = self.backend.call_vlm(
                VLM_SUMMARY_SYSTEM,
                VLM_SUMMARY_USER.format(instruction=demo.get("language_instruction", "")),
                frames,
            )
            if "[[Summarized trajectory:]]" not in reply:
                # Repair the envelope rather than lose the content: upstream's
                # is_summarized()/prepare_demo_query() parse for these exact strings.
                log(f"[demo2code-vlm] demo {demo.get('demo_id')}: reply missing upstream tags; wrapping.")
                reply = ("[[Is the new trajectory sufficiently summarized? (yes/no):]]\nyes\n"
                         f"[[Summarized trajectory:]]\n{reply.strip()}")
            summaries.append(reply)
        return summaries

    def _spec_from_summaries(self, summaries: List[str], lang_goal: str, concept: str,
                             stats_block: str = "") -> str:
        '''Hand the VLM summaries to upstream's own recursion + merge.'''
        out_dir = Path(self.configs.run_dir) / "demo2code_artifacts" / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        cot_path = out_dir / "cot.txt"
        cot_path.write_text("", encoding="utf-8")

        # Upstream's summarize() appends its own rounds to cot.txt; record the vision
        # summaries first so the artifact shows the full chain, including the step that
        # replaced upstream's text summarization.
        with open(cot_path, "w", encoding="utf-8") as f:
            for i, summary in enumerate(summaries, 1):
                f.write(f"=================== VLM summary of demo {i} ===================\n")
                f.write(summary)
                f.write("\n\n")

        stage1 = self.prompt_dict["stage1"]
        if stats_block:
            stage1 = {key: {**value, "main": value["main"] + f"\n\n{stats_block}"}
                      for key, value in stage1.items()}
        return self.upstream.summarize(
            summaries, lang_goal,
            stage1["recursive_summarization"], stage1["summary_2_spec"],
            str(cot_path),
        )

    # ------------------------------------------------------------------ #
    def generate(self, demos: List[dict], sketch_infos: List[dict], shared) -> GeneratedConcept:
        # sketch_infos is None in the no-sketch condition; the model names the class.
        concept = sketch_infos[0]["concept"] if sketch_infos else None
        artifact_name = concept or demos[0].get("concept") or "unnamed"
        # One binding per demonstration: the argument value differs between demos, and
        # that variation is the evidence for a loop. SPL's Generalize gets the same list.
        demo_specs = [(d.get("language_instruction", ""),
                       shared.initialized_sketch(si) if si is not None else None)
                      for d, si in zip(demos, sketch_infos or [None] * len(demos))]

        stats_block = build_stats_block(shared.spl.executor, demos,
                                        enabled=self.configs.include_primitive_stats)
        # Concepts learned so far, so this one can be built out of them.
        library_block = dsl_prompt.build_library_block(shared.spl.concept_library)

        try:
            summaries = self._summarize_visually(demos)
            # Upstream takes a single high-level goal for the whole set, so stage 1 keeps
            # the first instruction; every demo's own instruction is used per-image in
            # _summarize_visually, and all of them reach stage 2 via demo_specs.
            spec = self._spec_from_summaries(summaries, demo_specs[0][0], artifact_name,
                                             stats_block)
        except Exception as exc:  # noqa: BLE001
            log(f"[demo2code-vlm] stage 1 failed for <{artifact_name}>: {exc}")
            return None

        # No privileged 3-D state may reach the model in this condition.
        assert "grid cell" not in spec, "3-D grid coordinates leaked into the VLM spec"

        code = codegen.generate_with_retries(
            self.backend,
            dsl_prompt.build_system_prompt(stats_block, require_signature=bool(sketch_infos),
                                           library_block=library_block),
            (f"{dsl_prompt.build_task_block(demo_specs)}\n"
             f"# Task specification, summarized from the demonstration videos\n{spec}\n"),
            wanted_name=concept,
            max_retries=self.configs.max_code_retries,
            log=log,
        )
        if code is None:
            return None

        info = {"variant": "vlm", "num_demos_summarized": len(summaries),
                "spec_chars": len(spec)}
        return self._result(code, concept, sketch_infos, shared, info)
