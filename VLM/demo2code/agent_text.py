'''
agent_text.py

Demo2Code baseline, text variant (Wang et al., NeurIPS 2023).

Stages 1 and 2 are the authors' released implementation, unmodified: we put the clone on
sys.path, import ``code_gen_helper.code_generator`` and hand it our domain prompt YAML.
Only two things are ours — the demonstration serializer (which emits their exact
``[Scenario i]`` / ``State 2:`` format, so their parser and recursive summarization run
untouched) and the LLM backend, substituted so every method in the comparison uses the
same model as SPL.

Stage 0 is the shared SketchAgent, identical across all methods.

Retries are structural only. SPL's execution-grounded evaluator is deliberately not
available here — that loop is an SPL contribution.
'''

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml

from baseline_spl.common import codegen, dsl_prompt
from baseline_spl.common.evaluator import ClassEvaluator
from baseline_spl.common.harness import GeneratedConcept, log
from baseline_spl.common.llm_backend import load_upstream_code_generator
from baseline_spl.common.primitive_stats import build_stats_block
from baseline_spl.common.serialize_text import to_demo2code_text

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "spl_demo2code_prompt.yaml"


class Demo2CodeTextAgent:
    '''Reads coordinate_mode and include_primitive_stats from the active config file.'''

    def __init__(self, configs, backend, prompt_path: str = None):
        self.configs = configs
        self.backend = backend
        self.prompt_path = Path(prompt_path) if prompt_path else PROMPT_PATH
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_dict = yaml.safe_load(f)
        # Imports the authors' code with our backend wired in.
        self.upstream = load_upstream_code_generator(backend, configs)

    # ------------------------------------------------------------------ #
    def _run_upstream(self, demo_text: str, concept: str, sketches_block: str,
                      stats_block: str = "", library_block: str = ""):
        '''Drive the authors' two-stage pipeline and return (code, spec_text).'''
        out_dir = Path(self.configs.run_dir) / "demo2code_artifacts" / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        code_path = out_dir / "code.txt"
        cot_path = out_dir / "cot.txt"

        # The direction statistics have to reach stage 1 as well: the summarizer sees bare
        # coordinates and cannot name a direction without them. env_info (the demo header)
        # reaches only stage 2, so it is not a substitute — inject into the prompt bodies.
        stage1 = self.prompt_dict["stage1"]
        if stats_block:
            stage1 = {key: {**value, "main": value["main"] + f"\n\n{stats_block}"}
                      for key, value in stage1.items()}

        # Stage 2 needs the target signature; upstream passes only the task spec through,
        # so the sketches are appended to the specification that reaches it. One per
        # demonstration, because the argument value differs between them. With no sketch,
        # the model is told the structural requirement instead and names the class itself.
        signature_note = (
            f"\n\n# Initialized sketches for THIS task. The argument VALUE\n"
            f"# differs per demonstration; the class name and argument NAMES\n"
            f"# must match exactly.\n{sketches_block}\n"
            if sketches_block else
            "\n\n# Choose the class name and argument names yourself. __init__ must take\n"
            "# one numeric argument (annotated int) and `objects: list`.\n")
        prompt_dict = {
            "stage1": stage1,
            "stage2": {
                "spec_2_highlevelcode": {
                    "main": self.prompt_dict["stage2"]["spec_2_highlevelcode"]["main"]
                            + (f"\n\n{stats_block}" if stats_block else "")
                            + (f"\n\n{library_block}" if library_block else "")
                            + signature_note,
                    "examples": self.prompt_dict["stage2"]["spec_2_highlevelcode"]["examples"],
                }
            },
        }

        self.upstream.code_generator(
            "demo2code", prompt_dict, demo_text,
            code_save_path=str(code_path), cot_save_path=str(cot_path),
        )

        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        spec = cot_path.read_text(encoding="utf-8") if cot_path.exists() else ""
        return code, spec

    @staticmethod
    def _clean(code: str) -> str:
        '''Undo upstream's fence stripping.

        ``spec_to_high_level_code`` does ``code.split('```')[1]``, which keeps the language
        tag when the model emits ```python. Our stage-2 prompt asks for an untagged fence,
        but models ignore that often enough to be worth handling.
        '''
        code = code.strip()
        for tag in ("python\n", "py\n"):
            if code.startswith(tag):
                code = code[len(tag):]
        return code.strip()

    # ------------------------------------------------------------------ #
    def generate(self, demos: List[dict], sketch_infos: List[dict], shared) -> GeneratedConcept:
        # sketch_infos is None in the no-sketch condition; the model names the class.
        concept = sketch_infos[0]["concept"] if sketch_infos else None
        # One binding per demonstration: the argument value differs between demos, and
        # that variation is the evidence for a loop. SPL's Generalize gets the same list.
        demo_specs = [(d.get("language_instruction", ""),
                       shared.initialized_sketch(si) if si is not None else None)
                      for d, si in zip(demos, sketch_infos or [None] * len(demos))]
        sketches_block = "\n".join(f"```\n{init.strip()}\n```"
                                   for _instruction, init in demo_specs
                                   if init is not None)

        # Artifacts are filed under the dataset's concept, which exists in both modes.
        artifact_name = concept or demos[0].get("concept") or "unnamed"

        demo_text = to_demo2code_text(demos, coordinate_mode=self.configs.coordinate_mode)
        stats_block = build_stats_block(shared.spl.executor, demos,
                                        enabled=self.configs.include_primitive_stats)
        # Concepts learned so far, so this one can be built out of them.
        library_block = dsl_prompt.build_library_block(shared.spl.concept_library)

        try:
            raw_code, spec = self._run_upstream(demo_text, artifact_name, sketches_block,
                                                stats_block, library_block)
        except Exception as exc:  # noqa: BLE001
            log(f"[demo2code-text] upstream pipeline failed for <{artifact_name}>: {exc}")
            return None

        code = self._clean(raw_code)
        info = {"variant": "text", "spec_chars": len(spec),
                "artifacts": str(Path(self.configs.run_dir) / "demo2code_artifacts"
                                 / artifact_name)}

        # Scores upstream's class and the repair attempts together (best_code spans both).
        evaluator = (ClassEvaluator(shared, demos, sketch_infos)
                     if self.configs.use_evaluator_feedback else None)
        if evaluator is not None:
            info["evaluator"] = evaluator.record

        # Upstream has no structural validation of its own; apply the same check and the
        # same retry budget every other baseline gets, using the produced spec as context.
        problem = "which was not a valid concept class"
        try:
            codegen.validate(code)
            if concept:
                code = codegen.ensure_class_name(code, concept)
                codegen.validate(code)
            if evaluator is None:
                return self._result(code, concept, sketch_infos, shared, info)
            passed, _score, report = evaluator(code)
            if passed:
                return self._result(code, concept, sketch_infos, shared, info)
            log(f"[demo2code-text] upstream class did not reproduce the demonstrations; "
                f"retrying with the report.\n{report}")
            problem = f"which did not reproduce the demonstrations.\nEvaluation report: {report}"
        except Exception as exc:  # noqa: BLE001
            log(f"[demo2code-text] upstream output invalid ({exc}); retrying structurally.")

        repaired = codegen.generate_with_retries(
            self.backend,
            dsl_prompt.build_system_prompt(stats_block, require_signature=bool(sketch_infos),
                                           library_block=library_block),
            (f"{dsl_prompt.build_task_block(demo_specs)}\n"
             f"# Task specification derived from the demonstrations\n{spec}\n\n"
             f"# A previous attempt produced this, {problem}\n"
             f"```python\n{code[:4000]}\n```\n"),
            wanted_name=concept,
            max_retries=self.configs.max_code_retries,
            evaluator=evaluator,
            log=log,
        )
        if repaired is None:
            return None
        info["repaired"] = True
        return self._result(repaired, concept, sketch_infos, shared, info)

    @staticmethod
    def _result(code, concept, sketch_infos, shared, info) -> GeneratedConcept:
        '''With a sketch the name and types come from it; without one they are whatever
        the model wrote, recovered from the class's own __init__.'''
        if sketch_infos:
            return GeneratedConcept(concept, shared.attributes(sketch_infos[0]), code, info)
        name, attributes = codegen.class_signature(code)
        return GeneratedConcept(name, attributes, code, info)
