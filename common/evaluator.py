'''
evaluator.py

Execution feedback for the class-generating baselines (``use_evaluator_feedback``).

Runs a candidate class on every demonstration and reports what went wrong, for the retry
loop in codegen.py. Built from the parts of ``SPL._build_concept_class_evaluator`` that need
only the demonstration and the shared executor: the run, crash messages, the bookkeeping
check and the per-block final-state report.

Not given, because they come from SPL's Plan stage: the training-mode reward (SRN
likelihood with teacher forcing) and the pass reference (the reward of SPL's best MCTS
plans). A class passes on distance instead: it places as many blocks as the demonstration,
and each ends within ``sketch_val_state_error_threshold`` of the demo's final keyframe.
'''

from __future__ import annotations

import traceback

import numpy as np


class ClassEvaluator:
    '''``evaluator(code) -> (passed, score, report)``.

    score is minus the centroid MSE over every compared block, or -inf if construct()
    raised. The best-scoring code of every call is kept in ``best_code``, so a caller with
    several generation stages (Demo2Code's upstream pipeline, then repair) ranks them together.

    with_numbers=False is for image-only variants: the report names the wrong blocks but
    gives no distances, which come from 3-D meshes those variants are never shown.
    transform is applied before running (CaP's helper expansion), so what is scored is what
    gets registered.
    '''

    def __init__(self, shared, demos, sketch_infos, with_numbers=True, transform=None):
        self.spl = shared.spl
        self.concept = sketch_infos[0]["concept"]
        self.attributes = shared.attributes(sketch_infos[0])
        self.demos = demos
        self.values = [shared.values(si) for si in sketch_infos]
        self.with_numbers = with_numbers
        self.transform = transform
        self.tolerance = float(self.spl.configs.sketch_val_state_error_threshold)
        self.spare = int(getattr(self.spl.configs.generalize_config, "class_check_spare_objects", 5))
        self.best_code, self.best_score = None, float("-inf")
        self.record = {"calls": 0, "passed": False, "best_score": None}   # for the metric record

    def __call__(self, code):
        if self.transform is not None:
            code = self.transform(code)
        passed, score, report = self._evaluate(code)
        self.record["calls"] += 1
        self.record["passed"] = self.record["passed"] or passed
        if self.best_code is None or score > self.best_score:
            self.best_code, self.best_score = code, score
            self.record["best_score"] = score if np.isfinite(score) else None
        return passed, score, report

    def _evaluate(self, code):
        from SPL.model.executor import _format_substructure_action, make_focus, mesh_centroid
        from SPL.utils.metrics import demo_step_objects

        try:
            self.spl.concept_library.register_inductive_concepts(
                (self.concept, self.attributes, code), anonymous=True)
        except Exception as exc:  # noqa: BLE001
            return False, float("-inf"), f"FAILED: the class could not be registered: {exc}"

        executor = self.spl.executor
        instance_key = f"_substructure_{self.concept}_0"
        lines, squared = [], []
        crashed = misplaced = wrong_bookkeeping = False
        for i, (demo, values) in enumerate(zip(self.demos, self.values), 1):
            meshes = demo["meshes"]
            executor.register_state("inference", meshes[0], None, None, None, None, None)
            state = executor.query_current_state()
            # Start where SPL._predict_final_state does: the block that moved most in keyframe 1.
            first = int(np.argmax([np.linalg.norm(mesh_centroid(a) - mesh_centroid(b))
                                   for a, b in zip(meshes[0], meshes[1])]))
            state.set_focus(make_focus(mesh_centroid(meshes[1][first])))
            # Spare ids in list arguments expose blocks = list(objects), as in SPL's evaluator.
            run_values = {k: v + list(range(max(v) + 1, max(v) + 1 + self.spare))
                          if isinstance(v, list) and v and all(isinstance(j, int) for j in v) else v
                          for k, v in values.items()}
            state.namespace.update(run_values)
            try:
                new_state, _ , _steps, _done = executor.step(
                    state, _format_substructure_action(self.concept, 0, run_values))
            except Exception as exc:  # noqa: BLE001
                crashed = True
                where = traceback.extract_tb(exc.__traceback__)[-1].name
                lines.append(f"Demonstration {i}: construct() raised {type(exc).__name__}: {exc} (in {where}).")
                continue

            parts = []
            placed = [int(j) for j in new_state.objects_moved]
            steps = demo_step_objects(meshes, demo.get("objects_moved"))
            if len(placed) != len(steps):
                misplaced = True
                parts.append(f"construct() placed {len(placed)} block(s), the demonstration placed {len(steps)}")

            # Final-state report, as SPL's _final_state_report: each demonstrated block vs the last keyframe.
            distances = [(j, float(np.linalg.norm(mesh_centroid(new_state.state[j]) - mesh_centroid(meshes[-1][j]))))
                         for j in dict.fromkeys(steps)]
            distances.sort(key=lambda x: x[1], reverse=True)   # worst first
            squared += [d * d for _, d in distances]
            far = [(j, d) for j, d in distances if d > self.tolerance]
            misplaced = misplaced or bool(far)
            if self.with_numbers:
                parts.append(f"final-state centroid MSE = {np.mean([d * d for _, d in distances]):.6f} over "
                             f"{len(distances)} block(s); per-block distance (worst first): "
                             + ", ".join(f"block {j}: {d:.4f}" for j, d in distances))
                parts.append(f"blocks NOT within {self.tolerance:.4f} m of the demonstration's final state: "
                             + ", ".join(f"block {j} ({d:.4f})" for j, d in far) if far else
                             f"all blocks are within {self.tolerance:.4f} m of the demonstration's final state")
            else:
                parts.append("blocks NOT close to the demonstration's final state: "
                             + ", ".join(f"block {j}" for j, _ in far) if far else
                             "all blocks are close to the demonstration's final state")

            # Bookkeeping, copied from SPL's _build_concept_class_evaluator.
            problems = []
            try:
                inst = new_state.namespace[instance_key]
                blocks, key_blocks = list(inst.blocks), list(inst.key_blocks)
                if sorted(blocks) != sorted(placed):
                    problems.append(f"`blocks` returned {blocks} but construct() placed {placed}")
                if (placed and not key_blocks) or not set(key_blocks) <= set(placed):
                    problems.append(f"`key_blocks` returned {key_blocks}, but they must be objects construct() placed ({placed})")
                for sub in inst.substructures or []:
                    if not set(sub.blocks) <= set(placed):
                        problems.append(f"substructure {type(sub).__name__} reports `blocks` {list(sub.blocks)}, not all placed")
            except Exception as exc:  # noqa: BLE001
                problems = [f"reading blocks/key_blocks/substructures raised {type(exc).__name__}: {exc}"]
            wrong_bookkeeping = wrong_bookkeeping or bool(problems)
            lines.append(f"Demonstration {i}: " + "; ".join(parts + problems) + ".")

        if wrong_bookkeeping:
            lines.append("Bookkeeping is wrong. `objects` also holds ids this concept should not use, so `blocks`, "
                         "`key_blocks` and each substructure's `blocks` must come from the objects construct() "
                         "actually placed.")
        report = "\n".join(lines)
        score = float("-inf") if crashed or not squared else -float(np.mean(squared))
        if crashed or misplaced:
            return False, score, f"FAILED:\n{report}"
        if wrong_bookkeeping:
            return False, score, f"BOOKKEEPING:\n{report}"
        return True, score, f"OK:\n{report}"
