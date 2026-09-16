'''
harness.py

``BaselineHarness`` — the shared_sketch spine for every baseline.

SPL already owns everything downstream of program generation: the executor, the concept
library, the final-state / plan / program metrics, stability simulation and the
inference loop. A baseline therefore *holds* an ``SPL`` instance and replaces exactly
one step — how the concept class code is produced. Nothing in ``SPL/`` is modified.

``learn_all`` and ``infer_all`` deliberately mirror
``SPL/pipelines/learn_spl_concept.py`` (which is hard-wired to ``SPL.learn``) so the
metric JSONs come out in the same shape and diff cleanly against SPL's own runs.
'''

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


def log(message: str = "") -> None:
    print(f"\n[BASELINE] {message}")


class GeneratedConcept:
    '''What a baseline agent must return from ``generate``.

    concept_name : str            - class name, taken from the shared_sketch sketch
    attributes   : dict[str,type] - argument schema for register_inductive_concepts
    code         : str            - the class source
    info         : dict           - free-form provenance (chain-of-thought paths, spec, ...)
    '''

    def __init__(self, concept_name: str, attributes: Dict[str, type], code: str,
                 info: Dict[str, Any] = None):
        self.concept_name = concept_name
        self.attributes = attributes
        self.code = code
        self.info = info or {}


class BaselineHarness:
    '''Input: configs: BaselineConfig
              agent:   object exposing ``generate(demos, sketch_infos, shared_sketch) -> GeneratedConcept``
    '''

    def __init__(self, configs, agent):
        from SPL.model.spl import SPL
        from baseline_spl.common.config import assert_not_spl_library
        from baseline_spl.common.sketch import SharedSketch

        # SPL.__init__ loads load_concept_checkpoint for us, minus skip_loading_concepts.
        # Re-check both paths here so no caller can bypass the invariant by building a
        # config object directly.
        if configs.load_concept_checkpoint:
            assert_not_spl_library(configs.load_concept_checkpoint)
        assert_not_spl_library(configs.concept_save_path)
        # The evaluator runs each class on the demos with their sketched arguments. SayCan
        # generates no class, so it is not checked.
        if getattr(configs, "use_evaluator_feedback", False) and hasattr(agent, "generate"):
            if configs.sketch_mode == "none":
                raise ValueError("use_evaluator_feedback needs each demo's arguments before the class is "
                                 "registered, but sketch_mode='none' sketches only afterwards.")
            if not getattr(configs, "use_demo", True):
                raise ValueError("use_evaluator_feedback runs the class on the demonstrations, so "
                                 "use_demo=False would no longer be the instruction-only control.")

        self.configs = configs
        self.spl = SPL(configs)
        self.shared_sketch = SharedSketch(self.spl)
        self.agent = agent

        self.learned = list(self.spl.concept_library.inductive_concepts)
        if self.learned:
            log(f"Loaded {configs.load_concept_checkpoint} with {self.learned}")
        elif configs.load_concept_checkpoint:
            log(f"{configs.load_concept_checkpoint} holds no concept to load "
                f"(skip_loading_concepts={list(configs.skip_loading_concepts)}).")
        elif getattr(configs, "learn", True) and os.path.exists(configs.concept_save_path):
            # Only when this run learns: an inference-only run writes no library, so the same
            # message there would be a false alarm on every rerun of a finished experiment.
            log(f"WARNING: load_concept_checkpoint is None and {configs.concept_save_path} "
                f"exists. This run will OVERWRITE it and the metric files in {configs.run_dir}.")

        self._records_path = os.path.join(configs.run_dir, "training_metrics.json")
        self._times_path = os.path.join(configs.run_dir, "learning_times.json")
        # Read unconditionally and prune to the library, as SPL does. Merging is what keeps a
        # second invocation from truncating the files to the concepts it happened to learn;
        # pruning is what keeps them describing the library that is actually loaded, so a
        # concept left out by skip_loading_concepts carries no stale timing and leaves no gap
        # in `order`. With load_concept_checkpoint=None the library is empty, so both start empty
        # and the run rewrites them — the warning above says so.
        self._metric_records = self._prune_to_library(self._load_records(self._records_path))
        self._time_records = self._prune_to_library(self._load_records(self._times_path),
                                                    reindex=True)

    @staticmethod
    def _load_records(path: str) -> Dict[str, dict]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            log(f"Could not read {os.path.basename(path)} ({exc}); starting fresh.")
            return {}

    def _prune_to_library(self, records: Dict[str, dict], *, reindex: bool = False) -> Dict[str, dict]:
        '''Keep the records whose concept is in the loaded library.

        Records are keyed by the DATASET's concept, but with sketch_mode='none' the library
        holds the name the model invented, so `pred_concept` counts as being in it too.
        `reindex` renumbers `order` densely (0..n-1, relative order kept) so a dropped concept
        leaves no gap and a newly learned one continues collision-free from len(records) —
        SPL._load_learning_times does the same for the same reason.
        '''
        loaded = set(self.spl.concept_library.inductive_concepts)
        kept = sorted((r for concept, r in records.items()
                       if concept in loaded or r.get("pred_concept") in loaded),
                      key=lambda r: r.get("order", 0))
        if reindex:
            for order, record in enumerate(kept):
                record["order"] = order
        return {r["concept"]: r for r in kept}

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def learn_concept(self, demos: List[dict]) -> dict:
        '''Generate, register and score one concept. Returns its metric record.'''
        backend = getattr(self.agent, "backend", None)
        if backend is not None:
            backend.reset_ledger()

        no_sketch = self.configs.sketch_mode == "none"

        t0 = time.perf_counter()
        if no_sketch:
            # The signature is withheld from generation; it is recovered afterwards, once
            # the class is in the library for the sketch agent to ground against.
            sketch_infos = None
        else:
            sketch_infos = [self.shared_sketch.signature(d["language_instruction"], d)
                            for d in demos]
            sketch_infos = self.shared_sketch.corrected(sketch_infos, demos)
        t_sketch = time.perf_counter() - t0

        # Records are keyed by the dataset's concept, not the predicted one: in no-sketch
        # mode the model invents the name, and keying by that would break resume's skip
        # and stop the table lining up with SPL's.
        gt_concept = demos[0].get("concept") or (sketch_infos and sketch_infos[0]["concept"])

        t0 = time.perf_counter()
        generated = self.agent.generate(demos, sketch_infos, self.shared_sketch)
        t_generate = time.perf_counter() - t0

        if generated is None or not generated.code:
            log(f"FAILED to generate a class for <{gt_concept}>.")
            record = {"concept": gt_concept, "status": "generation_failed"}
            self._metric_records[gt_concept] = record
            self._flush()
            return record

        concept_name = generated.concept_name
        self.spl.concept_library.register_inductive_concepts(
            (concept_name, generated.attributes, generated.code))
        log(f"Registered concept <{concept_name}>.")

        status = "ok"
        if no_sketch:
            # Ground each instruction onto the class we just registered. Demonstrations
            # that will not ground are dropped with a warning rather than failing the run.
            t0 = time.perf_counter()
            grounded = [(self.shared_sketch.ground(d["language_instruction"], d, concept_name,
                                                   generated.attributes, log=log), d)
                        for d in demos]
            t_sketch += time.perf_counter() - t0
            pairs = [(si, d) for si, d in grounded if si is not None]
            if not pairs:
                status = "sketch_ungrounded"
                log(f"WARNING no demonstration grounded onto <{concept_name}>; "
                    f"the concept is registered but has no training metrics.")
            elif len(pairs) < len(demos):
                status = f"partially_grounded ({len(pairs)}/{len(demos)})"
            sketch_infos = [si for si, _ in pairs]
            demos = [d for _, d in pairs]

        metrics = {}
        if sketch_infos and getattr(self.configs.evaluation_config, "evaluate_metrics", True):
            try:
                metrics = self.spl._evaluate_training_metrics(sketch_infos, demos)
                def display_metric(name, precision):
                    value = metrics.get(name)
                    return "n/a" if value is None else format(value, precision)

                log(f"Metrics <{concept_name}>: "
                    f"final_mse={display_metric('final_state_mse', '.5f')} "
                    f"step_mse={display_metric('per_step_mse', '.5f')} "
                    f"iou={display_metric('mean_iou', '.4f')} "
                    f"plan_acc={display_metric('plan_accuracy', '.4f')} "
                    f"plan_len={display_metric('plan_length', '.3f')} "
                    f"program_acc={display_metric('program_accuracy', '.4f')}")
            except Exception as exc:  # noqa: BLE001
                log(f"Training-metric evaluation failed for <{concept_name}>: {exc}")
                status = f"metrics_failed: {exc}"

        # concept = the dataset's name (keeps resume and cross-run diffs working);
        # pred_concept = what the model actually registered. They differ only when the
        # model invented the name, i.e. sketch_mode="none". infer_all records both too.
        record = {"concept": gt_concept, "pred_concept": concept_name,
                  "status": status, **metrics}
        if "evaluator" in generated.info:   # calls, passed, best_score (use_evaluator_feedback)
            record["evaluator"] = generated.info["evaluator"]
        self._metric_records[gt_concept] = record

        self._time_records[gt_concept] = {
            "concept": gt_concept,
            "pred_concept": concept_name,
            "order": len(self._time_records),
            "num_demos": len(demos),
            "sketch_total": t_sketch,
            "generate_total": t_generate,
            "concept_total": t_sketch + t_generate,
            **(backend.ledger() if backend is not None else {}),
        }
        self._flush()

        if self.configs.concept_save_path:
            self.spl.save(self.configs.concept_save_path)
        return record

    def learn_all(self) -> None:
        '''Mirrors pipelines/learn_spl_concept.py:learn_spl_concept.'''
        from SPL.dataloader.datasets import iter_concept_demos

        cfg = self.configs
        concepts = list(cfg.concepts_to_learn or cfg.concepts_space)
        log("Attempting to learn: " + ", ".join(concepts))

        # Already in the library, as learn_spl_concept decides it with the same flag: the
        # library is the only source, and names are matched as-is, so a concept the model
        # registered under an invented name (sketch_mode="none") is not recognised and gets
        # relearned. With the flag off a loaded concept is relearned and
        # register_inductive_concepts raises, exactly as in SPL — drop it from the library
        # first with skip_loading_concepts.
        pending = []
        for concept in concepts:
            if cfg.ignore_learnt_concepts and concept in self.spl.concept_library.inductive_concepts:
                log(f"<{concept}> already learned; skipping.")
            else:
                pending.append(concept)

        # One concept's demonstrations in memory at a time: the whole dataset does not fit.
        for concept, demos in iter_concept_demos(
                cfg.dataset_name, cfg.train_dataset_dir, cfg.assets_dir,
                camera_view=cfg.camera_view, concepts=pending,
                num_demos=cfg.num_demos_per_concept, load_images=True):
            if not demos:
                log(f"No demonstrations found for <{concept}>; skipping.")
                continue
            log(f"Learning <{concept.upper()}> from {len(demos)} demonstration(s).")
            try:
                self.learn_concept(demos)
            except Exception as exc:  # noqa: BLE001
                log(f"Learning <{concept}> raised: {exc}")
                self._metric_records[concept] = {"concept": concept, "status": f"error: {exc}"}
                self._flush()
            del demos  # release before the next concept is loaded

        if cfg.concept_save_path:
            self.spl.save(cfg.concept_save_path)
        log(f"Done. Metrics -> {self._records_path}")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def infer_all(self) -> None:
        '''Mirrors pipelines/learn_spl_concept.py:infer, minus robot execution.'''
        from SPL.dataloader.datasets import build_inductive_structure_dataloader
        from SPL.utils.metrics import (compute_state_metrics, compute_plan_metrics,
                                       run_gt_program, run_predicted_program)
        from SPL.model.executor import mesh_centroid, make_focus

        cfg = self.configs
        ev = cfg.evaluation_config
        concepts = list(cfg.concepts_to_infer or cfg.concepts_space)

        # Without a library every instruction scores 'ungrounded', which looks like a
        # result but is a configuration mistake. Say so instead of producing the numbers.
        if not self.spl.concept_library.inductive_concepts:
            raise RuntimeError(
                "Inference with an empty concept library: nothing has been learned. "
                f"Set learn=True, or point load_concept_checkpoint at a learned library "
                f"such as {cfg.concept_save_path}."
            )

        log("Inferring: " + ", ".join(concepts))

        dataloader = build_inductive_structure_dataloader(
            cfg.dataset_name, cfg.test_dataset_dir, cfg.assets_dir,
            camera_view=cfg.camera_view, batch_size=1,
            num_workers=getattr(cfg, "num_workers", 4), shuffle=False,
            load_images=False, load_only=concepts,
        )

        records: List[dict] = []
        for batch in tqdm(dataloader, desc="Inferring", total=len(dataloader)):
            for data in batch:
                concept, instruction = data["concept"], data["language_instruction"]
                gt_states = data.get("meshes")
                try:
                    sketch_info = self.shared_sketch.signature(instruction, data)
                except Exception as exc:  # noqa: BLE001
                    records.append({"concept": concept, "demo_id": data.get("demo_id"),
                                    "status": f"sketch_failed: {exc}"})
                    continue

                if sketch_info["concept"] not in self.spl.concept_library.operators:
                    # Not a construction failure — the baseline never learned this concept.
                    records.append({"concept": concept, "demo_id": data.get("demo_id"),
                                    "status": "ungrounded", "pred_concept": sketch_info["concept"]})
                    continue

                try:
                    # As in SPL's infer(): start at the block that moved most between keyframes
                    # 0 and 1 (the first one placed). None lets SPL.execute fall back to a
                    # configured INIT_FOCUS_INFERENCE.
                    anchor_focus = None
                    if cfg.INIT_FOCUS_INFERENCE is None:
                        first = int(np.argmax([np.linalg.norm(mesh_centroid(a) - mesh_centroid(b))
                                               for a, b in zip(gt_states[0], gt_states[1])]))
                        anchor_focus = make_focus(mesh_centroid(gt_states[1][first]))
                    _plan, final_state = self.spl.execute(
                        instruction, gt_states[0], sketch_info, init_focus=anchor_focus)
                except Exception as exc:  # noqa: BLE001
                    records.append({"concept": concept, "demo_id": data.get("demo_id"),
                                    "status": f"execution_failed: {exc}"})
                    continue

                rec = {"concept": concept, "demo_id": data.get("demo_id"), "status": "ok",
                       "pred_concept": sketch_info["concept"]}
                try:
                    m = compute_state_metrics(
                        final_state.state, final_state.objects_moved, gt_states,
                        data.get("objects_moved"),
                        min_movement=ev.min_movement_distance, max_mse=ev.max_mse)
                    num_objects = len(gt_states[0])
                    init_code = self.shared_sketch.initialized_sketch(sketch_info)
                    m.update(compute_plan_metrics(
                        run_predicted_program(
                            self.spl.concept_library.inductive_concepts_definition,
                            init_code, num_objects),
                        run_gt_program(data.get("program"), num_objects)))
                    m.update(self.spl._simulate_stability(final_state, data, cfg.test_dataset_dir))
                    rec.update({k: m[k] for k in ev.metric_keys if k in m})
                    rec["_demo"] = data
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = f"metrics_failed: {exc}"
                records.append(rec)

        self._write_inference_metrics(records)

    def program_equivalence(self, record: dict) -> dict:
        '''Is the learned program correct for every argument? A hook so a baseline that
        produces no program (SayCan) can say so, instead of it looking like a failed check.'''
        return self.spl._check_program_equivalence(
            record.get("pred_concept", record.get("concept")), record.get("_demo") or {})

    def _write_inference_metrics(self, records: List[dict]) -> None:
        cfg = self.configs
        keys = cfg.evaluation_config.metric_keys
        ok = [r for r in records if r.get("status") == "ok"]

        def _mean(recs, k):
            vals = [r[k] for r in recs if k in r and r[k] is not None]
            return float(np.mean(vals)) if vals else None

        by_concept = defaultdict(list)
        for r in ok:
            by_concept[r["concept"]].append(r)

        per_concept = {c: {**{k: _mean(rs, k) for k in keys}, "num_demos": len(rs)}
                       for c, rs in by_concept.items()}
        # program_accuracy is a property of the concept, not of one demo.
        for c, rs in by_concept.items():
            eq = self.program_equivalence(rs[0])
            per_concept[c].update({k: v for k, v in eq.items() if k != "program_accuracy"})
            per_concept[c]["program_accuracy"] = eq.get("program_accuracy")

        for r in records:
            r.pop("_demo", None)

        decided = [v["program_accuracy"] for v in per_concept.values()
                   if v.get("program_accuracy") is not None]
        overall = {k: _mean(ok, k) for k in keys}
        overall["program_accuracy"] = float(np.mean(decided)) if decided else None
        overall["program_concepts_decided"] = len(decided)
        overall["program_concepts_undecided"] = len(per_concept) - len(decided)
        overall["num_ok"] = len(ok)
        overall["num_total"] = len(records)
        # Failures are categorised, not silently averaged away.
        status_counts = defaultdict(int)
        for r in records:
            status_counts[str(r.get("status", "?")).split(":")[0]] += 1
        overall["status_counts"] = dict(status_counts)

        path = os.path.join(cfg.run_dir, "inference_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"overall": overall, "per_concept": per_concept, "per_demo": records},
                      f, indent=2)
        log(f"Inference metrics -> {path}")
        log("Overall: " + "  ".join(
            f"{k}={'n/a' if overall.get(k) is None else format(overall[k], '.5f')}" for k in keys))
        log(f"Statuses: {dict(status_counts)}")

    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        os.makedirs(self.configs.run_dir, exist_ok=True)
        with open(self._records_path, "w", encoding="utf-8") as f:
            json.dump(self._metric_records, f, indent=2)
        with open(self._times_path, "w", encoding="utf-8") as f:
            json.dump(self._time_records, f, indent=2)
