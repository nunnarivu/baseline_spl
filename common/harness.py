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
from collections import OrderedDict, defaultdict
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

        # SPL.__init__ loads load_concept_checkpoint for us; config resolved it from
        # resume / resume_from. Re-check here so no caller can bypass the invariant.
        if configs.load_concept_checkpoint:
            assert_not_spl_library(configs.load_concept_checkpoint)

        self.configs = configs
        self.spl = SPL(configs)
        self.shared_sketch = SharedSketch(self.spl)
        self.agent = agent

        self.learned = list(self.spl.concept_library.inductive_concepts)
        if self.learned:
            log(f"Resumed from {configs.load_concept_checkpoint} with {self.learned}")
        elif configs.resume:
            log("Nothing to resume from; starting with an empty concept library.")
        elif os.path.exists(configs.concept_save_path):
            log(f"WARNING: resume=False and {configs.concept_save_path} exists. "
                f"This run will OVERWRITE it and the metric files in {configs.run_dir}.")

        self._records_path = os.path.join(configs.run_dir, "training_metrics.json")
        self._times_path = os.path.join(configs.run_dir, "learning_times.json")
        # Merge with previous results when resuming, otherwise a second invocation
        # truncates the metric files to only the concepts it happened to learn.
        self._metric_records = self._load_records(self._records_path) if configs.resume else {}
        self._time_records = self._load_records(self._times_path) if configs.resume else {}

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
            sketch_infos = [self.shared_sketch.signature(d["language_instruction"])
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
            grounded = [(self.shared_sketch.ground(d["language_instruction"], concept_name,
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
            except Exception as exc:  # noqa: BLE001
                log(f"Training-metric evaluation failed for <{concept_name}>: {exc}")
                status = f"metrics_failed: {exc}"

        # concept = the dataset's name (keeps resume and cross-run diffs working);
        # pred_concept = what the model actually registered. They differ only when the
        # model invented the name, i.e. sketch_mode="none". infer_all records both too.
        record = {"concept": gt_concept, "pred_concept": concept_name,
                  "status": status, **metrics}
        self._metric_records[gt_concept] = record
        breakpoint()

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
        from SPL.dataloader.datasets import build_inductive_structure_dataloader

        cfg = self.configs
        concepts = list(cfg.concepts_to_learn or cfg.concepts_space)
        log("Attempting to learn: " + ", ".join(concepts))

        dataloader = build_inductive_structure_dataloader(
            cfg.dataset_name, cfg.train_dataset_dir, cfg.assets_dir,
            camera_view='fixed_robot_diag_45', batch_size=1,
            num_workers=getattr(cfg, "num_workers", 4), shuffle=False,
            load_images=True, load_only=concepts,
        )

        demos_by_concept = OrderedDict({c: [] for c in concepts})
        for batch in tqdm(dataloader, desc="Loading demonstrations", total=len(dataloader)):
            for data in batch:
                if data["concept"] in demos_by_concept:
                    demos_by_concept[data["concept"]].append(data)
        for concept, demos in demos_by_concept.items():
            if not demos:
                log(f"No demonstrations found for <{concept}>; skipping.")
                continue
            # Already learned by an earlier run. Skipping is required, not just thrifty:
            # register_inductive_concepts asserts the name is new. Check the metric records
            # as well as the library, because with sketch_mode="none" the registered name
            # is the model's invention and need not equal the dataset's concept.
            if (concept in self.spl.concept_library.inductive_concepts
                    or concept in self._metric_records):
                log(f"<{concept}> already learned; skipping.")
                continue
            n = min(len(demos), cfg.num_demos_per_concept)
            log(f"Learning <{concept.upper()}> from {n} demonstration(s).")
            try:
                self.learn_concept(demos[:n])
            except Exception as exc:  # noqa: BLE001
                log(f"Learning <{concept}> raised: {exc}")
                self._metric_records[concept] = {"concept": concept, "status": f"error: {exc}"}
                self._flush()

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
                "Set learn=True, or resume=True to load "
                f"{cfg.resume_from or cfg.concept_save_path}."
            )

        log("Inferring: " + ", ".join(concepts))

        dataloader = build_inductive_structure_dataloader(
            cfg.dataset_name, cfg.test_dataset_dir, cfg.assets_dir,
            camera_view='fixed_robot_diag_45', batch_size=1,
            num_workers=getattr(cfg, "num_workers", 4), shuffle=False,
            load_images=False, load_only=concepts,
        )

        records: List[dict] = []
        for batch in tqdm(dataloader, desc="Inferring", total=len(dataloader)):
            for data in batch:
                concept, instruction = data["concept"], data["language_instruction"]
                gt_states = data.get("meshes")
                try:
                    sketch_info = self.shared_sketch.signature(instruction)
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
                    anchor_focus = make_focus(mesh_centroid(gt_states[1][0]))
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
            eq = self.spl._check_program_equivalence(rs[0].get("pred_concept", c),
                                                     rs[0].get("_demo") or {})
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
