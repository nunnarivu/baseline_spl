'''
harness.py

SayCan's run loop. Subclasses ``BaselineHarness`` to reuse the SPL instance, the fairness
asserts, metric merging and the metric writers, and overrides only learn and inference —
because SayCan produces no concept class, so the generate/register path does not apply.

    learn      solve each training instruction blind and cache the action sequence
    inference  retrieve that concept's cached plans as worked examples, solve, execute, score

The cached plans are the agent's own output. They are never ground truth and never derived
from the demonstration's keyframes, so no planning of SPL's leaks into them.
'''

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List

from tqdm import tqdm

from baseline_spl.common.harness import BaselineHarness, log

PLAN_LIBRARY = "plan_library.json"


class SayCanHarness(BaselineHarness):

    def __init__(self, configs, agent):
        super().__init__(configs, agent)
        self._plans_path = os.path.join(configs.run_dir, PLAN_LIBRARY)
        # Keyed by concept -> list of {demo_id, instruction, actions, stop_reason}.
        self._plans: Dict[str, List[dict]] = (
            self._load_records(self._plans_path) if configs.resume else {})
        if self._plans:
            log(f"Loaded cached plans for {sorted(self._plans)}")

    # ------------------------------------------------------------------ #
    def _prepare(self, demo):
        '''Put the executor on this demo's initial scene with the anchor focus, and
        return the live state. Mirrors what SPL.execute does before running a class.'''
        from SPL.model.executor import make_focus, mesh_centroid

        meshes = demo["meshes"]
        self.spl.executor.register_state("inference", meshes[0], None, None, None, None, None)
        state = self.spl.executor.query_current_state()
        state.set_focus(make_focus(mesh_centroid(meshes[1][0])))
        return state

    def _solve(self, demo, cached_plans, recursive):
        '''Run the agent on one instruction and return (actions, stop_reason, final state).

        `recursive` is the caller's, not the config's: learning and inference each pick
        their own mode, since inference runs over the whole test set and is where the
        per-action cost actually lands.
        '''
        from baseline_spl.common.primitive_stats import build_stats_block

        self._prepare(demo)
        stats = build_stats_block(self.spl.executor, [demo],
                                  enabled=self.configs.include_primitive_stats)
        actions, stop_reason = self.agent.solve(
            demo["language_instruction"], [demo], self.spl.executor,
            self.spl.executor.action_space, cached_plans=cached_plans, stats_block=stats,
            recursive=recursive)
        return actions, stop_reason, self.spl.executor.query_current_state()

    @staticmethod
    def _replay(actions: List[str], num_objects: int):
        '''Replay the flat action list on the ideal executor to get a trace, the same
        object compute_plan_metrics expects from a program.'''
        from SPL.utils.metrics import IdealExecutor

        executor = IdealExecutor(num_objects)
        namespace = dict(executor.namespace())
        for action in actions:
            try:
                exec(action, namespace)
            except Exception as exc:  # noqa: BLE001
                log(f"[saycan] replay stopped at {action!r}: {exc}")
                break
        return executor.trace()

    # ------------------------------------------------------------------ #
    # Learn: cache a plan per training instruction
    # ------------------------------------------------------------------ #
    def _missing_demos(self, concept: str, demos: List[dict]) -> List[dict]:
        '''The demos in this list that have no cached plan yet.

        Matched on demo_id rather than counted, so a library holding the right number of
        plans for a different set of demonstrations is not mistaken for this one.
        '''
        cached_ids = {p.get("demo_id") for p in self._plans.get(concept, [])}
        return [d for d in demos if d.get("demo_id") not in cached_ids]

    def _cache_plan(self, concept: str, entry: dict) -> None:
        '''Append one plan and write the library out.

        Per demonstration, not once per concept: an interrupted run has to keep the plans
        it already paid for, and resume can only skip a demo whose plan reached disk.
        '''
        self._plans.setdefault(concept, []).append(entry)
        with open(self._plans_path, "w", encoding="utf-8") as f:
            json.dump(self._plans, f, indent=2)

    def learn_concept(self, demos: List[dict]) -> dict:
        backend = getattr(self.agent, "backend", None)
        if backend is not None:
            backend.reset_ledger()

        concept = demos[0].get("concept")
        started = time.perf_counter()

        todo = self._missing_demos(concept, demos)
        if len(todo) < len(demos):
            log(f"<{concept}> already has {len(demos) - len(todo)} of {len(demos)} "
                f"plan(s); solving the rest.")

        failures = []
        for demo in todo:
            try:
                actions, stop_reason, _final = self._solve(
                    demo, cached_plans=[], recursive=self.configs.recursive_learn)
            except Exception as exc:  # noqa: BLE001
                # Kept in the record, not only in the log: a concept that failed outright
                # otherwise leaves an empty plan list and no reason for it.
                log(f"[saycan] solving <{concept}/{demo.get('demo_id')}> failed: {exc}")
                failures.append(f"{demo.get('demo_id')}: {exc}")
                continue
            self._cache_plan(concept, {"demo_id": demo.get("demo_id"),
                                       "instruction": demo["language_instruction"],
                                       "actions": actions, "stop_reason": stop_reason})

        # Diagnostic only: how good are the plans inference will reuse as examples?
        # Computed from the demo's ground-truth program, never fed back into the agent.
        # Scored over every cached plan whose demonstration is in hand, so a resumed
        # concept reports on all of its plans, not just the ones solved this run.
        by_id = {d.get("demo_id"): d for d in demos}
        cached = self._plans.get(concept, [])
        accuracies, stops = [], []
        for plan in cached:
            stops.append(plan.get("stop_reason"))
            demo = by_id.get(plan.get("demo_id"))
            if demo is not None:
                accuracies.append(self._plan_accuracy(plan["actions"], demo))
        scored = [a for a in accuracies if a is not None]
        mean_accuracy = sum(scored) / len(scored) if scored else None

        if not cached:
            status = "no_plan_cached"
        elif len(cached) < len(demos):
            status = "incomplete"
        else:
            status = "ok"
        record = {"concept": concept, "status": status,
                  "num_cached_plans": len(cached), "num_demos": len(demos),
                  "cached_plan_accuracy": mean_accuracy,
                  "stop_reasons": stops, "failures": failures}
        self._metric_records[concept] = record
        self._time_records[concept] = {
            "concept": concept, "order": len(self._time_records),
            "num_demos": len(demos), "concept_total": time.perf_counter() - started,
            **(backend.ledger() if backend is not None else {}),
        }
        self._flush()
        log(f"Cached {len(cached)} plan(s) for <{concept}>; "
            f"plan_accuracy={mean_accuracy}")
        return record

    def _plan_accuracy(self, actions, demo):
        from SPL.utils.metrics import compute_plan_metrics, run_gt_program

        num_objects = len(demo["meshes"][0])
        gt = run_gt_program(demo.get("program"), num_objects)
        metrics = compute_plan_metrics(self._replay(actions, num_objects), gt)
        return metrics.get("plan_accuracy")

    def learn_all(self) -> None:
        from SPL.dataloader.datasets import build_inductive_structure_dataloader

        cfg = self.configs
        concepts = list(cfg.concepts_to_learn or cfg.concepts_space)
        log("Caching plans for: " + ", ".join(concepts))

        loader = build_inductive_structure_dataloader(
            cfg.dataset_name, cfg.train_dataset_dir, cfg.assets_dir,
            camera_view="fixed_robot_diag_45", batch_size=1,
            num_workers=getattr(cfg, "num_workers", 4), shuffle=False,
            load_images=True, load_only=concepts)

        by_concept = OrderedDict({c: [] for c in concepts})
        for batch in tqdm(loader, desc="Loading demonstrations", total=len(loader)):
            for data in batch:
                if data["concept"] in by_concept:
                    by_concept[data["concept"]].append(data)

        for concept, demos in by_concept.items():
            if not demos:
                log(f"No demonstrations found for <{concept}>; skipping.")
                continue
            n = min(len(demos), cfg.num_demos_per_concept)
            wanted = demos[:n]
            missing = self._missing_demos(concept, wanted)
            if not missing:
                log(f"<{concept}> already has plans for all {n} demonstration(s); skipping.")
                continue
            log(f"Solving <{concept.upper()}> from {len(missing)} of {n} "
                f"demonstration(s), blind.")
            try:
                self.learn_concept(wanted)
            except Exception as exc:  # noqa: BLE001
                log(f"Caching plans for <{concept}> raised: {exc}")
                self._metric_records[concept] = {"concept": concept, "status": f"error: {exc}"}
                self._flush()
        log(f"Done. Plans -> {self._plans_path}")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def program_equivalence(self, record: dict) -> dict:
        '''SayCan emits a flat action sequence, so there is no program to check for
        every argument. Said explicitly, because a blank column here is the finding.'''
        return {"program_accuracy": None, "program_verdict": "no_program",
                "program_reason": "SayCan emits an action sequence, not a program",
                "program_counterexample": None}

    def infer_all(self) -> None:
        from SPL.dataloader.datasets import build_inductive_structure_dataloader
        from SPL.utils.metrics import (compute_plan_metrics, compute_state_metrics,
                                       run_gt_program)

        cfg = self.configs
        ev = cfg.evaluation_config
        concepts = list(cfg.concepts_to_infer or cfg.concepts_space)
        log("Inferring: " + ", ".join(concepts))

        loader = build_inductive_structure_dataloader(
            cfg.dataset_name, cfg.test_dataset_dir, cfg.assets_dir,
            camera_view="fixed_robot_diag_45", batch_size=1,
            num_workers=getattr(cfg, "num_workers", 4), shuffle=False,
            load_images=self.agent.demo_modality == "images", load_only=concepts)

        records: List[dict] = []
        for batch in tqdm(loader, desc="Inferring", total=len(loader)):
            for data in batch:
                concept = data["concept"]
                cached = self._plans.get(concept, [])[: cfg.plan_library_top_k]
                rec = {"concept": concept, "demo_id": data.get("demo_id"), "status": "ok",
                       "num_cached_plans": len(cached)}
                try:
                    actions, stop_reason, final_state = self._solve(
                        data, cached_plans=cached, recursive=cfg.recursive_infer)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = f"execution_failed: {exc}"
                    records.append(rec)
                    continue

                rec["num_actions"] = len(actions)
                # Only 'done' is the agent judging the structure complete; 'all_placed'
                # means the object list ran out and handed it the count.
                rec["stop_reason"] = stop_reason
                try:
                    gt_states = data["meshes"]
                    m = compute_state_metrics(
                        final_state.state, final_state.objects_moved, gt_states,
                        data.get("objects_moved"),
                        min_movement=ev.min_movement_distance, max_mse=ev.max_mse)
                    num_objects = len(gt_states[0])
                    m.update(compute_plan_metrics(self._replay(actions, num_objects),
                                                  run_gt_program(data.get("program"),
                                                                 num_objects)))
                    m.update(self.spl._simulate_stability(final_state, data,
                                                          cfg.test_dataset_dir))
                    rec.update({k: m[k] for k in ev.metric_keys if k in m})
                    rec["_demo"] = data
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = f"metrics_failed: {exc}"
                records.append(rec)

        self._write_inference_metrics(records)
