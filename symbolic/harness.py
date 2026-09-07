'''
harness.py

Plug the search into SPL's evaluation, so B3-a is scored by exactly the metrics every other
baseline is scored by.

A DreamCoder run is not per-concept independent -- it needs every task at once and grows a
shared library -- so the `agent.generate(...)` contract `BaselineHarness` expects does not
fit. Follow the SayCan precedent (`VLM/saycan/harness.py`) and override `learn_all` only:

    load demos -> sketch -> build tasks -> run the wake/sleep loop
        -> lower each solution to a concept class
        -> hand it to the inherited `learn_concept` path via a PrecomputedAgent

Going back through `learn_concept` rather than registering directly is what keeps timings,
resume, record keying, `spl.save` and the training-metric call identical to the other
baselines. Inference, `program_equivalence` and `_write_inference_metrics` are inherited
untouched.
'''

from __future__ import annotations

import gc
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional

from baseline_spl.common.harness import BaselineHarness, GeneratedConcept, log
from baseline_spl.symbolic import driver
from baseline_spl.symbolic.lower import lower, saved_is_exact
from baseline_spl.symbolic import evaluate
from baseline_spl.symbolic.tasks import build_tasks, summarise_demo


class PrecomputedAgent:
    '''Returns a class the search already found. `learn_concept` calls `generate`; by then
    the work is done, so this just hands back the right entry.'''

    def __init__(self):
        self.results: Dict[str, GeneratedConcept] = {}
        self.backend = None

    def add(self, concept: str, generated: GeneratedConcept) -> None:
        self.results[concept] = generated

    def generate(self, demos, sketch_infos, shared_sketch) -> Optional[GeneratedConcept]:
        concept = demos[0].get("concept")
        return self.results.get(concept)


class SearchHarness(BaselineHarness):
    '''B3-a: DreamCoder-style program search over SPL's DSL.'''

    def __init__(self, configs):
        super().__init__(configs, PrecomputedAgent())
        # Built here rather than in learn_all so an invalid (observation_mode, evaluator)
        # pair fails before any demonstration is loaded, and so B3-b's proposer can gate on
        # the same rule the driver accepts by.
        self.evaluator = evaluate.build(configs)
        self.observation_mode = getattr(configs, "observation_mode", "lattice")

    # ------------------------------------------------------------------ #

    # Payload a demonstration carries that the search never reads: trimesh geometry for every
    # keyframe, and the image channels. For 16 concepts this is ~53 GB, and it is what makes
    # the wake phase's worker pool ruinous -- `fork` shares it copy-on-write, but CPython
    # refcounts every object it touches, so each of 16 workers steadily copies the lot. Three
    # concurrent runs reached 231 GB of 251 GB and one was OOM-killed with no traceback.
    HEAVY_KEYS = ("meshes", "rgbs", "depths", "masks")

    def _load_demos(self, concepts, *, summarise: bool = False) -> "OrderedDict":
        """Demonstrations grouped by concept.

        With `summarise=True` each demonstration is reduced to what the search needs -- its
        placement offsets and, under a continuous observation mode, its SRN table -- and its
        geometry is dropped **as it streams**, so peak memory is one batch rather than the
        whole dataset. That is what makes the run's footprint independent of how many concepts
        the dataset holds: 16 concepts held ~53 GB of trimesh at once, so 100 would have wanted
        ~330 GB against 251 GB of RAM.

        Called twice per run: once summarised, to build tasks and search; once in full, to
        score (which genuinely needs the meshes back -- see `learn_all`).
        """
        from SPL.dataloader.datasets import build_inductive_structure_dataloader

        cfg = self.configs
        dataloader = build_inductive_structure_dataloader(
            cfg.dataset_name, cfg.train_dataset_dir, cfg.assets_dir,
            camera_view="fixed_robot_diag_45", batch_size=1,
            num_workers=getattr(cfg, "num_workers", 4), shuffle=False,
            # B3-a never needs pixels; B3-b does when its demonstration modality includes
            # images. Loading them unconditionally would slow the control run for nothing.
            load_images=getattr(cfg, "demo_modality", "coords") in ("images", "both"),
            load_only=list(concepts))

        want_srn = summarise and self.observation_mode == "continuous"
        executor = self.spl.executor if want_srn else None

        grouped = OrderedDict({c: [] for c in concepts})
        for batch in dataloader:
            for data in batch:
                if data["concept"] not in grouped:
                    continue
                if summarise:
                    try:
                        summarise_demo(data, executor, want_srn)
                    except Exception as exc:  # noqa: BLE001
                        log(f"  cannot summarise demo {data.get('demo_id')} of "
                            f"<{data.get('concept')}>: {type(exc).__name__}: {exc}")
                        continue
                    for key in self.HEAVY_KEYS:
                        data[key] = None
                grouped[data["concept"]].append(data)
        if summarise:
            self._reclaim()
        return grouped

    @staticmethod
    def _literal_ceiling(tasks, cfg) -> int:
        """Largest integer literal to put in the grammar.

        DreamCoder's tower domain hard-codes `Primitive(str(j), tint, j) for j in range(1, 50)`
        because its towers reach that size. The right analogue is not to copy 49 but to cover
        *our* range: a literal larger than anything a task needs cannot make a program
        expressible, only dearer, because every extra production dilutes the probability mass
        of the rest.

        Derived from the task parameters rather than hard-coded, so a dataset that grows from
        16 concepts to 100 -- with structures larger than anything here -- widens the grammar
        automatically instead of silently running with literals too small to express them.
        The headroom covers derived sizes: `pyramid`'s row length is 2n-1, so a parameter of n
        needs literals up to about twice that.

        `int_literals_upto` in the config overrides, for ablations.
        """
        override = getattr(cfg, "int_literals_upto", None)
        if override:
            return int(override)
        largest = max((n for task in tasks for n in task.parameters), default=6)
        return max(12, 2 * int(largest) + 1)

    @staticmethod
    def _reclaim() -> None:
        """Hand freed pages back to the OS.

        Dropping references is not enough: CPython returns freed arenas to its own allocator,
        so RSS stays put and every later `fork` still maps that footprint. Measured during the
        16-concept runs: parent RSS 53.4 GB against a PSS of 3.3 GB.
        """
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as exc:  # noqa: BLE001 - glibc-only; never fatal
            log(f"malloc_trim unavailable ({type(exc).__name__}); "
                f"freed memory stays in the allocator.")

    @classmethod
    def _release_demo_payload(cls, grouped) -> int:
        '''Drop the geometry and images from loaded demonstrations, in place.

        Everything the search needs has been extracted by this point: placement centroids and
        the SRN table live in the tasks, the sketch has supplied the parameter, and B3-b's
        keyframes are already encoded PNG bytes. Holding the meshes through a multi-hour
        driver only feeds them to the fork.

        Returns how many demonstrations were stripped. They are reloaded afterwards, because
        scoring genuinely needs them: `_evaluate_training_metrics` runs the ground-truth
        program and the physics stability check against the real geometry.
        '''
        stripped = 0
        for demos in grouped.values():
            for demo in demos:
                touched = False
                for key in cls.HEAVY_KEYS:
                    if demo.get(key) is not None:
                        demo[key] = None
                        touched = True
                stripped += bool(touched)

        # Dropping the references is not enough on its own. CPython returns freed arenas to
        # its own allocator, not to the OS, so RSS stays at ~53 GB and every later `fork`
        # still maps that footprint -- measured: parent RSS 53.4 GB with a PSS of only
        # 3.3 GB, i.e. the pages were shared rather than copied (so COW was working) but the
        # machine was still accounting for them. Returning them explicitly is what actually
        # lets several runs coexist.
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as exc:  # noqa: BLE001 - glibc-only; never fatal
            log(f"malloc_trim unavailable ({type(exc).__name__}); "
                f"freed memory stays in the allocator.")
        return stripped

    def learn_all(self) -> None:
        cfg = self.configs
        concepts = list(cfg.concepts_to_learn or cfg.concepts_space)
        log("Attempting to learn: " + ", ".join(concepts))

        demos_by_concept = self._load_demos(concepts, summarise=True)

        pending = OrderedDict()
        for concept, demos in demos_by_concept.items():
            if not demos:
                log(f"No demonstrations found for <{concept}>; skipping.")
            elif (concept in self.spl.concept_library.inductive_concepts
                    or concept in self._metric_records):
                log(f"<{concept}> already learned; skipping.")
            else:
                pending[concept] = demos
        if not pending:
            log("Nothing left to learn.")
            return

        # The sketch supplies the parameter name and value the task needs. It is cached by
        # instruction, so `learn_concept` re-reading it below costs nothing.
        log(f"Sketching {sum(len(d) for d in pending.values())} instruction(s)")
        sketches = {c: [self.shared_sketch.signature(d["language_instruction"]) for d in demos]
                    for c, demos in pending.items()}

        evaluator, observation_mode = self.evaluator, self.observation_mode
        log(f"Observations: {observation_mode}; acceptance: {evaluator.describe()}")

        tasks, used_demos = build_tasks(
            pending, sketches,
            granularity=getattr(cfg, "task_granularity", "concept"),
            selection=getattr(cfg, "demo_selection", "distinct_params"),
            demos_per_concept=cfg.num_demos_per_concept,
            observation_mode=observation_mode,
            # SPL's own executor, so the search's noise model is the SRN the rest of the
            # system uses rather than a second one that could drift from it.
            executor=self.spl.executor if observation_mode == "continuous" else None)
        self._write(f"demo_selection.json", used_demos)
        granularity = getattr(cfg, "task_granularity", "concept")
        for concept, ids in used_demos.items():
            # Concept-level names a task after its concept; demo-level names it
            # "<concept>_<demo_id>" and produces one per demonstration. Matching on equality
            # alone raised StopIteration the first time demo-level ran end to end.
            owned = [t for t in tasks
                     if t.name == concept or t.name.startswith(f"{concept}_")]
            params = [n for t in owned for n, _ in t.examples]
            log(f"  <{concept}>: demos {ids} at parameters {params} "
                f"({len(owned)} task(s))")

        # Hooks first: B3-b's proposer reads the demonstrations here to build its delta table
        # and encode its keyframes, and both outputs are small (a string and PNG bytes).
        hooks = self._driver_hooks(pending, tasks)

        # Then let the geometry go, before the wake phase forks its workers. Everything the
        # search needs is already in `tasks`.
        released = self._release_demo_payload(pending)
        # Usually zero: `_load_demos(summarise=True)` already dropped the geometry as each
        # demonstration streamed past. This is the safety net for any path that loaded demos
        # in full, and it must run before the wake phase forks its workers.
        log(f"Geometry released before the search "
            f"({released} demonstration(s) still holding it at this point)." )

        t0 = time.perf_counter()
        result = driver.run(
            tasks,
            **hooks,
            evaluator=evaluator,
            level=getattr(cfg, "grammar_level", "standard"),
            continuation=getattr(cfg, "use_continuation_type", True),
            int_literals_upto=self._literal_ceiling(tasks, cfg),
            iterations=getattr(cfg, "search_iterations", 3),
            timeout=getattr(cfg, "enumeration_timeout", 60.0),
            max_mdl=getattr(cfg, "max_mdl", 100.0),
            use_library=getattr(cfg, "use_library", True),
            pseudo_counts=getattr(cfg, "pseudo_counts", 1.0),
            max_arity=getattr(cfg, "stitch_max_arity", 3),
            maximum_frontier=getattr(cfg, "maximum_frontier", 5),
            cpus=getattr(cfg, "cpus", 1),
            use_recognition=getattr(cfg, "use_recognition", True),
            recognition_epochs=getattr(cfg, "recognition_epochs", None),
            recognition_steps=getattr(cfg, "recognition_steps", 10000),
            recognition_timeout=getattr(cfg, "recognition_timeout", 1800.0),
            recognition_hidden=getattr(cfg, "recognition_hidden", 64),
            recognition_contextual=getattr(cfg, "recognition_contextual", True),
            recognition_bias_optimal=getattr(cfg, "recognition_bias_optimal", True),
            recognition_auxiliary_loss=getattr(cfg, "recognition_auxiliary_loss", True),
            helmholtz_ratio=getattr(cfg, "helmholtz_ratio", 0.5),
            log=log)
        search_seconds = time.perf_counter() - t0

        stats = result.stats()
        stats["search_seconds"] = round(search_seconds, 2)
        stats["demo_selection"] = used_demos
        stats["task_granularity"] = granularity
        self._write("search_stats.json", stats)
        self._write("library.json", {"abstractions": [str(a) for a in result.abstractions],
                                     "documentation": result.documentation})
        self._after_run(result)

        if granularity != "concept":
            # Demo-level solutions are CLOSED programs -- request type `tstate -> tstate`,
            # with the structure's size baked in as a literal. That is DreamCoder as
            # published, and it is the point of running this: their own benchmarks are shaped
            # this way ("arch leg 1" ... "arch leg 8" are eight separate tasks, none of them
            # parameterised). But a closed program cannot be lowered into an SPL concept
            # class, which takes the size as an argument, so there is nothing to register and
            # `program_accuracy` is not defined for it. Recovering a parameterised form would
            # mean anti-unifying each concept's demo-level solutions, which is a separate
            # piece of work and not what this run is measuring.
            #
            # So this run reports the SEARCH result only -- how many closed tasks the
            # enumerator solves, and how deep it reaches -- which is exactly the quantity that
            # answers "was the parameterised framing handicapping the search?". Compare
            # search_stats.json against the matching concept-level run.
            solved = len(result.solved)
            log(f"Demo-level run: {solved}/{len(tasks)} task(s) solved. Skipping concept "
                f"registration and program_accuracy -- a closed program has no parameter to "
                f"lower into an SPL class. See search_stats.json for the comparable numbers.")
            if cfg.concept_save_path:
                self.spl.save(cfg.concept_save_path)
            return

        # Register and score whatever was found. An unsolved concept is recorded, not skipped:
        # an empty column by construction is the result, and hiding it would misreport it.
        #
        # Scoring needs the geometry back -- `_evaluate_training_metrics` runs the ground-truth
        # program and the physics stability check against real meshes -- but reloading every
        # concept at once would undo the whole point of streaming. So reload in chunks and
        # release each chunk before the next, keeping the peak at `scoring_chunk` concepts
        # regardless of how many the dataset holds.
        # Reload when the demonstrations in hand have no geometry, NOT when this run happened
        # to release some. Keying off the release count was wrong the moment streaming started
        # doing the releasing: the counter came back 0, the reload was skipped, and scoring ran
        # on mesh-less demos -- `program_accuracy` still worked (it compares against
        # `run_gt_program`) while `plan_accuracy`, `mean_iou` and the stability checks all went
        # silently to None.
        needs_reload = any(demo.get("meshes") is None
                           for demos in pending.values() for demo in demos)
        chunk = max(1, int(getattr(cfg, "scoring_chunk", 8)))
        names = list(pending)
        for start in range(0, len(names), chunk):
            window = names[start:start + chunk]
            if needs_reload:
                log(f"Reloading demonstrations for scoring "
                    f"({start + 1}-{start + len(window)} of {len(names)}).")
                window_demos = self._load_demos(window)
            else:
                window_demos = OrderedDict((c, pending[c]) for c in window)
            self._score_window(window_demos, used_demos, result, tasks, cfg)
            window_demos = None
            self._reclaim()

        if cfg.concept_save_path:
            self.spl.save(cfg.concept_save_path)
        log(f"Done. {len(result.solved)} solved, {len(result.approximate)} approximate, "
            f"{len(result.unsolved)} unsolved. Metrics -> {self._records_path}")

    def _score_window(self, pending, used_demos, result, tasks, cfg) -> None:
        """Lower, register and score one chunk of concepts.

        Split out of `learn_all` so the reload can be chunked: each call sees only the
        concepts whose demonstrations are currently in memory.
        """
        evaluator = self.evaluator
        for concept, demos in pending.items():
            solution = result.solutions.get(concept)
            chosen = [d for d in demos if str(d.get("demo_id")) in set(used_demos[concept])]
            chosen = chosen or demos[:cfg.num_demos_per_concept]

            if solution is None or solution.term is None:
                if solution is None or solution.program is None:
                    status, reason = "search_unsolved", "no program found"
                else:
                    status = "search_untranslatable"
                    reason = "the program found could not be read back as a term"
                log(f"<{concept}>: {reason}; recording without a program.")
                self._metric_records[concept] = {
                    "concept": concept, "status": status,
                    "search_status": solution.status if solution else "unsolved",
                    "search_score": round(solution.distance, 4) if solution else None,
                    "evaluator": evaluator.name,
                    "program_accuracy": None, "program_verdict": "not_found"}
                self._flush()
                continue

            param = next(t for t in tasks if t.name == concept).param_name
            code = lower(solution.term, concept, param)
            exact, why = saved_is_exact(solution.term)
            if not exact:
                log(f"<{concept}>: WARNING focus restore is not exactly lowerable ({why}); "
                    f"the class may fail on the live executor.")

            self.agent.add(concept, GeneratedConcept(
                concept_name=concept,
                attributes={param: int, "objects": list},
                code=code,
                info={"mdl": -solution.log_prior, "status": solution.status,
                      "program": str(solution.program),
                      "live_execution": "ok" if exact else f"unsupported: {why}"}))

            log(f"Learning <{concept.upper()}> from the {solution.status} search result "
                f"(MDL {-solution.log_prior:.1f}).")
            try:
                self.learn_concept(chosen)
            except Exception as exc:  # noqa: BLE001
                log(f"Scoring <{concept}> raised: {exc}")
                self._metric_records[concept] = {"concept": concept,
                                                 "status": f"error: {exc}"}
                self._flush()
                continue

            # `learn_concept` reports the *scoring* outcome, which is "ok" even when the
            # program registered was the search's best miss rather than a solution. Record
            # which it was: a reviewer has to be able to tell a solved concept from one whose
            # approximate program simply scored zero.
            record = self._metric_records.get(concept)
            if record is not None:
                record["search_status"] = solution.status
                record["search_mdl"] = round(-solution.log_prior, 2)
                record["search_score"] = round(solution.distance, 4)
                record["evaluator"] = evaluator.name
                if evaluator.name == "exact":
                    record["trace_distance"] = round(solution.distance, 4)
                record["solved_by"] = result.solved_by.get(concept)
                if not solution.exact:
                    record["status"] = f"approximate ({record.get('status', 'ok')})"
                # The integrity check a probabilistic evaluator makes necessary. `search_status`
                # is what the run accepted; `program_accuracy` is proven equivalence to ground
                # truth. Under `exact` these cannot disagree; under a soft evaluator they can,
                # and that disagreement is the false-accept rate -- a number to report, not a
                # bug to hide.
                if solution.exact and record.get("program_accuracy") == 0.0:
                    record["false_accept"] = True
                    log(f"<{concept}>: WARNING accepted by the {evaluator.name} evaluator but "
                        f"NOT equivalent to ground truth (program_accuracy 0.0).")
                self._annotate_record(concept, record, result)
                self._flush()

    # ------------------------------------------------------------------ #

    def _driver_hooks(self, pending, tasks) -> Dict:
        '''Extra keyword arguments for `driver.run`.

        Empty for B3-a, which is the definition of the control: no LLM touches the search.
        `LiloHarness` overrides this to supply `propose` and `document`, so the two baselines
        share one loop and differ by exactly LILO's contributions.
        '''
        return {}

    def _after_run(self, result) -> None:
        '''Called once the driver returns, before scoring. B3-a writes nothing extra.'''

    def _annotate_record(self, concept: str, record: dict, result) -> None:
        '''Called after `learn_concept` has scored one concept. B3-a adds nothing.'''

    def _write(self, name: str, payload) -> None:
        path = os.path.join(self.configs.run_dir, name)
        os.makedirs(self.configs.run_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
