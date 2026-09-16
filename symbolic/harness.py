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
from baseline_spl.symbolic.search import Solution
from baseline_spl.symbolic.settings import SearchSettings
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
            camera_view=cfg.camera_view, batch_size=1,
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
                    # Sketch now: filter() reads the demo's scene, which is released just below.
                    # Guarded for the same reason `summarise_demo` is: one concept the sketch
                    # cannot handle must cost that concept, not the whole run. The dataset keeps
                    # growing, so "a new concept broke loading" has to degrade to a skip.
                    try:
                        data["sketch_info"] = self.shared_sketch.signature(
                            data["language_instruction"], data)
                    except Exception as exc:  # noqa: BLE001
                        log(f"  cannot sketch demo {data.get('demo_id')} of "
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
        # Every integer of every argument tuple, not just the first: `room(4, 3, 2)` needs a
        # ceiling covering 4, and a concept whose second argument is the large one would
        # otherwise get a grammar too small to express it.
        from baseline_spl.symbolic.ir import args as _args

        largest = max((v for task in tasks for n in task.parameters for v in _args(n)),
                      default=6)
        return max(12, 2 * int(largest) + 1)

    def _register_and_score(self, pending, used_demos, result, tasks, cfg) -> None:
        """Register each concept's program and score it, reloading geometry in chunks.

        Scoring needs the meshes back -- `_evaluate_training_metrics` runs the ground-truth
        program and the physics stability check against real geometry -- but reloading every
        concept at once would undo the whole point of streaming. So reload a window, score it,
        release it, and move on: the peak stays at `scoring_chunk` concepts however many the
        dataset holds.

        Reload when the demonstrations in hand have NO GEOMETRY, not when this run happened to
        release some. Keying off the release count was wrong the moment streaming started doing
        the releasing: the counter came back 0, the reload was skipped, and scoring ran on
        mesh-less demos -- `program_accuracy` still worked (it compares against
        `run_gt_program`) while `plan_accuracy`, `mean_iou` and the stability checks all went
        silently to None.
        """
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

    def generalise_all(self) -> None:
        """Measure generalisation to unseen sizes over a FINISHED demo-level run.

        Post-processing, so `learn = False`: the solved closed programs are read back from
        `search_stats.json` rather than re-searched. Two arms over one held-out set, each
        behind its own config knob:

          arm A  `heldout_enumeration`      DreamCoder's own route -- enumerate each held-out
                                            (concept, size) task afresh under the learnt
                                            library, exactly as `dreamcoder.py:567` does.
          arm B  `recover_concept_classes`  anti-unify the solved demos into a parameterised
                                            class, then instantiate it at the held-out size.

        `infer_all` cannot serve here: it raises when no concept is registered, which is
        precisely the state a demo-level run leaves behind.
        """
        from dreamcoder.program import Program

        from baseline_spl.symbolic import heldout
        from baseline_spl.symbolic.bridge import grammar, to_term

        cfg = self.configs
        stats_path = os.path.join(cfg.run_dir, "search_stats.json")
        if not os.path.exists(stats_path):
            log(f"No search_stats.json in {cfg.run_dir}; nothing to generalise from.")
            return
        stats = json.loads(open(stats_path).read())
        if stats.get("task_granularity") == "concept":
            log("This run is concept-level; its programs already take the size as an "
                "argument, so there is nothing to recover.")
            return

        concepts = list(cfg.concepts_to_learn or cfg.concepts_space)
        sizes = tuple(getattr(cfg, "heldout_sizes", (8, 10, 12)))

        # The tasks carry the parameters and SRN tables recovery needs, and the artifacts do
        # not record them; rebuilding is cheaper than a second bespoke path.
        demos_by_concept = self._load_demos(concepts, summarise=True)
        pending = OrderedDict((c, d) for c, d in demos_by_concept.items() if d)
        # `.get`: a demo whose sketch failed during loading carries none, and one such
        # demo must not KeyError the run.
        sketches = {c: [d.get("sketch_info") for d in demos]
                    for c, demos in pending.items()}
        tasks, used_demos, _skipped = build_tasks(
            pending, sketches, granularity="demo",
            selection=getattr(cfg, "demo_selection", "distinct_params"),
            demos_per_concept=cfg.num_demos_per_concept,
            observation_mode=self.observation_mode,
            executor=self.spl.executor if self.observation_mode == "continuous" else None)

        # Build the grammar FIRST: `extra_int_literals` registers 3..ceiling as primitives,
        # and without them `Program.parse` cannot read back a closed program like `(loop 6 ...)`.
        # Doing this after the parse loop silently lost all 18 solved programs.
        g = grammar(getattr(cfg, "grammar_level", "standard"),
                    getattr(cfg, "use_continuation_type", True),
                    int_literals_upto=self._literal_ceiling(tasks, cfg))

        # Rebuild the solved solutions from the artifact, with their terms.
        result = driver.RunResult()
        for name, entry in stats.get("per_task", {}).items():
            if entry.get("status") != "solved" or not entry.get("program"):
                continue
            solution = Solution(task=name)
            try:
                program = Program.parse(entry["program"])
                solution.term = to_term(program, concept_level=False)
            except Exception as exc:  # noqa: BLE001
                log(f"  {name}: solved program will not translate ({exc}); skipping")
                continue
            # The frontier is what makes `Solution.status` say "solved" -- it is derived from
            # `exact`, which is `bool(self.frontier)`. Setting only `term` and `distance` left
            # every rebuilt solution reporting "unsolved", so recovery saw nothing to work with
            # and reported 0/0 while 18 programs sat right there.
            solution.frontier = [(-float(entry.get("mdl") or 0.0), program)]
            solution.distance = 0.0
            result.solutions[name] = solution
        log(f"Read {len(result.solutions)} solved closed program(s) from search_stats.json")

        recovered = self._recover_concept_classes(result, tasks, used_demos, cfg, log) \
            if getattr(cfg, "recover_concept_classes", True) else {}

        # Register and score the recovered classes, so a demo-level run gets the metrics that
        # were structurally undefined for it -- `program_accuracy` above all, which proves the
        # class equivalent to the ground-truth program rather than merely fitting its demos.
        if recovered:
            for concept, term in recovered.items():
                demo_solutions = [s for name, s in result.solutions.items()
                                  if name.rsplit("_", 1)[0] == concept and s.status == "solved"]
                best = max(demo_solutions, key=lambda s: s.log_prior, default=None)
                entry = Solution(task=concept)
                if best is not None:
                    entry.frontier = list(best.frontier)
                entry.term, entry.distance = term, 0.0
                result.solutions[concept] = entry
            scored = OrderedDict((c, pending[c]) for c in recovered if c in pending)
            self._register_and_score(
                scored, used_demos, result, self._concept_level_tasks(tasks, recovered,
                                                                      used_demos), cfg)
            if cfg.concept_save_path:
                self.spl.save(cfg.concept_save_path)

        # No evaluator is passed: the held-out tasks carry oracle-generated integer cells, so
        # `heldout` scores them exactly. Handing over this run's continuous evaluator scored
        # every one as None -- it needs SRN tables, which oracle targets do not have -- and
        # arm A reported 0/48 with nothing visibly wrong.
        results = heldout.run(
            g, concepts, sizes,
            recovered=recovered if recovered else None,
            search=bool(getattr(cfg, "heldout_enumeration", True)),
            timeout=float(getattr(cfg, "heldout_timeout", 300.0)),
            cpus=getattr(cfg, "cpus", 1), log=log)

        summary = heldout.summarise(results)
        summary["recovered_concepts"] = sorted(recovered)
        summary["heldout_sizes"] = list(sizes)
        # Arm B uses the concept grouping and size labels that DreamCoder's task formulation
        # never receives, so the artifact says so rather than leaving it to be inferred.
        summary["disclosure"] = (
            "class_solved is arm B: our anti-unification post-processing, which consumes the "
            "concept grouping and demo size labels. It is NOT published DreamCoder. "
            "search_solved is arm A, DreamCoder's own route (enumerate the held-out task).")
        self._write("heldout_metrics.json", summary)
        log(f"Held-out generalisation -> {os.path.join(cfg.run_dir, 'heldout_metrics.json')}")

    def _recover_concept_classes(self, result, tasks, used_demos, cfg, log) -> "OrderedDict":
        """Anti-unify each concept's SOLVED demo programs into a parameterised class.

        Only solved demos are used. An `approximate` program is a near miss, and generalising
        two near misses yields a confident-looking class that is simply wrong -- measured on
        `staircase`, whose two approximate solutions differ in the direction as well as the
        size, inventing a spurious direction argument.

        The gate is this run's own evaluator, so a recovered class must reproduce every demo
        under the same acceptance test the search used. `generalise.reproduces` would compare
        lattice cells only, which would be the wrong question in a continuous run.
        """
        from baseline_spl.symbolic import generalise
        from baseline_spl.symbolic.bridge import from_term

        by_concept = OrderedDict()
        for concept, ids in used_demos.items():
            owned = [t for t in tasks if str(t.demo_ids and t.demo_ids[0]) in set(map(str, ids))
                     and t.name.rsplit("_", 1)[0] == concept]
            solved = [(t, result.solutions.get(t.name)) for t in owned]
            solved = [(t, s) for t, s in solved if s is not None and s.status == "solved"
                      and s.term is not None]
            if not solved:
                continue
            by_concept[concept] = solved

        recovered = OrderedDict()
        for concept, solved in by_concept.items():
            terms = [s.term for _t, s in solved]
            params = [n for t, _s in solved for n, _ in t.examples]
            probe = self._concept_task(concept, [t for t, _s in solved])

            def gate(candidate, _probe=probe):
                try:
                    return self.evaluator.accepts(
                        self.evaluator.score(
                            from_term(candidate, concept_level=_probe.arity), _probe))
                except Exception:  # noqa: BLE001 - a wrong generalisation may fail any way
                    return False

            outcome = generalise.recover(
                concept, terms, params, accepts=gate,
                allow_single_demo=getattr(cfg, "recover_allow_single_demo", True),
                holdout_validate=getattr(cfg, "recover_holdout_validate", True))
            if outcome.ok:
                recovered[concept] = outcome.term
                log(f"  <{concept}>: recovered a parameterised class from "
                    f"{len(terms)} solved demo(s) at {params} [{outcome.route}]")
            else:
                log(f"  <{concept}>: no class recovered -- "
                    f"{generalise.REASONS.get(outcome.reason, outcome.reason)}")

        log(f"  recovered {len(recovered)}/{len(by_concept)} concept(s) with solved demos")
        return recovered

    @staticmethod
    def _concept_task(concept: str, owned):
        """A concept-level task built from the demo-level ones, for validation and lowering.

        Each demo task holds exactly one example; concatenating them is precisely the
        concept-level task the parameterised search would have been given. `srn_tables` must
        come along or the continuous evaluators score every example as None.
        """
        from baseline_spl.symbolic.search import SearchTask

        first = owned[0]
        return SearchTask(
            name=concept,
            examples=[t.examples[0] for t in owned],
            param_name=first.param_name,
            demo_ids=tuple(d for t in owned for d in t.demo_ids),
            instruction=first.instruction,
            observation_mode=first.observation_mode,
            srn_tables=[t.srn_tables[0] for t in owned if t.srn_tables],
            closed=False)

    def _concept_level_tasks(self, tasks, recovered, used_demos):
        """Concept-level tasks for the recovered concepts; `_score_window` looks up
        `param_name` by `t.name == concept`."""
        out = []
        for concept in recovered:
            owned = [t for t in tasks
                     if t.demo_ids and str(t.demo_ids[0]) in set(map(str, used_demos[concept]))
                     and t.name.rsplit("_", 1)[0] == concept]
            if owned:
                out.append(self._concept_task(concept, owned))
        return out

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

        cls._reclaim()          # dropping references is not enough -- see _reclaim
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
            elif cfg.ignore_learnt_concepts and concept in self.spl.concept_library.inductive_concepts:
                log(f"<{concept}> already learned; skipping.")
            else:
                pending[concept] = demos
        if not pending:
            log("Nothing left to learn.")
            return

        # The sketch supplies the parameter name and value the task needs. `_load_demos` took it
        # while the scene was still loaded; it is cached by instruction, so `learn_concept`
        # re-reading it below costs no LLM call.
        # `.get`: a demo whose sketch failed during loading carries none, and one such
        # demo must not KeyError the run.
        sketches = {c: [d.get("sketch_info") for d in demos]
                    for c, demos in pending.items()}

        evaluator, observation_mode = self.evaluator, self.observation_mode
        log(f"Observations: {observation_mode}; acceptance: {evaluator.describe()}")

        tasks, used_demos, skipped = build_tasks(
            pending, sketches,
            granularity=getattr(cfg, "task_granularity", "concept"),
            selection=getattr(cfg, "demo_selection", "distinct_params"),
            demos_per_concept=cfg.num_demos_per_concept,
            observation_mode=observation_mode,
            # SPL's own executor, so the search's noise model is the SRN the rest of the
            # system uses rather than a second one that could drift from it.
            executor=self.spl.executor if observation_mode == "continuous" else None)
        self._write(f"demo_selection.json", used_demos)

        # A concept the grammar cannot type gets a record, not silence. Without one it would
        # simply be absent from the results, indistinguishable from a concept that was searched
        # and failed -- which misreports the baseline's coverage.
        for concept, reason in skipped.items():
            # Two different things end up here, and conflating them would hide the second.
            # "not an integer" is the EXPECTED case -- an argument the grammar has no type for,
            # which is a property of the concept. Anything else is a failure worth reading, so
            # it keeps the exception type in the record rather than being filed as by-design.
            expected = "not an integer" in reason
            log(f"<{concept}>: {'inexpressible' if expected else 'could not be prepared'} "
                f"for this baseline -- {reason}")
            self._metric_records[concept] = {
                "concept": concept,
                "status": "inexpressible" if expected else "task_build_failed",
                "reason": reason,
                "search_status": "not_attempted", "program_accuracy": None,
                "program_verdict": "no_program", "evaluator": evaluator.name}
            pending.pop(concept, None)
        if skipped:
            self._flush()
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
        settings = SearchSettings.from_config(cfg)
        # Derived from the tasks, not the config, so a larger dataset widens the grammar on
        # its own; `int_literals_upto` in the config overrides for an ablation.
        settings.int_literals_upto = self._literal_ceiling(tasks, cfg)

        result = driver.run(tasks, settings, evaluator=evaluator, log=log, **hooks)
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
            # published: their benchmarks are shaped this way ("arch leg 1" ... "arch leg 8"
            # are eight separate tasks, none parameterised).
            #
            # A closed program cannot be lowered into an SPL class on its own, because the
            # class takes the size as an argument and the program has none. But a concept's
            # SOLVED demos differ only in that size, so anti-unifying them recovers the
            # parameterised form (`generalise.py`). That is strictly more than published
            # DreamCoder does -- it uses the concept grouping and the size labels, which
            # DreamCoder's task formulation never receives -- so it is reported as a
            # disclosed variant and gated behind `recover_concept_classes`.
            solved = len(result.solved)
            log(f"Demo-level run: {solved}/{len(tasks)} task(s) solved.")
            recovered = self._recover_concept_classes(result, tasks, used_demos, cfg, log) \
                if getattr(cfg, "recover_concept_classes", True) else {}
            if not recovered:
                log("  no concept class recovered; reporting the search result only. "
                    "See search_stats.json for the comparable numbers.")
                if cfg.concept_save_path:
                    self.spl.save(cfg.concept_save_path)
                return
            # Fall through: the recovered classes are parameterised, so the concept-level
            # registration and scoring path below applies to them unchanged.
            # `result.solutions` is keyed by TASK name (`row_0000`), so a concept-level entry
            # has to be synthesised for `_score_window` to find. It carries the recovered term
            # and the best of the concept's demo frontiers, so the reported MDL is a real
            # search result rather than an invented one.
            for concept, term in recovered.items():
                demo_solutions = [s for name, s in result.solutions.items()
                                  if name.rsplit("_", 1)[0] == concept and s.status == "solved"]
                best = max(demo_solutions, key=lambda s: s.log_prior, default=None)
                entry = Solution(task=concept)
                if best is not None:
                    entry.frontier = list(best.frontier)
                    entry.seconds = best.seconds
                    entry.programs_tried = best.programs_tried
                entry.term = term
                entry.distance = 0.0
                result.solutions[concept] = entry
            pending = OrderedDict((c, pending[c]) for c in recovered if c in pending)
            tasks = self._concept_level_tasks(tasks, recovered, used_demos)

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
        self._register_and_score(pending, used_demos, result, tasks, cfg)

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

            from baseline_spl.symbolic.ir import args as _args

            param = next(t for t in tasks if t.name == concept).param_name
            code = lower(solution.term, concept, param)
            exact, why = saved_is_exact(solution.term, arity=len(_args(param)))
            if not exact:
                log(f"<{concept}>: WARNING focus restore is not exactly lowerable ({why}); "
                    f"the class may fail on the live executor.")

            # One entry per integer argument. A 2- or 3-argument concept whose attributes named
            # only the first would be instantiated with the wrong signature by
            # `check_program_equivalence`, which binds by position.
            attributes = {str(n): int for n in _args(param)}
            attributes["objects"] = list

            self.agent.add(concept, GeneratedConcept(
                concept_name=concept,
                attributes=attributes,
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
