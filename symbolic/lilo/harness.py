'''
harness.py

B3-b = B3-a's harness plus two hooks. Everything else -- demo loading, sketching, task
construction, lowering, scoring, resume, `spl.save` -- is inherited unchanged, which is what
makes the two baselines comparable: they differ by LILO's contributions and by nothing else.

One trap worth naming, because it would have silently zeroed the cost table. `learn_concept`
resets and reads the token ledger through `self.agent.backend` (`common/harness.py:101-103`),
but B3-b spends its tokens inside `learn_all`'s driver loop, *before* any `learn_concept`
call. So `agent.backend` stays None -- nothing zeroes the ledger mid-run -- and the
proposer's per-task deltas are merged onto each record afterwards instead.
'''

from __future__ import annotations

from typing import Dict

from baseline_spl.common.harness import log
from baseline_spl.symbolic.harness import SearchHarness
from baseline_spl.symbolic.lilo.namer import LibraryNamer
from baseline_spl.symbolic.lilo.proposer import LLMProposer


class LiloHarness(SearchHarness):
    '''B3-b: LLM synthesis + STITCH + library auto-documentation.'''

    def __init__(self, configs, backend):
        super().__init__(configs)
        self.backend = backend
        cfg = configs
        self.proposer = LLMProposer(
            backend,
            queries_per_task=getattr(cfg, "llm_queries_per_task", 4),
            samples_per_query=getattr(cfg, "llm_samples_per_query", 4),
            max_tokens=getattr(cfg, "llm_max_tokens", 3000),
            temperature=getattr(cfg, "llm_temperature", 0.7),
            allow_named_variables=getattr(cfg, "allow_named_variables", False),
            # Gate 7 must agree with what the driver will accept, or the rejection histogram
            # would report candidates as wrong-trace that the run then accepts.
            evaluator=self.evaluator,
            demo_modality=getattr(cfg, "demo_modality", "coords"),
            log=log)
        self.namer = (LibraryNamer(backend, log=log)
                      if getattr(cfg, "auto_document", True) else None)

    # ------------------------------------------------------------------ #

    def _driver_hooks(self, pending, tasks) -> Dict:
        # The proposer is built in __init__, before any demonstration is loaded, so the
        # demonstration-derived halves of the prompt are attached here instead.
        self._attach_demonstration_channels(pending, tasks)
        hooks = {"propose": self.proposer}
        if self.namer is not None:
            hooks["document"] = self.namer
        return hooks

    def _attach_demonstration_channels(self, pending, tasks) -> None:
        '''Give the proposer whatever its modality needs from the demonstrations.

        Two channels, matching what the LLM baselines get:

          the shift_focus delta table  -- `include_primitive_stats` in CaP / Demo2Code. Only
              meaningful alongside coordinates: with a lattice the directions are already
              implicit in the integers. It is also the same mu/sigma the Gaussian evaluator
              scores with, so the model is told the noise model it is being judged under.
          keyframe images -- `demo_modality='images'` in CaP, `variant='vlm'` in Demo2Code.

        Failures here are logged and non-fatal: a missing SRN checkpoint or a demo loaded
        without images should cost the run a prompt section, not the run.

        Both channels need real geometry `pending`'s own demos don't carry -- `_load_demos
        (summarise=True)` stripped every HEAVY_KEY before this hook ever runs, the same strip
        `_register_and_score` (harness.py) reloads around for scoring. At iteration 0 `pending`
        is most or all of the concept set, and loading that many concepts' geometry in one shot
        reproduces the incident documented at `SearchHarness.HEAVY_KEYS`: ~53 GB for 16
        concepts, three concurrent runs at 231 GB of 251 GB, one OOM-killed with no traceback.
        So this reloads in `scoring_chunk`-sized windows, releasing each before the next --
        exactly the pattern `_register_and_score` already uses, not a one-shot load.
        '''
        cfg = self.configs
        modality = getattr(cfg, "demo_modality", "coords")
        stats_wanted = modality in ("coords", "both") and self.observation_mode == "continuous"
        stats_enabled = stats_wanted and getattr(cfg, "include_primitive_stats", True)
        needs_images = modality in ("images", "both")
        if not (stats_wanted or needs_images):
            return

        names = {t.name for t in tasks}
        concepts_needed = [c for c in pending if c in names]
        per_demo_stats: list = []
        attached = 0

        if (stats_enabled or needs_images) and concepts_needed:
            from baseline_spl.common.primitive_stats import direction_stats
            from baseline_spl.common.serialize_visual import demo_frames

            chunk = max(1, int(getattr(cfg, "scoring_chunk", 8)))
            for start in range(0, len(concepts_needed), chunk):
                window = concepts_needed[start:start + chunk]
                window_demos = self._load_demos(window, summarise=False)

                if stats_enabled:
                    demos = [d for group in window_demos.values() for d in group]
                    try:
                        per_demo_stats.extend(direction_stats(self.spl.executor, demos))
                    except Exception as exc:  # noqa: BLE001
                        log(f"Proposer: shift_focus stats failed on {window} "
                            f"({type(exc).__name__}: {exc}); skipping this window.")

                if needs_images:
                    for concept in window:
                        demos = window_demos.get(concept) or []
                        try:
                            frames = []
                            for demo in demos[:cfg.num_demos_per_concept]:
                                frames.extend(demo_frames(
                                    demo, max_px=getattr(cfg, "vlm_max_image_px", 512),
                                    max_keyframes=getattr(cfg, "vlm_max_keyframes", 40)))
                            self.proposer.demo_frames[concept] = frames
                            attached += 1
                        except Exception as exc:  # noqa: BLE001
                            log(f"Proposer: no keyframes for <{concept}>: "
                                f"{type(exc).__name__}: {exc}")

                window_demos = None    # let this window's meshes/images go before the next
                self._reclaim()

        if stats_wanted:
            from baseline_spl.common.primitive_stats import render_stats_block

            block = render_stats_block(per_demo_stats) if stats_enabled else ""
            self.proposer.stats_block = block
            log(f"Proposer: {'attached' if block else 'no'} shift_focus delta table")

        if needs_images:
            log(f"Proposer: attached keyframes for {attached} concept(s)")

    def _after_run(self, result) -> None:
        stats = self.proposer.stats()
        if self.namer is not None:
            stats["naming"] = {"calls": self.namer.calls, "failures": self.namer.failures,
                               "documentation": self.namer.documentation}
        stats["solved_by_llm"] = sorted(n for n, s in result.solved_by.items() if s == "llm")
        stats["solved_by_enumeration"] = sorted(n for n, s in result.solved_by.items()
                                                if s == "enumeration")
        self._write("lilo_stats.json", stats)

        accepted = stats["accepted"]
        examined = stats["candidates_examined"]
        log(f"Proposer: {accepted}/{examined} candidates accepted "
            f"({stats['acceptance_rate']:.0%}) over {stats['llm_calls']} call(s); "
            f"{len(stats['solved_by_llm'])} concept(s) solved by the LLM, "
            f"{len(stats['solved_by_enumeration'])} by enumeration")
        for gate, count in stats["rejected_by_gate"].items():
            log(f"  rejected at {gate}: {count}")

    def _annotate_record(self, concept: str, record: dict, result) -> None:
        '''Attribute the LLM spend to the concept it was spent on.

        Without this the cost table would show B3-b as free, because the harness's own
        per-concept ledger snapshot happens after the spending is over.
        '''
        ledger = self.proposer.ledger_by_task.get(concept)
        if ledger:
            for key, value in ledger.items():
                record[key] = record.get(key, 0) + value
