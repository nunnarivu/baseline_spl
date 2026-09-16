'''
test_resume.py

Checks the checkpoint interface the baselines share with SPL: which concept library gets
loaded, that neither side of the load/save pair can touch SPL's own tree, and that resuming
appends to the metric files instead of truncating them.

Run: python -m baseline_spl.tests.test_resume
'''

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types

from baseline_spl.common.config import BaselineConfig, RUNS_ROOT
from baseline_spl.common.harness import BaselineHarness
from baseline_spl.config import CapConfig


def cfg_for(tmp_name, **overrides):
    return BaselineConfig.from_run_config(
        type("Tmp", (CapConfig,), overrides), tmp_name)


def pruned(records, learnt, **kwargs):
    '''_prune_to_library against a stub library, so no SPL instance has to be built.'''
    stub = types.SimpleNamespace(
        spl=types.SimpleNamespace(
            concept_library=types.SimpleNamespace(inductive_concepts=learnt)))
    return BaselineHarness._prune_to_library(stub, records, **kwargs)


def main() -> int:
    failures = []
    run_name = "test_resume"
    run_dir = RUNS_ROOT / run_name

    def check(label, got, want):
        if got == want:
            print(f"  OK   {label}")
        else:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    try:
        # Nothing named: an empty library, never SPL's.
        shutil.rmtree(run_dir, ignore_errors=True)
        cfg = cfg_for(run_name)
        check("no checkpoint named -> None", cfg.load_concept_checkpoint, None)
        check("save path stays in this run",
              cfg.concept_save_path, str(run_dir / "concept_library.pt"))

        # An explicit path elsewhere is honoured as-is; nothing is derived from the save path.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            elsewhere = f.name
        try:
            cfg = cfg_for(run_name, load_concept_checkpoint=elsewhere)
            check("explicit checkpoint honoured", cfg.load_concept_checkpoint, elsewhere)
        finally:
            os.unlink(elsewhere)

        # Its own library is not loaded unless it is named, even when it is on disk.
        open(cfg.concept_save_path, "w").close()
        check("own library on disk is still not loaded",
              cfg_for(run_name).load_concept_checkpoint, None)

        # A path that does not exist is a mistake, not an empty library.
        try:
            cfg_for(run_name, load_concept_checkpoint="/nonexistent/library.pt")
            failures.append("a missing load_concept_checkpoint was accepted")
        except FileNotFoundError:
            print("  OK   missing load_concept_checkpoint raises")

        # SPL's own library must be refused on both sides of the pair.
        from SPL.config.spl_config import SPLConfig
        spl_ckpt = SPLConfig.load_concept_checkpoint
        if spl_ckpt and os.path.exists(spl_ckpt):
            try:
                cfg_for(run_name, load_concept_checkpoint=spl_ckpt)
                failures.append("load_concept_checkpoint accepted SPL's own concept library")
            except ValueError:
                print("  OK   loading SPL's own library is refused")
        else:
            print("  SKIP SPL checkpoint not on disk, cannot test the load refusal")
        try:
            cfg_for(run_name, concept_save_path=SPLConfig.concept_save_path)
            failures.append("concept_save_path accepted SPL's own concept library")
        except ValueError:
            print("  OK   saving over SPL's own library is refused")

        # The knobs SPL reads at load time must reach the config object.
        cfg = cfg_for(run_name, skip_loading_concepts=("row",), ignore_learnt_concepts=False)
        check("skip_loading_concepts reaches the config", tuple(cfg.skip_loading_concepts), ("row",))
        check("ignore_learnt_concepts reaches the config", cfg.ignore_learnt_concepts, False)
        check("ignore_learnt_concepts defaults on", cfg_for(run_name).ignore_learnt_concepts, True)

        # Metric files must be merged, not truncated.
        os.makedirs(run_dir, exist_ok=True)
        records_path = run_dir / "training_metrics.json"
        with open(records_path, "w") as f:
            json.dump({"row": {"concept": "row", "status": "ok"}}, f)

        merged = BaselineHarness._load_records(str(records_path))
        check("previous metrics load for merging", list(merged), ["row"])

        merged["tower"] = {"concept": "tower", "status": "ok"}
        check("merged record keeps both concepts", sorted(merged), ["row", "tower"])

        missing = BaselineHarness._load_records(str(run_dir / "nope.json"))
        check("absent metric file -> empty dict", missing, {})

        # Pruning: records describe the library that was actually loaded.
        records = {"row": {"concept": "row", "order": 0},
                   "tower": {"concept": "tower", "order": 1},
                   "psi": {"concept": "psi", "order": 2, "pred_concept": "psi_shape"}}
        check("record of a concept left out by skip_loading_concepts is dropped",
              sorted(pruned(records, ["tower"])), ["tower"])
        check("a record is kept through the name the model registered",
              sorted(pruned(records, ["psi_shape"])), ["psi"])
        check("an empty library prunes everything", pruned(records, []), {})
        check("order is re-indexed densely",
              [r["order"] for r in pruned(records, ["tower", "psi_shape"], reindex=True).values()],
              [0, 1])
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all checkpoint checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
