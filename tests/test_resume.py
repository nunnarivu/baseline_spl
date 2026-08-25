'''
test_resume.py

Checks the resume logic that decides which concept library gets loaded, and that
resuming appends to the metric files instead of truncating them.

Run: python -m baseline_spl.tests.test_resume
'''

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

from baseline_spl.common.config import BaselineConfig, RUNS_ROOT
from baseline_spl.common.harness import BaselineHarness
from baseline_spl.config import CapConfig


def cfg_for(tmp_name, **overrides):
    return BaselineConfig.from_run_config(
        type("Tmp", (CapConfig,), overrides), tmp_name)


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
        # No checkpoint on disk yet.
        shutil.rmtree(run_dir, ignore_errors=True)
        cfg = cfg_for(run_name, resume=True)
        check("resume=True, no checkpoint -> None", cfg.load_concept_checkpoint, None)

        # Checkpoint present.
        open(cfg.concept_save_path, "w").close()
        cfg = cfg_for(run_name, resume=True)
        check("resume=True, checkpoint exists -> own path",
              cfg.load_concept_checkpoint, cfg.concept_save_path)

        cfg = cfg_for(run_name, resume=False)
        check("resume=False -> None even though checkpoint exists",
              cfg.load_concept_checkpoint, None)

        # Explicit path elsewhere.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            elsewhere = f.name
        try:
            cfg = cfg_for(run_name, resume=True, resume_from=elsewhere)
            check("resume_from honoured", cfg.load_concept_checkpoint, elsewhere)
        finally:
            os.unlink(elsewhere)

        # A path that does not exist must not be silently accepted.
        cfg = cfg_for(run_name, resume=True, resume_from="/nonexistent/library.pt")
        check("resume_from missing file -> None", cfg.load_concept_checkpoint, None)

        # SPL's own library must be refused.
        from SPL.config.spl_config import SPLConfig
        spl_ckpt = SPLConfig.load_concept_checkpoint
        if spl_ckpt and os.path.exists(spl_ckpt):
            try:
                cfg_for(run_name, resume=True, resume_from=spl_ckpt)
                failures.append("resume_from accepted SPL's own concept library")
            except ValueError:
                print("  OK   resume_from refuses SPL's own library")
        else:
            print("  SKIP SPL checkpoint not on disk, cannot test refusal")

        # Metric files must be merged, not truncated, when resuming.
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
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all resume checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
