'''
compare_artifacts.py

Diff a run directory against a recorded golden, ignoring wall-clock fields.

Why this exists: on this codebase "the tests pass" has repeatedly failed to catch a behaviour
change. The `plan_accuracy -> None` regression exited 0 and passed all 79 tests; the recognition
model trained for 45 gradient steps instead of 10,000 for several runs without a single failure.
Both would have been caught in seconds by diffing a run's artifacts against a known-good one.

    python -m baseline_spl.tests.compare_artifacts <run_dir> <golden_dir>
    python -m baseline_spl.tests.compare_artifacts <run_dir> <golden_dir> --update

Timing fields are excluded because they vary by nature. Everything else -- the solved list,
every program string, every MDL, every metric -- must match exactly.

The companion config `configs/determinism_pin.py` makes the MDL band rather than the clock end
the search, so its `search_stats.json` is reproducible run to run; the smoke configs are
clock-bound: a faster machine enumerates further and picks a different near-miss for anything
it cannot solve, so `--clock-bound` compares outcomes (what was solved, how it scored) rather
than the path taken.
'''

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

# Vary by nature; never evidence of a behaviour change.
TIMING_KEYS = {"seconds", "total_seconds", "programs_per_second", "search_seconds",
               "seconds_by_stage", "learning_times"}

# When the CLOCK ends the search rather than the MDL band, the machine's speed decides how far
# enumeration gets, so a faster run finds a *different* best near-miss for any concept it cannot
# solve. Every number derived from that choice then moves without anything being wrong.
#
# So `--clock-bound` compares OUTCOMES, not the path taken: which concepts were solved, and how
# the registered program scored. An allowlist rather than a denylist, because the failure mode
# to avoid is a comparator that cries wolf until its warnings get ignored.
CLOCK_BOUND_COMPARED = {
    # search_stats.json
    "solved", "approximate", "unsolved", "status", "solved_by",
    "solved_by_llm", "solved_by_enumeration", "task_granularity", "evaluator",
    # training_metrics.json
    "concept", "pred_concept", "program_accuracy", "program_verdict", "search_status",
}

ARTIFACTS = ("search_stats.json", "training_metrics.json")


def _strip(value: Any, drop: set) -> Any:
    '''Drop timing fields, recursively.'''
    if isinstance(value, dict):
        return {k: _strip(v, drop) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [_strip(v, drop) for v in value]
    return value


def _outcomes_only(value: Any) -> Any:
    """Keep just the fields listed in CLOCK_BOUND_COMPARED, wherever they appear."""
    if isinstance(value, dict):
        kept = {}
        for k, v in value.items():
            if k in CLOCK_BOUND_COMPARED:
                kept[k] = v if not isinstance(v, (dict, list)) else _outcomes_only(v)
            elif isinstance(v, (dict, list)):
                inner = _outcomes_only(v)
                if inner not in ({}, []):
                    kept[k] = inner
        return kept
    if isinstance(value, list):
        return [_outcomes_only(v) for v in value]
    return value


def _differences(a: Any, b: Any, path: str = "") -> List[str]:
    '''Every leaf where the two disagree, named by its path.'''
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}.{key}: missing in run")
            elif key not in b:
                out.append(f"{path}.{key}: missing in golden")
            else:
                out.extend(_differences(a[key], b[key], f"{path}.{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in _differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def compare(run_dir: Path, golden_dir: Path, clock_bound: bool = False) -> List[str]:
    problems: List[str] = []
    shape = (lambda o: _outcomes_only(_strip(o, TIMING_KEYS))) if clock_bound \
        else (lambda o: _strip(o, TIMING_KEYS))
    for name in ARTIFACTS:
        run, golden = run_dir / name, golden_dir / name
        if not golden.exists():
            continue
        if not run.exists():
            problems.append(f"{name}: produced by the golden run but not by this one")
            continue
        problems.extend(f"{name}{d}" for d in _differences(shape(json.loads(run.read_text())),
                                                           shape(json.loads(golden.read_text()))))
    return problems


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    run_dir, golden_dir = Path(argv[1]), Path(argv[2])
    if "--update" in argv:
        golden_dir.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACTS:
            if (run_dir / name).exists():
                (golden_dir / name).write_text((run_dir / name).read_text())
        print(f"golden updated from {run_dir}")
        return 0

    problems = compare(run_dir, golden_dir, clock_bound="--clock-bound" in argv)
    if not problems:
        print(f"MATCH  {run_dir.name} == {golden_dir.name}")
        return 0
    print(f"DIFFERS  {run_dir.name} != {golden_dir.name}  ({len(problems)} difference(s))")
    for line in problems[:40]:
        print(f"   {line}")
    if len(problems) > 40:
        print(f"   ... and {len(problems) - 40} more")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
