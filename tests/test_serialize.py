'''
test_serialize.py

Two things that fail silently if they are wrong.

  1. The direction constants. `_DIRECTION_VECTORS` comes from the simulator's own
     ParameterSettings.DIRECTIONS, and the SRN's learned mean delta for each direction is
     checked against it. If the two disagree, the statistics we hand the model describe
     different directions than the DSL actually performs, and every baseline builds a
     rotated structure while still producing plausible numbers.
  2. The serializer's output: per-scenario object lists, no direction labels, and both
     coordinate modes rendering correctly in upstream's expected format.

Run: python -m baseline_spl.tests.test_serialize
'''

from __future__ import annotations

import sys

import numpy as np

from SPL.config.primitive_config import DEFAULT_ACTIONS
from SPL.dataloader.datasets import build_inductive_structure_dataset
from baseline_spl.common.serialize_text import (_DIRECTION_VECTORS, demo_to_scenario,
                                                to_demo2code_text)

DATA_DIR = "/home/nsei/Namas/DATA/structures"
ASSETS = "/home/nsei/Namas/DATA/assets"
CONCEPTS = ["row", "tower", "column"]


def check_direction_table(failures):
    '''Our vectors must BE the simulator's, and cover exactly the DSL's actions.'''
    from nsei_simulator.dataset.spg.configs import ParameterSettings

    if set(_DIRECTION_VECTORS) != {a.upper() for a in DEFAULT_ACTIONS}:
        failures.append(f"direction table {sorted(_DIRECTION_VECTORS)} != DSL actions "
                        f"{sorted(a.upper() for a in DEFAULT_ACTIONS)}")
        return
    for action, vector in _DIRECTION_VECTORS.items():
        expected = ParameterSettings.DIRECTIONS[action.lower()]
        if not np.allclose(vector, expected):
            failures.append(f"{action}: {vector} != simulator's {expected}")
    if not failures:
        print(f"  OK   direction vectors match ParameterSettings.DIRECTIONS "
              f"for all {len(_DIRECTION_VECTORS)} DSL actions")


def check_srn_agrees(failures, demo):
    '''The SRN's mean delta per direction must point the way the simulator says.'''
    from baseline_spl.common.config import BaselineConfig
    from baseline_spl.common.primitive_stats import direction_stats
    from baseline_spl.config import CapConfig
    from SPL.model.spl import SPL

    configs = BaselineConfig.from_run_config(CapConfig, "test_serialize")
    spl = SPL(configs)
    stats = direction_stats(spl.executor, [demo])
    if not stats:
        failures.append("direction_stats returned nothing")
        return

    for action, (mean, _std) in stats[0]["stats"].items():
        expected = np.asarray(_DIRECTION_VECTORS[action], dtype=float)
        axis = int(np.argmax(np.abs(expected)))
        if int(np.argmax(np.abs(mean))) != axis:
            failures.append(f"SRN {action}: dominant axis {int(np.argmax(np.abs(mean)))} "
                            f"!= expected {axis} (mean {np.round(mean, 3)})")
        elif np.sign(mean[axis]) != np.sign(expected[axis]):
            failures.append(f"SRN {action}: sign {np.sign(mean[axis])} on axis {axis} "
                            f"!= expected {np.sign(expected[axis])}")
    if not failures:
        print(f"  OK   SRN mean deltas agree with the simulator on all "
              f"{len(stats[0]['stats'])} directions")


def check_serialization(failures, demos):
    text = to_demo2code_text(demos, coordinate_mode="lattice")

    for marker in ('"""', "[Scenario 1]", "State 2:"):
        if marker not in text:
            failures.append(f"serialized text lacks upstream marker {marker!r}")

    # Per-scenario object lists: one per demo, and disjoint (different scenes).
    header = text.split('"""')[0]
    lines = [ln for ln in header.splitlines() if ln.startswith("scenario_")]
    if len(lines) != len(demos):
        failures.append(f"expected {len(demos)} scenario_N_objects lines, got {len(lines)}")
    elif len(demos) > 1:
        first, second = (set(eval(ln.split("=", 1)[1])) for ln in lines[:2])  # noqa: S307
        if first & second:
            failures.append(f"scenarios share object names: {sorted(first & second)}")
        else:
            print(f"  OK   {len(lines)} per-scenario object lists, no shared names")

    # No direction labels anywhere: naming the direction is the model's job now.
    leaked = [d for d in DEFAULT_ACTIONS if f" {d.upper()} of " in text]
    if leaked:
        failures.append(f"serializer still labels directions: {leaked}")
    else:
        print("  OK   no direction labels in the serialized demonstration")

    # Both coordinate modes render, and raw starts at the origin.
    for mode, label in (("lattice", "grid cell"), ("raw", "position")):
        rendered = demo_to_scenario(demos[0], 1, coordinate_mode=mode)
        if label not in rendered:
            failures.append(f"coordinate_mode={mode!r} did not render a {label!r}")
            continue
        first = rendered.split(f"is at {label} ")[1].splitlines()[0]
        origin = (0, 0, 0) if mode == "lattice" else (0.0, 0.0, 0.0)
        if eval(first) != origin:  # noqa: S307 — our own output
            failures.append(f"{mode}: first placement should be the origin, got {first}")
        else:
            print(f"  OK   coordinate_mode={mode!r} renders, first placement at origin")

    try:
        demo_to_scenario(demos[0], 1, coordinate_mode="bogus")
        failures.append("an invalid coordinate_mode was accepted")
    except ValueError:
        print("  OK   invalid coordinate_mode is rejected")

    print("\n--- serialized (raw mode) ---")
    print(to_demo2code_text(demos, coordinate_mode="raw"))


def main() -> int:
    failures = []
    check_direction_table(failures)

    dataset = build_inductive_structure_dataset(
        "spl_sim", DATA_DIR, ASSETS, "fixed_robot_diag_45", False, load_only=CONCEPTS)

    by_concept = {}
    for i in range(len(dataset)):
        demo = dataset[i]
        by_concept.setdefault(demo["concept"], []).append(demo)

    rows = by_concept.get("row", [])
    if len(rows) < 2:
        failures.append("need two 'row' demonstrations for the serializer checks")
    else:
        check_serialization(failures, rows[:2])
        check_srn_agrees(failures, rows[0])

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all serializer checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
