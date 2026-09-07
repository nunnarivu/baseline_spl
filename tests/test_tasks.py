'''
test_tasks.py

Does the keyframe -> lattice-cell conversion recover what the demonstration actually did?

The search never sees `demo.json:program`; its target comes from the keyframes via
`tasks.demo_cells`. This test uses the ground-truth program purely as an *oracle* to check
that conversion: for every demo, the cells read off the meshes must equal the cells
`run_gt_program` produces from the program that generated them.

If this drifts, the search would be chasing a target that does not match what the demo shows,
and every downstream number would be quietly wrong.

Also checks the demo-selection rule, which is an input asymmetry against SPL and therefore
has to behave exactly as documented.

Run: python -m baseline_spl.tests.test_tasks
'''

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from SPL.config.spl_config import ALL_CONCEPTS
from SPL.utils.metrics import run_gt_program
from baseline_spl.symbolic.tasks import (_parameter, demo_cells, global_pitch,
                                         select_demos)

DATA_DIR = "DATA/structures"


def load_demo_metadata():
    '''concept -> [(demo_id, parameter, gt_program, num_objects)], in dataloader order.'''
    grouped = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*", "demo.json"))):
        j = json.load(open(path))
        concept = j.get("concept")
        if concept not in ALL_CONCEPTS:
            continue
        grouped[concept].append((
            os.path.basename(os.path.dirname(path)),
            j["initializations"]["__param_0__"],
            j["program"],
            len(j["scene_info"]["object_ids"]),
        ))
    return grouped


def load_demos(concepts):
    from SPL.config.spl_config import SPLConfig
    from SPL.dataloader.datasets import build_inductive_structure_dataloader
    loader = build_inductive_structure_dataloader(
        SPLConfig.dataset_name, SPLConfig.train_dataset_dir, SPLConfig.assets_dir,
        camera_view="fixed_robot_diag_45", batch_size=1, num_workers=0, shuffle=False,
        load_images=False, load_only=list(concepts))
    grouped = defaultdict(list)
    for batch in loader:
        for demo in batch:
            grouped[demo["concept"]].append(demo)
    return grouped


def check_conversion(concepts):
    '''Returns {label: reason} for demos whose keyframe cells disagree with ground truth.'''
    metadata = load_demo_metadata()
    demos = load_demos(concepts)
    # One pitch for the whole run, exactly as build_tasks does. Per-concept estimation is
    # wrong for pins -- see tasks.global_pitch.
    pitch = global_pitch([d for group in demos.values() for d in group])
    problems = {}
    checked = 0
    for concept in concepts:
        by_id = {d: (p, prog, n) for d, p, prog, n in metadata.get(concept, [])}
        for demo in demos.get(concept, []):
            demo_id = str(demo.get("demo_id"))
            if demo_id not in by_id:
                continue
            _param, program, num_objects = by_id[demo_id]
            gt = run_gt_program(program, num_objects)
            if gt is None:
                problems[f"{concept}/{demo_id}"] = "ground truth failed to run"
                continue
            try:
                cells = demo_cells(demo, pitch)
            except Exception as exc:  # noqa: BLE001
                problems[f"{concept}/{demo_id}"] = f"demo_cells raised: {exc}"
                continue
            checked += 1
            if cells != gt["positions"]:
                first = next((i for i, (a, b) in enumerate(zip(cells, gt["positions"]))
                              if a != b), min(len(cells), len(gt["positions"])))
                problems[f"{concept}/{demo_id}"] = (
                    f"{len(cells)} cells vs {len(gt['positions'])}, first differs at {first}: "
                    f"keyframes {cells[first:first + 2]} vs gt {gt['positions'][first:first + 2]}")
    return problems, checked


# ----------------------------------------------------------------------------------- #

def test_keyframe_cells_match_ground_truth():
    problems, checked = check_conversion(sorted(ALL_CONCEPTS))
    assert checked, "no demos were checked"
    assert not problems, "\n".join(f"{k}: {v}" for k, v in problems.items())


def test_demo_selection_prefers_distinct_parameters():
    def sketch(value):
        return {"concept": "row", "arguments": {"length": {"type": "int", "value": value}}}

    demos = [{"demo_id": "a"}, {"demo_id": "b"}, {"demo_id": "c"}]
    infos = [sketch(5), sketch(5), sketch(3)]

    assert select_demos(demos, infos, 2, "parity") == [0, 1]
    assert select_demos(demos, infos, 2, "distinct_params") == [0, 2]
    # Already distinct: both rules agree, so only `row` is affected.
    assert select_demos(demos, [sketch(4), sketch(3), sketch(5)], 2, "distinct_params") == [0, 1]


def test_parameter_is_read_from_the_sketch():
    assert _parameter({"arguments": {"steps": {"type": "int", "value": 4}}}) == ("steps", 4)
    assert _parameter({"arguments": {"height": {"type": "int", "value": "6"}}}) == ("height", 6)


def main() -> int:
    print("demo-selection rules")
    test_demo_selection_prefers_distinct_parameters()
    test_parameter_is_read_from_the_sketch()
    print("  OK\n")

    print("keyframes -> lattice cells, cross-checked against ground truth")
    problems, checked = check_conversion(sorted(ALL_CONCEPTS))
    print(f"  checked {checked} demos across {len(ALL_CONCEPTS)} concepts")
    if problems:
        print(f"\n{len(problems)} MISMATCH(ES):")
        for label, why in sorted(problems.items()):
            print(f"  {label}: {why}")
        return 1
    print("  all demos agree with ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_scoring_reload_is_triggered_by_missing_geometry_not_by_a_counter():
    '''Regression: the reload before scoring must key off whether the demonstrations in hand
    actually lack their meshes.

    It used to key off how many this run had released. Once `_load_demos(summarise=True)`
    started releasing during streaming, that counter came back 0, the reload was skipped, and
    scoring ran on mesh-less demonstrations. `program_accuracy` still looked right -- it is
    computed against `run_gt_program` and needs no geometry -- while `plan_accuracy`,
    `mean_iou` and both stability metrics silently became None. A metric that quietly turns
    into None is exactly the failure this codebase keeps producing.
    '''
    import inspect

    from baseline_spl.symbolic.harness import SearchHarness

    source = inspect.getsource(SearchHarness.learn_all)
    assert "needs_reload" in source, "the reload condition was removed"
    assert 'demo.get("meshes") is None' in source, (
        "the reload must test for absent geometry, not for a release count")
    # And the release counter must not be what gates it.
    gating = [line for line in source.splitlines()
              if "if released:" in line and "needs_reload" not in line]
    assert not gating, f"reload still gated on the release counter: {gating}"
