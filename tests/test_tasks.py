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

from SPL.config.spl_config import ORIGINAL_CONCEPTS
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
        if concept not in ORIGINAL_CONCEPTS:
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
    # DATA_DIR, not SPLConfig.train_dataset_dir: demos are paired by id with the ground
    # truth read from DATA_DIR above, so both must come from the same dataset
    loader = build_inductive_structure_dataloader(
        SPLConfig.dataset_name, DATA_DIR, SPLConfig.assets_dir,
        camera_view=SPLConfig.camera_view, batch_size=1, num_workers=0, shuffle=False,
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
    problems, checked = check_conversion(sorted(ORIGINAL_CONCEPTS))
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
    '''Every integer argument, in sketch order -- that order is how SPL's
    `check_program_equivalence` pairs them, so it is the order a recovered class must use.'''
    assert _parameter({"arguments": {"steps": {"type": "int", "value": 4}}}) == (("steps",), (4,))
    assert _parameter({"arguments": {"height": {"type": "int", "value": "6"}}}) \
        == (("height",), (6,))

    # 12 of the 99 concepts take two integers and 2 take three.
    assert _parameter({"arguments": {"length": {"type": "int", "value": 5},
                                     "breadth": {"type": "int", "value": 3}}}) \
        == (("length", "breadth"), (5, 3))

    # The object list is not a size, however it is spelled.
    assert _parameter({"arguments": {"length": {"value": 5},
                                     "objects": {"value": [1, 2, 3]}}}) \
        == (("length",), (5,))


def test_fixed_size_concepts_have_no_integer_argument():
    '''15 of the 99 concepts take none. That is a CLOSED program, not an error -- raising was
    what made a default run over ALL_CONCEPTS die after loading every demonstration.'''
    assert _parameter({"arguments": {"objects": {"value": [1]}}}) == ((), ())


def test_a_concept_argument_is_inexpressible_not_a_crash():
    '''`wall` takes another concept, which the grammar has no type for. It must be recorded so
    one such concept cannot abort a run over the other 98.'''
    from baseline_spl.symbolic.tasks import Inexpressible

    try:
        _parameter({"arguments": {"along": {"type": "concept", "value": "row"}}})
    except Inexpressible as exc:
        assert "along" in str(exc)
    else:
        raise AssertionError("a concept-valued argument must be reported as inexpressible")


def main() -> int:
    print("demo-selection rules")
    test_demo_selection_prefers_distinct_parameters()
    test_parameter_is_read_from_the_sketch()
    print("  OK\n")

    print("keyframes -> lattice cells, cross-checked against ground truth")
    problems, checked = check_conversion(sorted(ORIGINAL_CONCEPTS))
    print(f"  checked {checked} demos across {len(ORIGINAL_CONCEPTS)} concepts")
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

    This asserts the BEHAVIOUR, not the source text. It used to grep
    `inspect.getsource(learn_all)` for two string literals, which broke the moment the loop
    moved into `_register_and_score` -- twice now, costing a session each time, while telling
    nobody whether the reload actually happens.
    '''
    from collections import OrderedDict

    from baseline_spl.symbolic.harness import SearchHarness

    class Stub:
        '''Only the three members `_register_and_score` calls.'''

        def __init__(self):
            self.reloaded = []

        def _load_demos(self, window):
            self.reloaded.append(list(window))
            return OrderedDict((c, [{"meshes": ["reloaded"]}]) for c in window)

        def _score_window(self, window_demos, *_args):
            self.scored = list(window_demos)

        def _reclaim(self):
            pass

    class Cfg:
        scoring_chunk = 8

    stripped = OrderedDict(row=[{"meshes": None}], tower=[{"meshes": None}])
    intact = OrderedDict(row=[{"meshes": ["geometry"]}], tower=[{"meshes": ["geometry"]}])

    without_geometry = Stub()
    SearchHarness._register_and_score(without_geometry, stripped, {}, None, [], Cfg)
    assert without_geometry.reloaded == [["row", "tower"]], (
        "demonstrations with no meshes must be reloaded before scoring, or plan_accuracy, "
        "mean_iou and the stability metrics silently become None")

    with_geometry = Stub()
    SearchHarness._register_and_score(with_geometry, intact, {}, None, [], Cfg)
    assert with_geometry.reloaded == [], (
        "demonstrations that still hold their geometry must not be reloaded")


def test_settings_resolve_identically_for_every_config():
    '''`SearchSettings.from_config` is the only place a config object is read, so a mistake in
    it changes every run at once and changes a *number*, not a behaviour that raises.

    The suite cannot catch that on its own: all four tests that call `driver.run` pass
    `use_recognition=False`, so the recognition settings are never exercised. This checks the
    resolved values directly instead, for every config in the package.

    Four of these settings appear in no config file at all -- `recognition_hidden`,
    `contextual`, `bias_optimal`, `auxiliary_loss`. Their default IS the production value for
    every run, and `train_recognizer` swallows exceptions, so a wrong `hidden` degrades to "no
    recognizer, global grammar" with a single log line.
    '''
    import os
    import pathlib
    import sys
    import tempfile

    import baseline_spl.configs as configs_pkg
    from baseline_spl.common import config as config_module
    from baseline_spl.common.config import BaselineConfig
    from baseline_spl.common.factory import resolve_run_name
    from baseline_spl.symbolic.settings import SearchSettings

    names = sorted(p.stem for p in pathlib.Path(configs_pkg.__file__).parent.glob("*.py")
                   if p.stem != "__init__")
    assert names, "no config files found at all"

    original = os.environ.get("BASELINE_CONFIG")
    # Building a config MAKES its run directory (`BaselineConfig.__init__` calls os.makedirs),
    # so probing every config here littered runs/ with a `<config>_probe` directory apiece --
    # 11 of them, recreated on every pytest run, which made deleting them pointless. Build into
    # a temporary root instead; nothing here ever reads the directory back.
    scratch = tempfile.TemporaryDirectory(prefix="baseline_probe_")
    real_runs_root = config_module.RUNS_ROOT
    config_module.RUNS_ROOT = pathlib.Path(scratch.name)
    checked = 0
    try:
        for name in names:
            for kind in ("DreamCoderConfig", "LiloConfig"):
                os.environ["BASELINE_CONFIG"] = name
                for module in [m for m in list(sys.modules) if m.startswith("baseline_spl.config")]:
                    del sys.modules[module]
                import baseline_spl.config as active

                cfg_cls = getattr(active, kind, None)
                if cfg_cls is None:
                    continue
                cfg = BaselineConfig.from_run_config(cfg_cls, resolve_run_name(cfg_cls, "probe"))
                settings = SearchSettings.from_config(cfg)
                recognition = settings.recognition

                # Every knob must come from the config where the config defines one.
                assert settings.timeout == cfg.enumeration_timeout, f"{name}:{kind} timeout"
                assert settings.cpus == cfg.cpus, f"{name}:{kind} cpus"
                assert settings.pseudo_counts == cfg.pseudo_counts, f"{name}:{kind} pseudo_counts"
                assert recognition.enabled == cfg.use_recognition, f"{name}:{kind} use_recognition"
                assert recognition.steps == cfg.recognition_steps, f"{name}:{kind} steps"

                # The architecture must never silently fall back to upstream's weaker defaults.
                assert recognition.hidden == 64, f"{name}:{kind} hidden"
                assert recognition.contextual is True, f"{name}:{kind} contextual"
                assert recognition.bias_optimal is True, f"{name}:{kind} bias_optimal"
                assert recognition.auxiliary_loss is True, f"{name}:{kind} auxiliary_loss"
                checked += 1
    finally:
        config_module.RUNS_ROOT = real_runs_root
        scratch.cleanup()
        if original is None:
            os.environ.pop("BASELINE_CONFIG", None)
        else:
            os.environ["BASELINE_CONFIG"] = original
        for module in [m for m in list(sys.modules) if m.startswith("baseline_spl.config")]:
            del sys.modules[module]

    assert checked, "no config/kind combination was checked, so this test proved nothing"
