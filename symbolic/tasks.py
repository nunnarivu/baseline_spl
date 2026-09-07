'''
tasks.py

Turn demonstrations into synthesis tasks.

A demonstration gives keyframes; a synthesis task needs an input/output example. The output
is the ordered list of integer lattice cells where blocks landed -- exactly the quantity
`check_program_equivalence` compares (metrics.py:495), so solving the task and scoring well
are the same thing by construction.

Two framings, per the plan:

  concept-level (default, the scored path)
      one task per concept, one example per demo: {steps=4 -> cells, steps=3 -> cells}.
      A program with the parameter hard-coded satisfies one example and fails the other, so
      the loop is forced and induction is the search objective.

  demo-level (`task_granularity="demo"`, a flag for future use)
      one task per demo, closed program. DreamCoder as published; the parameter only appears
      when STITCH anti-unifies two solutions.

Leakage note: the target trace is derived from the *keyframes*, never from
`demo.json:program`. The ground-truth program is used only as a test oracle in
tests/test_tasks.py, never as an input. The parameter value comes from the shared
SketchAgent's reading of the instruction ("a staircase of steps 4"), which is the same source
every other baseline uses.
'''

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from baseline_spl.common.serialize_text import (HORIZONTAL_PITCH, VERTICAL_PITCH,
                                                _lattice, estimate_pitch)
from baseline_spl.symbolic.search import SearchTask

Cell = Tuple[int, int, int]


# Key under which a streamed demonstration caches its extracted placement offsets. Once this
# is present the geometry can be dropped: it is the only thing the search ever reads from a
# demonstration, and it is a few hundred bytes against a few gigabytes of trimesh.
DELTA_CACHE_KEY = "_placement_deltas"
SRN_CACHE_KEY = "_srn_table"


def summarise_demo(demo: dict, executor=None, want_srn: bool = False) -> dict:
    """Extract everything the search needs from one demonstration, in place.

    Called while the demonstration is still streaming out of the dataloader, so its meshes can
    be released immediately afterwards. This is what makes the memory cost independent of the
    number of concepts: 16 concepts held ~53 GB of geometry at once, so 100 would want ~330 GB
    against 251 GB of RAM. Extracting first turns that into a few hundred bytes per demo.

    Two things are needed and both require the meshes:
      * the placement offsets, which every task target is derived from;
      * the SRN table, which needs the reference object's mesh in order to voxelize it.
    """
    demo[DELTA_CACHE_KEY] = demo_deltas(demo)
    if want_srn:
        from baseline_spl.symbolic import srn

        demo[SRN_CACHE_KEY] = srn.table_for(
            executor, demo, (HORIZONTAL_PITCH, VERTICAL_PITCH))
    return demo


def pitch_from_deltas(all_deltas: Sequence[Sequence]) -> tuple:
    """`estimate_pitch`, computed from cached offsets instead of live meshes.

    Same statistic and the same guards as `common/serialize_text.estimate_pitch`: the median
    single-axis step per axis, with a fallback whenever a run took no unit step on that axis
    (which is `pins`, whose every horizontal move is two cells -- see `global_pitch`).
    """
    import numpy as np

    horizontal, vertical = [], []
    for deltas in all_deltas:
        for previous, current in zip(deltas, deltas[1:]):
            step = np.abs(np.asarray(current) - np.asarray(previous))
            planar = float(max(step[0], step[1]))
            if step[2] < 0.02 and planar > 0.05:
                horizontal.append(planar)
            elif planar < 0.02 and step[2] > 0.02:
                vertical.append(float(step[2]))
    h = float(np.median(horizontal)) if horizontal else HORIZONTAL_PITCH
    v = float(np.median(vertical)) if vertical else VERTICAL_PITCH
    if h > 1.5 * HORIZONTAL_PITCH:
        h = HORIZONTAL_PITCH
    if v > 1.5 * VERTICAL_PITCH:
        v = VERTICAL_PITCH
    return h, v


def demo_deltas(demo: dict) -> List:
    '''The observed placement centroids, relative to the first placed block.

    This is the raw observation, before any decision about how to represent it. Both
    `demo_cells` (integers, Round 1) and `demo_centroids` (metres, Round 2) are views on it,
    which keeps them from drifting apart.

    The origin is the first placed block so that demos whose scenes sit at different table
    positions are directly comparable, and so the origin matches the executors'.
    '''
    cached = demo.get(DELTA_CACHE_KEY)
    if cached is not None:
        return cached

    from SPL.model.executor import mesh_centroid
    from SPL.utils.metrics import demo_step_objects

    meshes: Sequence = demo.get("meshes") or []
    if len(meshes) < 2:
        raise ValueError(f"demo {demo.get('demo_id')} has fewer than 2 keyframes")

    steps = demo_step_objects(meshes, demo.get("objects_moved"))
    if not steps:
        raise ValueError(f"demo {demo.get('demo_id')} has no detectable placements")

    origin = mesh_centroid(meshes[1][steps[0]])

    out = []
    for k, obj in enumerate(steps):
        after = meshes[k + 1]
        if obj is None or obj >= len(after):
            continue
        out.append(mesh_centroid(after[obj]) - origin)
    return out


def demo_cells(demo: dict, pitch: Optional[tuple] = None) -> List[Cell]:
    '''The ordered lattice cells a demonstration places blocks on.

    Reuses `estimate_pitch` / `_lattice` from common/serialize_text.py rather than
    re-deriving them: the domain is anisotropic (a horizontal step is ~0.109 m, a vertical
    one ~0.05 m) and that module already measures the pitch from the demos themselves.

    Round 2 note: this rounding was measured to be lossless on this dataset -- the worst
    residual over 93 placements was 0.209 of a cell, against the 0.5 that would misassign
    one. So it removes a *decision* the other baselines have to make, not a *difficulty*
    they have to survive. `observation_mode="continuous"` is the fair default; this remains
    as the ablation that quantifies the difference.
    '''
    if pitch is None:
        pitch = estimate_pitch([demo])
    import numpy as np

    zero = np.zeros(3)
    return [_lattice(delta, zero, pitch) for delta in demo_deltas(demo)]


def demo_centroids(demo: dict) -> List[tuple]:
    '''The ordered placement centroids in metres, relative to the first placed block.

    The same quantity CaP / Demo2Code receive at `coordinate_mode="raw"`, and the same one
    SPL's planner works from. Plain tuples rather than arrays, because the Gaussian executor
    that consumes them is arithmetic on 3-vectors and numpy overhead would dominate.
    '''
    return [tuple(float(v) for v in delta) for delta in demo_deltas(demo)]


def _parameter(sketch_info: dict) -> Tuple[str, int]:
    '''(name, value) of the single integer argument the sketch read off the instruction.'''
    arguments = (sketch_info or {}).get("arguments") or {}
    for name, spec in arguments.items():
        value = spec.get("value") if isinstance(spec, dict) else spec
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return name, int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return name, int(value.strip())
    raise ValueError(f"no integer parameter in sketch {sketch_info!r}")


def select_demos(demos: Sequence[dict], sketch_infos: Sequence[dict], limit: int,
                 selection: str = "distinct_params") -> List[int]:
    '''Indices of the demos to learn from.

    "parity"           the first `limit`, exactly what SPL sees.
    "distinct_params"  prefer demos whose parameters differ, falling back to order.

    Only `row` is affected: its first two demos are both length 5 while its third is 3, so
    with "parity" the concept-level task cannot force the loop. Every other concept's first
    two demos already differ. The choice is recorded in the run directory because it is an
    input asymmetry against SPL and has to be disclosed rather than buried.
    '''
    if selection not in ("parity", "distinct_params"):
        raise ValueError(f"unknown demo selection {selection!r}")
    if selection == "parity":
        return list(range(min(limit, len(demos))))

    chosen: List[int] = []
    seen: set = set()
    for i, info in enumerate(sketch_infos):
        try:
            _, value = _parameter(info)
        except ValueError:
            continue
        if value not in seen:
            chosen.append(i)
            seen.add(value)
        if len(chosen) == limit:
            return chosen
    # Not enough distinct values: top up in order so we still return `limit` demos.
    for i in range(len(demos)):
        if len(chosen) >= limit:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def _target(demo: dict, pitch: tuple, observation_mode: str):
    '''One example's target, in whichever representation the run is configured for.'''
    if observation_mode == "lattice":
        return demo_cells(demo, pitch)
    if observation_mode == "continuous":
        return demo_centroids(demo)
    raise ValueError(f"unknown observation_mode {observation_mode!r}")


def _tables(demos: Sequence[dict], pitch: tuple, observation_mode: str,
            executor) -> List[dict]:
    '''Per-demo SRN tables, or [] when nothing will read them.

    Only the probabilistic evaluator needs these, but they are built alongside the task
    rather than on demand: the SRN runs once per demo here, and the alternative is a network
    forward pass reachable from inside enumeration.
    '''
    if observation_mode != "continuous":
        return []
    from baseline_spl.symbolic import srn

    tables = []
    for demo in demos:
        cached = demo.get(SRN_CACHE_KEY)
        if cached is not None:                      # extracted while streaming
            tables.append(cached)
        elif executor is None:
            tables.append(srn.analytic_table(pitch))
        else:
            tables.append(srn.table_for(executor, demo, pitch))
    return tables


def concept_task(concept: str, demos: Sequence[dict], sketch_infos: Sequence[dict],
                 pitch: Optional[tuple] = None, *, observation_mode: str = "lattice",
                 executor=None) -> SearchTask:
    '''One task per concept: examples are (parameter, target), one per demo.'''
    if pitch is None:
        pitch = estimate_pitch(list(demos))

    examples: List[Tuple[int, List]] = []
    names: set = set()
    ids: List[str] = []
    for demo, info in zip(demos, sketch_infos):
        name, value = _parameter(info)
        names.add(name)
        examples.append((value, _target(demo, pitch, observation_mode)))
        ids.append(str(demo.get("demo_id")))

    if len(names) > 1:
        # The sketch disagreed across demos; `SharedSketch.corrected` should have settled
        # this upstream, so surface it rather than silently picking one.
        raise ValueError(f"{concept}: demos disagree on the parameter name: {sorted(names)}")

    return SearchTask(name=concept, examples=examples,
                      param_name=next(iter(names)), demo_ids=tuple(ids),
                      instruction=str(demos[0].get("language_instruction", "")).strip(),
                      observation_mode=observation_mode,
                      srn_tables=_tables(demos, pitch, observation_mode, executor))


def demo_tasks(concept: str, demos: Sequence[dict], sketch_infos: Sequence[dict],
               pitch: Optional[tuple] = None, *, observation_mode: str = "lattice",
               executor=None) -> List[SearchTask]:
    '''One task per demo -- DreamCoder as published. Kept behind `task_granularity="demo"`.'''
    if pitch is None:
        pitch = estimate_pitch(list(demos))
    tables = _tables(demos, pitch, observation_mode, executor)
    tasks = []
    for i, (demo, info) in enumerate(zip(demos, sketch_infos)):
        name, value = _parameter(info)
        tasks.append(SearchTask(name=f"{concept}_{demo.get('demo_id')}",
                                examples=[(value, _target(demo, pitch, observation_mode))],
                                param_name=name, demo_ids=(str(demo.get("demo_id")),),
                                instruction=str(demo.get("language_instruction", "")).strip(),
                                observation_mode=observation_mode,
                                srn_tables=[tables[i]] if tables else [],
                                # DreamCoder as published: the program is closed and the size
                                # appears only as a literal, so `tstate -> tstate`.
                                closed=True))
    return tasks


def global_pitch(all_demos: Sequence[dict]) -> tuple:
    '''Lattice pitch measured across *every* demonstration, not per concept.

    Per-concept estimation is wrong for `pins`: its consecutive placements are always two
    shifts apart (`shift right; shift right; place`), so the measured horizontal step comes
    out at 0.221 m -- exactly twice the true 0.109 -- and every pins cell is halved. Measured
    globally the median lands on 0.111, because most concepts do take single steps.

    The pitch is a property of the simulator (block extent plus
    ParameterSettings.SPATIAL_TARGET_MARGIN), not of a concept, so a global estimate is also
    the more principled choice. The guard below catches the pathological case where the whole
    concept set is pins-like.
    '''
    # From cached offsets when the demonstrations have been summarised and their geometry
    # released (the streaming path); from the meshes otherwise. Same statistic either way.
    demos = list(all_demos)
    if demos and demos[0].get(DELTA_CACHE_KEY) is not None:
        horizontal, vertical = pitch_from_deltas([d[DELTA_CACHE_KEY] for d in demos])
    else:
        horizontal, vertical = estimate_pitch(demos)
    if horizontal > 1.5 * HORIZONTAL_PITCH:
        import warnings
        warnings.warn(
            f"estimated horizontal pitch {horizontal:.4f} is more than 1.5x the expected "
            f"{HORIZONTAL_PITCH}; no demonstration appears to take a single horizontal step. "
            f"Falling back to the constant.")
        horizontal = HORIZONTAL_PITCH
    return horizontal, vertical


def build_tasks(grouped: Dict[str, Sequence[dict]], sketches: Dict[str, Sequence[dict]],
                *, granularity: str = "concept", selection: str = "distinct_params",
                demos_per_concept: int = 2, observation_mode: str = "lattice",
                executor=None) -> Tuple[List[SearchTask], Dict[str, List[str]]]:
    '''Build every task, and report which demos each one used.

    Returns (tasks, {concept: [demo_id, ...]}). The second value is written to
    demo_selection.json so the demo choice is auditable.
    '''
    if granularity not in ("concept", "demo"):
        raise ValueError(f"unknown task granularity {granularity!r}")

    # One pitch for the whole run -- see global_pitch for why this must not be per concept.
    pitch = global_pitch([d for demos in grouped.values() for d in demos])

    tasks: List[SearchTask] = []
    used: Dict[str, List[str]] = {}
    for concept, demos in grouped.items():
        infos = list(sketches[concept])
        picked = select_demos(demos, infos, demos_per_concept, selection)
        chosen_demos = [demos[i] for i in picked]
        chosen_infos = [infos[i] for i in picked]
        if granularity == "concept":
            tasks.append(concept_task(concept, chosen_demos, chosen_infos, pitch,
                                      observation_mode=observation_mode, executor=executor))
        else:
            tasks.extend(demo_tasks(concept, chosen_demos, chosen_infos, pitch,
                                    observation_mode=observation_mode, executor=executor))
        used[concept] = [str(d.get("demo_id")) for d in chosen_demos]
    return tasks, used
