'''
serialize_text.py

Turn an SPL demonstration into the exact text format Demo2Code's own parser expects,
so upstream's ``[Scenario i]`` splitting and recursive summarization run unmodified.

Fairness is the whole point of this file. The baseline must receive **the same
information SPL's planner receives** — the keyframe meshes — just rendered as text.
Two choices follow from that:

  * ``coordinate_mode`` picks how a placement's position is written. ``lattice`` gives an
    integer grid cell, ``raw`` the centroid in metres to 2 decimals. Both are relative to
    the first placed block, so scenes at different table positions stay comparable.
  * The direction between placements is **not** labelled. A step can be diagonal, or have
    no single LEFT/RIGHT/... name at all, so naming it is the model's job; the primitive's
    per-direction statistics (``common/primitive_stats.py``) are what it uses to do so.

The upstream parser requires the first post-action block to be labelled literally
``State 2:`` (``code_gen_helper.code_generator`` partitions on that string), so keyframe
k is labelled ``State k+1``.
'''

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from SPL.config.primitive_config import DEFAULT_ACTIONS
from SPL.model.executor import mesh_centroid
from SPL.utils.metrics import demo_step_objects


# Unit direction vectors, taken from the simulator's own definition rather than
# reproduced here — nsei_simulator/dataset/spg/configs.py is what actually generated the
# demonstrations, so importing it makes drift impossible.
#
# These are NOT the obvious x=LEFT/RIGHT, y=FRONT/BEHIND guess: RIGHT is -y and FRONT is
# -x. Getting it wrong silently makes every baseline build a rotated structure and score
# badly for a reason unrelated to its capability, which is why this is imported and then
# cross-checked against the ground-truth programs in tests/test_serialize.py.
#
# Only the directions the DSL actually exposes (DEFAULT_ACTIONS) are offered as candidate
# labels; the simulator also defines "below", which shift_focus cannot produce.
def _load_direction_vectors() -> Dict[str, np.ndarray]:
    from nsei_simulator.dataset.spg.configs import ParameterSettings

    source = {name.upper(): vector for name, vector in ParameterSettings.DIRECTIONS.items()}
    missing = [a for a in DEFAULT_ACTIONS if a.upper() not in source]
    if missing:
        raise RuntimeError(
            f"DSL actions {missing} have no entry in ParameterSettings.DIRECTIONS; "
            f"the simulator's direction table and SPL's action vocabulary disagree."
        )
    return {action.upper(): np.asarray(source[action.upper()], dtype=float)
            for action in DEFAULT_ACTIONS}


_DIRECTION_VECTORS: Dict[str, np.ndarray] = _load_direction_vectors()


def nearest_direction(delta: np.ndarray) -> Optional[str]:
    '''Name the DSL direction best matching a displacement, or None if the
    displacement is essentially zero. Only directions in DEFAULT_ACTIONS are offered,
    so the vocabulary handed over is exactly the one the DSL exposes.'''
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return None
    unit = delta / norm
    best, best_dot = None, -np.inf
    for name in DEFAULT_ACTIONS:
        vec = _DIRECTION_VECTORS.get(name.upper())
        if vec is None:
            continue
        dot = float(np.dot(unit, vec))
        if dot > best_dot:
            best, best_dot = name.upper(), dot
    return best


# Fallback lattice pitch. The domain is anisotropic by construction: the simulator adds
# ParameterSettings.SPATIAL_TARGET_MARGIN (0.06 m) to horizontal targets but stacks
# vertically flush, so with ~0.05 m blocks a row step is ~0.11 m while a tower step is
# ~0.05 m. Using one scalar pitch for both axes yields non-integer "grid cells" like
# (0, -2.25, 0), which is worse than useless in a prompt.
#
# These are only fallbacks — block extents vary per asset (0.044-0.056 m), so
# estimate_pitch() below measures the actual pitch from the demonstrations themselves and
# overrides them whenever a demo exercises the relevant axis.
HORIZONTAL_PITCH = 0.109
VERTICAL_PITCH = 0.050


def estimate_pitch(demos: Sequence[dict]) -> tuple:
    '''Estimate (horizontal, vertical) lattice pitch from the demonstrations themselves,
    so the rendering adapts if the assets or spacing change. Falls back to the measured
    constants for whichever axis a demo never exercises (a tower has no horizontal
    steps, a row no vertical ones).

    The pitch is the *unit* step, so it can only be measured from a demonstration that
    actually takes one. ``pins`` never does -- it moves ``shift right; shift right; place``,
    so every consecutive pair of placements is two cells apart and the naive estimate comes
    out at 0.221 m, exactly twice the true 0.109. Every cell it produced was then halved.
    No statistic over pins' own deltas can recover the unit (they are all doubled), so when
    the estimate is an implausible multiple of the expected pitch we fall back to the
    constant. Passing several concepts' demos at once avoids the situation entirely, because
    most concepts do take single steps.
    '''
    horizontal, vertical = [], []
    for demo in demos:
        meshes = demo.get("meshes") or []
        if len(meshes) < 2:
            continue
        steps = demo_step_objects(meshes, demo.get("objects_moved"))
        previous = None
        for k, obj in enumerate(steps):
            if previous is not None and obj is not None and obj < len(meshes[k + 1]):
                delta = np.abs(mesh_centroid(meshes[k + 1][obj])
                               - mesh_centroid(meshes[k + 1][previous]))
                planar = float(max(delta[0], delta[1]))
                if delta[2] < 0.02 and planar > 0.05:
                    horizontal.append(planar)
                elif planar < 0.02 and delta[2] > 0.02:
                    vertical.append(float(delta[2]))
            previous = obj
    h = float(np.median(horizontal)) if horizontal else HORIZONTAL_PITCH
    v = float(np.median(vertical)) if vertical else VERTICAL_PITCH

    # Implausible multiples mean no demo took a unit step on that axis (see the docstring).
    if h > 1.5 * HORIZONTAL_PITCH:
        h = HORIZONTAL_PITCH
    if v > 1.5 * VERTICAL_PITCH:
        v = VERTICAL_PITCH
    return h, v


def _lattice(centroid: np.ndarray, origin: np.ndarray, pitch: tuple) -> tuple:
    horizontal, vertical = pitch
    scale = np.array([horizontal, horizontal, vertical])
    cell = np.round((centroid - origin) / scale).astype(int)
    return tuple(int(v) for v in cell)


def _raw(centroid: np.ndarray, origin: np.ndarray) -> tuple:
    '''Centroid relative to the first placed block, rounded to 2 decimals.
    ``+ 0.0`` normalises the -0.0 that rounding a tiny negative produces.'''
    return tuple(round(float(v), 2) + 0.0 for v in (centroid - origin))


def demo_to_scenario(demo: dict, index: int, *, coordinate_mode: str = "lattice",
                     pitch: Optional[tuple] = None) -> str:
    '''Render one demonstration as a ``[Scenario i]`` block.

    Positions only — the direction between placements is deliberately not labelled.
    A step can be diagonal or otherwise have no single LEFT/RIGHT/... name, so naming
    it is the model's job; the primitive's per-direction statistics (see
    common/primitive_stats.py) are what it uses to do so.
    '''
    if coordinate_mode not in ("lattice", "raw"):
        raise ValueError(f"coordinate_mode must be 'lattice' or 'raw', got {coordinate_mode!r}")
    if pitch is None:
        pitch = estimate_pitch([demo])
    meshes: Sequence = demo.get("meshes") or []
    if len(meshes) < 2:
        raise ValueError(f"demo {demo.get('demo_id')} has fewer than 2 keyframes")

    names: List[str] = list(demo.get("object_names") or [])

    def name_of(i: int) -> str:
        return names[i] if 0 <= i < len(names) else f"object_{i}"

    steps = demo_step_objects(meshes, demo.get("objects_moved"))

    # Both modes are relative to the first placed block, so coordinates are comparable
    # across demos whose scenes sit at different table positions.
    first_obj = steps[0] if steps else 0
    origin = mesh_centroid(meshes[1][first_obj])

    lines = [f"[Scenario {index}]", demo.get("language_instruction", "").strip(), ""]

    for k, obj in enumerate(steps):
        after = meshes[k + 1]
        if obj is None or obj >= len(after):
            continue
        centroid = mesh_centroid(after[obj])
        position = (_lattice(centroid, origin, pitch) if coordinate_mode == "lattice"
                    else _raw(centroid, origin))
        label = "grid cell" if coordinate_mode == "lattice" else "position"
        lines.append(f"State {k + 2}:")
        lines.append(f"'{name_of(obj)}' has moved")
        lines.append(f"'{name_of(obj)}' is at {label} {position}")

    return "\n".join(lines).strip()


def to_demo2code_text(demos: Sequence[dict], *, coordinate_mode: str = "lattice") -> str:
    '''Render a concept's demonstrations in upstream's demonstration-file format:

        scenario_1_objects=[...]
        scenario_2_objects=[...]
        """
        [Scenario 1]
        <instruction>

        State 2:
        ...
        """

    Each demonstration is a different scene with its own objects, so the header lists
    them per scenario. It has to go here rather than inside the scenario: upstream's
    ``code_generator`` discards everything between ``[Scenario i]`` and ``State 2:`` for
    scenarios after the first, but passes this header through as ``env_info``.
    '''
    if not demos:
        raise ValueError("no demonstrations to serialize")
    header = "\n".join(
        f"scenario_{i + 1}_objects=" + repr(list(demo.get("object_names") or []))
        for i, demo in enumerate(demos))
    # One pitch estimate across all of a concept's demos: more samples, and it keeps the
    # lattice comparable between the scenarios the model is asked to generalize over.
    pitch = estimate_pitch(demos)
    scenarios = [demo_to_scenario(d, i + 1, coordinate_mode=coordinate_mode, pitch=pitch)
                 for i, d in enumerate(demos)]
    body = "\n\n".join(scenarios)
    return f'{header}\n"""\n{body}\n"""'
