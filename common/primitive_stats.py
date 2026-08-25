'''
primitive_stats.py

Per-direction statistics of the placement primitive (the Spatial Relation Network), so a
model reading bare coordinates can work out which ``shift_focus`` direction a movement
corresponds to.

The SRN maps (reference-object voxel, direction) to a diagonal Gaussian over the focus
delta. ``executor._get_ref_voxel`` centres the mesh before voxelizing, so the prediction
depends only on the reference block's geometry — computable per demonstration from its
first keyframe.

We deliberately do NOT call ``executor._get_ref_voxel``: its ``_voxel_cache`` is keyed by
object index and cleared only in ``register_state``, so calling it across demos would
hand back the first demo's voxel for every later one. The voxelization below mirrors it.
'''

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from SPL.config.primitive_config import DEFAULT_ACTIONS

# Means closer than this across demos are reported as one row rather than per-demo.
AGREEMENT_TOL = 0.005


def _reference_voxel(executor, mesh):
    from SPL.config.primitive_config import DatasetConfig, SRNConfig
    from SPL.dataloader.voxelize import voxelize_mesh

    centred = mesh.copy()
    centred.apply_translation(-centred.centroid)
    srn_cfg = SRNConfig(load_checkpoint=executor.configs.primitive_checkpoint)
    return voxelize_mesh(
        mesh=centred, mesh_scale=1.0, mesh_transform=np.eye(4),
        context_window_size=DatasetConfig().context_window_size,
        grid_size=srn_cfg.grid_size, centering=[1, 1, 1],
    )


def _forward(executor, ref_voxel, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    '''(mean, std) of the predicted delta for one direction.'''
    import torch
    from SPL.primitives.neural.neural3d import build_action_vocab

    action_idx = build_action_vocab(DEFAULT_ACTIONS).get(direction.upper(), 0)
    device = next(executor._move_focus.parameters()).device
    ref_t = torch.from_numpy(ref_voxel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    act_t = torch.tensor([action_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        mu, log_sigma = executor._move_focus(ref_t, act_t)
    return mu.squeeze(0).cpu().numpy(), np.exp(log_sigma.squeeze(0).cpu().numpy())


def direction_stats(executor, demos: Sequence[dict]) -> List[Dict]:
    '''One entry per demonstration: {"demo_id": ..., "stats": {DIRECTION: (mean, std)}}.'''
    out = []
    for demo in demos:
        meshes = demo.get("meshes") or []
        if not meshes or not meshes[0]:
            continue
        voxel = _reference_voxel(executor, meshes[0][0])
        stats = {d.upper(): _forward(executor, voxel, d) for d in DEFAULT_ACTIONS}
        out.append({"demo_id": demo.get("demo_id"), "stats": stats})
    return out


def _agree(per_demo: List[Dict]) -> bool:
    if len(per_demo) < 2:
        return True
    first = per_demo[0]["stats"]
    return all(
        np.max(np.abs(entry["stats"][d][0] - first[d][0])) <= AGREEMENT_TOL
        for entry in per_demo[1:] for d in first
    )


def _table(stats: Dict[str, Tuple[np.ndarray, np.ndarray]], indent: str = "  ") -> str:
    def fmt(v):
        return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"

    rows = [f"{indent}{'direction':<10} {'mean (x, y, z)':<26} std (x, y, z)"]
    for direction in DEFAULT_ACTIONS:
        mean, std = stats[direction.upper()]
        rows.append(f"{indent}{direction.upper():<10} {fmt(mean):<26} {fmt(std)}")
    return "\n".join(rows)


def render_stats_block(per_demo: List[Dict]) -> str:
    '''The prompt section. One table when the demonstrations agree, otherwise one per
    demonstration — the demos use different block assets (0.044-0.056 m), which moves
    the predicted deltas.'''
    if not per_demo:
        return ""
    header = (
        "# shift_focus delta statistics\n"
        "Calling shift_focus(direction) moves the focus by the delta below, in metres.\n"
        "Use these to decide which direction (or sequence of directions) accounts for the\n"
        "movement between two placements. A movement may need more than one shift.\n"
    )
    if _agree(per_demo):
        return f"{header}\n{_table(per_demo[0]['stats'])}\n"
    blocks = [f"  Scenario {i} (demo {entry['demo_id']}):\n{_table(entry['stats'], indent='    ')}"
              for i, entry in enumerate(per_demo, 1)]
    return f"{header}\n" + "\n".join(blocks) + "\n"


def build_stats_block(executor, demos: Sequence[dict], enabled: bool = True) -> str:
    '''Convenience wrapper: returns "" when disabled or when the SRN is unavailable.'''
    if not enabled:
        return ""
    try:
        return render_stats_block(direction_stats(executor, demos))
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(f"[primitive_stats] could not compute direction stats: {exc}")
        return ""
