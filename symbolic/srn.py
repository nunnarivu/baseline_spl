'''
srn.py

The placement primitive's per-direction Gaussian, as a small lookup table.

SPL does not move a focus by a fixed lattice step. `executor._predict_focus` runs the Spatial
Relation Network over (reference-object voxel, direction) and returns a diagonal Gaussian; the
focus is then updated as a Kalman prediction, `mu += mu_d` and `cov += Q_d`, so uncertainty
compounds along a program. That is the model B3's evaluator has to share if the comparison
against SPL is to be a comparison of methods rather than of representations.

Why this is cheap enough to sit inside enumeration
--------------------------------------------------
`_get_ref_voxel(state, 0)` always voxelizes object 0 and caches the result, so for one
demonstration the SRN has exactly five distinct outputs -- one per direction. Extract them
once per demo and the rest of the executor is arithmetic. Measured on row/tower/staircase the
five were identical across concepts, because every demo in this dataset shares a reference
asset; the table is still built per demo, since `primitive_stats._agree` exists precisely
because they need not be.

Measured values (metres), for reference:

    left    mu=(+0.0005,+0.1088,+0.0006)   sigma=(0.0100, 0.0339, 0.0067)
    right   mu=(+0.0002,-0.1106,+0.0006)   sigma=(0.0108, 0.0300, 0.0081)
    front   mu=(-0.1025,+0.0003,+0.0006)   sigma=(0.0293, 0.0102, 0.0067)
    behind  mu=(+0.1116,-0.0007,+0.0017)   sigma=(0.0278, 0.0111, 0.0082)
    top     mu=(+0.0007,-0.0003,+0.0516)   sigma=(0.0067, 0.0067, 0.0067)

The along-axis sigma is 0.030 m against a 0.109 m pitch -- 28% of a lattice cell -- and it
compounds, so a row of five reaches sigma_y ~ 0.061 m by its last block, over half a cell.
This is the uncertainty the lattice representation was quietly discarding.

Keys are lower-case, matching what `bridge` passes to `lattice.shift` (the primitive's *name*
is upper-case, its *value* is not).
'''

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]
# {direction: (mean, variance)}. Diagonal throughout -- see gaussian.py for why that matters.
Table = Dict[str, Tuple[Vec, Vec]]

# Fraction of a lattice step used as sigma when the SRN is unavailable. Set from the measured
# along-axis value above (0.0300 / 0.109) so an analytic table is a stand-in for the real one
# rather than an arbitrary choice.
DEFAULT_SIGMA_FRACTION = 0.28


def _to_table(stats: Dict[str, Tuple]) -> Table:
    '''{DIR: (mu, sigma)} from primitive_stats -> {dir: (mu, sigma**2)}.'''
    table: Table = {}
    for name, (mu, sigma) in stats.items():
        mu_t = tuple(float(v) for v in mu)
        var_t = tuple(float(v) ** 2 for v in sigma)
        table[name.lower()] = (mu_t, var_t)
    return table


def demo_table(executor, demo: dict) -> Table:
    '''The five (mean, variance) pairs for one demonstration, from the real SRN.'''
    from baseline_spl.common.primitive_stats import direction_stats

    entries = direction_stats(executor, [demo])
    if not entries:
        raise ValueError(f"no SRN statistics for demo {demo.get('demo_id')}")
    return _to_table(entries[0]["stats"])


def demo_tables(executor, demos: Sequence[dict]) -> list:
    '''One table per demonstration, in order.'''
    from baseline_spl.common.primitive_stats import direction_stats

    return [_to_table(entry["stats"]) for entry in direction_stats(executor, list(demos))]


def analytic_table(pitch: Tuple[float, float],
                   sigma_fraction: float = DEFAULT_SIGMA_FRACTION) -> Table:
    '''A table derived from the measured lattice pitch instead of the SRN.

    Two uses, both legitimate: unit tests that must not depend on a network checkpoint, and
    the `tolerance` evaluator, which wants the means but has no use for the variances. It is
    NOT a substitute for the real table in a scored run -- the SRN's sigma is anisotropic
    (0.030 along the direction of travel, 0.010 across it) and this is not.
    '''
    from baseline_spl.symbolic.lattice import DIRECTIONS

    horizontal, vertical = pitch
    table: Table = {}
    for name, step in DIRECTIONS.items():
        scale = (horizontal, horizontal, vertical)
        mu = tuple(float(step[i] * scale[i]) for i in range(3))
        var = tuple(float((sigma_fraction * scale[i]) ** 2) for i in range(3))
        table[name.lower()] = (mu, var)
    return table


def table_for(executor, demo: dict, pitch: Optional[Tuple[float, float]] = None) -> Table:
    '''The real table when the SRN is reachable, an analytic one otherwise.

    Falling back rather than raising is deliberate: a missing primitive checkpoint should
    degrade the evaluator, not end the run. It is logged loudly by the caller, because a run
    that silently swapped its noise model would be the exact failure this codebase keeps
    producing (see the plan's Trap 2).
    '''
    try:
        return demo_table(executor, demo)
    except Exception as exc:  # noqa: BLE001
        if pitch is None:
            raise
        import warnings

        warnings.warn(f"[srn] falling back to an analytic table for demo "
                      f"{demo.get('demo_id')}: {type(exc).__name__}: {exc}")
        return analytic_table(pitch)
