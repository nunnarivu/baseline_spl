'''
test_gaussian.py

Round 2's pre-flight. Three questions, in the order they have to be answered:

  1. Does the Gaussian executor reproduce SPL's? If `shifted` and `restored` do not match
     `executor._predict_focus` and `assign_focus`, the fairness claim is false and every
     number downstream is measuring something else.
  2. Are the ground-truth programs accepted? A threshold that rejects an oracle is wrong;
     the oracle is not.
  3. Does the threshold SEPARATE oracles from near-misses? This is the one that decides
     whether a soft evaluator is usable at all. Exact match could not accept a wrong
     program; a likelihood can, and if no threshold admits every oracle while excluding the
     known near-misses, the approach fails and that is the result.

Question 3 is why this file exists rather than a handful of assertions bolted onto
test_search. The plan makes it an explicit stop-and-report gate.
'''

from __future__ import annotations

import math

import pytest

from baseline_spl.symbolic import evaluate, gaussian, ir, oracles, srn
from baseline_spl.symbolic.bridge import from_term as to_program
from baseline_spl.symbolic.search import SearchTask

PITCH = (0.109, 0.05)


def _table():
    return srn.analytic_table(PITCH)


def _task(name, term, params, *, noise=0.0, table=None, seed=0):
    '''A continuous task whose targets are what `term` itself predicts.

    Generating the targets from the oracle makes the test independent of the dataset while
    still exercising the real scoring path. `noise` displaces every placement by a fixed
    fraction of one sigma, which is what turns "does it accept the exact answer" into "does
    it accept an answer that is right but observed imperfectly".
    '''
    import random

    rng = random.Random(seed)
    table = table or _table()
    examples = []
    for n in params:
        fn = to_program(term).evaluate([])
        state = gaussian.run(fn, n, table, (), False)
        target = []
        for mu, var in state.gaussians:
            if noise:
                target.append(tuple(mu[i] + noise * math.sqrt(var[i]) *
                                    rng.choice((-1.0, 1.0)) for i in range(3)))
            else:
                target.append(mu)
        examples.append((n, target))
    return SearchTask(name=name, examples=examples, observation_mode="continuous",
                      srn_tables=[table] * len(params))


# --------------------------------------------------------------------------------------- #
# 1. The executor
# --------------------------------------------------------------------------------------- #

def test_shift_accumulates_mean_and_variance_like_a_kalman_step():
    '''SPL's _predict_focus returns (cfm + mu_m, cfv + Q). Nothing else.'''
    table = _table()
    mu_d, var_d = table["right"]
    state = gaussian.initial(table)
    start_var = state.var

    once = state.shifted("right")
    assert once.mu == pytest.approx(mu_d)
    assert once.var == pytest.approx(tuple(start_var[i] + var_d[i] for i in range(3)))

    twice = once.shifted("right")
    assert twice.mu == pytest.approx(tuple(2 * mu_d[i] for i in range(3)))
    # Variance compounds -- the property the lattice representation discarded.
    assert twice.var == pytest.approx(tuple(start_var[i] + 2 * var_d[i] for i in range(3)))
    assert twice.var[1] > once.var[1] > start_var[1]


def test_compounding_variance_reaches_half_a_cell_within_one_row():
    '''The measurement that motivates the whole round: after four shifts the along-axis
    sigma exceeds half a lattice cell, so a placement's position is genuinely uncertain.'''
    table = srn.analytic_table(PITCH, sigma_fraction=0.28)
    state = gaussian.initial(table)
    for _ in range(4):
        state = state.shifted("right")
    sigma_y = math.sqrt(state.var[1])
    assert sigma_y > 0.5 * PITCH[0]


def test_saved_resets_variance_the_way_assign_focus_does():
    '''assign_focus writes cov_scale=0.0001 regardless of how uncertain the focus had become.
    That makes `saved` the DSL's only variance-resetting operation.'''
    table = _table()
    state = gaussian.initial(table)
    for _ in range(5):
        state = state.shifted("top")
    assert state.var[2] > gaussian.RESET_VAR[2]

    restored = state.restored(gaussian.initial(table))
    assert restored.var == pytest.approx(gaussian.RESET_VAR)


def test_saved_resync_snaps_to_the_observation_not_the_prediction():
    '''SPL resolves assign_focus(object_id=k) to mesh_centroid of the *placed* block, so the
    faithful restore uses what was observed. resync=False is the stricter alternative.'''
    table = _table()
    observed = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]

    at_save = gaussian.initial(table, observed, resync=True).placed()
    # A body that places nothing has no block to focus on -- the lowered Python would raise
    # IndexError -- so the restore falls back to the saved mean rather than snapping to an
    # observation this program never produced.
    drifted = at_save.shifted("right").shifted("right")
    assert drifted.restored(at_save).mu == pytest.approx(at_save.mu)

    # With a placement made by the body, the restore takes the observed centroid.
    body = at_save.shifted("top").placed()
    assert body.restored(at_save).mu == pytest.approx(observed[1])

    strict = gaussian.GaussianState(table, observations=tuple(observed), resync=False)
    strict = strict.placed()
    strict_body = strict.shifted("top").placed()
    assert strict_body.restored(strict).mu == pytest.approx(strict.mu)


def test_the_two_executors_agree_on_placement_count_for_every_oracle():
    '''The Gaussian executor is the lattice one with a different state. If they ever disagree
    on how many blocks a program places, one of them has the semantics wrong.'''
    table = _table()
    for name, term in oracles.ORACLES.items():
        for n in (1, 3, 5):
            cells = ir.positions(term, n)
            fn = to_program(term).evaluate([])
            produced = gaussian.run(fn, n, table, (), False).positions
            assert len(produced) == len(cells), f"{name} at n={n}"


def test_gaussian_means_track_the_lattice_up_to_pitch():
    '''With an analytic table the mean of the k-th placement is its lattice cell times the
    pitch. This is what makes the continuous and lattice runs comparable rather than two
    unrelated experiments.'''
    table = _table()
    scale = (PITCH[0], PITCH[0], PITCH[1])
    for name in ("row", "tower", "staircase", "diagonal_45"):
        term = oracles.ORACLES[name]
        fn = to_program(term).evaluate([])
        produced = gaussian.run(fn, 4, table, (), False).positions
        for cell, mu in zip(ir.positions(term, 4), produced):
            expected = tuple(cell[i] * scale[i] for i in range(3))
            assert mu == pytest.approx(expected, abs=1e-9), name


# --------------------------------------------------------------------------------------- #
# 2. Oracles are accepted
# --------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("criterion", ["mahalanobis", "penalised_loglik"])
def test_every_oracle_is_accepted_when_observed_exactly(criterion):
    ev = evaluate.LikelihoodEvaluator(criterion=criterion, resync=False)
    for name, term in oracles.ORACLES.items():
        task = _task(name, term, (3, 5))
        value = ev.score(to_program(term), task)
        assert value is not None, name
        assert ev.accepts(value), f"{name}: {value}"


@pytest.mark.parametrize("criterion", ["mahalanobis", "penalised_loglik"])
def test_every_oracle_survives_realistic_observation_noise(criterion):
    '''One sigma of displacement on every placement must not reject the right program --
    otherwise the threshold is measuring noise rather than correctness.'''
    ev = evaluate.LikelihoodEvaluator(criterion=criterion, resync=False)
    for name, term in oracles.ORACLES.items():
        task = _task(name, term, (3, 5), noise=1.0, seed=hash(name) % 1000)
        value = ev.score(to_program(term), task)
        assert ev.accepts(value), f"{name}: {value}"


def test_tolerance_evaluator_accepts_oracles_and_rejects_a_shifted_program():
    ev = evaluate.ToleranceEvaluator(epsilon=0.03, resync=False)
    term = oracles.ORACLES["row"]
    assert ev.accepts(ev.score(to_program(term), _task("row", term, (3, 5))))

    # A row built in the opposite direction: same shape, wrong axis sign.
    wrong = oracles.ORACLES["inverted_row"]
    assert not ev.accepts(ev.score(to_program(wrong), _task("row", term, (3, 5))))


def test_wrong_placement_count_is_never_accepted():
    '''The check exact match got for free. Without it a program placing three blocks scores
    well against a five-block demonstration by matching a prefix.'''
    for ev in (evaluate.LikelihoodEvaluator(resync=False),
               evaluate.ToleranceEvaluator(resync=False)):
        term, short = oracles.ORACLES["row"], oracles.ORACLES["tower"]
        task = _task("row", term, (5,))
        # `staircase` places n(n+1)/2 blocks against row's n -- a guaranteed count mismatch.
        value = ev.score(to_program(oracles.ORACLES["staircase"]), task)
        assert value is not None and value >= ev.miss_value
        assert not ev.accepts(value)
        assert ev.score(to_program(short), task) is not None


# --------------------------------------------------------------------------------------- #
# 3. Separation -- the gate
# --------------------------------------------------------------------------------------- #

def _confusable_pairs():
    '''(target concept, wrong program) pairs that place the SAME number of blocks.

    Count mismatch is a trivial rejection, so a separation test built on it would prove
    nothing. These pairs are the real question: programs the evaluator must tell apart on
    geometry alone.
    '''
    return [("row", "inverted_row"), ("row", "column"), ("row", "tower"),
            ("column", "inverted_column"), ("tower", "row"),
            ("diagonal_45", "diagonal_135"), ("diagonal_45", "row"),
            ("staircase", "inverted_staircase")]


@pytest.mark.parametrize("criterion", ["mahalanobis", "penalised_loglik"])
def test_threshold_separates_oracles_from_confusable_programs(criterion):
    '''The decisive test. Every oracle accepted, every same-length wrong program rejected,
    at the shipped default threshold.'''
    ev = evaluate.LikelihoodEvaluator(criterion=criterion, resync=False)

    for name, term in oracles.ORACLES.items():
        task = _task(name, term, (3, 5), noise=1.0, seed=7)
        assert ev.accepts(ev.score(to_program(term), task)), f"oracle {name} rejected"

    for target, wrong in _confusable_pairs():
        task = _task(target, oracles.ORACLES[target], (3, 5))
        value = ev.score(to_program(oracles.ORACLES[wrong]), task)
        assert not ev.accepts(value), f"{wrong} was accepted as {target} ({value})"


def test_separation_sweep_reports_a_usable_window(capsys):
    '''Print accept/reject counts across thresholds, so the default is chosen from a
    measurement rather than tuned until the results look good. Fails only if NO threshold
    separates the two populations -- which would mean the soft evaluator is unusable.'''
    rows = []
    usable = []
    for tau in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0):
        ev = evaluate.LikelihoodEvaluator(criterion="mahalanobis", tau=tau, resync=False)
        accepted_oracles = sum(
            ev.accepts(ev.score(to_program(term), _task(n, term, (3, 5), noise=1.0, seed=7)))
            for n, term in oracles.ORACLES.items())
        accepted_wrong = sum(
            ev.accepts(ev.score(to_program(oracles.ORACLES[w]),
                                _task(t, oracles.ORACLES[t], (3, 5))))
            for t, w in _confusable_pairs())
        rows.append((tau, accepted_oracles, len(oracles.ORACLES), accepted_wrong,
                     len(_confusable_pairs())))
        if accepted_oracles == len(oracles.ORACLES) and accepted_wrong == 0:
            usable.append(tau)

    with capsys.disabled():
        print("\n  tau   oracles accepted   confusable accepted")
        for tau, ok, total, bad, bad_total in rows:
            print(f"  {tau:5.1f}   {ok:2d}/{total:<14d} {bad:2d}/{bad_total}")
        print(f"  usable thresholds: {usable}")

    assert usable, ("no threshold accepts every oracle while rejecting every confusable "
                    "program; the probabilistic evaluator is not usable as configured")
    assert 2.0 in usable, f"the shipped default tau=2.0 is outside the usable window {usable}"


def test_the_default_threshold_clears_the_real_near_miss_margin():
    '''Finding M, pinned. This sweep's synthetic negatives put the usable window at [2, 5],
    but on the real 8-concept run the correct programs topped out at 0.62 sigma while the
    nearest WRONG program sat at 3.11. So the upper half of the synthetic window is unsafe:
    tau=5 would have accepted staircase's near-miss as solved at program_accuracy 0.0.

    The lesson generalises past this project -- a threshold calibrated on synthetic negatives
    is trustworthy only at its strict end, because invented wrong answers are wrong in
    obvious ways while the ones a real search produces are the hard cases by construction.
    '''
    from baseline_spl.configs.default import DreamCoderConfig

    worst_correct, nearest_wrong = 0.62, 3.11        # measured, symbolic_sweep8 continuous
    tau = DreamCoderConfig.accept_tau
    assert worst_correct < tau < nearest_wrong, (
        f"tau={tau} does not separate the measured populations "
        f"({worst_correct} correct / {nearest_wrong} wrong)")
    # And it is not hugging either edge -- 3.0 was inside the window but only by 0.11.
    assert tau - worst_correct > 1.0 and nearest_wrong - tau > 1.0


# --------------------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------------------- #

def test_invalid_observation_and_evaluator_pairs_are_rejected_loudly():
    for bad in [("lattice", "tolerance"), ("lattice", "srn_likelihood"),
                ("continuous", "exact")]:
        with pytest.raises(ValueError):
            evaluate.validate(*bad)
    for good in evaluate.VALID_COMBINATIONS:
        evaluate.validate(*good)


def test_build_returns_the_configured_evaluator():
    class Cfg:
        observation_mode = "continuous"
        evaluator = "srn_likelihood"
        accept_criterion = "penalised_loglik"
        accept_margin = 4.5

    ev = evaluate.build(Cfg)
    assert isinstance(ev, evaluate.LikelihoodEvaluator)
    assert ev.criterion == "penalised_loglik" and ev.threshold == 4.5

    class Lattice:
        observation_mode = "lattice"
        evaluator = "exact"

    assert isinstance(evaluate.build(Lattice), evaluate.ExactEvaluator)


def test_mahalanobis_and_loglik_thresholds_are_calibrated():
    '''A Gaussian's log-density falls by d**2/2 at Mahalanobis distance d, so tau=3 and
    margin=4.5 are the same threshold. If this drifts, the two knobs stop being comparable.'''
    var = (0.001, 0.002, 0.0005)
    offset = tuple(3.0 * math.sqrt(v) for v in var)   # exactly 3 sigma away, per axis...
    # ...which is sqrt(3) * 3 sigma in 3-D, so scale back to a true 3-sigma displacement.
    scale = 3.0 / gaussian.mahalanobis(offset, (0.0, 0.0, 0.0), var)
    offset = tuple(o * scale for o in offset)

    assert gaussian.mahalanobis(offset, (0.0, 0.0, 0.0), var) == pytest.approx(3.0)
    shortfall = (gaussian.penalised_ceiling(var)
                 - gaussian.penalised_score(offset, (0.0, 0.0, 0.0), var))
    assert shortfall == pytest.approx(4.5, abs=1e-6)
