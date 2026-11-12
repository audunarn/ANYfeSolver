"""Cancellation-resistant flat Q4 point arithmetic; no gate acceptance claim."""

from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from anyfesolver import corotational as cr, stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_ge_checkpoint_binary import encode
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import POLICY
from scripts.experimental_stiffened_ge_multi_member_panel import make_definition, actions_for
from test_corotational import _single_shell_model, _rigid_rotation_field
from test_stiffened_shell_ge_experimental import _panel


def test_rotation_difference_resolves_quadratic_terms_and_matches_so3():
    rng = np.random.default_rng(516)
    for angle in (*np.logspace(-14, -1, 14), .7, 1., 1.1, 2., np.pi-1e-9):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        matrix = Rotation.from_rotvec(angle*axis).as_matrix()
        result = cr._stable_shell_rotation_difference(matrix)
        np.testing.assert_allclose(result, matrix.T-np.eye(3), rtol=2e-14, atol=8e-16)
        if angle <= .7:
            # Independent Rodrigues coefficients, evaluated without I subtraction.
            hat = cr._skew(axis)
            expected = (-np.sin(angle)*hat + 2*np.sin(angle/2)**2*(hat@hat))
            np.testing.assert_allclose(np.diag(result), np.diag(expected),
                                       rtol=3e-14, atol=1e-40)
    tiny = Rotation.from_rotvec((1e-10, 0., 0.)).as_matrix()
    assert (tiny.T-np.eye(3))[1, 1] == 0.
    assert cr._stable_shell_rotation_difference(tiny)[1, 1] == pytest.approx(
        -5e-21, rel=1e-14, abs=1e-35)


@pytest.mark.parametrize("matrix", (np.diag((1.01, 1., 1.)),
    np.diag((-1., 1., 1.)), np.full((3, 3), np.nan)))
def test_nonrotation_uses_historical_subtraction(matrix):
    np.testing.assert_array_equal(cr._stable_shell_rotation_difference(matrix),
                                  matrix.T-np.eye(3))


def test_moderate_stable_shell_tangent_and_default_arithmetic():
    model = _single_shell_model()
    element = model.mesh.elements[1]
    u = _rigid_rotation_field(model, 35., (.2, .4, 1.))
    u[-6] += .003
    u[-4] += .007
    u[-2] -= .015
    kwargs = dict(tangent_mode="consistent", centered_shell_reference=True)
    default = cr.corotational_element_response(model, 1, element, u, True, **kwargs)
    old = cr.corotational_element_response(model, 1, element, u, True,
                                         stable_shell_rotation=False, **kwargs)
    for a, b in zip(default[:2], old[:2]):
        np.testing.assert_array_equal(a, b)
    force, tangent, _ = cr.corotational_element_response(
        model, 1, element, u, True, stable_shell_rotation=True, **kwargs)
    assert np.linalg.norm(force-old[0]) <= 1e-10*np.linalg.norm(old[0])
    direction = np.random.default_rng(18).normal(size=u.size)
    h = 2e-7
    fp = cr.corotational_element_response(model, 1, element, u+h*direction,
        False, stable_shell_rotation=True, **kwargs)[0]
    fm = cr.corotational_element_response(model, 1, element, u-h*direction,
        False, stable_shell_rotation=True, **kwargs)[0]
    numeric = (fp-fm)/(2*h)
    assert np.linalg.norm(numeric-tangent@direction)/np.linalg.norm(numeric) < 1e-6


def test_small_multistiffener_history_actual_update_and_exact_restart():
    with ExitStack() as cleanup:
        owner = ss.create_analysis(definition=make_definition(8, 4))
        cleanup.callback(owner.close)
        actions = actions_for(owner._definition)
        driver = FinitePointDriver(owner)
        pattern = driver._load(actions)
        r0, k0, *_ = driver._assemble(driver._state, .01,
            driver._shell_origins, driver._beam_origins, pattern=pattern)
        free = np.r_[owner._free, np.arange(owner._K.shape[0], len(r0))]
        direction = np.zeros_like(r0)
        direction[free] = spsolve(k0[free, :][:, free], -r0[free])
        accepted = owner.solve(actions, (.01, .02), max_iterations=8, max_backtracks=4)
        assert accepted.free_equilibrium_ratio <= 1e-8
        assert accepted.constraint_max <= 1e-9
        assert not accepted.production_qualified
        driver = owner._finite_driver
        state = driver._state
        r, k, *_ = driver._assemble(state, .02, driver._shell_origins,
                                   driver._beam_origins, pattern=pattern)
        expected = k@direction
        weights = owner._coordinate_scale[owner._free]
        for h in (.1, .03):
            rp = driver._assemble(driver._advance(state, direction, h), .02,
                driver._shell_origins, driver._beam_origins, pattern=pattern)[0]
            rm = driver._assemble(driver._advance(state, direction, -h), .02,
                driver._shell_origins, driver._beam_origins, pattern=pattern)[0]
            error = (rp-rm)/(2*h)-expected
            assert np.linalg.norm(error[owner._free]*weights)/np.linalg.norm(
                expected[owner._free]*weights) <= 1e-6
            assert np.linalg.norm(error[owner._K.shape[0]:]/owner._constraint_scale) <= 1e-9
        saved = owner.checkpoint()
        assert b"ELASTIC_POINT_STABLE_Q4" in saved[:8192]
        resumed = ss.create_analysis(definition=owner._definition)
        cleanup.callback(resumed.close)
        resumed.restore(saved)
        assert resumed._finite_driver._stable_q4
        assert resumed.checkpoint() == saved
        continued = owner.solve(actions, (.03,), max_iterations=8, max_backtracks=4)
        restarted = resumed.solve(actions, (.03,), max_iterations=8, max_backtracks=4)
        assert continued.accepted_factors == (.01, .02, .03)
        for name in ("shell_displacement", "beam_positions", "beam_frames",
                     "cell_frames", "cell_resultants", "multipliers", "support_reactions"):
            np.testing.assert_array_equal(getattr(continued, name), getattr(restarted, name))
        assert owner.checkpoint() == resumed.checkpoint()


@pytest.mark.parametrize("kind", ("algorithmic", "mixed_s3", "all_s3", "no_points"))
def test_other_pullbacks_keep_their_arithmetic(kind):
    definition = _panel()
    if kind == "algorithmic":
        definition = replace(definition, q4_pullback_policy=POLICY)
    elif kind == "no_points":
        definition = replace(definition, points=())
        with pytest.raises(ValueError, match="explicit native connection required"):
            ss.create_analysis(definition=definition)
        return
    else:
        definition = replace(definition, shell=replace(definition.shell,
            q4_elements=((2, 3, 6, 5),) if kind == "mixed_s3" else (),
            s3_elements=((1, 2, 5), (1, 5, 4)) if kind == "mixed_s3" else
                         ((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
    owner = ss.create_analysis(definition=definition)
    try:
        driver = FinitePointDriver(owner)
        assert not driver._stable_q4
        with pytest.raises(ValueError, match="stable Q4 checkpoint"):
            driver.restore({}, stable_q4=True)
        assert owner._generation == 0
        state = driver._state
        forged = encode("finite_stable_q4", owner._identity, 1,
            (("q", state.q), ("frames", state.frames), ("cells", state.cells),
             ("resultants", state.resultants), ("multipliers", state.multipliers),
             ("pattern", np.zeros_like(state.q)), ("accepted_factors", np.array([1.]))))
        with pytest.raises(ValueError, match="stable Q4 checkpoint"):
            owner.restore(forged)
        assert owner._generation == 0 and owner._finite_driver is None
    finally:
        owner.close()


@pytest.mark.parametrize("material_kind", (False, True))
def test_stable_chart_snapshot_and_cancelled_successor_preserve_prefix(material_kind):
    from anyfesolver.control import CancellationToken, SolveCancelled
    with ExitStack() as cleanup:
        owner = ss.create_analysis(definition=_panel())
        cleanup.callback(owner.close)
        actions = (ss.NodalAction("beam", 105, (0., 0., 1000., 0., 0., 0.)),)
        factors = (.5, 1., .5) if material_kind else (.5, 1.)
        first = owner.solve(actions, factors)
        owner.rebase_rotations(expected_generation=first.generation)
        saved = owner.checkpoint()
        assert (b"ANCHORED_GENERALIZED_POINT_STABLE_Q4" if material_kind else
                b"ANCHORED_ELASTIC_POINT_STABLE_Q4") in saved
        resumed = ss.create_analysis(definition=owner._definition)
        cleanup.callback(resumed.close)
        resumed.restore(saved)
        assert resumed.checkpoint() == saved
        token = CancellationToken()
        token.cancel()
        with pytest.raises(SolveCancelled):
            resumed.solve(actions, (2.,), cancellation_token=token)
        assert resumed.checkpoint() == saved
        a = owner.solve(actions, (2.,))
        b = resumed.solve(actions, (2.,))
        assert a.accepted_factors == b.accepted_factors == factors+(2.,)
        assert a.free_equilibrium_ratio <= 1e-8
        np.testing.assert_array_equal(a.shell_displacement, b.shell_displacement)
        assert owner.checkpoint() == resumed.checkpoint()


def test_previous_installed_checkpoint_retains_exact_legacy_continuation():
    data = Path(__file__).parent/"data"/"stiffened_shell_q4_legacy"
    raw = (data/"checkpoint.bin").read_bytes()
    identity = json.loads((data/"identity.json").read_text(encoding="utf-8"))
    assert sha256(raw).hexdigest() == identity["checkpoint_sha256"]
    assert sha256((data/"continued.npz").read_bytes()).hexdigest() == identity["continued_sha256"]
    expected = np.load(data/"continued.npz")
    owner = ss.create_analysis(definition=_panel())
    try:
        owner.restore(raw)
        assert not owner._finite_driver._stable_q4
        assert owner.checkpoint() == raw
        result = owner.solve((ss.NodalAction("beam", 105, (0., 0., 1000., 0., 0., 0.)),), (2.,))
        for name in expected.files:
            np.testing.assert_array_equal(getattr(result, name), expected[name])
        assert b"STABLE_Q4" not in owner.checkpoint()[:8192]
        with pytest.raises(ValueError, match="live finite driver"):
            owner._finite_driver.restore({}, stable_q4=True)
    finally:
        expected.close()
        owner.close()
