"""Development checks of the selected Q4 fixed-predecessor force map."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._stiffened_shell_q4_algorithmic_pullback import local_coordinates, response
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


@pytest.fixture(scope="module")
def yielded():
    model = FEModel("q4-full-pullback-development")
    material = model.add_material("shell", 210e9, .3, density=7850.,
        hardening_curve=DNVC208MaterialCurve(100e6, 105e6, 110e6, .005, .01, 400e6, .2))
    reference = np.array(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
    for node, point in enumerate(reference, 1):
        model.add_node(node, *point)
    element = create_shell_element(1, (1, 2, 3, 4), "shell", formulation="e4-pl",
                                  thickness=.02, reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    accepted = np.zeros(24)
    accepted[6] = accepted[12] = .004
    accepted[8], accepted[14] = .0004, .0007
    local = local_coordinates(model, element, accepted)
    _, _, candidate = element.compute_nonlinear_response(
        model.mesh, material, local, element.init_nonlinear_state(3), 3, True)
    origin = element.seal_committed_current_tangent_state(
        model.mesh, material, local, candidate, 3)
    assert np.max(origin["alpha"]) > 0.
    trial = accepted.copy()
    trial[[6, 12, 8, 14]] += (.0002, .00015, .00004, .00005)
    trial.reshape(4, 6)[:, 3:] = ((.003, -.002, .001), (-.001, .002, -.003),
                                (.001, .001, .002), (.002, -.001, .003))
    return model, material, element, reference, origin, trial


def _local(reference, u, anchors=None):
    """Independent SciPy best-fit pose, including owned physical anchors."""
    rows = u.reshape(-1, 6)
    current = reference+rows[:, :3]
    fit = Rotation.align_vectors(current-current.mean(0), reference-reference.mean(0))[0]
    frames = Rotation.from_rotvec(rows[:, 3:].copy()).as_matrix()
    if anchors is not None:
        frames = frames@anchors
    result = np.empty_like(rows)
    result[:, :3] = fit.inv().apply(current-current.mean(0))-(reference-reference.mean(0))
    result[:, 3:] = (fit.inv()*Rotation.from_matrix(frames)).as_rotvec()
    return result.ravel()


def _rigid(reference, u):
    motion = Rotation.from_rotvec((.7, -.4, .5))
    moved = u.reshape(-1, 6).copy()
    moved[:, :3] = motion.apply(reference+moved[:, :3])+(.1, -.2, .3)-reference
    moved[:, 3:] = (motion*Rotation.from_rotvec(moved[:, 3:])).as_rotvec()
    return motion, moved.ravel()


def test_q4_objective_local_oracle_and_fixed_predecessor(yielded):
    model, material, element, reference, origin, trial = yielded
    before = canonical_json_bytes(origin)
    motion, moved = _rigid(reference, trial)
    force, _, candidate = response(model, element, trial, origin, 3)
    rotated, _, rotated_candidate = response(model, element, moved, origin, 3)
    np.testing.assert_allclose(rotated.reshape(-1, 3), motion.apply(force.reshape(-1, 3)),
                               rtol=1e-9, atol=2e-6)
    independent = _local(reference, moved)
    np.testing.assert_allclose(local_coordinates(model, element, moved), independent,
                               rtol=1e-10, atol=2e-14)
    _, _, oracle = element.compute_nonlinear_response(
        model.mesh, material, independent, origin, 3, True)
    for name in ("layer_strain", "layer_stress", "plastic_strain", "alpha"):
        np.testing.assert_allclose(candidate[name], oracle[name], rtol=1e-9, atol=2e-6)
        np.testing.assert_allclose(rotated_candidate[name], oracle[name], rtol=1e-9, atol=2e-6)
    assert canonical_json_bytes(origin) == before


@pytest.mark.parametrize("anchored", [False, True])
def test_q4_smooth_active_branch_actual_chart_derivative_and_work(yielded, anchored):
    model, material, element, reference, origin, trial = yielded
    before = canonical_json_bytes(origin)
    _, trial = _rigid(reference, trial)
    anchors = None
    if anchored:
        anchors = Rotation.from_rotvec(trial.reshape(4, 6)[:, 3:].copy()).as_matrix()
        trial = trial.copy()
        trial.reshape(4, 6)[:, 3:] = ((.001, -.002, .003), (-.003, .002, .001),
                                     (.002, .001, -.002), (.001, -.001, .002))
    force, tangent, candidate = response(model, element, trial, origin, 3,
                                         rotation_anchors=anchors)
    active = np.asarray(candidate["alpha"]) > np.asarray(origin["alpha"])
    assert active.any()
    local = _local(reference, trial, anchors)
    local_force = element.compute_nonlinear_response(
        model.mesh, material, local, origin, 3, True)[0]
    for direction in (np.sin(np.arange(24)+.3), np.cos(.4+.73*np.arange(24))):
        direction /= np.linalg.norm(direction)
        predicted = tangent@direction
        errors = []
        for h in (2e-7, 1e-7):
            plus, _, a = response(model, element, trial+h*direction, origin, 3,
                                   rotation_anchors=anchors)
            minus, _, b = response(model, element, trial-h*direction, origin, 3,
                                    rotation_anchors=anchors)
            np.testing.assert_array_equal(np.asarray(a["alpha"]) > origin["alpha"], active)
            np.testing.assert_array_equal(np.asarray(b["alpha"]) > origin["alpha"], active)
            errors.append(np.linalg.norm((plus-minus)/(2*h)-predicted)/np.linalg.norm(predicted))
        assert max(errors) <= 1e-5, errors
        h = 1e-7
        local_velocity = (_local(reference, trial+h*direction, anchors)-
                          _local(reference, trial-h*direction, anchors))/(2*h)
        rows = trial.reshape(4, 6)
        spatial = direction.reshape(4, 6).copy()
        # Differentiate the actual independent rotation update; no production
        # Jacobian is used in this work-conjugacy oracle.
        plus = Rotation.from_rotvec(rows[:, 3:]+h*spatial[:, 3:])
        minus = Rotation.from_rotvec(rows[:, 3:]-h*spatial[:, 3:])
        spatial[:, 3:] = (plus*minus.inv()).as_rotvec()/(2*h)
        expected = float(local_force@local_velocity)
        actual = float(force@spatial.ravel())
        assert abs(actual-expected)/max(1., abs(actual), abs(expected)) <= 1e-8
    assert canonical_json_bytes(origin) == before


def test_q4_anchor_reset_keeps_physical_force_and_all_layer_state(yielded):
    model, _, element, _, origin, trial = yielded
    force, _, candidate = response(model, element, trial, origin, 3)
    anchors = Rotation.from_rotvec(trial.reshape(4, 6)[:, 3:].copy()).as_matrix()
    reset = trial.copy()
    reset.reshape(4, 6)[:, 3:] = 0.
    rebased, _, state = response(model, element, reset, origin, 3, rotation_anchors=anchors)
    np.testing.assert_allclose(rebased, force, rtol=1e-9, atol=2e-6)
    for name in ("layer_strain", "layer_stress", "plastic_strain", "alpha"):
        np.testing.assert_allclose(state[name], candidate[name], rtol=1e-9, atol=2e-6)
        assert not np.shares_memory(state[name], origin[name])


def test_q4_pullback_requires_owned_admitted_predecessor(yielded):
    model, _, element, _, _, trial = yielded
    with pytest.raises(ValueError, match="accepted predecessor"):
        response(model, element, trial, None, 3)
