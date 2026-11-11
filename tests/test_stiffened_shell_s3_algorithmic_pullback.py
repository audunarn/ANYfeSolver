"""Fixed-predecessor mechanics checks for the experimental S3 J2 pullback."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_variational_shell import deformation, response as elastic_response
from anyfesolver._stiffened_shell_s3_algorithmic_pullback import (
    local_coordinates, response,
)
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


def _fixture():
    model = FEModel("experimental-s3-algorithmic-pullback")
    material = model.add_material("shell", 210e9, .3, density=7850.,
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=80e6, sigma_yield=85e6, sigma_yield_2=100e6,
            eps_p_y1=.002, eps_p_y2=.02, K=280e6, n=.18))
    reference = np.array(((0., 0., 0.), (1.2, .05, 0.), (.2, .95, 0.)))
    for node, point in enumerate(reference, 1):
        model.add_node(node, *point)
    element = create_shell_element(1, [1, 2, 3], "shell",
        formulation="e4-pl-s3-v2d", thickness=.018,
        reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    accepted_u = np.zeros(18)
    accepted_u[6], accepted_u[12] = .008, -.002
    _, _, candidate = response(model, element, accepted_u, virgin, 3)
    assert np.max(candidate["alpha"]) > 0.
    accepted_local = local_coordinates(model, element, accepted_u)
    accepted = element.seal_solver_integrated_nonlinear_state(
        model.mesh, material, candidate, 3, accepted_local,
        kinematics="LINEAR")
    trial = accepted_u.copy()
    trial[6] += .0002
    trial[12] -= .00005
    trial[9:12] = (.003, -.002, .001)
    return model, material, element, reference, accepted, trial


def _independent_local(reference, u):
    coordinates = u.reshape(3, 6)
    current = reference+coordinates[:, :3]
    fit = Rotation.align_vectors(current-current.mean(axis=0),
                                 reference-reference.mean(axis=0))[0]
    local = np.empty((3, 6))
    local[:, :3] = (fit.inv().apply(current-current.mean(axis=0))-
                    (reference-reference.mean(axis=0)))
    local[:, 3:] = (fit.inv()*Rotation.from_rotvec(coordinates[:, 3:])).as_rotvec()
    return local.ravel()


def _moved(reference, u, motion):
    moved = u.reshape(3, 6).copy()
    moved[:, :3] = (motion.apply(reference+moved[:, :3])+
                    np.array((.1, -.2, .3))-reference)
    moved[:, 3:] = (motion*Rotation.from_rotvec(moved[:, 3:])).as_rotvec()
    return moved.ravel()


def test_yielded_pullback_preserves_rigid_motion_and_local_material_oracle():
    model, material, element, reference, origin, trial = _fixture()
    before = element.serialize_nonlinear_state(model.mesh, material, origin, 3)
    motion = Rotation.from_rotvec((.7, -.4, .5))
    rotated = _moved(reference, trial, motion)
    force, _, candidate = response(model, element, trial, origin, 3)
    rotated_force, _, rotated_candidate = response(
        model, element, rotated, origin, 3)
    np.testing.assert_allclose(rotated_force.reshape(-1, 3),
        motion.apply(force.reshape(-1, 3)), rtol=1e-9, atol=2e-6)
    oracle_local = _independent_local(reference, rotated)
    np.testing.assert_allclose(local_coordinates(model, element, rotated),
                               oracle_local, rtol=1e-10, atol=2e-14)
    _, _, oracle = element.compute_nonlinear_response(
        model.mesh, material, oracle_local, origin, 3, True)
    for name in ("layer_strain", "layer_stress", "plastic_strain", "alpha",
                 "station_generalized_resultant"):
        for actual in (candidate[name], rotated_candidate[name]):
            np.testing.assert_allclose(actual, oracle[name], rtol=1e-9, atol=2e-6)
    assert element.serialize_nonlinear_state(model.mesh, material, origin, 3) == before


def test_yielded_algorithmic_tangent_uses_same_accepted_predecessor():
    model, material, element, reference, origin, trial = _fixture()
    before = element.serialize_nonlinear_state(model.mesh, material, origin, 3)
    trial = _moved(reference, trial, Rotation.from_rotvec((.7, -.4, .5)))
    _, tangent, candidate = response(model, element, trial, origin, 3)
    active = np.asarray(candidate["alpha"]) > np.asarray(origin["alpha"])
    assert active.any()
    direction = np.random.default_rng(9129).normal(size=18)
    predicted = tangent@direction
    errors = []
    for step in (2e-7, 1e-7):
        positive, _, plus_state = response(model, element, trial+step*direction, origin, 3)
        negative, _, minus_state = response(model, element, trial-step*direction, origin, 3)
        for state in (plus_state, minus_state):
            np.testing.assert_array_equal(
                np.asarray(state["alpha"]) > np.asarray(origin["alpha"]), active)
        observed = (positive-negative)/(2*step)
        errors.append(np.linalg.norm(observed-predicted)/np.linalg.norm(predicted))
    assert max(errors) <= 1e-5, errors
    assert element.serialize_nonlinear_state(model.mesh, material, origin, 3) == before


def test_algorithmic_route_preserves_existing_elastic_guard():
    model, _, element, _, origin, trial = _fixture()
    with pytest.raises(ValueError, match="owned elastic section"):
        elastic_response(model, element, trial, origin, 3)
    with pytest.raises(ValueError, match="accepted predecessor"):
        response(model, element, trial, None, 3)


def test_stable_map_preserves_complete_finite_chart_derivatives():
    _, _, _, reference, _, trial = _fixture()
    moved = _moved(reference, trial, Rotation.from_rotvec((.7, -.4, .5)))
    original = deformation(reference, moved)
    stable = deformation(reference, moved, stable_translation=True)
    for actual, expected in zip(stable, original):
        # Compare the whole operator; individual zero/near-zero Hessian entries
        # are not a scale for accumulated eigensolve roundoff.
        assert np.linalg.norm(actual-expected)/max(1., np.linalg.norm(expected)) < 1e-12


def test_small_elastic_trial_is_insensitive_to_nanometre_common_translation():
    model, material, element, _, _, _ = _fixture()
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    trial = np.zeros(18)
    trial[6], trial[12], trial[13] = 1e-5, -2e-6, 3e-6
    moved = trial.reshape(3, 6).copy()
    moved[:, :3] += (1e-9, -2e-9, 3e-9)
    force, _, candidate = response(model, element, trial, virgin, 3)
    moved_force, _, moved_candidate = response(
        model, element, moved.ravel(), virgin, 3)
    np.testing.assert_allclose(moved_force, force, rtol=1e-11, atol=1e-9)
    np.testing.assert_array_equal(candidate["alpha"], moved_candidate["alpha"])
