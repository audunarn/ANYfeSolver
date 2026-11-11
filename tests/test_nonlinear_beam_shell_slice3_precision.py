"""Beam force precision with the unchanged objective work-conjugate contract."""
from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import FEModel
from anyfesolver.elements import BeamElement
import anyfesolver.corotational as cr


def _beam(origin):
    model = FEModel("beam_precision")
    model.add_material("elastic", 210e9, 0.)
    model.add_node(1, origin, 0., .002)
    model.add_node(2, origin+.0625, 0., .002)
    element = BeamElement(1, [1, 2], "elastic", dict(
        area=2e-5, Iy=.01*.002**3/12, Iz=.002*.01**3/12,
        J=1e-11, b=.01, h=.002))
    model.add_element(1, element)
    return model, element


@pytest.mark.parametrize("origin", [0., .75, 1e6])
def test_beam_retains_axial_force_below_absolute_coordinate_ulp(origin):
    model, element = _beam(origin)
    extension = 2.**-56
    u = np.zeros((2, 6))
    u[:, 0] = [-extension/2, extension/2]
    before = u.copy()
    expected = 210e9*2e-5*extension/.0625
    force, _, _ = cr.corotational_work_conjugate_response(model, 1, element, u.ravel())
    np.testing.assert_array_equal(u, before)
    np.testing.assert_allclose(force[[0, 6]], [-expected, expected], rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.delete(force, [0, 6]), 0., rtol=0., atol=1e-22)


def test_beam_local_geometry_and_force_are_translation_invariant():
    u = np.array([[0., 0., 0., .02, -.015, .01],
                  [2.**-18, 2.**-10, -2.**-11, .01, -.02, .03]])
    outputs = []
    for origin, translation in ((0., np.zeros(3)), (1e6, np.array([.5, -.25, .125]))):
        model, element = _beam(origin)
        uu = u.copy()
        uu[:, :3] += translation
        preimage = uu.copy()
        reference = cr._corotational_cache(model)[1]
        coordinates_before = reference.coordinates.copy()
        ud, R, geometry = cr._corotational_deformation_state(reference, element, uu.ravel())
        force, tangent, _ = cr.corotational_work_conjugate_response(model, 1, element, uu.ravel())
        np.testing.assert_array_equal(uu, preimage)
        np.testing.assert_array_equal(reference.coordinates, coordinates_before)
        outputs.append((ud, R, geometry, force, tangent))
    for first, second in zip(*outputs):
        np.testing.assert_array_equal(first, second)


def test_centered_beam_pullback_and_force_tangent_differentiate_same_state():
    model, element = _beam(1e6)
    reference = cr._corotational_cache(model)[1]
    u = np.array([[0., 0., 0., .02, -.015, .01],
                  [2.**-18, 2.**-10, -2.**-11, .01, -.02, .03]]).ravel()
    before = u.copy()
    ud, R, geometry = cr._corotational_deformation_state(reference, element, u)
    D, U, G = cr._pullback_jacobian_factors(reference, element, geometry, u.reshape(2, 6)[:, 3:], R)
    B = D+U@G
    force, tangent, _ = cr.corotational_work_conjugate_response(model, 1, element, u)
    numerical_B, numerical_K = np.zeros((12, 12)), np.zeros((12, 12))
    h = 1e-7
    for column in range(12):
        plus, minus = u.copy(), u.copy()
        plus[column] += h
        minus[column] -= h
        dp = cr._corotational_deformation_state(reference, element, plus)[0]
        dm = cr._corotational_deformation_state(reference, element, minus)[0]
        fp = cr.corotational_work_conjugate_response(model, 1, element, plus, tangent=False)[0]
        fm = cr.corotational_work_conjugate_response(model, 1, element, minus, tangent=False)[0]
        numerical_B[:, column] = (dp-dm)/(2*h)
        numerical_K[:, column] = (fp-fm)/(2*h)
    np.testing.assert_array_equal(u, before)
    assert np.linalg.norm(B-numerical_B)/np.linalg.norm(numerical_B) < 1e-9
    assert np.linalg.norm(tangent-numerical_K)/np.linalg.norm(numerical_K) < 1e-7
    # Match the existing objective contract: the material BLAS product need
    # not be bitwise symmetric, while its relative skew is roundoff-sized.
    assert np.linalg.norm(tangent-tangent.T)/np.linalg.norm(tangent) < 1e-12
