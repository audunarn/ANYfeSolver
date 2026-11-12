"""Preserve the scalar work-conjugate mechanics while batching its stencils."""
from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.corotational as cr
from anyfesolver import _corotational_batch as batch
from test_corotational_work_conjugate import (
    _make_beam_model, _make_q4_shell_model, _make_s3_shell_model,
    _superimpose_se3_motion,
)


@pytest.mark.parametrize("builder", [_make_beam_model, _make_q4_shell_model, _make_s3_shell_model],
                         ids=["beam", "q4", "s3"])
@pytest.mark.parametrize("angle", [0., 35., 100.])
def test_batched_geometric_tangent_preserves_scalar_stencils(builder, angle):
    model, element_id, element = builder()
    reference = cr._corotational_cache(model)[element_id]
    coordinates = np.asarray(element.get_node_coordinates(model.mesh))
    rng = np.random.default_rng(20261018)
    local = np.zeros((element.num_nodes, 6))
    local[:, :3] = .006*rng.standard_normal((element.num_nodes, 3))
    local[:, 3:] = .035*rng.standard_normal((element.num_nodes, 3))
    u = _superimpose_se3_motion(coordinates, local.ravel(),
            np.deg2rad(angle)*np.array([.6, -.5, .6244997998398398]),
            np.array([.3, -.4, .2]))
    ud, R, deformed = cr._corotational_deformation_state(reference, element, u)
    rotations = u.reshape(-1, 6)[:, 3:]
    D, U, G = cr._pullback_jacobian_factors(reference, element, deformed, rotations, R)
    force, tangent, _ = element.compute_nonlinear_response(
        model.mesh, model.get_material(element.material_name), ud, None, 5, True)
    args = (reference, element, deformed, rotations, R, G, D+U@G, force, tangent)
    arrays = [a for a in args[2:] if isinstance(a, np.ndarray)]
    arrays += [reference.coordinates]
    if reference.frame is not None:
        arrays.append(reference.frame)
    if reference.center_dN is not None:
        arrays.extend(reference.center_dN)
    preimages = [a.copy() for a in arrays]
    old_material, old_geometric = cr._scalar_work_conjugate_corotational_tangent(*args)
    material, geometric = cr._work_conjugate_corotational_tangent(*args)
    for array, preimage in zip(arrays, preimages):
        np.testing.assert_array_equal(array, preimage)
    np.testing.assert_array_equal(material, old_material)
    assert np.linalg.norm(geometric-old_geometric)/np.linalg.norm(old_material+old_geometric) < 1e-7
    assert np.linalg.norm(geometric-old_geometric)/np.linalg.norm(old_geometric) < 1e-6
    np.testing.assert_array_equal(geometric, geometric.T)


def test_batch_so3_primitives_preserve_zero_and_pi_branches():
    vectors = np.array([[0., 0., 0.], [1e-12, -2e-12, 3e-12],
                        [.01, -.02, .03], [np.pi-1e-8, 0., 0.],
                        [0., np.pi+1e-8, 0.]])
    rotations = batch.exp(vectors)
    for index, vector in enumerate(vectors):
        np.testing.assert_allclose(rotations[index], cr.rotation_matrix_from_vector(vector), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(batch.log(rotations)[index], cr.rotation_vector_from_matrix(rotations[index]), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(batch.right_jacobian(vectors)[index], cr._rotation_right_jacobian(vector), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(batch.inverse_jacobian(vectors)[index], cr._rotation_right_jacobian_inverse(vector), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(batch.inverse_jacobian(vectors, left=True)[index], cr._rotation_left_jacobian_inverse(vector), atol=1e-14, rtol=1e-14)


def test_batch_axis_alignment_preserves_antiparallel_branch():
    a = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    b = np.array([[-1., 0., 0.], [1e-7, -1., 0.], [.1, .2, .9]])
    rotations = batch.minimal(a, b)
    for index in range(len(a)):
        np.testing.assert_allclose(rotations[index], cr._minimal_rotation(a[index], b[index]), atol=1e-14, rtol=1e-14)
