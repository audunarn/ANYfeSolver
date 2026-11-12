"""Qualified planar Q4 physical-skin contact work mapping."""

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.overlap import (
    current_planar_shell_face, planar_projected_overlap,
)
from anyfesolver.general_contact.q4_skin_work import Q4SkinWorkAdapter
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
from anyfesolver.general_contact.triangle_mortar_energy import MovingS3PrimalMortarPenalty


def _model():
    model = FEModel("q4_skin_contact_work")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    q4 = ((0., 0., 0.), (2., 0., 0.),
          (1.8, 1.8, 0.), (0., 2., 0.))
    s3 = ((0.2, 0.2, 0.08), (0.8, 0.2, 0.08),
          (0.2, 0.8, 0.08))
    for node, position in enumerate((*q4, *s3), 1):
        model.add_node(node, *position)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
        reference_normal=(0, 0, 1),
    ))
    model.add_element(2, create_shell_element(
        2, (5, 6, 7), "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.01,
    ))
    return model


def test_q4_skin_preserves_bilinear_face_and_physical_offset_work():
    model = _model()
    adapter = Q4SkinWorkAdapter.from_model(model, 1, 1, "top")
    assert adapter.physical_offset == pytest.approx(0.025)
    np.testing.assert_allclose(adapter.reference_positions[:, 2],
                               0.025, atol=0.0)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(3)[0]] = -0.01
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[3]] = 0.002
    mapped = adapter.current_positions(u)
    separate = current_planar_shell_face(model, 1, 1, "top", u)
    np.testing.assert_allclose(mapped, separate.corners, atol=1e-15)
    source = S3SkinWorkAdapter.from_model(model, 2, 2, "bottom")
    overlap = planar_projected_overlap(separate, current_planar_shell_face(
        model, 2, 2, "bottom", np.zeros_like(u),
    ))
    assert overlap is not None
    point = overlap.polygon[0]
    shape = overlap.shape_values(separate, point)
    assert shape.shape == (4,)
    assert np.min(shape) >= -1e-12
    np.testing.assert_allclose(shape@mapped, point, atol=1e-12)
    drill = np.zeros_like(u)
    drill[model.mesh.dof_manager.get_node_dofs(1)[5]] = 0.1
    np.testing.assert_allclose(adapter.work_map@drill, 0.0, atol=0.0)
    gradient = np.zeros(12)
    gradient[1] = 10.0
    global_gradient = adapter.work_map.T@gradient
    assert abs(global_gradient[model.mesh.dof_manager.get_node_dofs(1)[3]]) > 0.0
    with pytest.raises(ValueError, match="not the admitted qualified Q4"):
        Q4SkinWorkAdapter.from_model(model, 1, 2, "top")


def test_s3_source_mortar_targets_original_planar_q4_without_triangulation():
    model = _model()
    source = S3SkinWorkAdapter.from_model(model, 1, 2, "bottom")
    target = Q4SkinWorkAdapter.from_model(model, 2, 1, "top")
    law = MovingS3PrimalMortarPenalty(
        source.reference_positions, target.reference_positions,
        1000.0, difference_step=2e-5,
        q4_weighted_gap_tolerance=1e-12,
        maximum_q4_integration_triangles=500,
    )
    current = source.reference_positions.copy()
    current[:, 2] = 0.015
    state = law.evaluate(current, target.reference_positions)
    assert state.pressure.shape == (3,)
    assert state.overlap_area > 0.0
    assert state.minimum_physical_gap < 0.0
    work = law.work(current, target.reference_positions, tangent=True)
    assert work.residual.shape == (21,)
    assert work.tangent.shape == (21, 21)
    np.testing.assert_allclose(work.residual.reshape(7, 3).sum(axis=0),
                               0.0, atol=1e-8*np.linalg.norm(work.residual))


def test_warped_trial_q4_weighted_gap_has_independent_polynomial_answer():
    source_reference = np.array(((0., 0., .1), (1., 0., .1), (0., 1., .1)))
    target_reference = np.array(((-.1, -.1, 0.), (1.1, -.1, 0.),
                                 (1.1, 1.1, 0.), (-.1, 1.1, 0.)))
    source = source_reference.copy()
    source[:, 2] = -.02
    target = target_reference.copy()
    target[2, 2] = .03*1.2**2
    law = MovingS3PrimalMortarPenalty(
        source_reference, target_reference, 1000.,
        difference_step=2e-5, q4_weighted_gap_tolerance=1e-12,
        maximum_q4_integration_triangles=100,
    )
    state = law.evaluate(source, target)
    expected = (-.02*np.full(3, 1./6.)
                -.03*(np.array((1./120., 1./60., 1./60.))
                      +.1*np.array((1./12., 1./8., 1./8.))
                      +.01/6.))
    np.testing.assert_allclose(state.reference_support, 1./6., atol=1e-12)
    np.testing.assert_allclose(state.reference_weighted_gap, expected, atol=1e-12)
    assert state.weighted_gap_integration_error <= 1e-12
    assert state.integration_triangles <= 100
    work = law.work(source, target, tangent=True)
    assert abs(work.residual[9+2*3+2]) > 0.0
    np.testing.assert_allclose(work.tangent, work.tangent.T, atol=1e-8)
