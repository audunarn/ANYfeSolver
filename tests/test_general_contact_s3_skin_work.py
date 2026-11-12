"""Formulation-owned S3 physical-skin contact work mapping."""

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.overlap import current_planar_shell_face
from anyfesolver.general_contact.friction_point import (
    objective_relative_tangent_increment,
    objective_relative_tangent_variation,
)
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
from anyfesolver.general_contact.triangle_pair_energy import MovingS3PairPenalty


def _model():
    model = FEModel("s3_moving_skin_work")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((0.4, 0.2, 0.065), (0.9, 0.2, 0.065),
              (0.4, 0.7, 0.065))
    target = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (0.0, 1.0, 0.0))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 3)), (2, (4, 5, 6))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    return model


def test_s3_skin_uses_formulation_offset_and_director_without_drill_work():
    model = _model()
    adapters = {
        side: S3SkinWorkAdapter.from_model(model, 1, 1, side)
        for side in ("bottom", "midsurface", "top")
    }
    for side, offset in (("bottom", -0.035),
                         ("midsurface", -0.01), ("top", 0.015)):
        adapter = adapters[side]
        assert adapter.physical_offset == pytest.approx(offset)
        np.testing.assert_allclose(adapter.reference_positions[:, 2],
                                   0.065 + offset, atol=1e-15)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    dofs = model.mesh.dof_manager.get_node_dofs(1)
    u[dofs[:3]] = (0.01, -0.02, -0.005)
    u[dofs[3:6]] = (0.002, -0.003, 0.004)
    bottom = adapters["bottom"]
    mapped = bottom.current_positions(u)
    separate = current_planar_shell_face(model, 1, 1, "bottom", u)
    np.testing.assert_allclose(mapped, separate.corners, atol=1e-15)
    drill = np.zeros_like(u)
    drill[dofs[5]] = 1.0
    np.testing.assert_allclose(bottom.work_map @ drill, 0.0, atol=0.0)
    np.testing.assert_allclose(bottom.current_positions(u + 0.1*drill),
                               mapped, atol=0.0)
    with pytest.raises(ValueError, match="not the admitted S3"):
        S3SkinWorkAdapter.from_model(model, 3, 99, "top")


def test_s3_pair_physical_work_pullback_and_tangent_use_authoritative_dofs():
    model = _model()
    source = S3SkinWorkAdapter.from_model(model, 1, 1, "bottom")
    target = S3SkinWorkAdapter.from_model(model, 2, 2, "top")
    law = MovingS3PairPenalty(
        source.reference_positions, target.reference_positions,
        1000.0, difference_step=2e-5,
    )
    mapping = sparse.vstack((source.work_map, target.work_map)).tocsr()
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.025
    u[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.01
    u[model.mesh.dof_manager.get_node_dofs(3)[1]] = -0.015

    def state_at(displacement, *, tangent):
        first = source.current_positions(displacement)
        second = target.current_positions(displacement)
        return law.work(first, second, tangent=tangent)

    physical = state_at(u, tangent=True)
    residual = np.asarray(mapping.T @ physical.residual).reshape(-1)
    stiffness = (mapping.T @ sparse.csr_matrix(physical.tangent)
                 @ mapping).tocsr()
    source_part, _ = source.pullback(physical.residual[:9])
    target_part, _ = target.pullback(physical.residual[9:])
    np.testing.assert_allclose(source_part + target_part, residual, atol=1e-14)
    assert abs(residual[model.mesh.dof_manager.get_node_dofs(2)[3]]) > 1e-5
    direction = np.zeros_like(u)
    direction[model.mesh.dof_manager.get_node_dofs(1)[1]] = 0.13
    direction[model.mesh.dof_manager.get_node_dofs(2)[0]] = -0.21
    direction[model.mesh.dof_manager.get_node_dofs(2)[3]] = 0.08
    direction[model.mesh.dof_manager.get_node_dofs(5)[2]] = -0.17
    step = 3e-5
    plus, minus = u + step*direction, u - step*direction
    plus_state = state_at(plus, tangent=False)
    minus_state = state_at(minus, tangent=False)
    measured_work = ((plus_state.energy.energy - minus_state.energy.energy)
                     / (2*step))
    assert measured_work == pytest.approx(float(residual @ direction),
                                          rel=1e-5, abs=1e-7)
    measured_tangent = np.asarray(mapping.T @ (
        (plus_state.residual - minus_state.residual) / (2*step)
    )).reshape(-1)
    relative_error = float(np.linalg.norm(
        measured_tangent - stiffness @ direction
    ) / np.linalg.norm(measured_tangent))
    assert relative_error <= 1e-5
    drill = np.zeros_like(u)
    drill[model.mesh.dof_manager.get_node_dofs(1)[5]] = 1.0
    np.testing.assert_allclose(mapping @ drill, 0.0, atol=0.0)
    assert float(residual @ drill) == 0.0


def test_s3_material_point_uses_physical_skin_and_authoritative_work_map():
    model = _model()
    point = S3SkinMaterialPoint.from_model(
        model, 1, 1, "bottom", (0.3, 0.3, 0.4),
    )
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    initial = point.physical_pose(u)
    expected = np.asarray(point.barycentric) @ point.adapter.reference_positions
    np.testing.assert_allclose(initial.position, expected, atol=1e-15)
    assert expected[2] == pytest.approx(0.03)
    np.testing.assert_allclose(initial.material_frame, np.eye(3), atol=1e-15)
    u[model.mesh.dof_manager.get_node_dofs(1)[:3]] = (0.01, -0.02, 0.005)
    u[model.mesh.dof_manager.get_node_dofs(2)[3:6]] = (0.002, -0.003, 0.004)
    current = point.physical_pose(u)
    np.testing.assert_allclose(
        np.asarray(current.position) - initial.position,
        np.asarray(point.position_map @ u).reshape(3), atol=1e-15,
    )
    frame = np.asarray(current.material_frame)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-15)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    traction = np.array((3.0, -2.0, 1.0))
    direction = np.linspace(-0.03, 0.04, u.size)
    nodal_force = np.asarray(point.position_map.T @ traction).reshape(-1)
    assert abs(nodal_force[model.mesh.dof_manager.get_node_dofs(1)[3]]) > 0.0
    assert float(nodal_force @ direction) == pytest.approx(
        float(traction @ (point.position_map @ direction)), abs=1e-14,
    )
    frame_tangent = point.frame_directional_variation(u, direction)
    for step in (1.0e-5, 3.0e-6):
        plus = np.asarray(point.physical_pose(u + step*direction).material_frame)
        minus = np.asarray(point.physical_pose(u - step*direction).material_frame)
        np.testing.assert_allclose(
            frame_tangent, (plus-minus)/(2.0*step),
            rtol=1.0e-7, atol=1.0e-9,
        )
    drill = np.zeros_like(u)
    drill[model.mesh.dof_manager.get_node_dofs(1)[5]] = 1.0
    np.testing.assert_allclose(point.position_map @ drill, 0.0, atol=0.0)
    np.testing.assert_allclose(point.physical_pose(u + 0.1*drill).position,
                               current.position, atol=0.0)
    np.testing.assert_allclose(point.physical_pose(u + 0.1*drill).material_frame,
                               current.material_frame, atol=0.0)
    np.testing.assert_allclose(point.frame_directional_variation(u, drill),
                               0.0, atol=0.0)
    with pytest.raises(ValueError, match="full FE direction"):
        point.frame_directional_variation(u, np.zeros(3))


def test_s3_material_pair_objective_motion_and_geometry_rejection():
    model = _model()
    source = S3SkinMaterialPoint.from_model(
        model, 1, 1, "bottom", (0.3, 0.3, 0.4),
    )
    target = S3SkinMaterialPoint.from_model(
        model, 2, 2, "top", (0.05, 0.55, 0.4),
    )
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    source_old = source.physical_pose(zero)
    target_old = target.physical_pose(zero)
    assert np.linalg.norm(np.asarray(source_old.position)-target_old.position) == pytest.approx(0.015)

    def increment(u):
        source_new = source.physical_pose(u)
        target_new = target.physical_pose(u)
        return objective_relative_tangent_increment(
            source_committed=source_old.position,
            target_committed=target_old.position,
            source_candidate=source_new.position,
            target_candidate=target_new.position,
            target_frame_committed=target_old.material_frame,
            target_frame_candidate=target_new.material_frame,
        )

    common = zero.copy()
    for node in range(1, 7):
        common[model.mesh.dof_manager.get_node_dofs(node)[:3]] = (
            0.01, -0.02, 0.005,
        )
    np.testing.assert_allclose(increment(common), 0.0, atol=1e-15)
    moving_source = zero.copy()
    for node in (1, 2, 3):
        moving_source[model.mesh.dof_manager.get_node_dofs(node)[0]] = 0.001
    np.testing.assert_allclose(increment(moving_source),
                               (0.001, 0.0, 0.0), atol=1e-15)

    candidate = moving_source.copy()
    candidate[model.mesh.dof_manager.get_node_dofs(5)[0]] = 0.005
    candidate[model.mesh.dof_manager.get_node_dofs(5)[2]] = -0.001
    candidate[model.mesh.dof_manager.get_node_dofs(6)[1]] = -0.003
    candidate[model.mesh.dof_manager.get_node_dofs(6)[3]] = 0.002
    source_new = source.physical_pose(candidate)
    target_new = target.physical_pose(candidate)
    direction = np.linspace(-0.02, 0.03, candidate.size)

    def directional(value):
        return objective_relative_tangent_variation(
            source_committed=source_old.position,
            target_committed=target_old.position,
            source_candidate=source_new.position,
            target_candidate=target_new.position,
            target_frame_committed=target_old.material_frame,
            target_frame_candidate=target_new.material_frame,
            source_position_variation=source.position_map @ value,
            target_position_variation=target.position_map @ value,
            target_frame_variation=target.frame_directional_variation(
                candidate, value,
            ),
        )

    for step in (1.0e-5, 3.0e-6):
        finite = (np.asarray(increment(candidate + step*direction))
                  - increment(candidate - step*direction))/(2.0*step)
        np.testing.assert_allclose(directional(direction), finite,
                                   rtol=1.0e-7, atol=1.0e-9)
    drill = np.zeros_like(candidate)
    drill[model.mesh.dof_manager.get_node_dofs(5)[5]] = 1.0
    np.testing.assert_allclose(directional(drill), 0.0, atol=0.0)
    with pytest.raises(ValueError, match="frame variation"):
        objective_relative_tangent_variation(
            source_committed=source_old.position,
            target_committed=target_old.position,
            source_candidate=source_new.position,
            target_candidate=target_new.position,
            target_frame_committed=target_old.material_frame,
            target_frame_candidate=target_new.material_frame,
            source_position_variation=(0, 0, 0),
            target_position_variation=(0, 0, 0),
            target_frame_variation=np.full((3, 3), np.nan),
        )

    inverted = zero.copy()
    inverted[model.mesh.dof_manager.get_node_dofs(3)[1]] = -0.6
    with pytest.raises(ValueError, match="inverted"):
        source.physical_pose(inverted)
    degenerate = zero.copy()
    degenerate[model.mesh.dof_manager.get_node_dofs(3)[1]] = -0.5
    with pytest.raises(ValueError, match="degenerate"):
        source.physical_pose(degenerate)
    with pytest.raises(ValueError, match="nonnegative weights"):
        S3SkinMaterialPoint.from_model(model, 1, 1, "bottom",
                                        (-0.1, 0.5, 0.6))
