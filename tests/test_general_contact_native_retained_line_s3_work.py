"""Native line/S3 physical contact mapped to admitted S3 nodal DOFs."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.native_retained_line_s3 import (
    NativeRetainedRoundLineS3Interior,
)
from anyfesolver.general_contact.native_retained_line_s3_work import (
    NativeRetainedLineS3Work,
)
from anyfesolver.general_contact.overlap import reference_shell_face
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter


def _case():
    model = FEModel("native_s3_work")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    for node, point in zip((4, 5, 6),
                           ((-5.0, -5.0, -0.025),
                            (5.0, -5.0, -0.025),
                            (0.0, 10.0, -0.025))):
        model.add_node(node, *point)
    model.add_element(21, create_shell_element(
        21, (4, 5, 6), "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.0,
    ))
    skin = S3SkinWorkAdapter.from_model(model, 7, 21, "top")
    face = reference_shell_face(model, 7, 21, "top")
    beam = np.array(((0.0, 0.0, 0.08),
                     (1.0, 0.0, 0.08),
                     (2.0, 0.0, 0.08)))
    reference = CenteredCurvedBeam3ReferenceGeometry(
        beam, np.tile(np.eye(3), (3, 1, 1)),
    )
    contact = NativeRetainedRoundLineS3Interior(
        11, reference, (1, 2, 3), 21, face, (4, 5, 6),
        radius_m=0.1, line_stiffness_n_per_m2=100.0,
        rim_clearance_m=0.1, geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-10, force_tolerance_n=1e-9,
        tangent_tolerance_n_per_m=1e-7,
        maximum_integration_intervals=100,
    )
    return model, skin, NativeRetainedLineS3Work(contact, skin), beam


def test_real_s3_top_skin_offset_and_authoritative_work():
    model, skin, work, beam = _case()
    assert skin.physical_offset == pytest.approx(0.025)
    np.testing.assert_allclose(skin.reference_positions[:, 2], 0, atol=1e-15)
    low = np.zeros((3, 3))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(5)[2]] = 0.002
    u[model.mesh.dof_manager.get_node_dofs(6)[3]] = 0.008
    center = work.evaluate(beam, low, rotations, u)
    assert center.contact.potential_j > 0
    assert center.residual.shape == (42+len(u),)
    assert center.tangent.shape == (42+len(u), 42+len(u))
    np.testing.assert_allclose(
        center.residual[42:],
        skin.work_map.T@center.contact.residual[42:], atol=1e-13,
    )

    direction = np.zeros_like(u)
    direction[model.mesh.dof_manager.get_node_dofs(4)[0]] = 0.13
    direction[model.mesh.dof_manager.get_node_dofs(5)[2]] = -0.21
    direction[model.mesh.dof_manager.get_node_dofs(6)[3]] = 0.08
    step = 1e-5
    forward = work.evaluate(beam, low, rotations, u+step*direction)
    backward = work.evaluate(beam, low, rotations, u-step*direction)
    full_direction = np.r_[np.zeros(42), direction]
    assert ((forward.contact.potential_j-backward.contact.potential_j)
            /(2*step)) == pytest.approx(float(center.residual@full_direction),
                                        rel=1e-6, abs=1e-8)
    measured = (forward.residual-backward.residual)/(2*step)
    np.testing.assert_allclose(measured, center.tangent@full_direction,
                               rtol=2e-5, atol=2e-5)

    drill = np.zeros_like(u)
    drill[model.mesh.dof_manager.get_node_dofs(4)[5]] = 1
    np.testing.assert_array_equal(work.evaluate(beam, low, rotations, u+drill).residual,
                                  center.residual)


def test_exact_shell_identity_and_definition_binding():
    _, skin, work, beam = _case()
    with pytest.raises(ValueError, match="does not bind"):
        NativeRetainedLineS3Work(work.contact, replace(skin, body_id=8))
    low = np.zeros((3, 3))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    u = np.zeros(skin.work_map.shape[1])
    skin.work_map.data[0] += 1
    with pytest.raises(ValueError, match="definition changed"):
        work.evaluate(beam, low, rotations, u)
