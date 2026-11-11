"""Actual physical-skin covectors under additional path constraints."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_interior_press_dev import build_interior_press


@pytest.fixture
def candidate():
    model, definition, settings, _ = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", reference_slope=0.)
    manager = GeneralS3PatchManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    _, _, certificate = manager.evaluate(u, tangent=False)
    pressure = np.zeros(len(certificate.source_pressure_node_ids))
    index = certificate.source_pressure_node_ids.index(13)
    pressure[index] = .01/certificate.reference_support_m2[index]
    certificate = replace(certificate, numerical_pressure_pa=tuple(pressure))
    transform = np.zeros((u.size, 2))
    transform[model.mesh.dof_manager.get_node_dofs(13)[2], 0] = 1.
    target_node = manager.law.target_node_ids[-1]
    transform[model.mesh.dof_manager.get_node_dofs(target_node)[2], 1] = 1.
    return manager, u, certificate, transform


def test_path_control_of_source_normal_is_rejected(candidate):
    manager, u, certificate, transform = candidate
    assert manager.d1_interior_admission_issue(certificate, u, transform) is None
    issue = manager.d1_interior_admission_issue(certificate, u, transform,
        control_rows=np.array(((1., 0.),)))
    assert "prescribed-normal source node 13" in issue


@pytest.mark.parametrize("controls", [((0., 1.),), ((0., 1.), (0., 1.)),
                                      ((1., 1.),), ((0., 0.),)])
def test_independent_or_coupled_control_does_not_falsely_prescribe_source(candidate, controls):
    manager, u, certificate, transform = candidate
    assert manager.d1_interior_admission_issue(certificate, u, transform,
        control_rows=np.array(controls)) is None


def test_control_admission_revision_is_bound_to_checkpoint(candidate):
    assert candidate[0].restart_contract()["settings"]["d1_normal_admission_revision"] == "model_and_path_constraints_v2"
