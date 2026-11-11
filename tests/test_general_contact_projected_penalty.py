"""Projected finite penalty: independent planar work and moving-cut controls."""
from dataclasses import replace
from scipy import sparse
import numpy as np
import pytest
from anyfesolver import validate_general_contact
from anyfesolver.general_contact.s3_patch_static import (
    GeneralS3PatchManager, REFINED_Q4_S3_PROJECTED_PENALTY_PROFILE,
)
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3
from test_general_contact_q4_s3_jet2_static_owner import _settings


def _candidate():
    model, definition, settings = _two_q4_two_s3()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], normal_law="projected_penalty"),))
    settings = replace(_settings(settings), q4_source_gap_certificate="overlap_local_dev")
    return model, definition, settings


def test_planar_cut_pressure_reactions_and_nonpotential_storage():
    model, definition, settings = _candidate()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == REFINED_Q4_S3_PROJECTED_PENALTY_PROFILE
    manager = GeneralS3PatchManager(model, definition, settings)
    u = np.zeros(manager.n_dof)
    for node in range(1, 7):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    free = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (2, 3, 5, 6)]
    free += [model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(7, 11)]
    manager.restrict_tangent_to_reduced_variations(sparse.eye(manager.n_dof, format="csr")[:, free])
    force, tangent, result = manager.evaluate(u)
    # Independent rectangle intersection, physical offsets and scalar law:
    # source bottom .030, target top .015, after pressing gap=-.005.
    area, gap, pressure = .9*.9, -.005, 1000*.005
    np.testing.assert_allclose(result.reference_overlap_area, area, atol=1e-13)
    np.testing.assert_allclose(result.numerical_pressure_pa, pressure, atol=1e-12)
    np.testing.assert_allclose(result.integration_force, (0, 0, pressure*area), atol=1e-12)
    np.testing.assert_allclose(result.target_force, (0, 0, -pressure*area), atol=1e-12)
    np.testing.assert_allclose(np.add(result.integration_moment, result.target_moment), 0, atol=1e-12)
    assert result.active_pressure_dofs == 6
    assert all(s > 0 for s in result.reference_support_m2)
    # The prescribed source-normal rows 1/4 remain in the residual, even
    # though their tangent variation columns were removed above.
    for node in (1, 4):
        index = result.source_pressure_node_ids.index(node)
        dof = model.mesh.dof_manager.get_node_dofs(node)[2]
        np.testing.assert_allclose(force[dof], -pressure*result.reference_support_m2[index], atol=1e-12)
        assert abs(force[dof]) > 0
    assert result.penalty_energy == 0
    assert result.augmented_merit_j == 0
    np.testing.assert_allclose(result.regularization_storage_j, .5*1000*gap**2*area, atol=1e-13)
    assert result.to_dict()["contact_work_convention"] == "projected_weak_gap_nonpotential_v1"
    assert result.normal_multiplier_estimate_pa == ()
    with pytest.raises(ValueError, match="no normal multipliers"):
        manager.set_normal_multipliers(np.ones(6))
    # Shift the target in x at constant physical gap. Projected cut work is
    # zero; storage changes by 1/2 k g^2 dA, independently known dA/dx=.9.
    direction = np.zeros(manager.n_dof)
    for node in range(7, 11):
        direction[model.mesh.dof_manager.get_node_dofs(node)[0]] = 1
    assert abs(force @ direction) < 1e-12
    step = 1e-6
    plus = manager.evaluate(u+step*direction, tangent=False)
    minus = manager.evaluate(u-step*direction, tangent=False)
    storage_derivative = (plus[2].regularization_storage_j-minus[2].regularization_storage_j)/(2*step)
    np.testing.assert_allclose(storage_derivative, .5*1000*gap**2*.9, rtol=1e-8)
    np.testing.assert_allclose(tangent @ direction, (plus[0]-minus[0])/(2*step), rtol=1e-5, atol=1e-7)
    # Contact opens without a retained multiplier or traction.
    opened = manager.evaluate(np.zeros(manager.n_dof), tangent=False)
    np.testing.assert_array_equal(opened[0], 0)
    assert opened[2].regularization_storage_j == 0


@pytest.mark.parametrize("change", [
    {"enforcement_solver": "monolithic_active_set", "complementarity_force_scale_n": 1.0},
    {"augmentation_support_resolution": "roundoff", "augmentation_support_force_budget_n": 1e-6},
    {"d1_interior_profile": True, "d1_interior_coverage_tolerance_m2": 1e-10, "d1_interior_boundary_force_tolerance_n": 1e-8},
    {"q4_source_gap_certificate": "whole_face"},
    {"general_s3_differentiation": "existing"},
])
def test_profile_rejects_other_enforcement_and_geometry(change):
    model, definition, settings = _candidate()
    assert not validate_general_contact(model, definition, replace(settings, **change)).valid
    with pytest.raises(ValueError, match="unsupported"):
        GeneralS3PatchManager(model, definition, replace(settings, **change))


def test_restart_and_old_result_contracts_remain_distinct():
    model, definition, settings = _candidate()
    new = GeneralS3PatchManager(model, definition, settings)
    old_definition = replace(definition, interactions=(replace(
        definition.interactions[0], normal_law="augmented"),))
    old = GeneralS3PatchManager(model, old_definition, settings)
    old_contract = old.restart_contract()
    new_contract = new.restart_contract()
    assert "projected_penalty_contract" not in old_contract
    assert new_contract["projected_penalty_contract"]["multipliers"] == "identically_zero"
    assert new_contract["interaction"]["normal_law"] == "projected_penalty"
    changed = GeneralS3PatchManager(model, replace(definition, interactions=(replace(
        definition.interactions[0], normal_stiffness=2000),)), settings)
    assert changed.restart_contract() != new_contract
    old_payload = old.evaluate(np.zeros(old.n_dof), tangent=False)[2].to_dict()
    assert "regularization_storage_j" not in old_payload
    assert "contact_work_convention" not in old_payload


def test_smooth_both_body_skin_directional_tangent_interval():
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    u = np.zeros(manager.n_dof)
    direction = np.zeros(manager.n_dof)
    for node in range(1, 7):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        x = model.mesh.get_node(node).x
        u[dofs[2]] = -.02 + .001*(x-.6)
        direction[dofs[2]] = .2 + .1*(x-.6)
        direction[dofs[3]] = .03  # Physical bottom skin displacement.
    for node in range(7, 11):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        x = model.mesh.get_node(node).x
        u[dofs[2]] = .0003*x
        direction[dofs[0]] = .1
        direction[dofs[2]] = -.1 + .05*x
        direction[dofs[4]] = -.02  # Offset S3 top skin displacement.
    force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs == 6
    analytic = tangent @ direction
    for step in (1e-6, 3e-7, 1e-7):
        plus = manager.evaluate(u+step*direction, tangent=False)
        minus = manager.evaluate(u-step*direction, tangent=False)
        assert plus[2].active_pressure_dofs == minus[2].active_pressure_dofs == 6
        numeric = (plus[0]-minus[0])/(2*step)
        assert np.linalg.norm(analytic-numeric)/np.linalg.norm(numeric) <= 1e-5


def test_ordinary_scope_rejects_unsupported_control_before_mutation():
    from anyfesolver.general_contact.core import contact_scope
    from anyfesolver import solve_static_nonlinear
    model, definition, settings = _candidate()
    with pytest.raises(ValueError, match="unsupported projected penalty"):
        with contact_scope(model, definition, settings, analysis_control="arc_length"):
            pytest.fail("unsupported analysis entered contact scope")
    with pytest.raises(ValueError, match="requires force or displacement"):
        solve_static_nonlinear(model, contact=definition, contact_settings=settings, control="arc_length")
    manager = GeneralS3PatchManager(model, definition, settings)
    assert not manager.normal_multipliers.flags.writeable
    np.testing.assert_array_equal(manager.normal_multipliers, 0)


def test_ordinary_rejects_unadmitted_finite_rotation_work_map():
    from anyfesolver import solve_static_nonlinear
    model, definition, settings = _candidate()
    with pytest.raises(ValueError, match="small-rotation kinematics"):
        solve_static_nonlinear(model, contact=definition, contact_settings=settings,
                              kinematics="corotational")
