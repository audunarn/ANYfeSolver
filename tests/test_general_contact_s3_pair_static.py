"""Bounded development checks for arbitrary-nodal S3/S3 static contact."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, DisplacementControl, LoadCase, create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import (
    ContactDefinition, ContactInteraction, ContactSolverSettings, ContactSurface,
    validate_general_contact,
)
from anyfesolver.general_contact.s3_pair_static import (
    GeneralS3PairManager, _swept_gap_lower,
)


def _case(*, gap_tolerance=0.011, maximum_swept_subdivisions=14,
          normal_discretization="pointwise_penalty"):
    model = FEModel("general_s3_pair")
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
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (1,), "bottom"),
                  ContactSurface("target", (2,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1000.0, gap_tolerance,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization=normal_discretization,
        general_s3_difference_step=2e-5,
        maximum_swept_subdivisions=maximum_swept_subdivisions,
    )
    return model, definition, settings


@pytest.mark.parametrize("discretization", ("pointwise_penalty", "primal_mortar"))
def test_general_s3_pair_maps_nonuniform_skin_work_and_restart_identity(discretization):
    model, definition, settings = _case(normal_discretization=discretization)
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PairManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.025
    u[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.01
    u[model.mesh.dof_manager.get_node_dofs(3)[1]] = -0.015
    force, stiffness, result = manager.evaluate(u)
    assert result.penalty_energy > 0.0
    assert result.minimum_gap <= -0.01
    assert result.gap_tolerance_met
    assert stiffness.shape == (u.size, u.size)
    np.testing.assert_allclose(np.array(result.integration_force)
                               + np.array(result.target_force), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.array(result.integration_moment)
                               + np.array(result.target_moment), 0.0, atol=1e-8)
    assert np.linalg.norm(force) > 0.0
    assert manager.restart_contract()["settings"]["general_s3_difference_step"] == 2e-5
    if discretization == "primal_mortar":
        assert result.active_pressure_dofs == 3
        assert len(result.normalized_weighted_gap_m) == 3
        assert all(item > 0.0 for item in result.reference_support_m2)
        assert result.sampled_overlap_gap is not None


def test_general_s3_swept_bound_catches_crossing():
    first = np.array(((0., 0., 1.), (1., 0., 1.), (0., 1., 1.)))
    target = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    # A linear trial reaches and crosses the target.
    middle = first.copy()
    middle[:, 2] = -1.0
    assert _swept_gap_lower(first, middle, target, target) < 0.0


def test_general_s3_swept_bound_is_below_independent_path_samples():
    source0 = np.array(((0.2, 0.1, 0.08), (0.8, 0.1, 0.08),
                        (0.2, 0.7, 0.08)))
    target0 = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    source1 = source0 + np.array(((0.01, 0., -0.01),
                                  (0., 0.02, -0.012),
                                  (-0.01, 0., -0.008)))
    target1 = target0 + np.array(((0., 0., 0.),
                                  (0., 0., 0.004),
                                  (0., 0., -0.003)))
    lower = _swept_gap_lower(source0, source1, target0, target1)
    assert np.isfinite(lower)
    for fraction in np.linspace(0.0, 1.0, 101):
        source = source0 + fraction*(source1-source0)
        target = target0 + fraction*(target1-target0)
        normal = np.cross(source[1]-source[0], source[2]-source[0])
        normal /= np.linalg.norm(normal)
        independently_sampled = min(float((a-b) @ normal)
                                     for a in source for b in target)
        assert lower <= independently_sampled


def test_general_s3_manager_cuts_back_through_crossing():
    model, definition, settings = _case(gap_tolerance=0.01)
    manager = GeneralS3PairManager(model, definition, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node in (1, 2, 3):
        end[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.05
    assert not manager.swept_increment_admissible(start, end, 0.0, 1.0)
    fraction = manager.admissible_step_fraction(start, end)
    assert 0.0 < fraction < 1.0
    assert manager.swept_increment_admissible(start, start + fraction*end,
                                              0.0, fraction)


def test_general_s3_mortar_reports_weak_open_gap_separately():
    model, definition, settings = _case(
        gap_tolerance=5e-5, normal_discretization="primal_mortar",
    )
    manager = GeneralS3PairManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[2]] = -0.012
    for node in (2, 3):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.016
    _force, _tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs == 0
    assert min(result.normalized_weighted_gap_m) > 0.0
    assert result.minimum_gap < -5e-5
    assert not result.gap_tolerance_met


@pytest.mark.parametrize("discretization", ("pointwise_penalty", "primal_mortar"))
def test_general_s3_coupled_displacement_and_restart(discretization):
    model, definition, settings = _case(
        gap_tolerance=0.02, normal_discretization=discretization,
    )
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (4, 5, 6),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive")
    load.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    common = dict(contact=definition, contact_settings=settings,
                  kinematics="von_karman", control="displacement",
                  displacement_control=DisplacementControl(
                      node_id=1, dof="uz", target_displacement=-0.02,
                  ), max_iterations=15)
    full = solve_static_nonlinear(
        model, load, num_steps=2, emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", full.failure_reason
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert full.info["general_contact"]["penalty_energy"] > 0.0
    assert full.restart_checkpoint["version"] == 2
    half = solve_static_nonlinear(
        model, load, num_steps=1, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=1,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
    changed = ContactSolverSettings(
        pair_motion="general", normal_discretization=discretization,
        general_s3_difference_step=1e-5,
    )
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        solve_static_nonlinear(
            model, load, num_steps=1,
            restart_checkpoint=half.to_restart_checkpoint(),
            **(common | {"contact_settings": changed}),
        )


def test_general_s3_rejects_initial_penetration_and_missing_controls():
    model, definition, settings = _case()
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3):
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    with pytest.raises(ValueError, match="initial penetration"):
        GeneralS3PairManager(model, definition, settings, initial)
    absent = ContactSolverSettings(
        pair_motion="general", normal_discretization="pointwise_penalty",
    )
    report = validate_general_contact(model, definition, absent)
    assert not report.valid
    assert any("general_s3_difference_step" in item for item in report.issues)
    outward = ContactDefinition(
        surfaces=(ContactSurface("source", (1,), "top"),
                  ContactSurface("target", (2,), "top")),
        rigid_bodies=(), interactions=definition.interactions,
    )
    report = validate_general_contact(model, outward, settings)
    assert not report.valid
    assert any("faces away" in item for item in report.issues)
