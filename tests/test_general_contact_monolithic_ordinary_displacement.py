"""Bounded ordinary displacement contact and its physical force reference."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import solve_static_nonlinear
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.nonlinear_static import DisplacementControl
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from anyfesolver.general_contact.static_path import (
    ContactPathControl, ContactPathProgram,
    _elastic_displacement_force_predictor, solve_static_contact_path,
)
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _case():
    model, definition, settings, _program = build_case(4)
    model.add_boundary_condition(BoundaryCondition(
        "source_x_support", (1, 6), {"ux": 0.0},
    ))
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=0.01,
        monolithic_linear_solver="block_response_dev",
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
    )
    control = DisplacementControl(
        node_id=5, dof="uz", target_displacement=-8.0e-5,
    )
    return model, definition, settings, control


def _clipped_linear_mass(knots, upper):
    """Exact two-node Q1 mass over this fixture's left-aligned overlap."""
    matrix = np.zeros((len(knots), len(knots)))
    for cell in range(len(knots)-1):
        left, right = knots[cell:cell+2]
        reach = min(right, upper)
        if reach <= left:
            continue
        length = right-left
        t = (reach-left)/length
        block = length*np.array([
            [t-t*t+t*t*t/3, t*t/2-t*t*t/3],
            [t*t/2-t*t*t/3, t*t*t/3],
        ])
        matrix[cell:cell+2, cell:cell+2] += block
    return matrix


def _independent_small_motion_oracle():
    """Use model linear K and analytic contact work on the reference footprint."""
    model, definition, _settings, control = _case()
    assert definition.surfaces[0].element_ids == (1, 2, 3, 4)
    assert definition.surfaces[1].element_ids == (5, 6)
    model.apply_boundary_conditions()
    stiffness, _info = assemble_stiffness_matrix(model)
    ndof = model.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(ndof), model,
    )
    assert np.linalg.norm(offset) == 0.0
    source_x = np.array([model.mesh.nodes[node].x for node in range(1, 6)])
    source_y = np.array([model.mesh.nodes[node].y for node in (1, 6)])
    assert np.all(np.diff(source_x) > 0.0)
    assert np.all(np.diff(source_y) > 0.0)
    assert all(model.mesh.nodes[node].x == source_x[(node-1) % 5]
               for node in range(1, 11))
    assert all(model.mesh.nodes[node].y == source_y[(node-1) // 5]
               for node in range(1, 11))
    target_x = max(model.mesh.nodes[node].x for node in range(11, 15))
    target_y = max(model.mesh.nodes[node].y for node in range(11, 15))
    assert min(model.mesh.nodes[node].x for node in range(11, 15)) <= source_x[0]
    assert min(model.mesh.nodes[node].y for node in range(11, 15)) <= source_y[0]
    source_element = model.mesh.elements[1]
    target_element = model.mesh.elements[5]
    np.testing.assert_allclose(source_element.physical_reference_director,
                               (0.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(target_element.physical_reference_director,
                               (0.0, 0.0, 1.0), atol=1e-12)
    for node in range(1, 11):
        for dof in model.mesh.dof_manager.get_node_dofs(node)[3:6]:
            assert transform[dof, :].nnz == 0
    for node in range(11, 15):
        for dof in model.mesh.dof_manager.get_node_dofs(node):
            assert transform[dof, :].nnz == 0
    reference_gap = (
        model.mesh.nodes[1].z-0.5*source_element.thickness
        -model.mesh.nodes[11].z
        -(0.5*target_element.thickness
          -target_element.reference_surface_offset)
    )
    assert reference_gap == pytest.approx(40e-6, abs=1e-12)
    mass = np.kron(_clipped_linear_mass(source_y, target_y),
                   _clipped_linear_mass(source_x, target_x))
    support = mass.sum(axis=1)
    assert np.min(support) > 0.0
    gap_work = np.zeros((len(support), ndof))
    source_z = [model.mesh.dof_manager.get_node_dofs(node)[2]
                for node in range(1, 11)]
    gap_work[:, source_z] = mass
    reduced_stiffness = (transform.T @ stiffness @ transform).toarray()
    reduced_gap_work = np.asarray(transform.T @ gap_work.T).T
    control_row = np.zeros(ndof)
    control_row[model.mesh.dof_manager.get_node_dofs(control.node_id)[2]] = 1.0
    reduced_control = np.asarray(control_row @ transform).reshape(1, -1)

    def equilibrate(active):
        active_gap_work = reduced_gap_work[active]
        size = reduced_stiffness.shape[0]
        count = int(active.sum())
        matrix = np.block([
            [reduced_stiffness, -active_gap_work.T, reduced_control.T],
            [active_gap_work, np.zeros((count, count)),
             np.zeros((count, 1))],
            [reduced_control, np.zeros((1, count)), np.zeros((1, 1))],
        ])
        rhs = np.r_[np.zeros(size), -reference_gap*support[active],
                    control.target_displacement]
        solved = np.linalg.solve(matrix, rhs)
        coordinates = solved[:size]
        pressure = np.zeros(len(support))
        pressure[active] = solved[size:size+count]
        return coordinates, pressure, float(solved[-1])

    active = np.zeros(len(support), dtype=bool)
    for _ in range(20):
        coordinates, pressure, multiplier = equilibrate(active)
        normalized_gap = reference_gap + (reduced_gap_work @ coordinates)/support
        updated = (pressure - definition.interactions[0].normal_stiffness
                   * normalized_gap) > 0.0
        if np.array_equal(updated, active):
            break
        active = updated
    else:
        raise AssertionError("independent Q1 active set did not settle")
    assert int(active.sum()) == 1
    assert np.min(pressure) >= -1e-12
    assert np.min(normalized_gap) >= -1e-12
    assert np.max(np.abs(pressure*normalized_gap)) < 1e-12
    equilibrium = (reduced_stiffness @ coordinates
                   - reduced_gap_work.T @ pressure
                   + reduced_control.T.ravel()*multiplier)
    assert np.linalg.norm(equilibrium) < 1e-10
    displacement = np.asarray(transform @ coordinates).reshape(-1)
    physical_contact_force = gap_work.T @ pressure
    return (pressure, displacement, float(support @ pressure), multiplier,
            float(physical_contact_force @ displacement))


def test_contact_free_elastic_force_predictor_matches_diagonal_spring():
    stiffness = sparse.diags([2.0, 3.0], format="csr")
    transform = sparse.eye(2, format="csr")
    reaction = _elastic_displacement_force_predictor(
        stiffness, transform, np.array([1.0, 0.0]), 0.5,
    )
    assert reaction == pytest.approx(1.0, rel=1e-13)


def test_contact_free_elastic_force_predictor_sums_two_control_reactions():
    stiffness = sparse.diags([2.0, 3.0], format="csr")
    transform = sparse.eye(2, format="csr")
    reaction = _elastic_displacement_force_predictor(
        stiffness, transform, np.eye(2),
        ((0.5, 0.25), (0.25, -0.5)),
    )
    # The second waypoint has the larger absolute reaction sum.
    assert reaction == pytest.approx(2.0, rel=1e-13)


def test_ordinary_displacement_contact_and_split_restart():
    model, definition, settings, control = _case()
    full = solve_static_nonlinear(
        model, control="displacement", displacement_control=control,
        num_steps=4, contact=definition, contact_settings=settings,
        max_iterations=18, emit_restart_checkpoint=True,
        record_increment_snapshots=True,
    )
    assert full.status == "completed", (
        full.failure_reason, full.info.get("contact_failure"),
    )
    assert [snapshot.step_index for snapshot in full.snapshots] == [1, 2, 3, 4]
    assert full.snapshots[-1].control_value == control.target_displacement
    assert full.info["contact_free_elastic_force_predictor_n"] >= (
        settings.complementarity_force_scale_n
    )
    assert full.info["general_contact"]["active_pressure_dofs"] > 0, (
        full.info["general_contact"]["minimum_gap"],
        full.info["contact_free_elastic_force_predictor_n"],
    )
    assert full.info["general_contact"]["gap_tolerance_met"]
    source_inplane = [abs(full.displacements[dof])
                      for node in range(1, 11)
                      for dof in model.mesh.dof_manager.get_node_dofs(node)[:2]]
    # This analytic oracle freezes the planar footprint, so bound that approximation.
    assert max(source_inplane) < 1e-8

    (oracle_pressure, oracle_displacement, oracle_force, oracle_control,
     oracle_work) = _independent_small_motion_oracle()
    reported = full.info["general_contact"]
    assert abs(reported["integration_force"][2]-oracle_force)/oracle_force < 1e-6
    assert (np.linalg.norm(np.asarray(reported["numerical_pressure_pa"])
                           - oracle_pressure)/np.linalg.norm(oracle_pressure)
            < 1e-6)
    assert (np.linalg.norm(full.displacements-oracle_displacement)
            / np.linalg.norm(oracle_displacement) < 1e-4)
    production_control = full.info["force_displacement_history"][-1][
        "reaction_generalized"][0]
    assert abs(production_control+oracle_control)/abs(oracle_control) < 1e-6
    manager = GeneralS3PatchManager(model, definition, settings)
    normal_forces = np.asarray(full.to_restart_checkpoint()["path_state"][
        "normal_forces"])
    contact = linearize(manager, full.displacements, normal_forces,
                        tangent=False)
    mapped_work = float(-contact.force @ full.displacements)
    assert abs(mapped_work-oracle_work)/abs(oracle_work) < 1e-6

    scaled = solve_static_nonlinear(
        model, control="displacement", displacement_control=control,
        num_steps=4, contact=definition,
        contact_settings=replace(settings, complementarity_force_scale_n=0.02),
        max_iterations=18,
    )
    assert scaled.status == "completed", scaled.failure_reason
    np.testing.assert_allclose(scaled.displacements, full.displacements,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        scaled.info["general_contact"]["integration_force"],
        full.info["general_contact"]["integration_force"],
        rtol=0, atol=1e-8,
    )

    half = solve_static_nonlinear(
        model, control="displacement", displacement_control=control,
        num_steps=4, max_load_factor=0.5,
        contact=definition, contact_settings=settings,
        max_iterations=18, emit_restart_checkpoint=True,
    )
    assert half.status == "completed" and len(half.steps) == 2
    resumed = solve_static_nonlinear(
        model, control="displacement", displacement_control=control,
        num_steps=4, contact=definition, contact_settings=settings,
        max_iterations=18, restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed"
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            model, control="displacement",
            displacement_control=replace(control, target_displacement=-7.5e-5),
            num_steps=4, contact=definition, contact_settings=settings,
            max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )


def test_ordinary_displacement_rejects_unphysical_force_scale(monkeypatch):
    model, definition, settings, control = _case()
    # The monolithic route keeps the strict predictor default; the spy
    # observes allow_zero=False while the original refusal is retained.
    import anyfesolver.general_contact.static_path as path_driver
    real = path_driver._elastic_displacement_force_predictor
    seen = []
    def spy(*args, **kwargs):
        seen.append(kwargs.get("allow_zero", False))
        return real(*args, **kwargs)
    monkeypatch.setattr(path_driver, "_elastic_displacement_force_predictor", spy)
    with pytest.raises(ValueError, match="exceeds the contact-free elastic"):
        solve_static_nonlinear(
            model, control="displacement", displacement_control=control,
            num_steps=2, contact=definition,
            contact_settings=replace(settings, complementarity_force_scale_n=1.0e9),
        )
    assert seen == [False]


@pytest.mark.parametrize("program", [
    ContactPathProgram(
        (ContactPathControl(((5, "uz", 2.0),)),),
        ((0.0,), (-8.0e-5,)),
    ),
    ContactPathProgram(
        (ContactPathControl(((5, "uz", 1.0), (4, "uz", 1.0))),),
        ((0.0,), (-8.0e-5,)),
    ),
    ContactPathProgram(
        (ContactPathControl.node(5, "uz"),),
        ((0.0,), (-4.0e-5,), (-8.0e-5,)),
    ),
])
def test_one_control_path_rejects_broader_unverified_programs(program):
    model, definition, settings, _control = _case()
    with pytest.raises(ValueError, match="one-control contact path requires"):
        solve_static_contact_path(model, definition, settings, program)
