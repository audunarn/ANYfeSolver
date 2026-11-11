"""Resolved source director work and active-branch Q4/S3 contact blocks."""

import numpy as np
from scipy import sparse

from anyfesolver.general_contact.core import suspend_contact
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import (
    complementarity, linearize, reference_nodal_support,
)
from anyfesolver.general_contact.semismooth_path import (
    _assemble_direct_matrix, _coupled_contact_blocks,
)
from anyfesolver.nonlinear_static import (
    _assemble_nonlinear_system, _discard_nonlinear_state_candidate,
)
from scripts.general_contact_d1_eccentric_source_director import (
    build_fixture, run,
)


def test_eccentric_source_director_moment_has_consistent_both_body_tangent():
    result = run()
    assert result["initial_active_pressure_modes"] == 0
    assert result["initial_minimum_gap_m"] > 0.0
    assert result["active_pressure_modes"] == 6
    assert result["pair_force_balance_n"] < 1.0e-6
    assert result["pair_moment_balance_nm"] < 1.0e-6
    for trial in result["trials"]:
        assert trial["smooth"], trial["branch_checks"]
        assert trial["groups"]["source_moment"]["action_scale"] > 1.0e-4
        assert trial["groups"]["target_moment"]["action_scale"] > 1.0e-4
        assert all(group["relative_error"] <= 1.0e-5
                   for group in trial["groups"].values())


def test_eccentric_active_contact_has_raw_displacement_force_blocks():
    """Check raw contact blocks on a smooth non-equilibrium active branch."""
    model, definition, settings = build_fixture()
    manager = GeneralS3PatchManager(model, definition, settings)
    displacement = np.zeros(manager.n_dof)
    source_director = np.zeros_like(displacement)
    for node in range(1, 7):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        displacement[dofs[2]] = -0.016
        source_director[dofs[4]] = 0.5
    for node in (8, 9):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = 0.001

    reference_support = reference_nodal_support(manager.law)
    nodal_force = 1000.0*reference_support
    force_scale = manager.stiffness*reference_support
    base = linearize(manager, displacement, nodal_force, tangent=True)
    _, active, constraint_u, constraint_z = complementarity(
        base.state, nodal_force, force_scale, base.gap_gradient,
    )
    assert np.all(active)
    assert np.all(base.state.reference_support > 0.0)

    def relative_error(actual, predicted):
        return np.linalg.norm(actual-predicted)/max(
            np.linalg.norm(predicted), 1.0e-12,
        )

    for step in (1.0e-5, 5.0e-6):
        plus = linearize(manager, displacement+step*source_director,
                         nodal_force, tangent=False)
        minus = linearize(manager, displacement-step*source_director,
                          nodal_force, tangent=False)
        for trial in (plus, minus):
            assert trial.state.overlap_signature == base.state.overlap_signature
            np.testing.assert_array_equal(
                trial.state.reference_support > 0.0,
                base.state.reference_support > 0.0,
            )
            assert np.array_equal(
                complementarity(trial.state, nodal_force, force_scale,
                                trial.gap_gradient, tangent=False)[1],
                active,
            )
        force_direction = (plus.force-minus.force)/(2.0*step)
        gap_direction = (
            plus.state.normalized_gap-minus.state.normalized_gap
        )/(2.0*step)
        plus_c = complementarity(plus.state, nodal_force, force_scale,
                                 plus.gap_gradient, tangent=False)[0]
        minus_c = complementarity(minus.state, nodal_force, force_scale,
                                  minus.gap_gradient, tangent=False)[0]
        predicted_force = np.asarray(base.tangent @ source_director)
        for nodes in (range(1, 7), range(7, 11)):
            for offsets, minimum_action in ((range(3), 1.0e-2),
                                            (range(3, 6), 1.0e-6)):
                rows = [model.mesh.dof_manager.get_node_dofs(node)[i]
                        for node in nodes for i in offsets]
                assert np.linalg.norm(predicted_force[rows]) > minimum_action
                assert relative_error(force_direction[rows],
                                      predicted_force[rows]) <= 1.0e-5
        assert relative_error(gap_direction,
                              base.gap_gradient @ source_director) <= 1.0e-5
        assert relative_error((plus_c-minus_c)/(2.0*step),
                              constraint_u @ source_director) <= 1.0e-5

    pressure_direction = np.arange(1, len(nodal_force)+1, dtype=float)
    pressure_direction /= np.linalg.norm(pressure_direction)
    pressure_step = 1.0e-3
    plus = linearize(manager, displacement,
                     nodal_force+pressure_step*pressure_direction,
                     tangent=False)
    minus = linearize(manager, displacement,
                      nodal_force-pressure_step*pressure_direction,
                      tangent=False)
    force_direction = (plus.force-minus.force)/(2.0*pressure_step)
    predicted_force = -base.gap_gradient.T @ pressure_direction
    for nodes in (range(1, 7), range(7, 11)):
        for offsets, minimum_action in ((range(3), 1.0e-2),
                                        (range(3, 6), 1.0e-6)):
            rows = [model.mesh.dof_manager.get_node_dofs(node)[i]
                    for node in nodes for i in offsets]
            assert np.linalg.norm(predicted_force[rows]) > minimum_action
            assert relative_error(force_direction[rows],
                                  predicted_force[rows]) <= 1.0e-5
    plus_c = complementarity(plus.state,
                             nodal_force+pressure_step*pressure_direction,
                             force_scale, plus.gap_gradient, tangent=False)[0]
    minus_c = complementarity(minus.state,
                              nodal_force-pressure_step*pressure_direction,
                              force_scale, minus.gap_gradient, tangent=False)[0]
    np.testing.assert_allclose(
        (plus_c-minus_c)/(2.0*pressure_step),
        constraint_z*pressure_direction, atol=1.0e-10, rtol=0.0,
    )


def test_eccentric_direct_newton_matrix_matches_coupled_residual():
    """Check driver block signs at one active, non-equilibrium Q4/S3 pose."""
    model, definition, settings = build_fixture()
    manager = GeneralS3PatchManager(model, definition, settings)
    displacement = np.zeros(manager.n_dof)
    source_director = np.zeros_like(displacement)
    for node in range(1, 7):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        displacement[dofs[2]] = -0.016
        source_director[dofs[4]] = 0.5
    for node in (8, 9):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = 0.001
    support = reference_nodal_support(manager.law)
    nodal_force = 1000.0*support
    scales = manager.stiffness*support
    states = {}

    def structural(at_displacement, *, tangent):
        try:
            with suspend_contact():
                return _assemble_nonlinear_system(
                    model, at_displacement, states, 5,
                    tangent=tangent, kinematics="von_karman",
                    require_full_coordinates=True,
                )[:2]
        finally:
            _discard_nonlinear_state_candidate(states)

    internal, structural_tangent = structural(displacement, tangent=True)
    contact = linearize(manager, displacement, nodal_force, tangent=True)
    _, active, _, _ = complementarity(
        contact.state, nodal_force, scales, contact.gap_gradient,
    )
    assert np.all(active)
    controlled_dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    control_translation = np.zeros_like(displacement)
    control_translation[controlled_dof] = 1.0
    target_director = np.zeros_like(displacement)
    target_director[model.mesh.dof_manager.get_node_dofs(8)[4]] = 0.5
    transform = sparse.csr_matrix(np.column_stack((
        source_director, control_translation, target_director,
    )))
    kqq, kqz, gap_q = _coupled_contact_blocks(
        structural_tangent, contact, transform,
    )
    q_size = transform.shape[1]
    control_rows = sparse.csr_matrix(
        ([1.0], ([0], [1])), shape=(1, q_size),
    )
    control_reactions = control_rows.T.tocsr()
    matrix = _assemble_direct_matrix(
        kqq, kqz, gap_q, control_reactions, control_rows, scales, active,
    )
    external = np.asarray(transform.T @ (internal+contact.force)).reshape(-1)
    target = np.zeros(1)

    def equations(at_q, reaction, at_force):
        at_displacement = displacement + np.asarray(transform @ at_q).reshape(-1)
        at_internal, _ = structural(at_displacement, tangent=False)
        at_contact = linearize(manager, at_displacement, at_force,
                               tangent=False)
        ncp = complementarity(at_contact.state, at_force, scales,
                              at_contact.gap_gradient, tangent=False)[0]
        return np.r_[
            np.asarray(transform.T @ (at_internal+at_contact.force)).reshape(-1)
            - external
            - np.asarray(control_reactions @ np.array([reaction])).reshape(-1),
            np.asarray(control_rows @ at_q).reshape(-1)-target,
            ncp,
        ]

    pressure_direction = np.arange(1, len(nodal_force)+1, dtype=float)
    pressure_direction /= np.linalg.norm(pressure_direction)
    directions = (
        (np.r_[[1.0, 0.0, 0.0, 0.0], np.zeros_like(nodal_force)], 1.0e-5),
        (np.r_[[0.0, 1.0, 0.0, 0.0], np.zeros_like(nodal_force)], 1.0e-7),
        (np.r_[[0.0, 0.0, 1.0, 0.0], np.zeros_like(nodal_force)], 1.0e-5),
        (np.r_[np.zeros(q_size+1), pressure_direction], 1.0e-3),
        (np.r_[np.zeros(q_size), 1.0, np.zeros_like(nodal_force)],
         1.0e-4),
    )
    for direction, step in directions:
        plus = equations(step*direction[:q_size],
                         step*direction[q_size],
                         nodal_force+step*direction[q_size+1:])
        minus = equations(-step*direction[:q_size],
                          -step*direction[q_size],
                          nodal_force-step*direction[q_size+1:])
        finite = (plus-minus)/(2.0*step)
        predicted = np.asarray(matrix @ direction).reshape(-1)
        for actual, expected in (
            (finite[:q_size], predicted[:q_size]),
            (finite[q_size:q_size+1], predicted[q_size:q_size+1]),
            (finite[q_size+1:], predicted[q_size+1:]),
        ):
            np.testing.assert_allclose(actual, expected, rtol=1.0e-5,
                                       atol=1.0e-8)
