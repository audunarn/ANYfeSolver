"""Opt-in Q4/S3 differentiated patch through the ordinary static owner."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3


def _settings(settings):
    return replace(
        settings,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        maximum_q4_source_integration_triangles=128,
    )


def _check_restricted_reference_and_reaction_rows(model, candidate, displacement,
                                                 tangent, old_tangent, transform):
    # The old FD route restricts both residual rows and variation columns.
    # Jet2 retains prescribed reaction rows: compare the common free operator,
    # then verify those additional rows against actual force variations.
    np.testing.assert_allclose((transform.T @ tangent @ transform).toarray(),
        (transform.T @ old_tangent @ transform).toarray(), rtol=3e-3, atol=3e-4)
    direction = np.zeros(candidate.n_dof)
    for node in range(1, 7):
        direction[model.mesh.dof_manager.get_node_dofs(node)[2]] = 1.
    action = np.asarray(tangent @ direction)
    free = np.flatnonzero(np.asarray(abs(transform).sum(axis=1)).reshape(-1))
    prescribed = np.setdiff1d(np.arange(candidate.n_dof), free)
    assert np.linalg.norm(action[prescribed]) > 1e-6
    for step in (1e-6, 3e-7, 1e-7):
        plus, _, _ = candidate.evaluate(displacement+step*direction, tangent=False)
        minus, _, _ = candidate.evaluate(displacement-step*direction, tangent=False)
        finite_difference = (plus-minus)/(2*step)
        assert np.linalg.norm(finite_difference-action)/np.linalg.norm(action) <= 1e-5


def test_q4_s3_jet2_mode_requires_explicit_derivative_resolution():
    model, definition, settings = _two_q4_two_s3()
    missing = replace(settings, general_s3_differentiation="jet2_q4_s3_dev")
    assert not validate_general_contact(
        model, definition, missing,
    ).valid
    with pytest.raises(ValueError, match="four explicit derivative tolerances"):
        GeneralS3PatchManager(model, definition, missing)
    assert validate_general_contact(model, definition, _settings(settings)).valid


def test_q4_s3_jet2_static_work_matches_existing_smooth_patch():
    model, definition, settings = _two_q4_two_s3()
    reference = GeneralS3PatchManager(model, definition, settings)
    candidate = GeneralS3PatchManager(model, definition, _settings(settings))
    free = [model.mesh.dof_manager.get_node_dofs(node)[axis]
            for node in range(1, 7) for axis in (0, 2)]
    transform = sparse.eye(candidate.n_dof, format="csr")[:, free]
    reference.restrict_tangent_to_reduced_variations(transform)
    candidate.restrict_tangent_to_reduced_variations(transform)
    displacement = np.zeros(candidate.n_dof)
    for node in range(1, 7):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    old_force, old_tangent, old_state = reference.evaluate(displacement)
    force, tangent, state = candidate.evaluate(displacement)
    assert state.active_pressure_dofs == old_state.active_pressure_dofs == 6
    np.testing.assert_allclose(force, old_force, rtol=3e-4, atol=1e-6)
    _check_restricted_reference_and_reaction_rows(
        model, candidate, displacement, tangent, old_tangent, transform)
    assert state.energy_evaluations < old_state.energy_evaluations


@pytest.mark.parametrize(("press_shift", "inactive_faces"), (
    (-0.01505, 0), (-0.01495, 1),
))
def test_q4_s3_jet2_partial_shared_pressure_work(press_shift, inactive_faces):
    model, definition, settings = _two_q4_two_s3()
    reference = GeneralS3PatchManager(model, definition, settings)
    candidate = GeneralS3PatchManager(model, definition, _settings(settings))
    free = [model.mesh.dof_manager.get_node_dofs(node)[axis]
            for node in range(1, 7) for axis in (0, 2)]
    transform = sparse.eye(candidate.n_dof, format="csr")[:, free]
    reference.restrict_tangent_to_reduced_variations(transform)
    candidate.restrict_tangent_to_reduced_variations(transform)
    displacement = np.zeros(candidate.n_dof)
    for node in range(1, 7):
        position = model.mesh.get_node(node)
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = (
            press_shift + 0.002*(position.x-0.6)
        )
    old_force, old_tangent, old_state = reference.evaluate(displacement)
    force, tangent, state = candidate.evaluate(displacement)
    assert 0 < state.active_pressure_dofs < 6
    assert state.active_pressure_dofs == old_state.active_pressure_dofs
    assert sum(not np.any(np.asarray(state.numerical_pressure_pa)[face])
               for face in candidate.law.source_face_indices) == inactive_faces
    np.testing.assert_allclose(force, old_force, rtol=3e-4, atol=1e-6)
    _check_restricted_reference_and_reaction_rows(
        model, candidate, displacement, tangent, old_tangent, transform)


def test_q4_s3_manager_reuses_its_candidate_geometry(monkeypatch):
    model, definition, settings = _two_q4_two_s3()
    manager = GeneralS3PatchManager(model, definition, _settings(settings))
    displacement = np.zeros(manager.n_dof)
    for node in range(1, 7):
        position = model.mesh.get_node(node)
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = (
            -0.01495 + 0.002*(position.x-0.6)
        )
    source, target = manager._positions(displacement)
    prepared = manager.law.prepare_work_state(
        source, target, manager.normal_multipliers,
    )
    ordinary = manager.law.work(source, target, manager.normal_multipliers)
    reused = manager.law.work(
        source, target, manager.normal_multipliers,
        _prepared_state=prepared,
    )
    np.testing.assert_array_equal(reused.residual, ordinary.residual)
    np.testing.assert_array_equal(reused.tangent, ordinary.tangent)
    changed_multipliers = manager.normal_multipliers.copy()
    changed_multipliers[0] += 0.1
    with pytest.raises(ValueError, match="does not match this candidate"):
        manager.law.work(
            source, target, changed_multipliers, _prepared_state=prepared,
        )
    changed_source = source.copy()
    changed_source[0, 2] += 1.0e-6
    with pytest.raises(ValueError, match="does not match this candidate"):
        manager.law.work(
            changed_source, target, manager.normal_multipliers,
            _prepared_state=prepared,
        )
    changed_target = target.copy()
    changed_target[0, 2] += 1.0e-6
    with pytest.raises(ValueError, match="does not match this candidate"):
        manager.law.work(
            source, changed_target, manager.normal_multipliers,
            _prepared_state=prepared,
        )
    other = GeneralS3PatchManager(model, definition, _settings(settings))
    with pytest.raises(ValueError, match="does not match this candidate"):
        other.law.work(
            source, target, other.normal_multipliers,
            _prepared_state=prepared,
        )

    original = manager.law.geometry.evaluate
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(manager.law.geometry, "evaluate", counted)
    pair_calls = [0]*len(manager.law.geometry.pair_laws)
    for index, law in enumerate(manager.law.geometry.pair_laws):
        original_pair = law.evaluate

        def counted_pair(*args, _index=index, _original=original_pair, **kwargs):
            pair_calls[_index] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(law, "evaluate", counted_pair)
    manager.evaluate(displacement)
    assert calls == 1
    assert pair_calls == [1]*len(pair_calls)


@pytest.mark.parametrize("corrupt_value", (0.01, np.nan))
def test_q4_s3_shared_support_rejects_corrupt_aggregate(monkeypatch, corrupt_value):
    model, definition, settings = _two_q4_two_s3()
    candidate = GeneralS3PatchManager(model, definition, _settings(settings))
    displacement = np.zeros(candidate.n_dof)
    for node in range(1, 7):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    geometry = candidate.law.geometry
    original = geometry.evaluate

    def corrupt_aggregate(*args):
        state = original(*args)
        support = state.reference_support.copy()
        support[0] = (support[0] + corrupt_value
                      if np.isfinite(corrupt_value) else np.nan)
        return replace(state, reference_support=support)

    monkeypatch.setattr(geometry, "evaluate", corrupt_aggregate)
    with pytest.raises(ValueError, match="shared pressure support"):
        candidate.evaluate(displacement, tangent=False)


def test_q4_s3_jet2_runs_coupled_static_path_and_restart():
    model, definition, settings = _two_q4_two_s3()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=0.005,
    ),))
    settings = replace(_settings(settings), maximum_augmentations=8)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
        steps_per_segment=1,
    )
    options = dict(contact=definition, contact_settings=settings,
                   contact_path_program=program, max_iterations=12,
                   emit_restart_checkpoint=True)
    result = solve_static_nonlinear(model, **options)
    assert result.status == "completed", result.failure_reason
    assert result.info["general_contact"]["gap_tolerance_met"]
    assert result.info["general_contact"]["active_pressure_dofs"] == 6
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=result.to_restart_checkpoint(), **options,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               atol=1.0e-10, rtol=0.0)
