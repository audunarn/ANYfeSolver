"""Bounded supported-pressure proposals in ordinary force control."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.control import CancellationToken
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.general_contact import linearized_pivot
from anyfesolver.general_contact.core import contact_scope
from anyfesolver.general_contact.supported_schur import SupportedSchurProposal
from scripts.general_contact_d1_finite_area_load import build_case


def test_schur_checkpoint_rejects_self_healing_token_before_element_work(monkeypatch):
    model, load, definition, settings = build_case(2, supported_schur=True)
    element = model.mesh.elements[1]
    token = CancellationToken()
    checkpoints = []
    response_hits = []

    def hostile_response(*_args, **_kwargs):
        response_hits.append(1)
        raise AssertionError("a checkpoint mutation must not reach element work")

    class SelfHealingFlag:
        def __bool__(self):
            checkpoints.append(1)
            # This deliberately bypasses counted instance mutation. The flag
            # is repaired before return, so trust must be checked beforehand.
            object.__setattr__(element, "compute_nonlinear_response", hostile_response)
            token._event._flag = False
            return False

    def arm_at_schur(*_args, check_cancel, **_kwargs):
        token._event._flag = SelfHealingFlag()
        check_cancel()
        return None

    monkeypatch.setattr(linearized_pivot, "refined_patch_supported_schur", arm_at_schur)
    try:
        with pytest.raises(ElementCapabilityError, match="augmentation Schur cancellation"):
            solve_static_nonlinear(
                model, load, contact=definition, contact_settings=settings,
                control="force", kinematics="von_karman", num_steps=1,
                max_load_factor=0.5, max_iterations=12,
                min_step_fraction=0.5, cancellation_token=token,
            )
        assert checkpoints == [1]
        assert response_hits == []
    finally:
        element.__dict__.pop("compute_nonlinear_response", None)
        token._event._flag = False


@pytest.mark.parametrize("divisions", (2, 8))
def test_central_uniform_pressure_reproduces_analytic_integrated_force(divisions):
    model, _load, definition, settings = build_case(
        divisions, total_load_n=2.0, normal_stiffness=100_000.0,
        geometry_profile="central",
    )
    with contact_scope(model, definition, settings, allow_augmented=True) as manager:
        displacement = np.zeros(manager.n_dof)
        for node in range(1, (divisions + 1)**2 + 1):
            displacement[model.mesh.get_node(node).dofs[2]] = -75.0e-6
        _force, _tangent, summary = manager.evaluate(displacement, tangent=True)
    # A rigid translation closes the initial 50 um skin clearance by 25 um.
    # P1/Q1 partition of unity gives pressure times the exact target area.
    expected = 100_000.0 * 25.0e-6 * (0.77 - 0.24) * (0.73 - 0.21)
    np.testing.assert_allclose(summary.integration_force[2], expected,
                               rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(summary.target_force[2], -expected,
                               rtol=1.0e-9, atol=1.0e-9)
    assert summary.gap_tolerance_met
    assert summary.active_pressure_dofs > 0


def test_ordinary_force_control_verifies_schur_before_committing():
    model, load, definition, settings = build_case(2, supported_schur=True)
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=1,
        max_load_factor=0.5, max_iterations=12,
        min_step_fraction=0.5, emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    assert len(result.steps) == 1
    attempts = result.info["contact_augmentation_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["equilibrium_converged"]
    trace = attempts[0]["trace"]
    assert any(row.get("schur_proposed") for row in trace)
    assert any(row.get("schur_verified") for row in trace)
    assert trace[-1]["normal_multiplier_change_pa"] <= settings.pressure_tolerance
    contact = result.info["general_contact"]
    assert contact["gap_tolerance_met"]
    assert contact["active_pressure_dofs"] > 0
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"])
        + np.asarray(contact["target_force"]),
        0.0, atol=1.0e-8,
    )
    assert result.restart_checkpoint is not None


def test_ordinary_schur_rejects_follower_load_before_solving():
    model, load, definition, settings = build_case(2, supported_schur=True)
    load.follower_pressure = True
    with pytest.raises(ValueError, match="one dead nodal load case"):
        solve_static_nonlinear(
            model, load, contact=definition, contact_settings=settings,
            control="force", kinematics="von_karman", num_steps=1,
        )


def test_ordinary_schur_rejects_penalty_only_before_solving():
    model, load, definition, settings = build_case(2, supported_schur=True)
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="penalty"),
    ))
    with pytest.raises(ValueError, match="one augmented contact interaction"):
        solve_static_nonlinear(
            model, load, contact=definition, contact_settings=settings,
            control="force", kinematics="von_karman", num_steps=1,
        )


def test_ordinary_schur_factorization_failure_falls_back(monkeypatch):
    model, load, definition, settings = build_case(2, supported_schur=True)
    monkeypatch.setattr(
        linearized_pivot, "factorize",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            solve=lambda _rhs: (_ for _ in ()).throw(
                RuntimeError("injected singular response")
            ),
        ),
    )
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=1,
        max_load_factor=0.5, max_iterations=12, min_step_fraction=0.5,
    )
    assert any(
        row.get("schur_response_failure") == "injected singular response"
        for attempt in result.info["contact_augmentation_attempts"]
        for row in attempt["trace"]
    )
    assert any(
        row.get("schur_proposed") is False
        for attempt in result.info["contact_augmentation_attempts"]
        for row in attempt["trace"]
    )


def test_ordinary_schur_rejects_bad_candidate_and_restores_state(monkeypatch):
    model, load, definition, settings = build_case(2, supported_schur=True)

    def bad_proposal(manager, *_args, **_kwargs):
        # Deliberately claim the wrong active branch. The analysis owner must
        # verify it after equilibrium rather than trusting this local claim.
        trial = manager.normal_multipliers.copy()
        source, target = manager._positions(_args[0])
        state = manager.law.evaluate(source, target, trial)
        trial[state.reference_support > 0.0] += 0.5
        return SupportedSchurProposal(
            trial, (0,), None, 1.0, 1.0, 0.1,
        )

    monkeypatch.setattr(linearized_pivot,
                        "refined_patch_supported_schur", bad_proposal)
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=1,
        max_load_factor=0.5, max_iterations=12,
        min_step_fraction=0.5,
    )
    assert result.status != "completed"
    # The quarter-load cutback can accept as an open state. The failed
    # half-load contact trial must leave that last accepted state intact.
    assert len(result.steps) == 1
    assert result.load_factor == 0.25
    assert any(row.get("schur_rejected") for attempt in
               result.info["contact_augmentation_attempts"]
               for row in attempt["trace"])
    assert np.linalg.norm(result.displacements) > 0.0
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        0.0,
    )
