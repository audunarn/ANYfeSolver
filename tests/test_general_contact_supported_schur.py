"""Independent projected-root controls for a many-mode pressure proposal."""

import numpy as np
import pytest
import json
from pathlib import Path

from anyfesolver.general_contact import supported_schur
from anyfesolver.general_contact.supported_schur import propose_supported_schur_step


def test_nineteen_supported_active_modes_take_a_bounded_projected_root_step():
    count = 19
    pressure = np.ones(count)
    gap = -0.1*np.ones(count)
    response = 0.1*np.eye(count)
    proposal = propose_supported_schur_step(
        pressure, gap, np.ones(count, dtype=bool), response,
        normal_stiffness=100.0, maximum_pressure=5.0,
        maximum_pressure_step=2.0, maximum_modes=32,
    )
    assert proposal is not None
    assert len(proposal.active_indices) == count
    np.testing.assert_allclose(proposal.multipliers, 2.0)
    assert proposal.predicted_residual_after < 1.0e-12


def test_projected_derivative_sign_matches_independent_directional_difference():
    pressure = np.array([1.0, 2.0, 0.0])
    gap = np.array([-0.1, -0.2, 0.3])
    response = np.array([[0.2, 0.03, 0.0],
                         [0.01, 0.3, 0.0],
                         [0.0, 0.0, 0.4]])
    direction = np.array([0.4, -0.2, 0.0])

    def residual(value):
        updated_gap = gap + response @ (value-pressure)
        return np.maximum(0.0, value-10.0*updated_gap)-value

    step = 1.0e-6
    derivative = (residual(pressure+step*direction)
                  - residual(pressure-step*direction))/(2.0*step)
    np.testing.assert_allclose(derivative, -10.0*response @ direction,
                               rtol=1.0e-8, atol=1.0e-9)


def test_singular_or_resource_exhausted_schur_does_not_soften_law():
    arguments = (np.ones(3), -np.ones(3), np.ones(3, dtype=bool))
    assert propose_supported_schur_step(
        *arguments, np.zeros((3, 3)), normal_stiffness=1.0,
        maximum_pressure=10.0, maximum_pressure_step=10.0,
        maximum_modes=3,
    ) is None
    with pytest.raises(MemoryError, match="Schur cap"):
        propose_supported_schur_step(
            *arguments, np.eye(3), normal_stiffness=1.0,
            maximum_pressure=10.0, maximum_pressure_step=10.0,
            maximum_modes=2,
        )


def test_rank_revealing_step_only_corrects_responsive_pressure_modes():
    pressure = np.ones(3)
    gap = np.array([-0.1, -0.1, -0.01])
    response = np.diag([0.1, 0.1, 0.0])
    diagnostics = {}
    proposal = propose_supported_schur_step(
        pressure, gap, np.ones(3, dtype=bool), response,
        normal_stiffness=10.0, maximum_pressure=5.0,
        maximum_pressure_step=2.0, maximum_modes=3,
        diagnostics=diagnostics,
    )
    assert proposal is not None
    assert diagnostics["schur_effective_rank"] == 2
    assert diagnostics["schur_truncated_response"] is True
    assert diagnostics["schur_scaled_condition"] is None
    assert proposal.scaled_condition is None
    json.dumps(diagnostics, allow_nan=False)
    np.testing.assert_allclose(proposal.multipliers, [2.0, 2.0, 1.0])
    equilibrated_gap = gap + response @ (proposal.multipliers-pressure)
    actual_residual = (np.maximum(0.0, proposal.multipliers
                                  - 10.0*equilibrated_gap)
                       - proposal.multipliers)
    assert np.linalg.norm(actual_residual, ord=np.inf) < 0.11


def test_bounded_step_below_verification_reduction_is_declined():
    diagnostics = {}
    proposal = propose_supported_schur_step(
        np.array([1.0]), np.array([-0.1]), np.array([True]),
        np.array([[0.1]]), normal_stiffness=10.0,
        maximum_pressure=5.0, maximum_pressure_step=0.05,
        maximum_modes=1, diagnostics=diagnostics,
    )
    assert proposal is None
    assert diagnostics["schur_ineligible"] == (
        "bounded_step_or_predicted_merit_failed"
    )
    assert diagnostics["schur_full_step_max_pa"] == pytest.approx(1.0)
    assert diagnostics["schur_best_fixed_branch_fraction"] == pytest.approx(1.0/32.0)
    assert diagnostics["schur_best_fixed_branch_residual_pa"] == pytest.approx(0.96875)
    assert diagnostics["schur_best_projected_fraction"] == pytest.approx(1.0/32.0)
    assert diagnostics["schur_best_projected_residual_pa"] == pytest.approx(0.96875)
    assert diagnostics["schur_backtrack_rejections"] == {
        "step_cap": 5,
        "negative_pressure": 0,
        "pressure_cap": 0,
        "active_branch": 0,
        "predicted_merit": 4,
    }


def test_projected_active_set_can_drop_a_nonnegative_pressure_mode():
    diagnostics = {}
    proposal = propose_supported_schur_step(
        np.array([0.04, 0.04]), np.array([-0.1, 0.001]),
        np.array([True, True]), np.diag([0.1, 0.01]),
        normal_stiffness=10.0, maximum_pressure=5.0,
        maximum_pressure_step=2.0, maximum_modes=2,
        diagnostics=diagnostics,
    )
    assert proposal is not None
    assert diagnostics["schur_proposal_kind"] == "projected_active_set"
    assert proposal.active_indices == (0,)
    np.testing.assert_allclose(proposal.multipliers, [1.04, 0.0])
    assert proposal.predicted_residual_after < 1.0e-12


def test_branch_candidate_beats_admissible_fixed_branch_half_step():
    # Both projected arguments are initially positive. The fixed-branch
    # Newton direction would make pressure 1 negative, so its admissible
    # half-step reduces the residual by half. Releasing that mode reaches
    # the exact local complementarity root under the same pressure caps.
    diagnostics = {}
    proposal = propose_supported_schur_step(
        np.array([0.2, 0.2]), np.array([-0.03, 0.03]),
        np.array([True, True]), 0.1*np.eye(2),
        normal_stiffness=1.0, maximum_pressure=2.0,
        maximum_pressure_step=1.0, maximum_modes=2,
        diagnostics=diagnostics,
    )
    assert proposal is not None
    assert diagnostics["schur_best_fixed_branch_fraction"] == 0.5
    assert diagnostics["schur_best_fixed_branch_residual_pa"] == pytest.approx(
        0.015,
    )
    assert diagnostics["schur_proposal_kind"] in {
        "projected_active_set", "bounded_semismooth_active_set",
    }
    np.testing.assert_allclose(proposal.multipliers, [0.5, 0.0], atol=1e-9)
    assert proposal.active_indices == (0,)
    assert proposal.predicted_residual_after < 1e-9


def test_bounded_branch_correction_finds_independent_six_mode_root():
    evidence = json.loads((Path(__file__).resolve().parents[1] / "reports" /
        "general_contact_d1_active_set_bounded_diagnostic_20260927.json"
    ).read_text(encoding="utf-8"))
    pressure = np.asarray(evidence["initial_pressure_pa"])
    gap = np.asarray(evidence["initial_normalized_gap_m"])
    response = np.asarray(evidence["equilibrated_gap_response_m_per_pa"])
    diagnostics = {}
    proposal = propose_supported_schur_step(
        pressure, gap, np.ones(len(pressure), dtype=bool), response,
        normal_stiffness=evidence["normal_stiffness_pa_per_m"],
        maximum_pressure=evidence["maximum_pressure_pa"],
        maximum_pressure_step=evidence["maximum_pressure_step_pa"],
        maximum_modes=16, diagnostics=diagnostics,
    )
    assert proposal is not None
    assert diagnostics["schur_proposal_kind"] == "bounded_semismooth_active_set"
    assert proposal.active_indices == (0,)
    assert proposal.predicted_residual_after < 1.0e-4
    assert proposal.predicted_residual_after < 0.75*proposal.predicted_residual_before
    assert proposal.multipliers[0] == pytest.approx(2.82775, rel=1.0e-4)
    assert np.max(proposal.multipliers[1:]) < 1.0e-5


def test_bounded_branch_correction_checks_cancellation():
    evidence = json.loads((Path(__file__).resolve().parents[1] / "reports" /
        "general_contact_d1_active_set_bounded_diagnostic_20260927.json"
    ).read_text(encoding="utf-8"))

    class CancelProbe(Exception):
        pass

    def cancel():
        raise CancelProbe()

    pressure = np.asarray(evidence["initial_pressure_pa"])
    with pytest.raises(CancelProbe):
        propose_supported_schur_step(
            pressure, np.asarray(evidence["initial_normalized_gap_m"]),
            np.ones(len(pressure), dtype=bool),
            np.asarray(evidence["equilibrated_gap_response_m_per_pa"]),
            normal_stiffness=evidence["normal_stiffness_pa_per_m"],
            maximum_pressure=evidence["maximum_pressure_pa"],
            maximum_pressure_step=evidence["maximum_pressure_step_pa"],
            maximum_modes=16, check_cancel=cancel,
        )


def test_sixteen_mode_selects_better_bounded_branch_than_projected_direction():
    evidence = json.loads((Path(__file__).resolve().parents[1] / "reports" /
        "general_contact_d1_16q4_first_response_20260927.json"
    ).read_text(encoding="utf-8"))
    indices = np.asarray(evidence["supported_indices"], dtype=int)
    pressure = np.zeros(indices.max()+1)
    gap = np.zeros_like(pressure)
    supported = np.zeros(len(pressure), dtype=bool)
    pressure[indices] = evidence["initial_pressure_pa"]
    gap[indices] = evidence["initial_normalized_gap_m"]
    supported[indices] = True
    diagnostics = {}
    proposal = propose_supported_schur_step(
        pressure, gap, supported,
        np.asarray(evidence["equilibrated_gap_response_m_per_pa"]),
        normal_stiffness=evidence["normal_stiffness_pa_per_m"],
        maximum_pressure=evidence["maximum_pressure_pa"],
        maximum_pressure_step=evidence["maximum_pressure_step_pa"],
        maximum_modes=16, diagnostics=diagnostics,
    )
    assert proposal is not None
    assert diagnostics["schur_proposal_kind"] == "bounded_semismooth_active_set"
    assert proposal.active_indices == (0,)
    assert proposal.predicted_residual_after < 1.0e-4
    assert (proposal.predicted_residual_after
            < diagnostics["schur_best_projected_residual_pa"])


@pytest.mark.parametrize("failure", (
    ValueError("optimizer failure"),
    np.linalg.LinAlgError("optimizer failure"),
    MemoryError("optimizer failure"),
))
def test_projected_candidate_survives_optional_optimizer_failure(
        monkeypatch, failure):
    evidence = json.loads((Path(__file__).resolve().parents[1] / "reports" /
        "general_contact_d1_16q4_first_response_20260927.json"
    ).read_text(encoding="utf-8"))
    indices = np.asarray(evidence["supported_indices"], dtype=int)
    pressure = np.zeros(indices.max()+1)
    gap = np.zeros_like(pressure)
    supported = np.zeros(len(pressure), dtype=bool)
    pressure[indices] = evidence["initial_pressure_pa"]
    gap[indices] = evidence["initial_normalized_gap_m"]
    supported[indices] = True

    def unavailable(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(supported_schur, "least_squares", unavailable)
    diagnostics = {}
    proposal = propose_supported_schur_step(
        pressure, gap, supported,
        np.asarray(evidence["equilibrated_gap_response_m_per_pa"]),
        normal_stiffness=evidence["normal_stiffness_pa_per_m"],
        maximum_pressure=evidence["maximum_pressure_pa"],
        maximum_pressure_step=evidence["maximum_pressure_step_pa"],
        maximum_modes=16, diagnostics=diagnostics,
    )
    assert proposal is not None
    assert diagnostics["schur_proposal_kind"] == "projected_active_set"
    assert "schur_ineligible" not in diagnostics
    assert proposal.predicted_residual_after < 0.75*proposal.predicted_residual_before
