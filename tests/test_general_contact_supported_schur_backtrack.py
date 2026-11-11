"""Bounded interior pressure proposals from an existing Schur direction."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.supported_schur import (
    SupportedSchurBacktrackData,
    SupportedSchurProposal,
    propose_supported_schur_backtrack,
    propose_supported_schur_secant_backtrack,
    propose_supported_schur_step,
)


def _saved_probe(report_name):
    path = (Path(__file__).resolve().parents[1] / "reports" /
            f"general_contact_d1_free_target_{report_name}_dev_20260928.json")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    modes = np.asarray(evidence["selected_pressure_ids"], dtype=int)
    base = np.asarray(evidence["base_pressure_pa"], dtype=float)
    gap = np.asarray(evidence["base_normalized_gap_m"], dtype=float)
    response = np.zeros((len(modes), len(modes)))
    active = np.asarray(evidence["active_indices"], dtype=int)
    response[:, np.searchsorted(modes, active)] = np.asarray(
        evidence["response_matrix_active_m_per_pa"], dtype=float,
    )
    stiffness = 10000.0
    argument = base[modes]-stiffness*gap
    before = float(np.max(np.abs(np.maximum(0.0, argument)-base[modes])))
    roundoff = 64.0*np.finfo(float).eps*max(
        1.0, 250.0, float(np.max(np.abs(argument))),
    )
    data = SupportedSchurBacktrackData(
        base, modes, gap, response, stiffness, 250.0, 200.0,
        len(modes), roundoff,
    )
    full = base+np.asarray(evidence["full_pressure_step_pa"], dtype=float)
    proposal = SupportedSchurProposal(
        full, tuple(evidence["predicted_active_indices"]), None, 1.0,
        before, evidence["predicted_projected_residual_pa"], data,
    )
    return proposal, evidence


def _saved_sixteen_q4_proposal():
    return _saved_probe("16q4_schur_full_step")[0]


def test_saved_sixteen_q4_map_selects_first_safe_interior_branch_change():
    full = _saved_sixteen_q4_proposal()
    diagnostics = {}
    trial = propose_supported_schur_backtrack(full, diagnostics=diagnostics)
    assert trial is not None
    assert trial.step_fraction == pytest.approx(0.8)
    assert trial.active_indices == (28,)
    assert trial.predicted_residual_after == pytest.approx(
        0.0005848947742124583,
    )
    assert trial.predicted_residual_after <= 0.70*trial.predicted_residual_before
    np.testing.assert_allclose(
        trial.multipliers,
        full.backtrack_data.base_multipliers
        + 0.8*(full.multipliers-full.backtrack_data.base_multipliers),
    )
    assert diagnostics["schur_backtrack_examined"] == 16
    assert trial.backtrack_data is None


def test_saved_twenty_four_q4_curvature_selects_interior_minimum():
    full, evidence = _saved_probe("24q4_schur_full_clip")
    diagnostics = {}
    trial = propose_supported_schur_secant_backtrack(
        full, np.asarray(evidence["actual_normalized_gap_m"], dtype=float),
        diagnostics=diagnostics,
    )
    assert trial is not None
    assert trial.step_fraction == pytest.approx(0.85)
    assert trial.predicted_residual_after == pytest.approx(
        0.00022706416731690287,
    )
    assert trial.active_indices == (20, 45)
    assert trial.predicted_residual_after <= 0.70*trial.predicted_residual_before
    assert diagnostics["schur_secant_backtrack_examined"] == 19
    assert diagnostics["schur_secant_backtrack_eligible"] > 0


def test_saved_sixteen_q4_curvature_declines_but_linear_fallback_remains():
    full, evidence = _saved_probe("16q4_schur_full_step")
    diagnostics = {}
    assert propose_supported_schur_secant_backtrack(
        full, np.asarray(evidence["actual_normalized_gap_m"], dtype=float),
        diagnostics=diagnostics,
    ) is None
    assert diagnostics["schur_secant_backtrack_reason"] == (
        "no_predicted_safe_interior_fraction"
    )
    assert propose_supported_schur_backtrack(full).step_fraction == pytest.approx(0.8)


def test_curvature_helper_rejects_missing_invalid_or_uncertain_inputs():
    full, evidence = _saved_probe("24q4_schur_full_clip")
    actual = np.asarray(evidence["actual_normalized_gap_m"], dtype=float)
    for proposal, gap in (
        (replace(full, backtrack_data=None), actual),
        (full, actual[:-1]),
        (full, np.full_like(actual, np.nan)),
        (replace(full, backtrack_data=replace(
            full.backtrack_data, maximum_pressure_step=0.001,
        )), actual),
        (replace(full, backtrack_data=replace(
            full.backtrack_data, kink_roundoff=1.0,
        )), actual),
    ):
        assert propose_supported_schur_secant_backtrack(proposal, gap) is None


def test_unavailable_or_insufficient_interior_merit_has_no_proposal():
    base = np.array([1.0])
    data = SupportedSchurBacktrackData(
        base, np.array([0]), np.array([-1.0]), np.ones((1, 1)),
        1.0, 3.0, 2.0, 1, 1e-12,
    )
    # The full local move reaches the 70% margin exactly; every interior
    # fraction leaves a larger residual and must not be sent to the owner.
    full = SupportedSchurProposal(np.array([1.3]), (0,), None, 1.0,
                                  1.0, 0.7, data)
    diagnostics = {}
    assert propose_supported_schur_backtrack(full, diagnostics=diagnostics) is None
    assert diagnostics["schur_backtrack_examined"] == 19
    assert diagnostics["schur_backtrack_reason"] == (
        "no_predicted_safe_interior_fraction"
    )
    assert propose_supported_schur_backtrack(replace(full, backtrack_data=None)) is None


def test_caps_fail_closed_before_scanning():
    full = _saved_sixteen_q4_proposal()
    data = full.backtrack_data
    for restricted in (
        replace(data, maximum_pressure=0.005),
        replace(data, maximum_pressure_step=0.001),
        replace(data, maximum_modes=33),
    ):
        diagnostics = {}
        assert propose_supported_schur_backtrack(
            replace(full, backtrack_data=restricted), diagnostics=diagnostics,
        ) is None
        assert diagnostics["schur_backtrack_reason"] == "inconsistent_full_proposal"


def test_uncertain_active_set_kinks_are_not_proposed():
    full = _saved_sixteen_q4_proposal()
    # Force every scanned local argument into the stated uncertainty band.
    guarded = replace(full, backtrack_data=replace(
        full.backtrack_data, kink_roundoff=1.0,
    ))
    diagnostics = {}
    assert propose_supported_schur_backtrack(
        guarded, diagnostics=diagnostics,
    ) is None
    assert diagnostics["schur_backtrack_examined"] == 19


def test_retained_map_is_opt_in_and_preserves_original_proposal():
    kwargs = dict(
        normal_stiffness=100.0, maximum_pressure=5.0,
        maximum_pressure_step=2.0, maximum_modes=2,
    )
    pressure = np.ones(2)
    gap = -0.1*np.ones(2)
    response = 0.1*np.eye(2)
    ordinary = propose_supported_schur_step(
        pressure, gap, np.ones(2, dtype=bool), response, **kwargs,
    )
    retained = propose_supported_schur_step(
        pressure, gap, np.ones(2, dtype=bool), response,
        retain_backtrack=True, **kwargs,
    )
    assert ordinary is not None and retained is not None
    assert ordinary.backtrack_data is None
    assert retained.backtrack_data is not None
    assert not retained.backtrack_data.base_multipliers.flags.writeable
    np.testing.assert_array_equal(retained.multipliers, ordinary.multipliers)
    assert retained.active_indices == ordinary.active_indices
