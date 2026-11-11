"""Opt-in long coupled Q4/S3 physical-gap and restart development case."""

import os

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear, validate_general_contact
from examples.general_contact_q4_s3_tight_path import (
    GAP_BUDGET_M, THICKNESS_M, make_case,
)


def test_q4_s3_tight_case_declares_physical_gap_and_valid_pair():
    model, definition, settings, path = make_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert GAP_BUDGET_M == pytest.approx(1e-3*THICKNESS_M)
    assert definition.interactions[0].gap_tolerance == GAP_BUDGET_M
    assert len(path.waypoints) == 5
    assert len(model.mesh.elements) == 3


@pytest.mark.skipif(
    os.environ.get("ANY_RUN_C1_TIGHT_PATH") != "1",
    reason="opt-in long Q4/S3 four-phase development route",
)
def test_q4_s3_tight_press_slide_open_recontact_and_split_restart():
    model, definition, settings, path = make_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert GAP_BUDGET_M == pytest.approx(1e-3*THICKNESS_M)
    options = dict(contact=definition, contact_settings=settings,
                   contact_path_program=path, max_iterations=15)
    full = solve_static_nonlinear(
        model, emit_restart_checkpoint=True, **options,
    )
    assert full.status == "completed", full.failure_reason
    phases = full.info["force_displacement_history"]
    assert len(phases) == 4
    assert [row["contact"]["active_pressure_dofs"] > 0 for row in phases] == [
        True, True, False, True,
    ]
    assert all(row["contact"]["gap_tolerance_met"] for row in phases)
    assert max(-row["contact"]["minimum_gap"] for row in phases) <= GAP_BUDGET_M
    assert all(phases[i]["contact"]["integration_force"][2] > 0
               for i in (0, 1, 3))
    assert phases[2]["contact"]["integration_force"][2] == 0.0
    assert full.restart_checkpoint["version"] == 2

    half = solve_static_nonlinear(
        model, max_load_factor=0.5, emit_restart_checkpoint=True, **options,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **options,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["integration_force"],
        full.info["general_contact"]["integration_force"],
        rtol=1e-6, atol=1e-8,
    )
    assert resumed.info["general_contact"]["gap_tolerance_met"]
