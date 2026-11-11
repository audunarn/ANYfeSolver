"""A stopped development run must retain its mechanical non-success."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import general_contact_d1_interior_press_dev as runner


def test_moving_clamp_controls_every_target_left_node():
    arguments = dict(source_columns=4, source_rows=4, target_columns=4,
                     target_rows=4, source_support="perimeter", slide_body="target",
                     target_reference_full_left_clamp=True, four_phase=True)
    fixed, _, _, fixed_path = runner.build_interior_press(**arguments)
    moving, _, _, moving_path = runner.build_interior_press(
        **arguments, target_left_edge_motion=True)
    assert len(fixed_path.controls) == 2
    assert len(moving_path.controls) == 2
    middle = next(b for b in moving.boundary_conditions
                  if b.name == "target_reference_middle_left_clamp")
    # Every middle clamp coordinate except lateral motion is still fixed.
    original = next(b for b in fixed.boundary_conditions
                    if b.name == middle.name)
    assert "ux" in original.dof_constraints
    assert "ux" not in middle.dof_constraints
    assert set(middle.dof_constraints) == {"uy", "uz", "rx", "ry", "rz"}
    assert moving_path.waypoints[0] == (0.,) * 2
    assert moving_path.waypoints[2] == (.05,) * 2
    assert len(moving.constraint_equations) == 3
    for equation in moving.constraint_equations:
        assert equation.value == 0.
        assert [coefficient for _, coefficient in equation.coefficients] == [1., -1.]
    from scipy import sparse
    from anyfesolver.assembly import build_constraint_transformation
    moving.apply_boundary_conditions()
    size = moving.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(
        sparse.eye(size, format="csr"), np.zeros(size), moving)
    reduced = np.zeros(transform.shape[1])
    for control in moving_path.controls:
        node = control.terms[0][0]
        dof = moving.mesh.dof_manager.get_node_dofs(node)[0]
        row = transform.getrow(dof)
        assert len(row.indices) == 1
        reduced[row.indices[0]] = .05/row.data[0]
    motion = transform @ reduced + offset
    for node in (26, 31, 36, 41, 46):
        dofs = moving.mesh.dof_manager.get_node_dofs(node)
        assert motion[dofs[0]] == .05
        np.testing.assert_array_equal(motion[dofs[1:]], np.zeros(5))


def test_stopped_result_is_saved_and_null_checkpoint_cannot_restart(tmp_path, monkeypatch):
    def stopped(model, **kwargs):
        assert kwargs["record_increment_snapshots"] is True
        displacement = np.zeros(model.mesh.dof_manager.total_dofs)
        def unavailable():
            raise AssertionError("must not request an unavailable checkpoint")
        return SimpleNamespace(
            status="stopped", failure_reason="injected geometry non-success",
            steps=(), snapshots=(SimpleNamespace(displacements=displacement),),
            displacements=displacement, restart_checkpoint=None,
            to_restart_checkpoint=unavailable,
            info={"contact_failure": {"kind": "geometry", "feature": 197}})

    monkeypatch.setattr(runner, "solve_static_nonlinear", stopped)
    arguments = dict(
        force_n=.04, reference_slope=1e-4, target_columns=4, target_rows=4,
        source_support="perimeter", target_young_multiplier=5., target_y0_m=-.1,
        source_reference_downshift_m=0., source_columns=4, source_rows=4,
        load_shape="physical_bump", source_y_profile=None,
        offline_structural_reference_y16=False, max_path_fraction=1.,
        four_phase=True, steps_per_segment=2, slide_distance_m=.05,
        slide_intervals=1, slide_body="target", target_reference_full_left_clamp=True,
        shared_edge_tie=False, capture_swept_failure=False, record_swept_fractions=False,
        q4_gap_certificate="overlap_local_dev", record_accepted_poses=True)
    output = tmp_path / "stopped.json"
    runner.main(output, **arguments)
    record = json.loads(output.read_text())
    assert record["status"] == "stopped"
    assert record["failure_reason"] == "injected geometry non-success"
    assert record["contact_failure"] == {"kind": "geometry", "feature": 197}
    assert record["accepted_pose_displacements"]
    assert record["restart_checkpoint"] is None
    assert record["restart_checkpoint_available"] is False
    with pytest.raises(ValueError, match="no committed checkpoint"):
        runner.main(tmp_path / "invalid_restart.json",
                    restart_checkpoint_path=output, **arguments)


def _fake_radial_fixture(monkeypatch):
    monkeypatch.setattr(runner, "build_radial_guided_case", lambda **kwargs:
        (object(), object(), object(), object(), object(), {"fixture": "recording_only"}))


def test_radial_solver_exception_retains_input_identity(tmp_path, monkeypatch):
    _fake_radial_fixture(monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError("injected geometry failure")
    monkeypatch.setattr(runner, "solve_static_nonlinear", fail)
    output = tmp_path / "radial_failure.json"
    with pytest.raises(ValueError, match="injected geometry failure"):
        runner.radial_solve(output)
    record = json.loads(output.read_text())
    assert record["status"] == "solver_exception"
    assert record["changed_kernel_sha256"]["semismooth_path.py"]
    assert record["restart_checkpoint"] is None


def test_radial_postprocessing_failure_retains_mechanical_result(tmp_path, monkeypatch):
    _fake_radial_fixture(monkeypatch)
    result = SimpleNamespace(status="completed", failure_reason=None, steps=(object(),),
        snapshots=(SimpleNamespace(displacements=np.array([1.])),),
        displacements=np.array([1.]), restart_checkpoint={"accepted": True}, info={})
    monkeypatch.setattr(runner, "solve_static_nonlinear", lambda *args, **kwargs: result)
    output = tmp_path / "radial_postprocessing_failure.json"
    with pytest.raises(IndexError):
        runner.radial_solve(output)
    record = json.loads(output.read_text())
    assert record["status"] == "completed"
    assert record["accepted_steps"] == 1
    assert record["displacements"] == [1.]
    assert record["accepted_pose_displacements"] == [[1.]]
    assert "force_relative_error" not in record


def test_invalid_rejected_geometry_is_reported_without_masking_failure():
    from anyfesolver.general_contact.semismooth_path import _proposal_gap_diagnostic
    def invalid(*args):
        raise ValueError("folded rejected proposal")
    manager = SimpleNamespace(_positions=lambda u: u, _gap_lower=invalid)
    diagnostic = _proposal_gap_diagnostic(manager, np.zeros(1))
    assert diagnostic["swept_proposed_gap_lower_m"] is None
    assert "folded rejected proposal" in diagnostic["swept_proposed_geometry_error"]
