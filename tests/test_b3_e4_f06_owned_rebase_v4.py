"""Synthetic G0 sanity checks for the separate F06 prototype owner."""

import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_g3c_local_shell import deformation
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "f06_v4_numba"))

from scripts.probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4 import (
    OwnedRebasePrototype, main, measure_pilot, physical_tangent,
)
from scripts.probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3 import numeric_state, shell_spatial
from anyfesolver._ge_beam3_shell_joint_state import decode


def _git_history_available():
    import subprocess
    probe = subprocess.run(("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).resolve().parents[1], capture_output=True)
    return probe.returncode == 0


def _physical_shell(owner, state, anchor):
    return deformation(np.asarray(owner.assembly.coordinates), state.shell_u, anchor)


def _close(a, b, label, tolerance=1.e-9):
    x, y = np.asarray(a), np.asarray(b)
    assert x.shape == y.shape, label
    assert np.isfinite(x).all() and np.isfinite(y).all(), label
    assert np.linalg.norm(x-y) <= tolerance*max(1., np.linalg.norm(x), np.linalg.norm(y)), label


def _physical_scale_close(a, b, floor, label, tolerance=1.e-9):
    x, y = np.asarray(a), np.asarray(b)
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), label
    scale = max(floor, float(np.linalg.norm(x)), float(np.linalg.norm(y)))
    assert float(np.linalg.norm(x-y))/scale <= tolerance, label


def _anchored_directional_plateau(prototype, predecessor, state, cursor):
    """Differentiate the complete anchored assembly along physical directions."""
    a = prototype.owner.assembly
    spatial = np.zeros(a.count)
    rows = spatial[:a.shell_count].reshape(-1, 6)
    rows[1] = (.13, -.09, .07, .31, -.23, .17)
    rows[2] = (-.08, .12, -.04, -.19, .29, .11)
    beam_direction = spatial[a.shell_count:-6]
    for index, dof in enumerate(a.beam.free[:9]):
        beam_direction[dof] = .11*np.sin(index+1)
    for index, dof in enumerate(a.beam.free[-8:]):
        beam_direction[dof] = .07*np.cos(index+1)
    spatial[-6:] = (.09, -.06, .04, -.03, .05, -.02)
    assert np.linalg.norm(beam_direction) > 0 and np.linalg.norm(spatial[-6:]) > 0
    spatial /= np.linalg.norm(spatial)
    p = np.eye(a.count)
    for index, row in enumerate(state.shell_u.reshape(-1, 6)):
        p[6*index+3:6*index+6, 6*index+3:6*index+6] = _exp_terms(row[3:])[1]
    chart = np.linalg.solve(p, spatial)
    response, _ = prototype._anchored_evaluate(predecessor, state.mechanical,
        state.shell_u, state.multipliers, cursor)
    expected = physical_tangent(response, state.shell_u, a.count)@spatial
    assert np.linalg.norm(expected/prototype.owner.scale) > 1.e-4
    errors = []
    for size in (2.e-3, 1.e-3, 5.e-4, 2.5e-4):
        forward = a.beam.advance(state.mechanical, size*chart[a.shell_count:-6])
        backward = a.beam.advance(state.mechanical, -size*chart[a.shell_count:-6])
        _, plus = prototype._anchored_evaluate(predecessor, forward,
            state.shell_u+size*chart[:a.shell_count],
            state.multipliers+size*chart[-6:], cursor)
        _, minus = prototype._anchored_evaluate(predecessor, backward,
            state.shell_u-size*chart[:a.shell_count],
            state.multipliers-size*chart[-6:], cursor)
        observed = (plus-minus)/(2*size)
        errors.append(float(np.linalg.norm((observed-expected)/prototype.owner.scale)
            /max(1., np.linalg.norm(expected/prototype.owner.scale))))
    assert all(np.isfinite(errors)), errors
    assert errors[-1] < 1.e-7, errors
    assert errors[-1] <= errors[0]/16 + 1.e-9, errors
    return errors


@pytest.fixture(scope="module")
def base():
    owner = OwnedRebasePrototype()
    run = owner.legacy_prefix()
    assert run.status == "paused" and run.state.cursor == 2
    return owner, run


def test_rebase_restores_and_continues_from_real_loaded_prefix(base, monkeypatch):
    source, prefix = base
    assert source.head_sha256 == __import__("hashlib").sha256(prefix.checkpoint).hexdigest()
    assert source.generation == 0
    frozen_state, frozen_checkpoint = canonical(prefix.state.descriptor()), prefix.checkpoint
    rebased = source.rebase(prefix.checkpoint,
        expected_sha256=__import__("hashlib").sha256(prefix.checkpoint).hexdigest())
    assert rebased.status == "REBASED_AT_LOADED_CURSOR_2"
    assert source.head_sha256 == rebased.checkpoint_sha256 and source.generation == 1
    assert rebased.checkpoint != prefix.checkpoint
    assert canonical(prefix.state.descriptor()) == frozen_state
    assert prefix.checkpoint == frozen_checkpoint
    assert rebased.state.shell_history == prefix.state.shell_history
    assert canonical(rebased.state.beam_histories) == canonical(prefix.state.beam_histories)
    assert np.linalg.norm(prefix.state.shell_u.reshape(-1, 6)[:, 3:]) > 1.e-8
    assert np.linalg.norm(prefix.state.multipliers) > 1.e-8
    assert np.count_nonzero(rebased.state.shell_u.reshape(-1, 6)[:, 3:]) == 0

    fresh = OwnedRebasePrototype()
    def forbid_legacy_replay(*args, **kwargs):
        raise AssertionError("normal F06 restoration replayed V1 history")
    monkeypatch.setattr(fresh.owner, "_restore", forbid_legacy_replay)
    restored = fresh.restore(rebased.checkpoint, expected_sha256=rebased.checkpoint_sha256)
    assert canonical(restored.descriptor()) == canonical(rebased.state.descriptor())
    assert fresh.head_sha256 == rebased.checkpoint_sha256 and fresh.generation == 1
    _anchored_directional_plateau(fresh, restored, restored, 2)
    changed = fresh.continue_loaded(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    assert changed.status == "ACCEPTED_LOADED_CURSOR_3" and changed.state.cursor == 3
    assert fresh.head_sha256 == changed.checkpoint_sha256 and fresh.generation == 2
    with pytest.raises(ValueError, match="stale successor predecessor"):
        fresh.continue_loaded(rebased.checkpoint,
            expected_sha256=rebased.checkpoint_sha256)
    with pytest.raises(ValueError, match="stale restore"):
        fresh.restore(rebased.checkpoint, expected_sha256=rebased.checkpoint_sha256)
    assert fresh.owner.targets[2] != 0.
    direct_successor = OwnedRebasePrototype()
    def forbid_successor_replay(*args, **kwargs):
        raise AssertionError("normal F06 cursor-three restoration replayed trial history")
    monkeypatch.setattr(direct_successor, "_accepted_successor", forbid_successor_replay)
    replayed = direct_successor.restore(changed.checkpoint,
        expected_sha256=changed.checkpoint_sha256)
    assert canonical(replayed.descriptor()) == canonical(changed.state.descriptor())
    assert direct_successor.head_sha256 == changed.checkpoint_sha256
    assert direct_successor.generation == 2
    wrong_generation = __import__("json").loads(changed.checkpoint)
    wrong_generation["successor"]["predecessor_sha256"] = "0"*64
    raw_wrong_generation = canonical(wrong_generation)
    with pytest.raises(ValueError, match="successor generation"):
        OwnedRebasePrototype().restore(raw_wrong_generation,
            expected_sha256=__import__("hashlib").sha256(raw_wrong_generation).hexdigest())
    wrong_history = __import__("json").loads(changed.checkpoint)
    wrong_history["successor"]["state"]["shell_history"]["layer_stress"][0][0] += 1.e-4
    raw_wrong_history = canonical(wrong_history)
    with pytest.raises(ValueError, match="direct successor shell history disagreement"):
        OwnedRebasePrototype().restore(raw_wrong_history,
            expected_sha256=__import__("hashlib").sha256(raw_wrong_history).hexdigest())
    zero_residual = __import__("json").loads(changed.checkpoint)
    zero_residual["successor"]["residual"] = [0.]*len(zero_residual["successor"]["residual"])
    zero_residual["successor"]["metric"] = 0.
    raw_zero_residual = canonical(zero_residual)
    with pytest.raises(ValueError, match="direct successor residual disagreement"):
        OwnedRebasePrototype().restore(raw_zero_residual,
            expected_sha256=__import__("hashlib").sha256(raw_zero_residual).hexdigest())
    audited = OwnedRebasePrototype().audit_legacy_replay(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    assert canonical(audited.descriptor()) == canonical(rebased.state.descriptor())
    successor_audit = OwnedRebasePrototype().audit_legacy_replay(changed.checkpoint,
        expected_sha256=changed.checkpoint_sha256)
    assert canonical(successor_audit.descriptor()) == canonical(changed.state.descriptor())
    _anchored_directional_plateau(fresh, restored, changed.state, 3)

    control = source.owner.solve(checkpoint=prefix.checkpoint,
        expected_sha256=__import__("hashlib").sha256(prefix.checkpoint).hexdigest(),
        stop_after=3)
    assert control.status == "completed" and control.state.cursor == 3
    n = len(source.owner.assembly.coordinates)
    original = _physical_shell(source.owner, control.state,
        np.tile(np.eye(3), (n, 1, 1)))
    altered = _physical_shell(fresh.owner, changed.state, changed.state.anchor)
    _close(original.current_positions, altered.current_positions, "accepted positions")
    _close(original.rotations, altered.rotations, "accepted physical rotations")
    _close(original.deformation, altered.deformation, "accepted shell deformation")
    _close(control.state.mechanical.positions, changed.state.mechanical.positions,
        "native beam positions")
    _close(control.state.mechanical.nodal_frames, changed.state.mechanical.nodal_frames,
        "native beam frames")
    assert np.linalg.norm(changed.state.mechanical.nodal_frames-
        prefix.state.mechanical.nodal_frames) > 1.e-8
    _close(control.state.multipliers, changed.state.multipliers, "joint multipliers")
    beam_control = source.owner.assembly.beam.assemble(control.state.mechanical,
        source.owner.targets[2], control.state.beam_origins)
    beam_rebased = fresh.owner.assembly.beam.assemble(changed.state.mechanical,
        fresh.owner.targets[2], changed.state.beam_origins)
    _close(beam_control[0], beam_rebased[0], "native beam section residual")
    _close(beam_control[1], beam_rebased[1], "native beam section tangent")
    assert np.linalg.norm(beam_rebased[0]) > 1.e-8
    assert np.linalg.norm(changed.state.shell_u.reshape(-1, 6)[:, 3:]) > 1.e-8
    assert np.linalg.norm(changed.state.mechanical.positions-prefix.state.mechanical.positions) > 1.e-8
    assert np.linalg.norm(changed.state.multipliers-prefix.state.multipliers) > 1.e-8
    assert canonical(control.state.beam_histories) == canonical(changed.state.beam_histories)
    old_paths, old_history = numeric_state(decode(control.state.shell_history))
    new_paths, new_history = numeric_state(decode(changed.state.shell_history))
    assert old_paths == new_paths
    _close(old_history, new_history, "accepted S3 layer history")
    old_shell_force, old_shell_jac, old_work, old_candidate = shell_spatial(
        source.owner.assembly, decode(control.state.shell_origin), original,
        control.state.shell_u)
    new_shell_force, new_shell_jac, new_work, new_candidate = shell_spatial(
        fresh.owner.assembly, decode(changed.state.shell_origin), altered,
        changed.state.shell_u)
    _close(old_shell_force, new_shell_force, "post-restart spatial shell force")
    _close(old_shell_jac, new_shell_jac, "post-restart spatial shell tangent")
    _close(old_work, new_work, "post-restart shell work")
    assert abs(new_work) > 1.e-8
    for name in ("layer_strain", "layer_stress"):
        assert np.linalg.norm(new_candidate[name]) > 1.e-8
        _close(old_candidate[name], new_candidate[name], name)
    direct, residual, _, _ = source.owner._evaluate(control.state,
        control.state.mechanical, control.state.shell_u, control.state.multipliers, 3)
    anchored, rebased_residual = fresh._anchored_evaluate(changed.state,
        changed.state.mechanical, changed.state.shell_u, changed.state.multipliers, 3)
    _close(direct.constraints, anchored.constraints, "post-restart joint gap")
    _close(residual, rebased_residual, "post-restart constrained residual")
    left = physical_tangent(direct, control.state.shell_u, source.owner.assembly.count)
    right = physical_tangent(anchored, changed.state.shell_u, source.owner.assembly.count)
    rng = np.random.default_rng(1606)
    for index in range(3):
        direction = rng.standard_normal(source.owner.assembly.count)
        direction /= np.linalg.norm(direction)
        _close(left@direction, right@direction, f"post-restart tangent direction {index}")
        direction[-6:] = 0.
        _close(np.array([residual@direction]), np.array([rebased_residual@direction]),
            f"post-restart virtual work {index}")
    for rows, label in ((slice(0, 3), "force"), (slice(3, 6), "moment")):
        _close(residual[:source.owner.assembly.shell_count].reshape(-1, 6)[:, rows],
            rebased_residual[:source.owner.assembly.shell_count].reshape(-1, 6)[:, rows],
            "post-restart shell " + label)
    fixed = np.setdiff1d(np.arange(source.owner.assembly.count),
        np.asarray(source.owner.free))
    assert np.array_equal(fixed, np.arange(6))
    _physical_scale_close(residual[fixed[:3]], rebased_residual[fixed[:3]],
        1.e-8, "supported force reaction")
    _physical_scale_close(residual[fixed[3:]], rebased_residual[fixed[3:]],
        1.e-8, "supported moment reaction")
    _, base_residual = fresh._anchored_evaluate(restored, restored.mechanical,
        restored.shell_u, restored.multipliers, 2)
    assert np.linalg.norm(rebased_residual[fixed[:3]]-base_residual[fixed[:3]]) > 1.e-8
    assert np.linalg.norm(rebased_residual[fixed[3:]]-base_residual[fixed[3:]]) > 1.e-8
    applied = fresh.owner.targets[2]*fresh.owner.forces.reshape(-1, 6)[-1, :3]
    force_balance = rebased_residual[fixed[:3]]+applied
    moment_balance = (rebased_residual[fixed[3:]]+
        np.cross(altered.current_positions[0], rebased_residual[fixed[:3]])+
        np.cross(changed.state.mechanical.positions[-1], applied))
    assert np.linalg.norm(force_balance)/max(1.e-8, np.linalg.norm(applied)) <= 1.e-8
    assert np.linalg.norm(moment_balance)/max(1.e-8,
        np.linalg.norm(rebased_residual[fixed[3:]])) <= 1.e-8


class _CancelAt(CancellationToken):
    def __init__(self, wanted):
        super().__init__()
        self.wanted = wanted

    def raise_if_cancelled(self, stage=""):
        if stage == self.wanted:
            self.cancel("synthetic F06 cancellation")
        super().raise_if_cancelled(stage)


def test_rejected_and_cancelled_rebase_leave_v1_prefix_replayable(base):
    _, prefix = base
    sha = __import__("hashlib").sha256(prefix.checkpoint).hexdigest()
    owner = OwnedRebasePrototype()
    with pytest.raises(ValueError, match="external F06 base checkpoint hash"):
        owner.rebase(prefix.checkpoint, expected_sha256="0"*64)
    with pytest.raises(SolveCancelled):
        owner.rebase(prefix.checkpoint, expected_sha256=sha,
            cancellation_token=_CancelAt("f06.rebase.before_commit"))
    assert owner.head_sha256 is None and owner.generation is None
    # Cancellation did not consume the generation or produce a partial event.
    accepted = owner.rebase(prefix.checkpoint, expected_sha256=sha)
    assert owner.head_sha256 == accepted.checkpoint_sha256 and owner.generation == 1
    with pytest.raises(ValueError, match="stale base"):
        owner.rebase(prefix.checkpoint, expected_sha256=sha)
    tampered = __import__("json").loads(accepted.checkpoint)
    tampered["anchor"][1][0][0] += .01
    tampered["rebased_state"]["anchor"][1][0][0] += .01
    tampered["rebased_state_sha256"] = __import__("hashlib").sha256(
        canonical(tampered["rebased_state"])).hexdigest()
    raw = canonical(tampered)
    with pytest.raises(ValueError, match="rebase event mismatch"):
        OwnedRebasePrototype().restore(raw,
            expected_sha256=__import__("hashlib").sha256(raw).hexdigest())
    cancelled_restore = OwnedRebasePrototype()
    with pytest.raises(SolveCancelled):
        cancelled_restore.restore(accepted.checkpoint,
            expected_sha256=accepted.checkpoint_sha256,
            cancellation_token=_CancelAt("f06.restore.before_commit"))
    assert cancelled_restore.head_sha256 is None and cancelled_restore.generation is None
    fresh = OwnedRebasePrototype()
    with pytest.raises(ValueError, match="external F06 prototype checkpoint hash"):
        fresh.restore(accepted.checkpoint, expected_sha256="0"*64)
    with pytest.raises(SolveCancelled):
        fresh.continue_loaded(accepted.checkpoint,
            expected_sha256=accepted.checkpoint_sha256,
            cancellation_token=_CancelAt("f06.successor.before_commit"))
    assert fresh.head_sha256 == accepted.checkpoint_sha256 and fresh.generation == 1
    assert fresh.restore(accepted.checkpoint,
        expected_sha256=accepted.checkpoint_sha256).cursor == 2
    resumed = OwnedRebasePrototype().owner.solve(checkpoint=prefix.checkpoint,
        expected_sha256=sha, stop_after=3)
    assert resumed.status == "completed" and resumed.state.cursor == 3


def test_unregistered_pilot_reports_raw_scaled_metrics_and_identity(tmp_path):
    if not _git_history_available():
        pytest.skip("actual immutable git history is absent from this source "
            "export; the unregistered pilot measurement records its source "
            "commit from git and must not fake or rebind provenance")
    value = measure_pilot(checkpoint_dir=tmp_path)
    metric = value["metrics"]
    assert metric["checkpoint_sidecars_written"] is True
    for name, row in metric["checkpoint_sidecars"].items():
        raw = (tmp_path / row["file"]).read_bytes()
        assert row["byte_count"] == len(raw)
        assert row["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
        assert row["sha256"] == metric["checkpoint_sha256"][name]
    assert value["metrics_sha256"] == __import__("hashlib").sha256(
        canonical(metric)).hexdigest()
    assert metric["schema"] == "G0_F06_CURVED_S3_V4_SCIENTIFIC_MEASUREMENT_V1"
    assert metric["scope"] == "G0_S3_OWNED_REBASE_MEASUREMENT_NOT_ADJUDICATED"
    assert metric["effective_environment"]["python_executable"]
    assert metric["effective_environment"]["anyfesolver_import"]
    assert "numba_cache_dir" in metric["effective_environment"]
    assert set(metric["effective_environment"]["python_environment"]) == {
        "PYTHONPATH", "PYTHONNOUSERSITE"}
    if __import__("sys").platform == "win32":
        assert metric["effective_environment"]["pywin32_version"]
        assert metric["effective_environment"]["win32job_import"]
    else:
        assert metric["effective_environment"]["pywin32_version"] is None
        assert metric["effective_environment"]["win32job_import"] is None
    assert metric["effective_environment"]["numba_version"]
    assert metric["effective_environment"]["llvmlite_version"]
    assert "blas" in metric["effective_environment"]
    assert len(metric["source_commit"]) == 40
    assert len(metric["source_sha256"]) == 25
    assert {"src/anyfesolver/_ge_beam3_native_generalized_element.py",
            "src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py",
            "src/anyfesolver/_ge_beam3_generalized_static_boundary.py"} <= set(
                metric["source_sha256"])
    assert metric["targets"] == (.5, 1., .75)
    assert metric["runtime_seconds"]["total"] > 0.
    assert set(metric["negative_checks"]) == {
        "bad_external_hash", "rehashed_anchor", "rehashed_successor_history",
        "rehashed_zero_residual", "stale_successor_commit",
        "cancel_rebase_before_commit", "cancel_successor_before_commit"}
    assert all(row["status"] == "REJECTED" for row in metric["negative_checks"].values())
    for name, row in metric["negative_checks"].items():
        assert row["before_head_sha256"] == row["after_head_sha256"], name
        assert row["before_generation"] == row["after_generation"], name
    assert metric["negative_checks"]["cancel_rebase_before_commit"][
        "after_head_sha256"] is None
    assert metric["negative_checks"]["cancel_successor_before_commit"][
        "after_head_sha256"] == metric["checkpoint_sha256"]["rebase"]
    assert all(row["status"] == "MATCHED" and
        row["direct_state_sha256"] == row["audited_state_sha256"]
        for row in metric["optional_replay_audit"].values())
    for cursor in ("cursor_2", "cursor_3"):
        paired = metric["paired"][cursor]
        for name in ("shell_translation", "beam_translation", "shell_rotation",
                     "beam_rotation", "shell_force", "shell_moment",
                     "shell_layer_strain", "shell_layer_stress",
                     "support_force", "support_moment", "shell_work",
                     "gap_translation", "gap_rotation", "free_solver_residual",
                     "physical_tangent_direction"):
            row = paired[name]
            assert row["floor"] > 0. and row["normalization_scale"] >= row["floor"]
            assert np.isfinite(row["absolute_difference"])
            assert np.isfinite(row["normalized_difference"])
            if name in ("shell_rotation", "beam_rotation"):
                assert np.asarray(row["left_frames"]).shape == np.asarray(
                    row["right_frames"]).shape
            else:
                assert np.asarray(row["left_values"]).shape == np.asarray(
                    row["right_values"]).shape
        samples = metric["tangent_fd"][cursor]["samples"]
        assert np.asarray(metric["tangent_fd"][cursor]["direction_spatial"]).shape == (
            np.asarray(metric["tangent_fd"][cursor]["direction_chart"]).shape)
        assert [row["step_size"] for row in samples] == [2.e-3, 1.e-3, 5.e-4, 2.5e-4]
        assert samples[-1]["assembled_scaled"]["normalized_difference"] < 1.e-7
        mixed_errors = [row["assembled_scaled"]["normalized_difference"] for row in samples]
        assert mixed_errors[0] > mixed_errors[1] > mixed_errors[2] > mixed_errors[3]
        assert all(metric["tangent_fd"][cursor][key] > 0. for key in (
            "direction_shell_norm", "direction_beam_retained_norm", "direction_multiplier_norm"))
        for kind in ("strain", "stress"):
            layer = paired["shell_layer_maxima"][kind]
            assert layer["index_names"] == ("integration_station", "layer", "component")
            assert layer["maximum"]["normalized_difference"] < 1.e-9
        beam = paired["beam_section_response"]
        assert len(beam["station_identity"]) == 2
        assert all(len(row["stations"]) == 8 for row in beam["station_identity"])
        assert len(beam["retained_resultants"]["component_labels"]) == 18
        for kind in ("retained_resultants", "section_strain", "section_resultants"):
            assert beam[kind]["maximum"]["normalized_difference"] < 1.e-9
        split = beam["section_split"]
        assert set(split) == {"strain_high", "strain_low", "resultant_high", "resultant_low"}
        for field, high, low in (("section_strain", "strain_high", "strain_low"),
                ("section_resultants", "resultant_high", "resultant_low")):
            for side_values in ("left_values", "right_values"):
                h, l = np.asarray(split[high][side_values]), np.asarray(split[low][side_values])
                assert h.shape == l.shape == (2, 8, 6)
                assert np.array_equal(h+l, np.asarray(beam[field][side_values]))
            assert np.isfinite(split[high]["maximum"]["normalized_difference"])
            assert np.isfinite(split[low]["maximum"]["normalized_difference"])
        for side in ("control", "rebased"):
            component = metric["component_fd"][cursor][side]
            assert len(component["direction_shell_chart"]) == len(component["direction_shell"])
            assert [row["step_size"] for row in component["samples"]] == [
                2.e-3, 1.e-3, 5.e-4, 2.5e-4]
            assert all(np.isfinite(row["normalized_difference"])
                for row in component["samples"][-1].values() if isinstance(row, dict))
            assert component["samples"][-1]["dead_external_force"]["absolute_difference"] == 0.
            assert component["samples"][-1]["dead_external_couple"]["absolute_difference"] == 0.
            assert component["external_operator_scope"] == "SHARED_CONSTANT_DEAD_LOAD_CALL_PATH"
            for key, row in component["samples"][-1].items():
                if not isinstance(row, dict):
                    continue
                errors = [sample[key]["normalized_difference"] for sample in component["samples"]]
                assert errors[-1] <= 1.e-6
                if errors[0] > 1.e-9:
                    assert errors[0] > errors[1] > errors[2]
                else:
                    assert max(errors) <= 1.e-9
            work = metric["interface_work"][cursor][side]
            assert set(work["variations"]) == {"shell_only", "beam_only", "mixed"}
            assert len(work["compatible_rigid_zero_work"]) == 6
            assert all(row["samples"][-1]["reference_scaled_error"] < 1.e-8
                for row in work["variations"].values())
            assert all(row["reference_scaled_work"] < 1.e-8
                for row in work["compatible_rigid_zero_work"].values())
            balance = metric["fixed_reference_balance"][cursor][side]
            reference_force = np.sqrt(5.)*1.e-4
            assert balance["reference_length"] == 1.
            assert balance["reference_force"] == reference_force
            assert balance["reference_moment"] == reference_force
            assert np.allclose(balance["force_closure"],
                np.asarray(balance["support_force"])+balance["applied_force"], atol=1.e-18)
            assert np.allclose(balance["moment_closure"],
                np.asarray(balance["support_moment"])+
                np.cross(balance["support_position"], balance["support_force"])+
                np.cross(balance["tip_position"], balance["applied_force"]), atol=1.e-18)
            assert balance["force_reference_scaled"] < 1.e-8
            assert balance["moment_reference_scaled"] < 1.e-8
    for name in ("beam_rotation", "support_force", "support_moment",
                 "shell_work", "multiplier_force"):
        assert metric["cursor_3_changes"][name]["absolute_difference"] > 1.e-8
    assert metric["cursor_3_balance"]["force"]["normalized"] < 1.e-8
    assert metric["cursor_3_balance"]["moment"]["normalized"] < 1.e-8
    for name, lower in (("beam_rotation", 5.e-4), ("support_force", 2.e-5),
            ("support_moment", 5.e-5), ("shell_work", 1.e-7),
            ("multiplier_force", 2.e-5)):
        assert metric["cursor_3_changes"][name]["absolute_difference"] > lower


def test_candidate_output_directory_is_exclusive(tmp_path):
    with pytest.raises(FileExistsError):
        main(["--output-dir", str(tmp_path)])
