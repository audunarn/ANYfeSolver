"""Read-only candidate screen for the proposed F06 v4 curved S3 contract.

This does not register, execute, accept, or release a scientific run. Missing
mandatory evidence fails the screen; no predicate is inferred from a status.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SCHEMA = "G0_F06_V4_SCIENTIFIC_REGISTRATION_V1"
ATTESTATION_SCHEMA = "G0_F06_V4_SUPERVISOR_ATTESTATION_V1"
METRICS_SCHEMA = "G0_F06_CURVED_S3_V4_SCIENTIFIC_MEASUREMENT_V1"
REBASE_SCHEMA = "G0_F06_CURVED_S3_OWNED_REBASE_PROTOTYPE_V1"
LEGACY_SCHEMA = "GE_BEAM3_COROTATIONAL_SHELL_COUPLED_ACCEPTED_CHAIN_V1"
STEPS = (2.e-3, 1.e-3, 5.e-4, 2.5e-4)
CURSORS = ("cursor_2", "cursor_3")
SIDECARS = ("base", "rebase", "successor", "control")
SIDES = ("control", "rebased")
F_REF = math.sqrt(5.)*1.e-4
FLOORS = dict(translation=(1.e-8, "length"), rotation=(1.e-9, "radian"),
    force=(1.e-10, "force"), moment=(1.e-10, "moment"),
    work=(1.e-12, "force_length"), strain=(1.e-12, "dimensionless_strain"),
    curvature=(1.e-12, "inverse_length"), stress=(1.e-10, "force_per_area"),
    gap_translation=(1.e-10, "length"), gap_rotation=(1.e-10, "radian"),
    solver_scaled_residual=(1.e-12, "mixed_solver_scale"),
    tangent_fd=(1.e-10, "scaled_directional_residual"))
THRESHOLDS = dict(floors={k: dict(value=v, unit=u) for k, (v, u) in FLOORS.items()},
    paired_pose=1.e-9, paired_force_moment=1.e-9, paired_work=1.e-8,
    layer=1.e-9, beam_section=1.e-9, gap_absolute=1.e-9,
    free_residual_absolute=1.e-11, balance=1.e-8, assembled_fd=1.e-6,
    component_fd=1.e-6, physical_tangent=1.e-9, port_work=1.e-8,
    nontrivial=dict(beam_rotation=5.e-4, support_force=2.e-5,
        support_moment=5.e-5, shell_work=1.e-7, multiplier_force=2.e-5))
REQUIRED_ROLES = (
    "scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py",
    "tests/test_b3_e4_f06_owned_rebase_v4.py",
    "docs/agent_plans/B3_GE_E4_PL_G0_F06_V4_SCIENTIFIC_CONTRACT_DRAFT.md",
    "scripts/adjudicate_b3_ge_e4_pl_g0_f06_v4.py",
    "scripts/supervise_b3_ge_e4_pl_g0_f06_v4.py",
    "docs/agent_plans/B3_GE_E4_PL_G0_F06_OWNED_REBASE_V4_PROTOCOL.md",
    "tests/test_b3_e4_f06_v4_adjudicator.py",
    "tests/test_b3_e4_f06_v4_supervisor.py",
)
RUNNER_SOURCES = (
    REQUIRED_ROLES[0], REQUIRED_ROLES[1], REQUIRED_ROLES[2],
    "scripts/probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3.py",
    "src/anyfesolver/_ge_beam3_shell_joint_state.py",
    "src/anyfesolver/_ge_beam3_shell_joint_trial.py",
    "src/anyfesolver/_ge_beam3_g3c_local_shell.py",
    "src/anyfesolver/_ge_beam3_coupled_beam_subdomain.py",
    "src/anyfesolver/_ge_beam3_pose_joint.py",
    "src/anyfesolver/_ge_beam3_retained_generalized.py",
    "src/anyfesolver/_ge_beam3_retained_generalized_state.py",
    "src/anyfesolver/_ge_beam3_retained_generalized_program.py",
    "src/anyfesolver/_ge_beam3_native_generalized_element.py",
    "src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py",
    "src/anyfesolver/_ge_beam3_generalized_cell.py",
    "src/anyfesolver/_ge_beam3_generalized_static_boundary.py",
    "src/anyfesolver/_ge_beam3_p5/algebra.py",
    "src/anyfesolver/_ge_beam3_p5_seeded/core.py",
    "src/anyfesolver/e4_pl_s3_v2d_element.py",
    "src/anyfesolver/control.py",
    "src/anyfesolver/_native_reference_modal.py",
    "src/anyfesolver/_ge_beam3_native_generalized_loading.py",
    "src/anyfesolver/_ge_beam3_native_line_loading.py",
    "src/anyfesolver/_ge_beam3_centered_reference.py",
    "src/anyfesolver/fe_core.py",
)


class EvidenceError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


def _require(condition, code, detail):
    if not condition:
        raise EvidenceError(code, detail)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def _nonfinite(value):
    raise EvidenceError("NONFINITE_JSON", str(value))


def _canonical(value, *, ascii_only=False):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=ascii_only, allow_nan=False)+"\n").encode(
            "ascii" if ascii_only else "utf-8")


def _read_json(path: Path, *, ascii_only=False):
    _require(path.is_file(), "MISSING_ARTIFACT", str(path))
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        expected = _canonical(value, ascii_only=ascii_only)
    except (UnicodeError, ValueError, TypeError) as error:
        raise EvidenceError("INVALID_JSON", str(path)) from error
    _require(raw == expected, "NONCANONICAL_JSON", str(path))
    return value, raw


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _keys(value, expected, label):
    _require(type(value) is dict and set(value) == set(expected),
        "INCOMPLETE_INVENTORY", f"{label}: expected {sorted(expected)}")


def _num(value, label, *, nonnegative=False):
    _require(type(value) in (int, float, np.float64, np.int64) and math.isfinite(value) and
        (not nonnegative or value >= 0.), "NONFINITE_OR_INVALID_NUMBER", label)
    return float(value)


def _array(value, shape, label):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise EvidenceError("INVALID_ARRAY", label) from error
    _require(a.shape == shape and np.isfinite(a).all(), "INVALID_ARRAY", label)
    return a


def _near(left, right, label, *, atol=1.e-15):
    _require(math.isclose(_num(left, label), _num(right, label),
        rel_tol=1.e-12, abs_tol=atol), "INCONSISTENT_METRIC", label)


class Screen:
    def __init__(self):
        self.failures = []
        self.check_count = 0

    def check(self, condition, code, detail):
        self.check_count += 1
        if not condition:
            self.failures.append(dict(code=code, detail=detail))

    def missing(self, code, detail):
        self.check(False, code, detail)


def _identity(registration_path, evidence_dir, attestation_path, screen):
    reg, reg_raw = _read_json(registration_path)
    att, _ = _read_json(attestation_path)
    _require(reg.get("schema") == REGISTRATION_SCHEMA,
        "REGISTRATION_SCHEMA", str(reg.get("schema")))
    _require(att.get("schema") == ATTESTATION_SCHEMA,
        "ATTESTATION_SCHEMA", str(att.get("schema")))
    _keys(att, ("schema", "registration_sha256", "command", "observed_sources",
        "observed_environment", "start_utc", "end_utc", "wall_seconds",
        "peak_tree_rss_bytes", "rss_sample_count", "root_pid", "last_job_pids",
        "exit_code", "resource_violation", "launch_error",
        "observed_child_environment", "child_environment_match",
        "stdout_sha256", "stderr_sha256", "test_log_sha256",
        "metrics_sha256", "artifacts"), "supervisor attestation")
    _require(registration_path.resolve() == evidence_dir/"registration.json",
        "REGISTRATION_LOCATION", str(registration_path))
    _require(attestation_path.resolve() == evidence_dir/"attestation.json",
        "ATTESTATION_LOCATION", str(attestation_path))
    _require(Path(reg.get("evidence_dir", "")).resolve() == evidence_dir,
        "EVIDENCE_DIRECTORY", str(reg.get("evidence_dir")))
    _require(att.get("registration_sha256") == _sha(reg_raw),
        "REGISTRATION_DIGEST", "attestation does not bind registration bytes")
    claim, _ = _read_json(evidence_dir/"execution.claim")
    _require(claim.get("registration_sha256") == _sha(reg_raw) and
        type(claim.get("claimed_at_utc")) is str,
        "EXECUTION_CLAIM", "claim does not bind registration")
    _require(reg.get("thresholds") == THRESHOLDS, "THRESHOLD_CONTRACT",
        "registered numerical bounds differ from reviewed draft")
    case = reg.get("case")
    _require(type(case) is dict and case.get("targets") == [.5, 1., .75] and
        case.get("L_ref") == 1. and case.get("F_ref") == F_REF and
        case.get("M_ref") == F_REF and type(case.get("definition_sha256")) is str,
        "CASE_CONTRACT", "fixed fixture targets/references/definition required")
    limits = reg.get("limits")
    _require(limits == dict(wall_seconds=180, sampled_tree_rss_bytes=2*1024**3),
        "RESOURCE_CONTRACT", "180 s and 2 GiB tree RSS required")
    sources = reg.get("sources")
    _require(type(sources) is dict and
        set(RUNNER_SOURCES) | set(REQUIRED_ROLES) == set(sources),
        "SOURCE_INVENTORY", "runner/test/protocol/adjudicator/supervisor missing")
    _require(att.get("observed_sources") == sources,
        "SOURCE_ATTESTATION", "observed source map differs")
    for name, expected in sources.items():
        _require(type(name) is str and type(expected) is str and len(expected) == 64,
            "SOURCE_INVENTORY", str(name))
        local = (ROOT/name).resolve()
        _require(local.is_relative_to(ROOT) and local.is_file(),
            "SOURCE_INVENTORY", str(name))
        screen.check(_sha(local.read_bytes()) == expected,
            "SOURCE_HASH", str(name))
    env = reg.get("environment")
    _require(type(env) is dict and att.get("observed_environment") == env,
        "ENVIRONMENT_ATTESTATION", "environment differs")
    _require(set(("python_executable", "python_version", "numpy_version",
        "scipy_version", "anyfesolver_import", "psutil_version",
        "threadpoolctl_version", "pywin32_version", "win32job_import",
        "numba_version", "llvmlite_version",
        "platform", "numba_cache_dir",
        "python_environment", "thread_environment", "blas")) <= set(env) and
        type(env.get("python_environment")) is dict and
        set(env["python_environment"]) == {"PYTHONPATH", "PYTHONNOUSERSITE"},
        "ENVIRONMENT_INVENTORY", "required import/thread/BLAS identities missing")
    expected_command = dict(argv=[env["python_executable"],
        str((ROOT/REQUIRED_ROLES[0]).resolve()), "--output-dir",
        str(evidence_dir/"measurement")], cwd=str(ROOT.resolve()))
    _require(reg.get("command") == expected_command and
        att.get("command") == expected_command, "COMMAND_IDENTITY",
        "registered/observed command differs")
    _require((evidence_dir/"test.log").is_file() and
        type(reg.get("test_log_sha256")) is str and
        att.get("test_log_sha256") == reg["test_log_sha256"] and
        _sha((evidence_dir/"test.log").read_bytes()) == reg["test_log_sha256"],
        "TEST_LOG_IDENTITY", "frozen test log missing or mismatched")
    for name in ("stdout", "stderr"):
        path = evidence_dir/f"{name}.log"
        _require(path.is_file() and att.get(f"{name}_sha256") == _sha(path.read_bytes()),
            "PROCESS_LOG_IDENTITY", name)
    for name in ("start_utc", "end_utc"):
        _require(type(att.get(name)) is str, "TIME_ATTESTATION", name)
    try:
        start = datetime.fromisoformat(att["start_utc"])
        end = datetime.fromisoformat(att["end_utc"])
    except ValueError as error:
        raise EvidenceError("TIME_ATTESTATION", "invalid UTC timestamps") from error
    _require(start.tzinfo is not None and end.tzinfo is not None and end >= start,
        "TIME_ATTESTATION", "timestamps unordered or timezone absent")
    screen.check(0. <= _num(att.get("wall_seconds"), "wall_seconds") <= 180.,
        "WALL_LIMIT", "supervisor duration")
    peak = att.get("peak_tree_rss_bytes")
    screen.check(type(peak) is int and 0 < peak <= 2*1024**3,
        "RSS_LIMIT", "sampled process-tree RSS")
    screen.check(type(att.get("rss_sample_count")) is int and
        att["rss_sample_count"] > 0 and type(att.get("root_pid")) is int,
        "RSS_SAMPLING", "no process-tree sample/root PID")
    screen.check(att.get("exit_code") == 0 and
        att.get("resource_violation") is None and att.get("launch_error") is None,
        "PROCESS_RESULT", "nonzero exit/resource/launch failure")
    screen.check(att.get("observed_child_environment") == env and
        att.get("child_environment_match") is True,
        "CHILD_ENVIRONMENT_ATTESTATION", "child process environment differs")
    return reg, att


def _record(row, label):
    _require(type(row) is dict and type(row.get("record_sha256")) is str,
        "CHECKPOINT_RECORD", label)
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    _require(_sha(_canonical(body, ascii_only=True)) == row["record_sha256"],
        "CHECKPOINT_RECORD", label)
    _require(type(row.get("state")) is dict and type(row.get("cursor")) is int and
        row["state"].get("cursor") == row["cursor"],
        "CHECKPOINT_RECORD", f"{label} cursor/state")


def _finite_tree(value, label):
    if type(value) is float:
        _require(math.isfinite(value), "NONFINITE_STATE", label)
    elif type(value) is dict:
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _rotation_inventory(state, label):
    mechanical = state.get("mechanical")
    _require(type(mechanical) is dict and "nodal_frames" in mechanical and
        "cell_rotations" in mechanical and "anchor" in state,
        "ROTATION_INVENTORY", label)
    frames = [*np.asarray(mechanical["nodal_frames"], dtype=float).reshape(-1, 3, 3),
        *np.asarray(mechanical["cell_rotations"], dtype=float).reshape(-1, 3, 3),
        *np.asarray(state["anchor"], dtype=float).reshape(-1, 3, 3)]
    for index, q in enumerate(frames):
        _require(np.isfinite(q).all() and
            np.linalg.norm(q.T@q-np.eye(3)) <= 1.e-9 and
            abs(np.linalg.det(q)-1.) <= 1.e-9,
            "ROTATION_CHART", f"{label} frame {index}")


def _artifacts(reg, att, evidence_dir, screen):
    measurement = evidence_dir/"measurement"
    metrics, metrics_raw = _read_json(measurement/"metrics.json", ascii_only=True)
    _keys(metrics, ("metrics_sha256", "metrics"), "metrics envelope")
    body = metrics["metrics"]
    _require(type(body) is dict and body.get("schema") == METRICS_SCHEMA and
        body.get("scope") == "G0_S3_OWNED_REBASE_MEASUREMENT_NOT_ADJUDICATED" and
        body.get("production_qualified") is False,
        "METRICS_SCHEMA", "candidate metrics identity")
    _require(metrics["metrics_sha256"] == _sha(_canonical(body, ascii_only=True)),
        "INTERNAL_METRICS_DIGEST", "metrics payload changed")
    raw_digest = _sha(metrics_raw)
    _require(att.get("metrics_sha256") == raw_digest,
        "METRICS_ARTIFACT_DIGEST", "supervisor raw-file digest differs")
    artifacts = att.get("artifacts")
    _require(type(artifacts) is dict and set(artifacts) == {"metrics", "checkpoints"},
        "ARTIFACT_INVENTORY", "metrics and four sidecars required")
    _require(artifacts["metrics"] == dict(file="metrics.json",
        byte_count=len(metrics_raw), sha256=raw_digest),
        "METRICS_ARTIFACT_DIGEST", "supervisor artifact descriptor differs")
    _require(body.get("checkpoint_sidecars_written") is True and
        type(body.get("checkpoint_sidecars")) is dict and
        set(body["checkpoint_sidecars"]) == set(SIDECARS) and
        type(body.get("checkpoint_sha256")) is dict and
        set(body["checkpoint_sha256"]) == set(SIDECARS) and
        type(artifacts["checkpoints"]) is dict and
        set(artifacts["checkpoints"]) == set(SIDECARS),
        "ARTIFACT_INVENTORY", "sidecar inventory incomplete")
    checkpoints = {}
    raw_checkpoints = {}
    for name in SIDECARS:
        value, raw = _read_json(measurement/f"{name}.checkpoint", ascii_only=True)
        digest = _sha(raw)
        descriptor = dict(file=f"{name}.checkpoint", byte_count=len(raw), sha256=digest)
        _require(body["checkpoint_sidecars"][name] == descriptor and
            artifacts["checkpoints"][name] == descriptor and
            body["checkpoint_sha256"][name] == digest,
            "SIDECAR_DIGEST", name)
        checkpoints[name], raw_checkpoints[name] = value, raw
    _require(body.get("source_commit") == reg.get("source_commit") and
        type(body.get("source_sha256")) is dict and
        set(body["source_sha256"]) == set(RUNNER_SOURCES) and
        all(reg["sources"][name] == digest for name, digest in
            body["source_sha256"].items()),
        "RUNNER_SOURCE_IDENTITY", "source commit/map differs")
    _require(body.get("provisional_floors") == THRESHOLDS["floors"] and
        body.get("targets") == [.5, 1., .75] and
        body.get("statuses") == dict(base="paused", control="completed",
            rebase="REBASED_AT_LOADED_CURSOR_2",
            successor="ACCEPTED_LOADED_CURSOR_3"),
        "RUNNER_CASE_INVENTORY", "targets/floors/statuses differ")
    base, rebased, successor, control = (checkpoints[name] for name in SIDECARS)
    _require(base.get("schema") == LEGACY_SCHEMA and
        control.get("schema") == LEGACY_SCHEMA and
        rebased.get("schema") == REBASE_SCHEMA and
        successor.get("schema") == REBASE_SCHEMA,
        "CHECKPOINT_SCHEMA", "checkpoint family differs")
    definition_hash = reg["case"]["definition_sha256"]
    _require(all(value.get("definition_sha256") == definition_hash
        for value in checkpoints.values()), "DEFINITION_IDENTITY",
        "four checkpoint definitions differ")
    _require(base.get("definition") == control.get("definition") and
        _sha(_canonical(base["definition"], ascii_only=True)) == definition_hash and
        base["definition"].get("targets") == [.5, 1., .75] and
        base["definition"].get("shell_fixed") == list(range(6)) and
        base["definition"].get("production_qualified") is False,
        "DEFINITION_IDENTITY", "fixture definition differs")
    forces = _array(base["definition"].get("nodal_forces"),
        (len(base["definition"]["nodal_forces"]),), "definition nodal forces")
    _require(forces.size % 6 == 0 and np.array_equal(forces[:-6], 0.*forces[:-6]) and
        np.array_equal(forces[-6:], [0., 1.e-4, 2.e-4, 0., 0., 0.]),
        "LOAD_INVENTORY", "one dead terminal force and zero couples required")
    _require(len(base.get("records", ())) == 2 and
        len(control.get("records", ())) == 3 and
        base["records"] == control["records"][:2],
        "ACCEPTED_PREFIX", "control does not share exact two-step prefix")
    for name, record in (("initial", base["initial"]),
            *((f"base_{index}", row) for index, row in enumerate(base["records"], 1)),
            ("control_3", control["records"][2])):
        _record(record, name)
    _require([row["parameter"] for row in control["records"]] == [.5, 1., .75] and
        [row["cursor"] for row in control["records"]] == [1, 2, 3],
        "TARGET_CHAIN", "legacy accepted target sequence differs")
    _require(rebased.get("successor") is None and rebased.get("cursor") == 2 and
        rebased.get("base_checkpoint") == raw_checkpoints["base"].decode("ascii") and
        rebased.get("base_sha256") == body["checkpoint_sha256"]["base"] and
        rebased.get("base_state") == base["records"][-1]["state"] and
        rebased.get("base_state_sha256") == _sha(_canonical(rebased["base_state"], ascii_only=True)) and
        rebased.get("rebased_state_sha256") == _sha(_canonical(rebased["rebased_state"], ascii_only=True)),
        "REBASE_GENERATION", "base/rebase accepted state binding differs")
    _require({k: value for k, value in successor.items() if k != "successor"} ==
        {k: value for k, value in rebased.items() if k != "successor"} and
        type(successor.get("successor")) is dict and
        successor["successor"].get("generation") == 2 and
        successor["successor"].get("predecessor_sha256") == body["checkpoint_sha256"]["rebase"] and
        successor["successor"].get("state", {}).get("cursor") == 3,
        "SUCCESSOR_GENERATION", "successor does not bind rebased head")
    for name, state in (("base", rebased["base_state"]),
            ("rebased", rebased["rebased_state"]),
            ("successor", successor["successor"]["state"]),
            ("control", control["records"][2]["state"])):
        _finite_tree(state, name)
        if name != "base" and name != "control":
            _rotation_inventory(state, name)
    return body, checkpoints


def _metric(row, floor_name, label, screen, *, raw_required=True):
    required = {"absolute_difference", "floor", "left_norm", "right_norm",
        "normalization_scale", "normalized_difference", "unit"}
    _require(type(row) is dict and required <= set(row),
        "METRIC_INVENTORY", label)
    floor, unit = FLOORS[floor_name]
    _require(row["floor"] == floor and row["unit"] == unit,
        "METRIC_UNIT", label)
    left, right, absolute, scale, normalized = (
        _num(row[key], f"{label}.{key}", nonnegative=True) for key in
        ("left_norm", "right_norm", "absolute_difference",
         "normalization_scale", "normalized_difference"))
    _near(scale, max(floor, left, right), f"{label}.scale")
    _near(normalized, absolute/scale, f"{label}.normalized")
    screen.check(absolute+1.e-14 >= abs(left-right) and
        absolute <= left+right+1.e-14,
        "METRIC_TRIANGLE", label)
    if raw_required:
        if floor_name == "rotation":
            if not {"left_frames", "right_frames", "reference_frames"} <= set(row):
                screen.missing("MISSING_PAIRED_FRAMES", label)
            else:
                try:
                    x, y, ref = (np.asarray(row[key], dtype=float) for key in
                        ("left_frames", "right_frames", "reference_frames"))
                    _require(x.shape == y.shape == ref.shape and x.shape[-2:] == (3, 3),
                        "METRIC_RAW_SHAPE", label)
                    def angles(a, b):
                        relative = a@np.swapaxes(b, -1, -2)
                        skew = np.stack((relative[..., 2, 1]-relative[..., 1, 2],
                            relative[..., 0, 2]-relative[..., 2, 0],
                            relative[..., 1, 0]-relative[..., 0, 1]), axis=-1)
                        sine = np.linalg.norm(skew, axis=-1)/2.
                        cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1.)/2., -1., 1.)
                        return np.arctan2(sine, cosine)
                    _near(absolute, float(np.linalg.norm(angles(x, y))),
                        f"{label}.raw_rotation", atol=1.e-12)
                    _near(left, float(np.linalg.norm(angles(x, ref))),
                        f"{label}.raw_left_rotation", atol=1.e-12)
                    _near(right, float(np.linalg.norm(angles(y, ref))),
                        f"{label}.raw_right_rotation", atol=1.e-12)
                except (TypeError, ValueError) as error:
                    raise EvidenceError("METRIC_RAW_SHAPE", label) from error
        elif not {"left_values", "right_values"} <= set(row):
            screen.missing("MISSING_PAIRED_RAW_ARRAYS", label)
        else:
            x, y = np.asarray(row["left_values"], dtype=float), np.asarray(
                row["right_values"], dtype=float)
            _require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(),
                "METRIC_RAW_SHAPE", label)
            _near(left, float(np.linalg.norm(x)), f"{label}.raw_left")
            _near(right, float(np.linalg.norm(y)), f"{label}.raw_right")
            _near(absolute, float(np.linalg.norm(x-y)), f"{label}.raw_difference")
    return row


def _indexed(row, shape, floors, index_names, label):
    keys = ("absolute_differences", "component_floors", "component_roles",
        "component_units", "index_names", "left_values", "maximum",
        "normalization_scales", "normalized_differences", "right_values")
    _require(type(row) is dict and set(keys) <= set(row) and
        set(row)-set(keys) <= {"component_labels"},
        "INCOMPLETE_INVENTORY", label)
    _require(row["index_names"] == list(index_names) and
        row["component_roles"] == list(floors) and
        row["component_floors"] == [FLOORS[name][0] for name in floors] and
        row["component_units"] == [FLOORS[name][1] for name in floors],
        "INDEXED_ROLE_UNIT", label)
    left = _array(row["left_values"], shape, label+" left")
    right = _array(row["right_values"], shape, label+" right")
    raw = np.abs(left-right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)),
        np.asarray([FLOORS[name][0] for name in floors]))
    normalized = raw/scale
    for key, expected in (("absolute_differences", raw),
            ("normalization_scales", scale),
            ("normalized_differences", normalized)):
        actual = _array(row[key], shape, label+" "+key)
        _require(np.allclose(actual, expected, rtol=1.e-12, atol=1.e-17),
            "INDEXED_RECOMPUTE", label+" "+key)
    worst = np.unravel_index(int(np.argmax(normalized)), shape)
    maximum = row["maximum"]
    _keys(maximum, ("indices", "absolute_difference", "normalized_difference",
        "left_value", "right_value", "scale"), label+" maximum")
    _require(maximum["indices"] == dict(zip(index_names, map(int, worst))),
        "INDEXED_MAXIMUM", label)
    for key, expected in (("absolute_difference", raw[worst]),
            ("normalized_difference", normalized[worst]),
            ("left_value", left[worst]), ("right_value", right[worst]),
            ("scale", scale[worst])):
        _near(maximum[key], float(expected), label+" max "+key)
    return float(normalized[worst])


def _plateau(rows, key, limit, label, screen, *, absolute=False):
    _require(type(rows) is list and len(rows) == 4 and
        [row.get("step_size") for row in rows] == list(STEPS),
        "FD_STEP_INVENTORY", label)
    values = []
    for row in rows:
        metric = row.get(key)
        _require(type(metric) is dict, "FD_BLOCK_INVENTORY", label+" "+key)
        values.append(_num(metric["absolute_difference" if absolute else
            "normalized_difference"], label+" "+key, nonnegative=True))
    screen.check(values[-1] <= limit, "FD_FINEST", label+" "+key)
    if values[0] > 1.e-9:
        screen.check(values[0] > values[1] > values[2],
            "FD_PLATEAU", label+" "+key)
    else:
        screen.check(max(values) <= 1.e-9, "FD_ROUNDOFF_PLATEAU", label+" "+key)


def _balance(row, cursor, label, screen):
    keys = ("applied_couple", "applied_force", "force_closure", "force_raw_norm",
        "force_reference_scaled", "load_factor", "moment_closure", "moment_raw_norm",
        "moment_reference_scaled", "reference_force", "reference_length",
        "reference_moment", "reference_origin", "support_force", "support_moment",
        "support_position", "tip_position")
    _keys(row, keys, label)
    _require(row["reference_length"] == 1. and row["reference_force"] == F_REF and
        row["reference_moment"] == F_REF and row["reference_origin"] == [0., 0., 0.] and
        row["load_factor"] == (1. if cursor == "cursor_2" else .75),
        "BALANCE_REFERENCE", label)
    vectors = {key: _array(row[key], (3,), label+" "+key) for key in (
        "applied_couple", "applied_force", "force_closure", "moment_closure",
        "support_force", "support_moment", "support_position", "tip_position")}
    _require(np.array_equal(vectors["applied_couple"], np.zeros(3)) and
        np.array_equal(vectors["applied_force"],
            row["load_factor"]*np.array([0., 1.e-4, 2.e-4])),
        "BALANCE_LOAD", label)
    force = vectors["support_force"]+vectors["applied_force"]
    moment = vectors["support_moment"]+np.cross(vectors["support_position"],
        vectors["support_force"])+np.cross(vectors["tip_position"],
        vectors["applied_force"])
    _require(np.allclose(force, vectors["force_closure"], rtol=1.e-12, atol=1.e-18) and
        np.allclose(moment, vectors["moment_closure"], rtol=1.e-12, atol=1.e-18),
        "BALANCE_RECOMPUTE", label)
    for key, actual in (("force", force), ("moment", moment)):
        _near(row[key+"_raw_norm"], float(np.linalg.norm(actual)), label+" "+key)
        _near(row[key+"_reference_scaled"], float(np.linalg.norm(actual))/F_REF,
            label+" "+key+" scaled")
        screen.check(row[key+"_reference_scaled"] <= THRESHOLDS["balance"],
            "BALANCE_GATE", label+" "+key)


def _left_jacobian(theta):
    theta = np.asarray(theta, dtype=float)
    length = float(np.linalg.norm(theta))
    x, y, z = theta
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    if length < 1.e-7:
        return np.eye(3)+.5*k+(1./6.)*k@k
    return (np.eye(3)+(1.-math.cos(length))/length**2*k+
        (length-math.sin(length))/length**3*k@k)


def _chart_direction(state, spatial, chart, label):
    u = np.asarray(state["shell_u"], dtype=float)
    _require(u.ndim == 1 and u.size % 6 == 0 and
        spatial.shape == chart.shape == u.shape,
        "TANGENT_DIRECTION", label)
    for node, row in enumerate(u.reshape(-1, 6)):
        sl = slice(6*node, 6*node+3)
        rot = slice(6*node+3, 6*node+6)
        _require(np.allclose(spatial[sl], chart[sl], atol=1.e-12, rtol=1.e-12) and
            np.allclose(spatial[rot], _left_jacobian(row[3:])@chart[rot],
                atol=1.e-11, rtol=1.e-11),
            "TANGENT_CHART_MAP", label+f" shell node {node}")


def _scientific(body, checkpoints, reg, att, screen):
    expected_top = {"checkpoint_sha256", "checkpoint_sidecars",
        "checkpoint_sidecars_written", "component_fd", "cursor_3_balance",
        "cursor_3_changes", "fixed_reference_balance", "interface_work",
        "negative_checks", "optional_replay_audit", "paired", "production_qualified",
        "provisional_floors", "runtime_seconds", "schema", "scope",
        "source_commit", "source_sha256", "statuses", "tangent_fd", "targets"}
    _require(expected_top <= set(body) and set(body)-expected_top <= {"effective_environment"},
        "METRICS_INVENTORY", "missing or unexpected metrics blocks")
    if "effective_environment" not in body:
        screen.missing("MISSING_EFFECTIVE_ENVIRONMENT", "runner child environment")
    else:
        screen.check(body["effective_environment"] == reg["environment"] ==
            att["observed_environment"], "EFFECTIVE_ENVIRONMENT",
            "runner child differs from prelaunch attestation")
    for collection in ("paired", "tangent_fd", "component_fd",
            "interface_work", "fixed_reference_balance"):
        _keys(body[collection], CURSORS, collection)
    paired_plan = dict(shell_translation=("translation", "paired_pose"),
        beam_translation=("translation", "paired_pose"),
        shell_rotation=("rotation", "paired_pose"),
        beam_rotation=("rotation", "paired_pose"),
        shell_force=("force", "paired_force_moment"),
        beam_nodal_force=("force", "paired_force_moment"),
        support_force=("force", "paired_force_moment"),
        shell_moment=("moment", "paired_force_moment"),
        beam_nodal_moment=("moment", "paired_force_moment"),
        support_moment=("moment", "paired_force_moment"),
        shell_work=("work", "paired_work"),
        beam_nodal_virtual_work=("work", "paired_work"),
        shell_layer_strain=("strain", "layer"),
        shell_layer_stress=("stress", "layer"),
        physical_tangent_direction=("tangent_fd", "physical_tangent"),
        native_beam_mixed_residual=("solver_scaled_residual", "beam_section"),
        full_solver_residual=("solver_scaled_residual", "paired_force_moment"),
        gap_translation=("gap_translation", None),
        gap_rotation=("gap_rotation", None),
        free_solver_residual=("solver_scaled_residual", None))
    for cursor in CURSORS:
        paired = body["paired"][cursor]
        _require(type(paired) is dict and set(paired) == set(paired_plan) | {
            "accepted_shell_history_equal", "accepted_beam_history_equal",
            "shell_layer_maxima", "beam_section_response"},
            "PAIRED_INVENTORY", cursor)
        screen.check(paired["accepted_beam_history_equal"] is True,
            "BEAM_HISTORY", cursor)
        if cursor == "cursor_2":
            screen.check(paired["accepted_shell_history_equal"] is True,
                "SHELL_HISTORY", cursor)
        for name, (floor, threshold) in paired_plan.items():
            row = _metric(paired[name], floor, cursor+" "+name, screen,
                raw_required=True)
            if threshold is not None:
                screen.check(row["normalized_difference"] <= THRESHOLDS[threshold],
                    "PAIRED_GATE", cursor+" "+name)
        for name in ("gap_translation", "gap_rotation"):
            row = paired[name]
            for key in ("left_norm", "right_norm", "absolute_difference"):
                screen.check(row[key] <= THRESHOLDS["gap_absolute"],
                    "GAP_ABSOLUTE", cursor+" "+name+" "+key)
        residual = paired["free_solver_residual"]
        for key in ("left_norm", "right_norm", "absolute_difference"):
            screen.check(residual[key] <= THRESHOLDS["free_residual_absolute"],
                "FREE_RESIDUAL", cursor+" "+key)
        layers = paired["shell_layer_maxima"]
        _keys(layers, ("strain", "stress"), cursor+" layers")
        for name in ("strain", "stress"):
            maximum = _indexed(layers[name], (3, 3, 3), [name]*3,
                ("integration_station", "layer", "component"), cursor+" layer "+name)
            screen.check(maximum <= THRESHOLDS["layer"],
                "LAYER_GATE", cursor+" "+name)
        beam = paired["beam_section_response"]
        _keys(beam, ("station_identity", "retained_resultants", "section_strain",
            "section_resultants", "section_split"), cursor+" beam recovery")
        identity = beam["station_identity"]
        _require(type(identity) is list and len(identity) == 2,
            "BEAM_STATION_INVENTORY", cursor)
        for member, row in enumerate(identity):
            _require(row.get("member_index") == member and
                row.get("element_id") == member+1 and len(row.get("stations", ())) == 8,
                "BEAM_STATION_INVENTORY", cursor+f" member {member}")
            for station, item in enumerate(row["stations"]):
                _require(item.get("cell") == station//4 and
                    item.get("station") == station%4 and
                    math.isfinite(item.get("xi", math.nan)),
                    "BEAM_STATION_INVENTORY", cursor+f" member {member} station {station}")
        retained = beam["retained_resultants"]
        _require(type(retained.get("component_labels")) is list and
            len(retained["component_labels"]) == 18 and
            len(set(retained["component_labels"])) == 18,
            "RETAINED_ROLES", cursor)
        for name, shape, floors, indexes in (
                ("retained_resultants", (2, 18), ["force"]*6+["moment"]*12,
                 ("member", "component")),
                ("section_strain", (2, 8, 6), ["strain"]*3+["curvature"]*3,
                 ("member", "station", "component")),
                ("section_resultants", (2, 8, 6), ["force"]*3+["moment"]*3,
                 ("member", "station", "component"))):
            maximum = _indexed(beam[name], shape, floors, indexes,
                cursor+" beam "+name)
            screen.check(maximum <= THRESHOLDS["beam_section"],
                "BEAM_SECTION_GATE", cursor+" "+name)
        split = beam["section_split"]
        _keys(split, ("strain_high", "strain_low", "resultant_high", "resultant_low"),
            cursor+" section split")
        for field, high, low, floors in (
                ("section_strain", "strain_high", "strain_low",
                 ["strain"]*3+["curvature"]*3),
                ("section_resultants", "resultant_high", "resultant_low",
                 ["force"]*3+["moment"]*3)):
            for part in (high, low):
                _indexed(split[part], (2, 8, 6), floors,
                    ("member", "station", "component"), cursor+" "+part)
            for side in ("left_values", "right_values"):
                recombined = np.asarray(split[high][side])+np.asarray(split[low][side])
                _require(np.array_equal(recombined, np.asarray(beam[field][side])),
                    "SECTION_SPLIT_RECOMBINE", cursor+" "+field+" "+side)
        _keys(body["fixed_reference_balance"][cursor], SIDES,
            cursor+" balance sides")
        for side in SIDES:
            _balance(body["fixed_reference_balance"][cursor][side],
                cursor, cursor+" "+side, screen)
        tangent = body["tangent_fd"][cursor]
        _require(type(tangent) is dict and {"direction_shell_norm",
            "direction_beam_retained_norm", "direction_multiplier_norm", "samples"} <= set(tangent),
            "TANGENT_INVENTORY", cursor)
        if not {"direction_spatial", "direction_chart"} <= set(tangent):
            screen.missing("MISSING_TANGENT_CHART_DIRECTION", cursor)
        else:
            spatial = np.asarray(tangent["direction_spatial"], dtype=float)
            chart = np.asarray(tangent["direction_chart"], dtype=float)
            _require(spatial.shape == chart.shape and spatial.ndim == 1 and
                np.isfinite(spatial).all() and np.isfinite(chart).all(),
                "TANGENT_DIRECTION", cursor)
            _near(np.linalg.norm(spatial), 1., cursor+" tangent direction")
            rebased_state = (checkpoints["rebase"]["rebased_state"] if
                cursor == "cursor_2" else checkpoints["successor"]["successor"]["state"])
            shell_size = len(rebased_state["shell_u"])
            _chart_direction(rebased_state, spatial[:shell_size], chart[:shell_size],
                cursor+" mixed")
            for key, values in (("direction_shell_norm", spatial[:shell_size]),
                    ("direction_beam_retained_norm", spatial[shell_size:-6]),
                    ("direction_multiplier_norm", spatial[-6:])):
                _near(tangent[key], np.linalg.norm(values), cursor+" "+key)
        for key in ("direction_shell_norm", "direction_beam_retained_norm",
                "direction_multiplier_norm"):
            screen.check(_num(tangent[key], cursor+" "+key) > 0.,
                "ZERO_DIRECTION_GROUP", cursor+" "+key)
        mixed_keys = {"assembled_scaled", "beam_nodal_force", "beam_nodal_moment",
            "gap_rotation", "gap_translation", "retained_mixed", "shell_force",
            "shell_moment", "step_size"}
        _require(type(tangent["samples"]) is list and len(tangent["samples"]) == 4 and
            all(type(row) is dict and set(row) == mixed_keys for row in tangent["samples"]),
            "TANGENT_BLOCK_INVENTORY", cursor)
        for key, floor in (("assembled_scaled", "tangent_fd"),
                ("beam_nodal_force", "force"), ("beam_nodal_moment", "moment"),
                ("gap_rotation", "gap_rotation"),
                ("gap_translation", "gap_translation"),
                ("retained_mixed", "tangent_fd"), ("shell_force", "force"),
                ("shell_moment", "moment")):
            for index, row in enumerate(tangent["samples"]):
                _metric(row[key], floor, f"{cursor} mixed {key} {index}", screen)
        _plateau(tangent["samples"], "assembled_scaled",
            THRESHOLDS["assembled_fd"], cursor+" mixed", screen)
        _keys(body["component_fd"][cursor], SIDES, cursor+" component sides")
        component_keys = {"dead_external_couple", "dead_external_force",
            "joint_port_force", "joint_port_moment", "joint_rotation_gap",
            "joint_translation_gap", "native_beam_nodal_force",
            "native_beam_nodal_moment", "native_beam_retained", "shell_force",
            "shell_moment", "step_size"}
        component_floors = dict(dead_external_couple="moment", dead_external_force="force",
            joint_port_force="force", joint_port_moment="moment",
            joint_rotation_gap="gap_rotation", joint_translation_gap="gap_translation",
            native_beam_nodal_force="force", native_beam_nodal_moment="moment",
            native_beam_retained="tangent_fd", shell_force="force",
            shell_moment="moment")
        for side in SIDES:
            component = body["component_fd"][cursor][side]
            _require(type(component) is dict and {
                "direction_beam_retained", "direction_joint_port", "direction_multiplier",
                "direction_shell", "external_operator_scope", "samples"} <= set(component) and
                component["external_operator_scope"] == "SHARED_CONSTANT_DEAD_LOAD_CALL_PATH",
                "COMPONENT_INVENTORY", cursor+" "+side)
            if "direction_shell_chart" not in component:
                screen.missing("MISSING_COMPONENT_CHART_DIRECTION", cursor+" "+side)
            else:
                state = (checkpoints["base"]["records"][-1]["state"] if
                    cursor == "cursor_2" and side == "control" else
                    checkpoints["control"]["records"][2]["state"] if
                    side == "control" else
                    checkpoints["rebase"]["rebased_state"] if
                    cursor == "cursor_2" else
                    checkpoints["successor"]["successor"]["state"])
                _chart_direction(state, np.asarray(component["direction_shell"], dtype=float),
                    np.asarray(component["direction_shell_chart"], dtype=float),
                    cursor+" "+side+" component")
                if side == "rebased" and {"direction_spatial", "direction_chart"} <= set(tangent):
                    full_spatial = np.r_[component["direction_shell"],
                        component["direction_beam_retained"],
                        component["direction_multiplier"]]
                    full_chart = np.r_[component["direction_shell_chart"],
                        component["direction_beam_retained"],
                        component["direction_multiplier"]]
                    _require(np.allclose(full_spatial, tangent["direction_spatial"],
                        atol=1.e-14, rtol=1.e-14) and
                        np.allclose(full_chart, tangent["direction_chart"],
                            atol=1.e-14, rtol=1.e-14),
                        "TANGENT_DIRECTION", cursor+" mixed/component")
            directions = (component["direction_shell"], component["direction_beam_retained"],
                component["direction_multiplier"])
            screen.check(all(np.linalg.norm(np.asarray(v, dtype=float)) > 0.
                for v in directions), "ZERO_DIRECTION_GROUP", cursor+" "+side)
            _require(type(component["samples"]) is list and
                len(component["samples"]) == 4 and
                all(type(row) is dict and set(row) == component_keys
                    for row in component["samples"]),
                "COMPONENT_BLOCK_INVENTORY", cursor+" "+side)
            for key, floor in component_floors.items():
                for index, row in enumerate(component["samples"]):
                    _metric(row[key], floor,
                        f"{cursor} {side} component {key} {index}", screen)
                if key.startswith("dead_external"):
                    screen.check(all(row[key]["absolute_difference"] <= FLOORS[floor][0]
                        for row in component["samples"]),
                        "DEAD_LOAD_DERIVATIVE", cursor+" "+side+" "+key)
                else:
                    _plateau(component["samples"], key,
                        THRESHOLDS["component_fd"], cursor+" "+side, screen)
        _keys(body["interface_work"][cursor],
            ("control", "rebased", "paired_virtual_work"), cursor+" interface")
        for side in SIDES:
            work = body["interface_work"][cursor][side]
            _require(type(work) is dict and {"positions", "frames", "multipliers",
                "constraints", "spatial_residual", "variations",
                "compatible_rigid_zero_work", "reference_length", "reference_force",
                "reference_work"} <= set(work) and
                work["reference_length"] == 1.,
                "PORT_INVENTORY", cursor+" "+side)
            _near(work["reference_force"], F_REF, cursor+" "+side+" F_ref")
            _near(work["reference_work"], F_REF, cursor+" "+side+" work_ref")
            positions = _array(work["positions"], (2, 3), cursor+" port positions")
            residual = _array(work["spatial_residual"], (12,), cursor+" port residual")
            _array(work["frames"], (2, 3, 3), cursor+" port frames")
            _array(work["multipliers"], (6,), cursor+" multipliers")
            _array(work["constraints"], (6,), cursor+" constraints")
            _keys(work["variations"], ("shell_only", "beam_only", "mixed"),
                cursor+" port variations")
            for name, row in work["variations"].items():
                direction = _array(row["direction"], (2, 6), cursor+" "+name)
                _require((name != "shell_only" or np.array_equal(direction[1], np.zeros(6))) and
                    (name != "beam_only" or np.array_equal(direction[0], np.zeros(6))),
                    "PORT_DIRECTION", cursor+" "+name)
                analytic = float(residual@direction.ravel())
                _near(row["analytic_work"], analytic, cursor+" "+name+" analytic")
                _near(row["reference_scaled_work"], analytic/F_REF,
                    cursor+" "+name+" scaled")
                _require([sample["step_size"] for sample in row["samples"]] == list(STEPS),
                    "PORT_FD_STEPS", cursor+" "+name)
                errors = []
                for sample in row["samples"]:
                    fd = (sample["plus_potential"]-sample["minus_potential"])/(
                        2*sample["step_size"])
                    _near(sample["derivative"], fd, cursor+" "+name+" FD")
                    error = abs(analytic-fd)
                    _near(sample["absolute_error"], error, cursor+" "+name+" error")
                    _near(sample["reference_scaled_error"], error/F_REF,
                        cursor+" "+name+" scaled error")
                    errors.append(error/F_REF)
                screen.check(errors[-1] <= THRESHOLDS["port_work"],
                    "PORT_CONJUGACY", cursor+" "+side+" "+name)
                screen.check(errors[0] > errors[1] > errors[2] or max(errors) <= 1.e-9,
                    "PORT_PLATEAU", cursor+" "+side+" "+name)
            rigid = work["compatible_rigid_zero_work"]
            _keys(rigid, tuple(f"{kind}_{axis}" for kind in ("translation", "rotation")
                for axis in range(3)), cursor+" rigid modes")
            for name, row in rigid.items():
                direction = _array(row["direction"], (2, 6), cursor+" "+name)
                kind, axis = name.rsplit("_", 1); axis = int(axis)
                v, omega = np.zeros(3), np.zeros(3)
                if kind == "translation": v[axis] = 1.
                else: omega[axis] = 1.
                expected = np.array([np.r_[v+np.cross(omega, x), omega]
                    for x in positions])
                _require(np.array_equal(direction, expected),
                    "RIGID_MODE_DIRECTION", cursor+" "+side+" "+name)
                value = float(residual@direction.ravel())
                _near(row["work"], value, cursor+" "+name+" work")
                _near(row["absolute_work"], abs(value), cursor+" "+name+" absolute")
                _near(row["reference_scaled_work"], abs(value)/F_REF,
                    cursor+" "+name+" scaled")
                screen.check(abs(value)/F_REF <= THRESHOLDS["port_work"],
                    "RIGID_ZERO_WORK", cursor+" "+side+" "+name)
        pair = body["interface_work"][cursor]["paired_virtual_work"]
        _keys(pair, ("shell_only", "beam_only", "mixed"), cursor+" work parity")
        for name, row in pair.items():
            old = body["interface_work"][cursor]["control"]["variations"][name]["analytic_work"]
            new = body["interface_work"][cursor]["rebased"]["variations"][name]["analytic_work"]
            _near(row["absolute_difference"], abs(old-new), cursor+" "+name+" parity")
            _near(row["reference_scaled_difference"], abs(old-new)/F_REF,
                cursor+" "+name+" scaled parity")
            screen.check(row["reference_scaled_difference"] <= THRESHOLDS["port_work"],
                "PORT_WORK_PARITY", cursor+" "+name)
    changes = body["cursor_3_changes"]
    _require(type(changes) is dict and set(changes) ==
        {"beam_rotation", "multiplier_force", "multiplier_moment", "shell_work",
         "support_force", "support_moment"},
        "SUCCESSOR_CHANGE_INVENTORY", "third-step metrics incomplete")
    for name, minimum in THRESHOLDS["nontrivial"].items():
        floor = ("rotation" if name == "beam_rotation" else "work" if name == "shell_work"
            else "moment" if name == "support_moment" else "force")
        row = _metric(changes[name], floor, "successor change "+name, screen)
        screen.check(row["absolute_difference"] > minimum,
            "NONTRIVIAL_SUCCESSOR", name)
    successor = checkpoints["successor"]["successor"]
    control = checkpoints["control"]["records"][2]
    screen.check(successor["metric"] <= THRESHOLDS["free_residual_absolute"] and
        successor["correction"] <= THRESHOLDS["free_residual_absolute"] and
        control["metric"] <= THRESHOLDS["free_residual_absolute"] and
        control["correction"] <= THRESHOLDS["free_residual_absolute"],
        "SUCCESSOR_EQUILIBRIUM", "cursor-three accepted metric/correction")
    negative = body["negative_checks"]
    expected = dict(bad_external_hash="external F06 prototype checkpoint hash",
        rehashed_anchor="rebase event mismatch",
        rehashed_successor_history="shell history disagreement",
        rehashed_zero_residual="residual disagreement",
        stale_successor_commit="stale successor predecessor",
        cancel_rebase_before_commit="disposable F06 pilot cancellation",
        cancel_successor_before_commit="disposable F06 pilot cancellation")
    _keys(negative, expected, "negative paths")
    for name, reason in expected.items():
        row = negative[name]
        _keys(row, ("status", "type", "reason", "before_head_sha256",
            "before_generation", "after_head_sha256", "after_generation"),
            "negative path "+name)
        _require(row.get("status") == "REJECTED" and
            row.get("type") == ("SolveCancelled" if name.startswith("cancel_") else "ValueError") and
            type(row.get("reason")) is str and reason in row["reason"],
            "NEGATIVE_PATH", name)
        head = (row["before_head_sha256"], row["before_generation"])
        _require((row["after_head_sha256"], row["after_generation"]) == head,
            "REJECTION_ATOMICITY", name)
        expected_head = ((body["checkpoint_sha256"]["successor"], 2)
            if name == "stale_successor_commit" else
            (body["checkpoint_sha256"]["rebase"], 1)
            if name == "cancel_successor_before_commit" else (None, None))
        _require(head == expected_head, "NEGATIVE_HEAD_IDENTITY", name)
    audits = body["optional_replay_audit"]
    _keys(audits, ("rebase", "successor"), "replay audit")
    for name, row in audits.items():
        screen.check(row.get("status") == "MATCHED" and
            row.get("direct_state_sha256") == row.get("audited_state_sha256") and
            type(row.get("direct_state_sha256")) is str and
            len(row["direct_state_sha256"]) == 64,
            "REPLAY_AUDIT", name)
    terminal = body["fixed_reference_balance"]["cursor_3"]["rebased"]
    for key, other in (("force", "force"), ("moment", "moment")):
        alias = body["cursor_3_balance"][key]
        _near(alias["raw_norm"], terminal[other+"_raw_norm"],
            "terminal balance alias "+key)
        _near(alias["physical_scale"], F_REF, "terminal balance scale "+key)
        _near(alias["normalized"], terminal[other+"_reference_scaled"],
            "terminal balance normalized "+key)


def adjudicate(registration_path: Path, evidence_dir: Path,
               attestation_path: Path, *, dry_run: bool = False) -> dict:
    """Read existing bytes and return a candidate report; never write evidence."""
    screen = Screen()
    evidence_dir = Path(evidence_dir).resolve()
    registration_path = Path(registration_path).resolve()
    attestation_path = Path(attestation_path).resolve()
    try:
        reg, att = _identity(registration_path, evidence_dir, attestation_path, screen)
        body, checkpoints = _artifacts(reg, att, evidence_dir, screen)
        _scientific(body, checkpoints, reg, att, screen)
    except EvidenceError as error:
        screen.missing(error.code, error.detail)
    except (KeyError, IndexError, TypeError, ValueError, FileNotFoundError) as error:
        screen.missing("INCOMPLETE_INVENTORY", str(error))
    return dict(schema="G0_F06_V4_CANDIDATE_ADJUDICATION_V1",
        disposition="CANDIDATE_SCREEN_PASSED" if not screen.failures else
            "CANDIDATE_SCREEN_FAILED", dry_run=bool(dry_run),
        qualification_accepted=False, check_count=screen.check_count,
        failure_count=len(screen.failures), failures=screen.failures,
        evidence_dir=str(evidence_dir))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = adjudicate(args.registration, args.evidence_dir,
        args.attestation, dry_run=args.dry_run)
    sys.stdout.buffer.write(_canonical(result))
    return 0 if result["disposition"] == "CANDIDATE_SCREEN_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
