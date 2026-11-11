"""Disposable evidence and corruption checks for the read-only F06 screen."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "f06_v4_numba"))

from anyfesolver._ge_beam3_p5_seeded.core import canonical
from scripts.adjudicate_b3_ge_e4_pl_g0_f06_v4 import (
    F_REF, ROOT, THRESHOLDS, _canonical, adjudicate,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_history_available():
    probe = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT,
                           capture_output=True)
    return probe.returncode == 0


@pytest.fixture(scope="module")
def pilot_measurement(tmp_path_factory):
    for name in ("f06-v4-candidate-92e-10", "f06-v4-candidate-92e-09", "f06-v4-candidate-92e-08", "f06-v4-candidate-92e-07", "f06-v4-candidate-92e-06", "f06-v4-candidate-92e-05", "f06-v4-candidate-92e-04",
                 "f06-v4-candidate-ac9f-03"):
        existing = Path(tempfile.gettempdir()) / name
        if (existing / "metrics.json").is_file():
            body = json.loads((existing / "metrics.json").read_bytes())["metrics"]
            if all((ROOT / path).is_file() and _digest(ROOT / path) == digest
                    for path, digest in body["source_sha256"].items()):
                return existing
    # Fallback remains disposable; it creates no registration or receipt.
    from scripts.probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4 import measure_pilot
    if not _git_history_available():
        pytest.skip("no saved f06-v4 candidate measurement bytes are present "
            "and the actual immutable git history is absent from this source "
            "export; a disposable pilot measurement cannot record honest "
            "source provenance, and archived raw arrays are not new acceptance")
    directory = tmp_path_factory.mktemp("f06_v4_disposable_measurement")
    (directory / "metrics.json").write_bytes(canonical(
        measure_pilot(checkpoint_dir=directory)))
    return directory


def _evidence(tmp_path, measurement):
    root = tmp_path / "copied_pilot"
    root.mkdir(parents=True)
    shutil.copytree(measurement, root / "measurement")
    body = json.loads((root / "measurement/metrics.json").read_bytes())["metrics"]
    base = json.loads((root / "measurement/base.checkpoint").read_bytes())
    (root / "test.log").write_text("synthetic dry-run test log; no registration consumed\n")
    (root / "stdout.log").write_text("synthetic copied pilot\n")
    (root / "stderr.log").write_text("")
    sources = dict(body["source_sha256"])
    for name in ("scripts/adjudicate_b3_ge_e4_pl_g0_f06_v4.py",
                 "scripts/supervise_b3_ge_e4_pl_g0_f06_v4.py",
                 "docs/agent_plans/B3_GE_E4_PL_G0_F06_OWNED_REBASE_V4_PROTOCOL.md",
                 "tests/test_b3_e4_f06_v4_adjudicator.py",
                 "tests/test_b3_e4_f06_v4_supervisor.py"):
        sources[name] = _digest(ROOT / name)
    environment = body.get("effective_environment") or dict(
        python_executable=str(Path(sys.executable).resolve()),
        python_version=sys.version.split()[0], numpy_version="dry-run",
        scipy_version="dry-run", anyfesolver_import=str((ROOT / "src/anyfesolver/__init__.py").resolve()),
        psutil_version="dry-run", threadpoolctl_version="dry-run", platform="dry-run",
        numba_cache_dir=None, thread_environment={}, blas=[])
    command = dict(argv=[environment["python_executable"],
        str((ROOT / "scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py").resolve()),
        "--output-dir", str(root / "measurement")], cwd=str(ROOT.resolve()))
    registration = dict(schema="G0_F06_V4_SCIENTIFIC_REGISTRATION_V1",
        source_commit=body["source_commit"], sources=sources,
        environment=environment, command=command,
        limits=dict(wall_seconds=180, sampled_tree_rss_bytes=2*1024**3),
        case=dict(targets=[.5, 1., .75], L_ref=1., F_ref=F_REF,
            M_ref=F_REF, definition_sha256=base["definition_sha256"]),
        thresholds=THRESHOLDS, evidence_dir=str(root),
        test_log_sha256=_digest(root / "test.log"))
    (root / "registration.json").write_bytes(_canonical(registration))
    _refresh(root)
    return root


def _refresh(root):
    registration_path = root / "registration.json"
    registration = json.loads(registration_path.read_bytes())
    registration_sha = _digest(registration_path)
    (root / "execution.claim").write_bytes(_canonical(dict(
        registration_sha256=registration_sha,
        claimed_at_utc="2026-09-23T12:00:00+00:00")))
    measurement = root / "measurement"
    artifacts = dict(metrics=dict(file="metrics.json",
        byte_count=(measurement / "metrics.json").stat().st_size,
        sha256=_digest(measurement / "metrics.json")),
        checkpoints={name: dict(file=name + ".checkpoint",
            byte_count=(measurement / (name + ".checkpoint")).stat().st_size,
            sha256=_digest(measurement / (name + ".checkpoint")))
            for name in ("base", "rebase", "successor", "control")})
    attestation = dict(schema="G0_F06_V4_SUPERVISOR_ATTESTATION_V1",
        registration_sha256=registration_sha, command=registration["command"],
        observed_sources=registration["sources"],
        observed_environment=registration["environment"],
        start_utc="2026-09-23T12:00:00+00:00",
        end_utc="2026-09-23T12:00:40+00:00", wall_seconds=40.,
        peak_tree_rss_bytes=1000000, rss_sample_count=10, root_pid=1234,
        last_job_pids=[1234], exit_code=0, resource_violation=None,
        launch_error=None, observed_child_environment=registration["environment"],
        child_environment_match=True,
        stdout_sha256=_digest(root / "stdout.log"),
        stderr_sha256=_digest(root / "stderr.log"),
        test_log_sha256=_digest(root / "test.log"),
        metrics_sha256=artifacts["metrics"]["sha256"], artifacts=artifacts)
    (root / "attestation.json").write_bytes(_canonical(attestation))


def _screen(root):
    return adjudicate(root / "registration.json", root,
        root / "attestation.json", dry_run=True)


def _codes(result):
    return {row["code"] for row in result["failures"]}


def test_copied_pilot_is_only_a_candidate_screen(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    result = _screen(root)
    assert result["qualification_accepted"] is False
    assert result["check_count"] > 500
    assert result["disposition"] == "CANDIDATE_SCREEN_PASSED", result["failures"]
    assert result["failure_count"] == 0


def test_corrupt_metrics_bytes_reject_before_science(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    path = root / "measurement/metrics.json"
    path.write_bytes(path.read_bytes() + b" ")
    result = _screen(root)
    assert "NONCANONICAL_JSON" in _codes(result)


def test_rehashed_balance_corruption_fails_recomputation(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    path = root / "measurement/metrics.json"
    envelope = json.loads(path.read_bytes())
    envelope["metrics"]["fixed_reference_balance"]["cursor_3"]["rebased"][
        "force_closure"][0] += 1.e-4
    envelope["metrics_sha256"] = hashlib.sha256(canonical(envelope["metrics"])).hexdigest()
    path.write_bytes(canonical(envelope))
    _refresh(root)
    result = _screen(root)
    assert "BALANCE_RECOMPUTE" in _codes(result)


def test_rehashed_fd_raw_summary_mismatch_fails(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    path = root / "measurement/metrics.json"
    envelope = json.loads(path.read_bytes())
    envelope["metrics"]["tangent_fd"]["cursor_2"]["samples"][0][
        "shell_force"]["right_values"][0][0] += 1.e-4
    envelope["metrics_sha256"] = hashlib.sha256(canonical(envelope["metrics"])).hexdigest()
    path.write_bytes(canonical(envelope))
    _refresh(root)
    assert "INCONSISTENT_METRIC" in _codes(_screen(root))


def test_rehashed_rejection_head_change_fails(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    path = root / "measurement/metrics.json"
    envelope = json.loads(path.read_bytes())
    envelope["metrics"]["negative_checks"]["stale_successor_commit"][
        "after_generation"] = 3
    envelope["metrics_sha256"] = hashlib.sha256(canonical(envelope["metrics"])).hexdigest()
    path.write_bytes(canonical(envelope))
    _refresh(root)
    assert "REJECTION_ATOMICITY" in _codes(_screen(root))


def test_missing_section_split_fails_inventory(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    path = root / "measurement/metrics.json"
    envelope = json.loads(path.read_bytes())
    del envelope["metrics"]["paired"]["cursor_2"]["beam_section_response"][
        "section_split"]["strain_low"]
    envelope["metrics_sha256"] = hashlib.sha256(canonical(envelope["metrics"])).hexdigest()
    path.write_bytes(canonical(envelope))
    _refresh(root)
    assert "INCOMPLETE_INVENTORY" in _codes(_screen(root))


def test_sidecar_and_threshold_corruptions_reject(tmp_path, pilot_measurement):
    root = _evidence(tmp_path, pilot_measurement)
    sidecar = root / "measurement/successor.checkpoint"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    assert "NONCANONICAL_JSON" in _codes(_screen(root))
    second = _evidence(tmp_path / "second", pilot_measurement)
    path = second / "registration.json"
    registration = json.loads(path.read_bytes())
    registration["thresholds"]["balance"] = 1.e-3
    path.write_bytes(_canonical(registration))
    _refresh(second)
    assert "THRESHOLD_CONTRACT" in _codes(_screen(second))
