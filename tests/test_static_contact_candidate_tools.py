"""Offline tool tests for the slice-2 static-contact candidate build/check adapters.

These exercise the actual repository tools (config schema, solver staging,
case construction, JUnit counting, cross-host comparison) without running the
solver: no installs, no network, no numerical execution.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import build_static_contact_candidate as build
from scripts import check_static_contact_candidate as check

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "scripts" / "contact_candidate_050.json"
CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def test_tools_load_the_050_successor_config() -> None:
    assert build.CONFIG_PATH == CONFIG_FILE
    assert check.CONFIG_PATH == CONFIG_FILE


def test_config_recipe_is_complete_and_matches_the_checkout() -> None:
    assert CONFIG["candidate_version"] == "0.5.0+contact.candidate.20261111.s7"
    assert CONFIG["repo_version"] == "0.5.0"
    assert CONFIG["private_versions"]["anyfesolver-results"] == "2.2.0"
    assert CONFIG["wheel_files"]["anyfesolver-results"] == (
        "anyfesolver_results-2.2.0-py3-none-any.whl")
    assert CONFIG["private_versions"]["ANYfeSolver"] == CONFIG["candidate_version"]
    assert CONFIG["wheel_files"]["ANYfeSolver"] == (
        "anyfesolver-" + CONFIG["candidate_version"] + "-py3-none-any.whl")
    assert len(CONFIG["base_selectors"]) == 26
    assert "elastic_sphere_panel_contact_skins" in CONFIG["bundle_examples"]
    assert CONFIG["example_commands"]["contact-skins"] == [
        "examples.elastic_sphere_panel_contact_skins", "--output",
        "examples/contact-skins.json"]
    # The s7 curved-skins regressions live in their own named additional
    # selector batch, never in the unchanged original base batch.
    batches = CONFIG["additional_selector_batches"]
    assert set(batches) == {"curved-skins", "elastic-initial-state"}
    curved = batches["curved-skins"]
    initial = batches["elastic-initial-state"]
    assert len(initial) == 9
    assert all(s.startswith("tests/test_elastic_sphere_stiffened_panel.py::")
               for s in initial)
    assert not set(initial) & set(CONFIG["base_selectors"])
    assert len(set(initial + CONFIG["base_selectors"])) == 35
    assert len(curved) == 13
    assert not set(curved) & set(CONFIG["base_selectors"])
    for selector in curved + initial:
        path_, node = selector.split("::")
        assert (ROOT / path_).is_file(), path_
        assert f"def {node}(" in (ROOT / path_).read_text(encoding="utf-8"), selector
    for selector in CONFIG["base_selectors"]:
        path, node = selector.split("::")
        assert (ROOT / path).is_file(), path
        assert f"def {node}(" in (ROOT / path).read_text(encoding="utf-8"), selector
    for node_key in ("friction_lifecycle_node", "friction_rollback_node",
                     "orientation_node"):
        path, node = CONFIG[node_key].split("::")
        assert f"def {node}(" in (ROOT / path).read_text(encoding="utf-8")
    assert CONFIG["friction_stages"] == ["full", "active-prefix", "prefix",
                                         "active-resume", "resume", "rollback"]
    for quantities in (CONFIG["cross_host_quantities"],):
        for _file, _path, family, scale in quantities:
            assert family and scale > 0
    assert CONFIG["timeouts"]["case_seconds"] == 120
    assert CONFIG["timeouts"]["host_numerical_seconds"] == 600
    assert CONFIG["timeouts"]["rss_limit_mib"] == 953
    for package, filename in CONFIG["wheel_files"].items():
        assert CONFIG["private_versions"][package] in filename


IDENTITY_KEYS = {"status", "note", "predecessor", "candidate_version",
                 "repo_version", "private_versions", "wheel_files"}
PREDECESSOR_FILE = ROOT / "scripts" / "contact_candidate_slice2.json"


def test_successor_numerical_fields_equal_the_frozen_predecessor() -> None:
    predecessor = json.loads(PREDECESSOR_FILE.read_text(encoding="utf-8"))
    # The predecessor keeps every historical identity and scope.
    assert predecessor["candidate_version"] == "0.4.7+contact.candidate.20261020.s7"
    assert predecessor["repo_version"] == "0.4.7"
    assert predecessor["private_versions"]["anysolver-results"] == "2.1.1"
    assert CONFIG["predecessor"]["config"] == "scripts/contact_candidate_slice2.json"
    assert CONFIG["predecessor"]["candidate_version"] == predecessor["candidate_version"]
    assert CONFIG["predecessor"]["authority_renewed"] is False
    assert CONFIG["predecessor"]["numerical_run_made"] is False
    # Same key set; every non-identity field (pins, selectors, batches, stages,
    # nodes, examples, caps, cross-host files/quantities/exact/bound, scales,
    # sibling dirs and refs) is exactly equal.
    assert set(CONFIG) - {"predecessor"} == set(predecessor)
    for key in sorted(set(predecessor) - IDENTITY_KEYS):
        assert CONFIG[key] == predecessor[key], key
    # Identity differs exactly where intended.
    assert CONFIG["candidate_version"] != predecessor["candidate_version"]
    assert CONFIG["timeouts"] == dict(case_seconds=120,
                                      host_numerical_seconds=600,
                                      rss_limit_mib=953)
    # Third-party private wheels keep the predecessor's versions and files.
    for package in ("ANYmaterial", "ANYgeometry", "ANYmesher", "ANYfileio"):
        assert CONFIG["private_versions"][package] == \
            predecessor["private_versions"][package]
        assert CONFIG["wheel_files"][package] == predecessor["wheel_files"][package]


def test_solver_staging_changes_only_the_version_string(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    identity = build.stage_solver(ROOT, stage, CONFIG["candidate_version"])
    staged = (stage / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{CONFIG["candidate_version"]}"' in staged
    assert f'version = "{CONFIG["repo_version"]}"' not in staged
    original = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{CONFIG["repo_version"]}"' in original
    assert identity["repo_version_unchanged"] and identity["version_change_only"]
    assert identity["staged_file_count"] > 100
    assert len(identity["staged_source_digest"]) == 64
    assert (stage / "src" / "anyfesolver" / "general_contact").is_dir()
    assert not list(stage.rglob("__pycache__"))


def test_build_cases_order_env_and_unique_outputs(tmp_path: Path) -> None:
    cases = check.build_cases(CONFIG, ROOT, tmp_path / "evidence",
                              tmp_path / "results", tmp_path / "out",
                              CONFIG["friction_stages"] + CONFIG["orientation_stages"],
                              True)
    names = [case["name"] for case in cases]
    batches = [f"{name}-batch" for name in CONFIG["additional_selector_batches"]]
    assert names[0] == "base-batch"
    assert names[1:1 + len(batches)] == batches
    offset = 1 + len(batches)
    assert names[offset:offset + 6] == [
        f"friction-{s}" for s in CONFIG["friction_stages"]]
    assert names[offset + 6] == "orientation-full-plus"
    assert names[offset + 7:offset + 10] == [
        "example-normal", "example-friction", "example-archive"]
    assert names[offset + 10:] == [
        "example-restart-partial", "example-restart", "example-contact-skins"]
    skin = next(c for c in cases if c["name"] == "example-contact-skins")
    assert skin["kind"] == "module"
    assert skin["argv"] == [
        "examples.elastic_sphere_panel_contact_skins", "--output",
        str(tmp_path / "evidence" / "examples" / "contact-skins.json")]
    outputs = [(case["name"], str(case.get("junit")), str(case["env"]))
               for case in cases]
    assert len(set(outputs)) == len(outputs)
    resume = next(c for c in cases if c["name"] == "friction-active-resume")
    assert resume["env"]["FRICTION_STAGE"] == "active-resume"
    assert resume["env"]["AUTOMATIC_DOMAIN_RESULT"].endswith("active-resume.json")
    rollback = next(c for c in cases if c["name"] == "friction-rollback")
    assert "rollback" in rollback["env"]["FRICTION_STAGE"]
    assert CONFIG["friction_rollback_node"].split("::")[1] in rollback["argv"][-1]
    orientation = next(c for c in cases if c["name"] == "orientation-full-plus")
    assert orientation["env"]["ANY_RUN_S3_PHYSICAL_ORIENTATION"] == "1"
    assert orientation["env"]["S3_PHYSICAL_ORIENTATION_STAGE"] == "full-plus"
    base = cases[0]
    assert len(base["argv"]) >= 14 and "CONTACT_APP_RESULT_RECORD" in base["env"]
    batch_case = cases[1]
    assert batch_case["name"] == "curved-skins-batch"
    assert batch_case["kind"] == "pytest"
    assert batch_case["argv"][2] == f"--junitxml={batch_case['junit']}"
    assert batch_case["argv"][3:] == [
        str(ROOT / s) for s in CONFIG["additional_selector_batches"]["curved-skins"]]
    assert batch_case["junit"].name == "curved-skins-junit.xml"
    assert batch_case["env"] == {}
    initial_case = cases[2]
    assert initial_case["name"] == "elastic-initial-state-batch"
    assert initial_case["kind"] == "pytest"
    assert initial_case["argv"][3:] == [str(ROOT / s) for s in CONFIG["additional_selector_batches"]["elastic-initial-state"]]
    assert initial_case["junit"].name == "elastic-initial-state-junit.xml"
    assert initial_case["env"] == {}


def test_junit_counts_classifies_failures_errors_and_skips(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text("""<testsuite><testcase/><testcase><failure/></testcase>
<testcase><error/></testcase><testcase><skipped/></testcase></testsuite>""",
                    encoding="utf-8")
    counts = check.junit_counts(junit)
    assert counts == dict(tests=4, failures=1, errors=1, skips=1)


def test_get_path_navigation() -> None:
    data = dict(a=[dict(b=[1.0, 2.0]), dict(b=[3.0, 4.0])], status="completed")
    assert check.get_path(data, "status") == "completed"
    assert check.get_path(data, "a[*].b[0]") == [1.0, 3.0]
    assert check.get_path(data, "a[*].b") == [[1.0, 2.0], [3.0, 4.0]]
    assert check.get_path(data, "a[1].b[1]") == 4.0


def set_path(obj: dict, path: str, value) -> None:
    segments = path.split(".")
    node = obj
    for segment in segments[:-1]:
        if segment.endswith("[*]"):
            key = segment[:-3]
            node.setdefault(key, [{}])
            node = node[key][0]
        else:
            node = node.setdefault(segment, {})
    last = segments[-1]
    if last.endswith("[*]"):
        node[last[:-3]] = [value]
    elif "[" in last:
        key, index = last[:-1].split("[")
        index = int(index)
        values = node.setdefault(key, [])
        values.extend([None] * (index + 1 - len(values)))
        values[index] = value
    else:
        node[last] = value


def synthetic_evidence(root: Path, delta: float = 0.0, exact_break: str | None = None):
    for file_key, relative in CONFIG["cross_host_files"].items():
        document: dict = {}
        for _file, path, _family, scale in CONFIG["cross_host_quantities"]:
            if _file == file_key:
                set_path(document, path, 0.5 * scale + delta)
        for _file, path in CONFIG["cross_host_exact"]:
            if _file == file_key:
                set_path(document, path, "exact-value" if path != exact_break else "broken")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(json.dumps(document), encoding="utf-8")


def test_compare_passes_within_and_fails_beyond_the_adopted_bound(tmp_path: Path):
    reference, candidate = tmp_path / "windows", tmp_path / "ubuntu"
    synthetic_evidence(reference)
    synthetic_evidence(candidate, delta=1e-15)
    assert check.compare(CONFIG, reference, candidate, tmp_path / "report.json") == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS" and len(report["entries"]) == \
        len(CONFIG["cross_host_quantities"]) + len(CONFIG["cross_host_exact"])
    synthetic_evidence(candidate, delta=1e-3)
    assert check.compare(CONFIG, reference, candidate, tmp_path / "report2.json") == 1
    report = json.loads((tmp_path / "report2.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    synthetic_evidence(candidate, delta=0.0, exact_break="status")
    assert check.compare(CONFIG, reference, candidate, tmp_path / "report3.json") == 1


def test_private_requirements_pin_all_six_packages() -> None:
    requirements = check.private_requirements(CONFIG)
    assert len(requirements) == 6
    assert "ANYgeometry[planar]==0.4.3" in requirements
    assert all("==" in r for r in requirements)


def fake_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    wheels = []
    for package, filename in CONFIG["wheel_files"].items():
        data = f"wheel-bytes-{package}".encode()
        (wheelhouse / filename).write_bytes(data)
        wheels.append(dict(package=package, file=filename,
                           sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(status="BUILT",
                   wheels=sorted(wheels, key=lambda w: w["package"]),
                   config_sha256=hashlib.sha256(
                       CONFIG_FILE.read_bytes()).hexdigest())
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def test_verify_manifest_rejects_corrupted_or_incomplete_bundles(tmp_path: Path) -> None:
    bundle = fake_bundle(tmp_path)
    assert check.verify_manifest(bundle, CONFIG)["status"] == "BUILT"
    (bundle / "wheelhouse" / CONFIG["wheel_files"]["ANYfeSolver"]).write_bytes(b"corrupt")
    with pytest.raises(SystemExit, match="digest mismatch"):
        check.verify_manifest(bundle, CONFIG)
    bundle = fake_bundle(tmp_path / "missing-wheel")
    (bundle / "wheelhouse" / CONFIG["wheel_files"]["ANYmesher"]).unlink()
    with pytest.raises(SystemExit, match="manifest wheel missing"):
        check.verify_manifest(bundle, CONFIG)


def test_compare_rejects_nonfinite_and_missing_configured_quantities(tmp_path: Path):
    reference, candidate = tmp_path / "windows", tmp_path / "ubuntu"
    synthetic_evidence(reference)
    synthetic_evidence(candidate)
    friction_file = candidate / CONFIG["cross_host_files"]["friction_full"]
    document = json.loads(friction_file.read_text(encoding="utf-8"))
    document["displacements"] = float("inf")
    friction_file.write_text(json.dumps(document), encoding="utf-8")
    assert check.compare(CONFIG, reference, candidate, tmp_path / "r1.json") == 1
    report = json.loads((tmp_path / "r1.json").read_text(encoding="utf-8"))
    assert any("nonfinite comparison input" in (e.get("failure") or "")
               for e in report["entries"])
    synthetic_evidence(candidate)
    (candidate / CONFIG["cross_host_files"]["orientation_full"]).unlink()
    assert check.compare(CONFIG, reference, candidate, tmp_path / "r2.json") == 1
    report = json.loads((tmp_path / "r2.json").read_text(encoding="utf-8"))
    assert any("missing configured quantity" in (e.get("failure") or "")
               for e in report["entries"])


def test_spawn_refuses_launch_without_psutil(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check, "psutil", None)
    result = check.spawn([sys.executable], {"jobfile": "unused"}, 1.0, {},
                        tmp_path / "out.txt", tmp_path / "err.txt", 0)
    assert result["status"] == "supervision_unavailable"
    assert "psutil unavailable" in result["failure"]
    assert result["survivors"] == []


@pytest.mark.parametrize("survives", (False, True))
@pytest.mark.parametrize("reason,cap", (("timeout", 0), ("rss_limit", 1)))
def test_timeout_waits_for_owned_members_and_reports_unknown_cleanup(tmp_path, monkeypatch, survives, reason, cap):
    from types import SimpleNamespace
    class Member:
        pid = 123
        alive = True
        def children(self, recursive=True): return []
        def memory_info(self): return SimpleNamespace(rss=1)
        def is_running(self): return self.alive
        def kill(self): pass  # Termination acknowledgement arrives at wait.
    member = Member()
    class Child:
        pid = 123
        returncode = None
        def poll(self): return None
        def wait(self, timeout=None): self.returncode = -9
    waits=[]
    def wait_procs(members, timeout):
        waits.append((members, timeout))
        member.alive = survives
        return ([], members) if survives else (members, [])
    monkeypatch.setattr(check.subprocess, 'Popen', lambda *a, **kw: Child())
    monkeypatch.setattr(check.psutil, 'Process', lambda *a: member)
    monkeypatch.setattr(check.psutil, 'wait_procs', wait_procs)
    result=check.spawn([sys.executable], {'jobfile':'unused'}, 0., {},
                       tmp_path/'out', tmp_path/'err', cap)
    assert len(waits)==1 and waits[0][0]==[member]
    assert 0 <= waits[0][1] <= 5.
    assert result['termination_reason']==reason
    assert result['status']==('cleanup_unknown' if survives else reason)
    assert result['survivors']==([123] if survives else [])


def test_real_timeout_reaps_worker_tree(tmp_path):
    if check.psutil is None:
        pytest.fail('process cleanup integration requires psutil')
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"
    result=check.spawn([sys.executable,'-c',code], {'jobfile':'unused'}, .5,
                      check.clean_env({}), tmp_path/'out', tmp_path/'err', 0)
    assert result['status']=='timeout'
    assert result['termination_reason']=='timeout'
    assert result['survivors']==[] and result['cleanup_errors']==[]


def test_worker_rejects_corrupt_argv_and_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown worker kind"):
        check.run_worker(dict(kind="bogus", inserts=[]))
    source = tmp_path / "partial.json"
    source.write_text(json.dumps(dict(status="ok")), encoding="utf-8")
    with pytest.raises(SystemExit, match="extract needs source and target"):
        check.run_worker(dict(kind="extract", inserts=[], argv=[str(source)]))
    with pytest.raises(SystemExit, match="no checkpoint field"):
        check.run_worker(dict(kind="extract", inserts=[],
                              argv=[str(source), str(tmp_path / "cp.json")]))


def test_example_commands_rebase_onto_transferred_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "windows" / "evidence"
    cases = check.build_cases(CONFIG, ROOT, evidence, tmp_path / "results",
                              tmp_path / "out", CONFIG["friction_stages"], True)
    by_name = {case["name"]: case for case in cases}
    friction = by_name["example-friction"]
    assert friction["argv"] == ["examples.general_contact_s3_friction_physical_path",
                               "--output", str(evidence / "examples" / "friction.json")]
    partial = by_name["example-restart-partial"]
    checkpoint = partial["argv"][partial["argv"].index("--checkpoint-output") + 1]
    assert checkpoint == str(evidence / "examples" / "restart-checkpoint.json")
    assert "example-restart-extract" not in by_name
    restart = by_name["example-restart"]
    assert restart["argv"][restart["argv"].index("--restart") + 1] == \
        str(evidence / "examples" / "restart-checkpoint.json")


def test_build_copies_exactly_the_delivered_examples(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    examples = checkout / "examples"
    examples.mkdir(parents=True)
    for name in CONFIG["bundle_examples"] + ["general_contact_experimental_dev"]:
        (examples / f"{name}.py").write_text("# example\n", encoding="utf-8")
    (examples / "__pycache__").mkdir()
    (examples / "__pycache__" / "stale.pyc").write_text("", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    delivered = build.copy_delivered_examples(checkout, bundle, CONFIG)
    assert delivered == CONFIG["bundle_examples"]
    assert sorted(p.name for p in (bundle / "examples").iterdir()) == \
        sorted(["__init__.py"] + [f"{name}.py" for name in CONFIG["bundle_examples"]])


def test_transferred_restart_uses_public_cli_and_refuses_bad_parity(tmp_path: Path) -> None:
    cases = check.build_cases(CONFIG, ROOT, tmp_path / "evidence", tmp_path / "results",
                              tmp_path, ["active-resume", "resume"], False)
    assert [c["kind"] for c in cases] == ["extract", "module", "extract", "module"]
    assert all(c["kind"] != "pytest" for c in cases)
    checkpoint = dict.fromkeys(("schema", "version", "integrity_id", "analysis_kind",
                               "model_fingerprint", "analysis_fingerprint", "total_dofs"), "same")
    checkpoint["displacements"] = [0.0, 1.0]
    phase = dict(step_index=1, reaction_generalized=[0.01, 0.02],
                 friction=dict(branch="stick", target_elements=[1], dissipation_increment_j=0.0))
    document = dict(status="completed", checkpoint=checkpoint, phases=[phase])
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(document), encoding="utf-8")
    right.write_text(json.dumps(document), encoding="utf-8")
    assert check.resume_parity(left, right, CONFIG)["status"] == "PASS"
    document["checkpoint"]["model_fingerprint"] = "changed"
    right.write_text(json.dumps(document), encoding="utf-8")
    assert check.resume_parity(left, right, CONFIG)["status"] == "FAIL"
    document["checkpoint"]["model_fingerprint"] = "same"
    document["checkpoint"]["displacements"] = [0.0, 1.01]
    right.write_text(json.dumps(document), encoding="utf-8")
    assert check.resume_parity(left, right, CONFIG)["status"] == "FAIL"
