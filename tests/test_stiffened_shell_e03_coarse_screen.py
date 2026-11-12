"""Synthetic coarse-runner controls; no native solve, subprocess or CCX."""
import copy
import hashlib
import json

import pytest

from scripts import r8_e03_coarse_screen as screen
from scripts import r8_e03_curved_reference as reference


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "_bindings", lambda bindings, phase: copy.deepcopy(bindings))
    monkeypatch.setattr(screen, "MODULE_MANIFEST", "modules.json")
    case = reference.build_case(0, 1)
    provider = tmp_path/"site"
    origin = provider/"anyfesolver/__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_bytes(b"synthetic installed origin")
    (tmp_path/"modules.json").write_text(json.dumps({"source_modules":{"anyfesolver/__init__.py":sha(origin.read_bytes())}}))
    resource = tmp_path/"resource.json"
    resource.write_bytes(b'{"status":"synthetic"}')
    bindings = dict(phase="native", case_id=case["case_id"], source_root=str(tmp_path),
        source_sha256={}, wheel_sha256="synthetic", module_manifest_sha256="synthetic",
        supervisor_id="synthetic", execution_scope="synthetic-controls-only", command=["synthetic"],
        installed_provider_root=str(provider), native_capability_law="synthetic-law")
    folder = screen.prepare_screen("native", tmp_path/"attempt", bindings=bindings)
    result = dict(status="COMPLETED_UNQUALIFIED_B2_CANDIDATE", profile=reference.B2,
        case_id=case["case_id"], level=0, offset_sign=1, production_qualified=False,
        external_work_j=1., force_balance_n=[0., 0., 0.], moment_balance_nm=[0., 0., 0.], scaled_balance=0.,
        root_reaction_wrench_world_origin=[0.]*6,
        tip_poses=[dict(rib_index=i, world_pose=[0.]*6) for i in range(2)],
        native_capability_law="synthetic-law",
        point_multiplier_coordinates="native ReferencePointLaw rows; not world spatial wrenches",
        point_gap_coordinates="native ReferencePointLaw rows; not world pose components")
    for name in ("point_multipliers_native_rows", "point_gaps_native_rows"):
        result[name] = [[f"rib{rib}-point{i}", [0.]*6] for rib in range(2) for i in range(4)]
    receipt = {key:bindings[key] for key in ("phase", "case_id", "supervisor_id", "execution_scope",
        "command", "wheel_sha256", "source_sha256", "module_manifest_sha256")}
    receipt.update(deck_sha256=json.loads((folder/"metadata.json").read_text())["deck_sha256"],
        exit_code=0, timed_out=False, process_wall_seconds=.01, resource_receipt_path=str(resource),
        resource_receipt_sha256=sha(resource.read_bytes()), anyfesolver_origin=str(origin),
        anyfesolver_origin_sha256=sha(origin.read_bytes()))
    return folder, receipt, result


def finish(prepared, *, result=None, receipt=None, raw=None):
    folder, original_receipt, original_result = prepared
    result = original_result if result is None else result
    receipt = original_receipt if receipt is None else receipt
    raw = dict(stdout=b"synthetic output", stderr=b"", native_result=json.dumps(result).encode()) if raw is None else raw
    for name, data in raw.items():
        if type(data) is bytes:
            receipt[name+"_sha256"] = sha(data)
    return screen.finish_screen(folder, receipt=receipt, raw=raw)


def test_complete_native_is_provisional_and_detached(prepared):
    result = finish(prepared)
    prepared[2]["tip_poses"][0]["world_pose"][0] = 123.
    assert result["tip_poses"][0]["world_pose"][0] == 0.
    outcome = json.loads((prepared[0]/"outcome.json").read_text())
    assert not outcome["production_qualified"] and not outcome["accuracy_scored"]
    assert outcome["full_e03"] == outcome["ge_point_correspondence"] == outcome["converged_accuracy"] == "OPEN"


@pytest.mark.parametrize("phase", ["paired", "ge", "", None])
def test_invalid_phase_refused_before_bindings(phase):
    with pytest.raises(ValueError, match="one native or external"):
        screen._bindings({}, phase)


@pytest.mark.parametrize("phase", ["native", "external"])
def test_binding_phase_mismatch(phase):
    with pytest.raises(ValueError, match="one-phase"):
        screen._bindings({"phase":"wrong"}, phase)


def test_actual_source_digest_mismatch(tmp_path):
    for name in screen.SOURCE_FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic source")
    bindings = dict(phase="native", case_id="synthetic", supervisor_id="synthetic", execution_scope="synthetic",
        source_root=str(tmp_path), source_sha256={name:"wrong" for name in screen.SOURCE_FILES})
    with pytest.raises(ValueError, match="source identity mismatch"):
        screen._bindings(bindings, "native")


def test_prepare_case_identity_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "_bindings", lambda bindings, phase: bindings)
    with pytest.raises(ValueError, match="case identity"):
        screen.prepare_screen("native", tmp_path/"bad", bindings={"case_id":"wrong"})
    assert not (tmp_path/"bad").exists()


@pytest.mark.parametrize("field,value", [
    ("exit_code", 1), ("timed_out", True), ("process_wall_seconds", -1.),
    ("case_id", "wrong"), ("source_sha256", {"wrong":"wrong"}),
])
def test_failed_receipt_retained(prepared, field, value):
    receipt = prepared[1]
    receipt[field] = value
    with pytest.raises((ValueError, reference.ReferenceParseError)):
        finish(prepared)
    retained = prepared[0]/"received"
    assert (retained/"native_result.raw").is_file() and (retained/"stdout.raw").read_bytes() == b"synthetic output"
    assert (retained/"receipt.raw.json").is_file() and (retained/"failure.json").is_file()
    assert not (prepared[0]/"outcome.json").exists()


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("external_work_j", 0.), ("force_balance_n", [1., 0., 0.]),
    ("root_reaction_wrench_world_origin", [0.]*5),
    ("tip_poses", [{"rib_index":0,"world_pose":[0.]*6}]*2),
    ("native_capability_law", "foreign-law"), ("point_multipliers_native_rows", []),
    ("point_gap_coordinates", "world"),
])
def test_partial_native_output_refused(prepared, field, value):
    result = prepared[2]
    result[field] = value
    with pytest.raises((ValueError, reference.ReferenceParseError)):
        finish(prepared)
    assert (prepared[0]/"received/failure.json").is_file()
    assert not (prepared[0]/"observables.json").exists()


@pytest.mark.parametrize("name", ["case.json", "metadata.json", "e03.inp"])
def test_prepared_input_mutation_preserves_raw(prepared, name):
    path = prepared[0]/name
    path.write_bytes(b"{}" if name.endswith(".json") else b"changed deck")
    with pytest.raises((ValueError, reference.ReferenceParseError)):
        finish(prepared)
    assert (prepared[0]/"received/native_result.raw").is_file()


@pytest.mark.parametrize("kind", ["missing", "nonbytes", "invalid-json"])
def test_malformed_raw_refused(prepared, kind):
    raw = dict(stdout=b"partial output", stderr=b"", native_result=b"invalid json")
    if kind == "missing":
        del raw["native_result"]
    elif kind == "nonbytes":
        raw["native_result"] = "not bytes"
    with pytest.raises((ValueError, reference.ReferenceParseError)):
        finish(prepared, raw=raw)
    assert (prepared[0]/"received/stdout.raw").read_bytes() == b"partial output"
    assert (prepared[0]/"received/failure.json").is_file()


def test_nonfinite_receipt_preserves_raw(prepared):
    prepared[1]["extra"] = float("nan")
    with pytest.raises(reference.ReferenceParseError, match="finite"):
        finish(prepared)
    assert (prepared[0]/"received/native_result.raw").is_file()


def test_actual_child_origin_hash_mismatch(prepared):
    prepared[1]["anyfesolver_origin_sha256"] = "wrong"
    with pytest.raises(ValueError, match="origin must match"):
        finish(prepared)
    assert (prepared[0]/"received/native_result.raw").is_file()


def test_attempt_cannot_be_finished_twice(prepared):
    finish(prepared)
    with pytest.raises(FileExistsError):
        finish(prepared)


@pytest.mark.parametrize("parser_fails", [False, True])
def test_external_raw_preserved_before_strict_parser(prepared, monkeypatch, parser_fails):
    folder, receipt, _ = prepared
    request = json.loads((folder/"request.json").read_text())
    request["phase"] = "external"
    (folder/"request.json").write_text(json.dumps(request))
    bindings = json.loads((folder/"bindings.json").read_text())
    bindings["phase"] = "external"
    (folder/"bindings.json").write_text(json.dumps(bindings))
    receipt.update(phase="external", executable_path=str(screen.CCX_PATH), executable_sha256=screen.CCX_SHA256, version="2.23")
    raw = dict(stdout=b"CalculiX Version 2.23\n", stderr=b"", frd=b"partial FRD", sta=b"partial STA")
    def parser(*args):
        assert (folder/"received/frd.raw").read_bytes() == raw["frd"]
        if parser_fails:
            raise reference.ReferenceParseError("synthetic strict refusal")
        return {"status":"synthetic parser result", "production_qualified":False}
    monkeypatch.setattr(reference, "parse_result", parser)
    if parser_fails:
        with pytest.raises(reference.ReferenceParseError, match="strict refusal"):
            finish(prepared, raw=raw)
        assert (folder/"received/failure.json").is_file()
    else:
        assert finish(prepared, raw=raw)["production_qualified"] is False
