"""Static G4 closeout guards; never execute mechanics or subprocesses."""
import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "docs/reference_cases/ge_beam3_g4_completion_confirmation_v1.json"
REVIEW = ROOT / "docs/reference_cases/ge_beam3_g4_completion_confirmation_review_v1.json"
STATUS = ROOT / "docs/reference_cases/ge_beam3_g4_completion_status_v1.json"


def strict(path):
    raw = path.read_bytes()
    def reject(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=reject,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode("ascii")
    return value


def test_confirmation_closes_exact_g4_scope():
    value = strict(CONFIRMATION)
    assert value["schema"] == "GE_BEAM3_G4_COMPLETION_CONFIRMATION_V1"
    assert value["terminal"] == "PROVISIONAL_GO_GE_BEAM3_G4_GENERAL_STATIC_MATERIAL_STATE_ONLY"
    assert value["candidate"] == {"commit": "fcbfe48a54bc0c15db64ab94553e42032d4a65c3",
        "tree": "b087bbb03b8c4e7536b88858bc2bd34e988b82ce"}
    assert value["closed_rows"] == ["S19", "S20", "S21", "S22", "S23", "S24"]
    assert value["checks"]["g4_qualified"] is True
    assert value["checks"]["g5_qualified"] is False
    assert value["checks"]["production_qualified"] is False
    assert value["remaining"] == {"G5": True, "PRODUCTION_QUALIFICATION": True, "PUBLIC_ROUTING": True}
    assert value["archive"]["formal_cycle_1"]["scientific_sha256"] == value["archive"]["formal_cycle_2"]["scientific_sha256"]


def test_review_and_status_preserve_boundary():
    digest = sha256(CONFIRMATION.read_bytes()).hexdigest()
    review = strict(REVIEW); status = strict(STATUS)
    assert digest == "a42c275d915d0c70eabc48185e2f23ac1fab1ff988a8a9a1036ee9713e2fa192"
    assert review["scope"]["confirmation_sha256"] == digest
    assert review["decision"] == "ACCEPTED_GE_BEAM3_G4_GENERAL_STATIC_MATERIAL_STATE_ONLY"
    assert review["findings"] == [] and review["reviewer"]["independent"] is True
    assert review["scope"]["g5_qualified"] is False
    assert review["scope"]["production_qualified"] is False
    assert status["confirmation_sha256"] == digest
    assert status["review_sha256"] == sha256(REVIEW.read_bytes()).hexdigest()
    assert status["next_gate"] == "G5_LOADS_INERTIA_SPECTRAL_TRANSIENT"
    assert status["g4_qualified"] is True and status["g5_qualified"] is False
    assert status["public_routing_authorized"] is False


def test_external_cycles_when_archive_is_available():
    value = strict(CONFIRMATION); archive = Path(value["archive"]["path"])
    if not archive.exists():
        pytest.skip("external qualification archive is host-local")
    science = []
    paths = {
        "formal_cycle_1": archive / "formal-cycle-1" / "anyfesolver-beam-qualification-fnb1q3e6",
        "formal_cycle_2": archive / "formal-cycle-2" / "anyfesolver-beam-qualification-s6d9p_57"}
    for key, directory in paths.items():
        raw = (directory / "scientific.json").read_bytes(); science.append(raw)
        expected = value["archive"][key]
        assert len(raw) == expected["scientific_bytes"]
        assert sha256(raw).hexdigest() == expected["scientific_sha256"]
        process = (directory / "process.json").read_bytes()
        assert len(process) == expected["process_bytes"]
        assert sha256(process).hexdigest() == expected["process_sha256"]
        assert json.loads(process)["active_processes"] == 0
    assert science[0] == science[1]


def test_closeout_has_no_mechanics_or_process_imports():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in modules)
    assert "subprocess" not in modules and "numpy" not in modules
