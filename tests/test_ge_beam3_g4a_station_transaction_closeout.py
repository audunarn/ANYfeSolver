"""Static G4a closeout guards; never execute mechanics or external processes."""
import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "docs/reference_cases/ge_beam3_g4a_station_transaction_confirmation_v1.json"
REVIEW = ROOT / "docs/reference_cases/ge_beam3_g4a_station_transaction_confirmation_review_v1.json"
STATUS = ROOT / "docs/reference_cases/ge_beam3_g4a_station_transaction_status_v1.json"


def strict(path):
    raw = path.read_bytes()
    def reject(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=reject,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode("ascii")
    return value


def test_confirmation_is_canonical_scoped_and_complete():
    value = strict(CONFIRMATION)
    assert value["schema"] == "GE_BEAM3_G4A_STATION_TRANSACTION_CONFIRMATION_V1"
    assert value["terminal"] == "PROVISIONAL_GO_GE_BEAM3_G4A_STATION_TRANSACTION_ONLY"
    assert value["candidate"] == {
        "commit": "d5783b48103066e5f17ac000b5aece6d3fe8a2bf",
        "tree": "0c3c4e3461fd09ba9927fef7dbdd5deb636ef139"}
    assert value["checks"] == {
        "accepted_epochs": 6, "atomic_publication": True,
        "cycle_science_byte_identical": True, "failure_matrix_entries": 16,
        "g4_qualified": False, "g4a_station_transaction_closed": True,
        "independent_response_replay": True, "physical_fibre_recovery": True,
        "production_qualified": False, "station_families": 3}
    assert value["archive"]["failed_rehearsal_inventory_v1"]["scientific_aggregate_created"] is False
    assert value["archive"]["formal_cycle_1"]["scientific_sha256"] == value["archive"]["formal_cycle_2"]["scientific_sha256"]
    assert all(value["remaining"].values())


def test_review_and_status_bind_confirmation_without_expanding_scope():
    confirmation_raw = CONFIRMATION.read_bytes()
    review = strict(REVIEW); status = strict(STATUS)
    digest = sha256(confirmation_raw).hexdigest()
    assert digest == "5ab81c37cb693d09221001d8887f9406e5b9de4ab314d84fd7e94190d0af798b"
    assert review["scope"]["confirmation_sha256"] == digest
    assert review["decision"] == "ACCEPTED_GE_BEAM3_G4A_STATION_TRANSACTION_ONLY"
    assert review["findings"] == [] and review["reviewer"]["independent"] is True
    assert review["scope"]["g4_qualified"] is False
    assert review["scope"]["production_qualified"] is False
    assert status["confirmation_sha256"] == digest
    assert status["review_sha256"] == sha256(REVIEW.read_bytes()).hexdigest()
    assert status["remaining_rows"] == ["S19", "S20", "S21", "S22", "S23", "S24"]
    assert status["next_gate"] == "G4B_ASSEMBLED_HISTORY_GRAPH_AND_SOLVER_CONTROL"
    assert status["g4_qualified"] is False and status["production_qualified"] is False


def test_external_formal_evidence_when_archive_is_available():
    value = strict(CONFIRMATION); archive = Path(value["archive"]["path"])
    if not archive.exists():
        pytest.skip("external qualification archive is host-local")
    scientific = []
    for cycle in ("formal-cycle-1", "formal-cycle-2"):
        directory = archive / cycle
        raw = (directory / "scientific.json").read_bytes()
        scientific.append(raw)
        expected = value["archive"][cycle.replace("-", "_")]
        assert len(raw) == expected["scientific_bytes"]
        assert sha256(raw).hexdigest() == expected["scientific_sha256"]
        process = (directory / "process.json").read_bytes()
        assert sha256(process).hexdigest() == expected["process_sha256"]
        assert json.loads(process)["active_processes"] == 0
    assert scientific[0] == scientific[1]


def test_closeout_is_static_and_has_no_mechanics_import():
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in modules)
    assert "subprocess" not in modules and "numpy" not in modules
