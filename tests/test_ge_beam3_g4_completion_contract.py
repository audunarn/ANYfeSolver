"""Static guards for the frozen G4 completion design."""
import ast
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/GE_BEAM3_G4_COMPLETION_CONTRACT.md"
CONTRACT = ROOT / "docs/reference_cases/ge_beam3_g4_completion_contract_v1.json"


def strict(path):
    raw = path.read_bytes()
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode("ascii")
    return value


def test_contract_is_canonical_and_scoped():
    value = strict(CONTRACT)
    assert value["schema"] == "GE_BEAM3_G4_COMPLETION_CONTRACT_V1"
    assert value["base"] == {"commit": "7999f6b7c3f3a8efd86839d6db9da335ea613458",
        "tree": "462e2e457d4938756570ed9118c73e5a9b714f8c"}
    assert value["gate_scope"]["closes"] == ["S19", "S20", "S21", "S22", "S23", "S24", "G4"]
    assert value["gate_scope"]["leaves_open"] == ["G5", "PRODUCTION_QUALIFICATION", "PUBLIC_ROUTING"]
    assert value["execution"] == {"automatic_retry": False, "child_seconds": 600,
        "inactivity_seconds": 120, "max_workers": 1, "memory_gib": 24,
        "numerical_threads": 1, "required_cycles": 2, "wave_seconds": 1800}
    assert len(value["restart_mutations"]) == 19


def test_bound_inputs_are_exact():
    value = strict(CONTRACT)
    for name, expected in value["bindings"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == expected["bytes"]
        assert sha256(raw).hexdigest() == expected["sha256"]


def test_plan_forbids_scope_expansion():
    text = PLAN.read_text(encoding="utf-8")
    assert "introduces no new beam or shell mechanics" in text
    assert "stores no executable class name" in text
    assert "G5, public routing" in text
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and name.startswith("anyfesolver") for name in modules)
