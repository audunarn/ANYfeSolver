from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = "2ac678a7f94c250fe433f66378a83508d86ee499"
BASE_TREE = "f7382e2b88343ac29c9a9e3c424f618a3652cc01"
TIER_IDENTITIES = {
    "e0": {
        "commit": "87b639499187736c59d87bc4aa8e6bd7f819d28b",
        "tree": "c01fd5cab7b63325e6cb5b70000f4586d4788563",
        "count": 94,
        "node_ids_canonical_lf_sha256": (
            "29EF584E9B51E8420934A519B3C1E71BDD3082EFDC89DBADA4FCE0FFE8997B9F"
        ),
    },
    "e1": {
        "commit": "281ed90e148c125edbec27e7336a8f9f0df08edc",
        "tree": "1ee60da4717055f5cc1b37ff9369877bb1867861",
        "count": 16,
        "node_ids_canonical_lf_sha256": (
            "9835FB4580C886B52BFF5961A30CD78E921B5CEED92A918312149032748A7F63"
        ),
    },
    "e2": {
        "commit": BASE,
        "tree": BASE_TREE,
        "count": 8,
    },
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _json(relative: str) -> dict[str, object]:
    raw = (ROOT / relative).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert raw == canonical
    return value


def _record(record: dict[str, object]) -> dict[str, object]:
    raw = (ROOT / str(record["path"])).read_bytes()
    assert len(raw) == record["bytes"] and _sha(raw) == record["sha256"]
    return json.loads(raw)


def test_e3_authority_and_three_closed_world_tiers_are_exact() -> None:
    baseline = _json("docs/reference_cases/e3_baseline.json")
    inventory = _json("docs/reference_cases/e3_test_inventory.json")
    assert baseline["authority"] == {
        "branch": "codex/s4-e3-hw29-mitc9i-route-selection",
        "commit": BASE,
        "tree": BASE_TREE,
    }
    assert baseline["attachment"]["sha256"] == (
        "7D86FE7A6D205BFEDDA4C884A2AFAD5C80EF0F3DE6BA350C48BBB2150BFC5108"
    )
    assert inventory["total_reference_count"] == 118
    assert inventory["total_reference_is_not_one_live_suite"] is True
    for name, expected in TIER_IDENTITIES.items():
        assert {key: inventory["tiers"][name][key] for key in expected} == expected
    assert inventory["tiers"]["e1"]["historical_report_statement_preserved"] == 15
    for record in inventory["tiers"]["e2"]["files"]:
        raw = (ROOT / record["path"]).read_bytes()
        assert len(raw) == record["bytes"] and _sha(raw) == record["sha256"]

    closeout = baseline["e2_closeout"]
    output = _record(closeout["output"])
    status = _record(closeout["status"])
    _record(closeout["contract"])
    review_raw = (ROOT / closeout["review"]["path"]).read_bytes()
    assert len(review_raw) == closeout["review"]["bytes"]
    assert _sha(review_raw) == closeout["review"]["sha256"]
    assert output["candidate_terminal"] == closeout["terminal"]
    assert status["independent_review"]["verdict"] == "ACCEPT_NO_P0_OR_P1"

    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=False
    )
    assert ancestry.returncode == 0
    tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", BASE],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=False,
    )
    assert tree.returncode == 0 and tree.stdout.decode().strip() == BASE_TREE


def test_e3_material_scope_and_production_boundary_are_frozen() -> None:
    materials = _json("docs/reference_cases/e3_material_fixtures.json")
    environment = _json("docs/reference_cases/e3_environment.json")
    inherited_environment = _record(environment["inherited_baseline_runtime"])
    inherited = _record(materials["inherited_fixture"])
    assert inherited["rp_c208"]["row_count"] == 17
    assert inherited["rp_c208"]["grades"] == ["S235", "S275", "S355", "S420", "S460"]
    assert materials["hw29_scope"] == {
        "density_is_not_used_by_linear_static_identity": True,
        "isotropic_G_formula": "E/[2*(1+nu)]",
        "new_drill_or_material_input": False,
        "section_scope": "homogeneous_isotropic_only",
    }
    assert environment["baseline_execution"]["one_live_118_suite"] is False
    assert environment["oracle"]["dependencies"] == "python_standard_library_only"
    expected_paths = [
        "C:/Github/ANYfeSolver/tmp/s4_candidate_a_mpmath_valid",
        "C:/Github/ANYfeSolver/.s4_candidate_a_pinned/fileio/src",
        "C:/Github/ANYfeSolver/.s4_candidate_a_pinned/material/src",
        "C:/Github/ANYfeSolver/.s4_candidate_a_pinned/mesh/src",
        "C:/Github/ANYfeSolver/.s4_candidate_a_pinned/geometry/src",
    ]
    assert inherited_environment["baseline_test_runtime"]["pythonpath_order_absolute"] == expected_paths
    assert environment["reproduced_e0_runtime"]["pythonpath_order_absolute"] == expected_paths
    assert environment["process"] == {
        "OMP_NUM_THREADS": "1",
        "OMP_STACKSIZE": "1G",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }

    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", "src", ".github", "pyproject.toml"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=False,
    )
    assert diff.returncode == 0 and diff.stdout == b""
