from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from anyfesolver.activity import ElementActivity


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "reference_cases" / "s4_restricted_release_contract.json"
POLICY_PATH = ROOT / "src" / "anyfesolver" / "shell_formulations" / "s4_restricted_policy.py"


def _normalized_git_blob_id(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in data:
        raise AssertionError(f"lone CR is outside the registered text domain: {path}")
    payload = f"blob {len(data)}\0".encode("ascii") + data
    return hashlib.sha1(payload).hexdigest()


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_activity_owned_blobs_are_unchanged() -> None:
    activity = _contract()["activity_contract"]
    assert activity["canonical_owner"] == "ElementActivity"
    assert activity["scaling"] == "element_local_pre_scatter"
    assert activity["changed_by_policy"] is False
    for relative, expected in activity["unchanged_blob_ids"].items():
        assert _normalized_git_blob_id(ROOT / relative) == expected


def test_canonical_activity_scaling_and_hard_deletion_remain_authoritative() -> None:
    activity = ElementActivity([10, 20], activity=[0.5, 1.0])
    stiffness = np.array([[2.0, 4.0], [3.0, 6.0]])
    np.testing.assert_allclose(
        activity.scale_stiffness(stiffness),
        [[1.0, 2.0], [3.0, 6.0]],
    )
    np.testing.assert_allclose(
        activity.scale_contributions(
            np.array([2.0, 3.0, 4.0]),
            [10, 20, 10],
            "stiffness",
        ),
        [1.0, 3.0, 2.0],
    )

    activity.hard_delete([20])
    assert activity.hard_deleted_mask.tolist() == [False, True]
    np.testing.assert_allclose(
        activity.scale_stiffness(stiffness),
        [[1.0, 2.0], [0.0, 0.0]],
    )


def test_policy_defines_no_duplicate_activity_or_assembly_map() -> None:
    tree = ast.parse(POLICY_PATH.read_text(encoding="utf-8"))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    assert all("activity" not in name.lower() for name in imported_names)
    assert all("assembly" not in name.lower() for name in imported_names)
    assert "ElementActivity" not in POLICY_PATH.read_text(encoding="utf-8")
