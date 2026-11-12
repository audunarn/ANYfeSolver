from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/reference_cases/e4_pl_s3_v5k_protocol_review.py"


def _module():
    spec = importlib.util.spec_from_file_location("_v5k_protocol_review_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_review_accepts_frozen_rule() -> None:
    result = _module().review()
    assert result["verdict"] == "ACCEPT_S3_V5K_PROTOCOL_NO_P0_P1"
    assert result["findings"] == {"P0": [], "P1": []}


def test_protocol_reviewer_is_standard_library_only() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)}
    roots |= {str(node.module).split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not roots & {"anyfesolver", "numpy", "scipy", "sympy"}
