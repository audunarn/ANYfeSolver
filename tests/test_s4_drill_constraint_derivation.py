"""Focused gates for the independent S4 drill-constraint derivation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "docs" / "reference_cases" / "s4_drill_constraint_oracle.py"
CASES = ROOT / "docs" / "reference_cases" / "s4_drill_constraint_cases.json"
OUTPUT = ROOT / "docs" / "reference_cases" / "s4_drill_constraint_oracle_output.json"
EXPECTED_CASES_SHA256 = "b4d663382302e971752f0757f6e869549a54234f485235e06dbef74085860f38"
EXPECTED_OUTPUT_SHA256 = "8005c6d285263e33ff7f6d4b5138d5fbe4efab6a95834c401f94af044acd9e1b"
EXPECTED_PLAN_SHA256 = "90b5c4903ee6a9c06056f7e1f3ab21dae0626c185a27627843a04bf289430e3a"
EXPECTED_TOR_SHA256 = "8e969863806461124510e7c31d99a3244fccf15dd67424517320ee819439aa90"
EXPECTED_SHARD_SHA256 = {
    "80": "321d9ae299b4d0be5f1f5fa49f6f3b6daa3e4ca8d1d46a766705eb715cefabe9",
    "160": "ed3951413ea8e655b9dc521536f06f3f0f7f18ce82403b0184384cfcb69459cb",
    "320": "e5c375415facbc181034a0eace2872f3a1fe5208562eb742cd37b2a203173b09",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle_module() -> types.ModuleType:
    os.environ["MPMATH_NOGMPY"] = "1"
    name = "_s4_drill_constraint_oracle_test"
    spec = importlib.util.spec_from_file_location(name, ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_numeric_reference() -> tuple[types.ModuleType, list[str]]:
    source = ROOT / "src" / "anyfesolver" / "shell_formulations"
    package = "_s4_drill_numeric"
    shell_package = f"{package}.shell_formulations"
    created: list[str] = []
    root_module = types.ModuleType(package)
    root_module.__path__ = []  # type: ignore[attr-defined]
    shell_module = types.ModuleType(shell_package)
    shell_module.__path__ = [str(source)]  # type: ignore[attr-defined]
    sys.modules[package] = root_module
    sys.modules[shell_package] = shell_module
    created.extend((package, shell_package))
    try:
        for short_name in ("protocol", "q4_common", "mitc4_plus_d_reference"):
            name = f"{shell_package}.{short_name}"
            path = source / f"{short_name}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            created.append(name)
            spec.loader.exec_module(module)
        reference = sys.modules[f"{shell_package}.mitc4_plus_d_reference"]
        assert Path(reference.__file__).resolve() == (source / "mitc4_plus_d_reference.py").resolve()
        return reference, created
    except BaseException:
        for name in reversed(created):
            sys.modules.pop(name, None)
        raise


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["MPMATH_NOGMPY"] = "1"
    local_dependency = ROOT / ".s4_mpmath"
    if local_dependency.is_dir():
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(local_dependency) + (os.pathsep + previous if previous else "")
    return environment


def test_registered_cases_and_quick_oracle_are_byte_repeatable() -> None:
    assert _sha256(CASES) == EXPECTED_CASES_SHA256
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert cases["governing_plan_sha256"] == EXPECTED_PLAN_SHA256
    assert cases["tor_plan_sha256"] == EXPECTED_TOR_SHA256
    command = [sys.executable, str(ORACLE), "--quick", "--summary"]
    first = subprocess.run(
        command,
        cwd=ROOT,
        env=_subprocess_environment(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    ).stdout
    second = subprocess.run(
        command,
        cwd=ROOT,
        env=_subprocess_environment(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    ).stdout
    assert first == second
    result = json.loads(first)
    assert result["status"] == "complete"
    assert result["scientific_summary"]["outcome"] == "CERTIFIED_FOR_LATER_LINEAR_ADAPTER_PLANNING"
    square = result["topologies"][0]
    assert square["free"] == {
        "rank_B": 16,
        "N": 8,
        "G": 1,
        "P": 7,
        "R": 6,
        "R_N": 6,
        "R_G": 0,
        "RQ": 6,
        "Z": 1,
    }
    assert square["C_D_rank"] == 4
    assert square["drill"]["Z_C"] == 0


def test_precision_shard_cli_has_closed_modes_and_paths() -> None:
    oracle = _oracle_module()
    assert oracle.PRECISION_SHARD_SCHEMA == "s4-drill-constraint-precision-shard-v1"
    assert set(oracle._allowed_shard_outputs()) == {
        "set1_080.json",
        "set1_160.json",
        "set1_320.json",
        "set2_080.json",
        "set2_160.json",
        "set2_320.json",
    }
    environment = _subprocess_environment()
    missing_full = subprocess.run(
        [sys.executable, str(ORACLE), "--precision", "80", "--output", str(ROOT / "escaped.json")],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert missing_full.returncode == 2
    escaped = subprocess.run(
        [
            sys.executable,
            str(ORACLE),
            "--full",
            "--precision",
            "80",
            "--output",
            str(ROOT / "escaped.json"),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert escaped.returncode == 1
    failure = json.loads(escaped.stdout)
    assert failure["status"] == "scientific_or_contract_failure"
    assert "allowlist" in failure["failure"]["message"]
    assert not (ROOT / "escaped.json").exists()


def test_whole_matrix_eq21_eq25_binary64_agreement_and_stored_packet() -> None:
    oracle = _oracle_module()
    reference_module, created = _load_numeric_reference()
    try:
        cases = oracle.load_cases()
        oracle.mp.mp.dps = 80
        epsilon64 = np.finfo(np.float64).eps
        limit = 8192.0 * 24.0 * epsilon64
        for case in cases["local_cases"]:
            coordinates = np.asarray(case["coordinates"], dtype=np.float64)
            seeds = np.asarray(case["director_seeds"], dtype=np.float64)
            directors = seeds / np.linalg.norm(seeds, axis=1)[:, None]
            thickness = np.asarray(case["thickness"], dtype=np.float64)
            actual_reference = reference_module.build_reference_data(
                coordinates, directors, thickness
            )
            expected_reference = oracle.build_reference(
                case["coordinates"],
                case["director_seeds"],
                case["thickness"],
                element_id=case["id"],
                connectivity=("0", "1", "2", "3"),
            )
            for point in cases["sample_points"]:
                r_mp, s_mp = oracle._sample_point(point)
                r_value, s_value = float(r_mp), float(s_mp)
                pairs = (
                    (
                        reference_module._drill_membrane_operator(
                            actual_reference, r_value, s_value
                        ),
                        np.asarray(
                            [
                                [float(oracle.eq21_drill(expected_reference, r_mp, s_mp)[row, column]) for column in range(24)]
                                for row in range(3)
                            ]
                        ),
                    ),
                    (
                        reference_module._assumed_mitc4_plus_membrane_2025_eq25(
                            actual_reference, r_value, s_value
                        ),
                        np.asarray(
                            [
                                [float(oracle.eq25_membrane(expected_reference, r_mp, s_mp)[row, column]) for column in range(24)]
                                for row in range(3)
                            ]
                        ),
                    ),
                )
                for actual, expected in pairs:
                    denominator = max(np.linalg.norm(actual), np.linalg.norm(expected))
                    if denominator == 0.0:
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        assert np.linalg.norm(actual - expected) / denominator <= limit
        assert OUTPUT.is_file()
        assert _sha256(OUTPUT) == EXPECTED_OUTPUT_SHA256
        packet = json.loads(OUTPUT.read_text(encoding="utf-8"))
        assert packet["schema"] == oracle.SCHEMA
        assert packet["status"] == "complete"
        assert packet["mode"] == "full"
        assert packet["execution_shards"] == EXPECTED_SHARD_SHA256
        assert packet["scientific_summary"]["outcome"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
        assert packet["identities"]["cases_sha256"] == EXPECTED_CASES_SHA256
        shard_directory = ROOT / ".s4_drill_constraint_shards"
        if shard_directory.exists():
            assert shard_directory.is_dir()
            repeat = shard_directory / "repeat_merged.json"
            assert repeat.read_bytes() == OUTPUT.read_bytes()
            for precision, digest in EXPECTED_SHARD_SHA256.items():
                for set_number in (1, 2):
                    shard = shard_directory / f"set{set_number}_{int(precision):03d}.json"
                    assert _sha256(shard) == digest
    finally:
        for name in reversed(created):
            sys.modules.pop(name, None)


def test_oracle_source_has_no_anyfesolver_import() -> None:
    source = ORACLE.read_text(encoding="utf-8")
    assert "import anyfesolver" not in source
    assert "from anyfesolver" not in source
