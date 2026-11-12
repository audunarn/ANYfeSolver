from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_finite_producer.py"
CHECKER = ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_finite_checker.py"
REFERENCE_DIRECTORY = ROOT / "docs" / "reference_cases"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _git_blob_sha256(relative_path: str) -> str:
    raw = subprocess.check_output(
        ("git", "ls-files", "--stage", "-z", "--", relative_path), cwd=ROOT
    )
    entries = [entry for entry in raw.split(b"\0") if entry]
    assert len(entries) == 1
    metadata, registered_path = entries[0].split(b"\t", 1)
    mode, oid, stage = metadata.split()
    assert mode in {b"100644", b"100755"}
    assert stage == b"0"
    assert registered_path.decode("utf-8").replace("\\", "/") == relative_path
    blob = subprocess.check_output(("git", "cat-file", "blob", oid), cwd=ROOT)
    assert b"\r" not in blob
    return hashlib.sha256(blob).hexdigest().upper()


def _rehash_record(record: dict) -> None:
    unhashed = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = hashlib.sha256(_canonical_bytes(unhashed)).hexdigest().upper()


def _run_producer(output: Path, *, mutation: str | None = None) -> None:
    command = [sys.executable, str(PRODUCER), "--output", str(output)]
    if mutation is not None:
        command.extend(("--mutate", mutation))
    subprocess.run(command, cwd=ROOT, check=True, timeout=30, env=os.environ.copy())


def _run_checker(proof: Path, output: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "--proof", str(proof), "--output", str(output)],
        cwd=ROOT,
        check=False,
        timeout=30,
        env=os.environ.copy(),
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = 0 if result["terminal"] == "NONCLASSIFYING_FINITE_GATE_PASS" else 2
    assert completed.returncode == expected
    return result


def test_finite_proof_and_independent_check_are_deterministic(tmp_path: Path) -> None:
    proof_a = tmp_path / "proof-a.json"
    proof_b = tmp_path / "proof-b.json"
    check_a = tmp_path / "check-a.json"
    check_b = tmp_path / "check-b.json"
    _run_producer(proof_a)
    _run_producer(proof_b)
    assert proof_a.read_bytes() == proof_b.read_bytes()
    result_a = _run_checker(proof_a, check_a)
    result_b = _run_checker(proof_b, check_b)
    assert check_a.read_bytes() == check_b.read_bytes()
    assert result_a == result_b
    assert result_a["terminal"] == "NONCLASSIFYING_FINITE_GATE_PASS"
    assert all(result_a["authority_bindings"].values())
    assert all(result_a["coverage"].values())
    assert all(all(case["predicates"].values()) for case in result_a["cases"])
    assert all(result_a["reversal"].values())
    for case in result_a["cases"]:
        assert len(case["registered_input_sha256"]) == 64
        assert case["registered_input_sha256"] == case["registered_input_sha256"].upper()
        assert case["predicates"]["tangent_full"] is True
        assert case["predicates"]["tangent_full_max"] is True
        assert case["predicates"]["source_tangent_symmetry"] is True
        assert float(case["metrics"]["tangent_full_relative"]) <= 1.0e-7
        assert float(case["metrics"]["tangent_full_max_relative"]) <= 1.0e-7


def test_finite_proof_binds_frozen_inputs_and_nonclassifying_coverage(tmp_path: Path) -> None:
    proof_path = tmp_path / "proof.json"
    _run_producer(proof_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    bindings = proof["bindings"]
    assert bindings["base"]["commit"] == "09351645ba17a0a5b130a1c7a48007d36dd08ada"
    assert bindings["base"]["tree"] == "cfb0cf9a19f6519abf335694253efa264cd2e695"
    base = {key: bindings["base"][key] for key in ("commit", "tree")}
    assert bindings["base"]["identity_sha256"] == hashlib.sha256(_canonical_bytes(base)).hexdigest().upper()
    environment = {
        key: value for key, value in bindings["environment"].items() if key != "identity_sha256"
    }
    assert bindings["environment"]["identity_sha256"] == hashlib.sha256(
        _canonical_bytes(environment)
    ).hexdigest().upper()
    for name, digest in bindings["authority_inputs"].items():
        assert digest == _git_blob_sha256(f"docs/reference_cases/{name}")
    for name, digest in bindings["programs"].items():
        assert digest == _git_blob_sha256(name)
    repository_paths = {
        *(f"docs/reference_cases/{name}" for name in bindings["authority_inputs"]),
        *bindings["programs"],
    }
    assert set(bindings["repository_git_blob_oids"]) == repository_paths
    assert set(bindings["repository_git_blobs_are_canonical_lf_text"]) == repository_paths
    assert set(bindings["repository_working_tree_matches_git_blobs"]) == repository_paths
    assert all(bindings["repository_git_blobs_are_canonical_lf_text"].values())
    assert all(bindings["repository_working_tree_matches_git_blobs"].values())
    assert bindings["source_artifacts"]["HUMER_STEINBRECHER_PECHSTEIN_2026"]["sha256"] == (
        "76AA9EDDDAE2EE16B47BF4E8255BDAA81678164B899E663E0B882B11C39BDB1E"
    )

    coverage = proof["coverage_diagnostics"]
    assert coverage["classification_authority"] is True
    assert coverage["geometry_scope"] == (
        "GLOBALLY_STRAIGHT_COLLINEAR_TWO_EQUAL_CELL_ZERO_REFERENCE_JUMP_ONLY"
    )
    assert [record["case_id"] for record in coverage["isolated_modes"]] == [
        "ISOLATED_AXIAL",
        "ISOLATED_SHEAR_Y",
        "ISOLATED_SHEAR_Z",
        "ISOLATED_TORSION",
        "ISOLATED_BENDING_Y",
        "ISOLATED_BENDING_Z",
    ]
    assert [record["case_id"] for record in coverage["physical_reference_orientations"]] == [
        "ORIENTATION_ROLLED_X",
        "ORIENTATION_SKEW_A",
        "ORIENTATION_SKEW_B",
    ]
    slenderness = coverage["slenderness"]
    assert [float.fromhex(record["l_over_h"]) for record in slenderness] == [
        10.0,
        100.0,
        10_000.0,
        1_000_000.0,
    ]
    assert all(record["anchored_complement_rank"] == 12 for record in slenderness)
    assert all(
        record["anchored_complement"] == "NODE_1_ALL_SIX_COORDINATES_ZERO"
        for record in slenderness
    )
    spectrum = coverage["multi_element_spectrum"]
    assert spectrum["dof_count"] == 30
    assert spectrum["element_count"] == 2
    assert spectrum["normalized_rank"] == 24
    unhashed = {key: value for key, value in coverage.items() if key != "content_sha256"}
    assert coverage["content_sha256"] == hashlib.sha256(_canonical_bytes(unhashed)).hexdigest().upper()


def test_authority_scope_forbids_reference_kinks_and_curvature() -> None:
    map_b = json.loads((REFERENCE_DIRECTORY / "ge_beam3_mixed_equation_map_b.json").read_text(encoding="utf-8"))
    contract = json.loads((REFERENCE_DIRECTORY / "ge_beam3_mixed_local_contract.json").read_text(encoding="utf-8"))
    limitations = map_b["limitations"]
    assert limitations["geometry_scope"].startswith("ONE GLOBALLY STRAIGHT COLLINEAR REFERENCE LINE")
    assert "MIDDLE KINK" in limitations["non_smooth_curved_limitation"]
    jump = map_b["macro_functional"]["reference_relative_jump"]
    assert jump["reference"].startswith("Eta0_c,v=0")
    assert "NONZERO REFERENCE JUMPS ARE OUTSIDE THIS CANDIDATE" in jump["reference"]
    assert "CLAIM_PIECEWISE_STRAIGHT_REFERENCE_KINK_QUALIFICATION" in contract["fatal_local_gate"]["forbidden"]
    assert "PIECEWISE_STRAIGHT_REFERENCE_WITH_MIDDLE_KINK" in contract["scope"]["deferred"]
    packet = contract["finite_packet_contract"]
    assert packet["classified_coverage"]["classification_authority"] is True
    assert packet["classified_coverage"]["slenderness_l_over_h_order"][-1] == 1_000_000
    assert "LOCAL_CONTRACT_SHA256" in packet["binding_groups"]


def test_finite_proof_mutations_are_rejected(tmp_path: Path) -> None:
    for mutation in ("energy", "local_moment", "residual", "section", "tangent"):
        proof = tmp_path / f"{mutation}-proof.json"
        check = tmp_path / f"{mutation}-check.json"
        _run_producer(proof, mutation=mutation)
        result = _run_checker(proof, check)
        assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"
        assert any(not case["predicates"]["content_hash"] for case in result["cases"])


def test_correctly_rehashed_registered_input_substitutions_are_rejected(tmp_path: Path) -> None:
    pristine = tmp_path / "pristine-proof.json"
    _run_producer(pristine)
    mutations = {
        "cell_length": ("registered_cell_length", lambda row: row.__setitem__("cell_length", float(0.51).hex())),
        "positions": (
            "registered_positions",
            lambda row: row["positions"][1].__setitem__(1, float(float.fromhex(row["positions"][1][1]) + 0.005).hex()),
        ),
        "section": (
            "registered_section",
            lambda row: row["section"][0].__setitem__(0, float(float.fromhex(row["section"][0][0]) + 0.125).hex()),
        ),
        "vertex_rotations": (
            "registered_vertex_rotations",
            lambda row: row["vertex_rotations"][0][0].__setitem__(0, float(float.fromhex(row["vertex_rotations"][0][0][0]) + 0.005).hex()),
        ),
    }
    for name, (predicate, mutate) in mutations.items():
        proof = json.loads(pristine.read_text(encoding="utf-8"))
        target = proof["records"][2]
        mutate(target)
        _rehash_record(target)
        proof_path = tmp_path / f"rehashed-{name}-proof.json"
        proof_path.write_bytes(_canonical_bytes(proof))
        result = _run_checker(proof_path, tmp_path / f"rehashed-{name}-check.json")
        checked = next(case for case in result["cases"] if case["case_id"] == "AXIAL_SHEAR")
        assert checked["predicates"]["content_hash"] is True
        assert checked["predicates"][predicate] is False
        assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"


def test_correctly_rehashed_full_tangent_perturbation_is_rejected(tmp_path: Path) -> None:
    pristine = tmp_path / "pristine-proof.json"
    _run_producer(pristine)
    proof = json.loads(pristine.read_text(encoding="utf-8"))
    target = proof["records"][3]
    target["tangent"][4][4] = float(float.fromhex(target["tangent"][4][4]) + 1.0e-3).hex()
    _rehash_record(target)
    changed = tmp_path / "rehashed-tangent-proof.json"
    changed.write_bytes(_canonical_bytes(proof))
    result = _run_checker(changed, tmp_path / "rehashed-tangent-check.json")
    checked = next(case for case in result["cases"] if case["case_id"] == "BEND_TWIST")
    assert checked["predicates"]["content_hash"] is True
    assert checked["predicates"]["tangent_symmetry"] is True
    assert checked["predicates"]["tangent_full"] is False
    assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"


def test_forged_bindings_and_rehashed_coverage_are_rejected(tmp_path: Path) -> None:
    pristine = tmp_path / "pristine-proof.json"
    _run_producer(pristine)

    binding_proof = json.loads(pristine.read_text(encoding="utf-8"))
    binding_proof["bindings"]["base"]["commit"] = "0" * 40
    binding_path = tmp_path / "forged-binding-proof.json"
    binding_path.write_bytes(_canonical_bytes(binding_proof))
    binding_result = _run_checker(
        binding_path, tmp_path / "forged-binding-check.json"
    )
    assert binding_result["authority_bindings"]["base"] is False
    assert binding_result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"

    checkout_proof = json.loads(pristine.read_text(encoding="utf-8"))
    checkout_map = checkout_proof["bindings"][
        "repository_working_tree_matches_git_blobs"
    ]
    checkout_map[next(iter(checkout_map))] = False
    checkout_path = tmp_path / "forged-checkout-binding-proof.json"
    checkout_path.write_bytes(_canonical_bytes(checkout_proof))
    checkout_result = _run_checker(
        checkout_path, tmp_path / "forged-checkout-binding-check.json"
    )
    assert (
        checkout_result["authority_bindings"][
            "repository_working_tree_matches_git_blobs"
        ]
        is False
    )
    assert checkout_result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"

    coverage_proof = json.loads(pristine.read_text(encoding="utf-8"))
    coverage = coverage_proof["coverage_diagnostics"]
    coverage["slenderness"][-1]["anchored_complement_rank"] = 11
    _rehash_record(coverage["slenderness"][-1])
    _rehash_record(coverage)
    coverage_path = tmp_path / "forged-coverage-proof.json"
    coverage_path.write_bytes(_canonical_bytes(coverage_proof))
    coverage_result = _run_checker(
        coverage_path, tmp_path / "forged-coverage-check.json"
    )
    assert coverage_result["coverage"]["coverage_content_hash"] is True
    assert coverage_result["coverage"]["slenderness_rank_and_reference_response"] is False
    assert coverage_result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"


def test_rehashed_coverage_inputs_and_complete_operators_are_independently_rejected(
    tmp_path: Path,
) -> None:
    pristine_path = tmp_path / "pristine-proof.json"
    _run_producer(pristine_path)
    pristine = json.loads(pristine_path.read_text(encoding="utf-8"))

    def check_changed(proof: dict, name: str) -> dict:
        coverage = proof["coverage_diagnostics"]
        _rehash_record(coverage)
        proof_path = tmp_path / f"{name}-proof.json"
        proof_path.write_bytes(_canonical_bytes(proof))
        return _run_checker(proof_path, tmp_path / f"{name}-check.json")

    forged_input = json.loads(json.dumps(pristine))
    coverage = forged_input["coverage_diagnostics"]
    isolated = coverage["isolated_modes"][0]
    isolated["nodal_input_local"][1][0] = float(
        float.fromhex(isolated["nodal_input_local"][1][0]) + 1.0e-5
    ).hex()
    _rehash_record(isolated)
    result = check_changed(forged_input, "coverage-input")
    assert result["coverage"]["coverage_content_hash"] is True
    assert result["coverage"]["isolated_modes_registered_inputs"] is False
    assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"

    forged_finite_tangent = json.loads(json.dumps(pristine))
    coverage = forged_finite_tangent["coverage_diagnostics"]
    isolated = coverage["isolated_modes"][1]
    isolated["tangent"][7][7] = float(
        float.fromhex(isolated["tangent"][7][7]) + 1.0e-3
    ).hex()
    _rehash_record(isolated)
    result = check_changed(forged_finite_tangent, "coverage-finite-tangent")
    assert result["coverage"]["coverage_content_hash"] is True
    assert result["coverage"]["isolated_modes_registered_inputs"] is True
    assert result["coverage"]["isolated_modes_independent_source"] is False
    assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"

    forged_slender = json.loads(json.dumps(pristine))
    coverage = forged_slender["coverage_diagnostics"]
    slender = coverage["slenderness"][1]
    slender["stiffness"][7][7] = float(
        float.fromhex(slender["stiffness"][7][7]) * 1.01
    ).hex()
    slender["stiffness_sha256"] = hashlib.sha256(
        _canonical_bytes(slender["stiffness"])
    ).hexdigest().upper()
    _rehash_record(slender)
    result = check_changed(forged_slender, "coverage-slender-tangent")
    assert result["coverage"]["coverage_content_hash"] is True
    assert result["coverage"]["slenderness_registered_inputs"] is True
    assert result["coverage"]["slenderness_independent_operator"] is False
    assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"

    forged_assembly = json.loads(json.dumps(pristine))
    coverage = forged_assembly["coverage_diagnostics"]
    spectrum = coverage["multi_element_spectrum"]
    spectrum["assembled_stiffness"][6][6] = float(
        float.fromhex(spectrum["assembled_stiffness"][6][6]) + 0.25
    ).hex()
    spectrum["stiffness_sha256"] = hashlib.sha256(
        _canonical_bytes(spectrum["assembled_stiffness"])
    ).hexdigest().upper()
    _rehash_record(spectrum)
    result = check_changed(forged_assembly, "coverage-assembled-tangent")
    assert result["coverage"]["coverage_content_hash"] is True
    assert result["coverage"]["multi_element_registered_inputs"] is True
    assert result["coverage"]["multi_element_independent_operator"] is False
    assert result["terminal"] == "NONCLASSIFYING_FINITE_GATE_FINDING"


def test_checker_import_boundary_is_independent() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"anyfesolver", "sympy", "mpmath"})
    text = CHECKER.read_text(encoding="utf-8")
    assert "from anyfesolver" not in text
    assert "import anyfesolver" not in text
