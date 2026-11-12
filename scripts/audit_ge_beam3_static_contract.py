"""Read-only, standard-library audit of the static-integration design freeze.

--render emits the proposed binding to stdout; it never creates authority files.
--check checks the canonical binding, source identities and planning-only extent.
Neither mode imports ANYfeSolver or executes mechanics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = "8808886f2394450a606a2e5972ac8f4e64b17a1f"
BASE_TREE = "321c5155befdcac010a39a43e35528e4aa429cc8"
RUNTIME = "74703a3202251edc0beafb21cd31f52c9304ceb8"
RUNTIME_TREE = "64a18a8fe137946ec2f0d4841821a19066a2707f"
DOCUMENT = "docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md"
MANIFEST = "docs/reference_cases/ge_beam3_general_static_integration_contract_v1.json"
AUDITOR = "scripts/audit_ge_beam3_static_contract.py"
TEST = "tests/test_ge_beam3_static_contract.py"
EXTENT = (DOCUMENT, MANIFEST, AUDITOR, TEST)
SOURCES = (
    "docs/GE_BEAM3_LEGACY_PARITY_MATRIX.md",
    "docs/reference_cases/ge_beam3_legacy_parity_matrix_v1.json",
    "docs/reference_cases/ge_beam3_legacy_parity_inventory_v1.json",
    "docs/GE_BEAM3_NATIVE_WORKFLOWS.md",
    "docs/GE_BEAM3_DELIVERY_STATUS.md",
    "src/anyfesolver/elements.py",
    "src/anyfesolver/assembly.py",
    "src/anyfesolver/constraint_audit.py",
    "src/anyfesolver/beam_sections.py",
    "src/anyfesolver/_native_rotation_state.py",
    "src/anyfesolver/_native_material_protocol.py",
    "src/anyfesolver/_ge_beam3_native_definition.py",
    "src/anyfesolver/_ge_beam3_native_analysis.py",
    "src/anyfesolver/_ge_beam3_native_generalized_element.py",
    "src/anyfesolver/_ge_beam3_native_fibre_static_element.py",
    "src/anyfesolver/_ge_beam3_native_generalized_restart.py",
    "src/anyfesolver/_ge_beam3_native_fibre_restart.py",
    "scripts/release_043_runtime.json",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON value: " + value)


def strict_load(data):
    value = json.loads(data.decode("ascii"), object_pairs_hook=_pairs,
                       parse_constant=_nonfinite)
    if canonical(value) != data:
        raise ValueError("noncanonical JSON or overflowing numeric value")
    return value


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def text_bytes(path):
    # Git working-tree CRLF conversion must not change the frozen text identity.
    return (ROOT / path).read_bytes().replace(b"\r\n", b"\n")


def git(*args):
    return subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=ROOT, check=True,
        capture_output=True, timeout=30,
    ).stdout


def build_expected():
    if git("rev-parse", BASE + "^{tree}").decode().strip() != BASE_TREE:
        raise ValueError("parity base tree mismatch")
    if git("rev-parse", RUNTIME + "^{tree}").decode().strip() != RUNTIME_TREE:
        raise ValueError("runtime base tree mismatch")
    git("merge-base", "--is-ancestor", RUNTIME, BASE)
    sources = []
    for path in SOURCES:
        raw = git("show", BASE + ":" + path)
        if text_bytes(path) != raw.replace(b"\r\n", b"\n"):
            raise ValueError("frozen input changed: " + path)
        sources.append({"path": path, "commit": BASE,
                        "git_blob": git("rev-parse", BASE + ":" + path).decode().strip(),
                        **digest(raw)})
    gate_ranges = {"G1": range(1, 9), "G2": range(9, 15),
                   "G3": range(15, 19), "G4": range(19, 25), "G5": range(25, 27)}
    return {
        "schema": "anysolver.ge-beam3.general-static-contract.v1",
        "contract_id": "GE_BEAM3_GENERAL_STATIC_INTEGRATION_V1",
        "status": "FROZEN_DESIGN_NOT_IMPLEMENTED_OR_QUALIFIED",
        "parent_commit": BASE, "parent_tree": BASE_TREE,
        "runtime_commit": RUNTIME, "runtime_tree": RUNTIME_TREE,
        "runtime_release": "0.4.3",
        "qualification_claim": False, "implementation_started": False,
        "default_changes": False, "mechanics_changes": False,
        "source_bindings": sources,
        "text_bindings": [{"path": p, "encoding": "UTF8_LF",
                           **digest(text_bytes(p))} for p in (DOCUMENT, AUDITOR, TEST)],
        "allowed_extent": list(EXTENT),
        "primary_parity_rows": ["P01", "P04", "P07", "P08", "P09", "P10", "P17", "P18", "P20"],
        "related_parity_rows": ["P05", "P06", "P11", "P15", "P16", "P19", "P28", "P29"],
        "ownership": {
            "nodal_pose": "ONE_ANALYSIS_OWNER_SHARED_SO3",
            "material_origin": "IMMUTABLE_COMMITTED_STATION_STATE",
            "acceptance": "PREPARE_ALL_THEN_ATOMIC_SNAPSHOT_PUBLICATION",
            "trial_base": "COMMITTED_NOT_PREVIOUS_REJECTED_TRIAL",
            "reference_triads": "ELEMENT_OWNED_PHYSICAL_MATERIAL_AXES",
        },
        "constraint_kinds": ["AFFINE_TRANSLATION", "ORIENTATION_SUPPORT", "RELATIVE_POSE"],
        "rotational_mpc_policy": "EXPLICIT_MANIFOLD_SEMANTICS_ANALYTIC_FIRST_AND_SECOND_DERIVATIVES",
        "orientation_chart_limit": "STRICTLY_LESS_THAN_0.9_PI_RELATIVE_ANGLE",
        "section_policy": "EXACT_ELASTIC_FIRST_EXISTING_NATIVE_HISTORY_LAWS_LATER",
        "internal_variables": {"external": 18, "internal": 24,
                               "resultants": 18, "physical_cell_rotations": 6,
                               "static_mass_substitution": False},
        "restart_policy": "EXTERNAL_DIGEST_AND_AUTHENTICATED_ACCEPTED_HISTORY_REPLAY",
        "restart_required_groups": [
            "schema_and_runtime_provenance", "graph_and_ordered_ids",
            "definitions_and_formulation_ids", "physical_reference_triads",
            "section_law_schema_parameters", "constraint_definitions_and_order",
            "load_and_control_history", "committed_translations_chart_and_shared_Q",
            "element_internal_state_and_station_origins", "epoch_and_hash_chain",
            "solver_chart_quadrature_policies", "acceptance_provenance",
        ],
        "gates": {g: [f"S{i:02d}" for i in ids] for g, ids in gate_ranges.items()},
        "case_status": "PREREGISTERED_OBLIGATIONS_NOT_EXECUTED",
        "execution": {"child_seconds": 600, "wave_seconds": 1800,
                      "process_tree_memory_gib": 24, "max_workers": 3,
                      "numerical_threads": 1, "inactivity_seconds": 120,
                      "automatic_retry": False, "formal_cycles": 2},
        "acceptance": {"normalized_invariant_error": "1e-11",
                       "directional_tangent_error": "1e-7",
                       "normalization": "REGISTERED_DIMENSIONLESS_SCALES_MAX_1_NORM_A_NORM_B",
                       "scientific_aggregate_agreement": "BYTE_IDENTICAL",
                       "existing_native_guards": "PRESERVE_WITHOUT_RELAXATION"},
        "terminal_precedence": [
            "BLOCKED_GE_BEAM3_STATIC_AUTHORITY",
            "BLOCKED_GE_BEAM3_STATIC_PROCESS_OR_EVIDENCE",
            "NO_GO_GE_BEAM3_STATIC_STATE_OR_RESTART",
            "NO_GO_GE_BEAM3_STATIC_CONSTRAINT_OR_WORK",
            "NO_GO_GE_BEAM3_STATIC_SECTION_OR_INTERNAL_ALGEBRA",
            "NO_GO_GE_BEAM3_STATIC_ASSEMBLY_OR_REGRESSION",
            "PROVISIONAL_GO_GE_BEAM3_SCOPED_STATIC_INTEGRATION",
        ],
        "next_gate": "G1_SHARED_TRANSACTION_TWO_ELEMENT_EXACT_ELASTIC_STATIC_ADAPTER",
    }


def validate(data, expected):
    # Canonical byte comparison also distinguishes bool/int and rejects every
    # missing/extra key, reordered gate, altered enum and changed binding.
    value = strict_load(data)
    if canonical(value) != canonical(expected):
        raise ValueError("contract schema, policy or binding mismatch")
    return value


def check_extent():
    # Fail closed on a missing base, including shallow local checkouts. This is
    # an explicit repository audit, not part of ordinary pure unit collection.
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    changed = set(git("diff", "--name-only", BASE, "--").decode().splitlines())
    untracked = set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    if (changed | untracked) - set(EXTENT):
        raise ValueError("changes outside planning-only extent")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_expected()
    check_extent()
    if args.render:
        sys.stdout.buffer.write(canonical(expected))
    else:
        validate(text_bytes(MANIFEST), expected)
        print("PASS: static design freeze; 26 planned cases, no mechanics execution")


if __name__ == "__main__":
    main()
