from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "reference_cases"
REFERENCE_PATH = CASES / "e4_pl_s3_v2_independent_reference.py"
CHECKER_PATH = CASES / "e4_pl_s3_v2_independent_checker.py"
DOMAIN_PATH = CASES / "e4_pl_s3_v2_independent_domain.py"
EQUATION_MAP_PATH = CASES / "e4_pl_s3_v2_dkmt_equation_map.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = _load("test_s3_v2_independent_reference", REFERENCE_PATH)
checker = _load("test_s3_v2_independent_checker", CHECKER_PATH)
domain = _load("test_s3_v2_independent_domain", DOMAIN_PATH)


NODES = np.asarray(((0.13, -0.21), (1.71, 0.18), (0.34, 1.27)), dtype=np.float64)


def _section(thickness: float = 0.35) -> np.ndarray:
    return reference.isotropic_generalized_section(210.0, 0.27, thickness, 5.0 / 6.0)


def test_published_dkmt_equations_patch_work_and_binary64_diagnostics() -> None:
    record = checker.check_reference_case(NODES, _section())
    assert record["classification"] == "PASS_E4_PL_S3_V2A_FLAT_DKMT_REFERENCE"
    assert all(record["checks"].values())
    assert max(record["residuals"].values()) < 3.0e-11
    assert record["rank_inertia_disposition"] == "EXACT_CANONICAL_CASE_ONLY_ARBITRARY_BINARY64_DIAGNOSTIC"
    assert record["diagnostic_ranks"] == {"physical": 9, "pl": 3, "condensed": 12, "saddle": 15, "rigid": 6}
    assert record["diagnostic_inertia"] == [12, 3, 6]
    assert record["limits"]["thin_rho_max"] < 1.0e-10
    assert record["limits"]["thick_rho_min"] > 1.0 - 1.0e-5

    assembled = reference.assemble_flat_reference(NODES, _section())
    expected_phi = 2.0 * assembled.section_parameters.thickness**2 / (
        assembled.section_parameters.shear_correction
        * (1.0 - assembled.section_parameters.poisson)
        * assembled.edge_lengths**2
    )
    np.testing.assert_allclose(assembled.phi, expected_phi, rtol=3e-15, atol=0.0)
    np.testing.assert_allclose(assembled.a_delta @ assembled.delta_beta_operator, assembled.au_operator, rtol=3e-15, atol=3e-15)

    probe = np.arange(-9, 9, dtype=np.float64) / 16.0
    direct = reference.direct_internal_force(assembled, probe)
    np.testing.assert_allclose(direct, assembled.physical_stiffness @ probe, rtol=8e-13, atol=8e-13)
    for station in range(3):
        fields = reference.generalized_fields(assembled, probe, station)
        np.testing.assert_allclose(fields["resultants"], np.concatenate((fields["N"], fields["M"], fields["Q"])))


def test_equation_27_shapes_and_equations_12_to_16_projection() -> None:
    assembled = reference.assemble_flat_reference(((0.0, 0.0), (3.0, 0.0), (0.0, 4.0)), _section(0.5))
    barycentric = (0.2, 0.3, 0.5)
    values, gradients = reference.quadratic_edge_shape_gradients(barycentric, assembled.shape_gradients)
    np.testing.assert_allclose(values, (4 * 0.2 * 0.3, 4 * 0.3 * 0.5, 4 * 0.2 * 0.5), rtol=0.0, atol=2e-16)
    assert gradients.shape == (3, 2)

    projection = reference.edge_shear_projection(barycentric, assembled.edge_directions)
    constant_shear = np.asarray((0.375, -0.625))
    edge_values = assembled.edge_directions @ constant_shear
    np.testing.assert_allclose(projection @ edge_values, constant_shear, rtol=3e-15, atol=3e-15)

    # Eq. (22): an affine w plus constant beta gives the exact edge average.
    vector = np.zeros(18)
    gradient_w = np.asarray((0.2, -0.4))
    beta = np.asarray((-0.1, 0.3))
    for node, (x, y) in enumerate(assembled.nodes):
        base = 6 * node
        vector[base + 2] = gradient_w @ np.asarray((x, y))
        vector[base + 3] = -beta[1]
        vector[base + 4] = beta[0]
    np.testing.assert_allclose(assembled.au_operator @ vector, assembled.edge_directions @ (gradient_w + beta), rtol=3e-15, atol=3e-15)


def test_exact_fraction_case_proves_rank_inertia_patch_and_d3() -> None:
    certificate = checker.exact_canonical_certificate()
    assert certificate["classification"] == "PASS_E4_PL_S3_V2A_DKMT_EXACT_CANONICAL"
    assert certificate["arithmetic"] == "FRACTION_GAUSSIAN_AND_SYMMETRIC_LDL"
    assert all(certificate["checks"].values())
    assert certificate["ranks"] == {"physical": 9, "pl": 3, "condensed": 12, "saddle": 15, "rigid": 6}
    assert certificate["inertia"] == {
        "physical": [9, 0, 9],
        "pl": [3, 0, 15],
        "condensed": [12, 0, 6],
        "saddle": [12, 3, 6],
    }
    assert len(certificate["hashes"]) == 4
    assert all(len(value) == 64 for value in certificate["hashes"].values())


def test_all_six_d3_binary64_transports() -> None:
    assembled = reference.assemble_flat_reference(NODES, _section())
    for permutation in itertools.permutations(range(3)):
        transform = reference.block_permutation(permutation)
        permuted = reference.assemble_flat_reference(NODES[np.asarray(permutation)], _section())
        np.testing.assert_allclose(permuted.condensed_stiffness, transform @ assembled.condensed_stiffness @ transform.T, rtol=2e-12, atol=2e-12)


def test_proof_is_canonical_repeatable_and_mutation_detecting() -> None:
    # The producer emits claims; the checker reconstructs them without loading
    # or invoking the producer module.
    proof = reference.make_proof(NODES, _section())
    left = reference.canonical_bytes(proof)
    right = reference.canonical_bytes(reference.make_proof(NODES, _section()))
    assert left == right
    assert checker.load_proof(left)["schema"] == checker.PROOF_SCHEMA
    assert checker.canonical_report_bytes(checker.verify_proof(left)) == checker.canonical_report_bytes(checker.verify_proof(right))

    mutated = copy.deepcopy(proof)
    mutated["nodes"][0, 0] += 0.125
    with pytest.raises(ValueError, match="claims disagree"):
        checker.verify_proof(mutated)
    with pytest.raises(ValueError, match="duplicate key"):
        checker.load_proof(b'{"schema":"x","schema":"y"}\n')
    with pytest.raises(ValueError, match="nonfinite"):
        checker.load_proof(b'{"schema":NaN}\n')


def _bounds_packet() -> dict[str, object]:
    packet: dict[str, object] = {
        "schema": domain.BOUNDS_PACKET_SCHEMA,
        "producer": "UNAPPROVED_TEST_INTERVAL_PRODUCER",
        "method": "TEST_ONLY_RATIONAL_INTERVALS",
        "source_pdf_sha256": domain.SOURCE_PDF_SHA256,
        "partition_sha256": domain.partition_sha256(),
        "bounds_by_path": {box.path: [["1/2", "1"]] * domain.EXPECTED_ORDERED_PIVOTS for box in domain.fixed_partition()},
        "packet_sha256": "",
    }
    packet["packet_sha256"] = domain.bounds_packet_sha256(packet)
    return packet


def test_domain_proves_quality_envelope_coercivity_by_compactness() -> None:
    leaves = domain.fixed_partition()
    assert len(leaves) == 2**domain.FIXED_DEPTH == 16
    assert domain.root_box() == domain.Box(
        (domain.Fraction(-5), domain.Fraction(5)),
        (domain.Fraction(1, 6), domain.Fraction(5)),
        0,
        "",
    )
    certificate = domain.certify_normalized_triangle_domain()
    assert certificate["classification"] == "PASS_E4_PL_S3_V2A_NORMALIZED_DOMAIN_CERTIFICATE"
    assert certificate["coverage"]["complete"] is True
    assert certificate["counts"] == {
        "ANALYTIC_COMPACTNESS_CERTIFIED": 16,
        "UNRESOLVED_GEOMETRY_SIGN": 0,
    }
    domain.verify_partition(certificate["leaves"])
    assert all("edge_reconstruction" not in str(row["geometry"]) for row in certificate["leaves"])
    assert all(row["geometry"]["a_delta_strictly_negative_for_supported_section"] for row in certificate["leaves"])

    analytic = certificate["analytic_certificate"]
    assert analytic["complete"] is True
    assert all(analytic["checks"].values())
    assert analytic["admission_superset"]["root_contains_complete_admitted_envelope"] is True
    area_floor = domain.NORMALIZED_AREA_THRESHOLD_BINARY64 - domain.QUALITY_COMPARISON_TOLERANCE_BINARY64
    assert area_floor > 0
    assert area_floor * area_floor > domain.Fraction(1, 3)
    assert domain.Fraction(4) + domain.QUALITY_COMPARISON_TOLERANCE_BINARY64 < domain.Fraction(5)
    assert analytic["pointwise_rank"] == {
        "physical_dimension": 15,
        "physical_rank": 9,
        "rigid_mode_count": 6,
        "total_dimension": 18,
        "total_rank": 12,
    }
    assert analytic["quotient"]["uniform_over_geometry"] is True
    assert analytic["quotient"]["numeric_lower_bound_claimed"] is False
    assert analytic["quotient"]["uniform_over_unbounded_or_degenerating_section_parameters"] is False
    assert analytic["smallest_unresolved_expression"] is None
    assert certificate["ordered_sign_diagnostics"] == {
        "approved": False,
        "approved_hash_count": 0,
        "classifying": False,
        "disposition": "NO_PACKET_SUPPLIED_NOT_REQUIRED",
        "packet_sha256": "",
    }


def test_unapproved_ordered_sign_packet_cannot_override_analytic_certificate() -> None:
    packet = _bounds_packet()
    unapproved = domain.certify_normalized_triangle_domain(packet)
    assert unapproved["classification"] == "PASS_E4_PL_S3_V2A_NORMALIZED_DOMAIN_CERTIFICATE"
    assert unapproved["ordered_sign_diagnostics"]["approved"] is False
    assert unapproved["ordered_sign_diagnostics"]["classifying"] is False
    assert unapproved["ordered_sign_diagnostics"]["disposition"] == (
        "UNAPPROVED_PACKET_RETAINED_AS_NONCLASSIFYING_DIAGNOSTIC"
    )
    assert domain.APPROVED_BOUNDS_PACKET_SHA256 == ()

    mutated = copy.deepcopy(packet)
    mutated["producer"] = "MUTATED"
    with pytest.raises(ValueError, match="self-hash mismatch"):
        domain.certify_normalized_triangle_domain(mutated)
    floating = copy.deepcopy(packet)
    floating["bounds_by_path"][next(iter(floating["bounds_by_path"]))][0] = [0.5, 1.0]
    floating["packet_sha256"] = domain.bounds_packet_sha256(floating)
    with pytest.raises(ValueError, match="exact rational"):
        domain.certify_normalized_triangle_domain(floating)


def test_domain_fails_closed_if_the_positive_height_gap_is_removed(monkeypatch) -> None:
    monkeypatch.setattr(domain, "ROOT_B", (domain.Fraction(0), domain.Fraction(5)))
    result = domain.certify_normalized_triangle_domain()
    assert result["classification"] == "UNCLASSIFIED_E4_PL_S3_V2A_DOMAIN_COERCIVITY"
    assert result["analytic_certificate"]["complete"] is False
    assert result["analytic_certificate"]["smallest_unresolved_expression"] == (
        "MIN_ROOT_LAMBDA_7_OF_FINAL_STIFFNESS"
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("variational_positive_terms", "MUTATED_POSITIVITY_ROUTE"),
        ("general_nullspace_proof", "MUTATED_NULLSPACE_ROUTE"),
        ("physical_rank", 8),
        ("total_rank", 11),
        ("ordered_spectrum_theorem", "MUTATED_SPECTRAL_ROUTE"),
        ("compact_minimum_theorem", "MUTATED_COMPACTNESS_ROUTE"),
        ("equation_map_sha256", "0" * 64),
    ),
)
def test_domain_fails_closed_if_a_theorem_premise_changes(
    monkeypatch, key: str, value: object
) -> None:
    authority = dict(domain.DOMAIN_THEOREM_AUTHORITY)
    authority[key] = value
    monkeypatch.setattr(domain, "DOMAIN_THEOREM_AUTHORITY", authority)
    result = domain.certify_normalized_triangle_domain()
    analytic = result["analytic_certificate"]
    assert result["classification"] == "UNCLASSIFIED_E4_PL_S3_V2A_DOMAIN_COERCIVITY"
    assert analytic["complete"] is False
    assert analytic["checks"]["theorem_authority_hash_matches"] is False
    assert analytic["smallest_unresolved_expression"] == (
        "MIN_ROOT_LAMBDA_7_OF_FINAL_STIFFNESS"
    )


def test_independent_modules_have_no_solver_or_production_imports() -> None:
    allowed_roots = {
        "__future__",
        "dataclasses",
        "fractions",
        "functools",
        "hashlib",
        "importlib",
        "itertools",
        "json",
        "math",
        "pathlib",
        "sys",
        "typing",
        "numpy",
    }
    for path in (REFERENCE_PATH, CHECKER_PATH, DOMAIN_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert imports <= allowed_roots
        assert "from anyfesolver" not in source
        assert "import anyfesolver" not in source

    checker_source = CHECKER_PATH.read_text(encoding="utf-8")
    checker_tree = ast.parse(checker_source)
    checker_imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(checker_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    checker_imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(checker_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "importlib" not in checker_imports
    assert "pathlib" not in checker_imports
    assert "e4_pl_s3_v2_independent_reference" not in checker_source
    assert hashlib.sha256(EQUATION_MAP_PATH.read_bytes()).hexdigest().upper() == domain.EQUATION_MAP_SHA256


def test_reference_fails_closed_for_degenerate_anisotropic_or_coupled_input() -> None:
    with pytest.raises(ValueError, match="nondegenerate"):
        reference.assemble_flat_reference(((0, 0), (1, 0), (2, 0)), _section())
    coupled = _section()
    coupled[0, 3] = coupled[3, 0] = 0.01
    with pytest.raises(ValueError, match="coupling"):
        reference.assemble_flat_reference(NODES, coupled)
    anisotropic = _section()
    anisotropic[0, 0] *= 1.1
    with pytest.raises(ValueError, match="non-isotropic membrane"):
        reference.assemble_flat_reference(NODES, anisotropic)
