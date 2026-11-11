from __future__ import annotations

import ast
from decimal import Decimal, getcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CERTIFIER_PATH = ROOT / "docs/reference_cases/e4_pl_s3_domain_coercivity.py"
CERTIFICATE_PATH = (
    ROOT / "docs/reference_cases/e4_pl_s3_domain_coercivity_certificate.json"
)


def _load_certifier():
    spec = importlib.util.spec_from_file_location(
        "e4_pl_s3_independent_domain_coercivity", CERTIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


certifier = _load_certifier()


def _load_binary64_reference():
    path = ROOT / "docs/reference_cases/e4_pl_s3_linear_reference.py"
    spec = importlib.util.spec_from_file_location(
        "e4_pl_s3_domain_crosscheck_reference", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contains(interval, exact: Fraction) -> bool:
    return (
        Fraction.from_float(interval.lo)
        <= exact
        <= Fraction.from_float(interval.hi)
    )


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key}")
        result[key] = value
    return result


def test_scalar_outward_intervals_enclose_exact_rational_operations() -> None:
    values = (
        Fraction(-13, 17),
        Fraction(-1, 10),
        Fraction(0),
        Fraction(1, 6),
        Fraction(7, 13),
        Fraction(19, 3),
    )
    for left in values:
        left_interval = certifier.Interval.rational(left)
        assert _contains(left_interval, left)
        for right in values:
            right_interval = certifier.Interval.rational(right)
            assert _contains(left_interval + right_interval, left + right)
            assert _contains(left_interval - right_interval, left - right)
            assert _contains(left_interval * right_interval, left * right)
            if right:
                assert _contains(left_interval / right_interval, left / right)
    with pytest.raises(ZeroDivisionError):
        certifier.Interval(-1.0, 1.0).reciprocal()
    with pytest.raises(ValueError):
        certifier.Interval(2.0, 1.0)


def test_exact_laurent_kernel_and_selected_minor_identities() -> None:
    identities = certifier._exact_rigid_identity()
    assert identities["physical_rigid_identity"] is True
    assert identities["pl_rigid_identity"] is True
    assert identities["physical_rigid_nonzero_terms"] == 0
    assert identities["pl_rigid_nonzero_terms"] == 0
    assert identities["rigid_dimension"] == 6
    assert identities["physical_selected_minor_identity"] is True
    assert identities["total_selected_minor_identity"] is True
    assert identities["selected_minor_determinant"] == "-C/B_POWER_4"
    registered = Fraction(*identities["selected_minor_constant"])
    assert registered > 0
    assert Fraction(
        *identities["derived_physical_selected_minor_constant"]
    ) == registered
    assert Fraction(
        *identities["derived_total_selected_minor_constant"]
    ) == registered


def test_assumed_shear_coefficients_bind_binary64_not_ideal_thirds(
    monkeypatch,
) -> None:
    expected_two_thirds = Fraction.from_float(2.0 / 3.0)
    expected_one_third = Fraction.from_float(1.0 / 3.0)
    assert certifier.SHEAR_TWO_THIRDS_BINARY64 == expected_two_thirds
    assert certifier.SHEAR_ONE_THIRD_BINARY64 == expected_one_third
    assert expected_two_thirds == 2 * expected_one_third
    assert expected_two_thirds != Fraction(2, 3)
    assert expected_one_third != Fraction(1, 3)

    frozen = certifier._binary64_input_record()["assumed_shear_interpolation"]
    assert Fraction(*frozen["two_thirds_exact_binary64_ratio"]) == expected_two_thirds
    assert Fraction(*frozen["one_third_exact_binary64_ratio"]) == expected_one_third
    baseline_hash = hashlib.sha256(
        certifier._canonical_bytes(certifier._binary64_input_record())
    ).hexdigest()

    monkeypatch.setattr(certifier, "SHEAR_TWO_THIRDS_BINARY64", Fraction(2, 3))
    monkeypatch.setattr(certifier, "SHEAR_ONE_THIRD_BINARY64", Fraction(1, 3))
    mutated_hash = hashlib.sha256(
        certifier._canonical_bytes(certifier._binary64_input_record())
    ).hexdigest()
    assert mutated_hash != baseline_hash

    identities = certifier._exact_rigid_identity()
    assert identities["physical_rigid_identity"] is True
    assert identities["pl_rigid_identity"] is True
    assert identities["physical_selected_minor_identity"] is False
    assert identities["total_selected_minor_identity"] is False
    assert Fraction(
        *identities["derived_physical_selected_minor_constant"]
    ) != certifier.SELECTED_MINOR_CONSTANT
    result = certifier.run_certificate(max_depth=2, max_processed=16)
    assert result["classification"] == "UNRESOLVED"
    assert result["first_unresolved"] == {
        "analytic_failures": [
            "physical_selected_minor_identity",
            "total_selected_minor_identity",
        ],
        "reason": "EXACT_ANALYTIC_IDENTITY_FAILED",
    }


@pytest.mark.parametrize("a,b", ((0.0, 1.0 / 6.0), (0.5, 0.8), (1.0, 1.0)))
def test_selected_maps_enclose_separately_authored_binary64_reference(a, b) -> None:
    reference = _load_binary64_reference()
    local = np.asarray(((0.0, 0.0), (1.0, 0.0), (a, b)), dtype=float)
    _local, inverse, _determinant = reference._geometry(local)
    operators = [
        reference._kinematic(local, inverse, r, s)
        for r, s, _weight in reference.SEVEN_POINT_RULE[:2]
    ]
    expected_physical = np.asarray(
        [
            operators[station][component][list(certifier.UNCONDENSED_QUOTIENT_INDICES)]
            for station, component in certifier.SELECTED_PHYSICAL_ROWS
        ]
    )
    physical, total = certifier._selected_interval_matrices(
        certifier.Interval.point(a), certifier.Interval.point(b)
    )
    assert len(total) == len(total[0]) == 14
    for row in range(11):
        for column in range(11):
            assert physical[row][column].lo <= expected_physical[row, column]
            assert expected_physical[row, column] <= physical[row][column].hi
    expected_determinant = -float(certifier.SELECTED_MINOR_CONSTANT) / b**4
    assert np.linalg.det(expected_physical) == pytest.approx(
        expected_determinant, rel=2.0e-12, abs=1.0e-15
    )


def test_root_is_a_strict_superset_of_the_complete_quality_envelope() -> None:
    assert certifier.ROOT == certifier.Box(
        Fraction(-5), Fraction(5), Fraction(1, 6), Fraction(5)
    )
    # A conservative proof-only margin is used for the superset derivation.
    # It is not mislabeled as an ANYmesh admission tolerance.  Since
    # sum(l_i^2)>=1 after longest-length scaling, eta >= 0.6-margin gives the
    # lower b there.  Edge-01 scaling can only increase b.  The edge-ratio
    # safety bound is strictly below five even with the proof margin.
    getcontext().prec = 80
    lower_from_area = (
        Decimal(3) / Decimal(5) - Decimal.from_float(1.0e-12)
    ) / (Decimal(2) * Decimal(3).sqrt())
    assert lower_from_area > Decimal(1) / Decimal(6)
    assert Decimal(4) + Decimal.from_float(1.0e-12) < Decimal(5)

    # This admitted shape has production edge 01 shorter than another edge and
    # a<0, directly demonstrating why the old longest-edge lens was invalid.
    a = -0.2
    b = 1.0
    lengths = np.asarray((1.0, math.hypot(a, b), math.hypot(a - 1.0, b)))
    cosines = np.asarray(
        (
            a / lengths[1],
            (1.0 - a) / lengths[2],
            (a * (a - 1.0) + b * b) / (lengths[1] * lengths[2]),
        )
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    normalized_area = 2.0 * math.sqrt(3.0) * b / float(lengths @ lengths)
    assert a < 0.0
    assert np.min(angles) >= 30.0
    assert np.max(angles) <= 150.0
    assert np.max(lengths) / np.min(lengths) <= 4.0
    assert np.min(np.sin(np.radians(angles))) >= 0.20
    assert normalized_area >= 0.60


def test_edge01_to_longest_metric_equivalence_is_explicit() -> None:
    reference = _load_binary64_reference()
    a = -0.2
    b = 1.0
    c = 0.4
    edge_nodes = np.asarray(((0.0, 0.0), (1.0, 0.0), (a, b)))
    longest_nodes = c * edge_nodes
    _edge_local, edge_inverse, _edge_det = reference._geometry(edge_nodes)
    _long_local, longest_inverse, _long_det = reference._geometry(longest_nodes)
    t17 = np.eye(17)
    for node in range(3):
        t17[5 * node : 5 * node + 3, 5 * node : 5 * node + 3] *= c
    strain_scale = np.diag((1.0, 1.0, 1.0, 1.0 / c, 1.0 / c, 1.0 / c, 1.0, 1.0))
    for r, s, _weight in reference.SEVEN_POINT_RULE:
        edge_operator = reference._kinematic(edge_nodes, edge_inverse, r, s)
        longest_operator = reference._kinematic(
            longest_nodes, longest_inverse, r, s
        )
        np.testing.assert_allclose(
            longest_operator @ t17,
            strain_scale @ edge_operator,
            rtol=4.0e-14,
            atol=4.0e-14,
        )

    def constraint(inverse: np.ndarray) -> np.ndarray:
        dr = np.asarray((-1.0, 1.0, 0.0))
        ds = np.asarray((-1.0, 0.0, 1.0))
        dx = inverse[0, 0] * dr + inverse[0, 1] * ds
        dy = inverse[1, 0] * dr + inverse[1, 1] * ds
        made = np.zeros((3, 18))
        for row in range(3):
            made[row, 0::6] = 0.5 * dy
            made[row, 1::6] = -0.5 * dx
            made[row, 6 * row + 5] = 1.0
        return made

    t18 = np.eye(18)
    for node in range(3):
        t18[6 * node : 6 * node + 3, 6 * node : 6 * node + 3] *= c
    np.testing.assert_allclose(
        constraint(longest_inverse) @ t18,
        constraint(edge_inverse),
        rtol=0.0,
        atol=2.0e-15,
    )

    def rigid(nodes: np.ndarray) -> np.ndarray:
        made = np.zeros((18, 6))
        for node, (x, y) in enumerate(nodes):
            base = 6 * node
            made[base : base + 3, :3] = np.eye(3)
            made[base + 2, 3] = y
            made[base + 2, 4] = -x
            made[base, 5] = -y
            made[base + 1, 5] = x
            made[base + 3 : base + 6, 3:] = np.eye(3)
        return made

    edge_rigid = rigid(edge_nodes)
    longest_rigid = rigid(longest_nodes)
    assert np.linalg.matrix_rank(t18 @ edge_rigid) == 6
    projection_edge = np.eye(18) - edge_rigid @ np.linalg.pinv(edge_rigid)
    projection_longest = (
        np.eye(18) - longest_rigid @ np.linalg.pinv(longest_rigid)
    )
    np.testing.assert_allclose(
        projection_longest @ (t18 @ edge_rigid),
        np.zeros((18, 6)),
        rtol=0.0,
        atol=2.0e-15,
    )
    q = np.linspace(-0.7, 1.1, 18)
    distance_edge = np.linalg.norm(projection_edge @ q)
    distance_longest = np.linalg.norm(projection_longest @ (t18 @ q))
    assert c * distance_edge <= distance_longest + 2.0e-15
    assert distance_longest <= distance_edge + 2.0e-15


def test_complete_certificate_closes_all_local_domain_obligations() -> None:
    started = time.perf_counter()
    result = certifier.run_certificate()
    assert time.perf_counter() - started < 5.0
    assert result["schema"] == certifier.SCHEMA
    assert result["classification"] == "CERTIFIED_COMPLETE"
    assert result["first_unresolved"] is None
    assert result["admission"]["complete_envelope_bound"] is True
    exact_constraints = result["admission"]["exact_constraints"]
    assert "comparison_tolerance_binary64_ratio" not in exact_constraints
    assert Fraction(
        *result["admission"]["proof_derivation_margin_binary64_ratio"]
    ) == Fraction.from_float(1.0e-12)
    assert result["cover"] == {
        "leaf_partition_sha256": (
            "0FF7E01B583874B04E982AD4B0568960DD8D83342D2AA774E1A43AA17132188F"
        ),
        "maximum_depth": 0,
        "partition_policy_id": certifier.PARTITION_POLICY_ID,
        "positive_leaf_count": 1,
        "processed_count": 1,
        "reason_counts": {},
        "unresolved_leaf_count": 0,
    }
    assert result["proof_obligations"] == {
        "bubble_spd": True,
        "condensed_physical_rank": 9,
        "full_saddle_inertia": [14, 3, 6],
        "full_saddle_rank": 17,
        "physical_uncondensed_rank": 11,
        "pl_rank": 3,
        "rigid_modes": 6,
        "schur_complement_spd_on_physical_quotient": True,
        "strictly_positive_quotient_lower_bound": True,
        "total_rank": 12,
        "total_uncondensed_rank": 14,
    }
    quotient = result["quotient"]
    assert quotient["metric_id"] == (
        "LOCAL_EDGE_01_ONE_EUCLIDEAN_DISTANCE_TO_RIGID_KERNEL_V1"
    )
    for key in (
        "minimum_baseline_bubble_eigenvalue_lower_ratio",
        "minimum_baseline_condensed_physical_eigenvalue_lower_ratio",
        "minimum_baseline_total_eigenvalue_lower_ratio",
        "minimum_baseline_uncondensed_eigenvalue_lower_ratio",
    ):
        assert Fraction(*quotient[key]) > 0
    equivalence = result["longest_edge_metric_equivalence"]
    edge_lower = Fraction(
        *quotient["minimum_baseline_total_eigenvalue_lower_ratio"]
    )
    longest_lower = Fraction(
        *equivalence["minimum_baseline_total_eigenvalue_lower_ratio"]
    )
    assert equivalence["edge_ratio_scale"] == {
        "c_definition": "L01_OVER_LMAX",
        "c_lower_open": [1, 5],
        "c_upper_closed": [1, 1],
    }
    assert 0 < longest_lower <= edge_lower / 25
    assert equivalence["transferred_lower_bound"].endswith("OVER_25")
    assert result["constitutive_scaling"]["baseline_drill_scale"] == [1, 2]
    assert result["formulation_input_authority"].startswith(
        "IMPLEMENTED_BINARY64_TYING_OFFSET"
    )
    assert result["reference_surface_offset"] == {
        "baseline_numeric_lower_bound_offset": [0, 1],
        "finite_offset_composition": (
            "INVERTIBLE_REFERENCE_SURFACE_STRAIN_AND_DOF_CONGRUENCE_"
            "PRESERVES_RANK_INERTIA_AND_POSITIVE_SEMIDEFINITENESS"
        ),
        "offset_uniform_euclidean_lower_bound": False,
        "scope": (
            "STRICT_COERCIVITY_EXTENDS_POINTWISE_TO_EACH_FINITE_OFFSET;"
            "NO_UNIFORM_CONSTANT_IS_CLAIMED_OVER_AN_UNBOUNDED_OFFSET_PARAMETER"
        ),
    }
    assert len(result["frozen_binary64_input_sha256"]) == 64


def test_analytic_or_bound_mutation_is_truthfully_unresolved(monkeypatch) -> None:
    monkeypatch.setattr(certifier, "SELECTED_MINOR_CONSTANT", Fraction(1))
    result = certifier.run_certificate(max_depth=2, max_processed=16)
    assert result["classification"] == "UNRESOLVED"
    assert result["proof_obligations"]["total_rank"] is None
    assert result["first_unresolved"]["reason"] == "EXACT_ANALYTIC_IDENTITY_FAILED"
    assert result["cover"]["unresolved_leaf_count"] == 1


def test_two_fresh_outputs_are_byte_identical_canonical_and_exclusive(tmp_path: Path) -> None:
    first = tmp_path / "cycle-1" / "certificate.json"
    second = tmp_path / "cycle-2" / "certificate.json"
    first_bytes = certifier.write_certificate(first, max_depth=20, max_processed=200_000)
    second_bytes = certifier.write_certificate(second, max_depth=20, max_processed=200_000)
    assert first_bytes == second_bytes
    assert first.read_bytes() == second.read_bytes() == first_bytes
    assert CERTIFICATE_PATH.read_bytes() == first_bytes
    assert hashlib.sha256(first_bytes).hexdigest().upper() == hashlib.sha256(
        second_bytes
    ).hexdigest().upper()
    decoded = json.loads(
        first_bytes,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert decoded["classification"] == "CERTIFIED_COMPLETE"
    assert first_bytes == certifier._canonical_bytes(decoded)
    with pytest.raises(FileExistsError):
        certifier.write_certificate(first, max_depth=20, max_processed=200_000)


def test_certifier_is_independently_authored_and_research_only() -> None:
    tree = ast.parse(CERTIFIER_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not any(
        name.startswith(("anyfesolver", "e4_pl_s3_", "sympy", "scipy", "numpy"))
        for name in imports
    )
    source = CERTIFIER_PATH.read_text(encoding="utf-8")
    assert "evalf" not in source
    assert "simplify(" not in source
    assert "e4_pl_s3_element" not in source
    assert "e4_pl_s3_linear_reference" not in source
    assert "e4_pl_s3_exact_oracle" not in source


@pytest.mark.parametrize(
    ("max_depth", "max_processed"),
    ((-1, 1), (1, 0), (True, 1), (1, False)),
)
def test_invalid_resource_bounds_fail_closed(max_depth, max_processed) -> None:
    with pytest.raises(ValueError):
        certifier.run_certificate(
            max_depth=max_depth,
            max_processed=max_processed,
        )
