"""Independent checks for the literal seed-gap reference.

These checks are deliberately self-contained: the reference module is
stdlib-only, no ``anyfesolver`` module is imported and no production
gap/witness kernel is called for any oracle value. The certified scope is
the single literal captured Q4/S3 pair path; nothing here asserts anything
about other faces, poses or the whole solver path.

Checks:
- artifact binding (sha256, embedded/artifact equality, tolerance);
- exact rational point evaluation sanity and two-formula consistency
  (contract gap vs plane height vs local-normal gap);
- directed-float interval containment of exact rational point values;
- the literal-path discriminator certificate (exact + float mirror) with a
  machine-readable result written to the attempt TEMP directory;
- synthetic deep-violation detection (the reference must be able to prove
  a violation when one exists);
- synthetic separation certification.
"""
import importlib.util
import json
import math
import os
import sys
import pytest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "scripts" / "general_contact_seed_gap_reference.py"
GAP_TOLERANCE = 5e-5


def _load_reference():
    spec = importlib.util.spec_from_file_location(
        "general_contact_seed_gap_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load_reference()


def test_artifact_binding_and_embedded_literal():
    if not os.path.isfile(ref.INPUT_SOURCE):
        pytest.skip('original seed-gap capture absent: archived artifact binding requires '+ref.INPUT_SOURCE)
    record = ref.load_literal_record()
    original = record.get('input_source')
    if not original or not os.path.isfile(original):
        pytest.skip('original seed-gap witness absent: archived source binding requires '+str(original))
    assert record['artifact_verified'] is True
    assert record['original_artifact_verified'] is True
    test_embedded_literal_with_optional_artifact_status()


def test_embedded_literal_with_optional_artifact_status():
    record = ref.load_literal_record()
    # Exact archive binding only when the original delivery
    # witness is present; the embedded literal checks below
    # continue on hosts without the archived witness.
    if os.path.isfile(ref.INPUT_SOURCE):
        assert record["artifact_verified"] is True
    else:
        assert record["artifact_verified"] is False
    original = record.get("input_source")
    if original and os.path.isfile(original):
        assert record["original_artifact_verified"] is True
    else:
        assert record["original_artifact_verified"] is False
    assert record["path_index"] == 0
    assert record["input_sha256"] == (
        "f8e7dee46a3135c621492364f5f727d0eb34d57873afeb7146606c4bda31a114")
    assert float(record["path"]["gap_tolerance_m"]) == GAP_TOLERANCE
    data = ref.PairPathData(record)
    assert data.tolerance == Fraction(float(GAP_TOLERANCE))
    assert len(data.a0) == 4 and len(data.b0) == 3
    assert len(data.e0) == 4 and len(data.e2) == 3


def test_exact_start_corner_gap_positive_foot_inside():
    data = ref.PairPathData(ref.load_literal_record())
    point = ref.exact_point(data, 0, 0, 0)
    assert point["gap"][0] > 0
    assert point["gap"][1] < 1e-4
    assert point["lambda1"] >= 0
    assert point["lambda2"] >= 0
    assert point["lambda_sum"] <= 1
    gap_mid = sum(float(x) for x in point["gap"]) / 2
    for other in ("plane_height", "local_normal_gap"):
        other_mid = sum(float(x) for x in point[other]) / 2
        relative = abs(gap_mid - other_mid) / max(1e-12, abs(other_mid))
        assert relative < 1e-8


def test_directed_float_box_contains_exact_point_values():
    data = ref.PairPathData(ref.load_literal_record())
    points = [
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
        (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)),
        (Fraction(1), Fraction(1) - Fraction(1, 2 ** 17), Fraction(1)),
    ]
    for t, u, v in points:
        exact = ref.exact_point(data, t, u, v)
        box = ref.evaluate_box_float(data, (t, t), (u, u), (v, v))
        assert box["chart_ok"]
        def contains(interval, value):
            return (Fraction(interval[0]) <= Fraction(value)
                    <= Fraction(interval[1]))
        def contains_interval(interval, exact_interval):
            return (Fraction(interval[0]) <= Fraction(exact_interval[0])
                    and Fraction(exact_interval[1]) <= Fraction(interval[1]))
        assert contains(box["height"], exact["height"])
        assert contains(box["projected"], exact["projected"])
        assert contains(box["parent_actual"], exact["parent_actual"])
        assert contains_interval(box["gap"], exact["gap"])
        assert contains(box["lambda1"], exact["lambda1"])
        assert contains(box["lambda2"], exact["lambda2"])

    # Non-degenerate boxes: the exact vertex enclosure must contain the
    # exact point values at interior points, and the directed-float mirror
    # must contain the exact enclosure (box-enclosure semantics).
    boxes = [
        ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)),
         (Fraction(0), Fraction(1))),
        ((Fraction(1, 2), Fraction(1)), (Fraction(1, 4), Fraction(3, 4)),
         (Fraction(0), Fraction(1, 2))),
    ]
    samples = [
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(3, 4), Fraction(1, 3), Fraction(2, 3)),
        (Fraction(9, 16), Fraction(5, 8), Fraction(1, 16)),
        (Fraction(1, 2), Fraction(1, 4), Fraction(0)),
    ]
    for t_box, u_box, v_box in boxes:
        exact_box = ref.evaluate_box_exact(data, t_box, u_box, v_box)
        float_box = ref.evaluate_box_float(data, t_box, u_box, v_box)
        assert exact_box["chart_ok"]
        assert float_box["chart_ok"]
        for key in ("height", "projected", "parent_actual", "gap",
                    "lambda1", "lambda2"):
            assert float_box[key][0] <= exact_box[key][0]
            assert exact_box[key][1] <= float_box[key][1]
        for t, u, v in samples:
            if not (t_box[0] <= t <= t_box[1] and u_box[0] <= u <= u_box[1]
                    and v_box[0] <= v <= v_box[1]):
                continue
            point = ref.exact_point(data, t, u, v)
            for d in range(3):
                assert exact_box["point"][d][0] <= point["point"][d]
                assert point["point"][d] <= exact_box["point"][d][1]
                assert float_box["point"][d][0] <= exact_box["point"][d][0]
                assert exact_box["point"][d][1] <= float_box["point"][d][1]
            assert exact_box["height"][0] <= point["height"] <= exact_box["height"][1]
            assert (exact_box["projected"][0] <= point["projected"]
                    <= exact_box["projected"][1])
            assert (exact_box["parent_actual"][0] <= point["parent_actual"]
                    <= exact_box["parent_actual"][1])
            assert exact_box["gap"][0] <= point["gap"][0]
            assert point["gap"][1] <= exact_box["gap"][1]
            assert exact_box["lambda1"][0] <= point["lambda1"] <= exact_box["lambda1"][1]
            assert exact_box["lambda2"][0] <= point["lambda2"] <= exact_box["lambda2"][1]
            if (point["lambda1"] > 0 and point["lambda2"] > 0
                    and point["lambda_sum"] < 1):
                assert exact_box["feasibility"] != "outside"

    # Partition-of-unity correlation is retained at the root box: the
    # derivative components stay at the skin edge length instead of the
    # [0, 1] hull a natural extension of the weighted sums produces.
    root = ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)),
            (Fraction(0), Fraction(1)))
    enclosure = ref.evaluate_box_exact(data, *root)
    assert enclosure["chart_ok"]
    du_x = enclosure["du"][0]
    dv_y = enclosure["dv"][1]
    assert du_x[1] - du_x[0] < Fraction(1, 10**4)
    assert dv_y[1] - dv_y[0] < Fraction(1, 10**4)
    assert Fraction(49, 100) < du_x[0] and du_x[1] < Fraction(51, 100)
    assert Fraction(49, 100) < dv_y[0] and dv_y[1] < Fraction(51, 100)


def test_literal_path_zero_discriminator_certificate():
    result = ref.run_discriminator()
    payload = json.dumps(result, indent=2)
    temp = os.environ.get("TEMP")
    if temp:
        with open(os.path.join(temp, "seed_gap_reference_result.json"),
                  "w", encoding="utf-8") as handle:
            handle.write(payload)
    print("literal path 0 discriminator:",
          json.dumps({key: result[key] for key in
                      ("gap_tolerance_m", "min_gap_interval_m", "decision",
                       "exact_rational", "float_directed",
                       "located_minimum", "witness_grid_27",
                       "certified_scope")}, indent=2))
    if os.path.isfile(ref.INPUT_SOURCE):
        assert result["input"]["artifact_verified"] is True
    else:
        assert result["input"]["artifact_verified"] is False
    original = result["input"]["source"]
    if original and os.path.isfile(original):
        assert result["input"]["original_artifact_verified"] is True
    else:
        assert result["input"]["original_artifact_verified"] is False
    exact = result["exact_rational"]
    assert exact["status"] == "safe_within_tolerance"
    assert exact["remaining_boxes"] == []
    assert exact["unresolved_chart_boxes"] == []
    lower = exact["min_gap_lower_m"]
    assert lower is not None and lower >= -GAP_TOLERANCE
    floated = result["float_directed"]
    assert floated["status"] == "safe_within_tolerance"
    assert floated["min_gap_lower_m"] >= -GAP_TOLERANCE
    assert abs(floated["min_gap_lower_m"] - lower) <= 1e-12
    interval = result["min_gap_interval_m"]
    assert interval[0] <= interval[1]
    assert interval[0] >= -GAP_TOLERANCE
    feasible = result["located_minimum"]["feasible_point"]
    assert feasible is not None
    assert feasible["foot_inside"] is True
    assert feasible["lambda_sum"][1] <= 1.0
    assert feasible["gap_upper_m"] >= interval[0]
    assert result["witness_grid_27"]["violating_samples"] == 0
    assert result["decision"]["path_safe_within_tolerance"] is True
    assert result["decision"]["violation_proven"] is False


def test_reference_detects_synthetic_deep_violation():
    record = ref.load_literal_record()
    for corner in record["path"]["source_end_m"]:
        corner[2] -= 4e-4
    data = ref.PairPathData(record)
    exact = ref.certify_exact(data)
    assert exact["status"] == "violation_proven"
    violation = exact["violation"]
    assert violation is not None
    witness = violation["point_witness"]
    assert witness is not None
    assert witness["enclosure"]["feasibility"] == "inside"
    assert witness["enclosure"]["gap"][1] < -data.tolerance
    floated = ref.certify_float(data)
    assert floated["status"] == "violation_proven"


def test_reference_certifies_synthetic_separation():
    record = ref.load_literal_record()
    for key in ("source_start_m", "source_end_m"):
        for corner in record["path"][key]:
            corner[2] += 1e-3
    data = ref.PairPathData(record)
    exact = ref.certify_exact(data)
    assert exact["status"] == "safe_within_tolerance"
    assert float(exact["min_gap_lower_m"]) > 0.0
    floated = ref.certify_float(data)
    assert floated["status"] == "safe_within_tolerance"
    assert floated["min_gap_lower_m"] > 0.0
