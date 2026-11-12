"""One parameterized literal-record diagnosis at the actual seed refusal.

Scope: D1_SEED_PATH_ADMISSION_DELIVERY_DEV600_V1 (Mistral-owned slice).

The independent stdlib-only reference (scripts/general_contact_seed_gap_reference.py,
extended in this slice with the ten-record capture loader, exact rational
string exports, end-pose gap-definition distinction and chart/budget
inconclusive refusal classification) certifies each of the ten literal
capture records (2 endpoint + 8 path witnesses, sha256-bound to the original
capture artifact). The unchanged production stage chain is then evaluated
per record with the exact production schedules and the per-record refusing
stage is located. All raw evidence is written incrementally to the guard's
unique TEMP directory; assertions here are structural invariants of the
enclosures only, never admission decisions.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "scripts" / "general_contact_seed_gap_reference.py"
CAPTURE_SHA256 = (
    "f8e7dee46a3135c621492364f5f727d0eb34d57873afeb7146606c4bda31a114"
)
GAP_TOLERANCE = 5e-5


def _load_reference():
    spec = importlib.util.spec_from_file_location(
        "general_contact_seed_gap_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load_reference()


def _atomic_write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def _arrays(record):
    path = record["path"]
    return (np.asarray(path["source_start_m"], dtype=float),
            np.asarray(path["source_end_m"], dtype=float),
            np.asarray(path["target_start_m"], dtype=float),
            np.asarray(path["target_end_m"], dtype=float),
            tuple(np.asarray(item, dtype=float)
                  for item in path["endpoint_errors"]),
            float(path["gap_tolerance_m"]))


def _timed(stage, name, stages):
    started = time.perf_counter()
    try:
        value = stage()
    except Exception as exc:  # a refusing stage is evidence, not a test error
        stages[name] = {"error": f"{type(exc).__name__}: {exc}"[:300],
                        "elapsed_s": time.perf_counter() - started}
        return None
    stages[name] = {"value": value,
                    "elapsed_s": time.perf_counter() - started}
    return value


def _production_stages(record):
    """Exact production schedules, per stage, for one literal record.

    Endpoint records are the captured refusers and get the complete
    production endpoint chain. Path records are the eight 0.25-path pair
    paths whose production admission is already established by the capture
    (fraction 0.25 returned, so every pair admitted at the matching or
    overlap-local stage); they get the cheap stages and the independent
    certificate, and the expensive matching/overlap-local schedules are
    recorded as skipped-known-admitted.
    """
    from anyfesolver.general_contact import q4_overlap_local_bound as bounds
    from anyfesolver.general_contact.q4_source_overlap import (
        q4_source_projected_gap_lower,
    )
    a0, a1, b0, b1, errors, tolerance = _arrays(record)
    stages = {}
    if record["kind"] == "path":
        _timed(
            lambda: bounds.q4_s3_whole_pair_swept_lower(
                a0, a1, b0, b1, endpoint_errors=errors),
            "whole_pair_swept_lower", stages)
        _timed(
            lambda: bounds.overlap_violating_point_witness(
                a0, a1, b0, b1, gap_tolerance=tolerance,
                endpoint_errors=errors),
            "overlap_violating_point_witness", stages)
        stages["matching_projection_swept_lower"] = {
            "skipped": "known_admitted_from_capture_fraction_0.25"}
        stages["overlap_local_swept_lower"] = {
            "skipped": "known_admitted_from_capture_fraction_0.25"}
        decision = "admitted_known_from_capture"
    else:
        projected = _timed(
            lambda: q4_source_projected_gap_lower(a0, b0),
            "q4_source_projected_gap_lower", stages)
        local = _timed(lambda: bounds.overlap_local_lower(a0, b0, 3),
                       "overlap_local_lower", stages)
        witness = _timed(
            lambda: bounds.overlap_violating_point_witness(
                a0, a0, b0, b0, gap_tolerance=tolerance),
            "overlap_violating_point_witness", stages)
        diagnostics = {}
        matching = _timed(
            lambda: bounds.matching_projection_swept_lower(
                a0, a0, b0, b0, 3, gap_budget=tolerance,
                adaptive_depth=7, node_budget=1024,
                overlap_dual=True, diagnostics=diagnostics),
            "matching_projection_swept_lower", stages)
        if projected is not None and projected >= -tolerance:
            decision = "admitted_at_projected_point"
        elif local is not None and local[0] >= -tolerance:
            decision = "admitted_at_overlap_local_point"
        elif witness is not None and witness[0]:
            decision = "refused_by_witness"
        elif matching is not None and matching[0] >= -tolerance:
            decision = "admitted_at_matching"
        else:
            decision = "refused_all_stages"
    return {"stages": stages, "production_decision": decision,
            "matching_diagnostics": {key: value for key, value
                                     in diagnostics.items()
                                     if isinstance(value, (int, float, bool))}
            if record["kind"] == "endpoint" else {}}


def test_literal_record_diagnosis_locates_the_refusing_stage():
    if not Path(ref.CAPTURE_SOURCE).is_file():
        pytest.skip('literal seed-path diagnosis needs local sha256-bound capture: '+ref.CAPTURE_SOURCE)
    records = ref.load_capture_records()
    assert len(records) == 10
    assert all(record["input_sha256"] == CAPTURE_SHA256
              for record in records)
    assert [record["kind"] for record in records] == (
        ["endpoint"]*2 + ["path"]*8)
    output = Path(os.environ["TEMP"]) / "seed_path_record_diagnosis.json"
    results = []
    for record in records:
        started = time.perf_counter()
        entry = ref.run_capture_record(record)
        entry["production"] = _production_stages(record)
        entry["record_elapsed_s"] = time.perf_counter() - started
        results.append(entry)
        _atomic_write(output, results)
    for entry in results:
        assert entry["gap_tolerance_m"] == GAP_TOLERANCE
        status = entry["float_directed"]["status"]
        assert status in {"safe_within_tolerance", "violation_proven",
                          "inconclusive"}
        lower = entry["float_directed"]["min_gap_lower_m"]
        upper = entry.get("min_gap_upper_m")
        if lower is not None and upper is not None:
            assert lower <= upper
        if status == "safe_within_tolerance":
            # A whole-domain foot-outside record is trivially safe and has
            # no finite gap interval at all (lower and upper both None).
            assert (lower is None and upper is None) or (
                lower is not None and lower >= -GAP_TOLERANCE)
        if status == "violation_proven":
            assert entry["float_directed"]["violation"] is not None
        assert entry["production"]["production_decision"] in {
            "admitted_known_from_capture", "refused_by_witness",
            "admitted_at_matching", "admitted_at_overlap_local_point",
            "admitted_at_projected_point", "refused_all_stages"}
    _atomic_write(output, results)
    print("record diagnosis:",
          json.dumps([{"kind": item["kind"],
                       "index": item["record_index"],
                       "float_status": item["float_directed"]["status"],
                       "min_gap_lower_m":
                           item["float_directed"]["min_gap_lower_m"],
                       "min_gap_upper_m": item.get("min_gap_upper_m"),
                       "production": item["production"]["production_decision"]}
                      for item in results], indent=1))
