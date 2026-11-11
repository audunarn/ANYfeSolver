"""Benchmark the qualified-S3 reference batch against its scalar fallback."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
THREAD_ENVIRONMENT_AT_IMPORT = {
    name: os.environ.get(name) for name in THREAD_ENVIRONMENT
}

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import anyfesolver.recovery_batches as recovery_batches  # noqa: E402
import anyfesolver.s3_reference_batch as s3_batch  # noqa: E402
from anyfesolver.e4_pl_s3_element import (  # noqa: E402
    QualifiedE4PLS3ShellElement,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement  # noqa: E402
from anyfesolver.fe_core import FEModel  # noqa: E402
from anyfesolver.matrix_assembly import assemble_stiffness_matrix  # noqa: E402
from anyfesolver.recovery import (  # noqa: E402
    RecoveryConfig,
    ResourceConfig,
    clear_recovery_batch_plan,
    recover_element_stresses_with_report,
)


def _build_model(element_count: int) -> FEModel:
    model = FEModel("qualified-s3-reference-benchmark")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    elements = []
    node_id = 1
    for index in range(int(element_count)):
        x = float(8 * index)
        node_ids = [node_id, node_id + 1, node_id + 2]
        for identifier, coordinate in zip(
            node_ids,
            ((x, 0.0, 0.0), (x + 3.0, 0.0, 0.0), (x, 3.0, 0.0)),
        ):
            model.add_node(identifier, *coordinate)
        element_id = index + 1
        elements.append(
            (
                element_id,
                QualifiedE4PLS3ShellElement(
                    element_id,
                    node_ids,
                    "steel",
                    thickness=0.05,
                    reference_normal=(0.0, 0.0, 1.0),
                ),
            )
        )
        node_id += 3
    for element_id, element in elements:
        model.add_element(element_id, element)
    return model


def _build_q4_model(element_count: int) -> FEModel:
    model = FEModel("qualified-q4-warm-stiffness-comparator")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    elements = []
    node_id = 1
    for index in range(int(element_count)):
        x = float(8 * index)
        node_ids = [node_id + offset for offset in range(4)]
        for identifier, coordinate in zip(
            node_ids,
            (
                (x, 0.0, 0.0),
                (x + 3.0, 0.0, 0.0),
                (x + 3.0, 3.0, 0.0),
                (x, 3.0, 0.0),
            ),
        ):
            model.add_node(identifier, *coordinate)
        element_id = index + 1
        elements.append(
            (
                element_id,
                QualifiedE4PLShellElement(
                    element_id,
                    node_ids,
                    "steel",
                    thickness=0.05,
                ),
            )
        )
        node_id += 4
    for element_id, element in elements:
        model.add_element(element_id, element)
    return model


def _summary(samples: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    median = float(statistics.median(float(value) for value in values))
    return {
        "samples_seconds": [float(value) for value in values],
        "median_seconds": median,
        "mad_seconds": float(
            statistics.median(abs(float(value) - median) for value in values)
        ),
        "p95_seconds": float(np.percentile(values, 95.0)),
    }


def _time(call: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = call()
    return float(time.perf_counter() - start), result


def _sparse_error(reference: Any, candidate: Any) -> Dict[str, Any]:
    same_structure = bool(
        np.array_equal(reference.indptr, candidate.indptr)
        and np.array_equal(reference.indices, candidate.indices)
    )
    difference = candidate - reference
    maximum = (
        float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    )
    scale = max(
        float(np.max(np.abs(reference.data))) if reference.nnz else 0.0,
        1.0,
    )
    return {
        "same_csr_structure": same_structure,
        "maximum_absolute_error": maximum,
        "maximum_scaled_error": maximum / scale,
    }


def _recovery_error(
    reference: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    if tuple(reference) != tuple(candidate):
        return {
            "same_order_and_fields": False,
            "maximum_absolute_error": float("inf"),
            "maximum_scaled_error": float("inf"),
        }
    maximum = 0.0
    scale = 1.0
    same = True
    for element_id, expected_fields in reference.items():
        actual_fields = candidate[element_id]
        if tuple(expected_fields) != tuple(actual_fields):
            same = False
            break
        for name, expected in expected_fields.items():
            actual = actual_fields[name]
            if isinstance(expected, np.ndarray):
                expected_values = np.asarray(expected, dtype=float)
                actual_values = np.asarray(actual, dtype=float)
                if expected_values.shape != actual_values.shape:
                    same = False
                    break
                maximum = max(
                    maximum,
                    float(
                        np.max(
                            np.abs(actual_values - expected_values),
                            initial=0.0,
                        )
                    ),
                )
                scale = max(
                    scale,
                    float(np.max(np.abs(expected_values), initial=0.0)),
                )
            elif actual != expected:
                same = False
                break
        if not same:
            break
    return {
        "same_order_and_fields": same,
        "maximum_absolute_error": maximum if same else float("inf"),
        "maximum_scaled_error": maximum / scale if same else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--return-global", action="store_true")
    parser.add_argument("--include-q4-comparator", action="store_true")
    args = parser.parse_args()
    if args.elements < s3_batch.MIN_REFERENCE_S3_RECOVERY_GROUP:
        parser.error(
            "elements must cover the qualified S3 recovery batch minimum "
            f"({s3_batch.MIN_REFERENCE_S3_RECOVERY_GROUP})"
        )
    if args.repeats < 11:
        parser.error("repeats must be at least 11")
    thread_environment = {
        name: os.environ.get(name) for name in THREAD_ENVIRONMENT
    }
    if (
        THREAD_ENVIRONMENT_AT_IMPORT != THREAD_ENVIRONMENT
        or thread_environment != THREAD_ENVIRONMENT
    ):
        parser.error(
            "benchmark requires OMP, OpenBLAS, MKL, and NumExpr thread "
            "environment variables to equal 1 before process startup"
        )

    batch_model = _build_model(args.elements)
    scalar_model = _build_model(args.elements)
    displacement = 2.0e-5 * np.sin(
        np.arange(batch_model.mesh.dof_manager.total_dofs, dtype=float) + 0.375
    )
    recovery = RecoveryConfig()
    resources = ResourceConfig(recovery_threads=1)

    original_get = s3_batch.get_reference_s3_stiffness_components
    original_prepare = s3_batch.prepare_reference_s3_components

    def scalar_prepare(model, items, **_kwargs):
        return (
            original_prepare(
                model,
                items,
                minimum_group_size=args.elements + 1,
                allow_exact_element_cache_reuse=False,
            ),
            False,
        )

    def batch_stiffness():
        s3_batch.get_reference_s3_stiffness_components = original_get
        return assemble_stiffness_matrix(batch_model)

    def scalar_stiffness():
        s3_batch.get_reference_s3_stiffness_components = scalar_prepare
        try:
            return assemble_stiffness_matrix(scalar_model)
        finally:
            s3_batch.get_reference_s3_stiffness_components = original_get

    # Freeze distinct recovery plans: one admitted batch, one public scalar
    # fallback.  Restore module policy immediately after scalar plan creation.
    batch_recovery = lambda: recover_element_stresses_with_report(
        batch_model,
        displacement,
        recovery,
        return_global=bool(args.return_global),
        resource_config=resources,
    )
    original_recovery_minimum = recovery_batches.MIN_REFERENCE_S3_RECOVERY_GROUP

    # One warm-up per measured route, outside all samples.
    batch_stiffness()
    scalar_stiffness()
    batch_recovery()
    recovery_batches.MIN_REFERENCE_S3_RECOVERY_GROUP = args.elements + 1
    clear_recovery_batch_plan(scalar_model)
    scalar_recovery = lambda: recover_element_stresses_with_report(
        scalar_model,
        displacement,
        recovery,
        return_global=bool(args.return_global),
        resource_config=resources,
    )
    scalar_recovery()
    recovery_batches.MIN_REFERENCE_S3_RECOVERY_GROUP = original_recovery_minimum

    stiffness_times = {"batch": [], "scalar": []}
    recovery_times = {"batch": [], "scalar": []}
    last_stiffness: Dict[str, Any] = {}
    last_recovery: Dict[str, Any] = {}
    try:
        for repetition in range(int(args.repeats)):
            stiffness_order = (
                (("batch", batch_stiffness), ("scalar", scalar_stiffness))
                if repetition % 2 == 0
                else (("scalar", scalar_stiffness), ("batch", batch_stiffness))
            )
            recovery_order = (
                (("scalar", scalar_recovery), ("batch", batch_recovery))
                if repetition % 2 == 0
                else (("batch", batch_recovery), ("scalar", scalar_recovery))
            )
            for name, call in stiffness_order:
                seconds, result = _time(call)
                stiffness_times[name].append(seconds)
                last_stiffness[name] = result
            for name, call in recovery_order:
                seconds, result = _time(call)
                recovery_times[name].append(seconds)
                last_recovery[name] = result
    finally:
        s3_batch.get_reference_s3_stiffness_components = original_get
        recovery_batches.MIN_REFERENCE_S3_RECOVERY_GROUP = original_recovery_minimum

    stiffness_batch = _summary(stiffness_times["batch"])
    stiffness_scalar = _summary(stiffness_times["scalar"])
    recovery_batch = _summary(recovery_times["batch"])
    recovery_scalar = _summary(recovery_times["scalar"])
    stiffness_error = _sparse_error(
        last_stiffness["scalar"][0],
        last_stiffness["batch"][0],
    )
    recovery_error = _recovery_error(
        last_recovery["scalar"][0],
        last_recovery["batch"][0],
    )
    batch_stiffness_diagnostics = last_stiffness["batch"][1]["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    scalar_stiffness_diagnostics = last_stiffness["scalar"][1]["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    batch_recovery_diagnostics = last_recovery["batch"][1].metadata
    scalar_recovery_diagnostics = last_recovery["scalar"][1].metadata
    payload = {
        "schema": "anysolver.qualified_s3_reference_batch_benchmark.v1",
        "status": "completed",
        "elements": int(args.elements),
        "warmups_per_route": 1,
        "repeats": int(args.repeats),
        "return_global": bool(args.return_global),
        "one_numerical_thread": (
            THREAD_ENVIRONMENT_AT_IMPORT == THREAD_ENVIRONMENT
            and thread_environment == THREAD_ENVIRONMENT
        ),
        "stiffness": {
            "batch": stiffness_batch,
            "scalar": stiffness_scalar,
            "median_speedup": (
                stiffness_scalar["median_seconds"]
                / stiffness_batch["median_seconds"]
            ),
            "equality": stiffness_error,
            "batch_element_count": batch_stiffness_diagnostics["element_count"],
            "batch_group_count": batch_stiffness_diagnostics[
                "exact_translation_group_count"
            ],
            "scalar_fallback_element_count": len(
                scalar_stiffness_diagnostics["fallback_reasons"].get(
                    "group_below_minimum_size", ()
                )
            ),
        },
        "recovery": {
            "batch": recovery_batch,
            "scalar": recovery_scalar,
            "median_speedup": (
                recovery_scalar["median_seconds"]
                / recovery_batch["median_seconds"]
            ),
            "equality": recovery_error,
            "batch_element_count": batch_recovery_diagnostics[
                "eligible_element_count"
            ],
            "batch_group_count": batch_recovery_diagnostics[
                "qualified_s3_reference_batch"
            ]["exact_translation_group_count"],
            "scalar_fallback_element_count": scalar_recovery_diagnostics[
                "fallback_element_count"
            ],
            "scalar_batch_count": scalar_recovery_diagnostics.get(
                "qualified_s3_reference_batch_count", 0
            ),
        },
        "speedup_claim_permitted": bool(
            THREAD_ENVIRONMENT_AT_IMPORT == THREAD_ENVIRONMENT
            and thread_environment == THREAD_ENVIRONMENT
            and stiffness_error["maximum_scaled_error"] <= 1.0e-12
            and recovery_error["maximum_scaled_error"] <= 1.0e-12
            and stiffness_scalar["median_seconds"]
            / stiffness_batch["median_seconds"]
            >= 1.5
            and recovery_scalar["median_seconds"]
            / recovery_batch["median_seconds"]
            >= 1.5
        ),
    }
    if args.include_q4_comparator:
        q4_model = _build_q4_model(args.elements)
        # Same one-warm-up plus N measured warm calls as the S3 stiffness
        # route, on the same process and numerical-thread policy.
        assemble_stiffness_matrix(q4_model)
        q4_times = []
        q4_result = None
        for _repetition in range(int(args.repeats)):
            seconds, q4_result = _time(
                lambda: assemble_stiffness_matrix(q4_model)
            )
            q4_times.append(seconds)
        assert q4_result is not None
        q4_summary = _summary(q4_times)
        s3_per_element = (
            stiffness_batch["median_seconds"] / float(args.elements)
        )
        q4_per_element = q4_summary["median_seconds"] / float(args.elements)
        payload["qualified_q4_comparator"] = {
            "warmups": 1,
            "repeats": int(args.repeats),
            "batch": q4_summary,
            "s3_median_seconds_per_element": s3_per_element,
            "q4_median_seconds_per_element": q4_per_element,
            "s3_over_q4_per_element_ratio": s3_per_element / q4_per_element,
            "s3_no_slower_than_q4": bool(s3_per_element <= q4_per_element),
            "q4_element_count": q4_result[1]["diagnostics"][
                "qualified_e4_pl_stiffness"
            ]["element_count"],
            "q4_path": q4_result[1]["diagnostics"][
                "qualified_e4_pl_stiffness"
            ]["path"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
