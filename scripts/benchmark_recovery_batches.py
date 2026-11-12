"""Benchmark compiled isotropic S4 recovery against the scalar element oracle."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable, Dict, Mapping

import numpy as np

from anyfesolver import RecoveryConfig, ResourceConfig, generate_simple_panel_mesh
from anyfesolver.recovery import (
    _compute_one_element_stress,
    recover_element_stresses_with_report,
)


def _timings(call: Callable[[], object], repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        values.append(float(time.perf_counter() - start))
    return values


def _maximum_error(
    reference: Mapping[int, Mapping[str, object]],
    candidate: Mapping[int, Mapping[str, object]],
) -> float:
    maximum = 0.0
    for element_id, expected_fields in reference.items():
        actual_fields = candidate[element_id]
        for field, expected in expected_fields.items():
            if isinstance(expected, str):
                if actual_fields[field] != expected:
                    return float("inf")
                continue
            actual = np.asarray(actual_fields[field], dtype=float)
            expected_array = np.asarray(expected, dtype=float)
            maximum = max(
                maximum,
                float(np.max(np.abs(actual - expected_array), initial=0.0)),
            )
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--return-global", action="store_true")
    args = parser.parse_args()
    if min(args.nx, args.ny, args.repeats, args.threads) <= 0:
        parser.error("nx, ny, repeats, and threads must be positive")

    model = generate_simple_panel_mesh(
        float(args.nx),
        float(args.ny),
        0.01,
        num_divisions_x=args.nx,
        num_divisions_y=args.ny,
    )
    displacements = np.linspace(
        0.0,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
    )
    recovery = RecoveryConfig(components=None)
    resources = ResourceConfig(recovery_threads=args.threads)

    def scalar() -> Dict[int, Mapping[str, object]]:
        result: Dict[int, Mapping[str, object]] = {}
        for element_id in model.mesh.elements:
            item = _compute_one_element_stress(
                model,
                displacements,
                element_id,
                return_global=args.return_global,
            )
            if item is not None:
                result[item[0]] = item[1]
        return result

    def compiled():
        return recover_element_stresses_with_report(
            model,
            displacements,
            recovery,
            return_global=args.return_global,
            resource_config=resources,
        )

    # Warm JIT compilation and the bounded plan outside measured samples.
    compiled()
    scalar_reference = scalar()
    compiled_reference, report = compiled()
    scalar_seconds = _timings(scalar, args.repeats)
    compiled_seconds = _timings(compiled, args.repeats)
    scalar_median = float(statistics.median(scalar_seconds))
    compiled_median = float(statistics.median(compiled_seconds))
    payload = {
        "status": "completed",
        "elements": len(model.mesh.elements),
        "return_global": bool(args.return_global),
        "threads": int(args.threads),
        "repeats": int(args.repeats),
        "scalar_seconds": scalar_seconds,
        "compiled_seconds": compiled_seconds,
        "scalar_median_seconds": scalar_median,
        "compiled_median_seconds": compiled_median,
        "speedup": scalar_median / max(compiled_median, np.finfo(float).tiny),
        "maximum_absolute_error": _maximum_error(
            scalar_reference,
            compiled_reference,
        ),
        "diagnostics": report.to_dict()["metadata"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
