"""Benchmark and qualify incremental impact-damage K/M matrix updates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from typing import Mapping

import numpy as np

from anyfesolver.contact import _assemble_damaged_linear_matrices, _linear_element_matrix_terms
from anyfesolver.damage_matrix_performance import DamageMatrixPlan
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel


def _panel(nx: int, ny: int) -> FEModel:
    model = FEModel("damage_matrix_benchmark")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    node_ids: dict[tuple[int, int], int] = {}
    node_id = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            model.add_node(node_id, float(i) / nx, float(j) / ny, 0.0)
            node_ids[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(ny):
        for i in range(nx):
            model.add_element(
                element_id,
                create_element(
                    "shell",
                    element_id,
                    [
                        node_ids[(i, j)],
                        node_ids[(i + 1, j)],
                        node_ids[(i + 1, j + 1)],
                        node_ids[(i, j + 1)],
                    ],
                    "steel",
                    thickness=0.012,
                ),
            )
            element_id += 1
    model.add_point_mass(node_ids[(nx // 2, ny // 2)], 125.0)
    return model


def _damage_events(element_ids: tuple[int, ...], count: int) -> tuple[dict[int, float], ...]:
    active: dict[int, float] = {}
    events = []
    changed_per_event = max(1, min(8, len(element_ids) // max(count, 1)))
    for event_index in range(count):
        for offset in range(changed_per_event):
            slot = (event_index * 17 + offset * 29) % len(element_ids)
            element_id = element_ids[slot]
            active[element_id] = max(0.05, active.get(element_id, 1.0) - 0.08)
        events.append(dict(active))
    return tuple(events)


def _elapsed(function) -> float:
    gc.collect()
    start = time.perf_counter()
    function()
    return float(time.perf_counter() - start)


def _matrix_error(candidate, reference) -> tuple[float, float]:
    delta = (candidate - reference).tocsr()
    absolute = float(np.max(np.abs(delta.data), initial=0.0))
    scale = max(float(np.max(np.abs(reference.data), initial=0.0)), 1.0)
    return absolute, absolute / scale


def _legacy_sequence(model: FEModel, terms, events: tuple[Mapping[int, float], ...]):
    matrices = None
    for scales in events:
        matrices = _assemble_damaged_linear_matrices(model, scales, cached_terms=terms)
    assert matrices is not None
    return matrices


def _incremental_sequence(model: FEModel, plan: DamageMatrixPlan, events: tuple[Mapping[int, float], ...]):
    matrices = None
    for scales in events:
        matrices = plan.update(model, scales)
    assert matrices is not None
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-steady-speedup", type=float, default=2.0)
    parser.add_argument("--min-amortized-speedup", type=float, default=1.1)
    parser.add_argument("--require-qualified", action="store_true")
    args = parser.parse_args()
    if args.nx <= 0 or args.ny <= 0:
        parser.error("--nx and --ny must be positive")
    if args.events <= 0 or args.repeats <= 0:
        parser.error("--events and --repeats must be positive")

    model = _panel(args.nx, args.ny)
    terms = _linear_element_matrix_terms(model)
    element_ids = tuple(term[0] for term in terms[1])
    events = _damage_events(element_ids, args.events)

    validation_plan = DamageMatrixPlan.build(model, terms)
    candidate_k, candidate_m = _incremental_sequence(model, validation_plan, events)
    reference_k, reference_m = _legacy_sequence(model, terms, events)
    k_absolute, k_relative = _matrix_error(candidate_k, reference_k)
    m_absolute, m_relative = _matrix_error(candidate_m, reference_m)

    plan = DamageMatrixPlan.build(model, terms)
    legacy_times: list[float] = []
    incremental_times: list[float] = []
    for repeat in range(args.repeats):
        runners = ("legacy", "incremental") if repeat % 2 == 0 else ("incremental", "legacy")
        for runner in runners:
            if runner == "legacy":
                legacy_times.append(_elapsed(lambda: _legacy_sequence(model, terms, events)))
            else:
                plan.update(model, {})
                incremental_times.append(_elapsed(lambda: _incremental_sequence(model, plan, events)))

    legacy_median = float(statistics.median(legacy_times))
    incremental_median = float(statistics.median(incremental_times))
    setup_seconds = float(plan.setup_seconds)
    steady_speedup = legacy_median / max(incremental_median, 1.0e-30)
    amortized_speedup = legacy_median / max(incremental_median + setup_seconds, 1.0e-30)
    savings_per_event = (legacy_median - incremental_median) / args.events
    break_even_events = (
        math.ceil(setup_seconds / savings_per_event)
        if savings_per_event > 0.0
        else None
    )
    qualified = bool(
        steady_speedup >= float(args.min_steady_speedup)
        and amortized_speedup >= float(args.min_amortized_speedup)
        and k_relative <= 1.0e-12
        and m_relative <= 1.0e-12
    )
    payload = {
        "benchmark": "incremental_impact_damage_matrix_updates",
        "model": {
            "nx": int(args.nx),
            "ny": int(args.ny),
            "element_count": len(element_ids),
            "total_dofs": int(terms[0]),
            "event_count": int(args.events),
        },
        "elapsed_median_s": {
            "legacy_cached_terms_full_coo_rebuild": legacy_median,
            "incremental_csr_updates": incremental_median,
            "incremental_setup": setup_seconds,
        },
        "steady_state_speedup": float(steady_speedup),
        "setup_amortized_speedup": float(amortized_speedup),
        "break_even_event_count": break_even_events,
        "matrix_error": {
            "stiffness_max_absolute": k_absolute,
            "stiffness_relative": k_relative,
            "mass_max_absolute": m_absolute,
            "mass_relative": m_relative,
        },
        "plan_diagnostics": plan.diagnostics(),
        "promotion_gate": {
            "minimum_steady_speedup": float(args.min_steady_speedup),
            "minimum_setup_amortized_speedup": float(args.min_amortized_speedup),
            "maximum_relative_matrix_error": 1.0e-12,
            "qualified": qualified,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_qualified and not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
