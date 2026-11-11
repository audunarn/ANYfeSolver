"""Benchmark compact contact trials against full public record assembly."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from typing import Any, Dict

import numpy as np

from anyfesolver.contact import (
    RigidSphereImpact,
    SphereContactConfig,
    SphereContactRecord,
    _assemble_sphere_contact_work_buffer,
    _contact_geometry,
    _two_shell_contact_verification_panel,
    _verification_contact_panel,
    assemble_sphere_contact_load_vector,
)
from anyfesolver.contact_performance import ContactWorkBuffer, ContactWorkCounters
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel


def _case_inputs(name: str):
    if name == "face":
        model = _verification_contact_panel()
        position = np.array([0.5, 0.5, 0.1])
        preferred = ()
    elif name == "shared_edge_sticky":
        model = _two_shell_contact_verification_panel()
        position = np.array([1.0, 0.5, 0.1])
        preferred = (2,)
        radius = 0.2
    elif name == "multi_candidate_patch":
        model = FEModel("multi_candidate_contact_patch")
        model.add_material("soft", 1.0e5, 0.3, density=20.0)
        division = 5
        node_of = {}
        node_id = 1
        for j in range(division + 1):
            for i in range(division + 1):
                model.add_node(node_id, i / division, j / division, 0.0)
                node_of[(i, j)] = node_id
                node_id += 1
        element_id = 1
        for j in range(division):
            for i in range(division):
                model.add_element(
                    element_id,
                    create_element(
                        "shell",
                        element_id,
                        [
                            node_of[(i, j)],
                            node_of[(i + 1, j)],
                            node_of[(i + 1, j + 1)],
                            node_of[(i, j + 1)],
                        ],
                        "soft",
                        thickness=0.05,
                    ),
                )
                element_id += 1
        position = np.array([0.5, 0.5, 0.3])
        preferred = (13,)
        radius = 0.7
    else:
        raise ValueError(name)
    if name == "face":
        radius = 0.2
    sphere = RigidSphereImpact(
        name,
        radius=radius,
        mass=1.0,
        start_point=position,
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )
    config = SphereContactConfig(penalty_stiffness=1000.0, max_active_contacts=1)
    return model, sphere, config, position, np.zeros(3), preferred


def _benchmark_case(
    name: str,
    *,
    iterations: int,
    batches: int,
    materialize_every: int,
) -> Dict[str, Any]:
    model, sphere, config, position, velocity, preferred = _case_inputs(name)
    geometry = _contact_geometry(model)
    case_iterations = max(int(iterations) // 10, 50) if name == "multi_candidate_patch" else int(iterations)

    public_load, public_force, public_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        position,
        velocity,
        preferred_element_ids=preferred,
    )
    verification_buffer = ContactWorkBuffer(model.mesh.dof_manager.total_dofs)
    compact = _assemble_sphere_contact_work_buffer(
        model,
        sphere,
        config,
        position,
        velocity,
        preferred_element_ids=preferred,
        work_buffer=verification_buffer,
    )
    compact_records = compact.materialize_records(SphereContactRecord, geometry.node_ids)
    if not np.array_equal(compact.load, public_load):
        raise AssertionError(f"{name}: compact load differs from public oracle")
    if not np.array_equal(compact.sphere_force, public_force):
        raise AssertionError(f"{name}: compact sphere force differs from public oracle")
    if tuple(record.to_dict() for record in compact_records) != tuple(
        record.to_dict() for record in public_records
    ):
        raise AssertionError(f"{name}: compact records differ from public oracle")

    # Warm geometry caches and allocation paths before recording.
    for _ in range(10):
        assemble_sphere_contact_load_vector(
            model,
            sphere,
            config,
            position,
            velocity,
            preferred_element_ids=preferred,
        )
        _assemble_sphere_contact_work_buffer(
            model,
            sphere,
            config,
            position,
            velocity,
            preferred_element_ids=preferred,
            work_buffer=verification_buffer,
        )

    public_times = []
    eager_times = []
    compact_times = []
    eager_counters = ContactWorkCounters()
    eager_work = ContactWorkBuffer(model.mesh.dof_manager.total_dofs, counters=eager_counters)
    lazy_counters = ContactWorkCounters()
    lazy_work = ContactWorkBuffer(model.mesh.dof_manager.total_dofs, counters=lazy_counters)
    gc.disable()
    try:
        for batch in range(int(batches)):
            orders = (
                ("public", "compact_eager", "compact_lazy"),
                ("compact_eager", "compact_lazy", "public"),
                ("compact_lazy", "public", "compact_eager"),
            )
            order = orders[batch % len(orders)]
            for path in order:
                start = time.perf_counter()
                if path == "public":
                    for _ in range(case_iterations):
                        assemble_sphere_contact_load_vector(
                            model,
                            sphere,
                            config,
                            position,
                            velocity,
                            preferred_element_ids=preferred,
                        )
                    public_times.append(time.perf_counter() - start)
                elif path == "compact_eager":
                    for _ in range(case_iterations):
                        result = _assemble_sphere_contact_work_buffer(
                            model,
                            sphere,
                            config,
                            position,
                            velocity,
                            preferred_element_ids=preferred,
                            work_buffer=eager_work,
                        )
                        result.materialize_records(SphereContactRecord, geometry.node_ids)
                    eager_times.append(time.perf_counter() - start)
                else:
                    for iteration in range(case_iterations):
                        result = _assemble_sphere_contact_work_buffer(
                            model,
                            sphere,
                            config,
                            position,
                            velocity,
                            preferred_element_ids=preferred,
                            work_buffer=lazy_work,
                        )
                        if (iteration + 1) % int(materialize_every) == 0:
                            result.materialize_records(SphereContactRecord, geometry.node_ids)
                    compact_times.append(time.perf_counter() - start)
    finally:
        gc.enable()

    public_median = float(statistics.median(public_times))
    eager_median = float(statistics.median(eager_times))
    compact_median = float(statistics.median(compact_times))
    return {
        "name": name,
        "iterations_per_batch": int(case_iterations),
        "batches": int(batches),
        "materialize_every": int(materialize_every),
        "public_median_s": public_median,
        "compact_eager_median_s": eager_median,
        "compact_median_s": compact_median,
        "speedup": public_median / max(compact_median, 1.0e-30),
        "lazy_materialization_speedup": eager_median / max(compact_median, 1.0e-30),
        "selected_element_ids": list(lazy_work.active_element_ids),
        "exact_load_parity": True,
        "exact_force_parity": True,
        "exact_record_parity": True,
        "compact_eager_counters": eager_counters.diagnostics(),
        "compact_lazy_counters": lazy_counters.diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--materialize-every", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batches <= 0 or args.materialize_every <= 0:
        parser.error("iterations, batches, and materialize-every must be positive")
    cases = [
        _benchmark_case(
            name,
            iterations=args.iterations,
            batches=args.batches,
            materialize_every=args.materialize_every,
        )
        for name in ("face", "shared_edge_sticky", "multi_candidate_patch")
    ]
    payload = {
        "benchmark": "contact_work_buffer",
        "cases": cases,
        "minimum_speedup": min(case["speedup"] for case in cases),
        "median_speedup": float(statistics.median(case["speedup"] for case in cases)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
