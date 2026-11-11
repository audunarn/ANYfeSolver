"""Local FE solver infrastructure benchmark helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import tracemalloc
import importlib.metadata
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy
from scipy import sparse

from .assembly import solve_linear, solve_linear_many
from .boundary import BoundaryCondition, FixedSupport, LoadCase
from .buckling import solve_eigenvalue_buckling
from .dynamics import TransientConfig, solve_transient_newmark
from .contact import RigidSphereImpact, SphereContactConfig, solve_transient_sphere_impact
from .elements import BeamElement
from .fe_core import FEModel
from .fracture import FractureConfig, ImpactDamageConfig, detect_new_deletions
from .matrix_assembly import assemble_load_vector, assemble_mass_matrix, assemble_stiffness_matrix
from .mesh_gen import generate_beam_mesh, generate_simple_panel_mesh
from .recovery import RecoveryConfig, ResourceConfig, recover_element_stresses_with_report
from .linalg import FactorizationCache, MatrixClass, factorize_cached


DEFAULT_BENCHMARK_PATH = Path("reports/benchmarks/fe_infrastructure_benchmarks.json")


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _optional_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _measure(func: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        payload = func()
    finally:
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    payload.setdefault("timing", {})
    payload["timing"]["wall_seconds"] = float(elapsed)
    payload["memory"] = {"current_bytes": int(current), "peak_bytes": int(peak)}
    return payload


def _topology(model: FEModel) -> Dict[str, int]:
    return {
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "dofs": int(model.mesh.dof_manager.total_dofs),
    }


def _static_beam_case() -> Dict[str, Any]:
    model = generate_beam_mesh(2.0, num_divisions=8, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    load_case = LoadCase("tip_load")
    load_case.add_nodal_load(9, [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0])
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    F, load_info = assemble_load_vector(model, load_case)
    u, solver_info = solve_linear(model, load_case)
    backend = (solver_info.get("convergence_info") or {}).get("backend", {})
    return {
        "topology": _topology(model),
        "matrix_nnz": {"K": int(K.nnz), "M": int(M.nnz)},
        "load_norm": float(np.linalg.norm(F)),
        "timing": {
            "stiffness_assembly_seconds": float(stiffness_info.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass_info.get("assembly_time", 0.0)),
            "load_assembly_seconds": float(load_info.get("assembly_time", 0.0)),
            "solve_seconds": float(solver_info.get("solve_time", 0.0)),
            "factorization_seconds": float(backend.get("factorization_time", 0.0)) if isinstance(backend, dict) else 0.0,
            "backend_solve_seconds": float(backend.get("solve_time", 0.0)) if isinstance(backend, dict) else 0.0,
        },
        "results": {"max_abs_displacement": float(np.max(np.abs(u)))},
        "backend": backend,
        "status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
    }


def _multi_rhs_case() -> Dict[str, Any]:
    model = generate_beam_mesh(2.0, num_divisions=8, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    tip = 9
    load_x = LoadCase("tip_x")
    load_x.add_nodal_load(tip, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_z = LoadCase("tip_z")
    load_z.add_nodal_load(tip, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
    load_m = LoadCase("tip_m")
    load_m.add_nodal_load(tip, [0.0, 0.0, 0.0, 0.0, 50.0, 0.0])
    U, info = solve_linear_many(model, [load_x, load_z, load_m])
    backend = info.get("backend", {})
    return {
        "topology": _topology(model),
        "num_rhs": 3,
        "timing": {
            "solve_many_seconds": float(info.get("solve_time", 0.0)),
            "factorization_seconds": float(backend.get("factorization_time", 0.0)) if isinstance(backend, dict) else 0.0,
            "backend_solve_seconds": float(backend.get("solve_time", 0.0)) if isinstance(backend, dict) else 0.0,
        },
        "results": {"solution_matrix_norm": float(np.linalg.norm(U))},
        "backend": backend,
        "status": str(info.get("status", "unknown")),
    }


def _set_reduced_integration_for_q8r(model: FEModel) -> None:
    for element in model.mesh.elements.values():
        if getattr(element, "_is_8node", False):
            element.reduced_integration = True


def _shell_stiffness_cold_warm_case(shell_order: str) -> Dict[str, Any]:
    from .jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED, jit_diagnostics

    order = str(shell_order).upper()
    use_8node = order in {"Q8", "Q8R", "S8", "S8R"}
    model = generate_simple_panel_mesh(1.5, 1.0, 0.01, num_divisions_x=2, num_divisions_y=2, use_8node_elements=use_8node)
    normalized_order = "Q8R" if order in {"Q8R", "S8R"} else ("Q8" if use_8node else "S4")
    if normalized_order == "Q8R":
        _set_reduced_integration_for_q8r(model)

    cold_start = time.perf_counter()
    K_cold, cold_info = assemble_stiffness_matrix(model)
    cold_seconds = time.perf_counter() - cold_start
    warm_start = time.perf_counter()
    K_warm, warm_info = assemble_stiffness_matrix(model)
    warm_seconds = time.perf_counter() - warm_start

    denominator = max(float(sparse.linalg.norm(K_cold)), 1.0)
    matrix_difference_norm = float(sparse.linalg.norm(K_cold - K_warm) / denominator)
    jit = jit_diagnostics()
    return {
        "topology": _topology(model),
        "shell_order": normalized_order,
        "element_count": int(model.mesh.num_elements),
        "jit_enabled": bool(JIT_ENABLED),
        "jit_disabled_reason": JIT_DISABLED_REASON,
        "parallel_threads": jit.get("num_threads"),
        "timing": {
            "cold_assembly_seconds": float(cold_seconds),
            "warm_assembly_seconds": float(warm_seconds),
            "warm_speedup": float(cold_seconds / warm_seconds) if warm_seconds > 0.0 else 0.0,
        },
        "results": {
            "matrix_difference_norm": matrix_difference_norm,
            "cold_diagnostics": cold_info.get("diagnostics", {}),
            "warm_diagnostics": warm_info.get("diagnostics", {}),
        },
        "status": "completed" if matrix_difference_norm < 1.0e-12 else "failed",
    }


def _shell_assembly_case() -> Dict[str, Any]:
    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=8, num_divisions_y=4)
    load_case = LoadCase("pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 1000.0)
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    _F, load_info = assemble_load_vector(model, load_case)
    return {
        "topology": _topology(model),
        "matrix_nnz": {"K": int(K.nnz), "M": int(M.nnz)},
        "timing": {
            "stiffness_assembly_seconds": float(stiffness_info.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass_info.get("assembly_time", 0.0)),
            "load_assembly_seconds": float(load_info.get("assembly_time", 0.0)),
        },
        "diagnostics": {"stiffness_symmetry": stiffness_info.get("diagnostics", {}).get("assembled_symmetry_error")},
        "status": "completed",
    }


def _buckling_case() -> Dict[str, Any]:
    model = FEModel("benchmark_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(9):
        model.add_node(i + 1, 4.0 * i / 8, 0.0, 0.0)
    for i in range(8):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(range(1, 10))
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pins", [1, 9], {"uy": 0.0}))
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    result = solve_eigenvalue_buckling(model, states, num_modes=2)
    return {
        "topology": _topology(model),
        "timing": {
            "stiffness_assembly_seconds": float(result.assembly_info.get("stiffness", {}).get("assembly_time", 0.0)),
            "geometric_assembly_seconds": float(result.assembly_info.get("geometric_stiffness", {}).get("assembly_time", 0.0)),
        },
        "results": {"critical_load_factor": float(result.critical_load_factor or 0.0), "num_modes": int(result.num_modes_returned)},
        "status": result.solver_status,
    }


def _transient_case() -> Dict[str, Any]:
    model = FEModel("benchmark_sdof")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load_case = LoadCase("step")
    load_case.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_transient_newmark(model, TransientConfig(dt=0.001, t_end=0.05), base_load_case=load_case)
    return {
        "topology": _topology(model),
        "timing": {
            "factorization_count": int(result.diagnostics.get("factorization_count", 0)),
            "solve_count": int(result.diagnostics.get("solve_count", 0)),
        },
        "results": {"peak_displacement": float(result.peak_displacement), "energy_drift": float(result.diagnostics.get("max_relative_energy_drift", 0.0))},
        "status": result.status,
    }


def _selective_recovery_case() -> Dict[str, Any]:
    model = generate_simple_panel_mesh(3.0, 2.0, 0.01, num_divisions_x=6, num_divisions_y=4)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    recovery = RecoveryConfig(components=["von_mises"])
    serial, serial_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    threaded, threaded_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=2),
    )
    results_match = sorted(serial) == sorted(threaded) and all(
        np.allclose(serial[element_id]["von_mises"], threaded[element_id]["von_mises"])
        for element_id in serial
    )
    return {
        "topology": _topology(model),
        "timing": {
            "serial_recovery_seconds": float(serial_report.elapsed_seconds),
            "threaded_recovery_seconds": float(threaded_report.elapsed_seconds),
            "observed_speedup": (
                float(serial_report.elapsed_seconds / threaded_report.elapsed_seconds)
                if threaded_report.elapsed_seconds > 0.0
                else 0.0
            ),
        },
        "resources": {
            "serial": serial_report.to_dict(),
            "threaded": threaded_report.to_dict(),
        },
        "results": {"num_stress_results": len(serial), "results_match": bool(results_match)},
        "status": "completed" if results_match else "failed",
    }


def _factorization_reuse_case() -> Dict[str, Any]:
    matrix = sparse.diags([1.0, 4.0, 1.0], offsets=[-1, 0, 1], shape=(200, 200), format="csr")
    rhs = np.ones(200, dtype=float)
    cache = FactorizationCache(name="benchmark_factorization_reuse", max_entries=2)
    first = factorize_cached(matrix, MatrixClass.SPD, cache=cache)
    first.solve(rhs)
    second = factorize_cached(matrix.copy(), MatrixClass.SPD, cache=cache)
    second.solve(rhs)
    changed = factorize_cached(matrix + sparse.eye(200, format="csr") * 0.01, MatrixClass.SPD, cache=cache)
    changed.solve(rhs)
    return {
        "topology": {"dofs": 200, "nnz": int(matrix.nnz)},
        "timing": {
            "first_factorization_seconds": float(first.factorization_time),
            "reused_solve_seconds_total": float(second.solve_time),
            "changed_factorization_seconds": float(changed.factorization_time),
        },
        "cache": cache.diagnostics(),
        "results": {"same_handle_reused": bool(first is second), "changed_matrix_new_handle": bool(changed is not first)},
        "status": "completed" if first is second and changed is not first else "failed",
    }


def _nonlinear_assembly_case() -> Dict[str, Any]:
    from . import nonlinear_performance, nonlinear_static
    from .nonlinear_performance_bootstrap import get_nonlinear_assembly_plan, nonlinear_performance_status

    model = generate_simple_panel_mesh(1.2, 0.8, 0.01, num_divisions_x=2, num_divisions_y=2)
    rng = np.random.default_rng(20260630)
    displacement = rng.normal(scale=2.0e-5, size=model.mesh.dof_manager.total_dofs)
    committed: Dict[int, Any] = {}
    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    if original is None:
        return {
            "topology": _topology(model),
            "timing": {},
            "results": {},
            "performance_layer": nonlinear_performance_status(),
            "status": "skipped",
            "reason": "legacy nonlinear assembler is unavailable",
        }

    start = time.perf_counter()
    force_reference, tangent_reference, _states_reference = original(
        model,
        displacement,
        committed,
        5,
        tangent=True,
    )
    legacy_seconds = time.perf_counter() - start

    plan = get_nonlinear_assembly_plan(model, 5)
    start = time.perf_counter()
    force_fast, tangent_fast, _states_fast = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        committed,
        5,
        tangent=True,
    )
    fast_seconds = time.perf_counter() - start

    force_error = float(
        np.linalg.norm(force_fast - force_reference) / max(float(np.linalg.norm(force_reference)), 1.0)
    )
    tangent_error = float(
        sparse.linalg.norm(tangent_fast - tangent_reference) / max(float(sparse.linalg.norm(tangent_reference)), 1.0)
    )
    return {
        "topology": _topology(model),
        "timing": {
            "legacy_assembly_seconds": float(legacy_seconds),
            "active_fast_path_seconds": float(fast_seconds),
            "observed_speedup": float(legacy_seconds / fast_seconds) if fast_seconds > 0.0 else 0.0,
        },
        "results": {
            "relative_force_error": force_error,
            "relative_tangent_error": tangent_error,
            "plan_diagnostics": plan.diagnostics(),
        },
        "performance_layer": nonlinear_performance_status(),
        "status": "completed" if force_error < 1.0e-10 and tangent_error < 1.0e-9 else "failed",
    }


def _fracture_damage_case() -> Dict[str, Any]:
    static_model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=8, num_divisions_y=4)
    states: Dict[int, Any] = {}
    element_ids = list(static_model.mesh.elements)
    for element_id in element_ids:
        states[int(element_id)] = {"alpha": np.zeros((4, 3), dtype=float)}
    for element_id in element_ids[: max(1, len(element_ids) // 16)]:
        states[int(element_id)]["alpha"][0, 0] = 0.02
    fracture_config = FractureConfig(threshold=0.01, max_deleted_fraction=1.0)
    scan_start = time.perf_counter()
    deletion_records, max_utilization = detect_new_deletions(
        static_model,
        states,
        fracture_config,
        (),
        step_index=1,
        load_factor=1.0,
    )
    scan_seconds = time.perf_counter() - scan_start

    impact_model = FEModel("benchmark_impact_damage_panel")
    impact_model.add_material("soft", 1.0e5, 0.3, density=20.0)
    impact_model.add_node(1, 0.0, 0.0, 0.0)
    impact_model.add_node(2, 1.0, 0.0, 0.0)
    impact_model.add_node(3, 1.0, 1.0, 0.0)
    impact_model.add_node(4, 0.0, 1.0, 0.0)
    from .elements import create_shell_element

    impact_model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "soft", thickness=0.05))
    impact_model.add_boundary_condition(
        BoundaryCondition(
            "restrain_shell_nonimpact_modes",
            [1, 2, 3, 4],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    sphere = RigidSphereImpact(
        "benchmark_damage",
        radius=0.1,
        mass=1.0,
        start_point=(0.5, 0.5, 0.25),
        travel_direction=(0.0, 0.0, -1.0),
        speed=2.0,
    )
    impact_result = solve_transient_sphere_impact(
        impact_model,
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(
            capacity_basis="user",
            user_capacity=1.0e6,
            softening_start=0.9,
            delete_at=1.0,
            max_deleted_fraction=1.0,
        ),
    )
    damage_summary = impact_result.diagnostics.get("impact_damage_summary", {})
    return {
        "topology": {
            "static_fracture_elements": int(static_model.mesh.num_elements),
            "impact_damage_elements": int(impact_model.mesh.num_elements),
        },
        "timing": {
            "static_deletion_scan_seconds": float(scan_seconds),
            "impact_wall_solver_steps": int(impact_result.diagnostics.get("num_steps", 0)),
            "impact_solve_count": int(impact_result.diagnostics.get("solve_count", 0)),
            "impact_factorization_count": int(impact_result.diagnostics.get("factorization_count", 0)),
            "impact_eroded_matrix_rebuild_count": int(impact_result.diagnostics.get("eroded_matrix_rebuild_count", 0)),
            "impact_damage_state_update_count": int(impact_result.diagnostics.get("damage_state_update_count", 0)),
        },
        "results": {
            "static_deleted_records": len(deletion_records),
            "static_max_fracture_utilization": float(max_utilization),
            "impact_status": impact_result.status,
            "impact_max_damage": float(damage_summary.get("max_damage", 0.0)),
            "impact_deleted_count": int(damage_summary.get("deleted_count", 0)),
            "sub_softening_rebuilds_skipped": int(impact_result.diagnostics.get("eroded_matrix_rebuild_count", 0)) == 0,
        },
        "status": "completed"
        if len(deletion_records) > 0 and int(impact_result.diagnostics.get("eroded_matrix_rebuild_count", 0)) == 0
        else "failed",
    }


BENCHMARK_CASES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "static_beam": _static_beam_case,
    "multi_rhs_static": _multi_rhs_case,
    "shell_stiffness_S4_cold_warm": lambda: _shell_stiffness_cold_warm_case("S4"),
    "shell_stiffness_Q8_cold_warm": lambda: _shell_stiffness_cold_warm_case("Q8"),
    "shell_stiffness_Q8R_cold_warm": lambda: _shell_stiffness_cold_warm_case("Q8R"),
    "shell_assembly": _shell_assembly_case,
    "beam_column_buckling": _buckling_case,
    "transient_newmark": _transient_case,
    "selective_recovery": _selective_recovery_case,
    "factorization_reuse": _factorization_reuse_case,
    "nonlinear_assembly": _nonlinear_assembly_case,
    "fracture_damage": _fracture_damage_case,
}


def run_infrastructure_benchmarks() -> Dict[str, Any]:
    """Run local benchmark smoke cases and return a serializable report."""
    cases = {name: _measure(builder) for name, builder in BENCHMARK_CASES.items()}
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_sha(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pypardiso": _optional_version("pypardiso"),
        },
        "cases": cases,
        "known_limitations": [
            "Benchmarks are local smoke measurements; compare trends on the same machine and Python environment.",
            "tracemalloc reports Python allocation peaks, not full process resident memory.",
            "Threaded recovery speedups are informational; deterministic correctness is the gate.",
        ],
    }


def write_benchmark_report(path: Path | str = DEFAULT_BENCHMARK_PATH) -> Dict[str, Any]:
    report = run_infrastructure_benchmarks()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
