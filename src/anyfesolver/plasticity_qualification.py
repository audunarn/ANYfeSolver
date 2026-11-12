"""Plasticity and nonlinear tangent qualification metrics."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .elements import BeamElement, ShellElement, create_shell_element
from .fe_core import FEModel
from .material_curves import DNVC208MaterialCurve, FiberSectionPlasticityConfig, dnv_c208_steel_curve
from .plasticity import (
    plane_stress_elastic_matrix,
    plane_stress_numerical_tangent,
    plane_stress_return_map,
    plane_stress_tangent_method,
    plane_stress_tangent_diagnostics,
)

DEFAULT_PLASTICITY_QUALIFICATION_PATH = Path("reports/plasticity_qualification/plasticity_qualification_report.json")

E_STEEL = 210.0e9
NU_STEEL = 0.3

_P_MATRIX = np.array(
    [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]],
    dtype=float,
) / 3.0


def reference_plastic_curve() -> DNVC208MaterialCurve:
    """Nearly elastic-perfectly plastic reference curve used by local checks."""
    return DNVC208MaterialCurve(
        sigma_prop=354.0e6,
        sigma_yield=355.0e6,
        sigma_yield_2=355.5e6,
        eps_p_y1=0.004,
        eps_p_y2=0.1,
        K=400.0e6,
        n=0.2,
    )


def yield_function_residual(stress: np.ndarray, alpha: float, curve: DNVC208MaterialCurve) -> float:
    """Scaled plane-stress J2 yield residual."""
    sigma = np.asarray(stress, dtype=float).reshape(3)
    sy = float(curve.flow_stress(np.array([float(alpha)], dtype=float))[0])
    residual = 0.5 * float(sigma @ _P_MATRIX @ sigma) - sy**2 / 3.0
    return float(residual / max(sy**2, 1.0))


def _material_tangent_fd_error(
    strain: np.ndarray,
    curve: DNVC208MaterialCurve,
    plastic: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    step: float = 1.0e-7,
) -> Dict[str, Any]:
    strain = np.asarray(strain, dtype=float).reshape(1, 3)
    plastic = np.zeros_like(strain) if plastic is None else np.asarray(plastic, dtype=float).reshape(1, 3)
    alpha = np.zeros(1, dtype=float) if alpha is None else np.asarray(alpha, dtype=float).reshape(1)
    stress, tangent, plastic_new, alpha_new = plane_stress_return_map(
        strain, plastic, alpha, E_STEEL, NU_STEEL, curve
    )
    fd = plane_stress_numerical_tangent(
        strain,
        plastic,
        alpha,
        E_STEEL,
        NU_STEEL,
        curve,
        step=step,
    )[0]
    error = float(np.linalg.norm(tangent[0] - fd) / max(np.linalg.norm(fd), 1.0))
    symmetry_error = float(
        np.linalg.norm(tangent[0] - tangent[0].T)
        / max(np.linalg.norm(tangent[0]), 1.0)
    )
    return {
        "stress": stress[0].tolist(),
        "alpha": float(alpha_new[0]),
        "max_plastic_strain_component": float(np.max(np.abs(plastic_new))),
        "yield_residual": yield_function_residual(stress[0], float(alpha_new[0]), curve),
        "tangent_fd_relative_error": error,
        "tangent_symmetry_relative_error": symmetry_error,
        "tangent_status": "tight" if error < 1.0e-3 else "diagnostic_oracle_mismatch",
    }


def algorithmic_tangent_path_metrics() -> Dict[str, Any]:
    """Compare analytical and numerical tangents over representative states.

    The path set deliberately spans the elastic branch, first yield, both
    piecewise-linear and power-law hardening, elastic unloading from a
    committed plastic state, and a highly ill-conditioned (but finite)
    plane-stress elastic matrix.
    """
    curve = reference_plastic_curve()
    zero_plastic = np.zeros((1, 3), dtype=float)
    zero_alpha = np.zeros(1, dtype=float)

    cases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {
        "elastic": (
            np.array([[2.0e-4, -0.5e-4, 0.25e-4]], dtype=float),
            zero_plastic,
            zero_alpha,
            NU_STEEL,
        ),
        "yielding": (
            np.array([[2.0e-3, 0.0, 0.0]], dtype=float),
            zero_plastic,
            zero_alpha,
            NU_STEEL,
        ),
        "linear_hardening": (
            np.array([[6.0e-3, 1.0e-3, 0.7e-3]], dtype=float),
            zero_plastic,
            zero_alpha,
            NU_STEEL,
        ),
        "power_law_hardening": (
            np.array([[0.16, 0.025, 0.01]], dtype=float),
            zero_plastic,
            zero_alpha,
            NU_STEEL,
        ),
        "near_singular_plane_stress": (
            np.array([[2.0e-10, -2.0e-10, 2.5e-11]], dtype=float),
            zero_plastic,
            zero_alpha,
            -0.999999,
        ),
    }

    preload_strain = np.array([[8.0e-3, 1.0e-3, 0.5e-3]], dtype=float)
    _preload_stress, _preload_tangent, preload_plastic, preload_alpha = (
        plane_stress_return_map(
            preload_strain,
            zero_plastic,
            zero_alpha,
            E_STEEL,
            NU_STEEL,
            curve,
        )
    )
    cases["unloading"] = (
        preload_strain - np.array([[1.0e-3, 0.25e-3, 0.1e-3]], dtype=float),
        preload_plastic,
        preload_alpha,
        NU_STEEL,
    )

    results: Dict[str, Any] = {}
    for name, (strain, plastic, alpha, nu) in cases.items():
        diagnostics = plane_stress_tangent_diagnostics(
            strain,
            plastic,
            alpha,
            E_STEEL,
            nu,
            curve,
        )
        stress, tangent, plastic_new, alpha_new = plane_stress_return_map(
            strain,
            plastic,
            alpha,
            E_STEEL,
            nu,
            curve,
        )
        results[name] = {
            **diagnostics,
            "elastic_matrix_condition": float(
                np.linalg.cond(plane_stress_elastic_matrix(E_STEEL, nu))
            ),
            "stress_norm": float(np.linalg.norm(stress)),
            "tangent_norm": float(np.linalg.norm(tangent)),
            "alpha": float(alpha_new[0]),
            "alpha_increment": float(alpha_new[0] - alpha[0]),
            "plastic_strain_increment_norm": float(
                np.linalg.norm(plastic_new - plastic)
            ),
        }

    return {
        "method": "analytical_implicit_consistent",
        "oracle": "central_finite_difference_discrete_return_map",
        "cases": results,
        "max_relative_error": max(
            float(case["max_relative_error"]) for case in results.values()
        ),
        "max_symmetry_relative_error": max(
            float(case["max_symmetry_relative_error"])
            for case in results.values()
        ),
        "max_fallback_count": max(
            int(case["fallback_count"]) for case in results.values()
        ),
    }


def _best_elapsed(call: Any, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(max(int(repeats), 1)):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
    return float(best)


def algorithmic_tangent_performance_metrics(
    num_points: int = 512,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Benchmark analytical and numerical tangents on one yielding batch."""
    count = max(int(num_points), 1)
    curve = reference_plastic_curve()
    phase = np.linspace(0.0, 1.0, count, dtype=float)
    strain = np.column_stack(
        (
            0.004 + 0.004 * phase,
            0.0005 + 0.001 * phase,
            0.0002 + 0.0005 * phase,
        )
    )
    plastic = np.zeros_like(strain)
    alpha = np.zeros(count, dtype=float)

    def analytical_call() -> None:
        plane_stress_return_map(
            strain,
            plastic,
            alpha,
            E_STEEL,
            NU_STEEL,
            curve,
            tangent_method="analytical",
        )

    def numerical_call() -> None:
        plane_stress_return_map(
            strain,
            plastic,
            alpha,
            E_STEEL,
            NU_STEEL,
            curve,
            tangent_method="numerical",
        )

    # Remove JIT compilation and allocator warm-up from steady-state timing.
    analytical_call()
    numerical_call()
    analytical_seconds = _best_elapsed(analytical_call, repeats=repeats)
    numerical_seconds = _best_elapsed(numerical_call, repeats=repeats)
    return {
        "num_points": count,
        "repeats": max(int(repeats), 1),
        "analytical_seconds": analytical_seconds,
        "numerical_seconds": numerical_seconds,
        "speedup": numerical_seconds / max(analytical_seconds, 1.0e-15),
        "return_map_evaluations_per_update": {
            "analytical": 1,
            "numerical": 7,
        },
        "tangent_derivative_samples": {
            "analytical": 0,
            "numerical": 6,
        },
    }


def _flatten_numeric_state(value: Any) -> np.ndarray:
    parts = []
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            parts.append(_flatten_numeric_state(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            parts.append(_flatten_numeric_state(item))
    else:
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            array = np.zeros(0, dtype=float)
        if array.size:
            parts.append(array)
    nonempty = [part for part in parts if part.size]
    return np.concatenate(nonempty) if nonempty else np.zeros(0, dtype=float)


def _global_shell_newton_run(tangent_method: str) -> Dict[str, Any]:
    """Run one force-controlled plastic shell solve without API plumbing."""
    from .boundary import BoundaryCondition, LoadCase
    from .elements import create_shell_element
    from .matrix_assembly import assemble_load_vector
    from .nonlinear_static import _assemble_nonlinear_system, solve_static_nonlinear

    model = FEModel(name=f"global_tangent_{tangent_method}")
    curve = dnv_c208_steel_curve("S355", 0.010)
    model.add_material("steel", E_STEEL, NU_STEEL, hardening_curve=curve)
    nx, ny = 4, 2
    nodes: Dict[Tuple[int, int], int] = {}
    node_id = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            model.add_node(node_id, i / nx, 0.2 * j / ny, 0.0)
            nodes[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(ny):
        for i in range(nx):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [
                        nodes[(i, j)],
                        nodes[(i + 1, j)],
                        nodes[(i + 1, j + 1)],
                        nodes[(i, j + 1)],
                    ],
                    "steel",
                    thickness=0.010,
                ),
            )
            element_id += 1
    all_nodes = sorted(nodes.values())
    left_nodes = [nodes[(0, j)] for j in range(ny + 1)]
    right_nodes = [nodes[(nx, j)] for j in range(ny + 1)]
    model.add_boundary_condition(
        BoundaryCondition("left_x", left_nodes, {"ux": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition("origin_y", [nodes[(0, 0)]], {"uy": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "in_plane",
            all_nodes,
            {"uz": 0.0, "rx": 0.0, "ry": 0.0},
        )
    )

    target_stress = 340.0e6
    total_force = target_stress * 0.2 * 0.010
    load = LoadCase(name="plastic_pull")
    edge_weights = np.ones(len(right_nodes), dtype=float)
    edge_weights[[0, -1]] = 0.5
    edge_weights /= np.sum(edge_weights)
    for right_node, weight in zip(right_nodes, edge_weights):
        load.add_nodal_load(
            right_node,
            [weight * total_force, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

    started = time.perf_counter()
    with plane_stress_tangent_method(tangent_method):
        result = solve_static_nonlinear(
            model,
            load,
            num_steps=6,
            num_layers=3,
            max_iterations=20,
            tolerance=1.0e-8,
            convergence_settings="legacy",
        )
    elapsed = time.perf_counter() - started

    internal_force = _assemble_nonlinear_system(
        model,
        result.displacements,
        result.element_states,
        num_layers=3,
        tangent=False,
    )[0]
    external_force = assemble_load_vector(
        model,
        load,
        displacements=result.displacements,
    )[0]
    imbalance = internal_force - result.load_factor * external_force
    constrained_dofs = [
        model.mesh.get_node(constrained_node).dofs[local_dof]
        for constrained_node in all_nodes
        for local_dof in (2, 3, 4)
    ]
    constrained_dofs.extend(
        model.mesh.get_node(constrained_node).dofs[0]
        for constrained_node in left_nodes
    )
    constrained_dofs.append(model.mesh.get_node(nodes[(0, 0)]).dofs[1])
    right_ux = np.array(
        [
            result.displacements[model.mesh.get_node(right_node).dofs[0]]
            for right_node in right_nodes
        ],
        dtype=float,
    )
    state_vector = _flatten_numeric_state(result.element_states)
    reaction_vector = imbalance[np.asarray(constrained_dofs, dtype=int)]
    left_x_reactions = np.array(
        [
            imbalance[model.mesh.get_node(left_node).dofs[0]]
            for left_node in left_nodes
        ],
        dtype=float,
    )
    return {
        "status": result.status,
        "load_factor": float(result.load_factor),
        "total_newton_iterations": int(
            result.info.get(
                "total_newton_iterations",
                sum(step.iterations for step in result.steps),
            )
        ),
        "num_converged_steps": len(result.steps),
        "elapsed_seconds": float(elapsed),
        "reported_solve_seconds": float(result.info.get("solve_time", elapsed)),
        "displacements": result.displacements.copy(),
        "right_ux": right_ux,
        "reaction_vector": reaction_vector,
        "left_x_reactions": left_x_reactions,
        "state_vector": state_vector,
        "state_summary": result.info.get("strain_summary", {}),
    }


def _relative_array_error(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        return float("inf")
    return float(
        np.linalg.norm(first - second)
        / max(np.linalg.norm(first), np.linalg.norm(second), 1.0)
    )


def global_newton_tangent_benchmark_metrics(repeats: int = 1) -> Dict[str, Any]:
    """Compare analytical/oracle tangents in an actual global shell solve."""
    # Warm compilation, cached geometry, and the nonlinear acceleration layer
    # with both modes before recording complete solve time.
    _global_shell_newton_run("analytical")
    _global_shell_newton_run("numerical")
    retained: Dict[str, Dict[str, Any]] = {}
    for method in ("analytical", "numerical"):
        candidates = [
            _global_shell_newton_run(method)
            for _ in range(max(int(repeats), 1))
        ]
        retained[method] = min(
            candidates, key=lambda candidate: candidate["elapsed_seconds"]
        )

    analytical = retained["analytical"]
    numerical = retained["numerical"]
    parity = {
        "displacement_relative_error": _relative_array_error(
            analytical["displacements"], numerical["displacements"]
        ),
        "right_ux_relative_error": _relative_array_error(
            analytical["right_ux"], numerical["right_ux"]
        ),
        "reaction_relative_error": _relative_array_error(
            analytical["reaction_vector"], numerical["reaction_vector"]
        ),
        "committed_state_relative_error": _relative_array_error(
            analytical["state_vector"], numerical["state_vector"]
        ),
    }

    def reportable(run: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": run["status"],
            "load_factor": run["load_factor"],
            "total_newton_iterations": run["total_newton_iterations"],
            "num_converged_steps": run["num_converged_steps"],
            "elapsed_seconds": run["elapsed_seconds"],
            "reported_solve_seconds": run["reported_solve_seconds"],
            "right_ux": run["right_ux"].tolist(),
            "reaction_norm": float(np.linalg.norm(run["reaction_vector"])),
            "left_x_reactions": run["left_x_reactions"].tolist(),
            "state_summary": run["state_summary"],
            "committed_state_norm": float(np.linalg.norm(run["state_vector"])),
        }

    return {
        "model": "eight_element_force_controlled_plastic_shell_strip",
        "analytical": reportable(analytical),
        "numerical": reportable(numerical),
        "parity": parity,
        "speedup": numerical["elapsed_seconds"]
        / max(analytical["elapsed_seconds"], 1.0e-15),
    }


def material_point_path_metrics() -> Dict[str, Any]:
    """Qualify return-map yield consistency over representative strain paths."""
    curve = reference_plastic_curve()
    elastic_curve = None
    elastic_strain = np.array([[1.0e-4, -0.5e-4, 0.25e-4]], dtype=float)
    elastic_stress, elastic_tangent, _, _ = plane_stress_return_map(
        elastic_strain, np.zeros_like(elastic_strain), np.zeros(1), E_STEEL, NU_STEEL, elastic_curve
    )
    elastic_expected = elastic_strain @ plane_stress_elastic_matrix(E_STEEL, NU_STEEL).T

    paths = {
        "uniaxial": _material_tangent_fd_error(np.array([0.002, 0.0, 0.0]), curve),
        "biaxial_shear": _material_tangent_fd_error(np.array([0.003, 0.001, 0.0005]), curve),
        "pure_shear": _material_tangent_fd_error(np.array([0.0, 0.0, 0.004]), curve),
    }

    first = _material_tangent_fd_error(np.array([0.003, 0.0, 0.0]), curve)
    plastic_old = np.asarray(first["stress"], dtype=float).reshape(1, 3) * 0.0
    stress, _tangent, plastic_new, alpha_new = plane_stress_return_map(
        np.array([[0.003, 0.0, 0.0]], dtype=float),
        np.zeros((1, 3), dtype=float),
        np.zeros(1, dtype=float),
        E_STEEL,
        NU_STEEL,
        curve,
    )
    unload_stress, _, _, unload_alpha = plane_stress_return_map(
        np.array([[0.001, 0.0, 0.0]], dtype=float),
        plastic_new,
        alpha_new,
        E_STEEL,
        NU_STEEL,
        curve,
    )
    paths["unload_from_plastic"] = {
        "stress": unload_stress[0].tolist(),
        "alpha": float(unload_alpha[0]),
        "alpha_change": float(unload_alpha[0] - alpha_new[0]),
        "yield_residual": yield_function_residual(stress[0], float(alpha_new[0]), curve),
    }

    return {
        "elastic": {
            "stress_relative_error": float(np.linalg.norm(elastic_stress - elastic_expected) / max(np.linalg.norm(elastic_expected), 1.0)),
            "tangent_relative_error": float(
                np.linalg.norm(elastic_tangent[0] - plane_stress_elastic_matrix(E_STEEL, NU_STEEL))
                / max(np.linalg.norm(plane_stress_elastic_matrix(E_STEEL, NU_STEEL)), 1.0)
            ),
        },
        "plastic_paths": paths,
        "max_abs_yield_residual": max(abs(path.get("yield_residual", 0.0)) for path in paths.values()),
        "max_material_tangent_fd_error": max(
            path.get("tangent_fd_relative_error", 0.0) for path in paths.values()
        ),
    }


def _finite_difference_element_tangent(
    element: Any,
    model: FEModel,
    u_elem: np.ndarray,
    state: Any = None,
    num_layers: int = 5,
    step: float = 1.0e-7,
) -> Dict[str, Any]:
    material = model.get_material(element.material_name)
    f, K, trial_state = element.compute_nonlinear_response(
        model.mesh, material, u_elem, state, num_layers=num_layers, tangent=True
    )
    fd = np.zeros_like(K)
    for col in range(K.shape[1]):
        perturb = np.zeros_like(u_elem)
        perturb[col] = step
        fp = element.compute_nonlinear_response(
            model.mesh, material, u_elem + perturb, state, num_layers=num_layers, tangent=False
        )[0]
        fm = element.compute_nonlinear_response(
            model.mesh, material, u_elem - perturb, state, num_layers=num_layers, tangent=False
        )[0]
        fd[:, col] = (fp - fm) / (2.0 * step)
    error = float(np.linalg.norm(K - fd) / max(np.linalg.norm(fd), 1.0))
    return {
        "tangent_fd_relative_error": error,
        "force_norm": float(np.linalg.norm(f)),
        "tangent_norm": float(np.linalg.norm(K)),
        "fd_tangent_norm": float(np.linalg.norm(fd)),
        "state_summary": _state_summary(trial_state),
    }


def _state_summary(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    summary: Dict[str, Any] = {}
    for key in ("alpha", "plastic_strain", "layer_strain", "fiber_strain"):
        value = np.asarray(state.get(key, []), dtype=float)
        if value.size:
            summary[f"{key}_max"] = float(np.max(value))
            summary[f"{key}_min"] = float(np.min(value))
            summary[f"{key}_max_abs"] = float(np.max(np.abs(value)))
    if "axial_force" in state:
        summary["axial_force"] = float(state["axial_force"])
    return summary


def _beam_model(curve: DNVC208MaterialCurve | None = None, fiber: bool = False) -> Tuple[FEModel, BeamElement]:
    model = FEModel("beam_tangent_metric")
    model.add_material("steel", E_STEEL, NU_STEEL, hardening_curve=curve)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {"area": 0.01, "Iy": 1.0e-5, "Iz": 1.0e-5, "J": 1.0e-5}
    if fiber:
        section["fiber_plasticity"] = FiberSectionPlasticityConfig(5, 5)
    element = BeamElement(1, [1, 2], "steel", section)
    model.add_element(1, element)
    return model, element


def _shell_model(curve: DNVC208MaterialCurve | None = None) -> Tuple[FEModel, ShellElement]:
    model = FEModel("shell_tangent_metric")
    model.add_material("steel", E_STEEL, NU_STEEL, hardening_curve=curve)
    for node_id, coord in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)), start=1):
        model.add_node(node_id, *coord)
    element = create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    model.add_element(1, element)
    return model, element


def element_tangent_metrics() -> Dict[str, Any]:
    curve = reference_plastic_curve()

    beam_elastic_model, beam_elastic = _beam_model()
    u_beam_elastic = np.zeros(12, dtype=float)
    u_beam_elastic[6] = 0.001
    u_beam_elastic[7] = 0.0003
    beam_elastic_metric = _finite_difference_element_tangent(beam_elastic, beam_elastic_model, u_beam_elastic)

    beam_plastic_model, beam_plastic = _beam_model(curve, fiber=True)
    u_beam_plastic = np.zeros(12, dtype=float)
    u_beam_plastic[6] = 0.006
    u_beam_plastic[7] = 0.0003
    beam_plastic_metric = _finite_difference_element_tangent(beam_plastic, beam_plastic_model, u_beam_plastic)

    shell_elastic_model, shell_elastic = _shell_model()
    u_shell_elastic = np.zeros(24, dtype=float)
    u_shell_elastic[2::6] = [0.0, 0.001, 0.001, 0.0]
    u_shell_elastic[4::6] = [0.001, 0.001, 0.001, 0.001]
    shell_elastic_metric = _finite_difference_element_tangent(
        shell_elastic, shell_elastic_model, u_shell_elastic, step=1.0e-8
    )

    shell_plastic_model, shell_plastic = _shell_model(curve)
    u_shell_plastic = np.zeros(24, dtype=float)
    coords = shell_plastic.get_node_coordinates(shell_plastic_model.mesh)
    for local, coord in enumerate(coords):
        x, y, _ = coord
        base = local * 6
        u_shell_plastic[base + 0] = 0.003 * x
        u_shell_plastic[base + 1] = -0.0008 * y
        u_shell_plastic[base + 4] = 0.002 * x
    shell_plastic_metric = _finite_difference_element_tangent(
        shell_plastic, shell_plastic_model, u_shell_plastic, num_layers=5
    )
    shell_plastic_metric["tangent_status"] = (
        "tight" if shell_plastic_metric["tangent_fd_relative_error"] < 1.0e-4 else "diagnostic_high_tangent_error"
    )

    max_algorithmic_error = max(
        beam_elastic_metric["tangent_fd_relative_error"],
        beam_plastic_metric["tangent_fd_relative_error"],
        shell_elastic_metric["tangent_fd_relative_error"],
        shell_plastic_metric["tangent_fd_relative_error"],
    )
    return {
        "beam_elastic": beam_elastic_metric,
        "beam_fiber_plastic": beam_plastic_metric,
        "shell_elastic": shell_elastic_metric,
        "shell_layered_plastic": shell_plastic_metric,
        "max_tight_tangent_error": max(
            beam_elastic_metric["tangent_fd_relative_error"],
            beam_plastic_metric["tangent_fd_relative_error"],
            shell_elastic_metric["tangent_fd_relative_error"],
        ),
        "max_algorithmic_tangent_error": max_algorithmic_error,
    }


def dnv_curve_metric() -> Dict[str, Any]:
    curves = {}
    for grade, thickness in (("S355", 0.010), ("S420", 0.020), ("S460", 0.050)):
        curve = dnv_c208_steel_curve(grade, thickness)
        curves[f"{grade}_{thickness:g}"] = {
            "sigma_prop": curve.sigma_prop,
            "sigma_yield": curve.sigma_yield,
            "sigma_yield_2": curve.sigma_yield_2,
            "eps_p_y1": curve.eps_p_y1,
            "eps_p_y2": curve.eps_p_y2,
            "K": curve.K,
            "n": curve.n,
        }
    return curves


def generate_plasticity_qualification_report() -> Dict[str, Any]:
    material = material_point_path_metrics()
    element = element_tangent_metrics()
    analytical_tangent = algorithmic_tangent_path_metrics()
    tangent_performance = algorithmic_tangent_performance_metrics()
    global_newton = global_newton_tangent_benchmark_metrics()
    return {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "dnv_curves": dnv_curve_metric(),
        "material_point": material,
        "algorithmic_tangent": analytical_tangent,
        "element_tangents": element,
        "performance": {
            "material_batch": tangent_performance,
            "global_nonlinear_shell_newton": global_newton,
        },
        "status": "passed"
        if (
            material["max_abs_yield_residual"] < 1.0e-8
            and analytical_tangent["max_relative_error"] < 1.0e-5
            and element["max_algorithmic_tangent_error"] < 1.0e-4
            and global_newton["analytical"]["status"] == "completed"
            and global_newton["numerical"]["status"] == "completed"
            and global_newton["analytical"]["total_newton_iterations"]
            <= global_newton["numerical"]["total_newton_iterations"]
            and max(global_newton["parity"].values()) < 1.0e-8
        )
        else "diagnostic",
        "known_limitations": [
            "The analytical tangent is branch-consistent; exactly at a "
            "piecewise hardening corner the derivative is directional.",
            "The numerical tangent remains the qualification oracle and "
            "automatic fallback for non-finite or ill-conditioned "
            "pathological states.",
            "Near loss of material ellipticity, a local tangent can be "
            "physically ill-conditioned even when its differentiation is "
            "accurate.",
        ],
    }


def write_plasticity_qualification_report(path: Path | str = DEFAULT_PLASTICITY_QUALIFICATION_PATH) -> Dict[str, Any]:
    report = generate_plasticity_qualification_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
