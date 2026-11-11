"""Run the frozen representative nonlinear-static evidence campaign.

The coordinator executes one bounded worker at a time.  Each worker imports an
explicit installed wheel, performs one unmeasured warm solve, and then records
one complete-route sample.  Baseline/candidate order alternates within every
pair.  Armijo comparisons use the candidate wheel with the same physical
increments and tangent policy as the residual-decrease method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_case(spec: Mapping[str, Any], method: str):
    import numpy as np

    from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
    from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from anyfesolver.elements import BeamElement, create_shell_element
    from anyfesolver.fe_core import FEModel
    from anyfesolver.material_curves import LinearHardeningCurve
    from anyfesolver.mesh_gen import generate_simple_panel_mesh
    from anyfesolver.nonlinear_static import NonlinearLoadProgram, NonlinearLoadStage

    builder = str(spec["builder"])
    options: dict[str, Any] = {
        "num_steps": int(spec["num_steps"]),
        "max_iterations": int(spec["max_iterations"]),
        "tolerance": float(spec["tolerance"]),
    }
    probes: dict[str, Any] = {}
    if builder == "clamped_shell":
        divisions = int(spec["divisions"])
        model = generate_simple_panel_mesh(
            1.0,
            1.0,
            float(spec["thickness"]),
            divisions,
            divisions,
        )
        model.name = str(spec["id"])
        model.clear_boundary_conditions()
        edge_nodes = []
        for node_id, node in model.mesh.nodes.items():
            x, y, _z = node.coords()
            if min(x, y, 1.0 - x, 1.0 - y) <= 1.0e-12:
                edge_nodes.append(int(node_id))
            elif abs(x - 0.5) <= 1.0e-12 and abs(y - 0.5) <= 1.0e-12:
                probes["centre_uz_dof"] = int(node.dofs[2])
        model.add_boundary_condition(
            BoundaryCondition(
                "clamped-translations",
                edge_nodes,
                {"ux": 0.0, "uy": 0.0, "uz": 0.0},
            )
        )
        load = LoadCase(
            "pressure",
            follower_pressure=bool(spec.get("follower_pressure", False)),
        )
        for element_id in model.mesh.elements:
            load.add_pressure_load(int(element_id), float(spec["pressure_pa"]))
        options["load_case"] = load
        if bool(spec.get("corotational", False)):
            options["kinematics"] = "corotational"
    elif builder == "rollup_shell":
        n = int(spec["elements"])
        width = float(spec["width"])
        thickness = float(spec["thickness"])
        elastic_modulus = float(spec["elastic_modulus_pa"])
        model = FEModel(str(spec["id"]))
        model.add_material("steel", elastic_modulus, 0.0, density=7850.0)
        node_ids: dict[tuple[int, int], int] = {}
        next_node = 1
        for i in range(n + 1):
            for j in range(2):
                model.add_node(next_node, i / n, width * j, 0.0)
                node_ids[(i, j)] = next_node
                next_node += 1
        for i in range(n):
            element_id = i + 1
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [
                        node_ids[(i, 0)],
                        node_ids[(i + 1, 0)],
                        node_ids[(i + 1, 1)],
                        node_ids[(i, 1)],
                    ],
                    "steel",
                    thickness=thickness,
                ),
            )
        model.add_boundary_condition(
            BoundaryCondition(
                "clamp",
                [node_ids[(0, 0)], node_ids[(0, 1)]],
                {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            )
        )
        inertia = width * thickness**3 / 12.0
        moment = float(spec["target_rotation_rad"]) * elastic_modulus * inertia
        load = LoadCase("end-moment")
        for j in (0, 1):
            load.add_nodal_load(
                node_ids[(n, j)], moments=np.array([0.0, -moment / 2.0, 0.0])
            )
        options.update(load_case=load, kinematics="corotational")
        tip = model.mesh.get_node(node_ids[(n, 0)])
        radius = elastic_modulus * inertia / moment
        probes.update(
            tip_x_dof=int(tip.dofs[0]),
            tip_z_dof=int(tip.dofs[2]),
            tip_ry_dof=int(tip.dofs[4]),
            expected_tip_x=float(radius * np.sin(float(spec["target_rotation_rad"])) - 1.0),
            expected_tip_z=float(radius * (1.0 - np.cos(float(spec["target_rotation_rad"])))),
            expected_tip_ry=-float(spec["target_rotation_rad"]),
        )
    elif builder == "plastic_s3_reversal":
        model = FEModel(str(spec["id"]))
        model.add_material(
            "steel",
            210.0e9,
            0.3,
            density=7850.0,
            hardening_curve=LinearHardeningCurve(
                float(spec["yield_stress_pa"]),
                float(spec["hardening_modulus_pa"]),
            ),
        )
        model.add_node(1, 0.0, 0.0, 0.0)
        model.add_node(2, 1.0, 0.0, 0.0)
        model.add_node(3, 0.2, 0.9, 0.0)
        model.add_element(
            1,
            QualifiedE4PLS3ShellElement(
                1,
                [1, 2, 3],
                "steel",
                thickness=float(spec["thickness"]),
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
        model.add_boundary_condition(FixedSupport("node-1", [1]))
        model.add_boundary_condition(
            BoundaryCondition(
                "node-2-guide",
                [2],
                {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            )
        )
        model.add_boundary_condition(FixedSupport("node-3", [3]))
        forward = LoadCase("forward")
        forward.add_nodal_load(
            2, [float(spec["forward_force_n"]), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        reverse = LoadCase("reverse")
        reverse.add_nodal_load(
            2, [float(spec["reverse_force_n"]), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        options.update(
            load_program=NonlinearLoadProgram(
                (
                    NonlinearLoadStage("forward", forward),
                    NonlinearLoadStage("reverse", reverse),
                )
            ),
            num_layers=int(spec["num_layers"]),
            record_increment_snapshots=True,
        )
        probes["loaded_ux_dof"] = int(model.mesh.get_node(2).dofs[0])
    elif builder == "prescribed_mpc_beam":
        n = int(spec["elements"])
        length = float(spec["length"])
        model = FEModel(str(spec["id"]))
        model.add_material("steel", 210.0e9, 0.3, density=7850.0)
        section = {
            "area": float(spec["area"]),
            "Iy": float(spec["inertia"]),
            "Iz": float(spec["inertia"]),
            "J": float(spec["torsion"]),
            "orientation": (0.0, 0.0, 1.0),
        }
        for i in range(n + 1):
            model.add_node(i + 1, length * i / n, 0.0, 0.0)
        for i in range(n):
            model.add_element(
                i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section)
            )
        model.add_boundary_condition(FixedSupport("fixed", [1]))
        model.add_boundary_condition(
            BoundaryCondition(
                "planar-guides",
                list(range(2, n + 2)),
                {"uz": 0.0, "rx": 0.0, "ry": 0.0},
            )
        )
        tip = model.mesh.get_node(n + 1)
        model.add_boundary_condition(
            BoundaryCondition(
                "prescribed-tip-x",
                [n + 1],
                {"ux": float(spec["prescribed_tip_x_m"])},
            )
        )
        for i in range(1, n):
            interior = model.mesh.get_node(i + 1)
            model.add_constraint_equation(
                terms=((interior.dofs[0], 1.0), (tip.dofs[0], -i / n)),
                rhs=0.0,
                source_id=f"axial-interpolation-{i}",
                dependent_dof=interior.dofs[0],
            )
        load = LoadCase("tip-lateral")
        load.add_nodal_load(
            n + 1,
            [0.0, float(spec["lateral_force_n"]), 0.0, 0.0, 0.0, 0.0],
        )
        options["load_case"] = load
        probes.update(
            tip_ux_dof=int(tip.dofs[0]),
            tip_uy_dof=int(tip.dofs[1]),
            expected_tip_ux=float(spec["prescribed_tip_x_m"]),
            mpc_axial_dofs=[
                [int(model.mesh.get_node(i + 1).dofs[0]), float(i / n)]
                for i in range(1, n)
            ],
        )
    else:
        raise ValueError(f"unknown case builder: {builder}")

    if method:
        options["convergence_settings"] = {
            "profile": "legacy",
            "line_search": method,
        }
        if bool(spec.get("force_consistent_tangent", False)):
            options["corotational_tangent"] = "consistent"
    return model, options, probes


def _state_summary(element_states: Mapping[Any, Any]) -> dict[str, float]:
    import numpy as np

    values: dict[str, list[float]] = {"alpha": [], "plastic_strain": []}

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif key in values:
            array = np.asarray(value, dtype=float)
            if array.size:
                values[key].append(float(np.max(np.abs(array))))

    visit(element_states)
    return {name: max(items, default=0.0) for name, items in values.items()}


def _state_observables(element_states: Mapping[Any, Any]) -> dict[str, list[float]]:
    import numpy as np

    retained = {"alpha", "plastic_strain", "committed_total_u"}
    observations: dict[str, list[float]] = {}

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                child_path = f"{path}/{key}" if path else str(key)
                visit(child, child_path)
            return
        if path.rsplit("/", 1)[-1] in retained:
            array = np.asarray(value, dtype=float)
            observations[path] = array.reshape(-1).tolist()

    visit(element_states)
    return observations


def _family_checks(
    result: Any,
    spec: Mapping[str, Any],
    probes: Mapping[str, Any],
) -> dict[str, bool]:
    import numpy as np

    displacement = np.asarray(result.displacements, dtype=float)
    checks = {
        "completed": str(result.status) == "completed",
        "target_load_factor": abs(
            float(result.load_factor)
            - (2.0 if spec["builder"] == "plastic_s3_reversal" else 1.0)
        )
        <= 1.0e-12,
    }
    builder = str(spec["builder"])
    if builder == "rollup_shell":
        checks.update(
            tip_rotation=np.isclose(
                displacement[int(probes["tip_ry_dof"])],
                float(probes["expected_tip_ry"]),
                rtol=1.0e-4,
                atol=1.0e-9,
            ),
            tip_x=np.isclose(
                displacement[int(probes["tip_x_dof"])],
                float(probes["expected_tip_x"]),
                rtol=5.0e-3,
                atol=1.0e-9,
            ),
            tip_z=np.isclose(
                displacement[int(probes["tip_z_dof"])],
                float(probes["expected_tip_z"]),
                rtol=5.0e-3,
                atol=1.0e-9,
            ),
        )
    elif builder == "plastic_s3_reversal":
        state = _state_summary(result.element_states)
        factors = result.info.get("load_program_stage_factors", {})
        displacement_history = [
            float(
                np.asarray(snapshot.displacements, dtype=float)[
                    int(probes["loaded_ux_dof"])
                ]
            )
            for snapshot in result.snapshots
        ]
        checks.update(
            committed_plastic_strain=state["plastic_strain"] > 0.0,
            committed_hardening=state["alpha"] > 0.0,
            forward_stage_complete=abs(float(factors.get("forward", 0.0)) - 1.0)
            <= 1.0e-12,
            reverse_stage_complete=abs(float(factors.get("reverse", 0.0)) - 1.0)
            <= 1.0e-12,
            history_snapshots=len(result.snapshots) == int(spec["num_steps"]),
            reversed_displacement_direction=bool(displacement_history)
            and max(displacement_history) - displacement_history[-1] > 1.0e-6,
        )
    elif builder == "prescribed_mpc_beam":
        tip_ux = displacement[int(probes["tip_ux_dof"])]
        mpc_error = max(
            (
                abs(displacement[int(dof)] - float(weight) * tip_ux)
                for dof, weight in probes["mpc_axial_dofs"]
            ),
            default=0.0,
        )
        checks.update(
            prescribed_tip=np.isclose(
                tip_ux,
                float(probes["expected_tip_ux"]),
                rtol=0.0,
                atol=1.0e-12,
            ),
            weighted_mpc=mpc_error <= 1.0e-12,
            nonzero_lateral_response=abs(
                displacement[int(probes["tip_uy_dof"])]
            )
            > 0.0,
        )
    elif builder == "clamped_shell" and bool(spec.get("follower_pressure", False)):
        centre = probes.get("centre_uz_dof")
        checks.update(
            positive_centre_response=centre is not None
            and displacement[int(centre)] > 0.0,
            effective_tangent=result.info.get("equilibrium_tangent")
            == "K_internal-K_external",
            current_external_load=result.info.get(
                "external_load_reduction", {}
            ).get("preprojected", False)
            is not True,
        )
    return {name: bool(value) for name, value in checks.items()}


def _physical_observables(
    result: Any,
    spec: Mapping[str, Any],
    probes: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    reaction_history = []
    step_history = []
    for step in result.steps:
        reaction_history.append(
            {
                str(name): np.asarray(values, dtype=float).tolist()
                for name, values in sorted(step.support_reactions.items())
            }
        )
        step_history.append(
            {
                "load_factor": float(step.load_factor),
                "residual_norm": float(step.residual_norm),
                "displacement_norm": float(step.displacement_norm),
                "max_equivalent_plastic_strain": float(
                    step.max_equivalent_plastic_strain
                ),
            }
        )
    snapshots = []
    for snapshot in result.snapshots:
        snapshots.append(
            {
                "load_factor": float(snapshot.load_factor),
                "displacements": np.asarray(
                    snapshot.displacements, dtype=float
                ).tolist(),
                "state_observables": _state_observables(
                    snapshot.element_states
                ),
            }
        )
    return {
        "status": str(result.status),
        "load_factor": float(result.load_factor),
        "displacements": np.asarray(result.displacements, dtype=float).tolist(),
        "steps": step_history,
        "support_reactions": reaction_history,
        "state_summary": _state_summary(result.element_states),
        "state_observables": _state_observables(result.element_states),
        "snapshots": snapshots,
        "family_checks": _family_checks(result, spec, probes),
    }


def _worker_sample(site: Path, spec: Mapping[str, Any], method: str) -> dict[str, Any]:
    sys.path.insert(0, str(site.resolve()))

    import importlib.metadata
    import numpy as np

    import anyfesolver
    from anyfesolver import nonlinear_static
    from anyfesolver.nonlinear_performance_bootstrap import nonlinear_performance_status

    nonlinear_static._ensure_nonlinear_acceleration()
    original_assemble = nonlinear_static._assemble_nonlinear_system
    original_factorize = nonlinear_static.factorize
    counters: dict[str, int] = {}

    def reset() -> None:
        counters.clear()
        counters.update(
            assembly_calls=0,
            tangent_calls=0,
            residual_only_calls=0,
            linear_factorizations=0,
            linear_solves=0,
        )

    def observed_assemble(*args, **kwargs):
        counters["assembly_calls"] += 1
        if bool(kwargs.get("tangent", True)):
            counters["tangent_calls"] += 1
        else:
            counters["residual_only_calls"] += 1
        return original_assemble(*args, **kwargs)

    class ObservedHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

        def solve(self, rhs):
            counters["linear_solves"] += 1
            return self._handle.solve(rhs)

    def observed_factorize(*args, **kwargs):
        counters["linear_factorizations"] += 1
        return ObservedHandle(original_factorize(*args, **kwargs))

    nonlinear_static._assemble_nonlinear_system = observed_assemble
    nonlinear_static.factorize = observed_factorize

    def solve_once() -> tuple[Any, Any, Mapping[str, Any], float, float]:
        route_start = time.perf_counter()
        model, options, probes = _build_case(spec, method)
        solve_start = time.perf_counter()
        result = nonlinear_static.solve_static_nonlinear(model, **options)
        solve_seconds = time.perf_counter() - solve_start
        return (
            result,
            model,
            probes,
            time.perf_counter() - route_start,
            solve_seconds,
        )

    reset()
    print(json.dumps({"event": "warm_start"}), flush=True)
    solve_once()
    print(json.dumps({"event": "warm_complete"}), flush=True)
    reset()
    print(json.dumps({"event": "measured_start"}), flush=True)
    result, _model, probes, route_seconds, solve_seconds = solve_once()
    observables = _physical_observables(result, spec, probes)
    solver = (
        result.info.get("nonlinear_performance", {}).get("solver", {})
        if isinstance(result.info, Mapping)
        else {}
    )
    recovery = (
        result.info.get("reaction_force_recovery", {})
        if isinstance(result.info, Mapping)
        else {}
    )
    failed_work = solver.get("failed_work", {})
    if not isinstance(failed_work, Mapping):
        failed_work = {}
    failed_work_units = sum(
        int(failed_work.get(name, 0))
        for name in (
            "newton_iterations",
            "rejected_trial_evaluations",
            "recoverable_trial_failures",
        )
    )
    sample = {
        "complete_route_wall_seconds": float(route_seconds),
        "solver_seconds": float(solve_seconds),
        "status": str(result.status),
        "load_factor": float(result.load_factor),
        "steps": len(result.steps),
        "iterations": int(result.info.get("total_newton_iterations", 0)),
        "work": {
            **{name: int(value) for name, value in counters.items()},
            "rejected_full_steps": int(solver.get("rejected_full_steps", 0)),
            "backtracks": int(solver.get("backtracks", 0)),
            "promotion_evaluations": int(solver.get("promotion_evaluations", 0)),
            "failed_increment_count": int(failed_work.get("increment_count", 0)),
            "failed_newton_iterations": int(
                failed_work.get("newton_iterations", 0)
            ),
            "failed_rejected_trial_evaluations": int(
                failed_work.get("rejected_trial_evaluations", 0)
            ),
            "failed_recoverable_trial_failures": int(
                failed_work.get("recoverable_trial_failures", 0)
            ),
            "failed_work_units": failed_work_units,
            "reaction_force_reuse_count": int(
                recovery.get("accepted_force_reuse_count", 0)
            ),
            "reaction_force_reassembly_count": int(
                recovery.get("full_reassembly_count", 0)
            ),
        },
        "physical": observables,
        "physical_sha256": _json_digest(observables),
        "dispatch": nonlinear_performance_status(),
        "identity": {
            "anyfesolver_version": importlib.metadata.version("anyfesolver"),
            "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
            "numpy_version": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    return sample


def _worker_main(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = {
        str(case["id"]): case
        for case in manifest["performance_cases"]
    }
    if args.case not in cases:
        raise SystemExit(f"unknown case: {args.case}")
    sample = _worker_sample(args.site, cases[args.case], args.method)
    print(json.dumps(sample, sort_keys=True), flush=True)
    return 0


def _terminate_tree(pid: int) -> None:
    import psutil

    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    processes = root.children(recursive=True)
    processes.append(root)
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=3.0)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def _tree_rss(pid: int) -> int:
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
        except psutil.Error:
            pass
    return total


def _run_bounded_worker(
    *,
    site: Path,
    manifest: Path,
    case: str,
    method: str,
    timeout_seconds: float,
    startup_timeout_seconds: float,
    worker_budget_seconds: float,
    memory_bytes: int,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--site",
        str(site.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--case",
        case,
        "--method",
        method,
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phase_started = started
    phase = "startup"
    peak_rss = 0
    output_lines: list[str] = []
    output_queue: queue.Queue[str] = queue.Queue()
    with log_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        assert process.stdout is not None

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        terminal_reason = "exit"
        while process.poll() is None:
            now = time.perf_counter()
            while True:
                try:
                    line = output_queue.get_nowait()
                except queue.Empty:
                    break
                output_lines.append(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_name = event.get("event")
                if event_name == "warm_start":
                    phase = "warm"
                    phase_started = now
                elif event_name == "warm_complete":
                    phase = "between_solves"
                    phase_started = now
                elif event_name == "measured_start":
                    phase = "measured"
                    phase_started = now
            peak_rss = max(peak_rss, _tree_rss(process.pid))
            if now - started > worker_budget_seconds:
                terminal_reason = "campaign_budget"
                _terminate_tree(process.pid)
                break
            phase_limit = (
                startup_timeout_seconds
                if phase in {"startup", "between_solves"}
                else timeout_seconds
            )
            if now - phase_started > phase_limit:
                terminal_reason = f"{phase}_timeout"
                _terminate_tree(process.pid)
                break
            if peak_rss > memory_bytes:
                terminal_reason = "memory_limit"
                _terminate_tree(process.pid)
                break
            time.sleep(0.05)
        process.wait(timeout=10.0)
        reader.join(timeout=2.0)
        while True:
            try:
                output_lines.append(output_queue.get_nowait())
            except queue.Empty:
                break
    elapsed = time.perf_counter() - started
    if terminal_reason != "exit":
        return {
            "ok": False,
            "terminal_reason": terminal_reason,
            "exit_code": process.returncode,
            "worker_wall_seconds": elapsed,
            "worker_peak_tree_rss_bytes": peak_rss,
            "stderr_log": str(log_path.resolve()),
        }
    if process.returncode != 0:
        return {
            "ok": False,
            "terminal_reason": "nonzero_exit",
            "exit_code": process.returncode,
            "worker_wall_seconds": elapsed,
            "worker_peak_tree_rss_bytes": peak_rss,
            "stderr_log": str(log_path.resolve()),
        }
    lines = [line for line in output_lines if line.strip()]
    try:
        sample = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "terminal_reason": "malformed_output",
            "detail": f"{type(exc).__name__}: {exc}",
            "exit_code": process.returncode,
            "worker_wall_seconds": elapsed,
            "worker_peak_tree_rss_bytes": peak_rss,
            "stderr_log": str(log_path.resolve()),
        }
    imported_module = Path(sample.get("identity", {}).get("anyfesolver_module", ""))
    try:
        imported_module.resolve().relative_to(site.resolve())
    except (ValueError, OSError):
        return {
            "ok": False,
            "terminal_reason": "identity_mismatch",
            "imported_module": str(imported_module),
            "expected_site": str(site.resolve()),
            "exit_code": process.returncode,
            "worker_wall_seconds": elapsed,
            "worker_peak_tree_rss_bytes": peak_rss,
            "stderr_log": str(log_path.resolve()),
        }
    sample.update(
        ok=True,
        terminal_reason="exit",
        exit_code=process.returncode,
        worker_wall_seconds=elapsed,
        worker_peak_tree_rss_bytes=peak_rss,
        stderr_log=str(log_path.resolve()),
    )
    return sample


def _flatten_reactions(history: list[dict[str, list[float]]]) -> tuple[list[str], list[float]]:
    keys: list[str] = []
    values: list[float] = []
    for step_index, step in enumerate(history):
        for name, components in sorted(step.items()):
            for component_index, value in enumerate(components):
                keys.append(f"{step_index}:{name}:{component_index}")
                values.append(float(value))
    return keys, values


def _max_difference(left: list[float], right: list[float]) -> tuple[float, float]:
    if len(left) != len(right):
        return float("inf"), float("inf")
    scale = max([1.0, *map(abs, left), *map(abs, right)])
    difference = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    return difference, scale


def _mapping_difference(
    left: Mapping[str, list[float]], right: Mapping[str, list[float]]
) -> tuple[bool, float, float]:
    if set(left) != set(right):
        return False, float("inf"), float("inf")
    difference = 0.0
    scale = 1.0
    for key in sorted(left):
        key_difference, key_scale = _max_difference(left[key], right[key])
        difference = max(difference, key_difference)
        scale = max(scale, key_scale)
    return True, difference, scale


def _compare_physics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if not left.get("ok") or not right.get("ok"):
        return {"numerical_match": False, "reason": "worker_failure"}
    lp = left["physical"]
    rp = right["physical"]
    displacement_difference, displacement_scale = _max_difference(
        lp["displacements"], rp["displacements"]
    )
    left_keys, left_reactions = _flatten_reactions(lp["support_reactions"])
    right_keys, right_reactions = _flatten_reactions(rp["support_reactions"])
    reaction_difference, reaction_scale = _max_difference(
        left_reactions, right_reactions
    )
    state_inventory, state_difference, state_scale = _mapping_difference(
        lp["state_observables"], rp["state_observables"]
    )
    plastic_difference = max(
        (
            abs(float(lp["state_summary"][key]) - float(rp["state_summary"][key]))
            for key in ("alpha", "plastic_strain")
        ),
        default=0.0,
    )
    same_step_path = len(lp["steps"]) == len(rp["steps"])
    max_step_load_difference = float("inf")
    max_step_displacement_difference = float("inf")
    max_step_plastic_difference = float("inf")
    if same_step_path:
        max_step_load_difference = max(
            (
                abs(float(a["load_factor"]) - float(b["load_factor"]))
                for a, b in zip(lp["steps"], rp["steps"])
            ),
            default=0.0,
        )
        max_step_displacement_difference = max(
            (
                abs(float(a["displacement_norm"]) - float(b["displacement_norm"]))
                for a, b in zip(lp["steps"], rp["steps"])
            ),
            default=0.0,
        )
        max_step_plastic_difference = max(
            (
                abs(
                    float(a["max_equivalent_plastic_strain"])
                    - float(b["max_equivalent_plastic_strain"])
                )
                for a, b in zip(lp["steps"], rp["steps"])
            ),
            default=0.0,
        )
    same_snapshots = len(lp["snapshots"]) == len(rp["snapshots"])
    max_snapshot_difference = 0.0
    max_snapshot_scale = 1.0
    snapshot_state_inventory = True
    max_snapshot_state_difference = 0.0
    max_snapshot_state_scale = 1.0
    if same_snapshots:
        for left_snapshot, right_snapshot in zip(lp["snapshots"], rp["snapshots"]):
            if (
                abs(
                    float(left_snapshot["load_factor"])
                    - float(right_snapshot["load_factor"])
                )
                > 1.0e-12
            ):
                same_snapshots = False
                break
            difference, scale = _max_difference(
                left_snapshot["displacements"],
                right_snapshot["displacements"],
            )
            max_snapshot_difference = max(max_snapshot_difference, difference)
            max_snapshot_scale = max(max_snapshot_scale, scale)
            inventory, difference, scale = _mapping_difference(
                left_snapshot["state_observables"],
                right_snapshot["state_observables"],
            )
            snapshot_state_inventory = snapshot_state_inventory and inventory
            max_snapshot_state_difference = max(
                max_snapshot_state_difference, difference
            )
            max_snapshot_state_scale = max(max_snapshot_state_scale, scale)
    family_targets_met = all(lp["family_checks"].values()) and all(
        rp["family_checks"].values()
    )
    path_displacement_scale = max(
        [
            1.0,
            *[abs(float(step["displacement_norm"])) for step in lp["steps"]],
            *[abs(float(step["displacement_norm"])) for step in rp["steps"]],
        ]
    )
    numerical_match = (
        lp["status"] == rp["status"] == "completed"
        and family_targets_met
        and same_step_path
        and max_step_load_difference <= 1.0e-12
        and max_step_displacement_difference
        <= 1.0e-12 + 1.0e-10 * path_displacement_scale
        and max_step_plastic_difference <= 1.0e-12
        and abs(float(lp["load_factor"]) - float(rp["load_factor"])) <= 1.0e-12
        and displacement_difference <= 1.0e-12 + 1.0e-10 * displacement_scale
        and left_keys == right_keys
        and reaction_difference <= 1.0e-8 + 1.0e-10 * reaction_scale
        and plastic_difference <= 1.0e-12
        and state_inventory
        and state_difference <= 1.0e-12 + 1.0e-10 * state_scale
        and same_snapshots
        and max_snapshot_difference
        <= 1.0e-12 + 1.0e-10 * max_snapshot_scale
        and snapshot_state_inventory
        and max_snapshot_state_difference
        <= 1.0e-12 + 1.0e-10 * max_snapshot_state_scale
    )
    return {
        "numerical_match": numerical_match,
        "exact_digest_match": left["physical_sha256"] == right["physical_sha256"],
        "max_displacement_difference": displacement_difference,
        "displacement_scale": displacement_scale,
        "max_reaction_difference": reaction_difference,
        "reaction_scale": reaction_scale,
        "max_plastic_state_difference": plastic_difference,
        "same_reaction_inventory": left_keys == right_keys,
        "same_step_path": same_step_path,
        "max_step_load_factor_difference": max_step_load_difference,
        "max_step_displacement_norm_difference": max_step_displacement_difference,
        "max_step_plastic_strain_difference": max_step_plastic_difference,
        "same_state_inventory": state_inventory,
        "max_state_observable_difference": state_difference,
        "same_snapshot_path": same_snapshots,
        "max_snapshot_displacement_difference": max_snapshot_difference,
        "same_snapshot_state_inventory": snapshot_state_inventory,
        "max_snapshot_state_difference": max_snapshot_state_difference,
        "family_targets_met": family_targets_met,
    }


def _median(samples: list[Mapping[str, Any]], key: str) -> float:
    return float(statistics.median(float(sample[key]) for sample in samples))


def _case_summary(
    baseline: list[Mapping[str, Any]], candidate: list[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_time = _median(baseline, "complete_route_wall_seconds")
    candidate_time = _median(candidate, "complete_route_wall_seconds")
    return {
        "baseline_median_seconds": baseline_time,
        "candidate_median_seconds": candidate_time,
        "median_reduction_fraction": (baseline_time - candidate_time) / baseline_time,
        "baseline_peak_tree_rss_bytes": max(
            int(sample["worker_peak_tree_rss_bytes"]) for sample in baseline
        ),
        "candidate_peak_tree_rss_bytes": max(
            int(sample["worker_peak_tree_rss_bytes"]) for sample in candidate
        ),
        "baseline_median_work": {
            key: statistics.median(sample["work"][key] for sample in baseline)
            for key in baseline[0]["work"]
        },
        "candidate_median_work": {
            key: statistics.median(sample["work"][key] for sample in candidate)
            for key in candidate[0]["work"]
        },
    }


def _convergence_physical_status(
    *,
    oracle: Mapping[str, Any],
    method_samples: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    samples = [sample for values in method_samples.values() for sample in values]
    successful = [
        sample
        for sample in samples
        if sample.get("ok") and sample.get("status") == "completed"
    ]
    if successful:
        matched = all(
            _compare_physics(oracle, sample).get("numerical_match", False)
            for sample in successful
        )
        return {
            "physical_match": matched,
            "basis": "completed_samples",
            "completed_samples": len(successful),
        }
    if samples and all(_resource_terminal(sample) for sample in samples):
        return {
            "physical_match": True,
            "basis": "resource_terminal_no_solution",
            "completed_samples": 0,
        }
    return {
        "physical_match": False,
        "basis": "missing_or_invalid_sample",
        "completed_samples": 0,
    }


def _terminal_evidence_complete(sample: Mapping[str, Any]) -> bool:
    return bool(sample.get("ok")) or sample.get("terminal_reason") in {
        "warm_timeout",
        "measured_timeout",
        "memory_limit",
        "campaign_budget",
    }


def _resource_terminal(sample: Mapping[str, Any]) -> bool:
    return sample.get("terminal_reason") in {
        "warm_timeout",
        "measured_timeout",
        "memory_limit",
        "campaign_budget",
    }


def _run_installed_regressions(
    *, site: Path, tests: list[str], timeout_seconds: float, log_path: Path
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env["PYTHONPATH"] = str(site.resolve())
    identity_test = log_path.parent / "test_installed_identity.py"
    expected_site = str(site.resolve())
    identity_test.parent.mkdir(parents=True, exist_ok=True)
    identity_test.write_text(
        "from pathlib import Path\n"
        "def test_anyfesolver_imports_from_frozen_site():\n"
        "    import anyfesolver\n"
        f"    expected = Path({expected_site!r}).resolve()\n"
        "    imported = Path(anyfesolver.__file__).resolve()\n"
        "    assert imported.is_relative_to(expected), (imported, expected)\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-c",
                "NUL",
                str(identity_test.resolve()),
                *tests,
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        timeout_stdout = exc.stdout or ""
        timeout_stderr = exc.stderr or ""
        if isinstance(timeout_stdout, bytes):
            timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
        if isinstance(timeout_stderr, bytes):
            timeout_stderr = timeout_stderr.decode("utf-8", errors="replace")
        log_path.write_text(
            timeout_stdout + "\n--- stderr ---\n" + timeout_stderr,
            encoding="utf-8",
        )
        return {
            "ok": False,
            "terminal_reason": "timeout",
            "exit_code": None,
            "wall_seconds": elapsed,
            "tests": tests,
            "installed_site": expected_site,
            "log": str(log_path.resolve()),
        }
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + "\n--- stderr ---\n" + completed.stderr,
        encoding="utf-8",
    )
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "tests": tests,
        "installed_site": expected_site,
        "log": str(log_path.resolve()),
        "summary_tail": completed.stdout.strip().splitlines()[-1]
        if completed.stdout.strip()
        else "",
    }


def _validate_campaign_identity(
    *,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    baseline_wheel: Path,
    candidate_wheel: Path,
) -> dict[str, Any]:
    expected = {
        "baseline_revision": str(manifest["baseline_revision"]),
        "candidate_revision": str(manifest["candidate_revision"]),
        "baseline_tree": str(manifest["source_trees"]["baseline"]),
        "candidate_tree": str(manifest["source_trees"]["candidate"]),
        "baseline_wheel_sha256": _sha256(baseline_wheel),
        "candidate_wheel_sha256": _sha256(candidate_wheel),
    }
    mismatches = {
        key: {"expected": value, "observed": provenance.get(key)}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "build provenance does not match frozen campaign identity: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def _install_frozen_wheels(
    *,
    install_root: Path,
    baseline_wheel: Path,
    candidate_wheel: Path,
    campaign_deadline: float,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any] | None]:
    try:
        install_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return {}, {}, {
            "terminal_reason": "install_root_error",
            "detail": f"{type(exc).__name__}: {exc}",
            "install_root": str(install_root.resolve()),
        }
    sites: dict[str, Path] = {}
    records: dict[str, Any] = {}
    for label, wheel in (
        ("baseline", baseline_wheel),
        ("candidate", candidate_wheel),
    ):
        site = install_root / label
        log_path = install_root / f"{label}.pip-install.log"
        remaining = campaign_deadline - time.perf_counter()
        if remaining <= 0.0:
            return sites, records, {
                "terminal_reason": "campaign_budget",
                "label": label,
                "log": str(log_path.resolve()),
            }
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(site),
                    str(wheel.resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=min(300.0, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = exc.stdout or ""
            timeout_stderr = exc.stderr or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
            if isinstance(timeout_stderr, bytes):
                timeout_stderr = timeout_stderr.decode("utf-8", errors="replace")
            log_path.write_text(
                timeout_stdout + "\n--- stderr ---\n" + timeout_stderr,
                encoding="utf-8",
            )
            return sites, records, {
                "terminal_reason": "install_timeout",
                "label": label,
                "log": str(log_path.resolve()),
            }
        log_path.write_text(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return sites, records, {
                "terminal_reason": "install_nonzero_exit",
                "label": label,
                "exit_code": completed.returncode,
                "log": str(log_path.resolve()),
            }
        sites[label] = site.resolve()
        records[label] = {
            "wheel": str(wheel.resolve()),
            "wheel_sha256": _sha256(wheel),
            "site": str(site.resolve()),
            "pip_log": str(log_path.resolve()),
        }
    return sites, records, None


def _adjudicate_armijo(
    *,
    convergence: Mapping[str, Mapping[str, Any]],
    convergence_repeats: int,
    timeout_seconds: float,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    complete = all(
        case["complete"] and case["physical_match"]
        for case in convergence.values()
    )
    new_solves = sum(
        int(case["newly_solved_by_armijo"]) for case in convergence.values()
    )
    total_residual_failed = sum(
        int(case["residual_decrease_failed_work"])
        for case in convergence.values()
    )
    total_armijo_failed = sum(
        int(case["armijo_failed_work"]) for case in convergence.values()
    )
    failed_reduction = (
        (total_residual_failed - total_armijo_failed) / total_residual_failed
        if total_residual_failed > 0
        else 0.0
    )
    residual_runtime = sum(
        float(case["residual_decrease_median_seconds"] or timeout_seconds)
        for case in convergence.values()
    )
    armijo_runtime = sum(
        float(case["armijo_median_seconds"] or timeout_seconds)
        for case in convergence.values()
    )
    armijo_all_completed = all(
        pair["armijo"].get("ok")
        and pair["armijo"].get("status") == "completed"
        for case in convergence.values()
        for pair in case["pairs"]
    )
    full_failed_work_inventory = all(
        len(case["pairs"]) == convergence_repeats
        and all(
            pair[method].get("ok")
            and pair[method].get("status") == "completed"
            for pair in case["pairs"]
            for method in ("always", "armijo")
        )
        for case in convergence.values()
    )
    runtime_ok = armijo_runtime <= residual_runtime
    new_solve_route = new_solves >= int(acceptance["armijo_new_difficult_solves"])
    failed_work_route = (
        full_failed_work_inventory
        and failed_reduction
        >= float(acceptance["armijo_failed_work_reduction_fraction"])
    )
    return {
        "complete": complete,
        "go": complete
        and armijo_all_completed
        and runtime_ok
        and (new_solve_route or failed_work_route),
        "new_solves": new_solves,
        "failed_work_reduction_fraction": failed_reduction,
        "armijo_runtime_seconds": armijo_runtime,
        "residual_decrease_runtime_seconds": residual_runtime,
        "armijo_all_samples_completed": armijo_all_completed,
        "runtime_not_worse": runtime_ok,
        "failed_work_inventory_complete": full_failed_work_inventory,
    }


def _write_report(result: Mapping[str, Any], report_path: Path) -> None:
    def seconds(value: Any) -> str:
        return "resource limit" if value is None else f"{float(value):.6f}"

    lines = [
        "# Representative nonlinear evidence",
        "",
        f"- Campaign started: `{result['started_utc']}`",
        f"- Campaign completed: `{result['completed_utc']}`",
        f"- Baseline: `{result['identity']['baseline_revision']}`",
        f"- Candidate: `{result['identity']['candidate_revision']}`",
        f"- Manifest SHA-256: `{result['identity']['manifest_sha256']}`",
        f"- Runner SHA-256: `{result['identity']['runner_sha256']}`",
        "",
        "## Installed-wheel performance",
        "",
        "| Case | Baseline median (s) | Candidate median (s) | Reduction | Physics |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for case_id, case in result["performance"].items():
        summary = case["summary"]
        if summary is None:
            lines.append(
                f"| {case_id} | resource limit | resource limit | n/a | FAIL |"
            )
        else:
            lines.append(
                f"| {case_id} | {summary['baseline_median_seconds']:.6f} | "
                f"{summary['candidate_median_seconds']:.6f} | "
                f"{100.0 * summary['median_reduction_fraction']:.2f}% | "
                f"{'PASS' if case['physical_match'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Candidate globalization comparison",
            "",
            "| Case | Residual-decrease median (s) | Armijo median (s) | "
            "Failed-work change | Physics |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for case_id, case in result["convergence"].items():
        lines.append(
            f"| {case_id} | {seconds(case['residual_decrease_median_seconds'])} | "
            f"{seconds(case['armijo_median_seconds'])} | "
            f"{case['failed_work_change_fraction']:.2%} | "
            f"{'PASS' if case['physical_match'] else 'FAIL'} |"
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Performance promotion: **{decision['performance']}**",
            f"- Armijo promotion: **{decision['armijo']}**",
            f"- Completeness: **{decision['completeness']}**",
            "",
            "Historical evidence remains unchanged. This campaign does not replace "
            "the earlier registered 4.35% small-shell NO-GO.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coordinator(args: argparse.Namespace) -> int:
    campaign_started = time.perf_counter()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limits = manifest["resource_limits"]
    campaign_budget_seconds = float(limits["campaign_wall_budget_seconds"])
    campaign_deadline = campaign_started + campaign_budget_seconds
    provenance_path = args.build_provenance.resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    frozen_identity = _validate_campaign_identity(
        manifest=manifest,
        provenance=provenance,
        baseline_wheel=args.baseline_wheel,
        candidate_wheel=args.candidate_wheel,
    )
    sites, install_records, setup_failure = _install_frozen_wheels(
        install_root=args.install_root.resolve(),
        baseline_wheel=args.baseline_wheel,
        candidate_wheel=args.candidate_wheel,
        campaign_deadline=campaign_deadline,
    )
    timeout_seconds = float(limits["per_solve_timeout_seconds"])
    startup_timeout_seconds = float(limits["worker_startup_timeout_seconds"])
    memory_bytes = int(limits["process_tree_memory_bytes"])
    repeats = int(manifest["execution"]["performance_pairs"])
    convergence_repeats = int(manifest["execution"]["convergence_repeats"])
    log_dir = args.log_dir.resolve()
    result: dict[str, Any] = {
        "schema": "anyfesolver.nonlinear_static.representative_evidence",
        "version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": {
            **frozen_identity,
            "baseline_wheel": str(args.baseline_wheel.resolve()),
            "candidate_wheel": str(args.candidate_wheel.resolve()),
            "build_provenance": str(provenance_path),
            "build_provenance_sha256": _sha256(provenance_path),
            "installed_sites": install_records,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "thread_environment": THREAD_ENV,
        },
        "resource_limits": limits,
        "performance": {},
        "convergence": {},
    }
    if setup_failure is not None:
        result["setup_failure"] = setup_failure
        result["failures"] = [{"phase": "wheel_install", "detail": setup_failure}]
        result["decision"] = {
            "completeness": "FAIL",
            "performance": "NOT-RUN",
            "armijo": "NOT-RUN",
        }
        result["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_report(result, args.report)
        return 1

    def bounded_sample(*, site: Path, case: str, method: str, log: Path) -> dict[str, Any]:
        remaining = campaign_budget_seconds - (time.perf_counter() - campaign_started)
        if remaining <= 0.0:
            return {
                "ok": False,
                "terminal_reason": "campaign_budget",
                "exit_code": None,
                "worker_wall_seconds": 0.0,
                "worker_peak_tree_rss_bytes": 0,
                "stderr_log": str(log.resolve()),
            }
        return _run_bounded_worker(
            site=site,
            manifest=manifest_path,
            case=case,
            method=method,
            timeout_seconds=min(timeout_seconds, remaining),
            startup_timeout_seconds=min(startup_timeout_seconds, remaining),
            worker_budget_seconds=remaining,
            memory_bytes=memory_bytes,
            log_path=log,
        )
    failures: list[dict[str, Any]] = []
    installed_regressions = _run_installed_regressions(
        site=sites["candidate"],
        tests=list(manifest["required_regressions"]),
        timeout_seconds=min(timeout_seconds, campaign_budget_seconds),
        log_path=log_dir / "installed-regressions.log",
    )
    result["installed_regressions"] = installed_regressions
    if not installed_regressions["ok"]:
        failures.append(
            {"phase": "installed_regressions", "detail": installed_regressions}
        )
        result["failures"] = failures
        result["decision"] = {
            "completeness": "FAIL",
            "performance": "NOT-RUN",
            "armijo": "NOT-RUN",
        }
        result["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_report(result, args.report)
        return 1
    for spec in manifest["performance_cases"]:
        case_id = str(spec["id"])
        print(f"performance case: {case_id}", flush=True)
        samples: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
        pairs = []
        for pair_index in range(repeats):
            order = (
                ("baseline", "candidate")
                if pair_index % 2 == 0
                else ("candidate", "baseline")
            )
            pair: dict[str, Any] = {"pair": pair_index + 1, "order": list(order)}
            for label in order:
                site = sites[label]
                log = log_dir / f"performance.{case_id}.p{pair_index + 1}.{label}.log"
                print(f"  pair {pair_index + 1}/{repeats}: {label}", flush=True)
                sample = bounded_sample(
                    site=site,
                    case=case_id,
                    method="never",
                    log=log,
                )
                pair[label] = sample
                samples[label].append(sample)
                if not sample.get("ok"):
                    failures.append(
                        {"phase": "performance", "case": case_id, "label": label, "sample": sample}
                    )
            pair["physical_comparison"] = _compare_physics(
                pair["baseline"], pair["candidate"]
            )
            pairs.append(pair)
            if any(
                _resource_terminal(sample)
                for sample in (pair["baseline"], pair["candidate"])
            ):
                break
        complete = all(sample.get("ok") for values in samples.values() for sample in values)
        physical_match = all(
            pair["physical_comparison"].get("numerical_match", False)
            for pair in pairs
        )
        result["performance"][case_id] = {
            "spec": spec,
            "pairs": pairs,
            "complete": complete,
            "physical_match": physical_match,
            "summary": _case_summary(samples["baseline"], samples["candidate"])
            if complete
            else None,
        }

    specs = {str(spec["id"]): spec for spec in manifest["performance_cases"]}
    for case_id in manifest["convergence_case_ids"]:
        print(f"convergence case: {case_id}", flush=True)
        method_samples: dict[str, list[dict[str, Any]]] = {
            "always": [],
            "armijo": [],
        }
        pairs = []
        for pair_index in range(convergence_repeats):
            order = ("always", "armijo") if pair_index % 2 == 0 else ("armijo", "always")
            pair: dict[str, Any] = {"pair": pair_index + 1, "order": list(order)}
            for method in order:
                log = log_dir / f"convergence.{case_id}.p{pair_index + 1}.{method}.log"
                print(f"  pair {pair_index + 1}/{convergence_repeats}: {method}", flush=True)
                sample = bounded_sample(
                    site=sites["candidate"],
                    case=case_id,
                    method=method,
                    log=log,
                )
                pair[method] = sample
                method_samples[method].append(sample)
                if not _terminal_evidence_complete(sample):
                    failures.append(
                        {"phase": "convergence", "case": case_id, "label": method, "sample": sample}
                    )
            pair["physical_comparison"] = _compare_physics(
                pair["always"], pair["armijo"]
            )
            pairs.append(pair)
            if any(
                _resource_terminal(sample)
                for sample in (pair["always"], pair["armijo"])
            ):
                break
        complete = all(
            _terminal_evidence_complete(sample)
            for values in method_samples.values()
            for sample in values
        )
        oracle = result["performance"][case_id]["pairs"][0]["candidate"]
        physical_status = _convergence_physical_status(
            oracle=oracle,
            method_samples=method_samples,
        )
        physical_match = bool(physical_status["physical_match"])
        residual_failed = sum(
            int(sample["work"]["failed_work_units"])
            for sample in method_samples["always"]
            if sample.get("ok")
        )
        armijo_failed = sum(
            int(sample["work"]["failed_work_units"])
            for sample in method_samples["armijo"]
            if sample.get("ok")
        )
        failed_change = (
            (residual_failed - armijo_failed) / residual_failed
            if residual_failed > 0
            else 0.0
        )
        result["convergence"][case_id] = {
            "spec": specs[case_id],
            "pairs": pairs,
            "complete": complete,
            "physical_match": physical_match,
            "physical_evidence_basis": physical_status["basis"],
            "residual_decrease_median_seconds": _median(
                [sample for sample in method_samples["always"] if sample.get("ok")],
                "complete_route_wall_seconds",
            ) if all(sample.get("ok") for sample in method_samples["always"]) else None,
            "armijo_median_seconds": _median(
                [sample for sample in method_samples["armijo"] if sample.get("ok")],
                "complete_route_wall_seconds",
            ) if all(sample.get("ok") for sample in method_samples["armijo"]) else None,
            "residual_decrease_failed_work": residual_failed,
            "armijo_failed_work": armijo_failed,
            "failed_work_change_fraction": failed_change,
            "newly_solved_by_armijo": bool(
                all(sample.get("status") != "completed" for sample in method_samples["always"])
                and all(sample.get("status") == "completed" for sample in method_samples["armijo"])
            ),
        }

    performance_complete = all(
        case["complete"] and case["physical_match"]
        for case in result["performance"].values()
    )
    easy_id = str(manifest["acceptance"]["easy_control_case"])
    nonlinear_ids = [
        case_id for case_id in result["performance"] if case_id != easy_id
    ]
    nonlinear_reductions = [
        result["performance"][case_id]["summary"]["median_reduction_fraction"]
        for case_id in nonlinear_ids
        if result["performance"][case_id]["summary"] is not None
    ]
    representative_reduction = (
        float(statistics.median(nonlinear_reductions))
        if len(nonlinear_reductions) == len(nonlinear_ids)
        else float("nan")
    )
    easy_reduction = (
        result["performance"][easy_id]["summary"]["median_reduction_fraction"]
        if result["performance"][easy_id]["summary"] is not None
        else float("nan")
    )
    performance_pass = (
        performance_complete
        and representative_reduction >= float(manifest["acceptance"]["performance_median_reduction_fraction"])
        and easy_reduction >= -float(manifest["acceptance"]["maximum_easy_regression_fraction"])
    )
    armijo_decision = _adjudicate_armijo(
        convergence=result["convergence"],
        convergence_repeats=convergence_repeats,
        timeout_seconds=timeout_seconds,
        acceptance=manifest["acceptance"],
    )
    convergence_complete = bool(armijo_decision["complete"])
    result["decision"] = {
        "completeness": "PASS" if performance_complete and convergence_complete and not failures else "FAIL",
        "performance": "GO" if performance_pass else "NO-GO",
        "armijo": "GO" if armijo_decision["go"] else "NO-GO",
        "representative_nonlinear_median_reduction_fraction": representative_reduction,
        "easy_control_reduction_fraction": easy_reduction,
        "armijo_new_difficult_solves": armijo_decision["new_solves"],
        "armijo_failed_work_reduction_fraction": armijo_decision[
            "failed_work_reduction_fraction"
        ],
        "armijo_runtime_seconds": armijo_decision["armijo_runtime_seconds"],
        "residual_decrease_runtime_seconds": armijo_decision[
            "residual_decrease_runtime_seconds"
        ],
        "armijo_all_samples_completed": armijo_decision[
            "armijo_all_samples_completed"
        ],
        "armijo_runtime_not_worse": armijo_decision["runtime_not_worse"],
        "failed_work_inventory_complete": armijo_decision[
            "failed_work_inventory_complete"
        ],
    }
    result["failures"] = failures
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(result, args.report)
    return 0 if result["decision"]["completeness"] == "PASS" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--site", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--method", default="never")
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--build-provenance", type=Path)
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if args.site is None or args.case is None:
            raise SystemExit("worker requires --site and --case")
        return _worker_main(args)
    required = (
        "baseline_wheel",
        "candidate_wheel",
        "build_provenance",
        "install_root",
        "output",
        "report",
        "log_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    return _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
