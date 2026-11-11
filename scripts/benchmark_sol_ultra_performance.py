"""Run reproducible Sol Ultra benchmark cases and write JSON/Markdown reports.

The runner deliberately uses only public or existing benchmark-facing APIs.  It
does not select pass/fail outcomes from wall-clock thresholds; numerical status
and timing evidence are recorded separately for matched baseline/candidate
comparison.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy import sparse

from anyfesolver import nonlinear_performance, nonlinear_static
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.benchmarks import BENCHMARK_CASES
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.corotational import rotation_matrix_from_vector
from anyfesolver.dynamics import TransientConfig, solve_transient_newmark
from anyfesolver.elements import BeamElement, Element, create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.materials import Hill48Yield, OrthotropicMaterial
from anyfesolver.matrix_assembly import (
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from anyfesolver.mesh_gen import generate_beam_mesh, generate_simple_panel_mesh
from anyfesolver.nonlinear_performance_batch_c import (
    assemble_reduced_system,
    build_reduced_assembly_plan,
)
from anyfesolver.nonlinear_performance_bootstrap import (
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
    nonlinear_performance_status,
)
from anyfesolver.recovery import (
    RecoveryConfig,
    ResourceConfig,
    recover_element_stresses_with_report,
)
from anyfesolver.shell_sections import GeneralizedShellSection


SCHEMA_NAME = "anyfesolver.sol_ultra.benchmark"
SCHEMA_VERSION = 1
DEFAULT_REPORT_DIRECTORY = ROOT / "reports" / "performance"

# Keep every case on the same phase vocabulary, even when the current API does
# not expose a timer.  Unavailable phases remain explicit null measurements.
PHASE_NAMES = (
    "model_preparation",
    "constraint_plan_construction",
    "linear_K_assembly",
    "linear_M_assembly",
    "KG_assembly",
    "nonlinear_local_response",
    "constitutive_update",
    "state_packing",
    "state_commit",
    "state_materialization",
    "full_coordinate_scatter",
    "reduced_coordinate_scatter",
    "T.T @ F_projection",
    "T.T @ K @ T_projection",
    "contact_search",
    "contact_load_construction",
    "factorization",
    "linear_solve",
    "full_vector_reconstruction",
    "stress_recovery",
    "history_output_storage",
    "total_wall_time",
)

_TIMING_ALIASES: Dict[str, tuple[str, ...]] = {
    "model_preparation": ("model_preparation_seconds",),
    "constraint_plan_construction": ("constraint_plan_construction_seconds",),
    "linear_K_assembly": (
        "stiffness_assembly_seconds",
        "warm_assembly_seconds",
        "cold_assembly_seconds",
    ),
    "linear_M_assembly": ("mass_assembly_seconds",),
    "KG_assembly": ("geometric_assembly_seconds",),
    "nonlinear_local_response": (
        "nonlinear_local_response_seconds",
        "active_fast_path_seconds",
        "legacy_assembly_seconds",
    ),
    "constitutive_update": ("constitutive_update_seconds",),
    "state_packing": ("state_pack_seconds",),
    "state_commit": ("state_commit_seconds",),
    "state_materialization": ("state_materialization_seconds",),
    "full_coordinate_scatter": ("full_coordinate_scatter_seconds",),
    "reduced_coordinate_scatter": ("reduced_coordinate_scatter_seconds",),
    "T.T @ F_projection": ("force_projection_seconds",),
    "T.T @ K @ T_projection": ("tangent_projection_seconds",),
    "contact_search": ("contact_search_seconds",),
    "contact_load_construction": ("contact_load_construction_seconds",),
    "factorization": (
        "factorization_seconds",
        "first_factorization_seconds",
    ),
    "linear_solve": (
        "linear_solve_seconds",
        "solve_seconds",
        "solve_many_seconds",
        "backend_solve_seconds",
    ),
    "full_vector_reconstruction": ("full_vector_reconstruction_seconds",),
    "stress_recovery": (
        "stress_recovery_seconds",
        "serial_recovery_seconds",
        "threaded_recovery_seconds",
    ),
    "history_output_storage": ("history_output_storage_seconds",),
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    description: str
    categories: tuple[str, ...]
    builder: Callable[[], Dict[str, Any]]


class _SofteningSpringElement(Element):
    """Small one-DOF oracle used to exercise the production arc-length API."""

    def __init__(self, element_id: int, node_id: int, k: float = 1.0, c: float = 1.0):
        super().__init__(element_id, [node_id], "default")
        self.k = float(k)
        self.c = float(c)

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray([mesh.get_node(self.node_ids[0]).coords()], dtype=float)

    def compute_stiffness_matrix(self, mesh, material):
        matrix = np.eye(6, dtype=float)
        matrix[0, 0] = self.k
        return matrix

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        del mesh, material, state, num_layers
        displacement = np.asarray(u_elem, dtype=float)
        active = float(displacement[0])
        force = displacement.copy()
        force[0] = self.k * active - self.c * active**3
        stiffness = None
        if tangent:
            stiffness = np.eye(6, dtype=float)
            stiffness[0, 0] = self.k - 3.0 * self.c * active**2
        return force, stiffness, {"spring_displacement": active}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str, cwd: Path = ROOT) -> Optional[str]:
    command = ["git", "-c", f"safe.directory={cwd}", "-C", str(cwd), *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if sparse.issparse(value):
        return {
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "nnz": int(value.nnz),
            "format": value.getformat(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _memory_and_cpu_counts() -> Dict[str, Optional[int]]:
    values: Dict[str, Optional[int]] = {
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "process_cpu_affinity_count": None,
        "physical_memory_bytes": None,
        "available_memory_bytes": None,
    }
    try:
        import psutil

        values.update(
            {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "process_cpu_affinity_count": len(psutil.Process().cpu_affinity())
                if hasattr(psutil.Process(), "cpu_affinity")
                else None,
                "physical_memory_bytes": int(psutil.virtual_memory().total),
                "available_memory_bytes": int(psutil.virtual_memory().available),
            }
        )
    except Exception:
        pass
    return values


def _sibling_revision(module_name: str) -> Dict[str, Optional[str]]:
    try:
        module = importlib.import_module(module_name)
        source = Path(module.__file__).resolve()
    except (ImportError, AttributeError, TypeError):
        return {"path": None, "sha": None}
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        return {"path": str(source), "sha": None}
    return {"path": str(repository), "sha": _git("rev-parse", "HEAD", cwd=repository)}


def collect_environment() -> Dict[str, Any]:
    """Collect comparison-relevant environment and revision provenance."""

    try:
        from threadpoolctl import threadpool_info

        native_pools = threadpool_info()
    except ImportError:
        native_pools = []
    try:
        from anyfesolver.jit_compiler import jit_diagnostics

        jit = jit_diagnostics()
    except Exception as exc:  # diagnostics should not block benchmark execution
        jit = {"enabled": False, "disabled_reason": f"{type(exc).__name__}: {exc}"}

    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    merge_base = _git("merge-base", "HEAD", "origin/main") if origin_main else None
    return {
        "schema_name": "anyfesolver.sol_ultra.environment",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "revision": {
            "head_sha": head,
            "origin_main_sha": origin_main,
            "merge_base_sha": merge_base,
            "branch": _git("branch", "--show-current"),
            "source_root": str(ROOT),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            **_memory_and_cpu_counts(),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "numba",
                "llvmlite",
                "pypardiso",
                "mkl",
                "threadpoolctl",
                "pytest",
                "ANYfeSolver",
                "ANYmaterial",
                "ANYmesher",
                "ANYgeometry",
                "ANYfileio",
            )
        },
        "sibling_revisions": {
            "ANYmaterial": _sibling_revision("anymaterial"),
            "ANYmesher": _sibling_revision("anymesher"),
            "ANYgeometry": _sibling_revision("anygeometry"),
            "ANYfileio": _sibling_revision("anyfileio"),
        },
        "native_threadpools": _jsonable(native_pools),
        "jit": _jsonable(jit),
        "required_environment": {
            "PYPARDISO_MKL_RT": os.environ.get("PYPARDISO_MKL_RT"),
        },
    }


def _topology(model: FEModel) -> Dict[str, int]:
    return {
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "dofs": int(model.mesh.dof_manager.total_dofs),
    }


def _relative_vector_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _relative_sparse_error(candidate, reference) -> float:
    denominator = max(float(sparse.linalg.norm(reference)), 1.0)
    return float(sparse.linalg.norm(candidate - reference) / denominator)


def _weighted_constraint_transformation(
    model: FEModel,
    weighted_mpc_rows: int,
) -> sparse.csr_matrix:
    mesh = model.mesh
    total_dofs = int(mesh.dof_manager.total_dofs)
    minimum_x = min(float(node.x) for node in mesh.nodes.values())
    fixed = {
        int(dof)
        for node in mesh.nodes.values()
        if np.isclose(float(node.x), minimum_x)
        for dof in node.dofs
    }
    independent = np.asarray(
        [dof for dof in range(total_dofs) if dof not in fixed],
        dtype=np.intp,
    )
    independent_index = {
        int(dof): index for index, dof in enumerate(independent)
    }
    rows = independent.tolist()
    columns = [independent_index[int(dof)] for dof in independent]
    values = [1.0] * independent.size
    for offset, slave in enumerate(sorted(fixed)[: max(int(weighted_mpc_rows), 0)]):
        first = offset % independent.size
        second = (offset + 1) % independent.size
        rows.extend((slave, slave))
        columns.extend((first, second))
        values.extend((0.75, 0.25))
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(total_dofs, independent.size),
    )


def _weighted_mpc_assembly_case() -> Dict[str, Any]:
    start = time.perf_counter()
    model = generate_simple_panel_mesh(
        4.0,
        2.0,
        0.012,
        num_divisions_x=12,
        num_divisions_y=6,
    )
    rng = np.random.default_rng(20260811)
    model_preparation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    transformation = _weighted_constraint_transformation(model, 12)
    reduced_displacement = rng.normal(scale=2.0e-5, size=transformation.shape[1])
    displacement = np.asarray(transformation @ reduced_displacement, dtype=float).reshape(-1)
    constraint_matrix_seconds = time.perf_counter() - start

    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    if original is None:
        raise RuntimeError("The nonlinear performance layer did not retain its scalar oracle")
    force_reference, tangent_reference, _ = original(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )

    start = time.perf_counter()
    reference_force = np.asarray(transformation.T @ force_reference, dtype=float).reshape(-1)
    force_projection_seconds = time.perf_counter() - start
    start = time.perf_counter()
    reference_tangent = (transformation.T @ tangent_reference @ transformation).tocsr()
    tangent_projection_seconds = time.perf_counter() - start

    clear_nonlinear_assembly_cache(model)
    nonlinear_plan = get_nonlinear_assembly_plan(model, 5)
    start = time.perf_counter()
    reduced_plan = build_reduced_assembly_plan(nonlinear_plan, transformation)
    constraint_plan_seconds = constraint_matrix_seconds + (time.perf_counter() - start)
    start = time.perf_counter()
    force_fast, tangent_fast, _ = assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        displacement,
        {},
        tangent=True,
    )
    reduced_seconds = time.perf_counter() - start

    force_error = _relative_vector_error(force_fast, reference_force)
    tangent_error = _relative_sparse_error(tangent_fast, reference_tangent)
    return {
        "status": "completed" if force_error <= 1.0e-11 and tangent_error <= 1.0e-10 else "failed",
        "topology": {
            **_topology(model),
            "reduced_dofs": int(transformation.shape[1]),
            "weighted_mpc_rows": 12,
        },
        "timing": {
            "model_preparation_seconds": model_preparation_seconds,
            "constraint_plan_construction_seconds": constraint_plan_seconds,
            "force_projection_seconds": force_projection_seconds,
            "tangent_projection_seconds": tangent_projection_seconds,
            "reduced_coordinate_scatter_seconds": reduced_seconds,
        },
        "results": {
            "relative_force_error": force_error,
            "relative_tangent_error": tangent_error,
        },
        "diagnostics": {
            "nonlinear_plan": nonlinear_plan.diagnostics(),
            "reduced_plan": reduced_plan.diagnostics(),
        },
    }


def _orthotropic_material(*, plastic: bool) -> OrthotropicMaterial:
    hill = None
    if plastic:
        hill = Hill48Yield(
            X=300.0e6,
            Y=240.0e6,
            Z=270.0e6,
            S12=130.0e6,
            S13=140.0e6,
            S23=120.0e6,
        )
    return OrthotropicMaterial(
        name="steel",
        elastic_modulus_1=150.0e9,
        elastic_modulus_2=90.0e9,
        elastic_modulus_3=70.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=40.0e9,
        shear_modulus_13=35.0e9,
        shear_modulus_23=30.0e9,
        density=1600.0,
        hill_yield=hill,
        hardening_curve=None,
    )


def _orthotropic_elastic_shell_case() -> Dict[str, Any]:
    start = time.perf_counter()
    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, 8, 4)
    model.register_material(_orthotropic_material(plastic=False))
    angles = (0.0, 30.0, 45.0, 90.0)
    for index, element in enumerate(model.mesh.elements.values()):
        element.material_angle_deg = angles[index % len(angles)]
    model.bump_revision("material")
    model_preparation_seconds = time.perf_counter() - start
    stiffness, stiffness_info = assemble_stiffness_matrix(model)
    mass, mass_info = assemble_mass_matrix(model)
    symmetric_error = float(
        sparse.linalg.norm(stiffness - stiffness.T)
        / max(float(sparse.linalg.norm(stiffness)), 1.0)
    )
    return {
        "status": "completed" if symmetric_error <= 1.0e-12 else "failed",
        "topology": _topology(model),
        "timing": {
            "model_preparation_seconds": model_preparation_seconds,
            "stiffness_assembly_seconds": float(stiffness_info.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass_info.get("assembly_time", 0.0)),
        },
        "results": {
            "stiffness_norm": float(sparse.linalg.norm(stiffness)),
            "mass_norm": float(sparse.linalg.norm(mass)),
            "relative_symmetry_error": symmetric_error,
        },
        "diagnostics": stiffness_info.get("diagnostics", {}),
    }


def _hill48_shell_case() -> Dict[str, Any]:
    start = time.perf_counter()
    model = generate_simple_panel_mesh(1.5, 1.0, 0.01, 4, 3)
    model.register_material(_orthotropic_material(plastic=True))
    for index, element in enumerate(model.mesh.elements.values()):
        element.material_angle_deg = (0.0, 30.0, 45.0, 90.0)[index % 4]
    model.bump_revision("material")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 0.0040 * float(node.x)
        displacement[node.dofs[1]] = -0.0005 * float(node.y)
    model_preparation_seconds = time.perf_counter() - start

    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    if original is None:
        raise RuntimeError("The nonlinear performance layer did not retain its scalar oracle")
    force_reference, tangent_reference, states_reference = original(
        model, displacement, {}, 5, tangent=True
    )
    start = time.perf_counter()
    force_active, tangent_active, states_active = nonlinear_static._assemble_nonlinear_system(
        model, displacement, {}, 5, tangent=True
    )
    active_seconds = time.perf_counter() - start
    force_error = _relative_vector_error(force_active, force_reference)
    tangent_error = _relative_sparse_error(tangent_active, tangent_reference)

    def maximum_alpha(states: Mapping[int, Any]) -> float:
        maxima = []
        for state in states.values():
            if isinstance(state, Mapping) and "alpha" in state:
                maxima.append(float(np.max(np.asarray(state["alpha"], dtype=float))))
        return max(maxima, default=0.0)

    alpha_reference = maximum_alpha(states_reference)
    alpha_active = maximum_alpha(states_active)
    alpha_error = abs(alpha_active - alpha_reference)
    passed = (
        force_error <= 1.0e-11
        and tangent_error <= 1.0e-10
        and alpha_error <= 1.0e-12
    )
    return {
        "status": "completed" if passed else "failed",
        "topology": _topology(model),
        "timing": {
            "model_preparation_seconds": model_preparation_seconds,
            "nonlinear_local_response_seconds": active_seconds,
        },
        "results": {
            "relative_force_error": force_error,
            "relative_tangent_error": tangent_error,
            "maximum_alpha_reference": alpha_reference,
            "maximum_alpha_active": alpha_active,
            "absolute_alpha_error": alpha_error,
            "state_element_count": len(states_active),
        },
        "performance_layer": nonlinear_performance_status(),
    }


def _generalized_shell_section() -> GeneralizedShellSection:
    thickness = 0.02
    elastic_modulus = 70.0e9
    poisson_ratio = 0.25
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    plane = elastic_modulus / (1.0 - poisson_ratio**2) * np.asarray(
        [
            [1.0, poisson_ratio, 0.0],
            [poisson_ratio, 1.0, 0.0],
            [0.0, 0.0, (1.0 - poisson_ratio) / 2.0],
        ]
    )
    return GeneralizedShellSection(
        A=thickness * plane,
        B=np.asarray(
            [[1.0e5, 2.0e4, -1.0e4], [1.5e4, -8.0e4, 1.0e4], [0.5e4, -0.4e4, 4.0e4]],
            dtype=float,
        ),
        D=thickness**3 / 12.0 * plane,
        As=(5.0 / 6.0) * thickness * shear_modulus * np.eye(2),
        mass_per_area=54.0,
        rotary_inertia_per_area=0.0018,
        name="benchmark_coupled_ABDAs",
    )


def _generalized_shell_case() -> Dict[str, Any]:
    start = time.perf_counter()
    model = generate_simple_panel_mesh(2.0, 1.0, 0.02, 6, 3)
    section = _generalized_shell_section()
    for element in model.mesh.elements.values():
        element.shell_section = section
    model.bump_revision("material")
    displacement = np.random.default_rng(818).normal(
        scale=1.0e-5,
        size=model.mesh.dof_manager.total_dofs,
    )
    model_preparation_seconds = time.perf_counter() - start
    stiffness, stiffness_info = assemble_stiffness_matrix(model)
    mass, mass_info = assemble_mass_matrix(model)

    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    if original is None:
        raise RuntimeError("The nonlinear performance layer did not retain its scalar oracle")
    force_reference, tangent_reference, _ = original(
        model, displacement, {}, 5, tangent=True
    )
    start = time.perf_counter()
    force_active, tangent_active, _ = nonlinear_static._assemble_nonlinear_system(
        model, displacement, {}, 5, tangent=True
    )
    nonlinear_seconds = time.perf_counter() - start
    force_error = _relative_vector_error(force_active, force_reference)
    tangent_error = _relative_sparse_error(tangent_active, tangent_reference)
    return {
        "status": "completed" if force_error <= 1.0e-11 and tangent_error <= 1.0e-10 else "failed",
        "topology": _topology(model),
        "timing": {
            "model_preparation_seconds": model_preparation_seconds,
            "stiffness_assembly_seconds": float(stiffness_info.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass_info.get("assembly_time", 0.0)),
            "nonlinear_local_response_seconds": nonlinear_seconds,
        },
        "results": {
            "relative_force_error": force_error,
            "relative_tangent_error": tangent_error,
            "stiffness_norm": float(sparse.linalg.norm(stiffness)),
            "mass_norm": float(sparse.linalg.norm(mass)),
            "nonzero_B_coupling": bool(np.any(section.B != 0.0)),
        },
        "diagnostics": stiffness_info.get("diagnostics", {}),
    }


def _single_corotational_model(kind: str) -> FEModel:
    model = FEModel(f"benchmark_corotational_{kind}")
    model.add_material("steel", 210.0e9, 0.0, density=7850.0)
    if kind == "shell":
        for node_id, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), 1):
            model.add_node(node_id, float(x), float(y), 0.0)
        model.add_element(
            1,
            create_element(
                "shell", 1, [1, 2, 3, 4], "steel", thickness=0.01
            ),
        )
    else:
        model.add_node(1, 0.0, 0.0, 0.0)
        model.add_node(2, 1.0, 0.0, 0.0)
        model.add_element(
            1,
            BeamElement(
                1,
                [1, 2],
                "steel",
                {"area": 1.0e-3, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
            ),
        )
    return model


def _rigid_rotation_displacement(model: FEModel) -> np.ndarray:
    axis = np.asarray((0.2, 0.4, 1.0), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = math.radians(75.0)
    rotation = rotation_matrix_from_vector(angle * axis)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        coordinate = np.asarray((node.x, node.y, node.z), dtype=float)
        displacement[np.asarray(node.dofs[:3], dtype=np.intp)] = rotation @ coordinate - coordinate
        displacement[np.asarray(node.dofs[3:], dtype=np.intp)] = angle * axis
    return displacement


def _corotational_case(kind: str) -> Dict[str, Any]:
    start = time.perf_counter()
    model = _single_corotational_model(kind)
    displacement = _rigid_rotation_displacement(model)
    model_preparation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    force_corotational, tangent_corotational, _ = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=True,
        kinematics="corotational",
        corotational_tangent="rotated",
    )
    response_seconds = time.perf_counter() - start
    force_von_karman, _, _ = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=False,
        kinematics="von_karman",
    )
    scale = max(float(np.linalg.norm(force_von_karman)), 210.0e9 * 1.0e-5)
    residual_ratio = float(np.linalg.norm(force_corotational) / scale)
    return {
        "status": "completed" if residual_ratio <= 1.0e-9 else "failed",
        "topology": _topology(model),
        "timing": {
            "model_preparation_seconds": model_preparation_seconds,
            "nonlinear_local_response_seconds": response_seconds,
        },
        "results": {
            "rigid_rotation_degrees": 75.0,
            "corotational_force_norm": float(np.linalg.norm(force_corotational)),
            "von_karman_force_norm": float(np.linalg.norm(force_von_karman)),
            "scaled_rigid_rotation_residual": residual_ratio,
            "tangent_norm": float(sparse.linalg.norm(tangent_corotational)),
        },
    }


def _arc_length_case() -> Dict[str, Any]:
    model = FEModel("benchmark_softening_spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, _SofteningSpringElement(1, 1))
    model.add_boundary_condition(
        BoundaryCondition(
            "one_dof",
            [1],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("unit_reference")
    load.add_nodal_load(1, load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    control = ArcLengthControl(
        initial_load_increment=0.02,
        minimum_load_increment=1.0e-5,
        maximum_load_increment=0.05,
        target_iterations=5,
        max_steps=120,
        stop_after_peak_steps=5,
        peak_drop_tolerance=1.0e-4,
    )
    result = solve_static_arc_length(
        model,
        load,
        control=control,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
    )
    exact_peak = 2.0 / (3.0 * math.sqrt(3.0))
    relative_error = abs(float(result.peak_load_factor or 0.0) - exact_peak) / exact_peak
    return {
        "status": "completed"
        if result.status == "peak_confirmed" and relative_error <= 0.03
        else "failed",
        "topology": _topology(model),
        "results": {
            "solver_status": result.status,
            "step_count": len(result.steps),
            "peak_load_factor": result.peak_load_factor,
            "exact_peak_load_factor": exact_peak,
            "relative_peak_error": relative_error,
        },
        "diagnostics": result.info,
    }


def _selected_transient_case() -> Dict[str, Any]:
    model = generate_beam_mesh(
        4.0,
        num_divisions=20,
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    # ``generate_beam_mesh`` preserves its historical massless default; a
    # transient benchmark must opt into physical density explicitly.
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    tip_node_id = max(model.mesh.nodes)
    load = LoadCase("selected_transient_tip")
    load.add_nodal_load(tip_node_id, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
    recovery = RecoveryConfig(
        node_ids=[tip_node_id],
        element_ids=[],
        include_displacements=True,
        include_stresses=False,
        include_reactions=False,
        history_mode="selected",
        store_full_histories=False,
    )
    result = solve_transient_newmark(
        model,
        TransientConfig(
            dt=0.001,
            t_end=0.5,
            save_every=5,
            output_nodes=[tip_node_id],
            output_elements=[],
            recovery=recovery,
            resource_config=ResourceConfig(solver_threads=1),
        ),
        base_load_case=load,
    )
    diagnostics = result.diagnostics
    stiffness = diagnostics.get("stiffness", {})
    mass = diagnostics.get("mass", {})
    factorization = diagnostics.get("effective_stiffness_factorization", {})
    return {
        "status": "completed" if result.status == "completed" else "failed",
        "topology": _topology(model),
        "timing": {
            "stiffness_assembly_seconds": float(stiffness.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass.get("assembly_time", 0.0)),
            "factorization_seconds": float(factorization.get("factorization_time", 0.0)),
        },
        "results": {
            "solver_status": result.status,
            "num_steps": int(diagnostics.get("num_steps", 0)),
            "num_saved_steps": int(diagnostics.get("num_saved_steps", 0)),
            "history_storage_mode": result.history_storage_mode,
            "saved_displacement_shape": list(result.displacements.shape),
            "saved_tip_history_shape": list(result.node_histories[tip_node_id].shape),
            "peak_displacement": float(result.peak_displacement),
        },
        "diagnostics": {
            "factorization_count": diagnostics.get("factorization_count"),
            "solve_count": diagnostics.get("solve_count"),
            "num_reduced_dofs": diagnostics.get("num_reduced_dofs"),
            "history_dof_indices": diagnostics.get("history_dof_indices"),
        },
    }


def _large_recovery_case() -> Dict[str, Any]:
    model = generate_simple_panel_mesh(4.0, 2.0, 0.01, 20, 10)
    displacement = np.random.default_rng(20260811).normal(
        scale=2.0e-6,
        size=model.mesh.dof_manager.total_dofs,
    )
    results, report = recover_element_stresses_with_report(
        model,
        displacement,
        RecoveryConfig(components=["von_mises"]),
        resource_config=ResourceConfig(recovery_threads=1),
    )
    finite = all(
        np.all(np.isfinite(np.asarray(state.get("von_mises", 0.0), dtype=float)))
        for state in results.values()
    )
    return {
        "status": "completed" if finite and len(results) == model.mesh.num_elements else "failed",
        "topology": _topology(model),
        "timing": {"stress_recovery_seconds": float(report.elapsed_seconds)},
        "results": {
            "recovered_element_count": len(results),
            "all_values_finite": finite,
        },
        "resources": report.to_dict(),
    }


CASE_SPECS: tuple[CaseSpec, ...] = (
    CaseSpec(
        "isotropic_s4_nonlinear_plate",
        "Isotropic S4 nonlinear plate",
        "Scalar-oracle parity and active persistent assembly on an isotropic S4 plate.",
        ("nonlinear", "shell", "isotropic"),
        BENCHMARK_CASES["nonlinear_assembly"],
    ),
    CaseSpec(
        "weighted_mpc_panel",
        "Weighted-MPC panel",
        "Direct reduced assembly with twelve synthetic weighted MPC rows.",
        ("nonlinear", "constraints", "reduced_assembly"),
        _weighted_mpc_assembly_case,
    ),
    CaseSpec(
        "orthotropic_elastic_s4_plate",
        "Orthotropic elastic S4 plate",
        "Mixed material angles through the qualified orthotropic shell path.",
        ("linear", "shell", "orthotropic"),
        _orthotropic_elastic_shell_case,
    ),
    CaseSpec(
        "hill48_plastic_s4_plate",
        "Hill-48 plastic S4 plate",
        "Yielded orthotropic shell response compared with the retained scalar oracle.",
        ("nonlinear", "shell", "plasticity", "hill48"),
        _hill48_shell_case,
    ),
    CaseSpec(
        "generalized_coupled_s4_plate",
        "Generalized coupled A/B/D/As S4 plate",
        "Pre-integrated section with nonzero B coupling and resultants-only semantics.",
        ("linear", "nonlinear", "shell", "generalized_section"),
        _generalized_shell_case,
    ),
    CaseSpec(
        "rotated_corotational_shell",
        "Rotated-corotational shell",
        "Large rigid shell rotation using the production rotated tangent.",
        ("nonlinear", "shell", "corotational"),
        lambda: _corotational_case("shell"),
    ),
    CaseSpec(
        "rotated_corotational_beam",
        "Rotated-corotational beam",
        "Large rigid beam rotation using the production rotated tangent.",
        ("nonlinear", "beam", "corotational"),
        lambda: _corotational_case("beam"),
    ),
    CaseSpec(
        "arc_length_post_buckling_oracle",
        "Arc-length post-peak oracle",
        "Softening one-DOF path with an analytical limit point.",
        ("nonlinear", "arc_length", "post_peak"),
        _arc_length_case,
    ),
    CaseSpec(
        "nonlinear_impact_damage",
        "Nonlinear impact with damage",
        "Sphere-impact and fracture/damage infrastructure smoke workload.",
        ("nonlinear", "impact", "damage"),
        BENCHMARK_CASES["fracture_damage"],
    ),
    CaseSpec(
        "repeated_multi_rhs_static",
        "Repeated multi-RHS static workflow",
        "Three load cases sharing the qualified solve-many path.",
        ("linear", "repeated_analysis", "factorization"),
        BENCHMARK_CASES["multi_rhs_static"],
    ),
    CaseSpec(
        "beam_column_buckling",
        "Beam-column buckling",
        "Linear K/KG assembly and eigenvalue buckling follow-on analysis.",
        ("linear", "buckling", "repeated_analysis"),
        BENCHMARK_CASES["beam_column_buckling"],
    ),
    CaseSpec(
        "long_transient_selected_output",
        "Long transient with selected output",
        "Five hundred Newmark steps with only the tip-node history retained.",
        ("transient", "selected_output"),
        _selected_transient_case,
    ),
    CaseSpec(
        "large_stress_recovery",
        "Large S4 stress recovery",
        "Serial deterministic von-Mises recovery for a 200-element plate.",
        ("recovery", "shell"),
        _large_recovery_case,
    ),
    CaseSpec(
        "factorization_cache_reuse",
        "Factorization cache reuse",
        "Same-signature reuse and changed-matrix invalidation.",
        ("linear", "factorization", "cache"),
        BENCHMARK_CASES["factorization_reuse"],
    ),
    CaseSpec(
        "linear_shell_K_M_assembly",
        "Linear shell K/M assembly",
        "Representative linear shell stiffness, mass, and pressure-load setup.",
        ("linear", "shell", "assembly"),
        BENCHMARK_CASES["shell_assembly"],
    ),
    CaseSpec(
        "selective_recovery_consistency",
        "Selective recovery consistency",
        "Serial/threaded selected-component parity and resource diagnostics.",
        ("recovery", "threading"),
        BENCHMARK_CASES["selective_recovery"],
    ),
)

CASE_BY_ID = {spec.case_id: spec for spec in CASE_SPECS}
SUITES: Dict[str, tuple[str, ...]] = {
    "smoke": (
        "isotropic_s4_nonlinear_plate",
        "weighted_mpc_panel",
        "hill48_plastic_s4_plate",
        "long_transient_selected_output",
        "large_stress_recovery",
    ),
    "full": tuple(spec.case_id for spec in CASE_SPECS),
}


def _process_memory() -> Dict[str, Optional[int]]:
    values = {
        "rss_bytes": None,
        "process_peak_rss_bytes": None,
    }
    try:
        import psutil

        info = psutil.Process().memory_info()
        values["rss_bytes"] = int(info.rss)
        peak = getattr(info, "peak_wset", None)
        values["process_peak_rss_bytes"] = None if peak is None else int(peak)
    except Exception:
        pass
    return values


def _invoke(builder: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    before = _process_memory()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        payload = builder()
        exception = None
    except Exception as exc:  # benchmark suite must preserve partial evidence
        payload = {
            "status": "error",
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exception = exc
    wall_seconds = time.perf_counter() - start
    _current_python_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = _process_memory()
    observed_status = str(payload.get("status", "completed"))
    return {
        "status": "error" if exception is not None else observed_status,
        "wall_seconds": float(wall_seconds),
        "python_peak_bytes": int(peak_python_bytes),
        "process_rss_before_bytes": before["rss_bytes"],
        "process_rss_after_bytes": after["rss_bytes"],
        "process_peak_rss_bytes": after["process_peak_rss_bytes"],
        "payload": _jsonable(payload),
    }


def _numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "median": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "samples": samples,
        "median": float(statistics.median(samples)),
        "minimum": float(min(samples)),
        "maximum": float(max(samples)),
        "mean": float(statistics.fmean(samples)),
    }


def _phase_value(payload: Mapping[str, Any], phase: str) -> tuple[Optional[float], Optional[str]]:
    explicit = payload.get("phase_seconds", {})
    if isinstance(explicit, Mapping) and phase in explicit:
        value = explicit[phase]
        if isinstance(value, (int, float)):
            return float(value), f"phase_seconds.{phase}"
    timing = payload.get("timing", {})
    if isinstance(timing, Mapping):
        for key in _TIMING_ALIASES.get(phase, ()):
            value = timing.get(key)
            if isinstance(value, (int, float)):
                return float(value), f"timing.{key}"
    return None, None


def _phase_report(invocations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    phases: Dict[str, Any] = {}
    for phase in PHASE_NAMES:
        if phase == "total_wall_time":
            values = [float(invocation["wall_seconds"]) for invocation in invocations]
            sources = ["runner.wall_seconds"]
        else:
            values = []
            sources = []
            for invocation in invocations:
                payload = invocation.get("payload", {})
                if not isinstance(payload, Mapping):
                    continue
                value, source = _phase_value(payload, phase)
                if value is not None:
                    values.append(value)
                if source is not None and source not in sources:
                    sources.append(source)
        summary = _numeric_summary(values)
        phases[phase] = {
            "available": bool(values),
            "samples_seconds": summary["samples"],
            "median_seconds": summary["median"],
            "minimum_seconds": summary["minimum"],
            "maximum_seconds": summary["maximum"],
            "mean_seconds": summary["mean"],
            "sources": sources,
        }
    return phases


def run_case(spec: CaseSpec, repeats: int) -> Dict[str, Any]:
    cold = _invoke(spec.builder)
    warm = [_invoke(spec.builder) for _ in range(int(repeats))]
    statuses = [cold["status"], *(invocation["status"] for invocation in warm)]
    failed = any(str(status).lower() in {"failed", "error"} for status in statuses)
    warm_wall = _numeric_summary([float(invocation["wall_seconds"]) for invocation in warm])
    warm_python_peak = _numeric_summary(
        [float(invocation["python_peak_bytes"]) for invocation in warm]
    )
    warm_process_peak = _numeric_summary(
        [
            float(invocation["process_peak_rss_bytes"])
            for invocation in warm
            if invocation["process_peak_rss_bytes"] is not None
        ]
    )
    representative = warm[-1]["payload"] if warm else cold["payload"]
    return {
        "case_id": spec.case_id,
        "title": spec.title,
        "description": spec.description,
        "categories": list(spec.categories),
        "status": "failed" if failed else "completed",
        "observed_statuses": statuses,
        "measurements": {
            "cold": {
                key: value
                for key, value in cold.items()
                if key != "payload"
            },
            "warm": {
                "repeats": int(repeats),
                "wall_seconds": warm_wall,
                "python_peak_bytes": warm_python_peak,
                "process_peak_rss_bytes": warm_process_peak,
            },
        },
        "phases": _phase_report(warm if warm else [cold]),
        "cold_payload": cold["payload"],
        "representative_warm_payload": representative,
    }


def run_suite(
    case_ids: Sequence[str],
    *,
    suite_name: str,
    repeats: int,
    label: str,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    environment = collect_environment()
    suite_start = time.perf_counter()
    cases = []
    for case_id in case_ids:
        case = run_case(CASE_BY_ID[case_id], repeats)
        cases.append(case)
        if fail_fast and case["status"] == "failed":
            break
    suite_seconds = time.perf_counter() - suite_start
    failed_case_ids = [case["case_id"] for case in cases if case["status"] == "failed"]
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "report_kind": str(label),
        "generated_at_utc": _utc_now(),
        "revision": environment["revision"],
        "suite": {
            "name": suite_name,
            "requested_case_ids": list(case_ids),
            "completed_case_ids": [case["case_id"] for case in cases],
            "repeats": int(repeats),
            "total_wall_seconds": float(suite_seconds),
        },
        "measurement_policy": {
            "cold": "First in-process invocation after runner initialization.",
            "warm": "Independent model rebuilds in the same interpreter after the cold invocation.",
            "warm_statistic": "median",
            "performance_gate": "informational; no wall-clock pass/fail threshold is applied",
            "memory": "tracemalloc peak is per invocation; process peak RSS is process-lifetime peak when exposed by the OS",
        },
        "phase_schema": list(PHASE_NAMES),
        "environment": environment,
        "performance_layer": nonlinear_performance_status(),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "completed_count": len(cases) - len(failed_case_ids),
            "failed_count": len(failed_case_ids),
            "failed_case_ids": failed_case_ids,
        },
        "known_limitations": [
            "Use matched revisions, native libraries, and thread policies for timing comparisons.",
            "Cold measurements are first-in-process measurements, not fresh-process startup measurements.",
            "Unavailable phase timers remain explicit null values; total wall time is always measured.",
            "Process peak RSS is cumulative on platforms that do not expose a resettable per-case peak.",
            "Representative workloads are intentionally bounded and do not replace numerical qualification tests.",
        ],
    }


def _seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.6f}s"


def _bytes(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GiB"


def render_markdown(report: Mapping[str, Any]) -> str:
    revision = report.get("revision", {})
    suite = report.get("suite", {})
    summary = report.get("summary", {})
    environment = report.get("environment", {}).get("runtime", {})
    lines = [
        "# ANYfeSolver Sol Ultra Performance Report",
        "",
        f"- Report kind: `{report.get('report_kind', 'unknown')}`",
        f"- Generated: {report.get('generated_at_utc', 'unknown')}",
        f"- Revision: `{revision.get('head_sha') or 'unknown'}`",
        f"- Suite: `{suite.get('name', 'unknown')}` ({suite.get('repeats', 0)} warm repeats)",
        f"- Python: {environment.get('python_version', 'unknown')}",
        f"- CPU: {environment.get('cpu', 'unknown')}",
        f"- Cases: {summary.get('completed_count', 0)} completed, {summary.get('failed_count', 0)} failed",
        "",
        "## Case summary",
        "",
        "| Case | Status | Cold wall | Warm median | Cold / warm | Warm Python peak |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report.get("cases", []):
        cold = case.get("measurements", {}).get("cold", {})
        warm = case.get("measurements", {}).get("warm", {})
        cold_wall = cold.get("wall_seconds")
        warm_wall = warm.get("wall_seconds", {}).get("median")
        ratio = (
            float(cold_wall) / float(warm_wall)
            if isinstance(cold_wall, (int, float))
            and isinstance(warm_wall, (int, float))
            and warm_wall > 0.0
            else None
        )
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}x"
        python_peak = warm.get("python_peak_bytes", {}).get("median")
        lines.append(
            f"| `{case.get('case_id')}` | {case.get('status')} | "
            f"{_seconds(cold_wall)} | {_seconds(warm_wall)} | {ratio_text} | {_bytes(python_peak)} |"
        )

    context = report.get("campaign_context", {})
    if isinstance(context, Mapping) and context:
        context_revision = context.get("revision", {})
        qualification = context.get("qualification", {})
        full_suite = qualification.get("full_test_suite", {}) if isinstance(qualification, Mapping) else {}
        lines.extend(["", "## Baseline qualification", ""])
        if isinstance(context_revision, Mapping) and context_revision:
            lines.extend(
                [
                    f"- Immutable `performance_2`: `{context_revision.get('initial_performance_2_sha', 'unknown')}`",
                    f"- Contemporaneous `origin/main`: `{context_revision.get('origin_main_sha', 'unknown')}`",
                    f"- Merge-base: `{context_revision.get('merge_base_sha', 'unknown')}`",
                ]
            )
        if isinstance(full_suite, Mapping) and full_suite:
            lines.append(
                f"- Full test suite: **{full_suite.get('passed', 0)} passed in "
                f"{float(full_suite.get('wall_seconds', 0.0)):.2f}s**"
            )
            if full_suite.get("command"):
                lines.append(f"- Full-suite command: `{full_suite.get('command')}`")
        incidents = context.get("setup_incidents", [])
        if incidents:
            lines.extend(
                [
                    "",
                    "### Setup incidents (not numerical regressions)",
                    "",
                    "| Stage | Classification | Outcome | Causes |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for incident in incidents:
                causes = "; ".join(str(value) for value in incident.get("causes", []))
                lines.append(
                    f"| `{incident.get('stage')}` | {incident.get('classification')} | "
                    f"{incident.get('outcome')} | {causes} |"
                )
        assembly_benchmarks = context.get("benchmarks", [])
        if assembly_benchmarks:
            lines.extend(
                [
                    "",
                    "### Mandated nonlinear assembly benchmarks",
                    "",
                    "| Case | Legacy median | Persistent median | Direct median | Direct vs legacy |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for benchmark in assembly_benchmarks:
                variants = benchmark.get("variants", {})
                legacy = variants.get("legacy_full_then_projection", {})
                persistent = variants.get("persistent_full_then_projection", {})
                direct = variants.get("direct_reduced", {})
                lines.append(
                    f"| `{benchmark.get('case_id')}` | {_seconds(legacy.get('median_seconds'))} | "
                    f"{_seconds(persistent.get('median_seconds'))} | {_seconds(direct.get('median_seconds'))} | "
                    f"{float(direct.get('speedup_vs_legacy', 0.0)):.3f}x |"
                )
            lines.extend(["", "Reproduction commands:", ""])
            for benchmark in assembly_benchmarks:
                lines.append(f"- `{benchmark.get('command')}`")

    lines.extend(["", "## Phase coverage", ""])
    for case in report.get("cases", []):
        available = [
            f"{name}={_seconds(phase.get('median_seconds'))}"
            for name, phase in case.get("phases", {}).items()
            if phase.get("available")
        ]
        lines.append(f"### {case.get('title', case.get('case_id'))}")
        lines.append("")
        lines.append(", ".join(available) if available else "No phase timers were exposed.")
        payload = case.get("representative_warm_payload", {})
        results = payload.get("results", {}) if isinstance(payload, Mapping) else {}
        if results:
            lines.append("")
            lines.append(
                "Correctness/result metrics: `"
                + json.dumps(_jsonable(results), sort_keys=True, separators=(",", ":"))
                + "`"
            )
        lines.append("")

    lines.extend(["## Known limitations", ""])
    for limitation in report.get("known_limitations", []):
        lines.append(f"- {limitation}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: Mapping[str, Any], json_path: Path, markdown_path: Optional[Path]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _selected_case_ids(suite_name: str, explicit: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(explicit)
    if not requested:
        return SUITES[suite_name]
    unknown = sorted(set(requested) - set(CASE_BY_ID))
    if unknown:
        raise ValueError(f"Unknown benchmark case(s): {', '.join(unknown)}")
    return requested


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=[],
        help="Run a named case instead of the suite; may be repeated.",
    )
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    parser.add_argument("--label", default="final")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--environment-output", type=Path, default=None)
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="Optional JSON evidence to preserve under campaign_context.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--render-existing",
        type=Path,
        default=None,
        help="Render Markdown from an existing JSON report without rerunning cases.",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        print(
            json.dumps(
                [
                    {
                        "case_id": spec.case_id,
                        "title": spec.title,
                        "categories": list(spec.categories),
                    }
                    for spec in CASE_SPECS
                ],
                indent=2,
            )
        )
        return 0

    if args.render_existing is not None:
        report = json.loads(args.render_existing.read_text(encoding="utf-8"))
        markdown = args.markdown or args.render_existing.with_suffix(".md")
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "completed",
                    "input": str(args.render_existing),
                    "markdown": str(markdown),
                },
                indent=2,
            )
        )
        return 0

    case_ids = _selected_case_ids(args.suite, args.cases)
    if not install_nonlinear_performance_optimizations():
        parser.error(
            "The nonlinear performance layer is disabled. Unset FE_SOLVER_DISABLE_FAST_NL "
            "to retain the scalar oracle and benchmark the active layer."
        )

    output = args.output or DEFAULT_REPORT_DIRECTORY / f"sol_ultra_{args.label}.json"
    if args.no_markdown:
        markdown = None
    else:
        markdown = args.markdown or output.with_suffix(".md")
    report = run_suite(
        case_ids,
        suite_name=args.suite if not args.cases else "selected",
        repeats=args.repeats,
        label=args.label,
        fail_fast=args.fail_fast,
    )
    if args.context is not None:
        campaign_context = json.loads(
            args.context.read_text(encoding="utf-8")
        )
        if (
            campaign_context.get("schema_name") == SCHEMA_NAME
            and isinstance(campaign_context.get("campaign_context"), Mapping)
        ):
            campaign_context = campaign_context["campaign_context"]
        report["campaign_context"] = campaign_context
    write_report(report, output, markdown)
    if args.environment_output is not None:
        args.environment_output.parent.mkdir(parents=True, exist_ok=True)
        args.environment_output.write_text(
            json.dumps(
                _jsonable(report["environment"]),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "completed" if report["summary"]["failed_count"] == 0 else "failed",
                "suite": report["suite"]["name"],
                "case_count": report["summary"]["case_count"],
                "failed_case_ids": report["summary"]["failed_case_ids"],
                "output": str(output),
                "markdown": None if markdown is None else str(markdown),
            },
            indent=2,
        )
    )
    return 0 if report["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
