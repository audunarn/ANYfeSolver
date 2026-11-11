"""Capture and independently compare bounded Sol Ultra numerical artifacts.

The harness deliberately lives outside the production package.  It exercises
public solver workflows plus a small number of stable scalar-oracle hooks, and
stores the numerical vectors/matrices needed for a real relative-norm
comparison.  Checksums alone are retained as provenance, never as a substitute
for tolerance-based qualification.

Typical use from two clean worktrees::

    python scripts/verify_sol_ultra_numerics.py capture \
        --solver-root ../ANYfeSolver-baseline \
        --label baseline --suite full --output baseline.json
    python scripts/verify_sol_ultra_numerics.py capture \
        --solver-root ../ANYfeSolver-candidate \
        --label candidate --suite full --output candidate.json
    python scripts/verify_sol_ultra_numerics.py compare \
        --baseline baseline.json --candidate candidate.json

``compare`` writes the campaign deliverables by default:

* ``reports/performance/sol_ultra_numerical_comparison.json``
* ``reports/performance/sol_ultra_independent_verification.md``

Exit codes are 0 for pass, 1 for a numerical/error failure, and 2 for an
incomplete comparison (for example a timed-out or unavailable case).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


HARNESS_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(
    os.environ.get("SOL_ULTRA_SOLVER_ROOT", str(HARNESS_ROOT))
).resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA_VERSION = 1
DEFAULT_JSON_REPORT = HARNESS_ROOT / "reports" / "performance" / "sol_ultra_numerical_comparison.json"
DEFAULT_MARKDOWN_REPORT = HARNESS_ROOT / "reports" / "performance" / "sol_ultra_independent_verification.md"


# These are the gates from section 7 of the supplied Sol Ultra plan.  The two
# qualification-derived values are called out explicitly rather than silently
# relaxing a plan gate: the Hill tangent uses the existing central-difference
# test tolerance and contact histories use a conservative deterministic path
# tolerance.  The baseline artifact remains authoritative during comparison.
ACCEPTANCE_CRITERIA: dict[str, dict[str, Any]] = {
    "global_matrix": {
        "method": "relative_l2",
        "rtol": 1.0e-12,
        "atol": 1.0e-12,
        "source": "Sol Ultra plan: K/M/KG relative matrix norm",
    },
    "internal_force": {
        "method": "relative_l2",
        "rtol": 1.0e-11,
        "atol": 1.0e-12,
        "source": "Sol Ultra plan: internal force relative norm",
    },
    "elastic_tangent": {
        "method": "relative_l2",
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "source": "Sol Ultra plan: elastic tangent relative norm",
    },
    "plastic_tangent": {
        "method": "relative_l2",
        "rtol": 2.0e-7,
        "atol": 1.0e-5,
        "source": "existing Hill-48 analytical/numerical tangent qualification",
    },
    "linear_displacement": {
        "method": "relative_l2",
        "rtol": 1.0e-10,
        "atol": 1.0e-14,
        "source": "Sol Ultra plan: linear displacement relative error",
    },
    "modal_frequency": {
        "method": "relative_l2",
        "rtol": 1.0e-9,
        "atol": 1.0e-12,
        "source": "Sol Ultra plan: modal frequency relative error",
    },
    "buckling_factor": {
        "method": "relative_l2",
        "rtol": 1.0e-8,
        "atol": 1.0e-12,
        "source": "Sol Ultra plan: buckling-factor relative error",
    },
    "nonlinear_history": {
        "method": "relative_l2",
        "rtol": 1.0e-8,
        "atol": 1.0e-10,
        "source": "solver path tolerance for deterministic continuation histories",
    },
    "plastic_state": {
        "method": "relative_l2",
        "rtol": 1.0e-9,
        "atol": 1.0e-12,
        "source": "existing Hill/J2 state qualification tolerance",
    },
    "recovery": {
        "method": "relative_l2",
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "source": "component-wise committed-state recovery parity",
    },
    "recovery_stress": {
        "method": "relative_l2",
        "rtol": 1.0e-10,
        "atol": 1.0e-7,
        "source": (
            "existing scalar/compiled shell stress-recovery qualification; "
            "absolute floor applies only to physical recovered stress fields"
        ),
    },
    "contact_history": {
        "method": "relative_l2",
        "rtol": 1.0e-6,
        "atol": 1.0e-9,
        "source": "existing deterministic contact-history qualification tolerance",
    },
}


class CaseUnavailable(RuntimeError):
    """A scientifically meaningful case cannot run in this checkout."""


@dataclass(frozen=True)
class CaseSpec:
    name: str
    builder: Callable[[], dict[str, Any]]
    minimum_suite: str
    description: str


SUITE_LEVEL = {"quick": 0, "standard": 1, "full": 2}


def configure_solver_root(path: Path | str) -> None:
    """Point subsequent case imports at an exact baseline/candidate checkout."""

    global ROOT, SRC
    target = Path(path).resolve()
    source = target / "src"
    if not (source / "anyfesolver").is_dir():
        raise ValueError(f"solver root {target} has no src/anyfesolver package")
    if any(
        name == "anyfesolver" or name.startswith("anyfesolver.")
        for name in sys.modules
    ):
        raise RuntimeError("configure_solver_root must run before importing anyfesolver")
    previous = str(SRC)
    sys.path[:] = [entry for entry in sys.path if entry != previous]
    ROOT = target
    SRC = source
    sys.path.insert(0, str(SRC))
    os.environ["SOL_ULTRA_SOLVER_ROOT"] = str(ROOT)


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "numba": _package_version("numba"),
        "pypardiso": _package_version("pypardiso"),
        "anymaterial": _package_version("ANYmaterial"),
        "anymesher": _package_version("ANYmesher"),
        "anyfileio": _package_version("ANYfileio"),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "pypardiso_mkl_rt_configured": bool(os.environ.get("PYPARDISO_MKL_RT")),
    }


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def numeric_metric(
    values: Any,
    gate: str,
    *,
    method: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    limit: float | None = None,
    allowed_increase: float | None = None,
) -> dict[str, Any]:
    """Create a self-describing numeric metric with its full comparison data."""

    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"metric {gate!r} contains non-finite values")
    criterion = dict(ACCEPTANCE_CRITERIA.get(gate, {}))
    comparison: dict[str, Any] = {
        "gate": gate,
        "method": method or criterion.get("method", "relative_l2"),
    }
    if rtol is not None or "rtol" in criterion:
        comparison["rtol"] = float(criterion.get("rtol", 0.0) if rtol is None else rtol)
    if atol is not None or "atol" in criterion:
        comparison["atol"] = float(criterion.get("atol", 0.0) if atol is None else atol)
    if limit is not None:
        comparison["limit"] = float(limit)
    if allowed_increase is not None:
        comparison["allowed_increase"] = float(allowed_increase)
    flat = array.reshape(-1)
    return {
        "kind": "numeric",
        "shape": list(array.shape),
        "values": flat.tolist(),
        "signature": {
            "sha256_float64_le": _sha256_array(array),
            "l2_norm": float(np.linalg.norm(flat)),
            "linf_norm": float(np.max(np.abs(flat))) if flat.size else 0.0,
            "sum": float(np.sum(flat)),
        },
        "comparison": comparison,
    }


def exact_numeric_metric(values: Any, gate: str = "exact") -> dict[str, Any]:
    return numeric_metric(values, gate, method="exact", rtol=0.0, atol=0.0)


def nonincrease_metric(value: int | float, gate: str = "iteration_or_cutback_count") -> dict[str, Any]:
    return numeric_metric(value, gate, method="nonincrease", allowed_increase=0.0)


def informational_numeric_metric(values: Any, gate: str) -> dict[str, Any]:
    """Retain a diagnostic history without turning improvement into failure."""

    return numeric_metric(values, gate, method="informational")


def upper_bound_metric(value: int | float, limit: float, gate: str) -> dict[str, Any]:
    return numeric_metric(value, gate, method="upper_bound", limit=limit)


def categorical_metric(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, bool, int, type(None))):
        raise TypeError(f"unsupported categorical metric value {type(value).__name__}")
    return {
        "kind": "categorical",
        "value": value,
        "comparison": {"gate": "categorical", "method": "exact"},
    }


def _relative_l2(candidate: Any, reference: Any) -> float:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), np.finfo(float).tiny))


def _append_numeric_tree(
    metrics: MutableMapping[str, dict[str, Any]],
    prefix: str,
    value: Any,
    *,
    gate: str,
) -> None:
    """Flatten public state/recovery dictionaries without using pickle."""

    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _append_numeric_tree(metrics, f"{prefix}.{key}", value[key], gate=gate)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biufc":
            metrics[prefix] = numeric_metric(np.real_if_close(value), gate)
        return
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception:
            array = np.asarray([], dtype=float)
        if array.dtype.kind in "biufc" and array.dtype != object:
            metrics[prefix] = numeric_metric(np.real_if_close(array), gate)
        else:
            for index, item in enumerate(value):
                _append_numeric_tree(metrics, f"{prefix}.{index}", item, gate=gate)
        return
    if isinstance(value, (float, np.floating)):
        metrics[prefix] = numeric_metric(float(value), gate)
        return
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        metrics[prefix] = numeric_metric(int(value), gate)
        return
    if isinstance(value, (str, bool, np.bool_)) or value is None:
        metrics[prefix] = categorical_metric(bool(value) if isinstance(value, np.bool_) else value)


def _json_summary(value: Any, *, depth: int = 0) -> Any:
    """Bound diagnostic observations so artifacts cannot balloon unexpectedly."""

    if depth > 4:
        return "<depth-limited>"
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size <= 24:
            return array.tolist()
        return {
            "shape": list(array.shape),
            "l2_norm": float(np.linalg.norm(array)),
            "sha256_float64_le": _sha256_array(array),
        }
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))[:50]
        return {str(key): _json_summary(item, depth=depth + 1) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_json_summary(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


_DIAGNOSTIC_TERMS = (
    "fast_path",
    "eligible",
    "fallback",
    "batch",
    "cache",
    "reuse",
    "factorization",
    "cutback",
    "retry",
    "thread_policy",
    "backend",
    "assembly_count",
    "materialization",
    "recovery_backend",
)


def _campaign_diagnostics(*objects: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 7 or len(found) >= 250:
            return
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key)
                child = f"{path}.{key}" if path else key
                lowered = key.lower()
                if any(term in lowered for term in _DIAGNOSTIC_TERMS):
                    found[child] = _json_summary(item)
                else:
                    visit(item, child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value[:30]):
                visit(item, f"{path}.{index}", depth + 1)

    for index, obj in enumerate(objects):
        visit(obj, f"source_{index}", 0)
    return found


def _topology(model: Any) -> dict[str, int]:
    return {
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "dofs": int(model.mesh.dof_manager.total_dofs),
    }


def _case_global_matrices() -> dict[str, Any]:
    from anyfesolver.matrix_assembly import (
        assemble_geometric_stiffness_matrix,
        assemble_mass_matrix,
        assemble_stiffness_matrix,
    )
    from anyfesolver.mesh_gen import generate_simple_panel_mesh

    model = generate_simple_panel_mesh(
        1.2,
        0.8,
        0.012,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    model.materials["steel"].density = 7850.0
    states: dict[int, Any] = {}
    for element_id, element in model.mesh.elements.items():
        count = len(element.gauss_points)
        states[int(element_id)] = {
            "membrane_compression_at_gauss": np.tile(
                np.asarray([1.2e5, 0.7e5, 0.15e5], dtype=float),
                (count, 1),
            ),
            "bending_compression_at_gauss": np.tile(
                np.asarray([80.0, 45.0, 12.0], dtype=float),
                (count, 1),
            ),
        }
    stiffness, stiffness_info = assemble_stiffness_matrix(model)
    mass, mass_info = assemble_mass_matrix(model)
    geometric, geometric_info = assemble_geometric_stiffness_matrix(model, states)
    metrics = {
        "global.K": numeric_metric(stiffness.toarray(), "global_matrix"),
        "global.M": numeric_metric(mass.toarray(), "global_matrix"),
        "global.KG": numeric_metric(geometric.toarray(), "global_matrix"),
    }
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "matrix_nnz": {
                "K": int(stiffness.nnz),
                "M": int(mass.nnz),
                "KG": int(geometric.nnz),
            },
            "campaign_diagnostics": _campaign_diagnostics(
                stiffness_info, mass_info, geometric_info
            ),
        },
    }


def _case_linear_static() -> dict[str, Any]:
    from anyfesolver.assembly import solve_linear
    from anyfesolver.boundary import LoadCase
    from anyfesolver.mesh_gen import generate_simple_panel_mesh

    model = generate_simple_panel_mesh(
        1.0,
        0.6,
        0.01,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    load = LoadCase("verification_pressure")
    for element_id in model.mesh.elements:
        load.add_pressure_load(element_id, 1250.0)
    displacement, info = solve_linear(model, load)
    convergence = info.get("convergence_info") or {}
    metrics = {
        "displacements": numeric_metric(displacement, "linear_displacement"),
        "solver_status": categorical_metric(str(convergence.get("status", ""))),
    }
    relative_residual = convergence.get("relative_residual")
    if relative_residual is not None:
        metrics["relative_residual"] = upper_bound_metric(
            float(relative_residual), 1.0e-8, "linear_residual"
        )
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "campaign_diagnostics": _campaign_diagnostics(info),
        },
    }


def _axial_modal_model() -> Any:
    from anyfesolver.boundary import BoundaryCondition, FixedSupport
    from anyfesolver.elements import BeamElement
    from anyfesolver.fe_core import FEModel

    model = FEModel("verification_axial_modal")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "slider",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _case_modal() -> dict[str, Any]:
    from anyfesolver.modal import solve_free_vibration

    model = _axial_modal_model()
    result = solve_free_vibration(model, num_modes=1)
    if not result.modes:
        raise CaseUnavailable(f"modal solver returned no modes ({result.solver_status})")
    metrics = {
        "frequencies_hz": numeric_metric(result.frequencies_hz, "modal_frequency"),
        "solver_status": categorical_metric(result.solver_status),
        "mode_residuals": upper_bound_metric(
            max(mode.residual_norm for mode in result.modes),
            1.0e-9,
            "modal_residual",
        ),
    }
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "campaign_diagnostics": _campaign_diagnostics(
                result.assembly_info, result.diagnostics
            ),
        },
    }


def _beam_column_model(num_elements: int = 4) -> Any:
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import BeamElement
    from anyfesolver.fe_core import FEModel

    length = 4.0
    model = FEModel("verification_beam_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for index in range(num_elements + 1):
        model.add_node(index + 1, length * index / num_elements, 0.0, 0.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for index in range(num_elements):
        model.add_element(
            index + 1,
            BeamElement(index + 1, [index + 1, index + 2], "steel", dict(section)),
        )
    all_nodes = list(range(1, num_elements + 2))
    model.add_boundary_condition(
        BoundaryCondition(
            "suppress_unrelated_dofs",
            all_nodes,
            {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition("pinned_lateral_ends", [1, num_elements + 1], {"uy": 0.0})
    )
    return model


def _case_buckling() -> dict[str, Any]:
    from anyfesolver.buckling import solve_eigenvalue_buckling

    model = _beam_column_model()
    states = {
        int(element_id): {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    result = solve_eigenvalue_buckling(model, states, num_modes=2)
    if not result.modes:
        raise CaseUnavailable(f"buckling solver returned no modes ({result.solver_status})")
    metrics = {
        "load_factors": numeric_metric(
            [mode.load_factor for mode in result.modes], "buckling_factor"
        ),
        "solver_status": categorical_metric(result.solver_status),
        "mode_residuals": upper_bound_metric(
            max(mode.residual_norm for mode in result.modes),
            1.0e-8,
            "buckling_residual",
        ),
    }
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "campaign_diagnostics": _campaign_diagnostics(
                result.assembly_info, result.diagnostics or {}
            ),
        },
    }


def _case_nonlinear_internal() -> dict[str, Any]:
    from anyfesolver import nonlinear_performance, nonlinear_static
    from anyfesolver.nonlinear_performance_bootstrap import (
        nonlinear_assembly_diagnostics,
    )
    from anyfesolver.mesh_gen import generate_simple_panel_mesh

    model = generate_simple_panel_mesh(
        1.2,
        0.8,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
    )
    rng = np.random.default_rng(20260811)
    displacement = rng.normal(
        scale=2.0e-5,
        size=model.mesh.dof_manager.total_dofs,
    )
    # Installation retains the pre-optimization assembler in
    # ``_ORIGINAL_ASSEMBLER`` and makes this call exercise the active path.
    nonlinear_static._ensure_nonlinear_acceleration()
    force, tangent, states = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    if tangent is None:
        raise CaseUnavailable("active nonlinear assembler returned no tangent")
    metrics: dict[str, dict[str, Any]] = {
        "active.internal_force": numeric_metric(force, "internal_force"),
        "active.tangent": numeric_metric(tangent.toarray(), "elastic_tangent"),
    }
    _append_numeric_tree(metrics, "active.state", states, gate="plastic_state")

    oracle = nonlinear_performance._ORIGINAL_ASSEMBLER
    metrics["scalar_oracle_available"] = categorical_metric(oracle is not None)
    observations: dict[str, Any] = {"topology": _topology(model)}
    if oracle is not None:
        oracle_force, oracle_tangent, oracle_states = oracle(
            model,
            displacement,
            {},
            5,
            tangent=True,
        )
        if oracle_tangent is None:
            raise CaseUnavailable("scalar nonlinear oracle returned no tangent")
        metrics["scalar.internal_force"] = numeric_metric(
            oracle_force, "internal_force"
        )
        metrics["scalar.tangent"] = numeric_metric(
            oracle_tangent.toarray(), "elastic_tangent"
        )
        _append_numeric_tree(
            metrics, "scalar.state", oracle_states, gate="plastic_state"
        )
        metrics["oracle.relative_force_error"] = upper_bound_metric(
            _relative_l2(force, oracle_force),
            1.0e-11,
            "internal_force",
        )
        metrics["oracle.relative_tangent_error"] = upper_bound_metric(
            _relative_l2(tangent.toarray(), oracle_tangent.toarray()),
            1.0e-10,
            "elastic_tangent",
        )
    else:
        observations["scalar_oracle_unavailable"] = (
            "anyfesolver.nonlinear_performance._ORIGINAL_ASSEMBLER is None"
        )
    observations["campaign_diagnostics"] = _campaign_diagnostics(
        nonlinear_assembly_diagnostics(model)
    )
    return {"metrics": metrics, "observations": observations}


@dataclass(frozen=True)
class _LinearHardening:
    initial_flow: float
    modulus: float

    def flow_stress(self, alpha: np.ndarray) -> np.ndarray:
        values = np.asarray(alpha, dtype=float)
        return self.initial_flow + self.modulus * values

    def hardening_modulus(self, alpha: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(alpha, dtype=float), self.modulus)


def _orthotropic_elastic_matrix() -> np.ndarray:
    e1, e2, nu12, g12 = 150.0e9, 90.0e9, 0.25, 40.0e9
    nu21 = nu12 * e2 / e1
    denominator = 1.0 - nu12 * nu21
    return np.asarray(
        [
            [e1 / denominator, nu12 * e2 / denominator, 0.0],
            [nu12 * e2 / denominator, e2 / denominator, 0.0],
            [0.0, 0.0, g12],
        ],
        dtype=float,
    )


def _hill_model() -> Any:
    from anyfesolver.materials import Hill48Yield

    return Hill48Yield(
        X=300.0e6,
        Y=240.0e6,
        Z=270.0e6,
        S12=130.0e6,
        S13=140.0e6,
        S23=120.0e6,
    )


def _case_hill48_material() -> dict[str, Any]:
    from anyfesolver.plasticity import (
        hill48_plane_stress_equivalent_stress,
        hill48_plane_stress_numerical_tangent,
        hill48_plane_stress_return_map,
    )

    elastic = _orthotropic_elastic_matrix()
    hill = _hill_model()
    curve = _LinearHardening(initial_flow=250.0e6, modulus=1.5e9)
    strain_1 = np.asarray(
        [
            [1.0e-5, -2.0e-6, 3.0e-6],
            [0.0040, -0.0005, 0.0010],
            [0.0065, 0.0004, -0.0017],
        ],
        dtype=float,
    )
    plastic_0 = np.zeros_like(strain_1)
    alpha_0 = np.zeros(strain_1.shape[0], dtype=float)
    stress_1, tangent_1, plastic_1, alpha_1 = hill48_plane_stress_return_map(
        strain_1,
        plastic_0,
        alpha_0,
        elastic,
        hill,
        curve,
    )
    numerical = hill48_plane_stress_numerical_tangent(
        strain_1,
        plastic_0,
        alpha_0,
        elastic,
        hill,
        curve,
        step=1.0e-8,
    )
    strain_2 = np.asarray(
        [
            [2.0e-5, -3.0e-6, 4.0e-6],
            plastic_1[1],
            [0.0090, -0.0010, 0.0020],
        ],
        dtype=float,
    )
    stress_2, tangent_2, plastic_2, alpha_2 = hill48_plane_stress_return_map(
        strain_2,
        plastic_1,
        alpha_1,
        elastic,
        hill,
        curve,
    )
    tangent_error = _relative_l2(tangent_1, numerical)
    metrics = {
        "path_1.stress": numeric_metric(stress_1, "plastic_state"),
        "path_1.tangent": numeric_metric(tangent_1, "plastic_tangent"),
        "path_1.plastic_strain": numeric_metric(plastic_1, "plastic_state"),
        "path_1.alpha": numeric_metric(alpha_1, "plastic_state"),
        "path_1.equivalent_stress": numeric_metric(
            hill48_plane_stress_equivalent_stress(stress_1, hill),
            "plastic_state",
        ),
        "path_1.analytical_tangent_fd_error": upper_bound_metric(
            tangent_error,
            2.0e-7,
            "plastic_tangent",
        ),
        "path_2.stress": numeric_metric(stress_2, "plastic_state"),
        "path_2.tangent": numeric_metric(tangent_2, "plastic_tangent"),
        "path_2.plastic_strain": numeric_metric(plastic_2, "plastic_state"),
        "path_2.alpha": numeric_metric(alpha_2, "plastic_state"),
    }
    return {
        "metrics": metrics,
        "observations": {
            "point_classification": [
                "elastic",
                "yield_then_unload",
                "yield_then_reload",
            ],
        },
    }


def _orthotropic_hill_shell_model() -> Any:
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import create_element
    from anyfesolver.fe_core import FEModel
    from anyfesolver.material_curves import DNVC208MaterialCurve
    from anyfesolver.materials import Hill48Yield

    model = FEModel("verification_orthotropic_hill_shell")
    strength = 100.0e6
    shear = strength / np.sqrt(3.0)
    model.add_orthotropic_material(
        "lamina",
        elastic_modulus_1=150.0e9,
        elastic_modulus_2=12.0e9,
        elastic_modulus_3=10.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.0e9,
        shear_modulus_23=3.8e9,
        density=1600.0,
        hill_yield=Hill48Yield(strength, strength, strength, shear, shear, shear),
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=100.0e6,
            sigma_yield=105.0e6,
            sigma_yield_2=110.0e6,
            eps_p_y1=0.005,
            eps_p_y2=0.010,
            K=400.0e6,
            n=0.20,
        ),
    )
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    for node_id, xyz in enumerate(coordinates, start=1):
        model.add_node(node_id, *xyz)
    model.add_element(
        1,
        create_element(
            "shell",
            1,
            [1, 2, 3, 4],
            "lamina",
            thickness=0.02,
            material_direction=(1.0, 0.0, 1.0),
            material_angle_deg=30.0,
        ),
    )
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_y", [1], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition(
            "in_plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _diagnostic_int(mapping: Mapping[str, Any], names: Iterable[str]) -> int | None:
    targets = {str(name).lower() for name in names}

    def find(value: Any) -> int | None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in targets and isinstance(item, (int, float, np.number)):
                    return int(item)
            for item in value.values():
                result = find(item)
                if result is not None:
                    return result
        return None

    return find(mapping)


def _case_hill48_shell_path() -> dict[str, Any]:
    from anyfesolver import nonlinear_static
    from anyfesolver.boundary import LoadCase
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import solve_static_nonlinear
    from anyfesolver.recovery import recover_stress_result

    model = _orthotropic_hill_shell_model()
    nonlinear_static._ensure_nonlinear_acceleration()
    stiffness, stiffness_info = assemble_stiffness_matrix(model)
    probe = np.linspace(
        -2.0e-4,
        2.0e-4,
        model.mesh.dof_manager.total_dofs,
        dtype=float,
    )
    probe_force, probe_tangent, probe_state = nonlinear_static._assemble_nonlinear_system(
        model,
        probe,
        {},
        5,
        tangent=True,
    )
    if probe_tangent is None:
        raise CaseUnavailable("orthotropic Hill shell probe returned no tangent")
    load = LoadCase("verification_hill_tension")
    load.add_nodal_load(2, [1.05e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [1.05e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=4,
        max_iterations=25,
        tolerance=1.0e-8,
        num_layers=5,
    )
    recovery = recover_stress_result(model, nonlinear_result=result)
    metrics: dict[str, dict[str, Any]] = {
        "orthotropic.K": numeric_metric(stiffness.toarray(), "global_matrix"),
        "probe.internal_force": numeric_metric(probe_force, "internal_force"),
        "probe.tangent": numeric_metric(probe_tangent.toarray(), "plastic_tangent"),
        "solution.displacements": numeric_metric(
            result.displacements, "nonlinear_history"
        ),
        "solution.load_factor": numeric_metric(
            result.load_factor, "nonlinear_history"
        ),
        "solution.status": categorical_metric(result.status),
        "solution.step_load_factors": numeric_metric(
            [step.load_factor for step in result.steps], "nonlinear_history"
        ),
        "solution.step_iterations": informational_numeric_metric(
            [step.iterations for step in result.steps], "iteration_count"
        ),
        "solution.total_iterations": nonincrease_metric(
            sum(step.iterations for step in result.steps)
        ),
    }
    _append_numeric_tree(metrics, "probe.state", probe_state, gate="plastic_state")
    _append_numeric_tree(
        metrics, "solution.committed_state", result.element_states, gate="plastic_state"
    )
    _append_numeric_tree(
        metrics,
        "recovery.element_stress",
        recovery.element_stresses,
        gate="recovery_stress",
    )
    provenance = recovery.provenance.to_dict()
    _append_numeric_tree(metrics, "recovery.provenance", provenance, gate="recovery")
    cutbacks = _diagnostic_int(
        result.info,
        ("cutback_count", "num_cutbacks", "total_cutbacks"),
    )
    unavailable_diagnostics: list[str] = []
    if cutbacks is not None:
        metrics["solution.cutback_count"] = nonincrease_metric(cutbacks)
    else:
        unavailable_diagnostics.append("cutback_count")
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "unavailable_diagnostics": unavailable_diagnostics,
            "campaign_diagnostics": _campaign_diagnostics(
                stiffness_info,
                result.info,
                recovery.execution_report.to_dict()
                if recovery.execution_report is not None
                else {},
            ),
        },
    }


def _generalized_shell_model() -> tuple[Any, Any]:
    from anyfesolver.elements import create_element
    from anyfesolver.fe_core import FEModel
    from anyfesolver.shell_sections import GeneralizedShellSection

    model = FEModel("verification_generalized_shell")
    model.add_material("dummy", 70.0e9, 0.25, density=2700.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    section = GeneralizedShellSection(
        A=np.asarray(
            [[120.0, 18.0, 4.0], [18.0, 90.0, -3.0], [4.0, -3.0, 35.0]]
        ),
        B=np.asarray(
            [[0.8, 0.1, 0.0], [0.05, -0.4, 0.08], [0.02, 0.0, 0.25]]
        ),
        D=np.asarray(
            [[10.0, 0.8, 0.1], [0.8, 8.0, -0.1], [0.1, -0.1, 3.0]]
        ),
        As=np.asarray([[20.0, 2.0], [2.0, 15.0]]),
        mass_per_area=17.0,
        rotary_inertia_per_area=0.08,
        name="verification_abd",
    )
    element = create_element(
        "shell",
        1,
        [1, 2, 3, 4],
        "dummy",
        thickness=0.02,
        shell_section=section,
        material_angle_deg=30.0,
    )
    model.add_element(1, element)
    return model, element


def _case_generalized_shell() -> dict[str, Any]:
    from anyfesolver.recovery import recover_stress_result

    model, element = _generalized_shell_model()
    material = model.get_material("dummy")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node_id in element.node_ids:
        node = model.mesh.nodes[node_id]
        x, y, _ = node.coords()
        displacement[node.dofs[0]] = 1.2e-3 * x + 0.2e-3 * y
        displacement[node.dofs[1]] = -0.4e-3 * y + 0.5e-3 * x
        displacement[node.dofs[2]] = 0.015 * x - 0.008 * y
        displacement[node.dofs[3]] = -0.003 * y - 0.002 * x
        displacement[node.dofs[4]] = 0.006 * x + 0.001 * y
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    local = displacement[mapping]
    stiffness = element.compute_stiffness_matrix(model.mesh, material)
    mass = element.compute_mass_matrix(model.mesh, material)
    force, tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        local,
        tangent=True,
    )
    if tangent is None or state is None:
        raise CaseUnavailable("generalized shell returned no tangent/state")
    nonlinear_result = SimpleNamespace(
        displacements=displacement.copy(),
        element_states={1: state},
        info={"kinematics": "von_karman"},
        status="converged",
        load_factor=1.0,
    )
    recovery = recover_stress_result(model, nonlinear_result=nonlinear_result)
    metrics: dict[str, dict[str, Any]] = {
        "element.K": numeric_metric(stiffness, "global_matrix"),
        "element.M": numeric_metric(mass, "global_matrix"),
        "element.internal_force": numeric_metric(force, "internal_force"),
        "element.tangent": numeric_metric(tangent, "elastic_tangent"),
    }
    _append_numeric_tree(metrics, "element.state", state, gate="recovery")
    _append_numeric_tree(
        metrics, "recovery.stress", recovery.element_stresses, gate="recovery"
    )
    _append_numeric_tree(
        metrics, "recovery.provenance", recovery.provenance.to_dict(), gate="recovery"
    )
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "recovery_scope": recovery.element_stresses[1].get("recovery_scope"),
            "campaign_diagnostics": _campaign_diagnostics(
                recovery.execution_report.to_dict()
                if recovery.execution_report is not None
                else {}
            ),
        },
    }


def _rigid_rotation_field(model: Any, angle_degrees: float, axis: Sequence[float]) -> np.ndarray:
    from anyfesolver.corotational import rotation_matrix_from_vector

    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    rotation_vector = np.radians(angle_degrees) * direction
    rotation = rotation_matrix_from_vector(rotation_vector)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        reference = np.asarray([node.x, node.y, node.z], dtype=float)
        displacement[np.asarray(node.dofs[:3])] = rotation @ reference - reference
        displacement[np.asarray(node.dofs[3:])] = rotation_vector
    return displacement


def _corotational_models() -> list[tuple[str, Any, Any]]:
    from anyfesolver.elements import BeamElement, create_element
    from anyfesolver.fe_core import FEModel

    shell_model = FEModel("verification_corotational_shell")
    shell_model.add_material("steel", 210.0e9, 0.0, density=7850.0)
    for node_id, (x, y) in enumerate(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        start=1,
    ):
        shell_model.add_node(node_id, x, y, 0.0)
    shell = create_element(
        "shell", 1, [1, 2, 3, 4], "steel", thickness=0.01
    )
    shell_model.add_element(1, shell)

    beam_model = FEModel("verification_corotational_beam")
    beam_model.add_material("steel", 210.0e9, 0.0, density=7850.0)
    beam_model.add_node(1, 0.0, 0.0, 0.0)
    beam_model.add_node(2, 1.0, 0.0, 0.0)
    beam = BeamElement(
        1,
        [1, 2],
        "steel",
        {"area": 1.0e-3, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    beam_model.add_element(1, beam)
    return [("shell", shell_model, shell), ("beam", beam_model, beam)]


def _case_corotational() -> dict[str, Any]:
    from anyfesolver.corotational import corotational_element_response
    from anyfesolver import nonlinear_static
    from anyfesolver.nonlinear_performance_bootstrap import nonlinear_assembly_diagnostics

    metrics: dict[str, dict[str, Any]] = {}
    observations: dict[str, Any] = {}
    nonlinear_static._ensure_nonlinear_acceleration()
    models = _corotational_models()
    for name, model, element in models:
        rigid = _rigid_rotation_field(model, 35.0, (0.2, 0.4, 1.0))
        deformed = rigid.copy()
        end = model.mesh.nodes[max(model.mesh.nodes)]
        deformed[end.dofs[0]] += 0.003
        deformed[end.dofs[2]] += 0.007
        deformed[end.dofs[4]] -= 0.015
        force, tangent, state = corotational_element_response(
            model,
            1,
            element,
            deformed,
            tangent=True,
            tangent_mode="rotated",
        )
        if tangent is None:
            raise CaseUnavailable(f"{name} corotational response returned no tangent")
        global_force, global_tangent, global_state = nonlinear_static._assemble_nonlinear_system(
            model,
            deformed,
            {},
            5,
            tangent=True,
            kinematics="corotational",
            corotational_tangent="rotated",
        )
        if global_tangent is None:
            raise CaseUnavailable(f"{name} global corotational response returned no tangent")
        rigid_cr, _, _ = nonlinear_static._assemble_nonlinear_system(
            model,
            rigid,
            {},
            5,
            tangent=False,
            kinematics="corotational",
        )
        rigid_vk, _, _ = nonlinear_static._assemble_nonlinear_system(
            model,
            rigid,
            {},
            5,
            tangent=False,
            kinematics="von_karman",
        )
        scale = max(float(np.linalg.norm(rigid_vk)), 210.0e9 * 1.0e-5)
        metrics[f"{name}.element_force"] = numeric_metric(force, "internal_force")
        metrics[f"{name}.element_tangent"] = numeric_metric(
            tangent, "elastic_tangent"
        )
        metrics[f"{name}.global_force"] = numeric_metric(
            global_force, "internal_force"
        )
        metrics[f"{name}.global_tangent"] = numeric_metric(
            global_tangent.toarray(), "elastic_tangent"
        )
        metrics[f"{name}.rigid_rotation_force_ratio"] = upper_bound_metric(
            float(np.linalg.norm(rigid_cr)) / scale,
            1.0e-9,
            "corotational_objectivity",
        )
        _append_numeric_tree(metrics, f"{name}.element_state", state, gate="plastic_state")
        _append_numeric_tree(
            metrics, f"{name}.global_state", global_state, gate="plastic_state"
        )
        observations[f"{name}_topology"] = _topology(model)
    observations["campaign_diagnostics"] = _campaign_diagnostics(
        *(nonlinear_assembly_diagnostics(model) for _, model, _ in models)
    )
    return {"metrics": metrics, "observations": observations}


def _case_arc_length() -> dict[str, Any]:
    from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
    from anyfesolver.boundary import BoundaryCondition, LoadCase
    from anyfesolver.elements import Element
    from anyfesolver.fe_core import FEModel

    class SofteningSpringElement(Element):
        def __init__(self, element_id: int, node_id: int) -> None:
            super().__init__(element_id, [node_id], "default")

        @property
        def num_nodes(self) -> int:
            return 1

        @property
        def dofs_per_node(self) -> int:
            return 6

        def get_node_coordinates(self, mesh: Any) -> np.ndarray:
            return np.asarray([mesh.get_node(self.node_ids[0]).coords()], dtype=float)

        def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
            return np.eye(6, dtype=float)

        def compute_nonlinear_response(
            self,
            mesh: Any,
            material: Any,
            u_elem: Any,
            state: Any = None,
            num_layers: int = 5,
            tangent: bool = True,
        ) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
            del mesh, material, state, num_layers
            values = np.asarray(u_elem, dtype=float)
            displacement = float(values[0])
            force = values.copy()
            force[0] = displacement - displacement**3
            stiffness = None
            if tangent:
                stiffness = np.eye(6, dtype=float)
                stiffness[0, 0] = 1.0 - 3.0 * displacement**2
            return force, stiffness, {"spring_displacement": displacement}

    model = FEModel("verification_softening_spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, SofteningSpringElement(1, 1))
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
    metrics = {
        "status": categorical_metric(result.status),
        "peak_load_factor": numeric_metric(
            result.peak_load_factor, "nonlinear_history"
        ),
        "load_factor_history": numeric_metric(
            [step.load_factor for step in result.steps], "nonlinear_history"
        ),
        "displacement_history": numeric_metric(
            [step.displacement_norm for step in result.steps], "nonlinear_history"
        ),
        "iteration_history": informational_numeric_metric(
            [step.iterations for step in result.steps], "iteration_count"
        ),
        "retry_history": informational_numeric_metric(
            [step.retries for step in result.steps], "retry_count"
        ),
        "total_iterations": nonincrease_metric(
            sum(step.iterations for step in result.steps)
        ),
        "total_retries": nonincrease_metric(sum(step.retries for step in result.steps)),
    }
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "campaign_diagnostics": _campaign_diagnostics(result.info),
        },
    }


def _contact_panel() -> Any:
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import create_element
    from anyfesolver.fe_core import FEModel

    model = FEModel("verification_contact_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    model.add_element(
        1,
        create_element(
            "shell", 1, [1, 2, 3, 4], "soft", thickness=0.05
        ),
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "restrain_shell_nonimpact_modes",
            [1, 2, 3, 4],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _yielding_contact_panel(*, divisions: int = 4) -> Any:
    """Small clamped panel with enough spatial freedom to yield under impact."""

    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import create_element
    from anyfesolver.fe_core import FEModel

    model = FEModel("verification_yielding_contact_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    node_of: dict[tuple[int, int], int] = {}
    node_id = 1
    for j in range(divisions + 1):
        for i in range(divisions + 1):
            model.add_node(node_id, i / divisions, j / divisions, 0.0)
            node_of[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(divisions):
        for i in range(divisions):
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
    edge_nodes = [
        node_of[(i, j)]
        for j in range(divisions + 1)
        for i in range(divisions + 1)
        if i in (0, divisions) or j in (0, divisions)
    ]
    model.add_boundary_condition(
        BoundaryCondition(
            "clamped_impact_edge",
            edge_nodes,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _case_contact_load() -> dict[str, Any]:
    from anyfesolver.contact import (
        RigidSphereImpact,
        SphereContactConfig,
        assemble_sphere_contact_load_vector,
    )

    model = _contact_panel()
    sphere = RigidSphereImpact(
        "verification_static_contact",
        radius=0.2,
        mass=1.0,
        start_point=(0.5, 0.5, 0.1),
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )
    vector, sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.asarray([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    if not records:
        raise CaseUnavailable("deterministic contact probe returned no records")
    record = records[0]
    nodal = np.asarray(
        [record.nodal_forces[node_id] for node_id in sorted(record.nodal_forces)],
        dtype=float,
    )
    metrics = {
        "structure_load_vector": numeric_metric(vector, "contact_history"),
        "sphere_force": numeric_metric(sphere_force, "contact_history"),
        "penetration": numeric_metric(record.penetration, "contact_history"),
        "normal_force": numeric_metric(record.normal_force, "contact_history"),
        "nodal_forces": numeric_metric(nodal, "contact_history"),
        "classification": categorical_metric(record.contact_classification),
        "element_id": categorical_metric(int(record.element_id)),
    }
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "record_count": len(records),
        },
    }


def _impact_penetration_history(active_history: Sequence[Sequence[Mapping[str, Any]]]) -> np.ndarray:
    values = []
    for records in active_history:
        values.append(
            max((float(record.get("penetration", 0.0)) for record in records), default=0.0)
        )
    return np.asarray(values, dtype=float)


def _append_aggregated_mapping_sequence(
    metrics: MutableMapping[str, dict[str, Any]],
    prefix: str,
    items: Sequence[Any],
    *,
    gate: str,
) -> None:
    """Compare state fields as full vectors rather than fragile scalar norms."""

    numeric_fields: dict[str, list[np.ndarray]] = {}

    def visit(value: Any, path: str, item_index: int) -> None:
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                child = f"{path}.{key}" if path else str(key)
                visit(value[key], child, item_index)
            return
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "biufc":
                numeric_fields.setdefault(path, []).append(
                    np.asarray(np.real_if_close(value), dtype=float).reshape(-1)
                )
            return
        if isinstance(value, (list, tuple)):
            try:
                array = np.asarray(value)
            except Exception:
                array = np.asarray([], dtype=float)
            if array.dtype.kind in "biufc" and array.dtype != object:
                numeric_fields.setdefault(path, []).append(
                    np.asarray(np.real_if_close(array), dtype=float).reshape(-1)
                )
            else:
                for index, child_value in enumerate(value):
                    visit(child_value, f"{path}.{index}", item_index)
            return
        if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            numeric_fields.setdefault(path, []).append(
                np.asarray([value], dtype=float)
            )
            return
        if isinstance(value, (str, bool, np.bool_)) or value is None:
            metric_name = f"{prefix}.{path}.category.{item_index}"
            metrics[metric_name] = categorical_metric(
                bool(value) if isinstance(value, np.bool_) else value
            )

    for item_index, item in enumerate(items):
        visit(item, "", item_index)
    for path, arrays in sorted(numeric_fields.items()):
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
        metrics[f"{prefix}.{path}"] = numeric_metric(values, gate)


def _append_plastic_impact_metrics(
    metrics: MutableMapping[str, dict[str, Any]],
    diagnostics: Mapping[str, Any],
) -> list[str]:
    """Record committed plastic state and compact damage/deletion histories."""

    unavailable: list[str] = []
    element_states = diagnostics.get("element_states")
    if isinstance(element_states, Mapping):
        state_ids = sorted(element_states, key=lambda value: int(value))
        metrics["plastic.element_states.element_ids"] = exact_numeric_metric(
            [int(value) for value in state_ids]
        )
        _append_aggregated_mapping_sequence(
            metrics,
            "plastic.element_states",
            [element_states[element_id] for element_id in state_ids],
            gate="plastic_state",
        )
    else:
        unavailable.append("element_states")

    element_state_history = diagnostics.get("element_state_history")
    if isinstance(element_state_history, (list, tuple)):
        _append_aggregated_mapping_sequence(
            metrics,
            "plastic.element_state_history",
            element_state_history,
            gate="plastic_state",
        )
    else:
        unavailable.append("element_state_history")

    state_von_mises_history = diagnostics.get("state_von_mises_history")
    if isinstance(state_von_mises_history, (list, tuple)):
        _append_aggregated_mapping_sequence(
            metrics,
            "plastic.state_von_mises_history",
            state_von_mises_history,
            gate="plastic_state",
        )
    else:
        unavailable.append("state_von_mises_history")

    if "plastic_work_proxy" in diagnostics:
        metrics["plastic.plastic_work_proxy"] = numeric_metric(
            diagnostics["plastic_work_proxy"], "plastic_state"
        )
    else:
        unavailable.append("plastic_work_proxy")

    summary = diagnostics.get("plastic_impact_damage_summary")
    if not isinstance(summary, Mapping):
        unavailable.append("plastic_impact_damage_summary")
        return unavailable

    for key in (
        "enabled",
        "deleted_count",
        "deleted_fraction",
        "deleted_element_ids",
        "softened_element_ids",
        "max_damage",
        "max_utilization",
        "max_equivalent_plastic_strain",
    ):
        if key in summary:
            _append_numeric_tree(metrics, f"damage.summary.{key}", summary[key], gate="plastic_state")
        else:
            unavailable.append(f"plastic_impact_damage_summary.{key}")

    records = summary.get("records", [])
    records = records if isinstance(records, (list, tuple)) else []
    metrics["damage.record_count"] = categorical_metric(len(records))
    history_event_ids: list[tuple[int, int]] = []
    history_times: list[float] = []
    history_values: list[tuple[float, float, float, float]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        element_id = int(record.get("element_id", -1))
        metrics[f"damage.record.{record_index}.element_id"] = categorical_metric(element_id)
        history = record.get("history", [])
        history = history if isinstance(history, (list, tuple)) else []
        metrics[f"damage.record.{record_index}.history_count"] = categorical_metric(len(history))
        for history_index, event in enumerate(history):
            if not isinstance(event, Mapping):
                continue
            history_event_ids.append((element_id, int(event.get("step_index", -1))))
            history_times.append(float(event.get("time", 0.0)))
            history_values.append(
                (
                    float(event.get("equivalent_plastic_strain", 0.0)),
                    float(event.get("utilization", 0.0)),
                    float(event.get("damage", 0.0)),
                    float(event.get("scale", 1.0)),
                )
            )
            metrics[
                f"damage.record.{record_index}.history.{history_index}.location"
            ] = categorical_metric(str(event.get("location", "")))
    metrics["damage.history.event_ids"] = exact_numeric_metric(history_event_ids)
    metrics["damage.history.times"] = numeric_metric(history_times, "contact_history")
    metrics["damage.history.values"] = numeric_metric(history_values, "plastic_state")

    deletions = summary.get("deletion_records", [])
    deletions = deletions if isinstance(deletions, (list, tuple)) else []
    metrics["damage.deletion_count"] = categorical_metric(len(deletions))
    deletion_ids: list[tuple[int, int]] = []
    deletion_values: list[tuple[float, float, float, float]] = []
    for index, record in enumerate(deletions):
        if not isinstance(record, Mapping):
            continue
        deletion_ids.append(
            (int(record.get("element_id", -1)), int(record.get("step_index", -1)))
        )
        deletion_values.append(
            (
                float(record.get("load_factor", 0.0)),
                float(record.get("trigger_value", 0.0)),
                float(record.get("threshold", 0.0)),
                float(record.get("measure", 0.0)),
            )
        )
        for key in ("element_type", "trigger_name", "location"):
            metrics[f"damage.deletion.{index}.{key}"] = categorical_metric(
                str(record.get(key, ""))
            )
    metrics["damage.deletion.event_ids"] = exact_numeric_metric(deletion_ids)
    metrics["damage.deletion.values"] = numeric_metric(
        deletion_values, "plastic_state"
    )

    erosion = diagnostics.get("erosion_summary")
    if isinstance(erosion, Mapping):
        _append_numeric_tree(metrics, "damage.erosion", erosion, gate="plastic_state")
    else:
        unavailable.append("erosion_summary")
    return unavailable


def _case_nonlinear_impact() -> dict[str, Any]:
    from anyfesolver.contact import (
        NonlinearTransientConfig,
        PlasticImpactDamageConfig,
        RigidSphereImpact,
        SphereContactConfig,
        solve_transient_sphere_impact,
    )
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.material_curves import DNVC208MaterialCurve

    model = _yielding_contact_panel()
    model.materials["soft"].hardening_curve = DNVC208MaterialCurve(
        sigma_prop=800.0,
        sigma_yield=1000.0,
        sigma_yield_2=1200.0,
        eps_p_y1=1.0e-5,
        eps_p_y2=1.0e-3,
        K=2000.0,
        n=0.1,
    )
    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0025, t_end=0.05, output_nodes=[13]),
        RigidSphereImpact(
            "verification_nonlinear_hit",
            radius=0.2,
            mass=20.0,
            start_point=(0.5, 0.5, 0.25),
            travel_direction=(0.0, 0.0, -1.0),
            speed=4.0,
        ),
        SphereContactConfig(penalty_stiffness=2000.0, max_contact_iterations=20),
        nonlinear_config=NonlinearTransientConfig(
            enabled=True,
            max_iterations=15,
            max_cutbacks=4,
            tangent_reuse_iterations=2,
        ),
        plastic_damage_config=PlasticImpactDamageConfig(
            threshold=0.2,
            max_deleted_fraction=1.0,
        ),
    )
    diagnostics = result.diagnostics
    metrics: dict[str, dict[str, Any]] = {
        "status": categorical_metric(result.status),
        "times": numeric_metric(result.times, "contact_history"),
        "displacements": numeric_metric(result.displacements, "contact_history"),
        "velocities": numeric_metric(result.velocities, "contact_history"),
        "accelerations": numeric_metric(result.accelerations, "contact_history"),
        "sphere_positions": numeric_metric(result.sphere_positions, "contact_history"),
        "sphere_velocities": numeric_metric(result.sphere_velocities, "contact_history"),
        "contact_force_history": numeric_metric(
            result.contact_force_history, "contact_history"
        ),
        "penetration_history": numeric_metric(
            _impact_penetration_history(result.active_contact_history),
            "contact_history",
        ),
        "sphere_impulse": numeric_metric(result.sphere_impulse, "contact_history"),
        "max_penetration": numeric_metric(result.max_penetration, "contact_history"),
        "peak_contact_force": numeric_metric(
            result.peak_contact_force, "contact_history"
        ),
        "contact_duration": numeric_metric(result.contact_duration, "contact_history"),
        "sphere_momentum_balance_error": numeric_metric(
            result.sphere_momentum_balance_error, "contact_history"
        ),
    }
    energy_keys = (
        "kinetic_energy",
        "strain_energy",
        "sphere_kinetic_energy",
        "internal_work",
    )
    unavailable_diagnostics: list[str] = []
    for key in energy_keys:
        if key in diagnostics:
            metrics[f"energy.{key}"] = numeric_metric(
                diagnostics[key], "contact_history"
            )
        else:
            unavailable_diagnostics.append(key)
    cutbacks = _diagnostic_int(diagnostics, ("cutback_count", "num_cutbacks"))
    if cutbacks is None:
        unavailable_diagnostics.append("cutback_count")
    else:
        metrics["cutback_count"] = nonincrease_metric(cutbacks)
    step_diagnostics = diagnostics.get("contact_step_diagnostics", [])
    iterations = [
        int(step["iterations"])
        for step in step_diagnostics
        if isinstance(step, Mapping) and "iterations" in step
    ]
    if iterations:
        metrics["iteration_history"] = informational_numeric_metric(
            iterations, "iteration_count"
        )
        metrics["total_iterations"] = nonincrease_metric(sum(iterations))
    else:
        unavailable_diagnostics.append("iteration_history")
    for key in (
        "tangent_assembly_count",
        "tangent_reuse_count",
        "factorization_count",
        "factorization_reuse_count",
    ):
        value = _diagnostic_int(diagnostics, (key,))
        if value is None:
            unavailable_diagnostics.append(key)
        else:
            # These are reported and compared for visibility.  An optimized
            # candidate may legitimately reduce them but must not increase.
            metrics[f"diagnostic.{key}"] = nonincrease_metric(value, key)
    unavailable_diagnostics.extend(
        _append_plastic_impact_metrics(metrics, diagnostics)
    )
    strain_summary = diagnostics.get("strain_summary", {})
    max_plastic = (
        float(strain_summary.get("max_equivalent_plastic_strain", 0.0) or 0.0)
        if isinstance(strain_summary, Mapping)
        else 0.0
    )
    damage_summary = diagnostics.get("plastic_impact_damage_summary", {})
    deletion_records = (
        damage_summary.get("deletion_records", [])
        if isinstance(damage_summary, Mapping)
        else []
    )
    if max_plastic <= 0.0:
        raise CaseUnavailable("nonlinear_impact_did_not_activate_plastic_history")
    if not deletion_records:
        raise CaseUnavailable("nonlinear_impact_did_not_produce_deletion_record")
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "unavailable_diagnostics": sorted(set(unavailable_diagnostics)),
            "campaign_diagnostics": _campaign_diagnostics(diagnostics),
        },
    }


def _case_nonlinear_impact_direct_reduced() -> dict[str, Any]:
    """Elastic impact sized to activate candidate direct-reduced assembly."""

    from anyfesolver.contact import (
        NonlinearTransientConfig,
        RigidSphereImpact,
        SphereContactConfig,
        solve_transient_sphere_impact,
    )
    from anyfesolver.dynamics import TransientConfig

    model = _contact_panel()
    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0025, t_end=0.12, output_nodes=[1]),
        RigidSphereImpact(
            "verification_direct_reduced_elastic_hit",
            radius=0.1,
            mass=1.0,
            start_point=(0.5, 0.5, 0.25),
            travel_direction=(0.0, 0.0, -1.0),
            speed=2.0,
        ),
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        nonlinear_config=NonlinearTransientConfig(
            enabled=True,
            max_iterations=15,
            max_cutbacks=4,
            tangent_reuse_iterations=2,
        ),
    )
    diagnostics = result.diagnostics
    direct_diagnostics = diagnostics.get("impact_reduced_assembly")
    # Baseline revisions predate this diagnostic and intentionally execute the
    # full-coordinate oracle.  A candidate that exposes the selector must
    # activate it here, otherwise this qualification case is not meaningful.
    if isinstance(direct_diagnostics, Mapping) and not bool(
        direct_diagnostics.get("activated", False)
    ):
        reason = direct_diagnostics.get("fallback_reason", "unknown")
        raise CaseUnavailable(f"direct_reduced_impact_not_active:{reason}")

    metrics: dict[str, dict[str, Any]] = {
        "status": categorical_metric(result.status),
        "times": numeric_metric(result.times, "contact_history"),
        "displacements": numeric_metric(result.displacements, "contact_history"),
        "velocities": numeric_metric(result.velocities, "contact_history"),
        "accelerations": numeric_metric(result.accelerations, "contact_history"),
        "sphere_positions": numeric_metric(result.sphere_positions, "contact_history"),
        "sphere_velocities": numeric_metric(result.sphere_velocities, "contact_history"),
        "contact_force_history": numeric_metric(
            result.contact_force_history, "contact_history"
        ),
        "penetration_history": numeric_metric(
            _impact_penetration_history(result.active_contact_history),
            "contact_history",
        ),
        "sphere_impulse": numeric_metric(result.sphere_impulse, "contact_history"),
        "max_penetration": numeric_metric(result.max_penetration, "contact_history"),
        "peak_contact_force": numeric_metric(
            result.peak_contact_force, "contact_history"
        ),
        "contact_duration": numeric_metric(result.contact_duration, "contact_history"),
        "sphere_momentum_balance_error": numeric_metric(
            result.sphere_momentum_balance_error, "contact_history"
        ),
    }
    for key in (
        "kinetic_energy",
        "strain_energy",
        "sphere_kinetic_energy",
        "internal_work",
    ):
        if key in diagnostics:
            metrics[f"energy.{key}"] = numeric_metric(
                diagnostics[key], "contact_history"
            )
    iterations = diagnostics.get("iteration_counts", [])
    if isinstance(iterations, (list, tuple)):
        metrics["iteration_history"] = informational_numeric_metric(
            iterations, "iteration_count"
        )
        metrics["total_iterations"] = nonincrease_metric(sum(iterations))
    cutbacks = _diagnostic_int(diagnostics, ("cutback_count", "num_cutbacks"))
    if cutbacks is not None:
        metrics["cutback_count"] = nonincrease_metric(cutbacks)
    return {
        "metrics": metrics,
        "observations": {
            "topology": _topology(model),
            "direct_reduced_assembly": _json_summary(direct_diagnostics),
            "campaign_diagnostics": _campaign_diagnostics(diagnostics),
        },
    }


CASE_SPECS: dict[str, CaseSpec] = {
    spec.name: spec
    for spec in (
        CaseSpec(
            "global_matrices",
            _case_global_matrices,
            "quick",
            "S4 global K/M/KG values and deterministic signatures",
        ),
        CaseSpec(
            "linear_static",
            _case_linear_static,
            "quick",
            "bounded pressure solve and full displacement vector",
        ),
        CaseSpec(
            "modal",
            _case_modal,
            "quick",
            "constrained axial mode frequency and residual",
        ),
        CaseSpec(
            "buckling",
            _case_buckling,
            "quick",
            "beam-column buckling factors and residuals",
        ),
        CaseSpec(
            "nonlinear_internal",
            _case_nonlinear_internal,
            "quick",
            "active/scalar internal force, tangent, and trial state",
        ),
        CaseSpec(
            "hill48_material",
            _case_hill48_material,
            "quick",
            "mixed Hill-48 load/unload/reload state and tangent path",
        ),
        CaseSpec(
            "generalized_shell",
            _case_generalized_shell,
            "quick",
            "coupled A/B/D/As force, tangent, state, and provenance",
        ),
        CaseSpec(
            "corotational",
            _case_corotational,
            "quick",
            "rotated shell/beam response and rigid-rotation objectivity",
        ),
        CaseSpec(
            "contact_load",
            _case_contact_load,
            "quick",
            "deterministic contact classification and nodal load distribution",
        ),
        CaseSpec(
            "hill48_shell_path",
            _case_hill48_shell_path,
            "standard",
            "global orthotropic Hill shell solve, committed state, and recovery",
        ),
        CaseSpec(
            "arc_length",
            _case_arc_length,
            "standard",
            "bounded softening path, peak load, iterations, and retries",
        ),
        CaseSpec(
            "nonlinear_impact",
            _case_nonlinear_impact,
            "full",
            "plastic impact histories, committed state, damage, and deletion records",
        ),
        CaseSpec(
            "nonlinear_impact_direct_reduced",
            _case_nonlinear_impact_direct_reduced,
            "full",
            "elastic direct-reduced candidate histories against the full-coordinate baseline",
        ),
    )
}


class _PeakRSSMonitor:
    def __init__(self) -> None:
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil  # type: ignore[import-not-found]

            process = psutil.Process(os.getpid())
        except Exception:
            return

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    rss = int(process.memory_info().rss)
                    self.peak = rss if self.peak is None else max(self.peak, rss)
                except Exception:
                    return
                self._stop.wait(0.01)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak


def execute_case(name: str) -> dict[str, Any]:
    spec = CASE_SPECS[name]
    monitor = _PeakRSSMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
        payload = spec.builder()
        status = "completed"
        reason = None
        error = None
    except CaseUnavailable as exc:
        payload = {"metrics": {}, "observations": {}}
        status = "unavailable"
        reason = str(exc)
        error = None
    except Exception as exc:  # keep the remaining independent cases running
        payload = {"metrics": {}, "observations": {}}
        status = "error"
        reason = None
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=30),
        }
    elapsed = time.perf_counter() - started
    peak_rss = monitor.stop()
    result = {
        "status": status,
        "description": spec.description,
        "metrics": payload.get("metrics", {}),
        "observations": payload.get("observations", {}),
        "performance": {
            "wall_seconds": float(elapsed),
            "peak_rss_bytes": peak_rss,
            "python_peak_allocated_bytes": None,
            "qualification_role": "informational_only",
        },
    }
    if reason is not None:
        result["reason"] = reason
    if error is not None:
        result["error"] = error
    return result


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_case_isolated(name: str, timeout_seconds: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sol_ultra_verify_") as tmp:
        output = Path(tmp) / f"{name}.json"
        numba_cache = Path(tmp) / "numba_cache"
        numba_cache.mkdir()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_case",
            "--name",
            name,
            "--output",
            str(output),
        ]
        try:
            environment = os.environ.copy()
            environment["SOL_ULTRA_SOLVER_ROOT"] = str(ROOT)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONHASHSEED"] = "0"
            environment["NUMBA_CACHE_DIR"] = str(numba_cache)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=float(timeout_seconds),
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "unavailable",
                "description": CASE_SPECS[name].description,
                "reason": f"case_timeout_after_{float(timeout_seconds):g}_seconds",
                "metrics": {},
                "observations": {"isolation": "subprocess"},
                "performance": {
                    "wall_seconds": float(timeout_seconds),
                    "peak_rss_bytes": None,
                    "python_peak_allocated_bytes": None,
                    "qualification_role": "informational_only",
                },
            }
        if completed.returncode != 0 or not output.exists():
            return {
                "status": "error",
                "description": CASE_SPECS[name].description,
                "metrics": {},
                "observations": {"isolation": "subprocess"},
                "performance": {
                    "wall_seconds": None,
                    "peak_rss_bytes": None,
                    "python_peak_allocated_bytes": None,
                    "qualification_role": "informational_only",
                },
                "error": {
                    "type": "IsolatedCaseProcessError",
                    "message": f"worker exited with code {completed.returncode}",
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
            }
        result = json.loads(output.read_text(encoding="utf-8"))
        result.setdefault("observations", {})["isolation"] = "subprocess"
        return result


def selected_case_names(suite: str, explicit: Sequence[str] | None = None) -> list[str]:
    if explicit:
        unknown = sorted(set(explicit) - set(CASE_SPECS))
        if unknown:
            raise ValueError(f"unknown cases: {', '.join(unknown)}")
        return list(dict.fromkeys(explicit))
    level = SUITE_LEVEL[suite]
    return [
        name
        for name, spec in CASE_SPECS.items()
        if SUITE_LEVEL[spec.minimum_suite] <= level
    ]


def capture_document(
    *,
    label: str,
    suite: str = "full",
    explicit_cases: Sequence[str] | None = None,
    isolate_cases: bool = True,
    case_timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    requested = selected_case_names(suite, explicit_cases)
    cases: dict[str, Any] = {}
    for name, spec in CASE_SPECS.items():
        if name not in requested:
            cases[name] = {
                "status": "unavailable",
                "description": spec.description,
                "reason": f"not_selected_by_suite:{suite}",
                "metrics": {},
                "observations": {},
                "performance": {
                    "wall_seconds": None,
                    "peak_rss_bytes": None,
                    "python_peak_allocated_bytes": None,
                    "qualification_role": "informational_only",
                },
            }
            continue
        cases[name] = (
            _run_case_isolated(name, case_timeout_seconds)
            if isolate_cases
            else execute_case(name)
        )
    status_counts = {
        status: sum(case.get("status") == status for case in cases.values())
        for status in ("completed", "unavailable", "error")
    }
    requested_unavailable = [
        name
        for name in requested
        if cases[name].get("status") == "unavailable"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "sol_ultra_numerical_capture",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": str(label),
        "source": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "dirty": bool(_git_output("status", "--porcelain")),
            "solver_root": str(ROOT),
        },
        "environment": _environment(),
        "suite": {
            "name": suite,
            "requested_cases": requested,
            "case_timeout_seconds": float(case_timeout_seconds),
            "isolated_cases": bool(isolate_cases),
            "status_counts": status_counts,
            "requested_unavailable_cases": requested_unavailable,
        },
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "cases": cases,
        "methodology": {
            "numeric_payload": "full float64 values plus SHA-256 provenance",
            "comparison": "baseline-authoritative tolerance-based metric comparison",
            "timing": "informational only; numerical qualification is pass/fail",
            "bounds": "fixed topology, fixed step counts, and optional subprocess timeout per case",
        },
    }


def _metric_array(metric: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in metric.get("shape", []))
    values = np.asarray(metric.get("values", []), dtype=float)
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if values.size != expected:
        raise ValueError(
            f"numeric metric payload has {values.size} values for shape {list(shape)}"
        )
    array = values.reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric metric contains non-finite values")
    expected_digest = (metric.get("signature") or {}).get("sha256_float64_le")
    if expected_digest and expected_digest != _sha256_array(array):
        raise ValueError("numeric metric checksum mismatch")
    return array


def validate_capture(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported capture schema {document.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if document.get("artifact_kind") != "sol_ultra_numerical_capture":
        raise ValueError("input is not a Sol Ultra numerical capture artifact")
    cases = document.get("cases")
    if not isinstance(cases, Mapping):
        raise ValueError("capture artifact has no case mapping")
    for case_name, case in cases.items():
        if not isinstance(case, Mapping):
            raise ValueError(f"case {case_name!r} is not a mapping")
        if case.get("status") not in {"completed", "unavailable", "error"}:
            raise ValueError(f"case {case_name!r} has invalid status")
        metrics = case.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError(f"case {case_name!r} has invalid metrics")
        for metric_name, metric in metrics.items():
            if not isinstance(metric, Mapping):
                raise ValueError(f"metric {case_name}.{metric_name} is invalid")
            kind = metric.get("kind")
            if kind == "numeric":
                _metric_array(metric)
            elif kind == "categorical":
                if "value" not in metric:
                    raise ValueError(f"categorical metric {case_name}.{metric_name} has no value")
            else:
                raise ValueError(f"metric {case_name}.{metric_name} has unknown kind {kind!r}")


def _compare_numeric_metric(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    ref = _metric_array(reference)
    cand = _metric_array(candidate)
    comparison = dict(reference.get("comparison") or {})
    method = comparison.get("method", "relative_l2")
    result: dict[str, Any] = {
        "method": method,
        "gate": comparison.get("gate"),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
    }
    if ref.shape != cand.shape:
        if method == "informational":
            result.update(
                status="passed",
                reason="informational_shape_change",
                qualification_role="informational_only",
            )
            return result
        result.update(
            status="failed",
            reason="shape_mismatch",
        )
        return result
    difference = cand - ref
    absolute_l2 = float(np.linalg.norm(difference))
    reference_l2 = float(np.linalg.norm(ref))
    relative_l2 = absolute_l2 / max(reference_l2, np.finfo(float).tiny)
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    result.update(
        absolute_l2_error=absolute_l2,
        relative_l2_error=relative_l2,
        max_absolute_error=max_abs,
    )
    if method == "relative_l2":
        rtol = float(comparison.get("rtol", 0.0))
        atol = float(comparison.get("atol", 0.0))
        threshold = atol + rtol * reference_l2
        passed = absolute_l2 <= threshold
        result.update(rtol=rtol, atol=atol, threshold=threshold)
    elif method == "exact":
        passed = bool(np.array_equal(cand, ref))
    elif method == "nonincrease":
        if ref.size != 1:
            result.update(status="failed", reason="nonincrease_requires_scalar")
            return result
        allowed = float(comparison.get("allowed_increase", 0.0))
        passed = float(cand.reshape(-1)[0]) <= float(ref.reshape(-1)[0]) + allowed
        result.update(
            reference_value=float(ref.reshape(-1)[0]),
            candidate_value=float(cand.reshape(-1)[0]),
            allowed_increase=allowed,
        )
    elif method == "upper_bound":
        if cand.size != 1 or ref.size != 1:
            result.update(status="failed", reason="upper_bound_requires_scalar")
            return result
        limit = float(comparison["limit"])
        reference_value = float(ref.reshape(-1)[0])
        candidate_value = float(cand.reshape(-1)[0])
        passed = reference_value <= limit and candidate_value <= limit
        result.update(
            reference_value=reference_value,
            candidate_value=candidate_value,
            limit=limit,
            baseline_within_limit=reference_value <= limit,
            candidate_within_limit=candidate_value <= limit,
        )
    elif method == "informational":
        passed = True
        result["qualification_role"] = "informational_only"
    else:
        result.update(status="failed", reason=f"unknown_comparison_method:{method}")
        return result
    result["status"] = "passed" if passed else "failed"
    return result


def _performance_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {"qualification_role": "informational_only"}
    for key in ("wall_seconds", "peak_rss_bytes", "python_peak_allocated_bytes"):
        base = baseline.get(key)
        cand = candidate.get(key)
        entry = {"baseline": base, "candidate": cand, "ratio_candidate_over_baseline": None}
        if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and float(base) > 0.0:
            entry["ratio_candidate_over_baseline"] = float(cand) / float(base)
        output[key] = entry
    return output


def compare_documents(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    validate_capture(baseline)
    validate_capture(candidate)
    case_results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    baseline_environment = baseline.get("environment") or {}
    candidate_environment = candidate.get("environment") or {}
    environment_differences: list[dict[str, Any]] = []
    for key in (
        "platform",
        "python",
        "numpy",
        "scipy",
        "numba",
        "pypardiso",
        "anymaterial",
        "anymesher",
        "anyfileio",
        "machine",
        "logical_cpu_count",
        "pypardiso_mkl_rt_configured",
    ):
        reference_value = baseline_environment.get(key)
        candidate_value = candidate_environment.get(key)
        if reference_value != candidate_value:
            item = {
                "field": key,
                "baseline": reference_value,
                "candidate": candidate_value,
            }
            environment_differences.append(item)
            warnings.append({"reason": "environment_mismatch", **item})
    baseline_cases = baseline["cases"]
    candidate_cases = candidate["cases"]
    for case_name in sorted(set(baseline_cases) | set(candidate_cases)):
        base_case = baseline_cases.get(case_name)
        cand_case = candidate_cases.get(case_name)
        if base_case is None or cand_case is None:
            item = {
                "case": case_name,
                "reason": "missing_baseline_case" if base_case is None else "missing_candidate_case",
            }
            unavailable.append(item)
            case_results[case_name] = {"status": "unavailable", **item}
            continue
        base_status = base_case.get("status")
        cand_status = cand_case.get("status")
        result: dict[str, Any] = {
            "baseline_status": base_status,
            "candidate_status": cand_status,
            "performance": _performance_comparison(
                base_case.get("performance", {}),
                cand_case.get("performance", {}),
            ),
            "baseline_observations": base_case.get("observations", {}),
            "candidate_observations": cand_case.get("observations", {}),
        }
        if base_status == "error" or cand_status == "error":
            result["status"] = "failed"
            result["reason"] = "capture_case_error"
            result["baseline_error"] = base_case.get("error")
            result["candidate_error"] = cand_case.get("error")
            failures.append({"case": case_name, "reason": "capture_case_error"})
            case_results[case_name] = result
            continue
        if base_status != "completed" or cand_status != "completed":
            result["status"] = "unavailable"
            result["reason"] = {
                "baseline": base_case.get("reason"),
                "candidate": cand_case.get("reason"),
            }
            unavailable.append({"case": case_name, "reason": result["reason"]})
            case_results[case_name] = result
            continue
        base_metrics = base_case.get("metrics", {})
        cand_metrics = cand_case.get("metrics", {})
        metric_results: dict[str, Any] = {}
        for metric_name, base_metric in base_metrics.items():
            cand_metric = cand_metrics.get(metric_name)
            if cand_metric is None:
                metric_result = {"status": "failed", "reason": "missing_candidate_metric"}
            elif base_metric.get("kind") != cand_metric.get("kind"):
                metric_result = {"status": "failed", "reason": "metric_kind_mismatch"}
            elif base_metric.get("kind") == "categorical":
                passed = base_metric.get("value") == cand_metric.get("value")
                metric_result = {
                    "status": "passed" if passed else "failed",
                    "method": "exact",
                    "reference": base_metric.get("value"),
                    "candidate": cand_metric.get("value"),
                }
            else:
                metric_result = _compare_numeric_metric(base_metric, cand_metric)
                if base_metric.get("comparison") != cand_metric.get("comparison"):
                    warnings.append(
                        {
                            "case": case_name,
                            "metric": metric_name,
                            "reason": "candidate_tolerance_metadata_differs; baseline gate used",
                        }
                    )
            metric_results[metric_name] = metric_result
            if metric_result.get("status") == "failed":
                failures.append(
                    {
                        "case": case_name,
                        "metric": metric_name,
                        "reason": metric_result.get("reason", "tolerance_exceeded"),
                    }
                )
        for metric_name in sorted(set(cand_metrics) - set(base_metrics)):
            warnings.append(
                {
                    "case": case_name,
                    "metric": metric_name,
                    "reason": "extra_candidate_metric",
                }
            )
        result["metrics"] = metric_results
        result["status"] = (
            "failed"
            if any(item.get("status") == "failed" for item in metric_results.values())
            else "passed"
        )
        case_results[case_name] = result
    overall = "failed" if failures else ("incomplete" if unavailable else "passed")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "sol_ultra_numerical_comparison",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": overall,
        "baseline": {
            "label": baseline.get("label"),
            "source": baseline.get("source"),
            "environment": baseline.get("environment"),
            "suite": baseline.get("suite"),
        },
        "candidate": {
            "label": candidate.get("label"),
            "source": candidate.get("source"),
            "environment": candidate.get("environment"),
            "suite": candidate.get("suite"),
        },
        "acceptance_criteria": baseline.get("acceptance_criteria", ACCEPTANCE_CRITERIA),
        "summary": {
            "case_counts": {
                status: sum(item.get("status") == status for item in case_results.values())
                for status in ("passed", "failed", "unavailable")
            },
            "metric_failures": len(failures),
            "unavailable_cases": len(unavailable),
            "warnings": len(warnings),
            "environment_differences": len(environment_differences),
        },
        "cases": case_results,
        "failures": failures,
        "unavailable": unavailable,
        "warnings": warnings,
        "environment_differences": environment_differences,
        "methodology": {
            "tolerance_authority": "baseline artifact",
            "relative_error": "||candidate-baseline||_2 / max(||baseline||_2, tiny)",
            "timing_and_memory": "reported for audit; never used as numerical pass/fail gates",
            "unavailable_semantics": "explicitly yields overall incomplete, not a fabricated pass",
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    baseline = report.get("baseline", {})
    candidate = report.get("candidate", {})
    lines = [
        "# Sol Ultra independent numerical verification",
        "",
        f"Overall status: **{str(report.get('status', 'unknown')).upper()}**",
        "",
        "This report compares complete stored numerical payloads. SHA-256 signatures are provenance only; acceptance uses baseline-authoritative numerical tolerances.",
        "",
        "## Revisions",
        "",
        "| Artifact | Label | Commit | Branch | Dirty at capture | Solver root |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Baseline | {_fmt(baseline.get('label'))} | {_fmt((baseline.get('source') or {}).get('commit'))} | {_fmt((baseline.get('source') or {}).get('branch'))} | {_fmt((baseline.get('source') or {}).get('dirty'))} | {_fmt((baseline.get('source') or {}).get('solver_root'))} |",
        f"| Candidate | {_fmt(candidate.get('label'))} | {_fmt((candidate.get('source') or {}).get('commit'))} | {_fmt((candidate.get('source') or {}).get('branch'))} | {_fmt((candidate.get('source') or {}).get('dirty'))} | {_fmt((candidate.get('source') or {}).get('solver_root'))} |",
    ]
    lines.extend(
        [
            "",
            "## Environment",
            "",
            "| Field | Baseline | Candidate |",
            "| --- | --- | --- |",
        ]
    )
    baseline_environment = baseline.get("environment") or {}
    candidate_environment = candidate.get("environment") or {}
    for key in (
        "platform",
        "python",
        "numpy",
        "scipy",
        "numba",
        "pypardiso",
        "anymaterial",
        "anymesher",
        "anyfileio",
        "logical_cpu_count",
        "pypardiso_mkl_rt_configured",
    ):
        lines.append(
            f"| {key} | {_fmt(baseline_environment.get(key))} | {_fmt(candidate_environment.get(key))} |"
        )
    lines.extend(
        [
            "",
            "## Case summary",
            "",
            "| Case | Result | Max relative L2 error | Candidate/baseline wall time | Candidate/baseline peak RSS |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name, case in sorted((report.get("cases") or {}).items()):
        relative_errors = [
            metric.get("relative_l2_error")
            for metric in (case.get("metrics") or {}).values()
            if isinstance(metric.get("relative_l2_error"), (int, float))
        ]
        maximum = max(relative_errors) if relative_errors else None
        performance = case.get("performance") or {}
        wall = (performance.get("wall_seconds") or {}).get("ratio_candidate_over_baseline")
        rss = (performance.get("peak_rss_bytes") or {}).get("ratio_candidate_over_baseline")
        lines.append(
            f"| {name} | {case.get('status')} | {_fmt(maximum)} | {_fmt(wall)} | {_fmt(rss)} |"
        )

    lines.extend(["", "## Numerical acceptance criteria", ""])
    lines.extend(
        [
            "| Gate | Method | Relative tolerance | Absolute tolerance | Source |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for name, criterion in sorted((report.get("acceptance_criteria") or {}).items()):
        lines.append(
            f"| {name} | {criterion.get('method')} | {_fmt(criterion.get('rtol'))} | {_fmt(criterion.get('atol'))} | {criterion.get('source', '')} |"
        )

    lines.extend(["", "## Failures", ""])
    failures = report.get("failures") or []
    if failures:
        for item in failures:
            metric = f" / {item.get('metric')}" if item.get("metric") else ""
            lines.append(f"- `{item.get('case')}{metric}`: {item.get('reason')}")
    else:
        lines.append("No numerical failures were detected.")

    lines.extend(["", "## Unavailable or incomplete coverage", ""])
    unavailable = report.get("unavailable") or []
    if unavailable:
        for item in unavailable:
            lines.append(f"- `{item.get('case')}`: `{json.dumps(item.get('reason'), sort_keys=True)}`")
    else:
        lines.append("All cases were available in both artifacts.")

    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings") or []
    if warnings:
        for item in warnings:
            lines.append(f"- `{json.dumps(item, sort_keys=True)}`")
    else:
        lines.append("No comparison warnings were emitted.")

    lines.extend(["", "## Candidate path and fallback observations", ""])
    any_diagnostics = False
    for case_name, case in sorted((report.get("cases") or {}).items()):
        observations = case.get("candidate_observations") or {}
        diagnostics = observations.get("campaign_diagnostics") or {}
        unavailable_diagnostics = observations.get("unavailable_diagnostics") or []
        if diagnostics or unavailable_diagnostics:
            any_diagnostics = True
            lines.append(f"### {case_name}")
            lines.append("")
            if unavailable_diagnostics:
                lines.append(
                    "Unavailable diagnostics: "
                    + ", ".join(f"`{item}`" for item in unavailable_diagnostics)
                )
                lines.append("")
            if diagnostics:
                lines.append("```json")
                lines.append(json.dumps(diagnostics, indent=2, sort_keys=True))
                lines.append("```")
                lines.append("")
    if not any_diagnostics:
        lines.append("No campaign-path diagnostics were exposed by the candidate artifact.")
        lines.append("")

    lines.extend(
        [
            "## Methodology and reproduction",
            "",
            "Cases use fixed meshes, fixed random seeds, fixed continuation/time-step bounds, and optional per-case subprocess timeouts. Iteration, retry, and cutback counts are compared where exposed. Timing and memory are informational and must be evaluated with the separate matched performance benchmark.",
            "",
            "```powershell",
            "$env:PYPARDISO_MKL_RT = '<path-to-mkl_rt.dll>'",
            "$harness = 'C:\\Github\\ANYfeSolver-verification\\scripts\\verify_sol_ultra_numerics.py'",
            "C:\\Github\\ANYfeSolver\\.venv\\Scripts\\python.exe $harness capture --solver-root C:\\Github\\ANYfeSolver-baseline --label baseline --suite full --output baseline.json",
            "C:\\Github\\ANYfeSolver\\.venv\\Scripts\\python.exe $harness capture --solver-root C:\\Github\\ANYfeSolver-candidate --label candidate --suite full --output candidate.json",
            "C:\\Github\\ANYfeSolver\\.venv\\Scripts\\python.exe $harness compare --baseline baseline.json --candidate candidate.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_case_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _capture_command(args: argparse.Namespace) -> int:
    if args.solver_root is not None:
        configure_solver_root(args.solver_root)
    document = capture_document(
        label=args.label,
        suite=args.suite,
        explicit_cases=_parse_case_list(args.cases),
        isolate_cases=not args.in_process,
        case_timeout_seconds=args.case_timeout_seconds,
    )
    _write_json(args.output, document)
    requested_unavailable = document["suite"]["requested_unavailable_cases"]
    errors = document["suite"]["status_counts"]["error"]
    print(
        f"Wrote {args.output}: {document['suite']['status_counts']}"
    )
    if errors:
        return 1
    if requested_unavailable:
        return 2
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_documents(baseline, candidate)
    _write_json(args.json_report, report)
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"{report['status'].upper()}: wrote {args.json_report} and {args.markdown_report}"
    )
    if report["status"] == "passed" or (
        report["status"] == "incomplete" and args.allow_incomplete
    ):
        return 0
    return 1 if report["status"] == "failed" else 2


def _case_command(args: argparse.Namespace) -> int:
    result = execute_case(args.name)
    _write_json(args.output, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture one bounded numerical artifact")
    capture.add_argument("--label", required=True, help="human-readable baseline/candidate label")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--solver-root",
        type=Path,
        default=None,
        help="exact ANYfeSolver checkout to import and record (defaults to the harness checkout)",
    )
    capture.add_argument("--suite", choices=tuple(SUITE_LEVEL), default="full")
    capture.add_argument(
        "--cases",
        help="comma-separated explicit case names; overrides suite selection",
    )
    capture.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=180.0,
        help="hard timeout per isolated case",
    )
    capture.add_argument(
        "--in-process",
        action="store_true",
        help="disable subprocess isolation/timeouts (development only)",
    )
    capture.set_defaults(func=_capture_command)

    compare = subparsers.add_parser("compare", help="compare baseline and candidate captures")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    compare.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    compare.add_argument("--allow-incomplete", action="store_true")
    compare.set_defaults(func=_compare_command)

    list_cases = subparsers.add_parser("list-cases", help="list deterministic case inventory")

    def show_cases(_args: argparse.Namespace) -> int:
        for spec in CASE_SPECS.values():
            print(f"{spec.name}\t{spec.minimum_suite}\t{spec.description}")
        return 0

    list_cases.set_defaults(func=show_cases)

    worker = subparsers.add_parser("_case", help=argparse.SUPPRESS)
    worker.add_argument("--name", choices=tuple(CASE_SPECS), required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.set_defaults(func=_case_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "case_timeout_seconds", 1.0) <= 0.0:
        parser.error("--case-timeout-seconds must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
