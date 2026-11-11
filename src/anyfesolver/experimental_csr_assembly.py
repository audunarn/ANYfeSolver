"""Opt-in topology-cached CSR assembly prototype.

This is a clean-room benchmark implementation.  Production K/M/KG assembly
continues to use the qualified COO path in :mod:`anyfesolver.matrix_assembly`.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .matrix_assembly import _check_element_matrix_shape, _get_cached_sparsity_pattern

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass(frozen=True)
class CSRPromotionGates:
    """Evidence thresholds required before the prototype can be promoted."""

    minimum_assembly_improvement: float = 0.20
    minimum_end_to_end_improvement: float = 0.05
    minimum_peak_memory_improvement: float = 0.15
    maximum_case_regression: float = 0.05
    maximum_relative_matrix_error: float = 1.0e-12

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class ExperimentalCSRAssemblyPlan:
    """Cached CSR topology and local-entry-to-global-data scatter map."""

    def __init__(self, total_dofs: int, indptr: np.ndarray, indices: np.ndarray, scatter: np.ndarray):
        self.total_dofs = int(total_dofs)
        self.indptr = np.asarray(indptr, dtype=np.intp)
        self.indices = np.asarray(indices, dtype=np.intp)
        self.scatter = np.asarray(scatter, dtype=np.intp)

    @classmethod
    def from_model(cls, model: "FEModel", matrix_type: str = "stiffness") -> "ExperimentalCSRAssemblyPlan":
        total_dofs = int(model.mesh.dof_manager.total_dofs)
        rows, cols = _get_cached_sparsity_pattern(model.mesh, str(matrix_type))
        if rows.size == 0:
            return cls(total_dofs, np.zeros(total_dofs + 1, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
        keys = rows.astype(np.int64) * np.int64(total_dofs) + cols.astype(np.int64)
        unique, inverse = np.unique(keys, return_inverse=True)
        unique_rows = (unique // total_dofs).astype(np.intp, copy=False)
        indices = (unique % total_dofs).astype(np.intp, copy=False)
        row_counts = np.bincount(unique_rows, minlength=total_dofs)
        indptr = np.empty(total_dofs + 1, dtype=np.intp)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        return cls(total_dofs, indptr, indices, inverse.astype(np.intp, copy=False))

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def assemble(self, flattened_local_matrices: np.ndarray) -> sparse.csr_matrix:
        values = np.asarray(flattened_local_matrices, dtype=float).reshape(-1)
        if values.size != self.scatter.size:
            raise ValueError(
                f"Local data contains {values.size} entries; topology expects {self.scatter.size}."
            )
        data = np.bincount(self.scatter, weights=values, minlength=self.nnz).astype(float, copy=False)
        return sparse.csr_matrix(
            (data, self.indices.copy(), self.indptr.copy()),
            shape=(self.total_dofs, self.total_dofs),
        )


def _local_matrix_data(
    model: "FEModel",
    matrix_type: str,
    element_states: Optional[Mapping[int, Any]],
) -> np.ndarray:
    blocks = []
    for element_id, element in model.mesh.elements.items():
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        if dofs.size == 0:
            continue
        material = model.get_material(element.material_name)
        if matrix_type == "stiffness":
            local = element.compute_stiffness_matrix(model.mesh, material)
        elif matrix_type == "mass":
            local = element.compute_mass_matrix(model.mesh, material)
        elif matrix_type == "geometric_stiffness":
            state = None if element_states is None else element_states.get(int(element_id))
            local = element.compute_geometric_stiffness_matrix(model.mesh, material, state)
        else:
            raise ValueError("matrix_type must be stiffness, mass or geometric_stiffness")
        blocks.append(
            _check_element_matrix_shape(int(element_id), matrix_type, local, int(dofs.size)).ravel()
        )
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=float)


def _measure_repetitions(operation, repetitions: int) -> Tuple[list[float], int, Any]:
    timings = []
    peak = 0
    output = None
    for _ in range(repetitions):
        tracemalloc.start()
        start = time.perf_counter()
        try:
            output = operation()
        finally:
            timings.append(time.perf_counter() - start)
            _current, measured_peak = tracemalloc.get_traced_memory()
            peak = max(peak, int(measured_peak))
            tracemalloc.stop()
    return timings, peak, output


def benchmark_experimental_csr_assembly(
    model: "FEModel",
    matrix_type: str = "stiffness",
    *,
    element_states: Optional[Mapping[int, Any]] = None,
    repetitions: int = 7,
    end_to_end_improvement: Optional[float] = None,
    gates: CSRPromotionGates = CSRPromotionGates(),
) -> Dict[str, Any]:
    """Compare identical local matrices through COO and persistent CSR scatter.

    Setup is reported separately and excluded from warmed scatter medians.  The
    caller may supply independently measured end-to-end improvement; absent
    that evidence, promotion is always rejected.
    """

    if int(repetitions) <= 0:
        raise ValueError("repetitions must be positive")
    repetitions = int(repetitions)
    matrix_type = str(matrix_type)
    rows, cols = _get_cached_sparsity_pattern(model.mesh, matrix_type)

    setup_start = time.perf_counter()
    plan = ExperimentalCSRAssemblyPlan.from_model(model, matrix_type)
    setup_seconds = time.perf_counter() - setup_start

    local_times, local_peak, local_values = _measure_repetitions(
        lambda: _local_matrix_data(model, matrix_type, element_states), repetitions
    )
    assert local_values is not None
    total_dofs = int(model.mesh.dof_manager.total_dofs)

    coo_times, coo_peak, coo_matrix = _measure_repetitions(
        lambda: sparse.coo_matrix(
            (local_values, (rows, cols)), shape=(total_dofs, total_dofs), dtype=float
        ).tocsr(),
        repetitions,
    )
    csr_times, csr_peak, csr_matrix = _measure_repetitions(
        lambda: plan.assemble(local_values), repetitions
    )
    assert coo_matrix is not None and csr_matrix is not None

    denominator = max(float(sparse.linalg.norm(coo_matrix)), 1.0)
    relative_error = float(sparse.linalg.norm(coo_matrix - csr_matrix) / denominator)
    symmetry_error = float(
        sparse.linalg.norm(csr_matrix - csr_matrix.T)
        / max(float(sparse.linalg.norm(csr_matrix)), 1.0)
    )
    local_median = float(statistics.median(local_times))
    coo_median = float(statistics.median(coo_times))
    csr_median = float(statistics.median(csr_times))
    coo_total = local_median + coo_median
    csr_total = local_median + csr_median
    assembly_improvement = (coo_total - csr_total) / max(coo_total, np.finfo(float).eps)
    peak_coo = local_peak + coo_peak
    peak_csr = local_peak + csr_peak
    memory_improvement = (peak_coo - peak_csr) / max(float(peak_coo), 1.0)
    end_to_end_pass = (
        end_to_end_improvement is not None
        and float(end_to_end_improvement) >= gates.minimum_end_to_end_improvement
    )
    memory_pass = memory_improvement >= gates.minimum_peak_memory_improvement
    promotion_eligible = bool(
        relative_error <= gates.maximum_relative_matrix_error
        and assembly_improvement >= gates.minimum_assembly_improvement
        and (end_to_end_pass or memory_pass)
    )
    return {
        "status": "completed",
        "prototype": "topology_cached_direct_csr_scatter",
        "production_path_changed": False,
        "matrix_type": matrix_type,
        "topology": {
            "dofs": total_dofs,
            "elements": int(model.mesh.num_elements),
            "coo_entries": int(local_values.size),
            "csr_nnz": int(plan.nnz),
        },
        "timing": {
            "repetitions": repetitions,
            "cold_csr_topology_setup_seconds": float(setup_seconds),
            "local_kernels_median_seconds": local_median,
            "coo_conversion_median_seconds": coo_median,
            "csr_scatter_median_seconds": csr_median,
            "coo_complete_assembly_median_seconds": coo_total,
            "csr_complete_assembly_median_seconds": csr_total,
            "complete_assembly_improvement": float(assembly_improvement),
            "end_to_end_improvement": end_to_end_improvement,
        },
        "memory": {
            "measurement": "python_allocation_peak_tracemalloc",
            "local_kernel_peak_bytes": int(local_peak),
            "coo_complete_peak_bytes": int(peak_coo),
            "csr_complete_peak_bytes": int(peak_csr),
            "peak_improvement": float(memory_improvement),
        },
        "equivalence": {
            "relative_matrix_error": relative_error,
            "same_shape": coo_matrix.shape == csr_matrix.shape,
            "same_nnz": int(coo_matrix.nnz) == int(csr_matrix.nnz),
            "symmetry_error": symmetry_error,
        },
        "promotion": {
            "eligible": promotion_eligible,
            "gates": gates.to_dict(),
            "assembly_gate_passed": assembly_improvement >= gates.minimum_assembly_improvement,
            "end_to_end_gate_passed": end_to_end_pass,
            "memory_gate_passed": memory_pass,
            "reason": (
                "all supplied evidence gates passed"
                if promotion_eligible
                else "prototype remains opt-in; complete promotion evidence is insufficient"
            ),
        },
    }
