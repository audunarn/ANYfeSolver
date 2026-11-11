import numpy as np
import pytest
from scipy import sparse

from anyfesolver.experimental_csr_assembly import (
    CSRPromotionGates,
    ExperimentalCSRAssemblyPlan,
    benchmark_experimental_csr_assembly,
)
from anyfesolver.mesh_gen import generate_beam_mesh, generate_simple_panel_mesh
from anyfesolver.matrix_assembly import _get_cached_sparsity_pattern


@pytest.mark.parametrize(
    "model",
    [
        generate_beam_mesh(2.0, num_divisions=3),
        generate_simple_panel_mesh(1.0, 0.5, 0.01, num_divisions_x=2, num_divisions_y=1),
    ],
)
def test_direct_csr_scatter_matches_coo_exactly(model):
    rows, cols = _get_cached_sparsity_pattern(model.mesh, "test")
    rng = np.random.default_rng(20260808)
    values = rng.normal(size=rows.size)
    reference = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(model.mesh.dof_manager.total_dofs,) * 2,
    ).tocsr()
    candidate = ExperimentalCSRAssemblyPlan.from_model(model, "test").assemble(values)
    assert candidate.shape == reference.shape
    assert candidate.nnz == reference.nnz
    assert sparse.linalg.norm(candidate - reference) <= 1.0e-12 * max(
        sparse.linalg.norm(reference), 1.0
    )


def test_benchmark_reports_separate_phases_and_rejects_incomplete_evidence():
    model = generate_simple_panel_mesh(
        1.0, 0.5, 0.01, num_divisions_x=2, num_divisions_y=1
    )
    report = benchmark_experimental_csr_assembly(model, repetitions=2)
    assert report["production_path_changed"] is False
    assert report["timing"]["local_kernels_median_seconds"] >= 0.0
    assert report["timing"]["coo_conversion_median_seconds"] >= 0.0
    assert report["timing"]["csr_scatter_median_seconds"] >= 0.0
    assert report["equivalence"]["relative_matrix_error"] <= 1.0e-12
    assert not report["promotion"]["eligible"]


def test_promotion_gates_are_the_release_thresholds():
    gates = CSRPromotionGates()
    assert gates.minimum_assembly_improvement == 0.20
    assert gates.minimum_end_to_end_improvement == 0.05
    assert gates.minimum_peak_memory_improvement == 0.15
    assert gates.maximum_case_regression == 0.05
    assert gates.maximum_relative_matrix_error == 1.0e-12
