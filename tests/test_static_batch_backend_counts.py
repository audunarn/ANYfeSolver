import numpy as np
import pytest
from scipy import sparse

from anyfesolver.linalg import (FactorizationCache, MatrixClass,
                             PyPardisoSolverBackend, SparseSolverBackend,
                             backend_operation_counts, _HAS_PYPARDISO)


def test_scipy_counts_actual_factorizations_and_cache_hits():
    before = backend_operation_counts()
    cache = FactorizationCache(backend=SparseSolverBackend())
    matrix = sparse.eye(3, format='csr')
    first = cache.factorize(matrix, MatrixClass.SPD)
    first.solve_many(np.ones((3, 4)))
    assert cache.factorize(matrix, MatrixClass.SPD) is first
    after = backend_operation_counts()
    assert after['scipy_factorization_attempts'] - before['scipy_factorization_attempts'] == 1
    assert first.diagnostics()['factorization_count'] == 1


@pytest.mark.skipif(not _HAS_PYPARDISO, reason='pypardiso not installed')
def test_pardiso_counts_hidden_rebuild_after_replacement_and_eviction():
    before = backend_operation_counts()
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    try:
        matrix = sparse.csr_matrix([[4., 1.], [1., 3.]])
        first = backend.factorize(matrix, MatrixClass.SPD)
        second = backend.factorize(matrix * 2, MatrixClass.SPD)
        assert first.status == second.status == 'ok'
        np.testing.assert_allclose(matrix @ first.solve(np.ones(2)), 1.)
        assert first.diagnostics()['factorization_count'] == 2
        backend.release_pattern_slots()
        np.testing.assert_allclose(matrix * 2 @ second.solve(np.ones(2)), 1.)
        assert second.diagnostics()['factorization_count'] == 2
        after = backend_operation_counts()
        assert after['pardiso_factorization_attempts'] - before['pardiso_factorization_attempts'] == 4
        assert after['pardiso_stale_rebuilds'] - before['pardiso_stale_rebuilds'] == 2
    finally:
        backend.release_pattern_slots()
