"""A factorization handle must never solve a matrix the caller edited after factorizing.

Regression for the 2026-11-24 finding: a GENERAL PARDISO handle aliased the
caller's CSR arrays, so after the caller scaled ``.data`` in place and the
pattern slot moved on, the stale-handle rebuild factorized the *edited* values
and a cache hit for the original matrix returned the edited matrix's solution
with status ``ok``.  (Explicit signatures are caller-owned revision stamps and
stay trusted; see ``FactorizationCache``.)

Under the D2 slot contract (merged 2026-11-25) a stale PARDISO handle is never
rebuilt: it raises :class:`~anyfesolver.linalg.PardisoStaleHandleError`, a
stale cache entry is a miss, and a holder that keeps solving one factorization
pins it with a lease.  The value assertions are unchanged; the PARDISO cases
assert the typed stale error where main rebuilt, and hold leases where main
relied on the rebuild.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import linalg

_HAS_PYPARDISO = importlib.util.find_spec("pypardiso") is not None

_EXPECTED = np.array([1.0, 0.5, 1.0 / 3.0])


def _backends():
    yield pytest.param(linalg.SparseSolverBackend, id="superlu")
    yield pytest.param(
        linalg.PyPardisoSolverBackend,
        id="pypardiso",
        marks=pytest.mark.skipif(not _HAS_PYPARDISO, reason="pypardiso not installed"),
    )


@pytest.mark.parametrize("backend_type", _backends())
@pytest.mark.parametrize("matrix_class", ["general", "spd", "symmetric_indefinite"])
def test_cache_hit_after_in_place_edit_solves_the_original_matrix(backend_type, matrix_class):
    cache = linalg.FactorizationCache(backend=backend_type())
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    cache.factorize(A, matrix_class)
    A.data *= 4.0
    edited = cache.factorize(A, matrix_class)
    np.testing.assert_allclose(edited.solve(np.ones(3)), _EXPECTED / 4.0, rtol=1e-14)

    fresh = cache.factorize(sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), matrix_class)
    assert fresh.status == "ok"
    # SuperLU: a hit.  PARDISO (D2): the edited matrix refactorized the shared
    # slot, so the original entry is stale and the lookup is a miss that
    # factorizes the original values again (never the edited ones).
    assert fresh.metadata["cache_hit"] is (backend_type is linalg.SparseSolverBackend)
    np.testing.assert_allclose(fresh.solve(np.ones(3)), _EXPECTED, rtol=1e-14)


@pytest.mark.skipif(not _HAS_PYPARDISO, reason="pypardiso not installed")
def test_pardiso_handle_owns_its_matrix_values():
    backend = linalg.PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    try:
        first = backend.factorize(A, linalg.MatrixClass.GENERAL)
        with linalg.lease_handle(first):
            A.data *= 4.0
            # Same pattern: the leased slot is never refactorized (a new slot
            # takes the edited values), and the holder solves its own values.
            backend.factorize(A, linalg.MatrixClass.GENERAL)
            np.testing.assert_allclose(first.solve(np.ones(3)), _EXPECTED, rtol=1e-14)
        # Unleased and stale: never rebuilt, neither from its snapshot nor
        # from the caller's arrays.
        with pytest.raises(linalg.PardisoStaleHandleError):
            first.solve(np.ones(3))
        second = backend.factorize(A, linalg.MatrixClass.GENERAL)
        A.data *= 0.5
        backend.factorize(A, linalg.MatrixClass.GENERAL)
        with pytest.raises(linalg.PardisoStaleHandleError):
            second.solve(np.ones(3))
    finally:
        backend.release_pattern_slots()


def _indefinite():
    # Symmetric, genuinely indefinite, with off-diagonal coupling.
    return sparse.csr_matrix(np.array([[2.0, 1.0, 0.0], [1.0, -3.0, 1.0], [0.0, 1.0, 4.0]]))


@pytest.mark.parametrize("backend_type", _backends())
@pytest.mark.parametrize("matrix_class", ["general", "symmetric_indefinite"])
def test_live_handle_keeps_solving_its_own_values_after_edit_and_replacement(backend_type, matrix_class):
    backend = backend_type()
    A = _indefinite()
    original = A.copy()
    rhs = np.array([1.0, -2.0, 0.5])
    expected = np.linalg.solve(original.toarray(), rhs)
    # A live holder pins its factorization (D2; a no-op lease for SuperLU).
    with linalg.leased_factorization(A, matrix_class, backend=backend) as lease:
        first = lease.handle
        np.testing.assert_allclose(first.solve(rhs), expected, rtol=1e-13)
        A.data *= -2.5
        linalg.factorize(A, matrix_class, backend=backend)  # same pattern, new values
        np.testing.assert_allclose(first.solve(rhs), expected, rtol=1e-13)


@pytest.mark.parametrize("backend_type", _backends())
def test_index_edit_after_factorize_does_not_reach_the_handle(backend_type):
    A = sparse.csr_matrix(np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]))
    rhs = np.ones(3)
    expected = np.linalg.solve(A.toarray(), rhs)
    cache = linalg.FactorizationCache(backend=backend_type())
    handle = cache.factorize(A, "general")
    A.indices[1] = 2  # move the (0, 1) coupling to (0, 2): same nnz, different matrix
    edited = cache.factorize(A, "general")
    assert edited is not handle
    np.testing.assert_allclose(edited.solve(rhs), np.linalg.solve(A.toarray(), rhs), rtol=1e-13)
    np.testing.assert_allclose(handle.solve(rhs), expected, rtol=1e-13)


@pytest.mark.parametrize("backend_type", _backends())
def test_cached_inverse_operator_after_in_place_edit(backend_type):
    A = _indefinite()
    rhs = np.array([1.0, -2.0, 0.5])
    expected = np.linalg.solve(A.toarray(), rhs)
    cache = linalg.FactorizationCache(backend=backend_type())
    # The first operator is held across the second factorization: leased (D2).
    with linalg.leased_inverse_operator(A, "general", cache=cache) as lease:
        operator = lease.operator
        A.data *= 3.0
        edited_operator, _ = linalg.cached_inverse_operator(A, "general", cache=cache)
        np.testing.assert_allclose(operator.matvec(rhs), expected, rtol=1e-13)
        np.testing.assert_allclose(edited_operator.matvec(rhs), expected / 3.0, rtol=1e-13)
    if backend_type is linalg.PyPardisoSolverBackend:
        # Unleased, the same sequence leaves the first handle stale (typed error).
        first, handle = linalg.cached_inverse_operator(_indefinite(), "general", cache=cache)
        B = _indefinite()
        B.data *= 5.0
        linalg.cached_inverse_operator(B, "general", cache=cache)
        with pytest.raises(linalg.PardisoStaleHandleError):
            first.matvec(rhs)
