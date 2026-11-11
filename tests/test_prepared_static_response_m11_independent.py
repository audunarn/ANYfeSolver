"""Independent small exact-system controls for the conditional M11 bound."""
from fractions import Fraction as F

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.assembly import _backward_error
from prepared_static_response_m11_gate import GateRefusal, gate_enclosure


def exact_two_by_two_solution(a, b):
    a = [[F(float(x)) for x in row] for row in a]
    b = [F(float(x)) for x in b]
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return [(a[1][1]*b[0]-a[0][1]*b[1])/determinant,
            (a[0][0]*b[1]-a[1][0]*b[0])/determinant]


@pytest.mark.parametrize('a,b', [
    ([[1., 0.], [0., 1.]], [1., -2.]),
    ([[0., 2.], [4., 1.]], [1., 3.]),
    ([[8., -2.], [-2., 4.]], [1., -1.]),
])
def test_unseen_bound_contains_accepted_adversaries_without_using_them(a, b):
    a = np.array(a); b = np.array(b)
    exact = exact_two_by_two_solution(a, b)
    candidate = np.array([float(x) for x in exact])
    result = gate_enclosure(sparse.csr_matrix(a), b, candidate)
    assert result['status'] == 'CONDITIONAL_ENCLOSURE'
    # Probe vectors are created after the enclosure and are not its inputs.
    admitted = 0
    for direction in (-1., 1.):
        for relative in (0., 1e-12, 1e-11, 1e-10):
            probe = candidate + direction * relative
            if _backward_error(sparse.csr_matrix(a), probe, b) <= 1e-10:
                admitted += 1
                assert max(abs(F(float(x))-F(float(y)))
                           for x, y in zip(probe, candidate)) <= result['radius']
    assert admitted >= 6
    assert result['uses_observed_direct'] is False
    assert max(abs(F(float(x))-y) for x, y in zip(candidate, exact)) <= result['radius']


def test_generic_gate_does_not_certify_physical_reactions():
    # Same unconstrained RHS, different constrained loads, identical solution.
    stiffness = np.diag([2., 3.])
    transform = np.array([[0.], [1.]])
    first = np.array([1., 6.]); second = np.array([7., 6.])
    assert np.array_equal(transform.T @ first, transform.T @ second)
    displacement = transform @ np.array([2.])
    assert np.array_equal((stiffness @ displacement-first)[1:],
                          (stiffness @ displacement-second)[1:])
    assert (stiffness @ displacement-first)[0] != (stiffness @ displacement-second)[0]


def test_domain_refusals_are_preserved():
    with pytest.raises(GateRefusal, match='zero_RHS'):
        gate_enclosure(sparse.eye(2, format='csr'), np.zeros(2), np.zeros(2))
    with pytest.raises(GateRefusal, match='canonical_CSR'):
        gate_enclosure(sparse.eye(2, format='csc'), np.ones(2), np.ones(2))
    result = gate_enclosure(sparse.diags([1., 1e-14], format='csr'),
                           np.ones(2), np.array([1., 1e14]))
    assert result['status'] == 'INCONCLUSIVE_GATE_CONTRACTION'
