"""Existing linear-static dispatch for the current private native families.

This verifies routing and reference linearization, not nonlinear state issuance
or a new engineering qualification. The uncondensed 42-coordinate comparison
is a distinct solve of the same original reference operator.
"""
import numpy as np
import pytest

from anyfesolver.assembly import solve_linear
from anyfesolver.boundary import LoadCase
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_native_generalized import problem as generalized
from test_ge_beam3_native_fibre_static_element import problem as fibre


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
@pytest.mark.parametrize('curved', (False, True))
def test_actual_linear_route_matches_full_stationary_solve(family, curved, monkeypatch, tmp_path):
    source, _ = (generalized if family == 'generalized' else fibre)(curved, False)
    owner = analysis(source)
    element = owner.model.mesh.elements[1]
    initial = owner._initial()
    before = canonical(initial)
    response = initial[1]['response']
    full = response.full_spatial_jacobian if family == 'generalized' else response.full_hessian
    assert full.shape == (42, 42)
    load = LoadCase('native-reference-linear')
    tip_force = np.array([.003, -.002, .001])
    load.add_nodal_load(3, forces=tip_force)
    rhs = np.zeros(42); rhs[12:15] = tip_force
    free = np.arange(6, 42)
    expected = np.zeros(42)
    expected[free] = np.linalg.solve(full[np.ix_(free, free)], rhs[free])
    calls = []
    original = type(element).compute_stiffness_matrix
    def observed(self, mesh, material):
        calls.append(self)
        return original(self, mesh, material)
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy beam mechanics reached by native linear route')
    monkeypatch.setattr(type(element), 'compute_stiffness_matrix', observed)
    monkeypatch.setattr(BeamElement, 'compute_stiffness_matrix', forbidden)
    monkeypatch.setattr(QuadraticBeamElement, 'compute_stiffness_matrix', forbidden)
    actual, info = solve_linear(owner.model, load)
    assert calls and all(item is element for item in calls)
    assert info['convergence_info']['status'] == 'converged'
    assert info['result_case']['analysis_case']['analysis_type'] == 'linear_static'
    assert relative(actual, expected[:18]) < 1e-11
    recovered = np.r_[actual, np.linalg.solve(full[18:, 18:], -full[18:, :18] @ actual)]
    residual = full @ recovered - rhs
    assert relative(residual[free], np.zeros(len(free))) < 1e-11
    assert relative(recovered, expected) < 1e-11
    assert abs(float(recovered @ full @ recovered - rhs @ recovered)) < 1e-11
    root_force = residual[:3]
    root_moment = residual[3:6]
    x = element.operator.reference.coordinates
    assert np.linalg.norm(root_force + tip_force) < 1e-11
    assert np.linalg.norm(root_moment + np.cross(x[2]-x[0], tip_force)) < 1e-11
    assert canonical(initial) == before
    assert canonical(owner._initial()) == before
    owner._guard()
    (tmp_path/'linear.json').write_bytes(canonical(dict(family=family, curved=curved,
        displacement=actual, internal_increment=recovered[18:], reactions=residual[:6],
        full_stationary_solution=expected, original_residual=residual,
        definition_sha256=owner.identity, production_qualified=False,
        nonlinear_checkpoint_issued=False, linearized_recovery_only=True)))


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_linear_cancellation_does_not_enter_native_mechanics(family, monkeypatch):
    model, element = (generalized if family == 'generalized' else fibre)(False, False)
    token = CancellationToken(); token.cancel()
    def forbidden(*args, **kwargs):
        raise AssertionError('linear cancellation entered mechanics')
    monkeypatch.setattr(type(element), 'compute_stiffness_matrix', forbidden)
    with pytest.raises(SolveCancelled):
        solve_linear(model, LoadCase('cancelled'), cancellation_token=token)
