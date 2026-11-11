"""Original-factor buckling and actual force-owner integration."""
from hashlib import sha256

import numpy as np
import pytest
from scipy.linalg import eig

from anyfesolver import _ge_beam3_native_buckling as buckling
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from test_ge_beam3_native_analysis import analysis, generalized_problem, fibre_problem


def roots(matrix, denominator, bounds=(0., 1e8)):
    """Independent QZ, not the candidate's material whitening algorithm."""
    values = eig(matrix, denominator, right=False)
    return np.sort(np.array([float(v.real) for v in values if np.isfinite(v)
        and abs(v.imag) <= 1e-10*max(1., abs(v.real)) and bounds[0] < v.real <= bounds[1]]))


def test_signed_multiplier_window_scaling_and_no_mass_substitution():
    f = np.diag([1e-6, 2., 3e6])
    g = np.diag([-1e-12/2, -4./3, 9e12/4])
    result = buckling.factor_buckling(np.eye(3), f, g, (0, 1, 2), bounds=(0., 10.))
    np.testing.assert_allclose(result.multipliers, [2., 3.], rtol=1e-12, atol=0.)
    np.testing.assert_allclose((f@result.modes).T@(f@result.modes), np.eye(2), atol=1e-12)
    assert result.positive_multipliers_in_window == 2 and not result.production_qualified
    assert not result.nonlinear_critical_load_authorized
    assert not result.modes.flags.writeable and not result.multipliers.flags.writeable
    positive = buckling.factor_buckling(np.eye(3), f, np.abs(g), (0, 1, 2), bounds=(0., 10.))
    assert positive.status == 'NO_MULTIPLIERS_IN_REQUESTED_WINDOW' and positive.modes.shape == (3, 0)
    zero = buckling.factor_buckling(np.eye(3), f, np.zeros((3, 3)), (0, 1, 2), bounds=(0., 10.))
    assert zero.multipliers.size == 0


def test_coupled_full_pencil_and_repeated_modes():
    rng = np.random.default_rng(761)
    transform = np.eye(5)+.12*rng.normal(size=(5, 5))
    left = np.diag([.7, .9, 1.2, 1.8, 2.])
    f = np.diag([1., 2., 3., 4., 5.]) @ transform
    right = np.linalg.solve(left, f)
    diagonal = np.diag([-1./2, -4./2, -9./5, 16./4, -25./9])
    g0 = transform.T @ diagonal @ transform
    g = .5*g0+.5*g0.T
    result = buckling.factor_buckling(left, right, g, tuple(range(5)), bounds=(0., 10.))
    np.testing.assert_allclose(result.multipliers, [2., 2., 5., 9.], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(result.multipliers, roots(f.T@f, -g), atol=1e-11, rtol=1e-11)
    assert max(result.original_factor_residuals) < 1e-11
    repeated = buckling.factor_buckling(left, right, g, tuple(range(5)), bounds=(0., 10.))
    assert canonical(result) == canonical(repeated)


def test_condensing_at_current_load_would_give_wrong_multipliers():
    material = np.array([[2., 1.], [1., 2.]])
    g = np.diag([-.5, -1.])
    result = buckling.factor_buckling(np.eye(2), np.linalg.cholesky(material).T,
        g, (0, 1), bounds=(0., 10.))
    np.testing.assert_allclose(result.multipliers, [3.-np.sqrt(3.), 3.+np.sqrt(3.)], atol=1e-11, rtol=1e-11)
    # Current-load (lambda=1) static lift, incorrectly held fixed in lambda.
    wrong_lift = np.array([1., -1.])
    wrong = float(wrong_lift@material@wrong_lift / (-wrong_lift@g@wrong_lift))
    assert abs(wrong/result.multipliers[0]-1.) > .02


def test_rounding_asymmetry_remains_visible_in_original_witness():
    g = np.diag([-1., -.5])
    g[0, 1] = 2e-13
    result = buckling.factor_buckling(np.eye(2), np.eye(2), g, (0, 1), bounds=(0., 10.))
    assert 0 < result.original_symmetry_error < 1e-11
    assert 0 < result.projected_symmetry_error < 1e-11
    assert max(result.original_factor_residuals) < 1e-11


@pytest.mark.parametrize('kind', ('singular', 'asymmetric', 'nonfinite', 'duplicate', 'bounds', 'mutation'))
def test_bad_pencils_and_changed_inputs_fail_closed(kind):
    left, right, g, free, bounds = np.eye(2), np.eye(2), -np.eye(2), (0, 1), (0., 10.)
    if kind == 'singular': right[:] = [[1., 2.], [2., 4.]]
    elif kind == 'asymmetric': g[0, 1] = .1
    elif kind == 'nonfinite': right[0, 0] = np.nan
    elif kind == 'duplicate': free = (0, 0)
    elif kind == 'bounds': bounds = (True, 2.)
    calls = []
    def check():
        calls.append(1)
        if kind == 'mutation' and len(calls) == 2:
            right[0, 0] = 2.
    with pytest.raises(ValueError):
        buckling.factor_buckling(left, right, g, free, bounds=bounds, check=check)


def make(family, curved=True, common=None, mass_scale=1.):
    source, _ = (generalized_problem if family == 'generalized' else fibre_problem)(curved, False)
    if common is not None:
        from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
        old = source.mesh.elements[1]
        reference = Reference(old.operator.reference.coordinates@common.T,
                              common@old.operator.reference.nodal_triads)
        element = type(old)(1, (1, 2, 3), reference, old.section, order=4)
        raw = NativeBeamDefinition.capture(element, mass_scale*np.diag([2., 2., 2., .07, .09, .11]))
        return NativeBeamAnalysis((raw,), tuple(source.boundary_conditions))
    return analysis(source, {1:mass_scale*np.diag([2., 2., 2., .07, .09, .11])})


def run(made, family, common=None):
    force = np.array([-.01, 0., 0.])
    if common is not None: force = common@force
    if family == 'generalized':
        return made.solve_distributed(DistributedPattern(LinePattern(((1, *map(float, force)),)), ()), steps=1)
    return made.solve_nodal(((3, *map(float, force)),), steps=1)


@pytest.fixture(scope='module', params=[(f, c) for f in ('generalized', 'fibre') for c in (False, True)])
def accepted(request):
    family, curved = request.param
    made = make(family, curved)
    result = run(made, family)
    assert result.status == 'completed', result.backend_result.info
    return family, curved, result


def test_actual_native_full_stationary_pencil_and_history(accepted, tmp_path):
    family, curved, state = accepted
    made = make(family, curved)
    before = canonical(made.recover(state.checkpoint, expected_sha256=state.checkpoint_sha256))
    result = made.buckling_modes(state.checkpoint, expected_sha256=state.checkpoint_sha256,
        bounds=(0., 1e8), num_modes=6)
    p, spectrum = result.pencil, result.spectrum
    assert spectrum.multipliers.size >= 2
    free = list(p.base.free_dofs)
    f = p.left@p.right[:, free]
    g = p.geometric[np.ix_(free, free)]
    independent = roots(f.T@f, -g)
    np.testing.assert_allclose(spectrum.multipliers, independent[:len(spectrum.multipliers)], rtol=1e-11, atol=1e-11)
    # Reconstruct the ORIGINAL stress-resultant saddle pencil, with stress
    # variables retained. Eliminating nodal/cell rotations at lambda=1 would
    # not reproduce these finite roots.
    chain = made._decode(state.checkpoint, state.checkpoint_sha256)
    response = chain[-1]['states'][1]['response']
    h = response.conservative_hessian if family == 'generalized' else response.full_hessian
    np.testing.assert_allclose(p.geometric, h[:24, :24], atol=1e-11, rtol=1e-11)
    b, c = h[24:, :24][:, free], -h[24:, 24:]
    full_material = np.block([[np.zeros_like(g), b.T], [b, -c]])
    full_stress = np.zeros_like(full_material)
    full_stress[:len(free), :len(free)] = -g
    saddle_roots = roots(full_material, full_stress)
    np.testing.assert_allclose(spectrum.multipliers, saddle_roots[:len(spectrum.multipliers)], rtol=1e-11, atol=1e-11)
    assert set(p.base.algebraic_dofs)&set(spectrum.free_dofs)
    assert spectrum.modes.shape[0] == 24
    assert result.definition_graph_sha256 == made.identity and result.checkpoint_sha256 == state.checkpoint_sha256
    assert canonical(made.recover(state.checkpoint, expected_sha256=state.checkpoint_sha256)) == before
    assert not result.production_qualified and not p.buckling_factor_authorized
    repeated = make(family, curved).buckling_modes(state.checkpoint, expected_sha256=state.checkpoint_sha256,
        bounds=(0., 1e8), num_modes=6)
    assert canonical(result) == canonical(repeated)
    (tmp_path/'force-checkpoint.json').write_bytes(state.checkpoint)
    (tmp_path/'buckling.json').write_bytes(canonical(dict(result=result,
        independent_condensed_roots=independent, independent_saddle_roots=saddle_roots)))


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_rigid_rotation_covariance_and_inertia_independence(family):
    first = run(make(family), family)
    result = make(family).buckling_modes(first.checkpoint, expected_sha256=first.checkpoint_sha256,
        bounds=(0., 1e8), num_modes=2)
    common = rotation([2.6, .8, -.3])
    changed = make(family, common=common)
    accepted = run(changed, family, common)
    assert accepted.status == 'completed'
    rotated = changed.buckling_modes(accepted.checkpoint, expected_sha256=accepted.checkpoint_sha256,
        bounds=(0., 1e8), num_modes=2)
    np.testing.assert_allclose(rotated.spectrum.multipliers, result.spectrum.multipliers, rtol=1e-11, atol=1e-11)
    heavy = make(family, mass_scale=3.)
    heavy_state = run(heavy, family)
    heavy_result = heavy.buckling_modes(heavy_state.checkpoint, expected_sha256=heavy_state.checkpoint_sha256,
        bounds=(0., 1e8), num_modes=2)
    np.testing.assert_array_equal(heavy_result.spectrum.multipliers, result.spectrum.multipliers)
    np.testing.assert_array_equal(heavy_result.spectrum.modes, result.spectrum.modes)


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_authority_and_cancellation_before_decode(family, monkeypatch):
    made = make(family)
    def forbidden(*a, **k): raise AssertionError('entered native mechanics before controls')
    monkeypatch.setattr(made, '_backend', forbidden)
    token = CancellationToken()
    token.cancel('stop buckling capture')
    with pytest.raises(SolveCancelled):
        made.buckling_modes(b'bad', expected_sha256='0'*64, bounds=(0., 1e8), cancellation_token=token)
    with pytest.raises(ValueError):
        made.buckling_modes(b'bad', expected_sha256='0'*64, bounds=(True, 10.))
    assert made._lock.acquire(blocking=False)
    made._lock.release()


def test_nonconservative_couple_not_admitted_as_buckling():
    from test_ge_beam3_native_generalized import pattern
    made = make('generalized')
    state = made.solve_distributed(pattern(), steps=1)
    assert state.status == 'completed'
    with pytest.raises(ValueError, match='nonconservative'):
        made.buckling_modes(state.checkpoint, expected_sha256=state.checkpoint_sha256, bounds=(0., 1e8))


def test_active_plastic_history_and_bad_authority_not_relabelled():
    made = analysis(fibre_problem(True, True)[0])
    state = made.solve_nodal(((3, .35, -.012, .006),), steps=2)
    assert state.status == 'completed'
    with pytest.raises(ValueError, match='authority'):
        made.buckling_modes(state.checkpoint, expected_sha256='0'*64, bounds=(0., 1e8))
    with pytest.raises(ValueError, match='history|yield|elastic'):
        made.buckling_modes(state.checkpoint, expected_sha256=state.checkpoint_sha256, bounds=(0., 1e8))


def test_connected_curved_fibres_retain_all_stationary_coordinates(tmp_path):
    from test_ge_beam3_retained_fibre import make_model
    from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
    source = make_model()
    definitions = tuple(NativeBeamDefinition.capture(NativeFibreStaticElement(
        i, tuple(e.node_ids), e.operator.reference, e.section, order=4),
        np.diag([2., 2., 2., .07, .09, .11])) for i, e in sorted(source.mesh.elements.items()))
    made = NativeBeamAnalysis(definitions, tuple(source.boundary_conditions))
    state = made.solve_nodal(((5, -.004, 0., 0.),), steps=1)
    assert state.status == 'completed'
    result = made.buckling_modes(state.checkpoint, expected_sha256=state.checkpoint_sha256,
        bounds=(0., 1e8), num_modes=4)
    p = result.pencil
    free = list(p.base.free_dofs)
    f = p.left@p.right[:, free]
    independent = roots(f.T@f, -p.geometric[np.ix_(free, free)])
    assert result.spectrum.modes.shape == (42, 4)
    assert len(set(p.base.algebraic_dofs)&set(free)) == 12
    np.testing.assert_allclose(result.spectrum.multipliers, independent[:4], rtol=1e-11, atol=1e-11)
    (tmp_path/'force-checkpoint.json').write_bytes(state.checkpoint)
    (tmp_path/'connected-buckling.json').write_bytes(canonical(dict(result=result, independent_roots=independent)))
