"""Current projected-map merit derivatives and direction-owned event safety."""

import numpy as np
import pytest

from anyfesolver.general_contact.semismooth_path import (
    _current_merit_slope, _original_direction_event_fraction,
)


def slope(delta, *, gap=(1.,), z=(2.,), support=(1.,), residual=(0.,),
          constraint=(), ncp=(1.,), k=-2., gradient=(1.,)):
    return _current_merit_slope(
        delta=np.asarray(delta), q_size=1, mu_size=len(constraint),
        kqq=np.array([[k]]), kqz=-np.asarray(gradient)[None, :],
        gap_q=np.asarray(gradient)[:, None],
        control_reactions=np.zeros((1, len(constraint))),
        control_rows=np.ones((len(constraint), 1)),
        residual=np.asarray(residual), constraint=np.asarray(constraint),
        ncp=np.asarray(ncp), z=np.asarray(z), gap=np.asarray(gap),
        support=np.asarray(support), c=np.ones(len(gap)), scale=1.,
        control_scales=np.ones(len(constraint)))


def test_predicted_branch_equations_do_not_guarantee_current_merit_descent():
    # Frozen indefinite nonlinear tangent; both directions exactly solve
    # their respective branch equations and structural residual derivative.
    # This is an algebra discriminator, not a claimed physical fixture.
    predicted = np.array([1., -2.])
    current = np.array([-1., 2.])
    assert -2.*predicted[0]-predicted[1] == 0.
    assert 2.+predicted[1] == 0.  # predicted inactive equation
    assert -2.*current[0]-current[1] == 0.
    assert 1.+current[0] == 0.  # current active equation
    assert slope(predicted) == 1.
    assert slope(current) == -1.


@pytest.mark.parametrize('dz,expected', [(1., 0.), (-1., -1.)])
def test_projected_map_tie_uses_one_sided_derivative(dz, expected):
    assert slope([0., dz], gap=(1.,), z=(1.,), ncp=(1.,),
                 k=0., residual=(0.,), gradient=(0.,)) == expected


def test_zero_ncp_row_contributes_absolute_direction():
    assert slope([-1., 0., 2.], gap=(1., 3.), z=(2., 0.),
                 support=(1., 1.), ncp=(1., 0.),
                 k=0., gradient=(1., 0.)) == 1.


def test_max_merit_tie_uses_largest_component_slope():
    # Equilibrium and NCP both equal1; their slopes -2 and +1.
    assert slope([1., -2.], residual=(1.,), k=0.) == 1.


def test_zero_equilibrium_norm_uses_direction_norm():
    assert slope([1., 0.], gap=(1.,), z=(0.,), ncp=(0.,),
                 k=2., gradient=(0.,)) == 2.


@pytest.mark.parametrize('constraints', [(1., -1.), (0., 0.)])
def test_control_linf_multiple_maxima_and_zero_rows(constraints):
    assert slope([1., 0., 0., 0.], gap=(1.,), z=(0.,), ncp=(0.,),
                 k=0., gradient=(0.,), constraint=constraints) == 1.


def test_unsupported_mode_uses_nodal_force_derivative():
    assert slope([0., 1.], gap=(-1.,), z=(2.,), support=(0.,),
                 ncp=(2.,), k=0., gradient=(0.,)) == 1.


def test_nonfinite_direction_does_not_claim_descent():
    assert np.isinf(slope([np.nan, 0.]))


@pytest.mark.parametrize('dq,dz,g,z,k,expected', [
    (1., -2., 1., 2., -2., 1.),
    (-1., 2., 1., 2., -2., -1.),
    (0., 1., 1., 1., 0., 0.),
    (0., -1., 1., 1., 0., -1.),
])
def test_independent_explicit_one_sided_merit_difference(dq, dz, g, z, k, expected):
    # Explicit scalar force balance and projected residual; no tested helper
    # is used to construct the independent finite-difference merit.
    def merit(t):
        force_residual = -k*t*dq + t*dz
        normal_gap = g+t*dq
        force = z+t*dz
        projected_residual = force-max(0., force-normal_gap)
        return max(abs(force_residual), abs(projected_residual))
    for h in (1e-4, 1e-6):
        assert (merit(h)-merit(0.))/h == pytest.approx(expected, abs=1e-8)
    base_ncp = z-max(0., z-g)
    assert slope([dq, dz], gap=(g,), z=(z,), ncp=(base_ncp,),
                 k=k) == expected


def test_event_uses_original_certificate_not_larger_fallback_trial():
    # Original admissible0.01; fallback last0.2 must not admit original0.1.
    assert _original_direction_event_fraction(.2, .01, .005) is None
    assert _original_direction_event_fraction(.002, .01, .005) == .001
    assert _original_direction_event_fraction(.2, .01, .02) == .01
    assert _original_direction_event_fraction(.2, 0., .2) is None
    assert _original_direction_event_fraction(.2, .01, None) is None


def _caller(monkeypatch, *, gaps, forces, gradient, directions, certificates):
    """Controlled linear responses exercise production selection/cleanup only."""
    from types import SimpleNamespace
    from scipy import sparse
    import anyfesolver.general_contact.semismooth_path as owner
    modes = len(gaps)
    origin = {7: {'origin': 11}}
    seen_origins, proposals, candidate_poses, discards = [], [], [], []
    def assemble(model, u, states, layers, **kw):
        assert states is origin
        seen_origins.append(states)
        return np.zeros(1), sparse.csr_matrix((1, 1)), {7: {'trial': 1}}
    state = SimpleNamespace(normalized_gap=np.asarray(gaps),
                            reference_support=np.ones(modes),
                            minimum_physical_gap=0., overlap_signature=())
    contact = SimpleNamespace(state=state, force=np.zeros(1),
        tangent=sparse.csr_matrix((1, 1)), gap_gradient=np.asarray(gradient)[:, None],
        force_gradient=np.zeros((modes, 1)), force_pressure_gradient=None)
    def linearize(manager, u, z, *, tangent):
        candidate_poses.append(u.copy())
        if not tangent:
            raise ValueError('controlled trial refusal')
        return contact
    answers = iter(directions)
    monkeypatch.setattr(owner, 'factorize', lambda *a, **k:
        SimpleNamespace(solve=lambda rhs: np.asarray(next(answers), dtype=float)))
    monkeypatch.setattr(owner, '_assemble_nonlinear_system', assemble)
    monkeypatch.setattr(owner, 'linearize', linearize)
    monkeypatch.setattr(owner, 'reference_nodal_support', lambda law: np.ones(modes))
    monkeypatch.setattr(owner, '_discard_nonlinear_state_candidate',
                        lambda states: discards.append(states is origin))
    fractions = iter(certificates)
    def fraction(start, end):
        proposals.append((start.copy(), end.copy()))
        return next(fractions)
    manager = SimpleNamespace(law=None, stiffness=1., gap_tolerance=50e-6,
        admissible_step_fraction=fraction, point_gap_lower=lambda u: 0.,
        _positions=lambda u: (np.zeros((1, 3)), np.zeros((1, 3))))
    result = owner.solve_increment(model=None, manager=manager,
        settings=SimpleNamespace(complementarity_force_scale_n=1.,
            monolithic_linear_solver='direct', monolithic_maximum_pivots=8),
        q=np.zeros(1), reactions=np.zeros(0), nodal_forces=np.asarray(forces),
        committed_states=origin, transform=sparse.eye(1, format='csr'),
        offset=np.zeros(1), control_rows=np.zeros((0, 1)),
        control_reactions=np.zeros((1, 0)), target=np.zeros(0),
        reduced_external=np.zeros(1), num_layers=1, max_iterations=1,
        tolerance=1e-6, control_scales=np.zeros(0),
        path_force_magnitude=1., cancellation_token=None, step=2)
    assert seen_origins and all(s is origin for s in seen_origins)
    assert discards and all(discards)
    assert origin == {7: {'origin': 11}}
    return result, proposals, candidate_poses


def test_production_caller_selects_existing_descent_branch_before_geometry(monkeypatch):
    result, proposals, _ = _caller(monkeypatch, gaps=[1., 3.], forces=[2., 0.],
        gradient=[1., 4.], directions=[[-1., -3., 0.], [1., -2., 8.]],
        certificates=[0.])
    assert not result.completed
    assert len(proposals) == 1 and proposals[0][1].tolist() == [-1.]
    selected = [r for r in result.diagnostics['pivot_trace']
                if r.get('fallback') == 'current_branch_before_line_search']
    assert len(selected) == 1
    assert selected[0]['predicted_merit_slope'] == 9.
    assert selected[0]['current_merit_slope'] == -1.


def test_production_event_cannot_borrow_fallback_certificate(monkeypatch):
    result, proposals, poses = _caller(monkeypatch, gaps=[.9998], forces=[1.],
        gradient=[1.], directions=[[2., 0.], [1., 0.]],
        certificates=[1e-5, .8])
    assert not result.completed and len(proposals) == 2
    assert proposals[0][1].tolist() == [1.]
    assert proposals[1][1].tolist() == [2.]
    assert len(result.diagnostics['attempts']) == 24
    assert all(a.get('kind') != 'before_active_event'
               for a in result.diagnostics['attempts'])
    assert len(poses) == 25  # base plus exactly the two existing searches


def test_production_current_direction_is_not_replaced_without_alternative(monkeypatch):
    result, proposals, _ = _caller(monkeypatch, gaps=[1.], forces=[2.],
        gradient=[1.], directions=[[-1., 2.]], certificates=[0.])
    assert not result.completed and len(proposals) == 1
    assert proposals[0][1].tolist() == [-1.]
    assert not any(r.get('fallback') == 'current_branch_before_line_search'
                   for r in result.diagnostics['pivot_trace'])
