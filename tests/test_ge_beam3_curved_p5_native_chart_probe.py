"""Native-chart compatibility checks, not independent beam qualification."""

import numpy as np
import pytest

from anyfesolver._native_rotation_state import NativeRotationStateStore, NativeRotationStateError
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases import ge_beam3_curved_p5_native_chart_probe as probe
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation
from docs.reference_cases.ge_beam3_curved_p5_implicit_dynamics_probe import left_jacobian
from docs.reference_cases.ge_beam3_curved_p5_section_probe import SectionHistory
from test_ge_beam3_curved_p5_algebra_probe import reference, assert_scaled_close
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law, mask


def store_for(ref, displacement=None, matrices=None):
    return NativeRotationStateStore(
        (0, 1, 2), rotational_dofs={i: tuple(range(6*i+3, 6*i+6)) for i in range(3)},
        coordinate_rows={i: i for i in range(3)},
        committed_full_displacement=np.zeros(18) if displacement is None else displacement,
        committed_full_coordinates=ref.coordinates, committed_rotation_matrices=matrices)


def sample():
    return np.array([[.01, -.02, .03, .04, -.02, .06],
                     [.04, .03, -.02, -.03, .05, .02],
                     [-.01, .04, .02, .02, .03, -.04]]).ravel()


def origins():
    return tuple(SectionHistory(.0001*i, .0002*i) for i in range(16))


def evaluate(store, token, ref, yield_force=1000., fixed=None):
    return probe.evaluate_native_trial(store, token, element_id=7, node_ids=(0, 1, 2),
        reference=ref, section=law(yield_force), origins=origins() if fixed is None else fixed)


def at_chart(store, ref, displacement, yield_force=1000.):
    coordinates=ref.coordinates+displacement.reshape(3, 6)[:, :3]
    with store.candidate(displacement, coordinates) as trial:
        return evaluate(store, trial.token, ref, yield_force)


@pytest.mark.parametrize('vector', [np.zeros(3), np.array([1e-9, -2e-9, 3e-9]), np.array([.5, -.3, .4])])
def test_analytic_exp_chart_and_derivative(vector):
    a, da=probe.exp_chart_terms(vector)
    assert_scaled_close(a, left_jacobian(vector))
    for k in range(3):
        step=np.eye(3)[k]*1e-7
        difference=(probe.exp_chart_terms(vector+step)[0]-probe.exp_chart_terms(vector-step)[0])/2e-7
        assert_scaled_close(da[:, :, k], difference, 1e-7)


@pytest.mark.parametrize('yield_force', [1000., .02])
def test_native_chart_energy_force_tangent_and_fixed_history(yield_force):
    ref=reference(.4, .05, .1);store=store_for(ref);u=sample()
    base=at_chart(store, ref, u, yield_force)
    direction=np.sin(np.arange(18)+.2)/8;step=1e-6
    plus=at_chart(store, ref, u+step*direction, yield_force)
    minus=at_chart(store, ref, u-step*direction, yield_force)
    assert mask(base.spatial)==mask(plus.spatial)==mask(minus.spatial)
    assert any(mask(base.spatial)) == (yield_force < 1.)
    assert abs((plus.spatial.potential-minus.spatial.potential)/(2*step)-base.force@direction)<=1e-7
    assert_scaled_close((plus.force-minus.force)/(2*step), base.tangent@direction, 1e-7)
    assert_scaled_close(base.tangent, base.tangent.T)
    assert abs(base.force@direction-base.spatial.residual@(base.chart@direction))<=1e-11
    # Both a raw spatial Hessian and just congruence miss the chart connection.
    assert np.linalg.norm(base.tangent-base.spatial.tangent)>1e-5
    assert np.linalg.norm(base.tangent-base.chart.T@base.spatial.tangent@base.chart)>1e-6
    assert base.origins==origins()
    assert [s.response.origin for s in base.spatial.stations]==list(origins())
    assert store.generation==0 and not store.has_active_trial
    assert np.array_equal(store.committed_full_displacement, np.zeros(18))
    for value in (base.force, base.tangent, base.positions, base.nodal_operators):
        assert not value.flags.writeable


def test_noncommuting_committed_rotation_is_not_accumulated_vector():
    ref=reference();store=store_for(ref);first=sample();second=first.copy()
    second.reshape(3, 6)[:, 3:]+=np.array([.05, .03, -.02])
    token=store.begin_trial(first, ref.coordinates)
    store.commit_trial(token, first, ref.coordinates)
    made=at_chart(store, ref, second)
    for i in range(3):
        a=first.reshape(3, 6)[i, 3:];b=second.reshape(3, 6)[i, 3:]-a
        assert_scaled_close(made.nodal_operators[i], rotation(b)@rotation(a))
        assert np.linalg.norm(made.nodal_operators[i]-rotation(a+b))>1e-5
    assert store.generation==1 and not store.has_active_trial


def test_bookkeeping_coordinate_offset_does_not_change_authoritative_pose():
    ref=reference();u=sample();offset=np.zeros((3, 6));offset[:, 3:]=[4., -2., 1.]
    matrices=np.tile(rotation([.1, -.07, .03]), (3, 1, 1))
    first=store_for(ref, matrices=matrices)
    second=store_for(ref, displacement=offset.ravel(), matrices=matrices)
    a=at_chart(first, ref, u)
    b=at_chart(second, ref, u+offset.ravel())
    assert_scaled_close(a.force, b.force)
    assert_scaled_close(a.tangent, b.tangent)
    assert_scaled_close(a.nodal_operators, b.nodal_operators)


@pytest.mark.parametrize('kind', ['discarded', 'foreign', 'old_serial'])
def test_invalid_token_rejected_before_mechanics(kind, monkeypatch):
    ref=reference();store=store_for(ref);token=store.begin_trial(sample(), ref.coordinates)
    if kind=='discarded': store.discard_trial(token)
    elif kind=='foreign': store=store_for(ref)
    else:
        store.discard_trial(token);store.begin_trial(sample(), ref.coordinates)
    def forbidden(*args, **kwargs): raise AssertionError('mechanics must not run')
    monkeypatch.setattr(probe.CompensatedMixedBeamProbe, 'solve', forbidden)
    with pytest.raises(NativeRotationStateError): evaluate(store, token, ref)


def test_token_revalidated_after_mechanics(monkeypatch):
    ref=reference();store=store_for(ref);token=store.begin_trial(sample(), ref.coordinates)
    original=probe.CompensatedMixedBeamProbe.solve
    def invalidate(*args, **kwargs):
        result=original(*args, **kwargs);store.discard_trial(token);return result
    monkeypatch.setattr(probe.CompensatedMixedBeamProbe, 'solve', invalidate)
    with pytest.raises(NativeRotationStateError): evaluate(store, token, ref)
    assert store.generation==0 and not store.has_active_trial


@pytest.mark.parametrize('bad_origins', [None, (), (SectionHistory(),)])
def test_missing_history_is_not_fabricated(bad_origins):
    ref=reference();store=store_for(ref)
    with store.candidate(sample(), ref.coordinates) as trial:
        with pytest.raises(ValueError):
            probe.evaluate_native_trial(store, trial.token, element_id=7, node_ids=(0, 1, 2),
                reference=ref, section=law(), origins=bad_origins)


@pytest.mark.parametrize('bad', [[np.nan, 0., 0.], [.9*np.pi, 0., 0.], [1., 2.]])
def test_invalid_increment_rejected(bad):
    with pytest.raises(ValueError): probe.exp_chart_terms(bad)


def test_nonsymmetric_hessian_rejected():
    h=np.eye(18);h[0, 1]=.1
    with pytest.raises(ValueError, match='symmetric'): probe.pullback(np.zeros(18), h, np.zeros((3, 3)))


def test_zero_chart_returns_original_symmetric_energy_hessian():
    force=np.sin(np.arange(18));factor=np.cos(np.arange(324)).reshape(18, 18)
    h=factor.T@factor
    g, tangent, a=probe.pullback(force, h, np.zeros((3, 3)))
    assert_scaled_close(g, force)
    assert_scaled_close(tangent, h)
    assert np.array_equal(a, np.eye(18))


def test_shared_native_node_has_one_operator_and_element_owned_material_frames():
    first=reference(.3);other=reference(.3, twist=.15)
    second=CurvedBeam3ReferenceGeometry(other.coordinates+[2., 0., 0.], other.nodal_triads)
    refs=(first, second);maps=((0, 1, 2), (2, 3, 4))
    coordinates=np.vstack((first.coordinates, second.coordinates[1:]))
    store=NativeRotationStateStore(tuple(range(5)),
        rotational_dofs={i: tuple(range(6*i+3, 6*i+6)) for i in range(5)},
        coordinate_rows={i: i for i in range(5)},
        committed_full_displacement=np.zeros(30), committed_full_coordinates=coordinates)
    def assembled(displacement):
        results=[];force=np.zeros(30);tangent=np.zeros((30, 30))
        with store.candidate(displacement, coordinates+displacement.reshape(5, 6)[:, :3]) as trial:
            for i, (ref, nodes) in enumerate(zip(refs, maps)):
                result=probe.evaluate_native_trial(store, trial.token, element_id=i,
                    node_ids=nodes, reference=ref, section=law(1000.), origins=origins())
                ids=np.array([6*n+j for n in nodes for j in range(6)])
                force[ids]+=result.force;tangent[np.ix_(ids, ids)]+=result.tangent
                results.append(result)
        return results, force, tangent
    u=np.sin(np.arange(30)+.7)*.015
    base, force, tangent=assembled(u)
    assert np.array_equal(base[0].nodal_operators[2], base[1].nodal_operators[0])
    assert not np.allclose(first.nodal_triads[2], second.nodal_triads[0])
    direction=np.zeros(30);direction[12:18]=[.1, -.2, .3, -.3, .2, .1]
    step=1e-6
    plus, pf, _=assembled(u+step*direction);minus, mf, _=assembled(u-step*direction)
    energy_difference=sum(p.spatial.potential-m.spatial.potential for p, m in zip(plus, minus))/(2*step)
    assert abs(energy_difference-force@direction)<=1e-7
    assert_scaled_close((pf-mf)/(2*step), tangent@direction, 1e-7)
    assert store.generation==0 and not store.has_active_trial


def test_local_failure_preserves_store_and_origins(monkeypatch):
    ref=reference();store=store_for(ref);fixed=origins()
    def failed(*args, **kwargs): raise RuntimeError('injected local failure')
    monkeypatch.setattr(probe.CompensatedMixedBeamProbe, 'solve', failed)
    with pytest.raises(RuntimeError, match='injected'):
        with store.candidate(sample(), ref.coordinates) as trial:
            evaluate(store, trial.token, ref, fixed=fixed)
    assert fixed==origins()
    assert store.generation==0 and not store.has_active_trial
    assert np.array_equal(store.committed_full_displacement, np.zeros(18))
