"""Actual mixed-owner pressure, external derivative and durable state."""
from dataclasses import replace
from copy import copy
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_ge_pressure_load import encode_checkpoint,decode_checkpoint,VECTOR_EQUILIBRIUM_POLICY
from test_stiffened_shell_ge_experimental import _panel


def definition(kind='q4'):
    d=_panel()
    if kind=='q4':return replace(d,q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    shell=replace(d.shell,q4_elements=() if kind=='s3' else ((1,2,5,4),),
        s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)) if kind=='s3' else ((2,3,6),(2,6,5)))
    return replace(d,shell=shell,**({} if kind=='s3' else {'q4_pullback_policy':ss.Q4_ALGORITHMIC_PULLBACK_V1}))


def owner(request,kind='q4'):
    a=ss.create_analysis(definition=definition(kind));request.addfinalizer(a.close);return a


def load(a,mode='current_surface',pressure=100.):
    return ss.GEShellPressureLoad(a._identity,tuple(ss.GEShellPressureAction(eid,pressure,mode)
        for eid,_,_,_ in a._shell_parts))


def balance(a,r):
    support=r.support_reactions if hasattr(r,'support_reactions') else r.reactions
    w=support[:a._shell_count].reshape(-1,6);points=np.asarray(a._shell_xyz)
    if hasattr(r,'beam_positions'):points=points+r.shell_displacement.reshape(-1,6)[:,:3]
    np.testing.assert_allclose(w[:,:3].sum(0),-np.asarray(r.pressure_load.applied_force),rtol=1e-8,atol=1e-6)
    np.testing.assert_allclose((np.cross(points,w[:,:3])+w[:,3:]).sum(0),
        -np.asarray(r.pressure_load.applied_moment_about_origin),rtol=1e-7,atol=2e-5)


@pytest.mark.parametrize('kind',['q4','mixed','s3'])
@pytest.mark.parametrize('mode',['reference_dead','current_surface'])
def test_reference_pressure_native_modes_restart_and_balance(request,kind,mode):
    a,b=owner(request,kind),owner(request,kind);q=load(a,mode)
    r=a.solve_linear(q);assert r.status=='completed';balance(a,r)
    np.testing.assert_allclose(r.pressure_load.applied_force,(0,0,200),atol=1e-10)
    restored=b.restore(a.checkpoint())
    np.testing.assert_array_equal(restored.beam_coordinates,r.beam_coordinates)
    assert b.checkpoint()==a.checkpoint()
    a.solve_linear(q);assert a._linear_factorizations==1
    descriptor,inner=decode_checkpoint(a.checkpoint());assert descriptor==q
    target=owner(request,kind);wrong=load(target,mode,101.).compile(target)
    with pytest.raises(ValueError):target.restore(encode_checkpoint(wrong,inner))
    with pytest.raises(ValueError):target.restore(inner)
    assert target._accepted is None and target._generation==0


@pytest.mark.parametrize('kind,mode',[('q4','reference_dead'),('q4','current_surface'),('mixed','current_surface'),('s3','current_surface')])
def test_finite_pressure_cancelled_prefix_chart_restart(request,kind,mode):
    a,b=owner(request,kind),owner(request,kind);q=load(a,mode);token=CancellationToken()
    if kind == 'mixed' and mode == 'current_surface':
        # Explicit approved vector policy for the mixed finite-pressure path;
        # preserve default checks in the other cases and refusal controls.
        q=replace(q,equilibrium_policy=VECTOR_EQUILIBRIUM_POLICY)
    control=ss.AdaptiveIncrementControl(2,16)
    def cancel(r):token.cancel('accepted pressure prefix')
    with pytest.raises(SolveCancelled):
        a.solve(q,(.1,.2),increment_control=control,on_accepted=cancel,cancellation_token=token)
    first=a.recover();assert first.accepted_factors==(.1,);balance(a,first)
    saved=a.checkpoint();b.restore(saved)
    with pytest.raises(ValueError):a.solve(load(a,mode,101.),(.1,.2),increment_control=control)
    assert a.checkpoint()==saved
    last=a.solve(q,(.1,.2),increment_control=control)
    resumed=b.solve(q,(.1,.2),increment_control=control)
    np.testing.assert_array_equal(last.beam_positions,resumed.beam_positions)
    assert a.checkpoint()==b.checkpoint();balance(a,last)
    if mode=='current_surface':assert last.pressure_load.potential is None
    # A separately charted restart exercises pressure's independence from directors.
    a.rebase_rotations(last.generation);c=owner(request,kind);c.restore(a.checkpoint())
    final=a.solve(q,(.3,),max_iterations=15);other=c.solve(q,(.3,),max_iterations=15)
    np.testing.assert_array_equal(final.beam_positions,other.beam_positions);balance(a,final)


@pytest.mark.parametrize('kind',['mixed','s3'])
def test_native_enhanced_dead_s3_not_silently_promoted_to_finite(request,kind):
    a=owner(request,kind);q=load(a,'reference_dead')
    context=q.compile(a)
    assert not context.finite_supported
    assert np.linalg.norm(context.reference_gradient[:a._shell_count].reshape(-1,6)[:,3:])>0
    prepared=copy(a);prepared._external_context=context
    # Restoration constructs the finite driver with the decoded external context.
    with pytest.raises(ValueError,match='S3'):FinitePointDriver(prepared)
    with pytest.raises(ValueError,match='S3'):a.solve(q,(.1,))
    assert a._finite_driver is None and a._generation==0
    assert a.solve_linear(q).status=='completed'


@pytest.mark.parametrize('kind',['q4','mixed','s3'])
def test_pressure_external_tangent_through_actual_owner_update(request,kind):
    a=owner(request,kind);driver=FinitePointDriver(a)
    pattern,_=driver._prepare_load(load(a),(.1,))
    q=np.array(driver._state.q);u=q[:a._shell_count].reshape(-1,6)
    u[:,2]=(.0,.03,.02,.01,-.02,.04)
    state=replace(driver._state,q=q)
    force,jac,_=driver._external.evaluate(state);n=len(force)
    assert np.linalg.norm((jac-jac.T).data)>1.
    r1,k1,*_=driver._assemble(state,1.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    r0,k0,*_=driver._assemble(state,0.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    np.testing.assert_allclose(r1[:n]-r0[:n],-force,atol=1e-6,rtol=1e-7)
    np.testing.assert_allclose((k1-k0)[:n,:n].toarray(),-jac.toarray(),atol=1e-6,rtol=1e-7)
    direction=np.zeros(len(r1));direction[:a._shell_count]=np.linspace(-.3,.4,a._shell_count)
    h=1e-5
    fp=driver._external.evaluate(driver._advance(state,direction,h))[0]
    fm=driver._external.evaluate(driver._advance(state,direction,-h))[0]
    np.testing.assert_allclose((fp-fm)/(2*h),jac@direction[:n],atol=1e-8,rtol=1e-7)


def test_follower_decrease_can_advance_newton_without_false_commit(request):
    from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
    a=owner(request)
    # The first full step decreases the parent's frozen merit, but its current
    # active-row ratio is still far above equilibrium. It must not be published.
    with pytest.raises(FiniteIncrementFailure,match='iteration limit'):
        a.solve(load(a),(.1,),max_iterations=1)
    assert a._generation==0 and a._accepted is None and a._finite_driver is None

def test_pressure_trial_work_norm_units_extremes_and_zero_load():
    from anyfesolver._stiffened_shell_ge_finite_experimental import _trial_work_ratio
    expected=np.sqrt(52.)/10.
    assert _trial_work_ratio(np.array([3.,4.]),np.array([0.,10.]),np.array([2.,1.]))==pytest.approx(expected)
    assert _trial_work_ratio(np.array([3.,4000.]),np.array([0.,10000.]),np.array([2000.,1.]))==pytest.approx(expected)
    for magnitude in (1e-200,1e200):
        assert _trial_work_ratio(magnitude*np.array([3.,4.]),magnitude*np.array([0.,10.]),np.array([2.,1.]))==pytest.approx(expected)
    assert _trial_work_ratio(np.zeros(2),np.zeros(2),np.ones(2))==0.
    assert np.isinf(_trial_work_ratio(np.ones(2),np.zeros(2),np.ones(2)))
    for scales in ([1.,0.],[1.,np.nan],[1.,-1.]):
        with pytest.raises(ValueError):_trial_work_ratio(np.ones(2),np.ones(2),np.array(scales))


def test_pressure_trial_merit_does_not_replace_component_acceptance():
    from anyfesolver._stiffened_shell_ge_finite_experimental import _trial_work_ratio
    from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium
    applied=np.array([10.,1e-10]);parent=.5*applied;candidate=np.array([1e-12,1e-10])
    assert _trial_work_ratio(candidate,applied,np.ones(2))<_trial_work_ratio(parent,applied,np.ones(2))
    assert _trial_work_ratio(candidate,applied,np.ones(2))<1e-8
    assert _linear_equilibrium(candidate,applied,np.ones(2))==1.


def test_equilibrated_pressure_trial_is_rechecked_despite_frozen_merit(request, monkeypatch):
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    original_assemble = finite.FinitePointDriver._assemble
    latest = {'equilibrated': False, 'obstructed': 0, 'certificates': 0,
              'assembling': False}
    def assemble(self, *args, **kwargs):
        latest['assembling'] = True
        try:
            result = original_assemble(self, *args, **kwargs)
        finally:
            latest['assembling'] = False
        latest['equilibrated'] = result[2] <= 1e-8 and result[3] <= 1e-9
        latest['certificates'] += int(latest['equilibrated'])
        return result
    def obstruct_comparison(original):
        def ratio(*args):
            value = original(*args)
            if latest['equilibrated'] and not latest['assembling']:
                latest['obstructed'] += 1
                return float('inf')
            return value
        return ratio
    monkeypatch.setattr(finite.FinitePointDriver, '_assemble', assemble)
    for name in ('_trial_work_ratio', '_linear_equilibrium'):
        monkeypatch.setattr(finite, name, obstruct_comparison(getattr(finite, name)))
    a = owner(request)
    result = a.solve(load(a), (.1,))
    assert latest['obstructed'] > 0
    assert latest['certificates'] >= 2
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    balance(a, result)
