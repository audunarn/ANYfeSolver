"""Native distributed work and durable state on experimental objective BL5."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_bl5_finite import _model
from test_stiffened_shell_ge_distributed_load import balance


def make_owner(request, basis='p0', objective=True):
    model = _model(basis=basis)
    if objective:
        model = replace(model, q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    owner = ss.create_analysis(definition=model)
    request.addfinalizer(owner.close)
    return owner


def actions(owner, sign=1.):
    return tuple(ss.GEBeamLineAction(beam.element_id, (0.,0.,100.*sign),
                                   (2.*sign,3.*sign,0.))
                 for beam, _, _ in owner._beam_parts)


@pytest.mark.parametrize('basis',['p0','continuous_p1_coarse'])
def test_distributed_bl5_signed_work_restart_and_rejection(request,basis,monkeypatch):
    a = make_owner(request,basis)
    load = ss.GEBeamDistributedLoad(a.preflight()['identity'],actions(a))
    linear = a.solve_linear(load)
    balance(a,linear)
    assert a.capabilities['beam_distributed_loads']
    assert a.capabilities['staged_beam_distributed_loads']
    # A fresh finite owner never inherits reference-linear accepted state.
    a = make_owner(request,basis)
    programme = ss.GEBeamDistributedProgramme(a.preflight()['identity'],
                                             (actions(a),actions(a,-1.)))
    token = CancellationToken()
    def observe(result):
        balance(a,result)
        assert result.constraint_max <= 1e-9
        assert result.free_equilibrium_ratio <= 1e-8
        assert result.beam_shell_line_work
        if result.load_position == 1.: token.cancel('accepted line programme prefix')
    targets=(.5,1.,1.25,1.75,2.)
    control=ss.AdaptiveIncrementControl(2,16)
    with pytest.raises(SolveCancelled):
        a.solve(programme,targets,cancellation_token=token,on_accepted=observe,
                increment_control=control)
    saved=a.checkpoint()
    old=make_owner(request,basis,False)
    with pytest.raises(ValueError):old.restore(saved)
    assert old._finite_driver is None
    original=FinitePointDriver._assemble
    def broken(self,*args,**kwargs):
        original(self,*args,**kwargs)
        raise RuntimeError('line distributed injected trial failure')
    with monkeypatch.context() as patch:
        patch.setattr(FinitePointDriver,'_assemble',broken)
        with pytest.raises(RuntimeError,match='injected'):
            a.solve(programme,targets,increment_control=control)
    assert a.checkpoint()==saved
    a.rebase_rotations(a.recover().generation)
    b=make_owner(request,basis)
    b.restore(a.checkpoint())
    direct=a.solve(programme,targets,increment_control=control,on_accepted=observe)
    restored=b.solve(programme,targets,increment_control=control)
    assert direct.load_programme_complete and restored.load_programme_complete
    np.testing.assert_array_equal(direct.beam_positions,restored.beam_positions)
    assert a.checkpoint()==b.checkpoint()
    final=make_owner(request,basis)
    fixed=ss.GEBeamDistributedLoad(final.preflight()['identity'],actions(final,-1.))
    oracle=final.solve(fixed)
    balance(final,oracle)
    np.testing.assert_allclose(direct.beam_positions,oracle.beam_positions,rtol=0.,atol=1e-9)


def test_bl5_native_distributed_actual_update_derivative(request):
    a=make_owner(request)
    driver=FinitePointDriver(a)
    load=ss.GEBeamDistributedLoad(a.preflight()['identity'],actions(a))
    pattern,_=driver._prepare_load(load,(1.,))
    rng=np.random.default_rng(192)
    count=a._K.shape[0]
    change=rng.normal(size=count+a._C.shape[0])*.003
    state=driver._advance(driver._state,change,1.)
    force,jac,_=driver._external.evaluate(state)
    direction=rng.normal(size=len(change)); direction[count:]=0.
    h=1e-5
    plus=driver._external.evaluate(driver._advance(state,direction,h))[0]
    minus=driver._external.evaluate(driver._advance(state,direction,-h))[0]
    np.testing.assert_allclose((plus-minus)/(2*h),jac@direction[:count],rtol=1e-6,atol=1e-8)
    r1,k1,*_=driver._assemble(state,1.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    r0,k0,*_=driver._assemble(state,0.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    np.testing.assert_allclose(r1[:count]-r0[:count],-force,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose((k1-k0)[:count,:count].toarray(),-jac.toarray(),rtol=1e-6,atol=1e-6)


def test_historical_bl5_distributed_scope_remains_rejected(request):
    a=make_owner(request,objective=False)
    load=ss.GEBeamDistributedLoad(a.preflight()['identity'],actions(a))
    assert not a.capabilities['beam_distributed_loads']
    with pytest.raises(ValueError):a.solve_linear(load)
    with pytest.raises(ValueError):a.solve(load)
    assert a._finite_driver is None


def test_mixed_point_and_bl5_native_distributed_restart(request):
    model=replace(_model(),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
                  points=(ss.PointConnection(4,105),))
    a=ss.create_analysis(definition=model); request.addfinalizer(a.close)
    b=ss.create_analysis(definition=model); request.addfinalizer(b.close)
    load=ss.GEBeamDistributedLoad(a.preflight()['identity'],actions(a))
    first=a.solve(load,(.25,)); balance(a,first)
    assert first.point_wrenches and first.beam_shell_line_work
    a.rebase_rotations(first.generation)
    b.restore(a.checkpoint())
    direct=a.solve(load,(1.,)); restored=b.solve(load,(1.,))
    balance(a,direct)
    assert a.checkpoint()==b.checkpoint()
    np.testing.assert_array_equal(direct.beam_positions,restored.beam_positions)
