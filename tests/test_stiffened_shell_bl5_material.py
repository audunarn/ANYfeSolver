"""Distinct generalized material history on experimental objective Q4 BL5."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_bl5_finite import _model, _action


def model(basis='p0', selected=True, hardening=1e6):
    base=_model(basis=basis);beams=[]
    for item in base.beams:
        element,inertia=item.instantiate()
        section=b3_ge.EllipsoidalGeneralizedSection(element.section.elastic,
                                                     element.section.metric,5.,hardening)
        beams.append(b3_ge.define_beam('b3-ge',element.element_id,element.node_ids,
                     element.operator.reference.coordinates,element.operator.reference.nodal_triads,
                     section,inertia,quadrature=element.operator.order))
    return replace(base,beams=tuple(beams),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
                   bl5_material_policy=ss.BL5_GENERALIZED_MATERIAL_V1 if selected else 'elastic')


def make(request,definition):
    a=ss.create_analysis(definition=definition);request.addfinalizer(a.close);return a


def test_selected_bl5_pre_yield_snapshot_and_legacy_isolation(request):
    definition=model();a=make(request,definition);b=make(request,definition)
    old=make(request,model(selected=False))
    assert a.preflight()['identity']!=old.preflight()['identity']
    assert 'generalized B3 plastic finite BL5' not in a.capabilities['excluded']
    assert 'generalized B3 plastic finite BL5' in old.capabilities['excluded']
    result=a.solve(_action(),(.001,))
    assert not np.any(result.beam_station_accumulated)
    b.restore(a.checkpoint());assert b.checkpoint()==a.checkpoint()
    with pytest.raises(ValueError):old.restore(a.checkpoint())
    assert old._finite_driver is None
    with pytest.raises(ValueError,match='elastic beam response'):old.solve(_action())
    for bad in (replace(definition,lines=()),replace(definition,bl5_material_policy='unknown')):
        with pytest.raises(ValueError):ss.create_analysis(definition=bad)


@pytest.mark.parametrize('basis',['p0','continuous_p1_coarse'])
def test_bl5_generalized_yield_reject_rebase_reverse_and_restart(request,monkeypatch,basis):
    definition=model(basis,hardening=1e8);a=make(request,definition)
    programme=ss.StagedDeadLoadProgramme(a.preflight()['identity'],(_action(),_action(-100.)))
    targets=(.5,1.,1.25,1.75,2.)
    control=ss.AdaptiveIncrementControl(4,64)
    token=CancellationToken()
    def observe(r):
        assert r.free_equilibrium_ratio<=1e-8 and r.constraint_max<=1e-9
        assert r.beam_shell_line_work
        count=a._shell_count+6*len(a._beam_ids)
        net=(r.support_reactions[:count]+r.applied_dead_load[:count]).reshape(-1,6)
        xyz=np.vstack((np.asarray(definition.shell.coordinates_m)+r.shell_displacement.reshape(-1,6)[:,:3],r.beam_positions))
        assert np.linalg.norm(net[:,:3].sum(0))<=1e-6
        assert np.linalg.norm((np.cross(xyz,net[:,:3])+net[:,3:]).sum(0))<=2e-6
        if r.load_position==1.:token.cancel('yielded BL5 accepted prefix')
    with pytest.raises(SolveCancelled):a.solve(programme,targets,on_accepted=observe,cancellation_token=token,max_iterations=20,max_backtracks=12,increment_control=control)
    prefix=a.recover();assert np.max(prefix.beam_station_accumulated)>0.
    saved=a.checkpoint()
    # Re-evaluate a yielded smooth branch from its actual accepted predecessor.
    driver=a._finite_driver;state=driver._state
    def evaluate(q):
        return driver._assemble(q,1.,driver._shell_predecessors,
            driver._beam_predecessors,applied_load=prefix.applied_dead_load)
    residual,tangent,*_=evaluate(state)
    direction=np.zeros(len(residual))
    for i in range(len(a._beam_ids)):
        direction[a._shell_count+6*i+2]=.001*(i+1)
    direction[a._fixed]=0.
    expected=tangent@direction
    weights=np.r_[a._coordinate_scale,np.ones(a._C.shape[0])]
    errors=[]
    for h in (1e-4,1e-5,1e-6):
        plus=evaluate(driver._advance(state,direction,h))[0]
        minus=evaluate(driver._advance(state,direction,-h))[0]
        errors.append(np.linalg.norm(((plus-minus)/(2*h)-expected)*weights)/
                      max(np.linalg.norm(expected*weights),1.))
    assert min(errors)<=1e-5,errors
    assert a.checkpoint()==saved
    original=FinitePointDriver._assemble
    def broken(self,*args,**kwargs):
        original(self,*args,**kwargs);raise RuntimeError('injected yielded-line trial')
    with monkeypatch.context() as patch:
        patch.setattr(FinitePointDriver,'_assemble',broken)
        with pytest.raises(RuntimeError,match='injected'):a.solve(programme,targets,increment_control=control,max_iterations=20,max_backtracks=12)
    assert a.checkpoint()==saved
    a.rebase_rotations(prefix.generation)
    b=make(request,definition);b.restore(a.checkpoint())
    final=a.solve(programme,targets,on_accepted=observe,max_iterations=20,max_backtracks=12,increment_control=control)
    resumed=b.solve(programme,targets,max_iterations=20,max_backtracks=12,increment_control=control)
    assert final.load_programme_complete
    assert np.max(final.beam_station_accumulated)>np.max(prefix.beam_station_accumulated)
    np.testing.assert_array_equal(final.beam_positions,resumed.beam_positions)
    np.testing.assert_array_equal(final.beam_station_accumulated,resumed.beam_station_accumulated)
    assert a.checkpoint()==b.checkpoint()


def test_generalized_bl5_distributed_yielded_checkpoint(request):
    definition=model(hardening=1e8)
    a=make(request,definition);b=make(request,definition)
    load=ss.GEBeamDistributedLoad(a.preflight()['identity'],tuple(
        ss.GEBeamLineAction(i,(0.,0.,100.),(2.,3.,0.)) for i in (1,2)))
    first=a.solve(load,(.5,),max_iterations=20,max_backtracks=12)
    assert np.max(first.beam_station_accumulated)>0.
    a.rebase_rotations(first.generation)
    b.restore(a.checkpoint())
    direct=a.solve(load,(1.,),max_iterations=20,max_backtracks=12)
    resumed=b.solve(load,(1.,),max_iterations=20,max_backtracks=12)
    np.testing.assert_array_equal(direct.beam_station_accumulated,resumed.beam_station_accumulated)
    assert a.checkpoint()==b.checkpoint()
    assert direct.free_equilibrium_ratio<=1e-8 and direct.constraint_max<=1e-9
