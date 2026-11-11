from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_distributed_programme import decode_checkpoint
from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
from test_stiffened_shell_ge_distributed_load import owner, balance, load


def programme(a):
    positive=load(a,force=(0.,0.,100.),couple=(0.,0.,0.)).actions
    negative=load(a,force=(0.,0.,-100.),couple=(0.,0.,0.)).actions
    return ss.GEBeamDistributedProgramme(a._identity,(positive,(),negative))


def test_descriptor_and_density_interpolation(request,monkeypatch):
    a=owner(request);p=programme(a)
    assert ss.GEBeamDistributedProgramme.deserialize(p.serialize())==p
    c=p.compile(a)
    state=SimpleNamespace(q=np.zeros(c.template.count),cells=np.tile(np.eye(3),(c.template.beam_count,2,1,1)))
    import anyfesolver._stiffened_shell_native_distributed_load as native
    def forbidden(*args,**kwargs):raise AssertionError('recompiled load geometry')
    monkeypatch.setattr(native,'compile_native_distributed',forbidden)
    for at,density in ((0,0),(.5,50),(1,100),(1.5,50),(2,0),(2.5,-50),(3,-100)):
        values=c.densities_at(at)
        np.testing.assert_array_equal(values[:,2],density)
        assert not values.flags.writeable
        force,jac,recovery=c.evaluate_at(state,at)
        np.testing.assert_allclose(recovery.applied_force,(0,0,2*density),atol=1e-12)
        assert recovery.identity==p.identity
    for targets in ((2.,),(1.,3.),(1.,1.),(-1.,),(1.,2.,4.)):
        with pytest.raises(ValueError):c.validate_targets(0.,targets)
    assert c.validate_targets(0.,(.5,1.,2.,3.))==(.5,1.,2.,3.)
    for endpoints in ((),((),),((p.endpoints[0][0],)*2,)):
        with pytest.raises(ValueError):ss.GEBeamDistributedProgramme(a._identity,endpoints)
    with pytest.raises(ValueError):ss.GEBeamDistributedProgramme.deserialize(p.serialize()+b' ')
    with pytest.raises(ValueError):replace(p,model_identity='0'*64).compile(a)
    with pytest.raises(ValueError,match='finite solve'):a.solve_linear(p)


def test_signed_programme_cancel_restore_rebase_and_complete(request):
    a,b=owner(request),owner(request);base=programme(a)
    p=replace(base,endpoints=(base.endpoints[0],load(a,force=(0.,0.,25.),couple=(0.,0.,0.)).actions,base.endpoints[2]))
    token=CancellationToken()
    def cancel(result):token.cancel('accepted programme prefix')
    with pytest.raises(SolveCancelled):
        a.solve(p,(1.,2.,3.),on_accepted=cancel,cancellation_token=token,max_iterations=15)
    first=a.recover();assert first.load_position==1. and first.load_stage_cursor==1
    assert first.load_programme_identity==p.identity and not first.load_programme_complete
    assert first.applied_dead_load is None
    balance(a,first);saved=a.checkpoint()
    descriptor,inner=decode_checkpoint(saved);assert descriptor==p
    restored=b.restore(saved)
    assert restored.load_position==1. and b.checkpoint()==saved
    with pytest.raises(ValueError):a.solve(p,(3.,))
    assert a.checkpoint()==saved
    unloaded=a.solve(p,(2.,),max_iterations=15)
    b.solve(p,(2.,),max_iterations=15)
    assert unloaded.distributed_load.applied_force==(0.,0.,50.)
    assert a.checkpoint()==b.checkpoint()
    a.rebase_rotations(unloaded.generation)
    c=owner(request);c.restore(a.checkpoint())
    final=a.solve(p,(3.,),max_iterations=15)
    resumed=c.solve(p,(3.,),max_iterations=15)
    assert final.load_programme_complete and final.load_stage_cursor==3
    assert final.accepted_factors==(1.,2.,3.)
    assert final.distributed_load.applied_force[2]<0.
    np.testing.assert_array_equal(final.beam_positions,resumed.beam_positions)
    assert a.checkpoint()==c.checkpoint();balance(a,final)
    fresh=owner(request)
    with pytest.raises(ValueError):fresh.restore(saved[:-1]+bytes([saved[-1]^1]))
    assert fresh._generation==0


def test_adaptive_programme_resumes_pending_sequence(request):
    a,b=owner(request),owner(request);base=programme(a)
    p=replace(base,endpoints=(base.endpoints[0],base.endpoints[2]))
    token=CancellationToken();control=ss.AdaptiveIncrementControl(2,16)
    with pytest.raises(SolveCancelled):
        a.solve(p,(1.,2.),increment_control=control,max_iterations=15,
            cancellation_token=token,on_accepted=lambda r:token.cancel('prefix'))
    b.restore(a.checkpoint())
    first=a.solve(p,(1.,2.),increment_control=control,max_iterations=15)
    second=b.solve(p,(1.,2.),increment_control=control,max_iterations=15)
    assert first.load_programme_complete and second.load_programme_complete
    assert a.checkpoint()==b.checkpoint()


def test_zero_stage_unresolved_acceptance_preserves_prefix(request):
    # Retain the known numerical limitation explicitly. Current assessment has
    # infinite relative residual for every nonzero response at zero total load.
    a=owner(request);p=programme(a)
    a.solve(p,(1.,),max_iterations=15);saved=a.checkpoint()
    with pytest.raises(FiniteIncrementFailure):a.solve(p,(2.,),max_iterations=15)
    assert a.checkpoint()==saved and a.recover().load_position==1.


def test_programme_checkpoint_cannot_be_relabelled(request):
    a=owner(request);p=programme(a);a.solve(p,(1.,),max_iterations=15)
    _,inner=decode_checkpoint(a.checkpoint())
    b=owner(request);changed=replace(p,endpoints=(p.endpoints[2],p.endpoints[0]))
    forged=changed.compile(b).checkpoint(inner)
    with pytest.raises(ValueError):b.restore(forged)
    assert b._generation==0 and b._accepted is None


def test_changed_element_inventory_and_couple_interpolation(request):
    a=owner(request);ids=sorted(e.element_id for e,_,_ in a._beam_parts)
    left=ss.GEBeamLineAction(ids[0],(2.,0.,-3.),(1.,-2.,3.))
    right=ss.GEBeamLineAction(ids[1],(-4.,5.,0.),(-3.,2.,1.))
    p=ss.GEBeamDistributedProgramme(a._identity,((right,),(left,)))
    c=p.compile(a)
    state=SimpleNamespace(q=np.zeros(c.template.count),cells=np.tile(np.eye(3),(c.template.beam_count,2,1,1)))
    assert [row[3] for row in c.template.beams]==ids
    np.testing.assert_array_equal(c.densities_at(1.)[0],np.zeros(6))
    np.testing.assert_array_equal(c.densities_at(2.)[1],np.zeros(6))
    rows=tuple(ss.GEBeamLineAction(row.element_id,
        tuple(.5*v for v in row.force_N_per_reference_m),
        tuple(.5*v for v in row.couple_Nm_per_reference_m)) for row in (right,left))
    expected=ss.GEBeamDistributedLoad(a._identity,rows).compile(a).evaluate(state)
    force,jac,recovery=c.evaluate_at(state,1.5)
    np.testing.assert_array_equal(force,expected[0])
    np.testing.assert_array_equal(jac.toarray(),expected[1].toarray())
    assert recovery.potential is None
    assert recovery.applied_moment_about_origin==expected[2].applied_moment_about_origin
    opposite=replace(left,couple_Nm_per_reference_m=tuple(-v for v in left.couple_Nm_per_reference_m))
    cancelled=ss.GEBeamDistributedProgramme(a._identity,((left,),(opposite,))).compile(a)
    _,_,recovery=cancelled.evaluate_at(state,1.5)
    assert recovery.potential is not None and recovery.potential==recovery.force_potential


@pytest.mark.parametrize('triangles',[False,True])
def test_mixed_shell_signed_programme_restart(request,triangles):
    from test_stiffened_shell_ge_experimental import _panel
    base=_panel()
    shell=replace(base.shell,q4_elements=() if triangles else ((1,2,5,4),),
        s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)) if triangles else ((2,3,6),(2,6,5)))
    definition=replace(base,shell=shell,**({} if triangles else {'q4_pullback_policy':ss.Q4_ALGORITHMIC_PULLBACK_V1}))
    a,b=owner(request,definition),owner(request,definition)
    base=programme(a);p=replace(base,endpoints=(base.endpoints[0],base.endpoints[2]))
    first=a.solve(p,(1.,),max_iterations=15);balance(a,first)
    saved=a.checkpoint();b.restore(saved)
    last=a.solve(p,(2.,),max_iterations=15)
    resumed=b.solve(p,(2.,),max_iterations=15)
    assert last.load_programme_complete and last.distributed_load.applied_force[2]<0.
    assert a.checkpoint()==b.checkpoint()
    np.testing.assert_array_equal(last.shell_displacement,resumed.shell_displacement)
    balance(a,last)
