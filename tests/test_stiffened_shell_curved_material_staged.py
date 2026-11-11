"""Curved material staged translations reuse the existing atomic owner."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_curved_material import curved_material_panel,make_owner,_actions
from test_stiffened_shell_curved_material_displacement import material_equal


OPTIONS=dict(max_iterations=20,max_backtracks=12)


def programme(owner):
    return ss.StagedDeadLoadProgramme(owner.preflight()['identity'],(
        _actions(),
        (ss.NodalAction('beam',105,(0.,1.,990.,0.,0.,0.)),),
        (ss.NodalAction('beam',105,(0.,2.,980.,0.,0.,0.)),)))


@pytest.mark.parametrize('shell_j2',[False,True])
@pytest.mark.parametrize('adaptive',[False,True])
def test_curved_staged_material_progress_restart_and_force_parity(request,monkeypatch,shell_j2,adaptive):
    d=replace(curved_material_panel(),retained_scale_policy='native_force_moment_roles_v1')
    if not shell_j2:d=replace(d,shell=replace(d.shell,j2_curve=None))
    owner=make_owner(request,d);plan=programme(owner)
    options=dict(OPTIONS)
    if adaptive:options['increment_control']=ss.AdaptiveIncrementControl(4,64)
    calls=[];compile=ss.StagedDeadLoadProgramme.compile
    def counted(self,own,load):
        calls.append(own);return compile(self,own,load)
    monkeypatch.setattr(ss.StagedDeadLoadProgramme,'compile',counted)
    snapshots=[];results=[]
    def capture(result):
        snapshots.append(owner.checkpoint());results.append(result)
    seed=owner.solve(plan,(.5,1.),**options,on_accepted=capture)
    assert calls==[owner] and np.max(seed.beam_station_accumulated)>0
    if shell_j2:assert np.max(seed.shell_layer_alpha)>0
    else:assert not seed.shell_layer_alpha and not seed.shell_q4_material
    force=make_owner(request,d);same=force.solve(_actions(),(.5,1.),**options)
    material_equal(seed,same)
    for field in ('shell_displacement','beam_positions','cell_resultants','support_reactions'):
        np.testing.assert_array_equal(getattr(seed,field),getattr(same,field))
    token=CancellationToken()
    def accepted(result):
        capture(result);token.cancel('complete staged material prefix')
    with pytest.raises(SolveCancelled):
        owner.solve(plan,(1.5,2.,3.),**options,on_accepted=accepted,cancellation_token=token)
    assert owner.recover().load_position==1.5 and calls==[owner]
    prefix=owner.checkpoint();fresh=make_owner(request,d);again=fresh.restore(prefix)
    assert fresh.checkpoint()==prefix;material_equal(results[-1],again)
    assert again.load_programme_identity==plan.identity and again.load_stage_cursor==1
    assert again.load_stage_fraction==.5 and not again.load_programme_complete
    remaining=(1.5,2.,3.) if adaptive else (2.,3.)
    left=owner.solve(plan,remaining,**options);right=fresh.solve(plan,remaining,**options)
    assert owner.checkpoint()==fresh.checkpoint() and left.accepted_factors==right.accepted_factors
    assert calls.count(owner)==1 and calls.count(fresh)==1
    material_equal(left,right)
    assert left.load_position==3. and left.load_programme_complete
    assert left.load_stage_cursor==3 and left.load_stage_fraction==0.
    expected=owner._finite_driver._programme.load_at(3.)
    np.testing.assert_array_equal(left.applied_dead_load,expected)
    for field in ('shell_displacement','beam_positions','cell_resultants','support_reactions'):
        np.testing.assert_array_equal(getattr(left,field),getattr(right,field))
    assert left.free_equilibrium_ratio<=1e-8 and left.constraint_max<=1e-9


def test_curved_staged_profile_and_progress_failures_are_atomic(request):
    from test_stiffened_shell_curved_reference_seam import curved_panel
    owner=make_owner(request,curved_material_panel());plan=programme(owner)
    assert owner.capabilities['staged_dead_loads']
    with pytest.raises(ValueError,match='force-control stepping'):
        owner.solve(plan,(.5,),displacement_control=ss.DisplacementControl('beam',105,2))
    with pytest.raises(ValueError,match='every crossed integer boundary'):
        owner.solve(plan,(1.5,),**OPTIONS)
    assert owner._finite_driver is None and owner._generation==0
    elastic=make_owner(request,curved_panel())
    with pytest.raises(ValueError,match='nodal dead translations'):
        elastic.solve(programme(elastic),(.5,),**OPTIONS)
    assert not elastic.capabilities['staged_dead_loads']
    foreign=ss.StagedDeadLoadProgramme('0'*64,plan.endpoints)
    with pytest.raises(ValueError,match='different model'):owner.solve(foreign,(.5,),**OPTIONS)
    assert owner._finite_driver is None and owner._generation==0
    owner.solve(plan,(.5,1.),**OPTIONS);raw=owner.checkpoint()
    changed=ss.StagedDeadLoadProgramme(plan.model_identity,plan.endpoints[:2])
    with pytest.raises(ValueError,match='changed after first increment'):
        owner.solve(changed,(1.5,2.),**OPTIONS)
    assert owner.checkpoint()==raw
    fresh=make_owner(request,replace(curved_material_panel(),shell=replace(curved_material_panel().shell,j2_curve=None)))
    with pytest.raises(ValueError):fresh.restore(raw)
    assert fresh._finite_driver is None and fresh._generation==0
    # Reframe a valid binary endpoint with valid digests: mechanical checks,
    # not checksum failure, must reject changed applied loads/predecessors.
    import json
    from anyfesolver._stiffened_shell_staged_load_experimental import decode_checkpoint,encode_checkpoint
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC,decode,encode
    descriptor,position,inner=decode_checkpoint(raw)
    start=len(MAGIC)+4;length=int.from_bytes(inner[len(MAGIC):start],'little')
    header=json.loads(inner[start:start+length])
    layout=tuple((name,tuple(shape)) for name,shape in header['arrays'])
    kind,generation,arrays=decode(inner,descriptor.augmented_model_identity,lambda k,g:layout)
    compiled=descriptor.compile(owner,owner._finite_driver._load)
    for field in ('applied_dead_load','beam_origin_accumulated','shell_origin_alpha'):
        changed={k:v.copy() for k,v in arrays.items()};changed[field].flat[0]+=1e-3
        altered=encode(kind,descriptor.augmented_model_identity,generation,tuple(changed.items()))
        candidate=encode_checkpoint(compiled,position,altered)
        fresh=make_owner(request,curved_material_panel())
        with pytest.raises(ValueError):fresh.restore(candidate)
        assert fresh._finite_driver is None and fresh._generation==0
    fresh=make_owner(request,curved_material_panel())
    with pytest.raises(ValueError):fresh.restore(inner)
    assert fresh._finite_driver is None and fresh._generation==0


def test_curved_staged_rebase_and_before_publish_cancellation(request,monkeypatch):
    from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
    owner=make_owner(request,curved_material_panel());plan=programme(owner)
    previous=owner.solve(plan,(.5,1.),**OPTIONS)
    before=owner.checkpoint();generation=owner._generation;token=CancellationToken()
    safe=finite.cancellation_safe_point
    def check(cancel,where):
        if where=='stiffened-shell.finite.controller-publish':token.cancel('before staged commit')
        return safe(cancel,where)
    with monkeypatch.context() as patch:
        patch.setattr(finite,'cancellation_safe_point',check)
        with pytest.raises(SolveCancelled):owner.solve(plan,(1.5,),**OPTIONS,cancellation_token=token)
    assert owner.checkpoint()==before and owner._generation==generation
    rebased=owner.rebase_rotations(generation)
    for field in ('shell_displacement','cell_resultants','support_reactions','shell_layer_alpha'):
        np.testing.assert_allclose(getattr(previous,field),getattr(rebased,field),rtol=1e-10,atol=1e-10)
    raw=owner.checkpoint();fresh=make_owner(request,curved_material_panel());restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw;material_equal(rebased,restored)
    left=owner.solve(plan,(1.5,2.),**OPTIONS);right=fresh.solve(plan,(1.5,2.),**OPTIONS)
    assert owner.checkpoint()==fresh.checkpoint();material_equal(left,right)
