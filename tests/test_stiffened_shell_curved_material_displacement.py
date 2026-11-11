"""Curved material displacement histories have distinct persistent identities."""
from dataclasses import replace
import json
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_curved_material import (
    curved_material_panel,make_owner,_actions,assert_generalized_records_equal)


def material_equal(a,b):
    assert_generalized_records_equal(a.beam_generalized_material,b.beam_generalized_material)
    np.testing.assert_array_equal(a.shell_layer_alpha,b.shell_layer_alpha)
    assert len(a.shell_q4_material)==len(b.shell_q4_material)
    for x,y in zip(a.shell_q4_material,b.shell_q4_material):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(x,name),getattr(y,name))


@pytest.mark.parametrize('sign',[-1,1])
def test_curved_displacement_active_prefix_restart_and_force_parity(request,monkeypatch,tmp_path,sign):
    from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot,encode_snapshot
    from anyfesolver import _stiffened_shell_displacement_control as publication
    d=replace(curved_material_panel(),retained_scale_policy='native_force_moment_roles_v1')
    owner=make_owner(request,d)
    seed=owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    seed_raw=owner.checkpoint()
    assert np.max(seed.shell_layer_alpha)>0 and np.max(seed.beam_station_accumulated)>0
    dof=owner._shell_count+6*owner._beam_index[105]+2
    initial=owner._finite_driver._state.q[dof]
    targets=(initial*(1+sign*.01),initial*(1+sign*.02))
    control=ss.DisplacementControl('beam',105,2)
    kw=dict(displacement_control=control,max_iterations=20,max_backtracks=12)
    token=CancellationToken();seen=[]
    def accepted(result):
        seen.append(result);token.cancel('accepted curved displacement prefix')
    with pytest.raises(SolveCancelled):
        owner.solve(_actions(),targets,**kw,cancellation_token=token,on_accepted=accepted)
    assert len(seen)==1
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    assert m['policy']=='B3_GE_CURVED_MATERIAL_POINT_DISPLACEMENT_V1_DEV'
    assert np.max(a['shell_origin_alpha'])>0 and np.max(a['beam_origin_accumulated'])>0
    from test_stiffened_shell_curved_reference_seam import curved_panel
    from test_stiffened_shell_bl5_coupled_material import coupled
    for foreign_definition in (curved_panel(),coupled(),replace(d,shell=replace(d.shell,j2_curve=None))):
        foreign=make_owner(request,foreign_definition)
        with pytest.raises(ValueError):foreign.restore(raw)
        assert foreign._generation==0 and foreign._finite_driver is None
    restored=make_owner(request,d);prefix=restored.restore(raw)
    assert restored.checkpoint()==raw;material_equal(prefix,seen[0])
    actual=owner.solve(_actions(),(targets[1],),**kw)
    repeated=restored.solve(_actions(),(targets[1],),**kw)
    assert owner.checkpoint()==restored.checkpoint() and actual.history==repeated.history
    np.testing.assert_array_equal(actual.cell_resultants,repeated.cell_resultants)
    material_equal(actual,repeated)
    assert actual.free_equilibrium_ratio<=1e-8 and actual.constraint_max<=1e-9
    # Match physical increments and predecessors; force history remains force history.
    force=make_owner(request,d);force.restore(seed_raw)
    for endpoint in (seen[0],actual):
        matched=force.solve(_actions(),(1+endpoint.amplitude,),max_iterations=20,max_backtracks=16)
        for name,floor in (('shell_displacement',1e-11),('cell_resultants',1e-7),
                ('support_reactions',1e-7),('shell_layer_alpha',1e-12),('beam_station_accumulated',1e-12)):
            np.testing.assert_allclose(getattr(endpoint,name),getattr(matched,name),rtol=1e-7,atol=floor)
    for name in ('shell_origin_alpha','beam_origin_accumulated'):
        arrays=dict(a);arrays[name]=a[name].copy();arrays[name].flat[0]+=1e-3
        poisoned=encode_snapshot(owner,generation=m['generation'],seed_generation=m['seed_generation'],
            seed_sha256=m['seed_sha256'],control_dof=m['control_dof'],arrays=arrays)
        fresh=make_owner(request,d)
        with pytest.raises(ValueError):fresh.restore(poisoned)
        assert fresh._generation==0 and fresh._finite_driver is None
    saved=owner.checkpoint();generation=owner._generation;cancel=CancellationToken()
    safe=publication.cancellation_safe_point
    def before_commit(token,where):
        if where=='stiffened-shell.displacement.publish':cancel.cancel('before commit')
        return safe(token,where)
    with monkeypatch.context() as patch:
        patch.setattr(publication,'cancellation_safe_point',before_commit)
        with pytest.raises(SolveCancelled):
            owner.solve(_actions(),(initial*(1+sign*.03),),**kw,cancellation_token=cancel)
    assert owner.checkpoint()==saved and owner._generation==generation
    with pytest.raises(ValueError,match='stale displacement publication'):
        publication.publish(owner,saved,expected_driver=None,expected_generation=generation)
    assert owner.checkpoint()==saved
    (tmp_path/'curved-displacement.json').write_text(json.dumps(dict(sign=sign,history=actual.history,
        equilibrium=actual.free_equilibrium_ratio,constraint=actual.constraint_max,exact_restart=True),indent=2))


def test_curved_displacement_elastic_shell_event_and_chart_restart(request):
    from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot
    d=curved_material_panel();d=replace(d,shell=replace(d.shell,j2_curve=None))
    owner=make_owner(request,d);owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    dof=owner._shell_count+6*owner._beam_index[105]+2
    initial=owner._finite_driver._state.q[dof]
    control=ss.DisplacementControl('beam',105,2,trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL)
    kw=dict(displacement_control=control,max_iterations=20,max_backtracks=12)
    result=owner.solve(_actions(),(initial*1.01,),**kw)
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    assert m['policy']=='B3_GE_CURVED_MATERIAL_POINT_DISPLACEMENT_EVENT_V1_DEV'
    assert not {'shell_plastic','shell_alpha','shell_origin_plastic','shell_origin_alpha'}&a.keys()
    assert not result.shell_q4_material and not result.shell_layer_alpha
    assert np.max(result.beam_station_accumulated)>0
    fresh=make_owner(request,d);restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw;material_equal(result,restored)
    rebased=owner.rebase_rotations(owner._generation)
    for name in ('shell_displacement','beam_positions','cell_resultants','support_reactions'):
        np.testing.assert_allclose(getattr(rebased,name),getattr(result,name),rtol=1e-10,atol=1e-10)
    changed=owner.checkpoint();m,a=decode_snapshot(owner,changed)
    assert m['chart_events']==1 and m['policy']=='B3_GE_CURVED_MATERIAL_POINT_DISPLACEMENT_EVENT_V1_DEV'
    restored_rebase=make_owner(request,d);again=restored_rebase.restore(changed)
    assert restored_rebase.checkpoint()==changed;material_equal(rebased,again)
    left=owner.solve(_actions(),(initial*1.02,),**kw)
    right=restored_rebase.solve(_actions(),(initial*1.02,),**kw)
    assert owner.checkpoint()==restored_rebase.checkpoint();material_equal(left,right)


def test_curved_displacement_requires_seed_and_matching_profile(request):
    d=curved_material_panel();owner=make_owner(request,d)
    control=ss.DisplacementControl('beam',105,2)
    with pytest.raises(ValueError,match='accepted force-controlled seed'):
        owner.solve(_actions(),(.001,),displacement_control=control)
    assert owner._generation==0 and owner._finite_driver is None
    from test_stiffened_shell_curved_reference_seam import curved_panel
    elastic=make_owner(request,curved_panel())
    with pytest.raises(ValueError,match='displacement control'):
        elastic.solve(_actions(),(.001,),displacement_control=control)
    assert elastic._generation==0 and elastic._finite_driver is None


def test_curved_displacement_coupled_chart_replay(request):
    d=curved_material_panel();owner=make_owner(request,d)
    owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    dof=owner._shell_count+6*owner._beam_index[105]+2;initial=owner._finite_driver._state.q[dof]
    kw=dict(displacement_control=ss.DisplacementControl('beam',105,2),max_iterations=20,max_backtracks=12)
    previous=owner.solve(_actions(),(initial*.99,),**kw)
    rebased=owner.rebase_rotations(previous.generation)
    for name in ('shell_displacement','cell_resultants','support_reactions','shell_layer_alpha','beam_station_accumulated'):
        np.testing.assert_allclose(getattr(previous,name),getattr(rebased,name),rtol=1e-10,atol=1e-10)
    raw=owner.checkpoint();fresh=make_owner(request,d);restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw;material_equal(rebased,restored)
    left=owner.solve(_actions(),(initial*.98,),**kw);right=fresh.solve(_actions(),(initial*.98,),**kw)
    assert owner.checkpoint()==fresh.checkpoint();material_equal(left,right)


@pytest.mark.parametrize('shell_j2',[False,True])
def test_curved_displacement_adaptive_accepted_prefix(request,shell_j2):
    d=curved_material_panel()
    if not shell_j2:d=replace(d,shell=replace(d.shell,j2_curve=None))
    owner=make_owner(request,d)
    owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    dof=owner._shell_count+6*owner._beam_index[105]+2;initial=owner._finite_driver._state.q[dof]
    targets=(initial*.99,initial*.98);control=ss.DisplacementControl('beam',105,2)
    limits=ss.AdaptiveIncrementControl()
    kw=dict(displacement_control=control,increment_control=limits,max_iterations=20,max_backtracks=12)
    token=CancellationToken();seen=[]
    def accepted(result):
        seen.append(result);token.cancel('complete adaptive displacement prefix')
    with pytest.raises(SolveCancelled):
        owner.solve(_actions(),targets,**kw,cancellation_token=token,on_accepted=accepted)
    assert len(seen)==1
    raw=owner.checkpoint();fresh=make_owner(request,d);restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw;material_equal(seen[0],restored)
    left=owner.solve(_actions(),targets,**kw);right=fresh.solve(_actions(),targets,**kw)
    assert owner.checkpoint()==fresh.checkpoint();material_equal(left,right)
    assert left.status=='completed' and left.target==targets[-1]


def test_curved_fixed_displacement_to_arc_transition(request):
    from test_stiffened_shell_curved_material_arc import control_for
    d=curved_material_panel();owner=make_owner(request,d)
    owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    dof=owner._shell_count+6*owner._beam_index[105]+2;initial=owner._finite_driver._state.q[dof]
    displacement=ss.DisplacementControl('beam',105,2)
    owner.solve(_actions(),(initial*.99,),displacement_control=displacement,max_iterations=20,max_backtracks=12)
    raw=owner.checkpoint();fresh=make_owner(request,d);fresh.restore(raw)
    control=control_for(owner,-1)
    left=owner.continue_arc_length(_actions(),control)
    right=fresh.continue_arc_length(_actions(),control)
    assert owner.checkpoint()==fresh.checkpoint();material_equal(left,right)
    assert left.history==right.history
    saved=owner.checkpoint()
    with pytest.raises(ValueError,match='arc history'):
        owner.solve(_actions(),(initial*.98,),displacement_control=displacement)
    assert owner.checkpoint()==saved
    restored=make_owner(request,d);again=restored.restore(saved)
    assert restored.checkpoint()==saved;material_equal(left,again)
