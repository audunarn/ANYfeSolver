"""Bounded material continuation, not limit-load or branch qualification."""
from dataclasses import replace
import json
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_curved_material import (
    curved_material_panel, make_owner, _actions, assert_generalized_records_equal)


def control_for(owner, orientation):
    info=owner.preflight();count=len(info['arc_metric_node_order'])
    return ss.ArcLengthControl('beam',105,2,
        ss.PhysicalChordMetric(np.full(count,1/count),info['reference_length_m']),
        1e-5,orientation)


@pytest.mark.parametrize('orientation',[-1,1])
def test_material_arc_prefix_restart(request,tmp_path,monkeypatch,orientation):
    from anyfesolver._stiffened_shell_arc_checkpoint import decode_snapshot,encode_snapshot
    d=curved_material_panel();owner=make_owner(request,d)
    seed=owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    seed_checkpoint=owner.checkpoint()
    assert np.max(seed.shell_layer_alpha)>0 and np.max(seed.beam_station_accumulated)>0
    control=control_for(owner,orientation);seen=[];token=CancellationToken()
    def stop(result):
        seen.append(result);token.cancel('complete material arc prefix')
    with pytest.raises(SolveCancelled):
        owner.continue_arc_length(_actions(),control,steps=2,
            cancellation_token=token,on_accepted=stop)
    assert len(seen)==1
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    assert m['policy']=='B3_GE_CURVED_MATERIAL_POINT_TANGENT_PLANE_V1_DEV'
    assert {'shell_plastic','shell_alpha','shell_origin_plastic','shell_origin_alpha'}<=a.keys()
    assert np.max(a['shell_origin_alpha'])>0 and np.max(a['beam_origin_accumulated'])>0
    assert m['max_backtracks']==12
    fresh=make_owner(request,d);restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw
    assert_generalized_records_equal(restored.beam_generalized_material,seen[0].beam_generalized_material)
    for left,right in zip(restored.shell_q4_material,seen[0].shell_q4_material):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    actual=owner.continue_arc_length(_actions(),control)
    repeated=fresh.continue_arc_length(_actions(),control)
    assert owner.checkpoint()==fresh.checkpoint()
    np.testing.assert_array_equal(actual.cell_resultants,repeated.cell_resultants)
    assert_generalized_records_equal(actual.beam_generalized_material,repeated.beam_generalized_material)
    assert actual.free_equilibrium_ratio<=1e-8 and actual.constraint_max<=1e-9
    # Same physical increments and accepted material predecessor, another driver.
    force=make_owner(request,d);force.restore(seed_checkpoint)
    for arc in (seen[0],actual):
        matched=force.solve(_actions(),(1.+arc.amplitude,),
                            max_iterations=20,max_backtracks=16)
        for name,floor in (('shell_displacement',1e-11),('cell_resultants',1e-7),
                           ('support_reactions',1e-7),('shell_layer_alpha',1e-12),
                           ('beam_station_accumulated',1e-12)):
            np.testing.assert_allclose(getattr(arc,name),getattr(matched,name),
                                       rtol=1e-7,atol=floor)
    # Exact material predecessor replay is mandatory even with valid framing.
    for name in ('shell_origin_alpha','beam_origin_accumulated'):
        arrays=dict(a);arrays[name]=a[name].copy();arrays[name].flat[0]+=1e-3
        poisoned=encode_snapshot(owner,m,arrays);target=make_owner(request,d)
        with pytest.raises(ValueError):target.restore(poisoned)
        assert target._generation==0 and target._finite_driver is None
    with pytest.raises(ValueError,match='metadata'):
        encode_snapshot(owner,dict(m,policy='B3_GE_CURVED_ELASTIC_POINT_TANGENT_PLANE_V1_DEV'),a)
    with pytest.raises(ValueError,match='arc history'):owner.solve(_actions())
    # Candidate validation finishes, but cancellation before commit is atomic.
    from anyfesolver import _stiffened_shell_arc_publication as publication
    prefix=owner.checkpoint();generation=owner._generation;cancel=CancellationToken()
    safe_point=publication.cancellation_safe_point
    def cancel_at_commit(token,where):
        if where=='stiffened-shell.arc.commit':cancel.cancel('before material arc commit')
        return safe_point(token,where)
    with monkeypatch.context() as patch:
        patch.setattr(publication,'cancellation_safe_point',cancel_at_commit)
        with pytest.raises(SolveCancelled):
            owner.continue_arc_length(_actions(),control,cancellation_token=cancel)
    assert owner.checkpoint()==prefix and owner._generation==generation
    with pytest.raises(ValueError,match='stale arc publication'):
        publication.publish(owner,prefix,expected_driver=None,expected_generation=generation)
    assert owner.checkpoint()==prefix and owner._generation==generation
    (tmp_path/'material-arc.json').write_text(json.dumps(dict(
        orientation=orientation,history=actual.history,seed_load_factor=1.,
        equilibrium=actual.free_equilibrium_ratio,constraint=actual.constraint_max,
        shell_alpha_max=float(np.max(actual.shell_layer_alpha)),
        beam_alpha_max=float(np.max(actual.beam_station_accumulated)),
        exact_restart=True),indent=2))


def test_material_arc_without_shell_j2(request):
    from anyfesolver._stiffened_shell_arc_checkpoint import decode_snapshot
    d=curved_material_panel();d=replace(d,shell=replace(d.shell,j2_curve=None))
    owner=make_owner(request,d);owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    result=owner.continue_arc_length(_actions(),control_for(owner,-1))
    assert not result.shell_q4_material and not result.shell_layer_alpha
    assert np.max(result.beam_station_accumulated)>0
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    assert m['policy']=='B3_GE_CURVED_MATERIAL_POINT_TANGENT_PLANE_V1_DEV'
    assert not {'shell_plastic','shell_alpha','shell_origin_plastic','shell_origin_alpha'}&a.keys()
    fresh=make_owner(request,d);restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw
    assert_generalized_records_equal(restored.beam_generalized_material,result.beam_generalized_material)
