"""Native fixed-predecessor continuation and lifecycle; no limit-point claim."""
from pathlib import Path
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_arc_checkpoint import decode_snapshot,encode_snapshot,identities,NATIVE_CHART_POLICY
from test_stiffened_shell_native_point_displacement import owner,native_state
from test_stiffened_shell_native_generalized_adaptive_history import exact
from test_stiffened_shell_q4_pullback_owner import _definition
from examples.stiffened_shell_coupled_s3_adaptive_history import definition as s3_definition


def seed(request,family):
    model=_definition(plastic=True,mixed=False) if family=='q4-generalized' else s3_definition()
    fixture='native_generalized_q4_j2_history/unloaded-v1.bin' if family=='q4-generalized' else 'native_s3_fibre_history/unloaded-v1.bin'
    source=owner(request,model)
    source.restore((Path(__file__).parent/'data'/fixture).read_bytes())
    # Seed from the mechanically validated endpoint, leaving the adaptive schedule.
    analysis=owner(request,model);analysis.restore(source._displacement_snapshot[0])
    nodes=len(analysis._shell_ids)+len(analysis._beam_ids)
    control=ss.ArcLengthControl('beam',105,2,
        ss.PhysicalChordMetric(np.full(nodes,1/nodes),1.),1e-6)
    actions=(ss.NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),)
    if family=='s3-fibre':actions+= (ss.NodalAction('beam',205,(0.,0.,100.,0.,0.,0.)),)
    return model,analysis,control,actions


def compare_arc(left,right):
    exact(left,right)
    assert left.increment_attempts==right.increment_attempts
    assert left.attempt_history==right.attempt_history
    for a,b in zip(left.shell_q4_material+left.shell_s3_material,
                   right.shell_q4_material+right.shell_s3_material,strict=True):
        assert a.element_id==b.element_id
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('family',('q4-generalized','s3-fibre'))
@pytest.mark.parametrize('rebase',(False,True))
def test_native_arc_fixed_predecessor_restart_and_cancel(request,monkeypatch,family,rebase):
    model,analysis,control,actions=seed(request,family)
    assert analysis.capabilities['arc_length']=='experimental_native_point_material_tangent_plane_v1'
    if rebase:analysis.rebase_rotations(analysis._generation)
    seed_raw=analysis.checkpoint();generation=analysis._generation
    # Native generalized/fibre increments must bypass BL5 event-cell selection.
    import anyfesolver._stiffened_shell_branch_event_trial as events
    def forbidden(*args,**kwargs):raise AssertionError('native arc used BL5 event fractions')
    monkeypatch.setattr(events,'trial_fraction',forbidden)
    token=CancellationToken();seen=[]
    def stop(result):seen.append(result);token.cancel('accepted native prefix')
    with pytest.raises(SolveCancelled):
        analysis.continue_arc_length(actions,control,steps=2,cancellation_token=token,on_accepted=stop)
    assert len(seen)==1 and analysis._generation==generation+1
    raw=analysis.checkpoint();m,a=decode_snapshot(analysis,raw)
    assert m['policy']==identities(analysis)[1] and m['native_chart_policy']==NATIVE_CHART_POLICY
    assert m['seed_generation']==generation and m['steps']==1
    assert m['attempts'][-1]['event_cell_evaluations']==0
    if family=='s3-fibre':
        assert len(a['s3_states'])==2*len(model.shell.s3_elements)
        assert seen[0].shell_s3_material and seen[0].beam_fibre_material
    else:assert seen[0].beam_generalized_material and seen[0].shell_q4_material
    resumed=owner(request,model);restored=resumed.restore(raw)
    compare_arc(restored,seen[0]);assert resumed.checkpoint()==raw
    for name,value in native_state(analysis).items():
        np.testing.assert_array_equal(value,native_state(resumed)[name])
    final=analysis.continue_arc_length(actions,control)
    repeated=resumed.continue_arc_length(actions,control)
    compare_arc(final,repeated);assert analysis.checkpoint()==resumed.checkpoint()
    for name,value in native_state(analysis).items():
        np.testing.assert_array_equal(value,native_state(resumed)[name])
    assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
    assert not final.production_qualified
    # Cancellation after isolated preparation leaves the live endpoint intact.
    import anyfesolver._stiffened_shell_arc_publication as publication
    prefix=analysis.checkpoint();driver=analysis._finite_driver;token=CancellationToken()
    safe=publication.cancellation_safe_point
    def cancel_at_commit(value,stage):
        if stage=='stiffened-shell.arc.commit':token.cancel('native commit')
        return safe(value,stage)
    with monkeypatch.context() as patch:
        patch.setattr(publication,'cancellation_safe_point',cancel_at_commit)
        with pytest.raises(SolveCancelled):analysis.continue_arc_length(actions,control,cancellation_token=token)
    assert analysis.checkpoint()==prefix and analysis._finite_driver is driver
    with pytest.raises(ValueError,match='stale arc publication'):
        publication.publish(analysis,raw,expected_driver=None,expected_generation=analysis._generation)
    assert analysis.checkpoint()==prefix
    assert seed_raw!=raw


@pytest.mark.parametrize('family',('q4-generalized','s3-fibre'))
def test_native_arc_corrupted_replay_is_atomic(request,family):
    model,analysis,control,actions=seed(request,family)
    analysis.continue_arc_length(actions,control)
    m,a=decode_snapshot(analysis,analysis.checkpoint())
    names=('beam_origin_accumulated','shell_origin_alpha') if family=='q4-generalized' else ('beam_origin_fibres','shell_origin_q')
    for name in names:
        forged=dict(a);forged[name]=a[name].copy();forged[name].flat[-1]+=.001
        raw=encode_snapshot(analysis,m,forged);fresh=owner(request,model)
        with pytest.raises(ValueError):fresh.restore(raw)
        assert fresh._generation==0 and fresh._finite_driver is None
    for metadata in (dict(m,native_chart_policy='unbound'),dict(m,generation=m['generation']+1)):
        with pytest.raises(ValueError):encode_snapshot(analysis,metadata,a)
    if family=='s3-fibre':
        from scipy.spatial.transform import Rotation
        forged=dict(a);forged['shell_origin_anchors']=a['shell_origin_anchors'].copy()
        forged['shell_origin_anchors'][0]=Rotation.from_rotvec((.01,0.,0.)).as_matrix()
        raw=encode_snapshot(analysis,m,forged);fresh=owner(request,model)
        with pytest.raises(ValueError,match='predecessor anchor violates rotational support'):fresh.restore(raw)
        assert fresh._generation==0 and fresh._finite_driver is None


def test_stable_elastic_s3_force_seed_arc_and_historical_rejection(request,monkeypatch):
    from test_stiffened_shell_native_fibre_s3_displacement import fibre_model
    from test_stiffened_shell_native_point_displacement import OPTIONS,DIRECTION
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode,encode
    from anyfesolver._stiffened_shell_beam_history import shapes
    model=fibre_model();analysis=owner(request,model);legacy=owner(request,model)
    zero=(ss.NodalAction('beam',105,(0.,0.,0.,0.,0.,0.)),)
    analysis.solve(zero,(1.,),**OPTIONS)
    assert analysis._finite_driver._stable_s3
    def layout(kind,generation):
        assert kind=='finite_native_material_stable'
        n=analysis._K.shape[0]
        return (('q',(n,)),('frames',(len(analysis._beam_ids),3,3)),
                ('cells',(len(analysis._beam_parts),2,3,3)),
                ('resultants',(len(analysis._beam_parts),18)),
                ('multipliers',(analysis._C.shape[0],)),('pattern',(n,)),
                ('accepted_factors',(generation,)))+shapes(analysis._beam_parts,'origin')+shapes(analysis._beam_parts,'accepted')
    _,generation,arrays=decode(analysis.checkpoint(),analysis._identity,layout)
    historical=encode('finite_native_material',analysis._identity,generation,tuple(arrays.items()))
    legacy.restore(historical);before=legacy.recover()
    nodes=len(analysis._shell_ids)+len(analysis._beam_ids)
    control=ss.ArcLengthControl('beam',105,2,ss.PhysicalChordMetric(np.full(nodes,1/nodes),1.),1e-6)
    result=analysis.continue_arc_length(DIRECTION,control)
    assert analysis._finite_driver._stable_s3 and result.beam_fibre_material
    assert not result.shell_s3_material and not result.production_qualified
    resumed=owner(request,model);restored=resumed.restore(analysis.checkpoint())
    compare_arc(result,restored)
    assert resumed._finite_driver._stable_s3 and resumed.checkpoint()==analysis.checkpoint()
    from anyfesolver import _stiffened_shell_arc_control as controller
    from anyfesolver._stiffened_shell_arc_trial import correct
    def forbidden(*args,**kwargs):raise AssertionError('historical S3 arc seed reached trial or assembly')
    with monkeypatch.context() as patch:
        patch.setattr(controller,'select_trial',forbidden)
        patch.setattr(legacy,'checkpoint',forbidden)
        patch.setattr(legacy._finite_driver,'_assemble',forbidden)
        with pytest.raises(ValueError,match='seed numerical policy mismatch'):
            legacy.continue_arc_length(DIRECTION,control)
        with pytest.raises(ValueError,match='seed numerical policy mismatch'):
            correct(legacy._finite_driver,load_direction=legacy._finite_driver._load(DIRECTION),
                metric=control.metric,radius=control.radius,predictor_dof=ss.DisplacementControl('beam',105,2).dof(legacy),orientation=1)
    assert legacy.recover() is before and legacy._generation==generation
    assert legacy.checkpoint()==historical and not legacy._finite_driver._stable_s3
    assert getattr(legacy,'_arc_snapshot',None) is None


@pytest.mark.parametrize('family',('q4-generalized','s3-fibre'))
def test_native_arc_v2_progress_metadata_and_v1_restore_continuation(request,family):
    model,analysis,control,actions=seed(request,family)
    analysis.continue_arc_length(actions,control)
    m,a=decode_snapshot(analysis,analysis.checkpoint())
    assert m['policy'].endswith('_V2_DEV')
    dof=m['predictor_dof'];node,component=divmod(dof,6)
    assert m['orientation']*(a['q'][dof]-a['origin_translations'][node,component])>0.
    # A checksum-valid orientation change preserves equilibrium and frozen plane,
    # but contradicts the V2 corrected-coordinate direction contract.
    with pytest.raises(ValueError,match='no oriented progress'):
        encode_snapshot(analysis,dict(m,orientation=-m['orientation']),a)
    for sign in (0.,-1.):
        forged=dict(a);forged['origin_translations']=a['origin_translations'].copy()
        forged['origin_translations'][node,component]=a['q'][dof]-sign*m['orientation']*control.radius
        with pytest.raises(ValueError,match='no oriented progress'):
            encode_snapshot(analysis,m,forged)
    legacy_metadata=dict(m,policy=identities(analysis,native_version=1)[1])
    legacy_raw=encode_snapshot(analysis,legacy_metadata,a)
    legacy=owner(request,model);legacy.restore(legacy_raw)
    assert legacy.checkpoint()==legacy_raw
    legacy.continue_arc_length(actions,control)
    continued,arrays=decode_snapshot(legacy,legacy.checkpoint())
    assert continued['policy']==legacy_metadata['policy']
    resumed=owner(request,model);resumed.restore(legacy_raw)
    resumed.continue_arc_length(actions,control)
    assert resumed.checkpoint()==legacy.checkpoint()
