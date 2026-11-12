"""Local continuation mechanics; no public or mesh-uniform qualification."""
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._stiffened_shell_arc_metric import PhysicalChordMetric
from anyfesolver._stiffened_shell_displacement_corrector import bordered_path_correction


def test_physical_chord_gradient_units_frame_and_ownership():
    from anyfesolver import stiffened_shell as ss
    class AlteredMetric(PhysicalChordMetric):pass
    with pytest.raises(ValueError,match='explicit physical arc metric'):
        ss.ArcLengthControl('beam',105,2,AlteredMetric([1.],1.),1e-5)
    weights=np.array([.2,.8]);metric=PhysicalChordMetric(weights,2.)
    weights[:]=9.
    with pytest.raises(ValueError):metric.weights.setflags(write=True)
    reference=np.array([rotation([.2,.1,-.3]),rotation([-.3,.2,.1])])
    frames=np.array([rotation([.1,-.2,.15])@r for r in reference])
    u=np.array([[.04,-.01,.02],[-.02,.03,.05]])
    du=np.array([[.2,.1,-.1],[-.1,.3,.2]])
    spin=np.array([[.3,-.2,.4],[.2,.1,-.3]])
    value,gt,gr=metric.evaluate(u,frames,reference)
    predicted=np.sum(gt*du)+np.sum(gr*spin)
    h=1e-5
    def changed(amount):
        return metric.evaluate(u+amount*du,
            np.array([rotation(amount*v)@r for v,r in zip(spin,frames)]),reference)[0]
    assert (changed(h)-changed(-h))/(2*h)==pytest.approx(predicted,rel=1e-8)
    basis=rotation([.5,-.1,.2])
    assert metric.evaluate(u@basis.T,basis@frames,basis@reference)[0]==pytest.approx(value,rel=1e-13)
    assert PhysicalChordMetric([.2,.8],2000.).evaluate(u*1000.,frames,reference)[0]==pytest.approx(value)
    with pytest.raises(ValueError,match='pi/2'):
        metric.evaluate(u,np.array([rotation([2.,0,0])@r for r in reference]),reference)


def test_general_border_at_fold_keeps_algebraic_equilibrium():
    # Singular physical stiffness plus an independent indefinite algebraic block.
    matrix=sparse.csc_matrix([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
    residual=np.array([-.4,2.,-3.]);pattern=np.array([1.,0.,0.])
    row=np.array([.6,0.,0.])
    step,load=bordered_path_correction(matrix,residual,pattern,row,-.12,np.arange(3))
    np.testing.assert_allclose(matrix@step-load*pattern,-residual,atol=1e-15)
    assert row@step==pytest.approx(.12)


def test_native_arc_trial_preserves_accepted_owner(request,tmp_path):
    from test_stiffened_shell_bl5_coupled_material import coupled
    from test_stiffened_shell_bl5_material import make
    from test_stiffened_shell_bl5_finite import _action
    from anyfesolver._stiffened_shell_arc_trial import correct,select_trial
    from anyfesolver.control import CancellationToken,SolveCancelled
    owner=make(request,coupled())
    seed=(Path(__file__).parent/'data/stiffened_shell_event_excursion/force-seed.bin').read_bytes()
    owner.restore(seed);driver=owner._finite_driver
    count=len(owner.preflight()['arc_metric_node_order'])
    metric=PhysicalChordMetric(np.full(count,1/count),owner._reference_length)
    args=dict(load_direction=driver._load(_action(-15.625)).copy(),metric=metric,
        radius=1e-5,predictor_dof=owner._shell_count+6*owner._beam_index[105]+2,orientation=-1)
    import json
    records=[];attempts=[]
    try:
        selected=select_trial(driver,**args,iteration_records=records,attempt_records=attempts)
        result=selected.trial
    finally:
        from dataclasses import asdict
        (tmp_path/'iterations.json').write_text(json.dumps(records,indent=2))
        (tmp_path/'attempts.json').write_text(json.dumps([asdict(row) for row in attempts],indent=2))
    assert selected.requested_radius==1e-5
    assert attempts[0].radius==1e-5 and not attempts[0].accepted
    assert result.radius<selected.requested_radius
    assert attempts[-1].accepted and attempts[-1].radius==result.radius
    assert result.evaluation[2]<=1e-8 and result.evaluation[3]<=1e-9
    assert abs(result.path_error)<=1e-9
    assert result.predecessor_state is driver._state
    assert result.shell_predecessor is driver._shell_origins
    assert result.beam_predecessor is driver._beam_origins
    np.testing.assert_array_equal(result.load_direction,args['load_direction'])
    args['load_direction'][:]=0.
    assert np.any(result.load_direction)
    for array in (result.base_load,result.load_direction,result.normal_translations,result.normal_spins):
        with pytest.raises(ValueError):array.setflags(write=True)
    assert owner.checkpoint()==seed
    token=CancellationToken();token.cancel('before trial')
    args['load_direction']=driver._load(_action(-15.625))
    with pytest.raises(SolveCancelled):correct(driver,**args,cancellation_token=token)
    assert owner.checkpoint()==seed
    from hashlib import sha256
    from anyfesolver._stiffened_shell_arc_checkpoint import from_selection,decode_snapshot,encode_snapshot
    from anyfesolver._stiffened_shell_arc_publication import publish
    raw=from_selection(owner,selected,seed_generation=owner._generation,
        seed_sha256=sha256(seed).hexdigest())
    metadata,arrays=decode_snapshot(owner,raw)
    wrong=dict(metadata,predecessor_sha256='0'*64)
    with pytest.raises(ValueError,match='predecessor'):
        publish(owner,encode_snapshot(owner,wrong,arrays),expected_driver=driver,expected_generation=owner._generation)
    wrong=dict(metadata,generation=metadata['generation']+1,seed_generation=metadata['seed_generation']+1)
    with pytest.raises(ValueError,match='predecessor'):
        publish(owner,encode_snapshot(owner,wrong,arrays),expected_driver=driver,expected_generation=owner._generation)
    assert owner.checkpoint()==seed
    with pytest.raises(SolveCancelled):
        publish(owner,raw,expected_driver=driver,expected_generation=owner._generation,cancellation_token=token)
    assert owner.checkpoint()==seed
    accepted=publish(owner,raw,expected_driver=driver,expected_generation=owner._generation)
    assert owner.checkpoint()==raw and accepted.arc_position==result.radius
    restored=make(request,coupled());repeated=restored.restore(raw)
    assert restored.checkpoint()==raw
    np.testing.assert_array_equal(accepted.cell_resultants,repeated.cell_resultants)
    np.testing.assert_array_equal(accepted.shell_layer_alpha,repeated.shell_layer_alpha)
    with pytest.raises(ValueError,match='arc history'):owner.solve(_action(-15.625))
    # Checksum-correct corruption still fails physical path validation.
    metadata,arrays=decode_snapshot(owner,raw);bad=dict(arrays)
    bad['normal_spins']=arrays['normal_spins']*2.
    with pytest.raises(ValueError,match='normal|path'):encode_snapshot(owner,metadata,bad)


def test_public_arc_cancel_restart_and_absolute_load(request,tmp_path,monkeypatch):
    from dataclasses import replace
    from anyfesolver import stiffened_shell as ss
    from anyfesolver.control import CancellationToken,SolveCancelled
    from test_stiffened_shell_bl5_coupled_material import coupled
    from test_stiffened_shell_bl5_material import make
    from test_stiffened_shell_bl5_finite import _action
    from anyfesolver._stiffened_shell_arc_checkpoint import decode_snapshot,encode_snapshot
    owner=make(request,coupled())
    seed=(Path(__file__).parent/'data/stiffened_shell_event_excursion/force-seed.bin').read_bytes()
    owner.restore(seed)
    preflight=owner.preflight();count=len(preflight['arc_metric_node_order'])
    control=ss.ArcLengthControl('beam',105,2,
        ss.PhysicalChordMetric(np.full(count,1/count),preflight['reference_length_m']),5e-6,-1)
    token=CancellationToken();seen=[]
    def stop(result):
        seen.append(result);token.cancel('accepted first arc step')
        with pytest.raises(RuntimeError,match='reentrant'):
            owner.continue_arc_length(_action(-15.625),control)
    with pytest.raises(SolveCancelled):
        owner.continue_arc_length(_action(-15.625),control,steps=2,cancellation_token=token,on_accepted=stop)
    assert len(seen)==1 and seen[0].status=='accepted_prefix'
    first=owner.checkpoint();m,a=decode_snapshot(owner,first)
    assert m['steps']==1 and m['amplitude_origin']==0.
    assert len(m['attempt_history'])==1
    for changed in (replace(control,radius=1e-5),replace(control,orientation=1),replace(control,max_iterations=21)):
        with pytest.raises(ValueError,match='differs'):
            owner.continue_arc_length(_action(-15.625),changed)
    assert owner.checkpoint()==first
    second=owner.continue_arc_length(_action(-15.625),control)
    assert second.status=='completed' and len(second.history)==2
    assert len(second.attempt_history)==2
    final=owner.checkpoint();m2,a2=decode_snapshot(owner,final)
    assert m2['attempt_history'][0]==m['attempt_history'][0]
    assert m2['amplitude_origin']==seen[0].amplitude
    np.testing.assert_array_equal(a2['base_load'],a['base_load'])
    np.testing.assert_array_equal(a2['applied_load'],a['base_load']+second.amplitude*a['load_direction'])
    restored=make(request,coupled());restored.restore(first)
    repeated=restored.continue_arc_length(_action(-15.625),control)
    assert restored.checkpoint()==final
    np.testing.assert_array_equal(second.cell_resultants,repeated.cell_resultants)
    np.testing.assert_array_equal(second.shell_layer_alpha,repeated.shell_layer_alpha)
    # A checksum-correct deletion of earlier failed/accepted work is invalid.
    bad=dict(m2,attempt_history=m2['attempt_history'][1:])
    with pytest.raises(ValueError,match='attempt history'):encode_snapshot(owner,bad,a2)
    from copy import deepcopy
    for key,value in (('amplitude_origin',1.),('max_iterations',21),('requested_radius',1e-5)):
        bad=deepcopy(m2);bad['attempt_history'][0][key]=value
        with pytest.raises(ValueError):encode_snapshot(owner,bad,a2)
    def forbidden(*args,**kwargs):raise AssertionError('selected arc recovery must not assemble')
    monkeypatch.setattr(owner._finite_driver,'_assemble',forbidden)
    selected=owner.recover(selection=ss.GERecoverySelection(beam_nodes=(105,),
        beam_material_elements=(second.beam_generalized_material[0].element_id,)))
    assert selected.amplitude==second.amplitude and selected.beam_generalized_material
    assert owner.checkpoint()==final
    with monkeypatch.context() as guard:
        guard.setattr(owner,'_external_context',object(),raising=False)
        assert 'arc_length' not in owner.capabilities
        with pytest.raises(ValueError,match='nodal dead translations'):
            owner.continue_arc_length(_action(-15.625),control)
    import json
    from dataclasses import asdict
    (tmp_path/'arc-public.json').write_text(json.dumps(dict(history=second.history,
        equilibrium=second.free_equilibrium_ratio,constraint=second.constraint_max,
        attempts=[[asdict(v) for v in group] for group in second.attempt_history],
        restart_bytes_equal=True),indent=2))
