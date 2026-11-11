"""Focused current D1 helper refusals; no acceptance/reference campaign."""
import copy,json,os
from pathlib import Path
import pytest
from scripts.general_contact_d1_displacement_cycle_dev import build_displacement_cycle,load_restart,validate_saved_displacement
from scripts.general_contact_d1_interior_press_dev import validate_restart_poses

FIXTURE=Path(__file__).resolve().parents[1]/'docs/d1_review_support/d1_final_installed_workflow_v1_20260930_prefix.json'

@pytest.mark.parametrize('field,value',[('steps_per_segment',1),('source_columns',2),('source_support','left'),('reference_slope',.01),('target_young_multiplier',1.),('target_left_edge_motion',False),('q4_gap_certificate',None)])
def test_registered_displacement_refuses_changed_fixture(field,value):
    saved=json.loads(FIXTURE.read_text());saved[field]=value
    with pytest.raises(ValueError,match='unsupported registered fixture'):build_displacement_cycle(saved)

def test_null_restart_is_refused(tmp_path):
    source=tmp_path/'null.json';source.write_text('{"restart_checkpoint":null}')
    with pytest.raises(ValueError,match='no committed checkpoint'):load_restart(source)

@pytest.mark.parametrize('kind',['null','invalid_prefix'])
def test_force_restart_refusal_preserves_global_sweep_method(kind,tmp_path,monkeypatch):
    import inspect
    from scripts import general_contact_d1_interior_press_dev as entry
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    fixture=json.loads(FIXTURE.read_text())
    payload=copy.deepcopy(fixture)
    if kind=='null':payload={'restart_checkpoint':None}
    else:payload['accepted_pose_displacements'][-1][38]+=1e-4
    restart=tmp_path/'restart.json';restart.write_text(json.dumps(payload))
    arguments={name:fixture[name] for name in inspect.signature(entry.main).parameters if name in fixture}
    arguments.update(capture_swept_failure=True,record_swept_fractions=True,record_accepted_poses=True,restart_checkpoint_path=restart)
    calls=[]
    def forbidden_solve(*args,**kwargs):
        calls.append(True)
        raise AssertionError('refused restart must not launch solve')
    monkeypatch.setattr(entry,'solve_static_nonlinear',forbidden_solve)
    original=GeneralS3PatchManager.admissible_step_fraction
    with pytest.raises(ValueError):entry.main(tmp_path/'unused-output.json',**arguments)
    assert GeneralS3PatchManager.admissible_step_fraction is original
    assert calls==[]

@pytest.mark.parametrize('kind',['weighted_row','load','endpoint','ncp','target','pose','terminal','history','step_norm','nan_pose','inf_target','nan_norm','pose_shape'])
@pytest.mark.skipif(not os.environ.get('D1_CURRENT_DISPLACEMENT_RESULT'),
    reason='archived displacement result D1_CURRENT_DISPLACEMENT_RESULT absent: mutated-contract refusal replay requires the original saved result')
def test_saved_displacement_refuses_mutated_contract(kind):
    import os
    original=json.loads(Path(os.environ['D1_CURRENT_DISPLACEMENT_RESULT']).read_text());saved=copy.deepcopy(original)
    if kind=='weighted_row':saved['control_terms'][0][2]*=2.
    elif kind=='load':saved['external_nodal_load_count']=1
    elif kind=='endpoint':saved['endpoint_m']=0.
    elif kind=='ncp':saved['ncp_scale_n']=.018
    elif kind=='target':saved['history'][0]['target'][-1]=0.
    elif kind=='pose':saved['accepted_pose_displacements'][0][38]+=1e-4
    elif kind=='terminal':saved['displacements'][38]+=1e-4
    elif kind=='history':saved['history']=[]
    elif kind=='step_norm':saved['restart_checkpoint']['path_state']['steps'][0]['displacement_norm']+=1e-4
    elif kind=='nan_pose':saved['accepted_pose_displacements'][0][38]=float('nan')
    elif kind=='inf_target':saved['history'][0]['target'][-1]=float('inf')
    elif kind=='nan_norm':saved['restart_checkpoint']['path_state']['steps'][0]['displacement_norm']=float('nan')
    elif kind=='pose_shape':saved['accepted_pose_displacements'][0].pop()
    with pytest.raises(ValueError):validate_saved_displacement(saved)

@pytest.mark.parametrize('kind',['endpoint','count','nan','shape'])
@pytest.mark.skipif(not os.environ.get('D1_CURRENT_DISPLACEMENT_RESULT'),
    reason='archived displacement result D1_CURRENT_DISPLACEMENT_RESULT absent: prefix-pose refusal replay requires the original saved result')
def test_prefix_poses_refused_before_advance(kind):
    import os
    saved=json.loads(Path(os.environ['D1_CURRENT_DISPLACEMENT_RESULT']).read_text());cp=saved['restart_checkpoint']
    if kind=='endpoint':saved['accepted_pose_displacements'][-1][38]+=1e-4
    elif kind=='count':saved['accepted_steps']=2
    elif kind=='nan':saved['accepted_pose_displacements'][0][38]=float('nan')
    elif kind=='shape':saved['accepted_pose_displacements'][0].pop()
    with pytest.raises(ValueError):validate_restart_poses(saved,cp)
