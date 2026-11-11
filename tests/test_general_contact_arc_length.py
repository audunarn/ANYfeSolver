"""Existing continuation engine, independent P1 response, contact transactions."""
from dataclasses import replace
import itertools,json,os,copy
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_arc_length import analysis
from examples.general_contact_triangulated_rigid_patch import case
from test_general_contact_triangulated_rigid_patch import independent_mass
from anyfesolver import solve_static_arc_length,ArcLengthControl,LoadCase
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact import report_general_contact_capability
from anyfesolver.general_contact.runtime import active_contact_manager
from anyfesolver.general_contact.arc_contact import PROFILE,ArcContactAdapter


def test_actual_arc_patch_response_restart(monkeypatch):
    import importlib
    arc=importlib.import_module('anyfesolver.arc_length');captured=[]
    real=arc._support_reaction_resultants_from_forces
    def recover(model,internal,constant,proportional,factor,**kwargs):
        captured.append(np.asarray(internal)-np.asarray(constant)-factor*np.asarray(proportional))
        return real(model,internal,constant,proportional,factor,**kwargs)
    monkeypatch.setattr(arc,'_support_reaction_resultants_from_forces',recover)
    model,definition,settings=case();m,s,_,_,_=independent_mass(model)
    baseline,_=assemble_stiffness_matrix(model)
    dofs=[model.mesh.dof_manager.get_node_dofs(i)[2] for i in (1,2,3,4)]
    k=baseline[np.ix_(dofs,dofs)].toarray();free=(1,2,3);gap=np.full(4,1e-5)
    result=analysis();assert result.status=='maximum_steps_reached' and len(result.steps)==4
    history=result.info['general_contact_history'];assert any(row['contact']['penalty_energy']>0 for row in history)
    errors=[];metric_errors=[];wrench_errors=[]
    for step,row,snapshot,imbalance in zip(result.steps,history,result.snapshots,captured):
        load=np.array((0.,-.05,-.05,-.05))*step.load_factor;references=[]
        for mask in itertools.product((False,True),repeat=4):
            active=np.array(mask);ma=m[active];sa=s[active]
            contact=1e6*ma.T@np.diag(1/sa)@ma if any(active) else np.zeros((4,4))
            u=np.zeros(4);u[list(free)]=np.linalg.solve((k+contact)[np.ix_(free,free)],(load-contact@gap)[list(free)])
            weighted=m@(gap+u)
            if np.all(weighted[active]<0) and np.all(weighted[~active]>=0):references.append(u)
        assert len(references)==1
        errors.append(float(np.linalg.norm(snapshot.displacements[dofs]-references[0])/1e-5))
        metric_errors.append(abs(step.path_increment_norm**2-step.arc_radius**2)/step.arc_radius**2)
        certificate=row['contact'];assert certificate['gap_tolerance_met'] and certificate['minimum_gap']>=-5e-5
        force=np.array(certificate['integration_force'])+certificate['target_force']
        moment=np.array(certificate['integration_moment'])+certificate['target_moment']
        wrench_errors.append(max(np.linalg.norm(force)/.15,np.linalg.norm(moment)/.15))
        # Arc step reactions use resultant force/moment about global origin.
        reaction=np.zeros(6)
        for node in (1,2,3,4):
            values=imbalance[model.mesh.dof_manager.get_node_dofs(node)].copy()
            if node!=1:values[2]=0.
            position=np.array((model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z))+snapshot.displacements[model.mesh.dof_manager.get_node_dofs(node)[:3]]
            reaction[:3]+=values[:3]
            reaction[3:]+=np.cross(position,values[:3])+values[3:]
        physical=np.array(certificate['integration_force']);torque=np.array(certificate['integration_moment'])
        loaded=np.array((0.,0.,-.15*step.load_factor));loadmoment=np.zeros(3)
        for node in (2,3,4):
            point=np.array((model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z))+snapshot.displacements[model.mesh.dof_manager.get_node_dofs(node)[:3]]
            loadmoment+=np.cross(point,(0.,0.,-.05*step.load_factor))
        wrench_errors.append(max(np.linalg.norm(reaction[:3]+loaded+physical)/.15,
                                 np.linalg.norm(reaction[3:]+loadmoment+torque)/.15))
    assert max(errors)<1e-6 and max(metric_errors)<1e-6 and max(wrench_errors)<1e-6
    first=analysis(2);resumed=analysis(2,first.to_restart_checkpoint())
    assert result.restart_checkpoint_bytes()==resumed.restart_checkpoint_bytes()
    with pytest.raises(ValueError):
        changed=copy.deepcopy(first.to_restart_checkpoint());changed['analysis_fingerprint']='wrong'
        analysis(1,changed)
    report={'status':result.status,'steps':len(result.steps),'profile':PROFILE,
        'response_errors':errors,'spherical_metric_errors':metric_errors,'wrench_errors':wrench_errors,
        'gaps':[row['contact']['minimum_gap'] for row in history],
        'energies':[row['contact']['penalty_energy'] for row in history],'restart_exact':True}
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path).write_text(json.dumps(report,indent=2))


def test_arc_refusal_cancel_and_exception(monkeypatch):
    model,definition,settings=case();load=LoadCase('test');load.add_nodal_load(2,forces=(0.,0.,-.05))
    kwargs=dict(contact=definition,contact_settings=settings,control=ArcLengthControl(initial_load_increment=2.5e-5,minimum_load_increment=5e-7,maximum_load_increment=2.5e-5,max_steps=1))
    assert report_general_contact_capability(model,definition,settings,analysis_control='arc_length').validation.profile==PROFILE
    for extra in ({'kinematics':'corotational'},{'imperfection':{}}):
        with pytest.raises(ValueError):solve_static_arc_length(model,load,**kwargs,**extra)
    from anyfesolver.control import CancellationToken,SolveCancelled
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):solve_static_arc_length(model,load,**kwargs,cancellation_token=token)
    assert active_contact_manager() is None
    original=ArcContactAdapter.evaluate
    def bad(self,pose,**kw):
        if np.any(pose):raise RuntimeError('candidate assembly failure')
        return original(self,pose,**kw)
    with monkeypatch.context() as scoped:
        scoped.setattr(ArcContactAdapter,'evaluate',bad)
        with pytest.raises(RuntimeError,match='candidate assembly failure'):
            solve_static_arc_length(model,load,**kwargs)
    assert active_contact_manager() is None
    adapter=ArcContactAdapter(model,definition,settings);adapter.manager.law.stiffness=0.
    with pytest.raises(ValueError):adapter.evaluate(np.zeros(adapter.manager.n_dof))


def test_no_contact_linear_response_compatibility():
    model,_,_=case();load=LoadCase('plain')
    for node in (2,3,4):load.add_nodal_load(node,forces=(0.,0.,-1e-5))
    result=solve_static_arc_length(model,load,control=ArcLengthControl(max_steps=1),emit_restart_checkpoint=True)
    assert result.status=='maximum_steps_reached'
    assert 'general_contact_profile' not in result.info
    assert 'contact_state' not in result.restart_checkpoint['path_state']
    assert active_contact_manager() is None


@pytest.mark.skipif(not os.environ.get('ARC_CONTACT_DIAGNOSTIC'),reason='explicit exploratory diagnostic only')
def test_first_arc_failure_discriminator(monkeypatch):
    import importlib
    arc=importlib.import_module('anyfesolver.arc_length')
    original=arc._finalize_arc_element_states
    sweeps=[];old_admissible=ArcContactAdapter.admissible
    def check(self,start,end):
        value=old_admissible(self,start,end)
        sweeps.append({'valid':value,'max_translation':float(np.max(np.abs(end))),
            'gap':self.manager._gap_lower(self.manager._positions(end))})
        return value
    monkeypatch.setattr(ArcContactAdapter,'admissible',check)
    def final(model,u,states,num_layers,info,**kwargs):
        import inspect
        loc=inspect.currentframe().f_back.f_locals
        report={'failure_reason':loc.get('failure_reason'),'adaptation':loc.get('adaptation_history'),'sweeps':sweeps,'u_norm':float(np.linalg.norm(u)),
                'states':len(states),'info':{k:v for k,v in info.items() if k in ('control','factorization_error')}}
        path=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if path:Path(path).write_text(json.dumps(report,indent=2))
        return original(model,u,states,num_layers,info,**kwargs)
    monkeypatch.setattr(arc,'_finalize_arc_element_states',final)
    analysis()


def test_inflight_cancel_and_rejected_retry(monkeypatch):
    import importlib
    arc=importlib.import_module('anyfesolver.arc_length')
    from anyfesolver.control import CancellationToken,SolveCancelled
    token=CancellationToken();real_safe=arc.cancellation_safe_point;seen=[];real_commit=ArcContactAdapter.commit
    def safe(value,stage=''):
        if stage=='arc_length.contact_before_commit':token.cancel('in flight')
        return real_safe(value,stage)
    monkeypatch.setattr(arc,'cancellation_safe_point',safe)
    monkeypatch.setattr(ArcContactAdapter,'commit',lambda self,candidate:seen.append(candidate))
    model,definition,settings=case();load=LoadCase('cancel');load.add_nodal_load(2,forces=(0.,0.,-.05))
    with pytest.raises(SolveCancelled):
        solve_static_arc_length(model,load,contact=definition,contact_settings=settings,
            control=ArcLengthControl(initial_load_increment=2.5e-5,minimum_load_increment=5e-7,
                                   maximum_load_increment=2.5e-5,max_steps=1),cancellation_token=token)
    assert seen==[] and active_contact_manager() is None
    monkeypatch.setattr(arc,'cancellation_safe_point',real_safe)
    monkeypatch.setattr(ArcContactAdapter,'commit',real_commit)
    real=ArcContactAdapter.admissible;rejected=[]
    def once(self,start,end):
        if not rejected and np.any(end):
            rejected.append((len(self.history),start.copy()));return False
        return real(self,start,end)
    monkeypatch.setattr(ArcContactAdapter,'admissible',once)
    retried=analysis(2)
    assert rejected[0][0]==0 and not np.any(rejected[0][1])
    assert len(retried.info['general_contact_history'])==2 and retried.steps[0].retries>=1


def test_recomputed_intermediate_history_mutation_refuses():
    import hashlib
    from anyfesolver.general_contact import transient_results_archive as codec
    prefix=analysis(2);capsule=prefix.to_restart_checkpoint()
    history=codec._decode(capsule['path_state']['contact_state']['history'],codec._Budget(codec.MAX_BYTES),{})
    manager=ArcContactAdapter(*case())
    # Remain in physical support chart, yet alter an earlier accepted pose and
    # regenerate its real contact summary. Structural path metrics must refuse.
    history[0]['displacements'][8]-=1e-6
    history[0]['contact']=manager.prepare(history[0]['displacements'],1,history[0]['load_factor'])['contact']
    capsule['path_state']['contact_state']['history']=codec._encode(history,codec._Budget(codec.MAX_BYTES),{})
    unsigned=dict(capsule);unsigned.pop('checkpoint_sha256')
    from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
    capsule['checkpoint_sha256']=hashlib.sha256(canonical_checkpoint_json_bytes(unsigned)).hexdigest().upper()
    with pytest.raises(ValueError,match='structural step metric|spherical constraint|predictor history'):
        analysis(1,capsule)


def test_contact_adapter_work_and_tangent():
    model,definition,settings=case();adapter=ArcContactAdapter(model,definition,settings)
    pose=np.zeros(adapter.manager.n_dof);direction=pose.copy()
    for index,node in enumerate((1,2,3,4)):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        pose[dofs[2]]=-2e-5;direction[dofs[2]]=(.2,-.3,.4,-.1)[index]
    direction[model.mesh.dof_manager.get_node_dofs(2)[3]]=.02
    force,tangent,summary=adapter.evaluate(pose);h=1e-8
    plus=adapter.evaluate(pose+h*direction);minus=adapter.evaluate(pose-h*direction)
    work=(plus[2].penalty_energy-minus[2].penalty_energy)/(2*h)
    assert abs(work-force@direction)/max(abs(force@direction),1e-12)<1e-5
    observed=(plus[0]-minus[0])/(2*h);expected=tangent@direction
    assert np.linalg.norm(observed-expected)/np.linalg.norm(expected)<1e-5


def test_zero_accepted_contact_failure_is_explicit():
    model,definition,settings=case();load=LoadCase('too_large')
    for node in (2,3,4):load.add_nodal_load(node,forces=(0.,0.,-.05))
    result=solve_static_arc_length(model,load,contact=definition,contact_settings=settings,
        control=ArcLengthControl(initial_load_increment=.25,minimum_load_increment=.0005,
            maximum_load_increment=.25,load_scaling=.01,growth_factor=1.,max_steps=1))
    assert result.status=='diverged' and result.steps==[] and not np.any(result.displacements)
    assert result.info['general_contact_history']==[]
    assert result.info['failure_reason']=='minimum_arc_radius_reached'


def test_intact_checkpoint_settings_incompatibility():
    from anyfesolver.nonlinear_restart import load_nonlinear_checkpoint
    prefix=analysis(2)
    capsule=prefix.to_restart_checkpoint()
    verified=load_nonlinear_checkpoint(capsule)
    assert verified['checkpoint_sha256']==capsule['checkpoint_sha256']
    model,definition,settings=case();load=LoadCase('arc_patch_press')
    for node in (2,3,4):load.add_nodal_load(node,forces=(0.,0.,-.05))
    changed=replace(settings,general_s3_difference_step=2e-6)
    with pytest.raises(ValueError,match='analysis|contract|fingerprint'):
        solve_static_arc_length(model,load,contact=definition,contact_settings=changed,
            control=ArcLengthControl(initial_load_increment=2.5e-5,minimum_load_increment=5e-7,
                maximum_load_increment=2.5e-5,load_scaling=.01,growth_factor=1.,max_steps=1),
            max_iterations=15,record_increment_snapshots=True,restart_checkpoint=capsule)
