"""Independent constrained P1 response for affine support and committed preload."""
import itertools,json,os,importlib
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_arc_length_affine import build,analysis
from test_general_contact_triangulated_rigid_patch import independent_mass
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.arc_contact import ArcContactAdapter
from anyfesolver.general_contact.runtime import active_contact_manager


@pytest.mark.parametrize('preload',[False,True])
def test_affine_preload_response_restart(preload,monkeypatch):
    arc=importlib.import_module('anyfesolver.arc_length');captured=[]
    real=arc._support_reaction_resultants_from_forces
    def recover(model,internal,constant,proportional,factor,**kwargs):
        captured.append(np.asarray(internal)-np.asarray(constant)-factor*np.asarray(proportional))
        return real(model,internal,constant,proportional,factor,**kwargs)
    monkeypatch.setattr(arc,'_support_reaction_resultants_from_forces',recover)
    result=analysis(preload=preload)
    assert result.status=='maximum_steps_reached' and len(result.steps)==3
    model,_,_,_,_=build(preload);K,_=assemble_stiffness_matrix(model)
    z=np.array([model.mesh.dof_manager.get_node_dofs(n)[2] for n in (1,2,3,4)])
    K=K.toarray()[np.ix_(z,z)];M,S,*_=independent_mass(model)
    history=result.info['general_contact_history'];response=[];sphere=[];wrenches=[]
    base=-2e-6*np.ones(4) if preload else np.zeros(4)
    # Independently enumerate all nodal pressure active sets, with imposed root.
    def oracle(lam):
        force=np.array([0.,-.05*lam,-.05*lam,-.05*lam])
        if preload:force[1:]-=.001
        fixed=-2e-6*((1. if preload else 0.)+lam)
        for bits in itertools.product((False,True),repeat=4):
            A=np.array(bits);C=1e6*M[A].T@np.diag(1/S[A])@M[A]
            H=K+C;rhs=force-C@np.full(4,1e-5)
            u=np.empty(4);u[0]=fixed;u[1:]=np.linalg.solve(H[1:,1:],rhs[1:]-H[1:,0]*fixed)
            weighted=M@(np.full(4,1e-5)+u)
            if np.all(weighted[A]<0.) and np.all(weighted[~A]>=0.):return u
        raise AssertionError('no independent active set')
    if preload:
        from anyfesolver.general_contact import transient_results_archive as codec
        base_record=codec._decode(result.restart_checkpoint['path_state']['contact_state']['base'],codec._Budget(codec.MAX_BYTES),{})
        base=base_record['displacements'][z]
        assert np.linalg.norm(base-oracle(0.))/1e-5<1e-6
    previous=np.zeros(model.mesh.dof_manager.total_dofs);previous[z]=base;last=0.
    for row,step,residual in zip(history,result.steps,captured):
        pose=row['displacements'];lam=row['load_factor'];u=pose[z]
        response.append(float(np.linalg.norm(u-oracle(lam))/1e-5))
        sphere.append(abs(np.dot(pose-previous,pose-previous)+.01**2*(lam-last)**2-step.arc_radius**2)/step.arc_radius**2)
        assert abs(u[0]-(-2e-6*((1. if preload else 0.)+lam)))<1e-20
        reaction=np.zeros(6);loadw=np.zeros(6)
        for i,node in enumerate((1,2,3,4)):
            point=np.array([model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z])+pose[model.mesh.dof_manager.get_node_dofs(node)[:3]]
            dofs=model.mesh.dof_manager.get_node_dofs(node);r=residual[dofs].copy()
            if node!=1:r[2]=0.
            reaction[:3]+=r[:3];reaction[3:]+=r[3:]+np.cross(point,r[:3])
            f=np.array([0.,0.,0. if node==1 else -.05*lam-(.001 if preload else 0.)])
            loadw[:3]+=f;loadw[3:]+=np.cross(point,f)
        c=row['contact'];contact=np.r_[c['integration_force'],c['integration_moment']]
        wrenches.append(max(np.linalg.norm((reaction+loadw+contact)[:3])/.15,np.linalg.norm((reaction+loadw+contact)[3:])/.15))
        assert c['gap_tolerance_met'] and c['minimum_gap']>=-5e-5
        assert max(np.linalg.norm(np.array(c['integration_force'])+c['target_force'])/.15,np.linalg.norm(np.array(c['integration_moment'])+c['target_moment'])/.15)<1e-6
        previous,last=pose,lam
    first=analysis(1,preload=preload);resumed=analysis(2,first.to_restart_checkpoint(),preload=preload)
    assert result.restart_checkpoint_bytes()==resumed.restart_checkpoint_bytes()
    assert max(response)<1e-6 and max(sphere)<1e-6 and max(wrenches)<1e-6
    if preload:
        with pytest.raises(ValueError):analysis(1,first.to_restart_checkpoint(),preload_force=.002)
        with pytest.raises(ValueError):analysis(1,first.to_restart_checkpoint(),preload=False)
        def forbidden(*args,**kwargs):raise AssertionError('preload rerun')
        monkeypatch.setattr(ArcContactAdapter,'preload',forbidden)
        assert analysis(2,first.to_restart_checkpoint()).restart_checkpoint_bytes()==result.restart_checkpoint_bytes()
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(str(Path(path).with_name('preload_result.json' if preload else 'affine_result.json'))).write_text(json.dumps({'preload':preload,'response':response,'spherical':sphere,'wrenches':wrenches,'gaps':[r['contact']['minimum_gap'] for r in history],'restart_exact':True},indent=2))
    assert active_contact_manager() is None


def test_preload_cancel_and_exception(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    static=importlib.import_module('anyfesolver.nonlinear_static');original=static._assemble_nonlinear_system
    calls=[]
    def broken(*args,**kwargs):
        calls.append(True)
        raise RuntimeError('preload assembly rejected')
    monkeypatch.setattr(static,'_assemble_nonlinear_system',broken)
    with pytest.raises(RuntimeError,match='preload assembly rejected'):analysis()
    assert calls and active_contact_manager() is None
    monkeypatch.setattr(static,'_assemble_nonlinear_system',original)
    token=CancellationToken();token.cancel('preload cancellation')
    with pytest.raises(SolveCancelled):analysis(cancellation_token=token)
    assert active_contact_manager() is None


def test_zero_force_prescribed_preload_and_inflight_cancel(monkeypatch):
    from anyfesolver.general_contact import transient_results_archive as codec
    from anyfesolver.control import CancellationToken,SolveCancelled
    result=analysis(1,preload_force=0.)
    base=codec._decode(result.restart_checkpoint['path_state']['contact_state']['base'],codec._Budget(codec.MAX_BYTES),{})
    model,_,_,_,_=build(True)
    z=[model.mesh.dof_manager.get_node_dofs(n)[2] for n in (1,2,3,4)]
    assert base['displacements'][z[0]] == -2e-6
    # Ordinary residual tolerance permits this soft fixture's free displacement
    # to remain zero; only the exact imposed stage is asserted here.
    K,_=assemble_stiffness_matrix(model)
    assert np.linalg.norm((K@base['displacements'])[z[1:]]) <= 1e-6
    assert base['history'][-1]['load_factor']==1.
    static=importlib.import_module('anyfesolver.nonlinear_static');real=static.cancellation_safe_point
    token=CancellationToken();observed=[]
    def cancel(value,stage=''):
        if stage!='nonlinear_static.start':
            observed.append(stage);token.cancel('in-flight static preload')
        return real(value,stage)
    monkeypatch.setattr(static,'cancellation_safe_point',cancel)
    with pytest.raises(SolveCancelled):analysis(cancellation_token=token)
    assert observed and active_contact_manager() is None


def test_resigned_preload_intermediate_constraint_refusal():
    import copy,hashlib
    from anyfesolver.general_contact import transient_results_archive as codec
    from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
    result=analysis(1);capsule=copy.deepcopy(result.to_restart_checkpoint())
    namespace=capsule['path_state']['contact_state']
    base=codec._decode(namespace['base'],codec._Budget(codec.MAX_BYTES),{})
    model,definition,settings,_,_=build(True);adapter=ArcContactAdapter(model,definition,settings,preload=True)
    row=base['history'][0];row['displacements'][2]-=1e-7
    row['contact']=adapter.prepare(row['displacements'],row['step_index'],row['load_factor'])['contact']
    namespace['base']=codec._encode(base,codec._Budget(codec.MAX_BYTES),{})
    unsigned=dict(capsule);unsigned.pop('checkpoint_sha256')
    capsule['checkpoint_sha256']=hashlib.sha256(canonical_checkpoint_json_bytes(unsigned)).hexdigest().upper()
    with pytest.raises(ValueError,match='preload base violates prescribed constraints'):
        analysis(1,capsule)


def test_unselected_plastic_preload_refuses_before_preload(monkeypatch):
    from anyfesolver import create_shell_element,solve_static_arc_length
    model,definition,settings,load,constant=build(True)
    model.add_material('unselected_plastic',1e5,.3,yield_stress=1.)
    model.add_element(3,create_shell_element(3,(1,2,4),'unselected_plastic',thickness=.01,
        formulation='e4-pl-s3-v2d',reference_normal=(0,0,1)))
    def forbidden(*args,**kwargs):raise AssertionError('physical preload started')
    monkeypatch.setattr(ArcContactAdapter,'preload',forbidden)
    with pytest.raises(ValueError,match='whole assembled model elastic'):
        solve_static_arc_length(model,load,constant_load_case=constant,
            contact=definition,contact_settings=settings)
    assert active_contact_manager() is None


def test_resigned_preload_history_summary_inventory_refusal():
    import copy,hashlib
    from anyfesolver.general_contact import transient_results_archive as codec
    from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
    capsule=copy.deepcopy(analysis(1).to_restart_checkpoint())
    namespace=capsule['path_state']['contact_state']
    base=codec._decode(namespace['base'],codec._Budget(codec.MAX_BYTES),{})
    base['history']=base['history'][1:]
    namespace['base']=codec._encode(base,codec._Budget(codec.MAX_BYTES),{})
    unsigned=dict(capsule);unsigned.pop('checkpoint_sha256')
    capsule['checkpoint_sha256']=hashlib.sha256(canonical_checkpoint_json_bytes(unsigned)).hexdigest().upper()
    with pytest.raises(ValueError,match='history/summary inventory mismatch'):
        analysis(1,capsule)
