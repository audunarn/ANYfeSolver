"""Independent return/work reference and actual nontrivial material subincrements."""
import os,json,copy
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pytest
from examples.general_contact_s3_mesh_history_events import build_case,solve_case
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager

CUTS={2:(2,4,6,8,10,12),4:(4,8,10,12,16,20,24)}
def case(ratio=2):return build_case(ratio,'sliding')
def save(v):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))
def evidence():return Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])

def test_prepare():
    m,d,s,f,p,ref=case()
    save(dict(program=p.contract(),settings=asdict(s),friction=asdict(f),reference=ref,
        partitions=CUTS,Fref_n=.02,Mref_nm=.02,gap_m=5e-5,criteria=dict(eq=1e-6,C=1e-8,tangent=1e-5,work_j=1e-12),
        slider=dict(p_pa=1.,mu=.3,k_pa_per_m=1000.,area_m2=.01,positions=[0,.0001,.01,.0099,.009],analytic_relative_target=.001)))

def test_slider():
    from anyfesolver.general_contact.s3_friction_history_path import summarize_work_history
    m,d,s,f,p,_=case();owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    u=np.zeros(60)
    for node in (2,3,4,5):u[m.mesh.dof_manager.get_node_dofs(node)[2]]=-2e-7
    z=np.array([.01/6,.01/3,.01/6,.01/3]);owner.commit(owner.validate_successor(u,z))
    e=0.;xold=0.;k=1000.;cap=.3;A=.01;rows=[]
    for x in (.0001,.01,.0099,.009):
        ds=x-xold;trial=e+ds;returned=np.clip(trial,-cap/k,cap/k);plastic=trial-returned
        knots=[0.,1.]+[(bound-e)/ds for bound in (-cap/k,cap/k) if 0<(bound-e)/ds<1]
        knots.sort();integral=0.
        for a,b in zip(knots,knots[1:]):
            integral+=.5*k*(np.clip(e+a*ds,-cap/k,cap/k)+np.clip(e+b*ds,-cap/k,cap/k))*ds*(b-a)
        exact_work=A*integral;deltaE=.5*A*k*(returned**2-e**2);D=A*cap*abs(plastic)
        assert abs(exact_work-deltaE-D)<=1e-12
        for node in range(1,6):u[m.mesh.dof_manager.get_node_dofs(node)[0]]=x
        prepared=owner.validate_successor(u,z);report=prepared[2]
        measured=report['dissipation_increment_j'];projection=report['return_projection_loss_j']
        endpoint=-report['material_relative_endpoint_contraction_j']
        expected_endpoint=A*k*returned*ds
        assert abs(measured-D)<=1e-12
        assert abs(projection-.5*A*k*plastic**2)<=1e-12
        assert abs(report['return_storage_drop_anchor_j']-(D+.5*A*k*plastic**2))<=1e-12
        assert abs(endpoint-expected_endpoint)<=1e-12
        assert abs(endpoint-deltaE-D-.5*A*k*(returned-e)**2)<=1e-12
        assert abs(report['return_algebra_error_j'])<=1e-12
        rows.append(dict(x=x,exact_integrated_work_j=exact_work,endpoint_work_j=endpoint,
            D_j=D,projection_excess_j=projection,storage_change_j=deltaE,branch=report['branch']))
        owner.commit(prepared);e=returned;xold=x
    uopen=u.copy()
    for node in (2,3,4,5):uopen[m.mesh.dof_manager.get_node_dofs(node)[2]]=2e-6
    released=owner.validate_successor(uopen,np.zeros(4));assert abs(released[2]['released_storage_j']-.5*A*k*e*e)<=1e-12
    owner.commit(released);birth=owner.validate_successor(u,z)
    assert all(b=='activation' for b in birth[2]['branch']) and birth[2]['storage_j']==0 and birth[2]['dissipation_increment_j']==0
    # Independent finite-gap common-spin cancellation: each separate piece nonzero.
    n=np.array([0.,0.,1.]);r=np.array([.002,.001,2e-5]);t=np.array([.3,0.,0.]);omega=np.array([0.,2.,0.])
    material=A*t@((np.eye(3)-np.outer(n,n))@np.cross(omega,r));numerical=-A*(n@r)*t@np.cross(omega,n)
    assert abs(material)>1e-12 and abs(numerical)>1e-12 and abs(material+numerical)<=1e-12
    empty=summarize_work_history([]);assert empty['status']=='no_accepted_rows'
    mixed=summarize_work_history([{'step_index':1,'friction':{}},{'step_index':2,'friction':birth[2]}])
    assert mixed['status']=='partial_schema_coverage' and mixed['covered_physical_steps']==[2] and mixed['excluded_physical_steps']==[1]
    save(dict(slider=rows,common_spin_material_j=material,common_spin_numerical_j=numerical,
        zero_pressure_release_j=released[2]['released_storage_j'],rebirth=birth[2]['branch'],mixed_schema=mixed,
        energy_scope='fixednormal_constantpressure_analytic_limit; variablepressure_endpoint_rule_separate'))

@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE'),
    reason='installed history stage input FRICTION_STAGE absent: staged nontrivial-work solve requires the original workflow stage selection')
def test_public():
    stage=os.environ['FRICTION_STAGE'];prefix=stage[0];ratio,step=map(int,stage[1:].split('-'));c=case(ratio)
    previous=0
    if prefix=='R':previous=max((v for v in CUTS[ratio] if v<step),default=0);prior=f'R{ratio}-{previous}'
    else:
        prior={5:'R2-4',6:'I2-5',11:'R2-10',12:'I2-11'}[step];previous=step-1
    incoming=json.loads((evidence()/(prior+'.json')).read_text())['checkpoint'] if previous else None
    result=solve_case(c,step,incoming);cp=result.to_restart_checkpoint() if result.status=='completed' else None
    output=dict(status=result.status,info=result.info,displacement=result.displacements,
        snapshots=[dict(step_index=x.step_index,displacement=x.displacements) for x in result.snapshots],checkpoint=cp)
    save(output);assert result.status=='completed',result.info
    assert [x['step_index'] for x in output['snapshots']]==list(range(previous+1,step+1))
    assert result.info['contact_path']['work_reporting']['status']=='complete'
    (evidence()/(os.environ.get('EVIDENCE_STAGE',stage)+'-checkpoint.json')).write_text(json.dumps(cp,indent=2))
    if prefix=='I':
        baseline=json.loads((evidence()/f'R{ratio}-{step if step in CUTS[ratio] else step+1}.json').read_text())
        pose=next(x['displacement'] for x in baseline['snapshots'] if x['step_index']==step)
        phase=baseline['checkpoint']['path_state']['history'][step-1]
        np.testing.assert_allclose(result.displacements,pose,rtol=0,atol=1e-10)
        np.testing.assert_allclose(cp['path_state']['normal_forces'],phase['friction']['supported_force_n'],rtol=1e-8,atol=1e-12)
        if step in CUTS[ratio]:assert cp['path_state']['friction_state']==baseline['checkpoint']['path_state']['friction_state']

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived nontrivial inputs AUTOMATIC_WORKFLOW_EVIDENCE R{ratio}-{6*ratio}.json absent: resolution replay requires the original staged evidence root')
def test_resolution():
    summary=[]
    for ratio in (2,4):
        data=json.loads((evidence()/f'R{ratio}-{6*ratio}.json').read_text());history=data['checkpoint']['path_state']['history'];fr=[x['friction'] for x in history]
        assert len(fr)==6*ratio
        assert any('slip' in x['branch'] for x in fr)
        assert all(b=='open' for b in fr[5*ratio-1]['branch'])
        assert all(b=='activation' for b in fr[5*ratio]['branch'])
        assert fr[5*ratio]['storage_j']==0 and fr[5*ratio]['dissipation_increment_j']==0
        assert any(a!=b for a,b in zip(fr[ratio-1]['target_elements'],fr[3*ratio-1]['target_elements']))
        D=sum(x['dissipation_increment_j'] for x in fr);assert D>1e-8
        absolute_material=sum(abs(x['material_relative_endpoint_contraction_j']) for x in fr);assert absolute_material>1e-8
        assert all(abs(x['return_algebra_error_j'])<=1e-12 for x in fr)
        assert all(abs(x['endpoint_work_identity_error_j'])<=1e-12 for x in fr)
        summary.append(dict(ratio=ratio,inventory=[x['step_index'] for x in history],D_j=D,
            release_j=sum(x['released_storage_j'] for x in fr),projection_excess_j=sum(x['return_projection_loss_j'] for x in fr),
            material_endpoint_j=sum(x['material_relative_endpoint_contraction_j'] for x in fr),absolute_material_endpoint_j=absolute_material,
            numerical_endpoint_j=sum(x['numerical_normal_variation_endpoint_contraction_j'] for x in fr),
            external_trapezoid_j=sum(x.get('external_nodal_work_trapezoid_j',0.) for x in history),
            control_trapezoid_j=sum(x.get('control_work_trapezoid_j',0.) for x in history),
            terminal_friction_state=data['checkpoint']['path_state']['friction_state'],phase_friction=[fr[i*ratio-1] for i in range(1,7)]))
    save(dict(resolutions=summary,relative_D_difference=abs(summary[1]['D_j']-summary[0]['D_j'])/summary[1]['D_j'],
        scope='two_discrete_equilibrated_resolutions_not_continuum_or_spatial_acceptance',
        work_scope='endpoint_contractions_and_trapezoids_not_exact_path_integrals'))


def test_spin():
    """Actual stored point maps under a common physical virtual rigid spin."""
    m,d,s,f,p,_=case();owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    u=np.zeros(60);z=np.array([.01/6,.01/3,.01/6,.01/3])
    for node in (2,3,4,5):u[m.mesh.dof_manager.get_node_dofs(node)[2]]=2e-5-2e-7
    owner.commit(owner.validate_successor(u,z))
    for node in range(1,6):u[m.mesh.dof_manager.get_node_dofs(node)[0]]=.0001
    force,history,records=owner._force(u,z);corners,n,frame,origin=owner._patch(u)
    omega=np.array([0.,2.,0.]);variation=np.zeros(60)
    for node in range(1,11):
        pt=m.mesh.get_node(node);dd=m.mesh.dof_manager.get_node_dofs(node)
        variation[dd[:3]]=np.cross(omega,np.array([pt.x,pt.y,pt.z])+u[dd[:3]])
        variation[dd[3:]]=omega
    local=variation[owner.work_columns];Dn=owner._normal_variation(corners,n,local=True)
    np.testing.assert_allclose(Dn@local,np.cross(omega,n),rtol=0,atol=1e-15)
    material=numerical=0.
    for j,(point,old,record) in enumerate(zip(owner.points,owner.history,records)):
        x=owner.point_reference[j]+owner.point_maps[j]@u[owner.work_columns]
        y=np.asarray(old.barycentric)@corners[old.target_element];r=x-y;g=n@r
        Jt=np.einsum('i,ijk->jk',old.barycentric,owner.target_maps[old.target_element])
        dx=owner.point_maps[j]@local;dy=Jt@local
        np.testing.assert_allclose(dx,np.cross(omega,x),rtol=0,atol=1e-15)
        np.testing.assert_allclose(dy,np.cross(omega,y),rtol=0,atol=1e-15)
        traction=np.array(record['traction_pa']);area=owner.weights[j]
        material+=area*traction@((np.eye(3)-np.outer(n,n))@(dx-dy))
        numerical-=area*g*traction@(Dn@local)
    save(dict(actual_pointmap_material_j=material,actual_normal_variation_j=numerical,
        full_friction_virtual_work_j=float(force@variation),scope='current_finitegap_common_virtual_spin_not_finitepose_admission'))
    assert abs(material)>1e-12 and abs(numerical)>1e-12
    assert abs(material+numerical)<=1e-12 and abs(force@variation)<=1e-12


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE'),
    reason='installed history stage input FRICTION_STAGE absent: staged nontrivial-work proof requires the original workflow stage selection')
def test_proof():
    from test_general_contact_s3_mesh_weak_pressure_friction import _weak_pose_gate
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.general_contact.s3_friction_history_path import expand_program
    proof=[]
    selected=(int(os.environ['FRICTION_STAGE'][-1]),) if os.environ['FRICTION_STAGE'] in ('proof2','proof4') else (2,4)
    for ratio in selected:
        c=case(ratio);m,d,s,f,p,_=c;expanded,_=expand_program(p,ratio)
        owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
        terminal=json.loads((evidence()/f'R{ratio}-{6*ratio}.json').read_text())
        history=terminal['checkpoint']['path_state']['history']
        snapshots=[x for cut in CUTS[ratio] for x in json.loads((evidence()/f'R{ratio}-{cut}.json').read_text())['snapshots']]
        assert [x['step_index'] for x in snapshots]==list(range(1,6*ratio+1))
        for snap,phase in zip(snapshots,history):
            step=snap['step_index'];u=np.array(snap['displacement']);z=np.array(phase['friction']['supported_force_n'])
            geometry=_weak_pose_gate(m,d,s,f,u,z);raw=linearize(owner,u,z,tangent=False)
            friction_force=owner._force(u,z)[0]
            full_endpoint=-float(friction_force@(u-owner.committed_displacement))
            reported=phase['friction']['material_relative_endpoint_contraction_j']+phase['friction']['numerical_normal_variation_endpoint_contraction_j']
            assert abs(full_endpoint-reported)<=1e-12
            structural,_,_=_assemble_nonlinear_system(m,u,{},5,tangent=False,require_full_coordinates=True)
            external=np.zeros(60)
            for load,value in zip(expanded.nodal_forces,expanded.force_values_at(step)):
                external[m.mesh.dof_manager.get_node_dofs(load.node_id)[('ux','uy','uz').index(load.dof)]]+=value
            residual=structural+raw.force-external;reaction=np.array(phase['reaction_generalized'])
            modes=[sum(residual[m.mesh.dof_manager.get_node_dofs(n)[a]] for n in range(1,6))-reaction[a] for a in (0,1)]
            modes += [residual[m.mesh.dof_manager.get_node_dofs(n)[2]] for n in (2,3,4,5)]
            for axis in (0,2):
                ru={n:residual[m.mesh.dof_manager.get_node_dofs(n)[axis]] for n in (7,8,9,10)}
                modes.extend((ru[7]-ru[9],ru[8]+ru[9],ru[10]+ru[9]))
            rows=[];moments=[];lever=[];virtual=0.;variation=np.cos(np.arange(60))*.001
            source,target=owner.normal._positions(u)
            for node,h,point in zip((2,3,4,5,7,8,9,10),(-.025,)*4+(.015,)*4,np.concatenate((source,target))):
                dd=m.mesh.dof_manager.get_node_dofs(node);F=raw.force[dd[:3]];M=raw.force[dd[3:]]
                rows.append(F);moments.append(np.cross(point,F));lever.extend(M-np.cross([0,0,h],F))
                virtual+=F@(variation[dd[:3]]+np.cross(variation[dd[3:]],[0,0,h]))
            metrics=dict(eq=float(np.linalg.norm(modes,1)/.02),wrench=float(np.linalg.norm(np.sum(rows,axis=0))/.02),moment=float(np.linalg.norm(np.sum(moments,axis=0))/.02),lever=float(np.linalg.norm(lever)/.02),virtual=float(abs(raw.force@variation-virtual)/max(abs(virtual),1e-12)),C=phase['complementarity_relative_force'])
            proof.append(dict(ratio=ratio,step=step,geometry=geometry,metrics=metrics,full_friction_endpoint_j=full_endpoint,full_endpoint_report_error_j=abs(full_endpoint-reported),friction=phase['friction']));save(dict(phases=proof))
            assert max(metrics[k] for k in ('eq','wrench','moment','lever'))<=1e-6
            assert metrics['virtual']<=1e-5 and metrics['C']<=1e-8
            assert abs(phase['friction']['return_algebra_error_j'])<=1e-12
            assert abs(phase['friction']['endpoint_work_identity_error_j'])<=1e-12
            owner.commit(owner.validate_successor(u,z))
    save(dict(phases=proof,Fref_n=.02,Mref_nm=.02,scope='complete_bothbody_operator_work_and_actual_accepted_binary_skin_gaps; no_path_integral'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived nontrivial inputs AUTOMATIC_WORKFLOW_EVIDENCE R2-4.json/R2-6.json absent: directional replay requires the original staged evidence root')
def test_directional():
    from anyfesolver.general_contact.semismooth import linearize
    m,d,s,f,p,_=case();owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    prior=json.loads((evidence()/'R2-4.json').read_text())['checkpoint'];owner.restore(prior['path_state']['friction_state'],np.array(prior['displacements']))
    saved=json.loads((evidence()/'R2-6.json').read_text());u=np.array(saved['snapshots'][0]['displacement']);z=np.array(saved['checkpoint']['path_state']['history'][4]['friction']['supported_force_n'])
    dq=np.zeros(60)
    for node,w in zip((2,3,4,5),(.3,.4,.2,.1)):dq[m.mesh.dof_manager.get_node_dofs(node)[2]]=w*.01
    for node,w in zip((7,8,9,10),(.1,.3,.4,.2)):dq[m.mesh.dof_manager.get_node_dofs(node)[0]]=w*.01
    dz=np.array([.2,.3,.4,.1])*1e-4;raw=linearize(owner,u,z);analytic=raw.tangent@dq+raw.force_pressure_gradient@dz;errors=[];probes=[]
    base=owner._force(u,z)[2];signature=lambda rows:[(r['branch'],r['target_element']) for r in rows]
    for h in (2e-5,1e-5):
        plus=linearize(owner,u+h*dq,z+h*dz,tangent=False);minus=linearize(owner,u-h*dq,z-h*dz,tangent=False)
        fd=(plus.force-minus.force)/(2*h)
        decisions=[signature(owner._force(u+sign*h*dq,z+sign*h*dz)[2]) for sign in (1,-1)]
        assert decisions[0]==decisions[1]==signature(base)
        assert np.all(plus.state.reference_support>0) and np.all(minus.state.reference_support>0)
        errors.append(float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd),1e-12)))
        probes.append(dict(h=h,max_q_perturbation_m=float(max(abs(h*dq))),max_z_perturbation_n=float(max(abs(h*dz))),directional_norm=float(np.linalg.norm(fd)),analytic_norm=float(np.linalg.norm(analytic)),branch_facet_decisions=decisions,positive_support_masks=True))
    save(dict(bothbody_qz_directional=errors,probes=probes,criterion=1e-5,committed_origin='R2-4',actual_pose=5,scope='bounded_smooth_directional_interval'))
    assert max(errors)<=1e-5


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived nontrivial inputs AUTOMATIC_WORKFLOW_EVIDENCE older-default-prefix.json/older-default-step3.json absent: mixed replay requires the original staged evidence root')
def test_mixed():
    incoming=json.loads((evidence()/'older-default-prefix.json').read_text())['checkpoint'];original=copy.deepcopy(incoming)
    assert all('return_work_reporting_schema' not in row['friction'] for row in incoming['path_state']['history'])
    c=build_case();explicit=build_case(2,'pressure_event')
    assert c[4].contract()==explicit[4].contract() and asdict(c[3])==asdict(explicit[3])
    result=solve_case(c,3,incoming);assert result.status=='completed';cp=result.to_restart_checkpoint()
    reporting=result.info['contact_path']['work_reporting']
    assert reporting['status']=='partial_schema_coverage'
    assert reporting['excluded_physical_steps']==[1,2] and reporting['covered_physical_steps']==[3]
    expected=json.loads((evidence()/'older-default-step3.json').read_text())
    np.testing.assert_allclose(result.displacements,expected['displacement'],rtol=0,atol=1e-10)
    assert cp['path_state']['friction_state']==expected['checkpoint']['path_state']['friction_state']
    assert incoming==original
    save(dict(status=result.status,reporting=reporting,authentic_old_default_checkpoint_no_reseal=True,default_program_unchanged=True,mechanics_match=True,checkpoint=cp))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived nontrivial input AUTOMATIC_WORKFLOW_EVIDENCE/R2-4.json absent: reporting rollback replay requires the original staged checkpoint')
def test_reporting(monkeypatch):
    import anyfesolver.general_contact.s3_friction_static as factory
    import anyfesolver.general_contact.s3_friction_history_path as module
    from anyfesolver.general_contact.runtime import _ACTIVE_CONTACT
    c=case();m,d,s,f,p,_=c
    incoming=json.loads((evidence()/'R2-4.json').read_text())['checkpoint'];original=copy.deepcopy(incoming)
    declared=copy.deepcopy((m.boundary_conditions,m.constraint_equations,m.load_cases));nodes=[(m.mesh.get_node(n).x,m.mesh.get_node(n).y,m.mesh.get_node(n).z) for n in range(1,11)]
    captured=[];reached=[];oldfactory=factory.create_s3_friction_manager
    def capture(*args):
        owner=oldfactory(*args);captured.append(owner);restore=owner.restore
        def restored(*a):
            v=restore(*a);owner.entry=copy.deepcopy(owner.checkpoint_state());owner.entry_normal=owner.normal.normal_multipliers.copy();return v
        owner.restore=restored;return owner
    monkeypatch.setattr(factory,'create_s3_friction_manager',capture)
    def broken(history):
        assert history[-1]['step_index']==5;reached.append('summary_after_accepted_commit');raise RuntimeError('injected work summary publication')
    monkeypatch.setattr(module,'summarize_work_history',broken);active=_ACTIVE_CONTACT.get()
    with pytest.raises(RuntimeError,match='injected work summary publication'):solve_case(c,5,incoming)
    assert reached==['summary_after_accepted_commit'] and len(captured)==1
    owner=captured[0];assert owner.checkpoint_state()==owner.entry
    np.testing.assert_array_equal(owner.normal.normal_multipliers,owner.entry_normal)
    assert incoming==original and _ACTIVE_CONTACT.get() is active
    assert (m.boundary_conditions,m.constraint_equations,m.load_cases)==declared
    assert [(m.mesh.get_node(n).x,m.mesh.get_node(n).y,m.mesh.get_node(n).z) for n in range(1,11)]==nodes
    save(dict(fault_reached=reached,actual_public_summary_failure_rollback=True,declared_inputs_checkpoint_context_unchanged=True))


def test_safety():
    c=case()
    with pytest.raises(ValueError):solve_case(c,0)
    save(dict(explicit_zero_refused=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not all((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(name+'.json')).is_file() for name in ('R2-4','R2-6','proof2')),
    reason='archived nontrivial inputs AUTOMATIC_WORKFLOW_EVIDENCE R2-4.json/R2-6.json/proof2.json absent: safety replay requires the original staged evidence root')
def test_archive_checkpoint_safety():
    from dataclasses import replace
    c=case();incoming=json.loads((evidence()/'R2-4.json').read_text())['checkpoint'];original=copy.deepcopy(incoming)
    changed=(*c[:3],replace(c[3],physical_subincrements_per_user_increment=4),*c[4:])
    with pytest.raises(ValueError):solve_case(changed,8,incoming)
    assert incoming==original
    geometry=json.loads((evidence()/'proof2.json').read_text())['phases']
    normals=np.array([row['geometry']['target_raw_normal'] for row in geometry]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    normal_delta=float(max(np.linalg.norm(row-normals[0]) for row in normals));assert normal_delta>0
    slid=np.array(json.loads((evidence()/'R2-6.json').read_text())['displacement'])
    target_ux=[float(slid[c[0].mesh.dof_manager.get_node_dofs(n)[0]]) for n in (7,8,9,10)];assert max(abs(v) for v in target_ux)>0
    save(dict(changed_ratio_refused=True,checkpoint_unchanged=True,actual_target_normal_delta_from_first_accepted=normal_delta,target_ux_m=target_ux))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived nontrivial inputs AUTOMATIC_WORKFLOW_EVIDENCE R{ratio}-{cut}.json absent: work-consumer replay requires the original staged evidence root')
def test_workconsumer():
    """Actual accepted structural/external/control quadrature, not an energy oracle."""
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    reports=[]
    for ratio in (2,4):
        m,*_=case(ratio);terminal=json.loads((evidence()/f'R{ratio}-{6*ratio}.json').read_text());history=terminal['checkpoint']['path_state']['history']
        snapshots=[x for cut in CUTS[ratio] for x in json.loads((evidence()/f'R{ratio}-{cut}.json').read_text())['snapshots']]
        previous=np.zeros(60);previous_force=np.zeros(60);rows=[]
        for snap,phase in zip(snapshots,history):
            u=np.array(snap['displacement']);force,_,_=_assemble_nonlinear_system(m,u,{},5,tangent=False,require_full_coordinates=True)
            du=u-previous;internal=float(.5*(force+previous_force)@du)
            rows.append(dict(step=snap['step_index'],structural_internal_trapezoid_j=internal,
                external_nodal_trapezoid_j=phase['external_nodal_work_trapezoid_j'],control_trapezoid_j=phase['control_work_trapezoid_j'],
                fixed_support_work_j=0.,friction_endpoint_contraction_j=phase['friction']['material_relative_endpoint_contraction_j']+phase['friction']['numerical_normal_variation_endpoint_contraction_j']))
            previous=u;previous_force=force
        reports.append(dict(ratio=ratio,rows=rows,totals={key:sum(row[key] for row in rows) for key in rows[0] if key!='step'}))
    save(dict(resolutions=reports,scope='accepted_subincrement_operator_quadrature_not_independent_nonlinear_energy_or_exact_variablepressure_path_work'))
