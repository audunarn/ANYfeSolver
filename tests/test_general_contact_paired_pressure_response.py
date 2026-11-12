"""Paired spin/chart diagnostics and ONE directional spatial refinement."""
import hashlib, importlib.util, json, math, os
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from anyfesolver import FEModel, BoundaryCondition, create_shell_element
from anyfesolver.general_contact import ContactSurface, ContactInteraction, resolve_automatic_contact_domain
from anyfesolver.general_contact.static_path import ContactPathControl, ContactPathProgram, ContactPathNodalForce
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize, complementarity, reference_nodal_support
from anyfesolver.general_contact.pressure_diagnostics import PressureResponseScales, paired_pressure_response, finite_pressure_pose
from examples.general_contact_tilted_relative_sliding import build as base_build, solve, rotation, rigid_pose

ROOT=Path(__file__).parents[1]
BASE=Path('C:/ANYworkspaceWork/contact_tilted_relative_sliding_20261008/outputs/v3/run-14.json')
BASE_SHA='a9a7cdefc74556964d4600009f42493dfefcc2eaced21e360d44fa8df66ccb29'
SCALES=PressureResponseScales(1e-4,1e-4,1.,1.)
spec=importlib.util.spec_from_file_location('_pressure_finite_proof',ROOT/'tests/test_general_contact_finite_rotation_shell_work.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
save,read,sha=prior.save,prior.read,prior.sha


def fixture(refined=False):
    if not refined:
        result=base_build();return (*result,dict(source_count=6,source_x=(.1,.6,1.1),target_start=7,yaw=9,source_faces=((1,2,5,4),(2,3,6,5)),target_faces=((7,8,11,10),(8,9,12,11),(10,11,14,13),(11,12,15,14))))
    _,domain,settings,_,_=base_build()
    model=FEModel('one-source-X-refinement-paired-pressure');model.add_material('elastic',1e5,.3,density=1.)
    xs=(.1,.35,.6,.85,1.1);source=tuple((x,y,.04+2e-7) for y in (.1,1.1) for x in xs)
    target=tuple((x,y,-.01) for y in (0.,.5,1.) for x in (0.,.5,1.))
    for n,p in enumerate((*source,*target),1):model.add_node(n,*p)
    sf=tuple((i,i+1,i+6,i+5) for i in range(1,5));tf=((11,12,15,14),(12,13,16,15),(14,15,18,17),(15,16,19,18))
    for eid,ids in enumerate((*sf,*tf),1):model.add_element(eid,create_shell_element(eid,ids,'elastic',thickness=.05,reference_normal=(0.,0.,1.)))
    model.add_boundary_condition(BoundaryCondition('target-yaw-anchor',(13,),{'uy':0.}))
    participants=tuple(replace(p,participant=replace(p.participant,element_ids=tuple(range(1,5)) if p.participant.name=='q4_source' else tuple(range(5,9)))) for p in domain.participants)
    domain=replace(domain,name='paired-pressure-source-X-refinement',participants=participants)
    settings=replace(settings,maximum_candidate_pairs=16)
    # Explicit supported public orientation: no renamed IDs or area-tie forcing.
    pair=ContactInteraction('declared-pressure-source','q4_source','q4_target',1000.,5e-5,'augmented')
    res=resolve_automatic_contact_domain(model,domain,settings,explicit_pairs=(pair,),analysis_control='contact_path')
    assert res.definition.interactions[0].surface=='q4_source'
    terms=[(n,name) for n in (1,6,11) for name in ('ux','uy','uz','rx','ry','rz')]+[(n,'uz') for n in (13,17,19)]
    controls=tuple(ContactPathControl.node(n,name,coordinate_profile='finite_shell_spin_chart_dev') for n,name in terms)
    names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5);waypoints=[]
    for step,angle in enumerate([math.pi*i/36 for i in range(13)]+[math.pi/3]*4):
        q=rigid_pose(model,angle)
        if step>=14:
            for n in range(1,11):q[model.mesh.dof_manager.get_node_dofs(n)[1]]+=.02
        waypoints.append(tuple(float(q[model.mesh.dof_manager.get_node_dofs(n)[names[name]]]) for n,name in terms))
    axis=rotation(math.pi/3)@np.array((0.,0.,1.));loads=(0.,)*13+(-5e-5,-5e-5,5e-5,-5e-5)
    forces=tuple(ContactPathNodalForce(n,name,tuple(v*axis[i] for v in loads)) for n in (5,10) for i,name in enumerate(('ux','uy','uz')) if axis[i]!=0.)
    program=ContactPathProgram(controls,tuple(waypoints),1,forces,control_profile='finite_shell_triplets_dev')
    return model,domain,settings,res,program,dict(source_count=10,source_x=xs,target_start=11,yaw=13,source_faces=sf,target_faces=tf)


def selectors(model,program):
    names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5);fixed=set();rows=np.zeros((len(program.controls),model.mesh.dof_manager.total_dofs))
    for bc in model.boundary_conditions:
        for n in bc.node_ids:
            for name in bc.dof_constraints:fixed.add(model.mesh.dof_manager.get_node_dofs(n)[names[name]])
    for i,c in enumerate(program.controls):
        assert len(c.terms)==1 and c.terms[0][2]==1.
        n,name,w=c.terms[0];d=model.mesh.dof_manager.get_node_dofs(n)[names[name]];fixed.add(d);rows[i,d]=w
    return np.array(sorted(set(range(rows.shape[1]))-fixed)),rows


def bins(model,meta,z,s):
    # FULL reference trace, intentionally not overlap traction/resultant.
    pressure=[F(float(v)) for v in np.asarray(z)/s];out=[]
    for left,right in ((.1,.6),(.6,1.1)):
        integral=F(0);area=F(0)
        for face in meta['source_faces']:
            x0=F(float(model.mesh.get_node(face[0]).x));x1=F(float(model.mesh.get_node(face[1]).x));a=max(x0,F(left));b=min(x1,F(right))
            if b<=a:continue
            y0=F(float(model.mesh.get_node(face[0]).y));y1=F(float(model.mesh.get_node(face[3]).y));height=y1-y0
            ix0=(x1*(b-a)-(b*b-a*a)/2)/(x1-x0);ix1=((b*b-a*a)/2-x0*(b-a))/(x1-x0)
            for node,w in zip(face,(ix0,ix1,ix1,ix0)):integral+=pressure[node-1]*w*height/2
            area+=(b-a)*height
        out.append(dict(material_x_bin=(left,right),area_m2=float(area),full_reference_trace_integral_n=float(integral),mean_numerical_pressure_pa=float(integral/area),exact_binary_fraction=str(integral)))
    return out


def bind_response(report,k,g,b):
    alpha=report['force_detection']['weakest_right_vector'];dz=report['pressure_force_scale']*alpha;s=report['support'];lam=dz/s
    norm=float(np.sqrt(np.sum(s*lam*lam))*np.sqrt(SCALES.reference_area_m2)/SCALES.force_n)
    assert abs(norm-1.)<=1e-12
    report.update(scales_si=dict(force_n=1e-4,moment_nm=1e-4,length_m=1.,reference_area_m2=1.,angle_rad=1.),
        frozen_norm_at_pose=True,weakest_signed_pressure_probe=dict(alpha=alpha,delta_supported_force_n=dz,delta_pressure_pa=lam,dimensionless_lumped_pressure_norm=norm,unconstrained_spatial_residual_load=b.T@dz,chart_displacement=report['displacement_per_force']@dz,weighted_gap_change_m=report['gap_per_force']@dz),
        matrix_inputs=dict(constrained_tangent=k.toarray(),gap_chart=g,force_spin=b))


def physical_gates(model,settings,res,program,meta,pose):
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    manager=GeneralS3PatchManager(model,res.definition,settings);free,rows=selectors(model,program)
    q=np.asarray(pose['u']);z=np.asarray(pose['z']);manager.require_finite_deformation(q);skin=manager.finite_skin.pose(q)
    independent=np.array([manager.finite_skin.reference[i]+q[d[:3]]+manager.finite_skin.offsets[i]*prior.decimal_director(q[d[3:]],manager.finite_skin.directors[i]) for i,d in enumerate(manager.finite_skin.dofs)])
    assert np.max(abs(independent-skin.positions))<=1e-13
    spec=importlib.util.spec_from_file_location('_pressure_gap_oracle',ROOT/'docs/d1_review_support/contact_finite_rotation_shell_work_20261008/oracle.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    oracle.SOURCE=meta['source_faces'];oracle.TARGET=meta['target_faces']
    axis=rotation(min(pose['step'],12)*math.pi/36)@np.array((0.,0.,1.));gap=oracle.posed_bounds(skin.positions,axis)
    assert F(gap['lower_fraction'])>=-F(float(5e-5))
    raw=linearize(manager,q,z,tangent=False);ncp,_,_,_=complementarity(raw.state,z,1000*reference_nodal_support(manager.law),raw.gap_gradient,tangent=False)
    r,_,trial=_assemble_nonlinear_system(model,q,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True);_discard_nonlinear_state_candidate(trial)
    ext=np.zeros(len(q));names=dict(ux=0,uy=1,uz=2)
    for f,v in zip(program.nodal_forces,program.force_values_at(pose['step'])):ext[model.mesh.dof_manager.get_node_dofs(f.node_id)[names[f.dof]]]+=v
    fixed=set()
    for bc in model.boundary_conditions:
        for n in bc.node_ids:
            for name in bc.dof_constraints:fixed.add(model.mesh.dof_manager.get_node_dofs(n)[dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5)[name]])
    staticfree=np.array(sorted(set(range(len(q)))-fixed))
    applied=ext+rows.T@np.asarray(pose['record']['reaction_generalized']);eq=float(np.linalg.norm((r+raw.force-applied)[staticfree],1)/1e-4)
    # Tolerance 1e-6 -> 6e-6 (owner waiver, Q4 single elastic law 2026-10-23): the
    # SHA-pinned run-14 pose was solved under the pre-single-law Q4 response; the
    # law's von Karman cross terms leave it with eq=3.92e-6 (|res|2 1.4e-10 N =
    # 4.5e-7 of the structural force) vs 5.9e-8 on 7455222f9. 6e-6 ~ 1.5x measured.
    assert max(1.,np.linalg.norm(applied[staticfree]))==1. and eq<=6e-6
    assert np.linalg.norm(ncp,1)/1e-4<=1e-8 and min(z)>=0 and min(raw.state.reference_support)>0
    physical=-raw.local_force.reshape(-1,3);pf=np.linalg.norm(physical.sum(axis=0),1)/1e-4;pm=np.linalg.norm(np.cross(skin.positions,physical).sum(axis=0),1)/1e-4
    mids=manager.finite_skin.reference+q[manager.finite_skin.dofs[:,:3]];nodal=raw.force[manager.finite_skin.dofs];sm=np.linalg.norm(np.cross(mids,-nodal[:,:3]).sum(axis=0)-nodal[:,3:].sum(axis=0),1)/1e-4
    assert max(pf,pm,sm)<=1e-6
    if pose['step']<=12:assert gap['strict_positive'] and np.linalg.norm(z,1)/1e-4<=1e-8
    rigid=rigid_pose(model,min(pose['step'],12)*math.pi/36)
    if pose['step']>=14:
        for n in range(1,meta['source_count']+1):rigid[model.mesh.dof_manager.get_node_dofs(n)[1]]+=.02
    bend=q-rigid
    if pose['step']>12:
        assert max(z)>0
        for ds in (manager.finite_skin.dofs[:meta['source_count']],manager.finite_skin.dofs[meta['source_count']:]):
            assert np.max(abs(bend[ds[:,:3]]))>1e-10 and np.max(abs(bend[ds[:,3:]]))>1e-10
    count=meta['source_count'];body_force=physical[:count].sum(axis=0);body_moment=np.cross(skin.positions[:count],physical[:count]).sum(axis=0)
    return dict(step=pose['step'],equilibrium_l1_relative=eq,complementarity_l1_relative=float(np.linalg.norm(ncp,1)/1e-4),force_relative=float(pf),physical_moment_relative=float(pm),spatial_moment_relative=float(sm),gap=gap,source_force_n=body_force,source_moment_nm=body_moment,pressure_support=raw.state.reference_support,normal_forces=z,full_reference_bins=bins(model,meta,z,raw.state.reference_support),free_bending=bend,positions=skin.positions)


def test_manufactured_paired_and_rectangular_nullspace():
    k=np.array(((2.,1.),(0.,3.)));g=np.array(((1.,1.),(3.,5.)));b=np.array(((1.,0.),(1.,2.)))
    r=paired_pressure_response(k,g,b,[1.,4.],[False,True],scales=PressureResponseScales(2.,3.,4.,5.))
    np.testing.assert_allclose(r['displacement_per_force'],((.5,1/6),(0.,2/3)),rtol=1e-15,atol=1e-15)
    np.testing.assert_allclose(r['gap_per_force'],((.5,5/6),(1.5,23/6)),rtol=1e-15,atol=1e-15)
    assert r['gap_response_relative_asymmetry']>0 and r['dimensionless_solve_backward_error']<1e-14
    null=paired_pressure_response([[2.]],[[1.],[2.]],[[1.],[2.]],[1.,1.],[False],scales=SCALES)
    detect=null['force_detection'];assert detect['domain_dimension']==2 and detect['numerically_resolved_nullity']==1 and math.isinf(detect['condition_number'])
    assert np.linalg.norm(np.array([[1.,2.]])@detect['weakest_right_vector'])<1e-14
    with pytest.raises(ValueError):paired_pressure_response(k,g,b,[1.,0.],[False,True],scales=SCALES)
    save(dict(verified=True,independent_nonsymmetric_fraction_solution=True,rectangular_pressure_nullspace_retained=True,no_support_dropped=True))


@pytest.mark.skipif(not BASE.is_file(),
    reason='archived BASE pose run-14.json absent (C:/ANYworkspaceWork/contact_tilted_relative_sliding_20261008/outputs/v3/run-14.json): authenticated sixty-degree paired response requires the saved sha-pinned BASE pose')
def test_authenticated_base_sixty_degree_paired_response():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
    stored=json.loads(BASE.read_text());model,domain,settings,res,program,meta=fixture();assert res.identity==stored['identity']
    pose=stored['poses'][-1];assert pose['step']==14 and pose['u']==stored['checkpoint']['displacements'] and pose['z']==stored['checkpoint']['path_state']['normal_forces']
    manager=GeneralS3PatchManager(model,res.definition,settings);free,_=selectors(model,program);q=np.asarray(pose['u']);z=np.asarray(pose['z'])
    report,raw,ks,rs,k,g,b=finite_pressure_pose(model,manager,q,z,free,program=program,scales=SCALES)
    bind_response(report,k,g,b)
    rng=np.random.default_rng(6014);du=np.zeros(len(q));du[free]=rng.normal(size=len(free));du/=np.linalg.norm(du);dz=rng.normal(size=len(z))*1e-5
    c=1000*reference_nodal_support(manager.law);_,active,cq,cz=complementarity(raw.state,z,c,raw.gap_gradient)
    errors=[]
    for h in (1e-6,3e-7):
        plus=linearize(manager,q+h*du,z+h*dz,tangent=False);minus=linearize(manager,q-h*du,z-h*dz,tangent=False)
        cp,ap,_,_=complementarity(plus.state,z+h*dz,c,plus.gap_gradient,tangent=False);cn,an,_,_=complementarity(minus.state,z-h*dz,c,minus.gap_gradient,tangent=False)
        assert np.array_equal(active,ap) and np.array_equal(active,an)
        def structural(at):
            r,_,trial=_assemble_nonlinear_system(model,at,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent');_discard_nonlinear_state_candidate(trial);return r
        expected=raw.tangent@du-raw.force_gradient.T@dz;df=(plus.force-minus.force)/(2*h)
        full=(structural(q+h*du)+plus.force-structural(q-h*du)-minus.force)/(2*h);expected_full=(ks+raw.tangent)@du-raw.force_gradient.T@dz
        gap=(np.asarray(plus.state.normalized_gap)-minus.state.normalized_gap)/(2*h);expected_gap=raw.gap_gradient@du
        row=dict(h=h,contact=float(np.linalg.norm(df-expected)/max(np.linalg.norm(expected),1e-12)),coupled=float(np.linalg.norm(full-expected_full)/max(np.linalg.norm(expected_full),1e-12)),chart_gap=float(np.linalg.norm(gap-expected_gap)/max(np.linalg.norm(expected_gap),1e-12)),complementarity=float(np.linalg.norm((cp-cn)/(2*h)-(cq@du+cz*dz))/max(np.linalg.norm(cq@du+cz*dz),1e-12)))
        assert max(row[n] for n in ('contact','coupled','chart_gap','complementarity'))<=1e-5;errors.append(row)
    # Independent physical-spin point work, using exponential series and L action.
    pointwork=0.;nodalwork=0.
    for i,d in enumerate(manager.finite_skin.dofs):
        theta=q[d[3:]];term=du[d[3:]].copy();omega=term.copy()
        for order in range(1,40):term=np.cross(theta,term)/(order+1);omega+=term
        director=prior.decimal_director(theta,manager.finite_skin.directors[i]);delta=du[d[:3]]+manager.finite_skin.offsets[i]*np.cross(omega,director)
        pointwork+=raw.local_force.reshape(-1,3)[i]@delta;nodalwork+=raw.force[d[:3]]@du[d[:3]]+raw.force[d[3:]]@omega
    workerror=abs(pointwork-nodalwork)/max(abs(pointwork),1e-10);assert workerror<=1e-12
    bad=free[free!=manager.finite_skin.dofs[2,3]]
    with pytest.raises(ValueError,match='angular triplets'):finite_pressure_pose(model,manager,q,z,bad,program=program,scales=SCALES)
    gates=physical_gates(model,settings,res,program,meta,pose)
    save(dict(verified=True,saved_pose_sha256=BASE_SHA,saved_pose_is_fixture_not_restart=True,diagnostic=report,sixty_degree_directional_errors=errors,independent_physical_spin_work_relative_error=workerror,physical=gates,identity=res.identity))


@pytest.mark.skipif(not os.environ.get('PRESSURE_STAGE'),
    reason='explicit PRESSURE_STAGE producer stage absent: refined public path requires the staged refinement input')
def test_refined_public_path():
    step=int(os.environ['PRESSURE_STAGE'].split('-')[1]);previous={4:None,8:4,12:8,13:12,14:13}[step]
    model,domain,settings,res,program,meta=fixture(True);kwargs=dict(max_load_factor=step/16.)
    if previous:kwargs['restart_checkpoint']=read('run-'+str(previous))['checkpoint']
    result=solve(model,settings,res,program,**kwargs);history={r['step_index']:r for r in result.info.get('force_displacement_history',[])};poses=[]
    for snap in result.snapshots:
        row=history[snap.step_index];contact=row['contact'];z=np.asarray(contact['numerical_pressure_pa'])*contact['reference_support_m2']
        if snap.step_index==result.restart_checkpoint['path_state']['accepted_step']:z=np.asarray(result.restart_checkpoint['path_state']['normal_forces'])
        poses.append(dict(step=snap.step_index,u=snap.displacements,z=z,record=row))
    save(dict(status=result.status,failure_reason=result.failure_reason,identity=res.identity,checkpoint=result.restart_checkpoint,info=result.info,poses=poses,metadata=meta,snapshot_inventory=[p['step'] for p in poses],previous_sha256=sha('run-'+str(previous)) if previous else None,explicit_integration_side='q4_source'))
    assert result.status=='completed',result.failure_reason
    assert [p['step'] for p in poses]==list(range((previous or 0)+1,step+1))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not all(
    (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(stage+'.json')).is_file()
    for stage in ('run-4','run-8','run-12','run-13','run-14','base')),
    reason='archived staged runs run-4/8/12/13/14.json and base.json under AUTOMATIC_WORKFLOW_EVIDENCE absent: saved refinement physical response requires the staged matrices')
def test_saved_refinement_physical_response_and_modes():
    stages=['run-'+str(i) for i in (4,8,12,13,14)];chunks=[read(s) for s in stages]
    assert all(c['status']=='completed' for c in chunks) and all(c['previous_sha256']==sha(stages[i-1]) for i,c in enumerate(chunks) if i)
    poses=[p for chunk in chunks for p in chunk['poses']];assert [p['step'] for p in poses]==list(range(1,15))
    model,domain,settings,res,program,meta=fixture(True);assert all(c['identity']==res.identity for c in chunks)
    gates=[physical_gates(model,settings,res,program,meta,p) for p in poses]
    manager=GeneralS3PatchManager(model,res.definition,settings);free,_=selectors(model,program);q=np.asarray(poses[-1]['u']);z=np.asarray(poses[-1]['z'])
    report,_,_,_,k,g,b=finite_pressure_pose(model,manager,q,z,free,program=program,scales=SCALES)
    bind_response(report,k,g,b)
    base=read('base');original=json.loads(BASE.read_text())['poses'][-1]
    common_source=[(1,1),(2,3),(3,5),(4,6),(5,8),(6,10)];common_target=[(n,n+4) for n in range(7,16)];axis=rotation(math.pi/3)@np.array((0.,0.,1.))
    old_model=fixture()[0];differences=[]
    for a,bnode in (*common_source,*common_target):
        oldd=old_model.mesh.dof_manager.get_node_dofs(a);newd=model.mesh.dof_manager.get_node_dofs(bnode)
        differences.append(dict(base_node=a,refined_node=bnode,normal_displacement_difference_m=float(axis@(q[newd[:3]]-np.asarray(original['u'])[oldd[:3]]))))
    save(dict(verified=True,source_sha256={s:sha(s) for s in stages},base_diagnostic_sha256=sha('base'),snapshot_inventory=list(range(1,15)),refined_diagnostic=report,refined_physical=gates,common_material_node_response_differences=differences,bin_pressure_trace_differences_pa=[r['mean_numerical_pressure_pa']-b['mean_numerical_pressure_pa'] for r,b in zip(gates[-1]['full_reference_bins'],base['physical']['full_reference_bins'])],source_resultant_difference_n=np.array(gates[-1]['source_force_n'])-base['physical']['source_force_n'],source_moment_difference_nm=np.array(gates[-1]['source_moment_nm'])-base['physical']['source_moment_nm'],space_change='source structural and pressure X spaces both refine; target fixed',spatial_convergence_gate=False,bin_physical_traction_gate_unresolved=True,mesh_uniform_stability_not_claimed=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE')
    or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'base.json').is_file()
    or not BASE.is_file()
    or not Path('C:/ANYworkspaceWork/contact_tilted_relative_sliding_20261008/current-install.json').is_file(),
    reason='archived base.json diagnostic under AUTOMATIC_WORKFLOW_EVIDENCE, BASE run-14.json, or current-install.json absent: installed saved-matrix consumer requires the saved matrices and install manifest')
def test_installed_saved_matrix_consumer_and_refusals(monkeypatch):
    import anyfesolver
    import anyfesolver.nonlinear_static as structural_owner
    saved=read('base')['diagnostic'];data=saved['matrix_inputs'];moment=np.array(saved['free_dofs'])%6>=3
    current=paired_pressure_response(data['constrained_tangent'],data['gap_chart'],data['force_spin'],saved['support'],moment,scales=SCALES,active_modes=np.isin(np.arange(len(saved['support'])),saved['active_mode_indices']))
    for key in ('force_detection','gap_response','active_gap_response'):
        np.testing.assert_array_equal(current[key]['singular_values'],saved[key]['singular_values'])
    site=Path(anyfesolver.__file__).resolve().parent.parent;manifest=json.loads((site.parent/'install.json').read_text())
    old=json.loads(Path('C:/ANYworkspaceWork/contact_tilted_relative_sliding_20261008/current-install.json').read_text())
    assert all(manifest['payload_sha256'][name]==digest for name,digest in old['payload_sha256'].items())
    model,_,settings,res,program,_=fixture();manager=GeneralS3PatchManager(model,res.definition,settings);free,_=selectors(model,program);pose=json.loads(BASE.read_text())['poses'][-1]
    calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('structural trial before diagnostic refusal')
    monkeypatch.setattr(structural_owner,'_assemble_nonlinear_system',forbidden)
    with monkeypatch.context() as context:
        context.setattr(manager,'profile','not-finite')
        with pytest.raises(ValueError,match='exact finite Q4 profile'):finite_pressure_pose(model,manager,pose['u'],pose['z'],free,program=program,scales=SCALES)
    with pytest.raises(ValueError,match='literal static and path'):finite_pressure_pose(model,manager,pose['u'],pose['z'],free[1:],program=program,scales=SCALES)
    model.add_constraint_equation(terms=((int(free[0]),1.),),source_id='diagnostic-negative-MPC',dependent_dof=int(free[0]))
    with pytest.raises(ValueError,match='not general MPC'):finite_pressure_pose(model,manager,pose['u'],pose['z'],free,program=program,scales=SCALES)
    assert not calls
    with pytest.raises(ValueError):PressureResponseScales(0.,1.,1.,1.)
    with pytest.raises(ValueError):paired_pressure_response([[1.]],[[1.]],[[float('nan')]],[1.],[False],scales=SCALES)
    save(dict(verified=True,current_installed_saved_matrix_consumer=True,common_mechanical_payloads_identical=len(old['payload_sha256']),wrong_profile_before_material=True,inexact_selector_before_material=True,MPC_before_material=True,invalid_scale_and_nonfinite_refusal=True,saved_base_sha256=sha('base'),no_mechanical_solver_replay=True))


def test_pressure_response_invalid_scale_and_nonfinite_refusals():
    # Input-free refusals mirrored from the guarded installed-consumer node
    # so they stay active without the saved matrices and install manifest.
    with pytest.raises(ValueError):PressureResponseScales(0.,1.,1.,1.)
    with pytest.raises(ValueError):paired_pressure_response([[1.]],[[1.]],[[float('nan')]],[1.],[False],scales=SCALES)
