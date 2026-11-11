"""Independent map/work, full q/z derivatives and actual finite static path."""
import copy,hashlib,json,math,os
from dataclasses import replace
from decimal import Decimal,localcontext
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_finite_rotation_shell_work import build,solve,rigid_pose
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support

FREF=1e-4;MREF=1e-4
def save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else x),encoding='utf-8')
def read(stage):return json.loads((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(stage+'.json')).read_text())
def sha(stage):return hashlib.sha256((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(stage+'.json')).read_bytes()).hexdigest()

def decimal_director(theta,dref):
    """Independent 80-digit vector exponential, no production SO3/map kernels."""
    with localcontext() as ctx:
        ctx.prec=80
        a=[Decimal(float(x)) for x in theta];term=[Decimal(float(x)) for x in dref];total=term.copy()
        for n in range(1,100):
            x,y,z=term;ax,ay,az=a
            term=[(ay*z-az*y)/n,(az*x-ax*z)/n,(ax*y-ay*x)/n]
            total=[v+w for v,w in zip(total,term)]
        return np.array([float(v) for v in total])


def test_director_work_and_coupled_chart_jacobian():
    from anyfesolver.corotational import rotation_matrix_from_vector,_rotation_right_jacobian
    from anyfesolver.general_contact.finite_rotation_skin import director_jets,director_value_error
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings)
    q=rigid_pose(model,math.pi/6.);normal=np.array((.5,0.,math.sqrt(3.)/2))
    for n in range(1,7):q[model.mesh.dof_manager.get_node_dofs(n)[:3]]-=2e-6*normal
    manager.require_finite_deformation(q);skin=manager.finite_skin;pose=skin.pose(q)
    expected=[];native=[]
    for theta,dref in zip(q[skin.dofs[:,3:]],skin.directors):
        expected.append(decimal_director(theta,dref));native.append(rotation_matrix_from_vector(theta)@dref)
    error=np.max(np.abs(pose.directors-np.array(expected)),axis=1)
    enclosure=director_value_error(q[skin.dofs[:,3:]],skin.directors)
    assert np.all(error<=enclosure)
    assert np.max(np.abs(pose.directors-native))<=1e-13
    zero,grad,hess=director_jets(np.zeros((1,3)),np.array(((0.,0.,1.),)))
    assert hess[0,2,0,0]==-1. and hess[0,2,1,1]==-1.
    # Native tiny-angle I+A differs from the adopted Exp field only by its
    # quadratic remainder; the contact map still retains nonzero curvature.
    tiny=np.array((2e-13,-3e-13,1e-13));d0=np.array((0.,0.,1.))
    td,tg,th=director_jets(tiny[None,:],d0[None,:])
    assert np.linalg.norm(td[0]-rotation_matrix_from_vector(tiny)@d0)<=np.linalg.norm(tiny)**2
    rng=np.random.default_rng(71);f=rng.normal(size=pose.positions.shape)
    pulled=np.asarray(pose.spin_map.T@f.ravel()).ravel();independent=np.zeros(manager.n_dof)
    delta=rng.normal(size=manager.n_dof);virtual=delta.copy()
    for i,dofs in enumerate(skin.dofs):
        independent[dofs[:3]]+=f[i]
        independent[dofs[3:]]+=skin.offsets[i]*np.cross(expected[i],f[i])
        virtual[dofs[3:]]=_rotation_right_jacobian(-q[dofs[3:]])@delta[dofs[3:]]
    np.testing.assert_allclose(pulled,independent,rtol=0,atol=1e-14)
    work_error=abs(float(pulled@virtual-f.ravel()@(pose.coordinate_map@delta)))
    assert work_error<=1e-13
    # Right multiplication by an axial reference drill leaves material skins.
    from anyfesolver.corotational import rotation_vector_from_matrix
    drill=q.copy()
    for i,dofs in enumerate(skin.dofs):
        qi=rotation_matrix_from_vector(q[dofs[3:]])
        drill[dofs[3:]]=rotation_vector_from_matrix(qi@rotation_matrix_from_vector(.12*skin.directors[i]))
    assert np.max(np.abs(skin.pose(drill).positions-pose.positions))<=1e-13
    z=np.full(6,1e-5);c=1000*reference_nodal_support(manager.law)
    raw=linearize(manager,q,z);_,active,cq,cz=complementarity(raw.state,z,c,raw.gap_gradient)
    du=rng.normal(size=manager.n_dof);du/=np.linalg.norm(du);dz=rng.normal(size=6)*1e-5
    def structural(at,tangent):
        r,k,st=_assemble_nonlinear_system(model,at,{},5,tangent=tangent,kinematics='corotational',corotational_tangent='consistent')
        _discard_nonlinear_state_candidate(st);return r,k
    rs,ks=structural(q,True);errors=[]
    for h in (1e-6,3e-7):
        p=linearize(manager,q+h*du,z+h*dz,tangent=False);n=linearize(manager,q-h*du,z-h*dz,tangent=False)
        cp,ap,_,_=complementarity(p.state,z+h*dz,c,p.gap_gradient,tangent=False)
        cn,an,_,_=complementarity(n.state,z-h*dz,c,n.gap_gradient,tangent=False)
        assert np.array_equal(active,ap) and np.array_equal(active,an)
        derivative=raw.tangent@du-raw.force_gradient.T@dz
        df=(p.force-n.force)/(2*h)
        full=(structural(q+h*du,False)[0]+p.force-structural(q-h*du,False)[0]-n.force)/(2*h)
        expected_full=(ks+raw.tangent)@du-raw.force_gradient.T@dz
        dc=(cp-cn)/(2*h);expected_c=cq@du+cz*dz
        row=dict(h=h,contact=np.linalg.norm(df-derivative)/max(np.linalg.norm(derivative),1e-12),coupled=np.linalg.norm(full-expected_full)/max(np.linalg.norm(expected_full),1e-12),complementarity=np.linalg.norm(dc-expected_c)/max(np.linalg.norm(expected_c),1e-12))
        assert max(row[k] for k in ('contact','coupled','complementarity'))<=1e-5;errors.append(row)
    save(dict(verified=True,director_error_max=float(max(error)),director_error_upper_max=float(max(enclosure)),physical_spin_work_error=work_error,directional_errors=errors,local_deformation_maxima=manager.last_finite_deformation_maxima,nonzero_zero_pose_second_variation=True,right_multiplicative_drill_invariant=True,pressure_ids=list(manager.law.pressure_mode_ids)))


def test_nonaffine_path_curvature_and_unsafe_chord():
    from anyfesolver.general_contact.finite_rotation_skin import curvature_radius,director_norm_upper
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings)
    a=np.zeros(manager.n_dof);safe=rigid_pose(model,math.pi/36.);unsafe=rigid_pose(model,math.pi/3.)
    skin=manager.finite_skin
    radius=curvature_radius(a[skin.dofs[:,3:]],safe[skin.dofs[:,3:]],skin.offsets,skin.directors)
    assert np.all(director_norm_upper(skin.directors)>=1.)
    midpoint=(a+safe)*.5;pa=skin.pose(a).positions;pb=skin.pose(safe).positions;pm=skin.pose(midpoint).positions
    assert np.all(np.abs(pm-(pa+pb)*.5)<=radius[:,None]+1e-13)
    assert manager._swept_admissible(a,safe)
    # Endpoint chord alone would retain positive clearance; actual rotating
    # directors penetrate at t=.5. Binary material-point witness is independent.
    uq=(a+unsafe)*.5;points=skin.pose(uq).positions
    source=np.array((1,2,5,4))-1;target=np.array((7,8,11,10))-1
    f=lambda p:tuple(F(float(x)) for x in p)
    sub=lambda p,q:tuple(x-y for x,y in zip(p,q))
    dot=lambda p,q:sum(x*y for x,y in zip(p,q))
    cross=lambda p,q:(p[1]*q[2]-p[2]*q[1],p[2]*q[0]-p[0]*q[2],p[0]*q[1]-p[1]*q[0])
    s=[f(x) for x in points[source]];t=[f(x) for x in points[target]]
    n=cross(sub(s[1],s[0]),sub(s[3],s[0]));numerator=dot(sub(s[0],t[0]),n)
    assert numerator<0 and numerator*numerator>F(float(5e-5))**2*dot(n,n)
    assert not manager._swept_admissible(a,unsafe)
    from anyfesolver.general_contact.finite_rotation_sweep import finite_whole_pair_lower
    from anyfesolver.general_contact.q4_q4_swept_bound import q4_q4_whole_pair_swept_lower
    from examples.general_contact_finite_rotation_shell_work import rotation
    square=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
    r=rotation(math.pi/4.);tn=square@r.T;sn=tn-45e-6*(r@np.array((0.,0.,1.)))
    old=q4_q4_whole_pair_swept_lower(sn,sn,tn,tn,square,square)
    new=finite_whole_pair_lower(sn,sn,tn,tn,square,square,endpoint_errors=(0.,)*4)
    assert old < -50e-6 and new >= -50e-6
    # Exact binary normal division, using a squared inequality without sqrt.
    ss=[f(x) for x in sn];tt=[f(x) for x in tn]
    nn=cross(sub(ss[1],ss[0]),sub(ss[3],ss[0]));num=dot(sub(ss[0],tt[0]),nn)
    assert num<0 and num*num<F(float(50e-6))**2*dot(nn,nn)
    assert manager._swept_admissible(rigid_pose(model,math.pi*8/36.),rigid_pose(model,math.pi*9/36.))
    save(dict(verified=True,safe_registered_five_degree_path=True,safe_tilted_five_degree_path=True,unsafe_sixty_degree_path_refused=True,independent_binary_midpoint_numerator=str(numerator),independent_binary_normal_squared=str(dot(n,n)),analytic_radius_max_m=float(max(radius)),tilted_negative_plane=dict(old_axis_lower_m=old,finite_euclidean_lower_m=new,exact_numerator=str(num),exact_normal_squared=str(dot(nn,nn))),endpoint_only_chord_not_certificate=True))


@pytest.mark.skipif(not os.environ.get('FINITE_STAGE'),
    reason='installed finite stage input FINITE_STAGE absent: staged finite-path solve requires the original workflow stage selection')
def test_installed_finite_path(monkeypatch):
    stage=os.environ['FINITE_STAGE'];step=int(stage.split('-')[1]);model,domain,settings,res,program=build()
    kwargs=dict(max_load_factor=step/15.)
    previous={1:None,4:1,8:4,12:8,13:12,14:13,15:14}[step]
    if stage=='run-1':
        import time,functools
        import anyfesolver.general_contact.semismooth_path as owner
        import anyfesolver.general_contact.static_path as path_owner
        from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
        counters={};accepted=[];last={}
        diagnostic=Path(os.environ['AUTOMATIC_RUNTIME_RESULT']).with_name('diagnostic.json')
        def flush():
            diagnostic.write_text(json.dumps(dict(counters=counters,accepted_inventory=accepted,last=last),indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x),encoding='utf-8')
        def instrument(obj,name,label,*,path=False,cap=False):
            original=getattr(obj,name)
            @functools.wraps(original)
            def wrapper(*args,**kw):
                item=counters.setdefault(label,dict(calls=0,completed=0,wall_s=0.,failures=0));item['calls']+=1
                if path:last['swept_start']=np.asarray(args[1]).tolist();last['swept_end']=np.asarray(args[2]).tolist();last['swept_result']='in_progress'
                flush();t=time.perf_counter()
                try:
                    result=original(*args,**kw)
                    if path:last['swept_result']=bool(result)
                    if cap:last['local_deformation_maxima']=getattr(args[0],'last_finite_deformation_maxima',None)
                    item['completed']+=1;return result
                except Exception as exc:
                    item['failures']+=1;last[label+'_error']=str(exc);raise
                finally:item['wall_s']+=time.perf_counter()-t;flush()
            monkeypatch.setattr(obj,name,wrapper)
        instrument(GeneralS3PatchManager,'_positions','physical_map')
        instrument(GeneralS3PatchManager,'_swept_admissible','whole_path',path=True)
        instrument(GeneralS3PatchManager,'require_finite_deformation','local_domain',cap=True)
        instrument(MovingS3PatchProjectedNormal,'_jet2_s3_work','contact_Jet2')
        instrument(owner,'_assemble_nonlinear_system','structural_material_tangent')
        original_factor=owner.factorize
        def factor(*args,**kw):
            item=counters.setdefault('sparse_factor_solve',dict(calls=0,wall_s=0.));t=time.perf_counter();response=original_factor(*args,**kw);item['wall_s']+=time.perf_counter()-t;item['calls']+=1;flush()
            class Response:
                def solve(self,*a,**k):
                    t=time.perf_counter()
                    try:return response.solve(*a,**k)
                    finally:item['wall_s']+=time.perf_counter()-t;flush()
            return Response()
        monkeypatch.setattr(owner,'factorize',factor)
        original_snapshot=path_owner._increment_snapshot
        def snapshot(*args,**kw):
            result=original_snapshot(*args,**kw);accepted.append(result.step_index);last['accepted_u']=np.asarray(result.displacements).tolist();flush();return result
        monkeypatch.setattr(path_owner,'_increment_snapshot',snapshot);flush()
    if previous is not None:kwargs['restart_checkpoint']=read('run-'+str(previous))['checkpoint']
    result=solve(model,settings,res,program,**kwargs)
    history={r['step_index']:r for r in result.info.get('force_displacement_history',[])}
    poses=[]
    for snapshot in result.snapshots:
        row=history[snapshot.step_index];contact=row['contact'];z=np.asarray(contact['numerical_pressure_pa'])*np.asarray(contact['reference_support_m2'])
        if result.restart_checkpoint is not None and snapshot.step_index==result.restart_checkpoint['path_state']['accepted_step']:z=np.array(result.restart_checkpoint['path_state']['normal_forces'])
        poses.append(dict(step=snapshot.step_index,u=snapshot.displacements,z=z,record=row))
    save(dict(status=result.status,failure_reason=result.failure_reason,identity=res.identity,checkpoint=result.restart_checkpoint,info=result.info,program=program.contract(),poses=poses,snapshot_inventory=[p['step'] for p in poses],previous_sha256=sha('run-'+str(previous)) if previous is not None else None))
    assert result.status=='completed',result.failure_reason
    assert [p['step'] for p in poses]==list(range((previous or 0)+1,step+1))


@pytest.mark.skipif(not os.environ.get('FINITE_STAGE'),
    reason='installed finite stage input FINITE_STAGE absent: saved finite-pose proof requires the original staged report and stage selection')
def test_saved_finite_poses_independent_physical_gates():
    from importlib.util import spec_from_file_location,module_from_spec
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    spec=spec_from_file_location('_finite_pose_oracle',Path(__file__).parents[1]/'docs/d1_review_support/contact_finite_rotation_shell_work_20261008/oracle.py')
    oracle=module_from_spec(spec);spec.loader.exec_module(oracle)
    stage=os.environ['FINITE_STAGE'].removeprefix('proof-')
    if stage=='chain':
        stages=['run-'+str(i) for i in (1,4,8,12,13,14,15)]
        chunks=[read(s) for s in stages]
        assert all(c['status']=='completed' for c in chunks)
        assert all(c['previous_sha256']==sha(stages[i-1]) for i,c in enumerate(chunks) if i)
        stored=dict(status='completed',identity=chunks[0]['identity'],poses=[p for c in chunks for p in c['poses']],snapshot_inventory=list(range(1,16)))
        source_hashes={s:sha(s) for s in stages}
    else:
        stored=read(stage);source_hashes={stage:sha(stage)}
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings)
    assert stored['status']=='completed' and stored['identity']==res.identity
    C=np.zeros((len(program.controls),manager.n_dof));names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5)
    for i,control in enumerate(program.controls):
        for node,name,weight in control.terms:C[i,model.mesh.dof_manager.get_node_dofs(node)[names[name]]]+=weight
    fixed=[model.mesh.dof_manager.get_node_dofs(n)[1] for n in (7,9)]
    free=np.setdiff1d(np.arange(manager.n_dof),fixed);reports=[]
    for saved in stored['poses']:
        step=saved['step'];q=np.array(saved['u']);z=np.array(saved['z']);manager.require_finite_deformation(q)
        pose=manager.finite_skin.pose(q);independent=[]
        for i,dofs in enumerate(manager.finite_skin.dofs):
            independent.append(manager.finite_skin.reference[i]+q[dofs[:3]]+manager.finite_skin.offsets[i]*decimal_director(q[dofs[3:]],manager.finite_skin.directors[i]))
        agreement=float(np.max(np.abs(pose.positions-independent)));assert agreement<=1e-13
        angle=min(step,12)*math.pi/36.;axis=np.array((math.sin(angle),0.,math.cos(angle)))
        gap=oracle.posed_bounds(pose.positions,axis)
        raw=linearize(manager,q,z);c=1000*reference_nodal_support(manager.law)
        ncp,active,_,_=complementarity(raw.state,z,c,raw.gap_gradient,tangent=False)
        ext=np.zeros(manager.n_dof)
        for force,value in zip(program.nodal_forces,program.force_values_at(step)):
            ext[model.mesh.dof_manager.get_node_dofs(force.node_id)[names[force.dof]]]+=value
        r,k,st=_assemble_nonlinear_system(model,q,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True)
        _discard_nonlinear_state_candidate(st)
        control_force=C.T@np.array(saved['record']['reaction_generalized'])
        equilibrium=np.linalg.norm((r+raw.force-ext-control_force)[free],1)/FREF
        complement=float(np.sum(np.abs(ncp))/FREF)
        assert equilibrium<=1e-6 and complement<=1e-8 and min(z)>=0
        physical=-raw.local_force.reshape(-1,3)
        force_error=np.linalg.norm(physical.sum(axis=0),1)/FREF
        moment=np.cross(pose.positions,physical).sum(axis=0)
        moment_error=np.linalg.norm(moment,1)/MREF
        mid=manager.finite_skin.reference+q[manager.finite_skin.dofs[:,:3]]
        nodal=raw.force[manager.finite_skin.dofs]
        structural_wrench=np.cross(mid,-nodal[:,:3]).sum(axis=0)-nodal[:,3:].sum(axis=0)
        structural_moment_error=np.linalg.norm(structural_wrench,1)/MREF
        assert max(force_error,moment_error,structural_moment_error)<=1e-6
        zero_contact=step<=12 or step==14
        if zero_contact:
            assert gap['strict_positive'] and np.linalg.norm(z,1)/FREF<=1e-8
        else:
            assert max(z)>0
            source=pose.positions[:6];target=pose.positions[6:]
            assert np.max(np.abs(source-(np.array(independent)[:6])))<=1e-13
            qrigid=rigid_pose(model,math.pi/3.)
            bending=q-qrigid
            upper=manager.finite_skin.dofs[:6];lower=manager.finite_skin.dofs[6:]
            assert np.max(np.abs(bending[upper[:,:3]]))>1e-10 and np.max(np.abs(bending[lower[:,:3]]))>1e-10
            assert np.max(np.abs(bending[upper[:,3:]]))>1e-10 and np.max(np.abs(bending[lower[:,3:]]))>1e-10
        virtual_work=[float(np.sum(raw.force[manager.finite_skin.dofs[:6,:3]]@axis)*1e-6),
                      float(np.sum(raw.force[manager.finite_skin.dofs[6:,:3]]@axis)*1e-6)]
        assert abs(sum(virtual_work))/(FREF*1e-6)<=1e-6
        if not zero_contact:assert min(abs(x) for x in virtual_work)>1e-12
        reports.append(dict(step=step,equilibrium_l1_relative=float(equilibrium),complementarity_l1_relative=complement,physical_force_relative=float(force_error),physical_moment_relative=float(moment_error),structural_spatial_moment_relative=float(structural_moment_error),independent_gap=gap,independent_skin_max_error_m=agreement,local_deformation_maxima=manager.last_finite_deformation_maxima,positive_support_count=int(np.count_nonzero(raw.state.reference_support>0)),both_body_physical_virtual_translation_work_j=virtual_work,both_body_coordinate_endpoint_contractions_j=[float(raw.force[manager.finite_skin.dofs[:6]].ravel()@q[manager.finite_skin.dofs[:6]].ravel()),float(raw.force[manager.finite_skin.dofs[6:]].ravel()@q[manager.finite_skin.dofs[6:]].ravel())]))
    assert [r['step'] for r in reports]==stored['snapshot_inventory']
    save(dict(verified=True,source_sha256=source_hashes,snapshot_inventory=stored['snapshot_inventory'],poses=reports,coordinate_endpoint_contractions_not_integrated_physical_work=True,opening_verified=any(r['step']==14 for r in reports)))


def test_finite_state_and_translation_seed():
    from scipy.sparse import eye
    from anyfesolver.general_contact.finite_rotation_predictor import finite_rigid_seed
    from anyfesolver.general_contact.finite_rotation_state import owner_local_coordinates,seal_finite_states,prepare_finite_states,finite_state_contract,COORDINATES,PROFILE
    from anyfesolver.general_contact.static_path import ContactPathControl,ContactPathProgram
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,domain,settings,res,program=build();n=model.mesh.dof_manager.total_dofs
    controls=tuple(ContactPathControl(((1,name,1.),),coordinate_profile='finite_shell_spin_chart_dev') for name in ('ux','uy','uz','rx','ry','rz'))
    shift=np.array((.001,-.002,.003));target=np.r_[shift,[0.,0.,0.]]
    translated=ContactPathProgram(controls,((0.,)*6,tuple(target)),1,control_profile='finite_shell_triplets_dev')
    rows=np.zeros((6,n));rows[:,model.mesh.dof_manager.get_node_dofs(1)]=np.eye(6)
    start=np.zeros(n);start[model.mesh.dof_manager.get_node_dofs(3)[3:]]=np.array((.01,-.02,.03))
    seed,report=finite_rigid_seed(model,translated,target,eye(n,format='csr'),np.zeros(n),start,rows)
    for node in model.mesh.nodes:
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        assert np.array_equal(seed[dofs[3:]],start[dofs[3:]])
        assert np.allclose(seed[dofs[:3]]-start[dofs[:3]],shift,rtol=0,atol=32*np.finfo(float).eps)
    assert report['common_translation_changed'] and not report['transported_free_nodal_frames']
    q=rigid_pose(model,math.pi/36.)
    r,k,trial=_assemble_nonlinear_system(model,q,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True)
    # Materialize the actual owner's trial record using the normal state-store API.
    from anyfesolver.nonlinear_static import _materialize_final_nonlinear_states
    states=_materialize_final_nonlinear_states(trial,{})
    info={};sealed=seal_finite_states(model,q,states,5,info)
    assert info['element_state_coordinates']==COORDINATES
    prepared=prepare_finite_states(model,q,sealed,5,{},restart=True)
    for eid,state in sealed.items():
        element=model.mesh.elements[eid];dofs=element.get_dof_mapping(model.mesh)
        assert np.linalg.norm(owner_local_coordinates(model,element,q[dofs]))<1e-12
        assert 'qualified_q4_committed_binding' not in prepared[eid]
        with pytest.raises(ValueError):element.validate_committed_current_tangent_state(model.mesh,model.get_material(element.material_name),q[dofs],state,5)
    bad=q.copy();bad[model.mesh.dof_manager.get_node_dofs(3)[2]]+=1e-4
    with pytest.raises(ValueError):prepare_finite_states(model,bad,sealed,5,{},restart=True)
    malformed=dict(profile=PROFILE)
    with pytest.raises(ValueError,match='coordinate/profile'):finite_state_contract(malformed)
    assert not finite_state_contract(dict(profile='legacy'))
    _discard_nonlinear_state_candidate(trial)
    save(dict(translation_only_seed=report,element_state_coordinates=COORDINATES,local_binding_count=len(sealed),global_binding_refused=True,changed_pose_refused=True,default_contract_unchanged=True))


def test_admission_and_restart_safety(monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled
    model,domain,settings,res,program=build();before=repr(model)
    token=CancellationToken();token.cancel('finite precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,res,program,cancellation_token=token)
    assert repr(model)==before
    from anyfesolver import BoundaryCondition
    import anyfesolver.general_contact.static_path as path_owner
    partial,_,ps,pr,pp=build()
    partial.add_boundary_condition(BoundaryCondition('unsupported-partial-spin',(3,),{'rx':0.}))
    material_calls=[]
    def no_material(*args,**kwargs):material_calls.append(True);raise AssertionError('material activation before finite admission')
    with monkeypatch.context() as local:
        local.setattr(path_owner,'_activate_nonlinear_state_storage',no_material)
        with pytest.raises(ValueError,match='all three constrained or all three free'):
            solve(partial,ps,pr,pp,max_load_factor=1/15.)
    assert not material_calls
    from anyfesolver.general_contact.finite_rotation_state import prepare_finite_states,finite_state_contract,COORDINATES,PROFILE
    assert not finite_state_contract(dict(profile='unrelated',element_state_coordinates='another_coordinate_contract'))
    with pytest.raises(ValueError,match='exact named'):finite_state_contract(dict(profile='unrelated',element_state_coordinates=COORDINATES))
    with pytest.raises(ValueError,match='coordinate/profile'):finite_state_contract(dict(profile=PROFILE,element_state_coordinates='global_q'))
    # Actual admitted legacy multi-beam public transient branch, before allocation.
    from examples.general_contact_multiple_beam_s3_transient import build as dynamic
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,_admit,ContactTransientSolverControls
    import anyfesolver.general_contact.multiple_beam_s3_transient as owner
    dm,dd,ds,velocity=dynamic();controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True)
    assert _admit(dm,dd,ds,controls)==owner.PROFILE
    calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('allocation before finite selector refusal')
    monkeypatch.setattr(owner,'pair_definitions',forbidden);monkeypatch.setattr(owner.MultipleBeamS3TransientManager,'__init__',forbidden);monkeypatch.setattr(dm,'apply_boundary_conditions',forbidden)
    with pytest.raises(ValueError,match='shell_skin_kinematics'):
        solve_transient_contact(dm,dd,TransientConfig(dt=1e-5,t_end=1e-5,initial_velocity=velocity),contact_settings=replace(ds,shell_skin_kinematics='corotational_nodal_director_dev'),solver_controls=controls)
    assert not calls
    save(dict(verified=True,public_precancel=True,declared_model_repr_unchanged=True,state_coordinate_named_profile_guard=True,legacy_valid_transient_before_allocation=True,partial_rotation_refused_before_material_activation=True,mutation_scope='declared model fields and no material/accepted-state mutation; BC preparation may occur'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'run-1.json').is_file(),
    reason='archived finite input AUTOMATIC_WORKFLOW_EVIDENCE/run-1.json absent: restart-safety replay requires the original staged checkpoint')
def test_archive_checkpoint_restart_safety():
    from anyfesolver.nonlinear_restart import _sha256,_checkpoint_without_hash
    model,domain,settings,res,program=build();stored=read('run-1');checkpoint=stored['checkpoint'];before=repr(model)
    changed=replace(program,waypoints=tuple([program.waypoints[0]]+[tuple(v+1e-8 if i==0 else v for i,v in enumerate(w)) for w in program.waypoints[1:]]))
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,res,changed,restart_checkpoint=checkpoint,max_load_factor=1/15.)
    damaged=copy.deepcopy(checkpoint);damaged['displacements'][0]+=1e-4
    with pytest.raises(ValueError,match='SHA-256'):solve(model,settings,res,program,restart_checkpoint=damaged,max_load_factor=1/15.)
    # Re-sealed outer bytes do not convert a stale local record into a current one.
    coherent=copy.deepcopy(checkpoint);free=model.mesh.dof_manager.get_node_dofs(3)[2];coherent['displacements'][free]+=1e-4
    coherent['checkpoint_sha256']=_sha256(_checkpoint_without_hash(coherent))
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,res,program,restart_checkpoint=coherent,max_load_factor=1/15.)
    save(dict(verified=True,checkpoint_sha256=sha('run-1'),program_incompatible=True,contents_refused=True,coherent_global_pose_local_record_mismatch_refused=True))


@pytest.mark.skipif(not Path('C:/ANYworkspaceWork/contact_q4_pair_monolithic_cycle_20261008/outputs/full.json').is_file(),
    reason='archived legacy cycle output contact_q4_pair_monolithic_cycle_20261008/outputs/full.json absent: legacy identity checkpoint replay requires the original saved run')
def test_legacy_default_identity_checkpoint():
    from examples.general_contact_q4_pair_monolithic_cycle import build as legacy,solve as legacy_solve
    from anyfesolver.general_contact.core import _contact_settings_contract
    path=Path('C:/ANYworkspaceWork/contact_q4_pair_monolithic_cycle_20261008/outputs/full.json')
    stored=json.loads(path.read_text());model,domain,settings,res,program=legacy()
    assert res.identity==stored['identity'] and 'shell_skin_kinematics' not in _contact_settings_contract(settings)
    result=legacy_solve(model,settings,res,program,restart_checkpoint=stored['checkpoint'])
    assert result.status=='completed'
    np.testing.assert_array_equal(result.displacements,np.array(stored['poses'][-1]['displacements']))
    save(dict(verified=True,legacy_identity_unchanged=True,legacy_global_state_checkpoint_continues=True,legacy_full_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_evidence=str(path)))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'run-1.json').is_file(),
    reason='archived finite input AUTOMATIC_WORKFLOW_EVIDENCE/run-1.json absent: local-state binding replay requires the original staged checkpoint')
def test_local_state_checkpoint_binding_discriminator(monkeypatch):
    model,domain,settings,res,program=build();checkpoint=copy.deepcopy(read('run-1')['checkpoint'])
    from anyfesolver.nonlinear_restart import _sha256,_checkpoint_without_hash
    dof=model.mesh.dof_manager.get_node_dofs(3)[2]
    fixed=[model.mesh.dof_manager.get_node_dofs(n)[1] for n in (7,9)]
    free=np.setdiff1d(np.arange(model.mesh.dof_manager.total_dofs),fixed)
    checkpoint['displacements'][dof]+=1e-4
    checkpoint['path_state']['reduced_coordinates'][int(np.flatnonzero(free==dof)[0])]+=1e-4
    checkpoint['checkpoint_sha256']=_sha256(_checkpoint_without_hash(checkpoint))
    import anyfesolver.general_contact.static_path as owner
    calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('material activation before invalid local record refusal')
    monkeypatch.setattr(owner,'_activate_nonlinear_state_storage',forbidden)
    with pytest.raises(ValueError,match='checkpoint state for element .* is incompatible'):
        solve(model,settings,res,program,restart_checkpoint=checkpoint,max_load_factor=1/15.)
    assert not calls
    save(dict(verified=True,source_sha256=sha('run-1'),outer_hash_resealed=True,global_and_reduced_pose_agree=True,specific_constitutive_local_record_refusal=True,before_material_activation=True,earlier_safety_claim_disposition='original generic incompatible matcher did not independently attribute its refusal to this guard'))
