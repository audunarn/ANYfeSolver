"""Frozen actual bending and independent linearized-Q1 contact reference."""
import json,os,itertools,copy,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import pytest
from examples.general_contact_q4_pair_bending import build,solve

FORCE_SCALE=1e-4
MOMENT_SCALE=1e-4
SOURCE=((1,2,5,4),(2,3,6,5))
TARGET=((7,8,11,10),(8,9,12,11),(10,11,14,13),(11,12,15,14))


def save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x) if isinstance(x,np.generic) else x),encoding='utf-8')


def free_dofs(model):
    fixed=set()
    for node in (1,4):fixed.update(model.mesh.dof_manager.get_node_dofs(node))
    for node in (7,9,13,15):fixed.add(model.mesh.dof_manager.get_node_dofs(node)[2])
    for node,axis in ((7,0),(7,1),(9,1)):fixed.add(model.mesh.dof_manager.get_node_dofs(node)[axis])
    return np.array(sorted(set(range(model.mesh.dof_manager.total_dofs))-fixed))


class P(dict):
    def __add__(self,b):
        if not isinstance(b,P):b=P({(0,0):F(b)})
        out=P(self)
        for key,v in b.items():out[key]=out.get(key,F(0))+v
        return out
    __radd__=__add__
    def __neg__(self):return P({k:-v for k,v in self.items()})
    def __sub__(self,b):return self+-b
    def __rsub__(self,b):return -self+b
    def __mul__(self,b):
        if not isinstance(b,P):b=P({(0,0):F(b)})
        out=P()
        for (i,j),v in self.items():
            for (k,l),w in b.items():out[i+k,j+l]=out.get((i+k,j+l),F(0))+v*w
        return out
    __rmul__=__mul__
    def integral(self,x0,x1,y0,y1):return sum(v*(x1**(i+1)-x0**(i+1))*(y1**(j+1)-y0**(j+1))/((i+1)*(j+1)) for (i,j),v in self.items())


def skin(model,u):
    points=[];mids=[];rotations=[]
    for node in range(1,16):
        n=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node);offset=-.025 if node<=6 else .025
        q=np.array([n.x,n.y,n.z])+u[d[:3]];theta=u[d[3:]]
        mids.append(q);rotations.append(offset*np.array([theta[1],-theta[0],0.]))
        points.append(q+np.array([0.,0.,offset])+rotations[-1])
    return np.array(points),np.array(mids),np.array(rotations)


def independent_linear_contact(model):
    x,y=P({(1,0):F(1)}),P({(0,1):F(1)})
    def shapes(ids):
        q=[model.mesh.get_node(n) for n in ids];x0,x1,y0,y1=map(lambda z:F(float(z)),(q[0].x,q[2].x,q[0].y,q[2].y))
        a,b=(x-x0)*(1/(x1-x0)),(y-y0)*(1/(y1-y0))
        return ((1-a)*(1-b),a*(1-b),a*b,(1-a)*b),(x0,x1,y0,y1)
    support=[F(0)]*6;g=[[F(0)]*model.mesh.dof_manager.total_dofs for _ in range(6)];gap0=[F(0)]*6
    for ids in SOURCE:
        ns,sbox=shapes(ids)
        for tids in TARGET:
            nt,tbox=shapes(tids);x0,x1=max(sbox[0],tbox[0]),min(sbox[1],tbox[1]);y0,y1=max(sbox[2],tbox[2]),min(sbox[3],tbox[3])
            if x1<=x0 or y1<=y0:continue
            source_z=F(float(model.mesh.get_node(ids[0]).z-.025));target_z=F(float(model.mesh.get_node(tids[0]).z+.025))
            for i,n in enumerate(ids):
                row=n-1;area=ns[i].integral(x0,x1,y0,y1);support[row]+=area;gap0[row]+=area*(source_z-target_z)
                for j,m in enumerate(ids):g[row][model.mesh.dof_manager.get_node_dofs(m)[2]]+=(ns[i]*ns[j]).integral(x0,x1,y0,y1)
                for j,m in enumerate(tids):g[row][model.mesh.dof_manager.get_node_dofs(m)[2]]-=(ns[i]*nt[j]).integral(x0,x1,y0,y1)
    return np.array([float(v) for v in support]),np.array([[float(v) for v in row] for row in g]),np.array([float(v) for v in gap0])


def linear_reference(model,load):
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    K=assemble_stiffness_matrix(model)[0].toarray();free=free_dofs(model);kf=K[np.ix_(free,free)]
    s,g,g0=independent_linear_contact(model);gf=g[:,free];force=load.get_load_vector(model.mesh,model.mesh.dof_manager)
    cf=np.linalg.solve(kf,gf.T);A=gf@cf;scaled=A/np.sqrt(s[:,None]*s[None,:]);eig,vec=np.linalg.eigh(scaled)
    assert eig[0]>1e-10*eig[-1],('signed pressure response unresolved',eig.tolist())
    signed=vec[:,0]/np.sqrt(s);work=float(signed@A@signed);assert work>0 and np.linalg.norm(gf.T@signed)>1e-14
    q=g0+gf@np.linalg.solve(kf,force[free]);H=A+np.diag(s/1000.)
    solutions=[]
    for bits in itertools.product((False,True),repeat=6):
        active=np.array(bits);p=np.zeros(6)
        if active.any():p[active]=np.linalg.solve(H[np.ix_(active,active)],-q[active])
        w=H@p+q
        if np.min(p)>=-1e-12 and np.min(w)>=-1e-14 and np.max(np.abs(p*w))<=1e-14:solutions.append(p)
    assert solutions
    pressure=solutions[0];u=np.zeros(len(force));u[free]=np.linalg.solve(kf,force[free]+gf.T@pressure)
    residual=K@u-g.T@pressure-force
    assert np.linalg.norm(residual[free])/FORCE_SCALE<=1e-6
    return dict(support=s,g=g,g0=g0,k=K,free=free,pressure=pressure,displacements=u,
        compliance_eigenvalues=eig,signed_weakest_mode=signed,signed_mode_work=work,
        reference_equilibrium_error=float(np.linalg.norm(residual[free])/FORCE_SCALE))


def exact_pose_gap(points):
    p=[tuple(F(float(v)) for v in q) for q in points]
    sub=lambda a,b:tuple(x-y for x,y in zip(a,b))
    cross=lambda a,b:(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    bounds=[]
    for ids in SOURCE:
        source=[p[n-1] for n in ids];normals=[cross(sub(source[(k+1)%4],source[k]),sub(source[(k-1)%4],source[k])) for k in range(4)]
        den=min(n[2] for n in normals);assert den>0
        for tids in TARGET:
            target=[p[n-1] for n in tids];num=min(dot(sub(a,b),n) for a in source for b in target for n in normals)
            bounds.append(min(F(0),num)/den)
    minimum=min(bounds);assert minimum>=-F(float(5e-5))
    return dict(kind='independent exact binary all_material_pair_Q1_normal_bound',lower_m=float(minimum),fraction=str(minimum),all_pairs=len(bounds),not_projection_or_finite_rotation_qualification=True)


def test_reference_skin_signed_response_and_linearized_operator():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.q4_q4_overlap_derivatives import q4_q4_overlap_derivatives
    model,domain,settings,resolution,load=build();ref=linear_reference(model,load);manager=GeneralS3PatchManager(model,resolution.definition,settings)
    zero=np.zeros(manager.n_dof);source,target=manager._positions(zero)
    local=np.zeros((6,manager.map.shape[0]));support=np.zeros(6)
    for i,si in enumerate(manager.law.source_face_indices):
        for j,ti in enumerate(manager.law.target_face_indices):
            # Independent exact reference rectangles conservatively exclude
            # zero-overlap pairs. The scalar local kernel is overlap-only.
            if any(max(F(float(v)) for v in source[si,axis])<=min(F(float(v)) for v in target[ti,axis])
                   or max(F(float(v)) for v in target[ti,axis])<=min(F(float(v)) for v in source[si,axis]) for axis in (0,1)):
                continue
            d=q4_q4_overlap_derivatives(manager.law.geometry.source_faces[i].reference_positions,source[si],target[ti],
                area_tolerance=1e-9,weighted_gap_tolerance=1e-10,maximum_triangles=128,
                support_gradient_tolerance=1e-5,support_hessian_tolerance=1e-5,weighted_gap_gradient_tolerance=1e-5,weighted_gap_hessian_tolerance=1e-5)
            coords=np.array([3*n+k for n in si for k in range(3)]+[3*(6+n)+k for n in ti for k in range(3)])
            local[np.ix_(si,coords)]+=d.weighted_gap_gradient;support[si]+=d.reference_support
    full=local@manager.map;error=float(np.max(np.abs(full-ref['g']))/max(1.,np.max(np.abs(ref['g']))));assert error<=1e-6
    np.testing.assert_allclose(support,ref['support'],atol=1e-9,rtol=0)
    # Admitted rotational map verified with a genuinely mixed probe.
    probe=np.random.default_rng(19).normal(size=manager.n_dof)*1e-6;independent=skin(model,probe)[0]
    actual=np.concatenate(manager._positions(probe));assert np.max(np.abs(actual-independent))<=1e-13
    save(dict(reference_operator_scaled_error=error,reference_equilibrium_error=ref['reference_equilibrium_error'],pressure=ref['pressure'],support=ref['support'],displacements=ref['displacements'],compliance_eigenvalues=ref['compliance_eigenvalues'],signed_weakest_mode=ref['signed_weakest_mode'],signed_mode_work=ref['signed_mode_work'],six_mode_response_detected=True,scope='independent contact linearization; same qualified structuralK reused; not nonlinearcodecomparison'))


@pytest.mark.skipif(not os.environ.get('BENDING_STAGE'),
    reason='installed bending stage input BENDING_STAGE absent: staged bending solve requires the original workflow stage selection')
def test_installed_bending_stage():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    stage=os.environ['BENDING_STAGE'].split('-')[0];evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    model,domain,settings,resolution,load=build();options={}
    if stage=='prefix':options['max_load_factor']=.5
    if stage=='resume':options['restart_checkpoint']=json.loads((evidence/'prefix.json').read_text())['checkpoint']
    result=solve(model,settings,resolution,load,**options);report=dict(stage=stage,status=result.status,failure_reason=result.failure_reason,identity=resolution.identity,displacements=result.displacements,checkpoint=result.restart_checkpoint,contact=result.info.get('general_contact'),snapshot_inventory=[s.load_factor for s in result.snapshots]);save(report)
    assert result.status=='completed',result.failure_reason
    u=result.displacements;manager=GeneralS3PatchManager(model,resolution.definition,settings);residual,tangent,contact=manager.evaluate(u)
    structural,_,states=_assemble_nonlinear_system(model,u,result.element_states,5,tangent=False,kinematics='von_karman');_discard_nonlinear_state_candidate(states)
    external=load.get_load_vector(model.mesh,model.mesh.dof_manager)*result.load_factor;free=free_dofs(model);error=float(np.linalg.norm((structural+residual-external)[free])/FORCE_SCALE);report['equilibrium_normalized_error']=error;save(report);assert error<=1e-6
    points,mids,rotations=skin(model,u);np.testing.assert_allclose(points,np.concatenate(manager._positions(u)),atol=1e-13,rtol=0)
    authoritative_points=np.concatenate(manager._positions(u))
    report['gap']=exact_pose_gap(authoritative_points)
    report['gap']['input']='authoritative binary manager positions, independently reconstructed first'
    uz=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in range(1,16)])
    theta=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[3:]] for n in range(1,16)])
    report['upper_uz_range_m']=float(np.ptp(uz[:6]));report['lower_uz_norm_m']=float(np.linalg.norm(uz[6:]));report['body_rotation_norms_rad']=[float(np.linalg.norm(theta[:6])),float(np.linalg.norm(theta[6:]))];report['skin_rotation_displacement_norm_m']=float(np.linalg.norm(rotations));save(report)
    assert report['upper_uz_range_m']>1e-9 and report['lower_uz_norm_m']>1e-10 and min(report['body_rotation_norms_rad'])>1e-9 and report['skin_rotation_displacement_norm_m']>1e-11
    source,target=manager._positions(u);work=manager.law.work(source,target,np.zeros(6),tangent=True);local=work.residual.reshape(-1,3);force=-local
    # Independent pullback includes all rotational conjugate moments.
    expected=np.zeros(manager.n_dof)
    for n,row in enumerate(local,1):
        d=model.mesh.dof_manager.get_node_dofs(n);offset=-.025 if n<=6 else .025
        expected[d[:3]]=row;expected[d[3:]]=offset*np.cross([0.,0.,1.],row)
    assert np.max(np.abs(expected-residual))<=1e-13
    physical_force=np.sum(force,axis=0);physical_moment=np.sum(np.cross(points,force),axis=0)
    structural_moment=np.sum(np.cross(mids,force),axis=0)+sum((-.025 if n<=6 else .025)*np.cross([0.,0.,1.],force[n-1]) for n in range(1,16))
    report['wrenches']=dict(origin_m=[0.,0.,0.],physical_force_error=float(np.linalg.norm(physical_force)/FORCE_SCALE),physical_moment_error=float(np.linalg.norm(physical_moment)/MOMENT_SCALE),structural_coordinate_moment_error=float(np.linalg.norm(structural_moment)/MOMENT_SCALE),second_order_skin_deficit=np.sum(np.cross(rotations,force),axis=0))
    save(report);assert max(report['wrenches'][k] for k in ('physical_force_error','physical_moment_error','structural_coordinate_moment_error'))<=1e-6
    assert contact.active_pressure_dofs>0
    report['reactions']=result.steps[-1].support_reactions
    report['external_work_j']=float(external@u);report['both_body_instantaneous_work_j']=[float(residual[[model.mesh.dof_manager.get_node_dofs(n)[k] for n in nodes for k in range(6)]]@u[[model.mesh.dof_manager.get_node_dofs(n)[k] for n in nodes for k in range(6)]]) for nodes in (range(1,7),range(7,16))]
    reference=linear_reference(model,load);report['nonlinear_vs_linear']=dict(displacement_relative=float(np.linalg.norm(u-reference['displacements']*result.load_factor)/max(np.linalg.norm(reference['displacements']*result.load_factor),1e-30)),interpretation='observed nonlinear contact/structural correction; no1e-6comparisonclaim')
    if stage=='full':
        direction=np.random.default_rng(27).normal(size=manager.n_dof);direction/=np.linalg.norm(direction);errors=[]
        for h in (1e-8,3e-9):
            plus=manager.evaluate(u+h*direction,tangent=False)[0];minus=manager.evaluate(u-h*direction,tangent=False)[0];fd=(plus-minus)/(2*h);exact=tangent@direction;err=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-12));errors.append(err);assert err<=1e-5
        report['both_body_directional_tangent_errors']=errors
    if stage=='resume':
        original_path=evidence/'full-accuracy.json';original=json.loads(original_path.read_text());assert original['verified']
        report['restart_reference']=dict(path=str(original_path),sha256=hashlib.sha256(original_path.read_bytes()).hexdigest())
        report['restart_displacement_max_error_m']=float(np.max(np.abs(u-np.array(original['displacements']))))
        assert report['restart_displacement_max_error_m']<=1e-9
    report['verified']=True;save(report)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'full-accuracy.json').is_file(),
    reason='archived bending input AUTOMATIC_WORKFLOW_EVIDENCE/full-accuracy.json absent: saved-pose replay requires the original evidence root')
def test_exact_saved_pose_and_inventory():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);path=evidence/'full-accuracy.json';report=json.loads(path.read_text());assert report['verified']
    assert report['snapshot_inventory']==[.5,1.]
    model,domain,settings,resolution,load=build();manager=GeneralS3PatchManager(model,resolution.definition,settings)
    u=np.array(report['displacements']);actual=np.concatenate(manager._positions(u));independent=skin(model,u)[0]
    error=float(np.max(np.abs(actual-independent)));assert error<=1e-13
    gap=exact_pose_gap(actual)
    save(dict(verified=True,solver_replayed=False,source_report_path=str(path),source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),skin_correspondence_max_error_m=error,authoritative_binary_pose_gap=gap,snapshot_inventory=report['snapshot_inventory'],nominal_reference_gap_m=float((.04-.025)-(-.01+.025))))


def test_actual_restart_refusal_and_precancel():
    from anyfesolver import CancellationToken,SolveCancelled
    model,domain,settings,resolution,load=build();before=repr(model)
    token=CancellationToken();token.cancel('bending public precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,resolution,load,cancellation_token=token)
    assert repr(model)==before
    save(dict(verified=True,actual_public_precancel=True,declared_FEModel_repr_unchanged=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'prefix.json').is_file(),
    reason='archived bending input AUTOMATIC_WORKFLOW_EVIDENCE/prefix.json absent: restart-refusal replay requires the original staged checkpoint')
def test_actual_archive_checkpoint_restart_refusals():
    from dataclasses import replace
    from anyfesolver.general_contact.automatic_domain import resolve_automatic_contact_domain
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);path=evidence/'prefix.json';prefix=json.loads(path.read_text());assert prefix['verified']
    model,domain,settings,resolution,load=build();before=repr(model)
    changed=replace(domain,name='bending-provenance-refusal');changed_resolution=resolve_automatic_contact_domain(model,changed,settings,analysis_control='force')
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,changed_resolution,load,restart_checkpoint=prefix['checkpoint'])
    assert repr(model)==before
    changed_settings=replace(settings,q4_source_weighted_gap_tolerance=settings.q4_source_weighted_gap_tolerance*2)
    changed_resolution=resolve_automatic_contact_domain(model,domain,changed_settings,analysis_control='force')
    with pytest.raises(ValueError,match='incompatible'):solve(model,changed_settings,changed_resolution,load,restart_checkpoint=prefix['checkpoint'])
    assert repr(model)==before
    tampered=copy.deepcopy(prefix['checkpoint']);tampered['displacements'][0]+=1e-6
    with pytest.raises(ValueError,match='SHA-256 does not match its contents'):solve(model,settings,resolution,load,restart_checkpoint=tampered)
    assert repr(model)==before
    old_default=json.loads((evidence/'full.json').read_text())
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,resolution,load,restart_checkpoint=old_default['checkpoint'])
    assert repr(model)==before
    save(dict(verified=True,provenance_refusal=True,settings_refusal=True,tamper_refusal=True,old_default_tolerance_restart_refusal=True,prefix_path=str(path),prefix_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),old_default_report_sha256=hashlib.sha256((evidence/'full.json').read_bytes()).hexdigest(),transaction_claim='declared FEModel fields only'))
