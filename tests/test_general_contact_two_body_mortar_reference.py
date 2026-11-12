"""Independent two-body original-Q1 mortar code and response discriminators."""
import hashlib,importlib.util,json,math,os
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from examples.general_contact_tilted_relative_sliding import rotation,rigid_pose

ROOT=Path(__file__).parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
old=module('_mortar_fixture',ROOT/'tests/test_general_contact_paired_pressure_response.py')
oracle=module('_independent_mortar',ROOT/'docs/d1_review_support/contact_two_body_mortar_reference_20261008/oracle.py')
RAW=Path(os.environ.get('MORTAR_EVIDENCE','C:/ANYworkspaceWork/contact_two_body_mortar_reference_20261008/outputs'))
BASE=old.BASE
REF=Path('C:/ANYworkspaceWork/contact_paired_pressure_response_20261008/outputs/v2/run-14.json')
REF_SHA='0f3ace2e13a1c3e4dbb597ad0f79d550cabbd400c300c91cbf5b2d11368750a7'

def save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v) if isinstance(v,np.generic) else v),encoding='utf-8')

def skin(manager,q):
    f=manager.finite_skin;directors=np.array([old.prior.decimal_director(q[d[3:]],ref) for d,ref in zip(f.dofs,f.directors)])
    points=f.reference+q[f.dofs[:,:3]]+f.offsets[:,None]*directors
    reference=f.reference+f.offsets[:,None]*f.directors
    assert np.max(abs(points-f.pose(q).positions))<=1e-13
    return reference,points,directors

def maps(manager,q,directors):
    f=manager.finite_skin;spin=np.zeros((len(directors)*3,len(q)));chart=spin.copy()
    for i,ds in enumerate(f.dofs):
        spin[3*i:3*i+3,ds[:3]]=np.eye(3);chart[3*i:3*i+3,ds[:3]]=np.eye(3)
        for k in range(3):
            axis=np.eye(3)[k];omega=axis.copy();term=axis.copy()
            for n in range(1,40):term=np.cross(q[ds[3:]],term)/(n+1);omega+=term
            spin[3*i:3*i+3,ds[3+k]]=f.offsets[i]*np.cross(axis,directors[i])
            chart[3*i:3*i+3,ds[3+k]]=f.offsets[i]*np.cross(omega,directors[i])
    return spin,chart

def posed(refined=False):
    path=REF if refined else BASE;digest=REF_SHA if refined else old.BASE_SHA
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    stored=json.loads(path.read_text());model,domain,settings,res,program,meta=old.fixture(refined)
    assert stored['identity']==res.identity
    pose=stored['poses'][-1];assert pose['step']==14
    q=np.asarray(pose['u']);z=np.asarray(pose['z']);manager=GeneralS3PatchManager(model,res.definition,settings)
    manager.require_finite_deformation(q)
    return model,manager,program,meta,pose,q,z,dict(path=str(path),sha256=digest,immutable_fixture_not_restart=True)

def richardson(reference,points,meta):
    args=(reference,points,meta['source_faces'],meta['target_faces'],meta['source_count'])
    coarse=oracle.derivatives(*args,h=2e-6,order=8);fine=oracle.derivatives(*args,h=1e-6,order=8)
    result=dict(fine)
    for name in ('gap_gradient','bin_gap_gradient'):result[name]=(4*fine[name]-coarse[name])/3
    return result,coarse,fine

def test_common_frame_exact_two_body_mortar():
    reports=[];normal=rotation(math.pi/3)@np.array((0.,0.,1.))
    for refined in (False,True):
        model,domain,settings,res,program,meta=old.fixture(refined);manager=GeneralS3PatchManager(model,res.definition,settings)
        q=rigid_pose(model,math.pi/3)
        for d in manager.finite_skin.dofs[:meta['source_count']]:q[d[1]]+=.02
        reference,points,directors=skin(manager,q)
        exact=oracle.exact_rectangles(meta['source_x']);S,a,ab,areas,_=oracle.integrals(reference,points,meta['source_faces'],meta['target_faces'],meta['source_count'],8)
        assert np.max(abs(S-exact['support']))<=1e-12
        assert abs(areas.sum()-.792)<=1e-12
        z=np.linspace(1e-6,1e-5,len(S));raw=linearize(manager,q,z)
        g=np.einsum('ij,k->ijk',exact['normal_work']/S[:,None],normal).reshape(len(S),-1)
        spin,chart=maps(manager,q,directors)
        support_error=float(np.max(abs(S-raw.state.reference_support)))
        code_gradient_error=float(np.max(abs(g@chart-raw.gap_gradient)))
        force_error=float(np.linalg.norm(raw.local_force+g.T@z,1)/1e-4)
        assert support_error<=1e-12 and max(code_gradient_error,force_error)<=1e-6
        assert np.max(abs(ab.sum(axis=0)-a))<=1e-18
        reports.append(dict(refined=refined,exact_support=exact['exact_support'],support_error_m2=support_error,exact_normal_work=exact['normal_work'],code_gradient_error=code_gradient_error,force_l1_error_relative=force_error,overlap_bin_areas_m2=areas,reference_clearance_m=float(a.sum()/S.sum()),norm_scales=dict(force_n=1e-4,moment_nm=1e-4)))
    save(dict(verified=True,common_frame_degrees=60,source_slide_m=.02,original_bilinear_both_body_work=True,exact_fraction_rectangle_proof=reports))

@pytest.mark.skipif(not os.environ.get('MORTAR_STAGE'),
    reason='installed mortar stage input MORTAR_STAGE absent: staged loaded-pose solve requires the original workflow stage selection')
def test_actual_loaded_pose_independent_work_and_bins():
    refined=os.environ['MORTAR_STAGE']=='refined';model,manager,program,meta,pose,q,z,provenance=posed(refined)
    reference,points,directors=skin(manager,q)
    # Fixed orders are self-resolution controls, not a mesh/physics sweep.
    args=(reference,points,meta['source_faces'],meta['target_faces'],meta['source_count'])
    lo=oracle.integrals(*args,order=8);hi=oracle.integrals(*args,order=12)
    quadrature=dict(support_error_m2=float(np.max(abs(lo[0]-hi[0]))),weighted_gap_error_m3=float(np.max(abs(lo[1]-hi[1]))))
    assert quadrature['support_error_m2']<=1e-12 and quadrature['weighted_gap_error_m3']<=1e-14
    independent,coarse,fine=richardson(reference,points,meta)
    raw=linearize(manager,q,z)
    spin,chart=maps(manager,q,directors);g=independent['gap_gradient'];f=-g.T@z
    bins=-np.einsum('imj,m->ij',independent['bin_gap_gradient'],z)
    derivative_resolution=float(np.linalg.norm((fine['gap_gradient']-coarse['gap_gradient']).T@z,1)/1e-4)
    code=dict(support_error_m2=float(np.max(abs(independent['support']-raw.state.reference_support))),weighted_gap_error_m3=float(np.max(abs(independent['weighted']-raw.state.reference_weighted_gap))),gap_gradient_error=float(np.linalg.norm(g@chart-raw.gap_gradient)/max(np.linalg.norm(raw.gap_gradient),1e-12)),physical_force_l1_relative=float(np.linalg.norm(f-raw.local_force,1)/1e-4),spatial_force_moment_l1_relative=float(np.linalg.norm(spin.T@f-raw.force,1)/1e-4))
    assert derivative_resolution<=1e-6
    assert code['support_error_m2']<=1e-10 and code['weighted_gap_error_m3']<=1e-12
    assert max(code[k] for k in ('gap_gradient_error','physical_force_l1_relative','spatial_force_moment_l1_relative'))<=1e-6
    assert np.linalg.norm(bins.sum(axis=0)-f,1)/1e-4<=1e-6
    physical=[]
    for force in bins:
        body=force.reshape(-1,3)[:meta['source_count']]
        physical.append(dict(allocated_source_force_n=-body.sum(axis=0),allocated_source_moment_nm=-np.cross(points[:meta['source_count']],body).sum(axis=0)))
    physical_total=-f.reshape(-1,3)[:meta['source_count']].sum(axis=0)
    # Independent point/spatial work: all couples and both bodies participate.
    rng=np.random.default_rng(14260);dq=rng.normal(size=len(q));dq/=np.linalg.norm(dq)
    pointwork=float(f@(chart@dq))
    # Work comparison against production uses its spatial spin residual with
    # the independently reconstructed physical-spin perturbation, not dq alone.
    physical_virtual=dq.copy()
    for ds in manager.finite_skin.dofs:
        term=dq[ds[3:]].copy();omega=term.copy()
        for n in range(1,40):term=np.cross(q[ds[3:]],term)/(n+1);omega+=term
        physical_virtual[ds[3:]]=omega
    actual_work=float(raw.force@physical_virtual)
    work_error=abs(pointwork-actual_work)/max(abs(pointwork),1e-10);assert work_error<=1e-6
    # Scientific equilibrium/gap remain the prior independently checked gates.
    gates=old.physical_gates(model,manager.settings if hasattr(manager,'settings') else old.fixture(refined)[2],old.fixture(refined)[3],program,meta,pose)
    save(dict(verified=True,refined=refined,provenance=provenance,quadrature_self_resolution=quadrature,fd_steps_m=(2e-6,1e-6),fd_force_resolution_relative=derivative_resolution,code_comparison=code,physical_spin_work_relative=work_error,reference=independent,physical_force=f,bin_force=bins,overlap_only_bin_work_force_allocation=physical,physical_source_resultant_n=physical_total,physical_gates=gates,positions=points,reference_positions=reference,chart_map=chart,spin_map=spin,pressure_space_unchanged=True,bin_definition='d(sum z_i*a_i_bin/S_i), including global S derivatives; not lambda*n traction'))

@pytest.mark.skipif(not (RAW/'refined.json').is_file(),
    reason='archived mortar trace input MORTAR_EVIDENCE/refined.json absent: fixed-structure pressure-trace replay requires the original saved trace')
def test_fixed_structure_pressure_trace_discriminator():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,manager,program,meta,pose,q,z,provenance=posed(True)
    stored=json.loads((RAW/'refined.json').read_text());ref=stored['reference'];S=np.asarray(ref['support']);G=np.asarray(ref['gap_gradient']);spin=np.asarray(stored['spin_map']);chart=np.asarray(stored['chart_map'])
    free,_=old.selectors(model,program)
    # Coarse basis is evaluated on the SAME fine material mesh; it exactly
    # reproduces coarse piecewise Q1 without changing physical surfaces.
    E=np.zeros((10,6));xs=(.1,.35,.6,.85,1.1)
    for row,x in enumerate(xs):
        cell=0 if x<=.6 else 1;left=(.1,.6)[cell];right=(.6,1.1)[cell];u=(x-left)/(right-left)
        for y in range(2):E[row+5*y,cell+3*y]=1-u;E[row+5*y,cell+1+3*y]=u
    a=np.asarray(ref['weighted']);coarseS=E.T@S
    # Recover da/ds from independent saved FD at fixed fine geometry. This
    # relation needs ds: normalized G alone cannot be coarsened with E.
    reference=np.asarray(stored['reference_positions']);points=np.asarray(stored['positions'])
    c1=oracle.derivatives(reference,points,meta['source_faces'],meta['target_faces'],10,h=2e-6,transfer=E)
    c2=oracle.derivatives(reference,points,meta['source_faces'],meta['target_faces'],10,h=1e-6,transfer=E)
    cg=(4*c2['gap_gradient']-c1['gap_gradient'])/3
    r,K,trial=_assemble_nonlinear_system(model,q,{},5,tangent=True,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True);_discard_nonlinear_state_candidate(trial)
    Kstruct=K[np.ix_(free,free)].toarray();raw=linearize(manager,q,z)
    K=Kstruct+raw.tangent[np.ix_(free,free)].toarray()
    # One registered physical signed trace on fine geometry and its independent
    # overlap-reference L2 projection into the coarse subspace. No alternative
    # nonnegative equilibrium is constructed.
    M=oracle.integrals(reference,points,meta['source_faces'],meta['target_faces'],10,order=8)[4]
    fine_lambda=np.tile(np.array((1.,-1.,1.,-1.,1.)),2)*1e-4
    coarse_lambda=np.linalg.solve(E.T@M@E,E.T@M@fine_lambda)
    trace_projection_error=np.sqrt((fine_lambda-E@coarse_lambda)@M@(fine_lambda-E@coarse_lambda))
    reports=[]
    for label,s,g,lam in (('coarse6',coarseS,cg,coarse_lambda),('fine10',S,G,fine_lambda)):
        B=g@spin[:,free];H=g@chart[:,free];D=np.linalg.solve(K,B.T);A=H@D
        dz=s*lam;du=D@dz;gap=H@du
        residual=float(np.linalg.norm(K@du-B.T@dz)/max(np.linalg.norm(B.T@dz),1e-30));assert residual<=1e-10
        reports.append(dict(space=label,positive_support=s,signed_pressure_pa=lam,delta_supported_forces_n=dz,chart_response=du,normalized_gap_response_m=gap,solve_relative_residual=residual,source_pressure_rhs_load_n=(g.T@dz).reshape(-1,3)[:10].sum(axis=0)))
    actualD=np.linalg.solve(K,(G@spin[:,free]).T);structuralD=np.linalg.solve(Kstruct,(G@spin[:,free]).T)
    nonlinear_tangent_difference=float(np.linalg.norm(actualD-structuralD)/np.linalg.norm(actualD))
    difference=np.asarray(reports[1]['chart_response'])-reports[0]['chart_response']
    base=json.loads((RAW/'base.json').read_text());fine=json.loads((RAW/'refined.json').read_text())
    assert base['verified'] and fine['verified']
    actual_traces=[]
    for label,s,g,data in (('base6_on_fine_geometry',coarseS,cg,base),('refined10_on_fine_geometry',S,G,fine)):
        lam=np.asarray(data['physical_gates']['normal_forces'])/np.asarray(data['reference']['support'])
        dz=s*lam;local_rhs=(g.T@dz).reshape(-1,3);rhs=(g@spin[:,free]).T@dz
        response=np.linalg.solve(K,rhs)
        actual_traces.append(dict(trace=label,saved_pressure_pa=lam,common_geometry_support_m2=s,common_geometry_supported_force_n=dz,spatial_unconstrained_rhs=rhs,chart_response=response,source_contact_rhs_n=local_rhs[:10].sum(axis=0),source_contact_rhs_moment_nm=np.cross(points[:10],local_rhs[:10]).sum(axis=0),local_rhs=local_rhs))
    common_delta=np.asarray(actual_traces[1]['source_contact_rhs_n'])-actual_traces[0]['source_contact_rhs_n']
    observed_delta=np.asarray(fine['physical_source_resultant_n'])-base['physical_source_resultant_n']
    remainder=observed_delta-common_delta
    common_response_delta=np.asarray(actual_traces[1]['chart_response'])-actual_traces[0]['chart_response']
    bin_differences=[];sum_closure=[]
    for data in (base,fine):
        f=np.asarray(data['physical_force']).reshape(-1,3);p=np.asarray(data['positions']);b=np.asarray(data['bin_force']).reshape(2,-1,3)
        force_balance=float(np.linalg.norm(f.sum(axis=0),1)/1e-4);moment_balance=float(np.linalg.norm(np.cross(p,f).sum(axis=0),1)/1e-4)
        closure=float(np.linalg.norm((b.sum(axis=0)-f).reshape(-1),1)/1e-4)
        assert max(force_balance,moment_balance,closure)<=1e-6
        sum_closure.append(dict(independent_force_balance_relative=force_balance,independent_physical_moment_balance_relative=moment_balance,allocation_force_sum_closure_relative=closure,force_scale_n=1e-4,moment_scale_nm=1e-4))
    for i in range(2):
        a=base['overlap_only_bin_work_force_allocation'][i];b=fine['overlap_only_bin_work_force_allocation'][i]
        bin_differences.append(dict(source_material_x_bin=((.1,.6),(.6,1.1))[i],allocated_force_difference_n=np.asarray(b['allocated_source_force_n'])-a['allocated_source_force_n'],allocated_moment_difference_nm=np.asarray(b['allocated_source_moment_nm'])-a['allocated_source_moment_nm']))
    save(dict(verified=True,fixed_model_pose_sha256=provenance,free_selector=free,pressure_transfer=E,overlap_reference_mass_m2=M,coarse_trace_projection_error_pa_m=float(trace_projection_error),reference_responses=reports,pressure_trace_chart_response_difference_l2=float(np.linalg.norm(difference)),actual_fixed_z_vs_structural_only_response_relative=nonlinear_tangent_difference,actual_trace_common_geometry_response=actual_traces,actual_trace_common_chart_response_difference_l2=float(np.linalg.norm(common_response_delta)),observed_actual_resultant_difference_n=observed_delta,common_geometry_trace_resultant_difference_n=common_delta,observed_minus_common_geometry_resultant_difference_n=remainder,overlap_work_allocation_bin_differences=bin_differences,independent_physical_sum_closure=sum_closure,model_scope='same local frozen-pose Kstruct+baselinefine fixed-z contact K for both traces; signed pressure probes and mapped accepted traces are not coarse nonlinear equilibria',interpretation='checkerboard probe alone does not attribute prior16.85percent shift; common actual-trace difference and observed remainder are diagnostic identities, not causal separation of structural/pressure effects',physical_spatial_convergence_gate=False,pressure_admission=False))

@pytest.mark.skipif(not BASE.is_file(),
    reason='archived mortar pose input contact_tilted_relative_sliding_20261008/outputs/v3/run-14.json absent: loaded directional-tangent replay requires the original saved pose')
def test_independent_loaded_directional_tangent():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,manager,program,meta,pose,q,z,provenance=posed(False)
    free,_=old.selectors(model,program);rng=np.random.default_rng(601420)
    dq=np.zeros(len(q));dq[free]=rng.normal(size=len(free));dq/=np.linalg.norm(dq)
    dz=rng.normal(size=len(z))*1e-5;raw=linearize(manager,q,z)
    # Frozen outer steps keep the independently resolved inner FD above noise.
    def independent(at,multipliers):
        reference,points,dirs=skin(manager,at);g,coarse,fine=richardson(reference,points,meta)
        spin,chart=maps(manager,at,dirs)
        return spin.T@(-g['gap_gradient'].T@multipliers),g['gap'],spin.T@(-coarse['gap_gradient'].T@multipliers),spin.T@(-fine['gap_gradient'].T@multipliers)
    observed=[]
    for h in (2e-4,1e-4):
        rp,gp,rcp,rfp=independent(q+h*dq,z+h*dz);rn,gn,rcn,rfn=independent(q-h*dq,z-h*dz)
        observed.append(((rp-rn)/(2*h),(gp-gn)/(2*h),(rcp-rcn)/(2*h),(rfp-rfn)/(2*h)))
    contact=(4*observed[1][0]-observed[0][0])/3;gap=(4*observed[1][1]-observed[0][1])/3
    expected=raw.tangent@dq-raw.force_gradient.T@dz
    tangent_error=float(np.linalg.norm(contact-expected)/max(np.linalg.norm(expected),1e-12))
    gap_error=float(np.linalg.norm(gap-raw.gap_gradient@dq)/max(np.linalg.norm(raw.gap_gradient@dq),1e-12))
    outer_contact_resolution=float(np.linalg.norm(observed[1][0]-observed[0][0])/max(np.linalg.norm(contact),1e-12))
    outer_gap_resolution=float(np.linalg.norm(observed[1][1]-observed[0][1])/max(np.linalg.norm(gap),1e-12))
    inner_coarse=(4*observed[1][2]-observed[0][2])/3;inner_fine=(4*observed[1][3]-observed[0][3])/3
    inner_contact_resolution=float(np.linalg.norm(inner_coarse-inner_fine)/max(np.linalg.norm(contact),1e-12))
    # Qualified structural derivative is openly reused; independent contact
    # discrepancy is also measured in the full coupled tangent scale.
    r,K,trial=_assemble_nonlinear_system(model,q,{},5,tangent=True,kinematics='corotational',corotational_tangent='consistent');_discard_nonlinear_state_candidate(trial)
    coupled_expected=K@dq+expected;coupled_error=float(np.linalg.norm(contact-expected)/max(np.linalg.norm(coupled_expected),1e-12))
    assert max(outer_contact_resolution,outer_gap_resolution,inner_contact_resolution)<=1e-5
    assert max(tangent_error,gap_error,coupled_error)<=1e-5
    save(dict(verified=True,provenance=provenance,outer_chart_steps=(2e-4,1e-4),inner_skin_steps_m=(2e-6,1e-6),outer_contact_resolution=outer_contact_resolution,outer_gap_resolution=outer_gap_resolution,inner_contact_resolution=inner_contact_resolution,contact_directional_relative=tangent_error,chart_gap_directional_relative=gap_error,coupled_directional_relative=coupled_error,force_scale_n=1e-4,moment_scale_nm=1e-4,qualified_structural_operator_reused=True,full_q_z_direction=dq.tolist()+dz.tolist()))

def test_oracle_admission_and_flat_scalar_closure():
    reports=[]
    for refined in (False,True):
        model,domain,settings,res,program,meta=old.fixture(refined);manager=GeneralS3PatchManager(model,res.definition,settings)
        q=rigid_pose(model,math.pi/3)
        for d in manager.finite_skin.dofs[:meta['source_count']]:q[d[1]]+=.02
        reference,points,_=skin(manager,q)
        args=(reference,points,meta['source_faces'],meta['target_faces'],meta['source_count'])
        S,a,ab,area,_=oracle.integrals(*args,order=8);z=np.linspace(1e-6,1e-5,len(S))
        scalar_error=float(np.max(abs(a/S-2e-7))/2e-7)
        psi=float(z@(a/S));expected=float(2e-7*sum(z));psi_error=abs(psi-expected)/expected
        assert max(scalar_error,psi_error)<=1e-6
        wrong=tuple(tuple(reversed(f)) for f in meta['target_faces'])
        with pytest.raises(ValueError,match='CCW'):oracle.integrals(reference,points,meta['source_faces'],wrong,meta['source_count'])
        with pytest.raises(ValueError,match='CCW'):oracle.convex_ccw(np.array(((0.,0.),(1.,0.),(.2,.2),(0.,1.))))
        reports.append(dict(refined=refined,weighted_gap_identity_relative=scalar_error,scalar_work_identity_relative=psi_error,nominal_clearance_m=2e-7,actual_weighted_gap_m3=a,actual_psi_j=psi,strict_projected_target_and_bin_guards=True))
    save(dict(verified=True,added_scalar_and_admission_checks=reports,previous_planar_force_gradient_proof_reused=True))
