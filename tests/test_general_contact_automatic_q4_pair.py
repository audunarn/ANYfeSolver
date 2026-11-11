"""Bounded installed guided-source press/20mm slide/open/recontact."""
from dataclasses import asdict,replace
import copy,hashlib,json,os
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact import ContactInteraction,resolve_automatic_contact_domain as resolve



def _footprint(model,resolution,settings,u):
    """Report actual projected pair polygons, not a new admission certificate."""
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.q4_source_overlap import projected_q4_source_face
    from anyfesolver.general_contact.overlap import PlanarFace,planar_projected_overlap
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    source,target=manager._positions(u)
    pairs=[]
    for si in manager.law.source_face_indices:
        face,normal=projected_q4_source_face(source[si])
        for ti in manager.law.target_face_indices:
            points=target[ti]
            projected=points-((points-source[si][0])@normal)[:,None]*normal
            overlap=planar_projected_overlap(face,PlanarFace((2,1),projected))
            if overlap is not None:
                pairs.append(dict(source_indices=list(map(int,si)),target_indices=list(map(int,ti)),
                    polygon_m=overlap.polygon,area_m2=overlap.area))
    return dict(area_m2=sum(p['area_m2'] for p in pairs),pairs=pairs,
        physical_source_nodes_m=source,physical_target_nodes_m=target,
        nominal_undeformed_rectangle_area_m2=.792 if u[model.mesh.dof_manager.get_node_dofs(1)[0]]>.01 else .81)


@pytest.mark.skipif(os.environ.get('ANY_RUN_Q4_PAIR')!='1',
                   reason='bounded installed refined automatic deformable lifecycle')
def test_installed_automatic_sliding_stage():
    """One guarded stage per invocation, sharing only retained JSON evidence."""
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    from anyfesolver import ContactDefinition,solve_static_nonlinear
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from examples.general_contact_automatic_q4_pair import build,solve_shell_pair
    stage=os.environ['Q4_PAIR_STAGE'].split('-')[0]
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    output=Path(os.environ['AUTOMATIC_DOMAIN_RESULT'])
    model,domain,settings,resolution,path=build(lifecycle=True,sliding=True,
        path_predictor=os.environ.get('AUTOMATIC_PATH_PREDICTOR','control_projection_dev'))
    report=dict(stage=stage,identity=resolution.identity,selections=[asdict(x) for x in resolution.selections],
        domain=domain.to_dict(),settings=asdict(settings),criteria=dict(
            force_scale_n=.02,moment_scale_nm=.02,virtual_translation_m=1e-6,
            force_moment_normalized_tolerance=1e-6,gap_budget_m=5e-5,
            displacement_parity_rtol=1e-7,displacement_parity_atol_m=1e-9,
            scale_basis='unchanged positive development scales k1000*area1*20um and lever1m',
            lateral_control_atol_m=1e-10,all_source_lateral_motion_min_m=.01,
            footprint_area_change_min_m2=1e-6,opening_clearance="strict_exact_rational_positive",
            target_return_atol_m=1e-10),
        physical_inputs=dict(source_midplane_z_m=.055,source_bottom_offset_m=-.025,
            target_midplane_z_m=-.01,target_top_offset_m=.025,source_director=(0.,0.,1.),target_director=(0.,0.,1.),
            path=path.contract()),
        runtime_before_solve=dict(anyfesolver_path=anyfesolver.__file__,
            automatic_domain_path=__import__('inspect').getfile(resolve),
            jit_disabled=bool(numba.config.DISABLE_JIT),numba_threads=int(numba.get_num_threads()),
            pools=threadpool_info()))
    def save():
        output.write_text(json.dumps(report,indent=2,
            default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    save()
    assert report['runtime_before_solve']['jit_disabled']
    assert report['runtime_before_solve']['numba_threads']==1
    assert all(pool['num_threads']==1 for pool in report['runtime_before_solve']['pools'])
    controls=dict(emit_restart_checkpoint=True,record_increment_snapshots=True)
    prefix_file='prefix.json'
    full_file='full.json'
    if stage=='prefix':controls['max_load_factor']=3/4
    if stage=='resume':
        prefix=json.loads((evidence/prefix_file).read_text(encoding='utf-8'))
        assert prefix['identity']==resolution.identity and prefix['verified']
        assert prefix['opening_verified']
        controls['restart_checkpoint']=prefix['checkpoint']
    if stage=='refusal':
        prefix=json.loads((evidence/prefix_file).read_text(encoding='utf-8'))
        before=repr(model)
        changed=replace(domain,name='changed-shell-provenance')
        changed_resolution=resolve(model,changed,settings,analysis_control='contact_path' if settings.path_predictor!='none' else 'displacement')
        with pytest.raises(ValueError,match='incompatible'):
            solve_shell_pair(model,changed,settings,changed_resolution,path,
                restart_checkpoint=prefix['checkpoint'])
        assert repr(model)==before
        if settings.path_predictor!='none':
            changed_settings=replace(settings,path_predictor='none')
            changed_resolution=resolve(model,domain,changed_settings,analysis_control='displacement')
            with pytest.raises(ValueError,match='incompatible'):
                solve_shell_pair(model,domain,changed_settings,changed_resolution,path,
                    restart_checkpoint=prefix['checkpoint'])
            assert repr(model)==before
            report['predictor_refusal']=True
        tampered=copy.deepcopy(prefix['checkpoint'])
        tampered['displacements'][0]+=1e-6
        with pytest.raises(ValueError,match='SHA-256 does not match its contents'):
            solve_shell_pair(model,domain,settings,resolution,path,restart_checkpoint=tampered)
        assert repr(model)==before
        report.update(verified=True,provenance_refusal=True,declared_repr_unchanged=True,tamper_refusal=True);save();return
    if stage=='explicit':
        definition=ContactDefinition(resolution.definition.surfaces,(),(
            ContactInteraction('ordinary-explicit','q4_source','q4_target',1000.,5e-5,
                               normal_law='projected_penalty'),))
        result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
            contact_path_program=path,max_iterations=15,**controls)
    else:result=solve_shell_pair(model,domain,settings,resolution,path,**controls)
    report.update(status=result.status,failure_reason=result.failure_reason,
        contact=result.info.get('general_contact'),phases=result.info.get('force_displacement_history'),
        displacements=result.displacements)
    if result.restart_checkpoint is not None:report['checkpoint']=result.to_restart_checkpoint()
    save()
    assert result.status=='completed',result.failure_reason
    phases=report['phases']
    source_x=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(1,7)]
    report['source_ux_m']=result.displacements[source_x].tolist()
    assert abs(result.displacements[source_x[0]]-.02)<=1e-10
    assert np.all(result.displacements[source_x]>.01)
    for index,row in enumerate(phases,1):
        np.testing.assert_allclose(row['target'],path.target_at(index),rtol=0.,atol=1e-10)
    expected_inventory={'full':[1,2,3,4],'explicit':[1,2,3,4],'prefix':[1,2,3],'resume':[4]}[stage]
    assert [x.step_index for x in result.snapshots]==expected_inventory
    report['accepted_snapshots']=[dict(step=x.step_index,source_ux_m=x.displacements[source_x],
        target_uz_m=x.displacements[model.mesh.dof_manager.get_node_dofs(11)[2]],
        footprint=_footprint(model,resolution,settings,x.displacements),
        independent_gap=_independent_pose_gap(model,x.displacements)) for x in result.snapshots]
    save()
    if stage in ('full','explicit','prefix'):
        by_step={x['step']:x for x in report['accepted_snapshots']}
        assert abs(by_step[1]['target_uz_m'])>1e-10
        assert np.all(np.asarray(by_step[2]['source_ux_m'])>.01)
        assert abs(by_step[2]['footprint']['area_m2']-by_step[1]['footprint']['area_m2'])>1e-6
        assert abs(by_step[2]['independent_gap']['overlap_area_m2']-by_step[1]['independent_gap']['overlap_area_m2'])>1e-6
        opened=by_step[3]
        assert F(opened['independent_gap']['numerator'],opened['independent_gap']['denominator'])>0
        assert opened['independent_gap']['lower_m']>0.
        assert abs(opened['target_uz_m'])<=1e-10
        assert phases[2]['contact']['regularization_storage_j']==0.
        report['opening_verified']=True;save()
    if settings.path_predictor!='none':
        assert result.info['contact_path']['path_predictor']==settings.path_predictor
        report['executed_path_predictor']=result.info['contact_path']['path_predictor'];save()
    expected={'full':[True,True,False,True],'explicit':[True,True,False,True],
              'prefix':[True,True,False],'resume':[True,True,False,True]}[stage]
    assert [row['contact']['active_pressure_dofs']>0 for row in phases]==expected
    wrench=[]
    for row,active in zip(phases,expected):
        contact=row['contact']
        errors=dict(force=float(np.linalg.norm(np.asarray(contact['integration_force'])+contact['target_force'])/.02),
                    moment=float(np.linalg.norm(np.asarray(contact['integration_moment'])+contact['target_moment'])/.02))
        wrench.append(errors)
        assert max(errors.values())<=1e-6
        assert contact['wrench_origin_m']==[0.,0.,0.]
        assert contact['gap_tolerance_met'] and contact['minimum_gap']>=-5e-5
        assert contact['contact_work_convention']=='projected_weak_gap_nonpotential_v1'
        assert (contact['regularization_storage_j']>0.)==active
        assert contact['penalty_energy']==0.
    report['phase_wrench_errors']=wrench;save()
    target_dof=model.mesh.dof_manager.get_node_dofs(11)[2]
    report['target_node11_uz_m']=float(result.displacements[target_dof])
    if stage!='prefix':
        assert abs(result.displacements[target_dof])>1e-10
        manager=GeneralS3PatchManager(model,resolution.definition,settings)
        residual,_,contact_result=manager.evaluate(result.displacements,tangent=False)
        from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
        structural,_,states=_assemble_nonlinear_system(model,result.displacements,result.element_states,
            5,tangent=False,kinematics='von_karman')
        _discard_nonlinear_state_candidate(states)
        equilibrium=(structural+residual)[target_dof]
        report['all_free_equilibrium_normalized_error']=float(np.linalg.norm((structural+residual)[[model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(2,7)]+[target_dof]])/.02)
        assert report['all_free_equilibrium_normalized_error']<=1e-6
        report['target_equilibrium_normalized_error']=abs(float(equilibrium))/.02
        report['support_reactions']=result.steps[-1].support_reactions
        report['actuator_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in range(1,7)]]))
        report['target_support_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (7,8,9,10,12,13,14,15)]]))
        save()
        assert abs(equilibrium)/.02<=1e-6
        assert abs(report['actuator_reaction_n'])>1e-6
        assert abs(report['target_support_reaction_n'])>1e-6
        assert abs(report['actuator_reaction_n']+report['target_support_reaction_n'])/.02<=1e-6
        source_probe=np.zeros(manager.n_dof);target_probe=np.zeros(manager.n_dof)
        for node in range(1,7):source_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        for node in range(7,16):target_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        ws=float(residual@source_probe);wt=float(residual@target_probe)
        source_work=float(residual@(result.displacements*(source_probe!=0)))
        target_work=float(residual@(result.displacements*(target_probe!=0)))
        report['both_body_work']=dict(source_virtual_work_j=ws,target_virtual_work_j=wt,
            common_translation_normalized_error=abs(ws+wt)/(.02*1e-6),
            source_residual_dot_displacement_j=source_work,target_residual_dot_displacement_j=target_work,
            interpretation='instantaneous residual work; not path-integrated work or regularization storage')
        save()
        assert abs(ws)>1e-12 and abs(wt)>1e-12
        assert abs(ws+wt)/(.02*1e-6)<=1e-6
        assert abs(source_work)>1e-12 and abs(target_work)>1e-12
        np.testing.assert_allclose(ws,-contact_result.integration_force[2]*1e-6,rtol=1e-6,atol=2e-14)
        np.testing.assert_allclose(wt,-contact_result.target_force[2]*1e-6,rtol=1e-6,atol=2e-14)
    if stage in ('explicit','resume'):
        full=json.loads((evidence/full_file).read_text(encoding='utf-8'))
        assert full['identity']==resolution.identity and full['verified']
        np.testing.assert_allclose(result.displacements,full['displacements'],rtol=1e-7,atol=1e-9)
        response_errors={}
        for key,scale in (('integration_force',.02),('target_force',.02),
                          ('integration_moment',.02),('target_moment',.02),
                          ('minimum_gap',5e-5),('regularization_storage_j',2e-7)):
            error=float(np.linalg.norm(np.asarray(report['contact'][key])-full['contact'][key])/scale)
            response_errors[key]=error
            assert error<=1e-6
        report['normalized_endpoint_response_parity']=response_errors
        phase_errors=[]
        for current,reference in zip(phases,full['phases']):
            errors={}
            assert current['contact']['active_pressure_dofs']==reference['contact']['active_pressure_dofs']
            for key,scale in (('integration_force',.02),('target_force',.02),('integration_moment',.02),
                              ('target_moment',.02),('minimum_gap',5e-5),('regularization_storage_j',2e-7)):
                errors[key]=float(np.linalg.norm(np.asarray(current['contact'][key])-reference['contact'][key])/scale)
                assert errors[key]<=1e-6
            phase_errors.append(errors)
        report['normalized_per_phase_response_parity']=phase_errors;save()
    assert report['checkpoint']
    report['verified']=True;save()




def _turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def _clip_exact(poly,target):
    for a,b in zip(target,target[1:]+target[:1]):
        if not poly:break
        out=[];previous=poly[-1];dp=_turn(a,b,previous)
        for current in poly:
            dc=_turn(a,b,current)
            if (dc>=0)!=(dp>=0):
                t=dp/(dp-dc);out.append(tuple(previous[k]+t*(current[k]-previous[k]) for k in range(2)))
            if dc>=0:out.append(current)
            previous,dp=current,dc
        poly=out
    return poly

def _bilinear_extrema(poly,coeff):
    """Exact stationary points on every edge and in the polygon interior."""
    a,b,c,d=coeff
    def value(x,y):return a+b*x+c*y+d*x*y
    points=list(poly)
    for p,q in zip(poly,poly[1:]+poly[:1]):
        dx,dy=q[0]-p[0],q[1]-p[1]
        linear=b*dx+c*dy+d*(p[0]*dy+p[1]*dx);quadratic=d*dx*dy
        if quadratic:
            t=-linear/(2*quadratic)
            if 0<t<1:points.append((p[0]+t*dx,p[1]+t*dy))
    if d:
        stationary=(-c/d,-b/d)
        if all(_turn(p,q,stationary)>=0 for p,q in zip(poly,poly[1:]+poly[:1])):points.append(stationary)
    values=[value(*p) for p in points]
    return min(values),max(values)

def _independent_pose_gap(model,u):
    points={}
    for node in range(1,16):
        n=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        assert np.all(u[d[3:]]==0.) and u[d[1]]==0.
        if node>6:assert u[d[0]]==0.
        assert n.z==(.055 if node<=6 else -.01)
        offset=-.025 if node<=6 else .025
        points[node]=tuple(F(float(v)) for v in (n.x+u[d[0]],n.y+u[d[1]],float(n.z+offset)+u[d[2]]))
    minima=[];areas=[];polygons=[]
    for ids in ((1,2,5,4),(2,3,6,5)):
        source=[points[n] for n in ids];assert len({q[2] for q in source})==1
        assert all(_turn(source[j],source[(j+1)%4],source[(j+2)%4])>0 for j in range(4))
        for tids in ((7,8,11,10),(8,9,12,11),(10,11,14,13),(11,12,15,14)):
            target=[points[n] for n in tids]
            # Fixed axis-aligned target XY is an exactly affine map. Refuse
            # a general nonaffine XY chart instead of using this oracle there.
            x0,y0=target[0][:2];x1,y1=target[2][:2]
            assert x1>x0 and y1>y0 and [q[:2] for q in target]==[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
            z0,z1,z2,z3=[q[2] for q in target]
            d=(z0-z1+z2-z3)/((x1-x0)*(y1-y0));b=(z1-z0)/(x1-x0)-d*y0;c=(z3-z0)/(y1-y0)-d*x0;a=z0-b*x0-c*y0-d*x0*y0
            poly=_clip_exact([q[:2] for q in source],[q[:2] for q in target])
            if len(poly)<3:continue
            area=sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(poly,poly[1:]+poly[:1]))/2
            if area==0:continue
            assert area>0
            lower,upper=_bilinear_extrema(poly,(source[0][2]-a,-b,-c,-d))
            minima.append(lower);areas.append(area);polygons.append(dict(source_nodes=ids,target_nodes=tids,minimum_gap_m=float(lower),polygon_xy_m=[[float(v) for v in q] for q in poly]))
    assert minima;minimum=min(minima);assert minimum>=-F(5e-5)
    return dict(kind='exact_rational_horizontal_source_original_target_Q1_gap',minimum_gap_m=float(minimum),
        lower_m=float(np.nextafter(float(minimum),-np.inf)),numerator=minimum.numerator,denominator=minimum.denominator,
        overlap_area_m2=float(sum(areas)),polygons=polygons,strict_positive=minimum>0,
        interpretation='binary accepted pose; all edge stationary points; not swept or full map-roundoff qualification')

class Poly(dict):
    def __add__(self,other):
        other=other if isinstance(other,Poly) else Poly({(0,0):F(other)})
        out=Poly(self)
        for k,v in other.items():out[k]=out.get(k,F(0))+v
        return out
    __radd__=__add__
    def __neg__(self):return Poly({k:-v for k,v in self.items()})
    def __sub__(self,other):return self+-other
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        other=other if isinstance(other,Poly) else Poly({(0,0):F(other)});out=Poly()
        for (i,j),v in self.items():
            for (k,l),w in other.items():out[i+k,j+l]=out.get((i+k,j+l),F(0))+v*w
        return out
    __rmul__=__mul__
    def derivative(self,axis):
        return Poly({(i-(axis==0),j-(axis==1)):v*(i if axis==0 else j) for (i,j),v in self.items() if (i if axis==0 else j)>0})
    def integral(self):return sum(v*F(4,(i+1)*(j+1)) for (i,j),v in self.items() if i%2==j%2==0)

def _scalar_exact(reference):
    r,s=Poly({(1,0):F(1)}),Poly({(0,1):F(1)})
    hats=[(1-r)*(1-s)*F(1,4),(1+r)*(1-s)*F(1,4),(1+r)*(1+s)*F(1,4),(1-r)*(1+s)*F(1,4)]
    x=sum(h*F(float(q[0])) for h,q in zip(hats,reference));y=sum(h*F(float(q[1])) for h,q in zip(hats,reference))
    jac=x.derivative(0)*y.derivative(1)-x.derivative(1)*y.derivative(0)
    gap=-(F('.003')+F('.0004')*x+F('.0002')*y+F('.0003')*x*y)
    return np.array([float((h*jac).integral()) for h in hats]),np.array([float((h*gap*jac).integral()) for h in hats])

CONTROLS=dict(area_tolerance=1e-9,weighted_gap_tolerance=1e-10,maximum_triangles=128,
    support_gradient_tolerance=1e-5,support_hessian_tolerance=1e-5,weighted_gap_gradient_tolerance=1e-5,weighted_gap_hessian_tolerance=1e-5)

def test_independent_scalar_trapezoid_work_and_tangent(monkeypatch):
    from anyfesolver.general_contact.q4_q4_overlap_derivatives import q4_q4_overlap_derivatives as derivative,q4_q4_swept_chart_admissible as chart
    from anyfesolver.general_contact.q4_source_overlap import q4_source_overlap_integrals
    from anyfesolver.general_contact.s3_patch_mortar_energy import MovingS3PatchPrimalMortarPenalty,S3PatchFace
    from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
    poly=[(F('.1'),F(0)),(F(1),F(0)),(F(0),F(1)),(F(0),F('.1'))]
    assert _bilinear_extrema(poly,(F(0),F(0),F(0),F(1)))==(0,F(1,4))
    xy=np.array([[-.4,-.3],[1.7,-.3],[1.7,1.8],[-.4,1.8]])
    target=np.column_stack((xy,.003+.0004*xy[:,0]+.0002*xy[:,1]+.0003*xy[:,0]*xy[:,1]))
    scalar=[]
    for xy in (np.array([[0.,0.],[.8,0.],[.8,.9],[0.,.9]]),np.array([[0.,0.],[1.1,0.],[.85,.9],[.05,.9]])):
        source=np.column_stack((xy,np.zeros(4)));actual=derivative(source,source,target,**{**CONTROLS,"maximum_triangles":512});expected=_scalar_exact(source)
        np.testing.assert_allclose(actual.reference_support,expected[0],atol=1e-9,rtol=0)
        np.testing.assert_allclose(actual.reference_weighted_gap,expected[1],atol=1e-10,rtol=0)
        value=q4_source_overlap_integrals(source,source,target,area_tolerance=1e-9,weighted_gap_tolerance=1e-10,maximum_triangles=512)
        np.testing.assert_allclose(value.weighted_gap,expected[1],atol=1e-10,rtol=0)
        scalar.append(dict(support_error_m2=float(np.max(np.abs(actual.reference_support-expected[0]))),gap_error_m3=float(np.max(np.abs(actual.reference_weighted_gap-expected[1]))),evaluated_cells=actual.integration_triangles,measured_resolution=actual.measured_errors))
        assert chart(source,source,target,target,source,target)
        reversed_target=target[[0,3,2,1]]
        assert not chart(source,source,reversed_target,reversed_target,source,target)
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(status="scalar_completed",scalar=scalar),indent=2))
    source=np.array([[0.,0.,0.],[.4,0.,0.],[.8,0.,0.],[0.,.9,0.],[.4,.9,0.],[.8,.9,0.]])
    target_reference=target.copy();target_reference[:,2]=.003
    geometry=MovingS3PatchPrimalMortarPenalty(tuple(S3PatchFace(ids,source[np.array(ids)-1]) for ids in ((1,2,5,4),(2,3,6,5))),
        (S3PatchFace((7,8,9,10),target_reference),),1000.,difference_step=2e-5,differentiation='jet2_q4_q4',
        q4_source_area_tolerance=1e-9,q4_source_weighted_gap_tolerance=1e-10,maximum_q4_source_integration_triangles=128,
        q4_source_derivative_tolerances=(1e-5,)*4,maximum_candidate_pairs=2)
    law=MovingS3PatchProjectedNormal(geometry)
    expected_support=np.zeros(6);expected_gap=np.zeros(6)
    for ids in ((1,2,5,4),(2,3,6,5)):
        values=_scalar_exact(source[np.array(ids)-1]);expected_support[np.array(ids)-1]+=values[0];expected_gap[np.array(ids)-1]+=values[1]
    independent=law.work(source,target,np.zeros(6))
    np.testing.assert_allclose(independent.state.reference_support,expected_support,atol=1e-9,rtol=0)
    np.testing.assert_allclose(independent.state.reference_weighted_gap,expected_gap,atol=1e-10,rtol=0)
    np.testing.assert_allclose(independent.state.pressure,1000*np.maximum(-expected_gap/expected_support,0),atol=1e-6,rtol=0)
    crossface=float(np.linalg.norm(independent.tangent[0:3,6:9]));assert crossface>1e-10
    source[:,2]=.0002*source[:,0]+.0001*source[:,1]+.0003*source[:,0]*source[:,1]
    pose=np.concatenate((source,target));direction=np.random.default_rng(72).normal(size=pose.shape);direction/=np.linalg.norm(direction)
    import anyfesolver.general_contact.s3_patch_augmented as owner
    agree=owner._q4_s3_pair_values_agree;operand=owner._q4_s3_gap_operand_roundoff
    forced=[];fallback=[]
    def force_fallback(*args,**kw):
        if kw.get('normwise') and not forced:forced.append(True);return False
        return agree(*args,**kw)
    def checked_operand(a,b,scale):
        assert b.shape==(4,3);fallback.append(True);return operand(a,b,scale)
    monkeypatch.setattr(owner,'_q4_s3_pair_values_agree',force_fallback)
    monkeypatch.setattr(owner,'_q4_s3_gap_operand_roundoff',checked_operand)
    work=law.work(source,target,np.zeros(6));errors=[]
    assert forced and fallback
    for h in (1e-7,3e-8):
        plus,minus=pose+h*direction,pose-h*direction
        fd=(law.work(plus[:6],plus[6:],np.zeros(6),tangent=False).residual-law.work(minus[:6],minus[6:],np.zeros(6),tangent=False).residual)/(2*h)
        exact=work.tangent@direction.reshape(-1);error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-12));errors.append(error);assert error<=1e-5
    common=np.tile((0.,0.,1e-6),len(pose));assert abs(work.residual@common)/(.02*1e-6)<=1e-6
    assert len(work.state.reference_support)==6 and np.all(work.state.reference_support>0)
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,scalar=scalar,both_body_directional_tangent_errors=errors,
        common_translation_work_normalized=abs(float(work.residual@common))/(.02*1e-6),shared_pressure_nodes=list(law.source_node_ids),independent_shared_sums=True,cross_face_tangent_block_norm=crossface,edge_stationary_counterexample=True,forced_four_corner_operand_fallback=True),indent=2))

def test_public_transient_refuses_unused_q4_pair_selector(monkeypatch):
    from examples.general_contact_multiple_beam_s3_transient import build
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,_admit,ContactTransientSolverControls
    import anyfesolver.general_contact.multiple_beam_s3_transient as owner
    model,definition,settings,velocity=build()
    controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True)
    before=repr(model)
    assert _admit(model,definition,settings,controls)==owner.PROFILE
    assert repr(model)==before
    class WrongSettings:
        general_s3_differentiation='jet2_q4_q4_dev'
    with pytest.raises(ValueError,match='requires exact contact definitions/settings'):
        _admit(model,definition,WrongSettings(),controls)
    with pytest.raises(ValueError,match='requires exact contact definitions/settings'):
        _admit(model,object(),replace(settings,general_s3_differentiation='jet2_q4_q4_dev'),controls)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(True)
        raise AssertionError('allocation or boundary mutation reached before selector refusal')
    monkeypatch.setattr(owner,'pair_definitions',forbidden)
    monkeypatch.setattr(owner.MultipleBeamS3TransientManager,'__init__',forbidden)
    monkeypatch.setattr(model,'apply_boundary_conditions',forbidden)
    with pytest.raises(ValueError,match='jet2_q4_q4_dev is unsupported'):
        solve_transient_contact(model,definition,TransientConfig(dt=1e-5,t_end=1e-5,initial_velocity=velocity),
            contact_settings=replace(settings,general_s3_differentiation='jet2_q4_q4_dev'),solver_controls=controls)
    assert not calls and repr(model)==before
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,
        actual_public_multiple_beam_flag=True,legacy_fixture_admitted=owner.PROFILE,
        named_selector_refusal=True,before_pair_inventory_and_manager_allocation=True,
        before_boundary_mutation=True,repr_fields_unchanged=True,wrong_type_contract_preserved=True),indent=2))


def test_default_identity_physical_map_and_public_unused_routes(monkeypatch):
    from anyfesolver import validate_general_contact,CancellationToken,SolveCancelled,LoadCase,solve_static_nonlinear,solve_static_arc_length,ArcLengthControl
    from anyfesolver.general_contact.core import _contact_settings_contract
    from examples.general_contact_automatic_s3_pair import build_s3_pair
    from examples.general_contact_automatic_q4_pair import build,solve_shell_pair
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,domain,settings,res,path=build_s3_pair()
    assert 'path_predictor' in _contact_settings_contract(settings) and 'shell_pair_orientation' not in _contact_settings_contract(settings)
    candidate=build();model,domain,settings,res,path=candidate
    manager=GeneralS3PatchManager(model,res.definition,settings);positions=manager._positions(np.zeros(manager.n_dof))
    np.testing.assert_allclose(positions[0][:,2],.03,atol=1e-17,rtol=0)
    np.testing.assert_allclose(positions[1][:,2],.015,atol=1e-17,rtol=0)
    assert len(manager.law.source_node_ids)==6 and len(manager.law.target_node_ids)==9
    before=repr(model);token=CancellationToken();token.cancel('Q4 pair public precancel')
    with pytest.raises(SolveCancelled):solve_shell_pair(*candidate,cancellation_token=token)
    with pytest.raises(ValueError,match='frictionless'):solve_shell_pair(*candidate,contact_friction=object())
    assert repr(model)==before
    from examples.general_contact_triangulated_rigid import case
    rigid,definition,defaults=case();assert validate_general_contact(rigid,definition,defaults).valid
    changed=replace(defaults,general_s3_differentiation='jet2_q4_q4_dev')
    load=LoadCase('refusal');load.add_nodal_load(2,forces=(0.,0.,-.01));before=repr(rigid)
    with pytest.raises(ValueError,match='jet2_q4_q4_dev requires'):
        solve_static_nonlinear(rigid,load,contact=definition,contact_settings=changed)
    assert repr(rigid)==before
    from examples.general_contact_triangulated_rigid_patch import case as arc_case
    arc,definition,defaults=arc_case();assert validate_general_contact(arc,definition,defaults,analysis_control='arc_length').valid
    load=LoadCase('arc_refusal');load.add_nodal_load(2,forces=(0.,0.,-.01));before=repr(arc)
    import anyfesolver.general_contact.arc_contact as arc_owner
    calls=[]
    def forbidden(*args,**kw):
        calls.append(True);raise AssertionError('arc allocation/mutation reached before selector refusal')
    monkeypatch.setattr(arc_owner.TriangulatedRigidManager,'__init__',forbidden)
    monkeypatch.setattr(arc,'apply_boundary_conditions',forbidden)
    assert not arc_owner.validate_arc_contact(arc,object(),replace(defaults,general_s3_differentiation='jet2_q4_q4_dev')).valid
    for name,value,match in (('general_s3_differentiation','jet2_q4_q4_dev','jet2_q4_q4_dev requires'),
                             ('path_predictor','control_projection_dev','path_predictor requires'),
                             ('shell_pair_orientation','physical_sides_dev','shell_pair_orientation requires')):
        with pytest.raises(ValueError,match=match):
            solve_static_arc_length(arc,load,control=ArcLengthControl(initial_load_increment=1e-5,minimum_load_increment=1e-6,maximum_load_increment=1e-5,max_steps=1),contact=definition,contact_settings=replace(defaults,**{name:value}))
    assert not calls and repr(arc)==before
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,
        actual_public_rigid_and_arc_refusal=True,arc_before_manager_allocation_and_BC_mutation=True,physical_reference_positions=True,precancel=True,friction_refusal=True,repr_fields_unchanged=True),indent=2))


@pytest.mark.skipif(not os.environ.get('Q4_PAIR_DEFAULT_BASELINE') or not Path(os.environ['Q4_PAIR_DEFAULT_BASELINE']).is_file(),
    reason='archived baseline input Q4_PAIR_DEFAULT_BASELINE absent: default-identity and terminal-restart replay requires the original baseline witness')
def test_default_identity_archive_baseline_terminal_replay():
    from examples.general_contact_automatic_s3_pair import build_s3_pair,solve_shell_pair as oldsolve
    old=json.loads(Path(os.environ['Q4_PAIR_DEFAULT_BASELINE']).read_text())
    model,domain,settings,res,path=build_s3_pair()
    assert res.identity==old['identity']
    terminal=oldsolve(model,domain,settings,res,path,restart_checkpoint=old['checkpoint'])
    assert terminal.status=='completed';np.testing.assert_array_equal(terminal.displacements,old['displacements'])
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,default_identity_unchanged=True,legacy_terminal_restart=True),indent=2))
