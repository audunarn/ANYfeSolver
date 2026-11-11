"""Independent rim shape/work and actual public V2 material replay."""
import os,json,copy
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import pytest
from scipy import sparse
from examples.general_contact_s3_rim_replay import build_case,solve_case
from anyfesolver.general_contact.s3_friction_replay import S3RimReplayFrictionManager,S3RimReplayFrictionSettings,S3ReplayFrictionManager,S3ReplayFrictionSettings
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager

def root():return Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
def save(v):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))
def mode(m,ns,axis):
    u=np.zeros(42)
    for n in ns:u[m.mesh.dof_manager.get_node_dofs(n)[axis]]=1.
    return u
def owner(case):
    m,d,s,f,p=case
    return S3RimReplayFrictionManager(m,GeneralS3PatchManager(m,d,s),f)

def test_unit():
    case=list(build_case());m,d,s,f,p=case
    # Local matched shape control avoids a source vertex on an internal seam.
    for n in (1,2,3):m.mesh.get_node(n).y+=.01
    a=owner(case);x=mode(m,(1,2,3),0);z=mode(m,(1,2,3),2);tx=mode(m,(4,5,6,7),0)
    a.restrict_tangent_to_reduced_variations(sparse.csr_matrix(np.column_stack((x,tx,z))))
    press=-5.2e-6*z;a.commit(a.prepare_commit(press));u=press+.6*x
    before=a.checkpoint_state();force,frame,summary=a._integrate(u)
    force_full,_,summary_full=a._integrate(u,full=True);prepared=a.prepare_commit(u)
    assert np.array_equal(force,force_full)
    for field in ('dissipation_increment_j','storage_j','released_storage_j'):assert summary[field]==summary_full[field]
    for field in ('dissipation_increment_j','storage_j','released_storage_j'):assert prepared[2][field]==summary[field]
    # Source material triangle b>=0,c>=0,b+c<=1, cut b<=.5.
    # Physical A=.08*(2*.5-.5^2)=.06; dA/d(source shift)=-.2.
    pressure=float(frame['pressure'][0]);cap=.3*pressure;expected=cap*.06;derivative=-cap*.2
    assert np.max(abs(frame['pressure']-pressure))<1e-12
    assert abs(x@force-expected)/abs(expected)<=.001
    # Independent reference-Q1 moments on b<=1/2. Target facet(5,7,6)
    # owns this complete overlap; the returned anchor is x-cap/k,y.
    from fractions import Fraction as Q
    a_cut=Q(1,2);jac=Q(4,25)
    area=float(jac*(a_cut-a_cut*a_cut/2));Ib=float(jac*(a_cut*a_cut/2-a_cut**3/3));Ic=float(jac*(a_cut-a_cut*a_cut+a_cut**3/3)/2)
    nodal=np.zeros((7,6));nodal[:3,0]=cap*np.array((area-Ib-Ic,Ib,Ic))
    elastic=cap/1000.;nodal[4,0]=-cap*((1-.21)*area-.4*Ic);nodal[5,0]=-cap*((1-.8+elastic)*area-.4*Ib);nodal[6,0]=-cap*((.8+.21-1-elastic)*area+.4*(Ib+Ic))
    # Independent derivative of summed two-triangle unit-square normal:
    # dn_x/d(z4,z5,z6,z7)=(1,-1,1,-1)/2. Full gDn work included.
    actual_gap=float(frame['source'][0][0,2]-frame['origin'][2]);nodal[3:,2]=(-actual_gap*cap*area)*np.array((.5,-.5,.5,-.5))
    nodal[:3,4]=-.035*nodal[:3,0];nodal[3:,4]=.015*nodal[3:,0]
    work_error=float(np.linalg.norm(force-nodal.ravel())/np.linalg.norm(nodal))
    assert work_error<=1e-5
    total,K,_=a.evaluate(u);normal,Knormal,_=a.normal.evaluate(u)
    actual=float(x@((K-Knormal)@x))
    assert abs(actual-derivative)/abs(derivative)<=1e-5
    # Independent complete skin-force couples and common rigid virtual work.
    F=force.reshape(7,6);skin=np.array([[q.x,q.y,q.z] for q in m.mesh.nodes.values()])+u.reshape(7,6)[:,:3]
    net=F[:,:3].sum(axis=0);moment=(np.cross(skin,F[:,:3])+F[:,3:]).sum(axis=0)
    assert np.linalg.norm(net)/.02<=1e-6 and np.linalg.norm(moment)/.02<=1e-6
    h=2e-7;FD=(a._integrate(u+h*x)[0]-a._integrate(u-h*x)[0])/(2*h)
    assert np.linalg.norm((K-Knormal)@x-FD)/max(np.linalg.norm(FD),1e-300)<=1e-5
    pressure_FD=(a._integrate(u+h*z)[0]-a._integrate(u-h*z)[0])/(2*h)
    assert np.linalg.norm((K-Knormal)@z-pressure_FD)/np.linalg.norm(pressure_FD)<=1e-5
    assert abs(float(x@((K-Knormal)@z))+18.)<=18e-5
    assert a.checkpoint_state()==before
    outside=press+1.05*x
    ordinary=a._integrate(outside)[0];a._rim_empty_probe_origin=True
    shortcut=a._tangent_probe_force(outside);a._rim_empty_probe_origin=False
    assert np.array_equal(shortcut,ordinary) and np.count_nonzero(shortcut)==0
    full,Kempty,_=a.evaluate(outside);normal_only,Konly,_=a.normal.evaluate(outside)
    assert np.array_equal(full,normal_only) and np.array_equal(Kempty.toarray(),Konly.toarray())
    save(dict(expected_force_n=expected,actual_force_n=float(x@force),expected_shape_derivative_n_per_m=derivative,actual_derivative=actual,pressure_coupling_n_per_m=float(x@((K-Knormal)@z)),independent_full_nodal_force_and_skin_work_relative_error=work_error,net_force_n=net,moment_nm=moment,common_partition_exact=True,pressure_pa=pressure,scope='matched fixednormal partial triangle; no independent nonlinear solution'))

@pytest.mark.skipif(not os.environ.get('FINITE_STAGE'),
    reason='installed rim stage input FINITE_STAGE absent: staged rim replay solve requires the original workflow stage selection')
def test_public(monkeypatch):
    stage=os.environ['FINITE_STAGE'];step=int(stage.removeprefix('A'));case=build_case();m,d,s,f,p=case
    incoming=json.loads((root()/f'A{step-1}.json').read_text())['checkpoint'] if step>1 else None
    import anyfesolver.general_contact.static_path as driver
    original=driver.create_s3_friction_manager if hasattr(driver,'create_s3_friction_manager') else None
    import anyfesolver.general_contact.s3_friction_static as factory
    original=factory.create_s3_friction_manager;captured=[]
    def make(*args):
        a=original(*args);commit=a.commit;restore=a.restore
        def record(prepared):
            a.saved_force=a._integrate(prepared[0])[0].copy();a.saved_normal=a.normal.evaluate(prepared[0],tangent=False)[0].copy();commit(prepared)
        # Restore copies the owner; instance-bound commit instrumentation would
        # mutate the live owner when scratch validation commits. Arm only after
        # restore and install instrumentation when restore has completed.
        def restored(*args,**kwargs):
            restore(*args,**kwargs);a.commit=record
        a.restore=restored
        if incoming is None:a.commit=record
        captured.append(a);return a
    monkeypatch.setattr(factory,'create_s3_friction_manager',make)
    result=solve_case(case,step,incoming);assert result.status=='completed',result.info
    checkpoint=result.to_restart_checkpoint();assert checkpoint['path_state']['accepted_step']==step
    save(dict(status=result.status,checkpoint=checkpoint,info=result.info,verification='saved before independent physical assertions'))
    snap=result.snapshots[-1];u=np.array(checkpoint['displacements']);a=captured[-1]
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    K=assemble_stiffness_matrix(m)[0];contact=a.saved_force+a.saved_normal
    phase=checkpoint['path_state']['history'][-1];reaction=np.array(phase['reaction_generalized']);res=K@u+contact
    res[np.asarray(m.mesh.dof_manager.get_node_dofs(1))[[0,2]]]-=reaction
    P=np.column_stack((mode(m,(1,2,3),0),mode(m,(1,2,3),2),*[mode(m,(n,),0) for n in (5,6,7)]))
    eq=float(np.sum(abs(P.T@res))/.02);assert eq<=1e-6
    forces=contact.reshape(7,6);mid=np.array([[q.x,q.y,q.z] for q in m.mesh.nodes.values()])+u.reshape(7,6)[:,:3]
    net=forces[:,:3].sum(axis=0);moment=(np.cross(mid,forces[:,:3])+forces[:,3:]).sum(axis=0)
    assert np.linalg.norm(net)/.02<=1e-6 and np.linalg.norm(moment)/.02<=1e-6
    assert phase['contact']['gap_tolerance_met']
    if step in (4,5,6):assert np.linalg.norm(contact)<=1e-12
    if step==6:
        skin=mid[:3,2]-.035;assert min(skin)-.015>0
    if step in (3,7):assert np.linalg.norm(u.reshape(7,6)[3:,:2])>0
    save(dict(status=result.status,checkpoint=checkpoint,info=result.info,snapshots=[dict(step_index=x.step_index,displacement=x.displacements) for x in result.snapshots],physical=dict(equilibrium=eq,net_force_n=net,moment_nm=moment,target_inplane_norm_m=float(np.linalg.norm(u.reshape(7,6)[3:,:2])),contact_norm=float(np.linalg.norm(contact))),scope='segmented public projectedpenalty; not hardKKT'))

def test_controls(monkeypatch):
    case=build_case();a=owner(case);before=a.checkpoint_state();u=np.zeros(42)
    changed=u.copy();changed[case[0].mesh.dof_manager.get_node_dofs(4)[2]]=1e-6
    with pytest.raises(ValueError,match='plane'):a.require_friction_path(u,changed)
    tilted=u.copy();tilted[case[0].mesh.dof_manager.get_node_dofs(2)[2]]=1e-6
    with pytest.raises(ValueError,match='parallel'):a.require_friction_path(u,tilted)
    sideways=u.copy();sideways[case[0].mesh.dof_manager.get_node_dofs(2)[1]]=1e-6
    with pytest.raises(ValueError,match='tangential'):a.require_friction_path(u,sideways)
    a.friction_settings=replace(a.friction_settings,work_integration_tolerance_j=2e-12)
    with pytest.raises(ValueError,match='mutated'):a.require_friction_path(u,u)
    assert a.checkpoint_state()==before
    save(dict(moving_plane_refused=True,cache_settings_mutation_refused=True,physical_history_unchanged=True))

def test_front():
    m,d,s,f,p=build_case();a=owner((m,d,s,f,p));x=mode(m,(1,2,3),0);z=mode(m,(1,2,3),2)
    press=-5.2e-6*z;a.commit(a.prepare_commit(press));a.commit(a.prepare_commit(press+1.05*x))
    u=press+.55*x;frame=a._frame(u);cap=.3*float(frame['pressure'][0]);kt=1000.;width=cap/kt
    distances=width*np.array((.1,.5,1.,2.,10.))
    b=(1.-.75-distances)/.4;c=np.full(len(b),.1);bary=np.column_stack((1-b-c,b,c))
    state=a.query_material(0,bary,candidate=frame)[0]
    anchors=np.zeros((len(b),3))
    for e,corners in frame['target'].items():
        mask=state['element']==e;anchors[mask]=state['bary'][mask]@corners
    points=bary@frame['source'][0];traction=-kt*(points-anchors);traction[:,2]=0
    expected=np.minimum(kt*distances,cap)
    assert np.max(abs(traction[:,0]-expected))<=1e-10
    # Exact independent strip integrals per unit rim length: missing saturated
    # approximation minus min(k*distance,cap), and elastic storage respectively.
    length=.4;force_error=.5*cap*width*length;storage_error=cap*cap/(2*kt)*width*length
    derivative_error=cap*width # straight triangle rim length changes at slope1
    assert force_error<2e-9 and storage_error<1e-12
    save(dict(cap_pa=cap,front_width_m=width,distances_m=distances,actual_traction_pa=traction[:,0],independent_traction_pa=expected,missing_strip_force_upper_n=force_error,missing_strip_storage_upper_j=storage_error,strip_shape_derivative_upper_n_per_m=derivative_error,scope='independent constant-pressure rebirth front, literal undeformed target; loaded variable-pressure front remains controlled numerically'))

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A6.json').is_file(),
    reason='archived rim input AUTOMATIC_WORKFLOW_EVIDENCE/A6.json absent: birth-front replay requires the original authenticated diagnostic input')
def test_birth():
    case=build_case();m=case[0];a=owner(case);cp=json.loads((root()/'A6.json').read_text())['checkpoint'];a.restore(cp['path_state']['friction_state'],np.array(cp['displacements']))
    u=.55*mode(m,(1,2,3),0)-5.2e-6*mode(m,(1,2,3),2)
    first=a.frames[-1];last=a._frame(u);g0=float((first['source'][0][0]-first['origin'])@first['normal']);g1=float((last['source'][0][0]-last['origin'])@last['normal']);fraction=-g0/(g1-g0)
    at_birth=first['source'][0]+fraction*(last['source'][0]-first['source'][0]);target=first['target'][3]+fraction*(last['target'][3]-first['target'][3]);rim=max(target[:,0]);b=(rim-at_birth[0,0])/(at_birth[1,0]-at_birth[0,0])
    force,_,base=a._integrate(u);control,_,fine=a._integrate(u,level=2)
    error={k:float(fine[k]-base[k]) for k in ('dissipation_increment_j','storage_j','released_storage_j')}
    before=a.checkpoint_state();probe=a._tangent_probe_force(u)
    assert np.array_equal(probe,force) and a.checkpoint_state()==before
    assert not a._rim_force_only_open_origin
    save(dict(normal_birth_fraction=fraction,material_birth_boundary_b=b,gaps_m=(g0,g1),pressure_pa=last['pressure'],work_difference_j=error,base=base,control=fine,force_difference_n=float(np.linalg.norm(control-force)),open_origin_full_force_exact_parity=True,scope='oldA6 authenticated diagnostic input only; newintegration publicrestart must refuse oldcontract'))
    if os.environ.get('BIRTH_REPAIRED')=='1':assert max(abs(v) for v in error.values())<=1e-12


def test_prepared_uniform(monkeypatch):
    from anyfesolver.general_contact.s3_friction_prepared_path import prepare_s3_plane_pressure_path,PreparedS3PlanePressurePath
    case=build_case();a=owner(case);m=case[0]
    x=mode(m,(1,2,3),0);z=mode(m,(1,2,3),2)
    tx=mode(m,(5,6,7),0)
    # Full, partial, empty, near-zero closure and target deformation skins.
    comparisons=[];activated=0
    for shift,gap,deformation in ((0.,-5.2e-6,0.),(.55,-5.2e-6,0.),
            (1.05,-5.2e-6,0.),(.55,5e-6,0.),(.55,-2.00001e-7,0.),
            (.55,-5.2e-6,1e-5)):
        u=shift*x+gap*z+deformation*tx;frame=a._frame(u)
        path=prepare_s3_plane_pressure_path(a,frame,frame,allow_partial=True)
        assert path is not None
        source,target=a.normal._positions(u)
        exact=a._partial_planar_pressure(source,target,frame['target'],frame['normal'],frame['origin'])
        fast=path._uniform_partial_pressure(source,target,frame['target'],frame['normal'],frame['origin'])
        if fast is not None:
            activated+=1
            assert np.max(abs(fast-exact))<=1e-12
        comparisons.append(dict(shift_m=shift,source_z_m=gap,fast_eligible=fast is not None))
    assert activated==6
    # Any nonuniform physical skin gap must use ordinary moments.
    bad=source.copy();bad[1,2]+=1e-10
    assert path._uniform_partial_pressure(bad,target,frame['target'],frame['normal'],frame['origin']) is None
    a.commit(a.prepare_commit(-5.2e-6*z));a.commit(a.prepare_commit(1.05*x-5.2e-6*z))
    u=.55*x-5.2e-6*z
    before=a.checkpoint_state();force,_,summary=a._integrate(u)
    a._event_frame_cache.clear()
    a._material_history_cache.clear()
    monkeypatch.setattr(PreparedS3PlanePressurePath,'_uniform_partial_pressure',lambda *args:None)
    ordinary,_,control=a._integrate(u)
    assert np.linalg.norm(force-ordinary)<=2e-9
    for field in ('dissipation_increment_j','storage_j','released_storage_j'):
        assert abs(summary[field]-control[field])<=1e-12
    assert a.checkpoint_state()==before and summary['prepared_uniform_pressure_samples']>0
    save(dict(comparisons=comparisons,nonuniform_fallback=True,force_difference_n=float(np.linalg.norm(force-ordinary)),
              accelerated=summary,ordinary=control,history_unchanged=True))

@pytest.mark.skipif(os.environ.get('ATOMIC_KIND')=='trial' and (not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A2.json').is_file()),
    reason='archived rim input AUTOMATIC_WORKFLOW_EVIDENCE/A2.json absent: trial-branch atomic replay requires the original staged checkpoint')
def test_atomic(monkeypatch):
    from anyfesolver.control import CancellationToken
    import anyfesolver.general_contact.s3_friction_static as factory
    import copy
    case=build_case();m=case[0];kind=os.environ.get('ATOMIC_KIND','commit');incoming=json.loads((root()/'A2.json').read_text())['checkpoint'] if kind=='trial' else None
    before=copy.deepcopy(incoming);declared=repr((m.boundary_conditions,m.constraint_equations,m.load_cases));nodes=[(n.x,n.y,n.z) for n in m.mesh.nodes.values()]
    original=factory.create_s3_friction_manager;token=CancellationToken();captured=[];fault=[];expected=[]
    def make(*args):
        a=original(*args);captured.append(a);restore=a.restore;commit=a.commit;integrate=a._integrate
        def arm():
            expected.append(copy.deepcopy(a.checkpoint_state()))
            if kind=='trial':
                def fail(*args,**kwargs):fault.append('trial');raise RuntimeError('injected rim trial failure')
                a._integrate=fail
            else:
                def fail(prepared):
                    commit(prepared);fault.append('commit returned')
                    if kind=='cancel':token.cancel('injected rim late cancel');return
                    raise RuntimeError('injected rim commit-return failure')
                a.commit=fail
        def restored(*args,**kwargs):restore(*args,**kwargs);arm()
        a.restore=restored
        if incoming is None:arm()
        return a
    monkeypatch.setattr(factory,'create_s3_friction_manager',make)
    with pytest.raises(Exception) as error:solve_case(case,3 if kind=='trial' else 1,incoming,token)
    assert fault and 'injected rim' in str(error.value),repr(error.value)
    if kind=='cancel':
        from anyfesolver.control import SolveCancelled
        assert isinstance(error.value,SolveCancelled)
    assert captured[0].checkpoint_state()==expected[0]
    assert incoming==before and repr((m.boundary_conditions,m.constraint_equations,m.load_cases))==declared
    assert [(n.x,n.y,n.z) for n in m.mesh.nodes.values()]==nodes
    assert getattr(m,'active_contact_manager',None) is None
    save(dict(kind=kind,fault_reached=fault,exception_type=type(error.value).__name__,reason=str(error.value),physical_history_entry_restored=True,declared_inputs_restored=True,scope='exact rim elastic owned mapping/publication guard'))

def test_refusal(monkeypatch):
    case=list(build_case());m,d,s,f,p=case
    old=S3ReplayFrictionSettings(**vars(f));case[3]=old
    with pytest.raises(ValueError,match='S3SlidingFrictionSettings'):solve_case(case,1)
    case[3]=f;case[2]=replace(s,shell_pair_orientation='physical_sides_dev')
    with pytest.raises(ValueError,match='legacy parallel'):solve_case(case,1)
    case[2]=s
    with pytest.raises(ValueError):solve_case(case,0)
    save(dict(old_replay_predictor_refused=True,nonlegacy_orientation_refused=True,zero_target_refused=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A2.json').is_file(),
    reason='archived rim inputs AUTOMATIC_WORKFLOW_EVIDENCE/A2.json absent: refusal replay requires the original staged checkpoint')
def test_archive_checkpoint_refusals():
    case=list(build_case());m,d,s,f,p=case
    cp=json.loads((root()/'A2.json').read_text())['checkpoint'];original=json.dumps(cp,sort_keys=True)
    case[2]=replace(s,path_predictor='none')
    with pytest.raises(ValueError,match='incompatible|match|contract|fingerprint'):solve_case(case,2,cp)
    assert json.dumps(cp,sort_keys=True)==original
    old_cp=json.loads((root().parent/'A2.json').read_text())['checkpoint'];old_bytes=json.dumps(old_cp,sort_keys=True);case[2]=s
    with pytest.raises(ValueError,match='incompatible|match|contract|fingerprint'):solve_case(case,2,old_cp)
    assert json.dumps(old_cp,sort_keys=True)==old_bytes
    save(dict(predictor_checkpoint_incompatible=True,supplied_checkpoint_unchanged=True))

@pytest.mark.skipif(not os.environ.get('FINITE_STAGE'),
    reason='installed rim stage input FINITE_STAGE absent: staged rim proof requires the original workflow stage selection')
def test_proof():
    from fractions import Fraction as Q
    case=build_case();m=case[0];a=owner(case);inventory=[];geometry=[]
    count=6 if os.environ['FINITE_STAGE']=='prefixproof' else 7
    for step in range(1,count+1):
        data=json.loads((root()/f'A{step}.json').read_text());inventory.extend(x['step_index'] for x in data['snapshots']);u=np.array(data['checkpoint']['displacements'])
        # Independent declared linear skin reconstruction, all rotations zero.
        source=np.array([[m.mesh.get_node(n).x,m.mesh.get_node(n).y,m.mesh.get_node(n).z-.035] for n in (1,2,3)])+u.reshape(7,6)[:3,:3]
        target=np.array([[m.mesh.get_node(n).x,m.mesh.get_node(n).y,m.mesh.get_node(n).z+.015] for n in (4,5,6,7)])+u.reshape(7,6)[3:,:3]
        actual_source,actual_target=a.normal._positions(u)
        assert np.max(abs(actual_source-source))<=1e-17 and np.max(abs(actual_target-target))<=1e-17
        gaps=[Q(float(x[2]))-Q(float(y[2])) for x in actual_source for y in actual_target]
        assert min(gaps)>=-Q(5e-5)
        if step==6:assert min(gaps)>0
        assert np.all(u.reshape(7,6)[:,3:]==0)
        geometry.append(dict(step=step,exact_binary_gap_lower_m=float(min(gaps)),target_deformation_norm_m=data['physical']['target_inplane_norm_m'],contact_norm=data['physical']['contact_norm']))
    assert inventory==list(range(1,count+1))
    if count==6:
        save(dict(inventory=inventory,independent_geometry=geometry,scope='accepted press/stick/partialslip/rimexit/reverse/strictopen prefix; coupledrecontact remains UNMET'))
        return
    terminal=json.loads((root()/'A7.json').read_text())['checkpoint'];prior=json.loads((root()/'A6.json').read_text())['checkpoint'];a.restore(prior['path_state']['friction_state'],np.array(prior['displacements']));u=np.array(terminal['displacements']);frame=a._frame(u)
    pressure=frame['pressure'];assert np.max(abs(pressure-pressure[0]))<1e-12
    cap=.3*max(pressure);width=cap/1000.;y=.25;rim=(1-y)*frame['target'][3][0,0]+y*frame['target'][3][1,0]
    delta=width*np.array((.1,.5,1.,2.,10.));source=frame['source'][0];b=(rim-source[0,0]-delta)/(source[1,0]-source[0,0]);c=np.full(5,(y-source[0,1])/(source[2,1]-source[0,1]));bary=np.column_stack((1-b-c,b,c))
    result=a.query_material(0,bary,candidate=frame);actual=result[1][:,0];expected=np.minimum(1000*delta,cap)
    assert np.max(abs(actual-expected))<=1e-10
    # UY and skinZ fixed: rim trace and its advection are affine in y and t.
    # Late rim birth has t_entry>t_normal; delta(t)/p(t) is monotone. Thus this
    # actual loaded front is the same scalar min(k*distance,mu*p_end) limit.
    length=.4*np.sqrt(1+(frame['target'][3][1,0]-frame['target'][3][0,0])**2)
    force_bound=.5*cap*width*length;storage_bound=cap*cap/(2*1000)*width*length
    assert force_bound<2e-9 and storage_bound<1e-12
    save(dict(inventory=inventory,independent_geometry=geometry,loaded_pressure_pa=pressure,loaded_front_actual_pa=actual,loaded_front_independent_pa=expected,loaded_front_width_m=width,loaded_strip_force_upper_n=force_bound,loaded_strip_storage_upper_j=storage_bound,scope='accepted binary nodal skins and scalar1D rimfront; not global nonlinear/workintegral acceptance'))

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not all((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(name+'.json')).is_file() for name in ('A2','A3')),
    reason='archived rim inputs AUTOMATIC_WORKFLOW_EVIDENCE A2.json/A3.json absent: facet replay requires the original staged checkpoints')
def test_facet(monkeypatch):
    import anyfesolver.general_contact.s3_friction_static as factory
    original=factory.create_s3_friction_manager;captured=[]
    def make(*args):a=original(*args);captured.append(a);return a
    monkeypatch.setattr(factory,'create_s3_friction_manager',make)
    states=[]
    for step in (2,3):
        cp=json.loads((root()/f'A{step}.json').read_text())['checkpoint'];result=solve_case(build_case(),step,cp)
        assert result.status=='completed';a=captured[-1];before=a.checkpoint_state();state=a.query_material(0,np.array(((1/3,1/3,1/3),)))
        assert a.checkpoint_state()==before
        states.append(dict(step=step,target_element=int(state['element'][0]),anchor_bary=state['bary'][0],slip_m=float(state['slip'][0])))
    assert [x['target_element'] for x in states]==[2,3]
    save(dict(material_source_nodes=(1,2,3),source_bary=(1/3,1/3,1/3),actual_public_restored_field=states,scope='same material feature crosses conforming target facet; accepted prefix only'))
