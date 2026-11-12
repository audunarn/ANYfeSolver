"""Prepared development controls; per-selector execution evidence is external.

The scalar reference uses physical element masses and the two distributed line
potentials. It does not call a contact kernel or reuse a production work map.
"""
from dataclasses import fields,replace
import json,os
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_multiple_beam_s3_transient import build,run
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import (ContactTransientSolverControls,
    solve_transient_contact)
from anyfesolver.general_contact.multiple_beam_s3_transient import (
    PROFILE,MultipleBeamS3TransientManager,multiple_hermite_gap_bounds,pair_definitions)

FORCE_SCALE=.0025
MOMENT_SCALE=.005
WORK_SCALE=1.e-6


def write_report(tmp_path,name,data):
    target=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/(name+'.json'))))
    # Preserve separate producer, metric and work evidence when pytest shares
    # one configured result path. The configured path is only a report index.
    stage='_'+data['stage'] if 'stage' in data else ''
    report=target.with_name(target.stem+'_'+name+stage+target.suffix)
    report.write_text(json.dumps(data,indent=2),encoding='utf-8')
    inventory=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'reports':{}}
    inventory['reports'][name+stage]=str(report)
    target.write_text(json.dumps(inventory,indent=2),encoding='utf-8')


def coordinates(model,result):
    masters=(1,6,3)
    ids=[model.mesh.dof_manager.get_node_dofs(n)[2] for n in masters]
    return result.displacements[:,ids],result.velocities[:,ids]


def reference(times):
    """Independent active-set Newmark solve of three physical uniform modes."""
    # Beam rho*A*L=.0005 each; triangle area=4.5, rho*t*A=.09.
    M=np.diag((.0005,.0005,.09));C=np.array(((1.,0.,-1.),(0.,1.,-1.)))
    k=1.e5*.5;g0=.000005
    q=np.zeros(3);v=np.array((-.04,-.04,.01));a=np.zeros(3)
    qs=[q.copy()];vs=[v.copy()]
    for h in np.diff(times):
        predictor=q+h*v+.25*h*h*a
        candidates=[]
        for mask in range(4):
            ids=[i for i in range(2) if mask&(1<<i)]
            active=C[ids];K=k*active.T@active
            force=k*g0*active.sum(axis=0)
            trial=np.linalg.solve(M+.25*h*h*K,M@predictor-.25*h*h*force)
            gaps=g0+C@trial
            if all((gaps[i]<=0.) if i in ids else (gaps[i]>=0.) for i in range(2)):
                candidates.append(trial)
        assert candidates,'independent scalar active-set inventory unresolved'
        qn=candidates[0];an=(qn-predictor)/(.25*h*h)
        v=v+.5*h*(a+an);q,a=qn,an
        qs.append(q.copy());vs.append(v.copy())
    return np.array(qs),np.array(vs)


def physical_body_balance(model,result):
    """Shared shell counted once, actual inertia/support/structural residual."""
    from anyfesolver.assembly import assemble_mass_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.nonlinear_state import discard_active_state_candidate
    from anyfesolver.general_contact.runtime import suspend_contact
    M,_=assemble_mass_matrix(model);errors=[]
    for index in (0,int(np.argmin([r.minimum_gap for r in result.contact])),len(result.times)-1):
        u=result.displacements[index];inertia=np.asarray(M@result.accelerations[index]).reshape(-1)
        origin={}
        try:
            with suspend_contact():
                internal,_,_=_assemble_nonlinear_system(model,u,origin,3,tangent=False,kinematics='von_karman')
            row=result.contact[index]
            # Aggregate each selected physical element's contact load once.
            for eid in (1,3,2):
                pairs=[p for _,p in row.pairs if (p.shell_element if eid==2 else p.beam_element)==eid]
                force=sum((np.array(p.shell_force if eid==2 else p.beam_force) for p in pairs),np.zeros(3))
                moment=sum((np.array(p.shell_moment if eid==2 else p.beam_moment) for p in pairs),np.zeros(3))
                w=np.r_[force,moment]
                for node in model.mesh.elements[eid].node_ids:
                    ids=model.mesh.dof_manager.get_node_dofs(node)
                    position=np.array(model.mesh.nodes[node].coords())+u[ids[:3]]
                    vector=result.constraint_reactions[index,ids]-inertia[ids]-internal[ids]
                    w[:3]+=vector[:3];w[3:]+=vector[3:]+np.cross(position,vector[:3])
                errors.append(dict(state=index,element=eid,force=float(np.linalg.norm(w[:3])/FORCE_SCALE),
                    moment=float(np.linalg.norm(w[3:])/MOMENT_SCALE)))
        finally:
            discard_active_state_candidate(origin)
    return errors


def test_shared_skin_impact_reference_and_restart(tmp_path):
    write_report(tmp_path,'shared_skin_impact',dict(stage='test_entered',verified=False))
    model,result=run()
    # Preserve the actual producer before secondary reference/restart recovery.
    write_report(tmp_path,'shared_skin_impact',dict(stage='full_producer_completed',verified=False,
        profile=result.profile,times=result.times.tolist(),displacements=result.displacements.tolist(),
        velocities=result.velocities.tolist(),accelerations=result.accelerations.tolist(),
        contact=[r.to_dict() for r in result.contact],checkpoint=result.restart_checkpoint,
        pair_gaps=result.pair_swept_minimum_gaps))
    q,v=coordinates(model,result);qr,vr=reference(result.times)
    dx=float(np.max(np.abs(q-qr))/.000005)
    dv=float(np.max(np.abs(v-vr))/.05)
    active=[i for i,r in enumerate(result.contact) if all(p.active_line_length>0 for _,p in r.pairs)]
    open_final=all(p.active_line_length==0 and p.minimum_gap>0 for _,p in result.contact[-1].pairs)
    wrench=0.
    for row in result.contact:
        for _,pair in row.pairs:
            wrench=max(wrench,float(np.linalg.norm(np.array(pair.beam_force)+pair.shell_force))/FORCE_SCALE,
                float(np.linalg.norm(np.array(pair.beam_moment)+pair.shell_moment))/MOMENT_SCALE)
    # Choose an ACTUALLY accepted active time, not a requested rounded timestamp.
    assert active,'no simultaneous physical impact was represented'
    cut=active[len(active)//2];_,prefix=run(end=float(result.times[cut]))
    _,resumed=run(restart=prefix.restart_checkpoint)
    continuation=result.displacements[cut:]
    exact=(np.array_equal(resumed.displacements,continuation)
        and np.array_equal(resumed.velocities,result.velocities[cut:])
        and np.array_equal(resumed.accelerations,result.accelerations[cut:])
        and resumed.restart_checkpoint==result.restart_checkpoint
        and [r.to_dict() for r in resumed.contact]==[r.to_dict() for r in result.contact[cut:]])
    # The resumed first row has endpoint gap provenance; following rows retain
    # complete actual segment bounds exactly, including the shared pair names.
    split_paths=resumed.pair_swept_minimum_gaps[1:]==result.pair_swept_minimum_gaps[cut+1:]
    balances=physical_body_balance(model,result)
    M=np.array((.0005,.0005,.09));C=np.array(((1.,0.,-1.),(0.,1.,-1.)))
    energy=.5*np.sum(M*v*v,axis=1)+.5*50000.*np.sum(np.minimum(.000005+q@C.T,0.)**2,axis=1)
    energy_defect=float(np.max(np.abs(energy-energy[0]))/energy[0])
    report=dict(profile=result.profile,states=len(result.times),active_states=len(active),
        final_open=open_final,displacement_error=dx,velocity_error=dv,wrench_error=wrench,
        exact_split=exact,exact_split_paths=split_paths,pair_gaps=result.pair_swept_minimum_gaps,
        physical_body_balance=balances,uniform_mode_energy_defect=energy_defect,
        reference='independent three physical uniform modes; not flexural qualification')
    write_report(tmp_path,'shared_skin_impact',report)
    assert result.profile==PROFILE and dx<=1.e-6 and dv<=1.e-6
    assert wrench<=1.e-6 and open_final and exact and split_paths
    assert all(max(r['force'],r['moment'])<=1.e-6 for r in balances)
    assert energy_defect<=.01
    assert all(r.gap_tolerance_met for r in result.contact)
    assert all(set(row)=={'line_a','line_b'} for row in result.pair_swept_minimum_gaps)
    changed=replace(build()[2],beam_shell_rim_clearance=.03)
    m,d,_,velocity=build()
    with pytest.raises(ValueError,match='checkpoint.*incompatible'):
        solve_transient_contact(m,d,TransientConfig(dt=1.e-5,t_end=.0005,initial_velocity=velocity),
            contact_settings=changed,solver_controls=ContactTransientSolverControls(
                allow_multiple_deformable_beam_s3_dev=True,save_constraint_reactions_dev=True),
            restart_checkpoint=prefix.restart_checkpoint)


def test_complete_shared_work_and_single_pair_hidden_motion(tmp_path):
    # This local operator pose/direction intentionally does not obey the
    # uniform fixture MPCs. It checks the complete unconstrained pullback;
    # actual coupled response and constraints are checked in the impact test.
    model,definition,settings,_=build();manager=MultipleBeamS3TransientManager(model,definition,settings)
    size=model.mesh.dof_manager.total_dofs;u=np.zeros(size)
    dofs=lambda n:model.mesh.dof_manager.get_node_dofs(n)
    for n in (1,2,6,8,7):u[dofs(n)[2]]=-.000055
    for n,shift in ((3,0.),(4,1.e-7),(5,-1.e-7)):u[dofs(n)[2]]=shift
    r,H,row=manager.evaluate(u)
    vector=np.zeros(size)
    for n,value in ((1,.2),(2,-.1),(6,.3),(8,-.4),(7,.2),(3,-.3),(4,.1),(5,.2)):
        vector[dofs(n)[2]]=value
    # Physical displacement direction in metres; h is DIMENSIONLESS.
    # Component magnitudes .5--2um, finite differences .05--20nm.
    vector*=5.e-6
    errors=[]
    for h in (1.e-2,1.e-3,1.e-4):
        rp,_,p=manager.evaluate(u+h*vector);rm,_,m=manager.evaluate(u-h*vector)
        work=abs((p.line_contact_energy-m.line_contact_energy)/(2*h)-r@vector)/WORK_SCALE
        difference=(rp-rm)/(2*h)-H@vector
        force_rows=np.array([dofs(n)[:3] for n in model.mesh.nodes]).ravel()
        moment_rows=np.array([dofs(n)[3:6] for n in model.mesh.nodes]).ravel()
        ferr=np.linalg.norm(difference[force_rows])/FORCE_SCALE
        merr=np.linalg.norm(difference[moment_rows])/MOMENT_SCALE
        # dE/dh is J, dR/dh translational rows N and rotational rows Nm,
        # since h is dimensionless and the direction has length units.
        # Keep conventional derivative-relative error as an additional gate.
        relative=np.linalg.norm(difference)/max(np.linalg.norm(H@vector),1.e-30)
        errors.append(dict(h=h,work_error=float(work),force_rate_error=float(ferr),
            moment_rate_error=float(merr),relative_hessian_error=float(relative)))
    zero=np.zeros(size);rates=zero.copy();rates[dofs(8)[2]]=-.4
    bound,valid,visited,pairs=multiple_hermite_gap_bounds(manager,zero,zero,rates,-rates,.001)
    write_report(tmp_path,'shared_work',dict(errors=errors,path_valid=valid,path_bounds=pairs,visited=visited))
    assert all(x['work_error']<=1.e-5 and x['relative_hessian_error']<=1.e-5
               and x['force_rate_error']<=1.e-5 and x['moment_rate_error']<=1.e-5 for x in errors)
    assert not valid and pairs['line_b']<-1.e-5
    assert pairs['line_a']>=-definition.interactions[0].gap_tolerance
    # Actual global A-S and B-S crossblocks, no independent-element condensation.
    assert H[dofs(1)[2],dofs(3)[2]]!=0 and H[dofs(8)[2],dofs(3)[2]]!=0


def test_inventory_tolerances_guards_and_controls():
    model,definition,settings,velocity=build()
    different=replace(definition,interactions=(definition.interactions[0],
        replace(definition.interactions[1],gap_tolerance=2.e-5)))
    manager=MultipleBeamS3TransientManager(model,different,settings)
    assert [c.gap_tolerance for _,c in manager.pairs]==[1.e-5,2.e-5]
    with pytest.raises(ValueError,match='budget'):
        pair_definitions(model,definition,replace(settings,maximum_candidate_pairs=1))
    with pytest.raises(ValueError,match='duplicate physical'):
        pair_definitions(model,replace(definition,surfaces=definition.surfaces+
            (replace(definition.surfaces[0],name='alias'),)),settings)
    with pytest.raises(ValueError,match='duplicate physical|unreferenced'):
        pair_definitions(model,replace(definition,interactions=(definition.interactions[0],
            replace(definition.interactions[1],surface='tube_a'))),settings)
    for mutate in ('routing','map','shell_map','profile'):
        owner=MultipleBeamS3TransientManager(model,definition,settings)
        if mutate=='routing':owner.pairs=owner.pairs[:1]
        elif mutate=='map':owner.pairs[0][1].work_map.data[0]+=1.
        elif mutate=='shell_map':owner.pairs[0][1].shell.work_map.data[0]+=1.
        else:owner.profile='wrong'
        with pytest.raises(ValueError,match='mutat'):
            owner.evaluate(np.zeros(model.mesh.dof_manager.total_dofs))
        with pytest.raises(ValueError,match='mutat'):owner.restart_contract()
    with pytest.raises(ValueError,match='boolean'):
        ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=1)
    for field in fields(ContactTransientSolverControls):
        if field.name.endswith('_dev') and field.name not in (
                'allow_multiple_deformable_beam_s3_dev','save_constraint_reactions_dev'):
            with pytest.raises(ValueError,match='separate fixed-director'):
                ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True,
                                               **{field.name:True})
    for rotation,rate in ((1.e-5,0.),(0.,1.e-5)):
        m,d,s,v=build();u=np.zeros_like(v);rid=m.mesh.dof_manager.get_node_dofs(3)[3]
        u[rid]=rotation;v[rid]=rate
        with pytest.raises(ValueError,match='rotation|structural constraints'):
            solve_transient_contact(m,d,TransientConfig(dt=1.e-5,t_end=1.e-5,
                initial_displacement=u,initial_velocity=v),contact_settings=s,
                solver_controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True))


def test_complete_model_section_skin_and_dof_binding_refusal(monkeypatch):
    for mutation in ('section','actual_A','actual_orientation','kinematics','fiber','skin','routing'):
        model,definition,settings,_=build()
        manager=MultipleBeamS3TransientManager(model,definition,settings)
        if mutation=='section':model.mesh.elements[1].cross_section['area']*=2.
        elif mutation=='actual_A':model.mesh.elements[1]._A*=2.
        elif mutation=='actual_orientation':model.mesh.elements[1]._orientation=np.array((0.,0.,1.))
        elif mutation=='kinematics':model.mesh.elements[1]._geometric_nonlinearity='corotational'
        elif mutation=='fiber':model.mesh.elements[1]._fiber_plasticity={}
        elif mutation=='skin':model.mesh.elements[2].thickness*=2.
        else:
            dofs=model.mesh.dof_manager.get_node_dofs
            monkeypatch.setattr(model.mesh.dof_manager,'get_node_dofs',
                lambda node: dofs(2) if node==1 else dofs(node))
        with pytest.raises(ValueError,match='mutat'):
            manager.guard()
        with pytest.raises(ValueError,match='mutat'):
            multiple_hermite_gap_bounds(manager,np.zeros(model.mesh.dof_manager.total_dofs),
                np.zeros(model.mesh.dof_manager.total_dofs),np.zeros(model.mesh.dof_manager.total_dofs),
                np.zeros(model.mesh.dof_manager.total_dofs),1.e-5)
        monkeypatch.undo()


def test_rejected_path_exception_and_cancellation_cleanup(monkeypatch):
    # Use the actual shared solver boundary, with a single failed proof, then
    # check it retries from unchanged accepted u/v and releases runtime state.
    from anyfesolver.general_contact import multiple_beam_s3_transient as aggregate
    from anyfesolver.general_contact.runtime import _ACTIVE_CONTACT
    original=aggregate.multiple_hermite_gap_bounds;seen=[];rejected=[]
    def active_start(owner,start):
        child=owner.pairs[0][1]
        return start[child.beam_dofs[0,2]]-start[child.shell_dofs[0,2]]< -5.e-6
    def reject_once(owner,start,end,vstart,vend,dt,**kwargs):
        seen.append((start.copy(),vstart.copy()))
        if not rejected and active_start(owner,start):
            rejected.append(len(seen)-1)
            return -np.inf,False,1,{name:-np.inf for name,_ in owner.pairs}
        return original(owner,start,end,vstart,vend,dt,**kwargs)
    monkeypatch.setattr(aggregate,'multiple_hermite_gap_bounds',reject_once)
    _,result=run(end=.00016,emit_checkpoint=False)
    assert rejected
    j=rejected[0]
    assert result.cutbacks>=1 and np.array_equal(seen[j][0],seen[j+1][0])
    assert np.array_equal(seen[j][1],seen[j+1][1]) and _ACTIVE_CONTACT.get() is None
    # Propagating exception at a real path boundary discards the pending trial.
    def fail(owner,start,*args,**kwargs):
        if active_start(owner,start):raise RuntimeError('owned candidate failure')
        return original(owner,start,*args,**kwargs)
    monkeypatch.setattr(aggregate,'multiple_hermite_gap_bounds',fail)
    with pytest.raises(RuntimeError,match='owned candidate failure'):run(end=.00016)
    assert _ACTIVE_CONTACT.get() is None


def test_inflight_cancel_preserves_accepted_origin(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    import anyfesolver.general_contact.transient as driver
    from anyfesolver.general_contact import multiple_beam_s3_transient as aggregate
    from anyfesolver.general_contact.runtime import _ACTIVE_CONTACT
    token=CancellationToken();commits=[];calls=0
    commit=driver.commit_state_candidate;path=aggregate.multiple_hermite_gap_bounds
    def saved(origin,trial):
        commits.append(origin)
        return commit(origin,trial)
    def cancel(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==13:
            owner,start=args[:2];child=owner.pairs[0][1]
            assert start[child.beam_dofs[0,2]]-start[child.shell_dofs[0,2]]< -5.e-6
            token.cancel('owned multiple beam/S3 path')
            kwargs['check_cancel']()
        return path(*args,**kwargs)
    monkeypatch.setattr(driver,'commit_state_candidate',saved)
    monkeypatch.setattr(aggregate,'multiple_hermite_gap_bounds',cancel)
    sentinel=object();context=_ACTIVE_CONTACT.set(sentinel)
    try:
        with pytest.raises(SolveCancelled):run(end=.00016,cancellation_token=token)
        assert len(commits)==13 and _ACTIVE_CONTACT.get() is sentinel
    finally:_ACTIVE_CONTACT.reset(context)
