"""Physical-fibre work/history in the retained whole-curve beam-pair owner."""
from pathlib import Path
from hashlib import sha256
import json,os,runpy
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken
from anyfesolver.general_contact.native_retained_pair_static import FIBRE_WHOLE_PROFILE
from test_general_contact_native_fibre_s3 import _independent_material_history


def _owner(*,plastic=True,targets=(.05,),force=-.05,separation=.101,line_stiffness=100.,gap_step_control=False,
           reference_rise_m=0.,certify_curved_ownership=False,initial_trial=None):
    builder=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_native_retained_whole_pair.py'))['analysis']
    return builder(physical_fibres=True,plastic_fibres=plastic,targets=targets,force=force,
        separation=separation,gap_tolerance_m=5e-5,line_stiffness_n_per_m2=line_stiffness,
        gap_step_control=gap_step_control,reference_rise_m=reference_rise_m,
        certify_curved_ownership=certify_curved_ownership,initial_trial=initial_trial)


def test_fibre_pair_context_step_complete_constrained_rank_one_reference():
    owner=_owner();c=owner.context;assert c.count==84
    d=np.arange(1,c.count+1,dtype=float);v=np.linspace(-.2,.3,c.count);r=np.linspace(.1,-.1,c.count)
    j=np.diag(d)+np.outer(v,v);free=np.array(c.free)
    expected=np.zeros(c.count);df=d[free];vf=v[free];rf=r[free]
    expected[free]=-rf/df+(vf/df)*(np.sum(vf*rf/df)/(1+np.sum(vf*vf/df)))
    step,correction=c.step(owner.initial.mechanical,r,j)
    np.testing.assert_allclose(step,expected,rtol=1e-13,atol=1e-15)
    assert correction>0. and np.count_nonzero(step[c.nodal_count:])==48
    assert owner.profile==FIBRE_WHOLE_PROFILE


def test_fibre_pair_family_binding_and_empty_restart():
    from test_general_contact_native_retained_whole_pair_static import _owner as generalized
    physical=_owner();elastic=_owner(plastic=False)
    assert physical.identity!=elastic.identity
    raw=physical._checkpoint(())
    for other in (elastic,generalized(targets=(.05,),separation=.101,force=-.05)):
        with pytest.raises(ValueError,match='binding'):other._restore(raw,sha256(raw).hexdigest())
    state,records=physical._restore(raw,sha256(raw).hexdigest())
    assert state.completed_targets==0 and physical._checkpoint(records)==raw


def test_gap_step_control_affine_event_and_restart_binding():
    from anyfesolver.general_contact.native_retained_pair_static import _gap_limited_fraction
    # Exact scalar affine gap is independent of native search/force kernels.
    tolerance=5e-5;g0=.001;alpha=.5;gtrial=g0-.00556805556*alpha
    fraction=_gap_limited_fraction(alpha,g0,gtrial,tolerance)
    assert abs(g0-.00556805556*fraction+.5*tolerance)<1e-15
    assert 0.<fraction<alpha
    # When already active, advance only inside the remaining budget.
    active=-.8*tolerance
    fraction=_gap_limited_fraction(.5,active,-.001,tolerance)
    assert .05<=fraction<=.45
    legacy=_owner();event=_owner(gap_step_control=True)
    raw=legacy._checkpoint(())
    with pytest.raises(ValueError,match='binding'):
        event._restore(raw,sha256(raw).hexdigest())
    guarded=event._checkpoint(())
    restored,records=event._restore(guarded,sha256(guarded).hexdigest())
    assert restored.completed_targets==0 and event._checkpoint(records)==guarded
    event.gap_step_control=False
    with pytest.raises(ValueError,match='settings changed'):event._guard()


@pytest.mark.parametrize('gap_step_control',(False,True))
def test_gap_geometry_query_observes_cancellation_before_consumption(monkeypatch,gap_step_control):
    from anyfesolver.control import SolveCancelled
    owner=_owner(gap_step_control=gap_step_control);token=CancellationToken()
    before=canonical(owner.initial.histories);raw=owner._checkpoint(())
    def cancel_inside_geometry(*args,**kwargs):
        token.cancel('during geometry-only pair query')
        return .001
    monkeypatch.setattr(owner.contact,'geometry_gap_lower',cancel_inside_geometry)
    with owner.context.operation(token):
        owner.context.activate_cancellation(token)
        with pytest.raises(SolveCancelled):owner._swept_gap(owner.initial.mechanical)
    assert canonical(owner.initial.histories)==before and owner._checkpoint(())==raw
    assert owner.context._token is None and not owner.context._lock.locked()


@pytest.mark.parametrize('gap_step_control',(False,True))
def test_swept_midpoint_certifies_full_pair_without_force_evaluation(monkeypatch,gap_step_control):
    owner=_owner(gap_step_control=gap_step_control)
    before=canonical(owner.initial.histories);raw=owner._checkpoint(())
    def forbidden_force(_pose):
        raise AssertionError('swept midpoint must not construct force/tangent')
    monkeypatch.setattr(owner,'_contact',forbidden_force)
    lower=owner._swept_gap(owner.initial.mechanical)
    # Two initially straight, parallel centreline tubes: independent gap.
    exact=.101-owner.contact.radius
    assert exact-1e-10<=lower<=exact
    assert canonical(owner.initial.histories)==before and owner._checkpoint(())==raw


@pytest.mark.skipif(not os.environ.get('CONTACT_PAIR_SAVED_TRIAL'),
    reason='explicit saved active candidate geometry-cost check')
def test_saved_active_pair_gap_query_matches_mechanical_result():
    import time
    path=Path(os.environ['CONTACT_PAIR_SAVED_TRIAL'])
    raw=path.read_bytes();owner=_owner(line_stiffness=4000.,gap_step_control=True)
    state=owner.context.make(json.loads(raw))
    before=canonical(owner.initial.histories)
    started=time.monotonic();full=owner._contact(state);full_seconds=time.monotonic()-started
    started=time.monotonic();gap=owner._swept_gap(state);geometry_seconds=time.monotonic()-started
    # A different conservative certificate can be tighter than the scan.
    # Force admission is unchanged and this pose must satisfy both checks.
    assert gap<=full.minimum_sampled_gap_m+1e-13
    assert gap>=-owner.gap_tolerance and full.conservative_lower_gap_m>=-owner.gap_tolerance
    assert canonical(owner.initial.histories)==before
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:Path(output+'-saved-gap.json').write_text(json.dumps(dict(
        scope='one saved candidate comparison; not whole-route performance',
        trial_sha256=sha256(raw).hexdigest(),full_seconds=full_seconds,
        geometry_seconds=geometry_seconds,gap_m=gap),indent=2))


@pytest.mark.skipif(not os.environ.get('CONTACT_PAIR_SAVED_TRIAL'),
    reason='explicit saved candidate global chord certificate discriminator')
def test_saved_active_pair_global_chord_gap_certificate():
    import time
    from anyfesolver.general_contact.beam_curve_pair import _source_interval_target_gap_lower
    raw=Path(os.environ['CONTACT_PAIR_SAVED_TRIAL']).read_bytes()
    owner=_owner(line_stiffness=4000.,gap_step_control=True)
    state=owner.context.make(json.loads(raw))
    curves=tuple(cell.evaluate(state.positions[owner.context.nodes[index]],
        state.position_low[owner.context.nodes[index]],state.cell_rotations[index]).positions
        for cells,index in zip((owner.contact.cells_a,owner.contact.cells_b),owner.indices)
        for cell in cells)
    started=time.monotonic()
    lower=min(_source_interval_target_gap_lower(source,0.,1.,target,owner.contact.radius)
        for source in curves[:2] for target in curves[2:])
    seconds=time.monotonic()-started
    reference=json.loads(Path(os.environ['CONTACT_PAIR_SAVED_GAP_RECORD']).read_text())
    assert sha256(raw).hexdigest()==reference['trial_sha256']
    # This saved fixture consists of straight physical half-cells. Compare
    # to their actual distance, not another independently conservative bound.
    for curve in curves:
        np.testing.assert_allclose(curve[1],.5*(curve[0]+curve[2]),rtol=0.,atol=1e-13)
    distances=[]
    for first in curves[:2]:
        for second in curves[2:]:
            p,p1=first[[0,2]];q,q1=second[[0,2]];v=p1-p;w=q1-q
            distances.extend(np.linalg.norm(x-(q+np.clip((x-q)@w/(w@w),0.,1.)*w))
                for x in (p,p1))
            distances.extend(np.linalg.norm(x-(p+np.clip((x-p)@v/(v@v),0.,1.)*v))
                for x in (q,q1))
            parameters=np.linalg.lstsq(np.column_stack((v,-w)),q-p,rcond=None)[0]
            if np.all(parameters>=0.) and np.all(parameters<=1.):
                distances.append(np.linalg.norm(p+parameters[0]*v-q-parameters[1]*w))
    independent_gap=min(distances)-owner.contact.radius
    assert lower<=independent_gap+1e-13
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:Path(output+'-global-chord.json').write_text(json.dumps(dict(
        scope='saved pose conservative-bound discriminator only',lower_m=lower,
        full_gap_m=reference['gap_m'],independent_gap_m=float(independent_gap),
        gap_budget_m=owner.gap_tolerance,
        seconds=seconds,trial_sha256=sha256(raw).hexdigest()),indent=2))


def test_fibre_pair_cancellation_and_precommit_exception_restore_origin(monkeypatch):
    owner=_owner();raw=owner._checkpoint(());origin=canonical(owner.initial.histories)
    token=CancellationToken();token.cancel('fibre-pair focused cancellation')
    failed=owner.solve(cancellation_token=token)
    assert failed.status=='failed' and 'SolveCancelled' in failed.failure
    assert failed.checkpoint==raw and canonical(failed.accepted.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked() and not owner.context._cancel_active
    owner=_owner(targets=(0.,));raw=owner._checkpoint(())
    def reject(*args,**kwargs):raise RuntimeError('fibre pair precommit interruption')
    monkeypatch.setattr(owner,'_record',reject)
    failed=owner.solve()
    assert failed.status=='failed' and 'precommit interruption' in failed.failure
    assert failed.checkpoint==raw and failed.accepted.completed_targets==0
    assert owner.context._token is None and not owner.context._lock.locked()


def test_fibre_pair_cancellation_after_checkpoint_staging_restores_origin(monkeypatch):
    owner=_owner(targets=(0.,));raw=owner._checkpoint(())
    origin=canonical(owner.initial.histories)
    token=CancellationToken();checkpoint=owner._checkpoint
    def stage_then_cancel(records):
        staged=checkpoint(records)
        if records:token.cancel('after staged fibre target')
        return staged
    monkeypatch.setattr(owner,'_checkpoint',stage_then_cancel)
    failed=owner.solve(cancellation_token=token)
    assert failed.status=='failed' and 'SolveCancelled' in failed.failure
    assert failed.checkpoint==raw and failed.accepted.completed_targets==0
    assert canonical(failed.accepted.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked() and not owner.context._cancel_active


def test_fibre_pair_complete_spatial_tangent_and_independent_material_trial():
    owner=_owner();c=owner.context
    step=np.zeros(c.count)
    step[6*c.node_ids.index(5)+1]=-.0008
    step[6*c.node_ids.index(6)+1]=-.002
    for slots in c.slots:step[np.asarray(slots)[24:42]]=np.linspace(-.001,.001,18)
    state=c.advance(owner.initial.mechanical,step);origins=owner.initial.histories
    before=canonical(origins)
    r,k,_,responses,contact=owner._assemble(state,.05,origins)
    assert contact.potential_j>0. and np.count_nonzero(contact.tangent[:42,42:])
    report,proposed=_independent_material_history(owner,state,origins)
    assert report['yielded']>0. and proposed==tuple(station for response in responses for station in response.history.stations)
    direction=np.random.default_rng(721).normal(size=c.count)*1e-4
    direction[list(c._backend.layout.fixed)]=0.
    errors=[]
    for h in (2e-6,1e-6):
        values=[owner._assemble(c.advance(state,sign*h*direction),.05,origins)[0] for sign in (1,-1)]
        fd=(values[0]-values[1])/(2*h);error=float(np.linalg.norm(k@direction-fd)/np.linalg.norm(fd));errors.append(error)
        assert error<=1e-5
    assert canonical(origins)==before
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record+'-fibre-pair-local.json').write_text(json.dumps(dict(
        material=report,directional_errors=errors,energy_j=contact.potential_j),indent=2))


def test_fibre_pair_prepared_contact_exact_pose_parity(monkeypatch):
    owner=_owner();state=owner.initial.mechanical;origins=owner.initial.histories
    reference=owner._assemble(state,.05,origins)
    def reject_recalculation(*args):raise AssertionError('duplicate candidate contact')
    monkeypatch.setattr(owner,'_contact',reject_recalculation)
    prepared=owner._assemble(state,.05,origins,prepared_contact=(state,reference[4]))
    np.testing.assert_array_equal(prepared[0],reference[0])
    np.testing.assert_array_equal(prepared[1],reference[1])
    assert prepared[2]==reference[2] and prepared[4] is reference[4]
    other=owner.context.advance(state,np.zeros(owner.context.count))
    with pytest.raises(ValueError,match='exact candidate pose'):
        owner._assemble(other,.05,origins,prepared_contact=(state,reference[4]))


def test_fibre_pair_cancel_before_complete_evaluation_reuse(monkeypatch):
    # Small, entirely open motion isolates transaction control from contact
    # admission. It does not replace the active coupled benchmark.
    owner=_owner(force=-.0001);raw=owner._checkpoint(())
    origin=canonical(owner.initial.histories);token=CancellationToken()
    assemble=owner._assemble;calls=[]
    def assemble_then_cancel(*args,**kwargs):
        calls.append(kwargs.get('prepared_contact') is not None)
        result=assemble(*args,**kwargs)
        if calls[-1]:token.cancel('before complete evaluation reuse')
        return result
    monkeypatch.setattr(owner,'_assemble',assemble_then_cancel)
    failed=owner.solve(cancellation_token=token)
    assert calls==[False,True]
    assert failed.status=='failed' and 'SolveCancelled' in failed.failure
    assert failed.checkpoint==raw and failed.accepted.completed_targets==0
    assert canonical(failed.accepted.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked()


def test_fibre_pair_small_open_motion_prepared_record_replay(monkeypatch):
    # Completing fibre route on entirely open motion: the converged
    # prepared record evaluation must publish the accepted target
    # without a second contact evaluation of the accepted pose,
    # and restore replay must re-assemble to byte-identical records.
    import sys
    owner=_owner(force=-.0001);origin=canonical(owner.initial.histories)
    contact=owner._contact;record=owner._record
    callers=[];records=[]
    def counted(state):
        callers.append(sys._getframe(1).f_code.co_name);return contact(state)
    def traced_record(*args,**kwargs):
        records.append(kwargs.get('prepared') is not None)
        return record(*args,**kwargs)
    monkeypatch.setattr(owner,'_contact',counted)
    monkeypatch.setattr(owner,'_record',traced_record)
    run=owner.solve()
    assert run.status=='completed' and run.accepted.completed_targets==1,run.failure
    assert records==[True] and callers
    restored,replayed=owner._restore(run.checkpoint,run.checkpoint_sha256)
    assert records==[True,False]
    assert restored.histories==run.accepted.histories
    assert owner._checkpoint(replayed)==run.checkpoint
    assert canonical(owner.initial.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked()

def test_fibre_pair_prepared_record_rejects_foreign_identity_and_rechecks_guard(monkeypatch):
    # Commit boundary of the cached record bundle: only the exact converged
    # pose, target parameter and material origins may supply it; every
    # foreign identity refuses before any checkpoint work, and the cached
    # branch rechecks the guard exactly as the assembly path does.
    owner=_owner(force=-.0001);origin=canonical(owner.initial.histories)
    captured=[]
    record=owner._record
    def traced(*args,**kwargs):
        if kwargs.get('prepared') is not None and not captured:
            captured.append(kwargs['prepared'])
        return record(*args,**kwargs)
    monkeypatch.setattr(owner,'_record',traced)
    run=owner.solve()
    assert run.status=='completed' and run.accepted.completed_targets==1,run.failure
    pose,parameter,origins,bundle=captured[0]
    for foreign in ((owner.initial.mechanical,parameter,origins,bundle),
                    (pose,parameter+1.,origins,bundle),
                    (pose,parameter,object(),bundle)):
        with pytest.raises(ValueError,
                match='exact accepted pose, target and origins'):
            owner._record(1,pose,origins,'0'*64,prepared=foreign)
    owner.gap_step_control=not owner.gap_step_control
    with pytest.raises(ValueError,match='settings changed'):
        owner._record(1,pose,origins,'0'*64,
            prepared=(pose,parameter,origins,bundle))
    owner.gap_step_control=not owner.gap_step_control
    restored,replayed=owner._restore(run.checkpoint,run.checkpoint_sha256)
    assert owner._checkpoint(replayed)==run.checkpoint
    assert canonical(owner.initial.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked()


def test_fibre_pair_actual_coupled_commit_and_restart():
    # Prospective resolved numerical enforcement, unchanged physical fixture
    # and dimensional gap/equilibrium/history gates. The former stiffness100
    # success expectation was contradicted by preserved gap-limit failures.
    _check_coupled_commit_and_restart(_owner(line_stiffness=4000.))


def test_fibre_pair_staged_coupled_commit_and_restart():
    _check_coupled_commit_and_restart(_owner(targets=(.005,.01,.02,.03,.04,.05),
        line_stiffness=4000.))


def test_fibre_pair_weak_normal_regularization_is_not_reported_completed():
    # Keep the original100 fixture as an explicit non-success regression.
    # No unsupported converged equilibrium is invented: the bounded solver
    # rejects its attempted path before committing the first physical target.
    owner=_owner();origin=canonical(owner.initial.histories)
    raw=owner._checkpoint(())
    run=owner.solve()
    assert run.status=='failed' and run.accepted.completed_targets==0
    assert 'line-search/gap limit' in run.failure,run.failure
    assert canonical(owner.initial.histories)==origin
    assert canonical(run.accepted.histories)==origin
    assert run.checkpoint==raw
    assert owner.context._token is None and not owner.context._lock.locked()


def test_fibre_pair_refined_coupled_commit_and_restart():
    _check_coupled_commit_and_restart(_owner(line_stiffness=4000.))


def test_fibre_pair_event_limited_coupled_commit_and_restart(monkeypatch):
    owner=_owner(line_stiffness=4000.,gap_step_control=True)
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        _trace_refined_pair(owner,monkeypatch)
    _check_coupled_commit_and_restart(owner)


@pytest.mark.skipif(not os.environ.get('CONTACT_PAIR_REFINEMENT_REFERENCE'),
    reason='explicit same-mechanics normal-enforcement sensitivity case')
def test_fibre_pair_event_normal_enforcement_refinement(monkeypatch):
    reference=json.loads(Path(os.environ['CONTACT_PAIR_REFINEMENT_REFERENCE']).read_text())
    owner=_owner(line_stiffness=8000.,gap_step_control=True)
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):_trace_refined_pair(owner,monkeypatch)
    record=_check_coupled_commit_and_restart(owner)
    errors={}
    for key in ('force_a_n','moment_a_nm'):
        previous=np.array(reference[key]);current=np.array(record[key])
        errors[key]=float(np.linalg.norm(current-previous)/np.linalg.norm(previous))
    before=np.array(reference['positions']);after=np.array(record['positions'])
    reference_positions=owner.initial.mechanical.positions+owner.initial.mechanical.position_low
    for index,name in ((2,'beam_a_tip'),(5,'beam_b_tip')):
        errors[name]=float(np.linalg.norm(after[index]-before[index])/
            np.linalg.norm(before[index]-reference_positions[index]))
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:Path(output+'-enforcement-sensitivity.json').write_text(json.dumps(dict(
        scope='one fixed-discretization development enforcement comparison only',
        reference_checkpoint_sha256=reference['checkpoint_sha256'],
        current_checkpoint_sha256=record['checkpoint_sha256'],
        stiffnesses_n_per_m2=[4000.,8000.],response_target=.02,errors=errors),indent=2))
    assert max(errors.values())<=.02


def _check_coupled_commit_and_restart(owner,*,restart_factory=None):
    origins=owner.initial.histories;before=canonical(origins)
    run=owner.solve();output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    record=dict(status=run.status,failure=run.failure,completed=run.accepted.completed_targets,
        profile=run.profile,targets=list(owner.program.targets),line_stiffness_n_per_m2=owner.contact.stiffness)
    if output:
        Path(output+'-fibre-pair.json').write_text(json.dumps(record,indent=2))

        Path(output+'-fibre-pair.checkpoint').write_bytes(run.checkpoint)
    assert run.status=='completed' and run.accepted.completed_targets==len(owner.program.targets),run.failure
    contact=run.accepted.contact
    assert contact.potential_j>0. and contact.conservative_lower_gap_m>=-5e-5
    assert np.linalg.norm(contact.beam_a_force_n+contact.beam_b_force_n)/.0025<=1e-6
    assert np.linalg.norm(contact.beam_a_moment_nm+contact.beam_b_moment_nm)/.005<=1e-6
    material,proposed=_pair_independent_history(owner,run.accepted,run.checkpoint)
    assert material['yielded']>0. and proposed==tuple(station for cell in run.accepted.histories for station in cell.stations)
    assert canonical(origins)==before
    resumed=(restart_factory() if restart_factory is not None else
        _owner(targets=owner.program.targets,line_stiffness=owner.contact.stiffness,
            gap_step_control=owner.gap_step_control))
    restored,records=resumed._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.histories==run.accepted.histories and owner._checkpoint(records)==run.checkpoint
    assert owner.context._token is None and not owner.context._lock.locked()
    if output:
        record.update(material=material,gap_lower_m=contact.conservative_lower_gap_m,
            checkpoint_sha256=run.checkpoint_sha256,force_a_n=contact.beam_a_force_n.tolist(),
            moment_a_nm=contact.beam_a_moment_nm.tolist(),
            positions=(run.accepted.mechanical.positions+run.accepted.mechanical.position_low).tolist())
        Path(output+'-fibre-pair.json').write_text(json.dumps(record,indent=2))
    return record


def _pair_independent_history(owner,accepted,checkpoint):
    if accepted.completed_targets==1:
        origins=owner.initial.histories
    else:
        records=tuple(json.loads(checkpoint)['records'][:-1])
        prefix=owner._checkpoint(records)
        preceding,_=owner._restore(prefix,sha256(prefix).hexdigest())
        origins=preceding.histories
    return _independent_material_history(owner,accepted.mechanical,origins)


@pytest.mark.skipif(not os.environ.get('CONTACT_PAIR_ACCEPTED_CHECKPOINT'),
    reason='explicit retained accepted-state verification, no coupled solve replay')
def test_event_limited_saved_checkpoint_material_work_and_restart():
    raw=Path(os.environ['CONTACT_PAIR_ACCEPTED_CHECKPOINT']).read_bytes()
    owner=_owner(line_stiffness=4000.,gap_step_control=True)
    origin=canonical(owner.initial.histories)
    accepted,records=owner._restore(raw,sha256(raw).hexdigest())
    assert accepted.completed_targets==1 and owner._checkpoint(records)==raw
    contact=accepted.contact
    assert contact.potential_j>0. and contact.conservative_lower_gap_m>=-5e-5
    force_error=float(np.linalg.norm(contact.beam_a_force_n+contact.beam_b_force_n)/.0025)
    moment_error=float(np.linalg.norm(contact.beam_a_moment_nm+contact.beam_b_moment_nm)/.005)
    assert max(force_error,moment_error)<=1e-6
    material,proposed=_pair_independent_history(owner,accepted,raw)
    assert material['yielded']>0. and proposed==tuple(
        station for cell in accepted.histories for station in cell.stations)
    assert canonical(owner.initial.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked()
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:Path(output+'-accepted-pair.json').write_text(json.dumps(dict(
        scope='accepted source-state verification; original postprocessing failure retained',
        completed_targets=accepted.completed_targets,material=material,
        checkpoint_sha256=sha256(raw).hexdigest(),force_error=force_error,moment_error=moment_error,
        potential_j=contact.potential_j,gap_lower_m=contact.conservative_lower_gap_m,
        force_a_n=contact.beam_a_force_n.tolist(),moment_a_nm=contact.beam_a_moment_nm.tolist(),
        positions=(accepted.mechanical.positions+accepted.mechanical.position_low).tolist()),indent=2))


@pytest.mark.skipif(os.environ.get('ANYSOLVER_CONTACT_PROFILE')!='1',
    reason='explicit bounded first-Newton cost diagnostic, not qualification')
def test_fibre_pair_first_newton_cost_discriminator():
    import cProfile,time
    output=Path(os.environ['CONTACT_APP_RESULT_RECORD']+'-first-newton-cost.json')
    rows=[]
    owner=_owner();origin=canonical(owner.initial.histories)
    def measured(label,call):
        profile=cProfile.Profile();started=time.monotonic();profile.enable()
        try:
            value=call();row=dict(operation=label,status='completed')
            return value
        except Exception as exc:
            row=dict(operation=label,status=type(exc).__name__,failure=str(exc))
            raise
        finally:
            profile.disable();profile.dump_stats(str(output)+f'.{len(rows)}.prof')
            row['elapsed_s']=time.monotonic()-started;rows.append(row)
            output.write_text(json.dumps(dict(scope='first Newton cost diagnostic only',operations=rows),indent=2))
    # The external bounded supervisor harness owns crash watchdog duty;
    # no recurring in-process faulthandler timer perturbs the measurements.
    try:
        r,k,metrics,_,contact=measured('initial_assembly',
            lambda:owner._assemble(owner.initial.mechanical,.05,owner.initial.histories))
        step,correction=measured('initial_step',
            lambda:owner.context.step(owner.initial.mechanical,r,k))
        for cut in range(owner.program.max_backtracks+1):
            increment=step*(.5**cut)
            candidate=owner.context.advance(owner.initial.mechanical,increment)
            try:
                endpoint=measured(f'endpoint_cut_{cut}',lambda:owner._contact(candidate))
            except (ValueError,MemoryError) as exc:
                if owner._recoverable_trial_contact_error(exc):continue
                raise
            rows[-1].update(gap_lower_m=endpoint.conservative_lower_gap_m,
                potential_j=endpoint.potential_j)
            output.write_text(json.dumps(dict(scope='first Newton cost diagnostic only',operations=rows),indent=2))
            if endpoint.conservative_lower_gap_m < -owner.gap_tolerance:continue
            safe=measured(f'swept_cut_{cut}',lambda:owner._swept_admissible(
                owner.initial.mechanical,increment,candidate,
                contact.conservative_lower_gap_m,endpoint.conservative_lower_gap_m))
            rows[-1]['admissible']=safe
            if not safe:continue
            changed=measured('candidate_assembly',lambda:owner._assemble(candidate,.05,
                owner.initial.histories,prepared_contact=(candidate,endpoint)))
            rows[-1]['metrics']=list(changed[2]);break
    finally:
        output.write_text(json.dumps(dict(scope='first Newton cost diagnostic only',operations=rows),indent=2))
    assert canonical(owner.initial.histories)==origin


@pytest.mark.skipif(os.environ.get('ANYSOLVER_CONTACT_PROFILE')!='1',
    reason='explicit ten-contact-call path diagnostic, not qualification')
def test_fibre_pair_contact_path_discriminator(monkeypatch):
    import sys,time
    output=Path(os.environ['CONTACT_APP_RESULT_RECORD']+'-contact-path.json')
    owner=_owner();rows=[];contact=owner._contact
    origin=canonical(owner.initial.histories);raw=owner._checkpoint(())
    def traced(state):
        if len(rows)>=10:raise RuntimeError('bounded ten-contact-call diagnostic stop')
        row=dict(caller=sys._getframe(1).f_code.co_name,index=len(rows))
        descriptor=state.descriptor()
        Path(str(output)+'.trial.json').write_text(json.dumps({
            key:np.asarray(value).tolist() for key,value in descriptor.items()},indent=2))
        started=time.monotonic()
        try:
            value=contact(state)
            row.update(status='completed',gap_lower_m=value.conservative_lower_gap_m,
                potential_j=value.potential_j)
            return value
        except Exception as exc:
            row.update(status=type(exc).__name__,failure=str(exc));raise
        finally:
            row['elapsed_s']=time.monotonic()-started;rows.append(row)
            output.write_text(json.dumps(dict(scope='ten-call diagnostic only',calls=rows),indent=2))
    monkeypatch.setattr(owner,'_contact',traced)
    # The external bounded supervisor harness owns crash watchdog duty;
    # no recurring in-process faulthandler timer perturbs the route.
    run=owner.solve()
    output.write_text(json.dumps(dict(scope='ten-call diagnostic only',calls=rows,
        solver_status=run.status,failure=run.failure,completed_targets=run.accepted.completed_targets),indent=2))
    assert canonical(owner.initial.histories)==origin
    assert run.checkpoint==raw and run.accepted.completed_targets==0
    assert run.status=='failed' and 'diagnostic stop' in run.failure


@pytest.mark.skipif(os.environ.get('ANYSOLVER_CONTACT_PROFILE')!='1',
    reason='explicit refined-law path diagnostic, not qualification')
def test_fibre_pair_refined_path_discriminator(monkeypatch):
    owner=_owner(line_stiffness=4000.)
    origin=canonical(owner.initial.histories);raw=owner._checkpoint(())
    rows,save=_trace_refined_pair(owner,monkeypatch,maximum_calls=10)
    run=owner.solve()
    rows.append(dict(event='result',status=run.status,failure=run.failure,
        completed_targets=run.accepted.completed_targets));save()
    assert canonical(owner.initial.histories)==origin
    assert run.checkpoint==raw and run.accepted.completed_targets==0
    assert run.status=='failed' and 'diagnostic stop' in run.failure


@pytest.mark.skipif(os.environ.get('ANYSOLVER_CONTACT_PROFILE')!='1',
    reason='explicit full-route contact cost diagnostic, not qualification')
def test_fibre_pair_route_cost_discriminator(monkeypatch):
    # One unmodified production solve traced at the owned contact boundary.
    # The cooperative deadline ends the run; no cap, tolerance or admission
    # is changed and no faulthandler/pose dump perturbs the route.
    import sys,time
    output=Path(os.environ['CONTACT_APP_RESULT_RECORD']+'-route-cost.json')
    owner=_owner();rows=[]
    contact=owner._contact
    origin=canonical(owner.initial.histories);raw=owner._checkpoint(())
    def traced(state):
        row=dict(caller=sys._getframe(1).f_code.co_name,index=len(rows))
        started=time.monotonic()
        try:
            value=contact(state)
            row.update(status='completed',
                gap_lower_m=value.conservative_lower_gap_m,
                potential_j=value.potential_j)
            return value
        except Exception as exc:
            row.update(status=type(exc).__name__,failure=str(exc)[:200]);raise
        finally:
            row['elapsed_s']=time.monotonic()-started;rows.append(row)
            output.write_text(json.dumps(dict(scope='full-route cost diagnostic only',
                calls=rows),indent=2))
    monkeypatch.setattr(owner,'_contact',traced)
    run=owner.solve()
    output.write_text(json.dumps(dict(scope='full-route cost diagnostic only',calls=rows,
        solver_status=run.status,failure=run.failure,
        completed_targets=run.accepted.completed_targets),indent=2))
    assert canonical(owner.initial.histories)==origin
    assert run.checkpoint==raw
    assert run.status=='failed' and 'cooperative deadline' in run.failure


def _trace_refined_pair(owner,monkeypatch,maximum_calls=None):
    import time
    output=Path(os.environ['CONTACT_APP_RESULT_RECORD']+'-refined-path.json')
    rows=[];calls=0
    contact=owner._contact;assemble=owner._assemble;step=owner.context.step
    swept_gap=owner._swept_gap
    def save():
        output.write_text(json.dumps(dict(scope='bounded refined-law path diagnostic only',
            line_stiffness_n_per_m2=owner.contact.stiffness,
            gap_step_control=owner.gap_step_control,events=rows),indent=2))
    def traced_contact(state):
        nonlocal calls
        if maximum_calls is not None and calls>=maximum_calls:
            raise RuntimeError('bounded refined-law diagnostic stop')
        calls+=1;row=dict(event='contact',index=calls,status='started');rows.append(row)
        Path(str(output)+'.trial.json').write_text(json.dumps({
            key:np.asarray(value).tolist() for key,value in state.descriptor().items()},indent=2))
        save();started=time.monotonic()
        try:
            value=contact(state)
            row.update(status='completed',gap_lower_m=value.conservative_lower_gap_m,
                potential_j=value.potential_j)
            return value
        except Exception as exc:
            row.update(status=type(exc).__name__,failure=str(exc));raise
        finally:
            row['elapsed_s']=time.monotonic()-started;save()
    def traced_assemble(*args,**kwargs):
        row=dict(event='assemble',parameter=float(args[1]),status='started')
        rows.append(row);save();started=time.monotonic()
        try:
            value=assemble(*args,**kwargs)
            row.update(status='completed',metrics=list(value[2]))
            return value
        except Exception as exc:
            row.update(status=type(exc).__name__,failure=str(exc));raise
        finally:
            row['elapsed_s']=time.monotonic()-started;save()
    def traced_step(*args,**kwargs):
        value=step(*args,**kwargs)
        rows.append(dict(event='step',correction=value[1],
            increment_norm=float(np.linalg.norm(value[0]))));save()
        return value
    def traced_swept_gap(state):
        row=dict(event='swept_gap',status='started');rows.append(row);save()
        started=time.monotonic()
        try:
            value=swept_gap(state);row.update(status='completed',gap_lower_m=value)
            return value
        except Exception as exc:
            row.update(status=type(exc).__name__,failure=str(exc));raise
        finally:
            row['elapsed_s']=time.monotonic()-started;save()
    monkeypatch.setattr(owner,'_contact',traced_contact)
    monkeypatch.setattr(owner,'_assemble',traced_assemble)
    monkeypatch.setattr(owner.context,'step',traced_step)
    monkeypatch.setattr(owner,'_swept_gap',traced_swept_gap)
    return rows,save
