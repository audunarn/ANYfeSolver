"""Opt-in replay integration through the actual retained native/S3 owner."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from test_general_contact_native_surface_patch import _patch
from test_general_contact_native_surface_normal import _settings as normal_settings
from test_general_contact_native_surface_friction_integral import _settings as friction_settings


def _replay_patch(activation_policy='transverse_events'):
    return _patch(activation_policy=activation_policy,gap_check='straight_patch',
        normal_integration='straight_segmented',normal_integration_settings=normal_settings(),
        friction_integration='replay',friction_integration_settings=friction_settings())


def test_saved_second_two_event_material_query(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment,NativeSurfaceReplayBudget
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    budget=NativeSurfaceReplayBudget(30)
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'two_event_material_query.json'
    point=(0.7244843513242318,3.1990185900472157)
    try:
        value=field.query(*point,budget=budget)
    except Exception as error:
        output.write_text(json.dumps(dict(point=point,error_type=type(error).__name__,error=str(error),
            point_paths=budget.point_paths,certified_intervals=budget.certified_intervals),indent=2)+'\n')
        raise
    output.write_text(json.dumps(dict(point=point,branch=value.last_contribution.trial.branch,
        gap=value.last_contribution.normal_gap_m,dissipation=value.history.cumulative_dissipation_density,
        point_paths=budget.point_paths,certified_intervals=budget.certified_intervals),indent=2)+'\n')
    assert value.history.cumulative_dissipation_density>=0


def test_saved_second_support_space_time_discriminator(tmp_path):
    import json,os
    from pathlib import Path
    from dataclasses import asdict
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_support import segment_gap_slab_enclosure,_minimum,_support_path
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    angles=(3.3238064996710945,3.323806598728249)
    rows=[]
    for xis in ((.5,.8),(.5,.65),(.65,.8)):
        enclose=segment_gap_slab_enclosure(field,1,xis,angles,1)
        for slab in ((0.,1.),(0.,.5),(.5,1.)):
            rows.append(dict(xis=xis,slab=slab,**asdict(enclose(*slab))))
    pose,segments=_support_path(field,1);segment=segments[0]
    times=np.linspace(0.,1.,17);paths=[]
    for xi in np.linspace(.5,.8,7):
        for angle in angles:
            gaps=[]
            for t in times:
                current=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,float(t))
                gaps.append(_minimum(current,field.patch.target,field.patch.source.reference.nodal_triads[0],
                    field.patch.rule.definition['radius'],1,(xi,xi),(angle,angle))[0])
            paths.append(dict(xi=float(xi),angle=angle,gaps=gaps,
                sampled_sign_changes=int(np.count_nonzero(np.diff(np.sign(gaps))))))
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'space_time_discriminator.json'
    output.write_text(json.dumps(dict(enclosures=rows,times=times.tolist(),sampled_paths=paths,
        limitation='Sampled paths diagnose only; no event or support certificate.'),indent=2)+'\n')
    assert all(np.isfinite(v) for row in paths for v in row['gaps'])


@pytest.mark.parametrize('region',[(.725,.734375,3.2201636457638987,3.225719419125236),
    (.7244140625000001,.7250000000000001,3.2007184389992185,3.201065674834302),
    (.7244140625000001,.724560546875,3.1990690687825714,3.1991558777413425)],ids=['previous','convex','correlated'])
def test_saved_second_stationary_front_discriminator(tmp_path,region):
    import json,os
    from pathlib import Path
    from scipy.optimize import brentq
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_support import _base,_support_path
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    pose,segments=_support_path(field,1);segment=segments[0];cell=1
    xi0,xi1,angle0,angle1=region
    angle=(angle0+angle1)/2
    target=field.patch.target;frame0=field.patch.source.reference.nodal_triads[0]
    motion=np.asarray(target.adapter.work_map@segment.shell_increment).reshape(3,3)
    da,db=motion[1]-motion[0],motion[2]-motion[0]
    rho=field.patch.rule.definition['radius']*np.array((0.,np.cos(angle),np.sin(angle)))
    def quantities(t,xi,theta=angle):
        rho=field.patch.rule.definition['radius']*np.array((0.,np.cos(theta),np.sin(theta)))
        current=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,float(t))
        corners=target.adapter.current_positions(current.shell_displacement)
        a,b=corners[1]-corners[0],corners[2]-corners[0]
        cross=np.cross(a,b);area=np.linalg.norm(cross);normal=cross/area
        cp=np.cross(da,b)+np.cross(a,db);normal_rate=(cp-normal*float(normal@cp))/area
        arm=current.cell_rotations[cell]@frame0@rho
        relative=_base(current,cell,xi,corners[0])+arm
        velocity=(1-xi)*segment.translations[cell]+xi*segment.translations[cell+1]-motion[0]+np.cross(segment.angular[cell],arm)
        return float(normal@relative),float(normal_rate@relative+normal@velocity)
    times=np.linspace(0.,1.,65)
    def roots(function):
        rows=[]
        values=[function(float(t)) for t in times]
        for left,right,fl,fr in zip(times[:-1],times[1:],values[:-1],values[1:]):
            if fl*fr<0:
                value=brentq(function,float(left),float(right),xtol=1e-14)
                rows.append(dict(bracket=[float(left),float(right)],root=value,residual=function(value)))
        return rows,values
    paths=[]
    for xi in np.linspace(xi0,xi1,9):
        stationary,_=roots(lambda t:quantities(t,xi)[1])
        crossings,_=roots(lambda t:quantities(t,xi)[0])
        for row in stationary:row['gap']=quantities(row['root'],xi)[0]
        paths.append(dict(xi=float(xi),stationary=stationary,crossings=crossings,
            endpoints=[quantities(t,xi) for t in (0.,1.)]))
    def determinant(t):
        gl,dl=quantities(t,xi0);gr,dr=quantities(t,xi1)
        return dl*gr-gl*dr
    folds,detvalues=roots(determinant)
    for row in folds:
        gl,_=quantities(row['root'],xi0);gr,_=quantities(row['root'],xi1)
        xi=xi0-gl*(xi1-xi0)/(gr-gl)
        row.update(xi=xi,gap_and_rate=quantities(row['root'],xi))
    angular_folds=[]
    for time in (.25,.5,.75):
        def angular_determinant(theta):
            gl,dl=quantities(time,xi0,theta);gr,dr=quantities(time,xi1,theta)
            return dl*gr-gl*dr
        lower,upper=3.1423828187051774,3.32016756626797
        if angular_determinant(lower)*angular_determinant(upper)<0:
            theta=brentq(angular_determinant,lower,upper,xtol=1e-14)
            gl,_=quantities(time,xi0,theta);gr,_=quantities(time,xi1,theta)
            xi=xi0-gl*(xi1-xi0)/(gr-gl)
            nearby=xi+1e-8
            crossings,_=roots(lambda t:quantities(t,nearby,theta)[0])
            angular_folds.append(dict(time=time,angle_bracket=[lower,upper],angle=theta,xi=xi,
                gap_and_rate=quantities(time,xi,theta),nearby_xi=nearby,nearby_crossings=crossings,
                nearby_endpoint_midpoint_gaps=[quantities(t,nearby,theta)[0] for t in (0.,time,1.)]))
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'stationary_front_discriminator.json'
    output.write_text(json.dumps(dict(angle=angle,paths=paths,fold_roots=folds,angular_folds=angular_folds,
        determinant_sample_range=[min(detvalues),max(detvalues)],
        limitation='Sign-bracket diagnostic at one angle,not exhaustive root isolation or a support certificate.'),indent=2)+'\n')
    assert all(np.isfinite(v) for row in paths for pair in row['endpoints'] for v in pair)


def test_replay_initialization_rejected_trial_and_checkpoint_identity():
    owner,bridge=_replay_patch();transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint();history=transaction.accepted.friction_history
    assert len(history.segments)==0
    proposal=transaction.propose(np.zeros(42),np.zeros(18),1.)
    assert len(proposal.assembly.material.contribution.trial.candidate_state.segments)==0
    assert proposal.assembly.material.contribution.friction_integral.diagnostics['queries']==0
    with pytest.raises(ValueError,match='equilibrium'):
        transaction.commit(proposal)
    assert transaction.checkpoint()==before
    delta=np.zeros(42);delta[18]=np.pi
    with pytest.raises(ValueError,match='requires cutback'):
        transaction.propose(delta,np.zeros(18),1.)
    assert transaction.checkpoint()==before
    foreign=replace(history,maximum_intervals=history.maximum_intervals+1)
    with pytest.raises(ValueError,match='interval binding'):
        bridge.evaluate(transaction.accepted,foreign,np.zeros(42),np.zeros(18),0.)
    restored=NativeS3MaterialTransaction.restore(bridge,before,sha256(before).hexdigest())
    assert restored.checkpoint()==before
    _,other=_patch(activation_policy='transverse_events',gap_check='straight_patch',
        normal_integration='straight_segmented',normal_integration_settings=normal_settings())
    with pytest.raises(ValueError,match='definition/settings'):
        NativeS3MaterialTransaction.restore(other,before,sha256(before).hexdigest())


@pytest.mark.parametrize('reason',[
    'native friction integration requires resolved historical contact envelope',
    'native friction integration requires swept whole-patch target ownership',
    'native friction aggregate support interval cap exhausted'])
def test_replay_geometry_rejection_backtracks_without_commit(monkeypatch,reason):
    owner,bridge=_replay_patch();transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint();original=transaction.propose;calls=0
    def rejected_once(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            assert transaction.checkpoint()==before
            raise (MemoryError if 'support interval cap' in reason else ValueError)(reason)
        if calls==3:assert transaction.checkpoint()==before
        return original(*args,**kwargs)
    monkeypatch.setattr(transaction,'propose',rejected_once)
    owner.solve_material_increment(transaction,.001)
    assert calls>=3 and transaction.accepted.sequence==1


def test_replay_unexpected_trial_error_remains_fatal(monkeypatch):
    owner,bridge=_replay_patch();transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint();original=transaction.propose;calls=0
    def unexpected(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise ValueError('unexpected replay defect')
        return original(*args,**kwargs)
    monkeypatch.setattr(transaction,'propose',unexpected)
    with pytest.raises(ValueError,match='unexpected replay defect'):
        owner.solve_material_increment(transaction,.001)
    assert transaction.checkpoint()==before


def test_material_newton_reuses_exact_candidate_but_rechecks_commit(monkeypatch):
    owner,bridge=_replay_patch();transaction=NativeS3MaterialTransaction(bridge)
    original_propose=transaction.propose;original_evaluate=bridge.evaluate
    proposed=[];evaluated=[]
    def key(native,shell,parameter):
        return (sha256(np.asarray(native).tobytes()+np.asarray(shell).tobytes()).hexdigest(),float(parameter))
    def propose(native,shell,parameter,**kwargs):
        proposed.append(key(native,shell,parameter))
        return original_propose(native,shell,parameter,**kwargs)
    def evaluate(origin,history,native,shell,parameter):
        evaluated.append(key(native,shell,parameter))
        return original_evaluate(origin,history,native,shell,parameter)
    monkeypatch.setattr(transaction,'propose',propose)
    monkeypatch.setattr(bridge,'evaluate',evaluate)
    owner.solve_material_increment(transaction,.001)
    assert len(proposed[:-1])==len(set(proposed[:-1]))
    # Commit alone repeats the final candidate's constitutive/contact work.
    assert proposed[-1]==proposed[-2]
    assert evaluated==proposed
    assert transaction.accepted.sequence==1


def test_replay_coupled_active_rebase_and_split_restart(monkeypatch):
    # Preserve a rejected integration path so subsequent diagnosis can replay
    # the local contribution without spending another structural solve.
    import json,os
    from dataclasses import asdict
    from pathlib import Path
    import anyfesolver.general_contact.native_surface_friction_integral as module
    original=module.integrate_material_friction
    calls=0;timings=[]
    def captured(field,settings,**kwargs):
        nonlocal calls
        import time
        calls+=1;started=time.perf_counter()
        try:
            result=original(field,settings,**kwargs)
            folder=os.environ.get('RUNNER_TEMP')
            if folder:
                timings.append(dict(call=calls,seconds=time.perf_counter()-started,
                    segments=len(field.segments),queries=result.diagnostics['queries'],
                    scaled_error=result.diagnostics['maximum_scaled_error_estimate']))
                Path(folder,'coupled_integral_timings.json').write_text(json.dumps(timings)+'\n')
            return result
        except (MemoryError,ValueError) as error:
            folder=os.environ.get('RUNNER_TEMP')
            if folder:
                payload=dict(calls=calls,error=str(error),initial=asdict(field.initial),
                    segments=[asdict(s) for s in field.segments],settings=asdict(settings),
                    definition=field.patch._definition(),maximum_intervals=field.maximum_intervals,
                    integration_diagnostics=getattr(error,'integration_diagnostics',None))
                Path(folder,'rejected_material_path.json').write_text(json.dumps(payload,
                    default=lambda value:value.tolist(),indent=2)+'\n')
            raise
    monkeypatch.setattr(module,'integrate_material_friction',captured)
    owner,bridge=_replay_patch();transaction=NativeS3MaterialTransaction(bridge)
    def stage(name,current=None):
        current=transaction if current is None else current
        folder=os.environ.get('RUNNER_TEMP')
        if folder:
            Path(folder,'coupled_stage.json').write_text(json.dumps(dict(stage=name,
                accepted_sequence=current.accepted.sequence,integral_calls=calls))+'\n')
            if current.accepted.sequence:
                Path(folder,'accepted_material_checkpoint.json').write_bytes(current.checkpoint())
    stage('initial')
    owner.solve_material_increment(transaction,1.)
    stage('first_load_accepted')
    accepted=transaction.accepted
    assert len(accepted.friction_history.segments)==1
    capsule=transaction.checkpoint()
    owner.solve_material_increment(transaction,1.05)
    uninterrupted_checkpoint=transaction.checkpoint()
    stage('uninterrupted_path_completed')
    # A real restart reconstructs its analysis; it does not reset the original
    # context's deadline. Each analysis retains its own unchanged120s guard.
    restarted_owner,restarted_bridge=_replay_patch()
    restored=NativeS3MaterialTransaction.restore(restarted_bridge,capsule,sha256(capsule).hexdigest())
    stage('active_checkpoint_restored',restored)
    # Perform the independent zero-increment diagnostic on this already
    # restored first-load origin, keeping the original analysis uninterrupted
    # for its two solver loads. Both analyses retain the unchanged120s guard.
    same=restored.propose(np.zeros(42),np.zeros(18),1.).assembly
    assert max(same.metrics)<restarted_owner.equilibrium_tolerance
    assert same.material.contribution.friction_integral.stored_energy_j>0
    assert same.material.contribution.friction_integral.dissipated_increment_j==0.
    assert same.material.contribution.friction_integral.diagnostics['queries']>0
    stage('restored_origin_rebased',restored)
    restarted_owner.solve_material_increment(restored,1.05)
    assert uninterrupted_checkpoint==restored.checkpoint()
    stage('split_restart_identical',restored)


def test_saved_replay_candidate_branch_discriminator(tmp_path):
    """Small saved-pose diagnostic; does not repeat the coupled failure."""
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import (
        NativeSurfaceHistoryField,NativeSurfaceMotionSegment,NativeSurfaceReplayBudget)
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    path=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_rejected_path.json'
    saved=json.loads(path.read_text())
    _,bridge=_replay_patch()
    initial=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()})
    segments=tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments'])
    field=NativeSurfaceHistoryField(bridge.pair,initial,segments,maximum_intervals=saved['maximum_intervals'])
    budget=NativeSurfaceReplayBudget(30);rows=[]
    for xi in (.79945,.7997,.79995):
        for angle in (3.48106,3.4812,3.48138):
            query=field.query(xi,angle,budget=budget);p=query.last_contribution
            rows.append(dict(xi=xi,angle=angle,gap=p.normal_gap_m,branch=p.trial.branch,
                pressure=p.normal_pressure_Pa,event_fraction=getattr(p,'event_fraction',None),
                friction_force=(p.residual-p.normal_residual).tolist(),
                tangent_norm=float(np.linalg.norm(p.spatial_tangent()-p.normal_spatial_tangent()))))
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'branch_discriminator.json'
    output.write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==9


@pytest.mark.parametrize('activation_policy',['transverse_events','transverse_event_sequence'])
@pytest.mark.parametrize('filename',[
    'replay_owner_rejected_path.json','replay_owner_support_rejected_path.json',
    'replay_owner_envelope_rejected_path.json','replay_owner_rebase_rejected_path.json',
    'replay_owner_second_load_rejected_path.json'])
def test_saved_replay_candidate_front_refinement(tmp_path,filename,monkeypatch,activation_policy):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'/filename).read_text())
    _,bridge=_replay_patch(activation_policy)
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    support_charges={}
    if os.environ.get('NATIVE_REPLAY_SUPPORT_TRACE'):
        import inspect
        from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget
        original_consume=SupportIntervalBudget.consume
        def counted_consume(budget):
            frame=inspect.currentframe().f_back
            names=[]
            while frame is not None and len(names)<4:
                names.append(frame.f_code.co_name)
                frame=frame.f_back
            key='/'.join(names)
            result=original_consume(budget)
            row=support_charges.setdefault(key,dict(consumed=0,rejected=0))
            row['consumed' if result else 'rejected']+=1
            return result
        monkeypatch.setattr(SupportIntervalBudget,'consume',counted_consume)
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'saved_candidate_result.json'
    try:
        if os.environ.get('NATIVE_REPLAY_PROFILE_PATH'):
            import cProfile
            profiler=cProfile.Profile()
            try:value=profiler.runcall(integrate_material_friction,field,friction_settings())
            finally:profiler.dump_stats(os.environ['NATIVE_REPLAY_PROFILE_PATH'])
        else:value=integrate_material_friction(field,friction_settings())
    except (MemoryError,ValueError) as error:
        snapshot=None
        traceback=error.__traceback__
        while traceback is not None:
            if (traceback.tb_frame.f_code.co_name=='integrate_material_friction'
                    and 'support_budget' in traceback.tb_frame.f_locals):
                local=traceback.tb_frame.f_locals
                scale=local.get('scale')
                rows=[]
                for bounds,_,estimate,safe_zero in local.get('leaves',[]):
                    scaled=scale*estimate
                    index=int(np.argmax(scaled))
                    rows.append(dict(bounds=list(bounds),safe_zero=bool(safe_zero),
                        maximum_scaled_error=float(scaled[index]),component=index,
                        error_kind='force' if index<60 else 'tangent' if index<3660 else 'work'))
                snapshot=dict(current_panel=local.get('panel'),last_parent=local.get('parent'),
                    visited=local.get('visited'),accepted=local.get('accepted'),
                    queries=len(local.get('cache',{})),pending_panels=len(local.get('panels',[])),
                    support_intervals=local['support_budget'].intervals,leaves=rows,
                    limitation='Cap snapshot only; no refinement ancestry or per-panel charge trace.')
            traceback=traceback.tb_next
        output.write_text(json.dumps(dict(error=str(error),diagnostics=getattr(error,'integration_diagnostics',None),
            cap_snapshot=snapshot,support_charges=support_charges),indent=2)+'\n')
        raise
    output.write_text(json.dumps(value.diagnostics,indent=2)+'\n')
    assert value.diagnostics['maximum_scaled_error_estimate']<=1.
    assert value.stored_energy_j>0


def test_saved_envelope_regional_cover(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_support import (
        SupportIntervalBudget,history_endpoint_support_certified,history_endpoint_support_cover_certified)
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_envelope_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    assert not history_endpoint_support_certified(field)
    budget=SupportIntervalBudget(friction_settings().maximum_support_intervals)
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'regional_cover.json'
    try:result=history_endpoint_support_cover_certified(field,budget=budget)
    except (MemoryError,ValueError) as error:
        output.write_text(json.dumps(dict(error=str(error),support_intervals=budget.intervals))+'\n')
        raise
    output.write_text(json.dumps(dict(certified=result,support_intervals=budget.intervals))+'\n')
    assert result
    with pytest.raises(MemoryError,match='aggregate support interval cap'):
        history_endpoint_support_cover_certified(field,budget=SupportIntervalBudget(1))


def test_saved_regional_candidate_directional_tangent():
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_envelope_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    initial=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()})
    segments=tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments'])
    def evaluate(shift):
        translation=segments[-1].translations.copy();translation[2,0]+=shift
        last=replace(segments[-1],translations=translation)
        field=NativeSurfaceHistoryField(bridge.pair,initial,(*segments[:-1],last),
            maximum_intervals=saved['maximum_intervals'])
        return integrate_material_friction(field,friction_settings())
    value=evaluate(0.)
    direction=np.zeros(60);direction[12]=1.
    predicted=value.tangent@direction
    for step in (1e-7,2e-8):
        numerical=(evaluate(step).residual-evaluate(-step).residual)/(2*step)
        assert np.linalg.norm(numerical-predicted)/np.linalg.norm(numerical)<=1e-5

def test_saved_rebase_branch_discriminator(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment,NativeSurfaceReplayBudget
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_rebase_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    initial=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()})
    segments=tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments'])
    assert len(segments)==2 and not np.any(segments[-1].translations) and not np.any(segments[-1].angular)
    fields=[NativeSurfaceHistoryField(bridge.pair,initial,s,maximum_intervals=saved['maximum_intervals']) for s in (segments[:-1],segments)]
    budget=NativeSurfaceReplayBudget(60);rows=[]
    for xi in (.25,.5,.75):
        for angle in (np.pi-.15,np.pi,np.pi+.15):
            points=[f.query(xi,angle,budget=budget).last_contribution for f in fields]
            rows.append(dict(xi=xi,angle=angle,steps=[dict(branch=p.trial.branch,gap=p.normal_gap_m,
                traction_norm=float(np.linalg.norm(p.trial.traction)),cap=.3*p.normal_pressure_Pa,
                friction_tangent_norm=float(np.linalg.norm(p.spatial_tangent()-p.normal_spatial_tangent())),
                dissipation=p.trial.dissipated_increment_density) for p in points]))
    Path(os.environ.get('RUNNER_TEMP',str(tmp_path)),'rebase_branch_discriminator.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==9


def test_saved_rebase_cover_prefix_cost(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget,history_endpoint_support_cover_certified
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_rebase_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    initial=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()})
    segments=tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments'])
    rows=[]
    for count in (1,2):
        field=NativeSurfaceHistoryField(bridge.pair,initial,segments[:count],maximum_intervals=saved['maximum_intervals'])
        budget=SupportIntervalBudget(128)
        result=history_endpoint_support_cover_certified(field,budget=budget)
        rows.append(dict(segments=count,certified=result,support_intervals=budget.intervals))
    Path(os.environ.get('RUNNER_TEMP',str(tmp_path)),'cover_prefix_cost.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(row['certified'] for row in rows)


def test_saved_second_load_half_trial(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    saved=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    initial=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()})
    segments=tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in row.items()}) for row in saved['segments'])
    last=segments[-1]
    half=NativeSurfaceMotionSegment(.5*last.translations,.5*last.angular,.5*last.shell_increment)
    field=NativeSurfaceHistoryField(bridge.pair,initial,(*segments[:-1],half),maximum_intervals=saved['maximum_intervals'])
    value=integrate_material_friction(field,friction_settings())
    Path(os.environ.get('RUNNER_TEMP',str(tmp_path)),'half_trial_result.json').write_text(json.dumps(value.diagnostics,indent=2)+'\n')
    assert value.diagnostics['maximum_scaled_error_estimate']<=1.


@pytest.mark.parametrize('persistent',[False,True])
def test_support_trial_limit_is_bounded_and_real_memory_error_is_fatal(monkeypatch,persistent):
    owner,bridge=_replay_patch();transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint();original=transaction.propose;calls=0
    message='native friction aggregate support interval cap exhausted' if persistent else 'allocator out of memory'
    def fail_trial(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls>1:raise MemoryError(message)
        return original(*args,**kwargs)
    monkeypatch.setattr(transaction,'propose',fail_trial)
    with pytest.raises(RuntimeError if persistent else MemoryError,match=message):
        owner.solve_material_increment(transaction,.001)
    assert calls==(owner.program.max_backtracks+2 if persistent else 2)
    assert transaction.checkpoint()==before


def test_saved_native_active_checkpoint_continues_second_load(tmp_path):
    import json,os
    from pathlib import Path
    path=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_first_accepted_checkpoint.json'
    capsule=path.read_bytes()
    # The saved container ends with CRLF while the producer checkpoint() emits
    # LF; correct only that container newline to the producer contract at load.
    if capsule.endswith(b'\r\n'):capsule=capsule[:-2]+b'\n'
    owner,bridge=_replay_patch()
    transaction=NativeS3MaterialTransaction.restore(bridge,capsule,sha256(capsule).hexdigest())
    assert transaction.accepted.sequence==1
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))
    (folder/'restored_stage.json').write_text(json.dumps(dict(stage='first_checkpoint_restored',sequence=1)))
    try:owner.solve_material_increment(transaction,1.05)
    except Exception as error:
        (folder/'restored_stage.json').write_text(json.dumps(dict(stage='second_load_failed',sequence=transaction.accepted.sequence,error=str(error))))
        raise
    (folder/'restored_stage.json').write_text(json.dumps(dict(stage='second_load_accepted',sequence=transaction.accepted.sequence)))
    (folder/'second_checkpoint.json').write_bytes(transaction.checkpoint())
    assert transaction.accepted.sequence==2
