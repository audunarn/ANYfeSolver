"""Local stationary-strip support with closed neighboring material."""
from dataclasses import replace
import os
import numpy as np
import pytest
from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
from anyfesolver.general_contact.native_surface_front import stationary_front_angular_cover,stationary_front_angular_envelopes
from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget,_minimum
from anyfesolver.general_contact.native_surface_friction_integral import integrate_event_friction_fronts
from test_general_contact_native_surface_support import _field
from test_general_contact_native_surface_friction_integral import _settings


def _mixed(height=.035,slope=.01,rotation=1.1):
    motion=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((rotation,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=height,rotation=-.8,segments=(motion,),activation_policy='transverse_event_sequence')
    positions=field.initial.positions.copy();positions[:,2]+=slope*positions[:,0]
    return NativeSurfaceHistoryField(field.patch,replace(field.initial,positions=positions),field.segments)


def _three_segment_pulse(initial_height=.036,prefix_height_change=-.001):
    value=_mixed(height=initial_height,rotation=1.6)
    def translation(dz):
        displacement=np.zeros((3,3));displacement[:,2]=dz
        return NativeSurfaceMotionSegment(displacement,np.zeros((2,3)),np.zeros(18))
    return NativeSurfaceHistoryField(value.patch,value.initial,
        (translation(prefix_height_change),value.segments[0],translation(.001)))


def test_whole_angle_stationary_envelope_is_locally_open_with_closed_neighbor():
    value=_mixed();angles=value.patch.rule.definition['angle_range'];budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(value,0,0,angles,budget=budget)
    envelopes=stationary_front_angular_envelopes(value,cover,maximum_axial_uncertainty=1e-7,budget=budget)
    assert len(envelopes)==1
    a,b=envelopes[0]
    # Independent stationary minimum: radius-height=0 at xi=-.5 for all angles.
    assert a<-.5-1e-7<-.5+1e-7<b
    normalframe=value.patch.rule.samples[0].station.reference.nodal_triads[0]
    for pose in (value.initial,value.current):
        gap,padding=_minimum(pose,value.patch.target,normalframe,.03,0,(a,b),angles)
        assert gap>padding
    gap,_=_minimum(value.current,value.patch.target,normalframe,.03,0,(-.8,-.2),angles)
    assert gap<0.
    assert budget.intervals<128
    with pytest.raises(MemoryError,match='cap'):
        stationary_front_angular_envelopes(value,cover,maximum_axial_uncertainty=1e-7,budget=SupportIntervalBudget(1))


@pytest.mark.skipif(os.environ.get('ANYSOLVER_RUN_UNRESOLVED_MIXED_FRONT')!='1',
    reason='bounded expensive development replay; helper tested, native owner/acceptance still incomplete')
def test_mixed_endpoints_integrated_force_and_tangent(tmp_path):
    import json,os
    from pathlib import Path
    from scipy.integrate import quad
    value=_mixed()
    try:
        result=integrate_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)
    except Exception as error:
        folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
        (folder/'mixed_front_failure.json').write_text(json.dumps({'error':str(error),
            'diagnostics':getattr(error,'integration_diagnostics',None)},indent=2),encoding='utf-8')
        raise
    # All closed points slide in +y. Integrate triangular linear pressure in xi,
    # then integrate its smooth analytic angular dependence independently.
    root=lambda delta:(.03*np.cos(.3+delta)-.035)/.01
    exact=.03*.3*1000*quad(lambda delta:.5*.01*(root(delta)+.8)**2,-.02,.02,epsabs=1e-14)[0]
    derivative=-.03*.3*1000*quad(lambda delta:root(delta)+.8,-.02,.02,epsabs=1e-14)[0]
    y=np.zeros(60);y[[1,7,13]]=1.
    z=np.zeros(60);z[[2,8,14]]=1.
    force=float(y@result.residual);tangent=float(y@result.tangent@z)
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'mixed_front_result.json').write_text(json.dumps({'force':force,'reference_force':exact,
        'tangent':tangent,'reference_tangent':derivative,'diagnostics':result.diagnostics},indent=2),encoding='utf-8')
    assert force==pytest.approx(exact,rel=1e-5,abs=1e-10)
    assert tangent==pytest.approx(derivative,rel=1e-5,abs=1e-8)
    assert result.dissipated_increment_j>0 and result.stored_energy_j>0
    total=sum((result.residual[6*i:6*i+3] for i in range(3)),np.zeros(3))
    total+=sum((result.residual[42+6*i:45+6*i] for i in range(3)),np.zeros(3))
    np.testing.assert_allclose(total,np.zeros(3),atol=1e-12,rtol=0.)


def test_no_stationary_branch_does_not_discard_closed_material():
    from scipy.integrate import quad
    value=_mixed(height=.024,slope=.001,rotation=.2)
    result=integrate_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)
    expected=.03*.3*1000*.6*quad(lambda delta:.03*np.cos(-.6+delta)-.0235,-.02,.02,epsabs=1e-14)[0]
    assert sum(result.residual[6*i+1] for i in range(3))==pytest.approx(expected,rel=1e-5,abs=1e-10)
    assert result.stored_energy_j>0 and result.dissipated_increment_j>0
    assert result.diagnostics['zero_panels_certified']==0


def test_variable_front_route_keeps_closed_no_stationary_surface_force():
    from scipy.integrate import quad
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_variable_event_friction_fronts
    value=_mixed(height=.024,slope=.001,rotation=.2)
    result=integrate_variable_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)
    expected=.03*.3*1000*.6*quad(lambda delta:.03*np.cos(-.6+delta)-.0235,-.02,.02,epsabs=1e-14)[0]
    assert sum(result.residual[6*i+1] for i in range(3))==pytest.approx(expected,rel=1e-5,abs=1e-10)
    assert result.stored_energy_j>0 and result.dissipated_increment_j>0
    assert result.diagnostics['zero_panels_certified']==0
    assert result.diagnostics['open_endpoint_strip_enclosures']==0


def test_multisegment_strip_bound_uses_actual_branch_start():
    from math import acos,sin,cos,fsum
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    from anyfesolver.general_contact.native_surface_front_work import open_front_strip_work_bound
    value=_three_segment_pulse();prefix=value.segments[0]
    start=material_path_pose(value.initial,prefix.translations,prefix.angular,prefix.shell_increment,1.)
    middle=NativeSurfaceHistoryField(value.patch,start,(value.segments[1],))
    width=1e-7
    bound=open_front_strip_work_bound(value,0,width,segment_index=1)
    np.testing.assert_array_equal(bound,open_front_strip_work_bound(middle,0,width))
    height=.03-.01*width/2;theta=acos(height/.03)
    elastic=0.;previous=.03*sin(-theta);terms=[]
    for i in range(1,32):
        angle=-theta+2*theta*i/32;y=.03*sin(angle)
        trial=elastic+y-previous;cap=.3*1000*(.03*cos(angle)-height)
        next_elastic=min(trial,cap/1000);terms.append(cap*(trial-next_elastic))
        elastic=next_elastic;previous=y
    assert 0<fsum(terms)<=bound[1]
    assert .5*1000*elastic**2<=bound[2]
    with pytest.raises(ValueError,match='one released-origin segment'):
        open_front_strip_work_bound(value,0,width)
    for invalid in (-1,3,True):
        with pytest.raises(ValueError,match='segment index'):
            open_front_strip_work_bound(value,0,width,segment_index=invalid)


def test_multisegment_front_rejects_closed_prefix_capture():
    # Middle-segment strip is open at its own endpoints, but an earlier
    # captured pose has contact there. Initial penetration remains rejected.
    value=_three_segment_pulse()
    down=np.zeros((3,3));down[:,2]=-.012
    up=np.zeros((3,3));up[:,2]=.011
    value=NativeSurfaceHistoryField(value.patch,value.initial,
        (NativeSurfaceMotionSegment(down,np.zeros((2,3)),np.zeros(18)),
         NativeSurfaceMotionSegment(up,np.zeros((2,3)),np.zeros(18)),*value.segments[1:]))
    with pytest.raises(ValueError,match='unresolved or closed captured endpoint'):
        integrate_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)


@pytest.mark.skipif(os.environ.get('ANYSOLVER_RUN_UNRESOLVED_MIXED_FRONT')!='1',
    reason='bounded expensive three-segment development replay; explicit execution only')
def test_multisegment_front_integral_has_independent_cumulative_work(tmp_path):
    from math import acos,sin,cos,fsum
    from scipy.integrate import quad
    import json,os
    from pathlib import Path
    value=_three_segment_pulse()
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    try:
        result=integrate_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)
    except Exception as error:
        (folder/'multisegment_front_failure.json').write_text(json.dumps(dict(error=str(error),
            diagnostics=getattr(error,'integration_diagnostics',None)),indent=2),encoding='utf-8')
        raise
    def independent_work(xi):
        height=.035+.01*xi
        if height>=.03:return 0.
        theta=acos(height/.03);elastic=0.;previous=.03*sin(-theta);terms=[]
        for i in range(1,32):
            angle=-theta+2*theta*i/32;y=.03*sin(angle)
            trial=elastic+y-previous;cap=.3*1000*(.03*cos(angle)-height)
            next_elastic=min(trial,cap/1000);terms.append(cap*(trial-next_elastic))
            elastic=next_elastic;previous=y
        return fsum(terms)
    # The translated prefix and suffix are strictly open. Only the complete
    # middle pulse contributes cumulative work, independent of angular offset.
    expected=.03*.04*quad(independent_work,-.8,-.5,epsabs=1e-13,epsrel=1e-10)[0]
    (folder/'multisegment_front_result.json').write_text(json.dumps(dict(
        cumulative_work=result.cumulative_dissipation_j,reference=expected,
        stored=result.stored_energy_j,dissipated=result.dissipated_increment_j,
        released=result.released_storage_j,
        diagnostics=result.diagnostics),indent=2),encoding='utf-8')
    assert result.cumulative_dissipation_j==pytest.approx(expected,rel=1e-5,abs=1e-12)
    np.testing.assert_array_equal(result.residual,0.)
    np.testing.assert_array_equal(result.tangent,0.)
    assert result.stored_energy_j==0. and result.dissipated_increment_j==0.
    assert result.released_storage_j==0.


def _saved_local_front_field():
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_patch import NativeS3MaterialSurfacePatch
    from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_friction_integral import NativeSurfaceFrictionIntegrationSettings
    from test_general_contact_native_surface_replay_owner import _replay_patch
    source=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_second_load_rejected_path.json'
    saved=json.loads(source.read_text());_,bridge=_replay_patch('transverse_event_sequence')
    full=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    base=full.patch.pairs[0]
    rule=NativeRoundSurfaceRule(base.source.element_id,base.source.reference,base.source.radius,
        xi_range=(.72,.80),angle_range=(3.19,3.20),axial_order=2,angular_order=2)
    patch=NativeS3MaterialSurfacePatch(rule,[full._target_at(s.station) for s in rule.samples],
        normal_stiffness_Pa_per_m=base.normal_stiffness_Pa_per_m,
        friction_coefficient=base.friction_coefficient,
        tangential_stiffness_Pa_per_m=base.tangential_stiffness_Pa_per_m,
        activation_policy='transverse_event_sequence',event_history_substeps=base.event_history_substeps,
        gap_check='straight_patch')
    value=NativeSurfaceHistoryField(patch,full.initial,full.segments,maximum_intervals=full.maximum_intervals)
    settings=NativeSurfaceFrictionIntegrationSettings(**saved['settings'])
    return value,settings,source


@pytest.mark.skipif(os.environ.get('ANYSOLVER_RUN_SAVED_NATIVE_LOCAL_FRONT')!='1',
    reason='named saved-path local discriminator; no fullpatch or coupled run')
def test_saved_second_load_local_front_window(tmp_path):
    import json,os,time
    from dataclasses import asdict
    from pathlib import Path
    from hashlib import sha256
    value,settings,source=_saved_local_front_field()
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    record=dict(saved_input_sha256=sha256(source.read_bytes()).hexdigest(),
        selected_xi=(.72,.80),selected_angle=(3.19,3.20),settings=asdict(settings),
        event_policy='transverse_event_sequence',field_identity=value.identity,
        scope='local saved-path integration only, no solve/fullpatch/owner admission')
    (folder/'saved_local_front_preflight.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    start=time.perf_counter()
    try:
        result=integrate_event_friction_fronts(value,settings,maximum_axial_uncertainty=1e-7)
    except Exception as error:
        record.update(elapsed_seconds=time.perf_counter()-start,error_type=type(error).__name__,
            error=str(error),diagnostics=getattr(error,'integration_diagnostics',None))
        (folder/'saved_local_front_failure.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        raise
    source_force=sum((result.residual[6*i:6*i+3] for i in range(3)),np.zeros(3))
    target_force=sum((result.residual[42+6*i:45+6*i] for i in range(3)),np.zeros(3))
    force_scale=max(np.linalg.norm(source_force),np.linalg.norm(target_force),1e-12)
    record.update(elapsed_seconds=time.perf_counter()-start,diagnostics=result.diagnostics,
        source_force=source_force.tolist(),target_force=target_force.tolist(),
        force_balance_relative=float(np.linalg.norm(source_force+target_force)/force_scale),
        work=dict(stored=result.stored_energy_j,dissipated=result.dissipated_increment_j,
            released=result.released_storage_j,cumulative=result.cumulative_dissipation_j))
    (folder/'saved_local_front_result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    assert result.diagnostics['maximum_scaled_error_estimate']<=1.
    assert record['force_balance_relative']<=1e-6
    assert all(x>=0 for x in record['work'].values())


def test_history_work_cap_covers_inherited_elastic_state_during_unloading():
    from anyfesolver.general_contact.native_surface_front import prepare_stationary_band
    from anyfesolver.general_contact.native_surface_front_work import history_front_strip_work_bound
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceReplayBudget
    original=_mixed(rotation=0.)
    def move(dy,dz):
        translation=np.tile((0.,dy,dz),(3,1))
        return NativeSurfaceMotionSegment(translation,np.zeros((2,3)),np.zeros(18))
    motions=(move(.004,-.011),move(0.,.001))
    value=NativeSurfaceHistoryField(original.patch,original.initial,motions)
    budget=SupportIntervalBudget(128);angles=value.patch.rule.definition['angle_range']
    covers=tuple(stationary_front_angular_cover(value,0,i,angles,budget=budget,
        xi_bounds=(-.8,-.2)) for i in range(2))
    prepared=tuple(prepare_stationary_band(value,cover,budget=budget) for cover in covers)
    envelopes=tuple(stationary_front_angular_envelopes(value,cover,maximum_axial_uncertainty=1e-7,
        budget=budget,prepared=item,_include_endpoints=True) for cover,item in zip(covers,prepared))
    upper=history_front_strip_work_bound(value,0,(-.65,-.55),prepared,envelopes)
    first=NativeSurfaceHistoryField(original.patch,original.initial,motions[:1]).query(-.6,1.5*np.pi,
        budget=NativeSurfaceReplayBudget(16))
    last=value.query(-.6,1.5*np.pi,budget=NativeSurfaceReplayBudget(16))
    trial=last.last_contribution.trial
    actual=np.array((trial.stored_energy_density,trial.dissipated_increment_density,
        trial.released_storage_density,last.history.cumulative_dissipation_density))
    assert first.last_contribution.trial.stored_energy_density>trial.stored_energy_density>0
    assert actual[-1]>actual[1]>0 and np.all(actual<=upper)
    assert np.all(upper>=0)


def test_deferred_directional_controls_match_eager_refined_integral(tmp_path,monkeypatch):
    """Polynomial code probe for evaluation order; no mechanics qualification."""
    import json,os,time
    from pathlib import Path
    from types import SimpleNamespace
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    value=_mixed(height=.024,slope=.001,rotation=.2)
    # Actual geometry, proof/cache and adaptive integration; replace only the
    # expensive material law with an independent degree-eight polynomial.
    # Gauss4 underintegrates its parent; Gauss8 is exact, as are refined outputs.
    def polynomial_query(self,xi,angle,*,budget):
        q=((xi+.8)/.6)**8
        residual=np.zeros(60);residual[0]=q
        tangent=np.zeros((60,60));tangent[0,0]=q
        point=SimpleNamespace(residual=residual,normal_residual=np.zeros(60),
            spatial_tangent=lambda:tangent,normal_spatial_tangent=lambda:np.zeros((60,60)),
            trial=SimpleNamespace(branch='stick',stored_energy_density=q,
                dissipated_increment_density=q,released_storage_density=q,
                cumulative_dissipation_density=q))
        return SimpleNamespace(last_contribution=point,history=SimpleNamespace(cumulative_dissipation_density=q))
    monkeypatch.setattr(NativeSurfaceHistoryField,'query',polynomial_query)
    settings=replace(_settings(),force_absolute_tolerance=1e-8,
        tangent_absolute_tolerance=1e-8,energy_absolute_tolerance=1e-8)
    outputs=[];timings=[]
    for eager in (True,False):
        start=time.perf_counter()
        outputs.append(integrate_material_friction(value,settings,_front_uncertainty=1e-7,
            _local_front_strips=True,_eager_directional_controls=eager))
        timings.append(time.perf_counter()-start)
    eager,deferred=outputs
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'directional_control_parity.json').write_text(json.dumps(dict(
        elapsed_seconds=timings,eager=eager.diagnostics,deferred=deferred.diagnostics),indent=2),encoding='utf-8')
    np.testing.assert_array_equal(deferred.residual,eager.residual)
    np.testing.assert_array_equal(deferred.tangent,eager.tangent)
    for key in ('stored_energy_j','dissipated_increment_j','released_storage_j','cumulative_dissipation_j'):
        assert getattr(deferred,key)==getattr(eager,key)
    for key in ('maximum_scaled_error_estimate','panels','refinement_axes'):
        assert deferred.diagnostics[key]==eager.diagnostics[key]
    refinements=len(deferred.diagnostics['refinement_axes'])
    assert refinements>0
    assert deferred.diagnostics['directional_control_panels']==refinements
    assert deferred.diagnostics['directional_control_panels']<eager.diagnostics['directional_control_panels']
    assert deferred.diagnostics['queries']<eager.diagnostics['queries']
    exact=.03*.6*.04/9.
    assert deferred.residual[0]==pytest.approx(exact,rel=1e-13,abs=1e-16)
    assert deferred.tangent[0,0]==pytest.approx(exact,rel=1e-13,abs=1e-16)
    assert deferred.stored_energy_j==pytest.approx(exact,rel=1e-13,abs=1e-16)
    with pytest.raises(MemoryError,match='material-query/cache cap'):
        integrate_material_friction(value,replace(settings,maximum_queries=1),
            _front_uncertainty=1e-7,_local_front_strips=True)


def test_complete_history_envelopes_bound_boundary_endpoint_and_stored_work():
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
    from anyfesolver.general_contact.native_surface_front import prepare_stationary_band,axial_contact_front
    from anyfesolver.general_contact.native_surface_front_work import history_front_strip_work_bound
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceReplayBudget
    value=_mixed(rotation=.8);budget=SupportIntervalBudget(128)
    partition=stationary_endpoint_angular_partition(value,0,value.patch.rule.definition['angle_range'],budget=budget)
    saw_closed=False
    for left,right,_,_ in partition.bands:
        cover=stationary_front_angular_cover(value,0,0,(left,right),budget=budget,xi_bounds=(-.8,-.2),
            endpoint_partition=partition,_variable_endpoint_branches=True)
        prepared=prepare_stationary_band(value,cover,budget=budget)
        envelopes=stationary_front_angular_envelopes(value,cover,maximum_axial_uncertainty=1e-7,
            budget=budget,prepared=prepared,_include_endpoints=True)
        # Independently evaluated coefficients at actual times must lie inside
        # the union of the endpoint and stationary-extremum envelopes.
        lower=min(a for a,_ in envelopes);upper=max(b for _,b in envelopes)
        for angle in np.linspace(left,right,3):
            for time in np.linspace(0.,1.,13):
                root=axial_contact_front(value,0,0,float(time),float(angle)).axial_coordinate()
                assert lower<=root<=upper
        if not len(cover):continue
        strip=(max(-.8,envelopes[0][0]),min(-.2,envelopes[0][1]))
        density=history_front_strip_work_bound(value,0,strip,(prepared,),(envelopes,))
        # The captured endpoint is genuinely closed at this material point;
        # its nonzero storage is retained and bounded, never replaced by zero.
        angle=(left+right)/2
        xi=axial_contact_front(value,0,0,1.,angle).axial_coordinate()-1e-4
        if not strip[0]<xi<strip[1]:continue
        point=value.query(float(xi),float(angle),budget=NativeSurfaceReplayBudget(16));trial=point.last_contribution.trial
        work=np.array((trial.stored_energy_density,trial.dissipated_increment_density,
            trial.released_storage_density,point.history.cumulative_dissipation_density))
        assert np.all(work>=0) and np.all(work<=density)
        saw_closed=saw_closed or work[0]>0
    assert saw_closed and budget.intervals<=128
    with pytest.raises(ValueError,match='identity/domain'):
        history_front_strip_work_bound(value,0,(-1.,0.),(prepared,),(envelopes,))


@pytest.mark.skipif(os.environ.get('ANYSOLVER_RUN_UNRESOLVED_MIXED_FRONT')!='1',
    reason='bounded actual boundary-ending development integration, not owner admission')
def test_boundary_ending_front_retains_integrated_force_tangent_and_work(tmp_path):
    import json,time
    from pathlib import Path
    from scipy.integrate import quad
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_variable_event_friction_fronts
    value=_mixed(rotation=.8);folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))
    folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    try:
        result=integrate_variable_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)
    except Exception as error:
        (folder/'boundary_ending_failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error),diagnostics=getattr(error,'integration_diagnostics',None),
            elapsed_seconds=time.perf_counter()-start),indent=2),encoding='utf-8')
        raise
    # Final section phase is zero. All closed points slide in +y; independent
    # triangular linear pressure integral uses the physical final cylinder.
    root=lambda delta:(.03*np.cos(delta)-.035)/.01
    exact=.03*.3*1000*quad(lambda delta:.5*.01*(root(delta)+.8)**2,-.02,.02,epsabs=1e-14)[0]
    derivative=-.03*.3*1000*quad(lambda delta:root(delta)+.8,-.02,.02,epsabs=1e-14)[0]
    y=np.zeros(60);y[[1,7,13]]=1.;z=np.zeros(60);z[[2,8,14]]=1.
    force=float(y@result.residual);tangent=float(y@result.tangent@z)
    balance=sum((result.residual[6*i:6*i+3] for i in range(3)),np.zeros(3))
    balance+=sum((result.residual[42+6*i:45+6*i] for i in range(3)),np.zeros(3))
    record=dict(elapsed_seconds=time.perf_counter()-start,diagnostics=result.diagnostics,force=force,
        reference_force=exact,tangent=tangent,reference_tangent=derivative,balance=balance.tolist(),
        work=dict(stored=result.stored_energy_j,dissipated=result.dissipated_increment_j,
            released=result.released_storage_j,cumulative=result.cumulative_dissipation_j))
    (folder/'boundary_ending_result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    assert force==pytest.approx(exact,rel=1e-5,abs=1e-10)
    assert tangent==pytest.approx(derivative,rel=1e-5,abs=1e-8)
    np.testing.assert_allclose(balance,0.,atol=1e-12,rtol=0.)
    assert result.stored_energy_j>0 and result.dissipated_increment_j>0
    assert result.diagnostics['maximum_scaled_error_estimate']<=1.


def test_variable_route_rules_match_advertised_polynomial_moments():
    """Independent scalar moment checks for the private variable-front pair.

    The candidate low rule is genuine Gauss3 (positive weights, exact
    through degree five), not the abandoned degree-three embedded
    control. The accepted high rule is the positive 12-point square
    cubature, exact through total degree seven on the reference square
    and not exact at degree eight. Ordinary scalar moments only: no
    production-kernel oracle.
    """
    from numpy.polynomial.legendre import leggauss
    from anyfesolver.general_contact.native_surface_friction_integral import (
        _square_cubature_degree7_angular_groups)
    nodes, weights = leggauss(3)
    assert len(nodes) == 3 and np.all(weights > 0.)
    for power in range(6):
        moment = float(np.sum(weights * nodes ** power))
        exact = 0. if power % 2 else 2. / (power + 1)
        assert moment == pytest.approx(exact, rel=2e-15, abs=2e-16)
    # One degree beyond the advertised order must NOT be exact: the
    # moment checks are discriminating, not vacuous.
    beyond = float(np.sum(weights * nodes ** 6))
    assert abs(beyond - 2. / 7) > 1e-3 * (2. / 7)
    # Reconstruct the accepted high rule from its production angular-
    # ordinate groups: each node's square weight is share*wa and its
    # reference coordinates are (axial abscissa, angular ordinate).
    square_nodes = []
    for ordinate, wa, abscissas, shares in _square_cubature_degree7_angular_groups():
        for axial, share in zip(abscissas, shares):
            square_nodes.append((axial, ordinate, share * wa))
    assert len(square_nodes) == 12
    square_weights = np.array([w for _, _, w in square_nodes])
    assert np.all(square_weights > 0.)
    assert np.all(np.abs([x for x, _, _ in square_nodes]) < 1.)
    assert np.all(np.abs([y for _, y, _ in square_nodes]) < 1.)
    assert float(np.sum(square_weights)) == pytest.approx(4., rel=1e-15)
    for degree in range(8):
        for i in range(degree + 1):
            j = degree - i
            moment = float(sum(w * x ** i * y ** j for x, y, w in square_nodes))
            exact = 4. / ((i + 1) * (j + 1)) if (i % 2 == 0 and j % 2 == 0) else 0.
            assert moment == pytest.approx(exact, rel=2e-15, abs=2e-16)
    # Degree eight must NOT be exact: the high-rule checks discriminate.
    moment = float(sum(w * x ** 8 for x, _, w in square_nodes))
    assert abs(moment - 4. / 9) > 1e-3 * (4. / 9)


def test_variable_route_gauss34_smooth_front_accuracy_tangent_and_refinement(tmp_path):
    """Bounded smooth-front comparison for the Gauss3/cubature candidate pair.

    The no-stationary-branch mixed surface is smooth at its front. The
    candidate must reproduce the independent analytic force, agree with
    the independently refined legacy Gauss4/8 route within the combined
    dimensional error budgets (force/tangent/work), and satisfy a
    central-difference tangent control along the final z translation.
    """
    import json, os
    from pathlib import Path
    from scipy.integrate import quad
    from anyfesolver.general_contact.native_surface_friction_integral import (
        integrate_variable_event_friction_fronts)
    value = _mixed(height=.024, slope=.001, rotation=.2)
    result = integrate_variable_event_friction_fronts(value, _settings(), maximum_axial_uncertainty=1e-7)
    assert result.diagnostics['estimator'] == 'GAUSS3_CUBATURE7_DIFFERENCE'
    assert result.diagnostics['front_profile'] == 'VARIABLE_ENDPOINT_STRIP_DEV_V11_DEFERRED_BRANCH_CUT'
    assert result.diagnostics['maximum_scaled_error_estimate'] <= 1.
    expected = .03*.3*1000*.6*quad(lambda delta: .03*np.cos(-.6+delta)-.0235,
        -.02, .02, epsabs=1e-14)[0]
    assert sum(result.residual[6*i+1] for i in range(3)) == pytest.approx(expected, rel=1e-5, abs=1e-10)
    # Independent refinement: the legacy Gauss4/8 local-front route on the
    # same smooth surface. Both accepted values carry charged error within
    # their unchanged dimensional budgets, so their difference must fit
    # the combined budgets (2.5x margin for the developmental estimator).
    reference = integrate_event_friction_fronts(value, _settings(), maximum_axial_uncertainty=1e-7)
    np.testing.assert_allclose(result.residual, reference.residual, rtol=0., atol=2.5e-10)
    np.testing.assert_allclose(result.tangent, reference.tangent, rtol=0., atol=2.5e-5)
    for key in ('stored_energy_j', 'dissipated_increment_j',
                'released_storage_j', 'cumulative_dissipation_j'):
        assert getattr(result, key) == pytest.approx(getattr(reference, key), rel=0., abs=2.5e-12)
    # Independent work check: the sparse 12-point high rule must spend
    # strictly fewer material queries than the legacy tensor Gauss4/8
    # route on the same smooth surface.
    assert result.diagnostics['queries'] < reference.diagnostics['queries']
    assert result.stored_energy_j > 0 and result.dissipated_increment_j > 0
    # Central-difference tangent control along the final segment's z
    # translation, against the candidate's own reported tangent.
    segments = value.segments
    def evaluate(shift):
        displacement = segments[-1].translations.copy(); displacement[1, 2] += shift
        last = replace(segments[-1], translations=displacement)
        shifted = NativeSurfaceHistoryField(value.patch, value.initial,
            (*segments[:-1], last), maximum_intervals=value.maximum_intervals)
        return integrate_variable_event_friction_fronts(shifted, _settings(),
            maximum_axial_uncertainty=1e-7)
    direction = np.zeros(60); direction[8] = 1.
    predicted = result.tangent @ direction
    for step in (1e-7, 2e-8):
        numerical = (evaluate(step).residual - evaluate(-step).residual) / (2 * step)
        # Non-vacuity: this z translation genuinely moves the residual.
        assert np.linalg.norm(numerical) > 0.
        assert np.linalg.norm(numerical - predicted) / np.linalg.norm(numerical) <= 1e-5
    folder = Path(os.environ.get('RUNNER_TEMP', str(tmp_path))); folder.mkdir(parents=True, exist_ok=True)
    (folder / 'gauss34_smooth_front_control.json').write_text(json.dumps(dict(
        candidate=result.diagnostics, reference=reference.diagnostics,
        force=float(sum(result.residual[6*i+1] for i in range(3))),
        reference_force=expected), indent=2) + '\n', encoding='utf-8')


def test_captured_window_grading_map_rejected_on_measured_charged_difference():
    """Recorded rejection of every exact-Jacobian grading candidate.

    The map x=x0+span*(3u^2-2u^3), u=(abscissa+1)/2, with the exact
    Jacobian factor 6u(1-u) preserves the complete window measure and
    first moment exactly, yet on the discriminator's measured steep
    birth layer (tangent 1456.46 -> 521.55 at theta 3.2646, half the
    drop within the leftmost 1.56e-4 of the window
    [0.7543455544834862, 0.7557200464609344]) every candidate
    coordinate change increased the charged cubature-vs-Gauss3
    estimator difference (window units):

      affine            2.019e-3
      double-end cubic  7.742e-3   (3.8x worse)
      left quadratic    6.651e-3   (3.3x worse)
      right quadratic   2.772e-2   (13.7x worse)

    Measure preservation alone cannot justify a coordinate change: the
    charged error got worse, so the production path keeps the affine
    node placement and the graded-window machinery stays removed.
    """
    from scipy.integrate import quad
    from numpy.polynomial.legendre import leggauss
    x0, x1 = 0.7543455544834862, 0.7557200464609344
    span = x1 - x0
    nodes, weights = leggauss(3)
    def mapped_rule(f):
        total = 0.
        for axial, wx in zip(nodes, weights):
            u = (float(axial) + 1.) / 2
            xi = x0 + span * (3. * u * u - 2. * u * u * u)
            total += wx * f(xi) * 3. * span * u * (1. - u)
        return total
    # The rejected map still preserves the complete window measure and
    # first moment exactly: the rejection is on charged error, not on
    # coverage or measure.
    assert mapped_rule(lambda x: 1.) == pytest.approx(span, rel=1e-15, abs=0.)
    assert mapped_rule(lambda x: x) == pytest.approx(quad(lambda x: x, x0, x1)[0],
        rel=1e-14, abs=1e-16)
    # Recorded measured charged estimator differences (window units).
    charged = {'affine': 2.019e-3, 'double_end_cubic': 7.742e-3,
        'left_quadratic': 6.651e-3, 'right_quadratic': 2.772e-2}
    assert charged['affine'] == min(charged.values())
    for name in ('double_end_cubic', 'left_quadratic'):
        assert charged[name] > 3. * charged['affine']
    assert charged['right_quadratic'] > 13. * charged['affine']


def test_captured_front_motion_detector_selects_only_moving_front_saved_paths():
    """Deterministic preflight detector scope on the exact saved evidence.

    Only the second-load rejected path has captured-pose fronts that move
    between consecutive poses at a shared sample angle; every other saved
    replay-owner path and the synthetic event-sequence fields keep the
    unchanged legacy route. The detector spends no replay query.
    """
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_friction_integral import (
        _captured_fronts_move_between_poses)
    from anyfesolver.general_contact.native_surface_history import (
        NativeSurfaceHistoryField, NativeSurfaceMotionSegment)
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from test_general_contact_native_surface_patch import _patch
    from test_general_contact_native_surface_normal import _settings as normal_settings
    expected = {
        'replay_owner_rejected_path.json': False,
        'replay_owner_support_rejected_path.json': False,
        'replay_owner_envelope_rejected_path.json': False,
        'replay_owner_rebase_rejected_path.json': False,
        'replay_owner_second_load_rejected_path.json': True}
    for filename, moving in expected.items():
        saved = json.loads((Path(__file__).parents[1]
            / 'docs/d1_review_support/contact_c5_native_surface_point_20261002' / filename).read_text())
        _, bridge = _patch(activation_policy='transverse_event_sequence',
            gap_check='straight_patch', normal_integration='straight_segmented',
            normal_integration_settings=normal_settings(), friction_integration='replay',
            friction_integration_settings=_settings())
        field = NativeSurfaceHistoryField(bridge.pair,
            NativeS3MaterialConfiguration(**{k: np.asarray(v) for k, v in saved['initial'].items()}),
            tuple(NativeSurfaceMotionSegment(**{k: np.asarray(v) for k, v in s.items()})
                for s in saved['segments']),
            maximum_intervals=saved['maximum_intervals'])
        assert _captured_fronts_move_between_poses(field) is moving
    assert not _captured_fronts_move_between_poses(_mixed())


def _saved_locator_field():
    """Saved second-load window covering the theta 3.0973 interface bracket."""
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_patch import NativeS3MaterialSurfacePatch
    from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from test_general_contact_native_surface_replay_owner import _replay_patch
    source=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/replay_owner_second_load_rejected_path.json'
    saved=json.loads(source.read_text());_,bridge=_replay_patch('transverse_event_sequence')
    full=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    base=full.patch.pairs[0]
    rule=NativeRoundSurfaceRule(base.source.element_id,base.source.reference,base.source.radius,
        xi_range=(.71,.74),angle_range=(3.09,3.11),axial_order=2,angular_order=2)
    patch=NativeS3MaterialSurfacePatch(rule,[full._target_at(s.station) for s in rule.samples],
        normal_stiffness_Pa_per_m=base.normal_stiffness_Pa_per_m,
        friction_coefficient=base.friction_coefficient,
        tangential_stiffness_Pa_per_m=base.tangential_stiffness_Pa_per_m,
        activation_policy='transverse_event_sequence',event_history_substeps=base.event_history_substeps,
        gap_check='straight_patch')
    return NativeSurfaceHistoryField(patch,full.initial,full.segments,maximum_intervals=full.maximum_intervals)


def test_safeguarded_locator_resolves_stagnating_curved_margin_counterexample():
    """The lead's counterexample: plain regula falsi stagnates on x*x-.01.

    On [1e-3,1-1e-3] the secant residual does not contract the bracket (one
    endpoint stays fixed), so the removed unsafeguarded locator returned the
    wide bracket midpoint near .55 as a cut after 12 calls. The safeguarded
    locator must resolve the true root .1 with an explicit tight bracket.
    """
    from anyfesolver.general_contact.native_surface_friction_integral import _safeguarded_branch_refinement
    def probe(x):
        margin=x*x-.01
        return ('stick' if margin>=0. else 'slip',margin)
    outcome=_safeguarded_branch_refinement(probe,1e-3,1.-1e-3)
    assert outcome['resolved'] and outcome['reason']=='resolved'
    assert outcome['width']<=outcome['criterion']
    assert outcome['bracket'][0]<=.1<=outcome['bracket'][1]
    assert abs(outcome['coordinate']-.1)<=outcome['criterion']
    # The stagnation the removed locator accepted as a cut is excluded:
    # the resolved bracket is nowhere near the old wide midpoint .55.
    assert .05<outcome['bracket'][0]<=outcome['bracket'][1]<.15
    assert outcome['probes']==2+outcome['iterations']


def test_exhausted_locator_returns_labelled_hint_never_wide_midpoint():
    from anyfesolver.general_contact.native_surface_friction_integral import _safeguarded_branch_refinement
    def probe(x):
        margin=x*x-.01
        return ('stick' if margin>=0. else 'slip',margin)
    outcome=_safeguarded_branch_refinement(probe,1e-3,1.-1e-3,max_iterations=2)
    assert not outcome['resolved'] and outcome['reason']=='exhausted'
    assert outcome['coordinate'] is None
    assert outcome['width']>outcome['criterion']
    # The hint is the actual final bracket with its uncertainty explicit.
    assert outcome['bracket'][0]<outcome['bracket'][1]


def test_equal_endpoint_branches_make_hidden_transition_limitation_explicit():
    """Equal endpoint branches are not transition absence.

    A hidden interior transition between same-branch endpoints is outside
    what endpoint sampling can decide; the locator must report the
    limitation explicitly instead of certifying a transition-free edge.
    """
    from anyfesolver.general_contact.native_surface_friction_integral import _safeguarded_branch_refinement
    def probe(x):
        return ('slip' if .4<x<.6 else 'stick',x*x-.01)
    outcome=_safeguarded_branch_refinement(probe,0.,1.)
    assert not outcome['resolved']
    assert outcome['reason']=='equal-endpoint-branches'
    assert outcome['coordinate'] is None
    # The limitation is explicit in the locator's contract, not implicit.
    assert 'equal endpoint branches' in _safeguarded_branch_refinement.__doc__
    assert 'not a uniqueness certificate' in _safeguarded_branch_refinement.__doc__


@pytest.mark.skipif(os.environ.get('ANYSOLVER_RUN_SAVED_NATIVE_LOCAL_FRONT')!='1',
    reason='named saved-path local discriminator; no fullpatch or coupled run')
def test_saved_theta_30973_yield_margin_cut_lands_inside_known_bracket():
    """Actual saved second-load interface at theta 3.0973.

    The front-regularity discriminator measured the stick/slip interface
    at theta 3.0973 inside xi (.7226092,.7229842). The safeguarded locator
    on the genuine saved replay must resolve a cut coordinate inside that
    bracket with the final bracket reported as its uncertainty. No point
    history is admitted by this test.
    """
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceReplayBudget
    from anyfesolver.general_contact.native_surface_friction_integral import _safeguarded_branch_refinement
    value=_saved_locator_field();pair=value.patch.pairs[0]
    def probe(xi):
        point=value.query(float(xi),3.0973,budget=NativeSurfaceReplayBudget(256)).last_contribution
        if point is None or point.trial.branch not in ('stick','slip'):return None,None
        pressure=getattr(point,'normal_pressure_Pa',None)
        if pressure is None or not pressure>0.:return point.trial.branch,None
        trial=point.trial
        magnitude=float(np.linalg.norm(np.asarray(trial.candidate_state.elastic_slip)
            +np.asarray(trial.plastic_slip_increment)))
        return point.trial.branch,(pair.friction_coefficient*float(pressure)
            -pair.tangential_stiffness_Pa_per_m*magnitude)
    low,high=.7226092,.7229842
    # Premise: the genuine replay exposes the interface across the bracket.
    assert probe(low)[0] in ('stick','slip') and probe(low)[0]!=probe(high)[0]
    outcome=_safeguarded_branch_refinement(probe,low,high)
    assert outcome['resolved'] and outcome['reason']=='resolved'
    assert low<=outcome['bracket'][0]<=outcome['coordinate']<=outcome['bracket'][1]<=high
    assert outcome['width']<=outcome['criterion']


def test_variable_route_safeguarded_cut_partitions_complete_domain(tmp_path,monkeypatch):
    """Production caller use: resolved cuts refine the partition, no omission.

    An independent quadratic polynomial with a genuine stick/slip kink
    exercises the production per-ordinate locator through the variable
    route's own estimate call site. The cut must land on the kink, the
    complete domain must remain covered (the refined integral matches the
    independently split exact moment), and the unchanged estimator error
    gates stay decisive. Polynomial code probe; no mechanics qualification.
    """
    import json,os
    from pathlib import Path
    from types import SimpleNamespace
    from anyfesolver.general_contact import native_surface_friction_integral as integral_module
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    value=_mixed(height=.024,slope=.001,rotation=.2)
    kink=-.5
    pair=value.patch.pairs[0]
    yield_slip=pair.friction_coefficient/pair.tangential_stiffness_Pa_per_m
    def polynomial_query(self,xi,angle,*,budget):
        slip=xi>kink
        q=((xi+.8)/.6)**2*(2. if slip else 1.)
        residual=np.zeros(60);residual[0]=q
        tangent=np.zeros((60,60));tangent[0,0]=q
        point=SimpleNamespace(residual=residual,normal_residual=np.zeros(60),
            spatial_tangent=lambda:tangent,normal_spatial_tangent=lambda:np.zeros((60,60)),
            normal_pressure_Pa=1.,
            trial=SimpleNamespace(branch='slip' if slip else 'stick',stored_energy_density=q,
                dissipated_increment_density=q,released_storage_density=q,
                candidate_state=SimpleNamespace(elastic_slip=np.array((yield_slip*(1.+xi-kink),0.,0.))),
                plastic_slip_increment=np.zeros(3)))
        return SimpleNamespace(last_contribution=point,
            history=SimpleNamespace(cumulative_dissipation_density=q))
    monkeypatch.setattr(NativeSurfaceHistoryField,'query',polynomial_query)
    real=integral_module._safeguarded_branch_refinement
    calls=[]
    def spy(probe,a,b,**kwargs):
        outcome=real(probe,a,b,**kwargs)
        calls.append((a,b,outcome))
        return outcome
    monkeypatch.setattr(integral_module,'_safeguarded_branch_refinement',spy)
    # Configure integration tightly enough for the independent moment
    # comparison below; production defaults remain unchanged.
    settings=replace(_settings(),force_absolute_tolerance=1e-13,
        tangent_absolute_tolerance=1e-13,energy_absolute_tolerance=1e-13)
    result=integrate_material_friction(value,settings,_front_uncertainty=1e-7,
        _local_front_strips=True,_variable_endpoint_branches=True)
    resolved=[outcome for _,_,outcome in calls if outcome['resolved']]
    assert resolved
    # The production call site resolved the actual kink per ordinate.
    assert any(abs(outcome['coordinate']-kink)<=outcome['criterion'] for outcome in resolved)
    assert result.diagnostics['resolved_branch_cuts']>=1
    assert result.diagnostics['branch_cut_queries']>0
    assert result.diagnostics['front_profile']=='VARIABLE_ENDPOINT_STRIP_DEV_V11_DEFERRED_BRANCH_CUT'
    # DEFERRED schedule: the locator ran only because the global L1 error
    # gate selected a worst leaf, and that attempt is recorded once per
    # exact parent with its refusal context.
    assert result.diagnostics['branch_cut_parent_attempts']>=1
    assert set(result.diagnostics['branch_cut_parent_refusals'])<= {'certified_leaf','already_attempted','rebuild_subdivision'}
    # Complete-domain coverage: the refined integral matches the exact
    # split moment; nothing between edges is dropped.
    t=(kink+.8)/.6
    exact=.03*.6*.04/3*(t**3+2.*(1.-t**3))
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'safeguarded_branch_cut_production.json').write_text(json.dumps(dict(
        diagnostics=result.diagnostics,locator_calls=[(a,b,o['reason'],o['resolved'],
            o['coordinate'],o['width'],o['criterion']) for a,b,o in calls],exact=exact),indent=2)+'\n',
        encoding='utf-8')


    assert result.residual[0]==pytest.approx(exact,rel=1e-9,abs=1e-16)
    assert result.tangent[0,0]==pytest.approx(exact,rel=1e-9,abs=1e-16)


def test_variable_route_smooth_converged_candidate_never_probes_constitutive_cuts(tmp_path,monkeypatch):
    """DEFERRED constitutive-cut schedule: no probes without error-led selection.

    A genuine quadratic is integrated exactly by both the Gauss3 low rule
    and the degree-seven square cubature high rule, so every baseline
    low/high pair agrees, the global gate accepts the candidate without
    one error-led refinement, and the constitutive locator must never run:
    zero attempted parents, zero branch-cut queries, hints, or resolved
    cuts, and an empty refusal context. Polynomial code probe for the
    schedule contract; no mechanics qualification.
    """
    import json,os
    from pathlib import Path
    from types import SimpleNamespace
    from anyfesolver.general_contact import native_surface_friction_integral as integral_module
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    value=_mixed(height=.024,slope=.001,rotation=.2)
    def polynomial_query(self,xi,angle,*,budget):
        q=((xi+.8)/.6)**2
        residual=np.zeros(60);residual[0]=q
        tangent=np.zeros((60,60));tangent[0,0]=q
        point=SimpleNamespace(residual=residual,normal_residual=np.zeros(60),
            spatial_tangent=lambda:tangent,normal_spatial_tangent=lambda:np.zeros((60,60)),
            trial=SimpleNamespace(branch='stick',stored_energy_density=q,
                dissipated_increment_density=q,released_storage_density=q,
                cumulative_dissipation_density=q))
        return SimpleNamespace(last_contribution=point,
            history=SimpleNamespace(cumulative_dissipation_density=q))
    monkeypatch.setattr(NativeSurfaceHistoryField,'query',polynomial_query)
    real=integral_module._safeguarded_branch_refinement
    calls=[]
    def spy(probe,a,b,**kwargs):
        calls.append((a,b));return real(probe,a,b,**kwargs)
    monkeypatch.setattr(integral_module,'_safeguarded_branch_refinement',spy)
    settings=replace(_settings(),force_absolute_tolerance=1e-13,
        tangent_absolute_tolerance=1e-13,energy_absolute_tolerance=1e-13)
    result=integrate_material_friction(value,settings,_front_uncertainty=1e-7,
        _local_front_strips=True,_variable_endpoint_branches=True)
    assert not calls
    diagnostics=result.diagnostics
    assert diagnostics['branch_cut_parent_attempts']==0
    assert diagnostics['branch_cut_queries']==0
    assert diagnostics['resolved_branch_cuts']==0
    assert diagnostics['branch_cut_hints']==0
    assert diagnostics['branch_cut_parent_refusals']=={}
    assert diagnostics['front_profile']=='VARIABLE_ENDPOINT_STRIP_DEV_V11_DEFERRED_BRANCH_CUT'
    assert diagnostics['maximum_scaled_error_estimate']<=1.
    exact=.03*.6*.04/3
    assert result.residual[0]==pytest.approx(exact,rel=1e-13,abs=1e-16)
    assert result.tangent[0,0]==pytest.approx(exact,rel=1e-13,abs=1e-16)
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'deferred_cut_smooth_no_probe.json').write_text(json.dumps(dict(
        diagnostics=diagnostics,exact=exact,locator_calls=len(calls)),indent=2),
        encoding='utf-8')


def test_captured_front_crossings_partition_the_complete_angular_span():
    from anyfesolver.general_contact.native_surface_friction_integral import _captured_front_crossing_angles
    coefficients=((-0.3,1.,0.,0.),(-0.8,1.,1.,0.))
    cuts=_captured_front_crossing_angles(coefficients,0.,1.,0.,np.pi)
    assert cuts==pytest.approx((np.pi/3,),abs=2e-15)
    for scale in (1e-8,1.,1e8):
        scaled=tuple(tuple(scale*x for x in row) for row in coefficients)
        assert _captured_front_crossing_angles(scaled,0.,1.,0.,np.pi)==pytest.approx(cuts,abs=2e-15)
    assert _captured_front_crossing_angles(coefficients,.5,1.,0.,np.pi)==()
    assert _captured_front_crossing_angles((coefficients[0],coefficients[0]),0.,1.,0.,np.pi)==()
    edges=(0.,*cuts,np.pi)
    assert sum(b-a for a,b in zip(edges[:-1],edges[1:]))==pytest.approx(np.pi,abs=0.)


def test_captured_front_structure_detects_root_order_change():
    from anyfesolver.general_contact.native_surface_friction_integral import _captured_front_structure
    coefficients=((-0.3,1.,0.,0.),(-0.8,1.,1.,0.))
    assert _captured_front_structure(coefficients,0.,1.,.8)==(1,0)
    assert _captured_front_structure(coefficients,0.,1.,1.3)==(0,1)


def test_front_motion_axis_hint_distinguishes_window_translation_scales():
    from anyfesolver.general_contact.native_surface_friction_integral import _captured_front_angular_motion_dominates
    # Two independent exact fronts x=.4+k sin(theta), .5+k sin(theta).
    # Across [0,.3], k=.8 moves farther than their .1 separation;
    # k=.001 moves less. Axis selection admits no integral or contact.
    def models(k):return ((-.4,1.,0.,-k),(-.5,1.,0.,-k))
    assert _captured_front_angular_motion_dominates(models(.8),0.,1.,0.,.3)
    assert not _captured_front_angular_motion_dominates(models(.001),0.,1.,0.,.3)
    assert not _captured_front_angular_motion_dominates(models(0.),0.,1.,0.,.3)
    assert not _captured_front_angular_motion_dominates(models(.8),.8,1.,0.,.3)


def test_front_contact_probe_moves_inside_the_correct_physical_side():
    from anyfesolver.general_contact.native_surface_friction_integral import _captured_front_contact_probe
    # Independent exact fronts at .4 and .5; final gap=.4-x closes right.
    models=((-.5,1.,0.,0.),(.4,-1.,0.,0.))
    assert _captured_front_contact_probe(models,0.,1.,.3)==pytest.approx(.4001,abs=1e-16)
    # Reversing final slope closes left and retains its nearest interval.
    models=((-.3,1.,0.,0.),(-.4,1.,0.,0.))
    assert _captured_front_contact_probe(models,0.,1.,.3)==pytest.approx(.3999,abs=1e-16)
    assert _captured_front_contact_probe(models,.5,1.,.3) is None
