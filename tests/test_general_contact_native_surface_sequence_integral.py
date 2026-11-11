"""Complete open/contact/open support and integrated discrete friction work."""
from math import acos,sin,cos,fsum
import numpy as np
import pytest
from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
from anyfesolver.general_contact.native_surface_support import classify_history_panel,SupportIntervalBudget
from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
from test_general_contact_native_surface_support import _field
from test_general_contact_native_surface_friction_integral import _settings


def field(height=.025):
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    return _field(height=height,rotation=-.8,segments=(segment,),activation_policy='transverse_event_sequence')


def test_sequence_panel_support_and_tangency():
    value=field();angles=value.patch.rule.definition['angle_range']
    support=classify_history_panel(value,0,(-.8,-.2),angles,budget=SupportIntervalBudget(128))
    assert support.status=='resolved'
    try:
        result=classify_history_panel(field(.03),0,(-.8,-.2),angles,budget=SupportIntervalBudget(128))
    except MemoryError as error:
        assert 'support interval cap' in str(error)
    else:
        assert result.status=='unresolved'


def test_integrated_hidden_pulse_retains_independent_discrete_work():
    value=field();result=integrate_material_friction(value,_settings())
    theta=acos(.025/.03);elastic=0.;previous=.03*sin(-theta);work=[]
    for index in range(1,32):
        angle=-theta+2*theta*index/32;y=.03*sin(angle)
        trial=elastic+y-previous;cap=.3*1000*(.03*cos(angle)-.025)
        next_elastic=min(trial,cap/1000);work.append(cap*(trial-next_elastic))
        elastic=next_elastic;previous=y
    area=.03*.6*.04
    assert result.dissipated_increment_j==pytest.approx(area*fsum(work),rel=1e-8,abs=1e-14)
    assert result.cumulative_dissipation_j==pytest.approx(result.dissipated_increment_j,rel=1e-12)
    assert result.released_storage_j==pytest.approx(area*.5*1000*elastic**2,rel=1e-8,abs=1e-14)
    assert result.dissipated_increment_j>0
    assert np.array_equal(result.residual,np.zeros(60))
    assert result.diagnostics['open_material_queries_certified']==0


def test_saved_two_event_neighborhood_has_complete_support(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from test_general_contact_native_surface_replay_owner import _patch,normal_settings,friction_settings
    _,bridge=_patch(activation_policy='transverse_event_sequence',gap_check='straight_patch',
        normal_integration='straight_segmented',normal_integration_settings=normal_settings(),
        friction_integration='replay',friction_integration_settings=friction_settings())
    folder=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved=json.loads((folder/'replay_owner_second_load_rejected_path.json').read_text())
    value=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    xi=.7244843513242318;angle=3.1990185900472157
    budget=SupportIntervalBudget(128)
    result=classify_history_panel(value,1,(xi-1e-10,xi+1e-10),(angle-1e-8,angle+1e-8),budget=budget)
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'saved_sequence_neighborhood.json'
    output.write_text(json.dumps(dict(status=result.status,reason=result.reason,region=result.region,
        support_intervals=budget.intervals),indent=2)+'\n')
    assert result.status=='resolved'


def test_saved_full_patch_endpoint_scope_discriminator(tmp_path):
    """Classify the first exhausted rectangle; no refinement or coupled retry."""
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    from anyfesolver.general_contact.native_surface_support import _minimum,segment_gap_slab_enclosure
    from test_general_contact_native_surface_replay_owner import _replay_patch
    folder=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved=json.loads((folder/'replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch('transverse_event_sequence')
    value=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    xis=(.7250000000000001,.7343750000000001);angles=(3.2146078724025617,3.2201636457638987)
    pose=value.initial;rows=[]
    for index,segment in enumerate(value.segments):
        end=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
        ranges=[]
        for endpoint in (pose,end):
            args=(endpoint,value.patch.target,value.patch.source.reference.nodal_triads[0],
                value.patch.rule.definition['radius'],1,xis,angles)
            lower,pad=_minimum(*args);negative_upper,_=_minimum(*args,gap_sign=-1.)
            ranges.append(dict(lower=lower,upper=-negative_upper,padding=pad))
        bounds=segment_gap_slab_enclosure(value,1,xis,angles,index)(0.,1.)
        rows.append(dict(segment=index,endpoints=ranges,derivative=[bounds.derivative_lower,bounds.derivative_upper]))
        pose=end
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'endpoint_scope_discriminator.json'
    output.write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==len(saved['segments'])
    assert rows[1]['endpoints'][0]['lower']<0<rows[1]['endpoints'][0]['upper']
    assert rows[1]['derivative'][1]<0
    assert classify_history_panel(value,1,xis,angles,budget=SupportIntervalBudget(128)).status=='resolved'


def test_moving_front_integral_across_pulse_appearance(tmp_path):
    from dataclasses import replace
    from scipy.integrate import quad
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_open_event_friction_fronts
    value=field(.035)
    positions=value.initial.positions.copy();positions[:,2]+=.01*positions[:,0]
    value=NativeSurfaceHistoryField(value.patch,replace(value.initial,positions=positions),value.segments)
    try:
        result=integrate_open_event_friction_fronts(value,_settings(),maximum_axial_uncertainty=1e-7)
    except Exception as error:
        import json,os
        from pathlib import Path
        output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'moving_front_failure.json'
        output.write_text(json.dumps(dict(error=str(error),diagnostics=getattr(error,'integration_diagnostics',None)),indent=2)+'\n')
        raise
    def work(xi):
        height=.035+.01*xi
        if height>=.03:return 0.
        theta=acos(height/.03);elastic=0.;previous=.03*sin(-theta);terms=[]
        for i in range(1,32):
            angle=-theta+2*theta*i/32;y=.03*sin(angle)
            trial=elastic+y-previous;cap=.3*1000*(.03*cos(angle)-height)
            next_elastic=min(trial,cap/1000);terms.append(cap*(trial-next_elastic))
            elastic=next_elastic;previous=y
        return fsum(terms)
    expected=.03*.04*quad(work,-.8,-.5,epsabs=1e-13,epsrel=1e-10)[0]
    assert result.dissipated_increment_j==pytest.approx(expected,rel=1e-5,abs=1e-12)
    assert result.dissipated_increment_j>0
    assert np.array_equal(result.residual,np.zeros(60))


def test_front_strip_bound_contains_independent_discrete_work():
    from dataclasses import replace
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
    from anyfesolver.general_contact.native_surface_front_work import open_front_strip_work_bound
    value=field(.035);positions=value.initial.positions.copy();positions[:,2]+=.01*positions[:,0]
    value=NativeSurfaceHistoryField(value.patch,replace(value.initial,positions=positions),value.segments)
    width=1e-7;bounds=open_front_strip_work_bound(value,0,width)
    height=.03-.01*width/2;theta=acos(height/.03)
    elastic=0.;previous=.03*sin(-theta);terms=[]
    for i in range(1,32):
        angle=-theta+2*theta*i/32;y=.03*sin(angle)
        trial=elastic+y-previous;cap=.3*1000*(.03*cos(angle)-height)
        next_elastic=min(trial,cap/1000);terms.append(cap*(trial-next_elastic))
        elastic=next_elastic;previous=y
    assert 0<fsum(terms)<=bounds[1]
    assert .5*1000*elastic**2<=bounds[2]
    assert bounds[0]==0 and bounds[3]==bounds[1]
    repeated=NativeSurfaceHistoryField(value.patch,value.initial,value.segments*2)
    with pytest.raises(ValueError,match='one released-origin segment'):
        open_front_strip_work_bound(repeated,0,width)
