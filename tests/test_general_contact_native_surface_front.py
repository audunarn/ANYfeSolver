"""Moving axial front geometry against authoritative gaps and time differences."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver.general_contact.native_surface_front import axial_contact_front
from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
from anyfesolver.general_contact.native_surface_path import material_path_pose
from test_general_contact_native_surface_replay_owner import _replay_patch


def test_front_nonfinite_derived_values_reject():
    from anyfesolver.general_contact.native_surface_front import AxialContactFront
    from dataclasses import replace
    front=AxialContactFront(1e308,1e-308,1.,1.,1.,1.,1.)
    with pytest.raises(ValueError,match='nonfinite'):front.axial_coordinate()
    front=replace(front,b=1e308,a_rate=1e308,a_acceleration=1e308)
    with pytest.raises(ValueError,match='nonfinite'):_=front.stationary_numerator
    with pytest.raises(ValueError,match='nonfinite'):_=front.stationary_rate


def test_whole_angular_stationary_branch_matches_analytic_geometry():
    from dataclasses import replace
    from test_general_contact_native_surface_support import _field
    from anyfesolver.general_contact.native_surface_front import stationary_front_angular_cover,stationary_front_slab,stationary_front_partition
    from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    original=_field(height=.035,rotation=-.8,segments=(segment,))
    positions=original.initial.positions.copy();positions[:,2]+=.01*positions[:,0]
    field=NativeSurfaceHistoryField(original.patch,replace(original.initial,positions=positions),original.segments)
    angles=field.patch.rule.definition['angle_range'];budget=SupportIntervalBudget(128)
    branches=stationary_front_angular_cover(field,0,0,angles,budget=budget)
    assert branches
    for angle in np.linspace(*angles,5):
        expected=.5-(angle-1.5*np.pi)/1.6
        assert sum(left<expected<right for left,right in branches)==1
        fronts,count,roots,endpoints=stationary_front_partition(field,0,0,float(angle))
        assert len(roots)==1 and roots[0].fraction==pytest.approx(expected,abs=1e-12)
    bounds=stationary_front_slab(field,0,0,sum(angles)/2,0.,1.,angle_bounds=angles)
    for time in np.linspace(0,1,7):
        for angle in np.linspace(*angles,5):
            point=axial_contact_front(field,0,0,float(time),float(angle))
            assert bounds[2]<=point.stationary_numerator<=bounds[3]
            assert bounds[4]<=point.stationary_rate<=bounds[5]
    with pytest.raises(MemoryError,match='interval cap'):
        stationary_front_angular_cover(field,0,0,angles,budget=SupportIntervalBudget(1))


@pytest.mark.parametrize('case',['two_roots','wide_point_b','endpoint','singular'])
def test_isolation_independent_polynomial_controls(monkeypatch,case):
    from anyfesolver.general_contact import native_surface_front as module
    def coefficient(t):
        if case=='endpoint':return module.AxialContactFront(.5*t*t,1.,t,0.,1.,0.,1.)
        return module.AxialContactFront(t**3/3-t*t/2+.16*t,
            0. if case=='singular' else 1.,t*t-t+.16,0.,2*t-1,0.,1.)
    def enclosure(_field,_cell,_segment,_angle,left,right,*,include_a=False):
        times=[left,right]+([.5] if left<.5<right else [])
        values=[coefficient(t) for t in times];pad=1e-14
        n=[v.stationary_numerator for v in values];d=[v.stationary_rate for v in values]
        b=values[0].b
        result=(b-pad,b+pad,min(n)-pad,max(n)+pad,min(d)-pad,max(d)+pad)
        if case=='wide_point_b' and left==right and abs(n[0])<1e-12:
            result=(-2.,2.,*result[2:])
        extrema=[left,right]+[t for t in (.2,.8) if left<t<right]
        a=[coefficient(t).a for t in extrema]
        return (*result,min(a)-pad,max(a)+pad) if include_a else result
    monkeypatch.setattr(module,'axial_contact_front',lambda f,c,s,t,a:coefficient(t))
    monkeypatch.setattr(module,'stationary_front_slab',enclosure)
    if case in ('two_roots','wide_point_b'):
        fronts,count,roots,endpoints=module.stationary_front_partition(None,0,0,0.,maximum_axial_uncertainty=1e-10)
        assert [v.fraction for v in roots]==pytest.approx([.2,.8],abs=1e-12)
        assert [v.axial_coordinate for v in roots]==pytest.approx([-coefficient(t).a for t in (.2,.8)],abs=1e-12)
        assert count<=128 and len(fronts)==4
        assert all(v.axial_interval[0]<=v.axial_coordinate<=v.axial_interval[1] for v in endpoints)
    elif case=='endpoint':
        with pytest.raises(ValueError,match='endpoint sign unresolved'):
            module.stationary_front_partition(None,0,0,0.)
    else:
        with pytest.raises(MemoryError,match='interval cap'):
            module.stationary_front_partition(None,0,0,0.,maximum_intervals=4)


def test_saved_stationary_front_geometry_and_derivatives(tmp_path):
    folder=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved=json.loads((folder/'replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch('transverse_event_sequence')
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    angle=3.1990185900472157
    get=lambda t:axial_contact_front(field,1,1,t,angle)
    centre=get(.5)
    for h in (1e-3,3e-4):
        left,right=get(.5-h),get(.5+h)
        assert (right.a-left.a)/(2*h)==pytest.approx(centre.a_rate,rel=1e-5,abs=1e-10)
        assert (right.b-left.b)/(2*h)==pytest.approx(centre.b_rate,rel=1e-5,abs=1e-10)
        assert (right.a_rate-left.a_rate)/(2*h)==pytest.approx(centre.a_acceleration,rel=1e-5,abs=1e-10)
        assert (right.stationary_numerator-left.stationary_numerator)/(2*h)==pytest.approx(centre.stationary_rate,rel=1e-5,abs=1e-10)
    root=brentq(lambda t:get(t).stationary_numerator,.3,.7,xtol=1e-12)
    front=get(root);xi=front.axial_coordinate()
    pose=field.initial
    for segment in field.segments[:1]:pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
    segment=field.segments[1]
    pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,root)
    for coordinate in (xi,xi-.002,xi+.002):
        pair=field.pair_at(coordinate,angle)
        x=pair.source.evaluate(pose.positions,pose.position_low,pose.cell_rotations).surface_position
        target=pair.target.physical_pose(pose.shell_displacement)
        gap=float((x-np.asarray(target.position))@np.asarray(target.material_frame)[:,2])
        assert (front.a+front.b*coordinate)/front.plane_area==pytest.approx(gap,abs=2e-16)
    assert xi==pytest.approx(.7244843413242318,abs=2e-9)
    assert abs(front.stationary_numerator)<1e-14
    from anyfesolver.general_contact.native_surface_front import stationary_front_slab,stationary_front_partition
    bounds=stationary_front_slab(field,1,1,angle,0.,1.)
    for t in np.linspace(0.,1.,9):
        sample=get(float(t))
        for scalar,(lower,upper) in zip((sample.b,sample.stationary_numerator,sample.stationary_rate),
                (bounds[:2],bounds[2:4],bounds[4:])):
            assert lower<=scalar<=upper
    partition,intervals,isolated,endpoints=stationary_front_partition(field,1,1,angle)
    assert len(partition)==3
    assert partition[1][0]==pytest.approx(root,abs=2e-8)
    assert partition[1][1]==pytest.approx(xi,abs=2e-12)
    assert len(isolated)==1
    assert isolated[0].time_bracket[0]<=root<=isolated[0].time_bracket[1]
    assert isolated[0].axial_interval[0]<=xi<=isolated[0].axial_interval[1]
    assert isolated[0].axial_interval[1]-isolated[0].axial_interval[0]<1e-7
    with pytest.raises(MemoryError,match='interval cap'):
        stationary_front_partition(field,1,1,angle,maximum_intervals=1)
    with pytest.raises(ValueError,match='axial uncertainty'):
        stationary_front_partition(field,1,1,angle,maximum_axial_uncertainty=1e-20)
    import os
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'axial_stationary_front.json'
    from dataclasses import asdict
    output.write_text(json.dumps(dict(time=root,xi=xi,angle=angle,stationary_numerator=front.stationary_numerator,
        axial_gap_derivative=front.b/front.plane_area,partition=partition,intervals=intervals,
        isolated=[asdict(v) for v in isolated]),indent=2)+'\n')


def test_rotating_round_surface_angular_coefficients_and_tight_bounds():
    # Independent cross-products at a directly integrated rigid rotation.
    from scipy.spatial.transform import Rotation
    from test_general_contact_native_surface_support import _field
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_angular_coefficients
    from anyfesolver.general_contact.native_surface_front import stationary_front_slab
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=.035,rotation=-.8,segments=(segment,))
    corners=field.patch.target.adapter.current_positions(field.initial.shell_displacement)
    normal=np.cross(corners[1]-corners[0],corners[2]-corners[0])
    reference=field.patch.source.reference.nodal_triads[0]
    for time in (0.,.31,1.):
        a,ap,app,_=stationary_angular_coefficients(field,0,0,time,4.7)
        frame=Rotation.from_rotvec((- .8+1.6*time,0.,0.)).as_matrix()@reference
        for angle in (4.69,4.71,4.73):
            arm=.03*(np.cos(angle)*frame[:,1]+np.sin(angle)*frame[:,2])
            omega=np.array((1.6,0.,0.))
            basis=np.array((1.,np.cos(angle),np.sin(angle)))
            assert np.dot(ap,basis)==pytest.approx(normal@np.cross(omega,arm),abs=1e-15)
            assert np.dot(app,basis)==pytest.approx(normal@np.cross(omega,np.cross(omega,arm)),abs=1e-15)
    band=field.patch.rule.definition['angle_range']
    bounds,_=stationary_front_slab(field,0,0,sum(band)/2,.2,.8,angle_bounds=band,include_a=True,_return_error_controls=True)
    for time in np.linspace(.2,.8,9):
        for angle in np.linspace(*band,9):
            value=axial_contact_front(field,0,0,float(time),float(angle))
            assert bounds[2]<=value.stationary_numerator<=bounds[3]
            assert bounds[4]<=value.stationary_rate<=bounds[5]
            assert bounds[6]<=value.a<=bounds[7]
