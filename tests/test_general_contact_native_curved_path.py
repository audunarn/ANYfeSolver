"""Whole-patch affine/Exp controls with independent hidden-path witnesses."""
from dataclasses import asdict
import json,os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver.general_contact.native_curved_plane_work import NativeCurvedMaterialPlaneWork
from anyfesolver.general_contact.native_curved_geometry import curved_patch_geometry_control
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
from test_general_contact_native_retained_station import _reference
from test_general_contact_native_surface_bounds import _independent_point


def test_projected_zero_roll_bounds_cover_independent_planar_derivatives():
    from anyfesolver.general_contact.native_curved_geometry import _reference_plane_derivative_bounds
    reference=_reference(True);K=np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,0.)))
    for U in (np.eye(3),Rotation.from_rotvec((.3,-.2,.4)).as_matrix()):
        for n in (U@np.array((0.,0.,1.)),U@np.array((1e-8,-2e-8,1.)),np.array((.2,.3,.7))):
            for cell,left,right in ((0,-.95,-.05),(1,.05,.95)):
                first,second=_reference_plane_derivative_bounds(reference,cell,U,n)
                for xi in np.linspace(left,right,31):
                    J=np.sqrt(1+.16*xi*xi)
                    frame=np.array(((1/J,.4*xi/J,0.),(-.4*xi/J,1/J,0.),(0.,0.,1.)))
                    rate=-.4/(J*J);rate2=.128*xi/(J**4)
                    prime=rate*K@frame;twice=rate2*K@frame+rate*rate*K@K@frame
                    assert np.linalg.norm(n@U@prime)<=first
                    assert np.linalg.norm(n@U@twice)<=second
                if np.allclose(n,U@np.array((0.,0.,1.)),rtol=0,atol=1e-15):
                    assert first<1e-8 and second<1e-8


def test_projected_bounds_nonzero_roll_fallback_and_clipping_ownership():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver.general_contact.native_curved_geometry import (
        _reference_plane_derivative_bounds,_reference_norm_bounds,
        _reference_second_bound,_reference_unclipped_interval)
    ref=_reference(True);frames=[]
    for frame,roll in zip(ref.nodal_triads,(.1,.4,-.2)):
        frames.append(np.column_stack((frame[:,0],np.cos(roll)*frame[:,1]+np.sin(roll)*frame[:,2],
                                       -np.sin(roll)*frame[:,1]+np.cos(roll)*frame[:,2])))
    twisted=CenteredCurvedBeam3ReferenceGeometry(ref.coordinates,np.array(frames))
    for cell in (0,1):
        assert twisted._half_frame_data[cell][1]!=0.
        assert _reference_plane_derivative_bounds(twisted,cell,np.eye(3),(0.,0.,1.))==(
            _reference_norm_bounds(twisted,cell,actual_roll=True)[2],_reference_second_bound(twisted,cell))
    assert not _reference_unclipped_interval(ref,0,-1.,-.99)
    assert _reference_unclipped_interval(ref,0,-.8,-.2)


def test_root_band_without_interior_float_reports_parameter_resolution():
    from anyfesolver.general_contact.native_curved_plane_work import _root_band_station
    law=_law();p=law.reference.coordinates;U=np.tile(np.eye(3),(2,1,1))
    with pytest.raises(ValueError,match='root-band parameter resolution exhausted'):
        _root_band_station(law.rule,p,np.zeros((3,3)),U,1,.5,np.nextafter(.5,np.inf),
                           .2,5.8,law.origin,law.normal)


def _law(origin=(0.,0.,0.),normal=(0.,0.,1.),radius=.005):
    reference=_reference(True)
    rule=NativeRoundSurfaceRule(7,reference,radius,xi_range=(-.9,.8),
        angle_range=(.2,5.8),axial_order=1,angular_order=1)
    return NativeCurvedMaterialPlaneWork(rule,plane_origin_m=origin,plane_normal=normal,
        normal_stiffness_Pa_per_m=1e4,
        integration_settings=StraightNormalIntegrationSettings(1e-10,1e-8,1e-6,64,1.))


def _pose(p,U):
    return NativeS3MaterialConfiguration(p,np.zeros((3,3)),np.tile(U,(2,1,1)),np.zeros(0))


def _independent_path_pose(start,move,spin,fraction):
    return (start.positions+start.position_low+fraction*move,
        np.stack([Rotation.from_rotvec(fraction*v).as_matrix()@U for v,U in zip(spin,start.cell_rotations)]))


def _independent_orientation(xi,angle,radius,p,U):
    # Analytic parabola/untwisted frame, no production derivative/area helper.
    cell=0 if xi<0 else 1;t=xi+1 if cell==0 else xi
    J=np.sqrt(1+.16*xi*xi)
    frame=np.array(((1/J,.4*xi/J,0.),(-.4*xi/J,1/J,0.),(0.,0.,1.)))
    K=np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,0.)))
    prime=(-.4/(J*J))*K@frame
    radial=radius*np.array((0.,np.cos(angle),np.sin(angle)))
    angular=radius*np.array((0.,-np.sin(angle),np.cos(angle)))
    dx=p[cell+1]-p[cell]+U[cell]@(np.array((0.,-.4,0.))*(t-.5)+prime@radial)
    da=U[cell]@frame@angular;out=U[cell]@frame@radial
    return float(np.cross(da,dx)@out/radius**2)


def _record(name,value):
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-'+name+'.json').write_text(json.dumps(value,indent=2))


@pytest.mark.parametrize('pressed',(False,True))
def test_whole_curved_path_allows_approach_with_explicit_gap_budget(pressed):
    law=_law();p=law.reference.coordinates.copy();p[:,2]=.04
    start=_pose(p,np.eye(3));move=np.zeros((3,3));spin=np.zeros((2,3))
    move[:,2]=-.03502 if pressed else -.0349
    result=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    assert result.certified and result.orientation_lower>0 and result.gap_lower_m>=-5e-5
    minimum=orientation=np.inf
    for fraction in np.linspace(0,1,9):
        p,U=_independent_path_pose(start,move,spin,fraction)
        for xi in (-.89,-.6,-.2,.2,.7):
            for angle in (.3,1.7,3.1,4.7,5.7):
                point=_independent_point(xi,angle,.005,p,np.zeros((3,3)),U,False)
                minimum=min(minimum,float(law.normal@(point-law.origin)))
                orientation=min(orientation,_independent_orientation(xi,angle,.005,p,U))
    assert result.gap_lower_m<=minimum and result.orientation_lower<=orientation
    _record('curved-approach-'+str(pressed),dict(control=asdict(result),
        independently_sampled_gap_m=minimum,independently_sampled_orientation=orientation))


def test_curved_internal_rotation_at_fixed_external_nodes_has_whole_path_control():
    law=_law();p=law.reference.coordinates.copy();p[:,2]=.04
    start=_pose(p,np.eye(3));move=np.zeros((3,3));spin=np.array(((.02,.04,0.),(-.01,.03,0.)))
    result=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    assert result.certified
    old=_independent_point(-.4,1.1,.005,p,np.zeros((3,3)),start.cell_rotations,False)
    _,U=_independent_path_pose(start,move,spin,1.)
    new=_independent_point(-.4,1.1,.005,p,np.zeros((3,3)),U,False)
    assert np.linalg.norm(new-old)>1e-4
    np.testing.assert_array_equal(start.positions,p)


def test_curved_whole_path_common_rigid_transform_preserves_geometry_admission():
    law=_law();p=law.reference.coordinates.copy();p[:,2]=.04
    start=_pose(p,np.eye(3));move=np.tile((.001,-.002,-.02),(3,1))
    spin=np.array(((.02,.03,0.),(-.01,.02,0.)))
    first=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    G=Rotation.from_rotvec((.3,-.2,.1)).as_matrix();offset=np.array((1.2,-.6,.3))
    transformed=_law(origin=G@law.origin+offset,normal=G@law.normal)
    moved=NativeS3MaterialConfiguration(start.positions@G.T+offset,start.position_low@G.T,
        G@start.cell_rotations,np.zeros(0))
    second=transformed.certify_path(moved,move@G.T,spin@G.T,
        gap_tolerance_m=5e-5,maximum_intervals=64)
    assert first.certified and second.certified
    assert second.gap_lower_m==pytest.approx(first.gap_lower_m,abs=2e-10)
    assert second.orientation_lower==pytest.approx(first.orientation_lower,abs=2e-10)


def test_endpoint_safe_curved_path_cannot_hide_internal_orientation_inversion():
    law=_law(origin=(0.,0.,-1.));ref=law.reference
    first=Rotation.from_rotvec((0.,0.,-1.4)).as_matrix()
    last=Rotation.from_rotvec((0.,0.,1.4)).as_matrix()
    deformation=ref.coordinates[:,0,None]*np.array((-.3,0.,0.))
    p=ref.coordinates@first.T+deformation;end=ref.coordinates@last.T+deformation
    start=_pose(p,first);move=end-p;spin=np.tile((0.,0.,2.8),(2,1))
    for fraction in (0.,1.):
        q,U=_independent_path_pose(start,move,spin,fraction)
        control=curved_patch_geometry_control(law.rule,q,np.zeros((3,3)),U,law.origin,law.normal)
        assert control.orientation_lower>0 and control.gap_lower_m>0
    q,U=_independent_path_pose(start,move,spin,.5)
    witness=_independent_orientation(-.4,1.1,.005,q,U)
    assert witness<-.05
    result=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    assert not result.certified and 'orientation' in result.reason
    _record('curved-hidden-orientation',dict(control=asdict(result),independent_witness=witness))


def test_endpoint_safe_curved_path_cannot_hide_plane_crossing():
    law=_law(origin=(0.,.045,0.),normal=(0.,-1.,0.));ref=law.reference
    first=Rotation.from_rotvec((-1.4,0.,0.)).as_matrix()
    last=Rotation.from_rotvec((1.4,0.,0.)).as_matrix()
    p=ref.coordinates@first.T;end=ref.coordinates@last.T
    start=_pose(p,first);move=end-p;spin=np.tile((2.8,0.,0.),(2,1))
    for fraction in (0.,1.):
        q,U=_independent_path_pose(start,move,spin,fraction)
        control=curved_patch_geometry_control(law.rule,q,np.zeros((3,3)),U,law.origin,law.normal)
        assert control.orientation_lower>0 and control.gap_lower_m>0
    q,U=_independent_path_pose(start,move,spin,.5)
    witness=law.normal@(_independent_point(-.5,.3,.005,q,np.zeros((3,3)),U,False)-law.origin)
    assert witness<-.02
    result=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    assert not result.certified
    _record('curved-hidden-crossing',dict(control=asdict(result),independent_gap_m=float(witness)))


def test_curved_positive_rotating_path_subdivision_limits_cancellation_and_identity():
    law=_law(origin=(0.,0.,-1.));ref=law.reference
    first=Rotation.from_rotvec((0.,0.,-1.)).as_matrix()
    last=Rotation.from_rotvec((0.,0.,1.)).as_matrix()
    p=ref.coordinates@first.T;start=_pose(p,first)
    move=ref.coordinates@last.T-p;spin=np.tile((0.,0.,2.),(2,1))
    limited=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=1)
    assert not limited.certified and 'interval budget' in limited.reason
    result=law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    assert result.certified and 1<result.intervals<=64
    assert law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)==result
    assert result.path_sha256!=limited.path_sha256
    token=CancellationToken();token.cancel('test curved path rollback')
    with pytest.raises(SolveCancelled):
        law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64,cancellation_token=token)
    np.testing.assert_array_equal(start.positions,p)
    with pytest.raises(ValueError,match='rotation chart'):
        law.certify_path(start,move,np.tile((0.,0.,3.),(2,1)),gap_tolerance_m=5e-5,maximum_intervals=64)
    with pytest.raises(ValueError,match='gap tolerance'):
        law.certify_path(start,move,spin,gap_tolerance_m=0.,maximum_intervals=64)
    law.stiffness*=2
    with pytest.raises(ValueError,match='definition/settings changed'):
        law.certify_path(start,move,spin,gap_tolerance_m=5e-5,maximum_intervals=64)
    _record('curved-rotating-path',dict(control=asdict(result),limited=asdict(limited)))


def test_root_band_angular_union_keeps_sub_resolution_endpoint_arc():
    from decimal import Decimal
    from anyfesolver.general_contact.native_normal_angles import PreciseNormalAngles
    from anyfesolver.general_contact.native_curved_plane_work import _negative_arc_width_bound
    # An acos seed rounds this genuine endpoint root onto zero. Midpoint-only
    # arc classification used to return zero; unresolved ownership must keep it.
    tiny=PreciseNormalAngles(Decimal('-1e-30'),0.,0.,0.,1.)
    assert tiny.value(0)<0 and tiny.value(Decimal('5e-31'))<0
    assert _negative_arc_width_bound(tiny,0.,np.pi/2)==np.pi/2
    broad=PreciseNormalAngles(-.5,0.,0.,1.,0.)
    width=_negative_arc_width_bound(broad,0.,2*np.pi)
    assert width>=4*np.pi/3 and width==pytest.approx(4*np.pi/3,abs=2e-14)
    endpoint=PreciseNormalAngles(-.2,0.,0.,0.,1.)
    width=_negative_arc_width_bound(endpoint,0.,np.pi/2)
    assert width>=np.arcsin(.2) and width==pytest.approx(np.arcsin(.2),abs=2e-14)
    assert _negative_arc_width_bound(PreciseNormalAngles(1.,0.,0.,0.,0.),0.,1.)==0.
    assert _negative_arc_width_bound(PreciseNormalAngles(0.,0.,0.,0.,0.),0.,1.)==1.


def test_root_band_grazing_gauss_node_charges_full_fallback_measure():
    from decimal import Decimal,localcontext
    from numpy.polynomial.legendre import leggauss
    from anyfesolver.general_contact.native_normal_angles import PreciseNormalAngles,_sincos,PRECISION
    from anyfesolver.general_contact.native_curved_plane_work import (
        _active_angular_rule,_negative_arc_width_bound,_root_band_measure_bound)
    points,weights=leggauss(16);al,ar=.2,5.8
    angle=Decimal((al+ar)/2)+Decimal((ar-al)/2)*Decimal(float(points[8]))
    with localcontext() as ctx:
        ctx.prec=PRECISION
        s,c=_sincos(angle);amplitude=(s*s+c*c).sqrt()
        field=PreciseNormalAngles(amplitude-Decimal('1e-24'),0.,0.,-c,-s)
        lower=PreciseNormalAngles(field.offset-Decimal('1e-10'),0.,0.,-c,-s)
    assert field.value(angle)<0
    sampled=sum(w for _,w,_ in _active_angular_rule(field,al,ar,points,weights,root_band=True))
    true_union=_negative_arc_width_bound(lower,al,ar)
    assert sampled>true_union and true_union<1e-4
    measure,fallback=_root_band_measure_bound(field,lower,al,ar)
    assert fallback and measure>=max(sampled,true_union)
