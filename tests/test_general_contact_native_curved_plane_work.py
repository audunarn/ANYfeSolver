"""Independent curved material area, wrench and retained work controls."""
from math import pi
import json,os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

from anyfesolver.general_contact.native_curved_geometry import curved_patch_geometry_control
from anyfesolver.general_contact.native_curved_plane_work import NativeCurvedMaterialPlaneWork
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from test_general_contact_native_retained_station import _reference
from test_general_contact_native_surface_bounds import _independent_point


def _case(settings=None):
    ref=_reference(True)
    rule=NativeRoundSurfaceRule(7,ref,.03,xi_range=(-.9,.8),angle_range=(.2,5.8),
        axial_order=1,angular_order=1)
    settings=settings or StraightNormalIntegrationSettings(1e-10,1e-8,1e-6,64,1.)
    law=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=(0.,0.,0.),
        plane_normal=(0.,0.,1.),normal_stiffness_Pa_per_m=10000.,
        integration_settings=settings)
    p=ref.coordinates.copy();p[:,2]=-.06
    p[1]+=(.003,-.002,-.001)
    U=np.stack((Rotation.from_rotvec((.025,-.015,.01)).as_matrix(),
                Rotation.from_rotvec((-.02,.01,-.015)).as_matrix()))
    return law,p,np.zeros((3,3)),U


def _reference_integral(law,p,low,U,order=64,twist=False):
    # Analytic parabola, planar frame and circular material surface; no production
    # station, area, derivative or contact integration helper.
    nodes,weights=leggauss(order)
    lo,hi=law.rule.definition['xi_range'];al,ar=law.rule.definition['angle_range']
    r=law.rule.definition['radius'];k=law.stiffness
    energy=0.;force=np.zeros(3);moment=np.zeros(3)
    for cell,left,right in ((0,lo,0.),(1,0.,hi)):
        for x,w in zip(nodes,weights):
            xi=(left+right)/2+(right-left)*x/2
            J=np.sqrt(1+.16*xi*xi)
            # r0'=(1,-.4xi,0); frame turn rate=-.4/J².
            for a,v in zip(nodes,weights):
                angle=(al+ar)/2+(ar-al)*a/2
                point=_independent_point(xi,angle,r,p,low,U,twist)
                gap=float(law.normal@(point-law.origin))
                t=xi+1 if cell==0 else xi
                roll=(1-t)*(.1,.4,-.2)[cell]+t*(.1,.4,-.2)[cell+1] if twist else 0.
                area=r*(J+.4*r*np.cos(angle+roll)/(J*J))
                factor=w*v*(right-left)*(ar-al)/4*area
                pressure=k*max(0.,-gap)
                energy+=.5*k*min(gap,0.)**2*factor
                reaction=pressure*law.normal*factor
                force+=reaction;moment+=np.cross(point,reaction)
    return energy,force,moment


def _moved(p,low,U,d,h):
    positions=p.copy()
    for i in range(3):positions[i]+=h*d[6*i:6*i+3]
    cells=np.stack([Rotation.from_rotvec(h*d[18+3*i:21+3*i]).as_matrix()@U[i] for i in range(2)])
    return positions,low,cells


def test_curved_whole_patch_orientation_plane_enclosure_and_independent_area_work():
    law,p,low,U=_case()
    value=law.evaluate(p,low,U)
    independent=_reference_integral(law,p,low,U)
    assert value.geometry.orientation_lower>0
    assert value.geometry.gap_upper_m<0
    assert value.normal_storage_j==pytest.approx(independent[0],rel=1e-9,abs=1e-10)
    np.testing.assert_allclose(value.source_force,independent[1],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(value.source_moment,independent[2],rtol=1e-9,atol=1e-8)
    np.testing.assert_array_equal(value.source_force+value.target_force,0.)
    np.testing.assert_array_equal(value.source_moment+value.target_moment,0.)
    np.testing.assert_array_equal(value.residual[24:],0.)
    for i in (3,9,15):np.testing.assert_array_equal(value.residual[i:i+3],0.)
    for xi in np.r_[np.linspace(-.9,-1e-8,37),np.linspace(1e-8,.8,33)]:
        for angle in np.linspace(.2,5.8,13):
            point=_independent_point(xi,angle,.03,p,low,U,False)
            gap=float(law.normal@point)
            assert value.geometry.gap_lower_m<=gap<=value.geometry.gap_upper_m
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-work.json').write_text(json.dumps(dict(
            energy_error_j=abs(value.normal_storage_j-independent[0]),
            force_error_n=float(np.linalg.norm(value.source_force-independent[1])),
            moment_error_nm=float(np.linalg.norm(value.source_moment-independent[2])),
            orientation_lower=value.geometry.orientation_lower,
            gap_lower_m=value.geometry.gap_lower_m,gap_upper_m=value.geometry.gap_upper_m,
            intervals=value.intervals,evaluations=value.integrand_evaluations,
            estimated_errors=value.estimated_errors),indent=2))


def test_curved_retained_virtual_work_and_full_spatial_tangent():
    law,p,low,U=_case();value=law.evaluate(p,low,U)
    d=np.random.default_rng(14837).normal(size=42)*.003
    d[24:]=0.
    errors=[];works=[]
    for h in (1e-5,2e-6):
        plus=law.evaluate(*_moved(p,low,U,d,h));minus=law.evaluate(*_moved(p,low,U,d,-h))
        derivative=(plus.residual-minus.residual)/(2*h)
        error=np.linalg.norm(value.tangent@d-derivative)/np.linalg.norm(derivative)
        assert error<=1e-5
        work=(plus.normal_storage_j-minus.normal_storage_j)/(2*h)
        assert work==pytest.approx(value.residual@d,rel=1e-5,abs=1e-9)
        errors.append(float(error));works.append(float(abs(work-value.residual@d)))
    assert np.linalg.norm(value.residual[18:24])>0.
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-curved-tangent.json').write_text(json.dumps(dict(tangent_errors=errors,
        work_errors_j=works),indent=2))


def test_curved_open_state_and_partial_contact():
    law,p,low,U=_case()
    p[:,2]=.06
    value=law.evaluate(p,low,U)
    assert value.geometry.gap_lower_m>0 and value.normal_storage_j==0
    np.testing.assert_array_equal(value.residual,0.)
    np.testing.assert_array_equal(value.tangent,0.)
    p[:,2]=.012;U[:]=np.eye(3)
    value=law.evaluate(p,low,U)
    assert value.normal_storage_j>0
    independent=_partial_reference_integral(law,p,low,U)
    assert value.normal_storage_j==pytest.approx(independent[0],rel=1e-9,abs=1e-10)
    np.testing.assert_allclose(value.source_force,independent[1],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(value.source_moment,independent[2],rtol=1e-9,atol=1e-8)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-partial-reference.json').write_text(json.dumps(dict(
            energy_error_j=abs(value.normal_storage_j-independent[0]),
            force_error_n=float(np.linalg.norm(value.source_force-independent[1])),
            moment_error_nm=float(np.linalg.norm(value.source_moment-independent[2])),
            intervals=value.intervals,evaluations=value.integrand_evaluations,
            estimated_errors=value.estimated_errors),indent=2))


def _partial_reference_integral(law,p,low,U):
    # Independent partial fixture: z=h+r*sin(alpha), U=I. Its negative arc
    # starts at pi+asin(h/r). Integrate the analytic parabola area and surface;
    # no production station, root, integration or derivative helper.
    assert np.array_equal(U,np.tile(np.eye(3),(2,1,1)))
    radius=law.rule.definition['radius'];height=p[0,2]
    left=max(law.rule.definition['angle_range'][0],np.pi+np.arcsin(height/radius))
    right=min(law.rule.definition['angle_range'][1],2*np.pi-np.arcsin(height/radius))
    nodes,weights=leggauss(64);lo,hi=law.rule.definition['xi_range']
    energy=0.;force=np.zeros(3);moment=np.zeros(3)
    for cell,xl,xr in ((0,lo,0.),(1,0.,hi)):
        for qx,wx in zip(nodes,weights):
            xi=(xl+xr)/2+(xr-xl)*qx/2;J=np.sqrt(1+.16*xi*xi)
            for qa,wa in zip(nodes,weights):
                angle=(left+right)/2+(right-left)*qa/2
                gap=height+radius*np.sin(angle)
                area=radius*(J+.4*radius*np.cos(angle)/(J*J))
                factor=wx*wa*(xr-xl)*(right-left)/4*area
                point=_independent_point(xi,angle,radius,p,low,U,False)
                reaction=-law.stiffness*gap*law.normal*factor
                energy+=.5*law.stiffness*gap**2*factor
                force+=reaction;moment+=np.cross(point,reaction)
    return energy,force,moment


def test_curved_partial_contact_moving_front_virtual_work_and_tangent():
    law,p,low,U=_case();p[:,2]=.012;U[:]=np.eye(3)
    value=law.evaluate(p,low,U)
    direction=np.random.default_rng(407).normal(size=42)
    direction[24:]=0.;direction/=np.linalg.norm(direction)
    errors=[];works=[]
    for h in (1e-5,2e-6):
        plus=law.evaluate(*_moved(p,low,U,direction,h))
        minus=law.evaluate(*_moved(p,low,U,direction,-h))
        derivative=(plus.residual-minus.residual)/(2*h)
        expected=value.tangent@direction
        error=np.linalg.norm(derivative-expected)/np.linalg.norm(expected)
        work=abs((plus.normal_storage_j-minus.normal_storage_j)/(2*h)-value.residual@direction)
        errors.append(float(error));works.append(float(work))
        assert error<1e-5 and work<1e-6
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-partial.json').write_text(json.dumps(dict(
            tangent_errors=errors,work_errors_j=works,intervals=value.intervals,
            evaluations=value.integrand_evaluations,energy_j=value.normal_storage_j),indent=2))


def test_curved_unobserved_thin_axial_contact_never_returns_zero():
    ref=_reference(True);radius=1e-7
    rule=NativeRoundSurfaceRule(7,ref,radius,xi_range=(-.95,-.05),
        angle_range=(np.pi/2-.001,np.pi/2+.001),axial_order=1,angular_order=1)
    p=ref.coordinates.copy();p[:,1]+=-.172*p[:,0]
    origin=np.array((0.,.2+.172**2/.8-1e-9,0.));normal=np.array((0.,-1.,0.))
    low=np.zeros((3,3));U=np.tile(np.eye(3),(2,1,1))
    law=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=origin,plane_normal=normal,
        normal_stiffness_Pa_per_m=1e7,integration_settings=StraightNormalIntegrationSettings(
            1e-7,1e-6,1e-4,64,1.))
    hidden=_independent_point(-.43,np.pi/2,radius,p,low,U,False)
    assert normal@(hidden-origin)<-9e-10
    # Both initial Gauss rules miss the independently located active pocket.
    for order in (8,16):
        for node in leggauss(order)[0]:
            point=_independent_point(-.5+.45*node,np.pi/2,radius,p,low,U,False)
            assert normal@(point-origin)>0
    try:
        value=law.evaluate(p,low,U)
    except (MemoryError,ValueError) as failure:
        assert 'interval budget' in str(failure) or 'unresolved' in str(failure)
        path=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if path:
            Path(path+'-curved-thin-axial.json').write_text(json.dumps(dict(
                result='explicit_non_success',reason=str(failure),
                diagnostics=getattr(failure,'integration_diagnostics',None)),indent=2))
    else:
        assert value.normal_storage_j>0 and np.linalg.norm(value.residual)>0


def test_curved_broad_contact_cannot_hide_second_thin_axial_pocket():
    ref=_reference(True);radius=1e-7
    rule=NativeRoundSurfaceRule(7,ref,radius,xi_range=(-.95,-.05),
        angle_range=(np.pi/2-.001,np.pi/2+.001),axial_order=1,angular_order=1)
    first_root=-.95+1e-7;second_root=-.5
    vertex=(first_root+second_root)/2;half=(second_root-first_root)/2
    p=ref.coordinates.copy();p[:,1]+=.4*vertex*p[:,0]
    origin=np.array((0.,.2+.2*vertex**2-.2*half**2,0.));normal=np.array((0.,1.,0.))
    low=np.zeros((3,3));U=np.tile(np.eye(3),(2,1,1))
    law=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=origin,plane_normal=normal,
        normal_stiffness_Pa_per_m=1e7,integration_settings=StraightNormalIntegrationSettings(
            1e-7,1e-6,1e-4,64,1.))
    hidden=_independent_point(-.95+2e-8,np.pi/2,radius,p,low,U,False)
    gap=float(normal@(hidden-origin))
    expected=-.2*(-.95+2e-8-first_root)*(-.95+2e-8-second_root)
    assert gap==pytest.approx(expected,abs=1e-15) and gap<-6e-9
    for order in (8,16):
        stations=-.5+.45*leggauss(order)[0]
        gaps=[normal@(_independent_point(x,np.pi/2,radius,p,low,U,False)-origin) for x in stations]
        assert min(stations)>first_root
        assert min(gaps)<0<max(gaps)  # Broad right pocket is sampled; thin left one is not.
    with pytest.raises((MemoryError,ValueError),match='interval budget|unresolved|root precision exhausted') as error:
        law.evaluate(p,low,U)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-two-pockets.json').write_text(json.dumps(dict(
            result='explicit_non_success',reason=str(error.value),
            diagnostics=getattr(error.value,'integration_diagnostics',None)),indent=2))


def test_curved_uniform_active_minimum_does_not_hide_angular_endpoint_crossing():
    from anyfesolver.general_contact.native_curved_plane_work import _rectangle_support
    ref=_reference(False);radius=.03;epsilon=1e-8
    rule=NativeRoundSurfaceRule(7,ref,radius,xi_range=(1e-8,.9),
        angle_range=(0.,np.pi),axial_order=1,angular_order=1)
    p=ref.coordinates.copy();p[:,1]=-.03-epsilon+.02*p[:,0]
    low=np.zeros((3,3));U=np.tile(np.eye(3),(2,1,1));normal=np.array((0.,1.,0.))
    # Literal circular field: -.03-epsilon+.02*xi+.03*cos(alpha).
    # Angular minimum is negative throughout, but endpoint alpha=0 crosses
    # at xi=5e-7, before either initial Gauss rule's first station.
    assert -.06-epsilon+.02*.9<0
    assert -epsilon+.02*1e-8<0<-epsilon+.02*.9
    for order in (8,16):
        stations=(.9+1e-8)/2+(.9-1e-8)*leggauss(order)[0]/2
        assert min(stations)>epsilon/.02
    assert _rectangle_support(rule,p,low,U,1,1e-8,.9,0.,np.pi,np.zeros(3),normal)=='unresolved'


def test_curved_partial_scalar_field_large_tangential_origin_invariance():
    from anyfesolver.general_contact.native_curved_plane_work import _station_field
    normal=np.array((1.,-1.,0.))/np.sqrt(2.)
    high=np.tile((1.011,1.,0.),(3,1));low=np.zeros((3,3))
    first=_station_field(high,low,0,.5,np.zeros(3),np.eye(3),np.eye(3),.03,
                         np.zeros(3),normal)
    second=_station_field(high,low,0,.5,np.zeros(3),np.eye(3),np.eye(3),.03,
                          np.array((2.**48,2.**48,0.)),normal)
    assert first.breakpoints(0.,2*np.pi)==second.breakpoints(0.,2*np.pi)
    for angle in np.linspace(0.,2*np.pi,21):
        assert first.value(angle)==second.value(angle)


def _independent_axial_front_integral(law,p,low,U,order=48):
    # Independent route: at each angle integrate xi only up to its exact affine
    # plane-gap root. Angular partitions follow analytic sine equations at the
    # cell boundaries. No production root, station, measure or work helper.
    assert np.array_equal(U,np.tile(np.eye(3),(2,1,1)))
    radius=law.rule.definition['radius'];lo,hi=law.rule.definition['xi_range']
    al,ar=law.rule.definition['angle_range'];height=p[1,2];slope=p[2,2]-p[1,2]
    points,weights=leggauss(order);energy=0.;force=np.zeros(3);moment=np.zeros(3)
    for cell,xl,xr in ((0,max(lo,-1.),min(hi,0.)),(1,max(lo,0.),min(hi,1.))):
        if xr<=xl:continue
        angles=[al,ar]
        for xi in (xl,xr):
            ratio=-(height+slope*xi)/radius
            if abs(ratio)<=1:
                root=np.arcsin(ratio)
                for base in (root,np.pi-root):
                    for turn in (-1,0,1):
                        a=base+2*np.pi*turn
                        if al<a<ar:angles.append(a)
        angles=sorted(angles)
        for left,right in zip(angles[:-1],angles[1:]):
            for qa,wa in zip(points,weights):
                angle=(left+right)/2+(right-left)*qa/2
                stop=min(xr,(-height-radius*np.sin(angle))/slope)
                if stop<=xl:continue
                for qx,wx in zip(points,weights):
                    xi=(xl+stop)/2+(stop-xl)*qx/2;J=np.sqrt(1+.16*xi*xi)
                    gap=height+slope*xi+radius*np.sin(angle)
                    area=radius*(J+.4*radius*np.cos(angle)/(J*J))
                    factor=wx*wa*(stop-xl)*(right-left)/4*area
                    point=_independent_point(xi,angle,radius,p,low,U,False)
                    reaction=-law.stiffness*gap*law.normal*factor
                    energy+=.5*law.stiffness*gap**2*factor
                    force+=reaction;moment+=np.cross(point,reaction)
    return energy,force,moment


def test_curved_axial_front_reference_area_complete_work():
    original,p,low,U=_case();p=original.reference.coordinates.copy()
    p[:,2]=.02+.08*p[:,0];U[:]=np.eye(3)
    law=NativeCurvedMaterialPlaneWork(original.rule,plane_origin_m=original.origin,
        plane_normal=original.normal,normal_stiffness_Pa_per_m=1e4,
        integration_settings=original.settings)
    try:value=law.evaluate(p,low,U)
    except Exception as failure:
        path=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if path:
            Path(path+'-curved-axial-front-failure.json').write_text(json.dumps(dict(
                reason=str(failure),diagnostics=getattr(failure,'integration_diagnostics',None)),indent=2))
        raise
    reference=_independent_axial_front_integral(law,p,low,U)
    assert value.axial_root_brackets and value.axial_isolation_intervals>0
    assert value.normal_storage_j==pytest.approx(reference[0],rel=1e-9,abs=1e-10)
    np.testing.assert_allclose(value.source_force,reference[1],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(value.source_moment,reference[2],rtol=1e-9,atol=1e-8)
    direction=np.random.default_rng(409).normal(size=42)*.002;direction[24:]=0.
    errors=[]
    for h in (1e-5,2e-6):
        plus=law.evaluate(*_moved(p,low,U,direction,h))
        minus=law.evaluate(*_moved(p,low,U,direction,-h))
        derivative=(plus.residual-minus.residual)/(2*h)
        error=np.linalg.norm(derivative-value.tangent@direction)/np.linalg.norm(derivative)
        errors.append(float(error));assert error<=1e-5
        work=(plus.normal_storage_j-minus.normal_storage_j)/(2*h)
        assert work==pytest.approx(value.residual@direction,rel=1e-5,abs=1e-9)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-axial-front.json').write_text(json.dumps(dict(
            energy_error_j=abs(value.normal_storage_j-reference[0]),
            force_error_n=float(np.linalg.norm(value.source_force-reference[1])),
            moment_error_nm=float(np.linalg.norm(value.source_moment-reference[2])),
            tangent_errors=errors,intervals=value.intervals,
            evaluations=value.integrand_evaluations,errors=value.estimated_errors,
            root_brackets=value.axial_root_brackets),indent=2))


def test_curved_reference_second_bound_against_independent_planar_twist_derivatives():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver.general_contact.native_curved_geometry import (
        _reference_second_bound,_reference_unclipped_interval)
    Kz=np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,0.)))
    Kx=np.array(((0.,0.,0.),(0.,0.,-1.),(0.,1.,0.)))
    base=_reference(True)
    for angles in ((0.,0.,0.),(.1,.4,-.2)):
        frames=base.nodal_triads.copy()
        for i,angle in enumerate(angles):frames[i]=frames[i]@Rotation.from_rotvec((angle,0.,0.)).as_matrix()
        ref=CenteredCurvedBeam3ReferenceGeometry(base.coordinates,frames)
        for cell,(left,right) in enumerate(((-1.,0.),(0.,1.))):
            bound=_reference_second_bound(ref,cell);roll=angles[cell+1]-angles[cell]
            assert _reference_unclipped_interval(ref,cell,left+.05,right-.05)
            assert not _reference_unclipped_interval(ref,cell,left,right)
            for xi in np.linspace(left+.001,right-.001,27):
                J=np.sqrt(1+.16*xi*xi);beta1=-.4/(J*J);beta2=.128*xi/(J**4)
                Rz=np.array(((1/J,.4*xi/J,0.),(-.4*xi/J,1/J,0.),(0.,0.,1.)))
                t=xi+1 if cell==0 else xi
                angle=angles[cell]+roll*t;c=np.cos(angle);s=np.sin(angle)
                Rx=np.array(((1.,0.,0.),(0.,c,-s),(0.,s,c)))
                derivative=Rz@((beta2*Kz+beta1**2*Kz@Kz)@Rx+
                    2*beta1*roll*Kz@Rx@Kx+roll**2*Rx@Kx@Kx)
                assert np.linalg.norm(derivative,2)<=bound


def test_curved_exact_binary_curvature_literal_cancellation_regression():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver.general_contact.native_curved_geometry import (
        _reference_curvature_coefficients,_reference_curvature_norm,_reference_second_bound)
    from anyfesolver.general_contact.native_curved_plane_work import (
        _projected_center_derivative,_projected_center_second)
    x=np.array((9007199254740991.,9007199254740994.,9007199254740996.))
    assert x[0]-2*x[1]+x[2]==0.  # Literal defect in the discarded arithmetic.
    coordinates=np.column_stack((x,(-1.,0.,1.),np.zeros(3)))
    frames=[]
    for xi in (-1.,0.,1.):
        tangent=np.array((2.5-xi,1.,0.));tangent/=np.linalg.norm(tangent)
        frames.append(np.column_stack((tangent,(-tangent[1],tangent[0],0.),(0.,0.,1.))))
    ref=CenteredCurvedBeam3ReferenceGeometry(coordinates,np.array(frames))
    assert _reference_curvature_coefficients(ref)==(-1,0,0)
    assert _reference_curvature_norm(ref)>=1.
    U=np.tile(np.eye(3),(2,1,1));normal=np.array((1.,0.,0.))
    for cell,xi in ((0,-.25),(1,.25)):
        assert _projected_center_second(ref,U,cell,normal)==-1.
        assert _projected_center_derivative(ref,coordinates,np.zeros((3,3)),U,cell,xi,normal)==2.5-xi
        v=2.5-xi;J2=v*v+1.;beta1=1/J2;beta2=2*v/J2**2
        assert np.hypot(beta2,beta1**2)<=_reference_second_bound(ref,cell)


def test_projected_gap_literal_origin_counterexample_and_resolution_guard():
    from anyfesolver.general_contact.native_curved_plane_work import _projected_gap
    n=np.array((1.,-1.,0.))/np.sqrt(2.)
    p=np.tile(np.array((1.011,.961,0.)),(3,1))
    arm=-.03*n
    first=_projected_gap(p,np.zeros_like(p),0,.5,arm,np.zeros(3),n)
    second=_projected_gap(p,np.zeros_like(p),0,.5,arm,np.array((2.**48,2.**48,0.)),n)
    assert first==second and first>0
    assert first==pytest.approx(.05/np.sqrt(2.)-.03,abs=2e-16)
    p[:]=0;p[0,2]=np.nextafter(0.,1.)
    with pytest.raises(ValueError,match='scalar projection unresolved'):
        _projected_gap(p,np.zeros_like(p),0,.75,np.zeros(3),np.zeros(3),np.array((0.,0.,1.)))


def test_curved_reference_measure_underflow_fails_explicitly():
    law,p,low,U=_case()
    tiny=NativeCurvedMaterialPlaneWork(law.rule,plane_origin_m=law.origin,
        plane_normal=law.normal,normal_stiffness_Pa_per_m=np.nextafter(0.,1.),
        integration_settings=law.settings)
    with pytest.raises(ValueError,match='reference measure/stiffness unresolved'):
        tiny.evaluate(p,low,U)


def test_whole_orientation_rejects_between_sample_inversion():
    from anyfesolver.general_contact.native_surface_quadrature import surface_geometry
    from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
    ref=_reference(True)
    rule=NativeRoundSurfaceRule(7,ref,1e-4,xi_range=(-.95,-.05),angle_range=(.2,5.8),
        axial_order=1,angular_order=1)
    p=ref.coordinates.copy();p[0]=(0.,0.,0.);p[1]=(.005,.01,0.)
    low=np.zeros((3,3));U=np.tile(np.eye(3),(2,1,1))
    # Sole quadrature sample stays outward. A different material station reverses.
    assert rule.evaluate(p,low,U)[0].area_jacobian>0
    hidden=NativeRetainedRoundStation(7,ref,-.25,1e-4,3.)
    with pytest.raises(ValueError,match='singular or reversed'):
        surface_geometry(hidden,p,low,U)
    with pytest.raises(ValueError,match='whole-patch orientation unresolved'):
        curved_patch_geometry_control(rule,p,low,U,np.zeros(3),np.array((0.,0.,1.)))


def test_curved_work_common_rigid_transform_and_numerical_limits():
    law,p,low,U=_case();value=law.evaluate(p,low,U)
    G=Rotation.from_rotvec((.3,-.4,.2)).as_matrix();offset=np.array((1.2,-.7,.3))
    moved=NativeCurvedMaterialPlaneWork(law.rule,plane_origin_m=offset,
        plane_normal=G@law.normal,normal_stiffness_Pa_per_m=law.stiffness,
        integration_settings=law.settings)
    transformed=moved.evaluate(p@G.T+offset,low,G@U)
    assert transformed.normal_storage_j==pytest.approx(value.normal_storage_j,rel=1e-12)
    np.testing.assert_allclose(transformed.source_force,G@value.source_force,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(transformed.source_moment,G@value.source_moment+
        np.cross(offset,G@value.source_force),rtol=1e-11,atol=1e-8)
    too_small=StraightNormalIntegrationSettings(1e-30,1e-30,1e-30,1,1.)
    limited,_,_,_=_case(too_small)
    with pytest.raises(MemoryError,match='interval budget'):
        limited.evaluate(p,low,U)
    law.stiffness*=2
    with pytest.raises(ValueError,match='definition/settings changed'):law.evaluate(p,low,U)


def test_curved_reference_derivative_interval_bound_with_twist():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver.general_contact.native_curved_geometry import _reference_norm_bounds
    base=_reference(True);frames=base.nodal_triads.copy()
    for i,roll in enumerate((.1,.4,-.2)):
        frames[i]=frames[i]@Rotation.from_rotvec((roll,0.,0.)).as_matrix()
    ref=CenteredCurvedBeam3ReferenceGeometry(base.coordinates,frames)
    for cell,(left,right) in enumerate(((-1.,0.),(0.,1.))):
        lower,upper,prime,_=_reference_norm_bounds(ref,cell)
        tight_prime=_reference_norm_bounds(ref,cell,actual_roll=True)[2]
        assert tight_prime<=prime
        for xi in np.linspace(left+1e-8,right-1e-8,199):
            derivative=ref.derivative(float(xi))
            assert lower<=np.linalg.norm(derivative)<=upper
            assert np.linalg.norm(ref.frame_derivative(float(xi)),2)<=tight_prime


def test_twisted_curved_material_reference_measure_and_work():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    law,p,low,U=_case()
    frames=law.reference.nodal_triads.copy()
    for i,roll in enumerate((.1,.4,-.2)):
        frames[i]=frames[i]@Rotation.from_rotvec((roll,0.,0.)).as_matrix()
    ref=CenteredCurvedBeam3ReferenceGeometry(law.reference.coordinates,frames)
    rule=NativeRoundSurfaceRule(7,ref,.03,xi_range=(-.9,.8),angle_range=(.2,5.8),
        axial_order=1,angular_order=1)
    twisted=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=law.origin,
        plane_normal=law.normal,normal_stiffness_Pa_per_m=law.stiffness,
        integration_settings=law.settings)
    value=twisted.evaluate(p,low,U)
    reference=_reference_integral(twisted,p,low,U,order=64,twist=True)
    assert value.normal_storage_j==pytest.approx(reference[0],rel=1e-9,abs=1e-10)
    np.testing.assert_allclose(value.source_force,reference[1],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(value.source_moment,reference[2],rtol=1e-9,atol=1e-8)


@pytest.mark.parametrize('open_state',(False,True))
def test_large_tangential_plane_origin_does_not_change_contact_work(open_state):
    # Same plane: n dot origin=0 exactly for both declared binary coordinates.
    law,p,low,U=_case()
    n=np.array((1.,-1.,0.))/np.sqrt(2.)
    plain=NativeCurvedMaterialPlaneWork(law.rule,plane_origin_m=(0.,0.,0.),
        plane_normal=n,normal_stiffness_Pa_per_m=law.stiffness,
        integration_settings=law.settings)
    distant=NativeCurvedMaterialPlaneWork(law.rule,plane_origin_m=(2.**48,2.**48,0.),
        plane_normal=n,normal_stiffness_Pa_per_m=law.stiffness,
        integration_settings=law.settings)
    # The complete source is placed on one side by its normal translation.
    p+= (2. if open_state else -2.)*n
    first=plain.evaluate(p,low,U);second=distant.evaluate(p,low,U)
    assert first.geometry.gap_lower_m==second.geometry.gap_lower_m
    assert first.geometry.gap_upper_m==second.geometry.gap_upper_m
    assert second.normal_storage_j==first.normal_storage_j
    np.testing.assert_array_equal(second.residual,first.residual)
    np.testing.assert_array_equal(second.tangent,first.tangent)


def test_large_tangential_origin_cannot_create_contact_on_open_curved_patch():
    ref=_reference(True);n=np.array((1.,-1.,0.))/np.sqrt(2.)
    rule=NativeRoundSurfaceRule(7,ref,.03,xi_range=(.9899,.9901),angle_range=(.2,5.8),
        axial_order=1,angular_order=1)
    G=Rotation.from_rotvec((0.,0.,np.pi/4+np.arctan(.4*.99))).as_matrix()
    chord_point=G@np.array((.99,.002,0.))
    offset=np.array((1.011,.961,0.))-chord_point
    p=ref.coordinates@G.T+offset;low=np.zeros((3,3));U=np.tile(G,(2,1,1))
    settings=StraightNormalIntegrationSettings(1e-10,1e-8,1e-6,64,1.)
    distant=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=(2.**48,2.**48,0.),
        plane_normal=n,normal_stiffness_Pa_per_m=10000.,integration_settings=settings)
    # Continuous plane slab proves positive gap; no force is allowed.
    geometry=curved_patch_geometry_control(rule,p,low,U,distant.origin,n)
    assert geometry.gap_lower_m>.003
    value=distant.evaluate(p,low,U)
    assert value.normal_storage_j==0.
    np.testing.assert_array_equal(value.residual,0.)
