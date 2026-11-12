"""Independent area, parameter derivative and integrated pressure references."""
from math import pi
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule, surface_geometry
from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
from test_general_contact_native_retained_station import _reference
from test_general_contact_native_surface_s3_friction import _case


@pytest.mark.parametrize('curved',(False,True))
def test_reference_surface_area_and_cell_partition(curved):
    reference=_reference(curved); radius=.03
    rule=NativeRoundSurfaceRule(7,reference,radius,axial_order=6,angular_order=16)
    # Independent analytic parabola x=xi,y=.2(1-xi^2), z=0.
    length=quad(lambda x:np.sqrt(1+(.4*x)**2) if curved else 1.,-1,1,epsabs=1e-13)[0]
    assert sum(s.reference_weight_m2 for s in rule.samples)==pytest.approx(2*pi*radius*length,rel=1e-10)
    assert all(s.station.xi!=0 and s.reference_weight_m2>0 for s in rule.samples)
    assert {s.station.cell for s in rule.samples}=={0,1}
    assert NativeRoundSurfaceRule(7,reference,radius,axial_order=6,angular_order=16).identity==rule.identity
    assert NativeRoundSurfaceRule(7,reference,radius,axial_order=5,angular_order=16).identity!=rule.identity


def test_deformed_curved_surface_parameter_tangents_and_nonradial_normal():
    reference=_reference(True)
    station=NativeRetainedRoundStation(7,reference,-.37,.04,1.2)
    positions=reference.coordinates.copy(); positions[1]+=(.03,-.02,.05)
    low=np.zeros((3,3)); rotations=np.tile(np.eye(3),(2,1,1))
    rotations[0]=Rotation.from_rotvec((.12,-.08,.05)).as_matrix()
    value=surface_geometry(station,positions,low,rotations)
    def point(x,a):
        return NativeRetainedRoundStation(7,reference,float(x),.04,float(a)).evaluate(positions,low,rotations).surface_position
    step=2e-6
    dx=(point(station.xi+step,station.angle)-point(station.xi-step,station.angle))/(2*step)
    da=(point(station.xi,station.angle+step)-point(station.xi,station.angle-step))/(2*step)
    np.testing.assert_allclose(value.xi_tangent,dx,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(value.angle_tangent,da,rtol=1e-8,atol=1e-10)
    radial=rotations[0]@reference.frame(station.xi)@np.array((0.,np.cos(station.angle),np.sin(station.angle)))
    assert np.linalg.norm(value.outward_normal-radial)>1e-3


def test_prepared_reference_area_matches_full_geometry_and_keeps_inversion_guard():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver.general_contact.native_surface_quadrature import _prepare_reference_area
    base=_reference(True);low=np.zeros((3,3));U=np.tile(np.eye(3),(2,1,1))
    rigid=Rotation.from_rotvec((.3,-.2,.1)).as_matrix()
    for twist in (False,True):
        frames=base.nodal_triads.copy()
        if twist:
            for i,roll in enumerate((.1,.4,-.2)):
                frames[i]=frames[i]@Rotation.from_rotvec((roll,0.,0.)).as_matrix()
        for transform in (np.eye(3),rigid):
            reference=CenteredCurvedBeam3ReferenceGeometry(
                base.coordinates@transform.T+np.array((2.,-3.,.5)),transform@frames)
            for xi in (-.95,-.4,.25,.85):
                prepared=_prepare_reference_area(reference,xi,.03)
                for angle in (.1,1.3,2.9,4.7,6.1):
                    station=NativeRetainedRoundStation(7,reference,xi,.03,angle)
                    expected=surface_geometry(station,reference.coordinates,low,U).area_jacobian
                    assert prepared.area_jacobian(angle)==pytest.approx(expected,rel=1e-13,abs=0.)
    with pytest.raises(ValueError,match='reversed'):
        _prepare_reference_area(base,-.5,4.).area_jacobian(pi)
    with pytest.raises(ValueError,match='reversed'):
        surface_geometry(NativeRetainedRoundStation(7,base,-.5,4.,pi),base.coordinates,low,U)


def test_surface_weighted_existing_contact_matches_cylinder_pressure_integral():
    _,pair,pose,history=_case()
    # Existing pose has center z=-.02, plane skin z=-.04: h=.02.
    lo,hi=4.1,5.3
    # The existing triangle owns x+y<0; select an interior physical patch.
    rule=NativeRoundSurfaceRule(7,pair.source.reference,.03,xi_range=(-.9,-.1),
        angle_range=(lo,hi),axial_order=2,angular_order=8)
    total=np.zeros(60)
    for sample in rule.samples:
        local=NativeS3MaterialPointFriction(sample.station,pair.target,
            reference_weight_m2=sample.reference_weight_m2,normal_stiffness_Pa_per_m=1000.,
            friction_coefficient=0.,tangential_stiffness_Pa_per_m=1000.)
        response=local.evaluate(pose,pose,history)
        assert response.normal_gap_m<0
        total+=response.residual
    force=1000*.8*.03*(-.02*(hi-lo)+.03*(np.cos(hi)-np.cos(lo)))
    assert -sum(total[[2,8,14]])==pytest.approx(force,rel=1e-11)
    assert sum(total[[44,50,56]])==pytest.approx(force,rel=1e-11)


def test_quadrature_caps_identity_and_surface_inversion():
    reference=_reference(False)
    with pytest.raises(MemoryError,match='point cap'):
        NativeRoundSurfaceRule(7,reference,.03,axial_order=20,angular_order=20)
    rule=NativeRoundSurfaceRule(7,reference,.03,axial_order=1,angular_order=2)
    rotations=np.tile(np.eye(3),(2,1,1)); low=np.zeros((3,3))
    positions=reference.coordinates.copy(); positions[:,0]*=-1
    with pytest.raises(ValueError,match='reversed'):
        rule.evaluate(positions,low,rotations)
    rule.samples[0].station.radius=.04
    with pytest.raises(ValueError,match='definition changed'):
        rule.evaluate(reference.coordinates,low,rotations)


def test_orientation_certificate_rejects_hidden_rotation_inversion():
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    reference=_reference(False)
    rule=NativeRoundSurfaceRule(7,reference,.03,axial_order=1,angular_order=1)
    axis=np.array((np.sqrt(.2),np.sqrt(.8),0.))
    cells=np.tile(Rotation.from_rotvec((np.pi-1.4)*axis).as_matrix(),(2,1,1))
    start=NativeS3MaterialConfiguration(reference.coordinates,np.zeros((3,3)),cells,np.zeros(18))
    angular=np.tile(2.8*axis,(2,1)); translation=np.zeros((3,3)); shell=np.zeros(18)
    end=material_path_pose(start,translation,angular,shell,1.)
    rule.evaluate(start.positions,start.position_low,start.cell_rotations)
    rule.evaluate(end.positions,end.position_low,end.cell_rotations)
    with pytest.raises(ValueError,match='source surface orientation'):
        rule.certify_orientation_path(start,translation,angular,shell)
