"""Independent spatial-update checks for the native swept-round work map."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
from anyfesolver.general_contact.friction_point import (
    initial_friction_point, objective_relative_tangent_increment, trial_coulomb_point,
)
from test_general_contact_native_retained_station import _reference


def test_native_section_frame_and_surface_chart_second_variations():
    reference=_reference(True)
    station=NativeRetainedRoundStation(7,reference,.37,.04,1.2)
    cells=Rotation.from_rotvec(((.2,-.1,.3),(-.3,.4,.2))).as_matrix()
    high=reference.coordinates.copy();low=np.zeros((3,3))
    value=station.surface_variations(high,low,cells)
    np.testing.assert_allclose(value.section_frame,cells[1]@reference.frame(.37),atol=2e-15)
    direction=np.array((.3,-.7,.4))
    expected_frame=np.einsum('ijk,k->ij',value.section_frame_gradient,direction)
    expected_frame_second=np.einsum('ijkl,k,l->ij',value.section_frame_hessian,direction,direction)
    expected_position_second=np.einsum('ijk,j,k->i',value.surface_rotation_hessian,direction,direction)
    for h in (3e-4,1e-4):
        def moved(sign):
            change=Rotation.from_rotvec(sign*h*direction).as_matrix()
            updated=cells.copy();updated[1]=change@cells[1]
            return station.evaluate(high,low,updated).surface_position,updated[1]@reference.frame(.37)
        xp,fp=moved(1);xm,fm=moved(-1)
        np.testing.assert_allclose((fp-fm)/(2*h),expected_frame,rtol=1e-7,atol=1e-9)
        np.testing.assert_allclose((fp-2*value.section_frame+fm)/h**2,
                                   expected_frame_second,rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose((xp-2*value.station.surface_position+xm)/h**2,
                                   expected_position_second,rtol=1e-5,atol=1e-8)
    assert value.angular_columns==(21,22,23)
    assert not value.section_frame_gradient.flags.writeable


def test_native_surface_spatial_force_tangent_includes_connection_term():
    reference=_reference(True)
    station=NativeRetainedRoundStation(5,reference,-.41,.07,.8)
    high=reference.coordinates.copy();low=np.zeros((3,3))
    cells=Rotation.from_rotvec(((.4,-.3,.2),(-.1,.2,.3))).as_matrix()
    value=station.surface_variations(high,low,cells)
    force=np.array((.7,-.4,.2))
    force_gradient=np.array(((2.,.3,-.1),(-.2,1.,.4),(.6,-.3,3.)))
    tangent=value.spatial_surface_force_tangent(force,force_gradient)
    direction=np.random.default_rng(173).normal(size=42)
    for h in (1e-5,2e-6):
        def moved(sign):
            x=high.copy();U=cells.copy()
            for n in range(3):x[n]+=sign*h*direction[6*n:6*n+3]
            for c in range(2):
                U[c]=Rotation.from_rotvec(sign*h*direction[18+3*c:21+3*c]).as_matrix()@U[c]
            updated=station.evaluate(x,low,U)
            f=force+force_gradient@(updated.surface_position-value.station.surface_position)
            return updated.generalized_surface_work(f)
        difference=(moved(1)-moved(-1))/(2*h)
        assert np.linalg.norm(tangent@direction-difference)/np.linalg.norm(difference)<=1e-5
    np.testing.assert_array_equal(tangent[24:,:],0.)
    np.testing.assert_array_equal(tangent[:,24:],0.)
    # The moving spatial covector has a nonsymmetric connection contribution.
    geometric=value.spatial_surface_force_tangent(force)
    assert np.linalg.norm(geometric-geometric.T)>1e-3


@pytest.mark.parametrize('angle,branch',[(.01,'stick'),(.2,'slip')])
def test_native_spin_at_fixed_centerline_creates_opposing_friction_torque(angle,branch):
    reference=_reference(False);radius=.03
    station=NativeRetainedRoundStation(3,reference,-.5,radius,1.5*np.pi)
    high=reference.coordinates;low=np.zeros((3,3));cells=np.tile(np.eye(3),(2,1,1))
    old=station.evaluate(high,low,cells)
    cells[0]=Rotation.from_rotvec((angle,0,0)).as_matrix()
    new=station.surface_variations(high,low,cells)
    np.testing.assert_array_equal(new.station.centerline_position,old.centerline_position)
    increment=objective_relative_tangent_increment(source_committed=old.surface_position,
        target_committed=old.surface_position,source_candidate=new.station.surface_position,
        target_candidate=old.surface_position,target_frame_committed=np.eye(3),target_frame_candidate=np.eye(3))
    history=initial_friction_point(np.eye(3))
    trial=trial_coulomb_point(history,material_frame=np.eye(3),relative_increment=increment,
        normal_pressure=5.,friction_coefficient=.3,tangential_stiffness=1000.,partner_key=station.identity)
    force=np.asarray(trial.traction);work=new.station.generalized_surface_work(force)
    expected=-radius*np.cos(angle)*min(1000*radius*np.sin(angle),1.5)
    assert trial.branch==branch
    assert work[18]==pytest.approx(expected,rel=1e-12,abs=1e-14)
    assert work[18]*angle<0
    assert trial.dissipated_increment_density>=0
    assert history==initial_friction_point(np.eye(3))
    np.testing.assert_array_equal(work[24:],0.)
    # Pressure is prescribed only for this local work oracle: no normal law or
    # actual surface contact admission is inferred from a material-point test.
    rigid=Rotation.from_rotvec((.4,-.2,.3)).as_matrix();shift=np.array((1.,-.4,.2))
    zero=objective_relative_tangent_increment(source_committed=old.surface_position,
        target_committed=old.surface_position,source_candidate=rigid@old.surface_position+shift,
        target_candidate=rigid@old.surface_position+shift,target_frame_committed=np.eye(3),
        target_frame_candidate=rigid)
    np.testing.assert_array_equal(zero,np.zeros(3))
