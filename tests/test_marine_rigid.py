"""Prospective instantaneous oracles; no integration/qualification campaign."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import (RigidBody6D, RigidBodyState,
    MarineRigidInputError, rotation_matrix, left_jacobian, left_jacobian_rate)

def close(actual, expected, tol=1e-10):
    assert np.linalg.norm(np.asarray(actual)-expected) <= tol*max(np.linalg.norm(expected), 1e-12)

def body():
    return RigidBody6D('body-a', 7., (1., -.4, .8),
                       ((4., .3, .2), (.3, 5., -.1), (.2, -.1, 6.)))

def pose_path(state, dt):
    qd, qdd = np.asarray(state.velocity), np.asarray(state.acceleration)
    return replace(state, position=np.asarray(state.position)+dt*qd[:3]+.5*dt*dt*qdd[:3],
                   rotation_vector=np.asarray(state.rotation_vector)+dt*qd[3:]+.5*dt*dt*qdd[3:],
                   velocity=qd+dt*qdd)

def test_physical_inertia_and_point_oracles_32_states():
    rng = np.random.default_rng(20261009)
    b = body()
    for _ in range(32):
        anchor = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        state = b.state(position=rng.normal(size=3), rotation_vector=rng.normal(size=3)*.35,
                        velocity=rng.normal(size=6), acceleration=rng.normal(size=6), anchor=anchor)
        R = Rotation.from_rotvec(state.rotation_vector).as_matrix() @ anchor
        close(state.rotation, R)
        c = R @ np.asarray(b.cg_body)
        Ic = R @ np.asarray(b.inertia_cg_body) @ R.T
        v, w = state.world_velocity[:3], state.world_velocity[3:]
        a, alpha = state.world_acceleration[:3], state.world_acceleration[3:]
        acg = a + np.cross(alpha, c) + np.cross(w, np.cross(w, c))
        F = b.mass*acg
        torque = Ic@alpha + np.cross(w, Ic@w) + np.cross(c, F)
        close(b.inertial_force(state), state.rate_map.T @ np.r_[F, torque])
        vCG = v + np.cross(w, c)
        expected_ke = .5*b.mass*(vCG@vCG) + .5*w@Ic@w
        assert abs(b.kinetic_energy(state)-expected_ke) <= 1e-10*max(expected_ke, 1e-12)
        h = 2e-6
        # Instantaneous energy power, includes changing orientation and velocities.
        power_fd = (b.kinetic_energy(pose_path(state,h))-b.kinetic_energy(pose_path(state,-h)))/(2*h)
        power = np.asarray(state.velocity) @ b.inertial_force(state)
        assert abs(power-power_fd) <= 1e-8*max(abs(power_fd), 1e-12)
        point = rng.normal(size=3)
        kin = b.point_kinematics(state, point)
        close(kin.velocity, kin.jacobian @ np.asarray(state.velocity))
        close(kin.acceleration, kin.jacobian @ np.asarray(state.acceleration)+kin.acceleration_bias)
        # Independent pose exponential for point displacement derivatives.
        q = np.r_[state.position, state.rotation_vector]
        def point_at(q):
            return q[:3] + Rotation.from_rotvec(q[3:]).as_matrix() @ anchor @ point
        Jfd = np.column_stack([(point_at(q+h*np.eye(6)[i])-point_at(q-h*np.eye(6)[i]))/(2*h) for i in range(6)])
        close(kin.jacobian, Jfd, 1e-8)
        acceleration_fd = (b.point_kinematics(pose_path(state,h),point).velocity
                           - b.point_kinematics(pose_path(state,-h),point).velocity)/(2*h)
        close(kin.acceleration, acceleration_fd, 1e-8)
        force = rng.normal(size=3)
        Q = b.point_force(state, point, force)
        power_expected = force @ kin.velocity
        assert abs(np.asarray(state.velocity)@Q-power_expected) <= 1e-10*max(abs(power_expected),1e-12)
        # Actual dry acceleration derivative, not only symmetry of a helper.
        Mfd = np.column_stack([(b.inertial_force(replace(state,acceleration=np.asarray(state.acceleration)+h*np.eye(6)[i]))
                               - b.inertial_force(replace(state,acceleration=np.asarray(state.acceleration)-h*np.eye(6)[i])))/(2*h) for i in range(6)])
        close(b.acceleration_tangent(state), Mfd, 1e-8)

def test_reference_shift_preserves_physical_mass_and_energy():
    b = body()
    s = b.state(rotation_vector=(.4,-.2,.3),velocity=(.3,.7,-.2,.4,-.6,.2))
    db = np.array([.8, -.3, .5]); d = s.rotation @ db
    shifted = RigidBody6D('shifted', b.mass, np.asarray(b.cg_body)-db, b.inertia_cg_body)
    T = np.eye(6); T[:3,3:] = np.array([[0,-d[2],d[1]],[d[2],0,-d[0]],[-d[1],d[0],0]])
    close(shifted.mass_matrix(s), T.T@b.mass_matrix(s)@T)
    new_v = np.asarray(s.velocity).copy()
    new_v[:3] += np.cross(s.world_velocity[3:], d)
    close(np.array([shifted.kinetic_energy(replace(s,velocity=new_v))]), np.array([b.kinetic_energy(s)]))

@pytest.mark.parametrize('theta', [(0.,0.,0.),(1e-10,-2e-10,3e-10),(.019,0.,0.),(.021,0.,0.),(.4,-.7,.2)])
def test_stable_series_rate_map(theta):
    close(rotation_matrix(theta), Rotation.from_rotvec(theta).as_matrix())
    rate=np.array([.2,-.4,.7]); h=2e-6
    derivative=(left_jacobian(np.asarray(theta)+h*rate)-left_jacobian(np.asarray(theta)-h*rate))/(2*h)
    close(left_jacobian_rate(theta,rate), derivative, 1e-8)

def test_gravity_is_negative_potential_gradient_and_internal_sign():
    b=body(); s=b.state(rotation_vector=(.5,-.4,.3),anchor=Rotation.from_rotvec([.7,.1,-.3]).as_matrix())
    q=np.r_[s.position,s.rotation_vector]; h=2e-6
    def potential(q):
        R=Rotation.from_rotvec(q[3:]).as_matrix()@np.asarray(s.anchor)
        return b.mass*9.81*(q[:3]+R@np.asarray(b.cg_body))[2]
    gradient=np.array([(potential(q+h*np.eye(6)[i])-potential(q-h*np.eye(6)[i]))/(2*h) for i in range(6)])
    close(b.gravity_force(s),-gradient,1e-8)
    translation=b.state(acceleration=(2.,-1.,3.,0.,0.,0.))
    force=b.mass*np.array([2.,-1.,3.])
    close(b.inertial_force(translation),np.r_[force,np.cross(b.cg_body,force)])

@pytest.mark.parametrize('tensor', [np.diag([0.,1.,1.]),np.diag([-1.,2.,2.]),np.diag([1.,1.,3.]),
    [[1.,1.,0.],[0.,1.,0.],[0.,0.,1.]],np.full((3,3),float('nan')),[[1.,0.],[0.,1.]]])
def test_invalid_inertia_is_typed_refusal(tensor):
    with pytest.raises(MarineRigidInputError):
        RigidBody6D('bad',1.,(0.,0.,0.),tensor)

def test_refusal_and_input_immutability():
    cg=[1.,2.,3.]; I=[[4.,0.,0.],[0.,5.,0.],[0.,0.,6.]]
    b=RigidBody6D('immutable',2.,cg,I); cg[0]=99.; I[0][0]=99.
    assert b.cg_body[0]==1. and b.inertia_cg_body[0][0]==4.
    vel=[0.]*6; s=b.state(velocity=vel); vel[0]=99.; assert s.velocity[0]==0.
    for theta in ((math.pi,0.,0.),(2*math.pi,0.,0.)):
        with pytest.raises(MarineRigidInputError,match='seam'): b.state(rotation_vector=theta)
    with pytest.raises(MarineRigidInputError): b.state(anchor=np.diag([1.,1.,-1.]))
    with pytest.raises(MarineRigidInputError): b.point_kinematics(s,[float('inf'),0.,0.])
    with pytest.raises(MarineRigidInputError): b.mass_matrix(None)

@pytest.mark.parametrize('mass', [0.,-1.,float('nan'),float('inf'),True,'7'])
def test_mass_admission_requires_real_positive_finite(mass):
    with pytest.raises(MarineRigidInputError):
        RigidBody6D('invalid-mass',mass,(0.,0.,0.),np.eye(3))

def test_near_seam_anchor_and_nonvertical_gravity():
    b=body(); s=b.state(rotation_vector=(math.pi-1e-6,0.,0.))
    close(s.rotation,Rotation.from_rotvec(s.rotation_vector).as_matrix())
    force=b.mass*np.array([1.2,-.7,.4])
    expected=np.r_[force, np.cross(s.rotation@np.asarray(b.cg_body),force)]
    close(b.gravity_force(s,(1.2,-.7,.4)),s.rate_map.T@expected)


# --- W4 appends: exact dry chart tangents and second Jl derivatives ---

def test_dry_tangents_match_finite_differences_8_states():
    """FD verification of the production analytic K/C/Ma (1e-8, unit floors)."""
    rng = np.random.default_rng(20261010)
    b = body()
    h = 1e-5
    for _ in range(8):
        anchor = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        state = b.state(position=rng.normal(size=3), rotation_vector=rng.normal(size=3)*.35,
                        velocity=rng.normal(size=6), acceleration=rng.normal(size=6), anchor=anchor)
        tangents = b.dry_tangents(state)
        q = np.r_[np.asarray(state.position), np.asarray(state.rotation_vector)]
        qd = np.asarray(state.velocity)
        qdd = np.asarray(state.acceleration)

        def qdry(q6, vel, acc):
            return b.inertial_force(replace(state, position=q6[:3], rotation_vector=q6[3:],
                                            velocity=vel, acceleration=acc))

        identity = np.eye(6)

        def fd_columns(perturb):
            columns = []
            for j in range(6):
                plus = qdry(*perturb(+1., j))
                minus = qdry(*perturb(-1., j))
                columns.append((plus - minus)/(2*h))
            return np.column_stack(columns)

        def perturb_q(sign, j):
            delta = sign*h*identity[j]
            return (q + delta, qd, qdd)

        def perturb_qd(sign, j):
            delta = sign*h*identity[j]
            return (q, qd + delta, qdd)

        def perturb_qdd(sign, j):
            delta = sign*h*identity[j]
            return (q, qd, qdd + delta)

        for tangent, perturb in ((tangents.stiffness, perturb_q),
                                 (tangents.damping, perturb_qd),
                                 (tangents.mass, perturb_qdd)):
            fd = fd_columns(perturb)
            for rows in ([0, 1, 2], [3, 4, 5]):
                expected = fd[rows]
                assert np.linalg.norm(expected - tangent[rows]) <= 1e-8*max(
                    np.linalg.norm(expected), 1.)
        # Dry stiffness translation columns are exactly zero.
        assert np.linalg.norm(tangents.stiffness[:, :3]) == 0.
        # Dry damping/mass remain symmetric to roundoff.
        assert np.linalg.norm(tangents.mass - tangents.mass.T) <= 1e-12*max(
            np.linalg.norm(tangents.mass), 1.)


@pytest.mark.parametrize('theta', [(0., 0., 0.), (1e-9, 2e-9, -1e-9), (.019, 0., 0.),
                                   (.021, 0., 0.), (.49, 0., -.3), (.5, .1, .2), (1.2, -.8, .4)])
def test_left_jacobian_second_rate_matches_fd_and_symmetry(theta):
    from anyfesolver.marine_rigid import left_jacobian_second_rate
    a = np.array([.3, -.7, .5])
    e = np.array([-.2, .4, .6])
    # Small theta makes the closed-form (1-cos)/t^2 cancellation the FD noise
    # floor dominate at h=2e-6; widen h there (still inside the series branch).
    h = 1e-4 if np.linalg.norm(theta) < .05 else 2e-6
    fd = (left_jacobian_rate(np.asarray(theta)+h*a, e)
          - left_jacobian_rate(np.asarray(theta)-h*a, e))/(2*h)
    close(left_jacobian_second_rate(theta, a, e), fd, 1e-8)
    close(left_jacobian_second_rate(theta, a, e),
          left_jacobian_second_rate(theta, e, a), 1e-12)
