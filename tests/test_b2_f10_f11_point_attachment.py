"""Bounded F10/F11 mechanics probe; no public profile admission."""

import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import PhysicalB2Member
from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.fe_core import FEModel


VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])


def specimen():
    section = LinearSection.isotropic(E=2000., nu=.25, area=.04,
        Iy=.00015, Iz=.0003, J=.0004, ky=.8, kz=.7)
    member = PhysicalB2Member("F10-beam", "F10-geometry", (10, 11),
        ((.2, .5, .2), (2.2, .5, .2)), (0., 1., 0.), "F10-section", section)
    law = compile_reference_point(member, VERTICES, station=.4,
                                  offset_local_m=(0., .08, -.03))
    model = FEModel("F10-physical-Q4")
    model.add_material("plate", 1000., .3)
    for i, x in enumerate(VERTICES, 1):
        model.add_node(i, *x)
    shell = QualifiedE4PLShellElement(1, (1, 2, 3, 4), "plate",
        thickness=.15, reference_normal=np.array([0., 0., 1.]))
    model.add_element(1, shell)
    shell_k = shell.compute_stiffness_matrix(model.mesh, model.get_material("plate"))
    transform = member.local_dof_map()
    beam_k = transform.T @ member.local_operator().evaluate(np.zeros(12)).tangent @ transform
    return member, law, shell_k, beam_k


def test_reference_derivative_virtual_work_and_action_reaction():
    member, law, _, _ = specimen()
    assert not law.production_qualified
    assert law.beam_reference_m[2] != law.shell_reference_m[2]
    assert np.linalg.matrix_rank(law.constraint) == 6
    # An independent linear rigid-motion fit must reproduce every prescribed
    # shell-port twist, including the normal-axis moment channel.
    v = np.array([.02, -.03, .01, .04, -.02, .03])
    translations = np.array([v[:3]+np.cross(v[3:], x-law.shell_reference_m)
                             for x in VERTICES])
    u_shell = np.zeros((4, 6)); u_shell[:, :3] = translations
    np.testing.assert_allclose(law.shell_port @ u_shell.ravel(), v, atol=2e-16)

    direction = np.sin(np.arange(36)+.3) * .01
    multiplier = np.array([.17, -.23, .31, .07, -.05, .11])
    shell_force, beam_force = law.forces(multiplier)
    predicted = np.r_[shell_force, beam_force]
    eps = 1e-6
    def work(s):
        u = s*direction
        return law.multiplier_work(u[:24], u[24:], multiplier)
    assert abs((work(eps)-work(-eps))/(2*eps)-predicted@direction) < 1e-11
    # The constraint Jacobian is also checked against the nonlinear objective
    # joint, independently of this module's stored composed rows.
    q_s = law.shell_port @ direction[:24]
    q_b = law.beam_port @ direction[24:]
    trace_plus = member.trace_global(eps*direction[24:], stations=(.4,),
        offset_local_m=(0., .08, -.03))
    trace_minus = member.trace_global(-eps*direction[24:], stations=(.4,),
        offset_local_m=(0., .08, -.03))
    np.testing.assert_allclose((trace_plus.pose[0]-trace_minus.pose[0])/(2*eps),
                               q_b, rtol=2e-10, atol=2e-12)
    x0 = np.array((law.shell_reference_m, law.beam_reference_m))
    frames0 = np.array((np.eye(3), member.local_frame_rows().T))
    joint = RigidPoseJoint(x0, frames0)
    def nonlinear_gap(sign):
        d = sign*eps*np.array((q_s, q_b))
        x = x0+d[:, :3]
        frames = np.array([Rotation.from_rotvec(row[3:]).as_matrix()@frame
                           for row, frame in zip(d, frames0)])
        return joint.evaluate(x, frames, np.zeros(6)).constraints
    np.testing.assert_allclose((nonlinear_gap(1)-nonlinear_gap(-1))/(2*eps),
                               law.constraint@direction, rtol=2e-8, atol=2e-10)
    # Sum of nodal forces and global moments vanishes for an internal action.
    sf = shell_force.reshape(4, 6); bf = beam_force.reshape(2, 6)
    beam_nodes = np.asarray(member.reference_points_m)
    force_balance = sf[:, :3].sum(axis=0)+bf[:, :3].sum(axis=0)
    moment_balance = (sf[:, 3:]+np.cross(VERTICES, sf[:, :3])).sum(axis=0)
    moment_balance += (bf[:, 3:]+np.cross(beam_nodes, bf[:, :3])).sum(axis=0)
    np.testing.assert_allclose(force_balance, 0., atol=2e-13)
    np.testing.assert_allclose(moment_balance, 0., atol=2e-13)
    assert abs(multiplier @ (law.constraint @ direction)-predicted @ direction) < 1e-14


def test_physical_q4_b2_sparse_kkt_matches_independent_reduced_solution():
    _, law, shell_k, beam_k = specimen()
    stiffness = np.zeros((36, 36))
    stiffness[:24, :24] = shell_k
    stiffness[24:, 24:] = beam_k
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    # Rotational Q4 DOFs remain in the actual physical shell stiffness; this
    # local sample supports them to isolate the finite translation port law.
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    k = stiffness[np.ix_(free, free)]
    c = law.constraint[:, free]
    assert np.linalg.matrix_rank(c) == 6
    load = np.zeros(36)
    load[12] = .15; load[13] = -.09; load[32] = .11; load[33] = .035
    f = load[free]
    sparse = bmat([[csc_matrix(k), csc_matrix(c.T)],
                   [csc_matrix(c), None]], format="csc")
    result = spsolve(sparse, np.r_[f, np.zeros(6)])
    u = result[:len(f)]; lam = result[len(f):]
    # Independent null-space route solves only compatible primal directions.
    z = null_space(c)
    u_oracle = z @ np.linalg.solve(z.T@k@z, z.T@f)
    lam_oracle = np.linalg.lstsq(c.T, f-k@u_oracle, rcond=None)[0]
    np.testing.assert_allclose(u, u_oracle, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(lam, lam_oracle, rtol=2e-10, atol=2e-11)
    assert np.linalg.norm(c@u) < 1e-11
    assert np.linalg.norm(k@u+c.T@lam-f) < 1e-10
    full_u = np.zeros(36); full_u[free] = u
    reactions = stiffness@full_u+law.constraint.T@lam-load
    oracle_full = np.zeros(36); oracle_full[free] = u_oracle
    reactions_oracle = stiffness@oracle_full+law.constraint.T@lam_oracle-load
    np.testing.assert_allclose(reactions, reactions_oracle, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(reactions[free], 0., atol=1e-10)
    assert np.linalg.norm(reactions[list(fixed)]) > 0.
    assert abs(full_u@load-full_u@stiffness@full_u) < 1e-10


@pytest.mark.parametrize("change", ("collinear", "bad-station"))
def test_invalid_reference_support_rejected(change):
    member, _, _, _ = specimen()
    vertices = VERTICES.copy()
    station = .4
    if change == "collinear":
        vertices[:, 1] = 0.
    else:
        station = 1.2
    with pytest.raises(ValueError):
        compile_reference_point(member, vertices, station=station)
