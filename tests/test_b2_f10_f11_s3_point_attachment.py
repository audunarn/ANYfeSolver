"""Bounded physical S3 support for the private B2 reference-point law."""

from itertools import permutations

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
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.fe_core import FEModel


TRIANGLE = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))


def _specimen():
    section = LinearSection.isotropic(E=2000., nu=.25, area=.04,
        Iy=.00015, Iz=.0003, J=.0004, ky=.8, kz=.7)
    member = PhysicalB2Member("F10-S3-beam", "F10-S3-geometry", (10, 11),
        ((.2, .5, .2), (2.2, .5, .2)), (0., 1., 0.), "F10-S3-section", section)
    vertices = TRIANGLE.copy()
    law = compile_reference_point(member, vertices, station=.4,
                                  offset_local_m=(0., .08, -.03))
    model = FEModel("F10-physical-S3")
    model.add_material("plate", 1000., .3)
    for i, x in enumerate(TRIANGLE, 1):
        model.add_node(i, *x)
    shell = QualifiedE4PLS3ShellElement(
        1, (1, 2, 3), "plate", thickness=.15,
        reference_normal=np.array((0., 0., 1.)))
    model.add_element(1, shell)
    shell_k = shell.compute_stiffness_matrix(model.mesh,
                                             model.get_material("plate"))
    transform = member.local_dof_map()
    beam_k = transform.T @ member.local_operator().evaluate(
        np.zeros(12)).tangent @ transform
    return member, law, vertices, shell_k, beam_k


def test_s3_reference_pose_work_wrench_and_owned_arrays():
    member, law, vertices, _, _ = _specimen()
    assert law.shell_port.shape == (6, 18)
    assert law.constraint.shape == (6, 30)
    assert law.law_id == "B2_FINITE_S3_POINT_REFERENCE_LINEAR_V1_DEV"
    assert np.linalg.matrix_rank(law.constraint) == 6
    assert not law.production_qualified
    v = np.array((.02, -.03, .01, .04, -.02, .03))
    shell = np.zeros((3, 6))
    shell[:, :3] = [v[:3]+np.cross(v[3:], x-law.shell_reference_m)
                     for x in TRIANGLE]
    np.testing.assert_allclose(law.shell_port@shell.ravel(), v,
                               rtol=0., atol=2e-16)
    saved_rows = law.constraint.copy()
    vertices[0, 0] = 100.
    np.testing.assert_array_equal(law.constraint, saved_rows)
    with pytest.raises(ValueError):
        law.constraint[0, 0] = 100.

    direction = np.sin(np.arange(30)+.3)*.01
    multiplier = np.array((.17, -.23, .31, .07, -.05, .11))
    shell_force, beam_force = law.forces(multiplier)
    predicted = np.r_[shell_force, beam_force]
    assert abs(multiplier@(law.constraint@direction)-predicted@direction) < 1e-14
    eps = 1e-6
    x0 = np.array((law.shell_reference_m, law.beam_reference_m))
    frames0 = law.reference_frames
    joint = RigidPoseJoint(x0, frames0)
    qs = law.shell_port@direction[:18]
    qb = law.beam_port@direction[18:]

    def actual_gap(sign):
        pose = sign*eps*np.array((qs, qb))
        rotated = np.array([
            Rotation.from_rotvec(row[3:]).as_matrix()@frame
            for row, frame in zip(pose, frames0)])
        return joint.evaluate(x0+pose[:, :3], rotated,
                              np.zeros(6)).constraints

    np.testing.assert_allclose((actual_gap(1)-actual_gap(-1))/(2*eps),
                               law.constraint@direction, rtol=2e-8, atol=2e-10)
    sf = shell_force.reshape(3, 6)
    bf = beam_force.reshape(2, 6)
    beam_nodes = np.asarray(member.reference_points_m)
    force_balance = sf[:, :3].sum(axis=0)+bf[:, :3].sum(axis=0)
    moment_balance = (sf[:, 3:]+np.cross(TRIANGLE, sf[:, :3])).sum(axis=0)
    moment_balance += (bf[:, 3:]+np.cross(beam_nodes, bf[:, :3])).sum(axis=0)
    np.testing.assert_allclose(force_balance, 0., atol=2e-13)
    np.testing.assert_allclose(moment_balance, 0., atol=2e-13)


def test_s3_physical_sparse_kkt_matches_reduced_solution_and_reactions():
    _, law, _, shell_k, beam_k = _specimen()
    stiffness = np.zeros((30, 30))
    stiffness[:18, :18] = shell_k
    stiffness[18:, 18:] = beam_k
    fixed = set(range(6)) | set(range(18, 24))
    # Keep the S3 bending rotations active; restrain only the two unneeded
    # drilling coordinates in this isolated, root-supported triangle.
    fixed.update(6*i+5 for i in (1, 2))
    free = np.array([i for i in range(30) if i not in fixed])
    k = stiffness[np.ix_(free, free)]
    c = law.constraint[:, free]
    assert np.linalg.matrix_rank(c) == 6
    load = np.zeros(30)
    load[6] = .15; load[7] = -.09; load[26] = .11; load[27] = .035
    f = load[free]
    sparse = bmat([[csc_matrix(k), csc_matrix(c.T)],
                   [csc_matrix(c), None]], format="csc")
    answer = spsolve(sparse, np.r_[f, np.zeros(6)])
    u, lam = answer[:len(f)], answer[len(f):]
    z = null_space(c)
    oracle = z@np.linalg.solve(z.T@k@z, z.T@f)
    dual = np.linalg.lstsq(c.T, f-k@oracle, rcond=None)[0]
    np.testing.assert_allclose(u, oracle, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(lam, dual, rtol=2e-10, atol=2e-11)
    assert np.linalg.norm(c@u) < 1e-11
    assert np.linalg.norm(k@u+c.T@lam-f) < 1e-10
    full = np.zeros(30); full[free] = u
    assert np.linalg.norm(full[[9, 10, 15, 16]]) > 1e-8
    reaction = stiffness@full+law.constraint.T@lam-load
    np.testing.assert_allclose(reaction[free], 0., atol=1e-10)
    assert np.linalg.norm(reaction[list(fixed)]) > 0.
    assert abs(full@load-full@stiffness@full) < 1e-10


def test_s3_invalid_support_and_input_mutation_rejected():
    member, law, _, _, _ = _specimen()
    bad = TRIANGLE.copy(); bad[2] = bad[1]
    with pytest.raises(ValueError, match="noncollinear|nondegenerate"):
        compile_reference_point(member, bad, station=.4)
    with pytest.raises(ValueError, match="shell displacement"):
        law.gap(np.zeros(24), np.zeros(12))


def test_s3_vertex_winding_preserves_physical_constraint_space():
    member, base, _, _, _ = _specimen()
    free_mode = null_space(base.constraint)
    for order in permutations(range(3)):
        candidate = compile_reference_point(
            member, TRIANGLE[list(order)], station=.4,
            offset_local_m=(0., .08, -.03))
        canonical = np.zeros_like(base.constraint)
        for input_row, original_row in enumerate(order):
            canonical[:, 6*original_row:6*original_row+6] = (
                candidate.constraint[:, 6*input_row:6*input_row+6])
        canonical[:, 18:] = candidate.constraint[:, 18:]
        assert np.linalg.matrix_rank(canonical) == 6
        assert np.linalg.norm(canonical@free_mode)/np.linalg.norm(canonical) < 1e-12
