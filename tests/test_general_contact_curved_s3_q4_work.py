"""Focused s9 tests: real skins, original Q4s and integrated-gap work.

One actual gravity preparation is shared across this process. No contact
solve, pressure-law calibration or mechanical reference is introduced.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.corotational import _rotation_right_jacobian, rotation_matrix_from_vector
from anyfesolver.general_contact.curved_shell_skin import CurvedShellSkinMap, TOTAL_ROTATION_BOUND_RAD
from anyfesolver.general_contact.curved_s3_q4_work import CurvedS3Q4WorkProbe, CurvedContactWorkError
from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.q4_target_overlap import q4_target_overlap_integrals
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from examples.elastic_sphere_panel_contact_work import (
    prepare_work_case, evaluate_case, first_discriminator, report_contact_work, physical_wrench,
)
from examples.elastic_sphere_panel_continuation import _model_snapshot, _states_equal


@pytest.fixture(scope="module")
def case():
    return prepare_work_case()


@pytest.fixture(scope="module")
def facet(case):
    return first_discriminator(case)["source_face_index"]


@pytest.fixture(scope="module")
def near(case, facet):
    return evaluate_case(case, "near", source_faces=(facet,))


def test_actual_ownership_seams_original_bilinear_and_scalar_parity(case, facet, near):
    assert (len(case.sphere_map.faces), len(case.panel_map.faces)) == (320, 100)
    assert len(near.pairs) >= 2
    sp = case.sphere_map.pose(case.poses["near"])
    tp = case.panel_map.pose(case.panel_q)
    for pair in near.pairs:
        i, j = pair["source_face_index"], pair["target_face_index"]
        source, target = case.sphere_map.faces[i], case.panel_map.faces[j]
        a, b = case.sphere_map.face_slices[i]
        c, d = case.panel_map.face_slices[j]
        order = pair["source_chart_corner_order"]
        assert order == [0, 2, 1]
        assert pair["source_chart_nodes"] == [source.node_ids[k] for k in order]
        assert pair["target_chart_nodes"] == list(target.node_ids)
        s, t = sp.positions[a:b][order], tp.positions[c:d]
        ref = source.reference_corners[order]
        ratio = np.linalg.norm(np.cross(ref[1]-ref[0], ref[2]-ref[0])) / np.linalg.norm(np.cross(s[1]-s[0], s[2]-s[0]))
        scalar = q4_target_overlap_integrals(PlanarFace((1, source.element_id), s), t,
                                            reference_area_ratio=ratio,
                                            weighted_gap_tolerance=1e-13, maximum_triangles=200)
        np.testing.assert_allclose(pair["reference_support_m2"], scalar.support, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(pair["reference_weighted_gap_m3"], scalar.weighted_gap, rtol=1e-11, atol=1e-16)
        assert pair["integration_triangles"] <= 200
        assert pair["derivative_order"] == 1 and pair["hessian_available"] is False
    area = near.coverage[0]["reference_facet_area_m2"]
    np.testing.assert_allclose(near.coverage[0]["support_m2"], np.full(3, area/3), rtol=1e-12, atol=1e-14)


def test_signs_empty_overlap_and_unresolved_refusal(case, facet, near, monkeypatch):
    assert min(g for p in near.pairs for g in p["source_corner_gaps_on_target_m"]) > 0
    closed = evaluate_case(case, "overclosure", source_faces=(facet,))
    assert min(g for p in closed.pairs for g in p["source_corner_gaps_on_target_m"]) < 0
    q = case.poses["near"].copy()
    for node in case.sphere.mesh.nodes.values():
        q[node.dofs[0]] += 10.
    empty = case.probe.evaluate(case.sphere_map.pose(q), case.panel_map.pose(case.panel_q), source_faces=(facet,))
    assert not empty.pairs and empty.empty_pairs == 100
    assert np.count_nonzero(empty.source_coordinate_work) == 0
    def fail(*args, **kwargs):
        raise UnsupportedQ4S3Branch("deliberate unresolved clipping branch")
    monkeypatch.setattr("anyfesolver.general_contact.curved_s3_q4_work._s3_q4_first_order_work", fail)
    with pytest.raises(UnsupportedQ4S3Branch, match="unresolved"):
        evaluate_case(case, "near", source_faces=(facet,))


def test_shared_sphere_corners_accumulate_without_merging(case, near):
    full = evaluate_case(case, "near")
    pose = case.sphere_map.pose(case.poses["near"])
    nonzero = np.flatnonzero(np.linalg.norm(full.source_corner_work, axis=1) > 0)
    found = False
    expected = np.zeros(case.sphere_map.n_dof)
    for node_id in set(case.sphere_map.corner_nodes):
        indices = [i for i in nonzero if case.sphere_map.corner_nodes[i] == node_id]
        if len(indices) > 1:
            points = pose.positions[indices]
            found |= np.max(np.linalg.norm(points - points[0], axis=1)) > 0
        dofs = np.asarray(case.sphere.mesh.nodes[node_id].dofs[:6], dtype=int)
        for i in indices:
            f = full.source_corner_work[i]
            expected[dofs[:3]] += f
            expected[dofs[3:]] += case.sphere_map.offsets[i] * np.cross(pose.directors[i], f)
    assert found
    np.testing.assert_allclose(full.source_spin_work, expected, rtol=0, atol=1e-14)


def test_coordinate_spin_virtual_work_and_resultants(case, facet, near):
    physical_wrench(case, near, "near")
    # A nonzero additive rotation distinguishes chart from spatial-spin work.
    q = case.poses["near"].copy()
    node = case.sphere.mesh.nodes[case.sphere_map.faces[facet].node_ids[0]]
    q[node.dofs[3:6]] = (.02, -.01, .015)
    work = case.probe.evaluate(case.sphere_map.pose(q), case.panel_map.pose(case.panel_q), source_faces=(facet,))
    rng = np.random.default_rng(910)
    for model, at, coordinate, spin, corners, skin in (
        (case.sphere, q, work.source_coordinate_work, work.source_spin_work, work.source_corner_work, case.sphere_map),
        (case.panel, case.panel_q, work.target_coordinate_work, work.target_spin_work, work.target_corner_work, case.panel_map),
    ):
        direction = rng.normal(size=skin.n_dof); direction /= np.linalg.norm(direction)
        virtual = direction.copy()
        for n in model.mesh.nodes.values():
            dofs = np.asarray(n.dofs[:6], dtype=int)
            virtual[dofs[3:]] = _rotation_right_jacobian(-at[dofs[3:]]) @ direction[dofs[3:]]
        assert abs(coordinate @ direction - spin @ virtual) <= 1e-13
        assert abs(coordinate @ direction - corners.ravel() @ (skin.pose(at).coordinate_map @ direction)) <= 1e-13


def test_integrated_gap_directional_derivative_both_deformable_bodies(case, facet, near):
    rng = np.random.default_rng(911)
    ds = rng.normal(size=case.sphere_map.n_dof); ds /= np.linalg.norm(ds)
    dt = rng.normal(size=case.panel_map.n_dof); dt /= np.linalg.norm(dt)
    expected = near.source_coordinate_work @ ds + near.target_coordinate_work @ dt
    for h in (1e-6, 3e-7):
        values = []
        for sign in (1, -1):
            values.append(case.probe.evaluate(
                case.sphere_map.pose(case.poses["near"] + sign*h*ds),
                case.panel_map.pose(case.panel_q + sign*h*dt), source_faces=(facet,)).integrated_gap_m3)
        finite = (values[0] - values[1])/(2*h)
        assert abs(finite - expected) / max(abs(expected), 1e-12) <= 1e-5


def test_common_rigid_motion_and_independent_deformation(case, facet, near):
    theta = np.array((.01, -.02, .015))
    rotation = rotation_matrix_from_vector(theta)
    shift = np.array((.1, -.2, .3))
    # New common physical rotation: use zero nodal base rotations for this
    # invariance check, so theta is the exact composed chart in both bodies.
    qs, qt = case.poses["near"].copy(), np.zeros(case.panel_map.n_dof)
    base = case.probe.evaluate(case.sphere_map.pose(qs), case.panel_map.pose(qt), source_faces=(facet,))
    for model, at in ((case.sphere, qs), (case.panel, qt)):
        for n in model.mesh.nodes.values():
            dofs = np.asarray(n.dofs[:6], dtype=int)
            position = np.asarray(n.coords()) + at[dofs[:3]]
            at[dofs[:3]] = rotation @ position + shift - np.asarray(n.coords())
            at[dofs[3:]] = theta
    moved = case.probe.evaluate(case.sphere_map.pose(qs), case.panel_map.pose(qt), source_faces=(facet,))
    np.testing.assert_allclose(moved.integrated_gap_m3, base.integrated_gap_m3, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(moved.coverage[0]["support_m2"], base.coverage[0]["support_m2"], rtol=1e-12, atol=1e-14)


def test_ambiguous_duplicate_target_support_refused(case, facet, near):
    first = near.pairs[0]["target_face_index"]
    face = case.panel_map.faces[first]
    other = next(f for f in case.panel_map.faces if not set(f.node_ids) & set(face.node_ids))
    skin = CurvedShellSkinMap(case.panel, [(face.body_id, face.element_id, "top"),
                                         (other.body_id, other.element_id, "top")])
    q = case.panel_q.copy()
    for a, b in zip(face.node_ids, other.node_ids):
        na, nb = case.panel.mesh.nodes[a], case.panel.mesh.nodes[b]
        q[nb.dofs[:3]] = np.asarray(na.coords()) + q[na.dofs[:3]] - np.asarray(nb.coords())
        q[nb.dofs[3:6]] = q[na.dofs[3:6]]
    probe = CurvedS3Q4WorkProbe(case.sphere, case.sphere_map, case.panel, skin)
    with pytest.raises(CurvedContactWorkError, match="partition is ambiguous"):
        probe.evaluate(case.sphere_map.pose(case.poses["near"]), skin.pose(q), source_faces=(facet,))


@pytest.mark.parametrize("mode", ("dimensions", "nonfinite", "rotation", "inverted", "wrong_pose", "wrong_model", "region"))
def test_invalid_inputs_refused(case, facet, mode):
    q = case.poses["near"].copy()
    sp, tp = case.sphere_map.pose(q), case.panel_map.pose(case.panel_q)
    with pytest.raises((ValueError, RuntimeError)):
        if mode == "dimensions":
            case.sphere_map.pose(q[:-1])
        elif mode == "nonfinite":
            q[0] = np.nan; case.sphere_map.pose(q)
        elif mode == "rotation":
            q[case.sphere_map.dofs[0, 3]] = TOTAL_ROTATION_BOUND_RAD + .01; case.sphere_map.pose(q)
        elif mode == "inverted":
            f = case.sphere_map.faces[facet]
            a, b = (case.sphere.mesh.nodes[i] for i in f.node_ids[:2])
            q[a.dofs[:3]] += np.asarray(b.coords()) - np.asarray(a.coords())
            q[b.dofs[:3]] += np.asarray(a.coords()) - np.asarray(b.coords())
            case.sphere_map.pose(q)
        elif mode == "wrong_pose":
            case.probe.evaluate(tp, sp, source_faces=(facet,))
        elif mode == "wrong_model":
            CurvedS3Q4WorkProbe(case.panel, case.sphere_map, case.sphere, case.panel_map)
        else:
            case.probe.evaluate(sp, tp, source_faces=(facet, facet))


def test_complete_report_preserves_models_preload_and_states(case):
    snapshots = (_model_snapshot(case.panel), _model_snapshot(case.sphere))
    q = case.panel_q.copy()
    states = copy.deepcopy(case.result.element_states)
    report = report_contact_work(case)
    assert report["status"] == "contact_geometry_prepared"
    assert report["sphere_solved"] is False and report["is_checkpoint"] is False
    assert report["contact_response"] == report["impact"] == "unsupported"
    assert report["model_and_preload_unchanged"]
    assert snapshots == (_model_snapshot(case.panel), _model_snapshot(case.sphere))
    assert np.array_equal(q, case.panel_q)
    assert _states_equal(states, case.result.element_states)
    if os.environ.get("CONTACT_WORK_EVIDENCE"):
        Path(os.environ["CONTACT_WORK_EVIDENCE"]).write_text(json.dumps(report, indent=2, allow_nan=False))
