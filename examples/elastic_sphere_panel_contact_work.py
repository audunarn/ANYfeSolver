"""Actual curved sphere / gravity-panel overlap and 1 Pa virtual-work probe.

Run with ``python -m examples.elastic_sphere_panel_contact_work --output work.json``.
All motions of the sphere are prescribed kinematic probes. No contact law,
equilibrium reaction, impact, swept search or restart checkpoint is provided.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.curved_s3_q4_work import (
    CurvedS3Q4WorkProbe, INTEGRATION_CONTROLS, PROBE_PRESSURE_PA,
)
from examples.elastic_sphere_panel_contact_skins import (
    _shared_edge_separations, build_skin_maps,
)
from examples.elastic_sphere_panel_continuation import _model_snapshot, _states_equal
from examples.elastic_sphere_stiffened_panel import (
    DROP_CLEARANCE, PLATE_TOP_Z, SPHERE_OUTSIDE_RADIUS,
    _runtime_identity, case_acceptance, prepare_case_models,
)

STATUS = "contact_geometry_prepared"
CAP_DEPTH_M = 0.02
NEAR_VERTICAL_CLEARANCE_M = 0.005
OVERCLOSURE_VERTICAL_CLEARANCE_M = -0.001


@dataclass
class WorkCase:
    panel: object
    sphere: object
    sphere_map: object
    panel_map: object
    probe: object
    panel_q: np.ndarray
    poses: dict
    cap: tuple
    acceptance: dict
    result: object


def prepare_work_case():
    panel, sphere, geometry, load, result, q = prepare_case_models()
    acceptance = case_acceptance(panel, sphere, geometry, load, result, q)
    sphere_map, panel_map = build_skin_maps(panel, sphere)
    zero = sphere_map.pose(np.zeros(sphere_map.n_dof))
    minimum = float(zero.positions[:, 2].min())
    cap = tuple(i for i, (a, b) in enumerate(sphere_map.face_slices)
                if float(zero.positions[a:b, 2].min()) <= minimum + CAP_DEPTH_M)
    centre = min(panel.mesh.nodes.values(), key=lambda n: abs(n.x) + abs(n.y))
    top = PLATE_TOP_Z + float(q[centre.dofs[2]])
    heights = dict(
        release=top + DROP_CLEARANCE + SPHERE_OUTSIDE_RADIUS,
        near=top + NEAR_VERTICAL_CLEARANCE_M - minimum,
        overclosure=top + OVERCLOSURE_VERTICAL_CLEARANCE_M - minimum,
    )
    poses = {}
    for name, height in heights.items():
        at = np.zeros(sphere_map.n_dof)
        for node in sphere.mesh.nodes.values():
            at[node.dofs[2]] = height
        poses[name] = at
    return WorkCase(panel, sphere, sphere_map, panel_map,
                    CurvedS3Q4WorkProbe(sphere, sphere_map, panel, panel_map),
                    np.array(q, copy=True), poses, cap, acceptance, result)


def evaluate_case(case, name, *, source_faces=None):
    return case.probe.evaluate(
        case.sphere_map.pose(case.poses[name]), case.panel_map.pose(case.panel_q),
        source_faces=case.cap if source_faces is None else source_faces)


def physical_wrench(case, work, name):
    """Resultant via independent nodal positions + spatial-spin moments."""
    force, moment = np.zeros(3), np.zeros(3)
    scale_force = scale_moment = 0.
    for model, q, vector in ((case.sphere, case.poses[name], work.source_spin_work),
                              (case.panel, case.panel_q, work.target_spin_work)):
        for node in model.mesh.nodes.values():
            dofs = np.asarray(node.dofs[:6], dtype=int)
            f = vector[dofs[:3]]
            m = np.cross(np.asarray(node.coords()) + q[dofs[:3]], f) + vector[dofs[3:]]
            force += f
            moment += m
            scale_force += float(np.linalg.norm(f))
            scale_moment += float(np.linalg.norm(m))
    result = dict(force_N=force.tolist(), moment_Nm=moment.tolist(),
                  force_norm_N=float(np.linalg.norm(force)),
                  moment_norm_Nm=float(np.linalg.norm(moment)),
                  force_scale_N=max(1., scale_force), moment_scale_Nm=max(1., scale_moment))
    if (result["force_norm_N"] > 1e-8 * result["force_scale_N"]
            or result["moment_norm_Nm"] > 1e-8 * result["moment_scale_Nm"]):
        raise RuntimeError("unit-gap-work probe resultant is unbalanced")
    return result


def first_discriminator(case):
    """Choose by geometry, not a successful numerical result or shifted pose."""
    pose = case.sphere_map.pose(case.poses["near"])
    crossing = []
    for i in case.cap:
        a, b = case.sphere_map.face_slices[i]
        corners = pose.positions[a:b]
        if any(corners[:, k].min() < 0 < corners[:, k].max() for k in (0, 1)):
            crossing.append((float(corners[:, 2].mean()), i))
    if not crossing:
        raise RuntimeError("no actual cap facet spans a central panel seam")
    index = min(crossing)[1]
    reports = {}
    for name in ("near", "overclosure"):
        work = evaluate_case(case, name, source_faces=(index,))
        coverage = work.coverage[0]
        if len(coverage["target_face_indices"]) < 2:
            raise RuntimeError("first actual facet did not span multiple Q4 targets")
        area = coverage["reference_facet_area_m2"]
        expected = np.full(3, area / 3)
        np.testing.assert_allclose(coverage["support_m2"], expected, rtol=1e-12, atol=1e-14)
        gaps = np.array([g for pair in work.pairs for g in pair["normalized_gap_m"]])
        corner_gaps = [g for pair in work.pairs for g in pair["source_corner_gaps_on_target_m"]]
        # The near pose is separated everywhere on this facet. A small vertical
        # overclosure need not put every sloping-facet pressure row in contact.
        if name == "near" and not np.all(gaps > 0):
            raise RuntimeError("separation gap sign is wrong")
        if not corner_gaps or (name == "overclosure" and min(corner_gaps) >= 0):
            raise RuntimeError("overclosure must have a measured negative physical corner gap")
        reports[name] = dict(coverage=coverage, wrench=physical_wrench(case, work, name),
                             minimum_normalized_gap_m=float(gaps.min()),
                             minimum_sampled_corner_gap_m=min(corner_gaps),
                             integrated_gap_m3=work.integrated_gap_m3)
    # The entire source facet need not penetrate at -1 mm. A declared extra
    # constant normal-direction translation is not introduced to force it.
    if not reports["overclosure"]["integrated_gap_m3"] < reports["near"]["integrated_gap_m3"]:
        raise RuntimeError("prescribed downward approach did not reduce signed gap")
    return dict(status="PASS", source_face_index=index, probes=reports)


def report_contact_work(case=None):
    case = prepare_work_case() if case is None else case
    before = (_model_snapshot(case.panel), _model_snapshot(case.sphere))
    q_before = case.panel_q.copy()
    poses_before = {k: q.copy() for k, q in case.poses.items()}
    states_before = copy.deepcopy(case.result.element_states)
    discriminator = first_discriminator(case)
    probes = {}
    for name in ("release", "near", "overclosure"):
        work = evaluate_case(case, name)
        if not work.pairs:
            raise RuntimeError("declared cap query returned no positive projected support")
        for coverage in work.coverage:
            area = coverage["reference_facet_area_m2"]
            np.testing.assert_allclose(coverage["support_m2"], np.full(3, area / 3),
                                       rtol=1e-12, atol=1e-14)
        probes[name] = dict(
            prescribed_displacements=case.poses[name].tolist(),
            pairs=list(work.pairs), coverage=list(work.coverage), empty_pairs=work.empty_pairs,
            integrated_gap_m3=work.integrated_gap_m3,
            source_coordinate_work=work.source_coordinate_work.tolist(),
            target_coordinate_work=work.target_coordinate_work.tolist(),
            source_spin_work=work.source_spin_work.tolist(),
            target_spin_work=work.target_spin_work.tolist(),
            wrench=physical_wrench(case, work, name),
        )
    if (before != (_model_snapshot(case.panel), _model_snapshot(case.sphere))
            or not np.array_equal(q_before, case.panel_q)
            or any(not np.array_equal(q, case.poses[k]) for k, q in poses_before.items())
            or not _states_equal(states_before, case.result.element_states)):
        raise RuntimeError("contact geometry probe mutated model/preload/material state")
    payload = dict(
        schema="anyfesolver.elastic-sphere-panel-contact-work-v1", status=STATUS,
        contact_response="unsupported", impact="unsupported", is_checkpoint=False,
        numerical_acceptance="local_windows_only; paired_host_pending",
        interpretation="1 Pa derivative of reference-area integrated gap; no contact reaction or pressure law",
        gap_convention="positive separation, negative overclosure; vertical placement is not the normal gap",
        pressure_Pa=PROBE_PRESSURE_PA, integration_controls=INTEGRATION_CONTROLS,
        region=dict(cap_depth_m=CAP_DEPTH_M, source_face_indices=list(case.cap),
                    all_target_faces_searched=True, complete_impact_search=False),
        inventory=dict(sphere_faces=[[f.body_id, f.element_id, list(f.node_ids)] for f in case.sphere_map.faces],
                       panel_faces=[[f.body_id, f.element_id, list(f.node_ids)] for f in case.panel_map.faces]),
        sphere_facet_edge_separations=_shared_edge_separations(
            case.sphere_map, case.sphere_map.pose(np.zeros(case.sphere_map.n_dof)).positions),
        panel_reference_coordinates_retained=True, sphere_solved=False,
        gravity_displacement=case.panel_q.tolist(), model_and_preload_unchanged=True,
        gravity_acceptance=case.acceptance, discriminator=discriminator,
        probes=probes, runtime=_runtime_identity(),
        next_work=["combined two-body/preload/work-conjugate contact integration",
                   "static contact law and indentation equilibrium", "elastic dynamics and full 10 m drop"],
    )
    # Reject any nonfinite diagnostic, including nested arrays, before export.
    json.dumps(payload, allow_nan=False)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = report_contact_work()
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=payload["status"], output=str(args.output),
                         contact_response="unsupported", impact="unsupported")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
