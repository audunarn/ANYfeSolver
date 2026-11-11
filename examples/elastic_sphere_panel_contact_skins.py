"""Curved sphere-panel contact skin geometry report (development-only).

This module assembles the delivered elastic sphere stiffened-panel case once
through the existing preparation (``prepare_case_models``: canonical panel
build, fixed sphere skin build, single qualified panel-only gravity solve) and
maps the actual physical exterior skins of both bodies with the
development-only face-local curved shell skin service:

- the sphere's 320 S3 triangles use their per-element outward reference
  directors and the S3 top exterior offset, so a shared structural node keeps
  one distinct physical corner per incident element;
- the panel's 100 original Q4 plate faces use their bilinear topside physical
  offset (the stiffener beams and beam-shell couplings are structural only
  and are not contact faces);
- the panel skin is recovered at the accepted gravity-preloaded displacement;
  the sphere skin is reported at its zero-pose reference and at the declared
  placement, which is a prescribed kinematic probe (a rigid translation), not
  a solved response;
- both bodies additionally carry clearly labelled prescribed kinematic
  probes: a small rigid rotation of the placed sphere and genuinely local
  deformation probes on a small patch of each body (the panel one on top of
  the accepted gravity state).  None of these is a solved response;
- the same fixed mass/reaction/global-balance acceptance as the
  initial-state workflow is applied to the single gravity solve, and its
  compact passed measurements are recorded in the report.

The output status is ``skin_geometry_prepared``; contact response and the
gravity impact remain explicitly unsupported.  The offset facets are honest
per-element skins: they are not claimed to form a watertight smooth sphere,
and no contact gap or force history is measured or implied.  This report is
not a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from anyfesolver import BeamElement
from anyfesolver.corotational import rotation_matrix_from_vector
from anyfesolver.general_contact.curved_shell_skin import CurvedShellSkinMap
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement

from examples.elastic_sphere_stiffened_panel import (
    DROP_CLEARANCE,
    FORCE_BALANCE_TOLERANCE,
    MOMENT_BALANCE_TOLERANCE,
    PLATE_THICKNESS,
    PLATE_TOP_Z,
    REACTION_AGREEMENT_TOLERANCE,
    RELATIVE_MASS_TOLERANCE,
    SOLVE_SETTINGS,
    SPHERE_OUTSIDE_RADIUS,
    TARGET_PANEL_MASS,
    _runtime_identity,
    case_acceptance,
    prepare_case_models,
)

SCHEMA = "anyfesolver.elastic-sphere-panel-contact-skins-v1"
STATUS = "skin_geometry_prepared"
CONTACT_RESPONSE_STATUS = "unsupported"
IMPACT_STATUS = "unsupported"

SPHERE_BODY_ID = 1
PANEL_BODY_ID = 2
SKIN_SIDE = "top"

UNSUPPORTED_NOTE = (
    "This document reports prepared skin geometry only. No contact response, "
    "gap history, contact force or impact execution is claimed or implied: "
    "the current solver has no admitted contact solve for this case, and the "
    "geometry service is development-only."
)

# Declared fixed inventory of the delivered case (fail closed on mismatch).
SPHERE_FACE_COUNT = 320
PANEL_FACE_COUNT = 100

# Deterministic prescribed kinematic probe: small rigid rotation of the
# stress-free sphere about a fixed normalized axis (not a solved response).
PROBE_ROTATION_RAD = 0.1
PROBE_ROTATION_AXIS = np.array((1.0, 2.0, 3.0))
PROBE_ROTATION_AXIS = PROBE_ROTATION_AXIS / np.linalg.norm(PROBE_ROTATION_AXIS)


def sphere_contact_faces(sphere_model):
    """All S3 sphere triangles as exterior top skin face declarations."""
    return tuple(
        (SPHERE_BODY_ID, int(element_id), SKIN_SIDE)
        for element_id in sorted(sphere_model.mesh.elements)
    )


def panel_contact_faces(panel_model):
    """The original Q4 plate tops as skin faces; beams/couplings excluded."""
    faces = []
    for element_id, element in sorted(panel_model.mesh.elements.items()):
        if isinstance(element, (BeamElement, InterpolatedBeamShellMPCElement)):
            continue
        faces.append((PANEL_BODY_ID, int(element_id), SKIN_SIDE))
    return tuple(faces)


def build_skin_maps(panel_model, sphere_model):
    """Build both curved skin maps from the actual delivered models."""
    sphere_map = CurvedShellSkinMap(sphere_model, sphere_contact_faces(sphere_model))
    panel_map = CurvedShellSkinMap(panel_model, panel_contact_faces(panel_model))
    if len(sphere_map.faces) != SPHERE_FACE_COUNT:
        raise RuntimeError(
            f"sphere skin inventory {len(sphere_map.faces)} != {SPHERE_FACE_COUNT}")
    if len(panel_map.faces) != PANEL_FACE_COUNT:
        raise RuntimeError(
            f"panel skin inventory {len(panel_map.faces)} != {PANEL_FACE_COUNT}")
    return sphere_map, panel_map


def _face_entries(skin_map, reference_pose, recovered_pose, recovered_label):
    """Per-face records with distinct reference and recovered geometries.

    ``reference_positions_m`` are the zero-pose reference corners and
    ``recovered_positions_m`` the corners of the recovered (or prescribed
    probe) state named by ``recovered_label``; the two poses are passed
    separately so zero-pose positions are never labelled as recovered
    preload or release geometry.  Each record also carries the current
    geometric normal of the interpolated face at its centre (the original
    linear/bilinear face geometry, not the rotated element director),
    evaluated with the map's geometry-only normal method for both states.
    """
    reference_normals = skin_map.face_centre_normals(reference_pose)
    recovered_normals = skin_map.face_centre_normals(recovered_pose)
    entries = []
    for index, (face, (start, end)) in enumerate(
            zip(skin_map.faces, skin_map.face_slices)):
        entries.append({
            "body_id": face.body_id,
            "element_id": face.element_id,
            "side": face.side,
            "formulation_id": face.formulation_id,
            "local_node_ids": [int(node) for node in face.node_ids],
            "physical_director": [float(v) for v in face.director],
            "physical_offset_m": float(face.physical_offset),
            "reference_positions_m": [
                [float(v) for v in row]
                for row in reference_pose.positions[start:end]],
            "reference_geometric_normal": [
                float(v) for v in reference_normals[index]],
            "recovered_positions_m": [
                [float(v) for v in row]
                for row in recovered_pose.positions[start:end]],
            "recovered_geometric_normal": [
                float(v) for v in recovered_normals[index]],
            "recovered_geometry": recovered_label,
        })
    return entries


def _local_deformation_probe(model, node_ids, seed):
    """Deterministic small local translation/rotation probe (kinematic only)."""
    rng = np.random.default_rng(seed)
    probe = np.zeros(int(model.mesh.dof_manager.total_dofs))
    for node_id in sorted(int(n) for n in node_ids):
        dofs = np.asarray(model.mesh.get_node(node_id).dofs[:6], dtype=np.intp)
        probe[dofs[:3]] = 1.0e-4 * rng.normal(size=3)
        probe[dofs[3:]] = 1.0e-3 * rng.normal(size=3)
    return probe


def _probe_patch(model, count, centre):
    """The ``count`` model nodes nearest a centre point (deterministic)."""
    centre = np.asarray(centre, dtype=float)
    nodes = sorted(
        model.mesh.nodes.values(),
        key=lambda node: float(np.linalg.norm(
            np.asarray(node.coords(), dtype=float) - centre)))
    return [int(node.id) for node in nodes[:count]]


def _shared_edge_separations(skin_map, positions):
    """Separation between offset copies of each topologically shared edge.

    Neighbouring elements carry their own directors, so the offset skin
    corners of one shared mesh edge generally differ between the two incident
    faces.  This measures that gap honestly; it does not claim watertightness.
    """
    edges: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(skin_map.faces):
        count = len(face.node_ids)
        for k in range(count):
            a, b = face.node_ids[k], face.node_ids[(k + 1) % count]
            edges.setdefault((min(a, b), max(a, b)), []).append(face_index)
    separations = []
    for face_list in edges.values():
        if len(face_list) < 2:
            continue
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                first, second = face_list[i], face_list[j]
                start_a, end_a = skin_map.face_slices[first]
                start_b, end_b = skin_map.face_slices[second]
                nodes_a = skin_map.faces[first].node_ids
                nodes_b = skin_map.faces[second].node_ids
                for local, node in enumerate(nodes_a):
                    if node in nodes_b:
                        other = nodes_b.index(node)
                        separations.append(float(np.linalg.norm(
                            positions[start_a + local] - positions[start_b + other])))
    if not separations:
        return {"shared_corner_comparisons": 0}
    values = np.asarray(separations)
    return {
        "shared_corner_comparisons": int(len(separations)),
        "max_corner_separation_m": float(np.max(values)),
        "mean_corner_separation_m": float(np.mean(values)),
    }


def prepare_skin_geometry() -> dict:
    """Assemble the case once and report both curved contact skins."""
    panel_model, sphere_model, sphere_geometry, load_case, result, displacements = (
        prepare_case_models()
    )
    # The same fixed mass/reaction/global-balance acceptance as the
    # initial-state workflow, applied to this single gravity solve (no second
    # solve); the compact passed measurements are recorded in the report.
    acceptance = case_acceptance(
        panel_model, sphere_model, sphere_geometry, load_case, result,
        displacements)
    sphere_map, panel_map = build_skin_maps(panel_model, sphere_model)

    zero_sphere = np.zeros(sphere_map.n_dof)
    sphere_reference_pose = sphere_map.pose(zero_sphere)

    # Declared placement (same convention as the initial-state example).
    centre_node = min(
        panel_model.mesh.nodes.values(),
        key=lambda node: abs(node.x) + abs(node.y),
    )
    centre_top_z = PLATE_TOP_Z + float(displacements[centre_node.dofs[2]])
    sphere_centre_z = centre_top_z + DROP_CLEARANCE + SPHERE_OUTSIDE_RADIUS

    # Prescribed kinematic probe: rigid translation of the whole sphere into
    # the case frame (its accepted displacement stays zero; it is never solved).
    placement = np.zeros(sphere_map.n_dof)
    for node in sphere_model.mesh.nodes.values():
        placement[node.dofs[2]] = sphere_centre_z
    sphere_placement_pose = sphere_map.pose(placement)

    # Prescribed kinematic probe: small rigid rotation of the placed sphere
    # about its own centre (nodes translate by (R - I)X and spin by theta).
    rotation = PROBE_ROTATION_RAD * PROBE_ROTATION_AXIS
    rotation_matrix = rotation_matrix_from_vector(rotation)
    spin = np.zeros(sphere_map.n_dof)
    for node in sphere_model.mesh.nodes.values():
        spin[node.dofs[:3]] = rotation_matrix @ node.coords() - node.coords()
        spin[node.dofs[3:]] = rotation
    sphere_rotation_pose = sphere_map.pose(placement + spin)
    probe_motion = float(np.max(np.linalg.norm(
        sphere_rotation_pose.positions - sphere_placement_pose.positions, axis=1)))

    # Prescribed kinematic probe: genuinely local deformation of a small
    # sphere patch near the top pole (a few nodes only; never solved).
    sphere_patch = _probe_patch(sphere_model, 6, (0.0, 0.0, 0.5))
    sphere_local = _local_deformation_probe(sphere_model, sphere_patch, seed=11)
    sphere_local_pose = sphere_map.pose(sphere_local)
    sphere_local_motion = float(np.max(np.linalg.norm(
        sphere_local_pose.positions - sphere_reference_pose.positions, axis=1)))

    # Panel skin at the accepted gravity-preloaded displacement.
    zero_panel = np.zeros(panel_map.n_dof)
    panel_reference_pose = panel_map.pose(zero_panel)
    panel_recovered_pose = panel_map.pose(displacements)

    # Prescribed kinematic probe: local deformation added to the accepted
    # gravity state on a small patch near the impact centre (never solved).
    panel_patch = _probe_patch(panel_model, 6, (0.0, 0.0, 0.0))
    panel_local = displacements + _local_deformation_probe(
        panel_model, panel_patch, seed=13)
    panel_local_pose = panel_map.pose(panel_local)
    panel_local_motion = float(np.max(np.linalg.norm(
        panel_local_pose.positions - panel_recovered_pose.positions, axis=1)))

    radii = np.linalg.norm(sphere_reference_pose.positions, axis=1)
    lowest_exterior_z = float(np.min(sphere_placement_pose.positions[:, 2]))
    measured_clearance = lowest_exterior_z - centre_top_z

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "contact_response": CONTACT_RESPONSE_STATUS,
        "impact_status": IMPACT_STATUS,
        "unsupported_note": UNSUPPORTED_NOTE,
        "case": {
            "gravity_solve_settings": dict(SOLVE_SETTINGS),
            "gravity_solve_status": str(result.status),
            "skin_service": (
                "development-only face-local curved shell skin map "
                "(anyfesolver.general_contact.curved_shell_skin); not an "
                "admitted contact solve"
            ),
            "sphere_skin": {
                "body_id": SPHERE_BODY_ID,
                "formulation": "e4-pl-s3-v2d S3 triangles, linear interpolation",
                "side": SKIN_SIDE,
                "face_count": len(sphere_map.faces),
                "corner_count": int(len(sphere_map.corner_keys)),
                "midsurface_radius_m": sphere_geometry["midsurface_radius_m"],
                "thickness_m": sphere_geometry["thickness_m"],
                "nominal_outside_radius_m": SPHERE_OUTSIDE_RADIUS,
            },
            "panel_skin": {
                "body_id": PANEL_BODY_ID,
                "formulation": "qualified Q4 plate shells, bilinear interpolation",
                "side": SKIN_SIDE,
                "face_count": len(panel_map.faces),
                "corner_count": int(len(panel_map.corner_keys)),
                "plate_thickness_m": PLATE_THICKNESS,
                "note": "stiffener beams and beam-shell couplings are structural only",
            },
        },
        "sphere": {
            "faces": _face_entries(
                sphere_map, sphere_reference_pose, sphere_placement_pose,
                "prescribed kinematic placement probe (rigid translation; "
                "not a solved response)"),
            "recovered_geometry": {
                "label": "prescribed kinematic placement probe (rigid translation)",
                "centre_m": [0.0, 0.0, float(sphere_centre_z)],
                "positions_m": [
                    [float(v) for v in row]
                    for row in sphere_placement_pose.positions
                ],
            },
            "probes": [
                {
                    "label": "prescribed rigid rotation probe (kinematic only)",
                    "rotation_rad": PROBE_ROTATION_RAD,
                    "axis": [float(v) for v in PROBE_ROTATION_AXIS],
                    "max_corner_motion_m": probe_motion,
                    "solved_response": False,
                },
                {
                    "label": (
                        "prescribed local deformation probe on a small top-pole "
                        "patch (kinematic only)"),
                    "patch_node_ids": sphere_patch,
                    "max_corner_motion_m": sphere_local_motion,
                    "solved_response": False,
                },
            ],
            "measurements": {
                "corner_radial_range_m": {
                    "min": float(np.min(radii)),
                    "max": float(np.max(radii)),
                    "nominal_outside_radius_m": SPHERE_OUTSIDE_RADIUS,
                    "note": (
                        "range of the offset face-corner radii only, not "
                        "face-interior extrema; per-element facet directors "
                        "are not exactly radial, so the offset corners stay "
                        "inside the nominal outside radius; no smooth-sphere "
                        "claim"
                    ),
                },
                "shared_edge_separations": _shared_edge_separations(
                    sphere_map, sphere_reference_pose.positions),
                "watertight_claim": False,
            },
        },
        "panel": {
            "faces": _face_entries(
                panel_map, panel_reference_pose, panel_recovered_pose,
                "accepted gravity-preloaded panel displacement"),
            "recovered_geometry": {
                "label": "accepted gravity-preloaded panel displacement",
                "positions_m": [
                    [float(v) for v in row]
                    for row in panel_recovered_pose.positions
                ],
            },
            "probes": [
                {
                    "label": (
                        "prescribed local deformation probe added to the "
                        "accepted gravity state near the impact centre "
                        "(kinematic only)"),
                    "patch_node_ids": panel_patch,
                    "max_corner_motion_m": panel_local_motion,
                    "solved_response": False,
                },
            ],
            "measurements": {
                "shared_edge_separations": _shared_edge_separations(
                    panel_map, panel_reference_pose.positions),
                "plate_top_z_at_impact_centre_m": float(centre_top_z),
            },
        },
        "acceptance": {
            "note": (
                "the same fixed mass/reaction/global-balance acceptance as "
                "the initial-state workflow, applied to this single gravity "
                "solve (no second solve)"
            ),
            "sphere_translation_masses_kg": [
                float(v) for v in acceptance["sphere_masses_kg"]],
            "sphere_faceted_mass_kg": float(
                acceptance["sphere_faceted_mass_kg"]),
            "panel_translation_masses_kg": [
                float(v) for v in acceptance["panel_masses_kg"]],
            "panel_target_mass_kg": float(TARGET_PANEL_MASS),
            "relative_mass_tolerance": RELATIVE_MASS_TOLERANCE,
            "max_support_resultant_relative_deviation": float(
                acceptance["reaction_check"]["max_relative_component_deviation"]),
            "reaction_agreement_tolerance": REACTION_AGREEMENT_TOLERANCE,
            "force_balance_norm_N": float(
                acceptance["equilibrium"]["force_balance_norm_N"]),
            "force_balance_tolerance_N": FORCE_BALANCE_TOLERANCE,
            "moment_balance_norm_Nm": float(
                acceptance["equilibrium"]["moment_balance_norm_Nm"]),
            "moment_balance_tolerance_Nm": MOMENT_BALANCE_TOLERANCE,
            "passed": True,
        },
        "placement": {
            "nominal_clearance_m": DROP_CLEARANCE,
            "measured_centre_referenced_clearance_m": float(measured_clearance),
            "sphere_lowest_exterior_z_m": lowest_exterior_z,
            "panel_top_z_at_impact_centre_m": float(centre_top_z),
            "note": (
                "the nominal 10 m clearance is declared between the lowest "
                "outer-surface point and the equilibrated panel top at the "
                "impact centre; the measured value is the centre-referenced "
                "vertical clearance (lowest exterior skin corner Z minus "
                "panel top Z at the impact centre) computed from the "
                "recovered offset skins, whose facet corners sit slightly "
                "inside the nominal outside radius; it is a labelled "
                "vertical offset, not a computed interbody contact gap"
            ),
        },
        "identity": _runtime_identity(),
    }
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the skin-geometry JSON document to this path")
    args = parser.parse_args(argv)
    payload = prepare_skin_geometry()
    document = json.dumps(payload, indent=2)
    if args.output is None:
        print(document)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "contact_response": payload["contact_response"],
        "impact_status": payload["impact_status"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
