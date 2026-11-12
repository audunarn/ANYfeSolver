"""Independent graph bound for the flat-target D1 development fixture.

The contact adapter supplies physical skin coordinates. This diagnostic does
not call contact projection, integration, gap-certificate or pressure kernels.
It applies only to the documented horizontal, fixed-target, rotation-locked
fixture and is not a general shell-contact certificate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from general_contact_d1_coupled_refinement import build_case
from anyfesolver.general_contact.s3_patch_static import _adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    model, definition, _, _ = build_case(
        record["columns"], record["rows"], record["edge_control"],
    )
    adapters = _adapters(model, definition.surfaces)
    displacements = record["final_nodal_displacements"]
    if record["status"] != "completed":
        raise ValueError("gap probe needs an accepted pose")
    if any(max(abs(value) for value in values[3:]) > 1e-12
           for values in displacements.values()):
        raise ValueError("gap probe needs rotation-locked shells")

    def current(adapter):
        return np.asarray([
            position + np.asarray(displacements[str(node_id)][:3])
            for node_id, position in zip(adapter.node_ids,
                                         adapter.reference_positions)
        ])

    source_faces = [current(item) for item in adapters[0]]
    target_faces = [current(item) for item in adapters[1]]
    target_z = np.concatenate(target_faces)[:, 2]
    if float(np.ptp(target_z)) > 1e-12:
        raise ValueError("gap probe needs a fixed planar horizontal target")
    if any(np.max(np.abs(face[:, 1] - face[0, 1])) < 1e-12
           for face in source_faces):
        raise ValueError("source face is degenerate")

    minimum_vertical = float("inf")
    maximum_x_slope = maximum_y_slope = 0.0
    lowest_nodes = []
    for adapter, face in zip(adapters[0], source_faces):
        if face.shape != (4, 3):
            raise ValueError("gap probe needs Q4 source faces")
        if abs(face[1, 1]-face[0, 1]) > 1e-12 or abs(face[2, 1]-face[3, 1]) > 1e-12:
            raise ValueError("gap probe needs x-aligned source edges")
        if abs(face[3, 0]-face[0, 0]) > 1e-5 or abs(face[2, 0]-face[1, 0]) > 1e-5:
            raise ValueError("source in-plane shear exceeds diagnostic bound")
        x_xi_min = min(face[1, 0]-face[0, 0], face[2, 0]-face[3, 0])
        y_eta_min = min(face[3, 1]-face[0, 1], face[2, 1]-face[1, 1])
        if x_xi_min <= 0.0 or y_eta_min <= 0.0:
            raise ValueError("gap probe needs a positive graph Jacobian")
        z_xi_max = max(abs(face[1, 2]-face[0, 2]),
                       abs(face[2, 2]-face[3, 2]))
        z_eta_max = max(abs(face[3, 2]-face[0, 2]),
                        abs(face[2, 2]-face[1, 2]))
        x_eta_max = max(abs(face[3, 0]-face[0, 0]),
                        abs(face[2, 0]-face[1, 0]))
        x_slope = z_xi_max/x_xi_min
        y_slope = (z_eta_max + z_xi_max*x_eta_max/x_xi_min)/y_eta_min
        maximum_x_slope = max(maximum_x_slope, x_slope)
        maximum_y_slope = max(maximum_y_slope, y_slope)
        for node_id, point in zip(adapter.node_ids, face):
            gap = float(point[2]-target_z[0])
            if gap < minimum_vertical - 1e-12:
                minimum_vertical = gap
                lowest_nodes = [node_id]
            elif abs(gap-minimum_vertical) <= 1e-12 and node_id not in lowest_nodes:
                lowest_nodes.append(node_id)

    cosine_lower = 1.0/np.sqrt(
        1.0 + maximum_x_slope**2 + maximum_y_slope**2,
    )
    # Bilinear Q4 height is a convex combination of corner heights. For any
    # point whose normal ray meets the target plane, a negative signed gap is
    # vertical gap divided by its positive normal cosine.
    physical_lower = (minimum_vertical/cosine_lower
                      if minimum_vertical < 0.0 else minimum_vertical*cosine_lower)
    report = {
        "input": str(args.input),
        "scope": "fixed horizontal planar target; rotation-locked Q4 graph source",
        "target_z_m": float(target_z[0]),
        "minimum_source_vertex_vertical_gap_m": minimum_vertical,
        "lowest_source_nodes": lowest_nodes,
        "maximum_graph_x_slope": maximum_x_slope,
        "maximum_graph_y_slope": maximum_y_slope,
        "source_normal_cosine_lower": float(cosine_lower),
        "independent_physical_gap_lower_m": float(physical_lower),
        "reported_whole_face_lower_m": record["force_displacement_history"][-1]["contact"]["minimum_gap"],
        "reported_sampled_overlap_gap_m": record["force_displacement_history"][-1]["contact"]["sampled_overlap_gap"],
    }
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
