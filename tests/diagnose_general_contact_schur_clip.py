"""Bounded projection-branch probe for the small coupled Schur fixture."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from anyfesolver.general_contact.linearized_pivot import _projected_clip_signature
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.q4_source_overlap import projected_q4_source_face
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3


def probe(label, manager, displacement):
    source, target = manager._positions(displacement)
    geometry = manager.law.geometry
    signature = _projected_clip_signature(geometry, source, target)
    print(f"{label}: clip_signature={'ambiguous' if signature is None else 'smooth'}")
    for si, source_nodes in enumerate(geometry.source_face_indices):
        first, normal = projected_q4_source_face(source[source_nodes])
        tangent = first.corners[1]-first.corners[0]
        tangent /= np.linalg.norm(tangent)
        binormal = np.cross(normal, tangent)
        origin = first.corners[0]
        for ti, target_nodes in enumerate(geometry.target_face_indices):
            projected = target[target_nodes]-(
                (target[target_nodes]-origin) @ normal
            )[:, None]*normal
            polygons = tuple(np.column_stack(((corners-origin) @ tangent,
                                              (corners-origin) @ binormal))
                             for corners in (first.corners, projected))
            for side, (vertices, edges) in enumerate((
                (polygons[0], polygons[1]), (polygons[1], polygons[0]),
            )):
                for vi, point in enumerate(vertices):
                    for ei, (start, end) in enumerate(zip(
                        edges, np.roll(edges, -1, axis=0),
                    )):
                        edge = end-start
                        along = float(np.dot(point-start, edge)/np.dot(edge, edge))
                        offset = point-start
                        distance = float((edge[0]*offset[1]-edge[1]*offset[0])
                                         /np.linalg.norm(edge))
                        if -1e-10 <= along <= 1+1e-10 and abs(distance) < 1e-8:
                            print(f"  pair={si},{ti} side={side} vertex={vi} "
                                  f"edge={ei} along={along:.12g} "
                                  f"signed_distance={distance:.12g}")


if __name__ == "__main__":
    model, definition, settings = _two_q4_two_s3()
    for node in (7, 8, 9, 10):
        model.mesh.get_node(node).x -= 0.5
    manager = GeneralS3PatchManager(model, definition, settings)
    zero = np.zeros(manager.n_dof)
    probe("reference", manager, zero)
    for node in range(1, 7):
        zero[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    probe("uniform_normal_displacement", manager, zero)
