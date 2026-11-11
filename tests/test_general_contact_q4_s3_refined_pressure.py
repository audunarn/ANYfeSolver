"""Independent support control for the multi-Q4/S3 pressure route."""

import numpy as np
import pytest

from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


@pytest.mark.parametrize("divisions,active_nodes", ((4, 0), (8, 1)))
def test_refined_q4_s3_affine_corner_pressure_follows_independent_q1_integrals(
        divisions, active_nodes):
    penetration = 1.0e-5
    slope = 1.0e-4
    coordinates = {}

    def node(i, j):
        return 1 + j*(divisions + 1) + i

    for j in range(divisions + 1):
        for i in range(divisions + 1):
            coordinates[node(i, j)] = (i/divisions, j/divisions)
    source_faces = []
    for j in range(divisions):
        for i in range(divisions):
            ids = (node(i, j), node(i + 1, j),
                   node(i + 1, j + 1), node(i, j + 1))
            source_faces.append(S3PatchFace(
                ids, np.array([(*coordinates[index], 0.0) for index in ids]),
            ))
    target_faces = (
        S3PatchFace((10001, 10002, 10003), np.array((
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
        ))),
        S3PatchFace((10001, 10003, 10004), np.array((
            (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        ))),
    )
    geometry = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 1.0e6,
        difference_step=1.0e-6,
        maximum_energy_evaluations=2*divisions**2*884,
        q4_source_area_tolerance=1.0e-10,
        q4_source_weighted_gap_tolerance=1.0e-12,
        maximum_q4_source_integration_triangles=300,
        differentiation="pair_local",
        maximum_candidate_pairs=2*divisions**2,
    )
    source = geometry.reference_source.copy()
    source[:, 2] = [-penetration + slope*sum(coordinates[index])
                    for index in geometry.source_node_ids]
    state = geometry.evaluate(source, geometry.reference_target)

    # Exact cellwise tensor-Q1 integrals over the square; this oracle does
    # not call the contact overlap or integration kernels.
    support = np.zeros(len(geometry.source_node_ids))
    weighted_gap = np.zeros_like(support)
    node_index = {index: local for local, index in
                  enumerate(geometry.source_node_ids)}
    local_side = ((0, 0), (1, 0), (1, 1), (0, 1))
    for face in source_faces:
        x0, y0 = coordinates[face.node_ids[0]]
        x1, y1 = coordinates[face.node_ids[2]]
        quarter_area = (x1 - x0)*(y1 - y0)/4.0
        for index, (right, upper) in zip(face.node_ids, local_side):
            x_mean = x0 + (2.0 if right else 1.0)*(x1 - x0)/3.0
            y_mean = y0 + (2.0 if upper else 1.0)*(y1 - y0)/3.0
            support[node_index[index]] += quarter_area
            weighted_gap[node_index[index]] += quarter_area*(
                -penetration + slope*(x_mean + y_mean)
            )

    np.testing.assert_allclose(state.reference_support, support,
                               rtol=1.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(state.reference_weighted_gap, weighted_gap,
                               rtol=1.0e-5, atol=1.0e-12)
    corner = node_index[node(0, 0)]
    assert state.normalized_gap[corner] == pytest.approx(
        weighted_gap[corner]/support[corner], abs=1.0e-9,
    )
    assert state.active_pressure_dofs == active_nodes
    assert state.minimum_physical_gap == pytest.approx(-penetration, abs=1.0e-9)
    assert state.overlap_area == pytest.approx(1.0, abs=5.0e-8)
