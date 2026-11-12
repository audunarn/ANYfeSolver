"""Bounded D1 face-pair preparation and candidate-cost screen."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


def faces(source_cells: int, target_cells: int):
    source = []
    target = []
    source_stride = source_cells + 1
    target_stride = target_cells + 1
    target_start = source_stride**2 + 1
    for j in range(source_cells):
        for i in range(source_cells):
            base = 1 + j*source_stride + i
            nodes = (base, base+1, base+source_stride+1,
                     base+source_stride)
            corners = np.asarray([
                (0.1+x/source_cells, 0.1+y/source_cells, 0.01504)
                for x, y in ((i, j), (i+1, j), (i+1, j+1), (i, j+1))
            ])
            source.append(S3PatchFace(nodes, corners))
    for j in range(target_cells):
        for i in range(target_cells):
            base = target_start + j*target_stride + i
            corners = {
                base: (i/target_cells, j/target_cells, 0.01),
                base+1: ((i+1)/target_cells, j/target_cells, 0.01),
                base+target_stride: (i/target_cells, (j+1)/target_cells, 0.01),
                base+target_stride+1:
                    ((i+1)/target_cells, (j+1)/target_cells, 0.01),
            }
            for nodes in ((base, base+1, base+target_stride),
                          (base+1, base+target_stride+1, base+target_stride)):
                target.append(S3PatchFace(nodes,
                                           np.asarray([corners[n] for n in nodes])))
    return tuple(source), tuple(target)


def main(source_cells: int, target_cells: int, output: Path,
         pair_broadphase: bool):
    if source_cells not in (4, 8, 12, 16) or target_cells not in (2, 3, 4, 6):
        raise ValueError("preflight mesh size is outside the bounded set")
    if output.exists():
        raise FileExistsError(output)
    source_faces, target_faces = faces(source_cells, target_cells)
    started = time.perf_counter()
    geometry = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 10000.0,
        difference_step=2e-5,
        maximum_energy_evaluations=14144,
        maximum_candidate_pairs=len(source_faces)*len(target_faces),
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
        q4_source_derivative_tolerances=(1e-5,)*4,
        differentiation="jet2_q4_s3",
        pair_broadphase=pair_broadphase,
    )
    preparation_s = time.perf_counter()-started
    started = time.perf_counter()
    state = geometry.evaluate(geometry.reference_source,
                              geometry.reference_target)
    evaluation_s = time.perf_counter()-started
    record = {
        "source_cells": source_cells,
        "target_cells": target_cells,
        "cartesian_pairs": len(geometry.pair_laws),
        "overlapping_pairs": state.overlapping_pairs,
        "evaluated_pairs": sum(pair is not None for pair in state.pair_states),
        "preparation_seconds": preparation_s,
        "evaluation_seconds": evaluation_s,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]),
         len(sys.argv) < 5 or sys.argv[4] != "--exhaustive")
