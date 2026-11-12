"""Bounded paired timing and shape check for prepared planar Q4 inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from anyfesolver.general_contact.overlap import (
    PlanarFace, _q4_shape, planar_projected_overlap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 1 <= args.repeats <= 8:
        parser.error("repeats must lie in [1, 8]")

    face = PlanarFace((1, 1), np.array((
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (1.75, 1.2, 0.0), (-0.1, 1.0, 0.0),
    )))
    overlap = planar_projected_overlap(face, face)
    if overlap is None:
        raise AssertionError("self-overlap is missing")
    natural = np.random.default_rng(20260929).uniform(-0.9, 0.9, (512, 2))
    points = np.asarray([_q4_shape(uv)[0] @ face.corners for uv in natural])
    expected = np.asarray([_q4_shape(uv)[0] for uv in natural])
    prepared = overlap.prepared_shape_values(face)
    evaluated = np.asarray([prepared(point) for point in points])
    maximum_error = float(np.max(np.abs(evaluated-expected)))
    if maximum_error > 2.0e-12:
        raise AssertionError("prepared shape inversion differs from Q4 reference")

    samples = []
    checksums = []
    for _ in range(args.repeats):
        start = perf_counter()
        checksum = 0.0
        for point in points:
            checksum += float(prepared(point)[0])
        samples.append(perf_counter()-start)
        checksums.append(checksum)
    if max(checksums)-min(checksums) > 1.0e-12:
        raise AssertionError("prepared inversion is not deterministic")
    result = {
        "case": "skew planar Q4, 512 fixed interior physical points",
        "repeats": args.repeats,
        "seconds": samples,
        "median_seconds": median(samples),
        "maximum_shape_error": maximum_error,
        "checksum": checksums[0],
    }
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "median_seconds", "maximum_shape_error", "checksum",
    )}))


if __name__ == "__main__":
    main()
