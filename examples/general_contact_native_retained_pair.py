"""Internal C3 development example: two coupled native beams in contact.

Install this checkout or set PYTHONPATH=src before running. This is not a
public native beam-pair capability or an engineering qualification example.
"""

from __future__ import annotations

import json

import numpy as np

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_generalized_ellipsoid_section import (
    EllipsoidalGeneralizedSection,
)
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import (
    NativeGeneralizedStaticElement,
)
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces, Program
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.general_contact.native_retained_cell_pair import (
    NativeRetainedSelectedCellPair,
)
from anyfesolver.general_contact.native_retained_pair_static import (
    NativeRetainedRoundPairAnalysis,
)


def analysis() -> NativeRetainedRoundPairAnalysis:
    first_points = np.array(((0.0, 0.0, 0.0),
                             (1.0, 0.1, 0.0),
                             (2.0, 0.0, 0.0)))
    first_frames = []
    for slope in (0.2, 0.0, -0.2):
        tangent = np.array((1.0, slope, 0.0))
        tangent /= np.linalg.norm(tangent)
        normal = np.array((0.0, 0.0, 1.0))
        first_frames.append(np.column_stack((tangent, normal,
                                             np.cross(tangent, normal))))
    second_points = np.array(((0.5, -1.5, 0.13),
                              (0.5, -0.5, 0.13),
                              (0.5, 0.5, 0.13)))
    second_tangent = np.array((0.0, 1.0, 0.0))
    second_normal = np.array((0.0, 0.0, 1.0))
    second_frame = np.column_stack((second_tangent, second_normal,
                                    np.cross(second_tangent, second_normal)))
    first_ref = CenteredCurvedBeam3ReferenceGeometry(
        first_points, np.array(first_frames),
    )
    second_ref = CenteredCurvedBeam3ReferenceGeometry(
        second_points, np.tile(second_frame, (3, 1, 1)),
    )
    section = EllipsoidalGeneralizedSection(
        np.diag((1000.0, 400.0, 350.0, 0.8, 1.0, 1.2)),
        np.eye(6), 1e6, 1.0,
    )
    inertia = np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11))
    first = NativeGeneralizedStaticElement(
        11, (1, 2, 3), first_ref, section, order=4,
    )
    second = NativeGeneralizedStaticElement(
        12, (4, 5, 6), second_ref, section, order=4,
    )
    definitions = (NativeBeamDefinition.capture(first, inertia),
                   NativeBeamDefinition.capture(second, inertia))
    fixed = {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}
    boundaries = (BoundaryCondition("first-root", [1], fixed),
                  BoundaryCondition("second-root", [4], fixed))
    program = Program(
        (0.5, 1.0), DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(((6, 0.0, 0.0, -0.02),)),
    )
    contact = NativeRetainedSelectedCellPair(
        11, first_ref, 0, 0.05, 12, second_ref, 1, 0.05,
        line_stiffness_n_per_m2=1000.0,
        integration_relative_tolerance=1e-5,
        maximum_subdivisions=10,
        gap_accuracy_m=1e-4,
        maximum_gap_cells=10000,
        maximum_gap_depth=28,
    )
    return NativeRetainedRoundPairAnalysis(
        definitions, boundaries, program, contact,
        gap_tolerance_m=1e-3, equilibrium_tolerance=1e-8,
        maximum_swept_intervals=128,
    )


def main() -> None:
    full = analysis().solve()
    if full.status != "completed":
        raise RuntimeError(full.failure)
    prefix = analysis().solve(stop_after=1)
    if prefix.status != "paused":
        raise RuntimeError(prefix.failure)
    resumed = analysis().solve(
        checkpoint=prefix.checkpoint,
        expected_sha256=prefix.checkpoint_sha256,
    )
    if resumed.status != "completed" or resumed.checkpoint != full.checkpoint:
        raise RuntimeError(resumed.failure or "native pair restart mismatch")
    print(json.dumps({
        "profile": full.profile,
        "completed_targets": full.accepted.completed_targets,
        "contact_force_on_first_beam_n": (
            full.accepted.contact.beam_a_force_n.tolist()
        ),
        "conservative_lower_gap_m": (
            full.accepted.contact.conservative_lower_gap_m
        ),
        "checkpoint_sha256": full.checkpoint_sha256,
        "split_restart_identical": True,
    }, indent=2))


if __name__ == "__main__":
    main()
