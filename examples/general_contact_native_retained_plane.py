"""Internal C3 development example: curved native beam against a fixed plane.

Run from an ANYfeSolver source checkout with ``python -m pip install -e .`` or
with a built wheel installed. This example is not a public C3 admission.
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
from anyfesolver.general_contact.native_retained_line_plane import (
    NativeRetainedRoundLinePlane,
)
from anyfesolver.general_contact.native_retained_static import (
    NativeRetainedRoundPlaneAnalysis,
)


def analysis() -> NativeRetainedRoundPlaneAnalysis:
    points = np.array(((0.0, 0.0, 0.0),
                       (1.0, 0.2, 0.0),
                       (2.0, 0.0, 0.0)))
    frames = []
    for slope in (0.4, 0.0, -0.4):
        tangent = np.array((1.0, slope, 0.0))
        tangent /= np.linalg.norm(tangent)
        normal = np.array((0.0, 0.0, 1.0))
        frames.append(np.column_stack((tangent, normal,
                                       np.cross(tangent, normal))))
    reference = CenteredCurvedBeam3ReferenceGeometry(points, np.array(frames))
    section = EllipsoidalGeneralizedSection(
        np.diag((1000.0, 400.0, 350.0, 0.8, 1.0, 1.2)),
        np.eye(6), 1e6, 1.0,
    )
    element = NativeGeneralizedStaticElement(
        1, (1, 2, 3), reference, section, order=4,
    )
    definition = NativeBeamDefinition.capture(
        element, np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11)),
    )
    root = BoundaryCondition(
        "clamped", [1],
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    )
    program = Program(
        (0.5, 1.0), DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(((3, 0.0, 0.0, -0.005),)),
    )
    contact = NativeRetainedRoundLinePlane(
        1, reference, radius_m=0.05,
        plane_origin_m=np.array((0.0, 0.0, -0.05)),
        plane_normal=np.array((0.0, 0.0, 1.0)),
        line_stiffness_n_per_m2=100.0,
        energy_tolerance_j=1e-10,
        force_tolerance_n=1e-9,
        tangent_tolerance_n_per_m=1e-8,
        maximum_intervals=64,
    )
    return NativeRetainedRoundPlaneAnalysis(
        (definition,), (root,), program, contact,
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
        raise RuntimeError(resumed.failure or "native contact restart mismatch")
    print(json.dumps({
        "profile": full.profile,
        "completed_targets": full.accepted.completed_targets,
        "contact_force_n": full.accepted.contact.beam_force_n.tolist(),
        "minimum_gap_m": full.accepted.contact.minimum_gap_m,
        "checkpoint_sha256": full.checkpoint_sha256,
        "split_restart_identical": True,
    }, indent=2))


if __name__ == "__main__":
    main()
