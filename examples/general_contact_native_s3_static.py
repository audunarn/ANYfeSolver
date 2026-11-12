"""Internal C3 development example: retained native beam on elastic S3 skin.

Run with ANYfeSolver installed or from an editable checkout. This restricted
analysis is not a public native beam–shell capability admission.
"""

from __future__ import annotations

import json

import numpy as np

from anyfesolver import create_shell_element
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
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.native_retained_line_s3 import (
    NativeRetainedRoundLineS3Interior,
)
from anyfesolver.general_contact.native_retained_line_s3_work import (
    NativeRetainedLineS3Work,
)
from anyfesolver.general_contact.native_retained_s3_static import (
    NativeRetainedRoundS3Analysis,
)
from anyfesolver.general_contact.overlap import reference_shell_face
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter


def analysis() -> NativeRetainedRoundS3Analysis:
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
        NodalDeadForces(((3, 0.0, 0.0, -0.01),)),
    )

    shell = FEModel("native_shell_contact")
    shell.add_material("elastic", 1e5, 0.3, density=1.0)
    for node, (x, y) in zip((4, 5, 6),
                            ((-5.0, -5.0), (5.0, -5.0), (0.0, 10.0))):
        shell.add_node(node, x, y, -0.076)
    shell.add_element(21, create_shell_element(
        21, (4, 5, 6), "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.0,
    ))
    shell.add_boundary_condition(FixedSupport("shell_edge", [4, 6]))
    shell.add_boundary_condition(BoundaryCondition(
        "shell_vertex", [5],
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    skin = S3SkinWorkAdapter.from_model(shell, 7, 21, "top")
    kernel = NativeRetainedRoundLineS3Interior(
        1, reference, (1, 2, 3),
        21, reference_shell_face(shell, 7, 21, "top"), (4, 5, 6),
        radius_m=0.05, line_stiffness_n_per_m2=1000.0,
        rim_clearance_m=0.1, geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-10, force_tolerance_n=1e-9,
        tangent_tolerance_n_per_m=1e-7,
        maximum_integration_intervals=100,
    )
    return NativeRetainedRoundS3Analysis(
        (definition,), (root,), program, shell,
        NativeRetainedLineS3Work(kernel, skin),
        gap_tolerance_m=0.001, equilibrium_tolerance=1e-8,
        maximum_swept_intervals=128,
    )


def main() -> None:
    owner = analysis()
    run = owner.solve()
    if run.status != "completed":
        raise RuntimeError(run.failure)
    prefix = analysis().solve(stop_after=1)
    if prefix.status != "paused":
        raise RuntimeError(prefix.failure)
    resumed = analysis().solve(
        checkpoint=prefix.checkpoint,
        expected_sha256=prefix.checkpoint_sha256,
    )
    if resumed.status != "completed" or resumed.checkpoint != run.checkpoint:
        raise RuntimeError(resumed.failure or "native/S3 restart mismatch")
    accepted = run.accepted
    contact = accepted.contact.contact
    print(json.dumps({
        "profile": run.profile,
        "status": run.status,
        "completed_targets": accepted.completed_targets,
        "contact_force_z_n": float(contact.beam_force_n[2]),
        "shell_displacement_norm_m": float(np.linalg.norm(
            accepted.shell_displacement)),
        "native_internal_contact_work_norm": float(np.linalg.norm(
            contact.residual[18:24])),
        "minimum_gap_m": float(contact.minimum_gap_m),
        "active_intervals": contact.active_intervals,
        "checkpoint_sha256": run.checkpoint_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
