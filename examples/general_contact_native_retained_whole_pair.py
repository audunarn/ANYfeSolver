"""Internal C3 example: active contact between two complete native B3-GE beams.

Install this checkout or set PYTHONPATH=src before running. The example uses
the restricted development profile and may take several minutes. It is not
an admitted public native contact capability or engineering qualification.
"""

from __future__ import annotations

import json
import argparse

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
from anyfesolver.general_contact.native_retained_pair_static import (
    NativeRetainedRoundPairAnalysis,
)
from anyfesolver.general_contact.native_retained_whole_pair import (
    NativeRetainedWholeCurvePair,
)


def analysis(*,physical_fibres=False,plastic_fibres=False,separation=0.13,force=-0.05,
             targets=(0.0,0.1,0.2,0.3),line_stiffness_n_per_m2=100.0,
             gap_tolerance_m=1e-3,gap_step_control=False,reference_rise_m=0.0,
             certify_curved_ownership=False,initial_trial=None,
             reuse_target_trials=False) -> NativeRetainedRoundPairAnalysis:
    if plastic_fibres and not physical_fibres:
        raise ValueError('plastic fibre pair requires the physical-fibre section family')
    if (isinstance(reference_rise_m, (bool, np.bool_))
            or not np.isfinite(reference_rise_m)):
        raise ValueError('reference rise must be a finite dimensional length')
    def reference(y: float) -> CenteredCurvedBeam3ReferenceGeometry:
        # Preserve the historical straight fixture and its restart identity.
        if reference_rise_m != 0.0:
            x = np.array((0.0, 1.0, 2.0))
            slope = 2.0*reference_rise_m*(1.0-x)
            frames = []
            for value in slope:
                tangent = np.array((1.0, value, 0.0))/np.hypot(1.0, value)
                normal = np.array((-tangent[1], tangent[0], 0.0))
                frames.append(np.column_stack((tangent, normal, (0.0, 0.0, 1.0))))
            return CenteredCurvedBeam3ReferenceGeometry(
                np.column_stack((x, y+reference_rise_m*(1.0-(x-1.0)**2), np.zeros(3))),
                np.array(frames),
            )
        return CenteredCurvedBeam3ReferenceGeometry(
            np.array(((0.0, y, 0.0), (1.0, y, 0.0), (2.0, y, 0.0))),
            np.tile(np.eye(3), (3, 1, 1)),
        )

    a, b = reference(0.0), reference(separation)
    if physical_fibres:
        from anyfesolver._ge_beam3_fibre_section import Fibre,FlowCurve,PhysicalFibreSection
        from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
        curve=FlowCurve.linear(2.,1e5) if plastic_fibres else FlowCurve((0.,),(1e6,),1.)
        fibres=tuple(Fibre(str(i),y,z,.00025,1e6,curve) for i,(y,z) in enumerate(
            ((.03,0.),(-.03,0.),(0.,.03),(0.,-.03))))
        section=PhysicalFibreSection(fibres,np.diag(np.sqrt((0.,400.,350.,.8,.55,.75))))
        element_type=NativeFibreStaticElement
    else:
        section = EllipsoidalGeneralizedSection(
            np.diag((1000.0, 400.0, 350.0, 0.8, 1.0, 1.2)),
            np.eye(6), 1e6, 1.0,
        )
        element_type=NativeGeneralizedStaticElement
    inertia = np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11))
    first = element_type(
        11, (1, 2, 3), a, section, order=4,
    )
    second = element_type(
        12, (4, 5, 6), b, section, order=4,
    )
    definitions = (NativeBeamDefinition.capture(first, inertia),
                   NativeBeamDefinition.capture(second, inertia))
    fixed = {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}
    boundaries = (BoundaryCondition("first-root", [1], fixed),
                  BoundaryCondition("second-root", [4], fixed))
    program = Program(
        targets,
        DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(((6, 0.0, force, 0.0),)),
        max_iterations=24,
    )
    contact = NativeRetainedWholeCurvePair(
        11, a, (1, 2, 3), 0.05,
        12, b, (4, 5, 6), 0.05,
        line_stiffness_n_per_m2=line_stiffness_n_per_m2,
        distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9,
        search_parameter_tolerance=1e-3,
        maximum_search_depth=20,
        maximum_search_cells=10000,
        energy_tolerance_j=1e-8,
        force_tolerance_n=1e-7,
        tangent_tolerance_n_per_m=1e-6,
        maximum_integration_intervals=1000,
        certify_open_intervals=physical_fibres,
        certify_curved_ownership=certify_curved_ownership,
    )
    return NativeRetainedRoundPairAnalysis(
        definitions, boundaries, program, contact,
        gap_tolerance_m=gap_tolerance_m, equilibrium_tolerance=1e-8,
        maximum_swept_intervals=128,
        gap_step_control=gap_step_control,
        initial_trial=initial_trial,
        reuse_target_trials=reuse_target_trials,
    )


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curved-fibre-pair',action='store_true',
        help='opt-in actual-curved plastic pair development candidate')
    args=parser.parse_args()
    gap_tolerance=5e-5 if args.curved_fibre_pair else 1e-3
    if args.curved_fibre_pair:
        resumed=analysis(physical_fibres=True,plastic_fibres=True,
            separation=.101,force=-.05,targets=(.05,),
            line_stiffness_n_per_m2=4000.,gap_tolerance_m=gap_tolerance,
            gap_step_control=True,reference_rise_m=.005,
            certify_curved_ownership=True).solve()
    else:
        prefix = analysis().solve(stop_after=3)
        if prefix.status != "paused":
            raise RuntimeError(prefix.failure)
        resumed = analysis().solve(
            checkpoint=prefix.checkpoint,
            expected_sha256=prefix.checkpoint_sha256,
        )
    if resumed.status != "completed":
        raise RuntimeError(resumed.failure)
    result = resumed.accepted.contact
    print(json.dumps({
        "profile": resumed.profile,
        "completed_targets": resumed.accepted.completed_targets,
        "potential_j": result.potential_j,
        "conservative_lower_gap_m": result.conservative_lower_gap_m,
        "declared_gap_tolerance_m": gap_tolerance,
        "curved_fibre_pair_development_candidate": args.curved_fibre_pair,
        "force_on_first_beam_n": result.beam_a_force_n.tolist(),
        "force_on_second_beam_n": result.beam_b_force_n.tolist(),
        "moment_on_first_beam_about_origin_nm": result.beam_a_moment_nm.tolist(),
        "moment_on_second_beam_about_origin_nm": result.beam_b_moment_nm.tolist(),
        "search_cells": result.search_cells,
        "integration_evaluations": result.integration_evaluations,
        "checkpoint_sha256": resumed.checkpoint_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
