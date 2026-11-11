"""Bounded C1 law witness with exactly unchanged shell-top positions.

The field is a supported E4 director pitch accompanied by midsurface
translations that cancel its top-surface displacement at every node. It
classifies the candidate's director-couple work; it does not qualify a weld.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from anyfesolver.corotational import rotation_matrix_from_vector
from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_distributed_surface_finite import FiniteSurfaceLineHybrid
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


def evaluate() -> dict:
    xs, ys = local_grid(8, transverse_cells=4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    patch = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    reference = patch.reference_nodes
    zero = np.zeros((patch.shell_count, 6))
    current = np.zeros_like(zero)
    pitch = 1e-4*(reference[:, 0]+.5)/1.5
    up = np.array((0., 0., 1.))
    frames = np.array([rotation_matrix_from_vector((0., angle, 0.))
                       for angle in pitch])
    current[:, 4] = pitch
    current[:, :3] = -patch.top*(frames@up-up)
    coordinates = reference+current[:, :3]
    top_reference = reference+patch.top*up
    top_current = coordinates+patch.top*(frames@up)
    top_nodal_difference = float(np.max(np.abs(top_current-top_reference)))
    beam_state = beam.context.initial.mechanical
    reference_gap = patch.evaluate_e4_total(zero.ravel(), beam_state)[:, 0]
    current_gap = patch.evaluate_e4_total(current.ravel(), beam_state)[:, 0]
    changed = current_gap-reference_gap
    top_sample_difference = float(max(np.max(np.abs(
        patch._shell(coordinates, frames, point)-
        patch._shell(reference, np.repeat(np.eye(3)[None], patch.shell_count, axis=0), point)))
        for point in patch.points))
    _, physical = raw_e4_tangent(xs, ys, thickness=.035)
    physical_energy = float(.5*current.ravel()@physical@current.ravel())
    root = reference[:, 0] <= -.5+1e-12
    root_max = float(np.max(np.abs(current[root])))
    position_max = float(np.max(np.abs(changed[[0, 1, 2, 3, 5]])))
    pitch_change = float(changed[4])
    if (top_nodal_difference > 1e-14 or top_sample_difference > 1e-14 or
            position_max > 1e-12 or abs(pitch_change) <= 1e-5 or
            physical_energy <= 0. or root_max != 0.):
        raise ArithmeticError("C1 compensated director witness does not meet guards")
    return {
        "classification": "bounded continuous-law classification, not a connection or multiplier qualification",
        "decision": "Does the C1 pitch channel act on a supported shell director field invisible to top-surface positional bond work?",
        "geometry": "C1 Q4 x8/y4, 35 mm plate, B3 retained 12-half-cell reference",
        "prescribed_field": "beta_y(x)=1e-4*(x+0.5)/1.5; midsurface translation exactly cancels top director offset",
        "maximum_shell_top_nodal_position_change_m": top_nodal_difference,
        "maximum_shell_top_quadrature_position_change_m": top_sample_difference,
        "continuous_gap_change": changed.tolist(),
        "maximum_non_pitch_gap_change": position_max,
        "pitch_gap_change_rad": pitch_change,
        "reference_physical_shell_quadratic_energy_j": physical_energy,
        "fixed_root_max_coordinate_change": root_max,
        "conclusion": "The pitch row is an independent director-couple law beyond a bond defined only by shell-top/beam-surface positions. Whether that rotational weld idealization is intended and accurate requires physical work evidence; no multiplier selection follows.",
        "limits": ["one kinematic supported field at a flat reference, not a loaded solution",
                   "quadratic E4 reference energy shows physical stiffness only, not finite-motion energy",
                   "a true weld may intentionally transmit an independent director couple"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"top_change": result["maximum_shell_top_quadrature_position_change_m"],
                      "nonpitch_gap": result["maximum_non_pitch_gap_change"],
                      "pitch_gap": result["pitch_gap_change_rad"],
                      "physical_energy": result["reference_physical_shell_quadratic_energy_j"]}))


if __name__ == "__main__":
    main()
