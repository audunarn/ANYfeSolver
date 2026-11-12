"""Bounded physical B2 eccentric-section mass metadata check; no spectral admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import MassMetadata, PhysicalB2Member


def _skew(x):
    return np.array(((0., -x[2], x[1]), (x[2], 0., -x[0]),
                     (-x[1], x[0], 0.)))


def _member(end, local_y, mass):
    section = LinearSection.isotropic(E=210e9, nu=.3, area=.01,
                                     Iy=1e-5, Iz=1e-5, J=2e-5,
                                     ky=.8, kz=.8)
    return PhysicalB2Member("eccentric-mass", "physical-cloud", (1, 2),
                            ((0., 0., 0.), tuple(end)), tuple(local_y),
                            "unrelated-stiffness", section, mass=mass)


def _analytic(mass_per_m, center, centroidal_inertia):
    cross = _skew(center)
    transform = np.block([[np.eye(3), -cross],
                          [np.zeros((3, 3)), np.eye(3)]])
    centroid = np.zeros((6, 6))
    centroid[:3, :3] = mass_per_m*np.eye(3)
    centroid[3:, 3:] = centroidal_inertia
    return transform.T@centroid@transform


def evaluate():
    # Four equal cross-section point masses per metre form an independent
    # kinetic-energy oracle. Their x coordinates are zero in the base frame.
    mass_per_m = 8.
    center_global = np.array((0., .25, -.125))
    offsets = np.array([(0., y, z) for y in (-.3, .3)
                        for z in (-.4, .4)])
    points_global = center_global+offsets
    particle_mass = mass_per_m/len(points_global)
    direct_global = np.zeros((6, 6))
    for point in points_global:
        velocity_map = np.column_stack((np.eye(3), -_skew(point)))
        direct_global += particle_mass*velocity_map.T@velocity_map
    section_frames = (
        ((1., 0., 0.), (0., 1., 0.)),
        ((1., 0., 0.), (0., 2.**-.5, 2.**-.5)),
        ((-1., 0., 0.), (0., 1., 0.)),
    )
    probes = []
    for end, local_y in section_frames:
        provisional = _member(end, local_y, None)
        frame = provisional.local_frame_rows()
        center = frame@center_global
        local_offsets = offsets@frame.T
        centroidal = np.zeros((3, 3))
        for point in local_offsets:
            centroidal += particle_mass*(_skew(point).T@_skew(point))
        analytic = _analytic(mass_per_m, center, centroidal)
        metadata = MassMetadata("four-physical-point-masses", (0., 0., 0.),
                                tuple(map(tuple, analytic)))
        member = _member(end, local_y, metadata)
        transform = member.local_dof_map()[:6, :6]
        global_mass = transform.T@np.asarray(metadata.matrix_per_m)@transform
        test_velocity = np.array((.13, -.21, .31, -.17, .29, .11))
        direct_energy = sum(
            .5*particle_mass*np.linalg.norm(
                test_velocity[:3]+np.cross(test_velocity[3:], point))**2
            for point in points_global)
        matrix_energy = .5*test_velocity@global_mass@test_velocity
        probes.append({
            "frame_rows": frame.tolist(),
            "mass_center_local_m": center.tolist(),
            "minimum_mass_eigenvalue": float(np.linalg.eigvalsh(analytic)[0]),
            "relative_global_matrix_error": float(
                np.linalg.norm(global_mass-direct_global)/np.linalg.norm(direct_global)),
            "relative_kinetic_energy_error": float(
                abs(matrix_energy-direct_energy)/direct_energy),
            "translation_rotation_coupling_norm": float(
                np.linalg.norm(analytic[:3, 3:])),
        })
    passed = all(p["minimum_mass_eigenvalue"] > 0 and
                 p["relative_global_matrix_error"] < 2e-15 and
                 p["relative_kinetic_energy_error"] < 2e-15 and
                 p["translation_rotation_coupling_norm"] > 0
                 for p in probes)
    source = Path(__file__)
    definition = Path("src/anyfesolver/_b2_physical_linear_definition.py")
    return {
        "classification": "bounded F12 physical metadata feasibility; not mass or spectral admission",
        "question": "Can eccentric-section inertia at the beam reference axis retain direct particle kinetic work under frame rotation and reversal?",
        "passed_bounded_screen": passed,
        "mass_per_m": mass_per_m,
        "points_global_m": points_global.tolist(),
        "probes": probes,
        "limits": ["four-point physical cross section and one station",
                   "metadata origin is the beam reference axis in every probe",
                   "no nonzero-origin metadata transformation, coupled shell, prestress or eigensolver"],
        "provenance": {
            "source_git": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  text=True).strip(),
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (source, definition)},
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__,
            "env": {key: os.environ.get(key) for key in
                    ("PYTHONPATH", "NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS")},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": result["passed_bounded_screen"],
                      "maximum_matrix_error": max(
                          p["relative_global_matrix_error"] for p in result["probes"]),
                      "maximum_energy_error": max(
                          p["relative_kinetic_energy_error"] for p in result["probes"])}))
    if not result["passed_bounded_screen"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
