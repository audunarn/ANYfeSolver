"""Bounded B2 rigid kinetic identity with explicit offset section inertia.

Four-point trace integration is exact for the current cubic local pose
polynomials. This is not a production mass matrix or a dynamics admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import MassMetadata, PhysicalB2Member


LENGTH = 2.
WIDTH = .08
HEIGHT = .04
DENSITY = 7850.
MASS_ORIGIN = (0., .01, -.005)
RELATIVE_TOL = 1e-10


def _skew(a: np.ndarray) -> np.ndarray:
    x, y, z = a
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _section() -> LinearSection:
    area = WIDTH*HEIGHT
    iy = WIDTH*HEIGHT**3/12
    iz = HEIGHT*WIDTH**3/12
    # Standard bounded rectangle torsion approximation; rigid mass checks
    # depend only on geometric mass data, not this stiffness approximation.
    j = WIDTH*HEIGHT**3*(1/3-.21*HEIGHT/WIDTH*(1-HEIGHT**4/(12*WIDTH**4)))
    return LinearSection.isotropic(E=210e9, nu=.3, area=area,
                                   Iy=iy, Iz=iz, J=j, ky=5/6, kz=5/6)


def _mass(origin: tuple[float, float, float]) -> MassMetadata:
    area = WIDTH*HEIGHT
    iy = WIDTH*HEIGHT**3/12
    iz = HEIGHT*WIDTH**3/12
    matrix = np.diag((DENSITY*area,)*3 +
                     (DENSITY*(iy+iz), DENSITY*iy, DENSITY*iz))
    return MassMetadata("uniform-rectangular-density", origin,
                        tuple(tuple(float(v) for v in row) for row in matrix))


def _member(*, reversed_order: bool = False) -> PhysicalB2Member:
    a, b = (0., 0., 0.), (LENGTH, 0., 0.)
    if reversed_order:
        a, b = b, a
    # Reversing x with the same y flips local z. Transform the same physical
    # mass center into that new frame; density and principal inertias stay put.
    origin = (0., .01, .005) if reversed_order else MASS_ORIGIN
    return PhysicalB2Member("b2-rigid-mass", "rectangular-source",
                            (1, 0) if reversed_order else (0, 1),
                            (a, b), (0., 1., 0.), "rectangular-section",
                            _section(), mass=_mass(origin))


def _integrated_trace_mass(member: PhysicalB2Member) -> tuple[np.ndarray, np.ndarray]:
    abscissae, weights = np.polynomial.legendre.leggauss(4)
    stations = (abscissae+1)/2
    trace = member.trace_global(np.zeros(12), stations=stations,
                                offset_local_m=member.mass.origin_local_m)
    frame = member.local_frame_rows()
    six = np.zeros((6, 6))
    six[:3, :3] = six[3:, 3:] = frame.T
    section_mass = six@np.asarray(member.mass.matrix_per_m)@six.T
    result = member.length_m*sum(
        weight/2 * (jac.T@section_mass@jac)
        for weight, jac in zip(weights, trace.jacobian))
    return result, trace.jacobian


def _rigid_nodal_map(member: PhysicalB2Member) -> np.ndarray:
    origin = np.zeros(3)
    result = np.zeros((12, 6))
    for i, point in enumerate(member.reference_points_m):
        result[6*i:6*i+3, :3] = np.eye(3)
        result[6*i:6*i+3, 3:] = -_skew(np.asarray(point)-origin)
        result[6*i+3:6*i+6, 3:] = np.eye(3)
    return result


def _analytic_rigid_mass(member: PhysicalB2Member) -> np.ndarray:
    frame = member.local_frame_rows()
    length = member.length_m
    axis = frame[0]
    mass_per_m = float(member.mass.matrix_per_m[0][0])
    r = np.asarray(member.reference_points_m[0]) + .5*length*axis + \
        frame.T@np.asarray(member.mass.origin_local_m)
    principal = np.asarray(member.mass.matrix_per_m)[3:, 3:]
    result = np.zeros((6, 6))
    result[:3, :3] = mass_per_m*length*np.eye(3)
    result[:3, 3:] = -mass_per_m*length*_skew(r)
    result[3:, :3] = result[:3, 3:].T
    result[3:, 3:] = length*(frame.T@principal@frame) + mass_per_m*length*(
        _skew(r).T@_skew(r) + length**2/12*_skew(axis).T@_skew(axis))
    return result


def _error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))


def evaluate() -> dict:
    forward, reverse = _member(), _member(reversed_order=True)
    mass, _ = _integrated_trace_mass(forward)
    reverse_mass, _ = _integrated_trace_mass(reverse)
    rigid = _rigid_nodal_map(forward)
    calculated = rigid.T@mass@rigid
    analytic = _analytic_rigid_mass(forward)
    rigid_error = _error(calculated, analytic)
    swap = np.block([[np.zeros((6, 6)), np.eye(6)],
                     [np.eye(6), np.zeros((6, 6))]])
    reversal_error = _error(mass, swap.T@reverse_mass@swap)
    symmetry_error = _error(mass, mass.T)
    trace = forward.trace_global(np.zeros(12), stations=(0., .19, .5, .81, 1.),
                                 offset_local_m=forward.mass.origin_local_m)
    station_error = 0.
    for point, jac in zip(trace.reference_points_m, trace.jacobian):
        expected = np.zeros((6, 6))
        expected[:3, :3] = np.eye(3)
        expected[:3, 3:] = -_skew(point)
        expected[3:, 3:] = np.eye(3)
        station_error = max(station_error, float(np.max(np.abs(jac@rigid-expected))))
    eigen_min = float(np.min(np.linalg.eigvalsh(mass)))
    passed = (max(rigid_error, reversal_error, symmetry_error, station_error)
              <= RELATIVE_TOL and eigen_min >= -1e-11)
    source = Path(__file__)
    return {
        "classification": "F12_LOCAL_RIGID_MASS_DEV_NOT_ADMISSION",
        "question": "Does the physical B2 station trace with independently supplied offset mass data reproduce rigid kinetic work and member reversal?",
        "physical_data": {"length_m": LENGTH, "width_m": WIDTH, "height_m": HEIGHT,
                          "density_kg_m3": DENSITY,
                          "mass_origin_local_m": MASS_ORIGIN},
        "four_point_rigid_spatial_inertia_relative_error": rigid_error,
        "reversed_member_mass_matrix_relative_error": reversal_error,
        "mass_matrix_symmetry_relative_error": symmetry_error,
        "rigid_station_trace_max_absolute_error": station_error,
        "minimum_element_mass_eigenvalue": eigen_min,
        "threshold": RELATIVE_TOL,
        "passed_bounded_identity": passed,
        "limits": ["one local straight elastic B2 member and independent rigid-motion inertia only",
                   "cubic static trace used as an exploratory mass interpolation; deformational dynamics unqualified",
                   "no attached shell, prestress, buckling, spectral modes or production mass assembly"],
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "passed_bounded_identity", "four_point_rigid_spatial_inertia_relative_error",
        "reversed_member_mass_matrix_relative_error", "rigid_station_trace_max_absolute_error")}))
    if not result["passed_bounded_identity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
