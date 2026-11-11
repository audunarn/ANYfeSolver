"""Bounded B2 static-trace mass interpolation against a slender beam mode.

This tests a candidate dynamic interpolation; it is not a production B2 mass
assembler or a qualification of coupled modes or buckling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
import scipy
from scipy import linalg, sparse
from threadpoolctl import threadpool_info, threadpool_limits

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import MassMetadata, PhysicalB2Member


LENGTH = 2.0
WIDTH = 0.04
HEIGHT = 0.01
ELASTIC_MODULUS = 210e9
POISSON = 0.3
DENSITY = 7850.0
BETA_ONE = 1.875104068711961
COUNTS = (1, 2, 4, 8)


def _section():
    area = WIDTH*HEIGHT
    iy = WIDTH*HEIGHT**3/12
    iz = HEIGHT*WIDTH**3/12
    aspect = HEIGHT/WIDTH
    j = WIDTH*HEIGHT**3*(1/3-.21*aspect*(1-HEIGHT**4/(12*WIDTH**4)))
    return LinearSection.isotropic(E=ELASTIC_MODULUS, nu=POISSON,
        area=area, Iy=iy, Iz=iz, J=j, ky=5/6, kz=5/6)


def _mass():
    area = WIDTH*HEIGHT
    iy = WIDTH*HEIGHT**3/12
    iz = HEIGHT*WIDTH**3/12
    data = np.diag((DENSITY*area,)*3 +
                   (DENSITY*(iy+iz), DENSITY*iy, DENSITY*iz))
    return MassMetadata("uniform-rectangular-density", (0., 0., 0.),
                        tuple(tuple(float(v) for v in row) for row in data))


def _analytic_mode(x):
    t = BETA_ONE*x/LENGTH
    ratio = (np.cosh(BETA_ONE)+np.cos(BETA_ONE))/(
        np.sinh(BETA_ONE)+np.sin(BETA_ONE))
    y = np.cosh(t)-np.cos(t)-ratio*(np.sinh(t)-np.sin(t))
    end = np.cosh(BETA_ONE)-np.cos(BETA_ONE)-ratio*(
        np.sinh(BETA_ONE)-np.sin(BETA_ONE))
    return y/end


def _element(length, section, mass):
    member = PhysicalB2Member("bounded-b2-mode", "physical-rectangle",
        (0, 1), ((0., 0., 0.), (length, 0., 0.)), (0., 1., 0.),
        "rectangular", section, mass=mass)
    k = member.local_operator().evaluate(np.zeros(12)).tangent
    abscissae, weights = np.polynomial.legendre.leggauss(4)
    stations = (abscissae+1)/2
    trace = member.trace_global(np.zeros(12), stations=stations)
    section_mass = np.asarray(mass.matrix_per_m)
    m = length*sum(weight/2 * (jac.T@section_mass@jac)
                   for weight, jac in zip(weights, trace.jacobian))
    return k, m


def _case(count, section, mass):
    h = LENGTH/count
    ke, me = _element(h, section, mass)
    size = 6*(count+1)
    k = sparse.lil_matrix((size, size), dtype=float)
    m = sparse.lil_matrix((size, size), dtype=float)
    for element in range(count):
        rows = np.r_[np.arange(6*element, 6*element+6),
                     np.arange(6*(element+1), 6*(element+1)+6)]
        k[np.ix_(rows, rows)] += ke
        m[np.ix_(rows, rows)] += me
    k, m = k.tocsr(), m.tocsr()
    free = np.arange(6, size)
    kr = k[free, :][:, free].toarray()
    mr = m[free, :][:, free].toarray()
    mass_min = float(np.linalg.eigvalsh(mr)[0])
    if mass_min <= 0.:
        raise ArithmeticError("candidate B2 dynamic mass is not positive definite")
    values, vectors = linalg.eigh(kr, mr)
    if values[0] <= 0.:
        raise ArithmeticError("supported B2 first mode is not positive")
    mode = np.zeros(size)
    mode[free] = vectors[:, 0]
    if mode[-4] < 0.:  # terminal z translation is node[-1] component 2
        mode = -mode
    z = mode.reshape(count+1, 6)[:, 2]
    z /= z[-1]
    samples = _analytic_mode(np.linspace(0., LENGTH, count+1))
    equation = float(np.linalg.norm(kr@vectors[:, 0]-values[0]*mr@vectors[:, 0]) /
        max(1., np.linalg.norm(kr@vectors[:, 0]),
            np.linalg.norm(values[0]*mr@vectors[:, 0])))
    return {"elements": count, "first_omega_rad_per_s": float(np.sqrt(values[0])),
            "minimum_reduced_mass_eigenvalue": mass_min,
            "mass_symmetry_relative": float(np.linalg.norm(mr-mr.T)/
                                             max(1., np.linalg.norm(mr))),
            "stiffness_symmetry_relative": float(np.linalg.norm(kr-kr.T)/
                                                  max(1., np.linalg.norm(kr))),
            "scaled_eigen_equation_residual": equation,
            "nodal_z_mode_error": float(np.max(np.abs(z-samples))),
            "second_omega_rad_per_s": float(np.sqrt(values[1]))}


def evaluate():
    started = monotonic()
    section, mass = _section(), _mass()
    iy = WIDTH*HEIGHT**3/12
    reference = BETA_ONE**2*np.sqrt(ELASTIC_MODULUS*iy /
                                   (DENSITY*WIDTH*HEIGHT*LENGTH**4))
    cases = []
    with threadpool_limits(limits=1):
        for count in COUNTS:
            if monotonic()-started > 120.:
                raise TimeoutError("bounded B2 dynamic interpolation deadline")
            cases.append(_case(count, section, mass))
        effective_pools = threadpool_info()
    errors = [abs(c["first_omega_rad_per_s"]/reference-1.) for c in cases]
    passed = (all(np.isfinite(errors)) and
              errors[-1] < 1e-3 and
              all(errors[i+1] < errors[i] for i in range(len(errors)-1)) and
              all(c["scaled_eigen_equation_residual"] < 1e-8 and
                  c["mass_symmetry_relative"] < 1e-12 and
                  c["stiffness_symmetry_relative"] < 1e-12 for c in cases) and
              cases[-1]["nodal_z_mode_error"] < 1e-3)
    sources = [Path(__file__),
        Path("src/anyfesolver/_b2_physical_flexibility_linear.py"),
        Path("src/anyfesolver/_b2_physical_linear_definition.py")]
    return {"classification": "bounded F12 dynamic interpolation screen, not spectral admission",
            "question": "Does the physical B2 static station trace yield a convergent deformational mass interpolation on a slender cantilever?",
            "geometry": {"length_m": LENGTH, "width_m": WIDTH,
                         "height_m": HEIGHT, "density_kg_m3": DENSITY},
            "analytic_eb_first_omega_rad_per_s": float(reference),
            "relative_errors": errors, "cases": cases,
            "passed_bounded_screen": passed,
            "limits": ["one centered uniform rectangle, slender Euler-Bernoulli asymptote",
                       "no attached shell, off-axis mass, buckling, prestress or production eigensolver",
                       "static trace remains a candidate dynamic interpolation until broader qualification"],
            "elapsed_s": monotonic()-started,
            "provenance": {"source_git": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources},
                "python": sys.version.split()[0], "executable": sys.executable,
                "numpy": np.__version__, "scipy": scipy.__version__,
                "effective_threadpools": effective_pools,
                "env": {key: os.environ.get(key) for key in
                        ("PYTHONPATH", "NUMBA_DISABLE_JIT",
                         "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": result["passed_bounded_screen"],
                      "errors": result["relative_errors"],
                      "elapsed_s": result["elapsed_s"]}))
    if not result["passed_bounded_screen"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
