"""Match the bounded B2 mass screen to an independent Timoshenko Ritz mode."""

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
from numpy.polynomial.legendre import Legendre
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits

from scripts import probe_b3_ge_e4_pl_g0_f12_b2_dynamic_mass as previous


def _ritz(order):
    """Conforming polynomial w and theta fields with w(0)=theta(0)=0."""
    nq = 2*order+5
    gauss, weights = np.polynomial.legendre.leggauss(nq)
    s = (gauss+1)/2
    weights = weights/2
    basis = np.zeros((nq, order))
    derivative = np.zeros_like(basis)
    for i in range(order):
        polynomial = Legendre.basis(i)
        integral = polynomial.integ()
        basis[:, i] = .5*(integral(gauss)-integral(-1.))
        derivative[:, i] = polynomial(gauss)/previous.LENGTH
    l = previous.LENGTH
    area = previous.WIDTH*previous.HEIGHT
    iy = previous.WIDTH*previous.HEIGHT**3/12
    g = previous.ELASTIC_MODULUS/(2*(1+previous.POISSON))
    shear = (5/6)*g*area
    bend = previous.ELASTIC_MODULUS*iy
    rhoa, rhoi = previous.DENSITY*area, previous.DENSITY*iy
    weighted = weights[:, None]
    dd = l*derivative.T@(weighted*derivative)
    db = l*derivative.T@(weighted*basis)
    bb = l*basis.T@(weighted*basis)
    k = np.block([[shear*dd, shear*db],
                  [shear*db.T, bend*dd+shear*bb]])
    m = linalg.block_diag(rhoa*bb, rhoi*bb)
    values, vectors = linalg.eigh(k, m)
    if values[0] <= 0.:
        raise ArithmeticError("independent Timoshenko Ritz mode is not positive")
    equation = float(np.linalg.norm(k@vectors[:, 0]-values[0]*m@vectors[:, 0]) /
        max(1., np.linalg.norm(k@vectors[:, 0]),
            np.linalg.norm(values[0]*m@vectors[:, 0])))
    return {"order": order, "first_omega_rad_per_s": float(np.sqrt(values[0])),
            "minimum_mass_eigenvalue": float(np.linalg.eigvalsh(m)[0]),
            "scaled_eigen_equation_residual": equation}


def evaluate():
    started = monotonic()
    section, mass = previous._section(), previous._mass()
    with threadpool_limits(limits=1):
        reference = [_ritz(order) for order in (4, 6, 8, 10)]
        cases = [previous._case(count, section, mass) for count in previous.COUNTS]
        effective = threadpool_info()
    continuum = reference[-1]["first_omega_rad_per_s"]
    reference_delta = abs(reference[-2]["first_omega_rad_per_s"]/continuum-1.)
    errors = [abs(c["first_omega_rad_per_s"]/continuum-1.) for c in cases]
    passed = (reference_delta < 1e-9 and
              all(r["minimum_mass_eigenvalue"] > 0 and
                  r["scaled_eigen_equation_residual"] < 1e-7
                  for r in reference) and
              all(errors[i+1] < errors[i] for i in range(len(errors)-1)) and
              errors[-1] < 1e-5 and
              all(c["scaled_eigen_equation_residual"] < 1e-8 and
                  c["minimum_reduced_mass_eigenvalue"] > 0 for c in cases))
    sources = [Path(__file__), Path(previous.__file__),
        Path("src/anyfesolver/_b2_physical_flexibility_linear.py"),
        Path("src/anyfesolver/_b2_physical_linear_definition.py")]
    return {"classification": "bounded F12 matched-continuum dynamic interpolation screen, not B2 spectral admission",
            "question": "Does the static-trace B2 mass converge to a separately assembled continuum Timoshenko mode rather than the shear-free EB asymptote?",
            "ritz": reference, "ritz_order_8_to_10_relative_change": reference_delta,
            "b2_cases": cases, "b2_relative_errors_to_matched_reference": errors,
            "analytic_eb_omega_rad_per_s": previous.BETA_ONE**2*np.sqrt(
                previous.ELASTIC_MODULUS*(previous.WIDTH*previous.HEIGHT**3/12)/
                (previous.DENSITY*previous.WIDTH*previous.HEIGHT*previous.LENGTH**4)),
            "passed_bounded_screen": passed,
            "elapsed_s": monotonic()-started,
            "limits": ["same one slender cantilever and physical section; not a mesh or section-family qualification",
                       "Ritz w/theta polynomial energy and inertia independently assembled from continuum, not the B2 trace",
                       "no attached shell, prestress, buckling or production sparse eigensolver"],
            "provenance": {"source_git": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources},
                "python": sys.version.split()[0], "executable": sys.executable,
                "numpy": np.__version__, "scipy": scipy.__version__,
                "effective_threadpools": effective,
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
                      "ritz": [v["first_omega_rad_per_s"] for v in result["ritz"]],
                      "errors": result["b2_relative_errors_to_matched_reference"]}))
    if not result["passed_bounded_screen"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
