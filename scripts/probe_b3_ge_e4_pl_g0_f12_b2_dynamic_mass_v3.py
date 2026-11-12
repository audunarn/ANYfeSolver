"""High-precision independent Timoshenko Ritz oracle for bounded B2 mass."""

from __future__ import annotations

import argparse
from decimal import Decimal as D, localcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from scripts import probe_b3_ge_e4_pl_g0_f12_b2_dynamic_mass as previous


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def _matvec(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _factor(matrix):
    size = len(matrix)
    lower = [[D(int(i == j)) for j in range(size)] for i in range(size)]
    diagonal = []
    for j in range(size):
        pivot = matrix[j][j]-sum((lower[j][k]**2*diagonal[k]
                                   for k in range(j)), D(0))
        if pivot <= 0:
            raise ArithmeticError("high-precision Ritz stiffness is not SPD")
        diagonal.append(pivot)
        for i in range(j+1, size):
            lower[i][j] = (matrix[i][j]-sum((
                lower[i][k]*lower[j][k]*diagonal[k]
                for k in range(j)), D(0)))/pivot
    return lower, diagonal


def _solve(lower, diagonal, rhs):
    size = len(rhs)
    y = []
    for i in range(size):
        y.append(rhs[i]-sum((lower[i][k]*y[k]
                              for k in range(i)), D(0)))
    z = [y[i]/diagonal[i] for i in range(size)]
    x = [D(0)]*size
    for i in range(size-1, -1, -1):
        x[i] = z[i]-sum((lower[k][i]*x[k]
                         for k in range(i+1, size)), D(0))
    return x


def _ritz(order):
    """Independent monomial w/theta continuum energies at 60-digit precision."""
    length = D(str(previous.LENGTH))
    width, height = D(str(previous.WIDTH)), D(str(previous.HEIGHT))
    young, poisson = D(str(previous.ELASTIC_MODULUS)), D(str(previous.POISSON))
    density = D(str(previous.DENSITY))
    area = width*height
    iy = width*height**3/D(12)
    shear = D(5)/D(6)*young/(D(2)*(D(1)+poisson))*area
    bend = young*iy
    rhoa, rhoi = density*area, density*iy
    size = 2*order
    k = [[D(0) for _ in range(size)] for _ in range(size)]
    m = [[D(0) for _ in range(size)] for _ in range(size)]
    for row in range(order):
        i = D(row+1)
        for col in range(order):
            j = D(col+1)
            dd = i*j/(length*(i+j-D(1)))
            db = i/(i+j)
            bb = length/(i+j+D(1))
            k[row][col] = shear*dd
            k[row][order+col] = shear*db
            k[order+col][row] = shear*db
            k[order+row][order+col] = bend*dd+shear*bb
            m[row][col] = rhoa*bb
            m[order+row][order+col] = rhoi*bb
    lower, diagonal = _factor(k)
    x = [D(int(i == 0)) for i in range(size)]
    prior = None
    iterations = 0
    for iterations in range(1, 81):
        z = _solve(lower, diagonal, _matvec(m, x))
        scale = max(abs(v) for v in z)
        if scale == 0:
            raise ArithmeticError("high-precision Ritz inverse iteration vanished")
        x = [v/scale for v in z]
        mx, kx = _matvec(m, x), _matvec(k, x)
        eigenvalue = _dot(x, kx)/_dot(x, mx)
        if prior is not None and abs(eigenvalue-prior)/eigenvalue < D("1e-45"):
            break
        prior = eigenvalue
    else:
        raise ArithmeticError("high-precision Ritz inverse iteration did not converge")
    residual = (max(abs(a-eigenvalue*b) for a, b in zip(kx, mx)) /
                max(D(1), max(abs(a) for a in kx)))
    return {"order": order, "first_omega_rad_per_s": float(eigenvalue.sqrt()),
            "eigenvalue_decimal": str(eigenvalue),
            "scaled_equation_residual_decimal": str(residual),
            "iterations": iterations}


def evaluate():
    started = monotonic()
    with localcontext() as context:
        context.prec = 60
        references = [_ritz(order) for order in (4, 6, 8, 10, 12)]
    section, mass = previous._section(), previous._mass()
    with threadpool_limits(limits=1):
        cases = [previous._case(count, section, mass) for count in previous.COUNTS]
        effective = threadpool_info()
    root = references[-1]["first_omega_rad_per_s"]
    ref_change = abs(references[-2]["first_omega_rad_per_s"]/root-1.)
    errors = [abs(c["first_omega_rad_per_s"]/root-1.) for c in cases]
    passed = (ref_change < 1e-9 and
              all(references[i+1]["first_omega_rad_per_s"] <=
                  references[i]["first_omega_rad_per_s"]+1e-11
                  for i in range(len(references)-1)) and
              all(errors[i+1] < errors[i] for i in range(len(errors)-1)) and
              errors[-1] < 1e-5 and
              all(c["scaled_eigen_equation_residual"] < 1e-8 and
                  c["minimum_reduced_mass_eigenvalue"] > 0 for c in cases))
    sources = [Path(__file__), Path(previous.__file__),
        Path("src/anyfesolver/_b2_physical_flexibility_linear.py"),
        Path("src/anyfesolver/_b2_physical_linear_definition.py")]
    return {"classification": "bounded F12 high-precision continuum comparison, not spectral admission",
            "question": "Does the B2 static-trace mass converge to a separately derived Timoshenko continuum frequency?",
            "reference_method": "60-digit Decimal conforming monomial w/theta Ritz energies and inverse iteration",
            "references": references,
            "order_10_to_12_relative_change": ref_change,
            "b2_cases": cases, "b2_relative_errors_to_matched_reference": errors,
            "passed_bounded_screen": passed, "elapsed_s": monotonic()-started,
            "limits": ["one slender homogeneous cantilever with centered mass",
                       "no attached shell, buckling, prestress or production eigensolver",
                       "Ritz and B2 share physical Timoshenko assumptions but not shape functions or mass assembly"],
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
                      "ritz": [v["first_omega_rad_per_s"] for v in result["references"]],
                      "errors": result["b2_relative_errors_to_matched_reference"],
                      "elapsed_s": result["elapsed_s"]}))
    if not result["passed_bounded_screen"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
