"""Private F09 feasibility of an exact local energy-rounding admission rule.

Decision: can a bounded, unit-aware check replace the provisional normalized
pivot ratio for the *local* six-basic-mode stiffness?  For exact rational A
and its binary64 rounding A_f, require exact positive definiteness of
epsilon*A +/- (A_f-A).  This proves, for every basic displacement b,
|b.T (A_f-A) b| < epsilon*b.T A b.  Use epsilon=1e-8, consistent with
the existing preparatory physical-work criterion.  Compare SI and numerical
millimetre descriptions, and retain explicit overflow/underflow negatives.

This only examines the private basic operator.  It is neither a public
section domain nor a graph/state solve rule and cannot qualify F09/G0B.
"""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from anyfesolver._b2_physical_flexibility_linear import (
    ConditioningAdmissionError, LinearOperator, LinearSection,
    RepresentabilityError,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (ROOT / "docs" / "agent_plans" /
          "B3_GE_E4_PL_G0_F09_LOCAL_ENERGY_ADMISSION_DEV_V1.json")
SOURCE = ROOT / "src" / "anyfesolver" / "_b2_physical_flexibility_linear.py"
EPSILON = F(1, 10**8)
UNIT = F(1000)
SECONDS_LIMIT = 120.
RSS_LIMIT = 2 * 1024**3


def _budget(start):
    if time.monotonic() - start > SECONDS_LIMIT:
        raise RuntimeError("120 s F09 diagnostic bound exceeded")
    if psutil.Process().memory_info().rss > RSS_LIMIT:
        raise RuntimeError("2 GiB F09 diagnostic bound exceeded")


def _exact_spd(matrix):
    """Exact LDL positivity, without the product's provisional pivot cut."""
    n = len(matrix)
    lower = [[F(int(i == j)) for j in range(n)] for i in range(n)]
    pivots = []
    for j in range(n):
        pivot = matrix[j][j] - sum(
            (lower[j][k]**2 * pivots[k] for k in range(j)), F(0))
        if pivot <= 0:
            return False
        pivots.append(pivot)
        for i in range(j+1, n):
            lower[i][j] = (matrix[i][j] - sum(
                (lower[i][k]*pivots[k]*lower[j][k] for k in range(j)),
                F(0))) / pivot
    return True


def _basic_matrix(operator):
    columns = [operator._basic_action([F(int(i == j)) for i in range(6)])
               for j in range(6)]
    result = [[columns[j][i] for j in range(6)] for i in range(6)]
    assert all(result[i][j] == result[j][i]
               for i in range(6) for j in range(6))
    return result


def _energy_guard(exact, rounded, epsilon):
    if not _exact_spd(exact):
        return False
    delta = [[F(float(rounded[i][j]))-exact[i][j]
              for j in range(6)] for i in range(6)]
    return all(_exact_spd([
        [epsilon*exact[i][j] + sign*delta[i][j] for j in range(6)]
        for i in range(6)]) for sign in (1, -1))


def _case(name, length, section, start):
    _budget(start)
    try:
        operator = LinearOperator(length, section)
        exact = _basic_matrix(operator)
        rounded = np.array([[float(value) for value in row]
                            for row in exact], dtype=float)
        if not np.isfinite(rounded).all() or any(
                exact[i][j] != 0 and rounded[i, j] == 0
                for i in range(6) for j in range(6)):
            raise RepresentabilityError("basic matrix overflow or underflow")
        guard = _energy_guard(exact, rounded, EPSILON)
        decades = (F(1, 10**12), F(1, 10**10), EPSILON,
                   F(1, 10**6), F(1, 10**4), F(1, 10**2))
        first_decade = next((str(epsilon) for epsilon in decades
                             if _energy_guard(exact, rounded, epsilon)), None)
        _budget(start)
        return {"name": name, "status": "energy_guard_pass" if guard else
                "energy_guard_reject", "energy_guard_pass": guard,
                "first_passing_epsilon_decade": first_decade,
                "basic_exact": [[str(value) for value in row] for row in exact],
                "basic_rounded_sha256": sha256(rounded.tobytes()).hexdigest(),
                "basic_rounded": rounded.tolist()}
    except (ConditioningAdmissionError, RepresentabilityError, ValueError) as exc:
        return {"name": name, "status": "current_operator_rejected",
                "error_type": type(exc).__name__, "error_message": str(exc)}


def _rectangle(width, depth):
    # Exact geometry relation; section inputs are rounded only at the API.
    w, h = F(width), F(depth)
    area = w*h
    iy, iz = w*h**3/12, h*w**3/12
    torsion = w*h**3*(F(1, 3)-F(21, 100)*h/w*(1-h**4/(12*w**4)))
    section = LinearSection.isotropic(
        E=210e9, nu=.3, area=float(area), Iy=float(iy), Iz=float(iz),
        J=float(torsion), ky=5/6, kz=5/6)
    return section, {"width_m": str(w), "depth_m": str(h),
                     "area_m2": str(area), "Iy_m4": str(iy),
                     "Iz_m4": str(iz), "J_approx_m4": str(torsion)}


def _convert(section, length):
    # Numerical N-mm equivalent of the same section.  The public definition
    # remains SI; this tests invariance of the private local admission rule.
    multipliers = (F(1), F(1), F(1), UNIT, UNIT, UNIT)
    source = [[F(value) for value in row] for row in section.stiffness]
    converted = [[float(source[i][j]*multipliers[i]*multipliers[j])
                  for j in range(6)] for i in range(6)]
    return LinearSection.generalized(converted), float(F(length)*UNIT)


def _unit_comparison(si, mm):
    if "basic_exact" not in si or "basic_exact" not in mm:
        return None
    # Basic axial displacement scales by 1000; the other five basic
    # deformations are rotations.  Numerical energy (N mm) scales by 1000.
    u = (UNIT, F(1), F(1), F(1), F(1), F(1))
    base = [[F(value) for value in row] for row in si["basic_exact"]]
    converted = [[F(value) for value in row] for row in mm["basic_exact"]]
    expected = [[UNIT*base[i][j]/(u[i]*u[j])
                 for j in range(6)] for i in range(6)]
    # Input conversion passes through binary64 section entries; measure that
    # roundoff separately from the operator's output-energy guarantee.
    discrepancy = max((float(abs(converted[i][j]-expected[i][j]) /
                             max(abs(expected[i][j]), F(1, 10**100)))
                       for i in range(6) for j in range(6)), default=0.)
    return {"max_component_relative_to_expected": discrepancy,
            "same_energy_guard_decision": si["energy_guard_pass"] ==
            mm["energy_guard_pass"]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"immutable diagnostic already exists: {OUTPUT}")
    start = time.monotonic()
    record = {
        "status": "running",
        "claim": "private local feasibility only; no F09/G0B acceptance",
        "decision_question": "Can exact relative-energy rounding guard admit physically linked sections, remain unit equivalent, and flag adverse local conditioning and unrepresentable output?",
        "candidate_rule": "exact SPD of epsilon*A +/- (Afloat-A), epsilon=1e-8, for the six-basic-mode stiffness A; this proves an all-basic-deformations relative energy bound",
        "principal_uncertainty": "whether local binary64 rounding preserves physical energy despite a basis-dependent normalized-pivot cutoff",
        "source_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_graph": [
            {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in (Path(__file__), SOURCE)],
        "python": sys.version.split()[0], "numpy": np.__version__,
        "limits": {"wall_seconds": SECONDS_LIMIT, "rss_bytes": RSS_LIMIT,
                   "numerical_threads": 1},
        "cases": [], "unit_pairs": [], "negatives": [],
    }
    try:
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            record["threadpools"] = [
                {"internal_api": row.get("internal_api"),
                 "num_threads": row.get("num_threads")} for row in pools]
            if any(row.get("num_threads") != 1 for row in pools):
                raise RuntimeError("numerical thread limit is not effective")
            for name, width, depth in (
                    ("rectangle_12x3_mm", F(12, 1000), F(3, 1000)),
                    ("rectangle_192x0p1875_mm", F(192, 1000), F(3, 16000))):
                section, metadata = _rectangle(width, depth)
                mm_section, mm_length = _convert(section, 2.)
                si = _case(name + "_SI", 2., section, start)
                mm = _case(name + "_N_mm", mm_length, mm_section, start)
                record["cases"].extend((si, mm))
                record["unit_pairs"].append({"name": name, "section": metadata,
                                             "comparison": _unit_comparison(si, mm)})
            near = np.eye(6)
            near[0, 4] = near[4, 0] = 1. - 2.**-40
            near_section = LinearSection.generalized(near)
            near_mm_section, near_mm_length = _convert(near_section, 2.)
            si = _case("near_coupled_SI", 2., near_section, start)
            mm = _case("near_coupled_N_mm", near_mm_length,
                       near_mm_section, start)
            record["cases"].extend((si, mm))
            record["unit_pairs"].append({"name": "near_coupled",
                                         "comparison": _unit_comparison(si, mm)})
            for name, action in (
                    ("overflow_section", lambda: LinearSection.isotropic(
                        E=1e308, nu=0., area=1e308, Iy=1., Iz=1., J=1.,
                        ky=1., kz=1.)),
                    ("underflow_response", lambda: LinearOperator(
                        1., LinearSection.isotropic(
                            E=1., nu=0., area=2.**-1074,
                            Iy=1e-12, Iz=2e-12, J=3e-12,
                            ky=.8, kz=.6)).evaluate(
                                [0., 0., 0., 0., 0., 0., 0., 0., 1e-100,
                                 0., 0., 0.]))):
                try:
                    action()
                    record["negatives"].append({"name": name,
                                                 "status": "unexpected_admission"})
                except RepresentabilityError as exc:
                    record["negatives"].append({"name": name,
                                                 "status": "explicit_rejection",
                                                 "message": str(exc)})
            _budget(start)
            record["status"] = "completed_development_diagnostic"
    except Exception as exc:
        record["status"] = "failed_development_diagnostic"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        record["elapsed_seconds"] = time.monotonic()-start
        record["peak_rss_bytes"] = (psutil.Process().memory_info().peak_wset
                                    if sys.platform == "win32" else
                                    psutil.Process().memory_info().rss)
        OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"],
                          "cases": [(c["name"], c["status"],
                                     c.get("first_passing_epsilon_decade"))
                                    for c in record["cases"]],
                          "negatives": record["negatives"],
                          "elapsed_seconds": record["elapsed_seconds"],
                          "peak_rss_bytes": record["peak_rss_bytes"],
                          "output": str(OUTPUT)}))
    return 0 if record["status"] == "completed_development_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
