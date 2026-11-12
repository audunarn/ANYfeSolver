"""One bounded physical thin-section F09 development diagnostic.

This is private source evidence, not section-range or F09 qualification.
The physical cut oracle uses rational geometry, material data and a rational
section-axis rotation. The separate Decimal solve uses the rounded graph.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import psutil
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
PROBE = runpy.run_path(str(ROOT / "scripts" /
                           "probe_b3_ge_e4_pl_g0_f09_graph_reversal.py"))
from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_batch_dev import CompiledB2AffineBatch


F = Fraction
ZERO = F(0)
ONE = F(1)
WIDTH_M = F(80, 1000)
DEPTH_M = F(3, 1000)
YOUNG_PA = F(210_000_000_000)
POISSON = F(3, 10)
COSINE = F(4, 5)
SINE = F(3, 5)
TIP_WRENCH_EXACT = (F(25, 1000), F(-80, 1000), F(60, 1000),
                    F(8, 1000), F(-9, 1000), F(7, 1000))
TIP_WRENCH = tuple(float(value) for value in TIP_WRENCH_EXACT)  # N, N m
SECONDS_LIMIT = 120.
RSS_LIMIT = 2 * 1024**3


def _check_budget(start: float) -> None:
    if time.monotonic() - start > SECONDS_LIMIT:
        raise RuntimeError("120 s thin-section diagnostic time bound exceeded")
    if psutil.Process().memory_info().rss > RSS_LIMIT:
        raise RuntimeError("2 GiB thin-section diagnostic RSS bound exceeded")


def _multiply(matrix, vector):
    return [sum((a*b for a, b in zip(row, vector)), ZERO) for row in matrix]


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _section():
    w, h = WIDTH_M, DEPTH_M
    shear = YOUNG_PA/(2*(1+POISSON))
    area = w*h
    iy, iz = w*h**3/12, h*w**3/12
    # Declared Saint-Venant engineering approximation for a solid rectangle.
    torsion = w*h**3*(F(1, 3)-F(21, 100)*h/w*(1-h**4/(12*w**4)))
    principal = (YOUNG_PA*area, shear*area*F(5, 6),
                 shear*area*F(5, 6), shear*torsion,
                 YOUNG_PA*iy, YOUNG_PA*iz)
    transform = [[F(int(i == j)) for j in range(6)] for i in range(6)]
    for first, second in ((1, 2), (4, 5)):
        transform[first][first] = transform[second][second] = COSINE
        transform[first][second], transform[second][first] = -SINE, SINE

    def rotated(diagonal):
        return [[sum((transform[i][k]*diagonal[k]*transform[j][k]
                      for k in range(6)), ZERO)
                 for j in range(6)] for i in range(6)]

    stiffness = rotated(principal)
    compliance = rotated(tuple(ONE/value for value in principal))
    stored = np.array([[float(value) for value in row] for row in stiffness])
    return stiffness, compliance, stored, principal, area, iy, iz, torsion


def _members(stored: np.ndarray, reverse: bool):
    base = PROBE["members"](reverse)
    sign = np.diag([-1., 1., -1., -1., 1., -1.])
    return (replace(base[0], section_id="steel-flat-bar-80x3-345",
                    section=LinearSection.generalized(stored)),
            replace(base[1], section_id="steel-flat-bar-80x3-345-reversed"
                    if reverse else "steel-flat-bar-80x3-345",
                    section=LinearSection.generalized(
                        sign@stored@sign if reverse else stored)))


def _frame(member):
    # Independent exact frames from the declared elbow geometry/axis choice.
    if member.member_id == "m0":
        return ((ONE, ZERO, ZERO), (ZERO, ONE, ZERO), (ZERO, ZERO, ONE))
    if member.node_ids == (2, 1):
        return ((ZERO, -ONE, ZERO), (-ONE, ZERO, ZERO),
                (ZERO, ZERO, -ONE))
    return ((ZERO, ONE, ZERO), (-ONE, ZERO, ZERO),
            (ZERO, ZERO, ONE))


def _cut(member, station: Fraction, load, compliance):
    start, end = (tuple(F(float(v)) for v in point)
                  for point in member.reference_points_m)
    tip = (F(2), F(3, 2), ZERO)
    point = tuple(start[i]+station*(end[i]-start[i]) for i in range(3))
    lever = tuple(tip[i]-point[i] for i in range(3))
    force, couple = load[:3], load[3:]
    lever_moment = _cross(lever, force)
    global_moment = tuple(couple[i]+lever_moment[i] for i in range(3))
    frame = _frame(member)
    wrench = [sum((frame[i][k]*value[k] for k in range(3)), ZERO)
              for value in (force, global_moment) for i in range(3)]
    sign = (-ONE, ONE, -ONE, -ONE, ONE, -ONE)
    local_compliance = ([[sign[i]*compliance[i][j]*sign[j]
                          for j in range(6)] for i in range(6)]
                        if member.node_ids == (2, 1) else compliance)
    return wrench, _multiply(local_compliance, wrench), lever


def _physical_oracle(members, compliance):
    load = TIP_WRENCH_EXACT
    energy, response = ZERO, [ZERO]*6
    station_data = {}
    for member in members:
        length = F(float(member.length_m))
        for station, weight in ((ZERO, 1), (F(1, 2), 4), (ONE, 1)):
            wrench, strain, lever = _cut(member, station, load, compliance)
            density = sum((wrench[i]*strain[i] for i in range(6)), ZERO)
            energy += length*weight*density/12
            frame = _frame(member)
            for i in range(6):
                direction = [ZERO]*6
                direction[i] = ONE
                dforce, dcouple = direction[:3], direction[3:]
                moment = _cross(lever, dforce)
                dglobal_moment = tuple(dcouple[k]+moment[k] for k in range(3))
                dw = [sum((frame[j][k]*value[k] for k in range(3)), ZERO)
                      for value in (dforce, dglobal_moment) for j in range(3)]
                response[i] += length*weight*sum(
                    (dw[k]*strain[k] for k in range(6)), ZERO)/6
            resultant = ([-value for value in wrench]
                         if member.node_ids == (2, 1) else wrench)
            physical_strain = ([-value for value in strain]
                               if member.node_ids == (2, 1) else strain)
            station_data[(member.member_id, str(station))] = (
                resultant, physical_strain)
    return energy, response, station_data


def _decimal_solve(matrix: np.ndarray, rhs: np.ndarray, start: float):
    with localcontext() as context:
        context.prec = 100
        rows = [[Decimal.from_float(float(v)) for v in row]
                + [Decimal.from_float(float(rhs[i]))]
                for i, row in enumerate(matrix)]
        n = len(rows)
        for k in range(n):
            _check_budget(start)
            pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
            rows[k], rows[pivot] = rows[pivot], rows[k]
            if rows[k][k] == 0:
                raise ArithmeticError("rounded reduced matrix is singular")
            for i in range(k+1, n):
                factor = rows[i][k]/rows[k][k]
                for j in range(k+1, n+1):
                    rows[i][j] -= factor*rows[k][j]
                rows[i][k] = Decimal(0)
        answer = [Decimal(0)]*n
        for i in range(n-1, -1, -1):
            answer[i] = (rows[i][-1]-sum(
                (rows[i][j]*answer[j] for j in range(i+1, n)),
                Decimal(0)))/rows[i][i]
        return np.array([float(value) for value in answer]), answer


def _exact_product_residual(matrix: np.ndarray, solution, rhs):
    return [sum((F(float(a))*F(float(x)) for a, x in zip(row, solution)),
                ZERO)-F(float(target))
            for row, target in zip(matrix, rhs)]


def _maximum_ratio(actual, expected, *, relative, absolute):
    return max(float(abs(F(float(a))-F(float(e))) /
                     F(float(absolute+relative*abs(float(e)))))
               for a, e in zip(actual, expected))


def _run_case(members, compliance, start):
    _check_budget(start)
    energy, physical_tip, station_data = _physical_oracle(members, compliance)
    matrix = PROBE["assemble"](members)
    free = np.arange(6, 18)
    reduced = matrix[free, :][:, free].tocsc()
    load = np.zeros(18)
    load[12:] = TIP_WRENCH
    rhs = load[free]
    dense = reduced.toarray()
    _check_budget(start)
    high_precision, _decimal = _decimal_solve(dense, rhs, start)
    scaling = 1/np.sqrt(reduced.diagonal())
    factor = splu((diags(scaling)@reduced@diags(scaling)).tocsc())
    stored_solution = scaling*factor.solve(scaling*rhs)
    if not np.isfinite(stored_solution).all():
        raise RuntimeError("scaled sparse state is nonfinite")
    exact_residual = _exact_product_residual(dense, stored_solution, rhs)
    rounded_reference_residual = _exact_product_residual(dense, high_precision, rhs)
    force_scale = max(abs(value) for value in TIP_WRENCH[:3])
    moment_scale = max(max(abs(value) for value in TIP_WRENCH[3:]),
                       max(member.length_m for member in members)*force_scale)
    forces = (0, 1, 2, 6, 7, 8)
    moments = (3, 4, 5, 9, 10, 11)
    force_ratio = max(float(abs(exact_residual[i])) for i in forces)/force_scale
    moment_ratio = max(float(abs(exact_residual[i])) for i in moments)/moment_scale
    reference_force_ratio = max(float(abs(rounded_reference_residual[i]))
                                for i in forces)/force_scale
    reference_moment_ratio = max(float(abs(rounded_reference_residual[i]))
                                 for i in moments)/moment_scale
    predicted_tip = [float(value) for value in physical_tip]
    linear_scale_ok = (max(abs(v) for v in predicted_tip[:3]) <=
                       .01*min(member.length_m for member in members) and
                       max(abs(v) for v in predicted_tip[3:]) <= .01)
    tip_ratio = _maximum_ratio(stored_solution[6:], predicted_tip,
                               relative=1e-8, absolute=1e-10)
    rounded_tip_ratio = _maximum_ratio(high_precision[6:], predicted_tip,
                                       relative=1e-8, absolute=1e-10)
    high_precision_ratio = _maximum_ratio(stored_solution, high_precision,
                                          relative=1e-9, absolute=1e-12)
    stored_work = float(rhs@stored_solution)/2
    physical_work = float(energy)
    work_error = abs(stored_work-physical_work)/physical_work
    full = np.zeros(18)
    full[free] = stored_solution
    station_resultant_ratio = 0.
    station_strain_ratio = 0.
    for member in members:
        indices = [6*node+j for node in member.node_ids for j in range(6)]
        recovered = member.local_operator().recover_sections(
            member.local_dof_map()@full[indices], stations=(0., .5, 1.))
        for i, station in enumerate((ZERO, F(1, 2), ONE)):
            expected_resultant, expected_strain = station_data[
                (member.member_id, str(station))]
            station_resultant_ratio = max(
                station_resultant_ratio,
                _maximum_ratio(recovered.resultants[i],
                               [float(v) for v in expected_resultant],
                               relative=1e-8, absolute=1e-7))
            station_strain_ratio = max(
                station_strain_ratio,
                _maximum_ratio(recovered.strains[i],
                               [float(v) for v in expected_strain],
                               relative=1e-8, absolute=1e-10))
    _check_budget(start)
    transform = sparse.eye(18, format="csr")[:, 6:]
    batch = None
    try:
        batch = CompiledB2AffineBatch(
            tuple(member.compile_local() for member in members),
            transform, np.zeros(18), np.arange(12, 18), max_chunk=1)
        batch_result = batch.solve_many((("thin-flat-bar-mixed", load),),
                                         recover_members={"m0": (0., .5, 1.)})
        batch_tip = list(batch_result.cases[0].values)
        batch_ratio = _maximum_ratio(batch_tip, high_precision[6:],
                                     relative=1e-9, absolute=1e-12)
        batch_status = batch_result.status
        factorization_count = batch_result.factorization_count
    finally:
        if batch is not None:
            batch.close()
    passed = (linear_scale_ok and force_ratio <= 1e-8 and moment_ratio <= 1e-8 and
              tip_ratio <= 1. and rounded_tip_ratio <= 1. and
              high_precision_ratio <= 1. and work_error <= 1e-8 and
              station_resultant_ratio <= 1. and station_strain_ratio <= 1. and
              batch_ratio <= 1. and batch_status == "completed" and
              factorization_count == 1)
    return {
        "orientation": "reversed-second" if members[1].node_ids == (2, 1) else "forward",
        "passed_development_diagnostic": bool(passed),
        "physical_tip_m_rad": predicted_tip,
        "small_rotation_and_displacement_eligibility": bool(linear_scale_ok),
        "stored_sparse_tip_m_rad": stored_solution[6:].tolist(),
        "rounded_matrix_100_digit_tip_m_rad": high_precision[6:].tolist(),
        "physical_cut_work_J": physical_work,
        "stored_sparse_work_J": stored_work,
        "tip_component_ratio_to_development_limit": tip_ratio,
        "rounded_tip_component_ratio_to_development_limit": rounded_tip_ratio,
        "sparse_vs_100_digit_ratio_to_development_limit": high_precision_ratio,
        "stored_exact_product_force_residual_ratio": force_ratio,
        "stored_exact_product_moment_residual_ratio": moment_ratio,
        "rounded_reference_stored_force_residual_ratio": reference_force_ratio,
        "rounded_reference_stored_moment_residual_ratio": reference_moment_ratio,
        "work_relative_error": work_error,
        "station_resultant_ratio_to_development_limit": station_resultant_ratio,
        "station_strain_ratio_to_development_limit": station_strain_ratio,
        "batch_vs_100_digit_tip_ratio_to_development_limit": batch_ratio,
        "batch_status": batch_status,
        "batch_factorization_count": factorization_count,
        "rounded_reduced_matrix_sha256": sha256(dense.tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "agent_plans" /
                        "B3_GE_E4_PL_G0_F09_THIN_SECTION_DEV_V1.json")
    args = parser.parse_args()
    start = time.monotonic()
    source_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "status": "running",
        "claim": "bounded private development diagnostic; no F09/G0B admission",
        "source_head": os.popen("git rev-parse HEAD").read().strip(),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": source_hash,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "geometry_m": {"width": str(WIDTH_M), "depth": str(DEPTH_M),
                       "member_lengths": [2., 1.5]},
        "material": {"E_Pa": str(YOUNG_PA), "nu": str(POISSON),
                     "shear_factors": ["5/6", "5/6"],
                     "torsion": "declared Saint-Venant rectangle approximation"},
        "axis_rotation": {"cos": str(COSINE), "sin": str(SINE)},
        "tip_wrench_N_Nm": TIP_WRENCH,
        "tip_wrench_exact_N_Nm": [str(value) for value in TIP_WRENCH_EXACT],
        "limits": {"physical_force_moment_relative": 1e-8,
                   "tip_component": "rtol=1e-8, atol=1e-10 (development)",
                   "station_resultant": "rtol=1e-8, atol=1e-7 (development)",
                   "station_strain": "rtol=1e-8, atol=1e-10 (development)",
                   "work_relative": 1e-8,
                   "linear_eligibility": "tip translation <= 1% shortest member; tip rotation <= 0.01 rad",
                   "wall_seconds": SECONDS_LIMIT, "rss_bytes": RSS_LIMIT,
                   "numerical_threads": 1},
        "cases": [],
    }
    try:
        with threadpool_limits(limits=1):
            info = threadpool_info()
            result["threadpools"] = [
                {"internal_api": row.get("internal_api"),
                 "num_threads": row.get("num_threads")} for row in info]
            if any(row.get("num_threads") != 1 for row in info):
                raise RuntimeError("numerical thread limit is not effective")
            stiffness, compliance, stored, principal, area, iy, iz, torsion = _section()
            result["section"] = {
                "A_m2": str(area), "Iy_m4": str(iy), "Iz_m4": str(iz),
                "J_approx_m4": str(torsion),
                "principal_stiffness": [str(value) for value in principal],
                "stored_stiffness_sha256": sha256(stored.tobytes()).hexdigest(),
            }
            forward = _members(stored, False)
            reversed_members = _members(stored, True)
            result["cases"].append(_run_case(forward, compliance, start))
            result["cases"].append(_run_case(reversed_members, compliance, start))
            result["status"] = ("passed_development_diagnostic" if
                                all(case["passed_development_diagnostic"]
                                    for case in result["cases"]) else
                                "failed_development_diagnostic")
    except Exception as exc:
        result["status"] = "failed_development_diagnostic"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["elapsed_seconds"] = time.monotonic()-start
        result["peak_rss_bytes"] = psutil.Process().memory_info().peak_wset if sys.platform == "win32" else psutil.Process().memory_info().rss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",
                               encoding="utf-8")
        print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                          "output": str(args.output),
                          "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if result["status"] == "passed_development_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
