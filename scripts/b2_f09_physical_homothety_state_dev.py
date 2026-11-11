"""Bounded F09 development discriminator for physical-section state storage.

This is private development evidence, not F09/G0B admission.  A fixed first
rectangle and one homothetically scaled second rectangle isolate three layers:
the exact physical flexibility oracle, the stored rounded graph, and the
binary64 sparse/factored state representations.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (ROOT / "docs" / "agent_plans" /
          "B3_GE_E4_PL_G0_F09_PHYSICAL_HOMOTHETY_STATE_DEV_V1.json")
THIN_PATH = ROOT / "scripts" / "b2_f09_thin_section_dev.py"
PROBE_PATH = ROOT / "scripts" / "probe_b3_ge_e4_pl_g0_f09_graph_reversal.py"
THIN = runpy.run_path(str(THIN_PATH))
G = THIN["_run_case"].__globals__

F = Fraction
ZERO = F(0)
ONE = F(1)
LAMBDAS = (ONE, F(1, 32), F(1, 1024))
FIRST_SECTION_M = (F(80, 1000), F(3, 1000))
SECOND_BASE_M = (F(12, 1000), F(3, 1000))
BASE_TIP_WRENCH = tuple(value / 10 for value in THIN["TIP_WRENCH_EXACT"])
PHYSICAL_RELATIVE = 1e-8
TIP_ABSOLUTE = 1e-10
STATION_RESULTANT_ABSOLUTE = 1e-7
STATION_STRAIN_ABSOLUTE = 1e-10
SPARSE_REFERENCE_RELATIVE = 1e-9
SPARSE_REFERENCE_ABSOLUTE = 1e-12
BALANCE_RELATIVE = 1e-9


def _source_graph():
    paths = [
        Path(__file__).resolve(),
        THIN_PATH,
        PROBE_PATH,
        ROOT / "src" / "anyfesolver" / "_b2_physical_flexibility_linear.py",
        ROOT / "src" / "anyfesolver" / "_b2_physical_linear_definition.py",
        ROOT / "src" / "anyfesolver" / "_b2_physical_linear_batch_dev.py",
        ROOT / "src" / "anyfesolver" / "control.py",
        ROOT / "src" / "anyfesolver" / "linalg.py",
        ROOT / "src" / "anyfesolver" / "threading_policy.py",
    ]
    return [{"path": str(path.resolve()),
             "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in paths]


def _section(width, depth):
    G["WIDTH_M"] = width
    G["DEPTH_M"] = depth
    _, compliance, stored, principal, area, iy, iz, torsion = THIN["_section"]()
    return {
        "compliance": compliance,
        "stored": stored,
        "metadata": {
            "width_m": str(width), "depth_m": str(depth),
            "A_m2": str(area), "Iy_m4": str(iy), "Iz_m4": str(iz),
            "J_approx_m4": str(torsion),
            "principal_stiffness": [str(value) for value in principal],
            "stored_stiffness_sha256": sha256(stored.tobytes()).hexdigest(),
        },
    }


def _members(first_section, second_section, reverse):
    first, second = THIN["PROBE"]["members"](reverse)
    sign = np.diag([-1., 1., -1., -1., 1., -1.])
    first = replace(
        first, section_id="steel-flat-bar-80x3-345",
        section=THIN["LinearSection"].generalized(first_section["stored"]))
    second_stored = second_section["stored"]
    if reverse:
        second_stored = sign @ second_stored @ sign
    second = replace(
        second,
        section_id=("steel-flat-bar-homothetic-reversed" if reverse
                    else "steel-flat-bar-homothetic"),
        section=THIN["LinearSection"].generalized(second_stored))
    return first, second


def _physical_oracle(members, compliance_by_member, tip_wrench):
    """Exact Simpson/Castigliano oracle for this serial straight-member graph."""
    energy, response = ZERO, [ZERO] * 6
    station_data = {}
    for member in members:
        length = F(float(member.length_m))
        compliance = compliance_by_member[member.member_id]
        frame = THIN["_frame"](member)
        for station, weight in ((ZERO, 1), (F(1, 2), 4), (ONE, 1)):
            wrench, strain, lever = THIN["_cut"](
                member, station, tip_wrench, compliance)
            density = sum((wrench[i] * strain[i] for i in range(6)), ZERO)
            energy += length * weight * density / 12
            for i in range(6):
                direction = [ZERO] * 6
                direction[i] = ONE
                dforce, dcouple = direction[:3], direction[3:]
                arm_moment = THIN["_cross"](lever, dforce)
                dmoment = tuple(dcouple[k] + arm_moment[k] for k in range(3))
                dw = [sum((frame[j][k] * vector[k] for k in range(3)), ZERO)
                      for vector in (dforce, dmoment) for j in range(3)]
                response[i] += length * weight * sum(
                    (dw[k] * strain[k] for k in range(6)), ZERO) / 6
            sign = -ONE if member.node_ids == (2, 1) else ONE
            station_data[(member.member_id, str(station))] = (
                [sign * value for value in wrench],
                [sign * value for value in strain])
    return energy, response, station_data


def _exact_coordinate_residual(matrix, coordinates, rhs):
    dense = np.asarray(matrix)
    return [sum((F(float(a)) * F(float(x)) for a, x in zip(row, coordinates)),
                ZERO) - F(float(target))
            for row, target in zip(dense, rhs)]


def _exact_factored_residual(matrix, scaling, scaled_state, rhs):
    dense = np.asarray(matrix)
    return [sum((F(float(a)) * F(float(scaling[j])) *
                 F(float(scaled_state[j])) for j, a in enumerate(row)), ZERO)
                - F(float(target))
            for row, target in zip(dense, rhs)]


def _balance(defects, tip_wrench, members):
    forces = (0, 1, 2, 6, 7, 8)
    moments = (3, 4, 5, 9, 10, 11)
    force_scale = max(abs(float(value)) for value in tip_wrench[:3])
    moment_scale = max(
        max(abs(float(value)) for value in tip_wrench[3:]),
        max(member.length_m for member in members) * force_scale)
    force_ratio = max(float(abs(defects[i])) for i in forces) / force_scale
    moment_ratio = max(float(abs(defects[i])) for i in moments) / moment_scale
    return {"force_ratio": force_ratio, "moment_ratio": moment_ratio,
            "passes_1e-9": bool(max(force_ratio, moment_ratio) <= BALANCE_RELATIVE)}


def _maximum_ratio(actual, expected, relative, absolute):
    return THIN["_maximum_ratio"](
        actual, expected, relative=relative, absolute=absolute)


def _station_ratios(members, full_state, station_data):
    resultant_ratio = 0.
    strain_ratio = 0.
    for member in members:
        indices = [6 * node + j for node in member.node_ids for j in range(6)]
        recovered = member.local_operator().recover_sections(
            member.local_dof_map() @ full_state[indices], stations=(0., .5, 1.))
        for i, station in enumerate((ZERO, F(1, 2), ONE)):
            expected_resultant, expected_strain = station_data[
                (member.member_id, str(station))]
            resultant_ratio = max(
                resultant_ratio,
                _maximum_ratio(recovered.resultants[i],
                               [float(v) for v in expected_resultant],
                               PHYSICAL_RELATIVE, STATION_RESULTANT_ABSOLUTE))
            strain_ratio = max(
                strain_ratio,
                _maximum_ratio(recovered.strains[i],
                               [float(v) for v in expected_strain],
                               PHYSICAL_RELATIVE, STATION_STRAIN_ABSOLUTE))
    return resultant_ratio, strain_ratio


def _reference_scaled_state(decimal_coordinates, scaling):
    with localcontext() as context:
        context.prec = 100
        return np.array([
            float(value / Decimal.from_float(float(scale)))
            for value, scale in zip(decimal_coordinates, scaling)
        ])


def _run_case(first_section, second_section, lam, reverse, start):
    THIN["_check_budget"](start)
    members = _members(first_section, second_section, reverse)
    tip_exact = tuple(value * lam**4 for value in BASE_TIP_WRENCH)
    tip = np.array([float(value) for value in tip_exact])
    compliance = {"m0": first_section["compliance"],
                  "m1": second_section["compliance"]}
    physical_energy, physical_tip, station_data = _physical_oracle(
        members, compliance, tip_exact)
    physical_tip_float = np.array([float(value) for value in physical_tip])

    matrix = THIN["PROBE"]["assemble"](members)
    reduced = matrix[6:, 6:].tocsc()
    dense = reduced.toarray()
    load = np.zeros(18)
    load[12:] = tip
    rhs = load[6:]
    reference, decimal_reference = THIN["_decimal_solve"](dense, rhs, start)

    scaling = 1. / np.sqrt(reduced.diagonal())
    scaled_matrix = (diags(scaling) @ reduced @ diags(scaling)).tocsc()
    factor = splu(scaled_matrix)
    scaled_state = np.asarray(factor.solve(scaling * rhs), dtype=float)
    initial_scaled_state = scaled_state.copy()
    correction_count = 0
    for _ in range(2):
        coordinates = scaling * scaled_state
        defect = _exact_coordinate_residual(dense, coordinates, rhs)
        if _balance(defect, tip_exact, members)["passes_1e-9"]:
            break
        correction = factor.solve(scaling * np.array([float(v) for v in defect]))
        scaled_state -= correction
        correction_count += 1
    coordinates = scaling * scaled_state

    reference_coordinate_balance = _balance(
        _exact_coordinate_residual(dense, reference, rhs), tip_exact, members)
    sparse_coordinate_balance = _balance(
        _exact_coordinate_residual(dense, coordinates, rhs), tip_exact, members)
    sparse_factored_balance = _balance(
        _exact_factored_residual(dense, scaling, scaled_state, rhs),
        tip_exact, members)
    reference_scaled_state = _reference_scaled_state(decimal_reference, scaling)
    reference_factored_balance = _balance(
        _exact_factored_residual(dense, scaling, reference_scaled_state, rhs),
        tip_exact, members)

    physical_energy_float = float(physical_energy)
    reference_tip_ratio = _maximum_ratio(
        reference[6:], physical_tip_float, PHYSICAL_RELATIVE, TIP_ABSOLUTE)
    sparse_tip_ratio = _maximum_ratio(
        coordinates[6:], physical_tip_float, PHYSICAL_RELATIVE, TIP_ABSOLUTE)
    sparse_reference_ratio = _maximum_ratio(
        coordinates, reference, SPARSE_REFERENCE_RELATIVE,
        SPARSE_REFERENCE_ABSOLUTE)
    reference_work = .5 * float(rhs @ reference)
    sparse_work = .5 * float(rhs @ coordinates)
    reference_work_error = abs(reference_work - physical_energy_float) / physical_energy_float
    sparse_work_error = abs(sparse_work - physical_energy_float) / physical_energy_float
    full_state = np.zeros(18)
    full_state[6:] = coordinates
    station_resultant_ratio, station_strain_ratio = _station_ratios(
        members, full_state, station_data)
    eligible = (max(abs(v) for v in physical_tip_float[:3]) <=
                .01 * min(member.length_m for member in members) and
                max(abs(v) for v in physical_tip_float[3:]) <= .01)

    batch_record = {"status": "not_started"}
    batch = None
    try:
        transform = sparse.eye(18, format="csr")[:, 6:]
        batch = THIN["CompiledB2AffineBatch"](
            tuple(member.compile_local() for member in members),
            transform, np.zeros(18), np.arange(12, 18), max_chunk=1)
        batch_result = batch.solve_many((("homothetic-mixed", load),))
        batch_tip = np.array(batch_result.cases[0].values)
        batch_record = {
            "status": batch_result.status,
            "factorization_count": batch_result.factorization_count,
            "solve_count": batch_result.solve_count,
            "tip_ratio_to_rounded_reference_limit": _maximum_ratio(
                batch_tip, reference[6:], SPARSE_REFERENCE_RELATIVE,
                SPARSE_REFERENCE_ABSOLUTE),
        }
    except Exception as exc:
        batch_record = {"status": "rejected", "error_type": type(exc).__name__,
                        "error_message": str(exc)}
    finally:
        if batch is not None:
            batch.close()

    physical_match = bool(
        reference_tip_ratio <= 1. and reference_work_error <= PHYSICAL_RELATIVE)
    if not physical_match:
        classification = "rounded_graph_differs_from_physical_oracle"
    elif (reference_coordinate_balance["passes_1e-9"] and
          not sparse_coordinate_balance["passes_1e-9"]):
        classification = "sparse_factorization_or_refinement_failure"
    elif (not reference_coordinate_balance["passes_1e-9"] and
          reference_factored_balance["passes_1e-9"]):
        classification = "ordinary_coordinate_storage_limit"
    elif (not reference_coordinate_balance["passes_1e-9"] and
          not reference_factored_balance["passes_1e-9"]):
        classification = "tested_binary64_state_representations_fail"
    elif batch_record["status"] != "completed":
        classification = "batch_route_rejected_despite_reference_balance"
    else:
        classification = "all_tested_layers_pass"

    passed = bool(
        physical_match and eligible and reference_coordinate_balance["passes_1e-9"] and
        sparse_coordinate_balance["passes_1e-9"] and
        sparse_factored_balance["passes_1e-9"] and sparse_tip_ratio <= 1. and
        sparse_reference_ratio <= 1. and sparse_work_error <= PHYSICAL_RELATIVE and
        station_resultant_ratio <= 1. and station_strain_ratio <= 1. and
        batch_record["status"] == "completed" and
        batch_record.get("factorization_count") == 1)
    THIN["_check_budget"](start)
    return {
        "lambda": str(lam),
        "orientation": "reversed-second" if reverse else "forward",
        "classification": classification,
        "passed_development_diagnostic": passed,
        "tip_wrench_exact_N_Nm": [str(value) for value in tip_exact],
        "physical_tip_m_rad": physical_tip_float.tolist(),
        "physical_cut_work_J": physical_energy_float,
        "rounded_reference_tip_m_rad": reference[6:].tolist(),
        "rounded_reference_work_J": reference_work,
        "sparse_tip_m_rad": coordinates[6:].tolist(),
        "sparse_work_J": sparse_work,
        "rounded_reference_tip_ratio_to_physical_limit": reference_tip_ratio,
        "sparse_tip_ratio_to_physical_limit": sparse_tip_ratio,
        "sparse_tip_ratio_to_rounded_reference_limit": sparse_reference_ratio,
        "rounded_reference_work_relative_error": reference_work_error,
        "sparse_work_relative_error": sparse_work_error,
        "rounded_reference_coordinate_balance": reference_coordinate_balance,
        "rounded_reference_factored_balance": reference_factored_balance,
        "sparse_coordinate_balance": sparse_coordinate_balance,
        "sparse_factored_balance": sparse_factored_balance,
        "sparse_initial_factored_balance": _balance(
            _exact_factored_residual(dense, scaling, initial_scaled_state, rhs),
            tip_exact, members),
        "sparse_correction_count": correction_count,
        "station_resultant_ratio_to_physical_limit": station_resultant_ratio,
        "station_strain_ratio_to_physical_limit": station_strain_ratio,
        "small_rotation_and_displacement_eligibility": bool(eligible),
        "batch": batch_record,
        "rounded_reduced_matrix_sha256": sha256(dense.tobytes()).hexdigest(),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"immutable diagnostic already exists: {OUTPUT}")
    start = time.monotonic()
    result = {
        "status": "running",
        "claim": "bounded private development diagnostic; no F09/G0B admission",
        "question": ("Does a coherent heterogeneous physical-section path fail in "
                     "ordinary binary64 coordinates, the factored D,z state, or the "
                     "scaled sparse factorization?"),
        "source_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_graph": _source_graph(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "fixture": {
            "member_lengths_m": [2., 1.5],
            "first_section": "80x3 mm fixed rectangle",
            "second_section": "lambda*(12x3 mm) homothetic rectangle",
            "lambdas": [str(value) for value in LAMBDAS],
            "axis_rotation": {"cos": "4/5", "sin": "3/5"},
            "base_tip_wrench_exact_N_Nm": [str(v) for v in BASE_TIP_WRENCH],
            "load_scaling": "complete tip wrench multiplied by lambda^4",
            "orientations": ["forward", "reversed-second"],
        },
        "criteria": {
            "physical_tip": "rtol=1e-8, atol=1e-10",
            "physical_work_relative": PHYSICAL_RELATIVE,
            "station_resultant": "rtol=1e-8, atol=1e-7",
            "station_strain": "rtol=1e-8, atol=1e-10",
            "sparse_vs_rounded_reference": "rtol=1e-9, atol=1e-12",
            "force_and_moment_balance_relative": BALANCE_RELATIVE,
            "linear_eligibility": ("tip translation <= 1% shortest member; "
                                   "tip rotation <= 0.01 rad"),
            "wall_seconds": 120,
            "rss_bytes": 2 * 1024**3,
            "numerical_threads": 1,
        },
        "cases": [],
    }
    try:
        with THIN["threadpool_limits"](limits=1):
            pools = THIN["threadpool_info"]()
            result["threadpools"] = [
                {"internal_api": row.get("internal_api"),
                 "num_threads": row.get("num_threads")} for row in pools]
            if any(row.get("num_threads") != 1 for row in pools):
                raise RuntimeError("numerical thread limit is not effective")
            first_section = _section(*FIRST_SECTION_M)
            result["first_section"] = first_section["metadata"]
            for lam in LAMBDAS:
                THIN["_check_budget"](start)
                second_section = _section(
                    SECOND_BASE_M[0] * lam, SECOND_BASE_M[1] * lam)
                for reverse in (False, True):
                    try:
                        case = _run_case(first_section, second_section, lam,
                                         reverse, start)
                    except Exception as exc:
                        case = {
                            "lambda": str(lam),
                            "orientation": ("reversed-second" if reverse else "forward"),
                            "classification": "case_execution_failed",
                            "passed_development_diagnostic": False,
                            "error": {"type": type(exc).__name__,
                                      "message": str(exc)},
                        }
                    case["second_section"] = second_section["metadata"]
                    result["cases"].append(case)
            result["status"] = ("passed_development_diagnostic" if all(
                case["passed_development_diagnostic"] for case in result["cases"])
                else "completed_with_development_failures")
    except Exception as exc:
        result["status"] = "failed_development_diagnostic"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["peak_rss_bytes"] = (
            THIN["psutil"].Process().memory_info().peak_wset
            if sys.platform == "win32" else THIN["psutil"].Process().memory_info().rss)
        with OUTPUT.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"status": result["status"],
                          "cases": len(result["cases"]),
                          "classifications": [case.get("classification")
                                              for case in result["cases"]],
                          "output": str(OUTPUT),
                          "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if result["status"] == "passed_development_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
