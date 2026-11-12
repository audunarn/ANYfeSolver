"""Local BL5 weak-P0 slip/work from saved CalculiX Fy/Mx and native solves.

This is a bounded development discriminator on one matched geometry. It does
not allocate the native/continuum body discrepancy or qualify the line law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anyfesolver
import numba
import numpy as np
import psutil
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell as ss
from scripts import r8_bl5_ccx_native_fy_mx_compare_dev as compare
from scripts import r8_bl5_ccx_weak_p0_dev as deck
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_weak_continuum_reference as weak
from scripts.r8_e02_double_output import read_double_output


CASES = ("fy", "mx")
ROOT = Path(__file__).resolve().parents[1]


def _summary(stations: list[dict], applied_work: float) -> dict:
    length = sum(row["measure_m"] for row in stations)
    if not np.isclose(length, 1., rtol=0., atol=1e-14):
        raise ValueError("one-metre physical bond measure required")
    gaps = np.asarray([row["gap"] for row in stations])
    measures = np.asarray([row["measure_m"] for row in stations])
    work = np.asarray([row["local_work_J"] for row in stations])
    if gaps.shape != (8, 5) or not np.isfinite(gaps).all() or not np.isfinite(work).all():
        raise ValueError("eight finite five-channel samples required")
    return {
        "translation_rms_m": float(np.sqrt(np.sum(measures * np.sum(gaps[:, :3] ** 2, axis=1)) / length)),
        "translation_max_m": float(np.max(np.linalg.norm(gaps[:, :3], axis=1))),
        "director_rms": float(np.sqrt(np.sum(measures * np.sum(gaps[:, 3:] ** 2, axis=1)) / length)),
        "director_max": float(np.max(np.linalg.norm(gaps[:, 3:], axis=1))),
        "signed_local_work_J": float(np.sum(work)),
        "absolute_local_work_J": float(np.sum(np.abs(work))),
        "absolute_local_work_over_applied": float(np.sum(np.abs(work)) / max(abs(applied_work), 1e-12)),
        "applied_work_J": float(applied_work),
    }


def _solid(case: str, saved: Path, report: Path) -> dict:
    binding = compare.checked_saved(saved, case, report)
    prior = json.loads(report.read_text(encoding="utf-8"))
    multiplier_parity = float(prior.get("physical_multiplier_relative_to_independent", float("nan")))
    if not np.isfinite(multiplier_parity) or multiplier_parity > 1e-6:
        raise ValueError("saved resolved multiplier density lacks independent parity")
    _, model = deck.solid.build_reference()
    metadata = json.loads((saved / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model_sha256"] != model["model_sha256"]:
        raise ValueError("resolved geometry identity changed")
    k, _ = weak.assemble_body(model)
    c, measures = weak.constraints(model, "weak")
    order, _, _, free, _ = weak.layout(model)
    field, _ = read_double_output((saved / "weak.frd").read_bytes(), metadata)
    q = np.asarray([value for node in order for value in field[node]], dtype=float)
    load = np.asarray(metadata["physical_tip_load_N"], dtype=float)
    residual = np.asarray(k @ q - load).ravel()
    cf = c[:, free]
    lam = np.linalg.lstsq(cf.T.toarray(), -residual[free], rcond=None)[0]
    augmented = residual[free] + np.asarray(cf.T @ lam).ravel()
    if (c.shape[0] != 20 or not np.allclose(measures, .25) or
            np.max(np.abs(c @ q)) > 1e-9 or
            np.linalg.norm(augmented) > 1e-6 * max(np.linalg.norm(load), 1.)):
        raise ValueError("saved resolved weak equilibrium/rows changed")
    points = []
    for edge in range(4):
        for alpha in (.5 - .5 / np.sqrt(3), .5 + .5 / np.sqrt(3)):
            station = (edge + alpha) * .25
            gap = weak.five_rows(model, station) @ q
            density = lam[[4 * channel + edge for channel in range(5)]]
            weight = .125
            points.append({"x_m": float(station), "edge": edge,
                           "measure_m": weight, "gap": gap.tolist(),
                           "multiplier_density": density.tolist(),
                           "local_work_J": float(weight * density @ gap)})
    row_work = float(lam @ (c @ q))
    quadrature_work = sum(row["local_work_J"] for row in points)
    work_scale = max(sum(abs(row["local_work_J"]) for row in points), 1e-12)
    if abs(row_work - quadrature_work) > 1e-10 * work_scale:
        raise AssertionError("resolved station work does not reconstruct assembled rows")
    applied = float(load @ q)
    return {"binding": binding, "stations": points, "summary": _summary(points, applied),
            "independent_multiplier_relative_error": multiplier_parity,
            "assembled_row_work_J": row_work,
            "station_row_work_error_J": abs(row_work - quadrature_work)}


def _native(owner, case: str, *, expected_rows: int = 20) -> dict:
    response = owner.solve_linear((ss.NodalAction("beam", 105, compare.CASES[case]),))
    if response.status != "completed" or response.multipliers.shape != (expected_rows,):
        raise ValueError("native complete declared-row solve required")
    q = np.r_[response.shell_displacement, response.beam_coordinates]
    state = native.reference_state(owner)
    points = []
    for (operator, slots, weight, edge, alpha), (row_ids, interpolation) in zip(
            owner._line_samples, owner._line_terms, strict=True):
        virgin = operator.evaluate(np.zeros((2, 6)), state, np.zeros(5))
        gap = virgin.gap + virgin.gap_jacobian @ q[slots]
        density = interpolation @ response.multipliers[row_ids]
        station = float(operator.reference[0, 0] +
                        operator.station * (operator.reference[1, 0] - operator.reference[0, 0]))
        points.append({"x_m": station, "edge": int(edge), "measure_m": float(weight),
                       "gap": gap.tolist(), "multiplier_density": density.tolist(),
                       "local_work_J": float(weight * density @ gap)})
    row_work = float(response.multipliers @ (owner._C @ q))
    quadrature_work = sum(row["local_work_J"] for row in points)
    work_scale = max(sum(abs(row["local_work_J"]) for row in points), 1e-12)
    if (len(points) != 8 or np.max(np.abs(owner._C @ q)) > 1e-9 or
            abs(row_work - quadrature_work) > 1e-10 * work_scale):
        raise AssertionError("native station work/weak-row reconstruction changed")
    applied = float(owner._accepted[1] @ q)
    return {"stations": points, "summary": _summary(points, applied),
            "multiplier_rows": expected_rows,
            "assembled_row_work_J": row_work,
            "station_row_work_error_J": abs(row_work - quadrature_work),
            "constraint_max": float(response.constraint_max),
            "free_equilibrium_ratio": float(response.free_equilibrium_ratio)}


def evaluate(cases: dict[str, tuple[Path, Path]]) -> dict:
    pools = threadpool_info()
    if (Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver" or
            psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(item["num_threads"] != 1 for item in pools)):
        raise ValueError("effective source/CPU/thread/JIT contract changed")
    definition = native.definition()
    _, geometry = deck.solid.build_reference()
    dimensions = geometry["geometry_m"]
    section = native.rectangular_section()
    if (definition.shell.thickness_m != dimensions["shell_thickness"] or
            definition.shell.young_pa != geometry["material"]["E_Pa"] or
            definition.shell.poisson != geometry["material"]["nu"] or
            section["area"] != geometry["section"]["area_m2"] or
            not np.isclose(section["Iy"], geometry["section"]["Iy_m4"], rtol=1e-14) or
            not np.isclose(section["Iz"], geometry["section"]["Iz_m4"], rtol=1e-14) or
            max(point[0] for point in definition.shell.coordinates_m) != dimensions["length"]):
        raise ValueError("native and resolved geometry/material/section differ")
    owner = ss.create_analysis(definition=definition)
    try:
        rows = {}
        for case in CASES:
            solid = _solid(case, *cases[case])
            beam = _native(owner, case)
            solid_x = [item["x_m"] for item in solid["stations"]]
            native_x = [item["x_m"] for item in beam["stations"]]
            if np.max(np.abs(np.asarray(solid_x) - native_x)) > 1e-14:
                raise AssertionError("resolved/native quadrature stations differ")
            rows[case] = {"resolved": solid, "native": beam}
    finally:
        owner.close()
    return {"status": "DEVELOPMENT_LOCAL_TRANSFER_NOT_QUALIFIED",
            "identity": "R8_BL5_SAVED_LOCAL_TRANSFER_DEV_V1", "cases": rows,
            "effective_runtime": {"cpu_affinity": psutil.Process().cpu_affinity(),
                                  "numba_disable_jit": int(numba.config.DISABLE_JIT),
                                  "numba_threads": int(numba.get_num_threads()),
                                  "threadpools": pools,
                                  "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": ["One coarse matched geometry and two loads only.",
                            "Continuum face traction and native tip wrench need not do identical warping work.",
                            "A weak-row work of zero is not pointwise bond fidelity or a stability proof.",
                            "The saved 2:1 auxiliary field uses another geometry and is not scored here."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, required=True)
        parser.add_argument(f"--{case}-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing development result")
    cases = {case: (getattr(args, case), getattr(args, case + "_result")) for case in CASES}
    result = evaluate(cases)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({case: {side: row[side]["summary"] for side in ("resolved", "native")}
                      for case, row in result["cases"].items()}))


if __name__ == "__main__":
    main()
