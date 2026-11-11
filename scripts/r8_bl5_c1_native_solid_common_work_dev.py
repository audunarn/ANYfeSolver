"""Check C1 native BL5 and solid hinge work on explicit common virtual fields.

This compares continuous five-channel *linearized* work at five terminal
stations. It cannot compare native broken half-cell fields or loaded accuracy.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import anyfesolver
import numba
import numpy as np
import psutil
from threadpoolctl import threadpool_info

from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator, LAW_ID
from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_plate_patch_work import local_grid
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from scripts.r8_bl5_c1_line_reference_dev import _five_channel_rows, _split_deck


ROOT = Path(__file__).resolve().parents[1]
SOLID = ROOT / ".development/bl5-c1-line-reference-20261008-v7/prepared"
TRANSLATION_SCALE_M = 1e-5
ROTATION_SCALE_RAD = 1e-4


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _local_rows(xs, ys, beam, shell_size):
    rows = []
    for x in (.9, .925, .95, .975, 1.):
        edge = min(int(np.searchsorted(xs, x, side="right")) - 1, len(xs) - 2)
        alpha = (x - xs[edge]) / (xs[edge + 1] - xs[edge])
        yzero = int(np.flatnonzero(np.isclose(ys, 0., atol=1e-14, rtol=0.))[0])
        nodes = (yzero * len(xs) + edge, yzero * len(xs) + edge + 1)
        shell_ref = np.array(((xs[edge], 0., 0.), (xs[edge + 1], 0., 0.)))
        element, cell, weight, _ = beam._locate(x)
        reference = beam.context.elements[element][1].operator.reference
        port = NativeHalfCellPort.from_reference(
            reference, beam.context.nodes[element], element, cell, weight,
            beam.context.nodal_count, beam.context.count)
        local = BL5LocalOperator.at_reference(
            shell_ref, np.zeros((2, 6)), .035, port,
            beam.context.initial.mechanical, station=alpha).evaluate(
                np.zeros((2, 6)), beam.context.initial.mechanical, np.zeros(5))
        if np.max(np.abs(local.gap)) > 2e-13:
            raise ArithmeticError("virgin native C1 gap changed")
        slots = np.r_[[6 * node + axis for node in nodes for axis in range(6)],
                      shell_size + local.beam_slots]
        rows.append((x, slots, local.gap_jacobian))
    return rows


def _mode(name, p, domain):
    x = float(p[0])
    zero = np.zeros(3)
    a, b = TRANSLATION_SCALE_M, ROTATION_SCALE_RAD
    if name == "shell_lateral_ramp":
        return (np.array((0., a * (x + .5), 0.)) if domain == "shell" else zero), zero
    if name == "beam_lateral_ramp":
        return (np.array((0., a * x, 0.)) if domain == "beam" else zero), zero
    if name == "shell_normal_ramp":
        return (np.array((0., 0., a * (x + .5))) if domain == "shell" else zero), zero
    if name == "beam_normal_ramp":
        return (np.array((0., 0., a * x)) if domain == "beam" else zero), zero
    if name == "shell_pitch_ramp":
        return zero, (np.array((0., b * (x + .5), 0.)) if domain == "shell" else zero)
    if name == "shell_roll_ramp":
        return zero, (np.array((b * (x + .5), 0., 0.)) if domain == "shell" else zero)
    if name in ("beam_pitch_constant", "beam_roll_constant", "beam_spin_constant"):
        axis = {"beam_pitch_constant": 1, "beam_roll_constant": 0,
                "beam_spin_constant": 2}[name]
        spin = np.eye(3)[axis] * b if domain == "beam" else zero
        return zero, spin
    if name in ("common_rigid_pitch", "common_rigid_spin"):
        spin = np.eye(3)[1 if name.endswith("pitch") else 2] * b
        return np.cross(spin, p), spin
    raise ValueError("unknown common virtual field")


def _native_vector(name, xs, ys, beam):
    shell_size = 6 * len(xs) * len(ys)
    q = np.zeros(shell_size + beam.context.count)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            u, spin = _mode(name, np.array((x, y, 0.)), "shell")
            q[6 * (j * len(xs) + i):6 * (j * len(xs) + i) + 6] = np.r_[u, spin]
    positions = beam.context.initial.mechanical.positions
    for node, p in enumerate(positions):
        u, spin = _mode(name, p, "beam")
        q[shell_size + 6 * node:shell_size + 6 * node + 6] = np.r_[u, spin]
    seen = set()
    for element, (_, part) in enumerate(beam.context.elements):
        for cell in (0, 1):
            left, right = beam.context.nodes[element][cell:cell + 2]
            x = float((positions[left, 0] + positions[right, 0]) / 2)
            reference = part.operator.reference
            port = NativeHalfCellPort.from_reference(
                reference, beam.context.nodes[element], element, cell, .5,
                beam.context.nodal_count, beam.context.count)
            slots = tuple(int(v) for v in port.active_beam_slots()[-3:])
            if any(v in seen for v in slots):
                raise ArithmeticError("retained cell rotation slot repeated")
            seen.update(slots)
            _, spin = _mode(name, np.array((x, 0., .0675)), "beam")
            q[shell_size + np.array(slots)] = spin
    if len(seen) != 3 * beam.n_half_cells:
        raise ArithmeticError("incomplete native retained rotation field")
    return q


def _solid_displacement(name, p, domain):
    if domain == "shell":
        reference = np.array((p[0], p[1], 0.))
    else:
        reference = np.array((p[0], 0., .0675))
    u, spin = _mode(name, reference, domain)
    return u + np.cross(spin, p - reference)


def evaluate():
    if Path(anyfesolver.__file__).resolve().parent != (ROOT / "src" / "anyfesolver").resolve():
        raise RuntimeError("wrong ANYfeSolver checkout")
    meta = json.loads((SOLID / "metadata.json").read_text(encoding="utf-8"))
    deck = SOLID / "fy.inp"
    if (_sha(deck) != meta["cases"]["fy"]["deck_sha256"] or
            meta["identity"] != "R8_BL5_C1_DETACHED_F2_FIVE_CHANNEL_DEV_V7"):
        raise ValueError("exact executed solid C1 deck required")
    lines, xyz, plate, bar = _split_deck(deck.read_text(encoding="ascii"))
    physical_rows = _five_channel_rows(xyz, plate, bar)
    if len(physical_rows) != 25:
        raise ArithmeticError("five solid channels at five stations required")
    start = lines.index("*NSET,NSET=Supports") + 1
    stop = next(j for j in range(start, len(lines)) if lines[j].startswith("*"))
    support = {int(v) for line in lines[start:stop] for v in line.split(",") if v}
    xs, ys = local_grid(8, transverse_cells=4)
    beam = build(14, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    shell_size = 6 * len(xs) * len(ys)
    native_rows = _local_rows(xs, ys, beam, shell_size)
    if [item[0] for item in native_rows] != meta["terminal_stations_m"]:
        raise ArithmeticError("native/solid station coordinates differ")
    names = ("shell_lateral_ramp", "beam_lateral_ramp", "shell_normal_ramp",
             "beam_normal_ramp", "shell_pitch_ramp", "shell_roll_ramp",
             "beam_pitch_constant", "beam_roll_constant", "beam_spin_constant",
             "common_rigid_pitch", "common_rigid_spin")
    modes = {}
    rng = np.random.default_rng(716)
    multiplier = rng.standard_normal(25)
    multiplier[np.arange(25) % 5 < 3] *= 10.
    multiplier[np.arange(25) % 5 >= 3] *= .1
    for name in names:
        q = _native_vector(name, xs, ys, beam)
        native_gap = np.concatenate([jac @ q[slots] for _, slots, jac in native_rows])
        solid_gap = np.array([sum(value * _solid_displacement(
            name, xyz[node], "shell" if node in plate else "beam")[axis - 1]
            for (node, axis), value in row.items()) for row in physical_rows])
        delta = native_gap + solid_gap
        normalized = float(np.max(np.abs(delta.reshape(5, 5)) /
                                  np.array((TRANSLATION_SCALE_M,) * 3 +
                                           (ROTATION_SCALE_RAD,) * 2)))
        work_native = float((-multiplier) @ native_gap)
        work_solid = float(multiplier @ solid_gap)
        support_max = float(max(np.max(np.abs(_solid_displacement(
            name, xyz[node], "shell" if node in plate else "beam")))
            for node in support))
        support_admissible = name not in (
            "beam_pitch_constant", "beam_roll_constant", "beam_spin_constant",
            "common_rigid_pitch", "common_rigid_spin")
        if support_admissible and support_max > 1e-13:
            raise ArithmeticError("declared supported common field moves a solid support")
        modes[name] = {"native_gap": native_gap.tolist(),
                       "solid_gap": solid_gap.tolist(),
                       "normalized_gap_difference": normalized,
                       "native_work_J": work_native,
                       "solid_work_J": work_solid,
                       "work_absolute_difference_J": abs(work_native - work_solid),
                       "solid_support_max_m": support_max,
                       "support_admissible": support_admissible}
    max_gap = max(item["normalized_gap_difference"] for item in modes.values())
    max_work = max(item["work_absolute_difference_J"] for item in modes.values())
    passed = max_gap <= 1e-7 and max_work <= 1e-10
    return {"identity": "R8_BL5_C1_NATIVE_SOLID_COMMON_WORK_DEV_V1",
            "status": "COMMON_WORK_MATCH_NOT_QUALIFIED" if passed else "COMMON_WORK_MISMATCH",
            "source_sha256": _sha(Path(__file__)),
            "solid_deck_sha256": _sha(deck), "solid_metadata_sha256": _sha(SOLID / "metadata.json"),
            "law": LAW_ID, "five_stations_m": meta["terminal_stations_m"],
            "multiplier_physical": multiplier.tolist(),
            "max_normalized_gap_difference": max_gap,
            "max_work_absolute_difference_J": max_work,
            "limits": {"normalized_gap_difference": 1e-7,
                       "work_absolute_difference_J": 1e-10},
            "modes": modes,
            "effective_runtime": {"affinity": psutil.Process().cpu_affinity(),
                                  "numba_disable_jit": int(numba.config.DISABLE_JIT),
                                  "numba_threads": numba.get_num_threads(),
                                  "threadpools": threadpool_info(),
                                  "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())},
            "limitation": "low-order common virtual fields only; no native broken-trace equivalence, load solution, physical accuracy or F03 acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous common-work result")
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "max_gap": result["max_normalized_gap_difference"],
                      "max_work_J": result["max_work_absolute_difference_J"]}))
    if result["status"] != "COMMON_WORK_MATCH_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
