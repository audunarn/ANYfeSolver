"""Compare native section loads with saved solid face tractions on common fields.

This is a load-map diagnostic, not a loaded-response or BL5 qualification test.
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

from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_plate_patch_work import local_grid
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from scripts.r8_bl5_c1_line_reference_dev import _split_deck
from scripts.r8_bl5_c1_native_solid_common_work_dev import (
    ROOT, SOLID, _native_vector, _solid_displacement,
)


MODES = (
    "shell_lateral_ramp", "beam_lateral_ramp", "shell_normal_ramp",
    "beam_normal_ramp", "shell_pitch_ramp", "shell_roll_ramp",
    "beam_roll_ramp",
    "beam_pitch_constant", "beam_roll_constant", "beam_spin_constant",
    "common_rigid_pitch", "common_rigid_spin",
)
SUPPORTED_MODES = set(MODES[:7])


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _face_loads(lines: list[str], bar: set[int]) -> dict[int, np.ndarray]:
    start = lines.index("*CLOAD") + 1
    stop = next(i for i in range(start, len(lines)) if lines[i].startswith("*"))
    loads: dict[int, np.ndarray] = {}
    for line in lines[start:stop]:
        node_text, axis_text, value_text = line.split(",")
        node, axis, value = int(node_text), int(axis_text), float(value_text)
        if node not in bar or axis not in (1, 2, 3) or not np.isfinite(value):
            raise ValueError("unexpected solid face load")
        loads.setdefault(node, np.zeros(3))[axis - 1] += value
    if not loads:
        raise ValueError("solid face has no nodal loads")
    return loads


def evaluate() -> dict:
    if Path(anyfesolver.__file__).resolve().parent != (ROOT / "src" / "anyfesolver").resolve():
        raise RuntimeError("wrong ANYfeSolver checkout")
    meta_path = SOLID / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["identity"] != "R8_BL5_C1_DETACHED_F2_FIVE_CHANNEL_DEV_V7":
        raise ValueError("exact saved C1 solid fixture required")

    xs, ys = local_grid(8, transverse_cells=4)
    beam = build(14, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    shell_size = 6 * len(xs) * len(ys)
    terminal = shell_size + 6 * beam.n_half_cells
    if abs(float(beam.context.initial.mechanical.positions[-1, 0]) - 1.) > 1e-12:
        raise ValueError("native end section has changed")

    cases = {}
    for case, axis, amplitude in (("fy", 1, 100.), ("mx", 3, 100.)):
        deck = SOLID / f"{case}.inp"
        if _sha(deck) != meta["cases"][case]["deck_sha256"]:
            raise ValueError("saved executed solid deck changed")
        lines, xyz, _, bar = _split_deck(deck.read_text(encoding="ascii"))
        loads = _face_loads(lines, bar)
        wrench = np.zeros(6)
        for node, force in loads.items():
            wrench[:3] += force
            wrench[3:] += np.cross(xyz[node], force)
        expected = np.array((0., 100., 0., -6.75, 0., 100.)) if case == "fy" else np.array((0., 0., 0., 100., 0., 0.))
        if np.max(np.abs(wrench - expected)) > 1e-8:
            raise ArithmeticError("solid face wrench differs from native section action")

        fields = {}
        for name in MODES:
            if name == "beam_roll_ramp":
                # A physical roll increasing linearly from the fixed root.
                # Its terminal section roll is 1e-4 rad, and it directly
                # exercises the Mx face-traction work that the first six
                # supported fields could not observe.
                terminal_pose = np.array((0., 0., 0., 1e-4, 0., 0.))
                solid_displacement = lambda p: np.cross(
                    np.array((1e-4, 0., 0.)), p - np.array((1., 0., .0675)))
            else:
                q = _native_vector(name, xs, ys, beam)
                terminal_pose = q[terminal:terminal + 6]
                solid_displacement = lambda p: _solid_displacement(name, p, "beam")
            native = float(amplitude * terminal_pose[axis])
            solid = float(sum(force @ solid_displacement(xyz[node])
                              for node, force in loads.items()))
            error = abs(solid - native)
            scale = max(abs(solid), abs(native), 1e-8)
            fields[name] = {
                "support_admissible": name in SUPPORTED_MODES,
                "native_section_work_J": native,
                "solid_face_work_J": solid,
                "absolute_difference_J": error,
                "scaled_difference": error / scale,
            }
        cases[case] = {
            "solid_deck_sha256": _sha(deck),
            "solid_loaded_node_count": len(loads),
            "solid_face_wrench_about_origin": wrench.tolist(),
            "modes": fields,
        }
        if max(abs(field["native_section_work_J"]) for field in fields.values()
               if field["support_admissible"]) < 1e-3:
            raise ArithmeticError("supported fields do not excite the declared load")

    supported = [field for case in cases.values() for field in case["modes"].values()
                 if field["support_admissible"]]
    max_absolute = max(field["absolute_difference_J"] for field in supported)
    max_scaled = max(field["scaled_difference"] for field in supported)
    passed = max_absolute <= 1e-10 and max_scaled <= 1e-8
    return {
        "identity": "R8_BL5_C1_COMMON_END_LOAD_WORK_DEV_V1",
        "status": "SUPPORTED_LOW_ORDER_LOAD_WORK_MATCH_NOT_QUALIFIED" if passed else "LOAD_WORK_MISMATCH",
        "source_sha256": _sha(Path(__file__)),
        "solid_metadata_sha256": _sha(meta_path),
        "native_terminal_slot": terminal,
        "limits": {"supported_absolute_difference_J": 1e-10,
                   "supported_scaled_difference": 1e-8},
        "max_supported_absolute_difference_J": max_absolute,
        "max_supported_scaled_difference": max_scaled,
        "cases": cases,
        "effective_runtime": {
            "affinity": psutil.Process().cpu_affinity(),
            "numba_disable_jit": int(numba.config.DISABLE_JIT),
            "numba_threads": numba.get_num_threads(),
            "threadpools": threadpool_info(),
            "anyfesolver_import": str(Path(anyfesolver.__file__).resolve()),
        },
        "limitation": "eleven hand-selected low-order fields, six support-admissible; no arbitrary face-warping or broken-port equivalence, loaded response, root parity, BL5/F03 acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior load-work evidence")
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "max_supported_absolute_difference_J": result["max_supported_absolute_difference_J"],
                      "max_supported_scaled_difference": result["max_supported_scaled_difference"]}))
    if result["status"] != "SUPPORTED_LOW_ORDER_LOAD_WORK_MATCH_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
