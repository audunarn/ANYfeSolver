"""Bounded independent B3/OpenSees beam control for the E01 point panel.

Run each backend in its installed Python environment. This compares the same
straight section and six tip-load channels without shell or connection terms.
It is a development discriminator, not E01 qualification.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


INTERVALS = 8
LOADS = tuple((name, tuple(value if i == axis else 0. for i in range(6)))
              for axis, (name, value) in enumerate((
                  ("Fx", 1000.), ("Fy", 1000.), ("Fz", 1000.),
                  ("Mx", 100.), ("My", 100.), ("Mz", 100.))))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _balance(root, applied, root_xyz, tip_xyz):
    arm = tuple(tip_xyz[i] - root_xyz[i] for i in range(3))
    moment = _cross(arm, applied[:3])
    force_error = max(abs(root[i] + applied[i]) for i in range(3))
    moment_error = max(abs(root[i + 3] + applied[i + 3] + moment[i])
                       for i in range(3))
    return force_error, moment_error


def native_cases():
    import numpy as np
    from scipy.sparse.linalg import splu
    from anyfesolver import _stiffened_shell_ge_experimental as ge
    from b3_e4_e01_native_point_refinement_screen import definition

    panel, _tip = definition(INTERVALS, 1, 4)
    ids, index, nodes, pieces, count = ge._beam_reference(panel.beams)
    reference, stiffness, reference_error = ge._assemble_beam(
        pieces, nodes, index, count, np.ones(count))
    root_id, tip_id = 101, 101 + 2 * INTERVALS
    if ids[0] != root_id or ids[-1] != tip_id:
        raise ValueError("native beam node inventory mismatch")
    root = np.arange(6 * index[root_id], 6 * index[root_id] + 6)
    tip = np.arange(6 * index[tip_id], 6 * index[tip_id] + 6)
    free = np.setdiff1d(np.arange(count), root)
    factor = splu(stiffness[free][:, free].tocsc())
    results = []
    for name, row in LOADS:
        applied = np.zeros(count)
        applied[tip] = row
        q = np.zeros(count)
        q[free] = factor.solve((applied - reference)[free])
        residual = np.asarray(stiffness @ q + reference - applied)
        tip_pose = q[tip]
        support = residual[root]
        force_error, moment_error = _balance(
            support, row, nodes[root_id][0], nodes[tip_id][0])
        applied_work = float(np.dot(applied, q))
        internal_work = float(np.dot(q, stiffness @ q + reference))
        work_error = abs(internal_work - applied_work)
        free_error = float(np.max(np.abs(residual[free])))
        if (not np.isfinite(q).all() or free_error > 1e-5
                or force_error > 1e-5 or moment_error > 1e-5
                or work_error > 1e-8 * max(abs(applied_work), 1e-12)):
            raise ValueError(f"native {name} equilibrium/work failed: "
                             f"{free_error=:.6g}, {force_error=:.6g}, "
                             f"{moment_error=:.6g}, {work_error=:.6g}")
        results.append({"load": name, "applied": row,
                        "tip_pose": tip_pose.tolist(),
                        "root_reaction": support.tolist(),
                        "free_error": free_error,
                        "force_error": force_error,
                        "moment_error": moment_error,
                        "work_error": work_error})
    return results, {
        "ge_adapter_sha256": sha256(Path(ge.__file__).read_bytes()).hexdigest(),
        "reference_error": reference_error,
        "physical_nodal_dofs": 6 * len(ids),
        "retained_dofs": count - 6 * len(ids),
    }


def opensees_cases(site: Path):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("pinned OpenSees binary requires Python 3.12")
    sys.path.insert(0, str(site))
    import openseespy.opensees as ops
    from openseespywin import opensees as core
    from b3_e4_e01_opensees_point_canary import (
        AREA, E, G, SECOND_MOMENT, SHEAR_AREA, TORSION)

    root_id, tip_id = 101, 101 + 2 * INTERVALS
    root_xyz = (0., -.5, .1)
    tip_xyz = (2., -.5, .1)
    results = []
    try:
        for name, row in LOADS:
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            for i in range(2 * INTERVALS + 1):
                ops.node(101 + i, i / INTERVALS, -.5, .1)
            ops.geomTransf("Linear", 1, 0., 0., 1.)
            for i in range(2 * INTERVALS):
                ops.element("ElasticTimoshenkoBeam", 1 + i,
                            101 + i, 102 + i, E, G, AREA, TORSION,
                            SECOND_MOMENT, SECOND_MOMENT,
                            SHEAR_AREA, SHEAR_AREA, 1)
            ops.fix(root_id, 1, 1, 1, 1, 1, 1)
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            ops.load(tip_id, *row)
            ops.constraints("Plain")
            ops.numberer("RCM")
            ops.system("BandGeneral")
            ops.test("NormUnbalance", 1e-8, 20)
            ops.algorithm("Linear")
            ops.integrator("LoadControl", 1.)
            ops.analysis("Static")
            status = ops.analyze(1)
            if status != 0:
                raise RuntimeError(f"OpenSees {name} analyze returned {status}")
            ops.reactions()
            tip_pose = tuple(float(v) for v in ops.nodeDisp(tip_id))
            support = tuple(float(v) for v in ops.nodeReaction(root_id))
            force_error, moment_error = _balance(
                support, row, root_xyz, tip_xyz)
            applied_work = sum(a * b for a, b in zip(row, tip_pose))
            internal_work = 0.
            for i in range(2 * INTERVALS):
                forces = tuple(float(v) for v in ops.eleResponse(
                    1 + i, "globalForce"))
                if len(forces) != 12:
                    raise ValueError("twelve beam end-force rows required")
                poses = (ops.nodeDisp(101 + i), ops.nodeDisp(102 + i))
                internal_work += sum(poses[j][k] * forces[6 * j + k]
                                     for j in range(2) for k in range(6))
            work_error = abs(internal_work - applied_work)
            if (not all(map(math.isfinite, (*tip_pose, *support)))
                    or force_error > 1e-5 or moment_error > 1e-5
                    or work_error > 1e-8 * max(abs(applied_work), 1e-12)):
                raise ValueError(f"external {name} balance/work failed: "
                                 f"{force_error=:.6g}, {moment_error=:.6g}, "
                                 f"{work_error=:.6g}")
            results.append({"load": name, "applied": row,
                            "tip_pose": tip_pose,
                            "root_reaction": support,
                            "force_error": force_error,
                            "moment_error": moment_error,
                            "work_error": work_error})
    finally:
        ops.wipe()
    return results, {"opensees": ops.version(),
                     "binary_sha256": sha256(Path(core.__file__).read_bytes()).hexdigest()}


def main(backend: str, output: Path, site: Path | None = None):
    if output.exists():
        raise FileExistsError("preserve earlier beam control")
    record = {"classification": "E01 beam-only development control; not accepted",
              "backend": backend, "python": sys.version.split()[0],
              "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "beam_elements": INTERVALS,
              "external_beam_segments": 2 * INTERVALS,
              "cases": [], "failure": None}
    try:
        if backend == "native":
            cases, metadata = native_cases()
        else:
            if site is None:
                raise ValueError("OpenSees site path required")
            cases, metadata = opensees_cases(site)
        record.update(metadata)
        record["cases"] = cases
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
    output.write_text(json.dumps(record, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "opensees"), required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = main(args.backend, args.output, args.site)
    print(json.dumps({"failure": result["failure"],
                      "completed_cases": len(result["cases"])}))
    if result["failure"]:
        raise SystemExit(1)
