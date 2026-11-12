"""Compare the free nine-station beam port compliance in the E01 model.

This isolates the beam and its retained/intermediate coordinates from both
shells and their point constraints.  It is a development discriminator, not
an E01 connection or physical-transfer qualification.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


INTERVALS = 8
STATIONS = tuple(101 + 2*i for i in range(1, INTERVALS + 1))
CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def native_ports():
    import numpy as np
    from scipy.sparse.linalg import splu
    from anyfesolver import _stiffened_shell_ge_experimental as ge
    from b3_e4_e01_native_point_refinement_screen import definition

    panel, _ = definition(INTERVALS, 1, 4)
    ids, index, nodes, pieces, count = ge._beam_reference(panel.beams)
    reference, stiffness, reference_error = ge._assemble_beam(
        pieces, nodes, index, count, np.ones(count))
    if ids[0] != 101 or any(node not in index for node in STATIONS):
        raise ValueError("E01 native port inventory changed")
    root = np.arange(6*index[101], 6*index[101]+6)
    free = np.setdiff1d(np.arange(count), root)
    ports = np.array([6*index[node]+d for node in STATIONS
                      for d in range(6)])
    if np.max(np.abs(reference)) > 1e-8:
        raise ValueError("nonzero native virgin beam residual")
    right = np.zeros((len(free), len(ports)))
    lookup = {int(slot): i for i, slot in enumerate(free)}
    for j, slot in enumerate(ports):
        right[lookup[int(slot)], j] = 1.
    solved = splu(stiffness[free][:, free].tocsc()).solve(right)
    full = np.zeros((count, len(ports)))
    full[free] = solved
    residual = stiffness@full
    residual[ports, np.arange(len(ports))] -= 1.
    if np.max(np.abs(residual[free])) > 1e-7:
        raise ValueError("native station solve lacks free equilibrium")
    return full[ports].tolist(), {
        "physical_nodal_dofs": 6*len(ids),
        "retained_dofs": count-6*len(ids),
        "reference_error": float(reference_error),
        "free_residual_max": float(np.max(np.abs(residual[free]))),
        "adapter_sha256": sha256(Path(ge.__file__).read_bytes()).hexdigest(),
    }


def external_ports(site: Path):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("pinned OpenSees binary requires Python 3.12")
    sys.path.insert(0, str(site))
    import openseespy.opensees as ops
    from openseespywin import opensees as core
    from b3_e4_e01_opensees_point_canary import (
        AREA, E, G, SECOND_MOMENT, SHEAR_AREA, TORSION)

    columns = []
    try:
        for station in STATIONS:
            for channel in range(6):
                ops.wipe()
                ops.model("basic", "-ndm", 3, "-ndf", 6)
                for i in range(2*INTERVALS+1):
                    ops.node(101+i, i/INTERVALS, -.5, .1)
                ops.geomTransf("Linear", 1, 0., 0., 1.)
                for i in range(2*INTERVALS):
                    ops.element("ElasticTimoshenkoBeam", 1+i,
                                101+i, 102+i, E, G, AREA, TORSION,
                                SECOND_MOMENT, SECOND_MOMENT,
                                SHEAR_AREA, SHEAR_AREA, 1)
                ops.fix(101, 1, 1, 1, 1, 1, 1)
                ops.timeSeries("Linear", 1)
                ops.pattern("Plain", 1, 1)
                load = [0.]*6
                load[channel] = 1.
                ops.load(station, *load)
                ops.constraints("Plain")
                ops.numberer("RCM")
                ops.system("BandGeneral")
                ops.test("NormUnbalance", 1e-9, 20)
                ops.algorithm("Linear")
                ops.integrator("LoadControl", 1.)
                ops.analysis("Static")
                if ops.analyze(1) != 0:
                    raise RuntimeError(f"OpenSees station {station} {CHANNELS[channel]} failed")
                columns.append([float(v) for node in STATIONS
                                for v in ops.nodeDisp(node)])
    finally:
        ops.wipe()
    matrix = [list(row) for row in zip(*columns)]
    return matrix, {
        "opensees_version": ops.version(),
        "binary_sha256": sha256(Path(core.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "opensees"), required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier E01 port comparison")
    if args.backend == "opensees" and args.site is None:
        raise ValueError("OpenSees site path required")
    matrix, metadata = (native_ports() if args.backend == "native" else
                        external_ports(args.site))
    result = {
        "classification": "E01 beam port development discriminator; not accepted",
        "backend": args.backend,
        "python": sys.version.split()[0],
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "stations": STATIONS,
        "channels": CHANNELS,
        "compliance": matrix,
        **metadata,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2,
                                      allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"backend": args.backend, "rows": len(matrix),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
