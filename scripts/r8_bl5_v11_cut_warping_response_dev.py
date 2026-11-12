"""V11: response to the saved V7 cut action and its six-resultant projection.

This is a bounded C3D20 beam-half development check. It leaves the original
V5 coupled load, native connection law, and formal F03 assessment untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import anyfesolver
import numba
import numpy as np
import psutil
from threadpoolctl import threadpool_info
from scripts import r8_bl5_two_sided_port_response_dev as ports

SAVED = ROOT / ".development/bl5-mz-common-beam-20261008/case-v7-nodal-port/result.json"
IDENTITY = "R8_BL5_V11_CUT_WARPING_RESPONSE_DEV_V1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_side(model, elements, saved, side):
    ids, _, index, k = ports._solid_half(model, elements, side)
    cut, remote = (.5, 0.) if side == "left" else (.5, 1.)
    _, lift, cut_meta = ports._face_map(model, ids, index, cut)
    c, _, remote_meta = ports._face_map(model, ids, index, remote)
    raw = saved["sides"][side]
    full = np.zeros(k.shape[0])
    for key, force in raw["cut_port_nodal_action_N"].items():
        node = int(key)
        if node not in index:
            raise ValueError("saved action contains an out-of-half node")
        full[index[node]:index[node] + 3] = force
    wrench = np.asarray(raw["cut_port_wrench_N_Nm"], dtype=float)
    rigid = lift @ wrench
    warping = full - rigid
    face = [node for node in index if abs(model["physical_nodes"][node][0] - cut) < 1e-14]
    def resultant(action):
        force = np.sum([action[index[n]:index[n]+3] for n in face], axis=0)
        moment = np.sum([np.cross(np.asarray(model["physical_nodes"][n])-ports.CENTER,
                                  action[index[n]:index[n]+3]) for n in face], axis=0)
        return np.r_[force, moment]
    if (np.max(abs(resultant(full)-wrench)) > 1e-8 or
            np.max(abs(resultant(warping))) > 1e-8):
        raise ValueError("saved full/zero-resultant action does not reconstruct")
    q_full, checks_full = ports._solve(k, c, full)
    q_rigid, checks_rigid = ports._solve(k, c, rigid)
    q_warping = q_full-q_rigid
    work_full = float(full @ q_full)
    work_rigid = float(rigid @ q_rigid)
    work_warping = float(warping @ q_warping)
    cross_a = float(rigid @ q_warping)
    cross_b = float(warping @ q_rigid)
    denominator = max(abs(work_full), abs(work_rigid), 1e-16)
    if (abs(work_full-work_rigid-work_warping-cross_a-cross_b) / denominator > 1e-8 or
            abs(cross_a-cross_b) / denominator > 1e-8 or
            abs(work_full-float(q_full @ (k @ q_full))) / denominator > 1e-8):
        raise ValueError("linear work decomposition or reciprocity failed")
    return dict(side=side, cut=cut_meta, remote=remote_meta,
        full_work_J=work_full, six_resultant_work_J=work_rigid,
        zero_resultant_self_work_J=work_warping, cross_work_each_J=cross_a,
        full_minus_six_work_J=work_full-work_rigid,
        full_over_six_work=work_full/work_rigid if work_rigid else None,
        full_action_wrench=wrench.tolist(), zero_resultant_wrench=resultant(warping).tolist(),
        full_equation=checks_full, six_equation=checks_rigid,
        full_action_norm_N=float(np.linalg.norm(full)),
        zero_resultant_action_norm_N=float(np.linalg.norm(warping)))


def main(output: Path):
    pools = threadpool_info()
    if not Path(anyfesolver.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("solver imported outside bound source checkout")
    if (psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(pool["num_threads"] != 1 for pool in pools)):
        raise ValueError("effective CPU/thread/JIT settings differ")
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    model = ports.weak.specimen()
    _, elements = ports.weak.assemble_body(model)
    sides = {name: one_side(model, elements, saved, name) for name in ("left", "right")}
    full = sum(row["full_work_J"] for row in sides.values())
    six = sum(row["six_resultant_work_J"] for row in sides.values())
    result = dict(identity=IDENTITY, status="DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFIED",
        saved_v7_sha256=digest(SAVED), source_sha256={name:digest(Path(module.__file__))
            for name,module in (("ports",ports),("weak",ports.weak),("anyfesolver",anyfesolver))},
        source_paths={name:str(Path(module.__file__).resolve()) for name,module in
            (("ports",ports),("weak",ports.weak),("anyfesolver",anyfesolver))},
        effective_runtime=dict(affinity=psutil.Process().cpu_affinity(),
            numba_disable_jit=int(numba.config.DISABLE_JIT),
            numba_threads=int(numba.get_num_threads()),threadpools=pools),
        compute_inventory=dict(resolved_half_RHS=4,native_RHS=0,external_runs=0,
                               formal_F03_runs=0),
        total_full_work_J=full,total_six_resultant_work_J=six,
        total_full_over_six_work=full/six if six else None,sides=sides,
        limitation="Incremental side-body response at V9 remote supports; no V5 body/connection causal allocation or BL5/F03 acceptance")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
