"""V10 BL5 cut-frame response and saved V7 warping-action discriminator.

Development diagnostic only. Reuses V9's declared two-sided body boundary,
then contrasts its retained half-cell orientation with the native cut nodal
frame. The saved V7 cut action is decomposed without another equilibrium run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import anyfesolver
import numpy as np
from scripts import r8_bl5_two_sided_port_response_dev as v9


IDENTITY = "R8_BL5_NODAL_PORT_AND_WARPING_DEV_V1"
V7 = ROOT / ".development/bl5-mz-common-beam-20261008/case-v7-nodal-port/result.json"
V5 = ROOT / ".development/bl5-mz-common-beam-20261008/case-v5/reference-beam-solved.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_nodal(side, action):
    """Cut nodal-frame moment, keeping all retained variables free."""
    definitions = v9.native.definition().beams
    _, _, nodes, pieces, _, _ = v9._beam_reference(definitions)
    element, _, origin = pieces[0 if side == "left" else 1]
    positions = np.array([nodes[node][0] for node in element.node_ids])
    frames = np.array([nodes[node][1] for node in element.node_ids])
    response = element.operator.evaluate(positions, np.zeros((3, 3)), frames,
        np.tile(np.eye(3), (2, 1, 1)), np.zeros(18),
        origin=origin, check=element.operator.guard)
    k = v9.spatial_jacobian(response.residual,
                            response.hessian + response.hessian_low)
    cut = 12 if side == "left" else 0
    remote = 0 if side == "left" else 12
    c = np.zeros((6, 42))
    c[:, remote:remote+6] = np.eye(6)
    port = np.zeros((6, 42))
    port[:, cut:cut+6] = np.eye(6)
    f = port.T @ action
    q, checks = v9._solve(k, c, f)
    pose = port @ q
    retained = 21 if side == "left" else 18
    retained_pose = np.r_[q[cut:cut+3], q[retained:retained+3]]
    work = float(f @ q)
    energy = float(q @ (k @ q))
    mismatch = abs(work-energy)/max(abs(work), abs(energy), 1e-20)
    if mismatch > 1e-8 or abs(work-float(action @ pose)) > max(abs(work), 1e-20)*1e-12:
        raise ValueError("native nodal-frame port work/energy failure")
    return dict(side=side, action=action.tolist(), nodal_pose=pose.tolist(),
        retained_pose=retained_pose.tolist(), applied_work_J=work,
        strain_work_J=energy, work_energy_relative=mismatch,
        nodal_rotation_minus_retained=(pose[3:]-retained_pose[3:]).tolist(),
        **checks)


def saved_warping(model, elements, side, saved_v7, saved_q, full_index):
    """Exact V7 discrete cut action = six-resultant lift + zero-resultant part."""
    ids, _, side_index, _ = v9._solid_half(model, elements, side)
    projection, lift, _ = v9._face_map(model, ids, side_index, .5)
    action = np.zeros(3 * len(side_index))
    for raw_node, force in saved_v7["sides"][side]["cut_port_nodal_action_N"].items():
        node = int(raw_node)
        if node not in side_index:
            raise ValueError("V7 cut port has a node outside its side")
        action[side_index[node]:side_index[node]+3] = force
    face_nodes = tuple(n for n in side_index
                       if abs(model["physical_nodes"][n][0] - .5) < 1e-14)
    force = np.sum([action[side_index[n]:side_index[n]+3] for n in face_nodes], axis=0)
    moment = np.sum([np.cross(np.asarray(model["physical_nodes"][n])-v9.CENTER,
                                action[side_index[n]:side_index[n]+3])
                     for n in face_nodes], axis=0)
    wrench = np.r_[force, moment]
    reference = np.asarray(saved_v7["sides"][side]["cut_port_wrench_N_Nm"])
    if np.max(abs(wrench-reference)) > 1e-8:
        raise ValueError("V7 cut resultant reconstruction failed")
    rigid_action = lift @ wrench
    warping_action = action-rigid_action
    residual_force = np.sum([warping_action[side_index[n]:side_index[n]+3]
                             for n in face_nodes], axis=0)
    residual_moment = np.sum([np.cross(np.asarray(model["physical_nodes"][n])-v9.CENTER,
                                    warping_action[side_index[n]:side_index[n]+3])
                              for n in face_nodes], axis=0)
    if np.max(abs(np.r_[residual_force,residual_moment])) > 1e-8:
        raise ValueError("warping cut action has nonzero resultant")
    q = np.zeros_like(action)
    for node in side_index:
        q[side_index[node]:side_index[node]+3] = saved_q[full_index[node]:full_index[node]+3]
    full_work = float(action @ q)
    rigid_work = float(rigid_action @ q)
    warping_work = float(warping_action @ q)
    if abs(full_work-rigid_work-warping_work) > max(abs(full_work),1e-15)*1e-10:
        raise ValueError("V7 cut work decomposition failed")
    return dict(side=side, cut_nodes=len(face_nodes), wrench=wrench.tolist(),
        zero_resultant_max=float(np.max(abs(np.r_[residual_force,residual_moment]))),
        saved_full_cut_work_J=full_work, rigid_six_port_work_J=rigid_work,
        zero_resultant_warping_work_J=warping_work,
        raw_nodal_force_norm_N=float(np.linalg.norm(action)),
        rigid_nodal_force_norm_N=float(np.linalg.norm(rigid_action)),
        warping_nodal_force_norm_N=float(np.linalg.norm(warping_action)),
        note="Euclidean nodal force norms are descriptive only, not a mesh-invariant dual norm")


def main(output):
    import anyfesolver._stiffened_shell_ge_experimental as ge_source
    model = v9.weak.specimen()
    _, elements = v9.weak.assemble_body(model)
    response = {}
    for name, action in (("v7_resultant", v9.ACTION),
                         ("pure_mz", np.array((0., 0., 0., 0., 0., 1.)))):
        sides = [native_nodal("left", action), native_nodal("right", -action)]
        response[name] = dict(left=sides[0], right=sides[1],
            nodal_frame_work_sum_J=sum(item["applied_work_J"] for item in sides),
            retained_port_work_sum_J=sum(v9._native(side, p)["work_J"]
                                         for side,p in (("left",action),("right",-action))),
            resolved_port_work_sum_J=sum(v9._continuum(model,elements,side,p)["work_J"]
                                         for side,p in (("left",action),("right",-action))))
    v7 = json.loads(V7.read_text(encoding="utf-8"))
    saved = json.loads(V5.read_text(encoding="utf-8"))
    # V5 saved the beam-only physical vector, ordered by model['beam_nodes'],
    # rather than the full shell+beam layout used by the coupled reference.
    beam_index = {node:3*i for i,node in enumerate(model["beam_nodes"])}
    saved_q = np.asarray(saved["physical_coordinates"])
    if saved_q.shape != (3*len(beam_index),):
        raise ValueError("saved V5 beam-only coordinate layout changed")
    warping = {side:saved_warping(model,elements,side,v7,saved_q,beam_index)
               for side in ("left","right")}
    sources = {name:dict(path=str(Path(module.__file__).resolve()),
                         sha256=sha(module.__file__)) for name,module in (
        ("anyfesolver",anyfesolver),("ge_compiler",ge_source),
        ("v9",v9),("native_fixture",v9.native),("solid_fixture",v9.solid),
        ("weak_fixture",v9.weak))}
    for name in ("anyfesolver","ge_compiler"):
        if not Path(sources[name]["path"]).is_relative_to(ROOT / "src"):
            raise ValueError("solver module imported outside bound worktree")
    result = dict(identity=IDENTITY,
        status="DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFIED",
        sources=sources, saved_inputs_sha256={"V5":sha(V5),"V7":sha(V7)},
        response=response, saved_v7_cut_action_decomposition=warping,
        caveat="nodal-frame and retained-spin loads are different generalized actions; saved V7 warping work is from the old conforming solution, not a V10 incremental response; no V5 allocation or F03/law acceptance")
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
