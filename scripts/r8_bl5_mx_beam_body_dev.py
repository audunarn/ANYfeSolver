"""Bounded beam-only Mx discriminator for the saved BL5 coupled reference.

Remove shell and weak connection from both bodies, retain the same section,
root clamp and declared tip torque. Resolved end-face traction and native
section moment remain distinct work fields; this diagnoses, not qualifies,
their body-level contribution to the coupled response difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from threadpoolctl import threadpool_info
from anyfesolver import stiffened_shell as ss
from scripts import r8_bl5_ccx_weak_p0_run_dev as previous
from scripts import r8_bl5_ccx_weak_p0_dev as deck
from scripts import r8_bl5_weak_continuum_reference as continuum
from scripts import r8_bl5_matched_native_reference as native

IDENTITY="R8_BL5_MX_BEAM_ONLY_BODY_DISCRIMINATOR_DEV_V1"
TORQUE_NM=10.


def wrench(positions,loads,origin=np.zeros(3)):
    values=np.zeros(6)
    for position,force in zip(positions,loads):
        values[:3]+=force
        values[3:]+=np.cross(np.asarray(position)-origin,force)
    return values


def resolved_body(model,load,k):
    order,index,_,_,_=continuum.layout(model)
    nodes=tuple(sorted(model["beam_nodes"]))
    slots=np.array([index[n]+axis for n in nodes for axis in range(3)])
    body=k[slots,:][:,slots].tocsr()
    applied=load[slots]
    fixed=np.array([3*i+axis for i,n in enumerate(nodes)
                    if abs(model["physical_nodes"][n][0])<1e-14
                    for axis in range(3)],dtype=int)
    free=np.setdiff1d(np.arange(len(slots)),fixed)
    q=np.zeros(len(slots))
    q[free]=spsolve(body[free,:][:,free],applied[free])
    if not np.isfinite(q).all():
        raise ValueError("resolved beam-only torque solve nonfinite")
    residual=np.asarray(body@q-applied).ravel()
    response=float(applied@q/TORQUE_NM)
    root=wrench([model["physical_nodes"][nodes[i//3]] for i in fixed[::3]],
                [residual[i:i+3] for i in fixed[::3]])
    tip=wrench([model["physical_nodes"][n] for n in nodes],
               applied.reshape(-1,3))
    if np.linalg.norm(root+tip)>1e-6:
        raise ValueError("resolved beam-only physical torque unbalanced")
    return dict(work_conjugate_tip_spin_rad=response,applied_work_J=float(applied@q),
        internal_work_J=float(q@(body@q)),
        free_equilibrium_ratio=float(np.linalg.norm(residual[free])/TORQUE_NM),
        support_wrench_about_origin=root.tolist(),
        applied_wrench_about_origin=tip.tolist(),
        root_fixed_translation_count=int(len(fixed)),body_dofs=int(len(slots)))


def constrained_response(body,c,load):
    """Exact weak KKT with a fully fixed shell and no added beam root support."""
    row_norms=np.sqrt(np.asarray(c.multiply(c).sum(axis=1)).ravel())
    if np.any(row_norms<=0) or not np.isfinite(row_norms).all():
        raise ValueError("finite nonzero rigid-shell constraint rows required")
    singular=np.linalg.svd((sparse.diags(1./row_norms)@c).toarray(),
                           compute_uv=False)
    system=sparse.bmat([[body,c.T],[c,None]],format="csc")
    rhs=np.r_[load,np.zeros(c.shape[0])]
    with warnings.catch_warnings():
        warnings.simplefilter("error",MatrixRankWarning)
        state=spsolve(system,rhs)
    if not np.isfinite(state).all():
        raise ValueError("rigid-shell weak KKT is nonfinite or rank deficient")
    q=state[:body.shape[0]]; lam=state[body.shape[0]:]
    residual=np.asarray(system@state-rhs).ravel()
    gap=np.asarray(c@q).ravel()
    if (np.linalg.norm(residual)/TORQUE_NM>1e-7 or
            np.max(abs(gap))>1e-9 or
            abs(float(q@(body@q))-float(load@q))/max(abs(float(load@q)),1e-12)>1e-7):
        raise ValueError("rigid-shell weak equilibrium, work or gap failed")
    return q,lam,dict(applied_work_J=float(load@q),
        internal_work_J=float(q@(body@q)),
        kkt_equation_ratio=float(np.linalg.norm(residual)/TORQUE_NM),
        weak_gap_max=float(np.max(abs(gap))),multiplier_rows=int(c.shape[0]),
        row_normalized_singular_min=float(singular[-1]),
        row_normalized_singular_max=float(singular[0]),
        row_normalized_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        row_metric_limitation="Local Euclidean row-normalized algebraic diagnostic only; mixed-unit native stability and F03 remain unassessed")


def resolved_rigid_shell(model,load,k):
    order,index,_,_,_=continuum.layout(model)
    beam_nodes=tuple(sorted(model["beam_nodes"]))
    shell_nodes=tuple(sorted(model["shell_nodes"]))
    beam_slots=np.array([index[n]+axis for n in beam_nodes for axis in range(3)])
    shell_slots=np.array([index[n]+axis for n in shell_nodes for axis in range(3)])
    c,_=continuum.constraints(model,"weak")
    q,lam,result=constrained_response(k[beam_slots,:][:,beam_slots],
        c[:,beam_slots],load[beam_slots])
    shell_action=-np.asarray(c[:,shell_slots].T@lam).ravel().reshape(-1,3)
    interface=wrench([model["physical_nodes"][n] for n in shell_nodes],shell_action)
    tip=wrench([model["physical_nodes"][n] for n in beam_nodes],
               load[beam_slots].reshape(-1,3))
    if np.linalg.norm(interface-tip)>1e-6:
        raise ValueError("resolved rigid-shell interface does not transfer tip torque")
    result.update(work_conjugate_tip_spin_rad=float(load[beam_slots]@q/TORQUE_NM),
        interface_shell_wrench_about_origin=interface.tolist(),
        applied_wrench_about_origin=tip.tolist(),body_dofs=int(len(q)))
    return result


def native_body(owner):
    ns=owner._shell_count
    body=owner._K[ns:,ns:].tocsr()
    count=body.shape[0]
    root=owner._beam_ids.index(101)
    tip=owner._beam_ids.index(105)
    fixed=np.arange(6*root,6*root+6)
    free=np.setdiff1d(np.arange(count),fixed)
    applied=np.zeros(count)
    applied[6*tip+3]=TORQUE_NM
    q=np.zeros(count)
    q[free]=spsolve(body[free,:][:,free],applied[free])
    if not np.isfinite(q).all():
        raise ValueError("native retained-coordinate beam-only solve nonfinite")
    residual=np.asarray(body@q-applied).ravel()
    response=float(q[6*tip+3])
    if abs(float(applied@q)-TORQUE_NM*response)>1e-12:
        raise ValueError("native torque is not conjugate to tip spin")
    root_position=np.asarray(owner._beam_nodes[101][0])
    root_wrench=np.r_[residual[fixed[:3]],residual[fixed[3:]]+
                      np.cross(root_position,residual[fixed[:3]])]
    tip_position=np.asarray(owner._beam_nodes[105][0])
    tip_wrench=np.r_[np.zeros(3),np.array((TORQUE_NM,0.,0.))]
    if np.linalg.norm(root_wrench+tip_wrench)>1e-6:
        raise ValueError("native beam-only physical torque unbalanced")
    return dict(tip_spin_x_rad=response,applied_work_J=float(applied@q),
        internal_work_J=float(q@(body@q)),
        free_equilibrium_ratio=float(np.linalg.norm(residual[free])/TORQUE_NM),
        support_wrench_about_origin=root_wrench.tolist(),
        applied_wrench_about_origin=tip_wrench.tolist(),
        root_fixed_pose_count=int(len(fixed)),body_dofs=int(count),
        retained_coordinate_count=int(count-6*len(owner._beam_ids)))


def native_rigid_shell(owner):
    ns=owner._shell_count
    body=owner._K[ns:,ns:].tocsr()
    c=owner._C[:,ns:].tocsr()
    applied=np.zeros(body.shape[0])
    tip=owner._beam_ids.index(105)
    applied[6*tip+3]=TORQUE_NM
    q,lam,result=constrained_response(body,c,applied)
    shell_action=-np.asarray(owner._C[:,:ns].T@lam).ravel().reshape(-1,6)
    shell_xyz=np.asarray(owner._definition.shell.coordinates_m)
    interface=np.r_[shell_action[:,:3].sum(axis=0),
        (shell_action[:,3:]+np.cross(shell_xyz,shell_action[:,:3])).sum(axis=0)]
    if np.linalg.norm(interface-np.array((0.,0.,0.,TORQUE_NM,0.,0.)))>1e-6:
        raise ValueError("native rigid-shell interface does not transfer tip torque")
    result.update(tip_spin_x_rad=float(q[6*tip+3]),
        interface_shell_wrench_about_origin=interface.tolist(),
        body_dofs=int(len(q)),retained_coordinate_count=int(len(q)-6*len(owner._beam_ids)))
    return result


def checked_saved(saved_mx, comparison):
    """Bind the exact saved load and model before any new body solve."""
    saved=json.loads((saved_mx/"metadata.json").read_text(encoding="utf-8"))
    checked=json.loads(comparison.read_text(encoding="utf-8"))
    expected,expected_meta=deck.deck("mx")
    if (saved["case"]!="mx" or (saved_mx/"weak.inp").read_bytes()!=expected or
            saved!=json.loads(json.dumps(expected_meta,allow_nan=False)) or
            checked["status"]!="DEVELOPMENT_COMPARISON_NOT_QUALIFIED" or
            checked["cases"]["mx"]["external"]["files_sha256"]["weak.inp"]!=
                previous.digest(saved_mx/"weak.inp") or
            checked["cases"]["mx"]["external"]["files_sha256"]["metadata.json"]!=
                previous.digest(saved_mx/"metadata.json")):
        raise ValueError("exact saved coupled-Mx comparison required")
    _,model=deck.solid.build_reference()
    if saved["model_sha256"]!=model["model_sha256"]:
        raise ValueError("resolved model identity changed")
    return model,np.asarray(saved["physical_tip_load_N"],dtype=float),checked


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--saved-mx",type=Path,required=True)
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    record=dict(identity=IDENTITY,status="FAILED",source_sha256={
        "body":previous.digest(__file__),"continuum":previous.digest(continuum.__file__),
        "native":previous.digest(native.__file__)})
    try:
        pools=threadpool_info()
        if (psutil.Process().cpu_affinity()!=[0] or numba.config.DISABLE_JIT!=1 or
                numba.get_num_threads()!=1 or not pools or
                any(item["num_threads"]!=1 for item in pools) or
                not Path(anyfesolver.__file__).resolve().is_relative_to(ROOT/"src")):
            raise ValueError("effective source/runtime differs from bounded development setting")
        model,load,checked=checked_saved(args.saved_mx,args.comparison)
        k,_=continuum.assemble_body(model)
        resolved=resolved_body(model,load,k)
        resolved_fixed_shell=resolved_rigid_shell(model,load,k)
        owner=ss.create_analysis(definition=native.definition())
        try:
            equivalent=native_body(owner)
            native_fixed_shell=native_rigid_shell(owner)
        finally:
            owner.close()
        coupled=checked["cases"]["mx"]
        body_ratio=equivalent["tip_spin_x_rad"]/resolved["work_conjugate_tip_spin_rad"]
        coupled_ratio=coupled["generalized_response_ratio_native_over_external"]
        rigid_shell_ratio=(native_fixed_shell["tip_spin_x_rad"]/
                           resolved_fixed_shell["work_conjugate_tip_spin_rad"])
        record.update(status="DEVELOPMENT_BODY_DISCRIMINATOR_NOT_QUALIFIED",
            effective_runtime=dict(affinity=psutil.Process().cpu_affinity(),
                numba_disable_jit=int(numba.config.DISABLE_JIT),
                numba_threads=int(numba.get_num_threads()),threadpools=pools),
            input_sha256={"saved_deck":previous.digest(args.saved_mx/"weak.inp"),
                "saved_metadata":previous.digest(args.saved_mx/"metadata.json"),
                "coupled_comparison":previous.digest(args.comparison)},
            resolved_beam_only=resolved,native_beam_only=equivalent,
            resolved_rigid_shell=resolved_fixed_shell,native_rigid_shell=native_fixed_shell,
            beam_only_native_over_resolved=float(body_ratio),
            rigid_shell_native_over_resolved=float(rigid_shell_ratio),
            coupled_native_over_resolved=float(coupled_ratio),
            difference_between_ratios=float(coupled_ratio-body_ratio),
            limitation="Beam-only comparison removes connection and shell but retains unequal face-traction/section and root-warping work. Rigid-shell comparison fixes every shell coordinate and restrains each beam along the full weak line; it is a different boundary-value problem and strongly mixes conforming versus broken trace response. Neither ratio allocates coupled error or selects BL5; Euclidean row singular values do not establish physical stability or F03.")
    except Exception as exc:
        record["error"]=repr(exc)
    record["process_wall_seconds"]=perf_counter()-started
    (args.output/"result.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({name:record.get(name) for name in ("status","process_wall_seconds",
          "beam_only_native_over_resolved","rigid_shell_native_over_resolved",
          "coupled_native_over_resolved","error")}))
    if record["status"]!="DEVELOPMENT_BODY_DISCRIMINATOR_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__=="__main__":
    main()
