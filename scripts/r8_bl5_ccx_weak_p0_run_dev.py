"""Bounded external development comparison for the explicit BL5 weak-P0 law.

This checks that a CalculiX *EQUATION representation reproduces the same
resolved C3D20 weak problem. It does not qualify the native B3-GE connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scripts import r8_bl5_ccx_weak_p0_dev as specimen
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as weak
from scripts.r8_e02_double_output import read_double_output

IDENTITY = "R8_BL5_CCX_WEAK_P0_EXECUTION_DEV_V1"
CCX = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe")
MAX_CCX_SECONDS = 30.
MAX_TOTAL_SECONDS = 60.
MAX_TREE_RSS_BYTES = 2 * 1024**3
EXPECTED_CCX_SHA256 = solid.CCX_SHA256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def executable_receipt(case_dir, case):
    if not CCX.is_file() or digest(CCX) != EXPECTED_CCX_SHA256:
        raise ValueError("pinned CalculiX 2.23 binary identity changed")
    env = os.environ.copy()
    env.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")})
    windows = Path(env.get("SystemRoot", "C:/Windows"))
    env["PATH"] = os.pathsep.join(map(str,(CCX.parent,windows/"System32",windows)))
    parent = psutil.Process()
    parent.cpu_affinity([0])
    if parent.cpu_affinity() != [0] or any(env[name] != "1" for name in
           ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")):
        raise ValueError("effective development resource settings changed")
    receipt = dict(identity=IDENTITY, case=case, executable=str(CCX),
                   executable_sha256=digest(CCX), command=[str(CCX), "-i", "weak"],
                   cwd=str(case_dir.resolve()), affinity=parent.cpu_affinity(),
                   environment={name:env[name] for name in ("OPENBLAS_NUM_THREADS",
                       "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                       "NUMBA_DISABLE_JIT", "PATH")}, max_ccx_seconds=MAX_CCX_SECONDS,
                   max_total_seconds=MAX_TOTAL_SECONDS,
                   max_tree_rss_bytes=MAX_TREE_RSS_BYTES,
                   status="NOT_STARTED")
    started = perf_counter()
    try:
        with (case_dir / "ccx.stdout.txt").open("wb") as stdout, (case_dir / "ccx.stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(receipt["command"], cwd=case_dir,
                                       stdout=stdout, stderr=stderr, env=env)
            child = psutil.Process(process.pid)
            peak = 0
            status = None
            while process.poll() is None:
                members = [child]
                try:
                    members += child.children(recursive=True)
                    rss = sum(member.memory_info().rss for member in members)
                    peak = max(peak, rss)
                except psutil.Error:
                    pass
                if peak > MAX_TREE_RSS_BYTES:
                    status = "MEMORY_LIMIT"
                elif perf_counter()-started > MAX_CCX_SECONDS:
                    status = "TIME_LIMIT"
                if status:
                    for member in reversed(members):
                        try: member.kill()
                        except psutil.Error: pass
                    break
                sleep(.05)
            if status:
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            receipt.update(status=status or ("EXITED" if process.returncode == 0 else "FAILED"),
                           exit_code=process.returncode, sampled_peak_tree_rss_bytes=peak,
                           ccx_process_wall_seconds=perf_counter()-started)
    except Exception as exc:
        receipt.update(status="SUPERVISOR_ERROR", error=repr(exc),
                       ccx_process_wall_seconds=perf_counter()-started)
    finally:
        (case_dir / "ccx.receipt.json").write_text(
            json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return receipt


def compare(case_dir, metadata):
    if "     1          1     1     1  0.100000E+01" not in (
            case_dir/"weak.sta").read_text(encoding="utf-8"):
        raise ValueError("one complete accepted external increment required")
    _, model = solid.build_reference()
    k, _ = weak.assemble_body(model)
    c, _ = weak.constraints(model, "weak")
    order, index, fixed, free, _ = weak.layout(model)
    load = np.asarray(metadata["physical_tip_load_N"],dtype=float)
    cf = c[:,free]
    system = sparse.bmat([[k[free,:][:,free],cf.T],[cf,None]],format="csc")
    solution = spsolve(system,np.r_[load[free],np.zeros(c.shape[0])])
    reference = np.zeros(k.shape[0]); reference[free] = solution[:len(free)]
    lagrange = solution[len(free):]
    frd = case_dir / "weak.frd"
    observed, reactions = read_double_output(frd.read_bytes(),metadata)
    actual = np.array([component for node in order for component in observed[node]])
    constraint = np.asarray(c @ actual).ravel()
    residual = np.asarray(k @ actual - load).ravel()
    fitted = np.linalg.lstsq(cf.T.toarray(),-residual[free],rcond=None)[0]
    augmented = residual + np.asarray(c.T @ fitted).ravel()
    root = np.array([component for node in model["root_nodes"] for component in reactions[node]])
    connection = -np.asarray(c.T @ fitted).ravel()
    constraint_force = np.asarray(c.T @ fitted).ravel()
    physical_support = root + constraint_force[fixed]
    independent_augmented = np.asarray(k @ reference - load + c.T @ lagrange).ravel()
    beam_action = np.zeros(6)
    center = np.array((0.,0.,0.))
    for node in model["beam_nodes"]:
        force = connection[index[node]:index[node]+3]
        beam_action[:3] += force
        beam_action[3:] += np.cross(np.asarray(model["physical_nodes"][node])-center,force)
    root_wrench = np.zeros(6)
    raw_root_wrench = np.zeros(6)
    for j,node in enumerate(model["root_nodes"]):
        force = physical_support[3*j:3*j+3]
        raw = root[3*j:3*j+3]
        root_wrench[:3] += force
        root_wrench[3:] += np.cross(np.asarray(model["physical_nodes"][node])-center,force)
        raw_root_wrench[:3] += raw
        raw_root_wrench[3:] += np.cross(np.asarray(model["physical_nodes"][node])-center,raw)
    tip_wrench = np.zeros(6)
    for node in model["beam_nodes"]:
        force = load[index[node]:index[node]+3]
        tip_wrench[:3] += force
        tip_wrench[3:] += np.cross(np.asarray(model["physical_nodes"][node])-center,force)
    denom = max(float(np.linalg.norm(reference)),1e-14)
    result=dict(status="DEVELOPMENT_COMPARISON_NOT_QUALIFIED",
        ccx_displacement_relative_to_independent_same_law=float(np.linalg.norm(actual-reference)/denom),
        ccx_displacement_max_abs=float(np.max(abs(actual-reference))),
        independent_equation_residual=float(np.linalg.norm(system @ solution - np.r_[load[free],np.zeros(c.shape[0])])),
        physical_weak_gap_max=float(np.max(abs(constraint))),
        free_augmented_residual_relative=float(np.linalg.norm(augmented[free])/max(np.linalg.norm(load),1.)),
        ccx_raw_rf_relative_to_body_internal=float(np.linalg.norm(root-residual[fixed])/max(np.linalg.norm(load),1.)),
        physical_support_relative_to_independent=float(np.linalg.norm(physical_support-independent_augmented[fixed])/max(np.linalg.norm(load),1.)),
        physical_multiplier_relative_to_independent=float(np.linalg.norm(fitted-lagrange)/max(np.linalg.norm(lagrange),1.)),
        interface_beam_wrench_about_origin=beam_action.tolist(),
        ccx_raw_rf_wrench_about_origin=raw_root_wrench.tolist(),
        physical_support_wrench_about_origin=root_wrench.tolist(),
        tip_wrench_about_origin=tip_wrench.tolist(),
        global_wrench_balance_about_origin=(root_wrench+tip_wrench).tolist(),
        interface_work_J=float(connection @ actual),
        full_strain_work_J=float(actual @ (k @ actual)),
        applied_work_J=float(load @ actual),
        model_dofs=int(k.shape[0]),physical_weak_rows=int(c.shape[0]),
        limitation="Conforming resolved C3D20 lacks independent native half-cell side traces; same-law deck parity does not establish native accuracy or F03 acceptance")
    denominator=max(abs(result["applied_work_J"]),1e-12)
    balance=np.asarray(result["global_wrench_balance_about_origin"])
    tip=np.asarray(result["tip_wrench_about_origin"])
    checks=dict(
        displacement_parity=result["ccx_displacement_relative_to_independent_same_law"],
        independent_equation=result["independent_equation_residual"]/max(np.linalg.norm(load),1.),
        weak_gap=result["physical_weak_gap_max"],
        free_equilibrium=result["free_augmented_residual_relative"],
        ccx_raw_reaction=result["ccx_raw_rf_relative_to_body_internal"],
        physical_support=result["physical_support_relative_to_independent"],
        force_balance=float(np.max(abs(balance[:3]))/max(np.linalg.norm(tip[:3]),1.)),
        moment_balance=float(np.max(abs(balance[3:]))/max(np.linalg.norm(tip[3:]),1.)),
        work_balance=abs(result["full_strain_work_J"]-result["applied_work_J"])/denominator)
    limits={name:(1e-9 if name=="weak_gap" else 1e-6) for name in checks}
    result["development_parity_checks"]=checks
    result["development_parity_limits"]=limits
    result["development_parity_passed"]=all(np.isfinite(value) and value<=limits[name]
        for name,value in checks.items())
    if not result["development_parity_passed"]:
        result["status"]="DEVELOPMENT_PARITY_FAILED"
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=sorted(specimen.LOADS),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    started=perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    deck,metadata=specimen.deck(args.case)
    (args.output/"weak.inp").write_bytes(deck)
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    receipt=executable_receipt(args.output,args.case)
    result=dict(identity=IDENTITY,case=args.case,source_sha256={
        "runner":digest(__file__),"specimen":digest(specimen.__file__)},
        deck_sha256=digest(args.output/"weak.inp"),ccx_receipt=receipt,
        status="EXTERNAL_EXECUTION_FAILED")
    try:
        if receipt["status"] == "EXITED":
            result.update(compare(args.output,metadata))
        if perf_counter()-started > MAX_TOTAL_SECONDS:
            result["status"]="TOTAL_TIME_LIMIT"
    except Exception as exc:
        result.update(status="COMPARISON_FAILED",error=repr(exc))
    result["total_process_wall_seconds"]=perf_counter()-started
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result.get(key) for key in ("case","status","total_process_wall_seconds",
        "ccx_displacement_relative_to_independent_same_law","error")},allow_nan=False))
    if result["status"] != "DEVELOPMENT_COMPARISON_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
