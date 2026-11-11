"""Compare the same declared end actions across native BL5 and saved C3D20.

The continuum end-face fields and native section wrenches have equal resultants
but need not perform identical warping work. This is a development comparison,
not a cross-solver acceptance or a selected line formulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import anyfesolver
import numba
import numpy as np
import psutil
from threadpoolctl import threadpool_info
from anyfesolver import stiffened_shell as ss
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_ccx_weak_p0_dev as deck_source
from scripts import r8_bl5_ccx_weak_p0_run_dev as external
from scripts.r8_e02_double_output import read_double_output

IDENTITY="R8_BL5_CCX_NATIVE_FY_MX_COMPARISON_DEV_V1"
CASES={"fy":(0.,100.,0.,0.,0.,0.),"mx":(0.,0.,0.,10.,0.,0.)}


def checked_saved(path,case,result_path):
    meta=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    receipt=json.loads((path/"ccx.receipt.json").read_text(encoding="utf-8"))
    prior=json.loads(result_path.read_text(encoding="utf-8"))
    deck,_=deck_source.deck(case)
    if (meta["case"]!=case or receipt["status"]!="EXITED" or
            receipt["exit_code"]!=0 or
            receipt["executable_sha256"]!=external.EXPECTED_CCX_SHA256 or
            prior.get("case")!=case or
            prior["status"]!="DEVELOPMENT_COMPARISON_NOT_QUALIFIED" or
            prior.get("development_parity_passed") is not True or
            (path/"weak.inp").read_bytes()!=deck or
            meta["deck_sha256"]!=external.digest(path/"weak.inp")):
        raise ValueError("exact saved external comparator identity required")
    checks=prior.get("development_parity_checks")
    limits=prior.get("development_parity_limits")
    expected_names={"displacement_parity","independent_equation","weak_gap",
        "free_equilibrium","ccx_raw_reaction","physical_support",
        "force_balance","moment_balance","work_balance"}
    if (not isinstance(checks,dict) or not isinstance(limits,dict) or
            set(checks)!=expected_names or set(limits)!=expected_names or
            any(not np.isfinite(checks[name]) or checks[name]>limits[name] or
                    limits[name]!=(1e-9 if name=="weak_gap" else 1e-6)
                for name in expected_names)):
        raise ValueError("saved same-law development parity checks missing or failed")
    if "saved_sha256" in prior and any(external.digest(path/name)!=value
            for name,value in prior["saved_sha256"].items()):
        raise ValueError("saved post-execution recovery input changed")
    if "     1          1     1     1  0.100000E+01" not in (
            path/"weak.sta").read_text(encoding="utf-8"):
        raise ValueError("one complete external increment required")
    displacement,_=read_double_output((path/"weak.frd").read_bytes(),meta)
    order=sorted(int(node) for node in meta["physical_nodes"])
    q=np.array([value for node in order for value in displacement[node]])
    load=np.asarray(meta["physical_tip_load_N"],dtype=float)
    return dict(path=str(path.resolve()),result_path=str(result_path.resolve()),
        files_sha256={name:external.digest(path/name)
        for name in ("weak.inp","weak.frd","weak.sta","metadata.json")},
        result_sha256=external.digest(result_path),
        applied_work_J=float(load@q),generalized_response=float(load@q/(100. if case=="fy" else 10.)),
        physical_support_wrench_about_origin=prior["physical_support_wrench_about_origin"],
        interface_beam_wrench_about_origin=prior["interface_beam_wrench_about_origin"],
        same_law_displacement_parity=prior["ccx_displacement_relative_to_independent_same_law"])


def native_result(owner,case):
    action=CASES[case]
    response=owner.solve_linear((ss.NodalAction("beam",105,action),))
    if response.status!="completed" or response.multipliers.shape!=(20,):
        raise ValueError("complete native 20-row reference-linear result required")
    q=np.r_[response.shell_displacement,response.beam_coordinates]
    applied=owner._accepted[1]
    if np.max(abs(np.asarray(owner._C@q)))>1e-9:
        raise ValueError("native weak compatibility failed")
    tip=owner._beam_ids.index(105)
    generalized=float(response.beam_coordinates[6*tip+(1 if case=="fy" else 3)])
    work=float(applied@q)
    if abs(work-(100. if case=="fy" else 10.)*generalized)>1e-9*max(abs(work),1e-12):
        raise ValueError("native declared tip action is not work conjugate")
    shell_reaction=response.reactions[:owner._shell_count].reshape(-1,6)
    physical_shell=np.asarray(owner._definition.shell.coordinates_m)
    support=np.r_[shell_reaction[:,:3].sum(axis=0),
        (shell_reaction[:,3:]+np.cross(physical_shell,shell_reaction[:,:3])).sum(axis=0)]
    exerted=-np.asarray(owner._C.T@response.multipliers).ravel()
    shell_exerted=exerted[:owner._shell_count].reshape(-1,6)
    interface=np.r_[shell_exerted[:,:3].sum(axis=0),
        (shell_exerted[:,3:]+np.cross(physical_shell,shell_exerted[:,:3])).sum(axis=0)]
    return dict(status=response.status,tip_generalized_response=generalized,
        applied_work_J=work,free_equilibrium_ratio=float(response.free_equilibrium_ratio),
        compatibility=float(response.constraint_max),
        support_wrench_about_origin=support.tolist(),
        interface_shell_wrench_about_origin=interface.tolist(),
        physical_dofs=int(len(q)),multiplier_rows=20)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fy",type=Path,required=True)
    parser.add_argument("--fy-result",type=Path,required=True)
    parser.add_argument("--mx",type=Path,required=True)
    parser.add_argument("--mx-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    result=dict(identity=IDENTITY,status="FAILED",source_sha256={
        "comparison":external.digest(__file__),"native":external.digest(native.__file__),
        "deck":external.digest(deck_source.__file__),"external_runner":external.digest(external.__file__)})
    try:
        pools=threadpool_info()
        if (psutil.Process().cpu_affinity()!=[0] or numba.config.DISABLE_JIT!=1 or
                numba.get_num_threads()!=1 or not pools or
                any(item["num_threads"]!=1 for item in pools) or
                not Path(anyfesolver.__file__).resolve().is_relative_to(ROOT/"src")):
            raise ValueError("effective native source/runtime identity changed")
        saved={name:checked_saved(path,name,report) for name,path,report in
               (("fy",args.fy,args.fy_result),("mx",args.mx,args.mx_result))}
        model=native.definition()
        _,geometry=deck_source.solid.build_reference()
        saved_model=json.loads((args.fy/"metadata.json").read_text(encoding="utf-8"))
        if geometry["model_sha256"]!=saved_model["model_sha256"]:
            raise ValueError("saved resolved model identity changed")
        dimensions=geometry["geometry_m"]
        section=native.rectangular_section()
        if (model.shell.thickness_m!=dimensions["shell_thickness"] or
                model.shell.young_pa!=geometry["material"]["E_Pa"] or
                model.shell.poisson!=geometry["material"]["nu"] or
                section["area"]!=geometry["section"]["area_m2"] or
                not np.isclose(section["Iy"],geometry["section"]["Iy_m4"],rtol=1e-14) or
                not np.isclose(section["Iz"],geometry["section"]["Iz_m4"],rtol=1e-14) or
                np.max(abs(np.asarray(model.shell.coordinates_m)[:,2]))!=0. or
                max(point[0] for point in model.shell.coordinates_m)!=dimensions["length"]):
            raise ValueError("native shell/section dimensions differ from resolved reference")
        owner=ss.create_analysis(definition=model)
        try:
            tip_position=np.asarray(owner._beam_nodes[105][0])
            if np.max(abs(tip_position-np.array((dimensions["length"],0.,
                    dimensions["centroid_z"]))))>1e-14:
                raise ValueError("native tip centroid differs from resolved bar")
            compared={name:native_result(owner,name) for name in CASES}
        finally:
            owner.close()
        result.update(status="DEVELOPMENT_COMPARISON_NOT_QUALIFIED",
            effective_runtime=dict(affinity=psutil.Process().cpu_affinity(),
                numba_disable_jit=int(numba.config.DISABLE_JIT),
                numba_threads=int(numba.get_num_threads()),threadpools=pools,
                anyfesolver_source=str(Path(anyfesolver.__file__).resolve())),
            matched_geometry=dict(shell_thickness_m=model.shell.thickness_m,
                shell_young_pa=model.shell.young_pa,shell_poisson=model.shell.poisson,
                beam_section_area_m2=section["area"],beam_iy_m4=section["Iy"],
                beam_iz_m4=section["Iz"],beam_tip_centroid_m=tip_position.tolist()),
            cases={name:dict(external=saved[name],native=compared[name],
                generalized_response_ratio_native_over_external=
                    compared[name]["tip_generalized_response"]/
                    saved[name]["generalized_response"])
                for name in CASES},
            limitations="Same declared resultant and weak P0 channel family; continuum face traction vs native section wrench differ in warping work, shell/beam body formulations and half-cell traces. No F03 or accuracy acceptance.")
    except Exception as exc:
        result["error"]=repr(exc)
    result["process_wall_seconds"]=perf_counter()-started
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"wall_seconds":result["process_wall_seconds"],
                      "ratios":{name:row["generalized_response_ratio_native_over_external"]
                      for name,row in result.get("cases",{}).items()},"error":result.get("error")}))
    if result["status"]!="DEVELOPMENT_COMPARISON_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__=="__main__":
    main()
