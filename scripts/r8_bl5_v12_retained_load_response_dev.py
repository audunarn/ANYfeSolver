"""V12: native V5 response to the self-balanced retained-spin load component.

This linear superposition check changes only the applied body action, not the
native operator, physical gauge, source material, or accepted V5 evidence.
It is a development discriminator, not a connection-law or F03 assessment.
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
from anyfesolver import stiffened_shell as ss
from scripts import r8_bl5_body_port_probe as port
from scripts import r8_bl5_matched_native_reference as native

SAVED = ROOT / ".development/bl5-mz-common-beam-20261008/case-v5"
IDENTITY = "R8_BL5_V12_RETAINED_LOAD_RESPONSE_DEV_V1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    pools = threadpool_info()
    if (psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(pool["num_threads"] != 1 for pool in pools)):
        raise ValueError("effective CPU/thread/JIT settings differ")
    if not Path(anyfesolver.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("solver imported outside bound source checkout")
    preflight = json.loads((SAVED / "preflight.json").read_text(encoding="utf-8"))
    saved = json.loads((SAVED / "beam-solved.json").read_text(encoding="utf-8"))
    if (preflight["source_sha256"]["native"] != digest(native.__file__) or
            preflight["source_sha256"]["port"] != digest(port.__file__) or
            saved["status"] != "DEVELOPMENT_BODY_SOLVED_NOT_ACCEPTED"):
        raise ValueError("V5 source or solved-state identity changed")
    q_saved = np.asarray(saved["physical_coordinates"], dtype=float)
    full = np.asarray(saved["full_physical_load_covector"], dtype=float)
    if (q_saved.shape != (78,) or full.shape != (78,) or
            not np.isfinite(q_saved).all() or not np.isfinite(full).all()):
        raise ValueError("V5 native 78-coordinate body state changed")
    retained_slots = np.r_[30:36, 54:60]
    retained = np.zeros(78)
    retained[retained_slots] = full[retained_slots]
    complement = full-retained
    if (np.max(abs(retained[np.setdiff1d(np.arange(78),retained_slots)])) != 0 or
            np.max(abs(complement[retained_slots])) != 0):
        raise ValueError("retained-spin and complementary load roles overlap")
    owner = ss.create_analysis(definition=native.definition())
    try:
        ns = owner._shell_count
        if ns != 90 or owner._beam_count != 78:
            raise ValueError("native V5 beam layout changed")
        if np.max(abs(full-np.asarray(preflight["native_beam_load_covector"]))) > 1e-12:
            raise ValueError("saved physical action differs from V5 preflight")
        rigid = port.native_rigid_fields(owner)
        wrench = {name:port.require_balanced(load,rigid).tolist() for name,load in
                  (("full",full),("retained",retained),("complement",complement))}
        k = owner._K[ns:,ns:]
        scale = owner._coordinate_scale[ns:]
        q_retained, r_retained = port.solve_body(k,retained,scale,rigid=rigid)
        q_complement, r_complement = port.solve_body(k,complement,scale,rigid=rigid)
    finally:
        owner.close()
    q_reconstructed = q_retained+q_complement
    delta = float(np.max(abs(q_reconstructed-q_saved)))
    relative = float(np.linalg.norm(q_reconstructed-q_saved)/
                     max(np.linalg.norm(q_saved),1e-15))
    if relative > 1e-8:
        raise ValueError("two balanced actions do not reconstruct V5 native state")
    w_full = float(full @ q_saved)
    w_retained = float(retained @ q_retained)
    w_complement = float(complement @ q_complement)
    cross_a = float(retained @ q_complement)
    cross_b = float(complement @ q_retained)
    denom = max(abs(w_full),1e-12)
    if (abs(cross_a-cross_b)/denom > 1e-8 or
            abs(w_full-w_retained-w_complement-cross_a-cross_b)/denom > 1e-8 or
            abs(float(retained @ q_saved)-w_retained-cross_a)/denom > 1e-8):
        raise ValueError("native superposition/work reciprocity failed")
    result = dict(identity=IDENTITY,status="DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFIED",
        source_paths={"anyfesolver":str(Path(anyfesolver.__file__).resolve()),
                      "port":str(Path(port.__file__).resolve()),
                      "native":str(Path(native.__file__).resolve())},
        source_sha256={"port":digest(port.__file__),"native":digest(native.__file__)},
        saved_v5_sha256={name:digest(SAVED/name) for name in
                         ("preflight.json","beam-solved.json","reference-beam-solved.json")},
        effective_runtime=dict(affinity=psutil.Process().cpu_affinity(),
            numba_disable_jit=int(numba.config.DISABLE_JIT),
            numba_threads=int(numba.get_num_threads()),threadpools=pools),
        compute_inventory=dict(native_beam_RHS=2,resolved_RHS=0,external_runs=0,
                               formal_F03_runs=0),
        balanced_rigid_wrench=wrench,
        reconstructed_state_max_abs=delta,reconstructed_state_relative=relative,
        full_original_work_J=w_full,retained_only_work_J=w_retained,
        complement_only_work_J=w_complement,cross_work_each_J=cross_a,
        direct_retained_pairing_on_saved_J=float(retained @ q_saved),
        work_change_when_retained_action_removed_J=w_full-w_complement,
        original_resolved_body_work_J=float(json.loads((SAVED/"reference-beam-solved.json").read_text())["paired_work_J"]),
        retained_response=dict(equation_ratio=r_retained["equation_ratio"],
                               work_relative_error=r_retained["work_relative_error"]),
        complement_response=dict(equation_ratio=r_complement["equation_ratio"],
                                 work_relative_error=r_complement["work_relative_error"]),
        limitation="Removing a balanced native load component changes the physical action; this cannot alone allocate native-versus-C3D20 body or coupled-connection error")
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
