"""Whole-element 8:1 BL5 actual mixed responses; no stability/accuracy acceptance.

One fresh installed process per frozen level; full retained variables survive.
The caller owns serial scheduling and 600 s/8 GiB resource supervision.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import zipfile

REPO = Path("C:/Github/ANYfeSolver/.perf2-worktrees/stiffened-main-reconcile-20261001")
WHEEL_SHA = "d99c59ac0464613b15461dcd23c0b8fe158a67fe3b27192a886b6daf6aa981ea"
LOADS = {"fy": (0., 100., 0., 0., 0., 0.), "mx": (0., 0., 0., 10., 0., 0.),
         "mz": (0., 0., 0., 0., 0., 10.)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def finite(value):
        if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return {"nonfinite": repr(value)}
        return value
    value = matched.plain(value) if "matched" in globals() else value
    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False)+"\n")


def installed_identity(site, manifest):
    expected = json.loads(manifest.read_text())
    source = {p.relative_to(REPO).as_posix() for p in (REPO/"src/anyfesolver").rglob("*.py")}
    installed = {"src/"+p.relative_to(site).as_posix() for p in (site/"anyfesolver").rglob("*.py")}
    require(len(expected) == 391 and source == installed == set(expected), "exact 391-module sets required")
    wheels = list((manifest.parent/"wheels").glob("*.whl"))
    require(len(wheels) == 1 and sha(wheels[0]) == WHEEL_SHA, "pinned wheel required")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = {n for n in wheel.namelist() if n.startswith("anyfesolver/") and n.endswith(".py")}
        require(names == {n.removeprefix("src/") for n in expected}, "wheel module set differs")
        for name, digest in expected.items():
            key = name.removeprefix("src/")
            require(sha(REPO/name) == sha(site/key) == digest and
                    hashlib.sha256(wheel.read(key)).hexdigest() == digest, "module mismatch: "+name)
    return {"manifest_path": str(manifest), "manifest_sha256": sha(manifest),
            "wheel_sha256": WHEEL_SHA, "module_sha256": expected}


def settings():
    pools = threadpool_info()
    require(pools and all(p["num_threads"] == 1 for p in pools) and
            numba.get_num_threads() == numba.config.NUMBA_NUM_THREADS == 1 and
            numba.config.DISABLE_JIT, "one effective BLAS/Numba thread and JIT off required")
    return {"pools": pools, "numba_threads": numba.get_num_threads(),
            "jit_disabled": bool(numba.config.DISABLE_JIT), "requested_environment":
            {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")}}


def imports(site):
    loaded = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
              if n.split(".")[0] == "anyfesolver" and getattr(m, "__file__", None)}
    require(loaded and all(Path(p).is_relative_to(site) for p in loaded.values()), "source product import forbidden")
    return loaded


def run(args):
    global np, numba, threadpool_info, matched
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "PREPARING", "qualification_claim": False, "nx": args.nx,
               "ny": args.nx//4, "halfcells": 16*args.nx, "whole_element_ratio": "8:1", "cases": {}}
    owner = None
    try:
        require(not any(n.split(".")[0] == "anyfesolver" for n in sys.modules), "fresh process required")
        site, manifest, fixtures = (p.resolve(strict=True) for p in (args.site, args.manifest, args.fixtures))
        paths = (fixtures/"scripts", fixtures/"tests") if (fixtures/"scripts").is_dir() else (fixtures,)
        sys.path[:0] = [str(site)]+[str(p) for p in paths]
        import numpy as np
        import numba
        from threadpoolctl import threadpool_info
        from anyfesolver import stiffened_shell as api
        import b3_e4_bl5_coherent_aggregation_dev as coherent
        import b3_e4_bl5_matched_mz_native_dev as matched
        import b3_e4_bl5_matched_p0_native_refinement_dev as inventory
        inventory.np = np  # Inventory only; no historical runtime/external collector.
        dependencies = {coherent: "scripts/b3_e4_bl5_coherent_aggregation_dev.py",
            matched: "scripts/b3_e4_bl5_matched_mz_native_dev.py",
            inventory: "scripts/b3_e4_bl5_matched_p0_native_refinement_dev.py",
            sys.modules["test_stiffened_shell_l1_assembly_experimental"]:
                "tests/test_stiffened_shell_l1_assembly_experimental.py"}
        for module, relative in dependencies.items():
            path = Path(module.__file__).resolve()
            require(path.parent in paths and sha(path) == sha(REPO/relative), "source/snapshot helper mismatch: "+relative)
        summary.update(identity=installed_identity(site, manifest), settings_before=settings(),
            helper_sha256={str(Path(m.__file__).resolve()): sha(m.__file__) for m in dependencies}, runner_sha256=sha(__file__))
        imports(site)
        definition, policy = coherent.model(args.nx, args.nx//4, 16*args.nx, 2)
        owner = api.create_analysis(definition=definition)
        xyz, layout = inventory.layout(owner, definition, policy, args.nx)
        layout["halfcell_count"] = 16*args.nx
        maps = matched.rigid_maps(owner, xyz)
        matched.save_npz(output/"operators.npz", inventory.sparse_arrays(owner, maps))
        write(output/"layout.json", layout)
        write(output/"traces.json", inventory.station_inventory(owner))
        summary.update(policy=policy, counts=layout["counts"], operators_sha256=sha(output/"operators.npz"),
                       norms={"reference_length_m": 1., "work_floor_J": 1e-12,
                              "recovery_absolute_cap": 1e-8, "recovery_rtol": 1e-10,
                              "weak_gap_limit": 1e-9,
                              "pointwise_policy": "sampled continuous g5; maxima over saved quadrature stations; no exact P0 tie"},
                       preflight_before=owner.preflight())
        write(output/"summary.json", summary)
        for case, wrench in LOADS.items():
            folder = output/case
            folder.mkdir()
            load = np.zeros(owner._K.shape[0])
            tip = owner._shell_count+6*owner._beam_index[101+16*args.nx]
            load[tip:tip+6] = wrench
            matched.save_npz(folder/"input.npz", {"load": load})
            try:
                solved = owner.solve_linear((api.NodalAction("beam", 101+16*args.nx, wrench),))
            except RuntimeError as error:
                if str(error) not in {"Factor is exactly singular", "finite mixed linear solution required",
                    "mixed reference linear residual is not representable", "mixed reference linear equilibrium failed"}:
                    raise
                summary["cases"][case] = {"status": "NUMERICAL_FAILURE", "error": str(error),
                    "traceback": traceback.format_exc(), "state_unavailable": "solve failed before publishing q"}
                write(folder/"result.json", summary["cases"][case])
                continue
            q = np.r_[solved.shell_displacement, solved.beam_coordinates]
            g, ct = owner._K@q, owner._C.T@solved.multipliers
            ns, nb, members = len(owner._shell_ids), len(owner._beam_ids), len(owner._beam_parts)
            raw = {"q": q, "load": load, "multipliers": solved.multipliers, "reactions": solved.reactions,
                "Kq": g, "constraint_resisting": ct, "connection_exerted": -ct, "weak_gaps": owner._C@q,
                "full_equilibrium": g+ct-load, "xyz_m": xyz,
                "shell_poses": solved.shell_displacement.reshape(ns, 6),
                "beam_nodal_poses": solved.beam_coordinates[:6*nb].reshape(nb, 6),
                "beam_retained": solved.beam_coordinates[6*nb:].reshape(members, 24)}
            matched.save_npz(folder/"raw-state.npz", raw)
            write(folder/"raw-state.json", {"status": "RETAINED_BEFORE_PREDICATES", "layout": "../layout.json",
                "operators": "../operators.npz", "traces": "../traces.json", "sha256": sha(folder/"raw-state.npz"),
                "signs": "Kq internal resisting; Ct*lambda connection resisting; -Ct*lambda exerted; reactions ground"})
            try:
                transfer = coherent.physical_transfer(owner, solved)
            except AssertionError as error:
                summary["cases"][case] = {"status": "PHYSICAL_CHECK_FAILED", "error": str(error),
                                          "traceback": traceback.format_exc(), "raw_retained": True}
                write(folder/"result.json", summary["cases"][case])
                continue
            write(folder/"transfer.json", transfer)
            internal, exerted, ground, applied = ({d: m@v for d, m in maps.items()}
                for v in (g, -ct, solved.reactions, load))
            cuts = {d: applied[d]+exerted[d]+ground[d] for d in maps}
            work, force_scale = float(q@load), max(map(abs, wrench))
            work_scale = max(abs(work), 1e-12)
            moment0 = np.array(transfer["polynomial_transfer_moments"][0])
            checks = {"equilibrium": solved.free_equilibrium_ratio, "weak_gap": solved.constraint_max,
                "energy_work": abs(float(q@g)-work)/work_scale,
                "connection_work": abs(float(q@(-ct)))/work_scale, "ground_work": abs(float(q@solved.reactions))/work_scale,
                "beam_transfer": float(np.max(abs(exerted["beam"]-moment0)))/force_scale,
                "shell_transfer": float(np.max(abs(exerted["shell"]+moment0)))/force_scale,
                "domain_cuts": max(float(np.max(abs(v)))/force_scale for v in cuts.values()),
                "root_balance": float(np.max(abs(sum(ground.values())+sum(applied.values()))))/force_scale}
            failures = {k: repr(v) for k, v in checks.items() if not math.isfinite(v) or v > (1e-9 if "gap" in k else 1e-8)}
            if not math.isfinite(work) or work <= 0.: failures["positive_work"] = repr(work)
            if solved.status != "completed" or owner.preflight()["reference_linear_factorizations"] != 1:
                failures["completion_factor"] = repr(owner.preflight())
            record = {"status": "PHYSICAL_CHECK_FAILED" if failures else "COMPLETED_DEVELOPMENT_ONLY",
                "failures": failures, "checks": {k: v if math.isfinite(v) else repr(v) for k, v in checks.items()},
                "final_pairing_J": work, "quadratic_energy_J": .5*float(q@g), "work_scale_J": work_scale,
                "pointwise_gap_diagnostics": {"translation_max_m": transfer["pointwise_translation_gap_max_m"],
                    "director_max": transfer["pointwise_director_gap_max"], "exact_pointwise_tie_claim": False,
                    "continuous_g5": "transfer.json:station_records:five_channel_gap"},
                "wrench_scale_N_equivalent": force_scale, "wrench_origin_m": xyz[ns], "domain_internal": internal,
                "connection_exerted": exerted, "ground": ground, "applied": applied, "complete_cuts": cuts,
                "tip_pose": q[tip:tip+6], "root_internal": g[owner._fixed], "root_interface": (-ct)[owner._fixed],
                "root_ground": solved.reactions[owner._fixed], "preflight": owner.preflight()}
            write(folder/"result.json", record)
            summary["cases"][case] = record
        summary.update(status="COMPLETED_DEVELOPMENT_ONLY" if all(v["status"] == "COMPLETED_DEVELOPMENT_ONLY"
            for v in summary["cases"].values()) else "DECLARED_FAILED", settings_after=settings(),
            canonical_imports=imports(site), preflight_after=owner.preflight(),
            remaining="Finite-envelope locking/convergence, physical accuracy and mixed-system acceptance remain open")
    except Exception as error:
        summary.update(status="ERROR_PROPAGATED", failure={"type": type(error).__name__, "message": str(error),
                                                           "traceback": traceback.format_exc()})
        raise
    finally:
        if owner is not None: owner.close()
        write(output/"summary.json", summary)
    print(json.dumps({"status": summary["status"], "nx": args.nx, "counts": summary["counts"]}))
    return 0 if summary["status"] == "COMPLETED_DEVELOPMENT_ONLY" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=REPO)
    parser.add_argument("--nx", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    sys.exit(run(parser.parse_args()))
