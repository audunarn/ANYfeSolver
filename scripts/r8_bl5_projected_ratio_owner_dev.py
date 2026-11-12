"""Installed complete projected BL5 reciprocal-ratio development owner.

One explicitly selected fixture per fresh process; historical guards unchanged.
No multiplier selection, uniform stability or formal F03 acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import zipfile


IDENTITY = "BL5_PROJECTED_COARSE_TRACE_RATIO_DEV_V1"
CASES = {"whole_half":16, "whole_double":64}
FIXTURE_CASE = None
NX, NY, SUPPORTS = 16, 4, 4
CELLS = SPAN = None
FIXTURE = None

def select_case(name):
    global FIXTURE_CASE, CELLS, SPAN, FIXTURE
    if FIXTURE_CASE is not None or name not in CASES:
        raise ValueError("one registered fixture per fresh process required")
    FIXTURE_CASE, CELLS = name, CASES[name]
    SPAN = 4*max(1, CELLS//NX)
    if SPAN % 4 or CELLS % SPAN or CELLS//SPAN != SUPPORTS:
        raise ValueError("coarser-trace support rule not representable")
    FIXTURE = dict(nx=NX,ny=NY,halfcells=CELLS,supports=SUPPORTS)
COORDINATE_NORM = {"length_m": 1., "width_m": .2, "ell_m": .4}
FIXED_CRITERIA = {"reference_error": 1e-3, "inverse_eigen_symmetry": 1e-10,
    "condition_budget": 1e4, "arithmetic_beta_margin": 1e-6, "variational_ceiling": (1+1e-6)**2}
BALANCE_LIMIT, GAP_LIMIT = 1e-8, 1e-9
LOADS = {"fy": (0., 100., 0., 0., 0., 0.), "mx": (0., 0., 0., 10., 0., 0.),
         "mz": (0., 0., 0., 0., 0., 10.)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    temporary = path.with_name(path.name+"."+str(os.getpid())+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def registration_identity(path, source):
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record["status"] == "REGISTERED_RATIO_DEVELOPMENT_BOUND" and
            record["identity"] == IDENTITY, "eligible reciprocal ratio independent scalar registration required")
    require(record["fixture_case"] == FIXTURE_CASE and record["fixture"] == FIXTURE and record["coordinate_norm"] == COORDINATE_NORM,
            "registered fixture and coordinate norm must match reciprocal ratio")
    require(record["fixed_criteria"] == FIXED_CRITERIA, "unchanged prospective criteria required")
    require(record["predicates"] and all(v is True for v in record["predicates"].values()) and
            math.isfinite(record["safe_case_floor"]) and record["safe_case_floor"] > 0.,
            "eligible positive independent scalar development bound required")
    norm_source = source/"scripts/r8_bl5_projected_ratio_norm_dev.py"
    require(record["source_sha256"] == sha(norm_source) and
            record["arrays_sha256"] == sha(path.with_suffix(".npz")),
            "registered independent norm source/arrays identity changed")
    require(isinstance(record["helper_sha256"], dict) and record["helper_sha256"],
            "registered independent helper identities required")
    expected_helpers={"r8_bl5_projected_auxiliary_norm_dev.py","b3_e4_bl5_riesz_controls_dev.py",
        "b3_e4_bl5_modal_reference_dev.py","b3_e4_bl5_reference_accuracy_dev.py",
        "r8_bl5_projected_reference_action_dev.py","r8_bl5_external_affine_reference_dev.py"}
    require(set(record["helper_sha256"])==expected_helpers and all(
        sha(source/"scripts"/name)==value for name,value in record["helper_sha256"].items()),
        "registered independent/transitive helper source identity changed")
    return record, {"json_sha256": sha(path), "arrays_sha256": record["arrays_sha256"],
        "source_sha256": record["source_sha256"], "source_path": str(norm_source.resolve()),
        "helper_sha256": record["helper_sha256"], "json_path": str(path.resolve()), "arrays_path": str(path.with_suffix(".npz").resolve())}


def installed_identity(site, wheel, wheel_sha, source):
    require(len(wheel_sha) == 64 and sha(wheel) == wheel_sha, "declared frozen installed wheel SHA required")
    source_files = {"anyfesolver/"+p.relative_to(source/"src/anyfesolver").as_posix(): p
                    for p in (source/"src/anyfesolver").rglob("*.py")}
    modules = {}
    with zipfile.ZipFile(wheel) as archive:
        names = {n for n in archive.namelist() if n.startswith("anyfesolver/") and n.endswith(".py")}
        require(len(names) == 417 and names == set(source_files), "complete417-module source/wheel graph required")
        installed_names = {"anyfesolver/"+p.relative_to(site/"anyfesolver").as_posix()
                           for p in (site/"anyfesolver").rglob("*.py")}
        require(installed_names == names, "complete417-module installed graph required")
        for name in sorted(names):
            data = archive.read(name)
            require(data == (site/name).read_bytes() == source_files[name].read_bytes(),
                    "source/wheel/site module mismatch: "+name)
            modules[name] = hashlib.sha256(data).hexdigest()
    return modules


def canonical_imports(site):
    imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.copy().items()
               if name.split(".")[0] == "anyfesolver" and getattr(module, "__file__", None)}
    require(imports and all(Path(p).is_relative_to(site) for p in imports.values()),
            "every product import must use declared installed site")
    return imports


def effective_settings(numba, threadpool_info):
    pools = threadpool_info()
    require(pools and all(p["num_threads"] == 1 for p in pools) and numba.get_num_threads() == 1
            and bool(numba.config.DISABLE_JIT), "effective native pools1/Numba1/JIT-off required")
    return {"pools": pools, "numba_threads": numba.get_num_threads(), "jit_disabled": bool(numba.config.DISABLE_JIT)}


def run(site, wheel, wheel_sha, source, registration, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAILED", "identity": IDENTITY, "fixture": FIXTURE, "fixture_case": FIXTURE_CASE,
        "coordinate_norm": COORDINATE_NORM, "source_sha256": sha(__file__), "cases": {},
        "formal_f03_authority_consumed": False,
        "classification": "EXPERIMENTAL_NOT_QUALIFIED; installed reference-linear development only",
        "limitations": "F03, M2, different-family qualification, growing spaces, nonlinear/material envelope, "
            "transfer accuracy and locking acceptance remain OPEN; no full-kernel or coarse-kernel transfer claim"}
    owner, earlier_error, cleanup_error = None, False, None
    try:
        site, source = site.resolve(strict=True), source.resolve(strict=True)
        require(not any(n.split(".")[0] == "anyfesolver" for n in sys.modules), "fresh process without preimported product required")
        record, registration_bound = registration_identity(registration, source)
        modules = installed_identity(site, wheel, wheel_sha, source)
        sys.path[:0] = [str(site), str(source/"scripts"), str(source/"tests")]
        import anyfesolver
        from anyfesolver import stiffened_shell as api
        import numpy as np
        import numba
        from threadpoolctl import threadpool_info
        import b3_e4_bl5_coherent_aggregation_dev as coherent
        import b3_e4_bl5_matched_mz_native_dev as matched
        import b3_e4_bl5_matched_p0_native_refinement_dev as inventory
        import r8_bl5_affine_flexible_native_dev as moments
        inventory.np = np  # Detached inventory helper has an explicit numerical dependency.
        require(coherent.LOADS == LOADS, "unchanged Fy100/Mx10/Mz10 loads required")
        require(all(Path(m.__file__).resolve() == (source/"scripts"/Path(m.__file__).name).resolve()
                    for m in (coherent, matched, inventory, moments)), "helpers must use declared source scripts")
        fixtures = {}
        for name, module in sys.modules.copy().items():
            if name.startswith("test_stiffened_shell_") and getattr(module, "__file__", None):
                path = Path(module.__file__).resolve()
                require(path == (source/"tests"/path.name).resolve(), "model fixtures must use declared source tests")
                fixtures[path.name] = sha(path)
        require("test_stiffened_shell_l1_assembly_experimental.py" in fixtures, "bound coherent model fixture required")
        settings = effective_settings(numba, threadpool_info)
        summary.update(registration=registration_bound, safe_case_floor=record["safe_case_floor"],
            fixed_criteria=FIXED_CRITERIA, wheel_sha256=wheel_sha, wheel=str(wheel.resolve()),
            source_root=str(source), site=str(site), module_sha256=modules, matching_modules=len(modules),
            helper_sha256={Path(m.__file__).name: sha(m.__file__) for m in (coherent, matched, inventory, moments)},
            fixture_sha256=fixtures, imported_anyfesolver=anyfesolver.__file__, product_imports=canonical_imports(site),
            python=sys.version, **settings)
        write(output/"identity.json", summary)
        base, policy = coherent.model(NX, NY, CELLS, 2, basis="discontinuous_p1")
        line = replace(base.lines[0], basis="discontinuous_p1_cell_average", multiplier_span=SPAN)
        definition = replace(base, lines=(line,))
        policy = dict(policy, span_halfcells=SPAN, c_lambda=SPAN/max(1,CELLS/NX), c_lambda_actual=SPAN/max(1,CELLS/NX),
            support_endpoints_m=[i/SUPPORTS for i in range(SUPPORTS+1)],
            support_membership=[list(range(a, a+SPAN)) for a in range(0, CELLS, SPAN)],
            basis="discontinuous_p1_cell_average", qualification="EXPERIMENTAL_NOT_QUALIFIED",
            density_basis=("1", "reference halfcell mean of 2*(x-center)/support_width"),
            scope="one reciprocal ratio fixture; four shell intervals per complete five-channel pair")
        owner = api.create_analysis(definition=definition)
        xyz, numbering = inventory.layout(owner, definition, policy, NX)
        numbering["halfcell_count"] = CELLS
        require(len(numbering["beam_node_ids"]) == CELLS+1 and numbering["counts"]["members"] == CELLS//2, "actual retained inventory differs")
        stations = inventory.station_inventory(owner)
        supports = []
        for index in range(SUPPORTS):
            actual = [r for r in stations if (2*int(r["beam_port"]["member_index"])+
                int(r["beam_port"]["native_cell"]))//SPAN == index]
            supports.append({"support_id": index, "interval_m": [index/SUPPORTS, (index+1)/SUPPORTS],
                "native_halfcells": sorted({2*int(r["beam_port"]["member_index"])+
                    int(r["beam_port"]["native_cell"]) for r in actual}),
                "sample_indices": [r["sample_index"] for r in actual],
                "multiplier_rows": sorted({int(row) for r in actual for row in r["multiplier_rows"]})})
        native = output/"native"
        native.mkdir()
        inputs = dict(fixture=FIXTURE, fixture_case=FIXTURE_CASE, coordinate_norm=COORDINATE_NORM, policy=policy,
            layout=numbering, xyz_m=xyz, shell=dict(thickness_m=definition.shell.thickness_m,
                young_pa=definition.shell.young_pa, poisson=definition.shell.poisson,
                q4_elements=definition.shell.q4_elements), beams=[json.loads(b.raw) for b in definition.beams],
            line=definition.lines[0].descriptor(), fixed=numbering["fixed_coordinates"],
            station_traces=stations, supports=supports, loads=LOADS, beam_tip=101+CELLS)
        maps = matched.rigid_maps(owner, xyz)
        arrays = {"fixed": owner._fixed, "free": owner._free, "coordinate_scale": owner._coordinate_scale,
            "constraint_scale": owner._constraint_scale, "shell_node_ids": owner._shell_ids,
            "beam_node_ids": owner._beam_ids, "xyz_m": xyz,
            "shell_rigid_work_map": maps["shell"], "beam_rigid_work_map": maps["beam"]}
        for name, matrix in (("K", owner._K), ("C", owner._C)):
            csr = matrix.tocsr()
            arrays.update({name+"_data": csr.data, name+"_indices": csr.indices,
                name+"_indptr": csr.indptr, name+"_shape": np.asarray(csr.shape, dtype=np.int64)})
        # Actual assembled inventory survives even if its following predicates fail.
        write(native/"inputs.json", inputs)
        matched.save_npz(native/"operators.npz", arrays)
        require(owner.capabilities["status"] == "EXPERIMENTAL_NOT_QUALIFIED", "qualification boundary changed")
        ns, nb, nm = (NX+1)*(NY+1), CELLS+1, CELLS//2
        physical, full = 6*(ns+nb), 6*(ns+nb)+24*nm
        expected_counts=dict(shell_nodes=ns,beam_nodes=nb,physical_dofs=physical,
            members=nm,retained_cell_rotations=6*nm,retained_resultants=18*nm,
            retained_dofs=24*nm,full_q=full,multipliers=10*SUPPORTS)
        require(owner._K.shape == (full,full) and owner._C.shape == (40,full)
            and numbering["counts"] == expected_counts, "actual reciprocal physical/retained accounting required")
        require(tuple(owner._beam_ids) == tuple(range(101,102+CELLS)), "fixed beam namespace required")
        require(len(owner._fixed)==35 and len(owner._free)==full-35 and
            len(set(owner._fixed)|set(owner._free))==full and not set(owner._fixed)&set(owner._free),
            "complete fixed/free partition required")
        require(all(s["native_halfcells"]==list(range(SPAN*s["support_id"],SPAN*(s["support_id"]+1)))
            and len(s["multiplier_rows"])==10 for s in supports) and
            len({row for s in supports for row in s["multiplier_rows"]})==40,
            "actual complete projected support/station inventory required")
        require(np.allclose(xyz[:ns],[(i/NX,.2*j/NY,0.) for j in range(NY+1) for i in range(NX+1)],rtol=0.,atol=1e-13),
            "unchanged physical shell geometry required")
        require(np.isfinite(owner._K.data).all() and np.isfinite(owner._C.data).all(), "finite actual sparse operators required")
        summary["native_input_sha256"] = {"inputs.json": sha(native/"inputs.json"), "operators.npz": sha(native/"operators.npz")}
        states = {}
        for case, wrench in LOADS.items():
            case_dir = native/case
            case_dir.mkdir()
            result = owner.solve_linear((api.NodalAction("beam", 101+CELLS, wrench),))
            q = np.r_[result.shell_displacement, result.beam_coordinates]
            load = np.array(owner._accepted[1], copy=True)
            lam, reactions = np.array(result.multipliers, copy=True), np.array(result.reactions, copy=True)
            g, ct = owner._K@q, owner._C.T@lam
            raw = dict(q=q, load=load, multipliers=lam, reactions=reactions, Kq=g,
                constraint_resisting=ct, connection_exerted=-ct, weak_gaps=owner._C@q,
                full_equilibrium=g+ct-load, shell_poses=result.shell_displacement.reshape(ns, 6),
                beam_nodal_poses=result.beam_coordinates[:6*nb].reshape(nb, 6),
                beam_retained=result.beam_coordinates[6*nb:].reshape(nm, 24),
                shell_node_ids=owner._shell_ids, beam_node_ids=owner._beam_ids,
                coordinate_scale=owner._coordinate_scale, constraint_scale=owner._constraint_scale,
                xyz_m=xyz, external_ids=[r["external_node"] for r in numbering["typed_physical_nodes"]])
            matched.save_npz(case_dir/"raw-state.npz", raw)
            write(case_dir/"raw-state.json", {"status": result.status, "generation": result.generation,
                "arrays_sha256": sha(case_dir/"raw-state.npz"), "counts": numbering["counts"],
                "sign_provenance": "Kq internal resisting; +Ct lambda connection resisting; -Ct lambda exerted; reactions actual ground"})
            require(result.status == "completed", "public installed reference-linear solve did not complete: "+case)
            require(all(np.isfinite(v).all() for v in (q, load, lam, reactions, g, ct)), "finite completed actual state required")
            transfer = coherent.physical_transfer(owner, result)
            exact = moments.exact_transfer_moments(transfer["station_records"])
            region = moments.exact_transfer_moments(transfer["station_records"], (.25, .75))[0]
            applied = {d: matrix@load for d, matrix in maps.items()}
            exerted = {d: matrix@(-ct) for d, matrix in maps.items()}
            ground = {d: matrix@reactions for d, matrix in maps.items()}
            work = float(q@load)
            scale, work_scale = max(map(abs, wrench)), max(abs(work), 1e-12)
            checks = dict(equilibrium_ratio=float(result.free_equilibrium_ratio), weak_gap=float(result.constraint_max),
                energy_work=abs(float(q@g)-work)/work_scale, compatible_work=abs(float(q@ct))/work_scale,
                ground_work=abs(float(q@reactions))/work_scale,
                domain_cuts=max(float(np.max(np.abs(applied[d]+exerted[d]+ground[d])))/scale for d in maps),
                global_balance=float(np.max(np.abs(sum(ground.values())+sum(applied.values()))))/scale,
                local_pullback=float(transfer["local_pullback_max_abs"]),
                beam_physical_workmap_scaled=float(np.max(np.abs(exerted["beam"]-np.asarray(exact[0]))))/scale,
                shell_physical_workmap_scaled=float(np.max(np.abs(exerted["shell"]+np.asarray(exact[0]))))/scale)
            case_record = dict(status=result.status, equilibrium=result.free_equilibrium_ratio,
                weak_gap=result.constraint_max, work_J=work, physical_checks=checks,
                physical_transfer=transfer, exact_transfer_moments=exact, exact_region_wrench=region,
                ground_wrenches=ground, domain_wrenches=dict(applied=applied, connection_exerted=exerted, ground=ground),
                scales=dict(length_m=1., force_N=scale, moment_Nm=scale, work_J=work_scale, work_floor_J=1e-12))
            write(case_dir/"physical-actions.json", case_record)
            summary["cases"][case] = case_record
            states.update({case+"_q": q, case+"_multipliers": lam, case+"_reactions": reactions, case+"_load": load})
            for name, value in checks.items():
                limit = GAP_LIMIT if name == "weak_gap" else BALANCE_LIMIT
                require(math.isfinite(value) and value <= limit, f"{case} {name} exceeds unchanged physical budget {limit}: {value}")
        matched.save_npz(native/"states.npz", states)
        summary.update(factorization_count=owner._linear_factorizations, accurate_refinements=owner._linear_refinements,
            counts=numbering["counts"], product_imports=canonical_imports(site), **effective_settings(numba, threadpool_info))
        require(owner._linear_factorizations == 1 and owner._linear_refinements == 6,
                "one factorization and two accurate corrections per fixed case required")
        summary["predicates"] = dict(actual_inventory=True, completed_finite=True, physical_work_and_balance=True,
            equilibrium=True, weak_gap=True, one_factorization=True, two_accurate_corrections_per_case=True)
        summary["status"] = "COMPLETED_INSTALLED_DEVELOPMENT_ONLY"
        return summary
    except BaseException as error:
        earlier_error = True
        summary["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                cleanup_error = error
                summary["status"] = "FAILED"
                summary["cleanup_error"] = traceback.format_exc()
        try:
            write(output/"summary.json", summary)
        except BaseException:
            if not earlier_error and cleanup_error is None:
                raise
        if cleanup_error is not None and not earlier_error:
            raise cleanup_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "wheel", "source", "registration", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--wheel-sha", required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    args = parser.parse_args()
    select_case(args.case)
    result = run(args.site, args.wheel, args.wheel_sha, args.source, args.registration, args.output_dir)
    print(json.dumps({name: result[name] for name in ("status", "counts", "factorization_count", "accurate_refinements", "predicates")}))
