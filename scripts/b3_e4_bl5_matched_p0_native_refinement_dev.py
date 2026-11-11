"""Detached native inventories for three fixed BL5 c2 refinement levels.

One compiled owner at nx16/32/64 (ny=nx/4, halfcells=2nx, span4) solves
Fy100N/Mx10Nm/Mz10Nm with one cached reference factor. Retain physical and
mixed retained states separately, full sparse operators/actions and typed IDs.
Optional external P0 comparisons share nodal fields, not retained rotations.
No mechanics, criteria, mesh policy, inverse or qualification gate is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import zipfile


LEVELS = (16, 32, 64)
LOADS = {"fy": (0., 100., 0., 0., 0., 0.), "mx": (0., 0., 0., 10., 0., 0.),
         "mz": (0., 0., 0., 0., 0., 10.)}
WHEEL_SHA = "0c45666609355e654cb726a0e4e0780d386bccfaafef1209ffd994bdd3b9d09c"
COHERENT_SHA = "9bd47696851110764486016f7d91f3de786818e87e4da88829dd76d5931f1fce"
MATCHED_SHA = "92fa4515f1c61008f24bafe932e8fa2c50ea298903f2a1e5499e5be6c7dfaefb"
EXTERNAL_REFINEMENT_SHA = "0e5435e0d38952642662eb1aad7dd374c341c866c9d099c87d2e66ce135a21c0"
SOURCE_ROOT = Path("C:/Github/ANYfeSolver/.perf2-worktrees/b3-ge-e4-pl-coupling")
BALANCE_LIMIT, GAP_LIMIT = 1e-8, 1e-9


class NativeReferenceContractError(ValueError):
    pass


class NativePhysicalCheckFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise NativeReferenceContractError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(site):
    """Select the canonical installed product before importing any numerical code."""
    global np, numba, stiffened_shell, coherent, matched, threadpool_info
    site = site.resolve(strict=True)
    fixtures = (site.parent/"fixtures").resolve(strict=True)
    require(not any(n.split('.')[0] == "anyfesolver" for n in sys.modules),
            "fresh process without preimported product required")
    sys.path.insert(0, str(fixtures))
    sys.path.insert(0, str(site))
    import numpy as np
    import numba
    from threadpoolctl import threadpool_info
    from anyfesolver import stiffened_shell
    import b3_e4_bl5_coherent_aggregation_dev as coherent
    import b3_e4_bl5_matched_mz_native_dev as matched
    require(sha(coherent.__file__) == COHERENT_SHA and sha(matched.__file__) == MATCHED_SHA,
            "unchanged coherent model/physical-transfer and matched work/field services required")
    module, path, service, service_path, reference, reference_path = matched.reference_services()
    for source, expected in ((path, "6e1b3f25c2b479e1ade3700b57d996cef2f9709812a72108ce7e6a41e346293a"),
                             (service_path, "506c5259afddfab1872c50c2a2544a7cb706b4cf14e6e7c6b4771d80a0ff98cc"),
                             (reference_path, "5d3616e5fc6f198adc8c0442e01c27c5b7b719e73059f5088561d0027e826f47")):
        require(sha(source) == expected, "unchanged physical reference constants/services required")
    return site, fixtures, service, reference


def canonical_imports(site):
    imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.copy().items()
               if name.split('.')[0] == "anyfesolver" and getattr(module, "__file__", None)}
    require(imports and all(Path(path).is_relative_to(site) for path in imports.values()),
            "all product imports must resolve to the canonical installed site")
    return imports


def installed_identity(site, fixtures):
    identity_path = site.parent/"identity.json"
    identity = json.loads(identity_path.read_text())
    require(identity["wheel_sha256"] == WHEEL_SHA and identity["matching_modules"] == 388,
            "unchanged declared 388-module installed wheel required")
    wheels = list((site.parent/"wheels").glob("*.whl"))
    require(len(wheels) == 1 and sha(wheels[0]) == WHEEL_SHA, "exact installed wheel bytes required")
    source_files = {"anyfesolver/"+p.relative_to(SOURCE_ROOT/"src/anyfesolver").as_posix(): p
                    for p in (SOURCE_ROOT/"src/anyfesolver").rglob("*.py")}
    modules = {}
    with zipfile.ZipFile(wheels[0]) as archive:
        names = {n for n in archive.namelist() if n.startswith("anyfesolver/") and n.endswith(".py")}
        require(len(names) == 388 and names == set(source_files), "complete source/wheel module set required")
        for name in sorted(names):
            data = archive.read(name)
            require(data == (site/name).read_bytes() == source_files[name].read_bytes(),
                    "source/wheel/site byte mismatch: "+name)
            modules[name] = hashlib.sha256(data).hexdigest()
    fixture = Path(sys.modules["test_stiffened_shell_l1_assembly_experimental"].__file__).resolve()
    require(fixture.parent == fixtures and sha(fixture) == identity["fixtures_sha256"][fixture.name],
            "isolated model fixture binding required")
    for name, module in sys.modules.copy().items():
        if name.startswith("test_stiffened_shell_") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            require(path.parent == fixtures and sha(path) == identity["fixtures_sha256"][path.name],
                    "all imported companion fixture bytes must match the installed campaign")
    return {"wheel_sha256": WHEEL_SHA, "wheel_path": str(wheels[0].resolve()),
            "matching_modules": len(modules), "module_sha256": modules, "source_root": str(SOURCE_ROOT),
            "identity_path": str(identity_path.resolve()), "identity_sha256": sha(identity_path),
            "model_fixture_path": str(fixture), "model_fixture_sha256": sha(fixture)}


def effective_settings():
    pools = threadpool_info()
    actual_numba = numba.get_num_threads()
    require(pools and all(p["num_threads"] == 1 for p in pools) and numba.config.DISABLE_JIT
            and numba.config.NUMBA_NUM_THREADS == 1 and actual_numba == 1,
            "one effective BLAS/Numba thread and disabled JIT required")
    return {"pools": pools, "numba_threads": actual_numba, "jit_disabled": bool(numba.config.DISABLE_JIT),
            "requested_environment": {key: os.environ.get(key) for key in
                ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")}}


def layout(owner, definition, policy, nx):
    ns, nb, members = len(owner._shell_ids), len(owner._beam_ids), len(owner._beam_parts)
    physical = 6*(ns+nb)
    xyz = np.vstack((owner._shell_xyz, [owner._beam_nodes[n][0] for n in owner._beam_ids]))
    typed = ([{"domain": "shell", "native_node": int(n), "external_node": int(n), "q_offset": 6*i}
              for i, n in enumerate(owner._shell_ids)]+
             [{"domain": "beam", "native_node": int(n), "external_node": 100001+int(n)-101,
               "q_offset": 6*(ns+i)} for i, n in enumerate(owner._beam_ids)])
    # Native node numbers overlap at finer levels. Never construct a dict keyed
    # only by the native integer, or identify physical and retained coordinates.
    retained = [{"member_index": i, "beam_nodes": part[0].node_ids,
                 "full_member_slots": owner._shell_count+part[1], "retained_24_offset": physical+24*i,
                 "cell_rotation_slots": tuple(range(physical+24*i, physical+24*i+6)),
                 "resultant_slots": tuple(range(physical+24*i+6, physical+24*(i+1)))}
                for i, part in enumerate(owner._beam_parts)]
    return xyz, {"typed_physical_nodes": typed, "shell_node_ids": owner._shell_ids, "beam_node_ids": owner._beam_ids,
        "retained_members": retained, "counts": {"shell_nodes": ns, "beam_nodes": nb, "physical_dofs": physical,
            "members": members, "retained_cell_rotations": 6*members, "retained_resultants": 18*members,
            "retained_dofs": 24*members, "full_q": owner._K.shape[0], "multipliers": owner._C.shape[0]},
        "policy": policy, "nx": nx, "ny": nx//4, "halfcell_count": 2*nx,
        "coordinate_role": "Physical shell/beam nodal6 then per-member24=[two3D cell rotations,18 resultants]. "
                           "Mixed K is indefinite; omitted C slots are not removed physical variables.",
        "external_ID_map": "shell n->n; typed beam n->100001+(n-101); no coordinate merge or integer-key domain flattening",
        "fixed_coordinates": [(f.domain, f.node, f.component) for f in definition.fixed]}


def station_inventory(owner):
    records = []
    for index, (sample, term) in enumerate(zip(owner._line_samples, owner._line_terms)):
        local, slots, weight, extra1, extra2 = sample
        rows, basis = term
        records.append({"sample_index": index, "shell_reference_m": local.reference,
            "shell_alpha": local.station, "thickness_m": local.thickness,
            "beam_port": {"node_indices": local.port.nodes, "member_index": local.port.element_index,
                          "native_cell": local.port.cell, "weight": local.port.weight,
                          "nodal_count": local.port.nodal_count, "retained_count": local.port.retained_count,
                          "lift_m": local.port.lift, "frame": local.port.frame},
            "reference_arm_m": local.reference_arm, "reference_director": local.reference_director,
            "director_basis": local.director_basis, "full_slots": slots, "measure_m": weight,
            "multiplier_rows": rows, "density_basis": basis, "compiler_extra": (extra1, extra2)})
    return records


def sparse_arrays(owner, maps):
    arrays = {"fixed": owner._fixed, "free": owner._free, "coordinate_scale": owner._coordinate_scale,
              "constraint_scale": owner._constraint_scale, "shell_node_ids": owner._shell_ids,
              "beam_node_ids": owner._beam_ids, "shell_rigid_work_map": maps["shell"], "beam_rigid_work_map": maps["beam"]}
    for name, matrix in (("K", owner._K), ("C", owner._C), ("A", owner._A)):
        csr = matrix.tocsr()
        arrays.update({name+"_data": csr.data, name+"_indices": csr.indices, name+"_indptr": csr.indptr,
                       name+"_shape": np.array(csr.shape, dtype=np.int64)})
    return arrays


def validate_model(owner, definition, policy, xyz, records, reference, nx):
    ny, halfcells, ns = nx//4, 2*nx, (nx+1)*(nx//4+1)
    require(tuple(owner._shell_ids) == tuple(range(1, ns+1))
            and tuple(owner._beam_ids) == tuple(range(101, 102+halfcells)), "exact typed native node orders required")
    require(len(owner._beam_parts) == nx and owner._C.shape == (5*nx//2, 6*(ns+halfcells+1)+24*nx)
            and policy["c_lambda"] == 2 and policy["span_halfcells"] == 4, "exact c2 native retained dimensions required")
    expected_shell = np.array([(i/nx, .2*j/ny, 0.) for j in range(ny+1) for i in range(nx+1)])
    expected_beam = np.array([(i/halfcells, 0., reference.BEAM_AXIS_Z_M) for i in range(halfcells+1)])
    require(np.allclose(xyz[:ns], expected_shell, rtol=0., atol=1e-13)
            and np.allclose(xyz[ns:], expected_beam, rtol=0., atol=1e-13), "fixed physical geometry required")
    expected_fixed = {("shell", 1+j*(nx+1), d) for j in range(ny+1) for d in range(6)}
    expected_fixed |= {("beam", 101, d) for d in range(5)}
    require({(f.domain, f.node, f.component) for f in definition.fixed} == expected_fixed,
            "full shell-root column/direct beam first5 ground/normalspinfree required")
    require(definition.shell.thickness_m == reference.SHELL_THICKNESS_M
            and definition.shell.young_pa == reference.E_PA and definition.shell.poisson == reference.NU,
            "unchanged physical shell section required")
    elastic = np.diag((reference.E_PA*reference.BEAM_AREA_M2, reference.G_PA*reference.BEAM_SHEAR_AREA_M2,
        reference.G_PA*reference.BEAM_SHEAR_AREA_M2, reference.G_PA*reference.BEAM_TORSION_M4,
        reference.E_PA*reference.BEAM_BENDING_M4, reference.E_PA*reference.BEAM_BENDING_M4))
    for element, _, _ in owner._beam_parts:
        require(np.array_equal(element.section.elastic, elastic), "unchanged generalized beam section required")
    require(len(records) == 2*halfcells and len(owner._line_terms) == len(records), "complete native station trace inventory required")
    for record in records:
        require(np.allclose(record["beam_port"]["frame"], np.eye(3), rtol=0., atol=1e-13)
            and np.allclose(record["reference_director"], (0., 0., 1.), rtol=0., atol=1e-13)
            and np.allclose(record["director_basis"], ((1., 0., 0.), (0., 1., 0.)), rtol=0., atol=1e-13)
            and np.allclose(record["reference_arm_m"], (0., 0., reference.BEAM_BOTTOM_ARM_M), rtol=0., atol=1e-13)
            and np.linalg.norm(record["beam_port"]["lift_m"]) <= 1e-13,
            "fixed native line physical frames/arms/director required")
    require(np.allclose([owner._beam_nodes[n][1] for n in owner._beam_ids], np.eye(3), rtol=0., atol=1e-13),
            "reference beam frames required")


def windows(transfer, halfcells):
    """Exact quartic-window times affine P0 wrench, generalized halfcell count."""
    records = transfer["station_records"]
    require(len(records) == 2*halfcells and (.25*halfcells).is_integer() and (.75*halfcells).is_integer(),
            "two samples per halfcell and aligned fixed window required")
    smooth = np.zeros(6)
    for edge in range(halfcells):
        a, b = records[2*edge:2*edge+2]
        left, right = edge/halfcells, (edge+1)/halfcells
        require(left < a["x_m"] < b["x_m"] < right and abs(a["measure_m"]-(right-left)/2) < 1e-13
                and abs(b["measure_m"]-(right-left)/2) < 1e-13, "native union quadrature mapping mismatch")
        fa, fb = np.array(a["beam_wrench_about_root_density"]), np.array(b["beam_wrench_about_root_density"])
        slope = (fb-fa)/(b["x_m"]-a["x_m"])
        require(np.allclose(slope, np.r_[np.zeros(3), np.cross((1., 0., 0.), fa[:3])], rtol=1e-10, atol=1e-8),
                "straight reference P0 wrench must be affine within halfcell")
        if left >= .25 and right <= .75:
            for xi, weight in ((-math.sqrt(3./5.), 5./9.), (0., 8./9.), (math.sqrt(3./5.), 5./9.)):
                x = (left+right)/2.+(right-left)*xi/2.
                smooth += weight*(right-left)/2.*256*(x-.25)**2*(.75-x)**2*(fa+slope*(x-a["x_m"]))
    closed = transfer["fixed_region_025_075_wrench"]
    return {"open": closed, "closed": closed, "endpoint_atoms": [0.]*6, "smooth_unit_peak": smooth.tolist(),
            "provenance": "Native continuous density, no endpoint atoms; 3Gauss per halfcell exactly integrates quartic times affine wrench."}


def old_first_level_parity(folder, arrays, write):
    summary = json.loads((folder/"summary.json").read_text())
    meta = json.loads((folder/"raw-state.json").read_text())
    require(summary["status"] == "completed" and summary["case"] == "mz"
            and summary["source_sha256"] == MATCHED_SHA and summary["expected_wheel_sha256"] == WHEEL_SHA,
            "completed unchanged old16 Mz collector required")
    require(meta["arrays_sha256"] == sha(folder/"raw-state.npz"), "old raw-state byte binding required")
    result = {"source_sha256": MATCHED_SHA, "summary_sha256": sha(folder/"summary.json"),
              "raw_state_sha256": sha(folder/"raw-state.npz"), "roles": {}}
    with np.load(folder/"raw-state.npz", allow_pickle=False) as previous:
        for key in ("q", "load", "multipliers", "reactions", "Kq", "constraint_resisting", "connection_exerted", "weak_gaps"):
            current, old = arrays[key], previous[key]
            require(current.shape == old.shape and np.isfinite(old).all(), "old raw role shape/finiteness mismatch: "+key)
            result["roles"][key] = {"bit_equal": bool(np.array_equal(current, old)),
                "maximum_absolute_difference": float(np.max(np.abs(current-old))),
                "relative_l2_difference": float(np.linalg.norm(current-old)/max(np.linalg.norm(old), 1e-12))}
    result["scope"] = "Instrumentation predecessor comparison only; raw deltas retained, no tuned acceptance threshold."
    write("old16-mz-raw-parity", result)


def external_comparison(folder, case, nx, owner, definition, xyz, q, transfer, native_windows):
    summary = json.loads((folder/"summary.json").read_text())
    inputs = json.loads((folder/"inputs.json").read_text())
    raw = json.loads((folder/"raw-inventory.json").read_text())
    actions = json.loads((folder/"recovered-actions.json").read_text())
    require(summary["status"] == "completed" and summary["case"] == case and summary["nx"] == nx
            and summary["source_sha256"] == EXTERNAL_REFINEMENT_SHA, "matching completed frozen external refinement case required")
    ns, nb = len(owner._shell_ids), len(owner._beam_ids)
    external_ids = list(owner._shell_ids)+[100001+n-101 for n in owner._beam_ids]
    require(len(set(external_ids)) == ns+nb and set(map(int, raw["poses"])) == set(external_ids),
            "complete disjoint external pose IDs required")
    coordinates = np.array([raw["nodes"][str(n)] for n in external_ids])
    require(np.allclose(coordinates, xyz, rtol=0., atol=1e-13) and inputs["load"] == list(LOADS[case]),
            "matching external coordinates/load required")
    fixed = {(f.domain, f.node if f.domain == "shell" else 100001+f.node-101, f.component) for f in definition.fixed}
    other_fixed = {("shell" if n in set(owner._shell_ids) else "beam", n, d)
        for key, flags in raw["fixes"].items() for n in (int(key),) for d, flag in enumerate(flags) if flag}
    require(fixed == other_fixed, "matching external typed root supports required")
    for name, value in actions["checks"].items():
        limit = GAP_LIMIT if name == "average_five_channel_gap_scaled" else BALANCE_LIMIT
        require(math.isfinite(value) and value <= limit, "external physical check failed: "+name)
    native = q[:6*(ns+nb)].reshape(ns+nb, 6)
    other = np.array([raw["poses"][str(n)] for n in external_ids])
    require(np.isfinite(other).all(), "finite external poses required")
    shell_index = {n: i for i, n in enumerate(owner._shell_ids)}
    shell_cells = [tuple(shell_index[n] for n in cell) for cell in definition.shell.q4_elements]
    fields = {"shell": matched.field_errors(native[:ns], other[:ns], shell_cells, xyz[:ns]),
              "beam": matched.field_errors(native[ns:], other[ns:], [(i, i+1) for i in range(nb-1)], xyz[ns:], True)}
    external_work = actions["work_J"]["applied"]
    native_work = float(q@owner._accepted[1])
    measures = actions["continuous_density_measures"]
    return {"field_errors": fields, "native_applied_work_J": native_work, "external_applied_work_J": external_work,
        "work_relative_signed": (native_work-external_work)/max(abs(external_work), 1e-12),
        "polynomial_transfer_difference": (np.array(transfer["polynomial_transfer_moments"])-
            np.array([measures["polynomial_x_moments"][str(p)] for p in range(3)])).tolist(),
        "fixed_region_difference": {k: (np.array(native_windows[k])-np.array(v)).tolist()
            for k, v in measures["fixed_region_025_075"].items()},
        "external_files": {name: sha(folder/name) for name in ("summary.json", "inputs.json", "raw-inventory.json", "recovered-actions.json")},
        "scope": "Common nodal Q1/P1 physical fields at1m rotation scale; retained-cell rotations remain distinct. "
                 "Continuous P0 actions compared, no discrepancy/1% acceptance predicate."}


def main(site, output, nx, external_level, old16):
    output.mkdir(parents=True, exist_ok=False)
    site, fixtures, service, reference = runtime(site)
    write = lambda name, value: service._atomic(output/(name+".json"), matched.plain(value))
    owner, original_error = None, None
    summary = {"status": "failed", "classification": "complete native raw refinement collector, development only",
               "nx": nx, "source_sha256": sha(__file__), "expected_wheel_sha256": WHEEL_SHA, "python": sys.version,
               "cases": {}, "claim_scope": "No mechanicalKKT/locking/convergedaccuracy/F03 acceptance; product equations/tolerances unchanged."}
    try:
        require(nx in LEVELS, "only frozen16/32/64 levels required")
        write("identity", installed_identity(site, fixtures))
        summary["initial_settings"] = effective_settings()
        write("initial-imports", canonical_imports(site))
        definition, policy = coherent.model(nx, nx//4, 2*nx, 2)
        owner = stiffened_shell.create_analysis(definition=definition)
        xyz, coordinate_layout = layout(owner, definition, policy, nx)
        maps = matched.rigid_maps(owner, xyz)
        trace = station_inventory(owner)
        matched.save_npz(output/"operators.npz", sparse_arrays(owner, maps))
        write("layout", coordinate_layout)
        write("station-traces", trace)
        write("inputs", {"loads": LOADS, "definition_profile": definition.profile,
            "model": {"nodes": {"shell": {int(n): xyz[i] for i, n in enumerate(owner._shell_ids)},
                                  "beam": {int(n): xyz[len(owner._shell_ids)+i] for i, n in enumerate(owner._beam_ids)}},
                      "cells": {"shell_q4": definition.shell.q4_elements,
                                "beam_halfcells": tuple(zip(owner._beam_ids, owner._beam_ids[1:])),
                                "beam_B3_members": tuple(part[0].node_ids for part in owner._beam_parts)},
                      "typed_physical_ID_map": coordinate_layout["typed_physical_nodes"],
                      "layout": coordinate_layout, "moment_origin_m": xyz[len(owner._shell_ids)]},
            "shell": {"node_ids": definition.shell.node_ids, "coordinates_m": definition.shell.coordinates_m,
                      "q4_elements": definition.shell.q4_elements, "thickness_m": definition.shell.thickness_m,
                      "young_pa": definition.shell.young_pa, "poisson": definition.shell.poisson,
                      "density_kg_m3": definition.shell.density_kg_m3},
            "beams": [json.loads(b.raw) for b in definition.beams], "line": definition.lines[0].descriptor(),
            "policy": policy, "fixed": coordinate_layout["fixed_coordinates"], "xyz_m": xyz,
            "compiled_preflight": owner.preflight(), "operators_sha256": sha(output/"operators.npz"),
            "helper_sources": {str(Path(module.__file__).resolve()): sha(module.__file__) for module in (coherent, matched)},
            "reference_constants": {key: getattr(reference, key) for key in ("E_PA", "NU", "G_PA", "SHELL_THICKNESS_M",
                "BEAM_AREA_M2", "BEAM_SHEAR_AREA_M2", "BEAM_BENDING_M4", "BEAM_TORSION_M4", "BEAM_AXIS_Z_M", "BEAM_BOTTOM_ARM_M")}})
        validate_model(owner, definition, policy, xyz, trace, reference, nx)
        ns, nb, members = len(owner._shell_ids), len(owner._beam_ids), len(owner._beam_parts)
        for case, wrench in LOADS.items():
            case_dir = output/case
            case_dir.mkdir(exist_ok=False)
            cw = lambda name, value: service._atomic(case_dir/(name+".json"), matched.plain(value))
            result = owner.solve_linear((stiffened_shell.NodalAction("beam", 101+2*nx, wrench),))
            q = np.array(np.r_[result.shell_displacement, result.beam_coordinates], copy=True)
            load = np.array(owner._accepted[1], copy=True)
            lam = np.array(result.multipliers, copy=True)
            g, ct = owner._K@q, owner._C.T@lam
            arrays = {"q": q, "load": load, "multipliers": lam, "reactions": result.reactions,
                "Kq": g, "constraint_resisting": ct, "connection_exerted": -ct,
                "weak_gaps": owner._C@q, "full_equilibrium": g+ct-load,
                "shell_poses": result.shell_displacement.reshape(ns, 6),
                "beam_nodal_poses": result.beam_coordinates[:6*nb].reshape(nb, 6),
                "beam_retained": result.beam_coordinates[6*nb:].reshape(members, 24),
                "shell_node_ids": owner._shell_ids, "beam_node_ids": owner._beam_ids,
                "shell_ids": owner._shell_ids, "beam_ids": owner._beam_ids,
                "xyz_m": xyz, "external_ids": [r["external_node"] for r in coordinate_layout["typed_physical_nodes"]],
                "shell_xyz_m": xyz[:ns], "beam_xyz_m": xyz[ns:],
                "beam_external_ids": [100001+n-101 for n in owner._beam_ids],
                "C_data": owner._C.data, "C_indices": owner._C.indices, "C_indptr": owner._C.indptr,
                "C_shape": owner._C.shape, "fixed": owner._fixed, "free": owner._free,
                "coordinate_scale": owner._coordinate_scale, "constraint_scale": owner._constraint_scale}
            matched.save_npz(case_dir/"raw-state.npz", arrays)
            cw("raw-state", {"status": result.status, "generation": result.generation,
                "arrays_sha256": sha(case_dir/"raw-state.npz"), "counts": coordinate_layout["counts"],
                "fixed_free_layout": "../operators.npz and ../layout.json", "forces_sign": "Kq internal resisting; "
                    "+Ct lambda connection resisting; -Ct lambda exerted; reactions actual ground on fixed slots",
                "preflight_after": owner.preflight()})
            transfer = coherent.physical_transfer(owner, result)
            cw("physical-transfer", transfer)
            native_windows = windows(transfer, 2*nx)
            cw("fixed-windows", native_windows)
            internal, exerted, ground, applied = ({d: matrix@vector for d, matrix in maps.items()}
                for vector in (g, -ct, result.reactions, load))
            cuts = {d: applied[d]+exerted[d]+ground[d] for d in maps}
            moment0 = np.array(transfer["polynomial_transfer_moments"][0])
            scale = max(map(abs, wrench))  # Force equivalent using frozen Lref1m.
            work = float(q@load)
            work_scale = max(abs(work), 1e-12)
            checks = {"equilibrium_ratio": result.free_equilibrium_ratio, "weak_gap": result.constraint_max,
                "energy_work_relative": abs(float(q@g)-work)/work_scale,
                "compatible_connection_work_relative": abs(float(q@(-ct)))/work_scale,
                "ground_work_relative": abs(float(q@result.reactions))/work_scale,
                "beam_physical_workmap_scaled": float(np.max(np.abs(exerted["beam"]-moment0)))/scale,
                "shell_physical_workmap_scaled": float(np.max(np.abs(exerted["shell"]+moment0)))/scale,
                "complete_domain_cuts_scaled": max(float(np.max(np.abs(v)))/scale for v in cuts.values()),
                "global_ground_applied_scaled": float(np.max(np.abs(sum(ground.values())+sum(applied.values()))))/scale}
            cw("physical-actions", {"domain_wrenches": {"internal": internal, "connection_exerted": exerted,
                "ground": ground, "applied": applied, "complete_cuts": cuts}, "wrench_origin_m": xyz[ns],
                "root_partition": {"native_G_nodal": g[owner._fixed], "native_interface_nodal": (-ct)[owner._fixed],
                                   "actual_ground_nodal": result.reactions[owner._fixed], "fixed_slots": owner._fixed},
                "work_J": {"applied": work, "internal": float(q@g), "connection_exerted": float(q@(-ct)),
                           "ground": float(q@result.reactions)}, "scales": {"length_m": 1., "force_N": scale,
                    "moment_Nm": scale, "work_J": work_scale, "work_floor_J": 1e-12}, "checks": checks,
                "provenance": "Full domain rigid virtual work includes both retained cell rotations; beam nodal sum alone omits cell torque covectors."})
            collected = {"status": "retained_before_predicates", "case": case, "nx": nx,
                "generation": result.generation, "applied_work_J": work, "counts": coordinate_layout["counts"],
                "physical_checks": checks, "windows": native_windows,
                "raw_state_sha256": sha(case_dir/"raw-state.npz"), "preflight": owner.preflight()}
            cw("result", collected)
            if external_level is not None:
                cw("external-comparison", external_comparison(external_level/case, case, nx, owner, definition, xyz,
                                                               q, transfer, native_windows))
            if old16 is not None and nx == 16 and case == "mz":
                old_first_level_parity(old16, arrays, cw)
            for name, value in checks.items():
                limit = GAP_LIMIT if name == "weak_gap" else BALANCE_LIMIT
                if not math.isfinite(value) or value > limit:
                    raise NativePhysicalCheckFailure(f"{case}: {name}={value} > {limit}")
            require(result.status == "completed" and owner.preflight()["reference_linear_factorizations"] == 1,
                    "completed RHS with one cached reference factor required")
            summary["cases"][case] = {"generation": result.generation, "applied_work_J": work,
                "checks": checks, "raw_state_sha256": sha(case_dir/"raw-state.npz"), "preflight": owner.preflight()}
            collected["status"] = "completed"
            cw("result", collected)
        require(set(summary["cases"]) == set(LOADS), "all three declared RHS inventories required")
        summary.update(status="completed", counts=coordinate_layout["counts"],
                       final_preflight=owner.preflight(), final_settings=effective_settings())
        summary["canonical_onsite_import_count"] = len(canonical_imports(site))
    except BaseException as error:
        original_error = error
        summary["status"] = "failed"
        summary["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        cleanup_error = None
        actions = [("retain_imports", lambda: write("imports", {name: str(Path(m.__file__).resolve())
            for name, m in sys.modules.copy().items() if name.split('.')[0] == "anyfesolver" and getattr(m, "__file__", None)}))]
        if owner is not None:
            actions.append(("close", owner.close))
        for name, action in actions:
            try:
                action()
            except BaseException as error:
                cleanup_error = error
                summary["status"] = "failed"
                summary.setdefault("retention_cleanup_failures", []).append({"operation": name, "type": type(error).__name__, "message": str(error)})
        try:
            write("summary", summary)
        except BaseException:
            if original_error is None:
                raise
            traceback.print_exc()
        if cleanup_error is not None and original_error is None:
            raise cleanup_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, choices=LEVELS, required=True)
    parser.add_argument("--external-level-dir", type=Path)
    parser.add_argument("--prior-16-mz-dir", type=Path)
    args = parser.parse_args()
    main(args.site, args.output_dir, args.nx, args.external_level_dir, args.prior_16_mz_dir)
