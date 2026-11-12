"""One saved joint32 projected mixed-system development oracle.

Recompute only 32 local stationary negative resultant blocks. Keep all nodal and
native cell rotations; diagnose the supported physical kernel and saddle inertia
on this fixture. Dense diagnostics are bounded by 2400 physical/3000 supported
KKT coordinates, 600 seconds, 1 GiB and eight threads (actual pools require one).
No production condensation, full-K SPD, mass/stiffness change, coarse-kernel
transfer, uniform/growing-space, M2, locking/accuracy or F03 claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import anyfesolver
import numba
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, _scaled_constraint_gap
from anyfesolver._b2_physical_linear_batch_dev import B2NumericalSolveFailure, _accurate_sparse_residual
from scripts.r8_bl5_nested_projected_kernel_dev import saved_csr, relative


IDENTITY = "BL5_PROJECTED_FOUR_SHELL_INTERVALS_JOINT32_DEV_V1"
SCHEMA = "BL5_SAVED_JOINT32_PROJECTED_KERNEL_DEV_V1"
CASES = ("fy", "mx", "mz")
FIXTURE = dict(nx=32, ny=8, halfcells=64, supports=8)
COUNTS = dict(shell_nodes=297, beam_nodes=65, physical_dofs=2172, members=32,
    retained_cell_rotations=192, retained_resultants=576, retained_dofs=768, full_q=2940, multipliers=80)
MAX_SECONDS, MAX_BYTES, MAX_PHYSICAL, MAX_SUPPORTED_KKT = 600., 1 << 30, 2400, 3000
EPS = np.finfo(float).eps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fro_bound(a):
    """Conservative Frobenius upper bound for the induced spectral norm."""
    value = float(sparse.linalg.norm(a) if sparse.issparse(a) else linalg.norm(a, "fro"))
    return math.nextafter(value*(1+64*EPS*max(a.shape, default=1)), math.inf) if value else 0.


def one_bound(a):
    value = float(np.max(np.sum(np.abs(a), axis=0), initial=0.))
    return math.nextafter(value*(1+64*EPS*max(a.shape)), math.inf) if value else 0.


def triangle_difference(a):
    # Measure the lower-triangle completion used by LAPACK; never change K/S.
    return fro_bound(a-(np.tril(a)+np.tril(a, -1).T))


def peak_rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]+[
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD)
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counter.PeakWorkingSetSize)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak*(1 if sys.platform == "darwin" else 1024))


def checkpoint(start, report):
    elapsed, peak = time.monotonic()-start, peak_rss_bytes()
    report.update(elapsed_seconds=elapsed, peak_rss_bytes=peak)
    require(elapsed <= MAX_SECONDS, "saved joint32 diagnostic600s ceiling exceeded")
    require(peak <= MAX_BYTES, "saved joint32 diagnostic1GiB ceiling exceeded")


def write_json(path, value):
    def serializable(item):
        if isinstance(item, dict):
            return {key: serializable(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [serializable(v) for v in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(item):
            # Preserve a diagnostic nonfinite token; exact raw arrays remain in NPZ.
            return "NONFINITE:"+str(item)
        return item
    temporary = path.with_name(path.name+"."+str(os.getpid())+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(serializable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_arrays(path, arrays):
    temporary = path.with_name(path.name+"."+str(os.getpid())+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def sparse_arrays(arrays, name, matrix):
    matrix = matrix.tocsr()
    arrays.update({name+"_data": matrix.data, name+"_indices": matrix.indices,
        name+"_indptr": matrix.indptr, name+"_shape": np.asarray(matrix.shape, dtype=np.int64)})


def effective_settings():
    pools = threadpool_info()
    require(pools and all(p["num_threads"] == 1 for p in pools) and numba.get_num_threads() == 1
            and numba.config.DISABLE_JIT, "effective numerical pools1/Numba1/JIT-off required")
    return dict(pools=pools, numba_threads=numba.get_num_threads(), jit_disabled=bool(numba.config.DISABLE_JIT))


def measure(q, lm, reactions, k, c, load, free, fixed, coordinate, constraint, workmaps, wrench_scale):
    action, g = c.T@lm, k@q
    residual = g+action-load
    work, energy = float(q@load), .5*float(q@g)
    resisting = np.vstack([matrix@action for matrix in workmaps])
    ground = np.vstack([matrix@reactions for matrix in workmaps])
    applied = np.vstack([matrix@load for matrix in workmaps])
    return dict(equilibrium=float(_linear_equilibrium(residual[free], load[free], coordinate[free])),
        weak_constraint=float(_scaled_constraint_gap(c@q, constraint)),
        energy_work_relative=abs(2*energy-work)/max(abs(work), 1e-12),
        compatible_work_relative=abs(float(q@action))/max(abs(work), 1e-12),
        ground_work_relative=abs(float(q@reactions))/max(abs(work), 1e-12),
        interface_action_balance=float(np.max(np.abs(resisting.sum(axis=0))))/wrench_scale,
        global_ground_applied_balance=float(np.max(np.abs((ground+applied).sum(axis=0))))/wrench_scale,
        domain_cut_balance=float(np.max(np.abs(applied-resisting+ground)))/wrench_scale,
        work_J=work, energy_J=energy, interface_resisting_wrenches=resisting.tolist(),
        ground_wrenches=ground.tolist(), residual_fixed_max=float(np.max(np.abs(residual[fixed]-reactions[fixed]))))


def parity(a, b, roles, pp, coordinate, constraint, workmaps, c, k, load):
    q, lm, reaction = a
    other, other_lm, other_reaction = b
    action, other_action = c.T@lm, c.T@other_lm
    result = {name: relative(q[slots]/coordinate[slots], other[slots]/coordinate[slots])
              for name, slots in roles.items()}
    result.update(resultants=relative(q[pp]/coordinate[pp], other[pp]/coordinate[pp]),
        multipliers=relative(lm*constraint, other_lm*constraint),
        reactions=relative(reaction*coordinate, other_reaction*coordinate),
        interface_resisting_wrenches=relative(np.vstack([m@action for m in workmaps]),
            np.vstack([m@other_action for m in workmaps])),
        ground_wrenches=relative(np.vstack([m@reaction for m in workmaps]),
            np.vstack([m@other_reaction for m in workmaps])),
        energy=relative(np.asarray(q@(k@q)), np.asarray(other@(k@other))),
        applied_work=relative(np.asarray(q@load), np.asarray(other@load)))
    return {name: float(value) for name, value in result.items()}


def evaluate(root, output):
    start = time.monotonic()
    root, output = Path(root), Path(output)
    require(output.suffix == ".json" and output.parent.is_dir(), "new JSON output with existing parent required")
    archive_path = output.with_suffix(".npz")
    require(not output.exists() and not archive_path.exists(), "existing diagnostic pair may not be replaced")
    arrays = {}
    report = dict(schema=SCHEMA, candidate=IDENTITY, status="FAILED", source_sha256=sha(__file__),
        scope="one saved joint32 reference-linear mixed-system oracle; experimental not qualified; "
            "no fullK SPD, production condensation, uniform/M2/growing-envelope/locking/accuracy/F03 claim",
        formal_f03_authority_consumed=False,
        resource_ceiling=dict(seconds=MAX_SECONDS, bytes=MAX_BYTES, threads=8,
            physical_order=MAX_PHYSICAL, supported_kkt_order=MAX_SUPPORTED_KKT),
        uncertainty_policy="Frobenius upper bounds for errors/leakage/orthogonality; "
            "max induced1norm(S,S.T) for operator size; unchanged minimum eigenvalue > summed uncertainty",
        full_general_factorizations=0, reduced_factorizations=0, full_accurate_residual_corrections=0)
    earlier_error = False
    try:
        report.update(**effective_settings())
        checkpoint(start, report)
        native = root/"native"
        names = ("inputs.json", "operators.npz", *(case+"/raw-state.npz" for case in CASES),
                 *(case+"/raw-state.json" for case in CASES))
        source_files = {name: sha(native/name) for name in names}
        source_files["summary.json"] = sha(root/"summary.json")
        owner_report = json.loads((root/"summary.json").read_text())
        require(owner_report["status"] == "COMPLETED_INSTALLED_DEVELOPMENT_ONLY" and
                owner_report["identity"] == IDENTITY and owner_report["fixture"] == FIXTURE and
                owner_report["predicates"] and all(v is True for v in owner_report["predicates"].values()),
                "completed actual installed joint32 owner evidence required")
        require(owner_report["coordinate_norm"] == dict(length_m=1., width_m=.2, ell_m=.4), "unchanged dimensional norm required")
        require(all(owner_report["native_input_sha256"][name] == source_files[name]
                for name in ("inputs.json", "operators.npz")), "actual native inputs differ from completed owner binding")
        source = Path(owner_report["source_root"])
        require(owner_report["source_sha256"] == sha(source/"scripts/r8_bl5_projected_joint_owner_dev.py"),
                "actual owner source identity changed")
        site = Path(owner_report["site"]).resolve(strict=True)
        modules = owner_report["module_sha256"]
        require(len(modules) == 417 and all(sha(site/name) == digest for name, digest in modules.items()),
                "complete417 saved installed module identities changed")
        imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.copy().items()
                   if name.split(".")[0] == "anyfesolver" and getattr(module, "__file__", None)}
        require(imports and all(Path(p).is_relative_to(site) and
            sha(p) == modules[Path(p).relative_to(site).as_posix()] for p in imports.values()),
            "all current product imports must match saved installed graph")
        inputs = json.loads((native/"inputs.json").read_text())
        require(inputs["fixture"] == FIXTURE and inputs["layout"]["counts"] == COUNTS and
                inputs["layout"]["nx"] == 32 and inputs["layout"]["ny"] == 8,
                "exact joint32 physical/retained fixture required")
        policy = inputs["policy"]
        require(policy["basis"] == "discontinuous_p1_cell_average" and policy["span_halfcells"] == 8 and
                policy["beam_halfcells"] == 64 and policy["support_endpoints_m"] == [i/8 for i in range(9)] and
                policy["support_membership"] == [list(range(8*i, 8*i+8)) for i in range(8)],
                "eight complete projected supports on fixed native halfcells required")
        report.update(source_files=source_files, owner_source_sha256=owner_report["source_sha256"],
            installed_identity=dict(site=str(site), wheel_sha256=owner_report["wheel_sha256"],
                module_sha256=modules, anyfesolver_import=anyfesolver.__file__, product_imports=imports),
            helpers={Path(f.__code__.co_filename).name: sha(f.__code__.co_filename) for f in
                (saved_csr, relative, _linear_equilibrium, _scaled_constraint_gap, _accurate_sparse_residual)})
        with np.load(native/"operators.npz", allow_pickle=False) as saved:
            k, c = saved_csr(saved, "K", (2940, 2940)), saved_csr(saved, "C", (80, 2940))
            free, fixed, coordinate, constraint, shell_work, beam_work = [saved[name].copy() for name in
                ("free", "fixed", "coordinate_scale", "constraint_scale", "shell_rigid_work_map", "beam_rigid_work_map")]
        arrays.update(free=free, fixed=fixed, coordinate_scale=coordinate, constraint_scale=constraint,
            shell_rigid_work_map=shell_work, beam_rigid_work_map=beam_work)
        sparse_arrays(arrays, "K", k); sparse_arrays(arrays, "C", c)
        save_arrays(archive_path, arrays)
        require(free.shape == (2881,) and fixed.shape == (59,) and free.dtype.kind in "iu" and fixed.dtype.kind in "iu"
                and np.array_equal(np.sort(np.r_[free, fixed]), np.arange(2940)), "exact full free/fixed partition required")
        require(coordinate.shape == (2940,) and constraint.shape == (80,) and
                np.isfinite(coordinate).all() and np.all(coordinate > 0.) and
                np.isfinite(constraint).all() and np.all(constraint > 0.) and
                shell_work.shape == beam_work.shape == (6, 2940) and
                np.isfinite(shell_work).all() and np.isfinite(beam_work).all(), "finite physical scales/work maps required")
        blocks = [2172+24*i+np.arange(6, 24) for i in range(32)]
        pp = np.concatenate(blocks); x = np.setdiff1d(free, pp)
        require(len(x) == 2305 and len(pp) == 576 and len(x)-80 == 2225 and len(free)+80 == 2961 and
                len(x) <= MAX_PHYSICAL and len(free)+80 <= MAX_SUPPORTED_KKT, "explicit fine diagnostic dimensions/caps required")
        require(np.isin(pp, free).all() and c[:, pp].nnz == 0, "all resultants free and absent from constraints required")
        offblock = k[pp][:, pp]-sparse.block_diag([k[p][:, p] for p in blocks])
        require(np.max(np.abs(offblock.data), initial=0.) == 0., "off-block resultant coupling is not locally eliminable")
        full_skew = float(sparse.linalg.norm(k-k.T)/max(sparse.linalg.norm(k), 1e-30))
        require(full_skew <= 1e-9, "saved fullK asymmetry exceeds unchanged diagnostic budget")
        report["full_K_skew_relative"] = full_skew
        s = k[x][:, x].tocsr(); free_map = {int(v): i for i, v in enumerate(free)}
        rows, columns, values = [free_map[int(v)] for v in x], list(range(len(x))), [1.]*len(x)
        local_maps, local_info = [], []
        for index, p in enumerate(blocks):
            checkpoint(start, report)
            d = k[p][:, p].toarray()
            local_ev = linalg.eigvalsh(d, lower=True)
            factor = linalg.cho_factor(-d, lower=True)
            chol = np.tril(factor[0]); factor_error = d+chol@chol.T
            budget = fro_bound(factor_error)+64*EPS*18*fro_bound(d)
            coupling = k[p][:, x]; neighbours = np.unique(coupling.nonzero()[1])
            px = coupling[:, neighbours].toarray()
            local_lift = linalg.cho_solve(factor, px)
            residual = d@local_lift+px
            backward = float(linalg.norm(residual, "fro")/max(
                linalg.norm(d, "fro")*linalg.norm(local_lift, "fro")+linalg.norm(px, "fro"), 1e-30))
            arrays.update({f"member{index}_negative_block": d, f"member{index}_negative_values": local_ev,
                f"member{index}_lift": local_lift, f"member{index}_residual": residual,
                f"member{index}_neighbours": neighbours, f"member{index}_factor_error": factor_error})
            save_arrays(archive_path, arrays)
            info = dict(member=index, maximum_negative_eigenvalue=float(local_ev[-1]),
                negative_block_budget=budget, stationary_backward_residual=backward,
                negative_block_skew_relative=relative(d, d.T), neighbours=len(neighbours))
            local_info.append(info); report["local_blocks"] = local_info
            require(info["negative_block_skew_relative"] <= 1e-9 and -local_ev[-1] > budget and
                    backward <= 1e-10 and len(neighbours) <= 24, "local negative/stationary block unresolved: "+str(index))
            update = k[x[neighbours]][:, p].toarray()@local_lift
            s += sparse.coo_matrix((update.ravel(), (np.repeat(neighbours, len(neighbours)),
                np.tile(neighbours, len(neighbours)))), shape=s.shape).tocsr()
            rows.extend(np.repeat([free_map[int(v)] for v in p], len(neighbours)))
            columns.extend(np.tile(neighbours, len(p))); values.extend(local_lift.ravel())
            local_maps.append((neighbours, local_lift, factor, residual, factor_error))
        lifted = sparse.coo_matrix((values, (rows, columns)), shape=(2881, 2305)).tocsr()
        kfree = k[free][:, free]; work_pullback = lifted.T@kfree@lifted
        work_parity = float(sparse.linalg.norm(work_pullback-s)/max(sparse.linalg.norm(s), 1e-30))
        cx = c[:, x].toarray()
        constraint_parity = relative((c[:, free]@lifted).toarray(), cx)
        identity_error = float(sparse.linalg.norm(lifted[[free_map[int(v)] for v in x]]-sparse.eye(len(x))))
        stationary = k[pp][:, free]@lifted
        stationary_backward = float(sparse.linalg.norm(stationary)/max(
            sparse.linalg.norm(k[pp][:, free])*sparse.linalg.norm(lifted), 1e-30))
        arrays.update(physical_slots=x, resultant_slots=pp)
        sparse_arrays(arrays, "S", s); sparse_arrays(arrays, "lift", lifted)
        save_arrays(archive_path, arrays)
        report.update(work_pullback_relative=work_parity, constraint_pullback_relative=constraint_parity,
            lift_identity_error=identity_error, stationary_backward_residual=stationary_backward)
        require(max(work_parity, constraint_parity, identity_error, stationary_backward) <= 1e-10,
                "fresh stationary work/C/identity pullback mismatch")
        scale = coordinate[x].copy(); diagonal = np.abs(s.diagonal()*scale**2)
        require(np.max(diagonal) > 0., "nonzero stationary diagonal scale required")
        scale /= np.sqrt(np.where(diagonal > 0., diagonal, np.max(diagonal)))
        sd = sparse.diags(scale); scaled = (sd@s@sd).toarray()
        require(relative(scaled, scaled.T) <= 1e-9, "retained physical asymmetry exceeds unchanged budget")
        cscaled = cx*scale; row_norm = np.linalg.norm(cscaled, axis=1)
        require(np.all(row_norm > 0.), "every actual constraint row must be nonzero")
        cscaled /= row_norm[:, None]
        singular = linalg.svdvals(cscaled)
        rank_budget = 64*EPS*max(cscaled.shape)*singular[0]
        arrays.update(physical_scale=scale, constraint_row_scale=row_norm, constraint_singular_values=singular)
        save_arrays(archive_path, arrays)
        require(singular[-1] > rank_budget, "all80 constraint rank unresolved; no row trimming")
        checkpoint(start, report)
        qbasis, _ = linalg.qr(cscaled.T, mode="full")
        z = qbasis[:, 80:].copy(); del qbasis
        restricted = z.T@scaled@z
        ev, vectors = linalg.eigh(restricted, lower=True)
        checkpoint(start, report)
        snorm = max(one_bound(scaled), one_bound(scaled.T))
        orthogonality = fro_bound(z.T@z-np.eye(2225))
        leakage = fro_bound(cscaled@z); eta = leakage/singular[-1]
        eliminated_error = fro_bound(sd@(work_pullback-s)@sd)
        for neighbours, local_lift, factor, residual, factor_error in local_maps:
            r = residual*scale[neighbours]
            sl = local_lift*scale[neighbours]
            eliminated_error += fro_bound(r.T@linalg.cho_solve(factor, r))+fro_bound(sl.T@factor_error@sl)
        terms = dict(rounding=64*EPS*2305*snorm, elimination_and_reconstruction=eliminated_error,
            kernel_leakage=(2*eta+eta**2)*snorm, kernel_orthogonality=(2*orthogonality+orthogonality**2)*snorm,
            physical_triangle_completion=triangle_difference(scaled),
            restricted_triangle_completion=triangle_difference(restricted),
            eigen_residual=fro_bound(restricted@vectors-vectors*ev))
        uncertainty = math.fsum(terms.values())
        saddle = np.block([[scaled, cscaled.T], [cscaled, np.zeros((80, 80))]])
        saddle_ev = linalg.eigvalsh(saddle, lower=True)
        checkpoint(start, report)
        saddle_budget = 64*EPS*2385*max(one_bound(saddle), one_bound(saddle.T))+uncertainty+triangle_difference(saddle)
        inertia = dict(positive=int(np.sum(saddle_ev > saddle_budget)), negative=int(np.sum(saddle_ev < -saddle_budget)),
            unresolved=int(np.sum(np.abs(saddle_ev) <= saddle_budget)))
        expected = dict(positive=2305, negative=80, unresolved=0)
        arrays.update(kernel_values=ev, kernel_basis=z, kernel_vectors=vectors,
            restricted_operator=restricted, saddle_values=saddle_ev)
        save_arrays(archive_path, arrays)
        kernel_positive = bool(ev[0] > uncertainty)
        report.update(physical_order=2305, native_supported_order=2881, constraint_rows=80, kernel_order=2225,
            full_supported_saddle_order=2961, constraint_min_singular=float(singular[-1]), rank_budget=float(rank_budget),
            physical_scale_policy="dimensional saved coordinate scale followed by stationary diagonal equilibration",
            kernel_leakage=leakage, kernel_orthogonality=orthogonality, operator_size_induced_one_bound=snorm,
            minimum_kernel_eigenvalue=float(ev[0]), spectral_uncertainty_terms=terms, spectral_uncertainty=uncertainty,
            reduced_saddle_inertia=inertia, expected_reduced_saddle_inertia=expected, saddle_budget=saddle_budget,
            implied_native_inertia=dict(positive=2305, negative=656, unresolved=0) if kernel_positive and inertia == expected else None)
        saved_cases = {}
        for case in CASES:
            metadata = json.loads((native/case/"raw-state.json").read_text())
            require(metadata["status"] == "completed" and metadata["arrays_sha256"] == source_files[case+"/raw-state.npz"],
                    "completed raw state binding required: "+case)
            with np.load(native/case/"raw-state.npz", allow_pickle=False) as saved:
                state = {name: saved[name].copy() for name in ("q", "load", "multipliers", "reactions")}
            require(state["q"].shape == state["load"].shape == state["reactions"].shape == (2940,) and
                    state["multipliers"].shape == (80,) and all(np.isfinite(v).all() for v in state.values()),
                    "finite saved installed state required: "+case)
            expected_load = np.zeros(2940)
            expected_load[2166+{"fy": 1, "mx": 3, "mz": 5}[case]] = 100. if case == "fy" else 10.
            require(np.array_equal(state["load"], expected_load), "fixed beamtip165 Fy100/Mx10/Mz10 load required: "+case)
            saved_cases[case] = state
            arrays.update({case+"_owner_"+name: value for name, value in state.items()})
        forces = np.column_stack([saved_cases[case]["load"] for case in CASES])
        require(np.max(np.abs(forces[pp])) == 0., "resultant RHS must be zero for local stationary oracle")
        roles = dict(physical=x, shell=x[x < 1782], beam_nodes=x[(x >= 1782) & (x < 2172)], cell_rotations=x[x >= 2172])
        require(len(roles["cell_rotations"]) == 192 and all(len(v) for v in roles.values()), "all physical roles must survive")
        results, raw_results, solve_failure = {}, {}, None
        try:
            full = sparse.bmat([[kfree, c[:, free].T], [c[:, free], None]], format="csc")
            try:
                full_factor = splu(full)
            except RuntimeError as error:
                if str(error) != "Factor is exactly singular":
                    raise
                raise B2NumericalSolveFailure("joint32 full sparse KKT: "+str(error)) from error
            report["full_general_factorizations"] = 1
            rhs = np.vstack((forces[free], np.zeros((80, 3))))
            raw_solution = full_factor.solve(rhs)
            arrays["raw_full_solution"] = raw_solution
            save_arrays(archive_path, arrays)
            if not np.isfinite(raw_solution).all():
                raise B2NumericalSolveFailure("nonfinite raw full three-load solution")
            defect = _accurate_sparse_residual(full, raw_solution, rhs)
            correction = full_factor.solve(-defect)
            solution = raw_solution+correction
            second_defect = _accurate_sparse_residual(full, solution, rhs)
            second_correction = full_factor.solve(-second_defect)
            arrays.update(first_corrected_full_solution=solution.copy(),
                second_full_accurate_defect=second_defect, second_full_refinement_correction=second_correction)
            solution += second_correction
            report["full_accurate_residual_corrections"] = 2
            arrays.update(raw_full_accurate_defect=defect, full_refinement_correction=correction, corrected_full_solution=solution)
            # Independent factor: no full-factor correction or tangent averaging.
            reduced_factor = linalg.lu_factor(saddle)
            reduced = linalg.lu_solve(reduced_factor,
                np.vstack((forces[x]*scale[:, None], np.zeros((80, 3)))))
            report["reduced_factorizations"] = 1
            arrays["raw_reduced_solution"] = reduced.copy()
            # Rounded stationary assembly can have small backward error against S
            # yet appreciable forward error against the ORIGINAL mixed equations.
            # One defect correction through the independent reduced factor also
            # needs local particulars for nonzero resultant residual RHS.
            reduced_native = np.vstack((lifted@(reduced[:2305]*scale[:, None]),
                reduced[2305:]/row_norm[:, None]))
            reduced_defect = _accurate_sparse_residual(full, reduced_native, rhs)
            correction_rhs = -reduced_defect
            particular = np.zeros((2881, 3))
            for p, (_, _, factor, _, _) in zip(blocks, local_maps):
                slots = [free_map[int(v)] for v in p]
                particular[slots] = -linalg.cho_solve(factor, correction_rhs[slots])
            physical_rhs = correction_rhs[[free_map[int(v)] for v in x]]-k[x][:, free]@particular
            reduced_delta = linalg.lu_solve(reduced_factor,
                np.vstack((physical_rhs*scale[:, None], correction_rhs[2881:]/row_norm[:, None])))
            reduced_native += np.vstack((lifted@(reduced_delta[:2305]*scale[:, None])+particular,
                reduced_delta[2305:]/row_norm[:, None]))
            arrays.update(reduced_solution=reduced, raw_lifted_reduced_solution=np.vstack((
                lifted@(reduced[:2305]*scale[:, None]), reduced[2305:]/row_norm[:, None])),
                reduced_original_mixed_defect=reduced_defect,
                reduced_defect_particular=particular, reduced_defect_correction=reduced_delta,
                corrected_lifted_reduced_solution=reduced_native)
            report.update(reduced_accurate_residual_corrections=1,
                reduced_refinement_policy="one original-mixed compensated residual correction through independent reduced LU and local resultant particulars")
            save_arrays(archive_path, arrays)
            if not (np.isfinite(solution).all() and np.isfinite(reduced).all()):
                raise B2NumericalSolveFailure("nonfinite corrected full or reduced three-load solution")
            for j, case in enumerate(CASES):
                load = forces[:, j]; state = saved_cases[case]
                triples = {}
                for name, solved in (("full", solution), ("raw", raw_solution)):
                    q = np.zeros(2940); q[free] = solved[:2881, j]
                    lm = solved[2881:, j]
                    residual = k@q+c.T@lm-load
                    reaction = np.zeros(2940); reaction[fixed] = residual[fixed]
                    triples[name] = (q, lm, reaction)
                q = np.zeros(2940); q[free] = reduced_native[:2881, j]
                lm = reduced_native[2881:, j]
                residual = k@q+c.T@lm-load
                reaction = np.zeros(2940); reaction[fixed] = residual[fixed]
                triples["reduced"] = (q, lm, reaction)
                triples["owner"] = (state["q"], state["multipliers"], state["reactions"])
                for name, triple in triples.items():
                    arrays.update({case+"_"+name+"_"+key: value for key, value in zip(("q", "multipliers", "reactions"), triple)})
                save_arrays(archive_path, arrays)
                workmaps = (shell_work, beam_work)
                wrench_scale = 100. if case == "fy" else 10.
                metrics = {name: measure(*triple, k, c, load, free, fixed, coordinate, constraint, workmaps, wrench_scale)
                           for name, triple in triples.items()}
                comparisons = {name: parity(triples[a], triples[b], roles, pp, coordinate, constraint, workmaps, c, k, load)
                    for name, a, b in (("full_vs_reduced", "full", "reduced"), ("full_vs_owner", "full", "owner"),
                        ("reduced_vs_owner", "reduced", "owner"))}
                raw_parity = parity(triples["raw"], triples["reduced"], roles, pp, coordinate, constraint, workmaps, c, k, load)
                if not all(math.isfinite(v) for comparison in comparisons.values() for v in comparison.values()):
                    raise B2NumericalSolveFailure("nonfinite corrected parity metric: "+case)
                results[case] = dict(parity=comparisons, metrics={name: metrics[name] for name in ("full", "reduced", "owner")})
                raw_results[case] = dict(parity=raw_parity, metrics=metrics["raw"],
                    accurate_full_residual_max=float(np.max(np.abs(defect[:, j]))))
        except (B2NumericalSolveFailure, linalg.LinAlgError) as error:
            solve_failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report.update(loads=results, raw_unrefined_loads=raw_results, solve_failure=solve_failure,
            full_refinement_policy="two compensated residual batch corrections using exactly the same sparse matrix/factor")
        save_arrays(archive_path, arrays)
        solved = solve_failure is None and set(results) == set(CASES)
        metric_rows = [v for case in results.values() for v in case["metrics"].values()]
        balance_keys = ("energy_work_relative", "compatible_work_relative", "ground_work_relative",
            "interface_action_balance", "global_ground_applied_balance", "domain_cut_balance")
        predicates = dict(kernel_positive=kernel_positive, saddle_inertia=inertia == expected,
            three_load_solves=solved,
            solution_parity=solved and all(math.isfinite(v) and v <= 1e-10 for case in results.values()
                for comparison in case["parity"].values() for v in comparison.values()),
            equilibrium=solved and all(math.isfinite(v["equilibrium"]) and v["equilibrium"] <= 1e-8 for v in metric_rows),
            weak_constraints=solved and all(math.isfinite(v["weak_constraint"]) and v["weak_constraint"] <= 1e-9 for v in metric_rows),
            energy_work_and_action_balance=solved and all(math.isfinite(v[key]) and v[key] <= 1e-8
                for v in metric_rows for key in balance_keys))
        report["raw_unrefined_predicates"] = dict(solution_parity=solved and all(max(v["parity"].values()) <= 1e-10
                for v in raw_results.values()),
            equilibrium=solved and all(v["metrics"]["equilibrium"] <= 1e-8 for v in raw_results.values()),
            weak_constraints=solved and all(v["metrics"]["weak_constraint"] <= 1e-9 for v in raw_results.values()),
            energy_work=solved and all(v["metrics"]["energy_work_relative"] <= 1e-8 for v in raw_results.values()))
        report.update(predicates=predicates, status="COMPLETED_DEVELOPMENT_ONLY" if all(predicates.values()) else "UNRESOLVED_DEVELOPMENT_CHECK",
            **effective_settings())
        checkpoint(start, report)
        return report
    except BaseException as error:
        earlier_error = True
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        try:
            if arrays:
                save_arrays(archive_path, arrays)
                report["output_npz_sha256"] = sha(archive_path)
            report["elapsed_seconds"] = time.monotonic()-start
            report["peak_rss_bytes"] = peak_rss_bytes()
            resource_error = None
            if report["elapsed_seconds"] > MAX_SECONDS or report["peak_rss_bytes"] > MAX_BYTES:
                resource_error = RuntimeError("saved diagnostic resource ceiling exceeded including final retention")
                report["status"] = "FAILED"
                report["resource_error"] = str(resource_error)
            write_json(output, report)
            if resource_error is not None and not earlier_error:
                raise resource_error
        except BaseException:
            if not earlier_error:
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.root, args.output)
    print(json.dumps({name: result[name] for name in ("status", "minimum_kernel_eigenvalue",
        "spectral_uncertainty", "reduced_saddle_inertia", "elapsed_seconds")}))
    raise SystemExit(0 if all(result["predicates"].values()) else 1)
