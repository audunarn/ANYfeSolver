"""Saved-matrix discriminator for four nested BL5 projected-density supports.

Reuse the previously checked stationary stress reduction without evaluating an
element or eliminating a block again. All free nodal and native cell rotations
remain. This diagnostic is neither a full-K SPD test nor qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info
from anyfesolver._stiffened_shell_ge_experimental import (
    _linear_equilibrium, _scaled_constraint_gap,
)
from anyfesolver._b2_physical_linear_batch_dev import (
    B2NumericalSolveFailure, _accurate_sparse_residual,
)


IDENTITY = "BL5_NESTED_REFERENCE_HALFCELL_AVERAGED_AFFINE_DENSITY_40ROW_DEV_V1"
SCHEMA = "BL5_SAVED_NESTED_PROJECTED_KERNEL_DEV_V1"
CASES = ("fy", "mx", "mz")
PRIOR_IDENTITY = "BL5_REFERENCE_HALFCELL_AVERAGED_AFFINE_DENSITY_DEV_V1"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(a, b, floor=1e-14):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), floor))


def norm2(a):
    return float(linalg.norm(a, 2))


def lower_completion_difference(a):
    return norm2(a-(np.tril(a)+np.tril(a, -1).T))


def nested_embedding():
    """Exact coefficient map: old eighty densities = A @ new forty densities.

    For each channel and adjacent equal-width support pair, the new modes are
    L0+R0 and (-L0+L1+R0+R1)/2. Include both modes in every channel.
    """
    rows, columns, values = [], [], []
    for channel in range(5):
        for support in range(4):
            old = 2*(channel*8+2*support)
            new = 2*(channel*4+support)
            rows.extend((old, old+2, old, old+1, old+2, old+3))
            columns.extend((new, new, new+1, new+1, new+1, new+1))
            values.extend((1., 1., -.5, .5, .5, .5))
    return sparse.coo_matrix((values, (rows, columns)), shape=(80, 40)).tocsr()


def saved_csr(archive, name, shape):
    """Validate dimensions before constructing a saved sparse operator."""
    required = {name+suffix for suffix in ("_data", "_indices", "_indptr", "_shape")}
    require(required.issubset(archive.files), f"missing saved {name} arrays")
    require(np.array_equal(archive[name+"_shape"], shape), f"saved {name} shape mismatch")
    data, indices, indptr = (archive[name+suffix].copy()
                            for suffix in ("_data", "_indices", "_indptr"))
    require(data.ndim == indices.ndim == indptr.ndim == 1,
            f"saved {name} must be CSR vectors")
    require(data.dtype.kind == "f" and indices.dtype.kind in "iu" and indptr.dtype.kind in "iu",
            f"saved {name} requires real coefficients and integer CSR indices")
    require(len(data) == len(indices) and len(indptr) == shape[0]+1 and
            indptr[0] == 0 and indptr[-1] == len(data) and
            np.all(np.diff(indptr) >= 0) and
            np.all((indices >= 0) & (indices < shape[1])) and np.isfinite(data).all(),
            f"invalid saved {name} CSR storage")
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def evaluate(prior, root, output):
    start = time.monotonic()
    prior, root, output = Path(prior), Path(root), Path(output)
    require(prior.suffix == ".json", "prior must use the saved mixed JSON/NPZ schema")
    require(output.suffix == ".json", "output must be a new JSON/NPZ pair")
    require(not output.exists() and not output.with_suffix(".npz").exists(),
            "existing diagnostic output may not be replaced")
    require(output.parent.is_dir(), "diagnostic output parent must already exist")
    pools = threadpool_info()
    require(pools and all(p["num_threads"] == 1 for p in pools), "numerical pools must be one")
    require(numba.get_num_threads() == 1 and numba.config.DISABLE_JIT,
            "Numba threads must be one and JIT disabled")
    previous = json.loads(prior.read_text())
    require(previous.get("schema") is None and
            previous.get("status") == "COMPLETED_DEVELOPMENT_ONLY" and
            previous.get("candidate") == PRIOR_IDENTITY and
            (previous.get("physical_order"), previous.get("native_supported_order"),
             previous.get("constraint_rows")) == (769, 1057, 80) and
            previous.get("predicates") and
            all(v is True for v in previous["predicates"].values()),
            "prior completed projected reduction identity/dimensions required")
    require(set(previous["predicates"]) == {"kernel_positive", "saddle_inertia",
            "solution_parity", "equilibrium", "weak_constraints", "energy_work", "action_balance"},
            "prior predicate schema mismatch")
    prior_source = Path(__file__).with_name("r8_bl5_projected_mixed_system_dev.py")
    require(previous.get("source_sha256") == sha(prior_source),
            "prior stationary-reduction source binding mismatch")
    native = root/"native"
    source_files = {"prior_json": sha(prior), "prior_npz": sha(prior.with_suffix(".npz"))}
    for name in ("inputs.json", "operators.npz", *(case+"/raw-state.npz" for case in CASES)):
        actual = sha(native/name)
        require(previous.get("source_files", {}).get(name) == actual,
                f"prior native input binding mismatch: {name}")
        source_files[name] = actual
    inputs = json.loads((native/"inputs.json").read_text())
    counts = inputs["layout"]["counts"]
    require((inputs["layout"]["nx"], inputs["layout"]["ny"],
             counts["full_q"], counts["physical_dofs"], counts["members"],
             counts["shell_nodes"], counts["beam_nodes"],
             counts["retained_cell_rotations"], counts["retained_resultants"],
             counts["multipliers"]) == (16, 4, 1092, 708, 16, 85, 33, 96, 288, 80),
            "bounded saved coarse physical layout mismatch")
    policy = inputs["policy"]
    require(policy["span_halfcells"] == 4 and policy["beam_halfcells"] == 32 and
            np.array_equal(policy["support_endpoints_m"], np.arange(9)/8.) and
            policy["support_membership"] == [list(range(4*i, 4*i+4)) for i in range(8)],
            "nested embedding requires eight complete equal physical supports")
    with np.load(native/"operators.npz", allow_pickle=False) as archive:
        k = saved_csr(archive, "K", (1092, 1092))
        free, fixed = (archive[name].copy() for name in ("free", "fixed"))
        coordinate = archive["coordinate_scale"].copy()
        shell_work, beam_work = (archive[name].copy() for name in
                                ("shell_rigid_work_map", "beam_rigid_work_map"))
    require(free.shape == (1057,) and fixed.shape == (35,) and
            free.dtype.kind in "iu" and fixed.dtype.kind in "iu" and
            np.array_equal(np.sort(np.r_[free, fixed]), np.arange(1092)),
            "saved full free/fixed partition mismatch")
    require(coordinate.shape == (1092,) and np.isfinite(coordinate).all() and
            np.all(coordinate > 0.) and shell_work.shape == beam_work.shape == (6, 1092) and
            np.isfinite(shell_work).all() and np.isfinite(beam_work).all(),
            "saved physical scales/rigid work maps mismatch")
    with np.load(prior.with_suffix(".npz"), allow_pickle=False) as archive:
        old_c = saved_csr(archive, "C", (80, 1092))
        s = saved_csr(archive, "S", (769, 769))
        lifted = saved_csr(archive, "lift", (1057, 769))
        x, pp, scale = (archive[name].copy() for name in
                        ("physical_slots", "resultant_slots", "physical_scale"))
    require(x.shape == scale.shape == (769,) and pp.shape == (288,) and
            x.dtype.kind in "iu" and pp.dtype.kind in "iu" and
            np.isfinite(scale).all() and np.all(scale > 0.),
            "saved physical/resultant coordinates or scale mismatch")
    expected_pp = np.concatenate([708+24*i+np.arange(6, 24) for i in range(16)])
    require(np.array_equal(pp, expected_pp) and np.isin(pp, free).all() and
            np.array_equal(x, np.setdiff1d(free, pp)),
            "must retain every free shell, beam-node, and native cell rotation")
    require(len(x) <= 1000 and len(free)+40 <= 1500, "bounded oracle order exceeded")
    blocks = previous["local_blocks"]
    require(len(blocks) == 16 and all(
        b["negative_block_budget"] >= 0. and
        b["maximum_negative_eigenvalue"] < -b["negative_block_budget"] and
        b["stationary_backward_residual"] <= 1e-10 for b in blocks),
        "prior checked eighteen-resultant negative block evidence required")
    inherited_error = previous["spectral_uncertainty_terms"]["elimination_and_reconstruction"]
    require(np.isfinite(inherited_error) and inherited_error >= 0.,
            "finite inherited stationary-reduction uncertainty required")
    full_skew = float(sparse.linalg.norm(k-k.T)/sparse.linalg.norm(k))
    require(full_skew <= 1e-9, "saved full K symmetry mismatch")
    a = nested_embedding()
    c = (a.T@old_c).tocsr(); c.eliminate_zeros()
    require(old_c[:, pp].nnz == c[:, pp].nnz == 0, "constraints may not use resultants")
    kfree = k[free][:, free]
    work_pullback = lifted.T@kfree@lifted
    work_parity = float(sparse.linalg.norm(work_pullback-s)/max(sparse.linalg.norm(s), 1e-30))
    cx = c[:, x].toarray()
    constraint_parity = relative((c[:, free]@lifted).toarray(), cx)
    old_constraint_parity = relative((old_c[:, free]@lifted).toarray(), old_c[:, x].toarray())
    free_map = {int(v): i for i, v in enumerate(free)}
    identity_error = sparse.linalg.norm(lifted[[free_map[int(v)] for v in x]]-sparse.eye(len(x)))
    stationary = k[pp][:, free]@lifted
    stationary_backward = float(sparse.linalg.norm(stationary)/max(
        sparse.linalg.norm(k[pp][:, free])*sparse.linalg.norm(lifted), 1e-30))
    require(max(work_parity, constraint_parity, old_constraint_parity,
                identity_error, stationary_backward) <= 1e-10,
            "saved stationary lift/K/C pullback mismatch")
    sd = sparse.diags(scale)
    scaled = (sd@s@sd).toarray()
    require(relative(scaled, scaled.T) <= 1e-9, "scaled retained physical symmetry mismatch")
    constraint = np.asarray(abs(c)@coordinate).ravel()
    require(np.all(constraint > 0.), "nonzero physical constraint row scale required")
    cscaled = cx*scale
    row_norm = np.linalg.norm(cscaled, axis=1)
    require(np.all(row_norm > 0.), "zero nested physical constraint row")
    cscaled /= row_norm[:, None]
    singular = linalg.svdvals(cscaled)
    eps = np.finfo(float).eps
    rank_budget = 64*eps*max(cscaled.shape)*singular[0]
    require(singular[-1] > rank_budget, "nested constraint rank unresolved; no row trimming")
    qbasis, _ = linalg.qr(cscaled.T, mode="full")
    z = qbasis[:, 40:]
    restricted = z.T@scaled@z
    ev, vectors = linalg.eigh(restricted, lower=True)
    physical_ev = linalg.eigvalsh(scaled, lower=True)
    snorm = float(np.max(np.abs(physical_ev)))
    orthogonality = norm2(z.T@z-np.eye(z.shape[1]))
    leakage = norm2(cscaled@z); eta = leakage/singular[-1]
    terms = dict(rounding=64*eps*len(x)*snorm,
        elimination_and_reconstruction=float(inherited_error),
        kernel_leakage=(2*eta+eta**2)*snorm,
        kernel_orthogonality=(2*orthogonality+orthogonality**2)*snorm,
        physical_triangle_completion=lower_completion_difference(scaled),
        eigen_residual=norm2(restricted@vectors-vectors*ev))
    uncertainty = sum(terms.values())
    kernel_positive = bool(ev[0] > uncertainty)
    saddle = np.block([[scaled, cscaled.T], [cscaled, np.zeros((40, 40))]])
    saddle_ev = linalg.eigvalsh(saddle, lower=True)
    saddle_budget = (64*eps*len(saddle)*float(np.max(np.abs(saddle_ev)))+
                     uncertainty+lower_completion_difference(saddle))
    inertia = dict(positive=int(np.sum(saddle_ev > saddle_budget)),
        negative=int(np.sum(saddle_ev < -saddle_budget)),
        unresolved=int(np.sum(np.abs(saddle_ev) <= saddle_budget)))
    expected = dict(positive=len(x), negative=40, unresolved=0)
    arrays = dict(physical_slots=x, resultant_slots=pp, physical_scale=scale,
        constraint_row_scale=row_norm, constraint_scale=constraint, kernel_values=ev,
        kernel_basis=z, kernel_vectors=vectors, saddle_values=saddle_ev,
        physical_values=physical_ev, constraint_singular_values=singular, free=free, fixed=fixed)
    for name, matrix in (("A", a), ("C", c), ("S", s), ("lift", lifted)):
        arrays.update({name+"_data": matrix.data, name+"_indices": matrix.indices,
                       name+"_indptr": matrix.indptr, name+"_shape": matrix.shape})
    loads = []
    for case in CASES:
        with np.load(native/case/"raw-state.npz", allow_pickle=False) as archive:
            load = archive["load"].copy()
        require(load.shape == (1092,) and np.isfinite(load).all(), f"invalid saved {case} load")
        loads.append(load)
    forces = np.column_stack(loads)
    require(np.max(np.abs(forces[pp])) == 0., "saved resultant RHS must be zero")
    results = {}; raw_results = {}; solve_failure = None
    full_factors = reduced_factors = full_refinements = 0
    try:
        full = sparse.bmat([[kfree, c[:, free].T], [c[:, free], None]], format="csc")
        try:
            full_factor = splu(full)
        except RuntimeError as error:
            # SuperLU has no separate numerical exception type. Recognize its
            # singular-factor message only; unrelated RuntimeErrors propagate.
            if str(error) != "Factor is exactly singular":
                raise
            raise B2NumericalSolveFailure("nested full sparse KKT: "+str(error)) from error
        full_factors = 1
        rhs = np.vstack((forces[free], np.zeros((40, 3))))
        raw_solution = full_factor.solve(rhs)
        if not np.isfinite(raw_solution).all():
            raise B2NumericalSolveFailure("nonfinite raw full three-load solution")
        # The installed owner mandates one compensated-residual correction,
        # using this SAME full matrix/factor and no reduced-oracle information.
        raw_defect = _accurate_sparse_residual(full, raw_solution, rhs)
        correction = full_factor.solve(-raw_defect)
        full_refinements = 1
        solution = raw_solution+correction
        arrays.update(raw_full_solution=raw_solution, raw_full_accurate_defect=raw_defect,
                      full_refinement_correction=correction)
        reduced = linalg.solve(saddle,
            np.vstack((forces[x]*scale[:, None], np.zeros((40, 3)))),
            assume_a="sym", lower=True)
        reduced_factors = 1
        if not (np.isfinite(solution).all() and np.isfinite(reduced).all()):
            raise B2NumericalSolveFailure("nonfinite full or reduced three-load solution")
        shell_end = 6*counts["shell_nodes"]
        roles = dict(shell=x[x < shell_end], beam_nodes=x[(x >= shell_end) & (x < 708)],
                     cell_rotations=x[x >= 708])
        require(all(len(slots) > 0 for slots in roles.values()), "missing physical coordinate role")
        for j, case in enumerate(CASES):
            q = np.zeros(1092); q[free] = lifted@(reduced[:len(x), j]*scale)
            lm = reduced[len(x):, j]/row_norm
            direct = np.zeros(1092); direct[free] = solution[:len(free), j]
            direct_lm = solution[len(free):, j]
            residual = k@q+c.T@lm-forces[:, j]
            direct_residual = k@direct+c.T@direct_lm-forces[:, j]
            support = np.zeros(1092); support[fixed] = residual[fixed]
            direct_support = np.zeros(1092); direct_support[fixed] = direct_residual[fixed]
            parity = dict(physical=relative(q[x]/coordinate[x], direct[x]/coordinate[x]),
                resultants=relative(q[pp]/coordinate[pp], direct[pp]/coordinate[pp]),
                multipliers=relative(lm*constraint, direct_lm*constraint),
                support=relative(support*coordinate, direct_support*coordinate))
            parity.update({name: relative(q[slots]/coordinate[slots], direct[slots]/coordinate[slots])
                           for name, slots in roles.items()})
            work = float(q@forces[:, j]); direct_work = float(direct@forces[:, j])
            energy = float(.5*q@(k@q)); direct_energy = float(.5*direct@(k@direct))
            action = c.T@lm; direct_action = c.T@direct_lm
            resisting = np.vstack((shell_work@action, beam_work@action))
            direct_resisting = np.vstack((shell_work@direct_action, beam_work@direct_action))
            parity.update(energy=relative(np.array(energy), np.array(direct_energy)),
                          work=relative(np.array(work), np.array(direct_work)),
                          interface_resisting_wrenches=relative(resisting, direct_resisting))
            raw = np.zeros(1092); raw[free] = raw_solution[:len(free), j]
            raw_lm = raw_solution[len(free):, j]
            raw_residual = k@raw+c.T@raw_lm-forces[:, j]
            raw_support = np.zeros(1092); raw_support[fixed] = raw_residual[fixed]
            raw_parity = dict(physical=relative(q[x]/coordinate[x], raw[x]/coordinate[x]),
                resultants=relative(q[pp]/coordinate[pp], raw[pp]/coordinate[pp]),
                multipliers=relative(lm*constraint, raw_lm*constraint),
                support=relative(support*coordinate, raw_support*coordinate))
            raw_parity.update({name: relative(q[slots]/coordinate[slots], raw[slots]/coordinate[slots])
                               for name, slots in roles.items()})
            raw_work = float(raw@forces[:, j]); raw_energy = float(.5*raw@(k@raw))
            raw_action = c.T@raw_lm
            raw_resisting = np.vstack((shell_work@raw_action, beam_work@raw_action))
            raw_parity.update(energy=relative(np.array(energy), np.array(raw_energy)),
                work=relative(np.array(work), np.array(raw_work)),
                interface_resisting_wrenches=relative(resisting, raw_resisting))
            raw_results[case] = dict(parity=raw_parity, equilibrium=_linear_equilibrium(
                raw_residual[free], forces[free, j], coordinate[free]),
                weak_constraint=_scaled_constraint_gap(c@raw, constraint),
                energy_work_relative=relative(np.array(2*raw_energy), np.array(raw_work)),
                work_J=raw_work, energy_J=raw_energy,
                accurate_full_residual_max=float(np.max(np.abs(raw_defect[:, j]))))
            results[case] = dict(parity=parity, work_J=work, direct_work_J=direct_work,
                energy_J=energy, direct_energy_J=direct_energy,
                equilibrium=_linear_equilibrium(residual[free], forces[free, j], coordinate[free]),
                direct_equilibrium=_linear_equilibrium(direct_residual[free], forces[free, j], coordinate[free]),
                weak_constraint=_scaled_constraint_gap(c@q, constraint),
                direct_weak_constraint=_scaled_constraint_gap(c@direct, constraint),
                energy_work_relative=relative(np.array(2*energy), np.array(work)),
                direct_energy_work_relative=relative(np.array(2*direct_energy), np.array(direct_work)),
                interface_resisting_wrenches=resisting.tolist(),
                interface_action_balance=float(np.max(np.abs(resisting.sum(axis=0)))))
            arrays.update({case+"_"+key: value for key, value in dict(q=q, load=forces[:, j],
                multipliers=lm, reactions=support, direct_q=direct,
                direct_multipliers=direct_lm, direct_reactions=direct_support,
                raw_q=raw, raw_multipliers=raw_lm, raw_reactions=raw_support).items()})
    except (B2NumericalSolveFailure, linalg.LinAlgError) as error:
        solve_failure = type(error).__name__+": "+str(error)
    solved = solve_failure is None and set(results) == set(CASES)
    predicates = dict(kernel_positive=kernel_positive, saddle_inertia=inertia == expected,
        three_load_solves=solved,
        solution_parity=solved and all(max(v["parity"].values()) <= 1e-10 for v in results.values()),
        equilibrium=solved and all(max(v["equilibrium"], v["direct_equilibrium"]) <= 1e-8 for v in results.values()),
        weak_constraints=solved and all(max(v["weak_constraint"], v["direct_weak_constraint"]) <= 1e-9 for v in results.values()),
        energy_work=solved and all(max(v["energy_work_relative"], v["direct_energy_work_relative"]) <= 1e-8 for v in results.values()),
        action_balance=solved and all(v["interface_action_balance"] <= 1e-8 for v in results.values()))
    raw_predicates = dict(
        solution_parity=solved and all(max(v["parity"].values()) <= 1e-10 for v in raw_results.values()),
        equilibrium=solved and all(v["equilibrium"] <= 1e-8 for v in raw_results.values()),
        weak_constraints=solved and all(v["weak_constraint"] <= 1e-9 for v in raw_results.values()),
        energy_work=solved and all(v["energy_work_relative"] <= 1e-8 for v in raw_results.values()))
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    helpers = {prior_source.name: sha(prior_source),
        "_stiffened_shell_ge_experimental.py": sha(_linear_equilibrium.__code__.co_filename),
        "_b2_physical_linear_batch_dev.py": sha(_accurate_sparse_residual.__code__.co_filename)}
    return dict(schema=SCHEMA, candidate=IDENTITY,
        status="COMPLETED_DEVELOPMENT_ONLY" if all(predicates.values()) else "UNRESOLVED_DEVELOPMENT_CHECK",
        scope="one nested saved coarse virgin kernel/parity discriminator; no old/new displacement equality, material-history, growing-space, locking, accuracy or F03 acceptance",
        source_sha256=sha(__file__), source_files=source_files, helpers=helpers,
        prior_helper_bindings=previous["helpers"], anyfesolver_import=anyfesolver.__file__,
        numba_threads=numba.get_num_threads(), jit_disabled=bool(numba.config.DISABLE_JIT), pools=pools,
        physical_order=len(x), native_supported_order=len(free), old_constraint_rows=80,
        constraint_rows=40, kernel_order=z.shape[1], full_supported_saddle_order=len(free)+40,
        nested_support_endpoints_m=(np.arange(5)/4.).tolist(), nested_span_halfcells=8,
        physical_scale_policy="unchanged saved scale; inherited elimination uncertainty only",
        local_blocks=blocks, full_K_skew_relative=full_skew, work_pullback_relative=work_parity,
        constraint_pullback_relative=constraint_parity, old_constraint_pullback_relative=old_constraint_parity,
        lift_identity_error=float(identity_error), stationary_backward_residual=stationary_backward,
        constraint_min_singular=float(singular[-1]), rank_budget=float(rank_budget),
        kernel_leakage=leakage, kernel_orthogonality=orthogonality,
        minimum_kernel_eigenvalue=float(ev[0]), spectral_uncertainty_terms=terms,
        spectral_uncertainty=float(uncertainty), reduced_saddle_inertia=inertia,
        expected_reduced_saddle_inertia=expected, saddle_budget=float(saddle_budget),
        implied_native_inertia=dict(positive=len(x), negative=len(pp)+40, unresolved=0)
            if kernel_positive and inertia == expected else None,
        eigenvalues_archive={"kernel": "kernel_values", "retained_physical": "physical_values",
                             "reduced_saddle": "saddle_values"},
        output_npz_sha256=sha(output.with_suffix(".npz")),
        full_general_factorizations=full_factors, reduced_factorizations=reduced_factors,
        full_accurate_residual_corrections=full_refinements,
        full_refinement_policy="one accurate A*x-rhs correction with unchanged cached full sparse factor",
        raw_unrefined_loads=raw_results, raw_unrefined_predicates=raw_predicates,
        solve_failure=solve_failure, loads=results, predicates=predicates,
        elapsed_seconds=time.monotonic()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True, help="immutable prior mixed.json (NPZ sibling)")
    parser.add_argument("--input", type=Path, required=True, help="saved native fixture parent")
    parser.add_argument("--output", type=Path, required=True, help="new JSON output (NPZ sibling)")
    args = parser.parse_args()
    result = evaluate(args.prior, args.input, args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "minimum_kernel_eigenvalue",
        "spectral_uncertainty", "reduced_saddle_inertia", "elapsed_seconds")}))
    raise SystemExit(0 if all(result["predicates"].values()) else 1)
