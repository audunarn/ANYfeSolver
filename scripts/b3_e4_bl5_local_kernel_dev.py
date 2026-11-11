"""Guarded local conservative-kernel premises for the fixed flat BL5 family.

Development only. All native nodal/cell coordinates are retained; only verified
negative 18-resultant blocks are eliminated. Complete local modes are recorded.
One small native owner supplies actual support/C maps, never 24 dense oracles.
Binary64 uncertainty accounting is nominal, not interval certification, a
coercivity bound, locking/accuracy acceptance or consumption of F03 authority.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_experimental import _shell_reference
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver.corotational import corotational_element_response
from scripts.b3_e4_bl5_coherent_aggregation_dev import model


EPS = np.finfo(float).eps
RATIOS = ((1, 8), (1, 4), (1, 2), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1))
GRIDS = ((16, 4), (32, 8), (64, 16))
MAX_LOCAL_EXTRACTIONS = 96


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def norm(a):
    return float(linalg.norm(a, 2))


def completion_error(a):
    return norm(a - (np.tril(a) + np.tril(a, -1).T))


def spectrum(a):
    values, vectors = linalg.eigh(a, lower=True)
    terms = dict(rounding=64 * EPS * len(a) * norm(a),
                 triangle_completion=completion_error(a),
                 eigen_residual=norm(a @ vectors - vectors * values))
    return values, vectors, terms


def rigid_columns(coordinates, beam=False, center=None):
    """Independent spatial u=a+omega cross X, with all rotations=omega.

    Centering only changes the rigid translation basis, not its six-dimensional
    span. Beam output includes all three nodal rotations and both cell rotations.
    """
    xyz = np.asarray(coordinates)
    centered = xyz - (xyz.mean(axis=0) if center is None else center)
    result = np.zeros((6 * len(xyz) + (6 if beam else 0), 6))
    for j in range(3):
        result[np.arange(len(xyz)) * 6 + j, j] = 1.
        omega = np.eye(3)[j]
        for i, point in enumerate(centered):
            result[6*i:6*i+3, 3+j] = np.cross(omega, point)
            result[6*i+3:6*i+6, 3+j] = omega
        if beam:
            result[-6:, 3+j] = np.tile(omega, 2)
    return result


def classify(matrix, xyz, beam, label):
    """Keep complete stored matrices/spectra; no favorable mode filtering."""
    if matrix.shape != ((42, 42) if beam else (24, 24)) or not np.isfinite(matrix).all():
        raise ValueError("unadmitted local tangent shape/nonfinite value")
    physical = matrix[:24, :24].copy()
    elimination_terms = {}
    stored = {f"{label}_native_matrix": matrix}
    record = dict(native_order=len(matrix), physical_order=24,
                  native_relative_skew=norm(matrix-matrix.T)/max(norm(matrix), 1e-30))
    if record["native_relative_skew"] > 1e-9:
        raise ArithmeticError("stored native local tangent skew exceeds conservative guard")
    raw_values, raw_vectors, raw_terms = spectrum(matrix)
    record.update(native_stored_eigenvalues=raw_values.tolist(),
                  native_stored_uncertainty_terms=raw_terms)
    stored[f"{label}_native_stored_vectors"] = raw_vectors
    if beam:
        d, px, xp = matrix[24:, 24:], matrix[24:, :24], matrix[:24, 24:]
        if norm(d-d.T)/max(norm(d), 1e-30) > 1e-9:
            raise ArithmeticError("stored negative-resultant block skew exceeds conservative guard")
        dv, dvec, dterms = spectrum(d)
        record["negative_resultant_eigenvalues"] = dv.tolist()
        stored[f"{label}_negative_vectors"] = dvec
        try:
            factor = linalg.cho_factor(-d, lower=True)
        except linalg.LinAlgError:
            record.update(negative_block_uncertainty_terms=dterms,
                negative_block_budget=sum(dterms.values()), negative_block_resolved=False,
                local_premise_resolved=False, obstruction="negative18 block Cholesky failed; no elimination")
            return record, stored
        chol = np.tril(factor[0])
        reconstruction = d + chol @ chol.T
        dterms["cholesky_reconstruction"] = norm(reconstruction)
        dbudget = sum(dterms.values())
        record.update(negative_resultant_eigenvalues=dv.tolist(),
                      negative_block_uncertainty_terms=dterms,
                      negative_block_budget=dbudget,
                      negative_block_resolved=bool(dv[-1] < -dbudget))
        stored[f"{label}_negative_vectors"] = dvec
        if not record["negative_block_resolved"]:
            record.update(local_premise_resolved=False,
                          obstruction="18-resultant block negativity unresolved; no elimination inference")
            return record, stored
        lift = linalg.cho_solve(factor, px)  # p=-D^-1 K_px x.
        residual = d @ lift + px
        physical += xp @ lift
        pullback = np.vstack((np.eye(24), lift))
        stored[f"{label}_stationary_lift"] = pullback
        stored[f"{label}_stationary_residual"] = residual
    else:
        record["negative_block_resolved"] = True
    # Invertible unit/diagonal congruence; never omit a zero-diagonal coordinate.
    length = float(np.max(np.linalg.norm(np.asarray(xyz)-xyz[0], axis=1)))
    if length <= 0.:
        raise ValueError("nondegenerate admitted local reference required")
    units = np.r_[np.tile((length, length, length, 1., 1., 1.), len(xyz)),
                  np.ones(6) if beam else np.empty(0)]
    diagonal = np.abs(np.diag(physical) * units**2)
    if not np.max(diagonal) > 0.:
        raise ArithmeticError("local form has no positive normalization scale")
    scale = units / np.sqrt(np.where(diagonal > 0., diagonal, np.max(diagonal)))
    scaled = physical * scale[:, None] * scale[None, :]
    record["physical_relative_skew"] = norm(scaled-scaled.T)/max(norm(scaled), 1e-30)
    if record["physical_relative_skew"] > 1e-9:
        raise ArithmeticError("stored stationary physical tangent skew exceeds conservative guard")
    if beam:
        r = residual * scale
        lifted = lift * scale
        delta = (pullback.T @ matrix @ pullback - physical) * scale[:, None] * scale[None, :]
        elimination_terms = dict(stationary_work_parity=norm(delta),
            stationary_residual_energy=norm(r.T @ linalg.cho_solve(factor, r)),
            local_factor_reconstruction=norm(lifted.T @ reconstruction @ lifted))
        # Complete native saddle spectrum in recorded invertible coordinates.
        pscale = 1. / np.sqrt(np.abs(np.diag(d)))
        native_scale = np.r_[scale, pscale]
        native_scaled = matrix * native_scale[:, None] * native_scale[None, :]
        nv, nvec, nt = spectrum(native_scaled)
        nbudget = sum(nt.values())
        record.update(native_scaled_eigenvalues=nv.tolist(), native_scaled_uncertainty_terms=nt,
                      native_scaled_budget=nbudget,
                      native_scaled_inertia=dict(positive=int(np.sum(nv > nbudget)),
                          negative=int(np.sum(nv < -nbudget)), unresolved=int(np.sum(abs(nv) <= nbudget))))
        stored.update({f"{label}_native_scale": native_scale,
                       f"{label}_native_scaled_vectors": nvec})
    rigid = rigid_columns(xyz, beam)
    normalized = rigid / scale[:, None]
    singular = linalg.svdvals(normalized)
    rank_budget = 64 * EPS * max(normalized.shape) * singular[0]
    if singular[-1] <= rank_budget:
        raise ArithmeticError("independent rigid basis rank unresolved")
    q, _ = linalg.qr(normalized, mode="full")
    qr, complement = q[:, :6], q[:, 6:]
    values, vectors, terms = spectrum(scaled)
    terms.update(elimination_terms)
    terms["rigid_basis_orthogonality"] = (2 * norm(q.T @ q - np.eye(24)) +
        norm(q.T @ q - np.eye(24))**2) * norm(scaled)
    budget = sum(terms.values())
    rv, rvec, rt = spectrum(complement.T @ scaled @ complement)
    complement_budget = budget + sum(rt.values())
    rigid_residual = norm(scaled @ qr)
    counts = dict(positive=int(np.sum(values > budget)), negative=int(np.sum(values < -budget)),
                  unresolved=int(np.sum(abs(values) <= budget)))
    resolved = (counts == dict(positive=18, negative=0, unresolved=6) and
                rigid_residual <= budget and rv[0] > complement_budget)
    if beam:
        resolved &= record["native_scaled_inertia"] == dict(positive=18, negative=18, unresolved=6)
    record.update(physical_eigenvalues=values.tolist(), physical_inertia=counts,
                  physical_uncertainty_terms=terms, physical_budget=budget,
                  rigid_basis_singular_values=singular.tolist(), rigid_basis_rank_budget=rank_budget,
                  rigid_residual=rigid_residual, complement_eigenvalues=rv.tolist(),
                  complement_uncertainty_terms=rt, complement_budget=complement_budget,
                  additional_unresolved_modes=max(0, counts["unresolved"]-6),
                  local_premise_resolved=bool(resolved),
                  obstruction=None if resolved else "complete local kernel is extra, negative or unresolved")
    stored.update({f"{label}_physical_matrix": physical, f"{label}_scale": scale,
                   f"{label}_physical_vectors": vectors, f"{label}_rigid_columns": rigid,
                   f"{label}_rigid_basis": qr, f"{label}_complement_basis": complement,
                   f"{label}_complement_vectors": rvec})
    return record, stored


def connected(cells):
    used = set().union(*(set(row) for row in cells))
    seen, pending = {next(iter(used))}, []
    pending.extend(seen)
    adjacency = {node: set() for node in used}
    for row in cells:
        for node in row:
            adjacency[node].update(row)
    while pending:
        for node in adjacency[pending.pop()] - seen:
            seen.add(node)
            pending.append(node)
    return seen == used


def reference_residual_ratio(matrix, force, coordinates, section=None):
    """Same-unit reference-zero check using physical m/rad/N/Nm scales."""
    force = np.asarray(force, dtype=float)
    if not np.isfinite(force).all():
        raise ArithmeticError("nonfinite local reference residual")
    length = float(np.max(np.linalg.norm(coordinates-coordinates[0], axis=1)))
    units = np.tile((length, length, length, 1., 1., 1.), len(coordinates))
    if section is not None:
        resultants = np.abs(section.elastic) @ np.array((1., 1., 1., 1./length, 1./length, 1./length))
        units = np.r_[units, np.ones(6), np.tile(resultants, 3)]
    rows = np.abs(matrix) @ units
    if np.any((rows == 0.) & (force != 0.)):
        raise ArithmeticError("nonzero reference residual in a zero tangent row")
    ratios = np.divide(np.abs(force), rows, out=np.zeros_like(force), where=rows != 0.)
    return float(np.max(ratios))


def family():
    cases, shell_sources, beam_sources = [], {}, {}
    common_beam_policy = None
    for nx, ny in GRIDS:
        for numerator, denominator in RATIOS:
            nb = nx*numerator//denominator
            definition, policy = model(nx, ny, nb, 2)
            shell, line = definition.shell, definition.lines[0]
            if (shell.s3_elements or shell.j2_curve is not None or definition.points or
                len(definition.lines) != 1 or line.basis != "p0" or
                (shell.young_pa, shell.poisson, shell.thickness_m, shell.density_kg_m3) !=
                    (210e9, .3, .035, 7850.) or definition.q4_pullback_policy != "historical_corotational"):
                raise ValueError("outside fixed elastic Q4/P0 historical reference policy")
            xyz = np.asarray(shell.coordinates_m)
            expected_xyz = np.array([(x, y, 0.) for y in np.linspace(0., .2, ny+1)
                                     for x in np.linspace(0., 1., nx+1)])
            if (not np.array_equal(xyz, expected_xyz) or not connected(shell.q4_elements) or
                set().union(*(set(c) for c in shell.q4_elements)) != set(shell.node_ids)):
                raise ValueError("nonflat/disconnected shell reference")
            tables = [json.loads(b.raw) for b in definition.beams]
            bcells = [row["node_ids"] for row in tables]
            if not connected(bcells):
                raise ValueError("disconnected native beam chain")
            beam_nodes = {}
            for beam_definition, row in zip(definition.beams, tables):
                reference = row["reference"]
                coords = np.asarray(reference["coordinates"])
                frames = np.asarray(reference["nodal_triads"])
                if (np.any(frames != np.tile(np.eye(3), (3, 1, 1))) or
                    np.any(coords[:, 1] != 0.) or np.any(coords[:, 2] != .0675) or
                    not np.all(np.diff(coords[:, 0]) > 0.) or
                    coords[1, 0] != (coords[0, 0]+coords[2, 0])/2.):
                    raise ValueError("outside straight constant-frame/reference-arm geometry")
                beam_policy = {k: row[k] for k in ("schema", "family", "formulation_id", "section",
                    "section_inertia", "quadrature", "static_policy", "dynamic_policy", "production_qualified")}
                if common_beam_policy is None:
                    common_beam_policy = canonical(beam_policy)
                if canonical(beam_policy) != common_beam_policy or row["family"] != "RESULTANT_ELLIPSOID":
                    raise ValueError("beam material/law/quadrature changed across family")
                e, g = 210e9, 210e9/(2*(1+.3))
                expected_elastic = np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                                           e*8.333333333333333e-6, e*8.333333333333333e-6))
                if (not np.array_equal(np.asarray(row["section"]["elastic"]), expected_elastic) or
                    not np.array_equal(np.asarray(row["section"]["metric"]), np.eye(6)) or
                    row["section"]["yield_force"] != 1e20 or row["section"]["hardening"] != 1e9 or
                    row["quadrature"] != 4 or not np.array_equal(np.asarray(row["section_inertia"]), np.eye(6))):
                    raise ValueError("outside fixed native elastic section/law/inertia/quadrature")
                for node, point in zip(row["node_ids"], coords):
                    old = beam_nodes.setdefault(node, point)
                    if not np.array_equal(old, point):
                        raise ValueError("shared native node coordinates differ")
                key = hashlib.sha256(canonical((coords-coords[0]).tolist()) + common_beam_policy).hexdigest()
                if key not in beam_sources:
                    beam_sources[key] = [beam_definition, beam_definition]
                beam_sources[key][1] = beam_definition
            expected_halfcells = tuple((i, cell) for i in range(len(tables)) for cell in (0, 1))
            expected_fixed = {("shell", 1+j*(nx+1), axis) for j in range(ny+1) for axis in range(6)}
            expected_fixed |= {("beam", 101, axis) for axis in range(5)}
            actual_fixed = {(f.domain, f.node, f.component) for f in definition.fixed}
            if (line.beam_halfcells != expected_halfcells or line.shell_nodes != tuple(range(1, nx+2)) or
                actual_fixed != expected_fixed or len(actual_fixed) != len(definition.fixed) or
                policy["support_membership"] != [list(range(a, a+line.multiplier_span))
                    for a in range(0, nb, line.multiplier_span)]):
                raise ValueError("actual connectivity/support/halfcell aggregation mismatch")
            if nx not in shell_sources:
                shell_sources[nx] = definition
            cases.append(dict(nx=nx, ny=ny, halfcells=nb, ratio=f"{numerator}:{denominator}", policy=policy,
                shell_node_ids=list(shell.node_ids), shell_cells=[list(c) for c in shell.q4_elements],
                beam_cells=bcells, beam_definition_sha256=[b.sha256 for b in definition.beams],
                fixed=[asdict(f) for f in definition.fixed],
                definition_sha256=hashlib.sha256(canonical(dict(shell=asdict(shell),
                    beams=[b.sha256 for b in definition.beams], line=line.descriptor(), fixed=sorted(actual_fixed),
                    q4_policy=definition.q4_pullback_policy))).hexdigest()))
    return cases, shell_sources, beam_sources, json.loads(common_beam_policy)


def representative():
    definition, policy = model(16, 4, 32, 2)
    owner = stiffened_shell.create_analysis(definition=definition)
    try:
        if owner._K.shape[0] > 1200 or owner._C.shape[0] > 64:
            raise ValueError("bounded actual-C representative exceeded")
        first = owner._line_samples[0][0]
        for sample in owner._line_samples:
            port = sample[0].port
            if (np.any(port.frame != np.eye(3)) or norm(port.lift) > 1e-13 or
                not np.allclose(sample[0].reference_arm, (0., 0., -.05), rtol=0., atol=1e-13) or
                not np.allclose(port.frame @ sample[0].reference_director, (0., 0., 1.), rtol=0., atol=1e-13) or
                sample[0].thickness != .035):
                raise ValueError("nonconstant frame/arm/nonzero reference lift")
        count, base = owner._K.shape[0], owner._shell_count+6*len(owner._beam_ids)
        r = np.zeros((count, 12))
        shell_xyz = np.asarray(definition.shell.coordinates_m)
        beam_xyz = np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
        r[:owner._shell_count, :6] = rigid_columns(shell_xyz, center=np.zeros(3))
        r[owner._shell_count:base, 6:] = rigid_columns(beam_xyz, center=np.zeros(3))
        for i in range(len(owner._beam_parts)):
            r[base+24*i:base+24*i+6, 9:] = np.tile(np.eye(3), (2, 1))
        support = r[owner._fixed]
        sv = linalg.svdvals(support)
        sbudget = 64*EPS*max(support.shape)*sv[0]
        support_rank = int(np.sum(sv > sbudget))
        common = r[:, :6]+r[:, 6:]
        action = np.asarray(owner._C @ common)
        cbudget = 64*EPS*max(owner._C.shape)*float(linalg.norm(owner._C.toarray(), 2))*norm(common)
        yaw = np.zeros(count)
        root = owner._beam_nodes[101][0]
        for i, point in enumerate(beam_xyz):
            yaw[owner._shell_count+6*i:owner._shell_count+6*i+3] = np.cross((0., 0., 1.), point-root)
            yaw[owner._shell_count+6*i+5] = 1.
        for i in range(len(owner._beam_parts)):
            yaw[base+24*i+2] = yaw[base+24*i+5] = 1.
        if (np.any(yaw[owner._fixed] != 0.) or any(owner._C[:, base+24*i+6:base+24*(i+1)].nnz
                                                for i in range(len(owner._beam_parts)))):
            raise ValueError("yaw/support or first resultant/action layout mismatch")
        cyaw = np.asarray(owner._C @ yaw)
        endpoints = np.asarray(policy["support_endpoints_m"])
        supports = len(endpoints)-1
        expected = np.zeros(5*supports)
        expected[supports:2*supports] = -.5*np.diff(endpoints**2)
        ybudget = 64*EPS*max(owner._C.shape)*float(linalg.norm(owner._C.toarray(), 2))*np.linalg.norm(yaw)
        combined = np.vstack((support, np.asarray(owner._C @ r)))
        complete = linalg.svdvals(combined)
        complete_budget = 64*EPS*max(combined.shape)*complete[0]
        predicates = dict(support_rigid_rank_11=support_rank == 11,
            common_rigid_annihilation=bool(norm(action) <= cbudget),
            yaw_force_work_parity=bool(np.linalg.norm(cyaw-expected) <= ybudget),
            yaw_detected=bool(np.linalg.norm(cyaw) > ybudget),
            supported_rigid_kernel_eliminated=bool(complete[-1] > complete_budget))
        report = dict(identity=owner.preflight()["identity"], policy=policy, predicates=predicates,
            support_singular_values=sv.tolist(), support_budget=sbudget,
            common_rigid_action=action.tolist(), common_rigid_action_budget=cbudget,
            yaw_action=cyaw.tolist(), yaw_expected=expected.tolist(), yaw_budget=ybudget,
            supported_constraint_singular_values=complete.tolist(), supported_constraint_budget=complete_budget,
            normal_director=(first.port.frame @ first.reference_director).tolist())
        return report, {"representative_rigid_columns": r, "representative_yaw": yaw,
                        "representative_rigid_support": support, "representative_rigid_constraint": owner._C @ r}
    finally:
        owner.close()


def evaluate(output):
    start, arrays = time.monotonic(), {}
    report = dict(classification="fixed-family local mechanical development; no qualification/F03",
                  status="ABORTED_DEVELOPMENT_CHECK", local_checks=[], translation_checks=[], local_source_bindings={},
                  row_independence_prerequisite="Reuse separately bound complete24 native norm/work results; not retested here",
                  limitations=["Nominal binary64 budgets, not certified intervals or h-uniform coercivity",
                      "Only stated stress-free flat straight elastic Q4/native laws, frames, lift and supports",
                      "Local rigidity cannot exclude locking or establish physical accuracy",
                      "Geometry-type reuse uses exact relative stored geometry and checked endpoint translations"])
    try:
        cases, shell_sources, beam_sources, beam_policy = family()
        report.update(cases=cases, beam_policy=beam_policy)
        sources = []
        for nx, definition in shell_sources.items():
            shell_model, ids, xyz, index, parts = _shell_reference(definition.shell)
            groups = {}
            for part in parts:
                coords = xyz[part[2][::6]//6]
                key = hashlib.sha256(canonical((coords-coords[0]).tolist())).hexdigest()
                groups.setdefault(key, [part, part])[1] = part
            for key, endpoints in groups.items():
                for role, (eid, element, slots, origin) in zip(("first", "last"), endpoints):
                    if len(sources) >= MAX_LOCAL_EXTRACTIONS:
                        raise ValueError("bounded local extraction inventory exceeded before evaluation")
                    coords = xyz[slots[::6]//6]
                    force, tangent, _ = corotational_element_response(shell_model, eid, element,
                        np.zeros(24), True, committed_state=origin, num_layers=3,
                        tangent_mode="consistent", centered_shell_reference=True)
                    label = f"shell_{nx}_{key[:12]}_{role}"
                    arrays[f"{label}_native_matrix"] = tangent
                    arrays[f"{label}_reference_residual"] = force
                    report["local_source_bindings"][label] = dict(element_id=eid, node_ids=list(element.node_ids),
                        shell_definition_sha256=hashlib.sha256(canonical(asdict(definition.shell))).hexdigest(),
                        reference_residual_ratio=reference_residual_ratio(tangent, force, coords))
                    sources.append((label, tangent, coords, False, key))
        for key, endpoints in beam_sources.items():
            for role, definition in zip(("first", "last"), endpoints):
                if len(sources) >= MAX_LOCAL_EXTRACTIONS:
                    raise ValueError("bounded local extraction inventory exceeded before evaluation")
                element, _ = definition.instantiate()
                ref = element.operator.reference
                if (any(np.linalg.norm(ref.half_cell_lift(cell, .5)) > 1e-13 for cell in (0, 1)) or
                    any(not np.allclose(ref.frame(x), np.eye(3), rtol=0., atol=1e-13) for x in (-1., 0., 1.))):
                    raise ValueError("native geometry prototype violates constant-frame/zero-lift eligibility")
                response = element.operator.evaluate(np.asarray(ref.coordinates), np.zeros((3, 3)),
                    np.asarray(ref.nodal_triads), np.tile(np.eye(3), (2, 1, 1)), np.zeros(18),
                    origin=element.operator.cell.virgin(), check=element.operator.guard)
                tangent = spatial_jacobian(response.residual, response.hessian+response.hessian_low)
                label = f"beam_{key[:12]}_{role}"
                arrays[f"{label}_native_matrix"] = tangent
                arrays[f"{label}_reference_residual"] = response.residual
                report["local_source_bindings"][label] = dict(element_id=element.element_id,
                    node_ids=list(element.node_ids), definition_sha256=definition.sha256,
                    element_identity=element.identity, operator_identity=element.operator.identity,
                    reference_residual_ratio=reference_residual_ratio(tangent, response.residual,
                        np.asarray(ref.coordinates), element.section))
                sources.append((label, tangent, np.asarray(ref.coordinates), True, key))
        if len(sources) > MAX_LOCAL_EXTRACTIONS:
            raise ValueError("bounded distinct local geometry extraction inventory exceeded")
        paired = {}
        for label, tangent, coords, beam, key in sources:
            record, stored = classify(np.asarray(tangent), coords, beam, label)
            record.update(label=label, geometry_key=key, coordinates_m=coords.tolist(), beam=beam)
            report["local_checks"].append(record)
            arrays.update(stored)
            pair_key = (beam, key)
            if record["local_premise_resolved"] and pair_key in paired:
                old, old_label = paired[pair_key]
                scale = arrays[f"{old_label}_scale"]
                other_scale = arrays[f"{label}_scale"]
                difference = (arrays[f"{label}_physical_matrix"]-arrays[f"{old_label}_physical_matrix"])*scale[:, None]*scale[None, :]
                budget = old["physical_budget"] + record["physical_budget"]*float(np.max(scale/other_scale)**2)
                report["translation_checks"].append(dict(first=old_label, last=label,
                    scaled_operator_difference=norm(difference), nominal_budget=budget,
                    resolved=bool(norm(difference) <= budget)))
            else:
                paired[pair_key] = (record, label)
        rep, stored = representative()
        arrays.update(stored)
        report["actual_rigid_support_constraint"] = rep
        predicates = dict(all_local_premises_resolved=all(r["local_premise_resolved"] for r in report["local_checks"]),
            reference_zero_state=all(r["reference_residual_ratio"] <= 32*EPS
                for r in report["local_source_bindings"].values()),
            translation_reuse_resolved=all(r["resolved"] for r in report["translation_checks"]),
            representative_actual_C_support=all(rep["predicates"].values()),
            complete_family_inventory=len(cases) == 24,
            complete_translation_pairs=len(report["translation_checks"]) == len(paired))
        report.update(predicates=predicates, status="LOCAL_PREMISES_RESOLVED_DEVELOPMENT_ONLY" if
            all(predicates.values()) else "UNRESOLVED_LOCAL_PREMISE", elapsed_seconds=time.monotonic()-start)
        return report
    finally:
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
        if report["status"] == "ABORTED_DEVELOPMENT_CHECK":
            output.with_name("partial-result.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists() or args.output.with_name("partial-result.json").exists():
        raise FileExistsError(args.output)
    pools = threadpool_info()
    if (not pools or any(p["num_threads"] != 1 for p in pools) or
        not numba.config.DISABLE_JIT or numba.config.NUMBA_NUM_THREADS != 1):
        raise RuntimeError("effective one-thread/JIT-off settings required")
    result = evaluate(args.output)
    paths = [Path(__file__), Path(inspect.getsourcefile(model)), Path(inspect.getsourcefile(_shell_reference)),
             Path(inspect.getsourcefile(spatial_jacobian)), Path(inspect.getsourcefile(corotational_element_response))]
    result.update(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  imported_anyfesolver=anyfesolver.__file__, threads=pools, jit_disabled=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"status": result["status"], "local_checks": len(result["local_checks"]),
                      "family_cases": len(result["cases"]), "elapsed_seconds": result["elapsed_seconds"]}))
    raise SystemExit(0 if all(result["predicates"].values()) else 1)
