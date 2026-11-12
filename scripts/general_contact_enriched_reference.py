"""Versioned independent238 contact/work/seven-gate assessment, V15.

The COMPLETE enriched structural FE branch is shared, as in registered V4.
Independent contact uses only the separately derived numeric geometry module.
No independent solve is initialized from the supplied production response.
"""
from __future__ import annotations
import contextlib
import hashlib
import inspect
import json
import marshal
import os
import sys
import types
from pathlib import Path
import numpy as np
from scripts.general_contact_enriched_reference_geometry import ContactOracle, finite, require, ReferenceRefused
from scripts.general_contact_enriched_reference_bounds import global_gap_certificate

SCHEMA = 'ENRICHED_MATCHED238_POLYNOMIAL_CONTACT_V15'
STEPS = (1e-5, 3e-6)
ORDERS = (4, 8)
GATE_NAMES = (
    'current_full_equilibrium', 'current_L1_NCP_and_independent_gap',
    'current_force_moment_balance', 'independent_current_integration_force_moment_support',
    'complete238_current_work_and_quadrature', 'independent_current_contact_KKT_equilibrium',
    'independent_nonlinear_reference')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def publish(path, value):
    """Durable partial output; no PARTIAL status can satisfy a gate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, default=serial, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def norms(value):
    a = finite(value)
    return dict(L1=float(np.linalg.norm(a, 1)), L2=float(np.linalg.norm(a)),
                Linf=float(np.linalg.norm(a, np.inf)))


def passed_norms(value, tolerance):
    return all(np.isfinite(v) and v <= tolerance for v in value.values())


def coordinate_inventory(model, system):
    labels = {}
    source_nodes = {n for cell in system.cells for n in cell.node_ids}
    for node in model.mesh.nodes:
        for axis, dof in enumerate(model.mesh.dof_manager.get_node_dofs(node)):
            labels[int(dof)] = dict(kind='retained_nodal', node=int(node),
                axis=('ux', 'uy', 'uz', 'rx', 'ry', 'rz')[axis],
                body='source' if node in source_nodes else 'target',
                coordinate_unit='m' if axis < 3 else 'rad',
                force_unit='N' if axis < 3 else 'Nm', normalization=.04)
    for ordinal, cell in enumerate(system.cells):
        for mode, name in enumerate(('B', 'Br', 'Bs', 'Brs')):
            labels[174+4*ordinal+mode] = dict(kind='physical_bubble', element=int(cell.element_id),
                axis=name, body='source', coordinate_unit='m', force_unit='N', normalization=.04)
    require(set(labels) == set(range(238)) and system.transform.shape == (238, 184),
            'complete174+64 and120+64 inventory')
    return labels


def full_work(model, system, definitions, u, z, production_force, *, progress=None, cancel=lambda: False,
              reuse=None):
    """All238 central AND both one-sided work derivatives, both h/orders."""
    def requested_cancel():
        if cancel():
            from anyfesolver.control import SolveCancelled
            raise SolveCancelled('independent full238 work cancelled')
    requested_cancel()
    labels = coordinate_inventory(model, system)
    rows = []
    for h in STEPS:
        orders = []
        for order in ORDERS:
            requested_cancel()
            oracle = None if reuse is None else reuse.get((h, order))
            if oracle is None:
                oracle = ContactOracle(model, definitions, u, z, order, cancel=cancel)
            else:
                oracle.require_exact_reuse(model, definitions, u, z, order)
            cache_hits_before = getattr(oracle, 'cache_hits', 0)
            base = oracle.base_value
            gradient, positive, negative = np.empty(238), np.empty(238), np.empty(238)
            G = np.empty((25, 238))
            changed = []
            for i in range(238):
                requested_cancel()
                plus, minus = np.array(u, copy=True), np.array(u, copy=True)
                plus[i] += h; minus[i] -= h
                p = oracle.evaluate(plus)
                requested_cancel()
                m = oracle.evaluate(minus)
                requested_cancel()
                gradient[i] = (p['work']-m['work'])/(2*h)
                positive[i] = (p['work']-base['work'])/h
                negative[i] = (base['work']-m['work'])/h
                G[:, i] = (p['gap']-m['gap'])/(2*h)
                if p['signature'] != base['signature'] or m['signature'] != base['signature']:
                    changed.append(i)
                if progress and ((i+1) % 25 == 0 or i == 237):
                    progress(dict(kind='full238_work', status='PARTIAL', h=h, order=order,
                                  measured_coordinates=i+1, authoritative_coordinates=238))
            errors = {name: norms((value-production_force)/.04)
                      for name, value in (('central', gradient), ('positive', positive), ('negative', negative))}
            orders.append(dict(order=order, gradient=gradient, positive=positive, negative=negative,
                gap_gradient=G, errors=errors, support=base['support'], gap=base['gap'],
                signature=base['signature'], changed_coordinates=changed,
                exact_prior_pose_hits=getattr(oracle, 'cache_hits', 0)-cache_hits_before,
                union_certificate=base['union_certificate'],
                all_finite=bool(all(np.all(np.isfinite(x)) for x in (gradient, positive, negative, G)))))
        quadrature = {key: norms((orders[0][key]-orders[1][key])/.04)
                      for key in ('gradient', 'positive', 'negative')}
        rows.append(dict(h=h, orders=orders, quadrature=quadrature))
    numerical = all(o['all_finite'] and all(passed_norms(v, 1e-6) for v in o['errors'].values())
                    for row in rows for o in row['orders'])
    quadrature = all(passed_norms(v, 1e-7) for row in rows for v in row['quadrature'].values())
    return dict(authoritative_coordinates=238, coordinate_inventory=labels, records=rows,
                numerical_work_predicate=numerical, quadrature_control_predicate=quadrature,
                development_complete=bool(numerical and quadrature), branch_method_pending=False,
                limitation='independent piecewise C0 FIRST work; empirical h/order uncertainty, no smooth tie Hessian theorem')


def gap_gradient(oracle, u, h):
    G = np.empty((25, 238))
    for i in range(238):
        p, m = np.array(u, copy=True), np.array(u, copy=True)
        p[i] += h; m[i] -= h
        G[:, i] = (oracle.evaluate(p)['gap']-oracle.evaluate(m)['gap'])/(2*h)
    return finite(G, (25, 238))


@contextlib.contextmanager
def observe_local_moments(system, u):
    """Production comparator ONLY: exact pass-through, all pair calls required.

    The independent geometry module never sees these jets or this hook.
    """
    from anyfesolver.general_contact import enriched_moment_batch as module
    from anyfesolver.general_contact.enriched_nonlinear import NonlinearEnrichedSystem
    require(type(system) is NonlinearEnrichedSystem, 'exact actual nonlinear profile for production comparator')
    caller_code = NonlinearEnrichedSystem.linearize.__code__
    caller_source = Path(inspect.getsourcefile(NonlinearEnrichedSystem.linearize)).resolve()
    compiled_caller = compile(caller_source.read_bytes(), caller_code.co_filename, 'exec', dont_inherit=True, optimize=0)
    class_codes = [c for c in compiled_caller.co_consts if isinstance(c, types.CodeType) and c.co_name == 'NonlinearEnrichedSystem']
    method_codes = [c for parent in class_codes for c in parent.co_consts if isinstance(c, types.CodeType) and c.co_name == 'linearize']
    require(len(method_codes) == 1 and caller_code == method_codes[0], 'authenticated actual production linearize callsite')
    saved = {name: getattr(module, name) for name in ('batch_moments', 'selected_batch_moments')}
    records = []
    for name, fn in saved.items():
        path = Path(inspect.getsourcefile(fn)).resolve()
        tree_code = compile(path.read_bytes(), fn.__code__.co_filename, 'exec', dont_inherit=True, optimize=0)
        expected = [c for c in tree_code.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        require(len(expected) == 1 and fn.__globals__ is vars(module)
                and fn.__module__ == module.__name__ and fn.__qualname__ == name
                and Path(fn.__code__.co_filename).resolve() == path
                and fn.__code__ == expected[0],
                'actual unmodified production moment source/code object')
    identities = {name: dict(module=fn.__module__, qualname=fn.__qualname__,
        source_sha256=sha(inspect.getsourcefile(fn)), first_line=fn.__code__.co_firstlineno)
        for name, fn in saved.items()}
    def wrap(name, original):
        def observed(*args, **kwargs):
            frame = inspect.currentframe().f_back
            try:
                require(frame.f_code is caller_code and frame.f_locals.get('self') is system,
                        'actual current linearize caller, no auxiliary/cache reconstruction')
            finally:
                del frame
            value = original(*args, **kwargs)
            def snapshot(a):
                out = np.array(a, copy=True)
                out.setflags(write=False)
                return out
            records.append((name, tuple(snapshot(a) for a in args[:5]),
                snapshot(kwargs['current_midsurface']),
                snapshot(value.support_gradient), snapshot(value.gap_integral_gradient), id(value)))
            return value
        return observed
    installed = []
    try:
        for name, fn in saved.items():
            setattr(module, name, wrap(name, fn)); installed.append(name)
        yield records, identities
    finally:
        primary = sys.exc_info()[1]
        errors = []
        for name in installed:
            try:
                setattr(module, name, saved[name])
            except BaseException as error:
                errors.append(type(error).__name__)
        restored = not errors and all(getattr(module, k) is v for k, v in saved.items())
        identities['restored'] = restored
        identities['restoration_errors'] = errors
        if not restored:
            if primary is not None:
                primary.add_note('secondary production comparator restoration failure: '+repr(errors))
            else:
                raise ReferenceRefused('all production-comparator hooks must be restored')


def production_field(system, u, z):
    """Actual production contact observer plus physical37 local wrench chain."""
    with observe_local_moments(system, u) as (records, identities):
        raw = system.linearize(u, z)
    require(len(records) == len(system.pairs), 'complete actual32 pair calls, no cache omissions')
    sf, sm, tf, tm = (np.zeros(3) for _ in range(4))
    for pair, (_, args, midsurface, dm, db, result_id) in zip(system.pairs, records, strict=True):
        x, _, _ = pair.map(u)
        require(np.array_equal(args[0], pair.reference_source)
                and np.array_equal(args[1], x[:12].reshape(4, 3))
                and np.array_equal(args[2], x[12:21].reshape(3, 3))
                and np.array_equal(args[3], x[21:25])
                and np.array_equal(midsurface, x[25:].reshape(4, 3)),
                'actual local physical37 argument coverage')
        modes = np.asarray(pair.mode_indices)
        # Global normalization is essential: neither local M nor local b/M is used.
        local = -np.sum((z[modes]/raw.support[modes])[:, None]
            *(db-raw.normalized_gap[modes, None]*dm), axis=0)
        require(local.shape == (37,) and np.all(np.isfinite(local)), 'complete finite local37 force')
        fq, ft, fm = local[:12].reshape(4, 3), local[12:21].reshape(3, 3), local[25:].reshape(4, 3)
        # local is dW, the residual covector. Original physical wrench is -dW.
        sf -= np.sum(fq+fm, axis=0); tf -= np.sum(ft, axis=0)
        sm -= np.sum(np.cross(x[:12].reshape(4, 3), fq)+np.cross(x[25:].reshape(4, 3), fm), axis=0)
        tm -= np.sum(np.cross(x[12:21].reshape(3, 3), ft), axis=0)
    return raw, dict(source_force=sf, source_moment=sm, target_force=tf, target_moment=tm,
                     observed_pair_count=len(records), function_identities=identities,
                     beta_cartesian_wrench_added=False, global_normalization=True)


def rigid_wrench(oracle, body, h=1e-5):
    """Physical rigid-work chain, NOT a surrogate nodal rotation.

    Corner skin AND midsurface normal rotate together, beta and reference area
    remain fixed. All moving frame/domain/support quantities are re-integrated.
    """
    source0, target0 = oracle.base_sources, oracle.base_targets
    results = []
    for rotation in (False, True):
        for axis in range(3):
            values = []
            for sign in (-1, 1):
                sources = {k: (v[0].copy(), v[1].copy(), v[2].copy()) for k, v in source0.items()}
                targets = {k: v.copy() for k, v in target0.items()}
                direction = np.eye(3)[axis]
                if rotation:
                    # Exact finite rigid rotation gives the same infinitesimal work.
                    angle = sign*h; c, s = np.cos(angle), np.sin(angle)
                    K = np.array([[0, -direction[2], direction[1]], [direction[2], 0, -direction[0]], [-direction[1], direction[0], 0]])
                    R = np.eye(3)+s*K+(1-c)*(K @ K)
                    if body == 'source':
                        sources = {k: (q @ R.T, b, n @ R.T) for k, (q, b, n) in sources.items()}
                    else:
                        targets = {k: q @ R.T for k, q in targets.items()}
                elif body == 'source':
                    sources = {k: (q+sign*h*direction, b, n) for k, (q, b, n) in sources.items()}
                else:
                    targets = {k: q+sign*h*direction for k, q in targets.items()}
                pairs = {key: oracle.pair(*key, sources[key[0]], targets[key[1]]) for key in oracle.pair_offsets}
                values.append(oracle.accumulate(sources, targets, pairs)['work'])
            # Physical wrench has the ORIGINAL sign -dW, not residual dW.
            results.append(-(values[1]-values[0])/(2*h))
    return np.asarray(results[:3]), np.asarray(results[3:])


def integration_wrench(model, system, definitions, u, z, raw, wrench):
    rows = []
    for order in ORDERS:
        oracle = ContactOracle(model, definitions, u, z, order)
        sf, sm = rigid_wrench(oracle, 'source')
        tf, tm = rigid_wrench(oracle, 'target')
        rows.append(dict(order=order, source_force=sf, source_moment=sm, target_force=tf, target_moment=tm,
            force_relative_error=max(float(np.linalg.norm(sf-wrench['source_force'])/.04), float(np.linalg.norm(tf-wrench['target_force'])/.04)),
            moment_relative_error=max(float(np.linalg.norm(sm-wrench['source_moment'])/.04), float(np.linalg.norm(tm-wrench['target_moment'])/.04)),
            mass_relative_max=float(np.max(np.abs(oracle.base_value['support']-raw.support)/oracle.base_value['support'])),
            union_certificate=oracle.base_value['union_certificate']))
    return rows


def independent_kkt(system, internal, load, z, work):
    values = []
    for row in work['records']:
        for item in row['orders']:
            residual = system.transform.T @ (internal-item['gap_gradient'].T @ z-load)
            force_error, moment_error = split_equilibrium(system, residual)
            values.append(dict(h=row['h'], order=item['order'], force_relative=force_error, moment_relative=moment_error))
    return dict(rows=values, equilibrium_predicate=bool(len(values) == 4 and all(max(v['force_relative'], v['moment_relative']) <= 1e-6 for v in values)))


def split_equilibrium(system, reduced_residual):
    projected = system.transform @ reduced_residual
    translations, rotations = [], []
    for node in system._nodal.model.mesh.nodes:
        dofs = system._nodal.model.mesh.dof_manager.get_node_dofs(node)
        translations.extend(dofs[:3]); rotations.extend(dofs[3:])
    translations.extend(range(174, 238))  # Every added work-conjugate N equation.
    return float(np.linalg.norm(projected[translations])/.04), float(np.linalg.norm(projected[rotations])/.04)


def control_predicate(descriptor, u):
    c = finite(descriptor['control_matrix'], (2, 174))
    target = finite(descriptor['control_values'], (2,))
    # Exact original one-press zero waypoint scaling, not an absolute1e-12 rule.
    scale = finite(descriptor['control_scales'], (2,))
    require(np.array_equal(scale, np.array([1e-9, 1e-9])) and np.all(target == 0), 'original exact zero-waypoint scales')
    error = float(np.max(np.abs(c @ u[:174]-target)/scale))
    require(error <= 1e-6, 'original scaled control predicate')
    return dict(error=error, scales=scale, target=target)


def independent_reference(model, system, definitions, load, descriptor, production_u, production_z,
                          production_force, production_wrench, *, progress=None):
    """Original max4 modified Newton steps; zero/unloaded, node12 branch explicit."""
    u, z = np.zeros(238), np.zeros(25)
    T, active, h, order = system.transform, np.array([11]), 3e-6, 8
    require(T.shape == (238, 184) and np.all(system.offset == 0), 'original184 free span/zero reference')
    trace, converged = [], False
    for iteration in range(4):
        oracle = ContactOracle(model, definitions, u, z, order, cache_pose_results=True)
        base, G = oracle.base_value, gap_gradient(oracle, u, h)
        internal, tangent, _ = system.structural_response(u)
        residual = T.T @ (internal-G.T @ z-load)
        gap = base['gap']
        C = z-np.maximum(0, z-system.scales*gap)
        eq, ncp = float(np.linalg.norm(residual)/.04), float(np.sum(np.abs(C))/.04)
        active_ok = bool(z[11] > 0 and abs(system.scales[11]*gap[11])/.04 <= 1e-8)
        inactive_ok = bool(np.all(np.delete(gap, active) >= 0) and np.all(np.delete(z, active) == 0))
        finite(residual); finite(gap)
        trace.append(dict(iteration=iteration, equilibrium=eq, NCP_L1=ncp,
            active_positive_and_scaled_gap=active_ok, all24_inactive_zero_force_nonnegative_gap=inactive_ok,
            u=u.copy(), z=z.copy(), union_certificate=base['union_certificate']))
        if progress:
            progress(dict(kind='independent_reference', status='PARTIAL', trace=trace))
        if eq <= 1e-6 and ncp <= 1e-8 and active_ok and inactive_ok and np.all(z >= 0):
            converged = True; break
        if iteration == 3:
            break
        A = G[active] @ T
        K = np.block([[T.T @ tangent @ T, -A.T], [A, np.zeros((1, 1))]])
        update = finite(np.linalg.solve(K, -np.r_[residual, gap[active]]), (185,))
        u += T @ update[:184]; z[active] += update[184:]
        require(np.all(z >= 0), 'independent reference negative force refusal')
        control_predicate(descriptor, u)
    result = dict(converged=converged, trace=trace, initialization='ZERO_UNLOADED_NO_PRODUCTION_SEED',
                  max_iterations=4, h=h, order=order, comparison_predicate=False)
    if not converged:
        return result
    # Complete original both-h/both-order endpoint controls are NOT transferred.
    raw, wrench = production_field(system, u, z)
    work = full_work(model, system, definitions, u, z, raw.force, progress=progress,
                     reuse={(h, order): oracle})
    if not work['development_complete']:
        result.update(final_control=work, later_reference_comparisons='UNRUN_AFTER_COMPLETED_WORK_FAILURE')
        if progress:
            progress(dict(kind='independent_reference_completed_work_failure', status='NONPASS', reference=result))
        return result
    gap = global_gap_certificate(ContactOracle(model, definitions, u, z, 8), u)
    force_comparisons = [norms((item['gradient']-production_force)/.04)
                         for row in work['records'] for item in row['orders']]
    z_comparison = norms((z-production_z)/.04)
    displacement = float(np.linalg.norm(u-production_u)/max(np.linalg.norm(u), np.finfo(float).tiny))
    sf, _ = rigid_wrench(ContactOracle(model, definitions, u, z, 8), 'source')
    physical_force = float(np.linalg.norm(sf-production_wrench['source_force'])/.04)
    result.update(displacements=u, normal_forces=z, final_control=work, gap_certificate=gap,
        complete238_contact_force_comparisons=force_comparisons, complete25_force_comparison=z_comparison,
        displacement_relative=displacement, physical_source_force_relative=physical_force,
        comparison_predicate=bool(work['development_complete'] and gap['lower_m'] >= -50e-6
            and displacement <= 1e-6 and physical_force <= 1e-6
            and len(force_comparisons) == 4 and all(passed_norms(v, 1e-6) for v in force_comparisons)
            and passed_norms(z_comparison, 1e-6)))
    return result


def assess(model, system, load, descriptor, production_u, production_z, *, progress=None):
    require(system.offset.shape == (238,) and system.transform.shape == (238, 184)
            and tuple(system.pressure_ids) == tuple(range(1, 26)), 'actual matched238/184/25 layout')
    u, z = finite(production_u, (238,)), finite(production_z, (25,))
    require(np.all(z >= 0), 'nonnegative production forces')
    definitions = tuple(p.definition for p in system.pairs)
    control = control_predicate(descriptor, u)
    result = dict(schema=SCHEMA, status='PARTIAL', gates={name: None for name in GATE_NAMES},
                  coordinate_count=238, free_count=184, pressure_count=25, control=control,
                  scientific_acceptance=False, full300_acceptance=False)
    def complete(name, predicate, **evidence):
        result.update(evidence)
        result['gates'][name] = bool(predicate)
        result['status'] = 'PARTIAL' if predicate else 'NONPASS'
        if progress:
            progress(dict(kind='completed_original_gate', name=name, predicate=bool(predicate), assessment=result))
        return bool(predicate)
    raw, wrench = production_field(system, u, z)
    internal, _, _ = system.structural_response(u)
    residual = system.transform.T @ (internal+raw.force-load)
    force_error, moment_error = split_equilibrium(system, residual)
    C = z-np.maximum(0, z-system.scales*raw.normalized_gap)
    ncp = float(np.sum(np.abs(C))/.04)
    require(np.all(np.isfinite(residual)) and np.all(np.isfinite(C)), 'finite full physical equations')
    if not complete(GATE_NAMES[0], max(force_error, moment_error) <= 1e-6,
                    equilibrium_force=force_error, equilibrium_moment=moment_error, wrench=wrench):
        return result
    if ncp > 1e-8:
        complete(GATE_NAMES[1], False, NCP_L1=ncp, gap_certificate='UNRUN_AFTER_COMPLETED_NCP_FAILURE')
        return result
    oracle = ContactOracle(model, definitions, u, z, 8)
    gap = global_gap_certificate(oracle, u)
    if not complete(GATE_NAMES[1], ncp <= 1e-8 and gap['lower_m'] >= -50e-6, NCP_L1=ncp, gap_certificate=gap):
        return result
    if not complete(GATE_NAMES[2], max(np.linalg.norm(wrench['source_force']+wrench['target_force']),
            np.linalg.norm(wrench['source_moment']+wrench['target_moment']))/.04 <= 1e-6):
        return result
    integration = integration_wrench(model, system, definitions, u, z, raw, wrench)
    if not complete(GATE_NAMES[3], len(integration) == 2 and [r['order'] for r in integration] == [4, 8]
            and all(max(r['force_relative_error'], r['moment_relative_error']) <= 1e-6
                    and r['mass_relative_max'] <= 1e-11 for r in integration), integration=integration):
        return result
    work = full_work(model, system, definitions, u, z, raw.force, progress=progress)
    if not complete(GATE_NAMES[4], work['development_complete'], complete_work=work):
        return result
    kkt = independent_kkt(system, internal, load, z, work)
    if not complete(GATE_NAMES[5], kkt['equilibrium_predicate'], independent_KKT=kkt):
        return result
    reference = independent_reference(model, system, definitions, load, descriptor, u, z,
                                     raw.force, wrench, progress=progress)
    complete(GATE_NAMES[6], reference['comparison_predicate'], independent_reference=reference)
    result['status'] = 'PASS' if all(v is True for v in result['gates'].values()) else 'NONPASS'
    result['shared_structural_scope'] = 'ACTUAL_CURRENT_NODAL_PLUS_FULL43_ENRICHMENT_DIFFERENCE_64_BETA'
    return result
