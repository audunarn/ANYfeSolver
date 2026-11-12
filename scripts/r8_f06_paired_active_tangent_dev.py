"""One frozen installed saved-state tangent assessment; no equilibrium/history solve.

Only the explicit supervised CLI grants numerical execution. Helpers import the
numerical stack lazily, so preparation and AST/hash inspection are stdlib-only.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'docs/agent_plans/development/r8-f06-log-domain-20261010/paired-displacement-v1'
LOADED = BASE / 'loaded-x-v1'
DOC = LOADED / 'active-tangent-v1'
PACKAGE = BASE / 'package'
ACCEPTED = LOADED / 'attempt-installed-history1/accepted.bin'
REBASED = LOADED / 'attempt-installed-controls1/rebased.bin'
INPUT_SHA = 'cbd6cddee612a1b2734b3a643f8f98b6b89391fc33247c4613b36b2fa05b4786'
REBASED_SHA = 'a824248c93cc8f2c91839d8a0b973ef9e5afc4ba99aaab925b7660aa32fe9a30'
SEED_SHA = '970e0360f22ed348ca2eaa7eee0d2d168d6d686d0a91a750f3ccd31794865f91'
WHEEL_SHA = '0ed73724bf3c8ccffcc0134628d94720d640840f9592ac8890779f54285ee3d8'
STEPS = (1e-4, 1e-5, 1e-6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        return plain(asdict(value))
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')


def checked(flags):
    if not flags or not all(type(v) is bool and v for v in flags.values()):
        raise AssertionError(flags)
    return True


def adjacent_plateau(errors, threshold=1e-5):
    return any(all(len(values) == 3 and values[i] is not None and
                   values[i+1] is not None and values[i] <= threshold and
                   values[i+1] <= threshold for values in errors.values())
               for i in (0, 1)) if errors else False


def metrics(error, action, rows, weights):
    import numpy as np
    e, a, w = error[rows], action[rows], weights[rows]
    raw_e, raw_a = float(np.linalg.norm(e)), float(np.linalg.norm(a))
    work_e, work_a = float(np.linalg.norm(e*w)), float(np.linalg.norm(a*w))
    return dict(raw_absolute_error=raw_e, raw_action_norm=raw_a,
                work_absolute_error=work_e, work_action_norm=work_a,
                raw_relative=None if raw_a == 0. else raw_e/raw_a,
                work_relative=None if work_a == 0. else work_e/work_a,
                exercised=bool(raw_a > 0. and work_a > 0.))


def frozen_direction(owner, residual):
    import numpy as np
    count = owner._K.shape[0]
    beam_end = owner._shell_count+6*len(owner._beam_ids)
    rng = np.random.default_rng(818)
    direction = np.zeros(len(residual))
    direction[:beam_end] = rng.normal(size=beam_end)*1e-4
    for i in range(len(owner._beam_parts)):
        base = beam_end+24*i
        direction[base:base+6] = rng.normal(size=6)*1e-4
        p = direction[base+6:base+24].reshape(3, 6)
        p[:, :3] = rng.normal(size=(3, 3))*1000.
        p[:, 3:] = rng.normal(size=(3, 3))*30.
    multipliers = direction[count:].reshape(-1, 6)
    multipliers[:, :3] = rng.normal(size=multipliers[:, :3].shape)*1000.
    multipliers[:, 3:] = rng.normal(size=multipliers[:, 3:].shape)*30.
    direction[owner._fixed] = 0.
    return direction


def row_groups(owner, size):
    import numpy as np
    count = owner._K.shape[0]
    beam_end = owner._shell_count+6*len(owner._beam_ids)
    groups = dict(shell=owner._free[owner._free < owner._shell_count],
                  beam=owner._free[(owner._free >= owner._shell_count)&(owner._free < beam_end)],
                  constraints=np.arange(count, size))
    for i, (element, _, _) in enumerate(owner._beam_parts):
        family = 'fibre' if hasattr(element.section, 'fibres') else 'generalized'
        key = family+'_retained'
        groups[key] = np.r_[groups.get(key, np.array([], dtype=int)),
                            np.arange(beam_end+24*i, beam_end+24*(i+1))]
    return groups


def identities(value):
    """IDs of every nested history/state object, including ndarray leaves."""
    from dataclasses import fields, is_dataclass
    if isinstance(value, dict):
        return (id(value), tuple((k, identities(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (id(value), tuple(identities(v) for v in value))
    if is_dataclass(value):
        return (id(value), tuple((f.name, identities(getattr(value, f.name))) for f in fields(value)))
    return id(value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture(owner):
    d = owner._finite_driver
    history = dict(shell_accepted=d._shell_origins, shell_predecessors=d._shell_predecessors,
                   beam_accepted=d._beam_origins, beam_predecessors=d._beam_predecessors)
    return dict(checkpoint=owner.checkpoint(), generation=owner._generation,
                accepted=owner._accepted, controller=d._adaptive_state,
                driver=d, driver_accepted=getattr(d,'_accepted',None),
                driver_accepted_hash=fingerprint(getattr(d,'_accepted',None)),
                state=d._state, state_hash=fingerprint(d._state),
                histories=history, history_ids={k: identities(v) for k, v in history.items()},
                history_hash={k: fingerprint(v) for k, v in history.items()},
                applied=d._displacement_applied_load,
                applied_hash=fingerprint(d._displacement_applied_load),
                seed=getattr(owner, '_displacement_snapshot', None))


def atomic_flags(owner, before):
    d = owner._finite_driver
    flags = dict(checkpoint=owner.checkpoint() == before['checkpoint'],
                 generation=owner._generation == before['generation'],
                 state_object=d._state is before['state'],
                 state_words=fingerprint(d._state) == before['state_hash'],
                 accepted_object=owner._accepted is before['accepted'],
                 driver_object=d is before['driver'],
                 driver_accepted_object=getattr(d,'_accepted',None) is before['driver_accepted'],
                 driver_accepted_words=fingerprint(getattr(d,'_accepted',None)) == before['driver_accepted_hash'],
                 controller=d._adaptive_state is before['controller'],
                 applied_load_object=d._displacement_applied_load is before['applied'],
                 applied_load_bytes=fingerprint(d._displacement_applied_load) == before['applied_hash'],
                 snapshot_object=getattr(owner, '_displacement_snapshot', None) is before['seed'])
    for key, value in dict(shell_accepted=d._shell_origins, shell_predecessors=d._shell_predecessors,
                           beam_accepted=d._beam_origins, beam_predecessors=d._beam_predecessors).items():
        flags[key+'_objects'] = identities(value) == before['history_ids'][key]
        flags[key+'_words'] = fingerprint(value) == before['history_hash'][key]
    flags['objects'] = all(flags[k] for k in flags if k.endswith(('_object','_objects')))
    return flags


def _native_signature(element, material):
    """Use actual canonical local material fields, including paired knot/sign data."""
    from decimal import Decimal as D, localcontext
    def pair(x):
        # Material vector pair is [high-list, low-list]; scalar pair is [high,low].
        return D.from_float(float(x[0]))+D.from_float(float(x[1]))
    result = []
    with localcontext() as ctx:
        ctx.prec = 80
        for j, row in enumerate(material['stations']):
            if hasattr(element.section, 'fibres'):
                old = material['origin']['stations'][j]['rows']
                new = material['history']['stations'][j]['rows']
                for i, (field, fibre) in enumerate(zip(row['fibres'], element.section.fibres)):
                    accumulated = pair(new[i][2:])
                    increment = pair(field['plastic_increment'])
                    result.append(dict(station=j, fibre=fibre.fibre_id, branch=field['branch'],
                        active=pair(new[i][2:])-pair(old[i][2:]) > 0,
                        flow_sign=int(increment > 0)-int(increment < 0),
                        hardening_segment=sum(D.from_float(x) <= accumulated for x in fibre.curve.plastic_strain)-1,
                        hardening_knot=accumulated in tuple(map(D.from_float, fibre.curve.plastic_strain[1:]))))
            else:
                old = material['origin']['stations'][j]
                new = material['history']['stations'][j]
                deltas = [pair(a)-pair(b) for a, b in zip(new['plastic'], old['plastic'])]
                result.append(dict(station=j, branch=row['branch'],
                    active=pair(new['accumulated'])-pair(old['accumulated']) > 0,
                    flow_sign=[int(x > 0)-int(x < 0) for x in deltas], hardening_segment='linear'))
    return result


class ObservedAssembly:
    """Read-only profiler of actual scatter returns; no constitutive replacement.

    Copies locals synchronously before the next assembly invalidates scratch.
    The previous profiler is restored even on refusal/cancellation/error.
    """
    def __init__(self, owner, out):
        self.owner, self.driver, self.out = owner, owner._finite_driver, out
        self.shell, self.beam = self.driver._shell_predecessors, self.driver._beam_predecessors
        self.applied = self.driver._displacement_applied_load
        self.calls = 0

    def evaluate(self, label, state, *, cancellation_token=None):
        import numpy as np
        from scipy import sparse
        from anyfesolver.control import cancellation_safe_point
        cancellation_safe_point(cancellation_token, 'paired-tangent.before')
        self.calls += 1
        locals_copy, shell_metadata = [], []
        target = self.driver._assemble.__func__.__code__
        previous_profile = sys.getprofile()
        def observed(frame, event, arg):
            if event != 'return':
                return
            code = frame.f_code
            if code.co_name == 'scatter' and frame.f_back.f_code is target:
                f, parent = frame.f_locals, frame.f_back.f_locals
                item = dict(slots=np.array(f['slots'], copy=True),
                            residual=np.array(f['force'], copy=True),
                            tangent=np.array(f['tangent'], copy=True))
                ordinal = len(locals_copy)
                if ordinal < len(self.owner._shell_parts):
                    item.update(kind='shell', index=ordinal, candidate=copy.deepcopy(parent['candidate']))
                elif ordinal < len(self.owner._shell_parts)+len(self.owner._beam_parts):
                    i = ordinal-len(self.owner._shell_parts)
                    response = parent['response']
                    item.update(kind='native', index=i, material=json.loads(response.material),
                                hessian=np.array(response.hessian, copy=True),
                                hessian_low=np.array(response.hessian_low, copy=True),
                                paired_input=dict(positions=np.array(parent['beam_positions'][parent['nodes']], copy=True),
                                    position_low=np.array(parent['beam_low'][parent['nodes']], copy=True),
                                    vertices=np.array(state.frames[parent['nodes']], copy=True),
                                    vertices_low=np.array(state.frames_low[parent['nodes']], copy=True),
                                    rotations=np.array(state.cells[i], copy=True), rotations_low=np.array(state.cells_low[i], copy=True),
                                    resultants=np.array(state.resultants[i], copy=True), resultants_low=np.array(state.resultants_low[i], copy=True)))
                else:
                    i = ordinal-len(self.owner._shell_parts)-len(self.owner._beam_parts)
                    trial = parent['trial']
                    item.update(kind='point', index=i,
                        spatial_jacobian=np.array(trial.pose.spatial_jacobian, copy=True),
                        constraint_jacobian=np.array(trial.pose.constraint_jacobian, copy=True),
                        coordinate_map=np.array(trial.coordinate_map, copy=True),
                        multipliers=np.array(state.multipliers[6*i:6*i+6], copy=True),
                        paired_input=dict(positions=np.array(parent['positions'], copy=True),
                            position_low=np.array(parent['position_low'], copy=True),
                            anchors=np.array(parent['anchors'], copy=True),
                            anchors_low=np.array((np.zeros((3,3)), state.frames_low[parent['beam_node']]), copy=True),
                            rotation_coordinates=np.array(parent['coordinates'], copy=True)))
                locals_copy.append(item)
            elif code.co_name == 'plane_stress_return_map' and code.co_filename.endswith('plasticity.py'):
                f = frame.f_locals
                shell_metadata.append({k: copy.deepcopy(f[k]) for k in (
                    'converged', 'scaled_residual', 'fallback_mask', 'method', 'max_iterations', 'tolerance') if k in f})
        try:
            sys.setprofile(observed)
            value = self.driver._assemble(state, 0., self.shell, self.beam, applied_load=self.applied)
            cancellation_safe_point(cancellation_token, 'paired-tangent.after')
        except BaseException as exc:
            write(self.out/(label+'-partial.json'), dict(error_type=type(exc).__name__, message=str(exc),
                  local_contributions=locals_copy, shell_metadata=shell_metadata))
            raise
        finally:
            sys.setprofile(previous_profile)
        np.savez(self.out/(label+'-state.npz'), **{k: getattr(state,k) for k in (
            'q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low',
            'multipliers','shell_anchors') if getattr(state,k) is not None}, R=value[0], applied_load=self.applied)
        sparse.save_npz(self.out/(label+'-K.npz'), value[1])
        # JSON copies include the actual low words, local tangent and parent material.
        write(self.out/(label+'-locals.json'), dict(contributions=locals_copy, shell_metadata=shell_metadata))
        branches = []
        curve = self.owner._shell_model.get_material('shell').hardening_curve
        for item in locals_copy:
            if item['kind'] == 'shell':
                old, new = self.shell[item['index']], item['candidate']
                deltas = np.asarray(new['plastic_strain'])-np.asarray(old['plastic_strain'])
                alpha = np.asarray(new['alpha'])
                knots = (curve.eps_p_y1, curve.eps_p_y2)
                item['branch_signature'] = dict(active=(alpha-np.asarray(old['alpha'])) > 0.,
                    flow_sign=np.sign(deltas).astype(int),
                    hardening_segment=np.sum(alpha[:,None] >= np.asarray(knots)[None,:], axis=1),
                    hardening_knot=np.any(alpha[:,None] == np.asarray(knots)[None,:], axis=1))
            elif item['kind'] == 'native':
                item['branch_signature'] = _native_signature(self.owner._beam_parts[item['index']][0], item['material'])
            if item['kind'] != 'point':
                branches.append(plain(item['branch_signature']))
        write(self.out/(label+'-branches.json'), dict(branches=branches, smooth_signature=smooth_signature(branches),
            shell_metadata=shell_metadata,
            shell_predecessors_sha256=fingerprint(self.shell), beam_predecessors_sha256=fingerprint(self.beam),
            applied_load_sha256=fingerprint(self.applied), same_parent=True,
            profiler_restored=sys.getprofile() is previous_profile,
            provenance=dict(shell='actual scatter candidate minus immediate predecessor; DNVC208 alpha segments',
                native='actual response.material canonical origin/history/stations bytes from scatter caller',
                convergence='actual plane_stress_return_map return-frame locals',
                bisection_fallback='per-point bisection-use mask not exposed by canonical return')))
        expected = len(self.owner._shell_parts)+len(self.owner._beam_parts)+len(self.owner._connections)
        if len(locals_copy) != expected:
            raise ValueError('incomplete local scatter capture')
        return value, locals_copy, branches, shell_metadata


def smooth_signature(branches):
    """Constitutive differentiability markers; vector signs remain raw diagnostics.

    Ellipsoid/J2 component signs do not select a return branch. Physical scalar
    fibres retain both their PLASTIC_POSITIVE/NEGATIVE selector and scalar sign.
    """
    selected=[]
    for row in branches:
        if isinstance(row,dict):
            selected.append({k:row[k] for k in ('active','hardening_segment','hardening_knot')})
        else:
            native=[]
            for item in row:
                markers={k:item[k] for k in ('station','branch','active','hardening_segment')}
                if 'fibre' in item:
                    markers.update(fibre=item['fibre'],flow_sign=item['flow_sign'],
                                   hardening_knot=item['hardening_knot'])
                native.append(markers)
            selected.append(native)
    return selected


def same_smooth_branch(left,right):
    return smooth_signature(left) == smooth_signature(right)


def branch_smooth(branches, metadata):
    import numpy as np
    for row in branches:
        if isinstance(row, dict):
            if any(row['hardening_knot']):
                return False
        else:
            if any(item['branch'] in ('YIELD_BOUNDARY', 'HARDENING_KNOT') or item.get('hardening_knot', False) for item in row):
                return False
    return bool(metadata) and all('converged' in row and 'fallback_mask' in row and
        np.all(row['converged']) and not np.any(row['fallback_mask']) for row in metadata)


def assess(out):
    import numpy as np
    from anyfesolver import stiffened_shell as ss
    from anyfesolver.control import CancellationToken, SolveCancelled
    from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot
    from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
    from anyfesolver._stiffened_shell_rotation_charts import physical_frames
    from scripts.r8_f06_cyclic_material_dev import definition
    assert sha(ACCEPTED) == INPUT_SHA and sha(REBASED) == REBASED_SHA
    owner, chart_owner = ss.create_analysis(definition=definition('coupled')), None
    before, report = None, dict(status='INCOMPLETE', scope='one installed saved-state endpoint/direction',
        steps=STEPS, rng_seed=818, accepted_sha256=INPUT_SHA, rebased_sha256=REBASED_SHA,
        accepted_publication=False, independent_acceptance=False, full_F05_F06_G3=False)
    try:
        owner.restore(ACCEPTED.read_bytes())
        d, before = owner._finite_driver, capture(owner)
        metadata, saved = decode_snapshot(owner, before['checkpoint'])
        np.savez(out/'input-arrays.npz', **{k:v for k,v in saved.items() if isinstance(v,np.ndarray)})
        write(out/'input-metadata.json', dict(metadata=metadata, force_seed_sha256=hashlib.sha256(saved['force_seed']).hexdigest(),
            history_hashes=before['history_hash'], history_object_ids=before['history_ids']))
        assert metadata['generation'] == owner._generation == 9 and metadata['steps'] == 1
        assert metadata['seed_sha256'] == SEED_SHA and hashlib.sha256(saved['force_seed']).hexdigest() == SEED_SHA
        assert owner.checkpoint() == ACCEPTED.read_bytes()
        assert np.array_equal(d._displacement_applied_load, saved['applied_load'])
        assert not np.any(d._state.q_low[:owner._shell_count])
        expected = list(history_arrays(owner._beam_parts, d._beam_predecessors, 'beam_origin'))
        expected += [('shell_origin_plastic', np.array([h['plastic_strain'] for h in d._shell_predecessors])),
                     ('shell_origin_alpha', np.array([h['alpha'] for h in d._shell_predecessors]))]
        assert all(np.array_equal(saved[k], v) for k,v in expected)
        observer = ObservedAssembly(owner, out)
        base, locals_base, branches, shell_meta = observer.evaluate('base', d._state)
        repeat, locals_repeat, branches_repeat, repeat_meta = observer.evaluate('repeat', d._state)
        report['reproducibility'] = bool(np.array_equal(base[0], repeat[0]) and
            np.array_equal(base[1].indptr, repeat[1].indptr) and np.array_equal(base[1].indices, repeat[1].indices) and
            np.array_equal(base[1].data, repeat[1].data) and branches == branches_repeat)
        # The installed cold restore already executed the authoritative endpoint
        # replay gate; the explicit repeated material data must obey that same gate.
        replay = list(history_arrays(owner._beam_parts, base[5], 'beam'))
        replay += [('shell_plastic', np.array([h['plastic_strain'] for h in base[4]])),
                   ('shell_alpha', np.array([h['alpha'] for h in base[4]]))]
        replay_flags = {}
        for key, actual in replay:
            accepted = saved[key]
            if key.startswith('shell_'):
                scale = max(float(np.max(abs(accepted),initial=0.)),float(np.max(abs(actual),initial=0.)),np.finfo(float).tiny)
                replay_flags[key] = bool(np.max(abs(actual-accepted),initial=0.) <= 64*np.finfo(float).eps*scale)
            else:
                replay_flags[key] = bool(np.array_equal(actual, accepted))
        report['saved_authoritative_history_replay'] = replay_flags
        direction = frozen_direction(owner, base[0])
        groups = row_groups(owner, len(base[0]))
        count = owner._K.shape[0]
        weights = np.r_[owner._coordinate_scale, np.ones(owner._C.shape[0])]
        action = base[1]@direction
        np.savez(out/'direction-and-row-map.npz', direction=direction, weights=weights, action=action,
                 free=owner._free, fixed=owner._fixed, constraint_scale=owner._constraint_scale, **groups)
        errors = {name: [] for name in groups}
        local_errors = {str(i): [] for i in range(len(locals_base))}
        smooth = branch_smooth(branches, shell_meta)
        from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
        report['local_mapping_exact'] = []
        report['metadata_observability'] = dict(shell_convergence=bool(shell_meta) and all('converged' in row for row in shell_meta),
            shell_analytical_tangent_fallback=bool(shell_meta) and all('fallback_mask' in row for row in shell_meta),
            shell_bisection_fallback='canonical return exposes convergence/residual but no per-point bisection-use mask',
            native='canonical material bytes expose iterations/evaluations/line_trials/residual/derivative_kind and branch fields')
        for item in locals_base:
            if item['kind'] == 'native':
                mapped_tangent = spatial_jacobian(item['residual'],item['hessian']+item['hessian_low'])
            elif item['kind'] == 'point':
                mapped_tangent = item['spatial_jacobian']@item['coordinate_map']
            else:
                continue
            report['local_mapping_exact'].append(dict(kind=item['kind'],index=item['index'],
                exact=bool(np.array_equal(mapped_tangent,item['tangent']))))
        report.update(base_smooth=smooth, evaluations=[], local_derivatives=[])
        f05_groups = dict(shell=np.arange(owner._shell_count), beam=np.arange(owner._shell_count,count), free_primal=owner._free)
        for h in STEPS:
            plus = observer.evaluate('plus-'+str(h), d._advance(d._state,direction,h))
            minus = observer.evaluate('minus-'+str(h), d._advance(d._state,direction,-h))
            stable = same_smooth_branch(plus[2],branches) and same_smooth_branch(minus[2],branches) and branch_smooth(plus[2],plus[3]) and branch_smooth(minus[2],minus[3])
            smooth = smooth and stable
            observed = (plus[0][0]-minus[0][0])/(2*h)
            error = observed-action
            np.savez(out/('actions-'+str(h)+'.npz'),observed=observed,predicted=action,error=error)
            step_report = dict(h=h, branches_stable=stable,
                raw_metadata_identical=plus[2] == branches == minus[2],
                groups={name:metrics(error,action,rows,weights) for name,rows in groups.items()},
                f05={name:metrics(error,action,rows,weights) for name,rows in f05_groups.items()},
                constraint_scaled_max=float(np.max(abs(error[count:])/owner._constraint_scale,initial=0.)))
            report['evaluations'].append(step_report)
            for name, values in errors.items():
                values.append(step_report['groups'][name]['work_relative'])
            for i,(b,p,m) in enumerate(zip(locals_base,plus[1],minus[1])):
                local_direction = direction[b['slots']]
                predicted = b['tangent']@local_direction
                actual = (p['residual']-m['residual'])/(2*h)
                measured = metrics(actual-predicted,predicted,np.arange(len(predicted)),weights[b['slots']])
                local_errors[str(i)].append(measured['raw_relative'])
                report['local_derivatives'].append(dict(h=h,kind=b['kind'],index=b['index'],metrics=measured))
            write(out/'assessment.json',report)
        report['smooth_assessment_resolved'] = smooth
        report['common_adjacent_plateau'] = adjacent_plateau(errors)
        report['f05_all_steps'] = all(s['constraint_scaled_max'] <= 1e-9 and
            all(m['exercised'] and m['raw_relative'] <= 1e-5 and m['work_relative'] <= 1e-5 for m in s['f05'].values()) for s in report['evaluations'])
        report['local_adjacent_plateau'] = {str(i): adjacent_plateau({str(i):local_errors[str(i)]},
            1e-5 if b['kind'] == 'shell' else 1e-6) for i,b in enumerate(locals_base)}
        point_controls = []
        for b in locals_base:
            if b['kind'] != 'point':
                continue
            i, local_direction = b['index'], direction[b['slots']]
            mapped = b['coordinate_map']@local_direction
            primal_work = float(b['residual'][:12]@mapped[:12])
            constraint_work = float(b['multipliers']@(b['constraint_jacobian'][:,:12]@mapped[:12]))
            zero = owner._connections[i].evaluate_ports(**b['paired_input'], multipliers=np.zeros(6), charted_ports=(True,False))
            zero_tangent = zero.pose.spatial_jacobian@zero.coordinate_map
            curvature = b['tangent'][:12,:12]-zero_tangent[:12,:12]
            omitted_action = curvature@local_direction[:12]
            np.savez(out/('point-curvature-'+str(i)+'.npz'), curvature=curvature,omitted_action=omitted_action,
                     zero_tangent=zero_tangent,coordinate_map=b['coordinate_map'],mapped_direction=mapped)
            point_controls.append(dict(index=i,primal_work=primal_work,constraint_work=constraint_work,
                work_absolute_error=abs(primal_work-constraint_work),curvature_norm=float(np.linalg.norm(curvature)),
                omitted_action_norm=float(np.linalg.norm(omitted_action))))
        report['point_controls'] = point_controls
        # Deliberate refusal and pre-cancellation controls do not evaluate another
        # physical state or invoke publication. All flags are independently saved.
        try:
            sample(observer, d._state, direction, 2e-6)
        except ValueError:
            report['refusal_atomic'] = atomic_flags(owner,before)
        else:
            raise AssertionError('nonfrozen step accepted')
        token = CancellationToken(); token.cancel('saved-state tangent cancellation control')
        try:
            observer.evaluate('cancelled',d._state,cancellation_token=token)
        except SolveCancelled:
            report['cancellation_atomic'] = atomic_flags(owner,before)
        else:
            raise AssertionError('cancelled evaluation accepted')
        chart_owner = ss.create_analysis(definition=definition('coupled'))
        chart_owner.restore(REBASED.read_bytes())
        chart_before = capture(chart_owner)
        chart_d = chart_owner._finite_driver
        cm, ca = decode_snapshot(chart_owner, chart_before['checkpoint'])
        chart_history_flags = {k: bool(np.array_equal(saved[k],ca[k])) for k in saved
                               if k.startswith(('beam_', 'shell_')) and isinstance(saved[k],np.ndarray)
                               and k not in ('shell_anchors', 'shell_predecessor_anchors', 'shell_frames')}
        representation_flags = dict(histories=all(chart_history_flags.values()),
            seed=saved['force_seed'] == ca['force_seed'],
            applied=np.array_equal(saved['applied_load'],ca['applied_load']),
            shell_frames=bool(np.allclose(physical_frames(d._state.q[:owner._shell_count],d._state.shell_anchors),
                physical_frames(chart_d._state.q[:owner._shell_count],chart_d._state.shell_anchors),rtol=0.,atol=1e-11)),
            shell_translations=np.array_equal(d._state.q[:owner._shell_count].reshape(-1,6)[:,:3],
                chart_d._state.q[:owner._shell_count].reshape(-1,6)[:,:3]))
        for key in ('q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers'):
            representation_flags[key] = bool(np.array_equal(getattr(d._state,key),getattr(chart_d._state,key)))
        report['chart_preconditions'] = representation_flags
        representation_flags['native_q'] = bool(np.array_equal(d._state.q[owner._shell_count:],
                                                               chart_d._state.q[owner._shell_count:]))
        write(out/'assessment.json',report)
        checked(representation_flags)
        from anyfesolver._ge_beam3_pose_joint import _exp_terms
        # Same production shell selector as spatial_response; no raw K subtraction.
        exp_terms = _exp_terms
        if owner._distributing_material:
            from anyfesolver._ge_beam3_g3c_stable.joint import _exp_terms as exp_terms
        mapped_direction = direction.copy()
        columns = np.eye(owner._shell_count)
        for j in range(owner._shell_count//6):
            slots = slice(6*j+3,6*j+6)
            old_a=exp_terms(d._state.q[slots])[1];new_a=exp_terms(chart_d._state.q[slots])[1]
            columns[slots,slots]=np.linalg.solve(new_a,old_a)
            mapped_direction[slots]=columns[slots,slots]@direction[slots]
        rebased = ObservedAssembly(chart_owner,out).evaluate('rebased',chart_d._state)
        new_action = rebased[0][1]@mapped_direction
        def invariant(a,b):
            return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(a),np.linalg.norm(b)))
        report['chart_comparison'] = dict(residual_work_invariant=invariant(base[0]*weights,rebased[0][0]*weights),
            action_work_invariant=invariant(action*weights,new_action*weights),criterion=1e-11,
            mapped_columns=True,branches_same=same_smooth_branch(branches,rebased[2]),
            raw_metadata_identical=branches == rebased[2])
        np.savez(out/'chart-comparison.npz',column_map=columns,mapped_direction=mapped_direction,
                 old_action=action,new_action=new_action,weights=weights)
        report['chart_atomic'] = atomic_flags(chart_owner,chart_before)
        report['atomic'] = atomic_flags(owner,before)
        report['assemblies'] = observer.calls+1
        report['numerical_criteria'] = dict(reproducibility=report['reproducibility'],
            replay=all(replay_flags.values()), plateau=report['common_adjacent_plateau'], f05=report['f05_all_steps'],
            local=all(report['local_adjacent_plateau'].values()),
            local_mapping=all(row['exact'] for row in report['local_mapping_exact']),
            point=bool(point_controls) and all(p['work_absolute_error'] <= 1e-12 and p['curvature_norm'] > 1e-8 and p['omitted_action_norm'] > 0. for p in point_controls),
            chart=all(report['chart_comparison'][k] <= 1e-11 for k in ('residual_work_invariant','action_work_invariant')) and report['chart_comparison']['branches_same'])
        report['status'] = 'UNRESOLVED_BRANCH' if not smooth else 'SCOPED_PASS' if all(report['numerical_criteria'].values()) else 'UNRESOLVED_DISCREPANCY'
        write(out/'assessment.json',report)
        for name in ('atomic','chart_atomic','refusal_atomic','cancellation_atomic'):
            checked(report[name])
        # A smooth discrepancy is retained for same-parent attribution; this is
        # not authority to alter product code, h, tolerances or physics.
        assert report['status'] == 'SCOPED_PASS', report['status']
    except BaseException as exc:
        if report['status'] == 'INCOMPLETE':
            report['status'] = 'UNRESOLVED_REFUSAL' if isinstance(exc,ValueError) else 'ASSESSMENT_ERROR'
        report['failure'] = dict(type=type(exc).__name__,message=str(exc))
        write(out/'assessment.json',report)
        raise
    finally:
        if before is not None:
            flags=atomic_flags(owner,before)
            write(out/'terminal-atomic.json',flags)
            (out/'terminal.bin').write_bytes(owner.checkpoint())
            report['terminal_atomic']=flags
            write(out/'assessment.json',report)
        if chart_owner is not None:
            if 'chart_before' in locals():
                write(out/'chart-terminal-atomic.json',atomic_flags(chart_owner,chart_before))
            chart_owner.close()
        owner.close()
        if before is not None:
            checked(flags)
    return report


def sample(observer,state,direction,h):
    if h not in STEPS:
        raise ValueError('only the frozen h grid is authorized')
    return observer.evaluate('sample-'+str(h),observer.driver._advance(state,direction,h))


def bindings(*,preparation=False):
    from scripts import r8_f06_paired_loaded_x_dev as previous
    # Inherited authority/slot files authorized historical runs, not this fresh
    # assessment. Only this directory's authority/slot records govern execution.
    paths = {ROOT/rel for rel in previous.bindings()
             if 'AUTHORITY' not in Path(rel).name and Path(rel).name != 'SLOT_TRANSFER.json'}
    paths.update((Path(__file__).resolve(),ROOT/'tests/test_r8_f06_paired_active_tangent_dev.py',ACCEPTED,REBASED,
                  LOADED/'NEXT_TANGENT_READINESS.md',PACKAGE/'snapshot-bindings.json',ROOT/'tests/conftest.py'))
    paths.update(DOC.glob('*AUTHORITY*.json'))
    paths.update(DOC/name for name in ('SOURCE_AUTHORITY.json','INSTALLED_AUTHORITY.json','SLOT_TRANSFER.json'))
    paths.update(ROOT/'tests'/name for name in ('test_stiffened_shell_distributing_native_tangent.py',
        'test_stiffened_shell_f05_assembled_plastic.py','test_r8_f06_fibre_log_domain_dev.py','test_r8_f06_log_domain_dev.py'))
    result={p.relative_to(ROOT).as_posix():(None if preparation and not p.exists() else sha(p)) for p in paths}
    snapshot_names={}
    for rel in result:
        key=Path(rel).name.casefold()
        if key in snapshot_names:
            raise ValueError('supervisor flat snapshot basename collision: '+snapshot_names[key]+' and '+rel)
        snapshot_names[key]=rel
    return result


def artifact():
    from scripts import r8_f06_paired_loaded_x_dev as previous
    result=previous.artifact()
    assert result['wheel_sha256'] == WHEEL_SHA and result['files'] == 633
    return result


def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0])
    assert json.loads((out/'bindings.json').read_text()) == bindings()
    write(out/'artifact-before.json',artifact())
    from scripts.r8_f06_log_domain_check_dev import runtime
    selected=PACKAGE/'site'
    sys.path[:0]=[str(selected),str(ROOT/'tests')]
    os.environ['F06_ACTIVE_TANGENT_OUTPUT']=str(out)
    runtime(out,'runtime-before',selected)
    import pytest
    try:
        return int(pytest.main([str(ROOT/'tests/test_r8_f06_paired_active_tangent_dev.py'),'-q','-x',
            '-k','source' if stage == 'source-controls' else 'installed','-o','pythonpath=',
            '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
    finally:
        rt=runtime(out,'runtime-after',selected)
        frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
        for name,path in rt['imports'].items():
            assert rt['import_sha256'][name] == frozen['src/'+Path(path).relative_to(selected).as_posix()]
        write(out/'artifact-after.json',artifact())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('source-controls','installed-assessment'))
    parser.add_argument('--child',action='store_true');parser.add_argument('--out')
    args=parser.parse_args()
    if args.child:
        raise SystemExit(child(Path(args.out),args.stage))
    assert (DOC/'SLOT_TRANSFER.json').exists(), 'root review and numerical slot transfer required'
    authority_name='SOURCE_AUTHORITY.json' if args.stage == 'source-controls' else 'INSTALLED_AUTHORITY.json'
    authority=json.loads((DOC/authority_name).read_text())
    assert authority['budget_s'] == 180. and authority['reserve_s'] == 60.
    assert authority['process_cap_s'] <= (20. if args.stage == 'source-controls' else 60.)
    assert authority['accepted_sha256'] == INPUT_SHA and authority['rebased_sha256'] == REBASED_SHA
    assert authority['wheel_sha256'] == WHEEL_SHA
    if args.stage == 'installed-assessment':
        assert not list(DOC.glob('attempt-installed-assessment*')), 'sole saved-state assessment; no automatic retry'
        receipts=sorted(DOC.glob('attempt-source-controls*/receipt.json'))
        assert receipts and json.loads(receipts[-1].read_text())['status'] == 'PASS'
    from scripts import r8_f06_fibre_log_domain_dev as supervisor
    raise SystemExit(supervisor.run(args.stage,doc=DOC,entry=__file__,bind=bindings,authority_file=authority_name))
