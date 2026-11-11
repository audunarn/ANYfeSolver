"""Actual controlled history/objectivity gates, not general qualification."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_translation_control as control
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_native_generalized_program import model_identity
from anyfesolver.control import CancellationToken
from docs.reference_cases.ge_beam3_controlled_history_cases import (
    CASES, make, pose, transformed_positions, position_error, axial_reference)
from docs.reference_cases.ge_beam3_generalized_ellipsoid_oracle import response as independent
from test_ge_beam3_schur_line_program import save


def close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    error = float(np.linalg.norm(a-b))/max(1., float(np.linalg.norm(b)))
    assert error <= 1e-11
    return error


def resume(m, p, old, **kw):
    return control.solve(m, p, checkpoint=old.checkpoint, expected_sha256=sha256(old.checkpoint).hexdigest(), **kw)


def progress(row): print(row, flush=True)


def run_saved(m, p, root, name, **kw):
    result = control.solve(m, p, progress=progress, **kw)
    save(root/(name+'.json'), result.checkpoint)
    save(root/(name+'-status.json'), dict(status=result.status, failure=result.failure,
        completed_targets=result.completed_targets, production_qualified=False))
    return result


def rigid_checks(operator, mechanical, nodes, element_index, origin):
    s, translation = pose('general'); rotations = mechanical.cell_rotations[element_index]
    x, low = mechanical.positions[nodes], mechanical.position_low[nodes]
    q = mechanical.nodal_frames[nodes]; p = mechanical.resultants[element_index]
    before = canonical(origin)
    base = operator.evaluate(x, low, q, rotations, p, origin=origin)
    high, tail = transformed_positions(x, low, s, translation)
    other = operator.evaluate(high, tail, s@q, s@rotations, p, origin=origin)
    transform = np.eye(42); transform[:24, :24] = np.kron(np.eye(8), s)
    errors = dict(potential=close(other.potential, base.potential),
        residual=close(other.residual, transform@base.residual),
        hessian=close(other.hessian+other.hessian_low, transform@(base.hessian+base.hessian_low)@transform.T),
        kinematics=close(other.kinematics, base.kinematics))
    assert canonical(other.history) == canonical(base.history) and canonical(origin) == before
    recovery = operator.recover(rotations, p, origin=origin)
    changed = operator.recover(s@rotations, p, origin=origin)
    for a, b in zip(changed, recovery, strict=True):
        for key in ('strain', 'resultants'):
            close(a[key]+a[key+'_low'], b[key]+b[key+'_low'])
        close(a['current_frame'], s@b['current_frame'])
    return errors


@pytest.mark.parametrize('case', ('axial', 'curved'))
def test_unloaded_rigid_objectivity(case, tmp_path):
    m, p = make(case); c = control.Context(m, p); phys = c.physical
    errors = rigid_checks(phys.elements[0][1].operator, c.initial.mechanical, phys.nodes[0], 0, c.initial.histories[0])
    assert np.linalg.norm(c.initial.mechanical.resultants) == 0.
    save(tmp_path/'rigid.json', dict(case=case, errors=errors, arbitrary_common_rotation=True, production_qualified=False))


def test_scalar_return_map_reverses():
    _, p = make('axial'); rows = axial_reference(p.targets)
    assert rows[0]['increment'] == 0. and rows[2]['increment'] > 0.
    assert rows[3]['increment'] == 0. and rows[6]['increment'] > 0.
    assert rows[6]['plastic'] < rows[2]['plastic']
    assert all(b['accumulated'] >= a['accumulated'] for a, b in zip(rows, rows[1:]))


@pytest.mark.parametrize('case', CASES)
def test_actual_plastic_history(case, tmp_path):
    m, p = make(case); original_model = model_identity(m)
    whole = run_saved(m, p, tmp_path, 'whole')
    assert whole.status == 'completed', whole.failure
    fresh, _ = make(case)
    partial = run_saved(fresh, p, tmp_path, 'prefix', stop_after=3)
    assert partial.status == 'paused', partial.failure
    fresh, _ = make(case); restarted = resume(fresh, p, partial, progress=progress)
    save(tmp_path/'restarted.json', restarted.checkpoint)
    assert restarted.status == 'completed', restarted.failure
    assert restarted.checkpoint == whole.checkpoint
    context = control.Context(m, p); state, records = context.restore(whole.checkpoint, expected_sha256=sha256(whole.checkpoint).hexdigest())
    before = canonical(state); recovery = context.recover(state)
    assert context.checkpoint(records) == whole.checkpoint and canonical(state) == before
    save(tmp_path/'recovery.json', recovery)
    rows = [json.loads(raw) for raw in records]; peak = rows[2]
    # Fixed-origin tangent at an actual accepted plastic loading state.
    probe = control.Context(m, p); physical = probe.physical
    mechanical = physical.make(peak['mechanical'], decoded=True)
    origins = physical.histories(peak['origins'], decoded=True)
    direction = np.sin(np.arange(physical.count)+.23)*.01; direction[list(physical.fixed)] = 0.
    _, jacobian, _, _, _ = probe.assemble(mechanical, peak['parameter'], origins, p.targets[2])
    expected = jacobian@direction; tangent_errors = []
    for h in (2e-5, 1e-5, 5e-6):
        plus = physical.advance(mechanical, h*direction); minus = physical.advance(mechanical, -h*direction)
        a = probe.assemble(plus, peak['parameter'], origins, p.targets[2])[0]
        b = probe.assemble(minus, peak['parameter'], origins, p.targets[2])[0]
        tangent_errors.append(float(np.linalg.norm((a-b)/(2*h)-expected))/max(1., float(np.linalg.norm(expected))))
    assert max(tangent_errors) <= 1e-7
    rigid = [rigid_checks(e.operator, mechanical, physical.nodes[i], i, origins[i])
             for i, (_, e) in enumerate(physical.elements)]
    scalar = axial_reference(p.targets) if case == 'axial' else None
    material_checks = []; plastic_counts = []; accumulated = []; plastic_signs = []
    predecessor = json.loads(whole.checkpoint)['initial']
    for cursor, row in enumerate(rows):
        print(dict(stage='independent-history-check', case=case, target=cursor+1), flush=True)
        assert canonical(row['origins']) == canonical(predecessor['histories'])
        assert row['previous_sha256'] == predecessor['record_sha256']
        # This is a read-only independent record inspection after the complete
        # controlled replay above, not a new Context or a committed trial.
        s = physical.make(row['mechanical'], decoded=True)
        old = physical.histories(row['origins'], decoded=True)
        current = physical.histories(row['histories'], decoded=True)
        count = 0; total = 0.; signed = []; station_errors = []
        force = row['parameter']*np.array(p.nodal_forces.rows[0][1:])
        reaction = np.array(row['residual'][:6])
        close(reaction[:3], -force)
        arm = s.positions[-1]-s.positions[0]+s.position_low[-1]-s.position_low[0]
        close(reaction[3:], -np.cross(arm, force))
        close(row['work'][-1], row['parameter']*np.linalg.norm(p.nodal_forces.rows[0][1:])*p.targets[cursor])
        for i, (_, element) in enumerate(physical.elements):
            nodes = physical.nodes[i]
            response = element.operator.evaluate(s.positions[nodes], s.position_low[nodes], s.nodal_frames[nodes],
                s.cell_rotations[i], s.resultants[i], origin=old[i])
            assert canonical(response.history) == canonical(current[i])
            for fields, prior, accepted in zip(json.loads(response.material)['stations'], old[i].stations, current[i].stations, strict=True):
                strain = np.sum(fields['strain'], axis=0); stress = np.sum(fields['resultants'], axis=0)
                law = element.section
                independent_result = independent(dict(elastic=law.elastic, metric=law.metric, strain=strain,
                    plastic=prior.plastic, accumulated=prior.accumulated, yield_force=law.yield_force, hardening=law.hardening))
                errors = dict(stress=close(stress, independent_result['stress']),
                    plastic=close([sum(v) for v in accepted.plastic], independent_result['plastic']),
                    accumulated=close(sum(accepted.accumulated), independent_result['accumulated']),
                    potential=close(sum(fields['incremental_potential'][k][0] for k in (0,1)), independent_result['potential']),
                    dissipation=close(sum(fields['dissipation'][k][0] for k in (0,1)), independent_result['dissipation']))
                assert sum(accepted.accumulated) >= sum(prior.accumulated)
                assert independent_result['dissipation'] >= 0.
                if fields['branch'] == 'PLASTIC': count += 1
                total += sum(accepted.accumulated); signed.append([sum(v) for v in accepted.plastic]); station_errors.append(errors)
                if scalar is not None:
                    exact = scalar[cursor]
                    close(stress, [exact['stress'], 0., 0., 0., 0., 0.])
                    close(strain, [exact['strain'], 0., 0., 0., 0., 0.])
                    close(sum(accepted.accumulated), exact['accumulated'])
                    close([sum(v) for v in accepted.plastic], [exact['plastic'], 0., 0., 0., 0., 0.])
                    close(row['parameter'], exact['stress'])
        plastic_counts.append(count); accumulated.append(total); plastic_signs.append(np.array(signed)); material_checks.append(station_errors)
        predecessor = row
    assert plastic_counts[2] > 0 and plastic_counts[6] > 0
    assert plastic_counts[3] == 0
    assert all(b >= a for a, b in zip(accumulated, accumulated[1:]))
    assert np.sum((plastic_signs[6]-plastic_signs[5])*plastic_signs[2]) < 0.
    assert model_identity(m) == original_model and canonical(state) == before
    save(tmp_path/'checks.json', dict(case=case, plastic_station_counts=plastic_counts, accumulated=accumulated,
        independent_material_checks=material_checks, tangent_errors=tangent_errors, rigid_errors=rigid,
        restart_identical=True, actual_controlled_history=True, production_qualified=False))


@pytest.mark.parametrize('pose_name', ('general', 'large-dyadic'))
def test_controlled_history_coordinate_covariance(pose_name, tmp_path):
    base, p = make('curved'); model, transformed = make('curved', pose_name); s, translation = pose(pose_name)
    original = run_saved(base, p, tmp_path, 'base'); other = run_saved(model, transformed, tmp_path, 'transformed')
    assert original.status == other.status == 'completed', (original.failure, other.failure)
    arows = json.loads(original.checkpoint)['records']; brows = json.loads(other.checkpoint)['records']
    ca, cb = control.Context(base, p), control.Context(model, transformed)
    errors = []
    for ar, br in zip(arows, brows, strict=True):
        a = ca.physical.make(ar['mechanical'], decoded=True); b = cb.physical.make(br['mechanical'], decoded=True)
        error = dict(positions=position_error(b, a, s, translation), parameter=close(br['parameter'], ar['parameter']),
            nodal_frames=close(b.nodal_frames, s@a.nodal_frames), cell_frames=close(b.cell_rotations, s@a.cell_rotations@s.T),
            work=close(br['work'], ar['work']))
        assert error['positions'] <= 1e-11
        expected = a.resultants.copy(); expected[:, :6] = (a.resultants[:, :6].reshape(-1, 2, 3)@s.T).reshape(-1, 6)
        error['resultants'] = close(b.resultants, expected)
        reaction = np.array(ar['residual'][:6]).reshape(2, 3)@s.T
        error['reaction'] = close(br['residual'][:6], reaction.ravel())
        for ah, bh in zip(ar['histories'], br['histories'], strict=True):
            for av, bv in zip(ah['stations'], bh['stations'], strict=True):
                close(np.sum(bv['plastic'], axis=1), np.sum(av['plastic'], axis=1))
                close(sum(bv['accumulated']), sum(av['accumulated']))
        oa = ca.physical.histories(ar['origins'], decoded=True); ob = cb.physical.histories(br['origins'], decoded=True)
        for i, ((_, ea), (_, eb)) in enumerate(zip(ca.physical.elements, cb.physical.elements, strict=True)):
            ra = ea.operator.recover(a.cell_rotations[i], a.resultants[i], origin=oa[i])
            rb = eb.operator.recover(b.cell_rotations[i], b.resultants[i], origin=ob[i])
            for av, bv in zip(ra, rb, strict=True):
                close(bv['current_frame'], s@av['current_frame'])
                for key in ('strain', 'resultants'): close(bv[key]+bv[key+'_low'], av[key]+av[key+'_low'])
        errors.append(error)
    save(tmp_path/'covariance.json', dict(pose=pose_name, rotation=s, translation=translation, errors=errors,
        complete_controlled_plastic_path=True, production_qualified=False))


@pytest.fixture(scope='module')
def plastic_prefix():
    m, p = make('axial'); result = control.solve(m, p, stop_after=3)
    assert result.status == 'paused', result.failure
    assert any(sum(h.accumulated) > 0. for cell in result.state.histories for h in cell.stations)
    return p, result


@pytest.mark.parametrize('stage', ('before_assembly', 'before_factorization', 'before_trial', 'before_commit', 'committed'))
def test_plastic_cancellation(plastic_prefix, stage, tmp_path):
    p, original = plastic_prefix; m, _ = make('axial'); token = CancellationToken(); before = canonical(original.state)
    def observe(row):
        if row['stage'] == 'retained-translation.'+stage: token.cancel()
    result = resume(m, p, original, cancellation_token=token, progress=observe)
    assert result.status == 'cancelled'
    if stage != 'committed':
        assert result.checkpoint == original.checkpoint and result.completed_targets == 3
    else:
        expected = control.solve(make('axial')[0], p, stop_after=4)
        assert expected.status == 'paused' and result.checkpoint == expected.checkpoint
    assert canonical(original.state) == before
    save(tmp_path/'cancel.json', dict(stage=stage, completed_targets=result.completed_targets,
        checkpoint_sha256=sha256(result.checkpoint).hexdigest(), preserved=True))


@pytest.mark.parametrize('field', ('origin', 'history', 'accumulated', 'material', 'target', 'parameter'))
def test_resealed_plastic_mutation(plastic_prefix, field):
    p, original = plastic_prefix; value = json.loads(original.checkpoint); row = value['records'][-1]
    if field == 'origin': row['origins'][0]['stations'][0]['plastic'][0][0] += .001
    elif field == 'history': row['histories'][0]['stations'][0]['plastic'][0][0] += .001
    elif field == 'accumulated': row['histories'][0]['stations'][0]['accumulated'][0] += .001
    elif field == 'material': row['material_sha256'] = '0'*64
    elif field == 'target': row['displacement_target'] += .001
    else: row['parameter'] += .001
    previous = value['initial']['record_sha256']
    for row in value['records']:
        row['previous_sha256'] = previous; row['record_sha256'] = sha({k:v for k,v in row.items() if k != 'record_sha256'})
        previous = row['record_sha256']
    value['checkpoint_sha256'] = sha({k:v for k,v in value.items() if k != 'checkpoint_sha256'}); raw = canonical(value)
    with pytest.raises(ValueError):
        control.solve(make('axial')[0], p, checkpoint=raw, expected_sha256=sha256(raw).hexdigest())
