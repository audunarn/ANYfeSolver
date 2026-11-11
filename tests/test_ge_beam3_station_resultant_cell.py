"""Small research cell checks; same-author audit, not qualification/review."""
from copy import deepcopy
from decimal import Decimal as D, localcontext
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5.section import SectionHistory
from docs.reference_cases.ge_beam3_station_resultant_cell import StationResultantCell, pair
from docs.reference_cases.ge_beam3_decimal_cell_force_audit import audit
from test_ge_beam3_plastic_cell_conjugate_probe import build


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def data_for(contrast, history=False):
    origins = tuple(SectionHistory(.01*(-1 if i % 2 else 1), .02) for i in range(16)) if history else None
    probe = build(contrast, origins); source = probe.source.section
    stations = []
    for cell in (0, 1):
        for index, t, _, w, v, _, _ in probe.source._stations[cell]:
            origin = probe.origins[cell*probe.source.order+index]
            stations.append(dict(cell=cell, t=float(t), weight=float(w), v=v.tolist(),
                origin=[origin.plastic_coordinate, origin.accumulated]))
    return dict(elastic=source._elastic.tolist(), direction=source._direction.tolist(),
        yield_force=source._yield, hardening=source._hardening, stations=stations)


def unpack(value):
    return [D.from_float(h)+D.from_float(l) for h, l in zip(*value)]


def mv(a, b): return [sum((x*y for x, y in zip(row, b)), D(0)) for row in a]
def dot(a, b): return sum((x*y for x, y in zip(a, b)), D(0))


def identities(data, p, response):
    """Original full section equations and cell constraints on paired output."""
    with localcontext() as ctx:
        ctx.prec = 90
        c = [[D.from_float(x) for x in row] for row in data['elastic']]
        a = [D.from_float(x) for x in data['direction']]
        p = [D.from_float(x) for x in p]; gradient = unpack(response['gradient'])
        force = [D(0)]*6; curvature = [D(0)]*12; primal = D(0)
        errors = dict(section=D(0), compatibility=D(0), moment=D(0), plastic=D(0))
        for row, station in zip(data['stations'], response['stations']):
            cell = row['cell']; w = D.from_float(row['weight']); t = D.from_float(row['t'])
            v = [[D.from_float(x) for x in r] for r in row['v']]
            stress = unpack(station['resultants']); strain = unpack(station['strain'])
            elastic = unpack(station['elastic_strain'])
            z = unpack(station['plastic_coordinate'])[0]; accumulated = unpack(station['accumulated'])[0]
            old_z, old_p = [D.from_float(h)+D.from_float(l) for h, l in zip(row['origin'], row.get('origin_low', [0., 0.]))]
            gamma = mv(v, gradient[3*cell:3*cell+3])
            moment = [(1-t)*p[6+6*cell+i]+t*p[9+6*cell+i] for i in range(3)]
            errors['section'] = max(errors['section'], *(abs(x-y) for x, y in zip(mv(c, elastic), stress)),
                *(abs(x+y*z-e) for x, y, e in zip(elastic, a, strain)))
            errors['compatibility'] = max(errors['compatibility'], *(abs(x-y) for x, y in zip(gamma, strain[:3])))
            errors['moment'] = max(errors['moment'], *(abs(x-y) for x, y in zip(moment, stress[3:])))
            drive = dot(a, stress); h = D.from_float(data['hardening']); y = D.from_float(data['yield_force'])
            delta = max(abs(drive)-y-h*old_p, D(0))/h*(1 if drive >= 0 else -1)
            errors['plastic'] = max(errors['plastic'], abs(z-old_z-delta), abs(accumulated-old_p-abs(delta)))
            assert accumulated >= abs(z)-D('1e-30')
            for i in range(3):
                force[3*cell+i] += w*sum((v[j][i]*stress[j] for j in range(3)), D(0))
                curvature[6*cell+i] += w*(1-t)*strain[3+i]
                curvature[6*cell+3+i] += w*t*strain[3+i]
            primal += w*(dot(stress, elastic)/2+h*abs(delta)*(2*old_p+abs(delta))/2+y*abs(delta))
        errors['force'] = sum(((x-y)**2 for x, y in zip(force, p[:6])), D(0)).sqrt()
        errors['curvature'] = max(abs(x-y) for x, y in zip(curvature, gradient[6:]))
        errors['work'] = abs(dot(p, gradient)-primal-unpack(response['potential'])[0])
        return {key: str(value) for key, value in errors.items()}


@pytest.mark.parametrize('contrast', [1., 10000., 1000000000000.])
@pytest.mark.parametrize('history', [False, True])
def test_constrained_stress_cell_against_source_equations(contrast, history, tmp_path):
    data = data_for(contrast, history); original = canonical(data)
    p = np.linspace(-.3, .5, 18).tolist(); cell = StationResultantCell(data)
    response = cell.response(p); errors = identities(data, p, response)
    independent_construction = dict(data, retained=p, old_gradient=response['gradient'][0],
        increment_pattern=response['increments'][0])
    comparison = audit(independent_construction, digits=90)
    with localcontext() as ctx:
        ctx.prec = 90
        gradient_error = max(abs(x-D(y)) for x, y in zip(unpack(response['gradient']), comparison['gradient']))
        increment_error = max(abs(x-D(y)) for x, y in zip(unpack(response['increments']), comparison['increments']))
    record = dict(contrast=contrast, history=history, errors=errors, gradient_error=str(gradient_error),
        increment_error=str(increment_error), response=response, production_qualified=False)
    (tmp_path/'input.json').write_bytes(original)
    (tmp_path/'result.json').write_bytes(canonical(record))
    (tmp_path/'source-audit.json').write_bytes(canonical(comparison))
    assert canonical(data) == original
    assert canonical(cell.response(p)) == canonical(response)
    assert max(D(x) for x in errors.values()) <= D('1e-11')
    assert gradient_error < D('1e-28') and increment_error < D('1e-28')
    assert D(response['kkt_residual']) <= D('1e-35')


@pytest.mark.parametrize('contrast', [1., 1000000000000.])
def test_cell_first_second_variations_and_fixed_origin(contrast):
    data = data_for(contrast, True); cell = StationResultantCell(data)
    p = np.linspace(-.3, .5, 18); d = np.cos(np.arange(18))*.1; eps = 1e-5
    centre = cell.response(p.tolist()); plus = cell.response((p+eps*d).tolist()); minus = cell.response((p-eps*d).tolist())
    with localcontext() as ctx:
        ctx.prec = 90
        gradient = unpack(centre['gradient']); hessian = [unpack(row) for row in centre['hessian']]
        delta = D.from_float(2*eps); direction = list(map(D.from_float, d.tolist()))
        expected = mv(hessian, direction)
        actual = [(x-y)/delta for x, y in zip(unpack(plus['gradient']), unpack(minus['gradient']))]
        assert max(abs(x-y) for x, y in zip(actual, expected)) < D('1e-7')
        first = (unpack(plus['potential'])[0]-unpack(minus['potential'])[0])/delta
        assert abs(first-dot(gradient, direction)) < D('1e-7')
        assert max(abs(hessian[i][j]-hessian[j][i]) for i in range(18) for j in range(18)) < D('1e-11')
    # Caller mutation cannot replace the already captured section or history.
    data['elastic'][0][0] *= 2; data['stations'][0]['origin'] = [99., 99.]
    assert canonical(cell.response(p.tolist())) == canonical(centre)


@pytest.mark.parametrize('p', [[0.]*17, [float('nan')]*18, [float('inf')]*18, [0]*18])
def test_bad_resultants_fail_closed(p):
    with pytest.raises(ValueError): StationResultantCell(data_for(1.)).response(p)


@pytest.mark.parametrize('value', [D('Infinity'), D('NaN'), D('1e400'), D('1e-400')])
def test_pair_cannot_fabricate_finite_or_zero_response(value):
    with pytest.raises(ValueError): pair([value])


def with_history(data, history):
    copied = deepcopy(data)
    for row, (zh, zl, ph, pl) in zip(copied['stations'], history):
        row['origin'] = [zh, ph]; row['origin_low'] = [zl, pl]
    return copied


@pytest.mark.parametrize('contrast', [1., 1000000000000.])
def test_paired_history_loading_unloading_reversal_and_replay(contrast, tmp_path):
    data = data_for(contrast); cell = StationResultantCell(data)
    p = np.linspace(-.3, .5, 18); history = [[0., 0., 0., 0.] for _ in data['stations']]
    records = []; previous_accumulated = [D(0)]*len(history)
    for factor in (0., 1., 2., 1., 0., -1., -2., 0.):
        origin_bytes = canonical(history); input_data = with_history(data, history)
        retained = (factor*p).tolist(); response = cell.response(retained, origins=history)
        errors = identities(input_data, retained, response)
        assert max(D(x) for x in errors.values()) <= D('1e-11')
        comparison = audit(dict(input_data, retained=retained, old_gradient=response['gradient'][0],
            increment_pattern=response['increments'][0]), digits=90)
        with localcontext() as ctx:
            ctx.prec = 90
            assert max(abs(x-D(y)) for x, y in zip(unpack(response['gradient']), comparison['gradient'])) < D('1e-28')
            current = [D.from_float(row[2])+D.from_float(row[3]) for row in response['history']]
            assert all(now >= old for now, old in zip(current, previous_accumulated))
            previous_accumulated = current
        # A trial does not publish history. A second trial or replay starts from
        # the identical accepted origin; response history is only a proposal.
        assert canonical(history) == origin_bytes
        assert canonical(cell.response(retained, origins=json.loads(origin_bytes))) == canonical(response)
        assert canonical(StationResultantCell(input_data).response(retained)) == canonical(response)
        records.append(dict(factor=factor, errors=errors, response=response))
        history = response['history']
    (tmp_path/'history-cycle.json').write_bytes(canonical(records))
    assert any(row[1] != 0. or row[3] != 0. for row in history)


@pytest.mark.parametrize('mutation', ['count', 'shape', 'nonfinite', 'inadmissible', 'unnormalized'])
def test_invalid_history_does_not_change_captured_origin(mutation):
    cell = StationResultantCell(data_for(1.)); p = np.linspace(-.3, .5, 18).tolist()
    before = canonical(cell.response(p)); histories = [[0., 0., 0., 0.] for _ in range(16)]
    if mutation == 'count': histories.pop()
    elif mutation == 'shape': histories[0].pop()
    elif mutation == 'nonfinite': histories[0][1] = float('nan')
    elif mutation == 'inadmissible': histories[0] = [1., 0., 0., 0.]
    elif mutation == 'unnormalized': histories[0] = [.5, .5, 1., 0.]
    with pytest.raises(ValueError): cell.response(p, origins=histories)
    assert canonical(cell.response(p)) == before


@pytest.mark.parametrize('mutation', ['elastic_shape', 'nonsymmetric', 'indefinite', 'yield',
    'hardening', 'weight', 'cell', 'missing_cell', 'map', 'station_count'])
def test_invalid_compiler_inputs_fail_closed(mutation):
    data = data_for(1.)
    if mutation == 'elastic_shape': data['elastic'].pop()
    elif mutation == 'nonsymmetric': data['elastic'][0][1] += 1.
    elif mutation == 'indefinite': data['elastic'][0][0] = -1.
    elif mutation == 'yield': data['yield_force'] = 0.
    elif mutation == 'hardening': data['hardening'] = -1.
    elif mutation == 'weight': data['stations'][0]['weight'] = -1.
    elif mutation == 'cell': data['stations'][0]['cell'] = True
    elif mutation == 'missing_cell': data['stations'] = data['stations'][:8]
    elif mutation == 'map': data['stations'][0]['v'].pop()
    elif mutation == 'station_count': data['stations'] *= 4
    with pytest.raises(ValueError): StationResultantCell(data)


def test_compiler_and_response_deadlines_fail_without_mutating_history(monkeypatch):
    import docs.reference_cases.ge_beam3_station_resultant_cell as implementation
    data = data_for(1.); cell = StationResultantCell(data); p = [0.]*18
    saved = canonical(cell.response(p)); clock = iter([0., 61.])
    with monkeypatch.context() as patch:
        patch.setattr(implementation, 'monotonic', lambda: next(clock))
        with pytest.raises(ValueError, match='compiler deadline'): StationResultantCell(data)
    clock = iter([0., 61.])
    with monkeypatch.context() as patch:
        patch.setattr(implementation, 'monotonic', lambda: next(clock))
        with pytest.raises(ValueError, match='conjugate deadline'): cell.response(p)
    assert canonical(cell.response(p)) == saved
