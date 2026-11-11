"""Coupled cell checks against uncondensed station work and fibre equations."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal as D, localcontext

import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_cell import FibreCellConjugate, CellHistory
from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection, FlowCurve, canonical
from test_ge_beam3_fibre_section import section, scalar_law
from test_ge_beam3_curved_contrast_probe import make_curved


def stations(order=4):
    reference = next(iter(make_curved(100.)[0].mesh.elements.values())).core.reference
    points, weights = np.polynomial.legendre.leggauss(order)
    rows = []
    for cell in (0, 1):
        for point, weight in zip(points, weights):
            t = float((point+1)/2); xi = cell-1+t; jacobian = reference.jacobian(xi)
            rows.append(dict(cell=cell, t=t, weight=float(weight*jacobian/2),
                             v=(reference.frame(xi).T/jacobian).tolist()))
    return rows


def model(contrast=1., tabular=False, order=4):
    law = section(contrast)
    if tabular:
        # Explicit positive tail keeps the strict conjugate coercive for this
        # test. The native flat-tail law is unchanged, not silently stabilized.
        curve = FlowCurve((0., .125, .5), (.25, .5, .75), .5)
        law = PhysicalFibreSection([replace(f, curve=curve) for f in law.fibres], law.background_factor)
    data = stations(order)
    return law, data, FibreCellConjugate(law, data)


def unpack(pair):
    return [D.from_float(h)+D.from_float(l) for h, l in zip(*pair)]


def exact_pair(value):
    return D.from_float(value[0])+D.from_float(value[1])


def dot(a, b): return sum((x*y for x, y in zip(a, b)), D(0))
def mv(a, b): return [dot(row, b) for row in a]


def identities(law, data, p, response):
    """Original full physical section + half-cell constraints, no cell compiler."""
    with localcontext() as context:
        context.prec = 90
        p = list(map(D.from_float, p)); g = unpack(response['gradient'])
        force = [D(0)]*6; curvature = [D(0)]*12; primal = D(0)
        errors = dict(force=D(0), moment=D(0), compatibility=D(0), curvature=D(0),
                      section=D(0), fibre=D(0), history=D(0), work=D(0))
        for source, fields, origin, proposed in zip(data, response['stations'], response['origin'].stations, response['history'].stations):
            cell = source['cell']; w = D.from_float(source['weight']); t = D.from_float(source['t'])
            v = [list(map(D.from_float, row)) for row in source['v']]
            strain = unpack(fields['strain']); stress = unpack(fields['resultants'])
            gamma = mv(v, g[3*cell:3*cell+3])
            moment = [(1-t)*p[6+6*cell+i]+t*p[9+6*cell+i] for i in range(3)]
            errors['moment'] = max(errors['moment'], *(abs(a-b) for a, b in zip(moment, stress[3:])))
            errors['compatibility'] = max(errors['compatibility'], *(abs(a-b) for a, b in zip(gamma, strain[:3])))
            # Different section entry, fed the full represented station strain.
            native = law.response(fields['strain'][0], origin, strain_low=fields['strain'][1])
            native_stress = [D.from_float(h)+D.from_float(l) for h, l in zip(native.resultants, native.resultants_low)]
            errors['section'] = max(errors['section'], *(abs(a-b) for a, b in zip(native_stress, stress)))
            primal += w*exact_pair(native.incremental_potential)
            for fibre, recovery, old, new, native_fibre in zip(law.fibres, fields['fibres'], origin.rows, proposed.rows, native.fibres):
                ef = strain[0]+D.from_float(fibre.z)*strain[4]-D.from_float(fibre.y)*strain[5]
                ee = exact_pair(recovery.elastic_strain); sigma = exact_pair(recovery.stress)
                z = exact_pair(new[:2]); accumulated = exact_pair(new[2:]); z0 = exact_pair(old[:2]); p0 = exact_pair(old[2:])
                delta = exact_pair(recovery.plastic_increment)
                errors['fibre'] = max(errors['fibre'], abs(ef-ee-z), abs(D.from_float(fibre.young)*ee-sigma),
                    abs(exact_pair(native_fibre.stress)-sigma))
                errors['history'] = max(errors['history'], abs(z-z0-delta), abs(accumulated-p0-abs(delta)))
            for i in range(3):
                force[3*cell+i] += w*sum((v[j][i]*stress[j] for j in range(3)), D(0))
                curvature[6*cell+i] += w*(1-t)*strain[3+i]
                curvature[6*cell+3+i] += w*t*strain[3+i]
        errors['force'] = max(abs(a-b) for a, b in zip(force, p[:6]))
        errors['curvature'] = max(abs(a-b) for a, b in zip(curvature, g[6:]))
        errors['work'] = abs(dot(p, g)-primal-unpack(response['potential'])[0])
        return {key: str(value) for key, value in errors.items()}


@pytest.mark.parametrize('contrast', [1., 1.e4, 1.e12])
@pytest.mark.parametrize('tabular', [False, True])
def test_curved_cell_force_moment_work_and_fibre_recovery(contrast, tabular, tmp_path):
    law, data, cell = model(contrast, tabular)
    p = np.linspace(-.3, .5, 18).tolist()
    response = cell.response(p); errors = identities(law, data, p, response)
    # Bind even a failed numerical attempt before asserting its scientific checks.
    with (tmp_path/'cell.json').open('xb') as stream:
        stream.write(canonical(dict(response=response, errors=errors)))
    assert max(D(v) for v in errors.values()) <= D('1e-11')
    assert D(response['residual']) <= D('1e-28')
    assert canonical(response) == canonical(cell.response(p))
    assert any(exact_pair(f.plastic_increment) != 0 for row in response['stations'] for f in row['fibres'])


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_coupled_first_second_variations_and_nonlocal_station_coupling(contrast):
    _, _, cell = model(contrast, True)
    p = np.linspace(-.3, .5, 18); direction = np.cos(np.arange(18))*.1; step = 1e-5
    centre = cell.response(p.tolist()); plus = cell.response((p+step*direction).tolist()); minus = cell.response((p-step*direction).tolist())
    assert centre['derivative_kind'] == 'CLASSICAL_SMOOTH_BRANCH'
    with localcontext() as context:
        context.prec = 90
        gradient = unpack(centre['gradient']); hessian = [unpack(row) for row in centre['hessian']]
        d = list(map(D.from_float, direction.tolist())); h = D.from_float(2*step)
        first = (unpack(plus['potential'])[0]-unpack(minus['potential'])[0])/h
        second = [(a-b)/h for a, b in zip(unpack(plus['gradient']), unpack(minus['gradient']))]
        assert abs(first-dot(gradient, d)) <= D('1e-7')
        assert max(abs(a-b) for a, b in zip(second, mv(hessian, d))) <= D('1e-7')
        assert max(abs(hessian[i][j]-hessian[j][i]) for i in range(18) for j in range(18)) <= D('1e-28')
        assert any(hessian[i][j] != 0 for i in range(3) for j in range(6, 12))


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_loading_unloading_reversal_replay_and_immutable_origin(contrast, tmp_path):
    law, data, cell = model(contrast, True, order=2)
    origin = cell.virgin(); pattern = np.linspace(-.3, .5, 18); records = []
    for factor in (.25, 1., .5, 0., -1., 0.):
        before = canonical(origin); p = (factor*pattern).tolist()
        result = cell.response(p, origin); errors = identities(law, data, p, result)
        assert canonical(origin) == before
        assert max(D(v) for v in errors.values()) <= D('1e-11')
        assert canonical(result) == canonical(cell.response(p, origin))
        records.append(dict(response=result, errors=errors)); origin = result['history']
    with (tmp_path/'cell-history.json').open('xb') as stream: stream.write(canonical(records))


@pytest.mark.parametrize('contrast', [1., 1.e12])
@pytest.mark.parametrize('general_rotation', [False, True])
def test_proper_spatial_reexpression_preserves_station_material_fields(contrast, general_rotation, tmp_path):
    law, data, cell = model(contrast)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    if general_rotation:
        vector = np.array([.4, -.3, .2]); angle = np.linalg.norm(vector); axis = vector/angle
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        rotation = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    p = np.linspace(-.3, .5, 18); a = cell.response(p.tolist())
    other_data = deepcopy(data)
    for row in other_data: row['v'] = (np.array(row['v'])@rotation.T).tolist()
    other = FibreCellConjugate(law, other_data); rotated_p = p.copy()
    rotated_p[:6] = (p[:6].reshape(2, 3)@rotation.T).ravel()
    b = other.response(rotated_p.tolist())
    ga = np.array(a['gradient'][0]); ga[:6] = (ga[:6].reshape(2, 3)@rotation.T).ravel()
    np.testing.assert_allclose(ga, b['gradient'][0], rtol=1e-11, atol=1e-11)
    for x, y in zip(a['stations'], b['stations']):
        np.testing.assert_allclose(x['strain'][0], y['strain'][0], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(x['resultants'][0], y['resultants'][0], rtol=1e-11, atol=1e-11)
    with (tmp_path/'cell-covariance.json').open('xb') as stream: stream.write(canonical(dict(base=a, rotated=b)))


def test_perfect_plastic_unbounded_stress_control_fails_without_regularization():
    law = scalar_law(FlowCurve.linear(1., 0.))
    data = [dict(cell=c, t=t, weight=.5, v=np.eye(3).tolist()) for c in (0, 1) for t in (.25, .75)]
    cell = FibreCellConjugate(law, data); origin = cell.virgin(); p = [0.]*18; p[0] = 2.
    with pytest.raises(ValueError, match='nonpositive current fibre cell tangent'): cell.response(p, origin)
    assert origin == cell.virgin()


def test_cancellation_and_timeout_never_advance_history(monkeypatch):
    _, _, cell = model(order=2); origin = cell.virgin(); calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 15: raise RuntimeError('cancelled coupled fixture')
    with pytest.raises(RuntimeError, match='cancelled coupled fixture'):
        cell.response(np.linspace(-.3, .5, 18).tolist(), origin, check=cancel)
    assert origin == cell.virgin()
    import anyfesolver._ge_beam3_fibre_cell as module
    ticks = 0
    def clock():
        nonlocal ticks
        ticks += 31
        return ticks
    monkeypatch.setattr(module, 'monotonic', clock)
    with pytest.raises(TimeoutError): cell.response([0.]*18, origin)
    assert origin == cell.virgin()


@pytest.mark.parametrize('mutation', ['identity', 'count', 'mutable', 'foreign_section'])
def test_foreign_or_mutated_cell_history_fails(mutation):
    law, _, cell = model(order=2); origin = cell.virgin()
    if mutation == 'identity': origin = replace(origin, cell_identity='0'*64)
    elif mutation == 'count': origin = replace(origin, stations=origin.stations[:-1])
    elif mutation == 'mutable': origin = replace(origin, stations=list(origin.stations))
    else: origin = replace(origin, stations=(section(2.).virgin(),)+origin.stations[1:])
    with pytest.raises(ValueError): cell.response([0.]*18, origin)


@pytest.mark.parametrize('mutation', ['unknown', 'cell', 'weight', 'station_map', 'missing_cell'])
def test_station_schema_and_geometry_mutations_fail(mutation):
    law, data, cell = model(order=2)
    if mutation == 'unknown': data[0]['extra'] = 1
    elif mutation == 'cell': data[0]['cell'] = True
    elif mutation == 'weight': data[0]['weight'] = 0.
    elif mutation == 'station_map': data[0]['v'][0][0] = float('nan')
    else:
        for row in data: row['cell'] = 0
    with pytest.raises(ValueError): FibreCellConjugate(law, data)
    assert cell.response([0.]*18)['iterations'] == 0


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_piecewise_line_minimum_satisfies_original_energy_directional_stationarity(contrast):
    _, _, cell = model(contrast, True, order=2)
    with localcontext() as context:
        context.prec = 80
        origins = cell._origins(cell.virgin())
        p = [D.from_float(v) for v in np.linspace(-.3, .5, 18).tolist()]
        load = mv(cell.load_map, p); x = [D(0)]*cell.size
        objective, _, residual, _, _ = cell._evaluate(x, load, origins, lambda: None)
        direction = [-v for v in residual]
        alpha, count = cell._line_minimum(x, direction, residual, load, origins, lambda: None)
        point = [a+alpha*b for a, b in zip(x, direction)]
        value, _, derivative, _, _ = cell._evaluate(point, load, origins, lambda: None)
        assert 0 < alpha <= 1 and 1 <= count <= 4096 and value < objective
        slope = dot(derivative, direction)
        assert abs(slope) <= D('1e-28') if alpha < 1 else slope <= D('1e-28')


@pytest.mark.parametrize('mutation', ['station_stress', 'fibre_stress', 'gradient', 'history', 'potential'])
def test_original_equation_auditor_detects_output_mutations(mutation):
    law, data, cell = model(order=2); p = np.linspace(-.3, .5, 18).tolist()
    response = deepcopy(cell.response(p))
    if mutation == 'station_stress': response['stations'][0]['resultants'][0][0] += 1.
    elif mutation == 'fibre_stress':
        fibres = list(response['stations'][0]['fibres']); fibres[0] = replace(fibres[0], stress=(100., 0.))
        response['stations'][0]['fibres'] = tuple(fibres)
    elif mutation == 'gradient': response['gradient'][0][0] += 1.
    elif mutation == 'potential': response['potential'][0][0] += 1.
    else:
        states = list(response['history'].stations); rows = list(states[0].rows); rows[0] = (1., 0., 1., 0.)
        states[0] = replace(states[0], rows=tuple(rows)); response['history'] = replace(response['history'], stations=tuple(states))
    assert max(D(v) for v in identities(law, data, p, response).values()) > D('1e-11')


def test_capture_and_cell_authority_change_detection():
    law, data, cell = model(order=2); baseline = canonical(cell.response([0.]*18))
    data[0]['v'][0][0] *= 1.1
    other = FibreCellConjugate(law, data)
    assert other.identity != cell.identity
    with pytest.raises(ValueError): other.response([0.]*18, cell.virgin())
    assert canonical(cell.response([0.]*18)) == baseline
    with pytest.raises(AttributeError): cell.load_map = ()


@pytest.mark.parametrize('incomplete', [False, True])
def test_endpoint_moment_work_map_requires_two_distinct_stations_per_half_cell(incomplete):
    law, data, _ = model(order=2)
    if incomplete: data = [data[0], data[-1]]
    else:
        for row in data: row['t'] = .5
    with pytest.raises(ValueError): FibreCellConjugate(law, data)
