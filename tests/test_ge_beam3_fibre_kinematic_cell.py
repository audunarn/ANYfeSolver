"""Constrained primal checks against the preserved resultant-entry cell."""
from decimal import Decimal as D, localcontext
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver._ge_beam3_fibre_kinematic_cell import response
from anyfesolver._ge_beam3_fibre_section import canonical
from anyfesolver._ge_beam3_station_resultant_cell import mul, mv, tr
from test_ge_beam3_fibre_cell import model, unpack, dot


@pytest.mark.parametrize('contrast', [1., 1.e4, 1.e12])
@pytest.mark.parametrize('tabular', [False, True])
def test_kinematic_entry_and_preserved_complementary_entry_agree(contrast, tabular, tmp_path):
    law, data, cell = model(contrast, tabular)
    origin = cell.response((.1*np.sin(np.arange(18))).tolist())['history']; before = canonical(origin)
    k = (.03*np.cos(np.arange(18))).tolist(); result = response(cell, k, origin)
    opposite = cell.response(result['resultants'][0], origin)
    with localcontext() as ctx:
        ctx.prec = 80
        constraint = max(abs(a-b) for a, b in zip(mv(tr(cell.load_map), unpack(result['primal_variables'])), map(D.from_float, k)))
        compatibility = max(abs(a-b) for a, b in zip(unpack(opposite['gradient']), map(D.from_float, k)))
        energy = abs(unpack(result['potential'])[0]+unpack(opposite['potential'])[0]-dot(list(map(D.from_float, result['resultants'][0])), list(map(D.from_float, k))))
        hessian = [unpack(row) for row in result['hessian']]; compliance = [unpack(row) for row in opposite['hessian']]
        inverse = max(abs(x-D(i == j)) for i, row in enumerate(mul(hessian, compliance)) for j, x in enumerate(row))
        station_error = D(0)
        for row, fields, history in zip(data, result['stations'], origin.stations):
            # Separate section entry uses the original station strains/history.
            native = law.response(fields['strain'][0], history, strain_low=fields['strain'][1])
            native_stress = [D.from_float(float(h))+D.from_float(float(l)) for h, l in zip(native.resultants, native.resultants_low)]
            station_error = max(station_error, *(abs(a-b) for a, b in zip(native_stress, unpack(fields['resultants']))))
        errors = {key: str(value) for key, value in dict(constraint=constraint, compatibility=compatibility,
            energy=energy, inverse=inverse, station=station_error).items()}
    (tmp_path/'primal.json').write_bytes(canonical(dict(response=result, errors=errors)))
    assert max(D(v) for v in errors.values()) <= D('1e-11')
    assert D(result['station_equilibrium_error']) <= D('1e-26')
    assert canonical(origin) == before and not result['production_qualified']
    assert canonical(result) == canonical(response(cell, k, origin))


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_primal_potential_first_and_second_variations(contrast):
    _, _, cell = model(contrast, True)
    k = .03*np.cos(np.arange(18)); direction = .1*np.sin(np.arange(18)); eps = 1e-6
    centre = response(cell, k.tolist()); plus = response(cell, (k+eps*direction).tolist()); minus = response(cell, (k-eps*direction).tolist())
    with localcontext() as ctx:
        ctx.prec = 80
        d = list(map(D.from_float, direction)); scale = D.from_float(2*eps)
        first = (unpack(plus['potential'])[0]-unpack(minus['potential'])[0])/scale
        expected = dot(unpack(centre['resultants']), d)
        second = [(a-b)/scale for a, b in zip(unpack(plus['resultants']), unpack(minus['resultants']))]
        wanted = mv([unpack(row) for row in centre['hessian']], d)
        assert abs(first-expected)/max(D(1), abs(expected)) <= D('1e-7')
        assert max(abs(a-b) for a, b in zip(second, wanted))/max(D(1), *(abs(v) for v in wanted)) <= D('1e-7')


@pytest.mark.parametrize('incident', ['shape', 'bool', 'nan', 'low_shape', 'low_pair', 'history'])
def test_kinematic_inputs_fail_closed(incident):
    _, _, cell = model(); k = [0.]*18; low = [0.]*18; origin = cell.virgin()
    if incident == 'shape': k.pop()
    elif incident == 'bool': k[0] = True
    elif incident == 'nan': k[0] = float('nan')
    elif incident == 'low_shape': low.pop()
    elif incident == 'low_pair': low[0] = .1
    else: origin = replace(origin, cell_identity='0'*64)
    with pytest.raises(ValueError): response(cell, k, origin, kinematics_low=low)


def test_cancelled_local_solve_does_not_change_history():
    _, _, cell = model(1.e12); origin = cell.virgin(); before = canonical(origin); calls = []
    def cancel():
        calls.append(1)
        if len(calls) == 20: raise RuntimeError('cancelled local construction')
    with pytest.raises(RuntimeError, match='cancelled local'): response(cell, [.01]*18, origin, check=cancel)
    assert canonical(origin) == before
