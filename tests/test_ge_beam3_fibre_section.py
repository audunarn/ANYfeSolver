"""Independent-equation small section checks, not independent authorship/review."""
from dataclasses import replace
from fractions import Fraction as Q
import json

import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_section import (
    Fibre, FibreHistory, FlowCurve, PhysicalFibreSection, PARAMETER_CAPTURE, canonical,
)


def section(contrast=1., tabular=False):
    curve = FlowCurve((0., .125, .5), (.25, .5, .75), 0.) if tabular else FlowCurve.linear(.25, 2.)
    fibres = tuple(Fibre(str(i), y, z, area, 30.*contrast, curve) for i, (y, z, area) in enumerate(
        ((-.5, -.25, .25), (.5, -.25, .125), (-.5, .25, .125), (.5, .25, .25))))
    # Includes genuine generalized elastic coupling, not an isotropic diagonal shortcut.
    background = np.array([[.1, .02, -.03, .04, .05, -.02],
        [0., 2., .05, -.02, .03, .01], [0., 0., 3., .04, -.01, .05],
        [0., 0., 0., 1.5, .05, .03]])
    return PhysicalFibreSection(fibres, background)


def exact_pair(pair):
    return Q(float(pair[0]))+Q(float(pair[1]))


def exact_array(high, low):
    return np.array([exact_pair(p) for p in zip(high.ravel(), low.ravel())], dtype=object).reshape(high.shape)


def flow_data(curve):
    x = list(map(Q, curve.plastic_strain)); y = list(map(Q, curve.flow_stress))
    return [(x[i], x[i+1] if i+1 < len(x) else None, y[i],
        (y[i+1]-y[i])/(x[i+1]-x[i]) if i+1 < len(x) else Q(curve.tail_slope)) for i in range(len(x))]


def integral(curve, a, b):
    total = Q(0)
    for start, end, value, slope in flow_data(curve):
        left = max(a, start); right = min(b, end) if end is not None else b
        if right > left:
            total += (right-left)*value+slope*((right-start)**2-(left-start)**2)/2
    return total


def rational_fibre(fibre, strain, z0, p0):
    """Enumerate both signs, affine branch stationary points and all endpoints.

    Select the least exact rational potential, independently of production's
    monotone fixed-sign interval traversal.
    """
    young = Q(fibre.young); candidates = {Q(0)}
    for start, end, value, slope in flow_data(fibre.curve):
        for sign in (-1, 1):
            delta = (young*(strain-z0)-sign*(value+slope*(p0-start)))/(young+slope)
            p = p0+abs(delta)
            if delta*sign >= 0 and p >= max(start, p0) and (end is None or p <= end):
                candidates.add(delta)
            for knot in (start, end):
                if knot is not None and knot >= p0: candidates.add(sign*(knot-p0))
    def energy(delta):
        return young*(strain-z0-delta)**2/2+integral(fibre.curve, p0, p0+abs(delta))
    delta = min(candidates, key=energy); p = p0+abs(delta)
    slope = next(h for a, b, _, h in reversed(flow_data(fibre.curve)) if p >= a)
    tangent = young if delta == 0 else young*slope/(young+slope)
    return dict(delta=delta, z=z0+delta, p=p, stress=young*(strain-z0-delta),
        modulus=tangent, potential=energy(delta), strain=strain, elastic=strain-z0-delta)


def oracle(law, value):
    e = exact_array(value.strain, value.strain_low)
    f = np.array([[Q(float(x)) for x in row] for row in law.background_factor], dtype=object)
    cb = f.T@f; s = cb@e; ct = cb.copy(); potential = sum(e*s)/2
    recovered = []
    for fibre, origin in zip(law.fibres, value.origin.rows):
        # Direct cross-product axial component supplies an independent sign check.
        rho = (Q(0), Q(fibre.y), Q(fibre.z)); kappa = e[3:]
        strain = e[0]+kappa[1]*rho[2]-kappa[2]*rho[1]
        b = np.array([Q(1), Q(0), Q(0), Q(0), rho[2], -rho[1]], dtype=object)
        item = rational_fibre(fibre, strain, exact_pair(origin[:2]), exact_pair(origin[2:]))
        s += Q(fibre.area)*b*item['stress']
        ct += Q(fibre.area)*item['modulus']*np.outer(b, b)
        potential += Q(fibre.area)*item['potential']; recovered.append(item)
    return s, ct, potential, recovered


def close_exact(actual, expected, tolerance=Q(1, 10**28)):
    assert abs(actual-expected) <= tolerance*max(Q(1), abs(expected))


@pytest.mark.parametrize('contrast', [1., 1.e4, 1.e12])
@pytest.mark.parametrize('tabular', [False, True])
def test_full_rational_potential_tangent_recovery_and_history(contrast, tabular, tmp_path):
    law = section(contrast, tabular); origin = law.virgin(); records = []
    for scale in (.25, 1., 3., .5, 0., -2., 0.):
        e = scale*np.array([.2, -.1, .05, .03, .4, -.3])
        low = np.array([1e-19, -2e-19, 0., 0., 2e-20, -1e-20])
        response = law.response(e, origin, strain_low=low)
        assert canonical(response) == canonical(law.response(e, origin, strain_low=low))
        stress, ct, potential, recovered = oracle(law, response)
        for actual, expected in zip(exact_array(response.resultants, response.resultants_low), stress):
            close_exact(actual, expected)
        for actual, expected in zip(exact_array(response.tangent, response.tangent_low).ravel(), ct.ravel()):
            close_exact(actual, expected)
        close_exact(exact_pair(response.incremental_potential), potential)
        for actual, expected, history, old, fibre in zip(response.fibres, recovered, response.history.rows, origin.rows, law.fibres):
            for key, name in (('total_strain', 'strain'), ('elastic_strain', 'elastic'), ('stress', 'stress'),
                    ('algorithmic_modulus', 'modulus'), ('plastic_increment', 'delta')):
                close_exact(exact_pair(getattr(actual, key)), expected[name])
            close_exact(exact_pair(history[:2]), expected['z'])
            close_exact(exact_pair(history[2:]), expected['p'])
            assert exact_pair(history[2:]) >= exact_pair(old[2:])
        previous_hardening = sum(Q(f.area)*(integral(f.curve, Q(0), exact_pair(row[2:]))
            -Q(f.curve.flow_stress[0])*exact_pair(row[2:])) for f, row in zip(law.fibres, origin.rows))
        close_exact(exact_pair(response.incremental_potential),
            exact_pair(response.stored_energy)-previous_hardening+exact_pair(response.dissipation))
        assert exact_pair(response.dissipation) >= 0
        assert response.origin == origin and not response.production_qualified
        records.append(response); origin = response.history
    with (tmp_path/'fibre-cycle.json').open('xb') as stream: stream.write(canonical(records))


@pytest.mark.parametrize('tabular', [False, True])
def test_work_and_tangent_directional_derivatives(tabular):
    law = section(tabular=tabular); origin = law.response([.3, 0., 0., 0., .4, -.2]).history
    e = np.array([-.25, .1, -.2, .1, .3, .4]); v = np.array([.4, -.3, .2, .1, -.5, .2])
    h = 1e-5; centre = law.response(e, origin); plus = law.response(e+h*v, origin); minus = law.response(e-h*v, origin)
    assert centre.derivative_kind == 'CLASSICAL_SMOOTH_BRANCH'
    derivative = float((exact_pair(plus.incremental_potential)-exact_pair(minus.incremental_potential))/Q(2*h))
    assert abs(derivative-centre.resultants@v) <= 1e-7
    np.testing.assert_allclose((plus.resultants-minus.resultants)/(2*h), centre.tangent@v, rtol=1e-7, atol=1e-7)
    assert np.array_equal(centre.tangent, centre.tangent.T)


@pytest.mark.parametrize('angle', [.37, np.pi/2, 2.6])
def test_material_roll_covariance_and_physical_fibre_work(angle):
    law = section(tabular=True); c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    transform = np.zeros((6, 6)); transform[:3, :3] = rotation; transform[3:, 3:] = rotation
    rotated = []
    for fibre in law.fibres:
        _, y, z = rotation@np.array([0., fibre.y, fibre.z]); rotated.append(replace(fibre, y=y, z=z))
    other = PhysicalFibreSection(rotated, law.background_factor@transform.T)
    old = law.response([.3, 0., 0., 0., -.5, .1]).history
    mapped = FibreHistory(other.identity, old.rows)  # Explicit same-physical-fibre correspondence, not implicit import.
    e = np.array([-.1, .2, -.3, .4, .6, -.5]); a = law.response(e, old); b = other.response(transform@e, mapped)
    np.testing.assert_allclose(b.resultants, transform@a.resultants, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(b.tangent, transform@a.tangent@transform.T, rtol=1e-11, atol=1e-11)
    assert abs(float(exact_pair(a.incremental_potential)-exact_pair(b.incremental_potential))) < 1e-11
    np.testing.assert_allclose([float(exact_pair(f.stress)) for f in a.fibres],
        [float(exact_pair(f.stress)) for f in b.fibres], rtol=1e-11, atol=1e-11)


def scalar_law(curve):
    # Background supplies all coordinates except axial; one centroid fibre supplies axial.
    return PhysicalFibreSection([Fibre('centroid', 0., 0., 1., 8., curve)], np.eye(6)[1:])


def test_yield_and_hardening_knots_have_explicit_derivative_policy():
    law = scalar_law(FlowCurve((0., .25, .5), (1., 2., 2.), 0.))
    yield_point = law.response([.125, 0., 0., 0., 0., 0.])
    assert yield_point.fibres[0].branch == 'YIELD_BOUNDARY'
    assert yield_point.tangent[0, 0] == 8.
    knot = law.response([.5, 0., 0., 0., 0., 0.])  # delta=.25, elastic strain=.25.
    assert knot.fibres[0].branch == 'HARDENING_KNOT'
    assert knot.derivative_kind == 'SEMISMOOTH_BRANCH_SELECTION'
    assert knot.tangent[0, 0] == 0.  # Right-hand plateau, no numerical floor.


def test_perfect_plasticity_preserves_zero_tangent_and_dissipation():
    value = scalar_law(FlowCurve.linear(1., 0.)).response([1., 0., 0., 0., 0., 0.])
    assert value.resultants[0] == 1. and value.tangent[0, 0] == 0.
    assert exact_pair(value.dissipation) == Q(7, 8)
    assert exact_pair(value.incremental_potential) == Q(15, 16)


def test_input_ownership_origin_rollback_replay_and_cancellation(tmp_path):
    law = section(); origin = law.virgin(); trial = law.response(np.ones(6), origin)
    assert origin == law.virgin()
    for array in (trial.strain, trial.strain_low, trial.resultants, trial.resultants_low, trial.tangent, trial.tangent_low,
                  law.background_factor):
        with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(AttributeError): law.identity = 'changed'
    for fail_at in (1, 3, 6):
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == fail_at: raise RuntimeError('cancelled fixture')
        with pytest.raises(RuntimeError, match='cancelled fixture'): law.response(np.ones(6), origin, check=cancel)
        assert origin == law.virgin()
    value = json.loads(canonical(trial.history))
    restored = FibreHistory(value['section_identity'], tuple(map(tuple, value['rows'])))
    next_value = law.response(-np.ones(6), trial.history)
    assert canonical(next_value) == canonical(law.response(-np.ones(6), restored))
    with (tmp_path/'fibre-replay.json').open('xb') as stream: stream.write(canonical(next_value))


@pytest.mark.parametrize('mutation', ['identity', 'count', 'mutable', 'boolean', 'nonfinite', 'inadmissible', 'unnormalized'])
def test_foreign_or_corrupted_history_rejected(mutation):
    law = section(); origin = law.virgin(); rows = list(origin.rows)
    if mutation == 'identity': origin = replace(origin, section_identity='0'*64)
    elif mutation == 'count': origin = replace(origin, rows=origin.rows[:-1])
    elif mutation == 'mutable': origin = replace(origin, rows=rows)
    else:
        rows[0] = {'boolean': (False, 0., 0., 0.), 'nonfinite': (float('nan'), 0., 0., 0.),
                   'inadmissible': (1., 0., 0., 0.), 'unnormalized': (.5, .5, 1., 0.)}[mutation]
        origin = replace(origin, rows=tuple(rows))
    with pytest.raises(ValueError): law.response(np.zeros(6), origin)


@pytest.mark.parametrize('mutation', ['area', 'young', 'coordinate', 'negative_curve', 'descending_curve',
    'bad_knots', 'negative_tail', 'duplicate', 'missing_shear', 'bad_shape', 'bool_strain', 'nan_strain'])
def test_invalid_material_or_section_inputs_rejected(mutation):
    law = section(); f = law.fibres[0]
    with pytest.raises(ValueError):
        if mutation == 'area': replace(f, area=0.)
        elif mutation == 'young': replace(f, young=-1.)
        elif mutation == 'coordinate': replace(f, y=float('inf'))
        elif mutation == 'negative_curve': FlowCurve.linear(-1., 0.)
        elif mutation == 'descending_curve': FlowCurve((0., 1.), (2., 1.), 0.)
        elif mutation == 'bad_knots': FlowCurve((1., 2.), (1., 2.), 0.)
        elif mutation == 'negative_tail': FlowCurve.linear(1., -1.)
        elif mutation == 'duplicate': PhysicalFibreSection([f, f], np.eye(6))
        elif mutation == 'missing_shear': PhysicalFibreSection(law.fibres, np.zeros((1, 6)))
        elif mutation == 'bad_shape': PhysicalFibreSection(law.fibres, np.eye(5))
        elif mutation == 'bool_strain': law.response([True, 0., 0., 0., 0., 0.])
        elif mutation == 'nan_strain': law.response([float('nan'), 0., 0., 0., 0., 0.])


def test_material_adapter_captures_only_declared_energy_curves():
    from anymaterial.curves import LinearHardeningCurve, PiecewiseLinearCurve
    assert FlowCurve.from_anymaterial(LinearHardeningCurve(3., 2.),
        parameter_interpretation=PARAMETER_CAPTURE) == FlowCurve.linear(3., 2.)
    table = PiecewiseLinearCurve((0., .25, .5), (1., 2., 3.))
    captured = FlowCurve.from_anymaterial(table, parameter_interpretation=PARAMETER_CAPTURE)
    assert captured == FlowCurve((0., .25, .5), (1., 2., 3.), 0.)
    with pytest.raises(ValueError): FlowCurve.from_anymaterial(object(), parameter_interpretation=PARAMETER_CAPTURE)
    with pytest.raises(ValueError): FlowCurve.from_anymaterial(table, parameter_interpretation='AUTOMATIC_TRUE_STRESS_CONVERSION')
    with pytest.raises(TypeError): FlowCurve.from_anymaterial(table)


def test_material_and_fibre_order_mutations_change_identity():
    law = section()
    for fibres in (law.fibres[::-1], (replace(law.fibres[0], area=.5),)+law.fibres[1:],
                   (replace(law.fibres[0], curve=FlowCurve.linear(.5, 2.)),)+law.fibres[1:]):
        other = PhysicalFibreSection(fibres, law.background_factor)
        assert other.identity != law.identity
        with pytest.raises(ValueError): other.response(np.zeros(6), law.virgin())
