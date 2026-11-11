"""Independent finite controls for analytic Pi8 + literal4/8 remainder, V15."""
from fractions import Fraction as Q
from math import factorial
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import general_contact_enriched_reference_polynomial as poly
from scripts import general_contact_enriched_reference_geometry as geometry


def exact_monomial(vertices, area, i, j):
    """Independent exact multinomial/barycentric expansion (not recurrence)."""
    total = Q(0)
    for a in range(i+1):
        for b in range(i-a+1):
            c = i-a-b
            first = Q(factorial(i), factorial(a)*factorial(b)*factorial(c))
            first *= vertices[0][0]**a*vertices[1][0]**b*vertices[2][0]**c
            for d in range(j+1):
                for e in range(j-d+1):
                    f = j-d-e
                    second = Q(factorial(j), factorial(d)*factorial(e)*factorial(f))
                    second *= vertices[0][1]**d*vertices[1][1]**e*vertices[2][1]**f
                    total += first*second*2*area*Q(factorial(a+d)*factorial(b+e)*factorial(c+f), factorial(i+j+2))
    return total


def test_all45_exact_degree8_moments_translated_clipped_and_B4():
    triangles = (
        ((Q(0), Q(0)), (Q(1), Q(0)), (Q(0), Q(1))),
        ((Q(-3, 4), Q(1, 8)), (Q(1, 4), Q(-3, 8)), (Q(7, 8), Q(3, 4))),
        ((Q(15, 16), Q(7, 8)), (Q(31, 32), Q(7, 8)), (Q(15, 16), Q(29, 32))))
    for vertices in triangles:
        a, b, c = vertices
        area = abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
        actual = poly.triangle_moments([[float(v) for v in p] for p in vertices], float(area))
        assert set(actual) == set(poly.EXPONENTS) and len(actual) == 45
        assert actual[0, 0] == float(area)  # catches the physical factor2.
        for i, j in poly.EXPONENTS:
            expected = float(exact_monomial(vertices, area, i, j))
            assert abs(actual[i, j]-expected) <= 2e-13*float(area)
        hats, bubble = poly.hats_and_bubble([.125, -.0625, .03125, .015625])
        polynomial = poly.multiply(hats[2], bubble)
        expected = sum(Q.from_float(v)*exact_monomial(vertices, area, *key) for key, v in polynomial.items())
        assert abs(poly.integrate(polynomial, actual)-float(expected)) <= 2e-13*float(area)
        assert any(sum(key) == 8 and value != 0 for key, value in polynomial.items())


def analytic_factory(monkeypatch):
    # Reuse only the existing synthetic fixture builder; no old tests execute.
    path = Path(__file__).with_name('test_general_contact_enriched_reference.py')
    spec = importlib.util.spec_from_file_location('v15_analytic_reference_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analytic_oracle(monkeypatch)


def posed_oracle(base, order, pose, *, cache=False):
    definitions = tuple(SimpleNamespace(source_element=1, target_element=t,
        source_offset=-.025, target_offset=.015000000000000001) for t in (2, 3))
    return geometry.ContactOracle(base.model, definitions, pose, base.z, order, cache_pose_results=cache)


def test_same_actual_domain_order_independent_P_and_nonpolynomial_remainder(monkeypatch, record_property):
    base = analytic_factory(monkeypatch)
    pose = base.base_pose.copy()
    pose[2] = .001; pose[10] = .002; pose[30] = .0001; pose[32] = .002
    pose[48:] = (.001, .0002, -.0003, .0001)
    original = geometry.control_polynomials
    records = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append(result)
        return result
    monkeypatch.setattr(geometry, 'control_polynomials', observed)
    low = posed_oracle(base, 4, pose)
    low_records = list(records); records.clear()
    high = posed_oracle(base, 8, pose)
    assert len(low_records) == len(records) == 2 and low_records == records
    assert low.base_value['quadrature_points']*4 == high.base_value['quadrature_points']
    assert low.base_value['integration_method'] == 'ANALYTIC_PI8_PLUS_LITERAL_GAUSS_REMAINDER_V15'
    assert all(max((sum(key) for key in p), default=0) <= 2 for p in records[0][0])
    assert any(sum(key) == 8 for p in records[0][1] for key in p)
    # Actual F is genuinely nonpolynomial: varying skin normal and nonlinear
    # inverse remain evaluated, not replaced by P or an affine source branch.
    q, beta, body = high.base_sources[1]
    normal = geometry.unit(np.cross(q[1]-q[0], q[3]-q[0]))
    tangent = geometry.unit(q[1]-q[0]); frame = np.column_stack((tangent, np.cross(normal, tangent)))
    p, pr, ps, n = geometry.skin(np.array([.31]), np.array([-.27]), q, beta, body)
    local_normal = np.cross(pr, ps); local_normal /= np.linalg.norm(local_normal, axis=1)[:, None]
    assert np.linalg.norm(local_normal[0]-normal) > 0
    xy = (p-q[0]) @ frame
    _, _, _, _, rs = geometry.inverse_projected(xy, q, beta, body, frame)
    assert np.max(np.abs(rs-np.array([[.31, -.27]]))) <= 1e-12
    corner = (q-q[0]) @ frame
    ar, ass = (corner[1]-corner[0])/2, (corner[3]-corner[0])/2
    center = corner[0]+ar+ass; det = geometry.cross2(ar, ass)
    ra = geometry.cross2(xy-center, ass)/det; sa = geometry.cross2(ar, xy-center)/det
    full_difference = None
    for index, tid in enumerate(high.targets):
        target = high.base_targets[tid]; t = (target-q[0]) @ frame
        a, b = t[1]-t[0], t[2]-t[0]; denominator = geometry.cross2(a, b)
        l1, l2 = geometry.cross2(xy-t[0], b)/denominator, geometry.cross2(a, xy-t[0])/denominator
        target_hat = np.column_stack((1-l1-l2, l1, l2))
        if np.min(target_hat) < 0:
            continue
        gap = np.sum(local_normal*(p-target_hat @ target), axis=1)
        _, rr, ss = geometry.q4(rs[:, 0], rs[:, 1])
        reference, director, offset, _, _ = high.source_data[1]
        reference = reference+offset*director
        measure = np.linalg.norm(np.cross(rr @ reference, ss @ reference), axis=1)/(np.cross(pr, ps) @ normal)
        full = measure[:, None]*n*gap[:, None]
        selected = np.column_stack([poly.evaluate(polynomial, ra, sa) for polynomial in records[index][1]])
        full_difference = float(np.max(np.abs(full-selected)))
        break
    assert full_difference is not None and full_difference > 0
    record_property('actual_nonpolynomial_remainder_max', full_difference)
    record_property('literal4_8_query_points', json.dumps([low.base_value['quadrature_points'], high.base_value['quadrature_points']]))


def test_control_variate_refusal_actual_pose_rebuild_mutation_and_cancel(monkeypatch):
    with pytest.raises(poly.PolynomialRefused, match='degree overflow'):
        poly.multiply({(8, 0): 1.}, {(1, 0): 1.})
    with pytest.raises(poly.PolynomialRefused, match='finite'):
        poly.checked({(0, 0): float('nan')})
    with pytest.raises(poly.PolynomialRefused, match='positive'):
        poly.triangle_moments([[0., 0.], [1., 0.], [0., 1.]], 0.)
    with pytest.raises(poly.PolynomialRefused, match='finite triangle'):
        poly.triangle_moments([[float('inf'), 0.], [1., 0.], [0., 1.]], .5)
    with pytest.raises(poly.PolynomialRefused, match='all45'):
        poly.integrate({(0, 0): 1.}, {(0, 0): .5})
    with pytest.raises(poly.PolynomialRefused, match='positive current affine'):
        poly.control_polynomials([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]],
            [0.]*4, [0., 0., 1.], [0., 0., 1.], [[-2., -2., -.2], [2., -2., -.2], [-2., 2., -.2]],
            [[-2., -2.], [2., -2.], [-2., 2.]], [0., 0.], [1., 0.], [2., 0.], 1.)
    base = analytic_factory(monkeypatch)
    pose = base.base_pose.copy(); pose[48:] = (.001, .0002, .0003, .0004)
    oracle = posed_oracle(base, 8, pose, cache=True)
    trial = pose.copy(); trial[49] += .0007; trial[26] += .0001
    changed = oracle.evaluate(trial)
    rebuilt = posed_oracle(base, 8, trial).base_value
    assert abs(changed['work']-rebuilt['work']) <= 1e-13
    assert np.max(np.abs(changed['gap']-rebuilt['gap'])) <= 1e-13
    assert changed['work'] != oracle.base_value['work']
    before_hits = oracle.cache_hits
    assert oracle.evaluate(trial)['work'] == changed['work']
    assert oracle.cache_hits == before_hits+1 and len(oracle._pose_cache) == 2
    with pytest.raises(ValueError):
        changed['gap'][0] = 0
    oracle.require_exact_reuse(base.model, oracle.definitions, pose, base.z, 8)
    foreign_force = base.z.copy(); foreign_force[0] += .01
    with pytest.raises(geometry.ReferenceRefused, match='reuse'):
        oracle.require_exact_reuse(base.model, oracle.definitions, pose, foreign_force, 8)
    old = base.model.mesh.nodes[1].z
    try:
        base.model.mesh.nodes[1].z = old+.01
        with pytest.raises(geometry.ReferenceRefused, match='mutated'):
            oracle.evaluate(trial)
    finally:
        base.model.mesh.nodes[1].z = old
    assert abs(oracle.evaluate(trial)['work']-changed['work']) <= 1e-13
    from anyfesolver.control import SolveCancelled
    from scripts import general_contact_enriched_reference as reference
    # The requested full_work callback must remain authoritative when a prior
    # oracle (whose own callback is false) supplies an exact cached pose.
    oracle.cancel = lambda: False
    cached_plus = pose.copy(); cached_plus[0] += 3e-6
    oracle.evaluate(cached_plus)
    before_hits = oracle.cache_hits
    requests = []
    def requested_cancel():
        requests.append(True)
        return len(requests) == 4
    with monkeypatch.context() as context:
        context.setattr(reference, 'coordinate_inventory', lambda model, system: {})
        context.setattr(reference, 'STEPS', (3e-6,))
        context.setattr(reference, 'ORDERS', (8,))
        with pytest.raises(SolveCancelled, match='full238 work'):
            reference.full_work(base.model, object(), oracle.definitions, pose, base.z,
                np.zeros(len(pose)), cancel=requested_cancel, reuse={(3e-6, 8): oracle})
    assert len(requests) == 4 and oracle.cache_hits == before_hits+1
    oracle.cancel = lambda: True
    with pytest.raises(SolveCancelled):
        oracle.evaluate(trial)
