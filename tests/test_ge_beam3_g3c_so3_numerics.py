"""Separate frozen static (3), scalar (3), and full-Jet (4) inventories.

38a983f author lanes passed but independent review found set-deduplicated
signed-zero coverage. Preserve its raw runs; this successor tests both signs
explicitly and checks value/gradient/Hessian groups separately. Kernel unchanged.
"""
import ast
from copy import deepcopy
from decimal import Decimal as D, localcontext
from hashlib import sha256
import importlib.util
import json
from math import factorial
from pathlib import Path
import sys

import numpy as np
import pytest

from anyfesolver import _ge_beam3_g3c_so3_numerics as kernel
from anyfesolver._ge_beam3_mixed_ad import Jet2, math_cos_limit
import run_ge_beam3_g3c_so3_numerics as runner

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('g3c_independent_decimal', ROOT/runner.ORACLE)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)
TOLERANCE = 1e-11


def norm_error(actual, reference):
    a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b)))


def flattened(jets):
    return [v for j in jets for v in [float(j.value),
            *(float(v) for v in j.gradient), *(float(v) for row in j.hessian for v in row)]]


def groups(jets):
    return ([float(j.value) for j in jets],
            [float(v) for j in jets for v in j.gradient],
            [float(v) for j in jets for row in j.hessian for v in row])


def jet_check(actual, reference, higher):
    errors = []
    for a, r, hi in zip(groups(actual), groups(reference), groups(higher)):
        assert norm_error(r, hi) <= 1e-60, 'independent high precision did not agree'
        error = norm_error(a, hi)
        assert error <= TOLERANCE, error
        errors.append(error)
    return max(errors)


def scalar_reference(kind, value):
    low, _ = oracle.coefficient(kind, float(value), 90)
    high, _ = oracle.coefficient(kind, float(value), 110)
    low, high = tuple(map(float, low)), tuple(map(float, high))
    assert low == high, 'independent scalar rounding disagreement'
    return high


def neighbours(values):
    return sorted({float(v) for x in values for v in
                   (np.nextafter(x, -np.inf), x, np.nextafter(x, np.inf))})


def checkpoint(name, **values):
    print('G3C_SO3 '+json.dumps(dict(name=name, **values), sort_keys=True, allow_nan=False), flush=True)


class TestStaticIsolation:
    def test_source_bindings_and_no_existing_routing(self):
        contract = runner.environment.strict((ROOT/runner.CONTRACT).read_bytes().replace(b'\r\n', b'\n'))
        runner.verify_sources(contract)
        changes = runner.git('diff', '--name-status', '--no-renames', contract['base']['commit'])
        for row in changes.splitlines():
            status, path = row.split('\t')
            assert status == 'A', path
            if path.startswith('src/'):
                assert path == 'src/anyfesolver/_ge_beam3_g3c_so3_numerics.py'
        tree = ast.parse((ROOT/'src/anyfesolver/_ge_beam3_g3c_so3_numerics.py').read_text())
        forbidden = {'_exp_coefficients', '_log_factor', 'so3_exp', 'so3_log', 'rotation_exponential', 'rotation_log'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name == 'math' for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module in ('math', '_ge_beam3_mixed_ad')
                assert not any(alias.name in forbidden for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'eval', 'exec', '__import__', 'setattr'}
        for path in (ROOT/'src').rglob('*.py'):
            if path.name == '_ge_beam3_g3c_so3_numerics.py':
                continue
            assert '_ge_beam3_g3c_so3_numerics' not in path.read_text(encoding='utf-8'), path
        assert kernel.Jet2 is Jet2

    def test_canonical_and_source_hash_mutations_rejected(self):
        for raw in (b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n',
                    b'{ "x":1}\n', b'{"x":1}', b'{"x":1e999}\n'):
            with pytest.raises((ValueError, OverflowError)):
                runner.environment.strict(raw)
        original = runner.environment.strict((ROOT/runner.CONTRACT).read_bytes().replace(b'\r\n', b'\n'))
        for key, value in (('sha256', '0'*64), ('bytes', 0)):
            changed = deepcopy(original)
            changed['sources'][0][key] = value
            with pytest.raises(ValueError, match='changed bound source'):
                runner.verify_sources(changed)

    def test_independent_oracle_identity_and_imports(self):
        raw = (ROOT/runner.ORACLE).read_bytes().replace(b'\r\n', b'\n')
        assert sha256(raw).hexdigest() == runner.ORACLE_SHA
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node, ast.ImportFrom):
                assert node.module in {'dataclasses', 'decimal', 'math'}
            if isinstance(node, ast.Import):
                assert all(alias.name in {'dataclasses', 'decimal', 'math'} for alias in node.names)
        assert oracle.self_test()['status'] == 'ORACLE_SELF_TEST_ONLY'


class TestScalarAccuracy:
    def test_a_priori_tail_majorants(self):
        with localcontext() as context:
            context.prec = 100
            first = D(4)*41*40/D(4)**41
            ratio = D(42)/40/4
            assert first/(1-ratio) < D('3e-21')
            first = D(17*16)/D(factorial(35))
            ratio = D(18)/16/D(36*37)
            assert first/(1-ratio) < D('1e-35')
        assert len(kernel.SINC) == len(kernel.COSC) == 17
        assert len(kernel.LOG) == 41

    def test_exp_coefficient_values_and_derivatives(self):
        grid = [-0., *sorted(set([0., np.nextafter(0., 1.),
            *np.logspace(-300, -1, 22), *np.linspace(1., (2*np.pi)**2, 32),
            *neighbours((1e-8, 1., np.pi**2, (1.4*np.pi)**2))]))]
        zero_signs = [bool(np.signbit(x)) for x in grid if x == 0.]
        assert zero_signs == [True, False]
        worst = 0.
        for x in grid:
            jets = kernel._exp_coefficients(Jet2.variable(float(x), 0, 1))
            for kind, jet in zip(('sinc', 'cosc'), jets):
                reference = scalar_reference(kind, x)
                # Compare each derivative separately; a large value must not
                # hide an inaccurate derivative in a combined norm.
                for a, b in zip((jet.value, jet.gradient[0], jet.hessian[0, 0]), reference):
                    error = norm_error([a], [b]); worst = max(worst, error)
                    assert error <= TOLERANCE, (kind, x, error)
        checkpoint('exp_scalar', arguments=len(grid), signed_zero_cases=len(zero_signs), maximum_error=worst)

    def test_log_coefficient_values_and_derivatives(self):
        lower = np.nextafter(math_cos_limit(), 1.)
        grid = sorted(set([lower, 1., np.nextafter(1., np.inf),
            *np.linspace(lower, 1., 32), *neighbours((.5, 1.-1e-7)),
            *(1.-float(d) for d in np.logspace(-16, -1, 18))]))
        worst = 0.
        for c in grid:
            jet = kernel._log_factor(Jet2.variable(float(c), 0, 1))
            reference = scalar_reference('log', c)
            for a, b in zip((jet.value, jet.gradient[0], jet.hessian[0, 0]), reference):
                error = norm_error([a], [b]); worst = max(worst, error)
                assert error <= TOLERANCE, (c, error)
        checkpoint('log_scalar', arguments=len(grid), maximum_error=worst)


def seeded(values):
    actual, reference = [], []
    for i, value in enumerate(values):
        gradient = np.array([.3*(i+1), -.2+i*.07, .11-i*.03])
        hessian = np.array([[.01*(i+1), .02, -.03], [.02, -.04, .05], [-.03, .05, .06]])
        actual.append(Jet2(float(value), gradient, hessian))
        reference.append(oracle.DJet(oracle.decimal(float(value)),
            tuple(oracle.decimal(float(v)) for v in gradient),
            tuple(tuple(oracle.decimal(float(v)) for v in row) for row in hessian)))
    return actual, reference


class TestFullJetAccuracy:
    def test_exp_full_jets_with_curved_input_seeds(self):
        axis = np.array([1., -.7, .4]); axis /= np.linalg.norm(axis)
        angles = sorted(set([0., 1e-8, 1e-4, .003, .1, np.pi, 1.4*np.pi,
                             *neighbours((1., np.sqrt(1e-8)))]))
        worst = 0.
        for angle in angles:
            a, r = seeded(axis*angle)
            actual = kernel.so3_exp(a)
            low, high = oracle.exp_jets(r, 100), oracle.exp_jets(r, 120)
            worst = max(worst, jet_check(sum(actual, []), sum(low, []), sum(high, [])))
        checkpoint('exp_full_jet', arguments=len(angles), maximum_error=worst)

    def test_log_full_matrix_jets(self):
        axis = np.array([1., -.7, .4]); axis /= np.linalg.norm(axis)
        cosines = [1., 1.-1e-12, 1.-1e-6, .2, -.4, math_cos_limit()+1e-8,
                   *neighbours((.5, 1.-1e-7))]
        worst = 0.
        for cosine in cosines:
            values = axis*float(np.arccos(cosine))
            q = np.array([[float(j.value) for j in row] for row in oracle.exp_from_vector(list(map(float, values)))])
            a, r = seeded(q.reshape(-1))
            a = [a[3*i:3*i+3] for i in range(3)]
            r = [r[3*i:3*i+3] for i in range(3)]
            worst = max(worst, jet_check(kernel.so3_log(a), oracle.log_jets(r, 100), oracle.log_jets(r, 120)))
        checkpoint('log_full_jet', arguments=len(cosines), maximum_error=worst)

    def test_principal_identity_and_proper_covariance(self):
        axis = np.array([1., -.7, .4]); axis /= np.linalg.norm(axis)
        transform = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for angle in (0., .001, .01, .6, 1.1, 2., .9*np.pi-1e-5):
            vector = axis*angle
            jets = [Jet2.variable(float(v), i, 3) for i, v in enumerate(vector)]
            result = kernel.so3_log(kernel.so3_exp(jets))
            assert norm_error([v.value for v in result], vector) <= TOLERANCE
            assert norm_error([v.gradient for v in result], np.eye(3)) <= TOLERANCE
            assert norm_error([v.hessian for v in result], np.zeros((3, 3, 3))) <= TOLERANCE
            moved = [sum((transform[i, j]*jets[j] for j in range(3)), start=Jet2.constant(0., 3)) for i in range(3)]
            a, b = kernel.so3_exp(moved), kernel.so3_exp(jets)
            expected = [[sum((transform[i, k]*b[k][l]*transform[j, l] for k in range(3) for l in range(3)),
                            start=Jet2.constant(0., 3)) for j in range(3)] for i in range(3)]
            for actual_group, expected_group in zip(groups(sum(a, [])), groups(sum(expected, []))):
                assert norm_error(actual_group, expected_group) <= TOLERANCE

    def test_domain_guards_and_no_clipping(self):
        for x in (-1., -np.nextafter(0., 1.), np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError):
                kernel._exp_coefficients(Jet2.variable(float(x), 0, 1))
        for c in (math_cos_limit(), np.nextafter(math_cos_limit(), -np.inf), -1., np.nan, np.inf):
            with pytest.raises(ValueError):
                kernel._log_factor(Jet2.variable(float(c), 0, 1))
        jet = kernel._log_factor(Jet2.variable(float(np.nextafter(1., np.inf)), 0, 1))
        assert norm_error([jet.gradient[0]], [-1./3.]) <= TOLERANCE
        assert norm_error([jet.hessian[0, 0]], [4./15.]) <= TOLERANCE
