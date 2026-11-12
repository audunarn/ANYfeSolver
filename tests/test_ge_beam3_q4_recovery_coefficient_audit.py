"""Five registered Q4 coefficient-audit tests; bounded runner owns execution.

These tests do not qualify physical recovery. Primitive/parser/source guards are
separate from the two complete source-equation fixture audits.

The separately registered fifth node actually alters producer assembly helpers
and exercises the same strict independent verifier against one independently
reconstructed square baseline. It does not label two positive replicas or the
primitive parser checks as mechanical mutation coverage.
"""
import ast
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'docs' / 'reference_cases'
sys.path.insert(0, str(REFERENCE))
from ge_beam3_q4_exact_field import Field, dot, matmul, positive_rational_root, solve
from ge_beam3_q4_recovery_coefficient_producer import audit

SCIENTIFIC_RECORDS = []
SOURCE_BINDINGS = {
    'src/anyfesolver/e4_pl_element.py': (293258, '7fc46a18046e043512a5fb2c3f61ed0b95b834e4dde764f63c500a885e87ea38'),
    'src/anyfesolver/elements.py': (244692, 'f8c59792947a3c9b84416c61a4d9db98e42926d1bf21e74e736afbf6a9a88b37'),
    'src/anyfesolver/_ge_beam3_g3c_local_shell.py': (19430, '69d9f27a96ed7e9a5959a42a081eeafbbe72021de6e4da1f91355c5fb3850f10'),
    'docs/GE_BEAM3_G3C_MATRIX_SHELL_CONTRACT.md': (9625, '59a09dc7db6256a145520ae7b4f2d325dee1abca5d5b5c381a7ada57a2ad92fd'),
    'docs/GE_BEAM3_G3C_MO16_SOURCE_ASSESSMENT.md': (7666, 'e6b702fba5c32ef22537a137fc150fa54b63028c6910df96c5c5f0c6a722026e'),
}


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _strict(raw):
    def pairs(values):
        made = {}
        for key, value in values:
            if key in made:
                raise ValueError('duplicate JSON key')
            made[key] = value
        return made
    def constant(value):
        raise ValueError('nonfinite JSON value: ' + value)
    value = json.loads(raw.decode('ascii'), object_pairs_hook=pairs, parse_constant=constant)
    if _canonical(value) != raw:
        raise ValueError('noncanonical JSON encoding')
    return value


def _regular_bytes(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError('not a regular non-reparse file')
    return path.read_bytes()


def _source_binding(raw, count, digest):
    normalized = raw.replace(b'\r\n', b'\n')
    if len(normalized) != count or sha256(normalized).hexdigest() != digest:
        raise ValueError('source authority mismatch')


def _write_exclusive(path, value):
    raw = _canonical(value)
    with path.open('xb') as stream:
        stream.write(raw)
    if _regular_bytes(path) != raw:
        raise ValueError('exclusive proof write did not preserve bytes')
    return raw


def test_exact_field_laws_and_schema():
    roots = [Field.root(prime) for prime in (2, 3, 5)]
    for value, prime in zip(roots, (2, 3, 5)):
        assert value * value == prime
        assert value * value.inverse() == 1
        assert value.inverse() * value == 1
    basis = [Field.from_coefficients([int(i == mask) for i in range(8)]) for mask in range(8)]
    for left in basis:
        for right in basis:
            assert left * right == right * left
            assert (left + right) * roots[0] == left * roots[0] + right * roots[0]
    x = 1 + roots[0] + roots[1] * roots[2]
    y = 2 - roots[2]
    z = Fraction(2, 7) + roots[1]
    assert (x * y) * z == x * (y * z)
    assert x * x.inverse() == x.inverse() * x == 1
    assert x - x == 0 and not (x - x) and bool(x)
    assert ((roots[0] + roots[1]) ** 2 - 5) ** 2 == 24
    assert positive_rational_root('18/5') ** 2 == Field('18/5')
    with pytest.raises(ZeroDivisionError):
        Field(0).inverse()
    with pytest.raises(ValueError):
        positive_rational_root(7)
    matrix = [[Field(2), roots[0]], [roots[0], Field(3)]]
    rhs = [[Field(1), Field(0)], [Field(0), Field(1)]]
    inverse = solve(matrix, rhs)
    assert matmul(matrix, inverse) == rhs
    assert matmul(inverse, matrix) == rhs
    assert dot([x, y], [1, 0]) == x
    payload = {'field': x.coefficients(), 'zero': Field(0).coefficients()}
    assert _strict(_canonical(payload)) == payload
    for invalid in (b'{"x":0,"x":1}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n', b'{ "x":1}\n', b'{"x":1}', b'{"x":-Infinity}\n'):
        with pytest.raises(ValueError):
            _strict(invalid)
    SCIENTIFIC_RECORDS.append({'test': 'exact_field_laws_and_schema',
                              'primitive_arithmetic_passed': True,
                              'canonical_negative_records': 6,
                              'mechanical_mutation_coverage_claimed': False})


def test_registered_source_boundary():
    for name, (count, digest) in SOURCE_BINDINGS.items():
        raw = _regular_bytes(ROOT / name)
        _source_binding(raw, count, digest)
        with pytest.raises(ValueError):
            _source_binding(raw + b'\n', count, digest)
        with pytest.raises(ValueError):
            _source_binding(raw, count, ('0' if digest[0] != '0' else '1') + digest[1:])
    permitted = {'fractions', 'itertools', 'math', 'json', 'ge_beam3_q4_exact_field'}
    for name in ('ge_beam3_q4_exact_field.py', 'ge_beam3_q4_recovery_coefficient_producer.py', 'ge_beam3_q4_recovery_coefficient_checker.py'):
        tree = ast.parse(_regular_bytes(REFERENCE / name).decode('utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name in permitted for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0 and node.module in permitted
            elif isinstance(node, ast.Constant):
                assert not isinstance(node.value, float)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'__import__', 'eval', 'exec', 'float', 'compile'}
    SCIENTIFIC_RECORDS.append({'test': 'registered_source_boundary',
                              'source_bindings_verified': len(SOURCE_BINDINGS),
                              'source_hash_negative_records': 2 * len(SOURCE_BINDINGS),
                              'mechanics_imports_absent': True,
                              'mechanical_mutation_coverage_claimed': False})


def _run_fixture(fixture_id):
    value = os.environ.get('BEAM_QUALIFICATION_OUTPUT')
    if not value:
        raise ValueError('reviewed bounded qualification output is required')
    out = Path(value)
    if not out.is_absolute():
        raise ValueError('qualification output must be absolute')
    info = out.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError('qualification output is not a non-reparse directory')
    lease = _strict(_regular_bytes(out / 'lease.json'))
    fixture_out = out / fixture_id
    fixture_out.mkdir(exist_ok=False)
    def progress(label):
        print('Q4 coefficient audit ' + fixture_id + ': ' + label, flush=True)
    proof = audit(fixture_id, progress)
    raw = _write_exclusive(fixture_out / 'proof.json', proof)
    proof_sha = sha256(raw).hexdigest()
    script = ROOT / 'scripts' / 'run_ge_beam3_qualification.py'
    processes, streams = [], []
    try:
        # Launch both fresh replicas before waiting for either. They inherit the
        # bounded parent Windows Job; no breakaway/new resource wrapper is used.
        for replica in (1, 2):
            target = fixture_out / ('checker' + str(replica) + '.json')
            if target.exists():
                raise ValueError('checker output already exists')
            stream = (fixture_out / ('checker' + str(replica) + '.log')).open('xb')
            streams.append(stream)
            command = [sys.executable, '-I', '-S', '-B', '-u', str(script),
                       '--q4-checker', str(out), fixture_id, str(replica), proof_sha]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                              stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 600
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise ValueError('independent checker process failed')
            if time.monotonic() >= deadline:
                raise TimeoutError('checker local wait ceiling exceeded')
            time.sleep(1)
        if any(process.returncode != 0 for process in processes):
            raise ValueError('independent checker nonzero exit')
    finally:
        # On test failure, drain the two children we created. The outer Job owns
        # all descendants and is the authoritative whole-tree resource guard.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=10)
        for stream in streams:
            stream.close()
    first = _regular_bytes(fixture_out / 'checker1.json')
    second = _regular_bytes(fixture_out / 'checker2.json')
    if first != second:
        raise ValueError('checker replicas are not byte-identical')
    wrapper = _strict(first)
    if set(wrapper) != {'schema', 'candidate', 'inputs_sha256', 'proof_sha256', 'verification'}:
        raise ValueError('checker wrapper schema keys differ')
    if (wrapper['schema'] != 'GE_BEAM3_Q4_CHECKER_RESULT_V1'
            or _canonical(wrapper['candidate']) != _canonical(lease['candidate'])
            or wrapper['inputs_sha256'] != sha256(_canonical(lease['inputs'])).hexdigest()
            or wrapper['proof_sha256'] != proof_sha):
        raise ValueError('checker wrapper authority mismatch')
    expected_verification = {
        'fixture_id': fixture_id, 'coefficient_count': 20150,
        'nonzero_count': proof['nonzero_count'], 'first_nonzero': proof['first_nonzero'],
        'independently_verified': True, 'recovery_qualified': False, 'full_g3c_qualified': False,
    }
    if _canonical(wrapper['verification']) != _canonical(expected_verification):
        raise ValueError('checker verification scope or result mismatch')
    if _regular_bytes(fixture_out / 'proof.json') != raw:
        raise ValueError('proof changed during independent checking')
    SCIENTIFIC_RECORDS.append({'test': fixture_id, 'proof': proof,
                              'verification': wrapper['verification'],
                              'checker_replicas_byte_identical': True})


def test_square_coefficient_proof():
    _run_fixture('MO16_SQUARE_EXACT')


def test_rhombus_coefficient_proof():
    _run_fixture('MO16_AFFINE_RHOMBUS_EXACT')


def test_assembly_and_proof_mutations(monkeypatch):
    from ge_beam3_q4_recovery_coefficient_checker import reconstruct, verify_against_reconstruction
    import ge_beam3_q4_recovery_coefficient_producer as producer
    fixture_id = 'MO16_SQUARE_EXACT'
    def progress(label):
        print('Q4 mutation audit: ' + label, flush=True)
    expected = reconstruct(fixture_id, progress)
    expected_raw = _canonical(expected)
    expected_sha = sha256(expected_raw).hexdigest()
    baseline = producer.audit(fixture_id, progress)
    verification = verify_against_reconstruction(baseline, _strict(expected_raw))
    assert verification['independently_verified'] is True
    original_frames = producer._frames
    original_engineering = producer._engineering_transform
    original_geometry = producer._geometry
    original_compatible = producer._compatible
    original_nonlinear = producer._nonlinear_coefficients

    def changed_frame(nodes):
        mixed, inherited = original_frames(nodes)
        # Rotate the reported mixed basis by +90 degrees, keeping the physical
        # director and orthogonality but violating frozen equation-7 authority.
        rotated = [[row[1], -row[0], row[2]] for row in mixed]
        return rotated, inherited
    def changed_engineering(rotation):
        value = original_engineering(rotation)
        value[2] = [2*x for x in value[2]]
        return value
    def changed_weight(local, r, s):
        shape, dx, dy, determinant, jacobian = original_geometry(local, r, s)
        return shape, dx, dy, 2*determinant, jacobian
    def changed_coupling(local, r, s):
        return [[-x for x in row] for row in original_compatible(local, r, s)]
    def changed_nonlinear(dx, dy):
        return [[2*x for x in row] for row in original_nonlinear(dx, dy)]

    assembly_cases = (
        ('geometry', None, ((-2,-1,0),(2,-1,0),(2,1,0),(-2,1,0))),
        ('frame', '_frames', changed_frame),
        ('engineering_shear', '_engineering_transform', changed_engineering),
        ('quadrature_weight', '_geometry', changed_weight),
        ('stationary_coupling_sign', '_compatible', changed_coupling),
        ('nonlinear_compatibility', '_nonlinear_coefficients', changed_nonlinear),
    )
    assembly_results = []
    for name, helper, replacement in assembly_cases:
        progress('assembly_mutation_' + name)
        with monkeypatch.context() as patch:
            if helper is None:
                patch.setitem(producer.FIXTURES, fixture_id, replacement)
            else:
                patch.setattr(producer, helper, replacement)
            try:
                altered = producer.audit(fixture_id, progress)
            except (ValueError, ArithmeticError) as error:
                # Source-identity guards are an acceptable fail-closed result;
                # unexpected process/runtime errors are not swallowed here.
                outcome = {'id': name, 'rejection': 'ASSEMBLY_GUARD',
                           'exception_type': type(error).__name__}
            else:
                with pytest.raises(ValueError, match='independent proof disagreement'):
                    verify_against_reconstruction(altered, _strict(expected_raw))
                outcome = {'id': name, 'rejection': 'INDEPENDENT_CHECKER'}
        assert _canonical(expected) == expected_raw
        assert sha256(expected_raw).hexdigest() == expected_sha
        assembly_results.append(outcome)

    def increment(values):
        values[0] = str(Fraction(values[0]) + 1)
    proof_cases = (
        ('coefficient', lambda p: increment(p['coefficient_records'][0]['coefficient'])),
        ('linear_operator', lambda p: increment(p['linear_physical_operator'][0][0])),
        ('frame_metadata', lambda p: increment(p['frames']['mixed'][0][0])),
        ('station_metadata', lambda p: increment(p['stations'][0]['weight'])),
        ('claimed_check', lambda p: p['exact_checks'].__setitem__('stationary_solve_residual', False)),
        ('zero_nonzero_count', lambda p: p.__setitem__('nonzero_count', p['nonzero_count'] + 1)),
        ('missing_coefficient', lambda p: p['coefficient_records'].pop()),
        ('extra_schema_key', lambda p: p.__setitem__('invented_qualification', True)),
    )
    proof_results = []
    for name, mutate in proof_cases:
        altered = _strict(expected_raw)
        mutate(altered)
        with pytest.raises(ValueError, match='independent proof disagreement'):
            verify_against_reconstruction(altered, _strict(expected_raw))
        assert _canonical(expected) == expected_raw
        proof_results.append({'id': name, 'rejection': 'INDEPENDENT_CHECKER'})
    SCIENTIFIC_RECORDS.append({
        'test': 'assembly_and_proof_mutations', 'fixture_id': fixture_id,
        'independent_baseline_sha256': expected_sha,
        'positive_baseline_verified': True,
        'assembly_mutations': assembly_results, 'assembly_mutation_count': len(assembly_results),
        'proof_mutations': proof_results, 'proof_mutation_count': len(proof_results),
        'independent_reconstructions': 1,
        'baseline_immutable': True,
        'recovery_qualified': False, 'full_g3c_qualified': False,
    })
