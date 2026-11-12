"""Static source/evidence boundary for the private reassembly checkpoint."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).parents[1]
RECORD = 'docs/reference_cases/ge_beam3_reassembled_modes_development_evidence.json'
EXTRA = {RECORD, 'docs/agent_plans/GE_BEAM3_REASSEMBLED_MODES_CHECKPOINT.md',
         'tests/test_ge_beam3_reassembled_modes_checkpoint.py'}


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def parse(raw):
    def nonfinite(value):
        raise ValueError('nonfinite')
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if raw != (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii'):
        raise ValueError('noncanonical')
    return value


def read():
    return parse((ROOT / RECORD).read_bytes().replace(b'\r\n', b'\n'))


def test_schema_and_no_qualification_claim():
    p = read()
    assert set(p) == {'schema', 'status', 'production_qualified', 'public_routing_changed',
        'independent_review', 'formal_resource_requests_consumed', 'base_commit', 'base_tree',
        'source_encoding', 'sources', 'archive', 'tests', 'limits', 'preserved_failures', 'remaining', 'notes'}
    assert p['schema'] == 'GE_BEAM3_REASSEMBLED_MODES_DEVELOPMENT_CHECKPOINT_V1'
    assert p['status'] == 'DEVELOPMENT_CHECKS_PASS_QUALIFICATION_INCOMPLETE'
    assert p['production_qualified'] is p['public_routing_changed'] is False
    assert p['independent_review'] == 'PENDING'
    assert p['formal_resource_requests_consumed'] == [] and len(p['remaining']) == 7
    assert p['limits']['certified_intervals'] is False
    assert p['limits']['hard_process_watchdog'] is False
    assert p['limits']['relative_and_absolute_curved_covariance_tolerance'] == 1e-11


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}\n', b'{"x":NaN}\n',
    b'{"x":Infinity}\n', b'{"x":-Infinity}\n', b'{"x": 1}\n'])
def test_noncanonical_or_ambiguous_evidence_rejected(raw):
    with pytest.raises(ValueError):
        parse(raw)


def test_eight_bound_sources_match_normalized_bytes():
    rows = read()['sources']
    assert len(rows) == len({r['path'] for r in rows}) == 8
    for row in rows:
        data = (ROOT / row['path']).read_bytes().replace(b'\r\n', b'\n')
        assert len(data) == row['bytes'], row['path']
        assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']


def test_external_inventory_and_two_matching_run_bindings():
    p = read(); archive = p['archive']; rows = archive['artifacts']
    assert len(rows) == len({r['path'] for r in rows}) == archive['file_count'] == 134
    assert sum(r['bytes'] for r in rows) == archive['bytes'] == 3545317
    assert archive['ordinary_context_hash_verified'] is True
    indexed = {r['path']: r for r in rows}
    for row in rows:
        assert '..' not in Path(row['path']).parts and not Path(row['path']).is_absolute()
        assert row['bytes'] > 0 and len(row['sha256']) == 64
    assert len([r for r in rows if r['path'].endswith('.py')]) == 2
    tests = p['tests']; assert sum(r['passed'] for r in tests['inventories']) == 46
    assert len(tests['runs']) == 2 and len(tests['deterministic_json_pairs']) == 23
    roots = []
    for run in tests['runs']:
        assert (run['passed'], run['failed'], run['errors'], run['skipped']) == (46, 0, 0, 0)
        assert run['junit'] in indexed
        roots.append('runs/'+Path(run['junit']).name.removesuffix('-junit.xml')+'/')
    for pair in tests['deterministic_json_pairs']:
        for root in roots:
            bound = indexed[root+pair['path']]
            assert (bound['bytes'], bound['sha256']) == (pair['bytes'], pair['sha256'])


def test_exact_additive_extent_without_existing_mechanics_changes():
    p = read()
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=30).stdout
    assert git('show', '-s', '--format=%T', p['base_commit']).strip() == p['base_tree']
    rows = [line.split('\t') for line in git('diff', '--name-status', p['base_commit'], '--').splitlines()]
    assert all(kind == 'A' for kind, _ in rows)
    untracked = set(git('ls-files', '--others', '--exclude-standard').splitlines())
    assert {path for _, path in rows} | untracked == {r['path'] for r in p['sources']} | EXTRA


def test_adapter_imports_preserved_mechanics_split_without_redefinition():
    path = ROOT / 'src/anyfesolver/_ge_beam3_reassembled_signed_modes.py'
    tree = ast.parse(path.read_text())
    assert any(isinstance(n, ast.ImportFrom) and n.module == '_ge_beam3_seeded_signed_modes'
        and [a.name for a in n.names] == ['_split'] for n in ast.walk(tree))
    assert not any(isinstance(n, ast.FunctionDef) and n.name == '_split' for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ('eval', 'exec', '__import__') for n in ast.walk(tree))


def test_historical_covariance_failure_test_is_not_modified_or_relaxed():
    p = read(); path = 'tests/test_ge_beam3_curved_contrast_probe.py'
    frozen = subprocess.run(['git', 'show', p['base_commit']+':'+path], cwd=ROOT,
        capture_output=True, check=True, timeout=30).stdout.replace(b'\r\n', b'\n')
    assert (ROOT/path).read_bytes().replace(b'\r\n', b'\n') == frozen
    tree = ast.parse(frozen)
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('skip', 'skipif', 'xfail') for n in ast.walk(tree))


def test_result_types_keep_private_scope_flags():
    from anyfesolver._native_reassembled_factor_modes import ReassembledFactorModes
    from anyfesolver._ge_beam3_reassembled_signed_modes import SignedLoadedModes
    for cls in (ReassembledFactorModes, SignedLoadedModes):
        assert cls.__dataclass_fields__['production_qualified'].default is False
        assert cls.__dataclass_fields__['certified_intervals'].default is False
        assert cls.__dataclass_fields__['negative_eigenvalues_retained'].default is True
    assert SignedLoadedModes.__dataclass_fields__['buckling_factor_authorized'].default is False
