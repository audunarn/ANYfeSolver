"""Static signed-mode implementation/evidence and copy-equivalence checks."""

import ast
import hashlib
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def pairs(rows):
    result = {}
    for key, value in rows:
        if key in result: raise ValueError('duplicate key')
        result[key] = value
    return result


def reject(_): raise ValueError('nonfinite value')


def evidence():
    raw = (ROOT/'docs/reference_cases/ge_beam3_signed_modes_evidence.json').read_text(encoding='utf-8')
    record = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    assert json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n' == raw
    return record


def test_source_and_baseline_bindings():
    record = evidence()
    assert record['baseline'] == dict(commit='2356029a52ead7d131155a4cb1c9f467865a72a7',
        tree='06113052c775a973c7dba413c2b54731d9788b4b')
    assert record['source_normalization'] == 'UTF8_LF' and len(record['sources']) == 5
    for row in record['sources']:
        p = (ROOT/row['path']).resolve(); assert p.is_relative_to(ROOT)
        raw = p.read_text(encoding='utf-8').encode('utf-8')
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']


def test_scope_and_separate_inventories():
    record = evidence()
    assert record['scope'] == dict(production_qualified=False, independent_review='PENDING',
        previous_sources_changed=False, defaults_changed=False, constitutive_laws_changed=False,
        public_route_added=False, installed_wheel_checked=False, certified_intervals=False,
        resource_requests_consumed=0, full_prestressed_modal_qualification=False)
    assert record['checks'] == dict(kernel_tests=14, slender_native_mode_tests=2,
        loaded_adapter_tests=11, preserved_loaded_modal_tests=23,
        preserved_signed_frequency_tests=7, preserved_signed_algebra_tests=11,
        final_suite_passed=True, repeated_new_suite_passed=True, six_rigid_modes_passed=True,
        mode_covariance_passed=True, unchanged_state_passed=True)


class StripCheckpoint(ast.NodeTransformer):
    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'checkpoint'): return None
        return self.generic_visit(node)


def normalized_function(path, name):
    tree = ast.parse((ROOT/path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node = StripCheckpoint().visit(node); node.name = 'same_function'
    node.args.args = [a for a in node.args.args if a.arg != 'checkpoint']
    return ast.dump(node, include_attributes=False)


def test_native_split_matches_preserved_research_expression_except_checkpoints():
    assert normalized_function('src/anyfesolver/_ge_beam3_signed_loaded_modes.py', '_split') == normalized_function(
        'docs/reference_cases/ge_beam3_loaded_factor_split_probe.py', 'split_accepted_elastic_operator')
    assert normalized_function('src/anyfesolver/_native_signed_factor_modes.py', '_reduce') == normalized_function(
        'docs/reference_cases/ge_beam3_signed_spectrum_probe.py', 'reduce_signed_split')


def test_private_source_has_no_research_import_or_public_export():
    names = ('_native_signed_factor_modes', '_ge_beam3_signed_loaded_modes')
    for name in names:
        tree = ast.parse((ROOT/'src/anyfesolver'/f'{name}.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('docs', 'tests'))
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(('docs', 'tests')) for n in node.names)
        assert name not in (ROOT/'src/anyfesolver/__init__.py').read_text(encoding='utf-8')


def test_archive_pairs_and_failed_sources_are_bound():
    record = evidence(); archive = record['archive']; rows = {r['path']: r for r in archive['files']}
    assert len(rows) == len(archive['files']) == archive['file_count']
    assert sum(r['bytes'] for r in rows.values()) == archive['bytes']
    assert len(archive['identical_pairs']) == 4
    for a, b in archive['identical_pairs']:
        assert a.startswith('accepted/') and b == a.replace('accepted/', 'repeat/', 1)
        assert rows[a]['sha256'] == rows[b]['sha256'] and rows[a]['bytes'] == rows[b]['bytes']
    for incident in record['incidents']:
        assert rows[incident['source_path']]['sha256'] == incident['source_sha256']
        assert incident['failed_tests'] >= 1


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_ambiguous_evidence_rejected(raw):
    with pytest.raises(ValueError): json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
