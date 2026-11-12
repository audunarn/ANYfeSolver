"""Static checkpoint checks, not independent review or full beam qualification."""
import ast
import hashlib
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def read():
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise ValueError('duplicate development key')
            out[key] = value
        return out
    def finite(value): raise ValueError('nonfinite development value')
    raw = (ROOT/'docs/reference_cases/ge_beam3_retained_fibre_development_evidence.json').read_bytes().replace(b'\r\n', b'\n')
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    assert raw == (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
    return value


def test_development_disposition_and_inventories():
    value = read()
    assert set(value) == {'schema', 'parent_commit', 'parent_tree', 'candidate', 'status', 'independent_review',
        'production_qualified', 'public_routing_changed', 'default_changes', 'resource_requests_consumed',
        'archive', 'sources', 'inventories', 'identical_pairs', 'limits', 'coverage', 'state_schema',
        'maximum_accepted_metric', 'maximum_observed_newton_updates', 'maximum_observed_checkpoint_bytes'}
    assert value['status'] == 'DEVELOPMENT_FEMODEL_FIBRE_STATE_CHECKS_PASS_NOT_QUALIFICATION'
    assert value['independent_review'] == 'PENDING'
    for key in ('production_qualified', 'public_routing_changed', 'default_changes'): assert value[key] is False
    assert value['resource_requests_consumed'] == []
    assert [(row['tests'], row['failures'], row['errors'], row['skips']) for row in value['inventories']] == [
        (11, 0, 0, 0), (2, 0, 0, 0), (30, 0, 0, 0), (50, 0, 0, 0), (54, 0, 0, 0), (54, 0, 0, 0)]
    assert len(value['identical_pairs']) == len({row['path'] for row in value['identical_pairs']}) == 13
    assert value['maximum_observed_newton_updates'] == 5
    assert value['maximum_observed_checkpoint_bytes'] == 135411
    assert value['archive']['manifest_sha256'] == '12524faf14b5f890a9a02444c55a7066f100dd2f17934bfb081a2a936f0650a8'


@pytest.mark.parametrize('index', list(range(6)))
def test_final_tested_source_hashes(index):
    item = read()['sources'][index]
    raw = (ROOT/item['path']).read_bytes().replace(b'\r\n', b'\n')
    assert len(raw) == item['bytes']
    assert hashlib.sha256(raw).hexdigest() == item['sha256']


def test_native_runtime_has_no_research_or_legacy_beam_mechanics_dependency():
    for record in read()['sources']:
        if not record['path'].startswith('src/'): continue
        tree = ast.parse((ROOT/record['path']).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): imports.append(node.module)
        assert not any(name.startswith(('docs.', 'tests.')) for name in imports)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names & {'QuadraticBeamElement', 'RetainedPlasticOperator', 'RetainedElasticOperator'}
    tree = ast.parse((ROOT/'src/anyfesolver/_ge_beam3_retained_fibre_state.py').read_text())
    layout = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Layout')
    reused = [node.value.attr for node in layout.body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
              and node.value.value.id == 'GeometricOperations']
    assert reused == ['make', 'advance'] and layout.bases == []
