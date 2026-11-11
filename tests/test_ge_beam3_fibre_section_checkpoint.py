"""Static preservation checks for an unqualified development component."""
import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/reference_cases/ge_beam3_fibre_section_development_evidence.json'


def read():
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise ValueError('duplicate development evidence key')
            out[key] = value
        return out
    def finite(value): raise ValueError('nonfinite development evidence')
    raw = EVIDENCE.read_bytes()
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    assert raw == (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
    return value


def test_exact_development_disposition_and_inventory():
    value = read()
    assert set(value) == {'schema', 'parent_commit', 'parent_tree', 'candidate', 'status', 'independent_review',
        'production_qualified', 'beam_adapter_implemented', 'default_changes', 'resource_requests_consumed',
        'archive', 'sources', 'inventories', 'identical_pairs', 'limits'}
    assert value['schema'] == 'GE_BEAM3_FIBRE_SECTION_DEVELOPMENT_CHECKPOINT_V1'
    assert value['status'] == 'DEVELOPMENT_SECTION_CHECKS_PASS_NOT_BEAM_QUALIFICATION'
    assert value['independent_review'] == 'PENDING'
    for key in ('production_qualified', 'beam_adapter_implemented', 'default_changes'): assert value[key] is False
    assert value['resource_requests_consumed'] == []
    assert [(x['tests'], x['failures'], x['errors'], x['skips']) for x in value['inventories']] == [
        (35, 0, 0, 0), (62, 0, 0, 0), (62, 0, 0, 0)]
    assert len(value['identical_pairs']) == 11
    assert len({x['path'] for x in value['identical_pairs']}) == 11
    assert value['archive']['content_files'] == 35
    assert value['archive']['manifest_sha256'] == '84bb929e1f820503640d63010d2069b841ad730a15c1ecff9594a10905cb782d'


@pytest.mark.parametrize('index', [0, 1, 2])
def test_final_tested_source_hashes(index):
    record = read()['sources'][index]
    raw = (ROOT/record['path']).read_bytes().replace(b'\r\n', b'\n')
    assert len(raw) == record['bytes']
    assert hashlib.sha256(raw).hexdigest() == record['sha256']


def test_new_component_has_no_legacy_or_research_mechanics_imports():
    tree = ast.parse((ROOT/'src/anyfesolver/_ge_beam3_fibre_section.py').read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add(node.module)
    assert imports == {'dataclasses', 'decimal', 'hashlib', 'json', 'math', 'numbers', 'time', 'numpy',
                       '_ge_beam3_station_resultant_cell', 'anymaterial.curves'}
    # The reused module supplies Decimal arithmetic only, not the fibre mechanics.
    arithmetic = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                      and n.module == '_ge_beam3_station_resultant_cell')
    assert {alias.name for alias in arithmetic.names} == {'cholesky', 'history_row', 'pair'}
