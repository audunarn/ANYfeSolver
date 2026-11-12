"""Static evidence checks; no assembled beam or qualification claim."""
import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read():
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate development evidence key')
            result[key] = value
        return result
    def finite(value): raise ValueError('nonfinite development evidence')
    # Normalize only Git working-copy line endings, not live execution input.
    raw = (ROOT/'docs/reference_cases/ge_beam3_fibre_cell_development_evidence.json').read_bytes().replace(b'\r\n', b'\n')
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    assert raw == (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
    return value


def test_disposition_and_preserved_failed_inventories():
    value = read()
    assert set(value) == {'schema', 'parent_commit', 'parent_tree', 'candidate', 'status', 'independent_review',
        'production_qualified', 'beam_adapter_implemented', 'default_changes', 'resource_requests_consumed',
        'archive', 'sources', 'inventories', 'identical_pairs', 'limits', 'preserved_incidents',
        'maximum_observed_original_equation_error', 'maximum_observed_newton_updates'}
    assert value['schema'] == 'GE_BEAM3_FIBRE_CELL_DEVELOPMENT_CHECKPOINT_V1'
    assert value['status'] == 'DEVELOPMENT_COUPLED_CELL_CHECKS_PASS_NOT_BEAM_QUALIFICATION'
    assert value['independent_review'] == 'PENDING'
    for key in ('production_qualified', 'beam_adapter_implemented', 'default_changes'): assert value[key] is False
    assert value['resource_requests_consumed'] == []
    assert [(x['tests'], x['failures'], x['errors'], x['skips']) for x in value['inventories']] == [
        (23, 23, 0, 0), (23, 10, 0, 0), (23, 0, 0, 0), (93, 0, 0, 0),
        (95, 0, 0, 0), (95, 0, 0, 0), (97, 0, 0, 0), (97, 0, 0, 0)]
    assert len(value['identical_pairs']) == len({x['path'] for x in value['identical_pairs']}) == 23
    assert value['maximum_observed_newton_updates'] == 15
    assert value['archive']['content_files'] == 143
    assert value['archive']['manifest_sha256'] == '2c648de68bd31c8c88ff67e4c725872ac678d3370775d4e32bcec734064065f6'


@pytest.mark.parametrize('index', [0, 1, 2])
def test_final_tested_source_hashes(index):
    item = read()['sources'][index]
    raw = (ROOT/item['path']).read_bytes().replace(b'\r\n', b'\n')
    assert len(raw) == item['bytes']
    assert hashlib.sha256(raw).hexdigest() == item['sha256']


def test_runtime_imports_do_not_include_research_or_legacy_mechanics():
    tree = ast.parse((ROOT/'src/anyfesolver/_ge_beam3_fibre_cell.py').read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add(node.module)
    assert imports == {'dataclasses', 'decimal', 'hashlib', 'time', '_ge_beam3_fibre_section',
                       '_ge_beam3_station_resultant_cell'}
    arithmetic = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                      and n.module == '_ge_beam3_station_resultant_cell')
    assert 'StationResultantCell' not in {alias.name for alias in arithmetic.names}
