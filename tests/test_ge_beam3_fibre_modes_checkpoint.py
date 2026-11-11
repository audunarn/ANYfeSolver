"""Static development preservation checks, not independent scientific review."""
import ast
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def record():
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise ValueError('duplicate checkpoint key')
            out[key] = value
        return out
    def finite(value): raise ValueError('nonfinite checkpoint value')
    raw = (ROOT/'docs/reference_cases/ge_beam3_fibre_modes_development_evidence.json').read_bytes().replace(b'\r\n', b'\n')
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    assert raw == (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
    return value


def test_checkpoint_is_development_only_and_failures_are_preserved():
    value = record()
    assert value['schema'] == 'GE_BEAM3_FIBRE_CURRENT_STATE_SPECTRA_DEVELOPMENT_V1'
    assert value['status'] == 'DEVELOPMENT_CURRENT_STATE_FIBRE_SPECTRA_PASS_NOT_QUALIFICATION'
    assert value['parent_commit'] == 'fbc66e6c79e48f5014c5c106c1c63cb9fc17c5fd'
    assert value['independent_review'] == 'PENDING'
    assert value['production_qualified'] is False and value['public_routing_changed'] is False
    assert value['buckling_factor_authorized'] is False and value['state_advanced'] is False
    assert value['resource_requests_consumed'] == []
    assert [(row['tests'], row['failures'], row['errors']) for row in value['inventories']] == [
        (4, 2, 0), (30, 0, 0), (36, 2, 0), (36, 0, 0), (36, 0, 0)]
    assert len(value['identical_pairs']) == len({row['path'] for row in value['identical_pairs']}) == 18
    assert len(value['incidents']) == 2
    assert float(value['maximum_covariance_error']) <= 1e-11


@pytest.mark.parametrize('index', [0, 1])
def test_final_tested_source_hashes(index):
    row = record()['sources'][index]
    raw = (ROOT/row['path']).read_bytes().replace(b'\r\n', b'\n')
    assert len(raw) == row['bytes']
    assert hashlib.sha256(raw).hexdigest() == row['sha256']


def test_native_fibre_spectra_does_not_import_old_material_or_research_operators():
    tree = ast.parse((ROOT/'src/anyfesolver/_ge_beam3_retained_fibre_modes.py').read_text())
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(('docs.', 'tests.')) for name in names)
    assert not any('retained_plastic' in name or 'retained_elastic' in name for name in names)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'solve_modes')
    assert function.args.kwonlyargs[0].arg == 'material_policy' and function.args.kw_defaults[0] is None
