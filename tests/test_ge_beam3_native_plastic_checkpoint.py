"""Static native-plastic development evidence audit; no mechanics rerun."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
from test_ge_beam3_station_resultant_checkpoint import strict, canonical, sha

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'docs/reference_cases/ge_beam3_native_plastic_development_evidence.json'


def test_canonical_development_disposition_preserves_scope():
    value = strict(RECORD.read_bytes())
    assert value['production_qualified'] is False and value['independent_review'] == 'PENDING'
    assert value['formulation_id'] == 'CANDIDATE_GE_BEAM3_RETAINED_DIRECTED_PLASTIC_V1'
    assert value['scope']['force_and_compatibility_tolerance'] == '1e-11'
    assert value['scope']['directional_tolerance'] == '1e-7'
    assert value['scope']['base_macros'] == 2
    assert len(value['incidents']) == 2 and len(value['remaining']) == 5


@pytest.mark.parametrize('raw', [b'{"x":0,"x":1}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n', b'{ "x":0}\n'])
def test_strict_evidence_rejects_duplicate_nonfinite_and_noncanonical(raw):
    with pytest.raises(ValueError): strict(raw)


def test_bound_source_hashes_and_private_runtime_import_boundary():
    record = strict(RECORD.read_bytes())
    for entry in record['sources']:
        path = ROOT/entry['path']; raw = path.read_bytes().replace(b'\r\n', b'\n')
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256']
        if entry['path'].startswith('src/'):
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Import): imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom): imports = [node.module]
                else: continue
                assert all(name.split('.')[0] in sys.stdlib_module_names | {'anyfesolver', 'numpy'} for name in imports)


def test_exact_external_archive_inventories_and_deterministic_outputs():
    record = strict(RECORD.read_bytes()); root = Path(record['archive']['path'])
    if not root.exists(): pytest.skip('explicit external native-plastic archive required')
    raw = (root/'archive-manifest.json').read_bytes()
    assert len(raw) == record['archive']['manifest_bytes'] and sha(raw) == record['archive']['manifest_sha256']
    manifest = json.loads(raw)
    assert len(manifest['files']) == 70 and sum(x['bytes'] for x in manifest['files']) == 4345071
    for entry in manifest['files']:
        content = (root/entry['path']).read_bytes()
        assert len(content) == entry['bytes'] and sha(content) == entry['sha256']
    for inventory in record['inventories']:
        suite = ET.parse(root/inventory['lane']/'junit.xml').getroot().find('testsuite')
        assert suite is not None
        for key in ('tests', 'errors', 'failures', 'skipped'): assert int(suite.attrib[key]) == inventory[key]
        assert suite.attrib['time'] == inventory['seconds']
    assert len(record['deterministic_pairs']) == 18
    for entry in record['deterministic_pairs']:
        a = (root/'cycle-a'/entry['path']).read_bytes(); b = (root/'cycle-b'/entry['path']).read_bytes()
        assert a == b and len(a) == entry['bytes'] and sha(a) == entry['sha256']


def test_exact_eight_path_extent_and_existing_mechanics_unchanged():
    value = strict(RECORD.read_bytes())
    def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).splitlines()
    assert git('rev-parse', value['base_commit']+'^{tree}') == [value['base_tree']]
    changed = set(git('diff', '--name-only', value['base_commit']))
    changed.update(git('ls-files', '--others', '--exclude-standard', '--', 'docs/', 'src/', 'tests/'))
    expected = {x['path'] for x in value['sources']} | {
        'docs/reference_cases/ge_beam3_native_plastic_development_evidence.json',
        'docs/agent_plans/GE_BEAM3_NATIVE_PLASTIC_DEVELOPMENT_CHECKPOINT.md',
        'tests/test_ge_beam3_native_plastic_checkpoint.py'}
    assert changed == expected
    assert git('diff', '--name-only', '--diff-filter=MDR', value['base_commit'], '--', 'src/') == []
