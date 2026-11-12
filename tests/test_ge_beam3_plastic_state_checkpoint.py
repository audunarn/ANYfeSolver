"""Static paired-plastic state evidence audit; no mechanical load rerun."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
from test_ge_beam3_station_resultant_checkpoint import strict, sha

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'docs/reference_cases/ge_beam3_plastic_state_development_evidence.json'


def test_canonical_record_preserves_development_scope():
    value = strict(RECORD.read_bytes())
    assert value['production_qualified'] is False and value['independent_review'] == 'PENDING'
    assert value['disposition'] == 'DEVELOPMENT_MODEL_BOUND_PLASTIC_RESTART_PASS_NOT_QUALIFICATION'
    assert value['state_schema'] == 'GE_BEAM3_RETAINED_PAIRED_PLASTIC_ACCEPTED_CHAIN_V1'
    assert value['scope']['equilibrium_compatibility_tolerance'] == '1e-11'
    assert value['scope']['checkpoint_bytes'] == [69006, 80274]
    assert len(value['remaining']) == 5


@pytest.mark.parametrize('raw', [b'{"a":0,"a":1}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n', b'{ "a":0}\n'])
def test_evidence_parser_is_strict(raw):
    with pytest.raises(ValueError): strict(raw)


def test_bound_sources_have_no_runtime_research_dependency():
    value = strict(RECORD.read_bytes())
    for entry in value['sources']:
        raw = (ROOT/entry['path']).read_bytes().replace(b'\r\n', b'\n')
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256']
        if entry['path'].startswith('src/'):
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Import): imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom): imports = [node.module]
                else: continue
                assert all(name.split('.')[0] in sys.stdlib_module_names | {'anyfesolver', 'numpy'} for name in imports)


def test_external_archive_and_cycle_equality():
    value = strict(RECORD.read_bytes()); root = Path(value['archive']['path'])
    if not root.exists(): pytest.skip('explicit external paired-plastic state archive required')
    raw = (root/'archive-manifest.json').read_bytes()
    assert len(raw) == value['archive']['manifest_bytes'] and sha(raw) == value['archive']['manifest_sha256']
    manifest = json.loads(raw)
    assert len(manifest['files']) == 24 and sum(row['bytes'] for row in manifest['files']) == 987356
    for row in manifest['files']:
        raw = (root/row['path']).read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    for run in value['inventories']:
        suite = ET.parse(root/run['lane']/'junit.xml').getroot().find('testsuite')
        assert suite is not None
        for key in ('tests', 'errors', 'failures', 'skipped'): assert int(suite.attrib[key]) == run[key]
        assert suite.attrib['time'] == run['seconds']
    assert len(value['deterministic_pairs']) == 6
    for row in value['deterministic_pairs']:
        a = (root/'cycle-a'/row['path']).read_bytes(); b = (root/'cycle-b'/row['path']).read_bytes()
        assert a == b and len(a) == row['bytes'] and sha(a) == row['sha256']


def test_exact_six_new_paths_and_no_existing_mechanics_delta():
    value = strict(RECORD.read_bytes())
    def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).splitlines()
    assert git('rev-parse', value['base_commit']+'^{tree}') == [value['base_tree']]
    changed = set(git('diff', '--name-only', value['base_commit']))
    changed.update(git('ls-files', '--others', '--exclude-standard', '--', 'docs/', 'src/', 'tests/'))
    expected = {row['path'] for row in value['sources']} | {
        'docs/reference_cases/ge_beam3_plastic_state_development_evidence.json',
        'docs/agent_plans/GE_BEAM3_PAIRED_PLASTIC_STATE_DEVELOPMENT_CHECKPOINT.md',
        'tests/test_ge_beam3_plastic_state_checkpoint.py'}
    assert changed == expected
    assert git('diff', '--name-only', '--diff-filter=MDR', value['base_commit'], '--', 'src/') == []
