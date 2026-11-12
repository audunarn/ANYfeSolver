"""Release metadata and explicit portable/extended GE test boundary."""
import ast
import json
from pathlib import Path
import tomllib
from hashlib import sha256
from zipfile import ZipFile

import pytest
from scripts import run_portable_ci as ci
from scripts import verify_release_runtime as bridge

ROOT = Path(__file__).resolve().parents[1]


def test_release_043_history_is_preserved_after_successor_release():
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    tree = ast.parse((ROOT / 'src/anyfesolver/__init__.py').read_text())
    version = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                   and t.id == '__version__' for t in n.targets))
    assert metadata['project']['name'] == 'ANYfeSolver'
    assert metadata['project']['version'] == version == '0.5.0'
    assert json.loads((ROOT / 'dependency-licenses.json').read_text())['release'] == version
    assert '## 0.4.3 - 2026-09-09' in (ROOT / 'CHANGELOG.md').read_text()
    assert (ROOT / 'scripts/release_043_runtime.json').is_file()


def test_successor_source_distribution_excludes_historical_release_harnesses():
    manifest = (ROOT / 'MANIFEST.in').read_text().splitlines()
    assert 'prune scripts' in manifest
    assert 'prune tests' in manifest
    assert 'include scripts/release_043_runtime.json' not in manifest


def test_ge_inventory_covers_every_module_and_keeps_runtime_tests():
    record = ci._ge_beam3_inventory()
    modules = ci.merge_test_modules()
    assert record['portable']
    assert set(record['portable']) <= set(modules)
    assert not set(record['extended']) & set(modules)
    assert 'tests/test_ge_beam3_public_workflows.py' in modules
    assert 'tests/test_ge_beam3_durable_coupled.py' in modules


def test_runtime_protocol_checks_are_process_isolated_from_exact_guards():
    modules = (
        'tests/test_generalized_shell_sections.py',
        'tests/test_plane_stress_analytical_tangent.py',
    )
    partitions = ci.execution_partitions(modules, 2)
    assert ('tests/test_generalized_shell_sections.py',) in partitions
    assert all(
        not (
            'tests/test_generalized_shell_sections.py' in partition
            and 'tests/test_plane_stress_analytical_tangent.py' in partition
        )
        for partition in partitions
    )


@pytest.mark.parametrize('mutation', ('new', 'missing', 'duplicate'))
def test_unknown_or_incomplete_ge_inventory_fails_closed(tmp_path, monkeypatch, mutation):
    record = ci._ge_beam3_inventory()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'scripts').mkdir()
    for name in record['portable'] + record['extended']:
        (tmp_path / name).touch()
    if mutation == 'new':
        (tmp_path / 'tests/test_ge_beam3_new_unclassified.py').touch()
    elif mutation == 'missing':
        (tmp_path / record['portable'][0]).unlink()
    else:
        record['extended'].append(record['portable'][0])
    (tmp_path / 'scripts/ge_beam3_release_test_inventory.json').write_text(json.dumps(record))
    monkeypatch.setattr(ci, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='incomplete or overlapping'):
        ci.merge_test_modules()


@pytest.mark.parametrize('fault', ('none', 'hash', 'mechanics', 'path', 'version'))
def test_release_bridge_rejects_more_than_version_delta(tmp_path, monkeypatch, fault):
    old, new = tmp_path / 'old.whl', tmp_path / 'new.whl'
    for path, version in ((old, '0.4.2'), (new, '0.4.3')):
        with ZipFile(path, 'w') as z:
            z.writestr('anysolver/__init__.py', '__version__ = "'+version+'"\n')
            z.writestr('anysolver/core.py', 'a = 2\n' if path == new and fault == 'mechanics' else 'a = 1\n')
            if path == new and fault == 'path':
                z.writestr('anysolver/extra.py', '')
            z.writestr('anysolver.dist-info/METADATA', 'Name: ANYsolver\nVersion: '+
                       ('0.4.4' if fault == 'version' else version)+'\n')
    monkeypatch.setattr(bridge, 'ACCEPTED_SHA256', '0'*64 if fault == 'hash' else sha256(old.read_bytes()).hexdigest())
    if fault == 'none':
        assert bridge.compare(old, new)['version_only'] is True
    else:
        with pytest.raises(ValueError):
            bridge.compare(old, new)


def test_publication_manifest_checks_all_accepted_runtime_files():
    """Preserve the immutable 0.4.3 authority without replaying current source.

    The 0.4.3 manifest predates the additive G7 runtime.  Reconstructing its
    wheel from a successor checkout would mix authorities, so successor release
    integrity is covered by ``test_release_046_bounded.py`` instead.
    """
    manifest = ROOT / 'scripts/release_043_runtime.json'
    record = json.loads(manifest.read_text())
    assert sha256(manifest.read_bytes()).hexdigest() == (
        '009b9cf1c664da1b6f7b2d450556d3e63adada3c05588c871e023b1350224945'
    )
    assert record['accepted_commit'] == '5fc032e48d25c0a0b866363514b73ac7baf8803c'
    assert record['accepted_wheel_sha256'] == bridge.ACCEPTED_SHA256
    assert len(record['runtime']) == 317
