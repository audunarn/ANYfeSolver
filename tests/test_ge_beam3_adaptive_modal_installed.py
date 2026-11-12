"""Offline fresh-venv correctness of private adaptive control and loaded modes."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_adaptive_modal_wheel_two_fresh_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    records = root/'docs/reference_cases'
    authority = json.loads((records/'ge_beam3_mixed_p3_package_execution_authority.json').read_text())
    wheelhouse = Path(os.environ.get('ANY_GE_BEAM3_TEST_WHEELHOUSE', authority['locations']['wheelhouse']))
    if not wheelhouse.is_dir(): pytest.skip('hash-bound offline dependency wheelhouse unavailable')
    for row in authority['wheelhouse']['files']:
        raw = (wheelhouse/row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest().upper() == row['sha256']
    # Bind source identities from preserved development records, not their old
    # resource requests. No historical scientific result is reclassified here.
    force = json.loads((records/'ge_beam3_force_program_development_evidence.json').read_text())
    native_raw = (records/'ge_beam3_native_load_source_map.json').read_text(encoding='utf-8').encode('utf-8')
    assert hashlib.sha256(native_raw).hexdigest() == force['preserved_native_source_map_sha256']
    outputs = {**json.loads(native_raw)['outputs'], **force['preserved_framework']}
    bindings = [force['source_bindings']]
    for name, key in (('displacement_program', 'source_bindings'), ('arc_geometry', 'bindings'),
                      ('arc_program', 'source_bindings'), ('adaptive_arc', 'source_bindings'), ('loaded_modal', 'source_bindings')):
        bindings.append(json.loads((records/f'ge_beam3_{name}_development_evidence.json').read_text())[key])
    for binding in bindings:
        for path, value in binding.items():
            if path.startswith('src/'):
                if path in outputs: assert outputs[path] == value
                outputs[path] = value
    for path, binding in outputs.items():
        raw = (root/path).read_text(encoding='utf-8').encode('utf-8')
        assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == binding

    def canonical(value):
        return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')

    source_map = tmp_path/'source-map.json'
    with source_map.open('xb') as stream:
        stream.write(canonical(dict(schema='GE_BEAM3_ADAPTIVE_MODAL_INSTALLED_SOURCE_MAP_V1',
            candidate_commit='d73b86bb3a1cc688c1776dd3ebae7bf74ff1d260',
            output_text_normalization='UTF8_LF', outputs=outputs, production_qualified=False)))
    build = tmp_path/'build'; build.mkdir()
    shutil.copytree(root/'src', build/'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    for path in outputs:
        item = build/path
        item.write_bytes(item.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'COPYRIGHT', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(root/name, build/name)
    wheels = tmp_path/'wheels'; wheels.mkdir()
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env.pop('PYTHONPATH', None)
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'; env['PIP_CONFIG_FILE'] = os.devnull

    def command(argv, cwd, name, limit=180):
        # Small correctness commands, not a benchmark or a formal resource wave.
        # Keep partial logs even on timeout, and terminate this child's tree.
        with (tmp_path/(name+'.stdout')).open('xb') as out, (tmp_path/(name+'.stderr')).open('xb') as err:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err,
                start_new_session=os.name != 'nt')
            try:
                code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                        capture_output=True, timeout=15, check=True)
                else:
                    import signal
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
                raise
        assert code == 0, ((tmp_path/(name+'.stdout')).read_text(errors='replace')+
                           (tmp_path/(name+'.stderr')).read_text(errors='replace'))

    command([sys.executable, '-I', '-B', '-c',
        'from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])', str(wheels)], build, 'build')
    candidates = list(wheels.glob('*.whl')); assert len(candidates) == 1
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        for name in ('_ge_beam3_adaptive_arc_program.py', '_ge_beam3_loaded_modal.py'):
            assert 'anyfesolver/'+name in archive.namelist()
        assert not any(name.startswith(('docs/', 'tests/')) for name in archive.namelist())
    environment = tmp_path/'environment'
    command([sys.executable, '-I', '-B', '-m', 'venv', str(environment)], tmp_path, 'venv')
    python = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command([str(python), '-I', '-B', '-m', 'pip', 'install', '--no-index', '--find-links',
        str(wheelhouse), str(wheel)], tmp_path, 'install')
    script = tmp_path/'installed_check.py'
    shutil.copy2(records/'ge_beam3_adaptive_modal_installed_smoke.py', script)
    results = []; checkpoints = []
    for index in (1, 2):
        directory = tmp_path/f'cycle-{index}'; directory.mkdir()
        output = directory/'result.json'
        command([str(python), '-I', '-B', str(script), '--source-map', str(source_map),
            '--output', str(output)], directory, f'cycle-{index}')
        results.append(output.read_bytes())
        checkpoints.append({name: (directory/(name+'-checkpoint.json')).read_bytes()
                            for name in ('adaptive', 'modal')})
    assert results[0] == results[1] and checkpoints[0] == checkpoints[1]
    result = json.loads(results[0])
    assert result['imports_isolated'] and not result['production_qualified']
    assert len(result['source_files']) == len(outputs) and len(outputs) >= 58
    for name, raw in checkpoints[0].items():
        assert result['controls'][name]['restart_exact']
        assert result['controls'][name]['checkpoint_bytes'] == len(raw)
        assert result['controls'][name]['checkpoint_sha256'] == hashlib.sha256(raw).hexdigest()
    receipt = dict(schema='GE_BEAM3_ADAPTIVE_MODAL_PACKAGE_TEST_RECEIPT_V1',
        wheel_file=wheel.name, wheel_bytes=wheel.stat().st_size, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        result_bytes=len(results[0]), result_sha256=hashlib.sha256(results[0]).hexdigest(),
        checkpoints={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                     for name, raw in checkpoints[0].items()}, cycles_identical=True,
        source_map_sha256=hashlib.sha256(source_map.read_bytes()).hexdigest(),
        dependencies=authority['wheelhouse']['files'], production_qualified=False, release_authorized=False)
    with (tmp_path/'receipt.json').open('xb') as stream: stream.write(canonical(receipt))
