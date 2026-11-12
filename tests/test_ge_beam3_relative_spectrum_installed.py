"""Offline wheel correctness for the private relative reference spectrum."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_relative_spectrum_wheel_two_fresh_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]; records = root/'docs/reference_cases'
    authority = json.loads((records/'ge_beam3_mixed_p3_package_execution_authority.json').read_text())
    wheelhouse = Path(os.environ.get('ANY_GE_BEAM3_TEST_WHEELHOUSE', authority['locations']['wheelhouse']))
    if not wheelhouse.is_dir(): pytest.skip('hash-bound offline dependency wheelhouse unavailable')
    for row in authority['wheelhouse']['files']:
        raw = (wheelhouse/row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest().upper() == row['sha256']
    # Reuse dependency ARTIFACT identities only. No historical request is run.
    force = json.loads((records/'ge_beam3_force_program_development_evidence.json').read_text())
    raw = (records/'ge_beam3_native_load_source_map.json').read_text(encoding='utf-8').encode('utf-8')
    assert hashlib.sha256(raw).hexdigest() == force['preserved_native_source_map_sha256']
    outputs = {**json.loads(raw)['outputs'], **force['preserved_framework']}
    bindings = [force['source_bindings']]
    for name, key in (('displacement_program', 'source_bindings'), ('arc_geometry', 'bindings'),
        ('arc_program', 'source_bindings'), ('adaptive_arc', 'source_bindings'), ('loaded_modal', 'source_bindings')):
        bindings.append(json.loads((records/f'ge_beam3_{name}_development_evidence.json').read_text())[key])
    for name in ('reference_factor_spectrum', 'relative_reference_spectrum'):
        value = json.loads((records/f'ge_beam3_{name}_evidence.json').read_text())
        bindings.append({r['path']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in value['sources']})
    for binding in bindings:
        for path, value in binding.items():
            if path.startswith('src/'):
                if path in outputs: assert outputs[path] == value
                outputs[path] = value
    assert len(outputs) == 65

    def canonical(value):
        return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')

    source_map = tmp_path/'source-map.json'
    with source_map.open('xb') as stream:
        stream.write(canonical(dict(schema='GE_BEAM3_RELATIVE_SPECTRUM_INSTALLED_SOURCE_MAP_V2',
            candidate_commit='72e095f5dba8bc0dca11b84131bcf1da4af06bc8',
            candidate_tree='664b2f518758f1011ea640988de5230f69e0394d',
            output_text_normalization='UTF8_LF', outputs=outputs, production_qualified=False)))
    env = dict(os.environ); env.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'; env['PIP_CONFIG_FILE'] = os.devnull

    def command(argv, cwd, name, limit=180):
        print('relative wheel: '+name+' started', flush=True)
        with (tmp_path/(name+'.stdout')).open('xb') as out, (tmp_path/(name+'.stderr')).open('xb') as err:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err,
                start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
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
        print('relative wheel: '+name+' completed', flush=True)

    # Build from the committed tree, not an editable checkout snapshot. Only
    # the explicit source-text line ending fixture is changed in this copy.
    candidate = '72e095f5dba8bc0dca11b84131bcf1da4af06bc8'
    source_zip = tmp_path/'candidate-source.zip'
    command(['git', 'archive', '--format=zip', '--output='+str(source_zip), candidate, '--',
        'src', 'pyproject.toml', 'README.md', 'LICENSE', 'COPYRIGHT', 'THIRD_PARTY_NOTICES.md'], root, 'archive')
    build = tmp_path/'build'; build.mkdir()
    with zipfile.ZipFile(source_zip) as archive:
        for name in archive.namelist():
            target = (build/name).resolve(); assert target.is_relative_to(build.resolve())
        archive.extractall(build)
    for path, expected in outputs.items():
        item = build/path; normalized = item.read_text(encoding='utf-8').encode('utf-8')
        assert dict(bytes=len(normalized), sha256=hashlib.sha256(normalized).hexdigest()) == expected
        item.write_bytes(normalized.replace(b'\n', b'\r\n'))
    wheels = tmp_path/'wheels'; wheels.mkdir()
    command([sys.executable, '-I', '-B', '-c',
        'from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])', str(wheels)], build, 'build')
    candidates = list(wheels.glob('*.whl')); assert len(candidates) == 1; wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        assert all('anyfesolver/'+name in archive.namelist() for name in (
            '_relative_reference_svd.py', '_native_relative_reference_factor_spectrum.py',
            '_ge_beam3_relative_virgin_reference_modes.py'))
        assert not any(name.startswith(('docs/', 'tests/')) for name in archive.namelist())
    environment = tmp_path/'environment'
    command([sys.executable, '-I', '-B', '-m', 'venv', str(environment)], tmp_path, 'venv')
    python = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command([str(python), '-I', '-B', '-m', 'pip', 'install', '--no-index', '--find-links',
        str(wheelhouse), str(wheel)], tmp_path, 'install')
    script = tmp_path/'installed_check.py'
    shutil.copy2(records/'ge_beam3_relative_spectrum_installed_smoke.py', script)
    results = []; diagnostics = []
    for index in (1, 2):
        directory = tmp_path/f'cycle-{index}'; directory.mkdir(); output = directory/'result.json'
        command([str(python), '-I', '-B', str(script), '--source-map', str(source_map),
            '--output', str(output)], directory, f'cycle-{index}')
        results.append(output.read_bytes())
        row = json.loads(results[-1]); files = {}
        for name, expected in row['diagnostics'].items():
            raw = (directory/name).read_bytes(); files[name] = raw
            assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == expected
        diagnostics.append(files)
    assert results[0] == results[1] and diagnostics[0] == diagnostics[1]
    result = json.loads(results[0])
    assert result['imports_isolated'] and result['loaded_state_rejected'] and result['tiny_positive_preserved']
    assert len(result['source_files']) == 65 and not result['production_qualified']
    receipt = dict(schema='GE_BEAM3_RELATIVE_SPECTRUM_PACKAGE_TEST_RECEIPT_V2',
        candidate_commit=candidate, wheel_file=wheel.name, wheel_bytes=wheel.stat().st_size,
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        result_bytes=len(results[0]), result_sha256=hashlib.sha256(results[0]).hexdigest(),
        diagnostics=result['diagnostics'], cycles_identical=True,
        source_archive_sha256=hashlib.sha256(source_zip.read_bytes()).hexdigest(),
        source_map_sha256=hashlib.sha256(source_map.read_bytes()).hexdigest(),
        dependencies=authority['wheelhouse']['files'], production_qualified=False, release_authorized=False)
    with (tmp_path/'receipt.json').open('xb') as stream: stream.write(canonical(receipt))
