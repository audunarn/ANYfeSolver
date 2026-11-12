"""Offline wheel isolation check for the private native force program."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_force_program_wheel_two_fresh_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    # Reuse dependency artifact identities only, never its consumed request or
    # historical authorization. This is a new small development correctness test.
    authority = json.loads((root/'docs/reference_cases/ge_beam3_mixed_p3_package_execution_authority.json').read_text())
    wheelhouse = Path(os.environ.get('ANY_GE_BEAM3_TEST_WHEELHOUSE',authority['locations']['wheelhouse']))
    if not wheelhouse.is_dir(): pytest.skip('hash-bound offline dependency wheelhouse unavailable')
    for row in authority['wheelhouse']['files']:
        raw = (wheelhouse/row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest().upper() == row['sha256']
    development = json.loads((root/'docs/reference_cases/ge_beam3_force_program_development_evidence.json').read_text())
    old_raw = (root/'docs/reference_cases/ge_beam3_native_load_source_map.json').read_text(encoding='utf-8').encode('utf-8')
    assert hashlib.sha256(old_raw).hexdigest() == development['preserved_native_source_map_sha256']
    outputs = {**json.loads(old_raw)['outputs'],**development['preserved_framework'],
        **{p:v for p,v in development['source_bindings'].items() if p.startswith('src/')}}
    for path,binding in outputs.items():
        raw = (root/path).read_text(encoding='utf-8').encode('utf-8')
        assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
    def canonical(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')
    source_map = tmp_path/'source-map.json'
    with source_map.open('xb') as stream:
        stream.write(canonical(dict(schema='GE_BEAM3_FORCE_PROGRAM_INSTALLED_SOURCE_MAP_V1',
            candidate_commit='eec5f5f4032cfddb82f9d8b6ecdef1d291d5d2ce',
            output_text_normalization='UTF8_LF',outputs=outputs,production_qualified=False)))

    build = tmp_path/'build'; build.mkdir()
    shutil.copytree(root/'src',build/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    for path in outputs:
        item = build/path
        item.write_bytes(item.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    for name in ('pyproject.toml','README.md','LICENSE','COPYRIGHT','THIRD_PARTY_NOTICES.md'):
        shutil.copy2(root/name,build/name)
    wheels = tmp_path/'wheels'; wheels.mkdir()
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[name] = '1'
    env.pop('PYTHONPATH',None); env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'; env['PIP_CONFIG_FILE'] = os.devnull
    def command(argv,cwd,name,limit=120):
        completed = subprocess.run(argv,cwd=cwd,env=env,capture_output=True,timeout=limit)
        with (tmp_path/(name+'.stdout')).open('xb') as stream: stream.write(completed.stdout)
        with (tmp_path/(name+'.stderr')).open('xb') as stream: stream.write(completed.stderr)
        assert completed.returncode == 0, (completed.stdout+completed.stderr).decode('utf-8',errors='replace')
    command([sys.executable,'-I','-B','-c','from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])',str(wheels)],build,'build')
    candidates = list(wheels.glob('*.whl')); assert len(candidates) == 1
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        assert 'anyfesolver/_ge_beam3_load_program.py' in archive.namelist()
        assert not any(name.startswith(('docs/','tests/')) for name in archive.namelist())
    environment = tmp_path/'environment'
    command([sys.executable,'-I','-B','-m','venv',str(environment)],tmp_path,'venv')
    python = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command([str(python),'-I','-B','-m','pip','install','--no-index','--find-links',str(wheelhouse),str(wheel)],tmp_path,'install')
    script = tmp_path/'installed_check.py'
    shutil.copy2(root/'docs/reference_cases/ge_beam3_force_program_installed_smoke.py',script)
    results = []; checkpoints = []
    for index in (1,2):
        directory = tmp_path/f'cycle-{index}'; directory.mkdir()
        output = directory/'result.json'
        command([str(python),'-I','-B',str(script),'--source-map',str(source_map),'--output',str(output)],directory,f'cycle-{index}')
        results.append(output.read_bytes()); checkpoints.append((directory/'accepted-checkpoint.json').read_bytes())
    assert results[0] == results[1] and checkpoints[0] == checkpoints[1]
    result = json.loads(results[0]); assert result['restart_exact'] and result['imports_isolated']
    assert not result['production_qualified'] and len(result['source_files']) == 52
    receipt = dict(schema='GE_BEAM3_FORCE_PROGRAM_PACKAGE_TEST_RECEIPT_V1',
        wheel_file=wheel.name,wheel_bytes=wheel.stat().st_size,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        result_bytes=len(results[0]),result_sha256=hashlib.sha256(results[0]).hexdigest(),
        checkpoint_bytes=len(checkpoints[0]),checkpoint_sha256=hashlib.sha256(checkpoints[0]).hexdigest(),
        cycles_identical=True,source_map_sha256=hashlib.sha256(source_map.read_bytes()).hexdigest(),
        dependencies=authority['wheelhouse']['files'],production_qualified=False,release_authorized=False)
    with (tmp_path/'receipt.json').open('xb') as stream: stream.write(canonical(receipt))
