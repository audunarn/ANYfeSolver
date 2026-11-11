"""Offline installed-wheel development test, isolated from repository imports."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_coordinate_wheel_two_fresh_process_checks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    authority = json.loads((root/'docs/reference_cases/ge_beam3_mixed_p3_package_execution_authority.json').read_text())
    wheelhouse = Path(os.environ.get('ANY_GE_BEAM3_TEST_WHEELHOUSE', authority['locations']['wheelhouse']))
    if not wheelhouse.is_dir():
        pytest.skip('installed-wheel test requires the hash-bound offline dependency wheelhouse')
    for record in authority['wheelhouse']['files']:
        raw = (wheelhouse/record['path']).read_bytes()
        assert len(raw) == record['bytes']
        assert hashlib.sha256(raw).hexdigest().upper() == record['sha256']
    # Only an external disposable build snapshot is written. No build products,
    # egg-info, installation or version change is made in the source repository.
    build = tmp_path/'build'; build.mkdir()
    shutil.copytree(root/'src',build/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    # Exercise the Windows checkout transform in the disposable snapshot.
    # Canonical source hashes and actual installed byte hashes are distinct.
    for path in [*(build/'src/anyfesolver/_ge_beam3_p5').glob('*.py'),
                 *(build/'src/anyfesolver/_ge_beam3_p5_coordinates').glob('*.py')]:
        raw = path.read_bytes().replace(b'\r\n',b'\n')
        path.write_bytes(raw.replace(b'\n',b'\r\n'))
    for name in ('pyproject.toml','README.md','LICENSE','COPYRIGHT','THIRD_PARTY_NOTICES.md'):
        shutil.copy2(root/name,build/name)
    wheels = tmp_path/'wheels'; wheels.mkdir()
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env.pop('PYTHONPATH',None); env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    def command(argv, cwd, limit=120):
        completed = subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,timeout=limit)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        return completed
    command([sys.executable,'-I','-B','-c',
             'from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])',str(wheels)],build)
    candidates = list(wheels.glob('*.whl')); assert len(candidates) == 1
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.startswith(('docs/','tests/')) for name in archive.namelist())
        assert 'anyfesolver/_ge_beam3_p5/element.py' in archive.namelist()
        assert 'anyfesolver/_ge_beam3_p5_coordinates/element.py' in archive.namelist()
    environment = tmp_path/'environment'
    command([sys.executable,'-I','-B','-m','venv',str(environment)],tmp_path)
    python = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command([str(python),'-I','-B','-m','pip','install','--no-index','--find-links',str(wheelhouse),str(wheel)],tmp_path)
    script = tmp_path/'installed_check.py'; source_map = tmp_path/'source-map.json'
    shutil.copy2(root/'docs/reference_cases/ge_beam3_curved_p5_coordinate_installed_smoke.py',script)
    source_map.write_text((root/'docs/reference_cases/ge_beam3_curved_p5_coordinate_source_map.json').read_text(encoding='utf-8'),
                          encoding='utf-8', newline='\n')
    outputs = []
    for index in (1,2):
        folder = tmp_path/f'cycle-{index}'; folder.mkdir(); output = folder/'result.json'
        command([str(python),'-I','-B',str(script),'--source-map',str(source_map),'--output',str(output)],folder)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
    record = json.loads(outputs[0]); assert record['restart_exact'] and record['imports_isolated']
    assert not record['production_qualified']
    assert record['sub_ulp_axial_commit_verified']
    assert record['formulation'] == 'CANDIDATE_GE_BEAM3_P5_NATIVE_COORDINATE_V2'
    receipt = dict(schema='GE_BEAM3_P5_COORDINATE_PACKAGE_TEST_RECEIPT_V2',
        wheel_file=wheel.name,wheel_bytes=wheel.stat().st_size,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        result_bytes=len(outputs[0]),result_sha256=hashlib.sha256(outputs[0]).hexdigest(),
        cycles_identical=True,source_map_sha256=hashlib.sha256(source_map.read_bytes()).hexdigest(),
        dependencies=authority['wheelhouse']['files'],release_authorized=False)
    (tmp_path/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n')
