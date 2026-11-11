"""Bounded isolated G3c MIXED OWNER development tests; never formal qualification."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import ge_beam3_g3b_environment as environment

CAPSULE=Path(r'C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\ge-beam3-g3b-rehearsal-history-20260912-0fe7f8f/capsule/environment.json')
CAPSULE_SHA='2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'
REVIEW='docs/reference_cases/ge_beam3_g3c_mixed_owner_design_review_v1.json'
REVIEW_SHA='c7efca597db0739b83eeeda98f55dd714cea1c4a05e35dbfc6a7f269cc8c91e3'
TEST='tests/test_ge_beam3_g3c_owner_bridge.py'
SMOKE=[TEST+'::test_b2_pair_zero_equilibrium_commits_actual_native_states']
DIAGNOSTIC=['tests/test_ge_beam3_g3c_owner_diagnostic.py::test_component_directional_diagnostic']
ELEMENTARY=['tests/test_ge_beam3_g3c_elementary_diagnostic.py::test_elementary_rotation_diagnostic']

def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()
def write(path,v):
    with path.open('xb') as f: f.write(canonical(v))
def inputs():
    env=environment.git_env()
    names=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        '-c','core.autocrlf=true','-c','core.eol=crlf',
        'ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,env=env,timeout=30).decode().splitlines()
    rows={}
    for name in sorted(set(names)):
        raw=environment.regular(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        rows[name]=dict(bytes=len(raw),sha256=sha256(raw).hexdigest())
    return rows
def verify_equation_bindings(contract,root=ROOT):
    for row in contract['sources']:
        raw=environment.regular(root/row['path']).read_bytes().replace(b'\r\n',b'\n')
        if len(raw)!=row['bytes'] or sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('changed equation source '+row['path'])
    for path,row in contract['payloads'].items():
        raw=environment.regular(root/path).read_bytes().replace(b'\r\n',b'\n')
        if dict(bytes=len(raw),sha256=sha256(raw).hexdigest())!=row:
            raise ValueError('changed equation payload '+path)
def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        '-c','core.autocrlf=true','-c','core.eol=crlf',*args],cwd=ROOT,
        env=environment.git_env(),timeout=30).decode().strip()

def clean_identity():
    if git('status','--porcelain','--untracked-files=all'):
        raise ValueError('clean committed candidate required')
    return dict(commit=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'))

def authority():
    raw=(ROOT/REVIEW).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=REVIEW_SHA: raise ValueError('independent equation review mismatch')
    r=environment.strict(raw)
    if r['decision']!='ACCEPTED_G3C_MIXED_OWNER_DESIGN_IMPLEMENTATION_ONLY' or r['findings']:
        raise ValueError('mixed owner design authority missing')
    contract=(ROOT/'docs/reference_cases/ge_beam3_g3c_mixed_owner_contract_v1.json').read_bytes().replace(b'\r\n',b'\n')
    if sha256(contract).hexdigest()!=r['scope']['contract_sha256']: raise ValueError('changed mixed owner contract')
    if git('rev-parse',r['subject_commit']+'^{tree}')!=r['scope']['subject_tree']:
        raise ValueError('reviewed tree mismatch')
    git('merge-base','--is-ancestor',r['subject_commit'],'HEAD')
    verify_equation_bindings(environment.strict(contract))
    environment.verify(CAPSULE,CAPSULE_SHA)

def worker(out,lease_hash):
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('isolated interpreter required')
    raw=(out/'lease.json').read_bytes()
    if sha256(raw).hexdigest()!=lease_hash: raise ValueError('lease hash')
    lease=environment.strict(raw)
    print('G3C CHECKPOINT authority',flush=True)
    authority()
    if inputs()!=lease['inputs'] or clean_identity()!=lease['candidate']: raise ValueError('candidate changed before test')
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(CAPSULE.parent/'site')]
    import pytest
    selected={'smoke':SMOKE,'full':[TEST],'diagnostic':DIAGNOSTIC,'elementary':ELEMENTARY}.get(lease['lane'])
    if selected is None: raise ValueError('unregistered mixed owner lane')
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),*selected])
    if inputs()!=lease['inputs'] or clean_identity()!=lease['candidate']: raise ValueError('candidate changed during test')
    print('G3C CHECKPOINT complete',flush=True)
    return int(code)
def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        return worker(Path(sys.argv[2]),sys.argv[3])
    if sys.argv[1:] not in ([],['--smoke'],['--diagnostic'],['--elementary-diagnostic']) or os.name!='nt': raise ValueError('fixed Windows mixed owner lane required')
    lane={'--smoke':'smoke','--diagnostic':'diagnostic','--elementary-diagnostic':'elementary'}.get(sys.argv[1] if len(sys.argv)>1 else '', 'full')
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('launch with -I -S -B')
    authority()
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-owner-development-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    write(out/'lease.json',dict(kind='LOCAL_DEVELOPMENT_NOT_QUALIFICATION',lane=lane,candidate=clean_identity(),inputs=inputs()))
    lease_hash=sha256((out/'lease.json').read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('g3c_process',ROOT/'docs/reference_cases/e4_pl_s3_v2_bounded_process.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    env=dict(os.environ)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','TBB_NUM_THREADS'): env[k]='1'
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONDONTWRITEBYTECODE='1')
    job=m._ProcessJob(24*1024**3); start=time.monotonic(); last=start; previous=None; status='FAILED'
    try:
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            process=job.launch([sys.executable,'-I','-S','-B','-u',str(Path(__file__).resolve()),
                 '--worker',str(out),lease_hash],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            while True:
                cpu,active,peak=job.accounting(); now=time.monotonic()
                progress=(cpu,(out/'stdout.log').stat().st_size,(out/'stderr.log').stat().st_size)
                if progress!=previous: previous=progress;last=now
                if now-start>=600 or now-last>=120 or peak>24*1024**3:
                    status='RESOURCE_BLOCKED';job.terminate();break
                if process.poll() is not None and active==0:
                    status='PASSED' if process.returncode==0 else 'FAILED';break
                time.sleep(.1)
            process.wait(timeout=15)
        record=dict(kind='LOCAL_DEVELOPMENT_NOT_QUALIFICATION',status=status,lane=lane,
            elapsed_seconds=time.monotonic()-start,returncode=process.returncode,
            active_processes=job.accounting()[1],peak_tree_bytes=peak,lease_sha256=lease_hash)
        write(out/'process.json',record)
        print((out/'stdout.log').read_text(errors='replace'));print((out/'stderr.log').read_text(errors='replace'))
        print(canonical(record).decode(),flush=True)
        return int(status!='PASSED')
    finally:
        if job.accounting()[1]: job.terminate()
        job.close()
if __name__=='__main__': raise SystemExit(main())
