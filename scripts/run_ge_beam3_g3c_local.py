"""Bounded isolated G3c LOCAL development test; never formal qualification."""
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
REVIEW='docs/reference_cases/ge_beam3_g3c_equation_review_v1.json'
REVIEW_SHA='4d92a728de4c012e1dd986ffbc2c21f674891809cad152c6a1b9b0fb64f8818f'
TEST='tests/test_ge_beam3_g3c_local_beam.py'
FINITE_TEST='tests/test_ge_beam3_g3c_finite_local.py'

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
def authority():
    raw=(ROOT/REVIEW).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=REVIEW_SHA: raise ValueError('independent equation review mismatch')
    r=environment.strict(raw)
    if r['decision']!='ACCEPTED_G3C_EQUATIONS_LOCAL_IMPLEMENTATION_ONLY' or r['findings']:
        raise ValueError('local implementation review authority missing')
    contract=(ROOT/'docs/reference_cases/ge_beam3_g3c_objective_beam_frame_contract_v1.json').read_bytes().replace(b'\r\n',b'\n')
    if sha256(contract).hexdigest()!=r['scope']['contract_sha256']: raise ValueError('changed equation contract')
    verify_equation_bindings(environment.strict(contract))
    recovery_review=(ROOT/'docs/reference_cases/ge_beam3_g3c_recovery_equation_review_v1.json').read_bytes().replace(b'\r\n',b'\n')
    if sha256(recovery_review).hexdigest()!='a765474f6fe0ed25412b63937444ddccb82113952143acea2c94776c98a8fc80':
        raise ValueError('changed independent recovery equation review')
    recovery_contract=(ROOT/'docs/GE_BEAM3_G3C_LOCAL_RECOVERY_EQUATIONS.md').read_bytes().replace(b'\r\n',b'\n')
    if sha256(recovery_contract).hexdigest()!='0f81a43804b4874402d7ff09dc158de95888bc4f7e3c734e76548688f7235e67':
        raise ValueError('changed recovery equations')
    foundation=(ROOT/'docs/reference_cases/ge_beam3_g3c_local_foundation_review_v1.json').read_bytes().replace(b'\r\n',b'\n')
    if sha256(foundation).hexdigest()!='b380258cfc54aa9725dfdb11596ce66c803a491410b19411518bd843b0bcff92':
        raise ValueError('changed foundation review')
    environment.verify(CAPSULE,CAPSULE_SHA)
def worker(out,lease_hash):
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('isolated interpreter required')
    raw=(out/'lease.json').read_bytes()
    if sha256(raw).hexdigest()!=lease_hash: raise ValueError('lease hash')
    lease=environment.strict(raw)
    print('G3C CHECKPOINT authority',flush=True)
    authority()
    if inputs()!=lease['inputs']: raise ValueError('candidate changed before test')
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(CAPSULE.parent/'site')]
    import pytest
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),TEST,FINITE_TEST])
    if inputs()!=lease['inputs']: raise ValueError('candidate changed during test')
    print('G3C CHECKPOINT complete',flush=True)
    return int(code)
def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        return worker(Path(sys.argv[2]),sys.argv[3])
    if len(sys.argv)!=1 or os.name!='nt': raise ValueError('fixed Windows local lane required')
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('launch with -I -S -B')
    authority()
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-local-development-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    write(out/'lease.json',dict(kind='LOCAL_DEVELOPMENT_NOT_QUALIFICATION',inputs=inputs()))
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
        record=dict(kind='LOCAL_DEVELOPMENT_NOT_QUALIFICATION',status=status,
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
