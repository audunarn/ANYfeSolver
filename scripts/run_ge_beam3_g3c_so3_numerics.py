"""Bounded isolated G3c SO(3) kernel tests; no graph or public routing."""
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
REVIEW='docs/reference_cases/ge_beam3_g3c_so3_numerics_design_review_v1.json'
REVIEW_SHA='1f04403843ade1f1eab0642c9fb14b1846852bf76100253f5540062321d78856'
CONTRACT='docs/reference_cases/ge_beam3_g3c_so3_numerics_contract_v1.json'
ORACLE='tests/ge_beam3_g3c_so3_oracle.py'
ORACLE_SHA='06244f1cf542267520172c587a8fa902c9c8892bf2f1c2bf9719913c96984a2a'
TEST='tests/test_ge_beam3_g3c_so3_numerics.py'
INVENTORIES={
    'static': [TEST+'::TestStaticIsolation::'+name for name in (
        'test_source_bindings_and_no_existing_routing',
        'test_canonical_and_source_hash_mutations_rejected',
        'test_independent_oracle_identity_and_imports')],
    'scalar': [TEST+'::TestScalarAccuracy::'+name for name in (
        'test_a_priori_tail_majorants', 'test_exp_coefficient_values_and_derivatives',
        'test_log_coefficient_values_and_derivatives')],
    'jet': [TEST+'::TestFullJetAccuracy::'+name for name in (
        'test_exp_full_jets_with_curved_input_seeds', 'test_log_full_matrix_jets',
        'test_principal_identity_and_proper_covariance', 'test_domain_guards_and_no_clipping')],
}

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
def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        '-c','core.autocrlf=true','-c','core.eol=crlf',*args],cwd=ROOT,
        env=environment.git_env(),timeout=30).decode().strip()

def clean_identity():
    if git('status','--porcelain','--untracked-files=all'):
        raise ValueError('clean committed candidate required')
    return dict(commit=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'))

def verify_sources(contract,root=ROOT):
    for row in contract['sources']:
        raw=environment.regular(root/row['path']).read_bytes().replace(b'\r\n',b'\n')
        if dict(bytes=len(raw),sha256=sha256(raw).hexdigest())!={k:row[k] for k in ('bytes','sha256')}:
            raise ValueError('changed bound source '+row['path'])

def authority():
    raw=environment.regular(ROOT/REVIEW).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=REVIEW_SHA: raise ValueError('independent design review mismatch')
    review=environment.strict(raw)
    if review['decision']!='ACCEPTED_G3C_ISOLATED_SO3_NUMERICS_DESIGN_ONLY' or review['findings']:
        raise ValueError('private numerics design authority missing')
    raw=environment.regular(ROOT/CONTRACT).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=review['scope']['contract_sha256']: raise ValueError('changed contract')
    contract=environment.strict(raw); subject=review['subject_commit']
    if git('rev-parse',subject+'^{tree}')!=review['scope']['subject_tree']: raise ValueError('design tree')
    if git('rev-parse',contract['base']['commit']+'^{tree}')!=contract['base']['tree']: raise ValueError('base tree')
    git('merge-base','--is-ancestor',subject,'HEAD')
    git('merge-base','--is-ancestor',contract['base']['commit'],subject)
    for path in contract['contract_extent']:
        original=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show',subject+':'+path],
                    cwd=ROOT,env=environment.git_env(),timeout=30).replace(b'\r\n',b'\n')
        if environment.regular(ROOT/path).read_bytes().replace(b'\r\n',b'\n')!=original:
            raise ValueError('changed design document')
    verify_sources(contract)
    allowed=set(contract['contract_extent']+contract['implementation_extent']+
        [REVIEW,'docs/GE_BEAM3_G3C_SO3_NUMERICS_DESIGN_REVIEW.md'])
    for line in git('diff','--name-status','--no-renames',contract['base']['commit']).splitlines():
        status,path=line.split('\t')
        if status!='A' or path not in allowed: raise ValueError('changed inherited file or extent '+path)
    raw=environment.regular(ROOT/ORACLE).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=ORACLE_SHA: raise ValueError('independent oracle changed')
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
    selected=INVENTORIES.get(lease['lane'])
    if selected is None or selected!=lease['inventory']: raise ValueError('unregistered private kernel lane')
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),*selected])
    if inputs()!=lease['inputs'] or clean_identity()!=lease['candidate']: raise ValueError('candidate changed during test')
    print('G3C CHECKPOINT complete',flush=True)
    return int(code)
def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        return worker(Path(sys.argv[2]),sys.argv[3])
    if len(sys.argv)!=3 or sys.argv[1]!='--lane' or sys.argv[2] not in INVENTORIES or os.name!='nt':
        raise ValueError('use --lane static|scalar|jet on Windows')
    lane=sys.argv[2]
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('launch with -I -S -B')
    authority()
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-so3-kernel-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    write(out/'lease.json',dict(kind='PRIVATE_KERNEL_DEVELOPMENT_NOT_GRAPH_QUALIFICATION',lane=lane,inventory=INVENTORIES[lane],candidate=clean_identity(),inputs=inputs()))
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
        record=dict(kind='PRIVATE_KERNEL_DEVELOPMENT_NOT_GRAPH_QUALIFICATION',status=status,lane=lane,
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
