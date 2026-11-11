"""Fresh bounded ordinary source/wheel/installed verification (not qualification)."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/agent_plans/development/r8-b2-s3-physical-footprint-20261010'
HERE=OUT/'evidence'
NEW=ROOT/'tests/test_r8_b2_s3_physical_footprint_dev.py'
OLD=ROOT/'tests/test_r8_b2_physical_footprint_static_foundation_dev.py'
ENV=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 NUMBA_NUM_THREADS='1',NUMBA_DISABLE_JIT='1',NUMEXPR_NUM_THREADS='1',
 VECLIB_MAXIMUM_THREADS='1',PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',
 PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONPATH='')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def identities():return {str(p.relative_to(ROOT)):sha(p) for p in [NEW,OLD,Path(__file__),*sorted((ROOT/'src/anyfesolver').rglob('*.py'))]}
def child(stage,out,focus=None):
 import psutil
 psutil.Process().cpu_affinity([0]);selected=ROOT/'src' if stage=='source' else HERE/'site'
 sys.path[:0]=[str(selected),str(ROOT),str(ROOT/'tests')]
 os.environ['B2_FOUNDATION_OUTPUT']=str(out);os.environ['B2_FOUNDATION_SUPERVISED']='r8-s3-root-adopted'
 from scripts.r8_f07_public_descriptor_dev import runtime
 runtime(out,'runtime-before',selected)
 import pytest
 args=[str(NEW),str(OLD),'-q','-x','-o','pythonpath=',
     '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'pytest-tmp')]
 if focus:args+=['-k',focus]
 try:return int(pytest.main(args))
 finally:runtime(out,'runtime-after',selected)
def run(stage,focus=None):
 import psutil
 HERE.mkdir(parents=True,exist_ok=True);lp=HERE/'ledger.json'
 ledger=json.loads(lp.read_text()) if lp.exists() else dict(status='OPEN',budget_s=300.,reserve_s=60.,used_s=0.,attempts=[])
 assert ledger['status']=='OPEN'
 name=stage+str(1+sum(p.is_dir() for p in HERE.glob(stage+'[0-9]*')));out=HERE/name;out.mkdir()
 ids=identities();write(out/'identities.json',ids)
 if stage in ('source','installed'):
  cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--stage',stage,'--out',str(out)]
  if focus:cmd+=['--focus',focus]
 elif stage=='wheel':
  assert any(a['stage']=='source' and a['status']=='PASS' for a in ledger['attempts'])
  prior=list(HERE.glob('wheel*/frozen-source.json'))
  if prior:
   frozen=json.loads(prior[0].read_text());build=Path(frozen['build_input'])
   assert all(sha(build/'src'/rel)==v for rel,v in frozen['source'].items())
  else:
   build=out/'input';build.mkdir();shutil.copytree(ROOT/'src/anyfesolver',build/'src/anyfesolver',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   for n in ('pyproject.toml','README.md','LICENSE','COPYRIGHT','THIRD_PARTY_NOTICES.md'):shutil.copy2(ROOT/n,build/n)
   snapshot={str(p.relative_to(build/'src')):sha(p) for p in (build/'src/anyfesolver').rglob('*.py')}
   assert all(v==ids[str(Path('src')/rel)] for rel,v in snapshot.items()),'source changed during immutable build snapshot'
   frozen=dict(source=snapshot,build_input=str(build))
  write(out/'frozen-source.json',frozen)
  cmd=[sys.executable,'-m','pip','wheel',str(build),'--no-deps','--no-build-isolation','--no-index','--no-cache-dir','--wheel-dir',str(out/'wheel')]
 elif stage=='install':
  wheels=list(HERE.glob('wheel*/wheel/*.whl'));assert len(wheels)==1
  assert not (HERE/'site').exists()
  cmd=[sys.executable,'-m','pip','install',str(wheels[0]),'--no-deps','--no-index','--no-cache-dir','--target',str(HERE/'site')]
 else:raise ValueError(stage)
 if stage=='installed':
  assert any(a['stage']=='source' and a['status']=='PASS' for a in ledger['attempts'])
  passed=[a for a in ledger['attempts'] if a['stage']=='wheel' and a['status']=='PASS'];assert len(passed)==1
  source=json.loads((HERE/passed[0]['name']/'frozen-source.json').read_text())['source']
  assert all(sha(HERE/'site'/rel)==v for rel,v in source.items())
  write(out/'installed-binding.json',dict(source=source,wheels={str(p):sha(p) for p in HERE.glob('wheel*/wheel/*.whl')}))
 cap=min(60.,300.-60.-ledger['used_s']);assert cap>0
 env=os.environ.copy();env.update(ENV)
 local_temp=out/'temp';local_temp.mkdir();env.update(TMP=str(local_temp),TEMP=str(local_temp),PIP_CACHE_DIR=str(out/'pip-cache'))
 receipt=dict(command=cmd,stage=stage,candidate_sha256=ids,environment={k:env[k] for k in (*ENV,'TMP','TEMP','PIP_CACHE_DIR')},
 cap_s=cap,memory_cap_bytes=2147483648,peak_tree_rss_bytes=0,python=sys.executable,python_sha256=sha(sys.executable))
 write(out/'launch.json',receipt);t=time.perf_counter_ns();code=None;failure=None;proc=None
 try:
  with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
   psutil.Process().cpu_affinity([0]);proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
   psutil.Process(proc.pid).cpu_affinity([0])
   while proc.poll() is None:
    try:members=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:members=[]
    rss=0
    for m in members:
     try:rss+=m.memory_info().rss
     except psutil.NoSuchProcess:pass
    receipt['peak_tree_rss_bytes']=max(receipt['peak_tree_rss_bytes'],rss)
    if (time.perf_counter_ns()-t)/1e9>cap:failure='timeout';break
    if rss>2147483648:failure='memory cap';break
    time.sleep(.04)
   if failure:
    for m in reversed(members):
     try:m.kill()
     except psutil.NoSuchProcess:pass
   code=proc.wait()
 finally:
  wall=(time.perf_counter_ns()-t)/1e9;ledger['used_s']+=wall
  receipt.update(status='PASS' if code==0 and failure is None else 'FAIL',exit_code=code,
   failure=failure,wall_s=wall,cumulative_wall_s=ledger['used_s'],remaining_s=300-ledger['used_s'])
  ledger['attempts'].append(dict(name=name,stage=stage,status=receipt['status'],wall_s=wall))
  write(out/'receipt.json',receipt);write(lp,ledger)
 print(json.dumps({k:receipt[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
 return int(receipt['status']!='PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('source','wheel','install','installed'),required=True)
 p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--focus');a=p.parse_args()
 raise SystemExit(child(a.stage,Path(a.out),a.focus) if a.child else run(a.stage,a.focus))
