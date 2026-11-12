"""Saved-data-only supervisor; requires a fresh adopted authority record."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/agent_plans/development/r8-f06-cell-moment-20261010'
ENV=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 NUMBA_NUM_THREADS='1',NUMBA_DISABLE_JIT='1',NUMEXPR_NUM_THREADS='1',
 VECLIB_MAXIMUM_THREADS='1',PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',
 PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONPATH='')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bindings():
 saved=json.loads((DOC/'saved-input-bindings.json').read_text())
 fixture=json.loads((DOC/'fixture-source-bindings.json').read_text())['bindings']
 for p,h in {**saved,**fixture}.items():assert sha(ROOT/p)==h
 current={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
  ROOT/'scripts/r8_f06_cell_moment_oracle_dev.py',ROOT/'tests/test_r8_f06_cell_moment_oracle_dev.py')}
 return dict(saved=saved,fixture=fixture,current=current)
def child(out,stage):
 import psutil,numpy,pytest
 psutil.Process().cpu_affinity([0]);sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
 os.environ['F06_CELL_MOMENT_OUTPUT']=str(out)
 from scripts.r8_f07_public_descriptor_dev import runtime
 runtime(out,'runtime-before',ROOT/'src')
 try:
  return int(pytest.main([str(ROOT/'tests/test_r8_f06_cell_moment_oracle_dev.py'),
   '-q','-x','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),
   '--basetemp',str(out/'pytest-tmp'),'-k','control' if stage=='controls' else 'complete']))
 finally:
  runtime(out,'runtime-after',ROOT/'src')
  assert not any(n=='anyfesolver' or n.startswith('anyfesolver.') for n in sys.modules), 'production import outside saved-data-only authority'
def run(stage):
 import psutil
 authority=json.loads((DOC/'AUTHORITY.json').read_text());assert authority['status']=='ADOPTED'
 assert authority['scope']=='saved-data-only F06 full48 missing12 oracle; no live physical jobs'
 base=DOC/'evidence';base.mkdir(exist_ok=True);lp=base/'ledger.json'
 ledger=json.loads(lp.read_text()) if lp.exists() else dict(status='OPEN',budget_s=authority['budget_s'],
  reserve_s=authority['reserve_s'],used_s=0.,authority_sha256=sha(DOC/'AUTHORITY.json'),attempts=[])
 assert ledger['status']=='OPEN' and ledger['authority_sha256']==sha(DOC/'AUTHORITY.json')
 cap=min(60.,authority['process_cap_s'],ledger['budget_s']-ledger['reserve_s']-ledger['used_s']);assert cap>0
 out=base/(stage+str(1+sum(a['stage']==stage for a in ledger['attempts'])));out.mkdir();binding=bindings();write(out/'bindings.json',binding)
 cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--out',str(out),'--stage',stage]
 env=os.environ.copy();env.update(ENV);temp=out/'temp';temp.mkdir();env.update(TEMP=str(temp),TMP=str(temp))
 receipt=dict(command=cmd,stage=stage,scope=authority['scope'],authority_sha256=sha(DOC/'AUTHORITY.json'),
  cap_s=cap,memory_cap_bytes=2147483648,environment={k:env[k] for k in (*ENV,'TEMP','TMP')},
  input_identities=binding,python=sys.executable,python_sha256=sha(sys.executable),peak_tree_rss_bytes=0)
 write(out/'launch.json',receipt);code=None;failure=None;t=time.perf_counter_ns()
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
   failure=failure,wall_s=wall,cumulative_wall_s=ledger['used_s'],remaining_s=ledger['budget_s']-ledger['used_s'])
  ledger['attempts'].append(dict(name=out.name,stage=stage,status=receipt['status'],wall_s=wall))
  write(out/'receipt.json',receipt);write(lp,ledger)
 print(json.dumps({k:receipt[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
 return int(receipt['status']!='PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',choices=('controls','saved'),required=True);a=p.parse_args()
 raise SystemExit(child(Path(a.out),a.stage) if a.child else run(a.stage))
