"""Saved-data-only supervisor; requires a fresh adopted authority record."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/agent_plans/development/r8-f06-compensated-state-20261010'
ENV=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 NUMBA_NUM_THREADS='1',NUMBA_DISABLE_JIT='1',NUMEXPR_NUM_THREADS='1',
 VECLIB_MAXIMUM_THREADS='1',PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',
 PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONPATH='')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bindings():
 saved=json.loads((ROOT/'docs/agent_plans/development/r8-f06-cell-moment-20261010/saved-input-bindings.json').read_text())
 for p,h in saved.items():assert sha(ROOT/p)==h
 current={p.relative_to(ROOT).as_posix():sha(p) for p in [*list((ROOT/'src/anyfesolver').glob('*.py')),
  Path(__file__),ROOT/'scripts/r8_f06_compensated_state_route_dev.py',ROOT/'tests/test_r8_f06_compensated_state_dev.py']}

 frozen=DOC/'package/snapshot-bindings.json'
 return dict(saved=saved,current=current,frozen_package=None if not frozen.exists() else json.loads(frozen.read_text()))

def runtime(out,name,selected):
 import psutil,numba,threadpoolctl,importlib.metadata
 settings=dict(affinity=psutil.Process().cpu_affinity(),optimize=sys.flags.optimize,
  python=dict(executable=sys.executable,version=sys.version,base_executable=getattr(sys,'_base_executable',None)),
  dependencies={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','pytest','psutil','threadpoolctl')},
  numba=dict(disable_jit=bool(numba.config.DISABLE_JIT),num_threads=numba.get_num_threads()),pools=threadpoolctl.threadpool_info())
 imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
  if (n=='anyfesolver' or n.startswith('anyfesolver.')) and getattr(m,'__file__',None)}
 settings.update(imports=imports,import_sha256={n:sha(p) for n,p in imports.items()})
 write(out/(name+'.json'),settings)
 assert settings['affinity']==[0] and settings['optimize']==0 and settings['numba']==dict(disable_jit=True,num_threads=1)
 assert all(p['num_threads']==1 for p in settings['pools'])
 for n,p in imports.items():assert Path(p).is_relative_to(selected/'anyfesolver'),(n,p)
 return settings

def child(out,stage):
 import psutil,numpy,pytest
 psutil.Process().cpu_affinity([0])
 package=DOC/'package'
 if stage=='wheelverify':
  frozen=json.loads((package/'snapshot-bindings.json').read_text());site=package/'site'
  verified={};excluded={}
  for rel,h in frozen['package'].items():
   if not rel.startswith('src/anyfesolver/'):
    excluded[rel]='generated packaging metadata; not an imported package source';continue
   target=site/Path(rel).relative_to('src')
   assert sha(target)==h,(str(target),h)
   assert sha(package/'snapshot'/rel)==h,'frozen package source altered during build'
   verified[str(target)]=h
  write(out/'installed-file-bindings.json',verified);write(out/'excluded-generated-metadata.json',excluded)
  wheels=list((package/'wheels').glob('*.whl'));assert len(wheels)==1
  wheel=wheels[0]
  write(package/'wheel.json',dict(wheel=str(wheel),sha256=sha(wheel),site=str(site),snapshot_sha256=sha(package/'snapshot-bindings.json')))
  return 0
 if stage=='wheel':
  snapshot=package/'snapshot';wheels=package/'wheels';site=package/'site'
  wheels.mkdir();site.mkdir()
  cmd=[sys.executable,'-m','pip','wheel',str(snapshot),'--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',str(wheels)]
  write(out/'build-command.json',cmd);subprocess.run(cmd,check=True)
  candidates=list(wheels.glob('*.whl'));assert len(candidates)==1
  wheel=candidates[0]
  cmd=[sys.executable,'-m','pip','install',str(wheel),'--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(site)]
  write(out/'install-command.json',cmd);subprocess.run(cmd,check=True)
  frozen=json.loads((package/'snapshot-bindings.json').read_text())
  verified={}
  for rel,h in frozen['package'].items():
   target=site/Path(rel).relative_to('src')
   assert sha(target)==h,(str(target),h)
   verified[str(target)]=h
  write(out/'installed-file-bindings.json',verified)
  write(package/'wheel.json',dict(wheel=str(wheel),sha256=sha(wheel),site=str(site),snapshot_sha256=sha(package/'snapshot-bindings.json')))
  return 0
 selected=package/'site' if stage=='installed' else ROOT/'src'
 sys.path[:0]=[str(selected),str(ROOT),str(ROOT/'tests')]
 os.environ['F06_COMPENSATED_OUTPUT']=str(out)
 runtime(out,'runtime-before',selected)
 try:
  return int(pytest.main([str(ROOT/'tests/test_r8_f06_compensated_state_dev.py'),
   '-q','-x','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),
   '--basetemp',str(out/'pytest-tmp'),'-k',dict(controls='control',saved='actual',lifecycle='lifecycle',installed='lifecycle or legacy')[stage]]))
 finally:
  runtime(out,'runtime-after',selected)
  if stage=='installed':
   frozen=json.loads((package/'snapshot-bindings.json').read_text())['package']
   checked={}
   for name,module in sys.modules.items():
    if name=='anyfesolver' or name.startswith('anyfesolver.'):
     path=getattr(module,'__file__',None)
     if path:
      relative=Path(path).relative_to(selected).as_posix();key='src/'+relative
      assert sha(path)==frozen[key],(name,path)
      checked[name]=dict(path=path,sha256=sha(path))
   write(out/'actual-installed-imports.json',checked)

def run(stage):
 import psutil
 authority=json.loads((DOC/'AUTHORITY.json').read_text());assert authority['status']=='ADOPTED'
 assert authority['scope']=='durable compensated native actual route focused development; saved original h=.002 full48; source and installed lifecycle'
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
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',choices=('controls','saved','lifecycle','wheel','wheelverify','installed'),required=True);a=p.parse_args()
 raise SystemExit(child(Path(a.out),a.stage) if a.child else run(a.stage))
