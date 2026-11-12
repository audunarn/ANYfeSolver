"""Bounded physical-fibre lifecycle; fresh authority, no historical input replay."""
from pathlib import Path
import argparse, json, os, sys, time, subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.r8_f06_log_domain_check_dev import ENV,sha,write,runtime
DOC=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/fibre-v1'
PRODUCTS=('_ge_beam3_retained_fibre.py','_ge_beam3_native_fibre_static_element.py',
 '_ge_beam3_native_definition.py','_ge_beam3_fibre_cell.py','_stiffened_shell_ge_experimental.py',
 '_stiffened_shell_ge_finite_experimental.py','_stiffened_shell_ge_checkpoint_binary.py',
 '_stiffened_shell_beam_history.py','_stiffened_shell_f06_increment_arithmetic.py')

def bindings():
 paths=[ROOT/'src/anyfesolver'/p for p in PRODUCTS]+[Path(__file__),
  ROOT/'tests/test_r8_f06_fibre_log_domain_dev.py',ROOT/'scripts/r8_f06_log_domain_check_dev.py',ROOT/'tests/conftest.py',
  ROOT/'tests/test_ge_beam3_native_definition.py',ROOT/'tests/test_ge_beam3_native_fibre_static_element.py']
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def element(*,paired=True,family='axial'):
 import numpy as np
 from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection,Fibre,FlowCurve
 from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
 from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
 from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY
 if family=='axial':fibres=(Fibre('0',0.,0.,.01,1e6,FlowCurve((0.,),(10.,),1e5)),)
 else:raise ValueError('this development fixture establishes axial fibres only')
 section=PhysicalFibreSection(fibres,np.diag(np.sqrt((0.,4e3,4e3,100.,100.,100.))))
 ref=CenteredCurvedBeam3ReferenceGeometry(np.array([[0.,-.5,.1],[.5,-.5,.1],[1.,-.5,.1]]),np.tile(np.eye(3),(3,1,1)))
 return NativeFibreStaticElement(1,(101,102,103),ref,section,order=4,arithmetic_policy=POLICY if paired else None)

def definition(*,paired=True):
 import numpy as np
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
 from anyfesolver._stiffened_shell_q4_algorithmic_pullback import POLICY
 shell=ss.ShellPanel((1,2,3,4),((0.,-.5,0.),(1.,-.5,0.),(1.,.5,0.),(0.,.5,0.)),
  ((1,2,3,4),),.07,1e6,.3,1000.)
 return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,shell,
  (NativeBeamDefinition.capture(element(paired=paired),np.eye(6)),),
  (ss.PointConnection(1,101),ss.PointConnection(2,103)),
  tuple(ss.FixedCoordinate('shell',n,k) for n in (1,4) for k in range(6)),q4_pullback_policy=POLICY)

def actions():
 from anyfesolver import stiffened_shell as ss
 return (ss.NodalAction('beam',103,(50.,0.,10.,0.,0.,0.)),)

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0])
 assert json.loads((out/'bindings.json').read_text())==bindings()
 package=DOC/'package'
 if stage=='wheel':
  import shutil
  snapshot=package/'snapshot';snapshot.mkdir(parents=True)
  source={}
  for p in (ROOT/'src/anyfesolver').rglob('*.py'):
   rel=p.relative_to(ROOT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
   content=p.read_bytes();target.write_bytes(content);source[rel.as_posix()]=sha(p)
   assert sha(target)==source[rel.as_posix()]
  metadata={}
  for name in ('pyproject.toml','setup.py','setup.cfg','README.md','LICENSE','COPYRIGHT','THIRD_PARTY_NOTICES.md','MANIFEST.in'):
   p=ROOT/name
   if p.exists():
    shutil.copyfile(p,snapshot/name);metadata[name]=sha(p);assert sha(snapshot/name)==metadata[name]
  write(package/'snapshot-bindings.json',source)
  write(package/'metadata-bindings.json',metadata)
  wheels=package/'wheels';site=package/'site';wheels.mkdir();site.mkdir()
  for label,cmd in (
   ('build',[sys.executable,'-m','pip','wheel',str(snapshot),'--no-index','--no-deps','--no-build-isolation','--no-cache-dir','-w',str(wheels)]),
   ('install',None)):
   if cmd is None:
    found=list(wheels.glob('*.whl'));assert len(found)==1
    cmd=[sys.executable,'-m','pip','install',str(found[0]),'--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(site)]
   write(out/(label+'-command.json'),cmd);subprocess.run(cmd,check=True)
  for rel,h in source.items():assert sha(site/Path(rel).relative_to('src'))==h and sha(ROOT/rel)==h
  write(package/'artifact.json',dict(wheel=str(found[0]),sha256=sha(found[0]),site=str(site),files=len(source)))
  return 0
 selected=package/'site' if stage=='installed' else ROOT/'src'
 sys.path[:0]=[str(selected),str(ROOT/'tests')]
 os.environ['F06_FIBRE_OUTPUT']=str(out)
 if stage in ('route','installed') and (DOC/'source-prefix.bin').exists():
  os.environ['F06_FIBRE_PREFIX']=str(DOC/'source-prefix.bin')
 runtime(out,'runtime-before',selected)
 import pytest
 try:
  targets=([str(ROOT/'tests/test_ge_beam3_native_definition.py')+'::test_reconstructible_native_definition',
   str(ROOT/'tests/test_ge_beam3_native_fibre_static_element.py')+'::test_real_store_commit_discard_and_fixed_origins',
   str(ROOT/'tests/test_ge_beam3_native_fibre_static_element.py')+'::test_native_subulp_common_translation_keeps_coordinate_low_part']
   if stage=='legacy' else [str(ROOT/'tests/test_r8_f06_fibre_log_domain_dev.py')])
  code=int(pytest.main(targets+['-q','-x','-o','pythonpath=',
   '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),
   '-k','public' if stage=='route' else '' if stage in ('installed','legacy') else 'spatial_derivative' if stage=='derivative' else 'not public']))
  return code
 finally:
  state=runtime(out,'runtime-after',selected)
  if stage=='installed':
   frozen=json.loads((package/'snapshot-bindings.json').read_text())
   for name,p in state['imports'].items():assert sha(p)==frozen['src/'+Path(p).relative_to(selected).as_posix()]

def scratch_temp(temp_root,name):
 root=Path(temp_root)
 if not root.is_absolute():raise ValueError('temp_root must be absolute: '+str(temp_root))
 resolved=root.resolve()
 if resolved!=ROOT and ROOT not in resolved.parents:raise ValueError('temp_root escapes the repository root: '+str(temp_root))
 if resolved.exists() and not resolved.is_dir():raise ValueError('temp_root exists and is not a directory: '+str(resolved))
 resolved.mkdir(parents=True,exist_ok=True)
 n=1
 while True:
  scratch=resolved/(name+'-tmp'+str(n))
  try:scratch.mkdir()
  except FileExistsError:
   if not scratch.is_dir():raise ValueError('pre-existing unsafe scratch allocation: '+str(scratch))
   n+=1
  else:return scratch

def run(stage, *, doc=None, entry=None, bind=None, authority_file='AUTHORITY.json', temp_root=None):
 import psutil
 doc=DOC if doc is None else Path(doc)
 authority=json.loads((doc/authority_file).read_text());assert authority['status']=='AUTHORIZED'
 doc.mkdir(parents=True,exist_ok=True);lp=doc/'ledger.json'
 budget=float(authority.get('budget_s',300.));reserve=float(authority.get('reserve_s',60.));process_cap=float(authority.get('process_cap_s',30.))
 assert 0.<reserve<budget and 0.<process_cap<=budget-reserve
 ledger=json.loads(lp.read_text()) if lp.exists() else dict(status='OPEN',budget_s=budget,reserve_s=reserve,used_s=0.,attempts=[])
 assert ledger['status']=='OPEN'
 assert ledger['budget_s']==budget and ledger['reserve_s']==reserve
 final=ledger.get('phase')=='FINAL_VERIFICATION'
 if final:assert stage=='wheel' or stage.startswith('installed') or (stage=='verify-framing' and (doc/'ARTIFACT_REPAIR.json').exists()), 'final reserve excludes primary numerical work'
 cap=min(process_cap,ledger['budget_s']-(0. if final else ledger['reserve_s'])-ledger['used_s']);assert cap>0
 out=doc/('attempt-'+stage+str(1+sum(a['stage']==stage for a in ledger['attempts'])));out.mkdir()
 bound=(bindings if bind is None else bind)();write(out/'bindings.json',bound)
 snapshots=out/'source-snapshots';snapshots.mkdir()
 for rel in bound:(snapshots/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
 inputs={}
 if stage in ('route','installed') and (doc/'source-prefix.bin').exists():inputs[str(doc/'source-prefix.bin')]=sha(doc/'source-prefix.bin')
 if temp_root is None:temp=out/'temp';temp.mkdir()
 else:temp=scratch_temp(temp_root,out.name)
 env=os.environ.copy();env.update(ENV);env.update(TEMP=str(temp),TMP=str(temp))
 cmd=[sys.executable,str(Path(__file__ if entry is None else entry).resolve()),'--child','--out',str(out),'--stage',stage]
 receipt=dict(command=cmd,stage=stage,bindings=bound,inputs=inputs,authority_sha256=sha(doc/authority_file),authority_file=authority_file,cap_s=cap,memory_cap_bytes=2147483648,
  environment={k:env[k] for k in (*ENV,'TEMP','TMP')},python_sha256=sha(sys.executable),peak_tree_rss_bytes=0,
  temp_root=None if temp_root is None else str(Path(temp_root).resolve()),temp=str(temp))
 write(out/'launch.json',receipt);write(lp,ledger)
 start=time.perf_counter_ns();proc=None;code=None;failure=None
 try:
  with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr);process=psutil.Process(proc.pid);process.cpu_affinity([0])
   while proc.poll() is None:
    try:members=[process]+process.children(recursive=True)
    except psutil.NoSuchProcess:members=[]
    rss=0
    for member in members:
     try:rss+=member.memory_info().rss
     except psutil.NoSuchProcess:pass
    receipt['peak_tree_rss_bytes']=max(receipt['peak_tree_rss_bytes'],rss)
    if rss>2147483648:failure='MEMORY_LIMIT'
    elif (time.perf_counter_ns()-start)/1e9>=cap:failure='TIMEOUT'
    if failure:
     for member in reversed(members):
      try:member.kill()
      except psutil.NoSuchProcess:pass
     break
    time.sleep(.02)
   code=proc.wait()
 except BaseException as exc:
  failure=repr(exc)
  if proc is not None and proc.poll() is None:
   for member in reversed([psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)):
    try:member.kill()
    except psutil.NoSuchProcess:pass
   proc.wait()
 finally:
  wall=(time.perf_counter_ns()-start)/1e9;ledger['used_s']+=wall
  receipt.update(status='PASS' if code==0 and failure is None else 'FAIL',exit_code=code,failure=failure,wall_s=wall,
   cumulative_wall_s=ledger['used_s'],remaining_s=ledger['budget_s']-ledger['used_s'])
  ledger['attempts'].append(dict(stage=stage,path=out.name,status=receipt['status'],wall_s=wall))
  for name in ('stdout.log','stderr.log','junit.xml','runtime-before.json','runtime-after.json'):
   if (out/name).exists():receipt[name+'_sha256']=sha(out/name)
  write(out/'receipt.json',receipt);write(lp,ledger)
 print(json.dumps({k:receipt[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
 return int(receipt['status']!='PASS')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',choices=('source','derivative','route','legacy','wheel','installed','close'),required=True);a=p.parse_args()
 if a.stage=='close':
  lp=DOC/'ledger.json';v=json.loads(lp.read_text());assert v['status']=='OPEN'
  v.update(status='CLOSED',unused_discarded_s=v['budget_s']-v['used_s']);write(lp,v)
 else:raise SystemExit(child(Path(a.out),a.stage) if a.child else run(a.stage))
