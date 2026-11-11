"""One bounded mixed/J2 nodal adaptive candidate and installed physical path."""
from pathlib import Path
import argparse,json,os,sys,hashlib,zipfile,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'mixed-adaptive-v1';PACKAGE=DOC/'artifact-repair-v3/package'
INPUT=BASE/'coupled-resume-v1/attempt-installed-reload1/anchored.bin'
INPUT_SHA='0e45fec011a86a0cf80be93d2a311d7eff2604d4dec01b5a3bc846a8b39359ec'

def bindings():
 paths=[ROOT/'src/anyfesolver'/p for p in (*infra.PRODUCTS,'_stiffened_shell_increment_control.py',
  '_stiffened_shell_staged_load_experimental.py','_stiffened_shell_q4_algorithmic_pullback.py','_stiffened_shell_algorithmic_pullback.py')]
 paths += [ROOT/'scripts'/p for p in ('r8_f06_cyclic_material_dev.py','r8_f06_native_material_breadth_dev.py',
  'r8_f06_coupled_resume_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_mixed_adaptive_dev.py',ROOT/'tests/test_r8_f06_coupled_resume_dev.py',
  INPUT,DOC/'recovered-cached-prefix.bin',DOC/'FRAMING_FAILURE.json',DOC/'RESUME_AUTHORITY.json',
  DOC/'attempt-installed-path1/accepted-00.npz',DOC/'attempt-installed-path1/accepted-00-recovery.json',DOC/'AUTHORITY.json',DOC/'WHEEL_AUTHORITY.json',DOC/'PATH_AUTHORITY.json',DOC/'ADOPTED_CONTRACT.md']
 for relative in ('attempt-installed-path1/accepted-00.bin','attempt-installed-path1/initial.npz',
  'attempt-installed-path1/accepted-journal.json','attempt-installed-path1/rejected-journal.json',
  'attempt-installed-controls2/actual-rebased.bin','attempt-installed-resume1/terminal.bin',
  'attempt-installed-resume1/terminal.npz','attempt-installed-resume1/terminal-recovery.json'):
  candidate=DOC/relative
  if candidate.exists():paths.append(candidate)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def artifact():
 frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text());wheel=next((PACKAGE/'wheels').glob('*.whl'))
 with zipfile.ZipFile(wheel) as z:
  for rel,h in frozen.items():
   name=rel.removeprefix('src/')
   assert sha(ROOT/rel)==sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/name)==h,rel
   assert hashlib.sha256(z.read(name)).hexdigest()==h,rel
 return dict(files=len(frozen),wheel_sha256=sha(wheel),manifest_sha256=sha(PACKAGE/'snapshot-bindings.json'))

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings();assert sha(INPUT)==INPUT_SHA
 if stage=='wheel':
  infra.DOC=DOC/'artifact-repair-v3';infra.bindings=bindings;original=subprocess.run;timings=[]
  def measured(*args,**kwargs):
   start=time.perf_counter_ns()
   try:
    with (out/('pip-%d.log'%len(timings))).open('wb') as stream:
     return original(*args,**kwargs,stdout=stream,stderr=subprocess.STDOUT)
   finally:timings.append(dict(command=args[0],wall_s=(time.perf_counter_ns()-start)/1e9));write(out/'pip-timings.json',timings)
  runtime(out,'runtime-before',ROOT/'src');subprocess.run=measured
  try:return infra.child(out,stage)
  finally:subprocess.run=original;runtime(out,'runtime-after',ROOT/'src')
 installed=stage.startswith('installed');selected=PACKAGE/'site' if installed else ROOT/'src'
 if installed:write(out/'artifact-before.json',artifact())
 sys.path.insert(0,str(selected));os.environ.update(F06_MIXED_ADAPTIVE_OUTPUT=str(out),F06_MIXED_ADAPTIVE_STAGE=stage)
 runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_mixed_adaptive_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),'-k',
  'physical' if stage in ('installed-path','installed-resume') else 'controls']))
 finally:
  rt=runtime(out,'runtime-after',selected)
  if installed:
   frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
   for n,p in rt['imports'].items():assert rt['import_sha256'][n]==frozen['src/'+Path(p).relative_to(selected).as_posix()]
   write(out/'artifact-after.json',artifact())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('source-controls','wheel','installed-controls','installed-path','installed-resume'))
 p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 if a.stage=='installed-resume':assert not list(DOC.glob('attempt-installed-resume*')),'one interrupted-path continuation only'
 if a.stage=='installed-path':assert not list(DOC.glob('attempt-installed-path*')),'one physical path only'
 for required in (() if a.stage=='source-controls' else ('source-controls',) if a.stage=='wheel' else ('source-controls','wheel') if a.stage=='installed-controls' else ('source-controls','wheel','installed-controls')):
  attempts=sorted(DOC.glob('attempt-'+required+'*/receipt.json'));assert attempts and json.loads(attempts[-1].read_text())['status']=='PASS'
 authority='RESUME_AUTHORITY.json' if a.stage=='installed-resume' else 'PATH_AUTHORITY.json' if a.stage=='installed-path' else 'WHEEL_AUTHORITY.json' if a.stage=='wheel' else 'AUTHORITY.json'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings,authority_file=authority))
