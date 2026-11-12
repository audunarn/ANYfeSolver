"""Fresh bounded public cyclic material development; no historical campaign."""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts import r8_f06_native_material_breadth_dev as breadth
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
DOC=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/cyclic-material-v1'

def bindings():
 paths=[ROOT/'src/anyfesolver'/p for p in (*infra.PRODUCTS,'_stiffened_shell_q4_algorithmic_pullback.py',
  '_stiffened_shell_algorithmic_pullback.py','_stiffened_shell_staged_load_experimental.py')]
 paths += [Path(__file__),ROOT/'scripts/r8_f06_fibre_log_domain_dev.py',ROOT/'scripts/r8_f06_native_material_breadth_dev.py',
  ROOT/'scripts/r8_f06_log_domain_check_dev.py',ROOT/'tests/test_r8_f06_cyclic_material_dev.py',ROOT/'tests/conftest.py']
 paths += sorted(DOC.glob('*-prefix.bin'))
 z=DOC/'attempt-source-bend-z2/failure-state.bin'
 if z.exists():paths.append(z)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def definition(case):
 base=breadth.definition('mixed' if case=='coupled' else 'biaxial')
 if case=='coupled':
  from anymaterial import DNVC208MaterialCurve
  p=1000.
  base=replace(base,shell=replace(base.shell,j2_curve=DNVC208MaterialCurve(p,1.05*p,1.1*p,.005,.01,4*p,.2)))
 return base

def actions(case,scale=1.):
 from anyfesolver import stiffened_shell as ss
 vector=(0.,10.,0.,0.,0.,0.) if case=='bend-y' else (0.,0.,10.,0.,0.,0.)
 if case=='coupled':vector=(50.,0.,10.,0.,0.,0.)
 nodes=(103,203) if case=='coupled' else (103,)
 result=tuple(ss.NodalAction('beam',node,tuple(scale*v for v in vector)) for node in nodes)
 # Explicit membrane action exercises the shell material independently of the
 # parallel beam load path: 160 N / .07 m2 exceeds the 1000 Pa shell yield.
 if case=='coupled':result+=tuple(ss.NodalAction('shell',node,(scale*200.,0.,0.,0.,0.,0.)) for node in (2,3))
 return result

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings()
 package_doc=DOC/'artifact-repair-v2' if (DOC/'ARTIFACT_REPAIR.json').exists() else DOC
 if stage=='wheel':
  infra.DOC=package_doc;infra.bindings=bindings
  return infra.child(out,stage)
 installed=stage.startswith('installed');selected=package_doc/'package/site' if installed else ROOT/'src'
 sys.path[:0]=[str(selected),str(ROOT/'tests')]
 os.environ.update(F06_CYCLIC_OUTPUT=str(out),F06_CYCLIC_CASE=stage.split('-',1)[1])
 os.environ['F06_CYCLIC_INSTALLED']='1' if installed else '0'
 runtime(out,'runtime-before',selected)
 import pytest
 selector='framing' if stage.endswith('-framing') else 'installed and not framing' if installed or stage.startswith('verify-') else 'not installed'
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_cyclic_material_dev.py'),'-q','-x','-k',selector,'-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
 finally:
  state=runtime(out,'runtime-after',selected)
  if installed:
   frozen=json.loads((package_doc/'package/snapshot-bindings.json').read_text())
   for name,p in state['imports'].items():assert sha(p)==frozen['src/'+Path(p).relative_to(selected).as_posix()]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.stage=='close':
  lp=DOC/'ledger.json';v=json.loads(lp.read_text());assert v['status']=='OPEN'
  v.update(status='CLOSED',unused_discarded_s=v['budget_s']-v['used_s']);write(lp,v)
 elif a.child:raise SystemExit(child(Path(a.out),a.stage))
 else:raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings))
