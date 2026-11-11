"""One installed same-parent coupled direction diagnostic, no accepted trial."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_coupled_resume_dev import artifact,PACKAGE
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'coupled-direction-v1'
INPUT=BASE/'coupled-resume-v1/attempt-installed-reload1/anchored.bin'
INPUT_SHA='0e45fec011a86a0cf80be93d2a311d7eff2604d4dec01b5a3bc846a8b39359ec'

def bindings():
 paths=[ROOT/'src/anyfesolver'/n for n in (*infra.PRODUCTS,'_stiffened_shell_q4_algorithmic_pullback.py',
  '_stiffened_shell_algorithmic_pullback.py','_stiffened_shell_increment_control.py','_ge_beam3_generalized_static_boundary.py')]
 paths += [ROOT/'scripts'/n for n in ('r8_f06_coupled_resume_dev.py','r8_f06_cyclic_material_dev.py',
  'r8_f06_native_material_breadth_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_coupled_direction_dev.py',INPUT,DOC/'AUTHORITY.json',PACKAGE/'snapshot-bindings.json']
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def child(out):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings();assert sha(INPUT)==INPUT_SHA
 write(out/'artifact-before.json',artifact());selected=PACKAGE/'site';sys.path.insert(0,str(selected))
 os.environ['F06_COUPLED_DIRECTION_OUTPUT']=str(out);runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_coupled_direction_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
 finally:
  state=runtime(out,'runtime-after',selected);frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
  for name,path in state['imports'].items():assert state['import_sha256'][name]==frozen['src/'+Path(path).relative_to(selected).as_posix()]
  write(out/'artifact-after.json',artifact())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',default='diagnostic');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out)))
 assert a.stage=='diagnostic' and not list(DOC.glob('attempt-diagnostic*')),'one physical diagnostic only'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings))
