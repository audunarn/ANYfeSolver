"""Bounded same-parent direction observations; explicit slot release required."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'pending-direction-v1'
PACKAGE=BASE/'rebased-continuation-v1/artifact-repair-v2/package'
INPUT=BASE/'pending-resume-v1/attempt-installed-pending1/initial-prefix.bin'
PARTIAL=BASE/'pending-resume-v1/partial-accepted-2_3125.bin'
HISTORY=BASE/'adaptive-material-v1/attempt-primary2/histories-00.npz'

def bindings():
 paths=[ROOT/'src/anyfesolver'/n for n in (*infra.PRODUCTS,'_stiffened_shell_increment_control.py',
  '_stiffened_shell_staged_load_experimental.py','_stiffened_shell_rotation_charts.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_pending_direction_dev.py',INPUT,PARTIAL,HISTORY,
  PACKAGE/'artifact.json',PACKAGE/'snapshot-bindings.json']
 paths += [ROOT/'scripts'/n for n in ('r8_f06_adaptive_material_dev.py','r8_f06_cyclic_material_dev.py',
  'r8_f06_native_material_breadth_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def child(out):
 import psutil
 psutil.Process().cpu_affinity([0])
 assert json.loads((out/'bindings.json').read_text())==bindings()
 selected=PACKAGE/'site';sys.path[:0]=[str(selected)]
 os.environ['F06_PENDING_DIRECTION_OUTPUT']=str(out)
 runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_pending_direction_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
 finally:
  state=runtime(out,'runtime-after',selected)
  frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
  for path in state['imports'].values():assert sha(path)==frozen['src/'+Path(path).relative_to(selected).as_posix()]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',default='direction');p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 assert (DOC/'SLOT_RELEASE.json').exists(),'numerical slot remains held'
 if a.child:raise SystemExit(child(Path(a.out)))
 assert not list(DOC.glob('attempt-direction*')),'one discriminator only'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings))
