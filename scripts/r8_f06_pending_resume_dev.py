"""One bounded installed material-pending resume; immutable prior inputs."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'pending-resume-v1'
PACKAGE=BASE/'rebased-continuation-v1/artifact-repair-v2/package'
INPUT=BASE/'adaptive-material-v1/attempt-primary2/accepted-00.bin'
ORACLE=BASE/'adaptive-material-v1/attempt-primary2'

def bindings():
 paths=[ROOT/'src/anyfesolver'/n for n in (*infra.PRODUCTS,'_stiffened_shell_increment_control.py','_stiffened_shell_staged_load_experimental.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_pending_resume_dev.py',INPUT,
  ORACLE/'final-prefix.bin',ORACLE/'histories-00.npz',ORACLE/'histories-01.npz',ORACLE/'accepted.json',
  PACKAGE/'artifact.json',PACKAGE/'snapshot-bindings.json',DOC/'READINESS.json']
 paths += [ROOT/'scripts'/n for n in ('r8_f06_adaptive_material_dev.py','r8_f06_cyclic_material_dev.py',
  'r8_f06_native_material_breadth_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 for name in ('partial-accepted-2_3125.bin','partial-histories.npz'):
  if (DOC/name).exists():paths.append(DOC/name)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0])
 assert json.loads((out/'bindings.json').read_text())==bindings()
 selected=PACKAGE/'site';sys.path[:0]=[str(selected)]
 os.environ['F06_PENDING_OUTPUT']=str(out)
 runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_pending_resume_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),'-k',
  'partial_restore' if stage=='installed-partial-restore' else 'material_pending_resume']))
 finally:
  state=runtime(out,'runtime-after',selected)
  frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
  for path in state['imports'].values():assert sha(path)==frozen['src/'+Path(path).relative_to(selected).as_posix()]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',default='installed-pending');p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 assert a.stage in ('installed-pending','installed-partial-restore')
 prior=list(DOC.glob('attempt-'+a.stage+'*'))
 assert not prior or (a.stage=='installed-partial-restore' and len(prior)==1 and
  (DOC/'RESTORE_REPAIR_AUTHORITY.json').exists()),'no unapproved repeat'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings))
