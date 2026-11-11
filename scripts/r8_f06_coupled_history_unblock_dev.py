"""One installed same-parent coupled direction diagnostic, no accepted trial."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
import hashlib,zipfile
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'coupled-history-unblock-v1'
PACKAGE=BASE/'mixed-adaptive-v1/artifact-repair-v2/package'
INPUT=BASE/'mixed-adaptive-v1/attempt-installed-resume1/terminal.bin'
INPUT_SHA='970e0360f22ed348ca2eaa7eee0d2d168d6d686d0a91a750f3ccd31794865f91'

def bindings():
 paths=[ROOT/'src/anyfesolver'/n for n in (*infra.PRODUCTS,'_stiffened_shell_q4_algorithmic_pullback.py',
  '_stiffened_shell_algorithmic_pullback.py','_stiffened_shell_increment_control.py','_ge_beam3_generalized_static_boundary.py')]
 paths += [ROOT/'scripts'/n for n in ('r8_f06_coupled_resume_dev.py','r8_f06_cyclic_material_dev.py',
  'r8_f06_native_material_breadth_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_coupled_history_unblock_dev.py',INPUT,DOC/'AUTHORITY.json',DOC/'DIAGNOSTIC_AMENDMENT.json',PACKAGE/'snapshot-bindings.json']
 if (DOC/'LOCAL_CHAIN_AMENDMENT.json').exists():
  paths += [DOC/'LOCAL_CHAIN_AMENDMENT.json',DOC/'LOCAL_CHAIN_REPAIR.json']+list((DOC/'attempt-diagnostic1').glob('production-13*'))+[DOC/'attempt-diagnostic1/last-parent-direction.npz',DOC/'attempt-diagnostic1/accepted-origin.npz']
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def artifact():
 frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text());wheel=next((PACKAGE/'wheels').glob('*.whl'))
 with zipfile.ZipFile(wheel) as z:
  for rel,h in frozen.items():
   name=rel.removeprefix('src/')
   assert sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/name)==hashlib.sha256(z.read(name)).hexdigest()==h,rel
 return dict(files=len(frozen),wheel_sha256=sha(wheel),manifest_sha256=sha(PACKAGE/'snapshot-bindings.json'),
  current_source_differences=[rel for rel,h in frozen.items() if not (ROOT/rel).exists() or sha(ROOT/rel)!=h],
  scope='Actual failed-action artifact2; current source differences recorded, not asserted equal')

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings();assert sha(INPUT)==INPUT_SHA
 write(out/'artifact-before.json',artifact());selected=PACKAGE/'site';sys.path.insert(0,str(selected))
 os.environ['F06_COUPLED_HISTORY_UNBLOCK_OUTPUT']=str(out);os.environ['F06_COUPLED_HISTORY_UNBLOCK_STAGE']=stage;runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_coupled_history_unblock_dev.py'),'-q','-x','-k','local_chain' if stage=='local-chain' else 'direction','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
 finally:
  state=runtime(out,'runtime-after',selected);frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
  for name,path in state['imports'].items():assert state['import_sha256'][name]==frozen['src/'+Path(path).relative_to(selected).as_posix()]
  write(out/'artifact-after.json',artifact())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',default='diagnostic');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 assert a.stage in ('diagnostic','local-chain')
 if a.stage=='diagnostic':assert not list(DOC.glob('attempt-diagnostic*'))
 else:assert len(list(DOC.glob('attempt-local-chain*')))<2,'one initial plus guard-preserving harness repair'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings,authority_file='LOCAL_CHAIN_REPAIR.json' if a.stage=='local-chain' else 'DIAGNOSTIC_AMENDMENT.json'))
