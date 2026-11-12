"""Installed, explicit prior-peak/reload history; existing fixed solver only."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_coupled_resume_dev import artifact,PACKAGE
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'coupled-staged-v1'
INPUT=BASE/'coupled-resume-v1/attempt-installed-reload1/anchored.bin'
INPUT_SHA='0e45fec011a86a0cf80be93d2a311d7eff2604d4dec01b5a3bc846a8b39359ec'

def bindings():
 paths=[ROOT/'src/anyfesolver'/p for p in (*infra.PRODUCTS,'_stiffened_shell_q4_algorithmic_pullback.py',
  '_stiffened_shell_algorithmic_pullback.py','_stiffened_shell_increment_control.py')]
 paths += [ROOT/'scripts'/p for p in ('r8_f06_coupled_resume_dev.py','r8_f06_cyclic_material_dev.py',
  'r8_f06_native_material_breadth_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_coupled_staged_dev.py',ROOT/'tests/test_r8_f06_coupled_resume_dev.py',
  INPUT,PACKAGE/'snapshot-bindings.json',DOC/'AUTHORITY.json',DOC/'CONTROL_AUTHORITY.json']
 for stage in ('peak','reload'):
  p=DOC/f'attempt-installed-{stage}1/accepted.bin'
  if p.exists():paths.append(p)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings();assert sha(INPUT)==INPUT_SHA
 write(out/'artifact-before.json',artifact());selected=PACKAGE/'site';sys.path.insert(0,str(selected))
 os.environ.update(F06_COUPLED_STAGED_OUTPUT=str(out),F06_COUPLED_STAGED_STAGE=stage)
 runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_coupled_staged_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),
  '-k',stage.split('-')[-1] if stage.endswith(('rollback','tamper')) else 'history']))
 finally:
  state=runtime(out,'runtime-after',selected);frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
  for name,path in state['imports'].items():assert state['import_sha256'][name]==frozen['src/'+Path(path).relative_to(selected).as_posix()]
  write(out/'artifact-after.json',artifact())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('installed-peak','installed-reload','installed-rollback','installed-tamper'))
 p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 assert not list(DOC.glob('attempt-'+a.stage+'*')),'one attempt per physical stage, no automatic retry'
 for stage in (() if a.stage.endswith('peak') else ('peak',) if a.stage.endswith('reload') else ('peak','reload')):
  assert json.loads((DOC/f'attempt-installed-{stage}1/receipt.json').read_text())['status']=='PASS'
 controls=a.stage.endswith(('rollback','tamper'));authority='CONTROL_AUTHORITY.json' if controls else 'AUTHORITY.json'
 ledger=json.loads((DOC/'ledger.json').read_text());cap=30 if controls else 90
 assert ledger['budget_s']-ledger['reserve_s']-ledger['used_s']>=cap,'full declared cap must fit remaining reserve'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings,authority_file=authority))
