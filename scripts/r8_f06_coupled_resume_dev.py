"""Bounded installed coupled unload/reload; no product or constitutive edits."""
from pathlib import Path
import argparse,json,os,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'coupled-resume-v1'
PACKAGE=BASE/'direction-fallback-v1/package'
PREFIX=BASE/'cyclic-material-v1/coupled-prefix.bin'
PREFIX_SHA='8a92fd3ad248b16c3f18e61b7c136b4e50e242e8ae9ed35328a25a4189055737'
WHEEL_SHA='275065004115481b295997328880ffef53c5a8cc0b2076fd76dd39332ade21ea'

def artifact():
 import hashlib
 frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
 wheel=next((PACKAGE/'wheels').glob('*.whl'));assert sha(wheel)==WHEEL_SHA
 with zipfile.ZipFile(wheel) as archive:
  for rel,digest in frozen.items():
   name=rel.removeprefix('src/')
   assert sha(ROOT/rel)==sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/name)==digest,rel
   assert hashlib.sha256(archive.read(name)).hexdigest()==digest,rel
 assert len(frozen)==633 and sha(PREFIX)==PREFIX_SHA
 return dict(files=len(frozen),wheel_sha256=WHEEL_SHA,manifest_sha256=sha(PACKAGE/'snapshot-bindings.json'),prefix_sha256=PREFIX_SHA)

def bindings():
 paths=[ROOT/'src/anyfesolver'/p for p in (*infra.PRODUCTS,'_stiffened_shell_q4_algorithmic_pullback.py',
  '_stiffened_shell_algorithmic_pullback.py','_stiffened_shell_increment_control.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_coupled_resume_dev.py',PREFIX,
  PACKAGE/'snapshot-bindings.json',DOC/'AUTHORITY.json',DOC/'CONTROL_AUTHORITY.json',DOC/'SLOT_TRANSFER.json']
 paths += [ROOT/'scripts'/p for p in ('r8_f06_cyclic_material_dev.py','r8_f06_native_material_breadth_dev.py',
  'r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 for stage in ('unload','reload'):
  p=DOC/f'attempt-installed-{stage}1/accepted.bin'
  if p.exists():paths.append(p)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings()
 write(out/'artifact-before.json',artifact())
 selected=PACKAGE/'site';sys.path.insert(0,str(selected))
 os.environ.update(F06_COUPLED_RESUME_OUTPUT=str(out),F06_COUPLED_RESUME_STAGE=stage)
 runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_coupled_resume_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),
  '-k',stage.split('-')[-1] if stage.endswith(('rollback','tamper')) else 'history']))
 finally:
  state=runtime(out,'runtime-after',selected);frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
  for name,path in state['imports'].items():assert state['import_sha256'][name]==frozen['src/'+Path(path).relative_to(selected).as_posix()]
  write(out/'artifact-after.json',artifact())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('installed-unload','installed-reload','installed-rollback','installed-tamper'))
 p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 assert (DOC/'SLOT_TRANSFER.json').exists(),'numerical slot has not been transferred'
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 assert not list(DOC.glob('attempt-'+a.stage+'*')),'no automatic physical retry'
 for stage in (() if a.stage.endswith('unload') else ('unload',) if a.stage.endswith('reload') else ('unload','reload')):
  assert json.loads((DOC/f'attempt-installed-{stage}1/receipt.json').read_text())['status']=='PASS'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings,
  authority_file='CONTROL_AUTHORITY.json' if a.stage.endswith(('rollback','tamper')) else 'AUTHORITY.json'))
