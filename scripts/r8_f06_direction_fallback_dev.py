"""One connected source/artifact verification ledger for direction fallback."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'direction-fallback-v1'
INPUTS={'pending':BASE/'pending-resume-v1/attempt-installed-pending1/initial-prefix.bin',
 'reversed':BASE/'adaptive-material-v1/attempt-installed-lifecycle1/reversed-rebased.bin'}

def bindings():
 paths=[ROOT/'src/anyfesolver'/n for n in (*infra.PRODUCTS,'_stiffened_shell_increment_control.py','_stiffened_shell_staged_load_experimental.py')]
 paths += [DOC/'AUTHORITY_DELTA1.json',Path(__file__),ROOT/'tests/test_r8_f06_direction_fallback_dev.py',*INPUTS.values()]
 paths += [ROOT/'scripts'/n for n in ('r8_f06_adaptive_material_dev.py','r8_f06_cyclic_material_dev.py',
  'r8_f06_native_material_breadth_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings()
 if stage=='wheel':
  infra.DOC=DOC;infra.bindings=bindings
  runtime(out,'runtime-before',ROOT/'src')
  try:return infra.child(out,stage)
  finally:runtime(out,'runtime-after',ROOT/'src')
 selected=DOC/'package/site' if stage.startswith('installed') else ROOT/'src'
 sys.path[:0]=[str(selected)];os.environ.update(F06_FALLBACK_OUTPUT=str(out),F06_FALLBACK_CASE=stage.split('-')[-1])
 runtime(out,'runtime-before',selected)
 import pytest
 try:return int(pytest.main([str(ROOT/'tests/test_r8_f06_direction_fallback_dev.py'),'-q','-x','-o','pythonpath=',
  '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),'-k',
  'controls' if stage.endswith('controls') else 'route']))
 finally:
  state=runtime(out,'runtime-after',selected)
  if stage.startswith('installed'):
   frozen=json.loads((DOC/'package/snapshot-bindings.json').read_text())
   for path in state['imports'].values():assert sha(path)==frozen['src/'+Path(path).relative_to(selected).as_posix()]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 assert a.stage in ('source-controls','source-pending','source-reversed','wheel','installed-controls','installed-pending','installed-reversed','source-observed-pending','installed-observed-pending')
 assert not list(DOC.glob('attempt-'+a.stage+'*')),'no automatic repeated stage'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings,
  authority_file='AUTHORITY_DELTA1.json' if a.stage.endswith('observed-pending') else 'AUTHORITY.json'))
