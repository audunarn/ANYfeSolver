"""Bounded adaptive material development; each run uses its own fresh ledger."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts import r8_f06_cyclic_material_dev as cyclic
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
DOC=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/adaptive-material-v1'
PREFIX=cyclic.DOC/'attempt-source-bend-z2/failure-state.bin'

def bindings():
 paths=[ROOT/'src/anyfesolver'/name for name in (*infra.PRODUCTS,'_stiffened_shell_increment_control.py','_stiffened_shell_staged_load_experimental.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_adaptive_material_dev.py',PREFIX,
  ROOT/'scripts/r8_f06_cyclic_material_dev.py',ROOT/'scripts/r8_f06_native_material_breadth_dev.py',
  ROOT/'scripts/r8_f06_fibre_log_domain_dev.py',ROOT/'scripts/r8_f06_log_domain_check_dev.py']
 for f in (DOC/'attempt-source-lifecycle2/accepted.bin',DOC/'attempt-source-lifecycle2/anchored.bin',DOC/'attempt-primary2/accepted-00.bin',DOC/'attempt-primary2/final-prefix.bin',DOC/'attempt-installed-lifecycle1/reversed-rebased.bin'):
  if f.exists():paths.append(f)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def programme(owner):
 from anyfesolver import stiffened_shell as ss
 return ss.StagedDeadLoadProgramme(owner.preflight()['identity'],tuple(cyclic.actions('bend-z',a) for a in (.4,.2,-.4,-.41)))

def paired_value(row, offset=0):
 from fractions import Fraction
 return Fraction(row[offset])+Fraction(row[offset+1])

def persist_accepted(out,owner,result,index):
 """Durably record a published endpoint before observer checks can fail."""
 import numpy as np
 from dataclasses import asdict
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 d=owner._finite_driver;prefix=out/('accepted-%02d.bin'%index)
 prefix.write_bytes(owner.checkpoint())
 np.savez(out/('histories-%02d.npz'%index),**dict(history_arrays(owner._beam_parts,d._beam_predecessors,'origin')+
  history_arrays(owner._beam_parts,d._beam_origins,'accepted')))
 applied=d._programme.load_at(result.load_factor)
 row=dict(index=index,factor=result.load_factor,prefix=prefix.name,sha256=sha(prefix),
  actual_load=applied[applied!=0].tolist(),equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
  attempts=[asdict(a) for a in result.increment_attempts])
 write(out/('observed-accepted-%02d.json'%index),row)
 return prefix,row

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings()
 if stage=='wheel':
  import time,subprocess
  infra.DOC=DOC;infra.bindings=bindings
  original_run=subprocess.run;timings=[]
  def measured(*args,**kwargs):
   start=time.perf_counter_ns()
   try:return original_run(*args,**kwargs)
   finally:
    timings.append(dict(command=args[0],wall_s=(time.perf_counter_ns()-start)/1e9))
    write(out/'pip-timings.json',timings)
  runtime(out,'runtime-before',ROOT/'src');subprocess.run=measured
  try:return infra.child(out,stage)
  finally:subprocess.run=original_run;runtime(out,'runtime-after',ROOT/'src')
 selected=DOC/'package/site' if stage.startswith('installed') else ROOT/'src'
 sys.path[:0]=[str(selected),str(ROOT/'tests')]
 os.environ['F06_ADAPTIVE_OUTPUT']=str(out);os.environ['F06_ADAPTIVE_STAGE']=stage
 runtime(out,'runtime-before',selected)
 import pytest
 try:
  return int(pytest.main([str(ROOT/'tests/test_r8_f06_adaptive_material_dev.py'),'-q','-x','-o','pythonpath=',
   '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),'-k',stage.replace('-','_')]))
 finally:
  state=runtime(out,'runtime-after',selected)
  if stage.startswith('installed'):
   frozen=json.loads((DOC/'package/snapshot-bindings.json').read_text())
   for path in state['imports'].values():assert sha(path)==frozen['src/'+Path(path).relative_to(selected).as_posix()]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 if a.stage=='primary':
  prior=list(DOC.glob('attempt-primary*'))
  assert not prior or (len(prior)==1 and (DOC/'REPAIR_AUTHORITY.json').exists()),'no unapproved repeated primary'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings))
