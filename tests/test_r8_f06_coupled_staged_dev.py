"""Separate explicit coupled .2 -> .4 -> .41 history, not the failed direct step."""
import os,json,runpy
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_COUPLED_STAGED_OUTPUT' not in os.environ,reason='bounded installed supervisor required')
# Reuse immutable capture/history/refusal helpers without executing their tests.
_helpers=runpy.run_path(str(Path(__file__).with_name('test_r8_f06_coupled_resume_dev.py')))
capture,exact,save,activity,identity,unchanged=(_helpers[k] for k in ('capture','exact','save','activity','identity','unchanged'))

def test_history():
 import numpy as np,sys
 from dataclasses import asdict
 from scripts.r8_f06_coupled_staged_dev import DOC,INPUT,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 out=Path(os.environ['F06_COUPLED_STAGED_OUTPUT']);peak=os.environ['F06_COUPLED_STAGED_STAGE'].endswith('peak')
 source=INPUT if peak else DOC/'attempt-installed-peak1/accepted.bin';target=.4 if peak else .41
 model=definition('coupled');owner=ss.create_analysis(definition=model)
 try:
  result=owner.restore(source.read_bytes());d=owner._finite_driver;initial=save(out,'initial',owner,result)
  assert owner.checkpoint()==source.read_bytes() and d._adaptive_state is None
  assert not np.any(d._state.q_low[:owner._shell_count])
  if not peak:
   owner.rebase_rotations(owner._generation);anchored=save(out,'anchored',owner,owner.recover())
   exact(initial,anchored,[k for k in initial if k.startswith(('origin_','accepted_')) or k.endswith('_low')]+['reactions'])
   chart=owner.checkpoint();fresh=ss.create_analysis(definition=model)
   try:fresh.restore(chart);assert fresh.checkpoint()==chart;exact(anchored,capture(fresh))
   finally:fresh.close()
  parent_beam=d._beam_origins;parent_shell=d._shell_origins;before=identity(owner)
  original=d._assemble;advance=d._advance;trials={};trace=out/'trace.jsonl'
  def advanced(state,delta,amount):
   frame=sys._getframe(1).f_locals;candidate=advance(state,delta,amount)
   trials[id(candidate)]=dict(amount=float(amount),backtrack=frame['backtrack']);return candidate
  def assembled(state,*args,**kwargs):
   frame=sys._getframe(2).f_locals;value=original(state,*args,**kwargs)
   row=dict(factor=float(args[0]),iteration=frame.get('iteration'),trial=trials.pop(id(state),{}),
    equilibrium=float(value[2]),gap=float(value[3]),counts=dict(frame.get('counts',{})))
   with trace.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
   return value
  d._assemble=assembled;d._advance=advanced
  try:result=owner.solve(actions('coupled'),(target,),max_iterations=20,max_backtracks=8,on_accepted=lambda r:save(out,'accepted',owner,r))
  except BaseException as error:
   save(out,'terminal-failure',owner,owner.recover());attempt=getattr(error,'attempt',None)
   write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempt=None if attempt is None else asdict(attempt),atomic=unchanged(owner,before)))
   raise
  finally:d._assemble=original;d._advance=advance
  current=save(out,'accepted',owner,result);increments=activity(initial,current);write(out/'activity.json',increments)
  assert d._beam_predecessors is parent_beam and d._shell_predecessors is parent_shell
  assert result.load_factor==target and result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  assert result.beam_fibre_material and result.beam_generalized_material
  assert all(increments[k]['negative']==0 for k in ('shell','fibre','generalized'))
  if not peak:assert all(increments[k]['positive']>0 for k in ('shell','fibre','generalized')),'reload must renew all three materials'
  assert all(sum(e.incremental_dissipation_j)>=0 for e in result.beam_fibre_energy)
  cold=ss.create_analysis(definition=model)
  try:cold.restore(owner.checkpoint());assert cold.checkpoint()==owner.checkpoint();exact(current,capture(cold))
  finally:cold.close()
  write(out/'result.json',dict(status='PASS',factor=target,accepted_factors=result.accepted_factors,activity=increments,
   equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,immediate_predecessors=True,
   exact_cold_state_histories=True,anchored_rebase=not peak,controller=None))
 finally:owner.close()

def test_rollback():
 from dataclasses import asdict
 from scripts.r8_f06_coupled_staged_dev import DOC,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure,AdaptiveIncrementControl
 out=Path(os.environ['F06_COUPLED_STAGED_OUTPUT']);owner=ss.create_analysis(definition=definition('coupled'))
 try:
  owner.restore((DOC/'attempt-installed-peak1/accepted.bin').read_bytes());before=identity(owner)
  class Cancel(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':self.visits+=1;self.cancel('owned uncommitted reload trial')
    super().raise_if_cancelled(stage)
  token=Cancel()
  with pytest.raises(SolveCancelled):owner.solve(actions('coupled'),(.41,),max_iterations=20,max_backtracks=8,cancellation_token=token)
  write(out/'cancellation.json',dict(visits=token.visits,atomic=unchanged(owner,before)))
  assert token.visits==1 and all(unchanged(owner,before).values())
  with pytest.raises(FiniteIncrementFailure) as rejected:owner.solve(actions('coupled'),(.41,),max_iterations=1,max_backtracks=0)
  write(out/'rejection.json',dict(attempt=asdict(rejected.value.attempt),atomic=unchanged(owner,before)))
  assert all(unchanged(owner,before).values())
  token=CancellationToken();token.cancel('rebase cancellation')
  with pytest.raises(SolveCancelled):owner.rebase_rotations(owner._generation,cancellation_token=token)
  assert all(unchanged(owner,before).values())
  with pytest.raises(ValueError):owner.solve(actions('coupled'),(.41,),increment_control=AdaptiveIncrementControl(4,12))
  assert all(unchanged(owner,before).values())
  (out/'terminal-prefix.bin').write_bytes(owner.checkpoint());write(out/'result.json',dict(status='PASS',atomic=unchanged(owner,before),adaptive_refused=True,rebase_cancelled=True))
 finally:owner.close()

def test_tamper():
 from scripts.r8_f06_coupled_staged_dev import DOC,write
 from scripts.r8_f06_cyclic_material_dev import definition
 from anyfesolver import stiffened_shell as ss
 out=Path(os.environ['F06_COUPLED_STAGED_OUTPUT']);raw=(DOC/'attempt-installed-reload1/accepted.bin').read_bytes()
 fresh=ss.create_analysis(definition=definition('coupled'))
 try:
  with pytest.raises(ValueError):fresh.restore(raw[:-1]+bytes([raw[-1]^1]))
  assert fresh._finite_driver is None and fresh._accepted is None
  fresh.restore(raw);before=identity(fresh)
  with pytest.raises(RuntimeError):fresh.restore(raw)
  assert all(unchanged(fresh,before).values())
  write(out/'result.json',dict(status='PASS',malformed_refused=True,live_restore_refused=True,atomic=unchanged(fresh,before)))
 finally:fresh.close()
