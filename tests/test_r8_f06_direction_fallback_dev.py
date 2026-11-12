"""Affected public callers and typed-refusal direction fallback controls."""
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_FALLBACK_OUTPUT' not in os.environ,reason='bounded fallback supervisor required')

@pytest.mark.parametrize('mode',('finite','rotation','domain','cancel-before','cancel-after','fixed','budget-zero'))
def test_controls(mode):
 import numpy as np
 from scripts.r8_f06_direction_fallback_dev import INPUTS,write
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,FiniteIncrementFailure,AdaptiveIncrementExhausted
 from anyfesolver._stiffened_shell_ge_finite_experimental import _RotationStepTooLarge,BL5TrialDomainError
 out=Path(os.environ['F06_FALLBACK_OUTPUT'])/mode;out.mkdir();owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  owner.restore(INPUTS['reversed' if mode in ('fixed','budget-zero') else 'pending'].read_bytes());d=owner._finite_driver
  before=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._adaptive_state)
  advance=d._advance;assemble=d._assemble;predictor=d._fibre_material_predictor;observations=[];directions=[];amounts=[];half=[False]
  def advanced(state,increment,amount):
   directions.append(increment.copy());amounts.append(amount);half[0]=(amount==.5)
   write(out/'amounts.json',amounts)
   if amount==.5 and mode=='rotation':raise _RotationStepTooLarge('owned half-trial refusal control')
   return advance(state,increment,amount)
  def assembled(state,*args,**kwargs):
   if half[0] and mode=='domain':half[0]=False;raise BL5TrialDomainError('owned half-trial domain control')
   value=assemble(state,*args,**kwargs);observations.append(dict(eq=value[2],gap=value[3]));write(out/'evaluations.json',observations);return value
  class Stop(CancellationToken):
   parent=0;predictor=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':
     self.parent+=1
     if self.parent==2:self.cancel('stop before second parent; no acceptance')
    if stage=='stiffened-shell.finite.predictor':
     self.predictor+=1
     if (mode=='cancel-before' and self.predictor==1) or (mode=='cancel-after' and self.predictor==2):self.cancel('predictor safe point control')
    super().raise_if_cancelled(stage)
  token=Stop();d._advance=advanced;d._assemble=assembled
  def forbidden(*args):raise AssertionError('fixed dispatch called adaptive predictor')
  if mode=='fixed':d._fibre_material_predictor=forbidden
  try:
   if mode=='fixed':
    with pytest.raises(FiniteIncrementFailure) as failure:owner.solve(programme(owner),(2.51,),max_iterations=1,max_backtracks=0)
    attempt=failure.value.attempt
   elif mode=='budget-zero':
    with pytest.raises(AdaptiveIncrementExhausted) as failure:owner.solve(programme(owner),(2.51,),increment_control=AdaptiveIncrementControl(0,1),max_iterations=1,max_backtracks=0)
    attempt=failure.value.attempts[-1]
   else:
    with pytest.raises(SolveCancelled):owner.solve(programme(owner),(2.5,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,cancellation_token=token)
  finally:d._advance=advance;d._assemble=assemble;d._fibre_material_predictor=predictor
  (out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
  atomic=dict(checkpoint=owner.checkpoint()==before[0],generation=owner._generation==before[1],state=d._state is before[2],
   accepted=d._beam_origins is before[3],predecessor=d._beam_predecessors is before[4],controller=d._adaptive_state==before[5])
  write(out/'result.json',dict(mode=mode,atomic=atomic,amounts=amounts,observations=observations,predictor_visits=token.predictor))
  assert all(atomic.values())
  if mode in ('fixed','budget-zero'):
   from dataclasses import asdict
   write(out/'attempt.json',asdict(attempt))
   assert amounts==[1.] and attempt.iterations==1 and attempt.evaluations==2 and attempt.rejected_trials==1
   assert attempt.linear_solves==(1 if mode=='fixed' else 2)
  elif mode.startswith('cancel'):
   assert amounts==[1.] and token.predictor==(1 if mode=='cancel-before' else 2)
  else:
   assert amounts==[1.,.5,.25,.125]
   assert not np.array_equal(directions[0],directions[1])
   assert all(np.array_equal(v,directions[0]) for v in directions[2:])
   assert max(observations[-1].values())<max(observations[0].values())
 finally:owner.close()

def test_route():
 import numpy as np
 from dataclasses import asdict
 from scripts.r8_f06_direction_fallback_dev import INPUTS,write,sha
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme,persist_accepted,paired_value
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,FiniteIncrementFailure
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 out=Path(os.environ['F06_FALLBACK_OUTPUT']);case=os.environ['F06_FALLBACK_CASE'];target={'pending':2.5,'reversed':2.51}[case]
 owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  raw=INPUTS[case].read_bytes();owner.restore(raw);d=owner._finite_driver
  def snapshot():return owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._adaptive_state
  def atomic(s):return dict(checkpoint=owner.checkpoint()==s[0],generation=owner._generation==s[1],state=d._state is s[2],
   accepted=d._beam_origins is s[3],predecessor=d._beam_predecessors is s[4],controller=d._adaptive_state==s[5])
  latest=[snapshot()];accepted=[];rejected=[];predictors=[]
  (out/'initial-prefix.bin').write_bytes(raw);(out/'terminal-prefix.bin').write_bytes(raw)
  original_fixed=d._solve_fixed;original_predictor=d._fibre_material_predictor
  def fixed(*args,**kwargs):
   try:return original_fixed(*args,**kwargs)
   except FiniteIncrementFailure as error:
    rejected.append(asdict(error.attempt));write(out/'rejected.json',rejected)
    if 'observed-pending' in out.name:raise RuntimeError('observed fixed attempt failed; stop before adaptive retry') from error
    raise
  def predictor(*args):
   value=original_predictor(*args);predictors.append(value[1]);write(out/'predictors.json',predictors);return value
  # Observation only: read the actual production frame; never reevaluate a trial.
  import sys,json
  original_advance=d._advance;original_assemble=d._assemble
  trace_path=out/'trace.jsonl';trial_info={};assembly_count=[0]
  retained=owner._shell_count+d._beam_nodal
  pslots=np.concatenate([np.arange(retained+24*i+6,retained+24*i+24) for i in range(len(owner._beam_parts))])
  blocks={'shell':owner._free[owner._free<owner._shell_count],
   'beam_nodal':owner._free[(owner._free>=owner._shell_count)&(owner._free<retained)],
   'retained_compatibility':pslots,'constraint':np.arange(owner._K.shape[0],owner._K.shape[0]+len(d._state.multipliers))}
  def trace(row):
   with trace_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
  def advanced(state,increment,amount):
   frame=sys._getframe(1).f_locals
   info=dict(event='advance',factor=float(frame['factor']),iteration=frame['iteration'],backtrack=frame['backtrack'],
    amount=float(amount),direction='original' if increment is frame['original_full'] else 'pp',counts=dict(frame['counts']))
   trace(info)
   value=original_advance(state,increment,amount);trial_info[id(value)]=info
   return value
  def assembled(state,*args,**kwargs):
   frame=sys._getframe(2).f_locals
   value=original_assemble(state,*args,**kwargs);assembly_count[0]+=1
   info=trial_info.pop(id(state),{})
   row=dict(event='assembly',index=assembly_count[0],factor=float(args[0]),iteration=frame.get('iteration'),
    trial=info,equilibrium=float(value[2]),gap=float(value[3]),counts=dict(frame.get('counts',{})),
    blocks={name:float(np.max(np.abs(value[0][ids]),initial=0.)) for name,ids in blocks.items()})
   trace(row)
   np.savez(out/'last-observed.npz',residual=value[0],q=state.q,resultants=state.resultants,resultants_low=state.resultants_low)
   return value
  d._advance=advanced;d._assemble=assembled
  def forbidden(*args):raise AssertionError('adaptive used fixed event heuristic')
  d._solve_fixed=fixed;d._fibre_material_predictor=predictor;d._native_boundary_amount=forbidden
  def observe(result):
   previous=latest[0][3];prefix,row=persist_accepted(out,owner,result,len(accepted));accepted.append(row);latest[0]=snapshot()
   (out/'terminal-prefix.bin').write_bytes(latest[0][0]);write(out/'accepted.json',accepted)
   assert d._beam_predecessors is previous
   for old,new in zip(previous[0].stations,d._beam_origins[0].stations):
    for a,b in zip(old.rows,new.rows):assert paired_value(b,2)>=paired_value(a,2)
  try:
   result=owner.solve(programme(owner),(target,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,on_accepted=observe)
  except BaseException as error:
   write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempts=[asdict(a) for a in getattr(error,'attempts',())]))
   raise
  finally:
   (out/'terminal-prefix.bin').write_bytes(owner.checkpoint());write(out/'terminal-atomic.json',atomic(latest[0]))
  assert result.load_factor==target and d._adaptive_state.completed and all(atomic(latest[0]).values())
  # Actual new accepted checkpoint must restore both generations and counts.
  arrays=dict(history_arrays(owner._beam_parts,d._beam_predecessors,'origin')+history_arrays(owner._beam_parts,d._beam_origins,'accepted'))
  cold=ss.create_analysis(definition=cyclic.definition('bend-z'))
  try:
   cold.restore(owner.checkpoint());assert cold.checkpoint()==owner.checkpoint()
   other=dict(history_arrays(cold._beam_parts,cold._finite_driver._beam_predecessors,'origin')+history_arrays(cold._beam_parts,cold._finite_driver._beam_origins,'accepted'))
   assert all(np.array_equal(v,other[k]) for k,v in arrays.items())
  finally:cold.close()
  write(out/'result.json',dict(status='ACCEPTED',case=case,factor=result.load_factor,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
   accepted=accepted,rejected=rejected,predictors=predictors,exact_cold_histories=True,completed=d._adaptive_state.completed,
   terminal_sha256=sha(out/'terminal-prefix.bin'),atomic=atomic(latest[0])))
 finally:owner.close()
