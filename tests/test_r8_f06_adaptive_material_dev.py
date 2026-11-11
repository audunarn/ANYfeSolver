"""Private supervised development controls, not a qualification campaign."""
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_ADAPTIVE_OUTPUT' not in os.environ,reason='bounded adaptive supervisor required')

def test_source_admission():
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme
 from scripts.r8_f06_native_material_breadth_dev import definition
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl
 from anyfesolver._stiffened_shell_ge_finite_experimental import supports_paired_fibre_adaptive
 owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  assert supports_paired_fibre_adaptive(owner)
  assert 'experimental_adaptive_halving' in owner.capabilities['increment_control']
  with pytest.raises(ValueError,match='compensated adaptive'):
   owner.solve(cyclic.actions('bend-z'),(1.,),increment_control=AdaptiveIncrementControl())
  assert owner._finite_driver is None
 finally:owner.close()
 from dataclasses import replace
 b=definition('biaxial');s3=replace(b,shell=replace(b.shell,q4_elements=(),s3_elements=((1,2,3),(1,3,4))),q4_pullback_policy='historical_corotational')
 for model in (definition('mixed'),s3,cyclic.definition('coupled')):
  owner=ss.create_analysis(definition=model)
  try:
   assert not supports_paired_fibre_adaptive(owner)
   with pytest.raises(ValueError,match='compensated adaptive'):
    owner.solve(programme(owner),(1.,),increment_control=AdaptiveIncrementControl())
   assert owner._finite_driver is None
  finally:owner.close()

def test_source_lifecycle():
 import numpy as np
 from dataclasses import replace,asdict
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme,PREFIX,write,sha
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,decode_control,encode_control
 from anyfesolver._stiffened_shell_staged_load_experimental import decode_checkpoint,encode_checkpoint
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 out=Path(os.environ['F06_ADAPTIVE_OUTPUT']);model=cyclic.definition('bend-z')
 owner=ss.create_analysis(definition=model)
 try:
  # Actual pending prefix generated in lifecycle2; preserve its distinct source identity.
  from scripts.r8_f06_adaptive_material_dev import DOC
  owner.restore((DOC/'attempt-source-lifecycle2/accepted.bin').read_bytes())
  load=programme(owner);targets=(.001,.002);control_request=AdaptiveIncrementControl(4,12)
  d=owner._finite_driver;result=owner.recover()
  assert d._beam_predecessors is not d._beam_origins and not d._adaptive_state.completed
  assert result.increment_attempts and result.increment_attempts[-1].factor==.001
  raw=owner.checkpoint();(out/'accepted.bin').write_bytes(raw)
  request,position,inner=decode_checkpoint(raw);key=d._checkpoint_identity()
  shapes=tuple((k,getattr(d._state,k).shape) for k in ('q_low','frames_low','cells_low','resultants_low'))
  controlled,lows=decode_compensated(inner,key,shapes);control,material=decode_control(controlled,key)
  assert control==d._adaptive_state and position==.001
  cold=ss.create_analysis(definition=model)
  try:cold.restore(raw);assert cold.checkpoint()==raw
  finally:cold.close()
  histories=canonical((d._beam_predecessors,d._beam_origins));before_generation=owner._generation
  owner.rebase_rotations(before_generation)
  assert canonical((d._beam_predecessors,d._beam_origins))==histories and d._adaptive_state==control
  with pytest.raises(ValueError,match='stale'):owner.rebase_rotations(before_generation)
  anchored=owner.checkpoint();(out/'anchored.bin').write_bytes(anchored)
  cold=ss.create_analysis(definition=model)
  try:cold.restore(anchored);assert cold.checkpoint()==anchored
  finally:cold.close()
  variants={
   'wrong_order':encode_checkpoint(d._programme,position,encode_control(key,control,encode_compensated(material,key,lows))),
   'missing_words':encode_checkpoint(d._programme,position,controlled),
   'wrong_cursor':encode_checkpoint(d._programme,position,encode_compensated(encode_control(key,replace(control,cursor=0),material),key,lows)),
   'wrong_target':encode_checkpoint(d._programme,position,encode_compensated(encode_control(key,replace(control,targets=(2.02,)),material),key,lows)),
   'wrong_depth':encode_checkpoint(d._programme,position,encode_compensated(encode_control(key,replace(control,depth=1),material),key,lows))}
  for name,bad in variants.items():
   cold=ss.create_analysis(definition=model)
   try:
    with pytest.raises(ValueError):cold.restore(bad)
    assert cold._finite_driver is None and cold._accepted is None
   finally:cold.close()
  assert owner.checkpoint()==anchored
  previous=d._beam_origins
  # This controller must not call the unqualified fixed-route event heuristic.
  def forbidden_event(*args):raise AssertionError('adaptive paired route used fixed event heuristic')
  d._native_boundary_amount=forbidden_event
  result=owner.solve(load,targets,increment_control=control_request,max_iterations=20,max_backtracks=8)
  assert d._adaptive_state.completed and d._beam_predecessors is previous
  write(out/'progress.json',dict(stage='pending-resumed-complete',factor=result.load_factor))
  write(out/'metrics.json',dict(status='PASS',physical_control_target=.002,actual_load=d._programme.load_at(.002).tolist(),
   attempts=[asdict(a) for a in control.attempts],normal_cold=True,anchored_cold=True,rebase_histories_preserved=True,
   stale_rebase_refused=True,refused=list(variants),pending_resume_completed=True,elastic_controller_fixture=True,accepted_sha256=sha(out/'accepted.bin')))
 finally:owner.close()

def test_primary():
 import numpy as np
 from dataclasses import asdict
 from scripts.r8_f06_adaptive_material_dev import paired_value,persist_accepted
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme,PREFIX,write,sha
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,AdaptiveIncrementExhausted,FiniteIncrementFailure
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 out=Path(os.environ['F06_ADAPTIVE_OUTPUT']);owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  raw=PREFIX.read_bytes();owner.restore(raw);assert owner.checkpoint()==raw
  d=owner._finite_driver;load=programme(owner);initial=d._beam_predecessors
  latest=[d._beam_origins,raw];accepted=[];rejected=[];opposite=[False];observed=[]
  np.savez(out/'initial-histories.npz',**dict(history_arrays(owner._beam_parts,initial,'loaded')+
   history_arrays(owner._beam_parts,d._beam_origins,'accepted')))
  def forbidden_event(*args):raise AssertionError('adaptive primary used fixed event heuristic')
  d._native_boundary_amount=forbidden_event
  fixed=d._solve_fixed
  def observed_fixed(*args,**kwargs):
   try:return fixed(*args,**kwargs)
   except FiniteIncrementFailure as error:
    row=asdict(error.attempt);row['actual_load']=d._programme.load_at(error.attempt.factor)[d._programme.load_at(error.attempt.factor)!=0].tolist()
    rejected.append(row);write(out/'rejected.json',rejected);raise
  d._solve_fixed=observed_fixed
  def observe(result):
   # Preserve the already-published endpoint before any observer validation.
   index=len(observed);prefix,observed_row=persist_accepted(out,owner,result,index)
   observed.append(observed_row)
   write(out/'observed-accepted.json',observed)
   assert d._beam_predecessors is latest[0]
   activity=[]
   for loaded,previous,current in zip(initial[0].stations,d._beam_predecessors[0].stations,d._beam_origins[0].stations):
    for first,before,after in zip(loaded.rows,previous.rows,current.rows):
     z1=paired_value(first);z0=paired_value(before);z=paired_value(after)
     da=paired_value(after,2)-paired_value(before,2)
     assert da>=0
     activity.append(dict(initial_plastic=float(z1),increment=float(z-z0),accumulated_increment=float(da),
      exact_accumulated_sign=(da>0)-(da<0),opposite_increment=bool(z1*(z-z0)<0 and da>0)))
   applied=d._programme.load_at(result.load_factor);negative=bool(np.min(applied)<0)
   opposite[0]=opposite[0] or (negative and any(a['opposite_increment'] for a in activity))
   row=dict(index=index,factor=result.load_factor,actual_load=applied[applied!=0].tolist(),activity=activity,
    equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
    dissipation_j=[sum(e.incremental_dissipation_j) for e in result.beam_fibre_energy],
    attempts=[asdict(a) for a in result.increment_attempts],prefix=prefix.name,sha256=sha(prefix),immediate_predecessor_owned=True)
   assert all(v>=0 for v in row['dissipation_j'])
   accepted.append(row);write(out/'accepted.json',accepted);latest[:]=[d._beam_origins,prefix.read_bytes()]
  try:
   for target in (2.5,3.,4.):
    result=owner.solve(load,(target,),increment_control=AdaptiveIncrementControl(4,12),
      max_iterations=20,max_backtracks=8,on_accepted=observe)
    if opposite[0]:break
   assert opposite[0],'negative actions alone do not establish plastic reversal'
  except AdaptiveIncrementExhausted as error:
   assert owner.checkpoint()==latest[1] and d._beam_origins is latest[0]
   (out/'failure-prefix.bin').write_bytes(latest[1])
   write(out/'result.json',dict(status='ADAPTIVE_EXHAUSTED',reason=str(error),attempts=[asdict(a) for a in error.attempts],
    accepted=accepted,latest_prefix_preserved=True,negative_plastic_reversal=opposite[0]))
   raise
  (out/'final-prefix.bin').write_bytes(owner.checkpoint())
  write(out/'result.json',dict(status='NEGATIVE_PLASTIC_REVERSAL_ACCEPTED',accepted=accepted,rejected=rejected,
   final_factor=d._load_factor,negative_plastic_reversal=True,final_prefix_sha256=sha(out/'final-prefix.bin'),
   direct_fixed_failure_unchanged=True))
 finally:owner.close()

def test_source_exact_pair():
 import math,json,numpy as np
 from scripts.r8_f06_adaptive_material_dev import paired_value,persist_accepted,cyclic,PREFIX,write
 from anyfesolver import stiffened_shell as ss
 out=Path(os.environ['F06_ADAPTIVE_OUTPUT']);owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  owner.restore(PREFIX.read_bytes());d=owner._finite_driver;raw=owner.checkpoint()
  rows=[r for station in d._beam_origins[0].stations for r in station.rows if r[2]!=0.]
  assert rows,'actual stored accumulated plastic history required'
  row=rows[0];assert paired_value(row,2)-paired_value(row,2)==0
  positive=list(row);positive[2]=math.nextafter(row[2],math.inf)
  negative=list(row);negative[2]=math.nextafter(row[2],-math.inf)
  assert paired_value(positive,2)-paired_value(row,2)>0
  assert paired_value(negative,2)-paired_value(row,2)<0
  # Same production observer helper persists real caller data before this injected assertion.
  with pytest.raises(AssertionError,match='injected observer'):
   persist_accepted(out,owner,owner.recover(),0)
   raise AssertionError('injected observer failure')
  assert (out/'accepted-00.bin').read_bytes()==raw
  observed=json.loads((out/'observed-accepted-00.json').read_text())
  assert observed['factor']==2.0 and observed['actual_load']==[2.0]
  with np.load(out/'histories-00.npz',allow_pickle=False) as saved:
   assert 'origin_fibres' in saved and 'accepted_fibres' in saved
   assert np.any(saved['origin_fibres'][:,2]!=0.) and np.any(saved['accepted_fibres'][:,2]!=0.)
  assert owner.checkpoint()==raw
  write(out/'exact-pair.json',dict(status='PASS',actual_nonzero_pairs=len(rows),identical_exact_zero=True,
   positive_negative_distinguished=True,save_before_assert_actual_caller=True,original_prefix_unchanged=True))
 finally:owner.close()

def test_installed_lifecycle():
 import numpy as np,json
 from dataclasses import replace
 from scripts.r8_f06_adaptive_material_dev import DOC,cyclic,programme,write,persist_accepted,paired_value
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,AdaptiveIncrementExhausted,decode_control,encode_control
 from anyfesolver._stiffened_shell_staged_load_experimental import decode_checkpoint,encode_checkpoint
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode_chart,encode_chart,decode,encode
 from anyfesolver._stiffened_shell_beam_history import shapes
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 from anyfesolver.control import CancellationToken,SolveCancelled
 out=Path(os.environ['F06_ADAPTIVE_OUTPUT']);model=cyclic.definition('bend-z');completed=[]
 def progress(label):completed.append(label);write(out/'progress.json',completed)
 pending=(DOC/'attempt-primary2/accepted-00.bin').read_bytes()
 owner=ss.create_analysis(definition=model)
 try:
  owner.restore(pending);d=owner._finite_driver;assert owner.checkpoint()==pending and not d._adaptive_state.completed
  control=d._adaptive_state;history=canonical((d._beam_predecessors,d._beam_origins));key=d._checkpoint_identity()
  request,position,comp=decode_checkpoint(pending)
  low_shapes=tuple((k,getattr(d._state,k).shape) for k in ('q_low','frames_low','cells_low','resultants_low'))
  controlled,lows=decode_compensated(comp,key,low_shapes);saved_control,charted=decode_control(controlled,key)
  event_generation,anchors,witness,material=decode_chart(charted,key,len(owner._shell_ids))
  layout=tuple((k,v.shape) for k,v in (('q',d._state.q),('frames',d._state.frames),('cells',d._state.cells),
   ('resultants',d._state.resultants),('multipliers',d._state.multipliers),('applied_dead_load',d._applied_load),
   ('accepted_factors',np.asarray(d._accepted_factors))))+shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted')
  kind,generation,values=decode(material,key,lambda k,g:layout)
  variants={}
  for name in ('origin_fibres','accepted_fibres'):
   changed={k:v.copy() for k,v in values.items()};index=np.flatnonzero(changed[name][:,0])[0]
   changed[name][index,0]*=.9;changed[name][index,1]=0.
   inner=encode_chart(key,event_generation,anchors,witness,encode(kind,key,generation,tuple(changed.items())))
   variants[name]=encode_checkpoint(d._programme,position,encode_compensated(encode_control(key,control,inner),key,lows))
  bad_lows={k:v.copy() for k,v in lows.items()};bad_lows['q_low'][0]=1e-6
  variants['shell_low']=encode_checkpoint(d._programme,position,encode_compensated(controlled,key,bad_lows))
  variants['target']=encode_checkpoint(d._programme,position,encode_compensated(encode_control(key,replace(control,targets=(3.,)),charted),key,lows))
  variants['order']=encode_checkpoint(d._programme,position,encode_control(key,control,encode_compensated(charted,key,lows)))
  for name,bad in variants.items():
   fresh=ss.create_analysis(definition=model)
   try:
    before=(fresh._identity,fresh._generation,fresh._finite_driver,fresh._accepted,getattr(fresh,'_staged_context',None))
    with pytest.raises(ValueError):fresh.restore(bad)
    assert (fresh._identity,fresh._generation,fresh._finite_driver,fresh._accepted,getattr(fresh,'_staged_context',None))==before
   finally:fresh.close()
  reaction=owner.recover().support_reactions.copy();owner.rebase_rotations(owner._generation)
  assert canonical((d._beam_predecessors,d._beam_origins))==history and d._adaptive_state==control
  assert np.linalg.norm(owner.recover().support_reactions-reaction)<=1e-8*np.linalg.norm(reaction)
  rebased=owner.checkpoint();(out/'pending-rebased.bin').write_bytes(rebased)
  fresh=ss.create_analysis(definition=model)
  try:fresh.restore(rebased);assert fresh.checkpoint()==rebased
  finally:fresh.close()
  progress('yielded-pending-cold-rebase-refusals-no-pending-resume')
 finally:owner.close()
 owner=ss.create_analysis(definition=model)
 try:
  negative=(DOC/'attempt-primary2/final-prefix.bin').read_bytes();owner.restore(negative);assert owner.checkpoint()==negative
  d=owner._finite_driver;load=programme(owner);assert d._load_factor==2.5 and d._adaptive_state.completed
  def snapshot():return (owner.checkpoint(),owner._generation,d._beam_origins,d._beam_predecessors,d._state)
  def unchanged(before):
   assert owner.checkpoint()==before[0] and owner._generation==before[1]
   assert d._beam_origins is before[2] and d._beam_predecessors is before[3] and d._state is before[4]
  class Cancel(CancellationToken):
   def __init__(self,point,visit=1):super().__init__();self.point=point;self.visit=visit;self.seen=0
   def raise_if_cancelled(self,stage=''):
    if stage==self.point:
     self.seen+=1
     if self.seen==self.visit:self.cancel('adaptive material injected cancellation')
    super().raise_if_cancelled(stage)
  before=snapshot();cancel=Cancel('stiffened-shell.rebase.commit')
  with pytest.raises(SolveCancelled):owner.rebase_rotations(owner._generation,cancellation_token=cancel)
  assert cancel.seen==1;unchanged(before)
  histories=canonical((d._beam_predecessors,d._beam_origins));old_generation=owner._generation
  reaction=owner.recover().support_reactions.copy();owner.rebase_rotations(old_generation)
  assert canonical((d._beam_predecessors,d._beam_origins))==histories
  assert np.linalg.norm(owner.recover().support_reactions-reaction)<=1e-8*np.linalg.norm(reaction)
  with pytest.raises(ValueError,match='stale'):owner.rebase_rotations(old_generation)
  anchored=owner.checkpoint();(out/'reversed-rebased.bin').write_bytes(anchored)
  fresh=ss.create_analysis(definition=model)
  try:fresh.restore(anchored);assert fresh.checkpoint()==anchored
  finally:fresh.close()
  progress('reversed-cold-rebase-cancel-stale-anchored-cold')
  before=snapshot();cancel=Cancel('stiffened-shell.finite.trial',2)
  with pytest.raises(SolveCancelled):owner.solve(load,(2.51,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,cancellation_token=cancel)
  assert cancel.seen==2;unchanged(before)
  with pytest.raises(AdaptiveIncrementExhausted):owner.solve(load,(3.9,),increment_control=AdaptiveIncrementControl(0,1),max_iterations=1,max_backtracks=0)
  unchanged(before);progress('actual-adaptive-trial-cancel-and-forced-rejection-atomic')
  actual=[];previous=[d._beam_origins];original_event=d._native_boundary_amount
  def forbidden_event(*args):raise AssertionError('installed adaptive path used fixed event heuristic')
  d._native_boundary_amount=forbidden_event
  def observe(result):
   prefix,row=persist_accepted(out,owner,result,len(actual));actual.append(row)
   assert d._beam_predecessors is previous[0]
   for old,new in zip(d._beam_predecessors[0].stations,d._beam_origins[0].stations):
    for a,b in zip(old.rows,new.rows):assert paired_value(b,2)>=paired_value(a,2)
   previous[0]=d._beam_origins;write(out/'adaptive-accepted.json',actual)
  result=owner.solve(load,(2.51,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,on_accepted=observe)
  d._native_boundary_amount=original_event
  assert d._adaptive_state.completed and result.load_factor==2.51
  adaptive=dict(factor=result.load_factor,load=d._programme.load_at(2.51)[d._programme.load_at(2.51)!=0].tolist(),
   equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max)
  progress('actual-installed-adaptive-negative-continuation')
  previous_fixed=d._beam_origins
  result=owner.solve(load,(2.5101,),max_iterations=20,max_backtracks=8)
  persist_accepted(out,owner,result,len(actual))
  assert d._adaptive_state is None and d._beam_predecessors is previous_fixed and owner.recover().load_factor==2.5101
  progress('fixed-caller-after-completed-adaptive')
  write(out/'metrics.json',dict(status='PASS',completed=completed,refused=list(variants),
   adaptive=adaptive,fixed=dict(factor=result.load_factor,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max),
   pending_material_resumed=False,source_elastic_pending_resume_separate=True,latest_prefix_preserved=True))
 finally:owner.close()

def test_installed_continuation():
 import json
 from scripts.r8_f06_adaptive_material_dev import DOC,cyclic,programme,write,persist_accepted,paired_value
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,AdaptiveIncrementExhausted
 out=Path(os.environ['F06_ADAPTIVE_OUTPUT']);completed=[]
 def progress(label):completed.append(label);write(out/'progress.json',completed)
 owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  raw=(DOC/'attempt-installed-lifecycle1/reversed-rebased.bin').read_bytes()
  owner.restore(raw);assert owner.checkpoint()==raw
  d=owner._finite_driver;load=programme(owner)
  before=(owner.checkpoint(),owner._generation,d._beam_origins,d._beam_predecessors,d._state)
  def unchanged(value):
   assert owner.checkpoint()==value[0] and owner._generation==value[1]
   assert d._beam_origins is value[2] and d._beam_predecessors is value[3] and d._state is value[4]
  with pytest.raises(AdaptiveIncrementExhausted):owner.solve(load,(2.9,),increment_control=AdaptiveIncrementControl(0,1),max_iterations=1,max_backtracks=0)
  unchanged(before);progress('forced-in-stage-rejection-atomic')
  actual=[];previous=[d._beam_origins];original_event=d._native_boundary_amount
  def forbidden_event(*args):raise AssertionError('installed adaptive path used fixed event heuristic')
  d._native_boundary_amount=forbidden_event
  def observe(result):
   prefix,row=persist_accepted(out,owner,result,len(actual));actual.append(row)
   assert d._beam_predecessors is previous[0]
   for old,new in zip(d._beam_predecessors[0].stations,d._beam_origins[0].stations):
    for a,b in zip(old.rows,new.rows):assert paired_value(b,2)>=paired_value(a,2)
   previous[0]=d._beam_origins;write(out/'adaptive-accepted.json',actual)
  result=owner.solve(load,(2.51,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,on_accepted=observe)
  d._native_boundary_amount=original_event
  assert d._adaptive_state.completed and result.load_factor==2.51
  adaptive=dict(factor=result.load_factor,load=d._programme.load_at(2.51)[d._programme.load_at(2.51)!=0].tolist(),
   equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max)
  progress('actual-installed-adaptive-negative-continuation')
  previous_fixed=d._beam_origins
  result=owner.solve(load,(2.5101,),max_iterations=20,max_backtracks=8)
  persist_accepted(out,owner,result,len(actual))
  assert d._adaptive_state is None and d._beam_predecessors is previous_fixed and owner.recover().load_factor==2.5101
  progress('fixed-caller-after-completed-adaptive')
  write(out/'metrics.json',dict(status='PASS',completed=completed,refused=['prior-installed1-material-tamper-controls'],
   adaptive=adaptive,fixed=dict(factor=result.load_factor,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max),
   pending_material_resumed=False,source_elastic_pending_resume_separate=True,latest_prefix_preserved=True))
 finally:owner.close()
