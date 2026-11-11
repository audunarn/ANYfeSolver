"""Candidate-domain controls and one genuine coupled adaptive history."""
import os,json,runpy
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_MIXED_ADAPTIVE_OUTPUT' not in os.environ,reason='bounded supervisor required')
_h=runpy.run_path(str(Path(__file__).with_name('test_r8_f06_coupled_resume_dev.py')))
capture,exact,save,activity,identity,unchanged=(_h[k] for k in ('capture','exact','save','activity','identity','unchanged'))

def test_controls_admission():
 from dataclasses import replace
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from scripts.r8_f06_native_material_breadth_dev import definition as breadth
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl
 from anyfesolver._stiffened_shell_ge_finite_experimental import supports_mixed_material_adaptive,supports_paired_fibre_adaptive
 models=(definition('coupled'),breadth('mixed'),definition('bend-z'))
 for i,model in enumerate(models):
  o=ss.create_analysis(definition=model)
  try:
   assert supports_mixed_material_adaptive(o)==(i==0)
   assert supports_paired_fibre_adaptive(o)==(i==2)
   if i==0:
    assert 'nodal translation' in o.capabilities['adaptive_load_scope']
    for load in (ss.StagedDeadLoadProgramme(o.preflight()['identity'],(actions('coupled'),)),
                 (ss.NodalAction('beam',103,(0.,0.,0.,1.,0.,0.)),)):
     with pytest.raises(ValueError):o.solve(load,(.41,),increment_control=AdaptiveIncrementControl(4,12))
     assert o._finite_driver is None and o._generation==0
   else:
    with pytest.raises(ValueError):o.solve(actions('coupled'),(.41,),increment_control=AdaptiveIncrementControl(4,12))
    assert o._finite_driver is None
  finally:o.close()

def test_controls_framing():
 import numpy as np
 from dataclasses import replace
 from scripts.r8_f06_mixed_adaptive_dev import INPUT,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,_AdaptiveState,IncrementAttempt,encode_control,decode_control,validate_control
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode_chart,encode_chart,decode,encode
 from anyfesolver._stiffened_shell_beam_history import shapes
 from anyfesolver.control import CancellationToken,SolveCancelled
 out=Path(os.environ['F06_MIXED_ADAPTIVE_OUTPUT']);model=definition('coupled');o=ss.create_analysis(definition=model)
 try:
  o.restore(INPUT.read_bytes());d=o._finite_driver;original=capture(o);key=o._identity
  lowshapes=tuple((k,getattr(d._state,k).shape) for k in ('q_low','frames_low','cells_low','resultants_low'))
  chart,lows=decode_compensated(INPUT.read_bytes(),key,lowshapes)
  # Explicit synthetic controller parser fixture around a genuine material state.
  # Counts/physical metrics come from the already saved actual fixed unload.
  row=IncrementAttempt(.2,.4,.2,'accepted',3,7,3,0,3.3618880113886744e-9,2.04983727703373e-12)
  control=_AdaptiveState(AdaptiveIncrementControl(4,12),(.2,.41),20,8,0,.4).accepted(row)
  validate_control(control,(.4,.2));raw=encode_compensated(encode_control(key,control,chart),key,lows)
  (out/'synthetic-controller.bin').write_bytes(raw)
  gen,anchors,witness,inner=decode_chart(chart,key,len(o._shell_ids))
  layout=tuple((k,getattr(d._state,k).shape) for k in ('q','frames','cells','resultants','multipliers'))
  layout+=(('pattern',d._pattern.shape),('accepted_factors',(2,)))+shapes(o._beam_parts,'origin')+shapes(o._beam_parts,'accepted')
  layout+=(('shell_plastic',(1,12,3)),('shell_alpha',(1,12)),('shell_origin_plastic',(1,12,3)),('shell_origin_alpha',(1,12)))
  kind,generation,values=decode(inner,key,lambda k,g:layout)
  variants={'order':encode_control(key,control,encode_compensated(chart,key,lows)),
   'missing_words':encode_control(key,control,chart),
   'cursor':encode_compensated(encode_control(key,replace(control,cursor=0),chart),key,lows),
   'targets':encode_compensated(encode_control(key,replace(control,targets=(.3,.41)),chart),key,lows),
   'extra_solve':encode_compensated(encode_control(key,replace(control,attempts=(replace(row,linear_solves=4,rejected_trials=1),)),chart),key,lows)}
  for name in ('pattern','accepted_fibres','origin_fibres','accepted_accumulated','origin_accumulated','shell_alpha','shell_origin_alpha'):
   changed={k:v.copy() for k,v in values.items()}
   if name=='pattern':changed[name][np.flatnonzero(changed[name])[0]]*=1.1
   else:changed[name].flat[0]+=1e-4
   changedinner=encode_chart(key,gen,anchors,witness,encode(kind,key,generation,tuple(changed.items())))
   variants[name]=encode_compensated(encode_control(key,control,changedinner),key,lows)
  refused=[]
  for name,bad in variants.items():
   fresh=ss.create_analysis(definition=model)
   try:
    with pytest.raises(ValueError):fresh.restore(bad)
    assert fresh._finite_driver is None and fresh._accepted is None and fresh._generation==0
    refused.append(name);write(out/'refusals.json',refused)
   finally:fresh.close()
  fresh=ss.create_analysis(definition=model)
  try:
   fresh.restore(raw);assert fresh.checkpoint()==raw;exact(original,capture(fresh));fd=fresh._finite_driver
   assert fd._adaptive_state==control and np.array_equal(fd._pattern,d._pattern)
   before=identity(fresh);controller=fd._adaptive_state
   for target,ctl in (((.41,),control.control),(control.targets,AdaptiveIncrementControl(3,12))):
    with pytest.raises(ValueError):fresh.solve(actions('coupled'),target,increment_control=ctl,max_iterations=20,max_backtracks=8)
    assert all(unchanged(fresh,before).values()) and fd._adaptive_state is controller
   changedload=list(actions('coupled'));changedload[0]=replace(changedload[0],wrench=(51.,0.,10.,0.,0.,0.))
   with pytest.raises(ValueError):fresh.solve(tuple(changedload),control.targets,increment_control=control.control,max_iterations=20,max_backtracks=8)
   assert all(unchanged(fresh,before).values())
   token=CancellationToken();token.cancel('mixed adaptive pretrial cancellation')
   with pytest.raises(SolveCancelled):fresh.solve(actions('coupled'),control.targets,increment_control=control.control,max_iterations=20,max_backtracks=8,cancellation_token=token)
   assert all(unchanged(fresh,before).values()) and fd._adaptive_state is controller
   write(out/'result.json',dict(status='PASS',synthetic_controller_only=True,refusals=refused,exact_state_histories=True,strict_counts=True,cancellation_atomic=True))
  finally:fresh.close()
 finally:o.close()

def test_controls_generated_cache():
 import numpy as np
 from scripts.r8_f06_mixed_adaptive_dev import DOC,write
 from scripts.r8_f06_cyclic_material_dev import definition
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated
 from anyfesolver._stiffened_shell_increment_control import MAGIC,decode_control
 from anyfesolver.control import CancellationToken,SolveCancelled
 out=Path(os.environ['F06_MIXED_ADAPTIVE_OUTPUT']);raw=(DOC/'recovered-cached-prefix.bin').read_bytes()
 original=DOC/'attempt-installed-path1';model=definition('coupled');o=ss.create_analysis(definition=model)
 try:
  with pytest.raises(ValueError):o.restore((original/'accepted-00.bin').read_bytes())
  assert o._finite_driver is None and o._accepted is None
  o.restore(raw);assert o.checkpoint()==raw;d=o._finite_driver
  saved=dict(np.load(original/'accepted-00.npz'));actual=save(out,'canonical-restored',o,o.recover());exact(saved,actual)
  assert json.loads((out/'canonical-restored-recovery.json').read_text())==json.loads((original/'accepted-00-recovery.json').read_text())
  control=d._adaptive_state;assert control.targets==(.41,) and not control.completed and d._load_factor==.305
  before=identity(o);token=CancellationToken();token.cancel('generated adaptive rebase cancellation')
  with pytest.raises(SolveCancelled):o.rebase_rotations(o._generation,cancellation_token=token)
  assert all(unchanged(o,before).values()) and d._adaptive_state is control
  generation=o._generation;o.rebase_rotations(generation);rebased=save(out,'actual-rebased',o,o.recover())
  exact(actual,rebased,[k for k in actual if k.startswith(('origin_','accepted_')) or k.endswith('_low')]+['reactions'])
  with pytest.raises(ValueError,match='stale'):o.rebase_rotations(generation)
  made=o.checkpoint();shapes=tuple((k,getattr(d._state,k).shape) for k in ('q_low','frames_low','cells_low','resultants_low'))
  inner,words=decode_compensated(made,o._identity,shapes);assert inner.startswith(MAGIC)
  restored_control,charted=decode_control(inner,o._identity);assert restored_control==control
  assert d.checkpoint(_encode_compensated=False)==inner
  assert d.checkpoint(_encode_control=False)==charted
  assert o.checkpoint()==made
  fresh=ss.create_analysis(definition=model)
  try:fresh.restore(made);assert fresh.checkpoint()==made;exact(rebased,capture(fresh))
  finally:fresh.close()
  write(out/'generated-cache-result.json',dict(status='PASS',actual_saved_history=True,all_saved_state_words_histories_recovery_equal=True,
   double_wrapper_refused=True,generated_cache_canonical=True,partial_encoding_exact=True,cold_rebase_cold=True,cancellation_stale_atomic=True))
  # Artifact3 checks the final real accepted prefix without any continuation.
  terminal=DOC/'attempt-installed-resume1/terminal.bin'
  final=ss.create_analysis(definition=model)
  try:
   final.restore(terminal.read_bytes());assert final.checkpoint()==terminal.read_bytes()
   saved_final=dict(np.load(terminal.with_suffix('.npz')));actual_final=save(out,'final-prefix-restored',final,final.recover());exact(saved_final,actual_final)
   assert json.loads((out/'final-prefix-restored-recovery.json').read_text())==json.loads(terminal.with_name('terminal-recovery.json').read_text())
   initial=dict(np.load(DOC/'attempt-installed-path1/initial.npz'));increments=activity(initial,actual_final)
   write(out/'final-prefix-result.json',dict(status='PASS',factor=final.recover().load_factor,activity_from_unloaded_input=increments,actual_cold_restore=True,no_physical_continuation=True))
  finally:final.close()
 finally:o.close()

def test_physical():
 import numpy as np,sys
 from dataclasses import asdict
 from scripts.r8_f06_mixed_adaptive_dev import DOC,INPUT,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,AdaptiveIncrementExhausted,FiniteIncrementFailure
 out=Path(os.environ['F06_MIXED_ADAPTIVE_OUTPUT']);model=definition('coupled');owner=ss.create_analysis(definition=model)
 accepted=[];rejected=[];completed=[];targets=(.41,);request=AdaptiveIncrementControl(4,12);trace=out/'trace.jsonl'
 class AcceptedStop(Exception):pass
 def persist(result):
  nonlocal previous,owner,initial
  index=len(accepted);current=save(out,'accepted-%02d'%index,owner,result)
  d=owner._finite_driver;row=dict(factor=result.load_factor,accepted_factors=result.accepted_factors,
   actual_nodal_actions=[asdict(v) for v in actions('coupled',result.load_factor)],
   attempts=[asdict(v) for v in result.increment_attempts],activity=activity(previous,current),
   equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max)
  accepted.append(row);write(out/'accepted-journal.json',accepted)
  assert all(row['activity'][k]['negative']==0 for k in ('fibre','generalized','shell'))
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  assert np.array_equal(d._pattern,pattern) and d._programme is None
  for key in previous:
   if key.startswith('accepted_'):assert np.array_equal(previous[key],current[key.replace('accepted_','origin_',1)])
  previous=current
  if len(accepted)==1 and not d._adaptive_state.completed:raise AcceptedStop('first genuine accepted subincrement saved')
 def solve_once():
  d=owner._finite_driver;assemble=d._assemble;fixed=d._solve_fixed
  def fixed_observed(*args,**kwargs):
   before=identity(owner);controller=d._adaptive_state
   try:return fixed(*args,**kwargs)
   except FiniteIncrementFailure as error:
    rejected.append(dict(attempt=asdict(error.attempt),atomic=unchanged(owner,before),controller_same=d._adaptive_state is controller))
    write(out/'rejected-journal.json',rejected)
    (out/'last-rejected-prefix.bin').write_bytes(owner.checkpoint())
    assert all(unchanged(owner,before).values()) and d._adaptive_state is controller
    raise
  def observed(state,*args,**kwargs):
   f=sys._getframe(2).f_locals;v=assemble(state,*args,**kwargs)
   with trace.open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(factor=float(args[0]),iteration=f.get('iteration'),equilibrium=float(v[2]),gap=float(v[3]),counts=dict(f.get('counts',{}))))+'\n')
   return v
  def forbidden(*args,**kwargs):raise AssertionError('mixed adaptive must not use fibre predictor or event heuristic')
  d._assemble=observed;d._solve_fixed=fixed_observed;d._native_boundary_amount=forbidden
  try:return owner.solve(actions('coupled'),targets,increment_control=request,max_iterations=20,max_backtracks=8,on_accepted=persist)
  finally:d._assemble=assemble;d._solve_fixed=fixed
 try:
  resume=os.environ['F06_MIXED_ADAPTIVE_STAGE']=='installed-resume'
  source=DOC/'attempt-installed-controls2/actual-rebased.bin' if resume else INPUT
  result=owner.restore(source.read_bytes());current=save(out,'initial',owner,result);previous=current;pattern=owner._finite_driver._pattern.copy()
  initial=dict(np.load(DOC/'attempt-installed-path1/initial.npz')) if resume else current
  if resume:
   accepted.extend(json.loads((DOC/'attempt-installed-path1/accepted-journal.json').read_text()))
   rejected.extend(json.loads((DOC/'attempt-installed-path1/rejected-journal.json').read_text()))
   completed.append('actual-generated-checkpoint-cold-rebase-cold-controls')
  try:result=solve_once()
  except AcceptedStop:
   raw=owner.checkpoint();saved=capture(owner);controller=owner._finite_driver._adaptive_state;owner.close()
   owner=ss.create_analysis(definition=model);owner.restore(raw);assert owner.checkpoint()==raw;exact(saved,capture(owner))
   assert owner._finite_driver._adaptive_state==controller
   owner.rebase_rotations(owner._generation);rebased=save(out,'pending-rebased',owner,owner.recover())
   exact(saved,rebased,[k for k in saved if k.startswith(('origin_','accepted_')) or k.endswith('_low')]+['reactions'])
   chart=owner.checkpoint();owner.close();owner=ss.create_analysis(definition=model);owner.restore(chart);assert owner.checkpoint()==chart
   exact(rebased,capture(owner));assert owner._finite_driver._adaptive_state==controller
   completed.append('genuine-pending-cold-rebase-cold');write(out/'progress.json',completed)
   result=solve_once()
  final=save(out,'final',owner,result);increments=activity(initial,final)
  write(out/'result.json',dict(status='PASS',factor=result.load_factor,path=result.accepted_factors,activity=increments,
   attempts=[asdict(v) for v in result.increment_attempts],completed=completed))
  assert result.load_factor==.41 and all(increments[k]['positive']>0 for k in ('fibre','generalized','shell'))
  raw=owner.checkpoint();fresh=ss.create_analysis(definition=model)
  try:fresh.restore(raw);assert fresh.checkpoint()==raw;exact(final,capture(fresh))
  finally:fresh.close()
 except BaseException as error:
  # A failed fresh restore has no accepted result: never mask the primary error.
  if owner._finite_driver is not None and owner._finite_driver._accepted is not None:
   save(out,'terminal',owner,owner.recover())
  else:
   candidates=sorted(out.glob('accepted-*.bin'))
   if candidates:(out/'terminal-last-saved.bin').write_bytes(candidates[-1].read_bytes())
  write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempts=[asdict(v) for v in getattr(error,'attempts',())],accepted=accepted,completed=completed))
  raise
 finally:owner.close()
