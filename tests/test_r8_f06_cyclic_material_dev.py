"""Actual public bending histories; no seeded accepted endpoint."""
import os,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif(not os.environ.get('F06_CYCLIC_CASE') or not os.environ.get('F06_CYCLIC_OUTPUT'),
 reason='requires the explicit bounded cyclic-material development supervisor')

def test_installed_framing_only():
 import numpy as np
 from scripts.r8_f06_cyclic_material_dev import definition,DOC
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import encode_compensated
 raw=(DOC/'attempt-source-bend-z2/failure-state.bin').read_bytes()
 owner=ss.create_analysis(definition=definition('bend-z'))
 try:
  zero={'q_low':np.zeros(owner._K.shape[0]),'frames_low':np.zeros((len(owner._beam_ids),3,3)),
   'cells_low':np.zeros((len(owner._beam_parts),2,3,3)),'resultants_low':np.zeros((len(owner._beam_parts),18))}
  wrong=encode_compensated(raw,owner._identity,zero)
  with pytest.raises(ValueError,match='envelope must be outermost'):owner.restore(wrong)
  assert owner._finite_driver is None and owner._accepted is None
  if os.environ['F06_CYCLIC_INSTALLED']=='1':
   owner.restore(raw);assert owner.checkpoint()==raw
 finally:owner.close()
 if os.environ['F06_CYCLIC_INSTALLED']=='1':
  owner=ss.create_analysis(definition=definition('coupled'))
  try:
   raw=(DOC/'coupled-prefix.bin').read_bytes();result=owner.restore(raw)
   assert owner.checkpoint()==raw and result.beam_fibre_material and result.beam_generalized_material
   assert max(v for s in result.shell_layer_alpha for v in s)>0.
  finally:owner.close()

def test_public_cyclic_material():
 import numpy as np
 from scripts.r8_f06_cyclic_material_dev import definition,actions,DOC
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 case=os.environ['F06_CYCLIC_CASE'];out=Path(os.environ['F06_CYCLIC_OUTPUT'])
 model=definition(case);owner=ss.create_analysis(definition=model)
 programme=ss.StagedDeadLoadProgramme(owner.preflight()['identity'],
  (actions(case,.4),actions(case,.2),actions(case,-.4),actions(case,-.41))) if case=='bend-z' else None
 load=actions(case) if programme is None else programme
 start,unload,continued=(.4,.2,.41) if programme is None else (1.,2.,3.)
 def phase(name):(out/'progress.json').write_text(json.dumps(dict(phase=name)))
 def rows(history):return np.asarray([[r[0]+r[1],r[2]+r[3]] for s in history[0].stations for r in s.rows])
 try:
  prefix=DOC/(case+'-prefix.bin')
  result=owner.restore(prefix.read_bytes()) if prefix.exists() else owner.solve(load,(start,),max_iterations=20,max_backtracks=8)
  driver=owner._finite_driver;active=rows(driver._beam_origins)
  (out/'first-equilibrium.bin').write_bytes(owner.checkpoint())
  (out/'first-equilibrium.json').write_text(json.dumps(dict(fibre_accumulation=float(active[:,1].max()),
   shell_alpha=result.shell_layer_alpha,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max)))
  assert np.max(active[:,1])>0.
  if case!='coupled':assert np.min(active[:,0])<0.<np.max(active[:,0]),'bending must activate opposite signed fibres'
  if case=='coupled':
   assert max(v for s in result.shell_layer_alpha for v in s)>0.
   assert result.beam_generalized_material and max(result.beam_station_accumulated[1])>0.
  raw=owner.checkpoint();(out/'active.bin').write_bytes(raw);phase('active-bending')
  derivative_errors={}
  if case=='coupled':
   # Actual update, fixed accepted predecessor, all raw residual row families.
   def evaluate(state):return driver._assemble(state,start,driver._shell_predecessors,driver._beam_predecessors)
   baseline=evaluate(driver._state);n=owner._K.shape[0];end=owner._shell_count+6*len(owner._beam_ids)
   direction=np.sin(np.arange(len(baseline[0]))+.7)*1e-4
   direction[end:n]*=100.;direction[n:]*=100.;direction[owner._fixed]=0.
   weight=np.r_[owner._coordinate_scale,np.ones(len(driver._state.multipliers))]
   predicted=(baseline[1]@direction)*weight;h=1e-3
   plus=evaluate(driver._advance(driver._state,direction,h));minus=evaluate(driver._advance(driver._state,direction,-h))
   def mask(response):return np.concatenate([a['alpha']>b['alpha'] for a,b in zip(response[4],driver._shell_predecessors)])
   assert np.array_equal(mask(baseline),mask(plus)) and np.array_equal(mask(baseline),mask(minus))
   difference=(plus[0]-minus[0])/(2*h)*weight-predicted
   groups=dict(shell=owner._free[owner._free<owner._shell_count],beam=owner._free[(owner._free>=owner._shell_count)&(owner._free<end)],
    retained=np.arange(end,n),constraints=np.arange(n,len(direction)))
   for name,index in groups.items():
    norm=np.linalg.norm(predicted[index]);assert norm>0.
    derivative_errors[name]=float(np.linalg.norm(difference[index])/norm)
   assert max(derivative_errors.values())<1e-6,derivative_errors
   # Replay the true accepted state before checkpoint/recovery capture.
   evaluate(driver._state)
   from anyfesolver._stiffened_shell_beam_history import shapes
   from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode,encode
   state=driver._state;lows=('q_low','frames_low','cells_low','resultants_low')
   inner,corrections=decode_compensated(raw,owner._identity,tuple((k,getattr(state,k).shape) for k in lows))
   layout=tuple((k,v.shape) for k,v in (('q',state.q),('frames',state.frames),('cells',state.cells),('resultants',state.resultants),
    ('multipliers',state.multipliers),('pattern',driver._pattern),('accepted_factors',np.asarray(driver._accepted_factors))))
   layout+=shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted')
   layout+=tuple((k,shape) for k,shape in (('shell_plastic',(1,12,3)),('shell_alpha',(1,12)),('shell_origin_plastic',(1,12,3)),('shell_origin_alpha',(1,12))))
   kind,generation,values=decode(inner,owner._identity,lambda k,g:layout)
   assert kind=='finite_compensated_mixed_coupled'
   fresh=ss.create_analysis(definition=model)
   try:
    for name in ('shell_alpha','shell_origin_alpha','origin_fibres','accepted_fibres','origin_accumulated','accepted_accumulated'):
     changed={k:v.copy() for k,v in values.items()};changed[name].flat[0]+=1e-4
     with pytest.raises(ValueError):fresh.restore(encode_compensated(encode(kind,owner._identity,generation,tuple(changed.items())),owner._identity,corrections))
     assert fresh._finite_driver is None and fresh._accepted is None
    for wrong in ('finite_compensated_mixed_material','finite_compensated_fibre_coupled','finite_compensated_generalized_coupled','finite_material','finite_shell_material'):
     with pytest.raises(ValueError):fresh.restore(encode_compensated(encode(wrong,owner._identity,generation,tuple(values.items())),owner._identity,corrections))
     assert fresh._finite_driver is None
    with pytest.raises(ValueError):fresh.restore(inner)
   finally:fresh.close()
  if not prefix.exists():prefix.write_bytes(raw)
  previous=driver._beam_origins
  result=owner.solve(load,(unload,),max_iterations=20,max_backtracks=8)
  assert driver._beam_predecessors is previous
  unloaded=rows(driver._beam_origins)
  assert np.all(unloaded[:,1]>=active[:,1]-1e-18)
  raw=owner.checkpoint();phase('unloaded')
  fresh=ss.create_analysis(definition=model)
  try:fresh.restore(raw);assert fresh.checkpoint()==raw
  finally:fresh.close()
  accepted=canonical(driver._beam_origins);parent=canonical(driver._beam_predecessors)
  shell_accepted=tuple((s['plastic_strain'].copy(),s['alpha'].copy()) for s in driver._shell_origins)
  reaction=owner.recover().support_reactions.copy()
  owner.rebase_rotations(owner._generation);assert canonical(driver._beam_origins)==accepted and canonical(driver._beam_predecessors)==parent
  for saved,current in zip(shell_accepted,driver._shell_origins):
   assert np.array_equal(saved[0],current['plastic_strain']) and np.array_equal(saved[1],current['alpha'])
  assert np.linalg.norm(owner.recover().support_reactions-reaction)<=1e-8*max(np.linalg.norm(reaction),np.finfo(float).tiny)
  raw=owner.checkpoint();fresh=ss.create_analysis(definition=model)
  try:fresh.restore(raw);assert fresh.checkpoint()==raw
  finally:fresh.close()
  previous=driver._beam_origins
  try:
   for step in ((2.5,continued) if programme is not None else (continued,)):
    previous=driver._beam_origins
    result=owner.solve(load,(step,),max_iterations=20,max_backtracks=8)
    assert driver._beam_predecessors is previous
  except Exception as error:
   from dataclasses import asdict,is_dataclass
   (out/'failure-state.bin').write_bytes(owner.checkpoint())
   attempt=getattr(error,'attempt',None)
   (out/'failure.json').write_text(json.dumps(dict(error=str(error),attempt=asdict(attempt) if is_dataclass(attempt) else repr(attempt))))
   raise
  assert driver._beam_predecessors is previous
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  assert owner.recover().beam_fibre_material
  dissipation=[sum(e.incremental_dissipation_j) for e in owner.recover().beam_fibre_energy]
  assert all(v>=0. for v in dissipation)
  if programme is not None:
   reversed_rows=rows(driver._beam_origins)
   assert np.any(reversed_rows[:,0]*active[:,0]<0.),'actual reversed plastic branch required'
   assert np.all(reversed_rows[:,1]>=unloaded[:,1]-1e-18)
  after=owner.checkpoint();phase('continued-or-reversed')
  final=DOC/(case+'-final-prefix.bin')
  if not final.exists():final.write_bytes(after)
  from anyfesolver.control import CancellationToken,SolveCancelled
  class Cancel(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':
     self.visits+=1
     if self.visits==2:self.cancel('cyclic material trial')
    super().raise_if_cancelled(stage)
  cancel=Cancel()
  with pytest.raises(SolveCancelled):owner.solve(load,(continued+.01,),cancellation_token=cancel)
  assert cancel.visits==2 and owner.checkpoint()==after
  from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
  with pytest.raises(FiniteIncrementFailure):owner.solve(load,((3.9 if programme is not None else 1.2),),max_iterations=1,max_backtracks=0)
  assert owner.checkpoint()==after
  (out/'metrics.json').write_text(json.dumps(dict(status='PASS',case=case,
   active_min=float(active[:,0].min()),active_max=float(active[:,0].max()),accumulation=float(active[:,1].max()),
   unloaded_accumulation=float(unloaded[:,1].max()),accepted_factors=result.accepted_factors,
   equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,derivative_errors=derivative_errors,
   shell_accumulation=max((v for s in result.shell_layer_alpha for v in s),default=0.),
   reversed=programme is not None,cancellation_rejection_atomic=True,fibre_incremental_dissipation_j=dissipation),indent=2))
 finally:owner.close()

def test_installed_material_continuation():
 """Current isolated artifact continues a provenance-bound true public prefix."""
 import numpy as np
 from scripts.r8_f06_cyclic_material_dev import definition,actions,DOC
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated
 from anyfesolver.control import CancellationToken,SolveCancelled
 case=os.environ['F06_CYCLIC_CASE'];out=Path(os.environ['F06_CYCLIC_OUTPUT']);model=definition(case)
 import anyfesolver._stiffened_shell_f06_increment_arithmetic as arithmetic
 original=arithmetic.full_log_coefficients;native_cosines=[]
 def capture(c):
  import sys
  if sys._getframe(2).f_code.co_name=='retained_evaluation':native_cosines.append(c.scalar())
  return original(c)
 arithmetic.full_log_coefficients=capture
 owner=ss.create_analysis(definition=model)
 prefix=DOC/(case+'-prefix.bin')
 if case=='bend-z':prefix=DOC/'attempt-source-bend-z2/failure-state.bin'
 try:
  result=owner.restore(prefix.read_bytes());driver=owner._finite_driver
  programme=ss.StagedDeadLoadProgramme(owner.preflight()['identity'],
   (actions(case,.4),actions(case,.2),actions(case,-.4),actions(case,-.41))) if case=='bend-z' else None
  load=programme if programme is not None else actions(case)
  before=owner.checkpoint();assert before==prefix.read_bytes()
  predecessor=canonical(driver._beam_predecessors);accepted=canonical(driver._beam_origins)
  lows={k:getattr(driver._state,k).copy() for k in ('q_low','frames_low','cells_low','resultants_low')}
  assert not np.any(lows['q_low'][:owner._shell_count]),'shell coordinates retain ordinary arithmetic'
  shell=tuple((s['plastic_strain'].copy(),s['alpha'].copy()) for s in driver._shell_origins)
  initial_rebase=bool(np.any(driver._state.q[:owner._shell_count].reshape(-1,6)[:,3:]))
  if initial_rebase:owner.rebase_rotations(owner._generation)
  else:
   with pytest.raises(ValueError,match='no nonzero shell chart'):owner.rebase_rotations(owner._generation)
   assert owner.checkpoint()==before
  assert canonical(driver._beam_predecessors)==predecessor and canonical(driver._beam_origins)==accepted
  for k,v in lows.items():assert np.array_equal(v,getattr(driver._state,k))
  for saved,current in zip(shell,driver._shell_origins):
   assert np.array_equal(saved[0],current['plastic_strain']) and np.array_equal(saved[1],current['alpha'])
  chart=owner.checkpoint();fresh=ss.create_analysis(definition=model)
  try:fresh.restore(chart);assert fresh.checkpoint()==chart
  finally:fresh.close()
  committed=driver._beam_origins;shell_committed=driver._shell_origins
  target=2.01 if programme is not None else (.401 if case=='coupled' else .41)
  result=owner.solve(load,(target,),max_iterations=20,max_backtracks=8)
  assert driver._beam_predecessors is committed and driver._shell_predecessors is shell_committed
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  assert result.beam_fibre_material
  if case=='coupled':assert result.beam_generalized_material and max(v for s in result.shell_layer_alpha for v in s)>0.
  if not initial_rebase:
   current=canonical(driver._beam_origins);previous=canonical(driver._beam_predecessors)
   owner.rebase_rotations(owner._generation)
   assert canonical(driver._beam_origins)==current and canonical(driver._beam_predecessors)==previous
   chart=owner.checkpoint();fresh=ss.create_analysis(definition=model)
   try:fresh.restore(chart);assert fresh.checkpoint()==chart
   finally:fresh.close()
  dissipation=[sum(e.incremental_dissipation_j) for e in result.beam_fibre_energy];assert all(v>=0. for v in dissipation)
  after=owner.checkpoint();fresh=ss.create_analysis(definition=model)
  try:
   with pytest.raises(ValueError):fresh.restore(after[:-1]+bytes([after[-1]^1]))
   assert fresh._finite_driver is None and fresh._accepted is None
   if programme is None:
    inner,_=decode_compensated(after,owner._identity,tuple((k,getattr(driver._state,k).shape) for k in lows))
    with pytest.raises(ValueError):fresh.restore(inner)
    assert fresh._finite_driver is None
   else:
    from anyfesolver._stiffened_shell_staged_load_experimental import decode_checkpoint
    _,_,inner=decode_checkpoint(after)
    with pytest.raises(ValueError):fresh.restore(inner)
    assert fresh._finite_driver is None
  finally:fresh.close()
  class Cancel(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':
     self.visits+=1
     if self.visits==2:self.cancel('installed material trial cancellation')
    super().raise_if_cancelled(stage)
  token=Cancel()
  with pytest.raises(SolveCancelled):owner.solve(load,(target+.01,),cancellation_token=token)
  assert token.visits==2 and owner.checkpoint()==after
  from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
  with pytest.raises(FiniteIncrementFailure):owner.solve(load,((2.2 if programme is not None else .8),),max_iterations=1,max_backtracks=0)
  assert owner.checkpoint()==after
  assert native_cosines and min(native_cosines)<1.-1e-7,'ordinary principal Log must reach retained evaluation'
  (out/'installed-metrics.json').write_text(json.dumps(dict(status='PASS',case=case,prefix=str(prefix),
   accepted_factors=result.accepted_factors,actual_next=target,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
   both_histories_and_corrections_preserved=True,cancel_refusal_atomic=True,
   shell_alpha=result.shell_layer_alpha,fibre_incremental_dissipation_j=dissipation,
   native_log_calls=len(native_cosines),native_log_min_cosine=min(native_cosines)),indent=2))
 finally:arithmetic.full_log_coefficients=original;owner.close()
