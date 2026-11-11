"""Observations only: no accepted state, controller cutback or second parent."""
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_PENDING_DIRECTION_OUTPUT' not in os.environ,reason='owned bounded supervisor required')

def test_same_parent_directions():
 import numpy as np
 from scipy import sparse
 from dataclasses import asdict
 from scripts.r8_f06_pending_direction_dev import INPUT,HISTORY,PACKAGE,write,runtime
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 out=Path(os.environ['F06_PENDING_DIRECTION_OUTPUT']);owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 fields=('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors')
 try:
  raw=INPUT.read_bytes();owner.restore(raw);d=owner._finite_driver
  before=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._adaptive_state)
  def atomic():return dict(checkpoint=owner.checkpoint()==before[0],generation=owner._generation==before[1],
   state=d._state is before[2],accepted_history=d._beam_origins is before[3],predecessor=d._beam_predecessors is before[4],controller=d._adaptive_state==before[5])
  (out/'initial-prefix.bin').write_bytes(raw);(out/'terminal-prefix.bin').write_bytes(raw)
  runtime(out,'runtime-cold-imports',PACKAGE/'site')
  actual=dict(history_arrays(owner._beam_parts,d._beam_predecessors,'origin')+history_arrays(owner._beam_parts,d._beam_origins,'accepted'))
  np.savez(out/'cold-histories.npz',**actual)
  assert raw==owner.checkpoint() and d._load_factor==2.25 and not d._adaptive_state.completed
  with np.load(HISTORY,allow_pickle=False) as saved:
   assert set(saved.files)==set(actual) and all(np.array_equal(v,saved[k]) for k,v in actual.items())
  assert d._external is None,'this staged point route uses the actual eq/gap merit, no frozen external norm'
  write(out/'parent-binding.json',dict(controller=asdict(d._adaptive_state),factor=2.25,target=2.5,
   source_histories_exact=True,footprint_chart=owner._footprint_increment_chart,physical_acceptance=dict(equilibrium=1e-8,gap=1e-9)))
  assemble=d._assemble;advance=d._advance;predict=d._fibre_material_predictor
  captured={};observations=[]
  count=owner._K.shape[0];free=np.r_[owner._free,np.arange(count,count+len(d._state.multipliers))]
  retained=owner._shell_count+d._beam_nodal
  pslots=np.concatenate([np.arange(retained+24*i+6,retained+24*i+24) for i in range(len(owner._beam_parts))])
  phislots=np.concatenate([np.arange(retained+24*i,retained+24*i+6) for i in range(len(owner._beam_parts))])
  blocks={'shell':owner._free[owner._free<owner._shell_count],
   'beam_nodal':owner._free[(owner._free>=owner._shell_count)&(owner._free<retained)],
   'retained_rotation':phislots,'retained_compatibility':pslots,'constraint':np.arange(count,count+len(d._state.multipliers))}
  def record(label,state,values,origin,amount=None,direction=None):
   residual,J,eq,gap,sh,bh=values
   np.savez(out/(label+'.npz'),residual=residual,**{k:getattr(state,k) for k in fields if getattr(state,k) is not None})
   material=[]
   for i,(part,_,_) in enumerate(owner._beam_parts):
    data=part.operator.recover_material(state.cells[i],state.resultants[i],origin=origin[i],
     rotations_low=state.cells_low[i],resultants_low=state.resultants_low[i])
    material.append([[asdict(f) for f in station['fibres']] for station in data.stations])
   block_values={}
   for name,ids in blocks.items():
    masked=np.zeros_like(residual);masked[ids]=residual[ids]
    block_values[name]=dict(raw_max_abs=float(np.max(np.abs(residual[ids]),initial=0.)),raw_l2=float(np.linalg.norm(residual[ids])),
     production_equilibrium_norm_ratio=None if name=='constraint' else d._equilibrium_ratio(masked,captured['norm_load']))
   row=dict(label=label,amount=amount,direction=direction,equilibrium=eq,gap=gap,merit=max(eq,gap),material=material,
    shell_anchors_present=state.shell_anchors is not None,
    residual_blocks=block_values)
   if 'parent_merit' in captured:
    row['would_pass_existing_trial_predicate']=bool((eq<=1e-8 and gap<=1e-9) or max(eq,gap)<captured['parent_merit'])
   observations.append(row);write(out/'observations.json',observations)
   return row
  def observed(state,*args,**kwargs):
   assert len(observations)<2,'no second parent/candidate during production capture'
   assert args[0]==2.5 and args[2] is before[3],'next increment must use the accepted2.25 origin'
   values=assemble(state,*args,**kwargs);label='parent' if not observations else 'original-full'
   if label=='parent':captured['norm_load']=kwargs['load_programme'].equilibrium_load(kwargs['applied_load']).copy()
   record(label,state,values,args[2],None if label=='parent' else 1.,'original')
   sparse.save_npz(out/(label+'-tangent.npz'),values[1])
   if label=='parent':
    assert state is before[2],'this point route must retain the exact cold accepted chart'
    captured.update(state=state,args=args,kwargs={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in kwargs.items()},
     residual=values[0].copy(),matrix=values[1].copy(),parent_merit=max(values[2],values[3]))
    np.savez(out/'row-map.npz',free=free,pslots=pslots,**blocks,applied=kwargs['applied_load'])
    assert np.array_equal(kwargs['applied_load'],d._programme.load_at(2.5))
   else:captured['donor']=values[1].copy()
   return values
  def advanced(state,increment,amount):
   assert amount==1. and 'original' not in captured,'capture stops before geometric candidates'
   captured['original']=increment.copy();return advance(state,increment,amount)
  def predicted(*args):
   result=predict(*args);captured['predicted']=None if result[0] is None else result[0].copy();captured['diagnostic']=dict(result[1])
   write(out/'production-predictor.json',dict(diagnostic=result[1],counts=dict(args[-1])))
   return result
  class StopAfterPredictor(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.predictor':
     self.visits+=1
     if self.visits==2:self.cancel('evaluation-only stop after actual pp solve')
    super().raise_if_cancelled(stage)
  token=StopAfterPredictor();d._assemble=observed;d._advance=advanced;d._fibre_material_predictor=predicted
  try:
   with pytest.raises(SolveCancelled):owner.solve(programme(owner),(2.5,),increment_control=AdaptiveIncrementControl(4,12),
     max_iterations=20,max_backtracks=8,cancellation_token=token)
  finally:d._assemble=assemble;d._advance=advance;d._fibre_material_predictor=predict
  write(out/'capture-atomic.json',atomic());assert token.visits==2 and all(atomic().values())
  assert len(observations)==2 and captured['predicted'] is not None
  J=captured['matrix'];R=captured['residual'];P=J.tolil()
  for i in range(len(owner._beam_parts)):
   slots=np.arange(retained+24*i+6,retained+24*i+24);P[np.ix_(slots,slots)]=captured['donor'][slots,:][:,slots].toarray().copy()
  P=P.tocsr();diff=(P-J).tocoo();assert all(i in pslots and j in pslots for i,j in zip(diff.row,diff.col))
  sparse.save_npz(out/'predictor-matrix.npz',P)
  np.savez(out/'directions.npz',original=captured['original'],predicted=captured['predicted'],
   original_linear_residual=J@captured['original']+R,predictor_linear_residual=P@captured['predicted']+R,
   predictor_parent_residual=J@captured['predicted']+R,free=free)
  try:
   for backtrack in range(1,9):
    amount=.5**backtrack
    for kind in ('original','predicted'):
     state=advance(captured['state'],captured[kind],amount)
     values=assemble(state,*captured['args'],**captured['kwargs'])
     record('%s-%02d'%(kind,backtrack),state,values,before[3],amount,kind)
  finally:
   (out/'terminal-prefix.bin').write_bytes(owner.checkpoint());write(out/'terminal-atomic.json',atomic())
  def branch(row):return [f['branch'] for member in row['material'] for station in member for f in station]
  donor=branch(observations[1]);parent=branch(observations[0])
  for row in observations:
   b=branch(row);row.update(donor_branch_matches=sum(x==y for x,y in zip(b,donor)),parent_branch_matches=sum(x==y for x,y in zip(b,parent)))
  write(out/'observations.json',observations)
  passes={kind:[r['amount'] for r in observations[2:] if r['direction']==kind and r['would_pass_existing_trial_predicate']] for kind in ('original','predicted')}
  write(out/'result.json',dict(status='EVALUATION_ONLY_COMPLETE',no_acceptance=True,parent_merit=captured['parent_merit'],
   existing_fraction_descent=passes,actual_assemblies=18,production_assemblies=2,production_linear_solves=2,
   original_linear_max=float(np.max(np.abs((J@captured['original']+R)[free]))),
   predictor_linear_max=float(np.max(np.abs((P@captured['predicted']+R)[free]))),atomic=atomic()))
  assert all(atomic().values())
 finally:
  if 'before' in locals():
   (out/'terminal-prefix.bin').write_bytes(owner.checkpoint());write(out/'terminal-atomic.json',atomic())
  owner.close()
