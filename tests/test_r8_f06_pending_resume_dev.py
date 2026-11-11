"""Installed saved material-pending continuation under a bounded supervisor."""
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_PENDING_OUTPUT' not in os.environ,reason='bounded pending-resume supervisor required')

def test_installed_partial_restore():
 import numpy as np
 from dataclasses import replace,asdict
 from scripts.r8_f06_pending_resume_dev import DOC,PACKAGE,write,runtime
 from scripts.r8_f06_adaptive_material_dev import cyclic
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 from anyfesolver._stiffened_shell_staged_load_experimental import decode_checkpoint,encode_checkpoint
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated
 from anyfesolver._stiffened_shell_increment_control import decode_control,encode_control
 out=Path(os.environ['F06_PENDING_OUTPUT']);owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  raw=(DOC/'partial-accepted-2_3125.bin').read_bytes();owner.restore(raw);d=owner._finite_driver
  # Persist actual loaded-site inventory and cold-owned state before assertions.
  runtime(out,'runtime-cold-imports',PACKAGE/'site')
  actual=dict(history_arrays(owner._beam_parts,d._beam_predecessors,'origin')+history_arrays(owner._beam_parts,d._beam_origins,'accepted'))
  np.savez(out/'cold-histories.npz',**actual)
  (out/'cold-prefix.bin').write_bytes(owner.checkpoint())
  write(out/'cold-controller.json',asdict(d._adaptive_state))
  before=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._adaptive_state)
  assert before[0]==raw and not d._adaptive_state.completed and d._adaptive_state.targets==(2.5,)
  assert d._accepted_factors[-1]==2.3125 and d._beam_predecessors is not d._beam_origins
  with np.load(DOC/'partial-histories.npz',allow_pickle=False) as saved:
   assert set(actual)==set(saved.files)
   assert all(np.array_equal(v,saved[k]) for k,v in actual.items())
  request,position,comp=decode_checkpoint(raw);key=d._checkpoint_identity()
  shapes=tuple((k,getattr(d._state,k).shape) for k in ('q_low','frames_low','cells_low','resultants_low'))
  controlled,lows=decode_compensated(comp,key,shapes);control,inner=decode_control(controlled,key)
  assert position==2.3125 and control==d._adaptive_state
  variants={'wrong_cursor':replace(control,cursor=1),'wrong_target':replace(control,targets=(2.6,))}
  refused=[]
  for name,bad in variants.items():
   wrong=encode_checkpoint(d._programme,position,encode_compensated(encode_control(key,bad,inner),key,lows))
   # Public restore requires a fresh owner even before inspecting the payload.
   with pytest.raises(RuntimeError,match='fresh experimental'):owner.restore(wrong)
   assert owner.checkpoint()==before[0] and owner._generation==before[1] and d._state is before[2]
   assert d._beam_origins is before[3] and d._beam_predecessors is before[4] and d._adaptive_state==before[5]
   cold=ss.create_analysis(definition=cyclic.definition('bend-z'))
   try:
    generation=cold._generation
    with pytest.raises(ValueError):cold.restore(wrong)
    assert cold._generation==generation and cold._finite_driver is None and cold._accepted is None
   finally:cold.close()
   refused.append(name)
  write(out/'result.json',dict(status='PASS',no_solve=True,factor=position,pending_target=list(control.targets),
   exact_checkpoint=True,exact_both_histories=True,distinct_owned_generations=True,refused=refused,
   atomic_live_owner=True,completed_endpoint=False,accepted_factors=list(d._accepted_factors)))
 finally:owner.close()

def test_installed_material_pending_resume():
 import numpy as np
 from dataclasses import asdict
 from scripts.r8_f06_pending_resume_dev import INPUT,ORACLE,write,sha
 from scripts.r8_f06_adaptive_material_dev import cyclic,programme,persist_accepted,paired_value
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl,FiniteIncrementFailure
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 out=Path(os.environ['F06_PENDING_OUTPUT']);owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  raw=INPUT.read_bytes();owner.restore(raw);d=owner._finite_driver;load=programme(owner)
  def snapshot():return owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._adaptive_state
  def unchanged(s):return dict(checkpoint=owner.checkpoint()==s[0],generation=owner._generation==s[1],
   state=d._state is s[2],accepted=d._beam_origins is s[3],predecessor=d._beam_predecessors is s[4],controller=d._adaptive_state==s[5])
  latest=[snapshot()];initial=d._beam_origins;accepted=[];rejected=[];predictors=[]
  (out/'initial-prefix.bin').write_bytes(raw);(out/'terminal-prefix.bin').write_bytes(raw)
  assert owner.checkpoint()==raw and not d._adaptive_state.completed
  assert d._adaptive_state.targets==(2.5,) and d._adaptive_state.max_iterations==20 and d._adaptive_state.max_backtracks==8
  assert d._adaptive_state.control==AdaptiveIncrementControl(4,12)
  assert d._beam_predecessors is not initial and canonical(d._beam_predecessors)!=canonical(initial)
  write(out/'progress.json',dict(stage='cold-material-pending-restored',factor=2.25,controller=asdict(d._adaptive_state)))
  # Pending request mismatch must refuse without consuming or replacing history.
  with pytest.raises(ValueError):owner.solve(load,(2.5,),increment_control=AdaptiveIncrementControl(3,12),max_iterations=20,max_backtracks=8)
  assert all(unchanged(latest[0]).values())
  token=CancellationToken();token.cancel('pending-resume pre-cancel control')
  with pytest.raises(SolveCancelled):owner.solve(load,(2.5,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,cancellation_token=token)
  assert all(unchanged(latest[0]).values())
  write(out/'controls.json',dict(cold_material_pending=True,wrong_control_refused=True,pre_cancel_atomic=True,ownership=unchanged(latest[0])))
  original_fixed=d._solve_fixed;original_predictor=d._fibre_material_predictor
  def fixed(*args,**kwargs):
   try:return original_fixed(*args,**kwargs)
   except FiniteIncrementFailure as error:
    row=asdict(error.attempt);row['actual_load']=d._programme.load_at(error.attempt.factor).tolist()
    rejected.append(row);write(out/'rejected.json',rejected);raise
  def predictor(*args):
   result=original_predictor(*args);predictors.append(result[1]);write(out/'predictors.json',predictors);return result
  def forbidden(*args):raise AssertionError('adaptive resume used fixed event heuristic')
  d._solve_fixed=fixed;d._fibre_material_predictor=predictor;d._native_boundary_amount=forbidden
  def observe(result):
   before=latest[0][3]
   path,row=persist_accepted(out,owner,result,len(accepted));accepted.append(row);latest[0]=snapshot()
   (out/'terminal-prefix.bin').write_bytes(latest[0][0]);write(out/'accepted.json',accepted)
   assert d._beam_predecessors is before
   activity=[]
   for old,new in zip(before[0].stations,d._beam_origins[0].stations):
    for a,b in zip(old.rows,new.rows):
     dz=paired_value(b)-paired_value(a);da=paired_value(b,2)-paired_value(a,2)
     activity.append(dict(plastic_increment=float(dz),accumulated_increment=float(da),
      opposite_to_parent=bool(paired_value(a)*dz<0 and da>0)))
     assert da>=0
   write(out/('activity-%02d.json'%(len(accepted)-1)),activity)
  try:
   result=owner.solve(load,(2.5,),increment_control=AdaptiveIncrementControl(4,12),max_iterations=20,max_backtracks=8,on_accepted=observe)
  except BaseException as error:
   write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempts=[asdict(a) for a in getattr(error,'attempts',())],
    cause=None if error.__cause__ is None else repr(error.__cause__)))
   raise
  finally:
   (out/'terminal-prefix.bin').write_bytes(owner.checkpoint());write(out/'terminal-atomic.json',unchanged(latest[0]))
  # Compare actual endpoint histories with the earlier source oracle. The new
  # predictor changes iteration/journal bytes; report differences, never assert
  # a manufactured bytewise identity or introduce a new scientific tolerance.
  current=dict(history_arrays(owner._beam_parts,d._beam_predecessors,'origin')+history_arrays(owner._beam_parts,d._beam_origins,'accepted'))
  with np.load(ORACLE/'histories-01.npz',allow_pickle=False) as oracle:
   comparison={k:dict(shape=list(v.shape),exact_equal=bool(np.array_equal(v,oracle[k])),max_abs_difference=float(np.max(np.abs(v-oracle[k])))) for k,v in current.items()}
  write(out/'source-oracle-comparison.json',dict(histories=comparison,source_prefix_sha256=sha(ORACLE/'final-prefix.bin'),
   current_prefix_sha256=sha(out/'terminal-prefix.bin'),source_accepted_path=[2.25,2.5],resumed_accepted_path=[r['factor'] for r in accepted],
   prefix_equal=(out/'terminal-prefix.bin').read_bytes()==(ORACLE/'final-prefix.bin').read_bytes(),new_predictor_journal_expected=True))
  write(out/'result.json',dict(status='ACCEPTED',factor=result.load_factor,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
   actual_load=d._programme.load_at(result.load_factor).tolist(),accepted=accepted,rejected=rejected,predictors=predictors,
   completed=d._adaptive_state.completed,rollback=unchanged(latest[0])))
  assert result.load_factor==2.5 and d._adaptive_state.completed and all(unchanged(latest[0]).values())
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
 finally:owner.close()
