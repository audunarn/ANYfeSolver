"""Fresh family-breadth controls under one bounded source/installed allowance."""
import json,os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_BREADTH_OUTPUT' not in os.environ,reason='bounded material-breadth supervisor required')

def test_family_classifier_and_partial_policy():
 from scripts.r8_f06_native_material_breadth_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_beam_history import compensated_kind
 for case,kind in (('axial','fibre'),('biaxial','fibre'),('mixed','mixed')):
  owner=ss.create_analysis(definition=definition(case))
  try:
   assert compensated_kind(owner._beam_parts)=='finite_compensated_'+kind+'_material'
   if case=='mixed':
    assert compensated_kind(tuple(reversed(owner._beam_parts)))=='finite_compensated_mixed_material'
    assert compensated_kind((owner._beam_parts[1],))=='finite_compensated_generalized_material'
  finally:owner.close()
 owner=ss.create_analysis(definition=definition('mixed',partial=True))
 try:
  with pytest.raises(ValueError,match='durable compensated'):owner.solve(actions('mixed'),(.4,))
  assert owner._finite_driver is None and owner._accepted is None
  with pytest.raises(ValueError,match='uniform'):compensated_kind(owner._beam_parts)
 finally:owner.close()

def test_biaxial_defining_law_and_spatial_derivative(monkeypatch):
 from scripts.r8_f06_native_material_breadth_dev import definition
 from scripts import r8_f06_fibre_log_domain_dev as old
 from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
 from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY
 import test_r8_f06_fibre_log_domain_dev as controls
 e,_=definition('biaxial').beams[0].instantiate()
 def physical(*,paired=True):
  return NativeFibreStaticElement(1,(101,102,103),e.operator.reference,e.section,order=4,arithmetic_policy=POLICY if paired else None)
 monkeypatch.setattr(old,'element',physical)
 monkeypatch.setenv('F06_FIBRE_OUTPUT',os.environ['F06_BREADTH_OUTPUT'])
 controls.test_active_paired_resultant_law_and_recovery()
 controls.test_active_spatial_derivative()

def test_public_material_lifecycle():
 import numpy as np
 from scripts.r8_f06_native_material_breadth_dev import definition,actions,DOC
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection
 from anyfesolver._stiffened_shell_beam_history import shapes,compensated_kind
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode,encode,SCHEMAS
 from anyfesolver._stiffened_shell_f06_increment_arithmetic import Pair
 import anyfesolver._stiffened_shell_f06_increment_arithmetic as arithmetic
 case=os.environ['F06_BREADTH_CASE'];out=Path(os.environ['F06_BREADTH_OUTPUT'])
 model=definition(case);owner=ss.create_analysis(definition=model);calls=[];native_calls=[];original=arithmetic.full_log_coefficients
 def phase(name):(out/'progress.json').write_text(json.dumps(dict(phase=name)))
 def capture(c):
  import sys
  calls.append(c.scalar())
  if sys._getframe(2).f_code.co_name=='retained_evaluation':native_calls.append(c.scalar())
  return original(c)
 arithmetic.full_log_coefficients=capture
 try:
  prefix=os.environ.get('F06_BREADTH_PREFIX')
  result=owner.restore(Path(prefix).read_bytes()) if prefix else owner.solve(actions(case),(.4,),max_iterations=16,max_backtracks=8)
  driver=owner._finite_driver;state=driver._state;expected=compensated_kind(owner._beam_parts)
  assert calls and min(calls)<1.-1e-7
  assert native_calls and min(native_calls)<1.-1e-7,'ordinary Log must be consumed by native retained evaluation'
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  accepted=canonical(driver._beam_origins);predecessor=canonical(driver._beam_predecessors)
  assert accepted!=predecessor
  material=[]
  for (element,_,_),current,previous in zip(owner._beam_parts,driver._beam_origins,driver._beam_predecessors):
   assert canonical(current)!=canonical(previous)
   fibre=type(element.section) is PhysicalFibreSection
   accumulation=max((row[2]+row[3] for s in current.stations for row in s.rows)) if fibre else max(sum(s.accumulated) for s in current.stations)
   assert accumulation>0.
   material.append(dict(member=element.element_id,family='physical_fibre' if fibre else 'generalized',accumulation=accumulation))
  recovery=owner.recover();assert recovery.beam_fibre_material
  if case.startswith('mixed'):assert recovery.beam_generalized_material
  raw=owner.checkpoint();(out/'accepted-active.bin').write_bytes(raw)
  phase('public-prefix-accepted')
  if prefix is None:
   target=DOC/(case+'-prefix.bin');assert not target.exists();target.write_bytes(raw)
  lows=('q_low','frames_low','cells_low','resultants_low')
  assert any(np.any(getattr(state,n)!=0) for n in lows)
  correction_shapes=tuple((n,getattr(state,n).shape) for n in lows)
  inner,corrections=decode_compensated(raw,owner._identity,correction_shapes)
  base_shapes=tuple((n,v.shape) for n,v in (('q',state.q),('frames',state.frames),('cells',state.cells),('resultants',state.resultants),
   ('multipliers',state.multipliers),('pattern',driver._pattern),('accepted_factors',np.asarray(driver._accepted_factors))))
  kind,generation,values=decode(inner,owner._identity,lambda k,g:base_shapes+shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted'))
  assert kind==expected
  def refuse(payload):
   fresh=ss.create_analysis(definition=model)
   try:
    with pytest.raises(ValueError):fresh.restore(payload)
    assert fresh._finite_driver is None and fresh._accepted is None
   finally:fresh.close()
  # The full mixed Q4 route covers both families' normal schemas/history arrays.
  # S3 and installed homogeneous companions keep their chart/physical lifecycle
  # checks without repeating the same six normal-format negative fresh owners.
  full_refusals=case!='mixed-s3' and not (prefix and case=='biaxial')
  if full_refusals:
   for wrong in ('finite_compensated_generalized_material','finite_compensated_fibre_material','finite_compensated_mixed_material'):
    if wrong!=kind:refuse(encode_compensated(encode(wrong,owner._identity,generation,tuple(values.items())),owner._identity,corrections))
   for name in ('origin_fibres','accepted_fibres','origin_accumulated','accepted_accumulated'):
    if not values[name].size:continue
    changed={k:v.copy() for k,v in values.items()};columns=slice(2,4) if name.endswith('fibres') else slice(0,2)
    pair=Pair(*changed[name][0,columns])+1e-4;changed[name][0,columns]=(pair.high,pair.low)
    refuse(encode_compensated(encode(kind,owner._identity,generation,tuple(changed.items())),owner._identity,corrections))
  if case=='mixed-s3':
   refuse(encode_compensated(encode('finite_compensated_fibre_material',owner._identity,generation,tuple(values.items())),owner._identity,corrections))
  phase('history-and-family-refusals-passed' if full_refusals else 'normal-refusals-covered-by-mixed-q4')
  fresh=ss.create_analysis(definition=model)
  try:
   fresh.restore(raw);assert fresh.checkpoint()==raw
   assert canonical(fresh._finite_driver._beam_origins)==accepted and canonical(fresh._finite_driver._beam_predecessors)==predecessor
   for n in ('q','frames','cells','resultants',*lows):np.testing.assert_array_equal(getattr(state,n),getattr(fresh._finite_driver._state,n))
   assert fresh._finite_driver._stable_s3==driver._stable_s3
  finally:fresh.close()
  saved_lows={n:getattr(state,n).copy() for n in lows}
  owner.rebase_rotations(owner._generation)
  assert canonical(driver._beam_origins)==accepted and canonical(driver._beam_predecessors)==predecessor
  for n,v in saved_lows.items():np.testing.assert_array_equal(getattr(driver._state,n),v)
  rebased=owner.checkpoint();phase('rebase-passed')
  from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_chart,encode_chart
  chart,chart_corrections=decode_compensated(rebased,owner._identity,correction_shapes)
  event,anchors,witness,numeric=decode_chart(chart,owner._identity,len(owner._shell_ids))
  chart_kind,steps,chart_values=decode(numeric,owner._identity,lambda k,g:base_shapes+shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted'))
  assert chart_kind==expected.replace('finite','finite_chart',1)
  wrong='finite_chart_compensated_fibre_material' if case.startswith('mixed') else 'finite_chart_compensated_mixed_material'
  refused_chart=encode_chart(owner._identity,event,anchors,witness,encode(wrong,owner._identity,steps,tuple(chart_values.items())))
  refuse(encode_compensated(refused_chart,owner._identity,chart_corrections))
  fresh=ss.create_analysis(definition=model)
  try:fresh.restore(rebased);assert fresh.checkpoint()==rebased
  finally:fresh.close()
  committed=driver._beam_origins;continued=owner.solve(actions(case),(.41,),max_iterations=16,max_backtracks=8)
  assert driver._beam_predecessors is committed and continued.accepted_factors[-1]==.41
  assert continued.free_equilibrium_ratio<=1e-8 and continued.constraint_max<=1e-9
  assert continued.beam_fibre_material
  if case.startswith('mixed'):assert continued.beam_generalized_material
  after=owner.checkpoint();phase('actual-continuation-passed');fresh=ss.create_analysis(definition=model)
  try:fresh.restore(after);assert fresh.checkpoint()==after
  finally:fresh.close()
  class Cancel(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':
     self.visits+=1
     if self.visits==2:self.cancel('mixed-family material candidate cancellation')
    super().raise_if_cancelled(stage)
  if case!='mixed-s3':
   cancel=Cancel()
   with pytest.raises(SolveCancelled):owner.solve(actions(case),(.42,),cancellation_token=cancel)
   assert cancel.visits==2 and owner.checkpoint()==after
   from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
   with pytest.raises(FiniteIncrementFailure):owner.solve(actions(case),(1.2,),max_iterations=1,max_backtracks=0)
   assert owner.checkpoint()==after
  metrics=dict(status='PASS',case=case,kind=expected,reused_prefix=prefix,material=material,
   ordinary_calls=len(calls),min_cosine=min(calls),native_ordinary_calls=len(native_calls),native_min_cosine=min(native_calls),equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
   continued_equilibrium=continued.free_equilibrium_ratio,continued_gap=continued.constraint_max,
   accepted_factors=list(continued.accepted_factors),both_histories_tamper_refused=full_refusals,wrong_family_kinds_refused=full_refusals,
   exact_restore_rebase=True,wrong_chart_kind_refused=True,s3_normal_kind_refused=case=='mixed-s3',
   cancellation_rejection_atomic=case!='mixed-s3',stable_s3=driver._stable_s3)
  (out/'public-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 finally:arithmetic.full_log_coefficients=original;owner.close()
