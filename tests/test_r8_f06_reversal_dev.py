"""Current paired-event controls; requires the bounded reversal supervisor."""
import os
import pytest
pytestmark=pytest.mark.skipif('F06_REVERSAL_OUTPUT' not in os.environ,reason='bounded reversal supervisor required')

def test_paired_event_witness_and_actual_update():
 import numpy as np
 from dataclasses import replace
 from decimal import Decimal as D,localcontext
 from pathlib import Path
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._ge_beam3_fibre_cell import PairedFibreTrialState
 from scripts import r8_f06_cyclic_material_dev as cyclic
 from scripts.r8_f06_log_domain_check_dev import write
 doc=cyclic.DOC.parent/'reversal-v1'
 owner=ss.create_analysis(definition=cyclic.definition('bend-z'))
 try:
  prefix=(cyclic.DOC/'attempt-source-bend-z2/failure-state.bin').read_bytes();owner.restore(prefix)
  driver=owner._finite_driver;assert driver._paired_fibre_event
  with np.load(doc/'attempt-capture1/parent-05.npz',allow_pickle=False) as data:
   keys=('q','frames','cells','resultants','multipliers','q_low','frames_low','cells_low','resultants_low')
   state=replace(driver._state,**{k:data[k].copy() for k in keys})
   direction=data['direction'].copy()
  driver._assemble(state,2.5,driver._shell_origins,driver._beam_origins,
   applied_load=driver._programme.load_at(2.5),load_programme=driver._programme)
  saved=driver._native_boundary_trial;trial=saved[3][0][1]
  assert type(trial) is PairedFibreTrialState and trial.smooth
  assert trial.resultants==tuple(state.resultants[0]) and trial.resultants_low==tuple(state.resultants_low[0])
  amount=driver._native_boundary_amount(state,direction)
  assert amount is not None and 0<amount<1
  # Independent affine event residual using the actual production state updater.
  advanced=driver._advance(state,direction,amount)
  cell=owner._beam_parts[0][0].operator.cell
  retained=owner._shell_count+driver._beam_nodal
  dp=direction[retained+6:retained+24]
  with localcontext() as ctx:
   ctx.prec=80;found=[]
   for strain,compliance,history in zip(trial.strains,trial.strain_compliances,cell._origins(trial.origin)):
    for fibre,(z0,p0) in zip(cell.section.fibres,history):
     b=cell.section._map(fibre)
     sensitivity=[sum(b[i]*compliance[i][j] for i in range(6)) for j in range(18)]
     e=sum(bi*ei for bi,ei in zip(b,strain));de=sum(a*D.from_float(float(v)) for a,v in zip(sensitivity,dp))
     if not de:continue
     for a in (p0,)+tuple(D.from_float(v) for v in fibre.curve.plastic_strain if D.from_float(v)>p0):
      for sign in (-1,1):
       boundary=z0+sign*(a-p0+fibre.curve._at(a)[0]/D.from_float(fibre.young))
       alpha=(boundary-e)/de
       if 0<alpha<1:found.append((alpha,e,de,boundary,sensitivity))
   alpha,e,de,boundary,sensitivity=min(found,key=lambda row:row[0])
   delta=[D.from_float(float(h))+D.from_float(float(l))-D.from_float(float(a))-D.from_float(float(b))
    for h,l,a,b in zip(advanced.resultants[0],advanced.resultants_low[0],state.resultants[0],state.resultants_low[0])]
   assert (e+sum(a*b for a,b in zip(sensitivity,delta))-boundary)*de>=0
  # Low-word mutation is invisible to ordinary rounded resultants but is refused.
  mutated=state.resultants_low.copy();mutated[0,0]=np.nextafter(mutated[0,0],np.inf)
  assert np.array_equal(state.resultants+mutated,state.resultants+state.resultants_low)
  with pytest.raises(ValueError,match='paired cell/origin/resultant'):
   cell.paired_boundary_amount(trial,dp,origin=trial.origin,resultants=state.resultants[0],resultants_low=mutated[0])
  with pytest.raises(ValueError,match='paired cell/origin/resultant'):
   cell.paired_boundary_amount(trial,dp,origin=replace(trial.origin),resultants=state.resultants[0],resultants_low=state.resultants_low[0])
  with pytest.raises(ValueError,match='stale native'):
   driver._native_boundary_amount(replace(state),direction)
  driver._native_boundary_trial=(saved[0],saved[1],saved[2]+1,saved[3])
  with pytest.raises(ValueError,match='stale native'):driver._native_boundary_amount(state,direction)
  driver._native_boundary_trial=saved
  assert cell.paired_boundary_amount(replace(trial,smooth=False),dp,origin=trial.origin,
   resultants=state.resultants[0],resultants_low=state.resultants_low[0]) is None
  assert cell.paired_boundary_amount(trial,np.zeros(18),origin=trial.origin,
   resultants=state.resultants[0],resultants_low=state.resultants_low[0]) is None
  assert owner.checkpoint()==prefix
  write(Path(os.environ['F06_REVERSAL_OUTPUT'])/'paired-control.json',dict(amount=amount,affine_first=str(alpha),
   both_halves_bound=True,actual_paired_crossing=True,stale_state_origin_generation_refused=True,accepted_prefix_unchanged=True))
 finally:owner.close()
