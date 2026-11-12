"""One same-parent loaded-x witness, no target chosen by a sweep."""
import os,json,runpy
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_LOADED_X_OUTPUT' not in os.environ,reason='bounded installed supervisor required')
OLD=runpy.run_path(str(Path(__file__).with_name('test_r8_f06_paired_displacement_dev.py')))
RAW_UNCHANGED=OLD['unchanged'];capture=OLD['capture'];identity=OLD['identity'];activity=OLD['activity']
def checked(flags):
 assert flags and all(type(v) is bool and v for v in flags.values()),flags
 return True

def install_checked_helpers():
 out=Path(os.environ['F06_LOADED_X_OUTPUT'])
 def atomic(owner,before):
  flags=RAW_UNCHANGED(owner,before)
  with (out/'atomic-flags.jsonl').open('a') as f:f.write(json.dumps(flags)+'\n')
  return checked(flags)
 for name in ('test_source_options_and_refusals','test_source_actual_trial_cancellation','test_history','test_installed_restore_rebase_and_refusals'):
  OLD[name].__globals__['unchanged']=atomic
 return atomic

def loaded_inputs():
 from scripts.r8_f06_paired_loaded_x_dev import DOC,INPUT,INPUT_SHA,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 target=json.loads((DOC/'attempt-installed-witness1/witness.json').read_text())['target']
 return ss.create_analysis(definition=definition('coupled')),actions('coupled'),ss.DisplacementControl('shell',2,0,force_seed_sha256=INPUT_SHA),target,INPUT,DOC,write

def test_source_atomic_negative_calibration():
 checked(dict(checkpoint=True,generation=True,objects=True))
 for key in ('checkpoint','generation','objects'):
  flags=dict(checkpoint=True,generation=True,objects=True);flags[key]=False
  with pytest.raises(AssertionError):checked(flags)
 Path(os.environ['F06_LOADED_X_OUTPUT'],'atomic-calibration.json').write_text(json.dumps({'false_checkpoint_generation_objects_each_refused':True}))

def test_source_refusals():
 install_checked_helpers();OLD['test_source_options_and_refusals']()

@pytest.mark.parametrize('visit',(1,2))
def test_source_actual_trial_cancellation(visit):
 install_checked_helpers();OLD['test_source_actual_trial_cancellation'](visit)

def test_witness():
 import numpy as np
 from scipy import sparse
 from scipy.sparse.linalg import spsolve
 from dataclasses import asdict
 from decimal import Decimal as D,localcontext
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_beam_history import arrays
 from anyfesolver.plasticity import plane_stress_elastic_matrix,_jit_flow_stress
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from scripts.r8_f06_paired_loaded_x_dev import DOC,INPUT,INPUT_SHA,write,sha
 out=Path(os.environ['F06_LOADED_X_OUTPUT']);owner=ss.create_analysis(definition=definition('coupled'));before=None
 def plain(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
  if isinstance(v,(tuple,list)):return [plain(x) for x in v]
  return v
 def pair(v):return D.from_float(float(v[0]))+D.from_float(float(v[1]))
 try:
  owner.restore(INPUT.read_bytes());d=owner._finite_driver;before=identity(owner);controller=d._adaptive_state
  assert sha(INPUT)==INPUT_SHA and d._load_factor==.396875
  original=capture(owner);np.savez(out/'initial.npz',**original);(out/'initial.bin').write_bytes(before[0])
  shell=d._shell_origins;beam=d._beam_origins;state=d._increment_origin(d._state)
  pattern=d._load(actions('coupled'));assert np.array_equal(pattern,d._pattern)
  n=owner._K.shape[0];free=np.r_[owner._free,np.arange(n,n+len(state.multipliers))];rhs=np.r_[pattern,np.zeros(len(state.multipliers))]
  def save_evaluation(label,s,e):
   packed=dict(arrays(owner._beam_parts,e[5],'accepted'))
   packed['accepted_shell_alpha']=np.asarray([x['alpha'] for x in e[4]]).copy();packed['accepted_shell_plastic_strain']=np.asarray([x['plastic_strain'] for x in e[4]]).copy()
   state_values={k:getattr(s,k).copy() for k in ('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors')}
   np.savez(out/(label+'.npz'),**state_values,**packed,R=e[0],pattern=pattern,free=free);sparse.save_npz(out/(label+'-K.npz'),e[1])
   rows={'fibre':[],'generalized':[],'shell':[]};recoveries=[]
   with localcontext() as ctx:
    ctx.prec=80
    for i,((element,_,_),parent) in enumerate(zip(owner._beam_parts,beam)):
     if hasattr(element.section,'fibres'):
      rec=element.operator.recover_material(s.cells[i],s.resultants[i],origin=parent,rotations_low=s.cells_low[i],resultants_low=s.resultants_low[i])
      for station_index,(row,origin) in enumerate(zip(rec.stations,parent.stations)):
       for fibre,field,history in zip(element.section.fibres,row['fibres'],origin.rows):
        z0=pair(history[:2]);p0=pair(history[2:]);strain=pair(field.total_strain);drive=D.from_float(fibre.young)*(strain-z0);radius=fibre.curve._at(p0)[0]
        rows['fibre'].append(dict(element=element.element_id,station=station_index,fibre=fibre.fibre_id,origin_plastic=str(z0),origin_accumulated=str(p0),strain=str(strain),drive=str(drive),radius=str(radius),margin=str(abs(drive)-radius),branch=field.branch,plastic_increment=list(field.plastic_increment)))
      recoveries.append(dict(element=element.element_id,stored_energy=rec.stored_energy,incremental_dissipation=rec.incremental_dissipation))
    # This formatter consumes the actual assembly cache; no extra generalized evaluation.
    for rec in d._generalized_records(s,beam,e[5]):
     i=next(i for i,(part,_,_) in enumerate(owner._beam_parts) if part.element_id==rec.element_id);section=owner._beam_parts[i][0].section
     for j,(row,origin) in enumerate(zip(rec.stations,beam[i].stations)):
      values=[pair((h,l)) for h,l in zip(row.local_resultants,row.local_resultants_low)];q=sum(values[a]*D.from_float(float(section.metric[a,b]))*values[b] for a in range(6) for b in range(6)).sqrt();radius=D.from_float(section.yield_force)+D.from_float(section.hardening)*pair(origin.accumulated)
      rows['generalized'].append(dict(element=rec.element_id,station=j,resultants=[str(x) for x in values],strain=plain(row.local_strain),strain_low=plain(row.local_strain_low),origin_accumulated=str(pair(origin.accumulated)),drive=str(q),radius=str(radius),margin=str(q-radius),branch=row.branch,plastic_increment=list(row.plastic_increment)))
   material=owner._shell_model.get_material('shell');curve=material.hardening_curve;C=plane_stress_elastic_matrix(material.elastic_modulus,material.poisson_ratio)
   for i,(origin,candidate) in enumerate(zip(shell,e[4])):
    trial=(candidate['layer_strain']-origin['plastic_strain'])@C.T
    flow=_jit_flow_stress(np.asarray(origin['alpha']),*(float(getattr(curve,k)) for k in ('sigma_prop','sigma_yield','sigma_yield_2','eps_p_y1','eps_p_y2','K','n','_power_offset')))
    for j,(stress,radius) in enumerate(zip(trial,flow)):
     q=float(np.sqrt(stress[0]**2-stress[0]*stress[1]+stress[1]**2+3*stress[2]**2))
     rows['shell'].append(dict(element=i,layer=j,strain=candidate['layer_strain'][j].tolist(),trial_stress=stress.tolist(),drive=q,radius=float(radius),margin=q-float(radius),origin_alpha=float(origin['alpha'][j]),alpha=float(candidate['alpha'][j])))
   write(out/(label+'-yield.json'),rows);write(out/(label+'-energy.json'),plain(recoveries))
   write(out/(label+'-physical.json'),dict(equilibrium=e[2],gap=e[3],constraint_work=float(s.multipliers@e[0][n:]),load_coefficient=.396875 if label=='parent' else .41,work=float(pattern@((s.q+ s.q_low)-(state.q+state.q_low)))))
   return packed,rows
  parent=d._assemble(state,.396875,shell,beam,pattern=pattern);parent_pack,parent_rows=save_evaluation('parent',state,parent)
  direction=np.zeros(len(rhs));direction[free]=spsolve(parent[1][free,:][:,free],rhs[free]);linear=parent[1]@direction-rhs
  delta=.41-.396875;predicted=d._advance(state,direction,delta);target=float(predicted.q[6]);np.savez(out/'loading-direction.npz',direction=direction,linear_residual=linear,free=free,delta=delta,target=target)
  actual=d._assemble(predicted,.41,shell,beam,pattern=pattern);predicted_pack,predicted_rows=save_evaluation('predictor',predicted,actual)
  change=activity(parent_pack,predicted_pack)
  margins={family:sum(float(x['margin'])>0 for x in rows) for family,rows in predicted_rows.items()}
  flags=RAW_UNCHANGED(owner,before);flags['controller']=d._adaptive_state is controller
  residual=float(np.linalg.norm(linear[free])/max(np.linalg.norm(rhs[free]),1.))
  eligible=bool(np.isfinite(direction).all() and target>state.q[6] and residual<=1e-8 and all(margins[x]>0 and change[x]['positive']>0 and change[x]['negative']==0 for x in ('shell','fibre','generalized')) and all(flags.values()))
  result=dict(status='WITNESS_ONLY',eligible=eligible,target=target,initial=float(state.q[6]),delta_force=delta,control=['shell',2,0],linear_relative_residual=residual,parent_equilibrium=parent[2],parent_gap=parent[3],activity=change,positive_trial_margin_counts=margins,atomic=flags,assembly_evaluations=2,fibre_recovery_cell_solves=2,accepted=False,force_seed_sha256=INPUT_SHA)
  write(out/'witness.json',result);(out/'terminal.bin').write_bytes(owner.checkpoint());assert all(flags.values());assert eligible,result
 finally:
  if before is not None:
   flags=RAW_UNCHANGED(owner,before);flags['controller']=owner._finite_driver._adaptive_state is controller;write(out/'terminal-atomic.json',flags);(out/'terminal.bin').write_bytes(owner.checkpoint())
  owner.close()

def test_history():
 install_checked_helpers();OLD['test_history'].__globals__['inputs']=loaded_inputs;OLD['test_history']()

def test_durability():
 install_checked_helpers();OLD['test_installed_restore_rebase_and_refusals'].__globals__['inputs']=loaded_inputs;OLD['test_installed_restore_rebase_and_refusals']()
