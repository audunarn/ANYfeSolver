"""Observe the actual exhausted fixed parent; evaluate only fixed same-parent probes."""
import os,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_COUPLED_DIRECTION_OUTPUT' not in os.environ,reason='explicit bounded supervisor required')

def test_direction():
 import sys,copy
 from dataclasses import asdict
 from fractions import Fraction as F
 import numpy as np
 from scipy import sparse
 from scripts.r8_f06_coupled_direction_dev import INPUT,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 import anyfesolver._stiffened_shell_ge_finite_experimental as finite
 out=Path(os.environ['F06_COUPLED_DIRECTION_OUTPUT']);owner=ss.create_analysis(definition=definition('coupled'))
 state_fields=('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors')
 originals={};d=None
 try:
  owner.restore(INPUT.read_bytes());d=owner._finite_driver
  before=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors,d._adaptive_state)
  def atomic():
   now=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors,d._adaptive_state)
   return dict(checkpoint=now[0]==before[0],generation=now[1]==before[1],objects=all(x is y for x,y in zip(now[2:],before[2:])))
  (out/'initial-prefix.bin').write_bytes(before[0]);(out/'terminal-prefix.bin').write_bytes(before[0])
  assert before[0]==INPUT.read_bytes() and d._load_factor==.2 and d._adaptive_state is None
  assert d._external is None and d._programme is None and not owner._definition.lines and not owner._definition.footprints
  count=owner._K.shape[0];size=count+len(d._state.multipliers);free=np.r_[owner._free,np.arange(count,size)]
  weight=np.r_[owner._coordinate_scale,np.ones(size-count)]
  groups={'shell':owner._free[owner._free<owner._shell_count], 'constraints':np.arange(count,size)}
  for i,(part,slots,_) in enumerate(owner._beam_parts):
   family='fibre' if type(part.operator).__name__=='RetainedFibreOperator' else 'generalized'
   groups[family]=np.intersect1d(slots+owner._shell_count,free)
  np.savez(out/'row-maps.npz',free=free,weight=weight,**groups)
  def packed(shell,beam):
   values=dict(history_arrays(owner._beam_parts,beam,'native'))
   values.update(shell_alpha=np.asarray([s['alpha'] for s in shell]).copy(),shell_plastic=np.asarray([s['plastic_strain'] for s in shell]).copy())
   return values
  origins=packed(d._shell_origins,d._beam_origins);predecessors=packed(d._shell_predecessors,d._beam_predecessors)
  np.savez(out/'accepted-origin.npz',**origins);np.savez(out/'checkpoint-predecessor.npz',**predecessors)
  def branch(values):
   pair=lambda x:F(float(x[0]))+F(float(x[1]))
   return dict(shell=(values['shell_alpha']>origins['shell_alpha']).ravel().tolist(),
    fibre=[pair(x[2:])-pair(y[2:])>0 for x,y in zip(values['native_fibres'],origins['native_fibres'])],
    fibre_sign=[int(pair(x[:2])>pair(y[:2]))-int(pair(x[:2])<pair(y[:2])) for x,y in zip(values['native_fibres'],origins['native_fibres'])],
    generalized=[pair(x)>pair(y) for x,y in zip(values['native_accumulated'],origins['native_accumulated'])])
  def embed(parts):
   residual=np.zeros(size);row=[];col=[];data=[]
   for slots,r,k in parts:
    np.add.at(residual,slots,r);row.extend(np.repeat(slots,len(slots)));col.extend(np.tile(slots,len(slots)));data.extend(k.ravel())
   return residual,sparse.coo_matrix((data,(row,col)),shape=(size,size)).tocsc()
  current_parts={};records=[];by_state={};parents=[];trial_info={};phase=['production']
  originals=dict(assemble=d._assemble,advance=d._advance,q4=finite.q4_algorithmic_response,spatial=finite.spatial_jacobian)
  def q4(*args,**kwargs):
   value=originals['q4'](*args,**kwargs);frame=sys._getframe(1).f_locals
   current_parts.setdefault('shell',[]).append((frame['slots'].copy(),value[0].copy(),value[1].copy()))
   return value
  def spatial(*args,**kwargs):
   value=originals['spatial'](*args,**kwargs);frame=sys._getframe(1).f_locals
   part=owner._beam_parts[frame['i']][0];family='fibre' if type(part.operator).__name__=='RetainedFibreOperator' else 'generalized'
   current_parts.setdefault(family,[]).append(((frame['slots']+owner._shell_count).copy(),frame['response'].residual.copy(),value.copy()))
   return value
  finite.q4_algorithmic_response=q4;finite.spatial_jacobian=spatial
  def persist(record):
   label=record['label'];np.savez(out/(label+'.npz'),residual=record['residual'],**record['state_arrays'],**record['histories'])
   sparse.save_npz(out/(label+'-K.npz'),record['matrix'])
   for family,(residual,matrix) in record['components'].items():
    np.save(out/(label+'-'+family+'-R.npy'),residual);sparse.save_npz(out/(label+'-'+family+'-K.npz'),matrix)
   write(out/(label+'.json'),record['metadata'])
  def assembled(state,*args,**kwargs):
   current_parts.clear();frame=sys._getframe(2).f_locals if phase[0]=='production' else {}
   value=originals['assemble'](state,*args,**kwargs)
   assert args[0]==.41 and args[1] is before[5] and args[2] is before[3]
   applied=.41*kwargs['pattern'];components={k:embed(v) for k,v in current_parts.items()}
   rest=value[0].copy();rest[:count]+=applied;restK=value[1].copy()
   for residual,matrix in components.values():rest-=residual;restK-=matrix
   components['point_constraints']=(rest,restK)
   info=trial_info.pop(id(state),{});histories=packed(copy.deepcopy(value[4]),copy.deepcopy(value[5]))
   label=('production-%02d'%len(records)) if phase[0]=='production' else phase[0]
   record=dict(label=label,original_state=state,state=copy.deepcopy(state),args=args,kwargs={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in kwargs.items()},
    state_arrays={k:getattr(state,k).copy() for k in state_fields if getattr(state,k) is not None},residual=value[0].copy(),matrix=value[1].copy(),histories=histories,components=components,
    metadata=dict(phase=phase[0],iteration=frame.get('iteration'),trial=info,equilibrium=float(value[2]),gap=float(value[3]),
     counts=dict(frame.get('counts',{})),branch=branch(histories)))
   persist(record);records.append(record);by_state[id(state)]=record
   with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(dict(label=label,**record['metadata']))+'\n')
   return value
  def advanced(state,direction,amount):
   frame=sys._getframe(1).f_locals
   if amount==1.:
    record=by_state[id(state)];parent=dict(record=record,direction=direction.copy(),matrix=frame['matrix'].copy(),residual=frame['residual'].copy(),
     iteration=frame['iteration'],old_merit=float(frame['old_merit']),merit_applied=None if frame['merit_applied'] is None else frame['merit_applied'].copy(),
     counts=dict(frame['counts']))
    assert frame['merit_applied'] is None,'declared dead-load path must use its actual physical eq/gap merit'
    parents.append(parent);np.savez(out/'last-parent-direction.npz',direction=direction,residual=frame['residual'],applied=.41*record['kwargs']['pattern'])
    sparse.save_npz(out/'last-parent-K.npz',frame['matrix']);persist(record)
    write(out/'last-parent.json',dict(label=record['label'],iteration=parent['iteration'],old_merit=parent['old_merit'],counts=parent['counts'],merit_applied=None))
   candidate=originals['advance'](state,direction,amount);trial_info[id(candidate)]=dict(amount=float(amount),backtrack=frame['backtrack'])
   return candidate
  d._assemble=assembled;d._advance=advanced
  try:
   owner.solve(actions('coupled'),(.41,),max_iterations=20,max_backtracks=8)
   pytest.fail('diagnostic expected original exhausted parent, not a new accepted endpoint')
  except FiniteIncrementFailure as error:
   write(out/'fixed-failure.json',dict(attempt=asdict(error.attempt),atomic=atomic()))
   assert error.attempt.failure=='finite mixed backtrack budget exhausted'
  finally:d._advance=originals['advance'];(out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
  assert all(atomic().values());parent=parents[-1];base=parent['record'];p=parent['direction'];K=parent['matrix'];R=parent['residual']
  assert np.array_equal(R,base['residual']) and (K-base['matrix']).nnz==0
  action=K@p;linear=action+R
  np.savez(out/'linear-equation.npz',action=action,residual=R,direction=p,backerror=linear)
  tiny=np.finfo(float).tiny
  linear_ratio=np.linalg.norm((weight*linear)[free])/max(np.linalg.norm((weight*action)[free])+np.linalg.norm((weight*R)[free]),tiny)
  actual_last=records[-1];assert actual_last['metadata']['trial']['amount']==1./256
  phase[0]='zero';zero_state=originals['advance'](base['state'],p,0.)
  assembled(zero_state,*base['args'],**base['kwargs']);zero=records[-1]
  zero_exact=np.array_equal(zero['residual'],R) and (zero['matrix']-K).nnz==0
  probes=[]
  for h in (1e-5,3e-6,1e-6):
   phase[0]='probe-'+str(h);state=originals['advance'](base['state'],p,h);assembled(state,*base['args'],**base['kwargs']);record=records[-1]
   groups_result={};component_result={}
   for name,ids in groups.items():
    expected=action[ids]*weight[ids];actual=(record['residual'][ids]-zero['residual'][ids])/h*weight[ids]
    error=float(np.linalg.norm(actual-expected));den=float(np.linalg.norm(expected))
    groups_result[name]=dict(absolute_error=error,action_norm=den,relative_error=error/max(den,tiny))
   for name,(r0,k0) in base['components'].items():
    expected=(k0@p)*weight;actual=(record['components'][name][0]-zero['components'][name][0])/h*weight
    error=float(np.linalg.norm((actual-expected)[free]));den=float(np.linalg.norm(expected[free]))
    component_result[name]=dict(absolute_error=error,action_norm=den,relative_error=error/max(den,tiny))
   row=dict(h=h,groups=groups_result,components=component_result,branch_equal=record['metadata']['branch']==base['metadata']['branch'],
    branch=record['metadata']['branch'],equilibrium=record['metadata']['equilibrium'],gap=record['metadata']['gap'],
    actual_merit=max(record['metadata']['equilibrium'],record['metadata']['gap']))
   probes.append(row);write(out/'probes.json',probes)
  result=dict(status='DIAGNOSTIC_COMPLETE',fixed_failure=True,failed_parent=parent['iteration'],old_merit=parent['old_merit'],
   linear_backward_error=float(linear_ratio),linear_absolute=float(np.max(np.abs(linear[free]),initial=0.)),zero_exact=zero_exact,
   zero_residual_max=float(np.max(np.abs(zero['residual']-R),initial=0.)),zero_branch_equal=zero['metadata']['branch']==base['metadata']['branch'],
   parent_branch=base['metadata']['branch'],probes=probes,last_existing_trial=actual_last['metadata'],
   smooth_plastic_reference_criterion=1e-5,derivative_denominator_floor=float(tiny),
   derivative_scaling='compiled coordinate work scale; unit multiplier rows, existing coupled derivative convention',
   production_assemblies=sum(r['metadata']['phase']=='production' for r in records),diagnostic_assemblies=4,
   accepted_trials=0,atomic=atomic())
  write(out/'result.json',result);assert all(atomic().values())
 finally:
  if originals:
   d._assemble=originals['assemble'];d._advance=originals['advance'];finite.q4_algorithmic_response=originals['q4'];finite.spatial_jacobian=originals['spatial']
  if d is not None:
   (out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
   write(out/'terminal-atomic.json',atomic())
  owner.close()
