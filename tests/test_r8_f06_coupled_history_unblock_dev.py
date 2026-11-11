"""Observe the actual exhausted fixed parent; evaluate only fixed same-parent probes."""
import os,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_COUPLED_HISTORY_UNBLOCK_OUTPUT' not in os.environ,reason='explicit bounded supervisor required')

def test_direction():
 import sys,copy
 from dataclasses import asdict
 from fractions import Fraction as F
 import numpy as np
 from scipy import sparse
 from scripts.r8_f06_coupled_history_unblock_dev import INPUT,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
 from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
 import anyfesolver._stiffened_shell_ge_finite_experimental as finite
 out=Path(os.environ['F06_COUPLED_HISTORY_UNBLOCK_OUTPUT']);owner=ss.create_analysis(definition=definition('coupled'))
 state_fields=('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors')
 originals={};d=None
 try:
  owner.restore(INPUT.read_bytes());d=owner._finite_driver
  before=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors,d._adaptive_state)
  def atomic():
   now=(owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors,d._adaptive_state)
   return dict(checkpoint=now[0]==before[0],generation=now[1]==before[1],objects=all(x is y for x,y in zip(now[2:],before[2:])))
  (out/'initial-prefix.bin').write_bytes(before[0]);(out/'terminal-prefix.bin').write_bytes(before[0])
  assert before[0]==INPUT.read_bytes() and d._load_factor==.396875 and d._adaptive_state is not None
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
  local_trials={};current_parts={};records=[];by_state={};parents=[];trial_info={};phase=['production']
  originals=dict(assemble=d._assemble,advance=d._advance,q4=finite.q4_algorithmic_response,spatial=finite.spatial_jacobian)
  def q4(*args,**kwargs):
   value=originals['q4'](*args,**kwargs);frame=sys._getframe(1).f_locals
   local_trials['shell']={k:v.copy() for k,v in value[2].items() if isinstance(v,np.ndarray)}
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
   for family,values in record['local_trials'].items():np.savez(out/(label+'-'+family+'-material.npz'),**values)
   for family,(residual,matrix) in record['components'].items():
    np.save(out/(label+'-'+family+'-R.npy'),residual);sparse.save_npz(out/(label+'-'+family+'-K.npz'),matrix)
   write(out/(label+'.json'),record['metadata'])
  def assembled(state,*args,**kwargs):
   current_parts.clear();local_trials.clear();frame=sys._getframe(2).f_locals if phase[0]=='production' else {}
   value=originals['assemble'](state,*args,**kwargs)
   assert args[0]==.41 and args[1] is before[5] and args[2] is before[3]
   applied=.41*kwargs['pattern'];components={k:embed(v) for k,v in current_parts.items()}
   rest=value[0].copy();rest[:count]+=applied;restK=value[1].copy()
   for residual,matrix in components.values():rest-=residual;restK-=matrix
   components['point_constraints']=(rest,restK)
   info=trial_info.pop(id(state),{});histories=packed(copy.deepcopy(value[4]),copy.deepcopy(value[5]))
   label=('production-%02d'%len(records)) if phase[0]=='production' else phase[0]
   record=dict(label=label,original_state=state,state=copy.deepcopy(state),args=args,kwargs={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in kwargs.items()},
    local_trials=copy.deepcopy(local_trials),state_arrays={k:getattr(state,k).copy() for k in state_fields if getattr(state,k) is not None},residual=value[0].copy(),matrix=value[1].copy(),histories=histories,components=components,
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
  def forbid_publication(attempt):
   write(out/'unexpected-equilibrium.json',asdict(attempt))
   raise RuntimeError('Diagnostic forbids equilibrium publication')
  np.save(out/'actual-pattern.npy',d._pattern)
  write(out/'controller-before.json',asdict(d._adaptive_state))
  d._assemble=assembled;d._advance=advanced
  try:
   d._solve_fixed(actions('coupled'),(.41,),max_iterations=20,max_backtracks=8,
    _prepared_pattern=d._pattern.copy(),_accepted_hook=forbid_publication)
   pytest.fail('diagnostic expected original exhausted parent, not a new accepted endpoint')
  except FiniteIncrementFailure as error:
   write(out/'fixed-failure.json',dict(attempt=asdict(error.attempt),atomic=atomic(),diagnostic_no_publication=True))
   assert error.attempt.failure=='finite mixed backtrack budget exhausted'
  finally:d._advance=originals['advance'];(out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
  assert all(atomic().values());parent=parents[-1];base=parent['record'];p=parent['direction'];K=parent['matrix'];R=parent['residual']
  assert np.array_equal(R,base['residual']) and (K-base['matrix']).nnz==0
  action=K@p;linear=action+R
  np.savez(out/'linear-equation.npz',action=action,residual=R,direction=p,backerror=linear)
  critical=np.argsort(np.abs((weight*R)[free]))[-12:]
  write(out/'critical-rows.json',dict(rows=[dict(row=int(free[i]),residual=float(R[free[i]]),weight=float(weight[free[i]]),direction=float(p[free[i]])) for i in critical],constraint_work=float(base['state'].multipliers@R[count:]),directional_residual_work=float(p[free]@R[free])))
  tiny=np.finfo(float).tiny
  linear_ratio=np.linalg.norm((weight*linear)[free])/max(np.linalg.norm((weight*action)[free])+np.linalg.norm((weight*R)[free]),tiny)
  actual_last=records[-1];assert actual_last['metadata']['trial']['amount']==1./256
  phase[0]='zero';zero_state=originals['advance'](base['state'],p,0.)
  assembled(zero_state,*base['args'],**base['kwargs']);zero=records[-1]
  zero_exact=np.array_equal(zero['residual'],R) and (zero['matrix']-K).nnz==0
  probes=[]
  for h in (1e-6,):
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
   production_assemblies=sum(r['metadata']['phase']=='production' for r in records),diagnostic_assemblies=2,
   accepted_trials=0,atomic=atomic())
  write(out/'result.json',result);assert all(atomic().values())
 finally:
  if originals:
   d._assemble=originals['assemble'];d._advance=originals['advance'];finite.q4_algorithmic_response=originals['q4'];finite.spatial_jacobian=originals['spatial']
  if d is not None:
   (out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
   write(out/'terminal-atomic.json',atomic())
  owner.close()

def test_local_chain():
 import numpy as np
 from scipy import sparse
 from scripts.r8_f06_coupled_history_unblock_dev import DOC,INPUT,write
 from scripts.r8_f06_cyclic_material_dev import definition
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._ge_beam3_variational_shell import deformation
 from anyfesolver._ge_beam3_pose_joint import _exp_terms
 from anyfesolver._stiffened_shell_algorithmic_pullback import spatial_response
 import anyfesolver.plasticity as pl
 out=Path(os.environ['F06_COUPLED_HISTORY_UNBLOCK_OUTPUT']);saved=DOC/'attempt-diagnostic1'
 owner=ss.create_analysis(definition=definition('coupled'));original_kernel=pl._jit_plane_stress_return_map;calls=[]
 try:
  owner.restore(INPUT.read_bytes());d=owner._finite_driver;before=owner.checkpoint();generation=owner._generation
  identities=(d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors,d._adaptive_state)
  with np.load(saved/'production-13.npz') as z:state={k:z[k].copy() for k in z.files}
  with np.load(saved/'last-parent-direction.npz') as z:p=z['direction'].copy()
  eid,element,slots,_=owner._shell_parts[0];u=state['q'][slots];v=p[slots];anchors=state['shell_anchors'][slots[::6]//6]
  origin=d._shell_origins[0];material=owner._shell_model.get_material('shell');curve=material.hardening_curve
  with np.load(saved/'accepted-origin.npz') as z:
   assert np.array_equal(origin['plastic_strain'],z['shell_plastic'][0]) and np.array_equal(origin['alpha'],z['shell_alpha'][0])
  reference=np.array([owner._shell_model.mesh.nodes[n].coords() for n in element.node_ids]);h=1e-6
  local,D,H=deformation(reference,u,stable_translation=True,rotation_anchors=anchors)
  import sys
  previous_profile=sys.getprofile();kernel_code=original_kernel.__code__
  def observed(frame,event,answer):
   if event=='return' and frame.f_code is kernel_code and answer is not None:
    calls.append(dict(converged=answer[4].tolist(),scaled_residual=answer[5].tolist(),active=(answer[3]>frame.f_locals['alpha']).tolist(),
     alpha=answer[3].tolist(),hardening_distance_y1=(answer[3]-curve.eps_p_y1).tolist(),hardening_distance_y2=(answer[3]-curve.eps_p_y2).tolist()))
  sys.setprofile(observed)
  def response(x):return element.compute_nonlinear_response(owner._shell_model.mesh,material,x,origin,3,True)
  f,C,history=response(local);r,K=spatial_response(u,f,C,D,H)
  savedK=sparse.load_npz(saved/'production-13-shell-K.npz')[slots][:,slots].toarray();savedR=np.load(saved/'production-13-shell-R.npy')[slots]
  with np.load(saved/'production-13-shell-material.npz') as z:strain=z['layer_strain'].copy()
  exact_parent=np.array_equal(K,savedK) and np.array_equal(r,savedR) and np.array_equal(strain,history['layer_strain'])
  write(out/'parent-identity.json',dict(exact=exact_parent,input_sha256=__import__('hashlib').sha256(INPUT.read_bytes()).hexdigest(),h=h))
  assert exact_parent
  curve_stage=lambda alpha:np.where(alpha<curve.eps_p_y1,0,np.where(alpha<curve.eps_p_y2,1,2))
  def branch(history):return (history['alpha']>origin['alpha']).tolist(),curve_stage(history['alpha']).tolist()
  actual=[];affine=[];covectors=[];strains=[];branches=[];maps=[]
  for sign in (-1,1):
   uq=u+sign*h*v;lq,Dq,Hq=deformation(reference,uq,stable_translation=True,rotation_anchors=anchors)
   fq,Cq,sq=response(lq);rq,_=spatial_response(uq,fq,Cq,Dq,Hq)
   fa=f+C@(lq-local);ra,_=spatial_response(uq,fa,C,Dq,Hq)
   actual.append(rq);affine.append(ra);covectors.append(Dq.T@fa);strains.append(sq['layer_strain']);branches.append(branch(sq));maps.append(lq)
  local_v=D@v;local_pair=[response(local+sign*h*local_v) for sign in (-1,1)]
  strain_v=(strains[1]-strains[0])/(2*h)
  sig,Cmat,ep,alpha=pl.plane_stress_return_map(strain,origin['plastic_strain'],origin['alpha'],material.elastic_modulus,material.poisson_ratio,curve)
  sig_pair=[pl.plane_stress_return_map(strain+sign*h*strain_v,origin['plastic_strain'],origin['alpha'],material.elastic_modulus,material.poisson_ratio,curve) for sign in (-1,1)]
  # Complementary existing full material oracle, with its own declared default strain step.
  oracle=pl.plane_stress_tangent_diagnostics(strain,origin['plastic_strain'],origin['alpha'],material.elastic_modulus,material.poisson_ratio,curve)
  fallback=pl._analytical_tangent_fallback_mask(Cmat,E=material.elastic_modulus,nu=material.poisson_ratio)
  derivative=D.T@C@D+np.einsum('i,ijk->jk',f,H);cov=D.T@f;dc=derivative@v
  chart_values=[]
  for sign in (-1,1):
   Aq=np.eye(24)
   for i,row in enumerate((u+sign*h*v).reshape(-1,6)):Aq[6*i+3:6*i+6,6*i+3:6*i+6]=_exp_terms(row[3:])[1]
   chart_values.append(np.linalg.solve(Aq.T,cov+sign*h*dc))
  def error(actual,expected):
   absolute=float(np.linalg.norm(actual-expected));norm=float(np.linalg.norm(expected))
   return dict(absolute_error=absolute,action_norm=norm,relative_error=absolute/max(norm,1e-12),absolute_floor=1e-12)
  def central(xs):return (xs[1]-xs[0])/(2*h)
  result=dict(status='LOCAL_DIAGNOSTIC_COMPLETE',h=h,parent_exact=exact_parent,centered_branch_same=all(x==branch(history) for x in branches),
   local_material_branch_same=all(((x[3]>origin['alpha']).tolist(),curve_stage(x[3]).tolist())==branch(history) for x in sig_pair),
   actual_shell=error(central(actual),K@v),local_element=error((local_pair[1][0]-local_pair[0][0])/(2*h),C@local_v),
   local_material=error((sig_pair[1][0]-sig_pair[0][0])/(2*h),np.einsum('nij,nj->ni',Cmat,strain_v)),
   deformation_covector=error(central(covectors),dc),affine_material_full_map=error(central(affine),K@v),chart_only=error(central(chart_values),K@v),
   deformation_direction=error(central(maps),local_v),material_oracle=oracle,fallback_mask=fallback.tolist(),
   criterion=1e-5,criterion_scope='diagnostic observations; one centered step does not establish limiting plateau or full scientific acceptance',
   atomic=dict(checkpoint=owner.checkpoint()==before,generation=owner._generation==generation,objects=all(a is b for a,b in zip(identities,(d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors,d._adaptive_state)))))
  np.savez(out/'chain.npz',u=u,direction=v,anchors=anchors,local=local,D=D,H=H,force=f,local_tangent=C,residual=r,tangent=K,strain=strain,strain_direction=strain_v,material_tangent=Cmat,plastic=origin['plastic_strain'],alpha=origin['alpha'],actual_minus=actual[0],actual_plus=actual[1])
  write(out/'local-solver-metadata.json',calls);write(out/'result.json',result);(out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
  assert all(result['atomic'].values())
 finally:
  if 'previous_profile' in locals():sys.setprofile(previous_profile)
  write(out/'profile-restoration.json',dict(kernel_identity=pl._jit_plane_stress_return_map is original_kernel,profile_restored=('previous_profile' not in locals() or sys.getprofile() is previous_profile)))
  owner.close()
