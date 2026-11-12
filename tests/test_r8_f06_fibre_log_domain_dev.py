"""New axial physical-fibre controls, only inside the fresh bounded supervisor."""
import os,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_FIBRE_OUTPUT' not in os.environ,reason='bounded fibre development supervisor required')

def test_legacy_definition_and_policy():
 import numpy as np
 from scripts.r8_f06_fibre_log_domain_dev import element
 from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition,SCHEMA,COMPENSATED_SCHEMA
 from anyfesolver._ge_beam3_retained_fibre import RetainedFibreOperator
 for paired,schema in ((False,SCHEMA),(True,COMPENSATED_SCHEMA)):
  e=element(paired=paired);d=NativeBeamDefinition.capture(e,np.eye(6))
  assert json.loads(d.raw)['schema']==schema
  rebuilt,_=d.instantiate()
  assert rebuilt.identity==e.identity and NativeBeamDefinition.capture(rebuilt,np.eye(6)).raw==d.raw
 e=element()
 for policy in (np.array([1,2]),{},False,'unknown'):
  with pytest.raises(ValueError):RetainedFibreOperator(e.operator.reference,e.section,order=4,arithmetic_policy=policy)

def test_active_paired_resultant_law_and_recovery():
 import numpy as np
 from decimal import Decimal as D,localcontext
 from scripts.r8_f06_fibre_log_domain_dev import element
 e=element();op=e.operator;p=np.zeros(18);p[0]=p[3]=1.;pl=np.zeros(18);pl[0]=pl[3]=2.**-55
 assert np.array_equal(p+pl,p)
 origin=op.cell.virgin();before=repr(origin)
 paired=op.cell.response(p.tolist(),origin,resultants_low=pl.tolist())
 dropped=op.cell.response(p.tolist(),origin)
 assert paired['history']!=dropped['history'],'low retained force did not reach active fibre return map'
 with localcontext() as ctx:
  ctx.prec=80
  stress=(D.from_float(1.)+D.from_float(2.**-55))/D.from_float(.01)
  alpha=(stress-D(10))/D(100000)
  tolerance=D(64)*D.from_float(np.finfo(float).eps)**2*max(abs(alpha),D('1e-30'))
  for station in paired['history'].stations:
   row=station.rows[0]
   assert abs(D.from_float(row[2])+D.from_float(row[3])-alpha)<=tolerance
 x=op.reference.coordinates.copy();x[1,0]=.5005;x[2,0]=1.001
 q=op.reference.nodal_triads;u=np.tile(np.eye(3),(2,1,1))
 response=op.evaluate(x,np.zeros_like(x),q,u,p,origin=origin,resultants_low=pl)
 rounded=op.evaluate(x,np.zeros_like(x),q,u,p,origin=origin)
 assert response.history==paired['history']
 assert response.residual[24]!=rounded.residual[24],'low force did not reach compatibility work'
 with localcontext() as ctx:
  ctx.prec=80
  strain=stress/D(1000000)+alpha
  exact=D.from_float(x[1,0])-D.from_float(x[0,0])-D('.5')-D('.5')*strain
  work_tolerance=D(64)*D.from_float(np.finfo(float).eps)**2*abs(strain)
  assert abs(D.from_float(float(response.residual[24]))-exact)<=work_tolerance
 rec=op.recover_material(u,p,origin=origin,rotations_low=np.zeros_like(u),resultants_low=pl)
 assert rec.history==response.history and repr(origin)==before
 for low in (np.ones(18),np.full(18,np.nan),np.zeros(17)):
  with pytest.raises(ValueError):op.cell.response(p.tolist(),origin,resultants_low=low.tolist())
 legacy=element(paired=False).operator
 with pytest.raises(ValueError):legacy.evaluate(x,np.zeros_like(x),q,u,p,resultants_low=pl)

def test_active_spatial_derivative():
 import numpy as np
 from scripts.r8_f06_fibre_log_domain_dev import element
 from anyfesolver._stiffened_shell_f06_increment_arithmetic import updated_rotation,Pair
 from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
 e=element();op=e.operator;x=op.reference.coordinates.copy();xl=np.zeros_like(x)
 q=np.tile(np.eye(3),(3,1,1));ql=np.zeros_like(q);u=np.tile(np.eye(3),(2,1,1));ul=np.zeros_like(u)
 for i in range(3):q[i],ql[i]=updated_rotation(q[i],ql[i],np.array([0.,.03*i,.01*i]))
 for i in range(2):u[i],ul[i]=updated_rotation(u[i],ul[i],np.array([0.,.04,.01]))
 p=np.zeros(18);p[0]=p[3]=1.;pl=np.zeros(18);pl[0]=pl[3]=2.**-55
 origin=op.cell.virgin();base=op.evaluate(x,xl,q,u,p,origin=origin,vertices_low=ql,rotations_low=ul,resultants_low=pl)
 direction=np.sin(np.arange(42)+1.)*.1;direction[24:]*=.01
 def at(h):
  xx=Pair(x,xl)+h*direction[:18].reshape(3,6)[:,:3]
  qq=q.copy();qql=ql.copy();uu=u.copy();uul=ul.copy()
  for i in range(3):qq[i],qql[i]=updated_rotation(q[i],ql[i],h*direction[6*i+3:6*i+6])
  for i in range(2):uu[i],uul[i]=updated_rotation(u[i],ul[i],h*direction[18+3*i:21+3*i])
  pp=Pair(p,pl)+h*direction[24:]
  return op.evaluate(xx.high,xx.low,qq,uu,pp.high,origin=origin,vertices_low=qql,rotations_low=uul,resultants_low=pp.low)
 h=1e-6;plus=at(h);minus=at(-h);actual=(plus.residual-minus.residual)/(2*h)
 expected=spatial_jacobian(base.residual,base.hessian+base.hessian_low)@direction
 error=np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny)
 assert error<=1e-6
 assert abs((plus.potential-minus.potential)/(2*h)-base.residual@direction)<=1e-6*max(1.,abs(base.residual@direction))
 Path(os.environ['F06_FIBRE_OUTPUT'],'derivative.json').write_text(json.dumps(dict(error=error,criterion=1e-6)))

def test_public_active_fibre_lifecycle():
 import numpy as np
 from scripts.r8_f06_fibre_log_domain_dev import definition,actions,DOC
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._ge_beam3_p5_seeded.core import canonical
 from anyfesolver._stiffened_shell_beam_history import shapes
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode,encode,SCHEMAS
 from anyfesolver._stiffened_shell_f06_increment_arithmetic import Pair
 import anyfesolver._stiffened_shell_f06_increment_arithmetic as arithmetic
 out=Path(os.environ['F06_FIBRE_OUTPUT']);owner=ss.create_analysis(definition=definition())
 calls=[];original=arithmetic.full_log_coefficients
 def capture(c):calls.append(c.scalar());return original(c)
 arithmetic.full_log_coefficients=capture
 try:
  prefix=os.environ.get('F06_FIBRE_PREFIX')
  result=owner.restore(Path(prefix).read_bytes()) if prefix else owner.solve(actions(),(.4,),max_iterations=16,max_backtracks=8)
  driver=owner._finite_driver;state=driver._state
  assert calls and min(calls)<1.-1e-7,'ordinary principal Log not consumed'
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  accepted=canonical(driver._beam_origins);predecessor=canonical(driver._beam_predecessors)
  assert accepted!=predecessor
  alpha=max(row[2]+row[3] for h in driver._beam_origins for station in h.stations for row in station.rows)
  assert alpha>0 and owner.recover().beam_fibre_material
  assert any(np.any(getattr(state,n)!=0) for n in ('q_low','frames_low','cells_low','resultants_low'))
  raw=owner.checkpoint();(out/'accepted-active.bin').write_bytes(raw)
  if prefix is None:
   assert not (DOC/'source-prefix.bin').exists(),'preserve first source prefix'
   (DOC/'source-prefix.bin').write_bytes(raw)
  correction_shapes=tuple((n,getattr(state,n).shape) for n in ('q_low','frames_low','cells_low','resultants_low'))
  inner,corrections=decode_compensated(raw,owner._identity,correction_shapes)
  base_shapes=tuple((n,v.shape) for n,v in (('q',state.q),('frames',state.frames),('cells',state.cells),('resultants',state.resultants),
   ('multipliers',state.multipliers),('pattern',driver._pattern),('accepted_factors',np.asarray(driver._accepted_factors))))
  kind,generation,values=decode(inner,owner._identity,lambda k,g:base_shapes+shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted'))
  assert kind=='finite_compensated_fibre_material'
  refused_payloads=(
   encode_compensated(encode('finite_compensated_generalized_material',owner._identity,generation,tuple(values.items())),owner._identity,corrections),
   encode_compensated(json.dumps(dict(schema=SCHEMAS[kind])).encode(),owner._identity,corrections))
  for payload in refused_payloads:
   fresh=ss.create_analysis(definition=definition())
   try:
    with pytest.raises(ValueError):fresh.restore(payload)
    assert fresh._finite_driver is None and fresh._accepted is None
   finally:fresh.close()
  for name in ('origin_fibres','accepted_fibres'):
   changed={k:v.copy() for k,v in values.items()};pair=Pair(*changed[name][0,2:])+1e-4
   changed[name][0,2:]=(pair.high,pair.low)
   tampered=encode_compensated(encode(kind,owner._identity,generation,tuple(changed.items())),owner._identity,corrections)
   fresh=ss.create_analysis(definition=definition())
   try:
    with pytest.raises(ValueError):fresh.restore(tampered)
    assert fresh._finite_driver is None and fresh._accepted is None
   finally:fresh.close()
  fresh=ss.create_analysis(definition=definition())
  try:
   fresh.restore(raw);assert fresh.checkpoint()==raw
   fd=fresh._finite_driver
   assert canonical(fd._beam_origins)==accepted and canonical(fd._beam_predecessors)==predecessor
   for n in ('q','frames','cells','resultants','q_low','frames_low','cells_low','resultants_low'):
    np.testing.assert_array_equal(getattr(state,n),getattr(fd._state,n))
  finally:fresh.close()
  lows={n:getattr(state,n).copy() for n in ('q_low','frames_low','cells_low','resultants_low')}
  owner.rebase_rotations(owner._generation)
  assert canonical(driver._beam_origins)==accepted and canonical(driver._beam_predecessors)==predecessor
  for n,v in lows.items():np.testing.assert_array_equal(getattr(driver._state,n),v)
  rebased=owner.checkpoint();fresh=ss.create_analysis(definition=definition())
  try:fresh.restore(rebased);assert fresh.checkpoint()==rebased
  finally:fresh.close()
  committed=driver._beam_origins
  continued=owner.solve(actions(),(.41,),max_iterations=16,max_backtracks=8)
  assert driver._beam_predecessors is committed and continued.beam_fibre_material
  assert continued.accepted_factors[-1]==.41 and continued.free_equilibrium_ratio<=1e-8 and continued.constraint_max<=1e-9
  after=owner.checkpoint();fresh=ss.create_analysis(definition=definition())
  try:fresh.restore(after);assert fresh.checkpoint()==after
  finally:fresh.close()
  class Cancel(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':
     self.visits+=1
     if self.visits==2:self.cancel('fibre candidate cancellation')
    super().raise_if_cancelled(stage)
  cancel=Cancel()
  with pytest.raises(SolveCancelled):owner.solve(actions(),(.42,),cancellation_token=cancel)
  assert cancel.visits==2 and owner.checkpoint()==after
  from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
  with pytest.raises(FiniteIncrementFailure):owner.solve(actions(),(1.2,),max_iterations=1,max_backtracks=0)
  assert owner.checkpoint()==after
  write=dict(status='PASS',family='axial',prefix_reused=prefix,ordinary_calls=len(calls),minimum_cosine=min(calls),
   accumulated=alpha,equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
   continued_equilibrium=continued.free_equilibrium_ratio,continued_gap=continued.constraint_max,
   accepted_factors=list(continued.accepted_factors),origin_and_accepted_tamper_refused=True,restore_rebase_exact=True,
   generalized_schema_refused=True,binary_only_refused=True,
   cancel_and_rejection_preserve_prefix=True)
  (out/'public-metrics.json').write_text(json.dumps(write,indent=2)+'\n')
 finally:arithmetic.full_log_coefficients=original;owner.close()
