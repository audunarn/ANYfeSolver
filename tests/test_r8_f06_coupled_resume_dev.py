"""Genuine coupled .4 -> .2 -> .41 material history through installed public owner."""
import os,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_COUPLED_RESUME_OUTPUT' not in os.environ,reason='explicit bounded supervisor required')

def capture(owner):
 import numpy as np
 from anyfesolver._stiffened_shell_beam_history import arrays
 d=owner._finite_driver;s=d._state
 values=dict(arrays(owner._beam_parts,d._beam_predecessors,'origin')+arrays(owner._beam_parts,d._beam_origins,'accepted'))
 for label,histories in (('origin',d._shell_predecessors),('accepted',d._shell_origins)):
  for key in ('plastic_strain','alpha'):values[label+'_shell_'+key]=np.asarray([h[key] for h in histories]).copy()
 for key in ('q','frames','cells','resultants','multipliers','q_low','frames_low','cells_low','resultants_low','shell_anchors'):
  if getattr(s,key) is not None:values[key]=getattr(s,key).copy()
 values['reactions']=owner.recover().support_reactions.copy()
 return values

def exact(a,b,keys=None):
 import numpy as np
 assert all(np.array_equal(a[k],b[k]) for k in (a if keys is None else keys) if k!='reactions')
 if keys is None or 'reactions' in keys:
  assert np.linalg.norm(a['reactions']-b['reactions'])<=1e-8*max(np.linalg.norm(a['reactions']),np.finfo(float).tiny)

def save(out,label,owner,result):
 import numpy as np
 from scripts.r8_f06_coupled_resume_dev import sha,write
 raw=owner.checkpoint();(out/(label+'.bin')).write_bytes(raw)
 from dataclasses import asdict
 from scripts.r8_f06_cyclic_material_dev import actions
 def plain(value):
  if isinstance(value,np.ndarray):return value.tolist()
  if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
  if isinstance(value,(list,tuple)):return [plain(v) for v in value]
  return value
 write(out/(label+'-recovery.json'),plain(dict(fibre=[asdict(v) for v in result.beam_fibre_material],
  generalized=[asdict(v) for v in result.beam_generalized_material],
  actual_nodal_actions=[asdict(v) for v in actions('coupled',result.load_factor)])))
 values=capture(owner);np.savez(out/(label+'.npz'),**values)
 write(out/(label+'.json'),dict(factor=result.load_factor,accepted_factors=result.accepted_factors,
  equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,generation=owner._generation,
  sha256=sha(out/(label+'.bin')),fibre_incremental_dissipation_j=[sum(e.incremental_dissipation_j) for e in result.beam_fibre_energy]))
 return values

def activity(old,new):
 from fractions import Fraction as F
 def paired(row,offset=0):return F(float(row[offset]))+F(float(row[offset+1]))
 result={}
 for label,key,offset in (('fibre','accepted_fibres',2),('generalized','accepted_accumulated',0)):
  changes=[paired(b,offset)-paired(a,offset) for a,b in zip(old[key],new[key])]
  result[label]=dict(positive=sum(x>0 for x in changes),negative=sum(x<0 for x in changes),maximum=float(max(changes)))
 shell=new['accepted_shell_alpha']-old['accepted_shell_alpha']
 result['shell']=dict(positive=int((shell>0).sum()),negative=int((shell<0).sum()),maximum=float(shell.max()))
 result['fibre_signed_changes']=[float(paired(b)-paired(a)) for a,b in zip(old['accepted_fibres'],new['accepted_fibres'])]
 result['generalized_plastic_changes']=[[float(paired(b)-paired(a)) for a,b in zip(x,y)] for x,y in zip(old['accepted_plastic'],new['accepted_plastic'])]
 return result

def identity(owner):
 d=owner._finite_driver
 return owner.checkpoint(),owner._generation,d._state,d._beam_origins,d._beam_predecessors,d._shell_origins,d._shell_predecessors

def unchanged(owner,before):
 now=identity(owner)
 return dict(checkpoint=now[0]==before[0],generation=now[1]==before[1],objects=all(a is b for a,b in zip(now[2:],before[2:])))

def test_history():
 import numpy as np,sys
 from dataclasses import asdict
 from scripts.r8_f06_coupled_resume_dev import DOC,PREFIX,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 out=Path(os.environ['F06_COUPLED_RESUME_OUTPUT']);unload=os.environ['F06_COUPLED_RESUME_STAGE'].endswith('unload')
 model=definition('coupled');owner=ss.create_analysis(definition=model)
 source=PREFIX if unload else DOC/'attempt-installed-unload1/accepted.bin';target=.2 if unload else .41
 try:
  result=owner.restore(source.read_bytes());d=owner._finite_driver;initial=save(out,'initial',owner,result)
  assert owner.checkpoint()==source.read_bytes()
  assert not np.any(d._state.q_low[:owner._shell_count])
  if not unload:
   before=identity(owner);reaction=initial['reactions'].copy()
   owner.rebase_rotations(owner._generation)
   anchored=save(out,'anchored',owner,owner.recover())
   history_keys=[k for k in initial if k.startswith(('origin_','accepted_')) or k.endswith('_low')]
   exact(initial,anchored,history_keys)
   assert np.linalg.norm(anchored['reactions']-reaction)<=1e-8*max(np.linalg.norm(reaction),np.finfo(float).tiny)
   chart=owner.checkpoint();fresh=ss.create_analysis(definition=model)
   try:
    fresh.restore(chart);assert fresh.checkpoint()==chart;exact(anchored,capture(fresh))
   finally:fresh.close()
  parent_beam=d._beam_origins;parent_shell=d._shell_origins;before=identity(owner)
  original=d._assemble;advance=d._advance;trial={};trace=out/'trace.jsonl'
  def record(row):
   with trace.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
  def advanced(state,delta,amount):
   frame=sys._getframe(1).f_locals;candidate=advance(state,delta,amount)
   trial[id(candidate)]=dict(amount=float(amount),backtrack=frame['backtrack']);return candidate
  def assembled(state,*args,**kwargs):
   frame=sys._getframe(2).f_locals;value=original(state,*args,**kwargs)
   record(dict(factor=float(args[0]),iteration=frame.get('iteration'),trial=trial.pop(id(state),{}),
    equilibrium=float(value[2]),gap=float(value[3]),counts=dict(frame.get('counts',{}))))
   return value
  d._assemble=assembled;d._advance=advanced
  def accepted(result):save(out,'accepted',owner,result)
  try:result=owner.solve(actions('coupled'),(target,),max_iterations=20,max_backtracks=8,on_accepted=accepted)
  except BaseException as error:
   save(out,'terminal-failure',owner,owner.recover());attempt=getattr(error,'attempt',None)
   write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempt=None if attempt is None else asdict(attempt),atomic=unchanged(owner,before)))
   raise
  finally:d._assemble=original;d._advance=advance
  current=save(out,'accepted',owner,result);increments=activity(initial,current)
  write(out/'activity.json',increments)
  assert d._beam_predecessors is parent_beam and d._shell_predecessors is parent_shell
  assert result.load_factor==target and result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
  assert result.beam_fibre_material and result.beam_generalized_material
  assert all(increments[k]['negative']==0 for k in ('fibre','generalized','shell'))
  if not unload:assert all(increments[k]['positive']>0 for k in ('fibre','generalized','shell')),'reload must renew each material family'
  assert all(sum(e.incremental_dissipation_j)>=0 for e in result.beam_fibre_energy)
  cold=ss.create_analysis(definition=model)
  try:cold.restore(owner.checkpoint());assert cold.checkpoint()==owner.checkpoint();exact(current,capture(cold))
  finally:cold.close()
  write(out/'result.json',dict(status='PASS',factor=target,accepted_factors=result.accepted_factors,activity=increments,
   equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,exact_cold_state_histories=True,immediate_predecessors=True,
   anchored_rebase=not unload,all_shell_low_zero=not bool(np.any(d._state.q_low[:owner._shell_count]))))
 finally:owner.close()

def test_controls_rollback():
 import numpy as np
 from scripts.r8_f06_coupled_resume_dev import DOC,PREFIX,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure,AdaptiveIncrementControl
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode,encode
 from anyfesolver._stiffened_shell_beam_history import shapes
 out=Path(os.environ['F06_COUPLED_RESUME_OUTPUT']);model=definition('coupled');owner=ss.create_analysis(definition=model)
 try:
  owner.restore((DOC/'attempt-installed-reload1/accepted.bin').read_bytes());d=owner._finite_driver;before=identity(owner)
  class Cancel(CancellationToken):
   visits=0
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.finite.trial':
     self.visits+=1
     if self.visits==2:self.cancel('coupled reload trial rollback')
    super().raise_if_cancelled(stage)
  token=Cancel()
  with pytest.raises(SolveCancelled):owner.solve(actions('coupled'),(.42,),max_iterations=20,max_backtracks=8,cancellation_token=token)
  write(out/'cancellation.json',dict(visits=token.visits,atomic=unchanged(owner,before)))
  assert token.visits==2 and all(unchanged(owner,before).values())
  with pytest.raises(FiniteIncrementFailure) as rejected:owner.solve(actions('coupled'),(.8,),max_iterations=1,max_backtracks=0)
  from dataclasses import asdict
  write(out/'rejection.json',dict(attempt=asdict(rejected.value.attempt),atomic=unchanged(owner,before)))
  assert all(unchanged(owner,before).values())
  with pytest.raises(ValueError):owner.solve(actions('coupled'),(.42,),increment_control=AdaptiveIncrementControl(4,12))
  assert all(unchanged(owner,before).values())
  (out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
  write(out/'result.json',dict(status='PASS',atomic=unchanged(owner,before),adaptive_refused=True))
 finally:owner.close()


def test_controls_tamper():
 import numpy as np
 from scripts.r8_f06_coupled_resume_dev import DOC,PREFIX,write
 from scripts.r8_f06_cyclic_material_dev import definition
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated,decode,encode
 from anyfesolver._stiffened_shell_beam_history import shapes
 out=Path(os.environ['F06_COUPLED_RESUME_OUTPUT']);model=definition('coupled');owner=ss.create_analysis(definition=model)
 try:
  owner.restore((DOC/'attempt-installed-reload1/accepted.bin').read_bytes());d=owner._finite_driver;before=identity(owner)
  # Model-bound normal-format input has one accepted factor; exercise both generations.
  state=d._state;lowkeys=('q_low','frames_low','cells_low','resultants_low')
  inner,low=decode_compensated(PREFIX.read_bytes(),owner._identity,tuple((k,getattr(state,k).shape) for k in lowkeys))
  layout=tuple((k,getattr(state,k).shape) for k in ('q','frames','cells','resultants','multipliers'))
  layout+=(('pattern',d._pattern.shape),('accepted_factors',(1,)))+shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted')
  layout+=(('shell_plastic',(1,12,3)),('shell_alpha',(1,12)),('shell_origin_plastic',(1,12,3)),('shell_origin_alpha',(1,12)))
  kind,generation,values=decode(inner,owner._identity,lambda k,g:layout)
  fresh=ss.create_analysis(definition=model);refusals=[]
  try:
   for key in ('shell_alpha','shell_origin_alpha','accepted_fibres','origin_fibres','accepted_accumulated','origin_accumulated'):
    altered={k:v.copy() for k,v in values.items()};altered[key].flat[0]+=1e-4
    bad=encode_compensated(encode(kind,owner._identity,generation,tuple(altered.items())),owner._identity,low)
    with pytest.raises(ValueError):fresh.restore(bad)
    assert fresh._finite_driver is None and fresh._accepted is None
    refusals.append(key);write(out/'refusals.json',refusals)
  finally:fresh.close()
  (out/'terminal-prefix.bin').write_bytes(owner.checkpoint())
  write(out/'result.json',dict(status='PASS',atomic=unchanged(owner,before),history_tamper_refusals=refusals))
  assert all(unchanged(owner,before).values())
 finally:owner.close()
