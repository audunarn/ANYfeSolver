"""Explicit paired force-seed displacement transfer, not force .41 acceptance."""
import os,json,runpy
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif('F06_PAIRED_DISPLACEMENT_OUTPUT' not in os.environ,reason='bounded supervisor required')
_h=runpy.run_path(str(Path(__file__).with_name('test_r8_f06_coupled_resume_dev.py')))
capture,activity,identity,unchanged,exact=(_h[x] for x in ('capture','activity','identity','unchanged','exact'))
def inputs():
 from scripts.r8_f06_paired_displacement_dev import DOC,INPUT,INPUT_SHA,write
 from scripts.r8_f06_cyclic_material_dev import definition,actions
 from anyfesolver import stiffened_shell as ss
 plan=json.loads((DOC/'PHYSICAL_ROUTE.json').read_text());owner=ss.create_analysis(definition=definition('coupled'))
 return owner,actions('coupled'),ss.DisplacementControl('shell',2,1,force_seed_sha256=INPUT_SHA),plan['target'],INPUT,DOC,write

def save(out,label,owner):
 import numpy as np
 from dataclasses import asdict
 from scripts.r8_f06_paired_displacement_dev import write,sha
 raw=owner.checkpoint();(out/(label+'.bin')).write_bytes(raw)
 values=capture(owner);np.savez(out/(label+'.npz'),**values);result=owner.recover()
 def plain(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
  if isinstance(v,(tuple,list)):return [plain(x) for x in v]
  return v
 write(out/(label+'-recovery.json'),plain(asdict(result)))
 write(out/(label+'.json'),dict(generation=owner._generation,sha256=sha(out/(label+'.bin')),equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
  target=getattr(result,'target',None),amplitude=getattr(result,'amplitude',None)))
 return values

def shared_arrays(owner):
 import numpy as np,hashlib
 found={};seen=set()
 def visit(value,path):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,np.ndarray):found[path]=(value.shape,value.dtype.str,hashlib.sha256(value.tobytes()).hexdigest())
  elif isinstance(value,dict):
   for k,v in value.items():visit(v,path+'/'+str(k))
  elif isinstance(value,(tuple,list)):
   for i,v in enumerate(value):visit(v,path+'/'+str(i))
 for name in ('_K','_C','_A'):
  matrix=getattr(owner,name)
  for key in ('data','indices','indptr'):visit(getattr(matrix,key),name+'/'+key)
 for i,(_,element,_,origin) in enumerate(owner._shell_parts):
  visit(vars(element),'shell/'+str(i));visit(origin,'origin/'+str(i))
 return found

def test_source_options_and_refusals():
 import numpy as np
 from dataclasses import replace
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._stiffened_shell_displacement_checkpoint import identities,_layout,PAIRED_KIND
 owner,loads,control,target,path,doc,write=inputs();out=Path(os.environ['F06_PAIRED_DISPLACEMENT_OUTPUT'])
 try:
  for value in ('',True,'A'*64,'g'*64,[]):
   with pytest.raises(ValueError):ss.DisplacementControl('shell',2,1,force_seed_sha256=value)
  assert ss.DisplacementControl('shell',2,1).force_seed_sha256 is None
  owner.restore(path.read_bytes());before=identity(owner);controller=owner._finite_driver._adaptive_state
  assert identities(owner)[0]==PAIRED_KIND and all(k in dict(_layout(owner,1)) for k in ('q_low','frames_low','cells_low','resultants_low'))
  refused=[]
  cases=[('missing_opt_in',replace(control,force_seed_sha256=None),loads,{}),('wrong_digest',replace(control,force_seed_sha256='0'*64),loads,{}),
   ('beam_control',replace(control,domain='beam',node=103),loads,{}),('changed_pattern',control,loads[:-1],{}),
   ('adaptive_displacement',control,loads,dict(increment_control=ss.AdaptiveIncrementControl(max_cutbacks=4,max_attempts=12)))]
  for label,c,a,kwargs in cases:
   with pytest.raises(ValueError):owner.solve(a,(target,),displacement_control=c,max_iterations=20,max_backtracks=8,**kwargs)
   assert unchanged(owner,before) and owner._finite_driver._adaptive_state is controller;refused.append(label)
  write(out/'source-refusals.json',dict(refused=refused,force_seed_preserved=True))
 finally:owner.close()

@pytest.mark.parametrize('visit',(1,2))
def test_source_actual_trial_cancellation(visit):
 from anyfesolver.control import CancellationToken,SolveCancelled
 owner,loads,control,target,path,doc,write=inputs();out=Path(os.environ['F06_PAIRED_DISPLACEMENT_OUTPUT'])
 class Cancel(CancellationToken):
  def __init__(self):super().__init__();self.visits=0
  def raise_if_cancelled(self,stage=''):
   if stage=='stiffened-shell.displacement.trial':
    self.visits+=1
    if self.visits==visit:self.cancel('paired displacement scratch cancellation')
   super().raise_if_cancelled(stage)
 try:
  owner.restore(path.read_bytes());before=identity(owner);controller=owner._finite_driver._adaptive_state;token=Cancel()
  with pytest.raises(SolveCancelled):owner.solve(loads,(target,),displacement_control=control,max_iterations=20,max_backtracks=8,cancellation_token=token)
  (out/('cancel-%d.bin'%visit)).write_bytes(owner.checkpoint());assert unchanged(owner,before) and owner._finite_driver._adaptive_state is controller
  write(out/('cancel-%d.json'%visit),dict(visits=token.visits,atomic=True))
 finally:owner.close()

def test_history():
 import numpy as np,sys
 from dataclasses import asdict
 from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot
 owner,loads,control,target,path,doc,write=inputs();out=Path(os.environ['F06_PAIRED_DISPLACEMENT_OUTPUT']);accepted=[]
 try:
  owner.restore(path.read_bytes());before=identity(owner);controller=owner._finite_driver._adaptive_state;initial=save(out,'initial',owner)
  import anyfesolver._stiffened_shell_displacement_control as dc
  original_encode=dc.encode_snapshot
  def persisted(*args,**kwargs):
   # Preserve the complete prospective result before publication/replay assertions.
   np.savez(out/'prospective-arrays.npz',**{k:v for k,v in kwargs['arrays'].items() if isinstance(v,np.ndarray)})
   encoded=original_encode(*args,**kwargs);(out/'prospective.bin').write_bytes(encoded);return encoded
  dc.encode_snapshot=persisted
  original=owner._finite_driver._assemble;counter=[0]
  def observed(*args,**kwargs):
   value=original(*args,**kwargs);counter[0]+=1;frame=sys._getframe(1).f_locals
   with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(dict(evaluation=counter[0],equilibrium=float(value[2]),gap=float(value[3]),amplitude=frame.get('value')))+'\n')
   return value
  owner._finite_driver._assemble=observed
  def on_accepted(result):
   accepted.append(save(out,'accepted',owner))
   write(out/'accepted-work.json',dict(history=result.history,generation=result.generation,amplitude=result.amplitude,target=result.target,activity=activity(initial,accepted[-1])))
  try:result=owner.solve(loads,(target,),displacement_control=control,max_iterations=20,max_backtracks=8,on_accepted=on_accepted)
  except BaseException as error:
   save(out,'terminal-failure',owner)
   write(out/'failure.json',dict(type=type(error).__name__,message=str(error),detail=asdict(error.detail) if getattr(error,'detail',None) is not None else None,atomic=unchanged(owner,before),controller_same=owner._finite_driver._adaptive_state is controller));raise
  values=save(out,'terminal',owner);assert accepted
  m,a=decode_snapshot(owner,owner.checkpoint());assert a['force_seed']==path.read_bytes()
  assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9 and result.target==target
  for k in initial:
   if k.startswith('accepted_'):np.testing.assert_array_equal(values[k.replace('accepted_','origin_',1)],initial[k])
  change=activity(initial,values);write(out/'result.json',dict(status='ACCEPTED_DISPLACEMENT',activity=change,history=result.history,force_seed_immutable=True,physical_assemblies=counter[0],force_target_claim=False))
  assert all(change[k]['positive']>0 and change[k]['negative']==0 for k in ('shell','fibre','generalized'))
 finally:
  if 'original_encode' in locals():dc.encode_snapshot=original_encode
  owner.close()

def test_installed_restore_rebase_and_refusals():
 import numpy as np
 from anyfesolver import stiffened_shell as ss
 from anyfesolver.control import CancellationToken,SolveCancelled
 from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot,encode_snapshot
 from anyfesolver._stiffened_shell_displacement_control import publish
 from scripts.r8_f06_cyclic_material_dev import definition
 owner,loads,control,target,path,doc,write=inputs();out=Path(os.environ['F06_PAIRED_DISPLACEMENT_OUTPUT']);raw=(doc/'attempt-installed-history1/accepted.bin').read_bytes()
 try:
  owner.restore(raw);before=identity(owner);original=capture(owner);m,a=decode_snapshot(owner,raw);shared=shared_arrays(owner)
  with np.load(doc/'attempt-installed-history1/accepted.npz') as z:exact(original,{k:z[k] for k in z.files})
  class CancelPublish(CancellationToken):
   def raise_if_cancelled(self,stage=''):
    if stage=='stiffened-shell.displacement.publish':self.cancel('paired displacement publication cancellation')
    super().raise_if_cancelled(stage)
  with pytest.raises(SolveCancelled):publish(owner,raw,expected_driver=owner._finite_driver,expected_generation=owner._generation,cancellation_token=CancelPublish())
  assert unchanged(owner,before) and shared_arrays(owner)==shared
  with pytest.raises(ValueError):publish(owner,raw,expected_driver=owner._finite_driver,expected_generation=owner._generation-1)
  assert unchanged(owner,before) and shared_arrays(owner)==shared
  token=CancellationToken();token.cancel('paired rebase cancel')
  with pytest.raises(SolveCancelled):owner.rebase_rotations(owner._generation,cancellation_token=token)
  assert unchanged(owner,before) and shared_arrays(owner)==shared
  rebased=owner.rebase_rotations(owner._generation);after=save(out,'rebased',owner)
  for k in original:
   if k not in ('q','shell_anchors','reactions'):np.testing.assert_array_equal(original[k],after[k])
  exact(original,after,keys=('reactions',))
  cold=ss.create_analysis(definition=definition('coupled'))
  try:cold.restore(owner.checkpoint());exact(after,capture(cold));assert decode_snapshot(cold,cold.checkpoint())[1]['force_seed']==path.read_bytes()
  finally:cold.close()
  refused=[];metadata={k:m[k] for k in ('generation','seed_generation','seed_sha256','control_dof')}
  for name in ('q_low','beam_origin_fibres','beam_fibres','shell_origin_alpha','shell_alpha'):
   changed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in a.items()};changed[name].flat[0]+=1e-4
   bad=encode_snapshot(owner,arrays=changed,**metadata);fresh=ss.create_analysis(definition=definition('coupled'))
   try:
    with pytest.raises(ValueError):fresh.restore(bad)
    assert fresh._finite_driver is None and fresh._generation==0;refused.append(name)
   finally:fresh.close()
  write(out/'result.json',dict(status='PASS',cold_restore=True,rebase=True,immutable_seed=True,cancel_publish=True,cancel_rebase=True,stale_atomic=True,tamper_refusals=refused,actual_continuation_not_repeated=True,shared_compiled_arrays_unchanged=shared_arrays(owner)==shared))
 finally:owner.close()
