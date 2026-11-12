"""Bounded paired mixed displacement lifecycle; one new physical history only."""
from pathlib import Path
import argparse,json,os,sys,hashlib,zipfile,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010'
DOC=BASE/'paired-displacement-v1';PACKAGE=DOC/'package'
INPUT=BASE/'mixed-adaptive-v1/attempt-installed-resume1/terminal.bin'
INPUT_SHA='970e0360f22ed348ca2eaa7eee0d2d168d6d686d0a91a750f3ccd31794865f91'
PRODUCTS=('_stiffened_shell_displacement_control.py','_stiffened_shell_displacement_adaptive.py',
 '_stiffened_shell_displacement_checkpoint.py','_stiffened_shell_displacement_corrector.py',
 '_stiffened_shell_affine_endpoint.py','_stiffened_shell_ge_checkpoint_binary.py','_stiffened_shell_ge_experimental.py')
def bindings():
 paths=[ROOT/'src/anyfesolver'/n for n in set((*PRODUCTS,*infra.PRODUCTS))]
 paths += [ROOT/'scripts'/n for n in ('r8_f06_cyclic_material_dev.py','r8_f06_native_material_breadth_dev.py','r8_f06_coupled_resume_dev.py','r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py')]
 paths += [Path(__file__),ROOT/'tests/test_r8_f06_paired_displacement_dev.py',ROOT/'tests/test_r8_f06_coupled_resume_dev.py',INPUT,DOC/'PHYSICAL_ROUTE.json']
 paths += [ROOT/'tests'/n for n in ('test_stiffened_shell_displacement_checkpoint.py','test_stiffened_shell_bl5_coupled_material.py','test_stiffened_shell_bl5_material.py','test_stiffened_shell_bl5_finite.py')]
 paths += list(DOC.glob('*AUTHORITY*.json'))
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def artifact():
 frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text());wheel=next((PACKAGE/'wheels').glob('*.whl'))
 with zipfile.ZipFile(wheel) as z:
  for rel,h in frozen.items():
   name=rel.removeprefix('src/')
   assert sha(ROOT/rel)==sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/name)==hashlib.sha256(z.read(name)).hexdigest()==h,rel
 return dict(files=len(frozen),wheel_sha256=sha(wheel),manifest_sha256=sha(PACKAGE/'snapshot-bindings.json'))
def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings();assert sha(INPUT)==INPUT_SHA
 if stage=='wheel':
  infra.DOC=DOC;infra.bindings=bindings;original=subprocess.run;timings=[]
  def measured(*args,**kwargs):
   start=time.perf_counter_ns()
   try:
    with (out/('pip-%d.log'%len(timings))).open('wb') as stream:return original(*args,**kwargs,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
   finally:timings.append(dict(command=args[0],wall_s=(time.perf_counter_ns()-start)/1e9));write(out/'pip-timings.json',timings)
  runtime(out,'runtime-before',ROOT/'src');subprocess.run=measured
  try:return infra.child(out,stage)
  finally:subprocess.run=original;runtime(out,'runtime-after',ROOT/'src')
 installed=stage.startswith('installed');selected=PACKAGE/'site' if installed else ROOT/'src'
 if installed:write(out/'artifact-before.json',artifact())
 sys.path[:0]=[str(selected),str(ROOT/'tests')];os.environ.update(F06_PAIRED_DISPLACEMENT_OUTPUT=str(out),F06_PAIRED_DISPLACEMENT_STAGE=stage)
 runtime(out,'runtime-before',selected)
 import pytest
 targets=[str(ROOT/'tests/test_r8_f06_paired_displacement_dev.py')]
 if stage in ('source-controls','installed-controls'):targets += [str(ROOT/'tests/test_stiffened_shell_displacement_checkpoint.py')+'::test_signed_amplitude_and_chart_generation_are_not_force_history']
 try:return int(pytest.main(targets+['-q','-x','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp'),'-k','history' if stage=='installed-history' else 'installed or signed_amplitude' if stage=='installed-controls' else 'source or signed_amplitude']))
 finally:
  rt=runtime(out,'runtime-after',selected)
  if installed:
   frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
   for n,p in rt['imports'].items():assert rt['import_sha256'][n]==frozen['src/'+Path(p).relative_to(selected).as_posix()]
   write(out/'artifact-after.json',artifact())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('source-controls','wheel','installed-history','installed-controls'));p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.child:raise SystemExit(child(Path(a.out),a.stage))
 assert (DOC/'SLOT_TRANSFER.json').exists(),'numerical slot not assigned'
 if a.stage=='installed-history':assert not list(DOC.glob('attempt-installed-history*')),'one physical history, no retry'
 for required in (() if a.stage=='source-controls' else ('source-controls',) if a.stage=='wheel' else ('source-controls','wheel') if a.stage=='installed-history' else ('source-controls','wheel','installed-history')):
  attempts=sorted(DOC.glob('attempt-'+required+'*/receipt.json'));assert attempts
  if required=='installed-history' and a.stage=='installed-controls' and json.loads(attempts[-1].read_text())['status']=='FAIL':
   prior=attempts[-1].parent;w=json.loads((prior/'accepted.json').read_text());result=json.loads((prior/'result.json').read_text());initial=json.loads((prior/'initial.json').read_text())
   assert (DOC/'ACCEPTED_CONTROLS_AUTHORITY.json').exists()
   assert result['status']=='ACCEPTED_DISPLACEMENT' and result['force_seed_immutable'] and result['physical_assemblies']>0
   assert w['equilibrium']<=1e-8 and w['gap']<=1e-9 and w['generation']==initial['generation']+1
   assert w['target']==json.loads((DOC/'PHYSICAL_ROUTE.json').read_text())['target']
   assert sha(prior/'accepted.bin')==sha(prior/'prospective.bin')==sha(prior/'terminal.bin')==w['sha256']
   assert (prior/'accepted.bin').read_bytes().startswith(b'ANYSSPD1') and sha(prior/'initial.bin')==INPUT_SHA
   artifact() # exact current/snapshot/wheel/site equality; child performs strict embedded-seed/mechanical restore
  else:assert json.loads(attempts[-1].read_text())['status']=='PASS'
 authority='PHYSICAL_AUTHORITY.json' if a.stage=='installed-history' else 'WHEEL_AUTHORITY.json' if a.stage=='wheel' else 'AUTHORITY.json'
 raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings,authority_file=authority))
