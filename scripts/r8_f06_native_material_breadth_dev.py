"""One fresh bounded material-family outcome; reuse existing supervisor/services."""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_f06_fibre_log_domain_dev as infra
from scripts.r8_f06_log_domain_check_dev import runtime,sha,write
DOC=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/material-breadth-v1'
AXIAL_PREFIX=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/fibre-v1/source-prefix.bin'
CASES=('axial','biaxial','mixed','mixed-s3')

def bindings():
 paths=[ROOT/'src/anyfesolver'/p for p in (*infra.PRODUCTS,'_stiffened_shell_affine_endpoint.py',
  '_ge_beam3_retained_generalized.py','_ge_beam3_native_generalized_element.py')]
 paths += [Path(__file__),ROOT/'scripts/r8_f06_fibre_log_domain_dev.py',ROOT/'scripts/r8_f06_log_domain_check_dev.py',
  ROOT/'tests/test_r8_f06_native_material_breadth_dev.py',ROOT/'tests/test_r8_f06_fibre_log_domain_dev.py',ROOT/'tests/conftest.py']
 paths.append(AXIAL_PREFIX)
 paths += sorted(DOC.glob('*-prefix.bin'))
 return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def definition(case='biaxial', *, partial=False):
 import numpy as np
 from anyfesolver import stiffened_shell as ss
 from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
 from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection,Fibre,FlowCurve
 from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
 from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
 from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
 from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
 from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY
 base=infra.definition()
 if case=='axial':return base
 old=infra.element();ref=old.operator.reference
 fibres=tuple(Fibre(str(i),y*.1,z*.1,.0025,1e6,FlowCurve((0.,),(10.,),1e5))
  for i,(y,z) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))))
 law=PhysicalFibreSection(fibres,np.diag(np.sqrt((0.,4e3,4e3,100.,0.,0.))))
 beam=NativeFibreStaticElement(1,(101,102,103),ref,law,order=4,arithmetic_policy=POLICY)
 members=(NativeBeamDefinition.capture(beam,np.eye(6)),);points=base.points
 if case in ('mixed','mixed-s3'):
  ref2=CenteredCurvedBeam3ReferenceGeometry(np.array([[0.,.5,.1],[.5,.5,.1],[1.,.5,.1]]),np.tile(np.eye(3),(3,1,1)))
  section=EllipsoidalGeneralizedSection(np.diag((1e4,4e3,4e3,100.,100.,100.)),np.eye(6),.2,1000.)
  general=NativeGeneralizedStaticElement(2,(201,202,203),ref2,section,order=4,arithmetic_policy=None if partial else POLICY)
  members += (NativeBeamDefinition.capture(general,np.eye(6)),)
  points += (ss.PointConnection(4,201),ss.PointConnection(3,203))
 elif case!='biaxial':raise ValueError('registered native material fixture required')
 model=replace(base,beams=members,points=points)
 if case=='mixed-s3':
  model=replace(model,shell=replace(model.shell,q4_elements=(),s3_elements=((1,2,3),(1,3,4))),q4_pullback_policy='historical_corotational')
 return model

def actions(case):
 from anyfesolver import stiffened_shell as ss
 result=infra.actions()
 return result+(ss.NodalAction('beam',203,(50.,0.,10.,0.,0.,0.)),) if case in ('mixed','mixed-s3') else result

def child(out,stage):
 import psutil
 psutil.Process().cpu_affinity([0]);assert json.loads((out/'bindings.json').read_text())==bindings()
 if stage=='wheel':
  # Existing wheel implementation gets this new namespace only, in this child.
  infra.DOC=DOC;infra.bindings=bindings
  return infra.child(out,stage)
 installed=stage.startswith('installed') or stage=='axial-regression'
 selected=DOC/'package/site' if installed else ROOT/'src';sys.path[:0]=[str(selected),str(ROOT/'tests')]
 os.environ['F06_BREADTH_OUTPUT']=str(out)
 if stage=='axial-regression':
  os.environ.update(F06_FIBRE_OUTPUT=str(out),F06_FIBRE_PREFIX=str(AXIAL_PREFIX))
 targets=[str(ROOT/'tests/test_r8_f06_native_material_breadth_dev.py')]
 if stage=='axial-regression':targets=[str(ROOT/'tests/test_r8_f06_fibre_log_domain_dev.py')+'::test_public_active_fibre_lifecycle']
 selector='not lifecycle'
 if '-' in stage and stage!='axial-regression':
  case=stage.split('-',1)[1];assert case in CASES
  os.environ['F06_BREADTH_CASE']=case;selector='lifecycle'
  prefix=DOC/(case+'-prefix.bin')
  if prefix.exists():os.environ['F06_BREADTH_PREFIX']=str(prefix)
 elif stage=='axial-regression':selector=''
 runtime(out,'runtime-before',selected)
 import pytest
 try:
  return int(pytest.main(targets+['-q','-x','-k',selector,'-o','pythonpath=','-p','no:cacheprovider',
   '--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
 finally:
  state=runtime(out,'runtime-after',selected)
  if installed:
   frozen=json.loads((DOC/'package/snapshot-bindings.json').read_text())
   for name,p in state['imports'].items():assert sha(p)==frozen['src/'+Path(p).relative_to(selected).as_posix()]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
 if a.stage=='close':
  lp=DOC/'ledger.json';v=json.loads(lp.read_text());assert v['status']=='OPEN'
  v.update(status='CLOSED',unused_discarded_s=v['budget_s']-v['used_s']);write(lp,v)
 elif a.child:raise SystemExit(child(Path(a.out),a.stage))
 else:raise SystemExit(infra.run(a.stage,doc=DOC,entry=__file__,bind=bindings))
