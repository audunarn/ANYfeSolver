"""Three-case workflow controls: no external executable launches."""
import json
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.sestra_three_case import prepare_three_cases,read_manifest,solve_case,compare_suite,Unsupported

BASELINE=Path(__file__).resolve().parent/'fixtures/sestra_three_case/20261008_140714_T1.FEM'

def test_generator_byte_identity_loads_order_and_tip_handcheck(tmp_path):
 suite=prepare_three_cases(BASELINE,tmp_path/'cases')
 assert suite['order']==['plate','flatbar','combined']
 assert [(v['nodes'],v['shells'],v['beams']) for v in suite['cases'].values()]==[(441,400,0),(21,0,20),(441,400,20)]
 for case in suite['cases'].values():
  data=Path(case['fem']['path']).read_bytes();assert b'\r\r\n' not in data and data.endswith(b'\r\n')
  control=(Path(case['manifest']).parent/'sestra_control.inp').read_text();assert 'ELOP                   1.   0.   1.' in control
 m=read_manifest(suite['cases']['flatbar']['manifest']);imported,u,translation,oracle,balance,info=solve_case(m)
 assert oracle['passed'] and balance['passed']
 assert oracle['analytic_timoshenko_uz_m']==pytest.approx(-.015275,rel=1e-3)
 np.testing.assert_allclose(balance['load'],[0,0,-1000,-2000,1000,0],atol=1e-8)
 assert len(imported.model.mesh.nodes)==21 and len(translation)==21

def test_no_reference_is_not_executed_and_source_mutation_refused(tmp_path):
 suite=prepare_three_cases(BASELINE,tmp_path/'cases')
 report=compare_suite(tmp_path/'cases/suite.json',tmp_path/'evidence')
 assert report['status']=='not_executed' and all(x['status']=='not_executed' for x in report['cases'].values())
 path=Path(suite['cases']['flatbar']['manifest']);m=json.loads(path.read_text());m['translation_tolerances']['absolute_m']=1e-5;path.write_text(json.dumps(m))
 with pytest.raises(Unsupported,match='criteria'):read_manifest(path)

def test_public_importer_refuses_new_nodal_multi_case_aggregation(tmp_path):
 from anyfesolver.sesam_fem import import_sesam_fem
 from anyfileio.sesam.records import read_raw_records,FemRawRecord,records_to_text
 suite=prepare_three_cases(BASELINE,tmp_path/'cases');path=Path(suite['cases']['flatbar']['fem']['path']);records=list(read_raw_records(path))
 records.insert(-1,FemRawRecord('BNLOAD',(2.,1.,0.,0.,21.,6.,0.,0.,-1000.,0.,0.,0.),(),0,0,()))
 path.write_bytes(records_to_text(records).encode('ascii'))
 with pytest.raises(ValueError,match='one shared source load case'):import_sesam_fem(path)

@pytest.mark.parametrize('mutation',['orientation','dimension','area'])
def test_named_flatbar_regime_refuses_changed_axes_and_section(tmp_path,mutation):
 from anyfesolver.sestra_three_case import admit_case
 from anyfileio.sesam.records import read_raw_records,records_to_text
 from dataclasses import replace
 suite=prepare_three_cases(BASELINE,tmp_path/'cases');m=read_manifest(suite['cases']['flatbar']['manifest']);path=Path(m['fem']);records=[]
 for r in read_raw_records(path):
  v=list(r.numeric_fields)
  if mutation=='orientation' and r.name=='GUNIVEC':v[1:4]=[1.,0.,0.]
  if mutation=='dimension' and r.name=='GBARM':v[1]*=2
  if mutation=='area' and r.name=='GBEAMG':v[2]*=2
  records.append(replace(r,numeric_fields=tuple(v)))
 path.write_bytes(records_to_text(records).encode('ascii'))
 with pytest.raises(Unsupported,match='local-z|FB100x10|area/inertias'):admit_case(m)
