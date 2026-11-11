"""Focused generated-model/analytical contracts; not SESTRA qualification."""
import json
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.sestra_structure_benchmarks import prepare_suite,read_manifest,admit,native_solution,frame_oracle
from anyfesolver.sestra_verification import Unsupported

@pytest.fixture(scope='module')
def suite(tmp_path_factory):return prepare_suite(tmp_path_factory.mktemp('structures')/'cases')

@pytest.mark.parametrize('family',['inclined_frame','crossing_stiffeners','folded_plate'])
@pytest.mark.parametrize('level',[4,8,16])
def test_public_decks_and_effective_admission(suite,family,level):
    m=read_manifest(suite['cases'][f'{family}_{level}']['manifest']);imp,a=admit(m)
    assert b'\r\r\n' not in Path(m['fem']).read_bytes()
    section=next(r for r in imp.document.raw_records if r.name=='GBEAMG')
    assert section.numeric_fields==(2.,0.,.00100000005,3.12333519e-8,8.33333331e-7,8.33333313e-9,0.,3.12530210e-6,1.66666669e-5,1.66666666e-6,.00066666666,.00066666666,0.,0.,1.24999997e-5,1.24999997e-6)
    assert len(section.raw_lines)==4
    td=next(r for r in imp.document.raw_records if r.name=='TDLOAD');assert td.text_fields==('LC1',) and len(td.raw_lines)==2 and td.raw_lines[1].strip()=='LC1'
    assert len(m['shell_ids'])==(0 if family=='inclined_frame' else level*level)
    assert len(m['beam_ids'])==(0 if family=='folded_plate' else 2*level)
    assert all(np.linalg.det(v['frame_columns'])>0 for v in a['beam_frames'].values())
    if family=='crossing_stiffeners':assert imp.document.nodes[m['model_contract']['crossing_node']].coordinates==(1.,1.,0.)
    if family=='folded_plate':assert m['model_contract']['tip_corner_coordinates_m']==[[1.,2.,0.],[0.,2.,1.]]

def test_castigliano_frame_native_tip_and_energy(suite):
    m=read_manifest(suite['cases']['inclined_frame_4']['manifest']);imp,_=admit(m);fields,u,balance,oracle,native=native_solution(imp,m)
    assert balance['passed'] and oracle['passed']
    assert m['model_contract']['tip_wrench']==[.2,-.3,-1.,0.,.1,0.]
    assert np.linalg.norm(oracle['tip_global_sixdof'][:3])<.01 and np.linalg.norm(oracle['tip_global_sixdof'][3:])<.01
    C=np.asarray(oracle['compliance']);np.testing.assert_allclose(C,C.T,atol=1e-14);assert np.linalg.eigvalsh(C).min()>0
    assert native['nodal_load_energy_j']==pytest.approx(oracle['strain_energy_j'],rel=1e-6)
    assert all(abs(v)>0 for v in oracle['tip_global_sixdof'])

def test_changed_load_and_control_binding_refuse(suite,tmp_path):
    source=Path(suite['cases']['inclined_frame_4']['manifest']);m=json.loads(source.read_text());m['fem']=str(source.parent/m['fem']);m['reference_sif']=str(source.parent/m['reference_sif']);m['fem_sha256']='0'*64;p=tmp_path/'bad.json';p.write_text(json.dumps(m))
    with pytest.raises(Unsupported):read_manifest(p)
    valid=read_manifest(source);valid['nodal_loads'][str(valid['model_contract']['tip_node'])][0]=201
    with pytest.raises(Unsupported,match='load differs'):admit(valid)

def test_no_existing_case_overwrite(suite):
    with pytest.raises(FileExistsError):prepare_suite(Path(suite['cases']['inclined_frame_4']['manifest']).parents[1])

def test_load_result_serialization_is_bounded_and_does_not_mutate_source(suite):
    from dataclasses import replace
    from anyfesolver.sestra_structure_benchmarks import assert_structure_reference_identity
    m=read_manifest(suite['cases']['inclined_frame_4']['manifest']);imp,_=admit(m);d=imp.document;r=next(x for x in d.load_records if x.record_name=='BNLOAD');rounded=tuple(r.raw_values[:6])+tuple(float(np.float32(v)) for v in r.raw_values[6:])
    def changed(v):return replace(d,load_records=tuple(replace(x,raw_values=tuple(v)) if x is r else x for x in d.load_records))
    evidence=assert_structure_reference_identity(d,changed(rounded),m)
    assert evidence['load_serialization']['records'][0]['result_payload'][0]!=r.raw_values[6]
    assert r.raw_values[6]==.2 and imp.model.load_cases[0].nodal_loads[m['model_contract']['tip_node']][0]==.2
    for index,value in ((6,.21),(9,1e-12),(6,-.2),(6,float('nan')),(0,2.),(4,999.)):
        bad=list(rounded);bad[index]=value
        with pytest.raises(Unsupported):assert_structure_reference_identity(d,changed(bad),m)
