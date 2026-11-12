"""Diagnostic model controls, not external solver evidence."""
from pathlib import Path
import json
import pytest
from anyfesolver.sestra_deep_diagnostics import prepare_stage1,load_case_manifest,STAGE1
from anyfesolver.sestra_verification import Unsupported,identity
from anyfileio.sesam import read_sesam_fem_document

@pytest.fixture
def base(tmp_path,monkeypatch):
    from anyfileio.sesam.records import FemRawRecord,records_to_text
    import anyfesolver.sestra_deep_diagnostics as diagnostic
    specs=[('IDENT',(1,1,3,0)),('UNITS',(5,1,1,1,1)),('MISOSEL',(1,210000003000,.300000012,7850,.03,.000012,1,355)),('GELTH',(1,.0199999996)),('GUNIVEC',(1,0,0,1)),('GBEAMG',(2,.001,8.333333e-7,8.333333e-9,3.123335e-8,0,0,0,0,0,.0006666666,.0006666666)),('GBARM',(2,.1,.01)),('IEND',(0,0,0,0))]
    p=tmp_path/'seed.FEM';p.write_bytes(records_to_text([FemRawRecord(n,tuple(map(float,v)),(),0,0,()) for n,v in specs]).encode('ascii'));monkeypatch.setattr(diagnostic,'BASELINE_SHA',identity(p)['sha256']);return p

def test_prepared_six_model_control_and_physics_identity(tmp_path,base):
    s=prepare_stage1(base,tmp_path/'cases')
    assert tuple(s['order'])==STAGE1
    for name,entry in s['cases'].items():
        m=load_case_manifest(entry['manifest']);d=read_sesam_fem_document(m['fem'],strict=True)
        assert b'\r\r\n' not in Path(m['fem']).read_bytes()
        assert len(d.nodes)==(42 if name.startswith('strip') else 441)
        assert d.materials[1].poisson_ratio==(0 if name.startswith('strip') else .300000012)
        thick=name.endswith('thick') or name.startswith('combined')
        assert m['model_contract']['sestra_thck']==int(thick)
        assert ('ELOP                   1.   1.   1.' if thick else 'ELOP                   1.   0.   1.') in Path(m['control_template']['path']).read_text()
        assert len(m['beam_ids'])==(20 if name.startswith('combined') else 0)
    with pytest.raises(FileExistsError):prepare_stage1(base,tmp_path/'cases')

def test_changed_source_refuses(tmp_path,base):
    s=prepare_stage1(base,tmp_path/'cases');p=Path(s['cases'][STAGE1[0]]['manifest']);m=json.loads(p.read_text());m['fem_sha256']='0'*64;p.write_text(json.dumps(m))
    with pytest.raises(Unsupported):load_case_manifest(p)

def test_stage2_refinement_and_source_driven_underplate_geometry(tmp_path,base):
    from anyfesolver.sestra_deep_diagnostics import prepare_stage2,STAGE2
    s=prepare_stage2(base,tmp_path/'stage2');assert tuple(s['order'])==STAGE2
    for name,entry in s['cases'].items():
        m=load_case_manifest(entry['manifest']);d=read_sesam_fem_document(m['fem'],strict=True)
        assert len(d.nodes)==(1681 if '40' in name else 441)
        assert m['model_contract']['sestra_thck']==1
        if name.endswith('underplate'):
            assert m['model_contract']['global_beam_offsets_m']==-(d.sections[1].thickness+d.sections[2].web_height)/2

def test_offset_serialization_identity_does_not_allow_physical_shift(tmp_path,base):
    from dataclasses import replace
    from anyfesolver.sestra_deep_diagnostics import prepare_stage2,assert_deep_reference_identity
    s=prepare_stage2(base,tmp_path/'offset');m=load_case_manifest(s['cases']['combined_regular20_underplate']['manifest']);d=read_sesam_fem_document(m['fem'],strict=True)
    value=d.eccentricities[1];rounded=tuple(float(__import__('numpy').float32(x)) for x in value)
    assert_deep_reference_identity(d,replace(d,eccentricities={1:rounded}),m)
    for bad in ((value[0],value[1],value[2]+1e-5),(1e-12,value[1],value[2])):
        with pytest.raises(Unsupported):assert_deep_reference_identity(d,replace(d,eccentricities={1:bad}),m)
