"""Synthetic parser/harness fixtures, never SESTRA acceptance evidence."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from anyfesolver.sestra_verification import (COMPONENTS, METRIC, Unsupported, admit, compare,
    execute_reference, load_manifest, native_stresses, run_verification, generate_hanging_shell, validate_execution_fem)


def formatted(line):
    name, *rest = line.split(' ',1)
    return name.ljust(8)+(' '+rest[0] if rest else '')


def fixture(tmp_path):
    lines = ["IDENT 1 1 3 0", "UNITS 5 1 1 1 1", "MISOSEL 1 210000000000 .3 7850 0 .000012 1 355", "GELTH 1 .02",
        "GNODE 1 1 6 123456", "GNODE 2 2 6 123456", "GNODE 3 3 6 123456", "GNODE 4 4 6 123456",
        "GCOORD 1 0 0 0", "GCOORD 2 1 0 0", "GCOORD 3 1 1 0", "GCOORD 4 0 1 0",
        "GELMNT1 100 1 24 0 1 2 3 4", "GELREF1 100 1 0 0 0 0 0 0 1 0 0 0",
        "BNBCD 1 6 0 0 1 1 1 1", "BNBCD 2 6 0 0 1 1 1 1",
        "BNBCD 3 6 0 1 1 1 1 1", "BNBCD 4 6 1 1 1 1 1 1", "TDLOAD 4 1 103 0 LC1", "BGRAV 1 0 0 0 0 -10 0", "IEND 0 0 0 0"]
    fem = tmp_path/"case.FEM"
    fem.write_bytes(('\r\n'.join(map(formatted,lines))+'\r\n').encode('ascii'))
    manifest = tmp_path/"case.json"
    m = {"schema_version":1,"case_id":"synthetic_harness_only","fem":"case.FEM","load_case":1,
         "units":"SI","frame":"GLOBAL","observable":METRIC,"element_ids":[100],
         "tolerances":{"absolute_pa":1e-5,"relative":1e-10},"reference_sif":"reference.SIF"}
    m["reference_fem_sha256"]=hashlib.sha256(fem.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(m))
    # Independent analytical mean axial self weight stress: rho*g*height/2.
    sif = tmp_path/"reference.SIF"
    # SESAM no-transform local y follows first edge (GLOBAL X), so local x
    # carries GLOBAL Y axial stress. Real Q4 payload has ten stress triplets.
    sif.write_text('\n'.join(map(formatted,["HIERARCH 1 1 0 0"]+lines[:-1]+["RVSTRESS -4 2 1 1", "RVSTRESS 35 1 1 1 24 "+' '.join(['39250 0 0']*10), "IEND 0 0 0 0"]))+'\n')
    return manifest,m


def test_native_axial_gravity_and_synthetic_sif(tmp_path):
    path,_ = fixture(tmp_path)
    m = load_manifest(path)
    tensors, _ = native_stresses(admit(m))
    np.testing.assert_allclose(tensors[100],[0,39250,0,0,0,0],atol=1e-7)
    report = run_verification(path,tmp_path/"reports",mode="compare")
    assert report["status"] == "passed", report.get("reason")
    assert report["compared_components"] == 6
    assert report["reference_origin"] == "imported_existing_file"
    assert not report["qualification"]


def test_prepare_does_not_solve_or_execute(tmp_path,monkeypatch):
    path,_ = fixture(tmp_path)
    monkeypatch.setattr("anyfesolver.sestra_verification.native_stresses",lambda _: pytest.fail("unexpected solve"))
    report = run_verification(path,tmp_path/"reports")
    assert report["status"] == "not_executed"
    assert report["admission"] == "admitted"


@pytest.mark.parametrize("mutation",["case","load","support","unknown","nonfinite","units","selector","missing_nu"])
def test_refuse_unadmitted_fem(tmp_path,mutation):
    path,_ = fixture(tmp_path)
    fem = tmp_path/"case.FEM"; text = fem.read_text()
    replacements = {"case":("BGRAV 1","BGRAV 2"), "load":("BGRAV","BNLOAD"),
        "support":("BNBCD 1 6 0","BNBCD 1 6 3"),"unknown":("IEND","FOOBAR 1\nIEND"),
        "nonfinite":("0 -10 0","0 1e309 0"),"units":("UNITS 5 1 1 1 1","UNITS 5 2 1 1 1"),
        "selector":("BGRAV 1 0 0 0","BGRAV 1 1 0 0"),"missing_nu":(".3 7850","7850")}
    old,new=replacements[mutation]
    old=formatted(old) if old.split()[0].isupper() else old
    new='\n'.join(formatted(line) for line in new.split('\n')) if new.split()[0].isupper() else new
    fem.write_text(text.replace(old,new))
    report = run_verification(path,tmp_path/"reports")
    assert report["status"] in {"unsupported","failed"}


def test_explicit_zero_poisson_ratio_public_admission(tmp_path):
    path,_=fixture(tmp_path)
    fem=tmp_path/'case.FEM';fem.write_text(fem.read_text().replace('.3 7850','0 7850'))
    report=run_verification(path,tmp_path/'reports')
    assert report['status']=='not_executed' and report['admission']=='admitted'
    imported=admit(load_manifest(path))
    assert imported.document.materials[1].poisson_ratio==0.
    material=imported.model.get_material(next(iter(imported.model.mesh.elements.values())).material_name)
    assert material.poisson_ratio==0.


@pytest.mark.parametrize("mutation",["wrong_case","wrong_ids","missing","duplicate","nonfinite","geometry","material","thickness"])
def test_reference_refusal(tmp_path,mutation):
    path,_ = fixture(tmp_path)
    sif = tmp_path/"reference.SIF"; text=sif.read_text()
    replacements = {"wrong_case":("RVSTRESS 35 1","RVSTRESS 35 2"),
        "wrong_ids":("RVSTRESS 35 1 1","RVSTRESS 35 1 999"),
        "missing":("RVSTRESS", "COMM"), "duplicate":("IEND","RVSTRESS 35 1 1 1 24 39250 0 0 39250 0 0\nIEND"),
        "nonfinite":("39250","1e309"),"geometry":("GCOORD 2 1 0 0","GCOORD 2 2 0 0"),
        "material":("210000000000","200000000000"),"thickness":(".02",".03")}
    old,new=replacements[mutation]
    old=formatted(old) if old.split()[0].isupper() else old
    new='\n'.join(formatted(line) for line in new.split('\n')) if new.split()[0].isupper() else new
    sif.write_text(text.replace(old,new))
    report = run_verification(path,tmp_path/"reports",mode="compare")
    assert report["status"] in {"unsupported","failed"}


def test_comparison_is_complete_and_absolute_plus_relative(tmp_path):
    _,m = fixture(tmp_path)
    a={100:[0,100,0,0,0,0]}; b={100:[0,101,0,0,0,0]}
    assert compare(a,b,m)["status"] == "failed"
    m["tolerances"]={"absolute_pa":0,"relative":.01}
    assert compare(a,b,m)["status"] == "passed"
    for bad in ({}, {100:[0]*5}, {100:[float('nan')]*6}):
        with pytest.raises(Unsupported): compare(bad,b,m)


def test_unconverged_native_is_refused(tmp_path,monkeypatch):
    path,_=fixture(tmp_path); imported=admit(load_manifest(path))
    monkeypatch.setattr("anyfesolver.assembly.solve_linear",lambda *a,**kw:(np.zeros(24),{"convergence_info":{"status":"failed"}}))
    with pytest.raises(Unsupported,match="did not converge"): native_stresses(imported)


def test_reference_source_binding_required(tmp_path):
    path,m=fixture(tmp_path); del m["reference_fem_sha256"]; path.write_text(json.dumps(m))
    report=run_verification(path,tmp_path/"reports",mode="compare")
    assert report["status"]=="unsupported" and "binding" in report["reason"]


@pytest.mark.parametrize("action,status",[("raise SystemExit(7)","process_failed"),("import time; time.sleep(5)","timeout"),("pass","missing_output")])
def test_execution_failure_and_timeout_keep_evidence(tmp_path,action,status):
    path,_=fixture(tmp_path); m=load_manifest(path)
    template=tmp_path/"control.inp"; template.write_text("INAM {prefix}\nRNAM {prefix} FORMATTED\n")
    m["execution"]={"argv":[sys.executable,"-c",action],"timeout_seconds":.3 if status=="timeout" else 5,
        "prefix":"test","control_template":str(template)}
    case=tmp_path/"run";case.mkdir()
    _, evidence=execute_reference(m,case)
    assert evidence["status"]==status
    assert evidence["process_wall_seconds"] > 0
    assert (case/"external"/"execution.json").is_file()
    assert (case/"external"/"stdout.log").is_file()
    with pytest.raises(FileExistsError): execute_reference(m,case)


def test_fresh_mock_execution_integrates_native_compare(tmp_path):
    path,_=fixture(tmp_path);m=json.loads(path.read_text())
    template=tmp_path/"control.inp";template.write_text("INAM {prefix}\nRNAM {prefix} FORMATTED\n")
    m["execution"]={"argv":[sys.executable,"-c", "import shutil; shutil.copyfile("+repr(str(tmp_path/"reference.SIF"))+", 'testR1.SIF')"],
        "timeout_seconds":5,"prefix":"test","control_template":"control.inp"}
    path.write_text(json.dumps(m))
    report=run_verification(path,tmp_path/"reports",mode="execute")
    assert report["status"]=="passed",report.get("reason")
    assert report["reference_origin"]=="fresh_execution"


def test_generated_fem_bytes_layout_and_native_properties(tmp_path):
    from anyfileio.sesam.records import read_raw_records
    fem=tmp_path/'generated.FEM'; report=generate_hanging_shell(fem); data=fem.read_bytes()
    assert b'\r\r\n' not in data and b'\n' not in data.replace(b'\r\n',b'')
    assert report.bytes_written==len(data)
    assert all(len(line)<=72 for line in data.decode('ascii').splitlines())
    d=validate_execution_fem(fem)
    assert len(d.nodes)==4 and len(d.elements)==1
    assert d.header.ident_values==(1.,1.,3.,0.) and d.header.unit_values==(5.,1.,1.,1.,1.)
    assert len(d.element_references[1].raw_values)==12
    assert d.materials[1].poisson_ratio==.3 and d.sections[1].thickness==.02
    records=read_raw_records(fem)
    assert next(r for r in records if r.name=='GELMNT1').source_line_end-next(r for r in records if r.name=='GELMNT1').source_line_start==1


def test_reference_float32_serialization_and_exact_zero_padding(tmp_path):
    from dataclasses import replace
    from anyfileio.sesam.records import read_raw_records, records_to_text
    path,_=fixture(tmp_path);sif=tmp_path/'reference.SIF';records=list(read_raw_records(sif))
    for i,r in enumerate(records):
        if r.name=='MISOSEL': records[i]=replace(r,numeric_fields=(1.,210000003000.,.300000012)+r.numeric_fields[3:])
        elif r.name=='GELTH':records[i]=replace(r,numeric_fields=(1.,.0199999996))
        elif r.name=='BGRAV':records[i]=replace(r,numeric_fields=r.numeric_fields+(0.,))
    load_records=[r for r in records if r.name in {'BGRAV','TDLOAD'}]
    records=[r for r in records if r.name not in {'BGRAV','TDLOAD'}]
    records[-1:-1]=list(reversed(load_records))
    sif.write_bytes(records_to_text(records).encode('ascii'))
    report=run_verification(path,tmp_path/'reports',mode='compare')
    assert report['status']=='passed',report.get('reason')
    pairs=report['reference_metadata']['serialization_normalization']['property_pairs']
    assert any(p['source']!=p['reference'] for p in pairs)
    assert all(p['matched'] and p['atol']==0 for p in pairs)
    for i,r in enumerate(records):
        if r.name=='BGRAV':records[i]=replace(r,numeric_fields=r.numeric_fields[:-1]+(1.,))
    sif.write_bytes(records_to_text(records).encode('ascii'))
    report=run_verification(path,tmp_path/'reports',mode='compare')
    assert report['status']=='unsupported' and 'load contract' in report['reason']


def test_reported_rotated_frame_and_relabelled_descriptor_integrate(tmp_path):
    from dataclasses import replace
    from anyfileio.sesam.records import read_raw_records, records_to_text, FemRawRecord
    path,_=fixture(tmp_path);sif=tmp_path/'reference.SIF';records=list(read_raw_records(sif))
    def record(name,v):return FemRawRecord(name,tuple(map(float,v)),(),0,0,())
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    p=(0,7,1,24,24,10,0,1,0)+tuple(x for i in range(1,11) for x in (i,0,0,0))+tuple(rotation.flatten(order='F'))
    records.insert(-1,record('RDPOINTS',(len(p),)+p[1:]))
    records.insert(-1,record('RDSTRESS',(6,99,3,4,2,1)))
    records.insert(-1,record('RDRESREF',(10,1,1,1,0,0,1,10,1,0)))
    for i,r in enumerate(records):
        if r.name=='RVSTRESS' and len(r.numeric_fields)>4:
            # Components reordered to shear/yy/xx. Reported 90deg frame places
            # local xx into GLOBAL yy; descriptor99 is unrelated to Q4 type24.
            records[i]=replace(r,numeric_fields=(35.,1.,1.,7.,99.)+tuple([0.,0.,39250.]*10))
    sif.write_bytes(records_to_text(records).encode('ascii'))
    report=run_verification(path,tmp_path/'reports',mode='compare')
    assert report['status']=='passed',report.get('reason')
    assert report['reference_metadata']['frame_provenance'][100]['stress_descriptor']==99
    for i,r in enumerate(records):
        if r.name=='RDRESREF':
            records[i]=replace(r,numeric_fields=r.numeric_fields[:3]+(2.,)+r.numeric_fields[4:])
    sif.write_bytes(records_to_text(records).encode('ascii'))
    report=run_verification(path,tmp_path/'reports',mode='compare')
    assert report['status']=='unsupported' and 'external result case' in report['reason']


@pytest.mark.parametrize('mutation',['double_cr','missing_gnode','short_ident','missing_continuation','short_iend','short_material','nonfinite'])
def test_external_layout_refuses_incomplete_before_launch(tmp_path,mutation,monkeypatch):
    from dataclasses import replace
    from anyfileio.sesam.records import read_raw_records, records_to_text
    fem=tmp_path/'generated.FEM'; generate_hanging_shell(fem)
    if mutation=='double_cr': fem.write_bytes(fem.read_bytes().replace(b'\r\n',b'\r\r\n'))
    else:
        names={'missing_gnode':'GNODE','short_ident':'IDENT','missing_continuation':'GELMNT1','short_iend':'IEND','short_material':'MISOSEL','nonfinite':'BGRAV'}
        records=list(read_raw_records(fem))
        for i,r in enumerate(records):
            if r.name==names[mutation]:
                if mutation=='missing_gnode':records.pop(i)
                elif mutation=='nonfinite':records[i]=replace(r,numeric_fields=r.numeric_fields[:-1]+(1e309,))
                else:records[i]=replace(r,numeric_fields=r.numeric_fields[:4] if mutation=='missing_continuation' else r.numeric_fields[:-1])
                break
        fem.write_bytes(records_to_text(records).encode('ascii'))
    monkeypatch.setattr('subprocess.Popen',lambda *a,**kw:pytest.fail('must refuse before process launch'))
    with pytest.raises((Unsupported,ValueError)):execute_reference({'fem':str(fem)},tmp_path)
