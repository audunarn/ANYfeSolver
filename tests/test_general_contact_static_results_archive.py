"""Closed codec controls and an actual static producer/consumer."""
from dataclasses import fields, replace, is_dataclass
import hashlib, json, os, struct
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.nonlinear_static import NonlinearStaticResult, NonlinearStaticStep, NonlinearIncrementSnapshot
from anyfesolver.general_contact import (StaticContactArchiveError as Error,
    serialize_static_contact_results as serialize, read_static_contact_results as read)
from anyfesolver.general_contact import transient_results_archive as codec


def sample():
    u=np.array([-.0, np.nextafter(0.,1.), 1.])
    step=NonlinearStaticStep(1,1.,2,.0,.0,.0,support_reactions={'root':(0.,)*6})
    snap=NonlinearIncrementSnapshot(1,1.,u,{1:{'plastic_strain':np.zeros((2,3))}})
    return NonlinearStaticResult([step],'completed',u,1.,{1:{'state':np.array([-.0])}},
        {'general_contact':{'minimum_gap':float('inf'),'physical_force':(0.,)*3}},
        (snap,),{'schema':'producer-checkpoint','binary':-.0})


def test_exact_detached_and_refusal():
    value=sample();raw=serialize(value);archive=read(raw)
    data=archive.to_dict()
    assert set(data)=={f.name for f in fields(value)}
    assert data['displacements'].tobytes()==value.displacements.tobytes()
    assert data['snapshots'][0]['fields']['displacements'].tobytes()==value.displacements.tobytes()
    assert np.signbit(data['restart_checkpoint']['binary'])
    data['displacements'][0]=2.;value.displacements[1]=4.
    assert archive.to_bytes()==raw and np.signbit(archive.to_dict()['displacements'][0])
    for bad in (replace(sample(),info={}),replace(sample(),displacements=np.array([True])),
                replace(sample(),element_states={1:object()}),replace(sample(),load_factor=np.nan),
                replace(sample(),steps=[object()])):
        with pytest.raises(Error):serialize(bad)
    with pytest.raises(Error):serialize(sample(),max_bytes=128)
    with pytest.raises(Error):read(raw,max_bytes=len(raw)-1)
    doc=json.loads(raw);doc['schema']='wrong'
    with pytest.raises(Error):read(codec._canonical(doc))
    assert codec._types()[0].__name__=='TransientContactResults'


def test_active_store_rejected_before_callback(monkeypatch):
    from anyfesolver.nonlinear_state import NonlinearStateStore
    store=NonlinearStateStore()
    monkeypatch.setattr(NonlinearStateStore,'has_active_trial',property(lambda self:True))
    monkeypatch.setattr(NonlinearStateStore,'materialize_owned',lambda self:pytest.fail('copy callback'))
    with pytest.raises(Error):serialize(replace(sample(),element_states=store))
    snapshot=object.__new__(NonlinearIncrementSnapshot)
    for key,value in {'step_index':1,'load_factor':1.,'displacements':np.zeros(3),
                      'element_states':store,'control_value':None}.items():
        object.__setattr__(snapshot,key,value)
    with pytest.raises(Error):serialize(replace(sample(),snapshots=(snapshot,)))


def assert_producer_data(expected, actual):
    """Independent preservation oracle; no archive encode/decode helpers."""
    from anyfesolver.nonlinear_state import NonlinearStateStore
    if type(expected) is NonlinearStateStore:
        assert not expected.has_active_trial
        expected = expected.materialize_owned()
    if is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is dict
        assert set(actual) == {'record_type', 'fields', 'summary'}
        module = type(expected).__module__.replace('anyfesolver.', 'anysolver.', 1)
        assert actual['record_type'] == module + '.' + type(expected).__qualname__
        assert_producer_data({field.name:getattr(expected,field.name) for field in fields(expected)},
                             actual['fields'])
        summary = expected.to_dict() if hasattr(expected, 'to_dict') else None
        assert_producer_data(summary, actual['summary'])
    elif type(expected) is np.ndarray:
        assert type(actual) is np.ndarray
        assert expected.dtype.str == actual.dtype.str and expected.shape == actual.shape
        assert expected.tobytes(order='C') == actual.tobytes(order='C')
    elif isinstance(expected, np.generic):
        assert type(actual) is type(expected) and expected.dtype.str == actual.dtype.str
        assert expected.tobytes() == actual.tobytes()
    elif type(expected) is float:
        assert type(actual) is float and struct.pack('>d',expected) == struct.pack('>d',actual)
    elif type(expected) is dict:
        assert type(actual) is dict
        assert {(type(k),k) for k in expected} == {(type(k),k) for k in actual}
        for key,value in expected.items(): assert_producer_data(value,actual[key])
    elif type(expected) in (tuple,list):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left,right in zip(expected,actual): assert_producer_data(left,right)
    else:
        assert type(expected) in (str,int,bool,type(None))
        assert type(actual) is type(expected) and actual == expected


def test_independent_preservation_oracle():
    value=sample()
    data=read(serialize(value)).to_dict()
    expected={field.name:getattr(value,field.name) for field in fields(value)}
    assert_producer_data(expected,data)
    for left,right in ((-.0,.0),(np.float32(-.0),np.float32(.0)),
                       (np.array([1.]),np.array([1.],dtype=np.float32)),
                       ({'x':1},{}),((1,),[1])):
        with pytest.raises(AssertionError): assert_producer_data(left,right)


def test_actual_static_producer_consumer():
    from examples.general_contact_static_results_archive import analysis
    result, archive=analysis()
    assert result.status=='completed' and result.info['general_contact']['penalty_energy']>0
    assert result.snapshots and result.restart_checkpoint is not None
    data=archive.to_dict()
    assert_producer_data({field.name:getattr(result,field.name) for field in fields(result)}, data)
    report={'status':result.status,'steps':len(result.steps),'snapshots':len(result.snapshots),
        'archive_bytes':len(archive.to_bytes()),'sha256':archive.sha256,
        'full_fields_equal':True,'checkpoint_equal':True,
        'preservation_oracle':'independent_recursive_producer_fields_and_raw_numeric_bytes','contact':result.info['general_contact']}
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path).write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


def test_closed_roles_and_wire_compatibility():
    for bad in (replace(sample(),steps=[replace(sample().steps[0],control_value='bad')]),
                replace(sample(),steps=[replace(sample().steps[0],support_reactions={'root':('x',)*6})]),
                replace(sample(),info={'general_contact':{'integration_force':('x',)*3}})):
        with pytest.raises(Error):serialize(bad)
    doc=json.loads(serialize(sample()))
    nodes={key:value for key,value in doc['payload']['items']}
    nodes['snapshots']['items'][0]['fields']['items'].append(['unknown',0])
    doc.pop('sha256');doc['sha256']=hashlib.sha256(codec._canonical(doc)).hexdigest()
    with pytest.raises(Error):read(codec._canonical(doc))
    baseline=os.environ.get('STATIC_ARCHIVE_LEGACY_CODEC')
    expected_baseline=os.environ.get('STATIC_ARCHIVE_EXPECTED_LEGACY_CODEC')
    if expected_baseline:
        assert baseline and Path(baseline).resolve()==Path(expected_baseline).resolve()
    if baseline:
        import importlib.util
        spec=importlib.util.spec_from_file_location('anyfesolver.general_contact._legacy_archive',baseline)
        module=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
        from test_general_contact_transient_results_archive import sample as transient_sample
        value=transient_sample()
        before=module.serialize_transient_contact_results(value)
        after=codec.serialize_transient_contact_results(value)
        record={'baseline_path':str(Path(baseline).resolve()),
                'baseline_sha256':hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
                'expected_baseline_required':bool(expected_baseline),
                'before_sha256':hashlib.sha256(before).hexdigest(),
                'after_sha256':hashlib.sha256(after).hexdigest(),
                'before_bytes':len(before),'after_bytes':len(after),'byte_equal':before==after}
        evidence=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if evidence:Path(evidence).with_name('wire_compatibility.json').write_text(json.dumps(record,indent=2))
        assert before==after
        # Baseline is evidence input, not part of the executing implementation.
        del sys.modules[spec.name]
