"""Independent arc producer preservation, inert export and closed refusals."""
from dataclasses import fields,replace
import hashlib,json,os,importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.arc_length import ArcLengthResult,ArcLengthStep
from anyfesolver.nonlinear_static import NonlinearIncrementSnapshot
from anyfesolver.general_contact import (ArcContactArchiveError as Error,
    serialize_arc_contact_results as serialize,read_arc_contact_results as read)
from anyfesolver.general_contact import transient_results_archive as codec
from test_general_contact_static_results_archive import assert_producer_data


def sample():
    u=np.array([-0.,np.nextafter(0.,1.),1.])
    step=ArcLengthStep(1,.1,2,1,.02,0.,-0.,.01,.1,.02,0.,True,{'root':(0.,)*6})
    snapshot=NonlinearIncrementSnapshot(1,.1,u,{1:{'strain':np.array([-.0],dtype=np.float32)}})
    return ArcLengthResult([step],'maximum_steps_reached',u,.1,.1,1,
        {1:{'bits':np.array([-.0])}},{'general_contact':{'minimum_gap':float('inf')},
        'numpy_scalar':np.float32(-.0)},(snapshot,),{'full':{'bits':-.0}})


def test_exact_detached_closed_roles_and_resources():
    value=sample();raw=serialize(value);archive=read(raw)
    assert_producer_data({f.name:getattr(value,f.name) for f in fields(value)},archive.to_dict())
    assert raw==serialize(value)
    view=archive.to_dict();view['displacements'][0]=2.;value.displacements[1]=4.
    assert archive.to_bytes()==raw and np.signbit(archive.to_dict()['displacements'][0])
    for bad in (replace(sample(),info={}),replace(sample(),peak_step_index=True),
        replace(sample(),peak_load_factor=np.nan),replace(sample(),element_states={1:object()}),
        replace(sample(),steps=[replace(sample().steps[0],is_peak=1)]),
        replace(sample(),steps=[replace(sample().steps[0],retries=-1)]),
        replace(sample(),steps=[replace(sample().steps[0],support_reactions={'root':('x',)*6})]),
        replace(sample(),snapshots=(NonlinearIncrementSnapshot(1,.1,np.zeros(2),{}),))):
        with pytest.raises(Error):serialize(bad)
    with pytest.raises(Error):serialize(object.__new__(ArcLengthResult))
    with pytest.raises(Error):serialize(replace(sample(),steps=[object.__new__(ArcLengthStep)]))
    with pytest.raises(Error):serialize(sample(),max_bytes=128)
    with pytest.raises(Error):read(raw,max_bytes=len(raw)-1)
    document=json.loads(raw);document['schema']='unknown'
    with pytest.raises(Error):read(json.dumps(document).encode())
    document=json.loads(raw);document['payload']['items'].append(['status','extra'])
    document.pop('sha256');document['sha256']=hashlib.sha256(codec._canonical(document)).hexdigest()
    with pytest.raises(Error):read(codec._canonical(document))


def test_unknown_and_active_material_before_callbacks(monkeypatch):
    class Malicious:
        def __deepcopy__(self,memo):pytest.fail('untrusted copy callback')
    with pytest.raises(Error):serialize(replace(sample(),element_states={1:Malicious()}))
    from anyfesolver.nonlinear_state import NonlinearStateStore
    store=NonlinearStateStore();monkeypatch.setattr(NonlinearStateStore,'has_active_trial',property(lambda self:True))
    monkeypatch.setattr(NonlinearStateStore,'materialize_owned',lambda *a,**k:pytest.fail('active copy'))
    with pytest.raises(Error):serialize(replace(sample(),element_states=store))
    snapshot=object.__new__(NonlinearIncrementSnapshot)
    for name,value in dict(step_index=1,load_factor=.1,displacements=np.zeros(3),element_states=store,control_value=None).items():
        object.__setattr__(snapshot,name,value)
    with pytest.raises(Error):serialize(replace(sample(),snapshots=(snapshot,)))


def test_actual_arc_producer_consumer():
    from examples.general_contact_arc_results_archive import analysis
    result,archive=analysis();data=archive.to_dict()
    assert len(fields(result))==10 and len(result.steps)==2
    assert result.status=='maximum_steps_reached' and result.snapshots
    assert result.info['general_contact']['penalty_energy']>0
    assert result.restart_checkpoint['path_state']['contact_state']['schema'].endswith('PRELOAD_V2')
    assert_producer_data({f.name:getattr(result,f.name) for f in fields(result)},data)
    assert archive.to_bytes()==result.to_contact_archive_bytes()
    record={'status':result.status,'fields':len(fields(result)),'steps':len(result.steps),
        'snapshots':len(result.snapshots),'archive_bytes':len(archive.to_bytes()),'sha256':archive.sha256,
        'full_fields_equal':True,'checkpoint_equal':True,'peak_load_factor':result.peak_load_factor,
        'peak_step_index':result.peak_step_index,'retries':[x.retries for x in result.steps],
        'arc_residuals':[x.arc_residual for x in result.steps],
        'oracle':'independent_recursive_producer_fields_and_raw_numeric_bytes'}
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path).write_text(json.dumps(record,indent=2))


@pytest.mark.skipif(not os.environ.get('ARC_ARCHIVE_LEGACY_ROOT'),
    reason='archived legacy source root ARC_ARCHIVE_LEGACY_ROOT absent: legacy wire byte-compare requires the original legacy archive sources')
def test_legacy_static_and_transient_wire_unchanged():
    from test_general_contact_static_results_archive import sample as static_sample
    from test_general_contact_transient_results_archive import sample as transient_sample
    from anyfesolver.general_contact import static_results_archive as static
    baseline=Path(os.environ['ARC_ARCHIVE_LEGACY_ROOT'])
    rows=[]
    for name,producer,current in (('transient_results_archive',transient_sample,codec),
                                  ('static_results_archive',static_sample,static)):
        path=baseline/(name+'.py');alias='anyfesolver.general_contact._legacy_'+name
        spec=importlib.util.spec_from_file_location(alias,path);module=importlib.util.module_from_spec(spec)
        sys.modules[alias]=module
        try:
            spec.loader.exec_module(module)
            fn='serialize_'+('transient' if name.startswith('transient') else 'static')+'_contact_results'
            value=producer();before=getattr(module,fn)(value);after=getattr(current,fn)(value)
            rows.append({'baseline':str(path),'baseline_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),
                'byte_equal':before==after})
            assert before==after
        finally:sys.modules.pop(alias,None)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path).with_name('arc_legacy_wire_compatibility.json').write_text(json.dumps(rows,indent=2))
