"""Authoritative observer witnesses actual source recovery before replay."""
from importlib.util import find_spec
import json
import numpy as np
import pytest
from anyfesolver import (PatternCase,RecoveryConfig,ResourceConfig,prepare_static_response_evaluator,
    export_recovery_capsule_to_sink,CancellationToken,SolveCancelled,PreparedStaticError)
from anyfesolver.result_capsule import _MemorySink,_CapsuleExport,RecoveryCapsuleError
from test_prepared_static_evaluator import _bar,_case,_shell_case
from test_result_capsule_export import _rotated_shell_model
from test_native_affinity_fallback import prepare as native_prepare

def export(evaluator,cases,observer=None,**kwargs):
    sink=_MemorySink()
    export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity='observer-model',
        mesh_identity='observer-mesh',source_field_observer=observer,**kwargs)
    return sink.capsule

def expected(record,key,ids):
    if key.startswith('displacement:'):
        column=('UX','UY','UZ','RX','RY','RZ').index(key.split(':')[1])
        return np.array([record['node_displacements'][int(n)][column] for n in ids])
    if key.startswith('reaction:'):
        column=('FX','FY','FZ','MX','MY','MZ').index(key.split(':')[1])
        return np.array([record['reactions'][int(n)][column] for n in ids])
    if key.startswith('principal_stress_'):
        name,surface=key.split(':');rank=int(name.removeprefix('principal_stress_'))-1
        source={eid:np.asarray(r['principal_stresses_'+surface])[:,rank] for eid,r in record['element_stresses'].items()}
    else:
        name=key.split(':')[0]
        source={eid:np.asarray(r[name]).reshape(-1) for eid,r in record['element_stresses'].items() if name in r}
    return np.concatenate([np.asarray(source[int(eid)]).reshape(-1) for eid in dict.fromkeys(ids)])

@pytest.mark.parametrize('execution',['auto','records'])
@pytest.mark.parametrize('fixture',['bar','rotated_pressure','native'])
def test_complete_actual_source_fields_and_absent_parity(execution,fixture):
    cases=[PatternCase('dense',{'p':1.25,'q':-.375}),PatternCase('empty',{}),
           PatternCase('canonical',{'p':1.},canonical_load_case=_case(9,5.).load_case)]
    if fixture=='native':
        manager=native_prepare(numerical_affinity_fallback=True);cases=[PatternCase('dense',{'p0':1.25,'p1':-.375}),PatternCase('empty',{})]
    else:
        model=_bar() if fixture=='bar' else _rotated_shell_model()
        factory=_case if fixture=='bar' else lambda i,v:_shell_case(i,pressure=v)
        if fixture!='bar':cases=cases[:2]
        manager=prepare_static_response_evaluator(model,{'p':factory(1,1.).load_case,'q':factory(2,.4).load_case},
            baseline_load_case=factory(0,.25).load_case,stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_stresses=True,include_displacements=True,include_reactions=True))
    with manager as evaluator:
        reference=next(evaluator.iter_batches(cases)).case_results
        original=export(evaluator,cases,recovery_execution=execution)
        snapshots=[]
        observed=export(evaluator,cases,snapshots.append,recovery_execution=execution)
        assert len(snapshots)==len(cases)
        for snapshot,record in zip(snapshots,reference):
            assert snapshot['schema']=='anysolver.source_fields' and snapshot['version']==1
            assert snapshot['case']['certified'] and snapshot['case']['status']=='converged'
            for key,field in snapshot['fields'].items():
                values=field['values'];ids=field['associations']
                descriptor=next(d for d in observed['descriptors'] if d['key']==key)
                assert all(field[n]==descriptor[n] for n in ('quantity_id','component','location','unit','basis'))
                assert np.array_equal(values.view(np.uint64),expected(record,key,ids).view(np.uint64)),key
                assert values.dtype==np.float64 and not values.flags.writeable
                assert ids.dtype==np.int64 and len(ids)==len(values)==len(field['native_ordinals'])
                for eid in dict.fromkeys(ids):
                    positions=np.flatnonzero(ids==eid)
                    assert field['native_ordinals'][positions].tolist()==list(range(len(positions)))
                with pytest.raises(ValueError):values.setflags(write=True)
                with pytest.raises(ValueError):values.base.setflags(write=True)
            with pytest.raises(TypeError):snapshot['case']['certified']=False
        # No-observer parity: scientific rows, associations and case eligibility unchanged.
        assert original['cases']==observed['cases']
        for key in original['responses']:assert np.array_equal(original['responses'][key].view(np.uint64),observed['responses'][key].view(np.uint64))
        for key in original['stored']:assert np.array_equal(original['stored'][key]['values'].view(np.uint64),observed['stored'][key]['values'].view(np.uint64))
        for key in original['associations']:assert np.array_equal(original['associations'][key],observed['associations'][key])

class FatalObserver(BaseException):pass
@pytest.mark.parametrize('kind',['ordinary','fatal','array','mapping','cancel','authority'])
def test_observer_abort_preserves_primary_target_and_diagnostics(tmp_path,kind):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    path=tmp_path/'prior.h5';path.write_bytes(b'previous target')
    token=CancellationToken();failure=RuntimeError('observer original') if kind=='ordinary' else FatalObserver('observer fatal')
    original_add=np.add
    def observer(payload):
        if kind in ('ordinary','fatal'):raise failure
        if kind=='array':next(iter(payload['fields'].values()))['values'].setflags(write=True)
        if kind=='mapping':payload['case']['status']='failed'
        if kind=='cancel':token.cancel('observer cancelled')
        if kind=='authority':np.add=lambda *a,**k:None
    error_type=RuntimeError if kind=='ordinary' else FatalObserver if kind=='fatal' else ValueError if kind=='array' else TypeError if kind=='mapping' else SolveCancelled if kind=='cancel' else PreparedStaticError
    try:
        with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',recovery_config=RecoveryConfig(include_stresses=True,include_displacements=True)) as evaluator:
            with pytest.raises(error_type) as caught:
                with ArchiveSink(path) as sink:
                    export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{'p':1.})],sink=sink,
                        model_identity='observer-model',mesh_identity='observer-mesh',source_field_observer=observer,cancellation_token=token)
            if kind in ('ordinary','fatal'):assert caught.value is failure
    finally:np.add=original_add
    assert path.read_bytes()==b'previous target'
    diagnostics=list(tmp_path.glob('*.failure.json'));assert diagnostics
    record=json.loads(diagnostics[0].read_text());assert record['input_case_ids']==['a']
    assert not list(tmp_path.glob('.*.tmp'))

def test_legacy_witness_never_replays_missing_actual_displacements():
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_stresses=True,include_displacements=True)) as evaluator:
        case=PatternCase('a',{'p':1.});record=dict(next(evaluator.iter_batches([case])).case_results[0])
        assert record['node_displacements']
        record.pop('node_displacements',None)
        snapshots=[]
        exporter=_CapsuleExport(evaluator,[case],model_identity='m',mesh_identity='g',replay_version=1,source_field_observer=snapshots.append)
        exporter._process_case(0,record)
        assert set(snapshots[0]['unavailable_components'])=={'displacement:'+c for c in ('UX','UY','UZ','RX','RY','RZ')}
        assert not any(k.startswith('displacement:') for k in snapshots[0]['fields'])

def test_invalid_observer_typed_refusal():
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
        with pytest.raises(RecoveryCapsuleError,match='observer'):export(evaluator,[PatternCase('a',{'p':1.})],3)

@pytest.mark.parametrize('secondary',['cancel','authority'])
def test_original_observer_error_remains_primary_with_secondary(tmp_path,secondary):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    path=tmp_path/'prior.h5';path.write_bytes(b'prior')
    token=CancellationToken();primary=FatalObserver('primary observer error');original=np.add
    def observer(payload):
        if secondary=='cancel':token.cancel('secondary cancel')
        else:np.add=lambda *args:None
        raise primary
    try:
        with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
            with pytest.raises(FatalObserver) as caught:
                with ArchiveSink(path) as sink:
                    export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{'p':1.})],sink=sink,
                        model_identity='m',mesh_identity='g',source_field_observer=observer,cancellation_token=token)
            assert caught.value is primary
            assert isinstance(primary.__cause__,SolveCancelled if secondary=='cancel' else PreparedStaticError)
    finally:np.add=original
    assert path.read_bytes()==b'prior'
    assert json.loads(next(tmp_path.glob('*.failure.json')).read_text())['error_type']=='FatalObserver'

def test_observer_resource_check_precedes_snapshot_allocation():
    from anyfesolver.recovery import ResourcePolicyError
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
        required=evaluator._preparation_memory_peak_required_bytes
    cfg=ResourceConfig(memory_limit_bytes=int(required)+32768)
    observed=[]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',resource_config=cfg) as evaluator:
        with pytest.raises(ResourcePolicyError):export(evaluator,[PatternCase('a',{'p':1.})],observed.append)
    assert observed==[]



def test_metadata_reuse_keeps_values_fresh_and_never_retains_history():
    from anyfesolver.result_capsule import _Layout
    from anyfesolver.prepared_static_numeric import NumericCase,NumericField
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_stresses=True,include_displacements=True,include_reactions=False)) as evaluator:
        cases=[PatternCase('a',{'p':1.}),PatternCase('b',{'p':2.})]
        numeric=next(evaluator._iter_capsule_batches(cases)).case_results[0]
        assert isinstance(numeric,NumericCase)
        snapshots=[];export(evaluator,cases,snapshots.append)
        for key in snapshots[0]['fields']:
            assert snapshots[0]['fields'][key]['associations'] is snapshots[1]['fields'][key]['associations'],key
            assert snapshots[0]['fields'][key]['values'] is not snapshots[1]['fields'][key]['values']
        assert not np.array_equal(snapshots[0]['fields']['displacement:UX']['values'],snapshots[1]['fields']['displacement:UX']['values'])
        exporter=_CapsuleExport(evaluator,cases,model_identity='m',mesh_identity='g',source_field_observer=lambda p:None)
        layout=_Layout();layout.add(1,2)
        ids=(1,);shape=((2,),);offsets=np.array([0,2],dtype=np.int64)
        field=NumericField(ids,shape,offsets,np.array([1.,2.]))
        record=NumericCase({}, {'q':field},(),None)
        first=exporter._observer_layout('q',record,layout)
        assert exporter._observer_layout('q',record,layout) is first
        changed=NumericCase({}, {'q':NumericField(ids,((1,2),),offsets,field.values)},(),None)
        second=exporter._observer_layout('q',changed,layout)
        assert second is not first and second['native_shapes']==((1,2),)
        layout.add(2,1)
        third=exporter._observer_layout('q',NumericCase({}, {'q':NumericField((1,2),((1,2),()),np.array([0,2,3]),np.array([1.,2.,3.]))},(),None),layout)
        assert third['associations'].tolist()==[1,1,2]
        assert len(exporter._observer_metadata)==1
        assert exporter._observer_metadata_bytes==exporter._observer_metadata['q'][4]

@pytest.mark.parametrize('name',['id','type'])
def test_observer_builtin_shadow_injection_refuses_before_next_snapshot(name):
    import anyfesolver.result_capsule as module
    calls=[]
    def observer(payload):
        calls.append(payload)
        setattr(module,name,lambda *a:0)
    try:
        with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
            with pytest.raises(RecoveryCapsuleError,match='namespace changed'):
                export(evaluator,[PatternCase('a',{'p':1.}),PatternCase('b',{'p':2.})],observer)
    finally:delattr(module,name)
    assert len(calls)==1

@pytest.mark.parametrize('mutation',['dtype','shape'])
def test_readonly_array_header_mutation_also_refuses(mutation):
    def observer(payload):
        value=payload['fields']['displacement:UX']['values']
        if mutation=='dtype':value.dtype=np.uint64
        else:value.shape=(1,len(value))
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_displacements=True)) as evaluator:
        with pytest.raises(RecoveryCapsuleError,match='array layout changed'):
            export(evaluator,[PatternCase('a',{'p':1.})],observer)

@pytest.mark.parametrize('name',['values','associations','native_ordinals'])
def test_stride_mutation_aborts_and_preserves_old_target(tmp_path,name):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    path=tmp_path/'prior.h5';path.write_bytes(b'old target')
    def observer(payload):payload['fields']['displacement:UX'][name].strides=(0,)
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_displacements=True)) as evaluator:
        with pytest.raises(RecoveryCapsuleError,match='array layout changed'):
            with ArchiveSink(path) as sink:
                export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{'p':1.})],sink=sink,
                    model_identity='m',mesh_identity='g',source_field_observer=observer)
    assert path.read_bytes()==b'old target'
    assert list(tmp_path.glob('*.failure.json'))

def test_cached_static_stride_mutation_refuses_before_reuse():
    from anyfesolver.result_capsule import _Layout
    from anyfesolver.prepared_static_numeric import NumericCase,NumericField
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
        exporter=_CapsuleExport(evaluator,[PatternCase('a',{'p':1.})],model_identity='m',mesh_identity='g',source_field_observer=lambda p:None)
        layout=_Layout();layout.add(1,2)
        record=NumericCase({}, {'q':NumericField((1,),((2,),),np.array([0,2]),np.array([1.,2.]))},(),None)
        first=exporter._observer_layout('q',record,layout)
        first['associations'].strides=(0,)
        with pytest.raises(RecoveryCapsuleError,match='cached array layout changed'):exporter._observer_layout('q',record,layout)

@pytest.mark.parametrize('target',['_observer_freeze','MappingProxyType','_require_observer_environment','_observer_layout',
                                  '_observer_array_facts','_require_observer_array_facts'])
def test_new_observer_executable_binding_mutations_refuse(target):
    import anyfesolver.result_capsule as rc
    owner=rc._CapsuleExport if target=='_observer_layout' else rc
    original=getattr(owner,target);calls=[]
    def observer(payload):
        calls.append(payload)
        setattr(owner,target,dict if target=='MappingProxyType' else lambda *a,**k:None)
    try:
        with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
            with pytest.raises(PreparedStaticError):export(evaluator,[PatternCase('a',{'p':1.}),PatternCase('b',{'p':2.})],observer)
    finally:setattr(owner,target,original)
    assert len(calls)==1

def test_large_intermediate_backing_remains_alive_during_validation():
    # Isolate the memory-safety assertion: a faulty implementation must not
    # crash the pytest process when an OS-sized backing allocation is released.
    import subprocess,sys
    from pathlib import Path
    import anyfesolver.result_capsule as rc
    package_root=str(Path(rc.__file__).resolve().parent.parent)
    code=r'''
import gc,sys
sys.path.insert(0,sys.argv[1])
import numpy as np
from anyfesolver.result_capsule import (_observer_freeze,_observer_array_facts,
    _require_observer_array_facts,RecoveryCapsuleError)
view=_observer_freeze(np.arange(262144,dtype=np.float64))
expected=float(view.sum())
facts=_observer_array_facts(view)
owner=view.base
replacement=np.full(owner.shape,-1.,dtype=owner.dtype).tobytes()
np.ndarray.__setstate__(owner,(1,owner.shape,owner.dtype,False,replacement))
owner.setflags(write=False)
del owner,replacement
gc.collect()
assert float(view.sum())==expected
try:_require_observer_array_facts(facts)
except RecoveryCapsuleError as error:assert 'backing chain' in str(error)
else:raise AssertionError('replacement backing was admitted')
'''
    result=subprocess.run([sys.executable,'-X','faulthandler','-c',code,package_root],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.mark.parametrize('name',['values','associations'])
def test_setstate_replacing_buffer_refuses_same_header(name):
    def observer(payload):
        value=payload['fields']['displacement:UX'][name]
        state=(1,value.shape,value.dtype,False,np.zeros(value.shape,dtype=value.dtype).tobytes())
        np.ndarray.__setstate__(value,state);value.setflags(write=False)
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_displacements=True)) as evaluator:
        with pytest.raises(RecoveryCapsuleError,match='array layout changed'):export(evaluator,[PatternCase('a',{'p':1.})],observer)

def test_existing_primary_cause_survives_secondary_cancel():
    primary=FatalObserver('primary');cause=ValueError('original cause');primary.__cause__=cause
    token=CancellationToken()
    def observer(payload):token.cancel('secondary');raise primary
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
        with pytest.raises(FatalObserver) as caught:export(evaluator,[PatternCase('a',{'p':1.})],observer,cancellation_token=token)
    assert caught.value is primary and primary.__cause__ is cause
    assert any('SolveCancelled' in note for note in primary.__notes__)

def test_batch_ledger_counts_python_containers_and_scalar_records():
    from anyfesolver.result_capsule import _observer_numeric_bytes
    record={'nodes':{i:[float(i)]*6 for i in range(32)},'provenance':['source']*32}
    assert _observer_numeric_bytes(record)>32*6*8

@pytest.mark.parametrize('name',['values','associations','native_ordinals'])
@pytest.mark.parametrize('primary_failure',[False,True])
def test_intermediate_base_replacement_second_callback_refuses(tmp_path,name,primary_failure):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    path=tmp_path/'prior.h5';path.write_bytes(b'old target')
    snapshots=[];primary=FatalObserver('original observer failure');cause=ValueError('original cause')
    primary.__cause__=cause
    def observer(payload):
        snapshots.append(payload)
        value=payload['fields']['displacement:UX'][name]
        base=value
        while type(base) is np.ndarray:
            assert not base.flags.owndata and not base.flags.writeable
            base=base.base
        assert type(base) is bytes
        if len(snapshots)==1:return
        if name!='values':assert value is snapshots[0]['fields']['displacement:UX'][name]
        before=value.tobytes()
        owner=value.base
        np.ndarray.__setstate__(owner,(1,owner.shape,owner.dtype,False,np.zeros(owner.shape,dtype=owner.dtype).tobytes()))
        owner.setflags(write=False)
        # Original terminal bytes is strongly held by facts; the payload view
        # remains valid even while its intermediate owner has replaced data.
        assert value.tobytes()==before
        if primary_failure:raise primary
    error_type=FatalObserver if primary_failure else RecoveryCapsuleError
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library',
            recovery_config=RecoveryConfig(include_displacements=True)) as evaluator:
        with pytest.raises(error_type) as caught:
            with ArchiveSink(path) as sink:
                export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{'p':1.}),PatternCase('b',{'p':2.})],
                    sink=sink,model_identity='m',mesh_identity='g',source_field_observer=observer)
    assert len(snapshots)==2
    if primary_failure:
        assert caught.value is primary and primary.__cause__ is cause
        assert any('backing chain' in note for note in primary.__notes__)
    else:assert 'backing chain' in str(caught.value)
    assert path.read_bytes()==b'old target'
    assert list(tmp_path.glob('*.failure.json'))

@pytest.mark.parametrize('name',['associations','native_ordinals'])
def test_cached_intermediate_base_replacement_refuses_before_reuse(name):
    from anyfesolver.result_capsule import _Layout
    from anyfesolver.prepared_static_numeric import NumericCase,NumericField
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},stress_execution='certified_library') as evaluator:
        exporter=_CapsuleExport(evaluator,[PatternCase('a',{'p':1.})],model_identity='m',mesh_identity='g',source_field_observer=lambda p:None)
        layout=_Layout();layout.add(1,2)
        record=NumericCase({}, {'q':NumericField((1,),((2,),),np.array([0,2]),np.array([1.,2.]))},(),None)
        metadata=exporter._observer_layout('q',record,layout)
        owner=metadata[name].base
        np.ndarray.__setstate__(owner,(1,owner.shape,owner.dtype,False,np.zeros(owner.shape,dtype=owner.dtype).tobytes()))
        owner.setflags(write=False)
        with pytest.raises(RecoveryCapsuleError,match='backing chain'):
            exporter._observer_layout('q',record,layout)
