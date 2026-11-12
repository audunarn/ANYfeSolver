"""Actual small source ownership/bit controls for storage-only publication."""
from importlib.util import find_spec
from copy import deepcopy
import numpy as np
import pytest
from anyfesolver import (PatternCase,LoadCase,RecoveryConfig,ResourceConfig,prepare_static_response_evaluator,
    export_recovery_capsule,export_recovery_capsule_to_sink)
from anyfesolver_results import validate_storage_prediction,CapsuleValidationError,validate_capsule
from test_native_affinity_fallback import original_fixture,prepare

def test_dense_twenty_pattern_source_actual_rows_and_metadata(tmp_path):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink,open_results
    model,baseline,_=original_fixture(True)
    patterns={f'p{i}':LoadCase(f'p{i}') for i in range(20)}
    for eid in (1,4808):
        for index,load in enumerate(patterns.values()):
            load.add_nodal_load(2*eid,[float(1000.*np.sin(eid*(index+1)*.017+(index+1)*.31)),0.,0.,0.,0.,0.])
    dense={f'p{i}':(-1 if i%2 else 1)*(i+.37) for i in range(20)}
    cases=[PatternCase('dense',dense),PatternCase('reversed',dict(reversed(list(dense.items())))),PatternCase('baseline',{})]
    with prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,
        recovery_config=RecoveryConfig(include_stresses=True,include_reactions=True),
        stress_execution='certified_library',numerical_affinity_fallback=True) as evaluator:
        before=evaluator.diagnostics()['source_response']
        assert before['element_library_probes']==24
        records=next(evaluator.iter_batches(cases,batch_size=3)).case_results
        path=tmp_path/'source.h5'
        with ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity='actual-two-beam-dense20',mesh_identity='actual-original-ids',batch_size=3)
        with open_results(path) as reader:
            for descriptor in reader.components('stress'):
                assert descriptor['availability']=='stored'
                ids=reader.associations('stress',descriptor['component'])
                for ci,record in enumerate(records):
                    expected=np.array([record['element_stresses'][int(eid)][descriptor['key']] for eid in ids])
                    actual=reader.read_field(cases[ci].case_id,'stress',descriptor['component'])
                    assert np.array_equal(actual.view(np.uint64),expected.view(np.uint64))
            assert 'storage_prediction' in reader.metadata()
            assert not reader._response_keys.intersection(d['key'] for d in reader.components('stress'))

def test_role_native_slot_is_not_an_engineering_basis():
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        c=export_recovery_capsule(evaluator,[PatternCase('a',{'p0':1.7})],model_identity='m',mesh_identity='s')
        role=c['storage_prediction'];validate_capsule(c)
        assert role['provenance']['native_element_ids']==[4808]
        for key,values in role['responses'].items():
            assert np.array_equal(values[role['associations'][key]==4808],np.zeros_like(values[role['associations'][key]==4808]))
            assert key not in c['responses'] and next(d for d in c['descriptors'] if d['key']==key)['availability']=='stored'
            with pytest.raises(ValueError):values.setflags(write=True)

def test_storage_allocation_refused_before_numeric_plan_or_publication(tmp_path):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    from anyfesolver.recovery import ResourcePolicyError
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        evaluator._resource_config=ResourceConfig(memory_limit_bytes=1)
        path=tmp_path/'previous.h5';path.write_bytes(b'previous')
        with pytest.raises(ResourcePolicyError) as caught,ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=sink,model_identity='m',mesh_identity='s')
        assert caught.value.memory_estimate.total_bytes_estimate>1
        assert evaluator._numeric_export_plan is None and path.read_bytes()==b'previous'
        assert sink.diagnostics['state']=='aborted'

def test_storage_validator_entrypoint_mutation_from_real_callback_refuses(tmp_path,monkeypatch):
    import anyfesolver_results as runtime
    from anyfesolver.prepared_static_loads import PreparedStaticError
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        path=tmp_path/'previous.h5';path.write_bytes(b'previous')
        native=ArchiveSink(path)
        class Sink:
            capsule_case=native.capsule_case
            capsule_stored=native.capsule_stored
            capsule_end=native.capsule_end
            capsule_failed=native.capsule_failed
            def capsule_begin(self,header):
                native.capsule_begin(header)
                monkeypatch.setattr(runtime,'validate_storage_prediction',lambda *a,**k:None)
        with pytest.raises(PreparedStaticError),native:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=Sink(),model_identity='m',mesh_identity='s')
        assert path.read_bytes()==b'previous' and native.diagnostics['state']=='aborted'

def test_cancel_after_new_codec_stored_callback_retains_previous_target(tmp_path):
    from anyfesolver import CancellationToken,SolveCancelled
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    token=CancellationToken()
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        path=tmp_path/'previous.h5';path.write_bytes(b'previous');native=ArchiveSink(path)
        class Sink:
            capsule_begin=native.capsule_begin
            capsule_case=native.capsule_case
            capsule_end=native.capsule_end
            capsule_failed=native.capsule_failed
            def capsule_stored(self,*args):native.capsule_stored(*args);token.cancel('new codec row written')
        with pytest.raises(SolveCancelled),native:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=Sink(),model_identity='m',mesh_identity='s',cancellation_token=token)
        assert path.read_bytes()==b'previous' and native.diagnostics['state']=='aborted'

def test_new_structural_role_refuses_escaping_scientific_targets():
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        c=export_recovery_capsule(evaluator,[PatternCase('a',{})],model_identity='m',mesh_identity='s')
        role=c['storage_prediction']
        descriptors=[dict(item) for item in c['descriptors']]
        descriptors[0]['availability']='reconstructable'
        with pytest.raises(CapsuleValidationError):validate_storage_prediction(role,patterns=c['patterns'],descriptors=descriptors,associations=c['associations'])

def test_unused_role_zero_cases_public_memory_and_stream(tmp_path):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink,open_results
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        capsule=export_recovery_capsule(evaluator,[],model_identity='m',mesh_identity='s')
        assert not capsule['cases'] and 'storage_prediction' not in capsule
        validate_capsule(capsule)
        path=tmp_path/'zero.h5'
        with ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[],sink=sink,model_identity='m',mesh_identity='s')
        with open_results(path) as reader:
            assert reader.cases()==() and 'storage_prediction' not in reader.metadata()
            assert all(d['availability']!='stored' for d in reader.components('stress'))

def test_all_failed_public_input_aborts_without_success_archive(tmp_path):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import ArchiveSink
    from anyfesolver import PreparedStaticError
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        path=tmp_path/'previous.h5';path.write_bytes(b'previous')
        with pytest.raises(PreparedStaticError),ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('failed',{'missing':1.})],sink=sink,model_identity='m',mesh_identity='s')
        assert path.read_bytes()==b'previous' and sink.diagnostics['state']=='aborted'
        import json
        assert json.loads(open(sink.diagnostic_path).read())['input_case_ids']==['failed']

def test_failed_dto_unused_role_has_no_fabricated_stored_fields():
    from anyfesolver.result_capsule import _CapsuleExport
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        exporter=_CapsuleExport(evaluator,[PatternCase('failed',{})],model_identity='m',mesh_identity='s')
        record,_=exporter._process_case(0,{'status':'failed','error':'source refusal','solver_info':{'status':'failed'}})
        final=exporter._finalize()
        assert not record['certified'] and 'storage_prediction' not in final
        assert not exporter.stored_rows
