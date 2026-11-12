"""Source staging copy/resource contracts; synthetic arrays are not certification."""
import numpy as np
import pytest
from anyfesolver import ResourceConfig
from anyfesolver.recovery import MemoryEstimate,ResourcePolicyError
from anyfesolver.prepared_static_recovery import (_verification_storage_bytes,
    _freeze_verification_value,_check_verification_memory)


def test_staged_native_cache_records_keep_source_precision_bits_and_no_aliases():
    values=np.asfortranarray(np.array([[-0.,1.25],[3.,4.]],dtype=np.float32))
    native={'values':values,'label':'native','constant':[1.,(2.,3.)]}
    first=_freeze_verification_value(native)
    before=values.tobytes()
    values[:]=9.
    second=_freeze_verification_value(native)
    values[:]=17.
    native['constant'][0]=-3.
    assert first['values'].dtype==np.float32 and first['values'].tobytes()==before
    assert np.all(second['values']==9.) and first['constant']==(1.,(2.,3.))
    with pytest.raises(ValueError):first['values'].setflags(write=True)
    with pytest.raises(TypeError):first['label']='changed'
    assert _verification_storage_bytes({'values':values})>=2*values.nbytes


def test_pending_copy_budget_refuses_before_copy_under_existing_policy():
    base=MemoryEstimate(12,2,1,16,128,0,0,0,0,1000)
    values={'field':np.zeros((7,3))}
    incoming=_verification_storage_bytes(values)
    timing={'pending_copy_peak_bytes':0}
    with pytest.raises(ResourcePolicyError) as caught:
        _check_verification_memory(base,2*incoming,ResourceConfig(memory_limit_bytes=1000+2*incoming-1),timing)
    assert caught.value.context=='prepared_static.recovery_verification_staging'
    assert caught.value.memory_estimate.total_bytes_estimate==1000+2*incoming
    assert timing['pending_copy_peak_bytes']==2*incoming
    _check_verification_memory(base,2*incoming,None,timing)


def test_staging_accepts_original_mapping_and_float_array_type_families():
    from collections import OrderedDict,UserDict
    from collections.abc import Mapping
    from anyfesolver.prepared_static_recovery import _array_value,_plain_constant
    class NativeArray(np.ndarray): pass
    values=np.asfortranarray(np.array([[-0.,1.25],[3.,4.]],dtype=np.float32)).view(NativeArray)
    native=OrderedDict(field=values,constant=UserDict(label='native',axis=(1.,2.)))
    assert _array_value(values) is not None
    assert _plain_constant(native['constant'])=={'label':'native','axis':(1.,2.)}
    estimate=_verification_storage_bytes(native)
    frozen=_freeze_verification_value(native)
    assert isinstance(frozen,Mapping) and type(frozen['field']) is np.ndarray
    assert frozen['field'].dtype==values.dtype and frozen['field'].shape==values.shape
    assert frozen['field'].tobytes()==values.tobytes() and estimate>=values.nbytes*2
    values[:]=19.
    native['constant']['label']='changed'
    assert frozen['constant']['label']=='native'
    assert frozen['field'][0,1]==1.25
    with pytest.raises(ValueError): frozen['field'].setflags(write=True)
    with pytest.raises(TypeError): frozen['constant']['label']='changed'


def test_midloop_scope_exit_authority_failure_clears_queue_and_is_global(monkeypatch):
    import sys,threading,time
    from anyfesolver import RecoveryConfig,PreparedStaticCapabilityError,prepare_static_response_evaluator
    from anyfesolver.elements import BeamElement
    import anyfesolver.prepared_static_recovery as recovery
    from test_prepared_static_response_preparation import _independent_beams
    model,load=_independent_beams(64)
    main_id=threading.get_ident()
    state={}
    threads=[]
    method=BeamElement.compute_stresses
    original_code=method.__code__
    def mutate_during_real_span():
        deadline=time.monotonic()+3.
        while time.monotonic()<deadline:
            frame=sys._current_frames().get(main_id)
            while frame is not None:
                if (frame.f_code.co_name=='_verify_staged_recipes' and 'entered' in frame.f_locals
                        and frame.f_locals['timing']['owner_verification_calls']>0):
                    state['pending']=frame.f_locals['pending']
                    state['compiled']=frame.f_locals['compiled']
                    # Same numerical body, changed executable identity. It must
                    # be rejected at the real span exit before publication.
                    method.__code__=original_code.replace(co_name='changed_native_stress_identity')
                    state['changed']=True
                    return
                frame=frame.f_back
            time.sleep(.0005)
        state['missed']=True
    class CallerRecovery(RecoveryConfig):
        observations=0
        def selected_components(self):
            assert recovery._OWNED_RECOVERY_SCOPE.get() is None
            type(self).observations+=1
            if type(self).observations==64*16:
                worker=threading.Thread(target=mutate_during_real_span)
                threads.append(worker);worker.start()
            return super().selected_components()
    try:
        with pytest.raises(PreparedStaticCapabilityError) as caught:
            prepare_static_response_evaluator(model,{'p':load},
                recovery_config=CallerRecovery(include_stresses=True,include_reactions=False),
                stress_execution='certified_library')
        assert state.get('changed') and not state.get('missed')
        assert 'refused' not in caught.value.diagnostics
        assert not state['pending'] and not state['compiled']
        assert recovery._OWNED_RECOVERY_SCOPE.get() is None
    finally:
        for worker in threads: worker.join(timeout=4.)
        method.__code__=original_code
        assert not any(worker.is_alive() for worker in threads)


@pytest.mark.parametrize('action',['observe','cancel','source_helper','checker_code'])
def test_public_staging_keeps_native_config_observations_outside_scope(monkeypatch,action):
    from anyfesolver import (RecoveryConfig,PreparedStaticCapabilityError,SolveCancelled,
                           prepare_static_response_evaluator)
    import anyfesolver.prepared_static_recovery as recovery
    from test_prepared_static_response_preparation import _independent_beams
    model,load=_independent_beams(2)
    observations=[]
    class CallerRecovery(RecoveryConfig):
        def selected_components(self):
            assert recovery._OWNED_RECOVERY_SCOPE.get() is None
            observations.append(len(observations))
            if len(observations)==20:
                if action=='cancel': raise SolveCancelled('native observation',stage='recovery compilation')
                if action=='source_helper':
                    def foreign(*args,**kwargs): pytest.fail('changed owned helper must not execute')
                    monkeypatch.setattr(recovery,'_recover_owned_local',foreign)
                if action=='checker_code':
                    checker=recovery._probe_element_stress.__kwdefaults__['_publication_authority'][0]
                    monkeypatch.setattr(checker,'__code__',checker.__code__.replace(co_name='changed_publication_checker'))
            return super().selected_components()
    kwargs=dict(recovery_config=CallerRecovery(include_stresses=True,include_reactions=False),
                stress_execution='certified_library')
    if action=='observe':
        with prepare_static_response_evaluator(model,{'p':load},**kwargs) as evaluator:
            phases=evaluator.diagnostics()['snapshot']['phase_timings_seconds']
            assert phases['affine_recovery.owner_verification_calls']==4
            assert phases['affine_recovery.verification_scope_count']==1
    else:
        with pytest.raises(SolveCancelled if action=='cancel' else PreparedStaticCapabilityError):
            prepare_static_response_evaluator(model,{'p':load},**kwargs)
    assert len(observations)>=20
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


def test_staged_comparison_keeps_two_defects_without_promising_refusal_order():
    from anyfesolver import RecoveryConfig,prepare_static_response_evaluator
    import anyfesolver.prepared_static_recovery as recovery
    from test_prepared_static_response_preparation import _independent_beams
    model,load=_independent_beams(2)
    with prepare_static_response_evaluator(model,{'p':load},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            stress_execution='certified_library') as evaluator:
        candidate=evaluator._snapshot.recovery_compilation.elements[1]
        local=np.ones(len(candidate.dof_mapping))*1e-8
        oracle=recovery._recover_owned_local(candidate.local_recipe,local)
        component='axial_stress'
        oracle[component]=np.asarray(oracle[component])+1e9
        refused=[dict(element_id=2,reason='an earlier native-admission defect')]
        compiled={}
        timing=dict(verification_scope_entry_ns=0,verification_scope_exit_ns=0,
                    verification_scope_count=0,verification_recipe_max=0,
                    owner_verification_ns=0,owner_verification_calls=0,admitted_elements=0)
        recovery._verify_staged_recipes([(candidate,((1e-8,
            recovery._freeze_verification_value(local),recovery._freeze_verification_value(oracle)),),0)],
            compiled,refused,timing)
        assert {item['element_id'] for item in refused}=={1,2}
        assert not compiled and timing['admitted_elements']==0
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


def test_public_resource_config_reaches_pre_copy_staging_refusal():
    from anyfesolver import RecoveryConfig,prepare_static_response_evaluator
    from anyfesolver.recovery import estimate_model_memory
    from test_prepared_static_response_preparation import _independent_beams
    model,load=_independent_beams(2)
    recovery=RecoveryConfig(include_stresses=True,include_reactions=False)
    base=estimate_model_memory(model,num_rhs=0,recovery_config=recovery)
    with pytest.raises(ResourcePolicyError) as caught:
        prepare_static_response_evaluator(model,{'p':load},recovery_config=recovery,
            resource_config=ResourceConfig(memory_limit_bytes=base.total_bytes_estimate),
            stress_execution='certified_library')
    assert caught.value.context=='prepared_static.recovery_verification_staging'
    assert caught.value.memory_estimate.total_bytes_estimate>base.total_bytes_estimate
