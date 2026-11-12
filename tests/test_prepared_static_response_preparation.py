"""Bounded library scopes preserve source probes and caller authority boundaries."""
import numpy as np
import pytest
from anyfesolver import (BoundaryCondition, FixedSupport, LoadCase, RecoveryConfig,
                       PreparedStaticCapabilityError, SolveCancelled,
                       prepare_static_response_evaluator)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.prepared_static_recovery import _OWNED_RECOVERY_SCOPE
from test_prepared_static_evaluator import _shell_model, _shell_case


def test_library_tiles_preserve_order_size_and_original_safe_point_positions():
    from anyfesolver.prepared_static_response import _library_tiles, _LIBRARY_SAFE_POINT_STRIDE
    entries={index:None for index in range(399,199,-1)}
    tiles=list(_library_tiles(entries))
    assert [start for start,_ in tiles]==[0,64,128,192]
    assert [len(tile) for _,tile in tiles]==[64,64,64,8]
    assert [eid for _,tile in tiles for eid,_ in tile]==list(entries)
    assert [start for start,_ in tiles if start % _LIBRARY_SAFE_POINT_STRIDE==0]==[0,128]
    assert list(_library_tiles({}))==[]


def test_preparation_timing_counts_native_probes_columns_and_snapshot_phases():
    with prepare_static_response_evaluator(_shell_model(),{'p':_shell_case(1,pressure=2.).load_case},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            stress_execution='certified_library') as evaluator:
        result=evaluator.diagnostics()
        timing=result['source_response']['preparation_timings']
        library=timing['element_library']
        assert library['schema_elements']==2 and library['probe_evaluations']==24
        assert library['column_evaluations']==4
        assert library['schema_scope_count']==1 and library['column_scope_count']==1
        assert library['scope_recipe_max']==2
        assert result['snapshot']['phase_timings_seconds']
        assert all(np.isfinite(value) and value>=0 for key,value in timing.items() if key.endswith('_seconds'))
        assert _OWNED_RECOVERY_SCOPE.get() is None
        timing['element_library']['schema_elements']=-1
        assert evaluator.diagnostics()['source_response']['preparation_timings']['element_library']['schema_elements']==2
        peak=evaluator.diagnostics()['source_response']['preparation_memory_peak_required_bytes']
        assert peak>0
        # Unlimited per-case checks retain their original fast path; preparation
        # telemetry does not turn into per-field snapshot copies or a live limit.
        evaluator._check_library_memory(10**12)
        assert evaluator.diagnostics()['source_response']['preparation_memory_peak_required_bytes']==peak


def test_memory_base_is_cached_with_factor_state_aware_policy(monkeypatch):
    from anyfesolver import ResourceConfig
    from anyfesolver.recovery import ResourcePolicyError
    with prepare_static_response_evaluator(_shell_model(),{'p':_shell_case(1,pressure=2.).load_case},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            stress_execution='certified_library') as evaluator:
        snapshot_type=type(evaluator._snapshot)
        original=snapshot_type.nbytes
        calls=[]
        def observed(snapshot):
            calls.append(snapshot)
            return original(snapshot)
        monkeypatch.setattr(snapshot_type,'nbytes',observed)
        evaluator._preparation_active=True
        peak=evaluator._preparation_memory_peak_required_bytes
        misses=evaluator._memory_base_cache_misses
        for _ in range(8): evaluator._check_library_memory(0)
        assert len(calls)==8 and evaluator._memory_base_cache_misses==misses
        handle=evaluator._operator_handle
        evaluator._operator_handle=None
        evaluator._check_library_memory(0)
        evaluator._operator_handle=handle
        evaluator._check_library_memory(0)
        assert len(calls)==10 and evaluator._memory_base_cache_misses==misses+2
        assert evaluator._memory_base_cache[3]>=original(evaluator._snapshot)
        assert evaluator._memory_base_cache[0][0] is evaluator._snapshot
        assert evaluator._memory_base_cache[0][1] is evaluator._operator_matrix
        evaluator._resource_config=ResourceConfig(memory_limit_bytes=1)
        with pytest.raises(ResourcePolicyError): evaluator._check_library_memory(0)
        assert len(calls)==11
        evaluator._preparation_active=False
        assert evaluator._preparation_memory_peak_required_bytes==peak


def test_memory_cache_accounts_operator_bytes_even_for_same_matrix_owner():
    from anyfesolver import ResourceConfig
    from anyfesolver.recovery import ResourcePolicyError
    with prepare_static_response_evaluator(_shell_model(),{'p':_shell_case(1,pressure=2.).load_case},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            stress_execution='certified_library') as evaluator:
        matrix=evaluator._operator_matrix
        original_data=matrix.data
        base=evaluator._memory_base_cache[3]
        misses=evaluator._memory_base_cache_misses
        evaluator._resource_config=ResourceConfig(memory_limit_bytes=base+1)
        # A private storage-policy probe only: never solve/publish this changed
        # CSR payload. Owner identity is unchanged while its storage grows.
        matrix.data=np.concatenate((original_data,np.zeros(16)))
        try:
            with pytest.raises(ResourcePolicyError) as caught:
                evaluator._check_library_memory(0)
            assert caught.value.memory_estimate.total_bytes_estimate>=base+128
            assert evaluator._memory_base_cache_misses==misses+1
            assert evaluator._memory_base_cache[0][1] is matrix
        finally:
            matrix.data=original_data


def test_memory_cache_accounts_snapshot_csr_growth_for_same_snapshot_owner():
    from anyfesolver import ResourceConfig
    from anyfesolver.recovery import ResourcePolicyError
    with prepare_static_response_evaluator(_shell_model(),{'p':_shell_case(1,pressure=2.).load_case},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            stress_execution='certified_library') as evaluator:
        snapshot=evaluator._snapshot
        matrix=snapshot.stiffness
        assert matrix is not evaluator._operator_matrix
        original_data=matrix.data
        base=evaluator._memory_base_cache[3]
        misses=evaluator._memory_base_cache_misses
        evaluator._resource_config=ResourceConfig(memory_limit_bytes=base+1)
        # Storage-policy probe only; no solve uses this altered CSR payload.
        matrix.data=np.concatenate((original_data,np.zeros(16)))
        try:
            with pytest.raises(ResourcePolicyError) as caught:
                evaluator._check_library_memory(0)
            assert caught.value.memory_estimate.total_bytes_estimate>=base+128
            assert evaluator._memory_base_cache_misses==misses+1
            assert evaluator._memory_base_cache[0][0] is snapshot
            assert evaluator._memory_base_cache[4]==snapshot.nbytes()
        finally:
            matrix.data=original_data


def _independent_beams(count):
    model=FEModel('caller-boundary-development');model.add_material('m',100.,.3,density=1.)
    roots=[];tips=[]
    for eid in range(1,count+1):
        a,b=2*eid-1,2*eid;roots.append(a);tips.append(b)
        model.add_node(a,0.,float(eid),0.);model.add_node(b,1.,float(eid),0.)
        model.add_element(eid,BeamElement(eid,[a,b],'m',{'area':1.,'Iy':.001,'Iz':.001,'J':.001}))
    model.add_boundary_condition(FixedSupport('bases',roots))
    model.add_boundary_condition(BoundaryCondition('axial',tips,{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
    load=LoadCase('p')
    for tip in tips: load.add_nodal_load(tip,[1.,0.,0.,0.,0.,0.])
    return model,load


@pytest.mark.parametrize('action',['cancel','source_helper','clock_alias'])
def test_actual_second_caller_boundary_outside_scope_refuses_without_success(monkeypatch,action):
    import anyfesolver.prepared_static_response as response
    model,load=_independent_beams(129)
    class CallerToken:
        schema_observations=0
        def raise_if_cancelled(self,stage):
            assert _OWNED_RECOVERY_SCOPE.get() is None
            if stage != 'prepared_response.element_schema': return
            self.schema_observations+=1
            if self.schema_observations!=2:return
            if action=='cancel': raise SolveCancelled('second original safe point',stage=stage)
            def foreign(*args,**kwargs): pytest.fail('caller replacement must never execute')
            monkeypatch.setattr(response,'_verify_library_element' if action=='source_helper' else '_preparation_clock_ns',foreign)
    token=CallerToken()
    with pytest.raises(SolveCancelled if action=='cancel' else PreparedStaticCapabilityError):
        prepare_static_response_evaluator(model,{'p':load},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            stress_execution='certified_library',cancellation_token=token)
    assert token.schema_observations==2
    assert _OWNED_RECOVERY_SCOPE.get() is None
