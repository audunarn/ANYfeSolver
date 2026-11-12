"""Opt-in native rows preserve failed eligibility and original source values."""
from importlib.util import find_spec
import numpy as np
import pytest
from anyfesolver import (BoundaryCondition,FixedSupport,LoadCase,PatternCase,RecoveryConfig,ResourceConfig,
    PreparedStaticCapabilityError,prepare_static_response_evaluator,export_recovery_capsule,export_recovery_capsule_to_sink)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement
from anyfesolver.prepared_static_numeric import NumericCase
from test_result_capsule_numeric import assert_identical
from test_result_capsule_replay import capsule_component

def original_fixture(mixed=True):
    result=FEModel('native-affinity-original-4808');result.add_material('steel',210.e9,.3,density=7850.)
    eids=(1,4808) if mixed else (4808,)
    for eid in eids:
        index=eid-1;a,b=2*eid-1,2*eid;y,z=float(index%400)*1.5,float(index//400)*1.5
        result.add_node(a,0.,y,z);result.add_node(b,1.,y,z)
        result.add_element(eid,BeamElement(eid,[a,b],'steel',{'area':.01,'Iy':1.e-5,'Iz':1.e-5,'J':1.e-5}))
    roots=[2*eid-1 for eid in eids];tips=[2*eid for eid in eids]
    result.add_boundary_condition(FixedSupport('bases',roots));result.add_boundary_condition(BoundaryCondition('axial',tips,{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
    baseline=LoadCase('baseline');patterns={f'p{i}':LoadCase(f'p{i}') for i in range(3)}
    for eid in eids:
        index=eid-1;baseline.add_nodal_load(2*eid,[125.*(1.+index%31/31.),0.,0.,0.,0.,0.])
        for p,load in enumerate(patterns.values()):load.add_nodal_load(2*eid,[float(1000.*np.sin((index+1)*(p+1)*.017+(p+1)*.31)),0.,0.,0.,0.,0.])
    return result,baseline,patterns

def prepare(mixed=True,**kwargs):
    model,baseline,patterns=original_fixture(mixed)
    return prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,
        stress_execution='certified_library',recovery_config=RecoveryConfig(include_stresses=True,include_reactions=True),**kwargs)

def test_default_original_source_refusal_is_unchanged():
    with pytest.raises(PreparedStaticCapabilityError) as caught:prepare()
    assert caught.value.diagnostics['element_id']==4808
    assert caught.value.diagnostics['component']=='axial_stress'
    assert caught.value.diagnostics['check']=='rigid_offset'
    assert caught.value.diagnostics['relative_residual']==6.355546660296293e-9

@pytest.mark.parametrize('mixed',[False,True])
def test_complete_native_rows_and_original_record_bits(mixed):
    cases=[PatternCase('baseline',{}),PatternCase('dense',{'p0':1.7,'p1':-2.3,'p2':.31}),PatternCase('reversed',{'p2':-1.2,'p0':.7})]
    with prepare(mixed,numerical_affinity_fallback=True) as evaluator:
        diagnostics=evaluator.diagnostics()['source_response'];n=2 if mixed else 1
        assert diagnostics['native_affinity_fallback_elements']==[4808]
        assert diagnostics['element_library_elements']==n-1
        assert diagnostics['element_library_probes']==12*n
        assert diagnostics['preparation_timings']['element_library']['column_evaluations']==4*n
        assert 4808 not in evaluator._element_library
        assert diagnostics['element_library_worst_verification_residual']<=1.e-9
        records=next(evaluator.iter_batches(cases,batch_size=3)).case_results
        before=evaluator.diagnostics()['source_response']
        actual=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',batch_size=3)
        after=evaluator.diagnostics()['source_response']
        expected=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',batch_size=3,recovery_execution='records')
        assert_identical(actual,expected)
        assert after['native_affinity_element_cases']-before['native_affinity_element_cases']==3
        assert after.get('numeric_element_cases',0)-before.get('numeric_element_cases',0)==(3 if mixed else 0)
        assert after['reconstructed_element_records']-before['reconstructed_element_records']==(3 if mixed else 0)
        assert all(case['certified'] and case['status']=='converged' for case in actual['cases'])
        assert all(case['portable']==mixed for case in actual['cases'])
        assert all(d['key']!='hill_utilization' for d in actual['descriptors'])
        numeric=next(evaluator._iter_capsule_batches(cases,batch_size=3)).case_results
        assert all(isinstance(case,NumericCase) for case in numeric)
        primitives=('axial_stress','bending_stress_y','bending_stress_z','shear_stress_y','shear_stress_z','torsional_stress')
        for field in (*primitives,'von_mises','equivalent_stress'):
            assert field not in actual['responses'] and field not in actual['derived']
            descriptor=next(d for d in actual['descriptors'] if d['key']==field)
            assert descriptor['availability']=='stored' and descriptor['reconstruction']=='explicit_per_case_values'
            assert descriptor['stored_case_count']==3
            assert descriptor['metadata']['native_affinity_fallback_elements']==[4808]
            ids=actual['associations'][field]
            assert ids.tolist()==([1,4808] if mixed else [4808])
            for index,record in enumerate(records):
                row=capsule_component(actual,index,field)
                original=np.array([record['element_stresses'][int(eid)][field] for eid in ids],dtype=np.float64)
                assert row.tobytes()==original.tobytes()
        assert actual['model_provenance']['native_affinity_fallback']['refusals']['4808'][0]['check']=='rigid_offset'

@pytest.mark.parametrize('value',[1,None,np.bool_(True)])
def test_exact_bool_opt_in(value):
    with pytest.raises(PreparedStaticCapabilityError,match='exact bool'):prepare(numerical_affinity_fallback=value)

def test_opt_in_is_restricted_to_library():
    model,baseline,patterns=original_fixture(False)
    with pytest.raises(PreparedStaticCapabilityError,match='requires certified_library'):
        prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,numerical_affinity_fallback=True)


@pytest.mark.parametrize('failure',['type','shape','constant','nonfinite','derived'])
def test_opt_in_probe_collector_never_admits_other_failures(monkeypatch,failure):
    # This private classifier test uses controlled records only to discriminate
    # refusal branches; it supplies no source evaluator/certification authority.
    import anyfesolver.prepared_static_response as response
    primitives=('axial_stress','bending_stress_y','bending_stress_z','shear_stress_y','shear_stress_z','torsional_stress')
    zero={name:np.float64(0.) for name in primitives};zero.update(von_mises=np.float64(0.),tag='fixed')
    schema=response._library_schema(7,zero)
    def record(recipe,local):
        result=dict(zero)
        if failure=='type':result['axial_stress']=np.array([0],dtype=np.int64)
        elif failure=='shape':result['axial_stress']=np.zeros(2)
        elif failure=='constant':result['tag']='changed'
        elif failure=='nonfinite':result['axial_stress']=np.float64(np.inf)
        elif failure=='derived':result['von_mises']=np.float64(1.)
        return result
    monkeypatch.setattr(response,'_recover_owned_local',record);reasons=[]
    with np.errstate(invalid='ignore'),pytest.raises(PreparedStaticCapabilityError):
        response._verify_library_element(7,None,12,zero,schema,numerical_failures=reasons)
    assert reasons==[]


def test_native_map_and_refusal_records_are_immutable_and_bound():
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        with pytest.raises(TypeError):evaluator._native_affinity_fallbacks[1]=None
        with pytest.raises(TypeError):evaluator._native_affinity_fallbacks[4808][5][0]['component']='changed'
        evaluator._native_affinity_fallbacks={}
        with pytest.raises(PreparedStaticCapabilityError,match='recipe binding changed'):
            list(evaluator.iter_batches([PatternCase('a',{})]))


@pytest.mark.parametrize('callback',['case','stored'])
@pytest.mark.parametrize('mutation',['option','map','helper'])
def test_callback_mutation_refuses_normal_archive_publication(tmp_path,monkeypatch,callback,mutation):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    import anyfesolver.prepared_static_response as response
    path=tmp_path/'existing.h5';previous=b'previous destination preserved';path.write_bytes(previous)
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        class CallerSink:
            def __init__(self,native):self.native=native;self.changed=False
            def capsule_begin(self,header):return self.native.capsule_begin(header)
            def mutate(self):
                if self.changed:return
                self.changed=True
                if mutation=='option':evaluator._numerical_affinity_fallback=False
                elif mutation=='map':evaluator._native_affinity_fallbacks={}
                else:monkeypatch.setattr(response,'_native_affinity_record',lambda *args:pytest.fail('foreign helper must not run'))
            def capsule_case(self,record,coefficients):
                self.native.capsule_case(record,coefficients)
                if callback=='case':self.mutate()
            def capsule_stored(self,key,indices,values):
                self.native.capsule_stored(key,indices,values)
                if callback=='stored':self.mutate()
            def capsule_end(self,footer):pytest.fail('mutated source must not reach final publication')
            def capsule_failed(self,record):return self.native.capsule_failed(record)
        with pytest.raises(PreparedStaticCapabilityError),anypost.ArchiveSink(path) as native:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=CallerSink(native),model_identity='m',mesh_identity='s')
        assert path.read_bytes()==previous
        assert native.diagnostics['state']=='aborted'


@pytest.mark.parametrize('execution',['auto','records'])
def test_native_cancel_is_outside_owned_scope_and_no_valid_archive(tmp_path,execution):
    from anyfesolver import SolveCancelled
    from anyfesolver.prepared_static_recovery import _OWNED_RECOVERY_SCOPE
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    class Token:
        armed=False
        def raise_if_cancelled(self,stage):
            assert _OWNED_RECOVERY_SCOPE.get() is None
            if self.armed and stage!='recovery_capsule.before_publication':raise SolveCancelled('native case',stage=stage)
    token=Token();path=tmp_path/'cancel.h5'
    with prepare(False,numerical_affinity_fallback=True,cancellation_token=token) as evaluator:
        token.armed=True
        with pytest.raises(SolveCancelled),anypost.ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=sink,model_identity='m',mesh_identity='s',cancellation_token=token,recovery_execution=execution)
        assert not path.exists() and sink.diagnostics['state']=='aborted'
        assert evaluator.diagnostics()['cases_certified']==0


@pytest.mark.parametrize('execution',['auto','records'])
@pytest.mark.parametrize('at',['pre','case','stored'])
def test_public_cancel_never_replaces_previous_destination(tmp_path,execution,at):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    from anyfesolver import CancellationToken,SolveCancelled
    token=CancellationToken();path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        native=anypost.ArchiveSink(path)
        class Sink:
            capsule_begin=native.capsule_begin
            capsule_end=native.capsule_end
            capsule_failed=native.capsule_failed
            def capsule_case(self,*args):
                native.capsule_case(*args)
                if at=='case':token.cancel('last case')
            def capsule_stored(self,*args):
                native.capsule_stored(*args)
                if at=='stored':token.cancel('last stored')
        if at=='pre':token.cancel('before export')
        with pytest.raises(SolveCancelled) as caught,native:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=Sink(),model_identity='m',mesh_identity='s',recovery_execution=execution,cancellation_token=token)
        assert path.read_bytes()==b'previous' and native.diagnostics['state']=='aborted'
        import json
        diagnostic_paths=list(tmp_path.glob('previous.h5.*.failure.json'))
        assert len(diagnostic_paths)==1,(native.diagnostic_error,getattr(caught.value,'__notes__',()))
        failure=json.loads(diagnostic_paths[0].read_text())
        assert failure['status']=='cancelled' and failure['sink']['hdf_published'] is False


@pytest.mark.parametrize('execution',['auto','records'])
def test_memory_export_observes_its_token_not_preparation_token(execution):
    from anyfesolver import CancellationToken,SolveCancelled
    prep=CancellationToken();export=CancellationToken()
    with prepare(False,numerical_affinity_fallback=True,cancellation_token=prep) as evaluator:
        prep.cancel('preparation token is not retained')
        actual=export_recovery_capsule(evaluator,[PatternCase('a',{})],model_identity='m',mesh_identity='s',recovery_execution=execution)
        assert actual['cases'][0]['certified']
        export.cancel('explicit export token')
        with pytest.raises(SolveCancelled):
            export_recovery_capsule(evaluator,[PatternCase('b',{})],model_identity='m',mesh_identity='s',recovery_execution=execution,cancellation_token=export)


def test_export_token_cannot_mutate_response_authority(tmp_path):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    from anyfesolver.prepared_static_recovery import _OWNED_RECOVERY_SCOPE
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        class Token:
            def raise_if_cancelled(self,stage):
                assert _OWNED_RECOVERY_SCOPE.get() is None
                evaluator._numerical_affinity_fallback=False
        path=tmp_path/'mutation.h5'
        with pytest.raises(PreparedStaticCapabilityError,match='option binding changed'),anypost.ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=sink,model_identity='m',mesh_identity='s',cancellation_token=Token())
        assert not path.exists()


def test_actual_unrelated_shell_basis_survives_beam_native_namespace():
    from test_result_capsule_export import _rotated_shell_model
    from test_prepared_static_evaluator import _shell_case
    model=_rotated_shell_model();model.add_material('beamsteel',210.e9,.3,density=7850.)
    model.add_node(9615,0.,10.5,18.);model.add_node(9616,1.,10.5,18.)
    model.add_element(4808,BeamElement(4808,[9615,9616],'beamsteel',{'area':.01,'Iy':1.e-5,'Iz':1.e-5,'J':1.e-5}))
    model.add_boundary_condition(FixedSupport('beam-base',[9615]));model.add_boundary_condition(BoundaryCondition('beam-axial',[9616],{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
    baseline=_shell_case(0,pressure=.5).load_case;baseline.add_nodal_load(9616,[125.,0.,0.,0.,0.,0.])
    load=_shell_case(1,pressure=2.).load_case;load.add_nodal_load(9616,[700.,0.,0.,0.,0.,0.])
    with prepare_static_response_evaluator(model,{'p':load},baseline_load_case=baseline,stress_execution='certified_library',
            numerical_affinity_fallback=True,recovery_config=RecoveryConfig(include_stresses=True,include_reactions=True)) as evaluator:
        cases=[PatternCase('a',{}),PatternCase('b',{'p':1.7})]
        expected=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',recovery_execution='records')
        actual=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s')
        assert_identical(actual,expected)
        assert all(case['portable'] for case in actual['cases'])
        assert 'membrane_xx' in actual['responses'] and 'von_mises:shell' in actual['derived']
        assert 'axial_stress' not in actual['responses'] and 'von_mises:beam' not in actual['derived']
        assert 'von_mises:beam' in actual['stored'] and actual['associations']['von_mises:beam'].tolist()==[4808]
        assert 'equivalent_stress:beam' in actual['stored']
        assert actual['associations']['equivalent_stress:beam'].tolist()==[4808]
        assert 'equivalent_stress:shell' in actual['derived']


def test_native_metadata_and_numeric_workspace_obey_resource_policy(tmp_path):
    from anyfesolver.recovery import ResourcePolicyError
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    with pytest.raises(ResourcePolicyError):prepare(False,numerical_affinity_fallback=True,resource_config=ResourceConfig(memory_limit_bytes=1))
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        evaluator._resource_config=ResourceConfig(memory_limit_bytes=1)
        path=tmp_path/'resource.h5'
        with pytest.raises(ResourcePolicyError),anypost.ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=sink,model_identity='m',mesh_identity='s')
        assert not path.exists() and evaluator._numeric_export_plan is None


def test_source_principal_refusal_and_public_extreme_certification_refusal(tmp_path):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost,json
    from anyfesolver.prepared_static_numeric import source_fields
    from anyfesolver.prepared_static_loads import PreparedStaticError
    from test_result_capsule_export import _rotated_shell_model
    from test_prepared_static_evaluator import _shell_case
    names=('membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz')
    linear=tuple((name,(1,),True) for name in names)
    values=np.zeros(8);values[2]=1.e308
    with np.errstate(over='ignore',invalid='ignore'),pytest.raises((ValueError,np.linalg.LinAlgError)):
        source_fields(values,linear,('principal_stresses_top',))
    # The actual extreme finite source case reaches the existing certification
    # refusal before recovery. This is not evidence of a certified principal
    # overflow; the literal source measure boundary is exercised above.
    with prepare_static_response_evaluator(_rotated_shell_model(),{'p':_shell_case(1).load_case},
            stress_execution='certified_library',numerical_affinity_fallback=True,
            recovery_config=RecoveryConfig(include_stresses=True,components=('principal_stresses_top',))) as evaluator:
        path=tmp_path/'principal.h5';path.write_bytes(b'previous')
        with np.errstate(over='ignore',invalid='ignore'),pytest.raises(PreparedStaticError) as caught,anypost.ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,[PatternCase('extreme',{'p':1.e303})],sink=sink,model_identity='m',mesh_identity='s')
        assert path.read_bytes()==b'previous' and sink.diagnostics['state']=='aborted'
        failure=json.loads(next(tmp_path.glob('principal.h5.*.failure.json')).read_text())
        assert failure['sink']['hdf_published'] is False
        assert caught.value.diagnostics
        assert caught.value.diagnostics['relative_backward_error_disposition']=='NONFINITE_OR_OVERFLOW'


def test_native_canonical_case_preserves_original_route_and_bits():
    canonical=LoadCase('canonical');canonical.add_nodal_load(9616,[17.,0.,0.,0.,0.,0.])
    cases=[PatternCase('canonical',{'p0':1.},canonical_load_case=canonical),PatternCase('native',{'p2':-1.7})]
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        expected=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',recovery_execution='records',batch_size=1)
        actual=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',batch_size=1)
        assert_identical(actual,expected)
        assert actual['cases'][0]['route'].endswith('canonical_fallback')
        assert not any(case['portable'] for case in actual['cases'])
        assert 'von_mises' in actual['stored'] and 'equivalent_stress' in actual['stored']


@pytest.mark.parametrize('mixed',[False,True])
def test_actual_archive_sink_native_complete_rows_roundtrip(tmp_path,mixed):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    cases=[PatternCase('baseline',{}),PatternCase('native',{'p2':-1.7,'p0':.2})]
    with prepare(mixed,numerical_affinity_fallback=True) as evaluator:
        original=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',recovery_execution='records')
        path=tmp_path/'native.h5'
        with anypost.ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity='m',mesh_identity='s')
        with anypost.open_results(path) as reader:
            for index,case in enumerate(cases):
                for descriptor in original['descriptors']:
                    key=descriptor['key']
                    expected=capsule_component(original,index,key)
                    actual=reader.read_field(case.case_id,descriptor['quantity_id'],descriptor['component'])
                    assert np.asarray(actual).tobytes()==np.asarray(expected).tobytes(),key
                    assert reader.associations(descriptor['quantity_id'],descriptor['component']).tobytes()==original['associations'][key].tobytes()


def test_conditional_equivalent_namespace_is_stored_when_vm_is_stored():
    from anyfesolver.result_capsule import _CapsuleExport
    with prepare(True,numerical_affinity_fallback=True) as evaluator:
        export=_CapsuleExport(evaluator,[PatternCase('a',{})],model_identity='m',mesh_identity='s')
        # Private descriptor discriminator, not admission of a partial-input
        # physical element: simulate VM stored because an input namespace lacks
        # a complete admitted basis, without an explicit VM refusal tag.
        export.basis['native_stored_derived']=set()
        export.basis['derived_elements']={'von_mises':{'beam':[1]},'equivalent_stress':{'beam':[1]}}
        export.layouts={};export.derived={};export._stored_derived_fields={};export.descriptor_info={}
        export._build_derived_components()
        assert export._stored_derived_fields['von_mises']=='von_mises'
        assert export._stored_derived_fields['equivalent_stress']=='equivalent_stress'
        assert not export.derived


def test_native_mapping_detached_and_accounting_is_conservative():
    from anyfesolver.prepared_static_response import _library_storage_bytes
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        native=evaluator._native_affinity_fallbacks[4808]
        mapping=evaluator._snapshot.recovery_compilation.elements[4808].dof_mapping
        assert not np.shares_memory(native[6],mapping)
        assert native[6].tobytes()==mapping.tobytes()
        with pytest.raises(ValueError):native[6].setflags(write=True)
        timing=evaluator.diagnostics()['source_response']['preparation_timings']['element_library']
        assert timing['native_metadata_entries_accounted']==1
        assert timing['native_metadata_conservative_bytes']>=_library_storage_bytes(evaluator._native_affinity_fallbacks)


@pytest.mark.parametrize('detailed_ids',[(1,),(4808,),()])
@pytest.mark.parametrize('enveloped',[False,True])
def test_native_actual_visits_subset_envelope_and_mixed_canonical_batch(detailed_ids,enveloped):
    model,baseline,patterns=original_fixture(True)
    recovery=RecoveryConfig(element_ids=detailed_ids,components=('axial_stress',),include_stresses=True,include_reactions=False)
    envelope=RecoveryConfig(element_ids=(4808,),components=('axial_stress',),include_stresses=True,include_reactions=False) if enveloped else None
    canonical=LoadCase('canonical');canonical.add_nodal_load(9616,[17.,0.,0.,0.,0.,0.])
    cases=[PatternCase('source',{'p0':.3}),PatternCase('canonical',{'p2':-.7},canonical_load_case=canonical)]
    with prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,recovery_config=recovery,
            envelope_recovery_config=envelope,stress_execution='certified_library',numerical_affinity_fallback=True) as evaluator:
        before=evaluator.diagnostics()['source_response'].get('native_affinity_element_cases',0)
        batch=next(evaluator.iter_batches(cases,batch_size=2))
        native=4808 in detailed_ids or enveloped
        assert evaluator.diagnostics()['source_response'].get('native_affinity_element_cases',0)-before==(2 if native else 0)
        assert batch.diagnostics['shared_recovery']['native_affinity_element_cases']==(2 if native else 0)
        for index,case in enumerate(batch.case_results):
            assert set(case['element_stresses'])==set(detailed_ids)
            provenance=case['solver_info']['source_response']
            assert provenance.get('native_affinity_fallback_elements',[])==([4808] if native else [])
            assert provenance['stress_execution']==('canonical_fallback' if index else 'certified_library_native_affinity_fallback' if native else 'certified_library')
        if enveloped:assert evaluator.envelope_result() is not None


def test_zero_refusal_opt_in_preserves_scientific_capsule_fields():
    from test_prepared_static_evaluator import _bar,_case
    cases=[PatternCase('baseline',{}),PatternCase('loaded',{'p':-.7})]
    def run(option):
        with prepare_static_response_evaluator(_bar(),{'p':_case(1,.7).load_case},baseline_load_case=_case(0,.5).load_case,
                stress_execution='certified_library',numerical_affinity_fallback=option,
                recovery_config=RecoveryConfig(include_stresses=True,include_reactions=True)) as evaluator:
            assert evaluator.diagnostics()['source_response']['native_affinity_fallback_elements']==[]
            assert evaluator.diagnostics()['source_response']['numerical_affinity_fallback'] is option
            return export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s')
    strict,native=run(False),run(True)
    for key in ('responses','stored','associations','coefficients','derived','descriptors','cases'):
        assert_identical(native[key],strict[key])


@pytest.mark.parametrize('selected',[(1,4808),(4808,)])
def test_public_export_mixed_batch_two_auto_records_and_actual_native_work(selected):
    model,baseline,patterns=original_fixture(True)
    recovery=RecoveryConfig(element_ids=selected,components=('axial_stress',),include_stresses=True,include_reactions=True)
    canonical=LoadCase('canonical');canonical.add_nodal_load(9616,[17.,0.,0.,0.,0.,0.])
    cases=[PatternCase('source',{'p0':.3}),PatternCase('canonical',{'p2':-.7},canonical_load_case=canonical)]
    with prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,recovery_config=recovery,
            stress_execution='certified_library',numerical_affinity_fallback=True) as evaluator:
        exports=[]
        for execution in ('auto','records'):
            before=evaluator.diagnostics()['source_response']
            exports.append(export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s',recovery_execution=execution,batch_size=2))
            after=evaluator.diagnostics()['source_response']
            assert after.get('native_affinity_element_cases',0)-before.get('native_affinity_element_cases',0)==2
            assert after.get('numeric_element_cases',0)-before.get('numeric_element_cases',0)==0
            assert after['reconstructed_element_records']-before['reconstructed_element_records']==(1 if 1 in selected else 0)
        assert_identical(exports[0],exports[1])
        actual=exports[0]
        assert actual['cases'][0]['route'].endswith('certified_library_native_affinity_fallback')
        assert actual['cases'][1]['route'].endswith('canonical_fallback')
        assert all(case['certified'] for case in actual['cases'])
        assert actual['associations']['axial_stress'].tolist()==list(selected)
        assert set(actual['stored']['axial_stress']['case_indices'])=={0,1}
        for case in actual['cases']:
            assert set(case['unavailable_components'])>={'von_mises','equivalent_stress'}


def test_missing_native_selected_field_chains_typed_helper_refusal():
    from anyfesolver.prepared_static_numeric import prepare_numeric_plan,recover_numeric_cases
    from anyfesolver.prepared_static_response import _native_affinity_record
    from anyfesolver.prepared_static_loads import PreparedStaticError
    with prepare(False,numerical_affinity_fallback=True) as evaluator:
        snapshot=evaluator._snapshot
        plan=prepare_numeric_plan(snapshot,evaluator._element_library,evaluator._recovery,evaluator._native_affinity_fallbacks)
        groups,layouts,output_bytes,retained_bytes=plan
        layouts=dict(layouts);layouts['missing_test_field']=((4808,),((),),np.array([0,1]),(False,))
        full=np.zeros((snapshot.stiffness.shape[0],1))
        # Deliberately malformed private layout; no public admission/certification,
        # evaluator subclass, executable monkeypatch, or archive publication.
        with pytest.raises(PreparedStaticError,match='selected field is missing') as caught:
            recover_numeric_cases(snapshot,full,full,evaluator._recovery,[('unit',None,0)],[{}],{},[None],evaluator._combine,
                (groups,layouts,output_bytes+8,retained_bytes),lambda **kwargs:None,evaluator._native_affinity_fallbacks,_native_affinity_record)
        assert isinstance(caught.value.__cause__,KeyError)
        assert caught.value.diagnostics['component']=='missing_test_field'
