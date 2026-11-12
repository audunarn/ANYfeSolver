"""Versioned replay, exact source corrections and legacy compatibility."""
import copy
import numpy as np
import pytest
from anyfesolver_results import (reconstruct_response,prepare_response,reconstruct_prepared_response,
                              validate_capsule,derive_field,ReconstructionError,CapsuleValidationError)
ORDERED='ordered_float64/1'


@pytest.mark.parametrize('order',['C','F'])
@pytest.mark.parametrize('active',[None,[20,1,7],[],[5]])
def test_ordered_replay_bitwise_row_tiles_prepared_and_duplicate_selectors(order,active):
    rng=np.random.default_rng(20261012)
    basis=np.array(rng.normal(size=(1307,21)),order=order)
    indices=list(range(1,21)) if active is None else active
    weights=rng.uniform(-3,3,len(indices))
    expected=reconstruct_response(basis,weights,active_indices=indices,combination=ORDERED)
    prepared=prepare_response(basis,active_indices=indices,combination=ORDERED)
    assert prepared.combination==ORDERED
    assert reconstruct_prepared_response(prepared,weights).tobytes()==expected.tobytes()
    for size in (1,3,17,64,407):
        actual=np.concatenate([reconstruct_response(basis[start:start+size],weights,
            active_indices=indices,combination=ORDERED) for start in range(0,len(basis),size)])
        assert actual.tobytes()==expected.tobytes()
    rows=np.array([1306,2,2,0,17])
    assert reconstruct_response(basis[rows],weights,active_indices=indices,combination=ORDERED).tobytes()==expected[rows].tobytes()
    with pytest.raises(ValueError): prepared.selected.setflags(write=True)
    with pytest.raises(AttributeError): prepared._combination='legacy_blas/1'
    with pytest.raises(ReconstructionError,match='mismatch'):
        reconstruct_prepared_response(prepared,weights,combination='legacy_blas/1')


def test_legacy_public_defaults_and_ordered_signed_zero_overflow_refusals():
    basis=np.random.default_rng(9).normal(size=(17,5));weights=np.array([.2,-.7,1.1])
    active=[4,2,1]
    expected=basis[:,0]+basis[:,active]@weights
    assert reconstruct_response(basis,weights,active_indices=active).tobytes()==expected.tobytes()
    assert reconstruct_prepared_response(prepare_response(basis,active_indices=active),weights).tobytes()==expected.tobytes()
    zeros=np.array([[-0.],[0.]])
    assert reconstruct_response(zeros,[],active_indices=[],combination=ORDERED).tobytes()==zeros[:,0].tobytes()
    with pytest.raises(ReconstructionError,match='nonfinite'):
        reconstruct_response(np.array([[0.,1.e308]]),[2.],combination=ORDERED)
    for bad in ('unknown',2,None):
        with pytest.raises(ReconstructionError): reconstruct_response(basis,[1,2,3,4],combination=bad)
        with pytest.raises(ReconstructionError): prepare_response(basis,combination=bad)
    with pytest.raises(ReconstructionError): reconstruct_response(basis,[1,1],active_indices=[1,1],combination=ORDERED)


def test_confirmed_f2_input_ordered_tiles_and_prepared_exact_bits():
    # NumPy's lognormal exponential varies by platform. Replay the original
    # Windows F2 basis and the 20 weights drawn immediately afterward.
    import hashlib
    from pathlib import Path
    with np.load(Path(__file__).parent/'data/confirmed_f2_replay.npz',allow_pickle=False) as fixture:
        basis=fixture['basis']
        weights=fixture['weights']
    assert basis.shape==(1307,21) and weights.shape==(20,)
    assert basis.dtype==weights.dtype==np.dtype('<f8')
    assert hashlib.sha256(basis.tobytes()).hexdigest()=='072b419c92146fcc920823896d76a045ac9fb50e202c756e17f58adbdf96844c'
    assert hashlib.sha256(weights.tobytes()).hexdigest()=='8dff1f57f5589fa1b156c445ce4fc9d38e180ba9cc742b5a42f8f63631c48262'
    active=np.array([20,1,11,3,7,15,2,9,6,19,4,12,8,14,10,17,16,13,18,5])
    packed=weights[active-1]
    expected=reconstruct_response(basis,packed,active_indices=active,combination=ORDERED)
    for values in (basis,np.asfortranarray(basis)):
        prepared=prepare_response(values,active_indices=active,combination=ORDERED)
        assert reconstruct_prepared_response(prepared,packed).tobytes()==expected.tobytes()
        for size in (1,3,17,64,407):
            actual=np.concatenate([reconstruct_response(values[start:start+size],packed,
                active_indices=active,combination=ORDERED) for start in range(0,len(values),size)])
            assert actual.tobytes()==expected.tobytes()
        rows=np.array([1306,2,2,0,17])
        assert reconstruct_response(values[rows],packed,active_indices=active,combination=ORDERED).tobytes()==expected[rows].tobytes()


def capsule_component(capsule,index,key):
    """Minimal runtime-only replay after case admission and stored precedence."""
    case=capsule['cases'][index]
    if key in case.get('unavailable_components',[]): raise KeyError(key)
    stored=capsule['stored'].get(key)
    if stored is not None and index in stored['case_indices']:
        return stored['values'][list(stored['case_indices']).index(index)]
    if not case['portable']: raise KeyError(key)
    if key.startswith('displacement:') and case.get('displacement_portable') is not True: raise KeyError(key)
    if key in capsule['derived']:
        spec=capsule['derived'][key]
        inputs={name:capsule_component(capsule,index,dependency) for name,dependency in spec['inputs'].items()}
        return derive_field(spec,inputs)
    active=np.asarray(case['active_indices'],dtype=np.intp)
    return reconstruct_response(capsule['responses'][key],capsule['coefficients'][index,active-1],
        active_indices=active,combination=capsule.get('combination','legacy_blas/1'))


@pytest.mark.parametrize('fixture',['beam2','beam3','shell'])
def test_runtime2_actual_source_every_field_uint64_and_spectral_storage(fixture):
    from anyfesolver import (PatternCase,RecoveryConfig,FixedSupport,BoundaryCondition,
                          prepare_static_response_evaluator,export_recovery_capsule)
    from test_prepared_static_evaluator import _bar,_case,_shell_case
    from test_result_capsule_export import _rotated_shell_model,_reference_fields
    from test_generalized_beam_sections import _model,_stiffness
    from anyfesolver.elements import QuadraticBeamElement
    from anyfesolver.beam_sections import GeneralizedBeamSection
    if fixture=='shell':
        model=_rotated_shell_model()
        patterns={f'p{i}':_shell_case(i+1,pressure=(i+1)*.7).load_case for i in range(4)}
        baseline=_shell_case(0,pressure=.5).load_case
    else:
        if fixture=='beam3':
            model,_=_model(QuadraticBeamElement,section=GeneralizedBeamSection(_stiffness()))
            model.add_boundary_condition(FixedSupport('base',[1]))
            model.add_boundary_condition(BoundaryCondition('axial',[2,3],{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
        else: model=_bar()
        patterns={f'p{i}':_case(i+1,(i+1)*.7).load_case for i in range(4)}
        baseline=_case(0,.5).load_case
    components=None if fixture!='shell' else ['membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz','von_mises','principal_stresses_top','principal_stresses_bottom']
    cases=[PatternCase('dense',{'p3':2.1,'p0':-.9,'p1':.3,'p2':1.7}),PatternCase('empty',{}),PatternCase('tie',{'p3':2.1,'p0':-.9,'p1':.3,'p2':1.7})]
    config=RecoveryConfig(include_stresses=True,include_reactions=True,components=components)
    with prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,recovery_config=config,stress_execution='certified_library') as evaluator:
        reference=next(evaluator.iter_batches(cases)).case_results
        capsule=export_recovery_capsule(evaluator,cases,model_identity='model',mesh_identity='mesh')
        legacy=export_recovery_capsule(evaluator,cases,model_identity='model',mesh_identity='mesh',replay_version=1)
    validate_capsule(capsule);validate_capsule(legacy)
    assert capsule['version']==2 and capsule['runtime']=='anysolver-results/2' and capsule['combination']==ORDERED
    assert legacy['version']==1 and legacy['runtime']=='anysolver-results/1' and 'combination' not in legacy
    for index,record in enumerate(reference):
        fields=_reference_fields(record)
        for key in fields:
            if key in capsule['responses'] or key in capsule['derived']:
                assert key not in capsule['cases'][index]['unavailable_components'],key
        for descriptor in capsule['descriptors']:
            key=descriptor['key']
            if key in capsule['cases'][index]['unavailable_components']: continue
            actual=capsule_component(capsule,index,key)
            if key.startswith('principal_stress_'):
                rank,surface=key.split(':');rank=int(rank.removeprefix('principal_stress_'))
                expected=np.concatenate([np.asarray(value['principal_stresses_'+surface])[:,rank-1] for eid,value in sorted(record['element_stresses'].items())])
                assert key not in capsule['derived'] and descriptor['availability']=='stored'
            elif key.startswith('displacement:'):
                position=['UX','UY','UZ','RX','RY','RZ'].index(key.split(':')[1])
                expected=np.array([value[position] for _,value in sorted(record['node_displacements'].items())])
            elif key.startswith('reaction:'):
                position=['FX','FY','FZ','MX','MY','MZ'].index(key.split(':')[1])
                expected=np.array([value[position] for _,value in sorted(record['reactions'].items())])
            else: expected=np.concatenate([value for _,value in sorted(fields[key].items())])
            assert actual.dtype==np.float64 and actual.shape==expected.shape
            assert actual.tobytes()==expected.tobytes(),(fixture,index,key)
    assert [case['input_index'] for case in capsule['cases']]==[0,1,2]
    for key in fields:
        if key in capsule['responses'] or key in capsule['derived']:
            if key in capsule['cases'][0]['unavailable_components']: continue
            assert capsule_component(capsule,0,key).tobytes()==capsule_component(capsule,2,key).tobytes()


@pytest.mark.parametrize('target',['runtime_alias','runtime_module','numpy_add','numpy_multiply'])
def test_real_sink_callback_replay_mutation_refuses_and_reports_once(target,monkeypatch):
    from anyfesolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator,export_recovery_capsule_to_sink,PreparedStaticCapabilityError
    import anyfesolver_results as runtime
    from anyfesolver_results import capsule as implementation
    from test_prepared_static_evaluator import _bar,_case
    from anyfesolver.result_capsule import _MemorySink
    class Sink(_MemorySink):
        failures=[]
        def capsule_case(self,record,coefficients):
            super().capsule_case(record,coefficients)
            if target=='runtime_alias': monkeypatch.setattr(runtime,'reconstruct_response',lambda *a,**k:np.zeros(1))
            elif target=='runtime_module': monkeypatch.setattr(implementation,'_combine_response',lambda *a,**k:np.zeros(1))
            elif target=='numpy_add': monkeypatch.setattr(np,'add',lambda *a,**k:None)
            else: monkeypatch.setattr(np,'multiply',lambda *a,**k:None)
        def capsule_end(self,*args): pytest.fail('must not complete after mutation')
        def capsule_failed(self,record): self.failures.append(record)
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        sink=Sink()
        with pytest.raises(PreparedStaticCapabilityError,match='authority'):
            export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{'p':1.})],sink=sink,model_identity='m',mesh_identity='s')
        assert len(sink.failures)==1


def test_missing_actual_rows_and_overflow_use_truthful_unavailability_or_correction():
    """Synthetic damage to a genuine source record/basis tests publication admission."""
    from anyfesolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator
    from anyfesolver.result_capsule import _CapsuleExport
    from test_prepared_static_evaluator import _bar,_case
    cases=[PatternCase('a',{'p':2.,'q':2.})]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case,'q':_case(2).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        record=copy.deepcopy(next(evaluator.iter_batches(cases)).case_results[0])
        missing=copy.deepcopy(record)
        for fields in missing['element_stresses'].values():
            fields.pop('axial_stress',None);fields.pop('von_mises',None)
        missing['node_displacements']={}
        owner=_CapsuleExport(evaluator,cases,model_identity='m',mesh_identity='s')
        metadata,_=owner._process_case(0,missing)
        assert {'axial_stress','von_mises',*owner.displacement_keys} <= set(metadata['unavailable_components'])
        assert metadata['displacement_portable'] is None
        assert 'axial_stress' not in owner.pending_rows and 'von_mises' not in owner.pending_rows
        owner=_CapsuleExport(evaluator,cases,model_identity='m',mesh_identity='s')
        owner.responses['axial_stress'][:]=np.array([[0.,1.e308,-1.e308]])
        metadata,_=owner._process_case(0,record)
        assert metadata['correction_reasons']['axial_stress']=='ordered_replay_refused'
        actual=owner._element_row('axial_stress',record)
        assert owner.pending_rows['axial_stress'][0][1].tobytes()==actual.tobytes()


def test_new_geometry_dense_indices_and_legacy_admission():
    from anyfesolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator,export_recovery_capsule
    from test_prepared_static_evaluator import _bar,_case
    geometry={'positions':np.array([[0.,0.,0.],[1.,0.,0.]]),'lines':np.array([[0,1]],dtype=np.int64),
              'line_to_element':np.array([0],dtype=np.int64),'element_ids':np.array([101],dtype=np.int64)}
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        capsule=export_recovery_capsule(evaluator,[PatternCase('a',{'p':1.})],model_identity='m',mesh_identity='s',geometry=geometry)
    validate_capsule(capsule)
    for mutation in ('index','identity','shape','node'):
        bad=dict(capsule);bad['geometry']=dict(geometry)
        if mutation=='index': bad['geometry']['line_to_element']=np.array([101],dtype=np.int64)
        elif mutation=='identity': bad['geometry'].pop('element_ids')
        elif mutation=='shape': bad['geometry']['line_to_element']=np.array([[0]],dtype=np.int64)
        else: bad['geometry']['lines']=np.array([[0,2]],dtype=np.int64)
        with pytest.raises(CapsuleValidationError): validate_capsule(bad)
        bad['version']=1;bad['runtime']='anysolver-results/1';bad.pop('combination')
        validate_capsule(bad)  # legacy numeric-only geometry behavior retained
    for version,runtime,combination in ((2,'anysolver-results/1',ORDERED),(1,'anysolver-results/2','legacy_blas/1'),(2,'anysolver-results/2','unknown')):
        bad=dict(capsule,version=version,runtime=runtime,combination=combination)
        with pytest.raises(CapsuleValidationError): validate_capsule(bad)


def test_source_delegate_preserves_original_scalar_and_array_bits():
    from anyfesolver.prepared_static_vectorized import _shell_von_mises,_beam_von_mises
    from anyfesolver.prepared_static_measures import SHELL_LINEAR_COMPONENTS
    from anyfesolver_results.formulas import BEAM_VON_MISES_COMPONENTS
    rng=np.random.default_rng(41)
    for shape in ((),(7,),(3,7)):
        fields={name:rng.normal(size=shape)*.017 for name in SHELL_LINEAR_COMPONENTS}
        mx,my,mxy,bx,by,bxy,sxz,syz=(np.asarray(fields[name],dtype=float) for name in SHELL_LINEAR_COMPONENTS)
        surfaces=[]
        for sign in (1.,-1.):
            sxx,syy,sxy=mx+sign*bx,my+sign*by,mxy+sign*bxy
            szz=0.
            surfaces.append(np.sqrt(.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)+3.*(sxy**2+syz**2+sxz**2)))
        original=np.maximum(surfaces[0],surfaces[1])
        assert np.asarray(_shell_von_mises(fields)).tobytes()==np.asarray(original).tobytes()
        beam={name:float(rng.normal()) if not shape else rng.normal(size=shape) for name in BEAM_VON_MISES_COMPONENTS}
        sigma=beam['axial_stress']+beam['bending_stress_y']+beam['bending_stress_z']
        original=np.sqrt(sigma*sigma+3.*(beam['shear_stress_y']**2+beam['shear_stress_z']**2+beam['torsional_stress']**2))
        assert np.asarray(_beam_von_mises(beam)).tobytes()==np.asarray(original).tobytes()


def test_variant_dependency_noncontiguous_ids_preserves_native_order_and_cached_selector():
    from types import SimpleNamespace
    from anyfesolver.result_capsule import _CapsuleExport,_Layout
    owner=object.__new__(_CapsuleExport)
    source,target=_Layout(),_Layout()
    for eid,size in ((101,2),(7,1),(303,3)): source.add(eid,size)
    for eid,size in ((303,3),(101,2)): target.add(eid,size)
    owner.layouts={'source':source,'target':target}
    owner._dependency_indices={};owner._numeric_indices={};owner.responses={}
    owner.displacement_columns=None;owner.pending_rows={}
    owner.evaluator=SimpleNamespace(_library_bytes=0,_check_library_memory=lambda value:None)
    values=np.arange(6,dtype=np.float64)
    assert owner._aligned_input('source','target',values).tobytes()==values[[3,4,5,0,1]].tobytes()
    selector=next(iter(owner._dependency_indices.values()))[2]
    assert owner._aligned_input('source','target',values+10).tobytes()==(values+10)[[3,4,5,0,1]].tobytes()
    assert next(iter(owner._dependency_indices.values()))[2] is selector
    target.add(900,1)
    assert owner._aligned_input('source','target',values) is None


def test_source_near_and_exact_tie_identity_is_preserved_by_corrected_runtime2_fields():
    from anyfesolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator,export_recovery_capsule
    from test_prepared_static_evaluator import _bar,_case
    config=RecoveryConfig(include_stresses=True,include_reactions=False,components=['axial_stress','von_mises'])
    coefficients=[1.5,np.nextafter(1.5,np.inf),np.nextafter(1.5,np.inf)]
    cases=[PatternCase(str(index),{'p':value}) for index,value in enumerate(coefficients)]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1,.7).load_case},baseline_load_case=_case(0,.3).load_case,
        recovery_config=config,envelope_recovery_config=config,stress_execution='certified_library') as evaluator:
        capsule=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='s')
        summary=evaluator.envelope_result()
    for entry in summary.extrema:
        if entry.quantity!='stress': continue
        rows=np.stack([capsule_component(capsule,index,entry.component) for index in range(3)])
        association=capsule['associations'][entry.component]
        samples=np.flatnonzero(association==entry.entity_id)
        chosen=rows[:,samples].reshape((3,)+entry.original_shape)
        assert np.asarray(entry.maximum.input_indices).tobytes()==np.asarray(np.argmax(chosen,axis=0),dtype=np.int64).tobytes()
        assert np.asarray(entry.maximum.values).tobytes()==np.max(chosen,axis=0).tobytes()
