"""Exact archive controls and an actual public transient producer/consumer."""
from dataclasses import replace,fields,FrozenInstanceError,MISSING
import hashlib,json,os
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact import (TransientContactResults,ContactResults,
    TransientContactArchiveError as Error,serialize_transient_contact_results as serialize,
    read_transient_contact_results as read)
from anyfesolver.general_contact.transient_results_archive import CONVENTIONS,MAX_BYTES,_types


def sample():
    contact=ContactResults('test',float('inf'),0.,(0.,0.,0.),(0.,0.,0.),
                           (0.,0.,0.),(0.,0.,0.),0,True)
    return TransientContactResults('codec-only',np.array([0.,.1]),np.zeros((2,6)),
        np.zeros((2,6)),np.zeros((2,6)),(contact,contact),np.array([float('inf'),-.0]),
        np.zeros(2),0,(1,),np.array([-.0,1.]),restart_checkpoint={'schema':'identity','bits':-.0})


def resign(document):
    payload=dict(document);payload.pop('sha256')
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
    document['sha256']=hashlib.sha256(raw).hexdigest()
    return json.dumps(document,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')


def payload_fields(doc):
    return {key:node for key,node in doc['payload']['items']}


def test_exact_bits_infinity_numpy_material_and_no_writable_alias():
    value=sample()
    material={1:{'plastic_strain':np.arange(6,dtype='>f8').reshape(2,3),
                 'alpha':np.array([np.nextafter(0.,1.),-.0]),'count':np.int64(3)}}
    value=replace(value,element_states=material,material_state_history=(material,material),
        prescribed_tool_work_j=({'tool':-.0},{'tool':1.}),
        prescribed_tool_work_error_j=({'tool':0.},{'tool':1e-12}),
        constraint_reactions=np.arange(12,dtype=float).reshape(2,6),
        pair_swept_minimum_gaps=({'a':float('inf')},{'a':-.0}),
        cutback_diagnostics=({'reason':'exact','count':np.int64(1)},))
    raw=serialize(value);archive=read(raw);data=archive.to_dict()
    assert archive.to_bytes()==raw
    assert np.signbit(data['external_nodal_work_j'][0])
    assert np.signbit(data['swept_minimum_gaps'][1])
    assert np.signbit(data['restart_checkpoint']['bits'])
    assert data['element_states'][1]['plastic_strain'].dtype.str=='>f8'
    np.testing.assert_array_equal(data['element_states'][1]['alpha'].view('u8'),material[1]['alpha'].view('u8'))
    value.displacements[0,0]=123.;material[1]['alpha'][0]=99.
    assert archive.to_dict()['displacements'][0,0]==0.
    data['constraint_reactions'][0,0]=999.
    assert archive.to_dict()['constraint_reactions'][0,0]==0.
    with pytest.raises(FrozenInstanceError):archive._raw=b'x'
    assert json.loads(raw)['conventions']==CONVENTIONS


def test_refusal_schema_integrity_dimensions_duplicates_and_resources():
    raw=serialize(sample())
    for change in ('schema','conventions','dimensions','record','mapping_duplicate','vector'):
        doc=json.loads(raw);nodes=payload_fields(doc)
        if change=='schema':doc['schema']='future'
        elif change=='conventions':doc['conventions']['constraint_reactions']='physical_moments'
        elif change=='dimensions':nodes['velocities']['shape']=[12]
        elif change=='record':nodes['contact']['items'][0]['record_type']='os.system'
        elif change=='mapping_duplicate':doc['payload']['items'].append(doc['payload']['items'][0])
        else:
            record=nodes['contact']['items'][0]['fields']['items']
            next(row[1] for row in record if row[0]=='shell_force')['items'].pop()
        with pytest.raises(Error):read(resign(doc))
    with pytest.raises(Error):read(raw.replace(b'"version":1',b'"version":1,"version":1'))
    with pytest.raises(Error):read(b'{"value":NaN}')
    with pytest.raises(Error):read(b'['*65+b']'*65)
    with pytest.raises(Error):read(raw,max_bytes=len(raw)-1)
    with pytest.raises(Error):serialize(sample(),max_bytes=128)
    with pytest.raises(Error):serialize(replace(sample(),displacements=np.broadcast_to(np.zeros(1),(2,10_000_000))))
    with pytest.raises(Error):serialize(replace(sample(),external_nodal_work_j=np.array([0.,np.nan])))
    with pytest.raises(Error):serialize(replace(sample(),element_states={'bad':object()}))
    with pytest.raises(Error):read(raw,max_bytes=MAX_BYTES+1)


def test_free_body_and_contact_variant_record_inventory():
    from anyfesolver.general_contact.free_rigid import FreeRigidBodyState
    state=FreeRigidBodyState((0.,0.,0.),(1.,0.,0.),(0.,0.,0.),
        ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),(0.,0.,0.),(0.,0.,0.),.5,(1.,0.,0.),(0.,0.,0.))
    value=replace(sample(),free_rigid_bodies=({'sphere':state},{'sphere':state}))
    data=read(serialize(value)).to_dict()
    assert data['free_rigid_bodies'][0]['sphere']['record_type'].endswith('.FreeRigidBodyState')
    assert data['free_rigid_bodies'][0]['sphere']['fields']['kinetic_energy_j']==.5
    # Each known contact class is represented by all exact dataclass fields;
    # no arbitrary import or result-object reconstruction is permitted.
    _,registry=_types()
    assert {name.rsplit('.',1)[-1] for name in registry}=={
        'RotationalBeamS3Results','DeformableBeamS3Results','ContactResults',
        '_PairResults','_PatchResults','_BeamPlaneResults','MultipleBeamRigidResults',
        'MultipleShellPlaneResults','_RadialResults','FreeRigidBodyState'}
    assert set(data)=={f.name for f in fields(TransientContactResults)}
    with pytest.raises(Error):serialize(replace(sample(),contact=(state,state)))
    malformed=replace(state,rotation=(('x','y','z'),)*3)
    with pytest.raises(Error):serialize(replace(value,free_rigid_bodies=({'sphere':malformed},)*2))
    with pytest.raises(Error):serialize(replace(value,free_rigid_bodies=({'sphere':1.},)*2))
    with pytest.raises(Error):serialize(replace(sample(),contact=(replace(sample().contact[0],shell_force=('x','y','z')),)*2))


def test_public_plane_impact_archive_release_reactions_and_restart_payload():
    from examples.general_contact_transient_results_archive import impact_archive
    result,archive=impact_archive();data=archive.to_dict()
    active=[x.active_pressure_dofs>0 for x in result.contact]
    report=dict(profile=result.profile,states=len(result.times),active_states=sum(active),
        final_active=active[-1],minimum_gap=float(min(x.minimum_gap for x in result.contact)),
        archive_bytes=len(archive.to_bytes()),archive_sha256=archive.sha256,
        external_work_j=float(result.external_nodal_work_j[-1]),
        reaction_shape=list(result.constraint_reactions.shape),checkpoint=result.restart_checkpoint)
    if os.environ.get('TRANSIENT_ARCHIVE_RESULT'):
        Path(os.environ['TRANSIENT_ARCHIVE_RESULT']).write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert any(active) and not active[-1]
    assert all(x.gap_tolerance_met for x in result.contact)
    for name in ('times','displacements','velocities','accelerations','swept_minimum_gaps',
                 'equilibrium_residuals','external_nodal_work_j','constraint_reactions'):
        source=getattr(result,name)
        assert source.dtype==data[name].dtype and source.shape==data[name].shape
        assert source.tobytes()==data[name].tobytes()
    assert data['restart_checkpoint']==result.restart_checkpoint
    assert len(data['contact'])==len(result.contact)
    for actual,decoded in zip(result.contact,data['contact']):assert decoded['summary']==actual.to_dict()
    assert data['constraint_reactions'].shape==result.displacements.shape
    assert result.external_nodal_work_j[-1]>0


def variant_records():
    from anyfesolver.general_contact.transient import (_PairResults,_PatchResults,_BeamPlaneResults,
        _RadialResults,MultipleBeamRigidResults,MultipleShellPlaneResults)
    def simple(cls):
        values={}
        for f in fields(cls):
            if f.default is not MISSING or f.default_factory is not MISSING:continue
            if f.name=='profile':values[f.name]='codec-variant'
            elif f.name=='obstacle_kind':values[f.name]='sphere'
            elif f.name.endswith(('_force','_moment')):values[f.name]=(-.0,1.,-1.)
            elif f.type in (bool,'bool'):values[f.name]=True
            elif f.type in (int,'int'):values[f.name]=1
            elif f.type in (float,'float'):values[f.name]=-.0
            elif f.name=='endpoint_stiffness':values[f.name]=1.
            elif f.name=='source_pressure_node_ids':values[f.name]=(1,)
            elif f.name in ('normalized_weighted_gap_m','numerical_pressure_pa','reference_support_m2'):
                values[f.name]=(1.,)
            elif f.name=='sampled_overlap_gap':values[f.name]=-.0
            else:raise AssertionError('uncovered producer field '+f.name)
        return cls(**values)
    plane=simple(_BeamPlaneResults);radial= simple(_RadialResults)
    return [sample().contact[0],simple(_PairResults),simple(_PatchResults),plane,radial,
        MultipleBeamRigidResults('multiple-beam',(('line',plane),('radial',radial)),('floor','sphere')),
        MultipleShellPlaneResults('multiple-shell',(('first',sample().contact[0]),('second',sample().contact[0])))]


@pytest.mark.parametrize('index',range(7))
def test_every_contact_variant_exact_fields_summary_and_nested_pairs(index):
    original=variant_records()[index]
    decoded=read(serialize(replace(sample(),contact=(original,original)))).to_dict()['contact'][0]
    module = type(original).__module__.replace('anyfesolver.', 'anysolver.', 1)
    assert decoded['record_type']==module+'.'+type(original).__qualname__
    assert set(decoded['fields'])=={f.name for f in fields(original)}
    assert decoded['summary']==original.to_dict()
    for f in fields(original):
        expected=getattr(original,f.name);actual=decoded['fields'][f.name]
        if f.name=='pairs':
            for (name,record),(decoded_name,data) in zip(expected,actual):
                assert decoded_name==name and data['summary']==record.to_dict()
        elif type(expected) is float:
            assert np.float64(actual).tobytes()==np.float64(expected).tobytes()
        else:assert actual==expected


def test_preflight_before_summary_clone_and_unknown_deepcopy(monkeypatch):
    from anyfesolver.general_contact.transient import _PatchResults
    from anyfesolver.nonlinear_state import NonlinearStateStore
    patch=next(x for x in variant_records() if type(x) is _PatchResults)
    patch=replace(patch,reference_support_m2=np.zeros(2_000_001))
    monkeypatch.setattr(_PatchResults,'to_dict',lambda self:pytest.fail('summary ran before oversized fields refused'))
    with pytest.raises(Error):serialize(replace(sample(),contact=(patch,patch)))
    store=object.__new__(NonlinearStateStore)
    store._batches={};store._fallback_sidecars={}
    monkeypatch.setattr(NonlinearStateStore,'has_active_trial',property(lambda self:False))
    monkeypatch.setattr(NonlinearStateStore,'materialize_owned',lambda self:pytest.fail('materialized before refusal'))
    shared=np.zeros(600_000)
    store._fallback_committed={1:{'x':shared},2:{'x':shared}}
    with pytest.raises(Error):serialize(replace(sample(),element_states=store))
    class Poison:
        def __deepcopy__(self,memo):pytest.fail('unsupported deepcopy callback ran')
    store._fallback_committed={1:Poison()}
    with pytest.raises(Error):serialize(replace(sample(),element_states=store))


def test_record_scalar_pressure_roles_and_astral_preflight(monkeypatch):
    for field,value in (('penalty_energy','not-a-number'),('minimum_gap','not-a-gap'),
                        ('active_pressure_dofs',1.),('gap_tolerance_met',1)):
        malformed=replace(sample().contact[0],**{field:value})
        with pytest.raises(Error):serialize(replace(sample(),contact=(malformed,malformed)))
    from anyfesolver.general_contact.transient import _PatchResults
    patch=next(x for x in variant_records() if type(x) is _PatchResults)
    for field in ('numerical_pressure_pa','reference_support_m2','normalized_weighted_gap_m'):
        malformed=replace(patch,**{field:('not-a-number',)})
        with pytest.raises(Error):serialize(replace(sample(),contact=(malformed,malformed)))
    valid=replace(sample().contact[0],penalty_energy=np.float64(-.0),
        active_pressure_dofs=np.int64(0),gap_tolerance_met=np.bool_(True))
    assert read(serialize(replace(sample(),contact=(valid,valid)))).to_dict()['contact'][0]['fields']['gap_tolerance_met']
    import anyfesolver.general_contact.transient_results_archive as codec
    monkeypatch.setattr(codec,'_canonical',lambda x:pytest.fail('canonical allocation before oversized Unicode refusal'))
    with pytest.raises(Error):serialize(replace(sample(),profile='\U0001f600'*1000),max_bytes=10_000)
    with pytest.raises(Error):codec._encode('\U0001f600'*1000,codec._Budget(11_000),{})


def test_zero_extent_does_not_allow_unbounded_shape_dimension():
    doc=json.loads(serialize(sample()))
    array=payload_fields(doc)['constraint_reactions']
    array.update(shape=[0,10**20],data='')
    with pytest.raises(Error):read(resign(doc))
