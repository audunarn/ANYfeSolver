"""Manufactured/source controls only. No family fixture is imported/executed."""
import numpy as np
import pytest
from scipy import sparse
from scripts.r8_f03_sparse_signed_family_dev import (Snapshot,SparseSignedPreparation,Refused,Unresolved,
    exact_work_witness,extract_owner,role_parity,stationary_system,matrices)

def toy(negative=False,skew=0.):
    k=np.array([[2.,.25,0.,1.],[.25,3.,.5,0.],[0.,.5,4.,1.],[1.,0.,1.,-2.]])
    if negative:k[0,0]=-4.
    k[0,1]+=skew
    c=np.array([[1.,1.,0.,0.]])
    return Snapshot(sparse.csr_matrix(k),sparse.csr_matrix(c),np.arange(4),np.array([],dtype=int),((3,),),
        dict(shell=np.array([0]),beam_nodes=np.array([1]),retained_rotations=np.array([2]),resultants=np.array([3])),np.ones(4),np.ones(1),dict(qualification=False))

@pytest.mark.parametrize('negative',[False,True])
def test_sparse_actions_transpose_nullspace_and_stationary_oracle(negative):
    s=toy(negative);p=SparseSignedPreparation(s)
    k=s.k.toarray();c=s.c[:,p.x].toarray();z=np.zeros((3,2));z[p.rest]=np.eye(2)
    z[p.pivot]=-np.linalg.solve(c[:,p.pivot],c[:,p.rest])
    reduced=k[:3,:3]-k[:3,3:]@np.linalg.solve(k[3:,3:],k[3:,:3])
    v=np.array([1.,0.]) if negative else np.array([.5,-.25]);w=np.array([.3,.7]);q=p.lift(p.z(v))
    np.testing.assert_allclose(p.z(v),z@v,atol=1e-15,rtol=0)
    np.testing.assert_allclose(p.zt(np.array([.3,.7,.2])),z.T@np.array([.3,.7,.2]),atol=1e-15)
    np.testing.assert_allclose(p.operator@v,z.T@reduced@z@v,atol=1e-15)
    np.testing.assert_allclose(p.operator.rmatvec(w),z.T@reduced.T@z@w,atol=1e-15)
    np.testing.assert_allclose(s.c@q,0.,atol=1e-15);np.testing.assert_allclose((s.k@q)[3],0.,atol=1e-15)
    rs,rc=stationary_system(p);np.testing.assert_allclose(rs.toarray(),reduced,atol=1e-15)
    out=p.probe(v)
    assert out['kernel_positive'] is None and out['inertia'] is None
    assert out['status']=='UNRESOLVED'
    if negative:assert out['witness']['exact_resultant_stationarity'] is False

def test_positive_sample_cannot_hide_negative_direction():
    p=SparseSignedPreparation(toy(True))
    assert p.probe(np.array([0.,1.]))['status']=='UNRESOLVED'
    negative=p.probe(np.array([1.,0.]))
    assert negative['full_work']<0 and negative['status']=='UNRESOLVED'
    exact=exact_work_witness(toy(True).k,toy(True).c,np.array([-1.,1.,0.,-.5]),resultant_indices=(3,))
    assert exact['status']=='NEGATIVE_STORED_WORK_WITNESS'

def test_bounded_skew_action_keeps_stored_operator_without_inertia():
    s=toy(skew=1e-10);old=s.k.copy();p=SparseSignedPreparation(s)
    assert p.skew>0 and (s.k-old).nnz==0
    v=np.array([.5,-.25]);w=np.array([.3,.7])
    assert abs(w@(p.action(v))-v@(p.transpose_action(w)))<1e-14
    assert p.probe(v)['inertia'] is None
    with pytest.raises(Refused,match='skew'):SparseSignedPreparation(toy(skew=1.))

def test_exact_witness_requires_actual_constraint_and_keeps_tiny_omission():
    k=sparse.diags([-1.,1.],format='csr');c=sparse.csr_matrix([[0.,1.]])
    assert exact_work_witness(k,c,[1.,0.])['status']=='NEGATIVE_STORED_WORK_WITNESS'
    assert exact_work_witness(k,c,[1.,1e-200])['status']=='UNRESOLVED'
    assert exact_work_witness(k,c,[1.,0.],term_cap=1)['status']=='UNRESOLVED'
    assert exact_work_witness(k,c,[1.5,0.],bit_cap=1)['status']=='UNRESOLVED'

def test_negative_stress_mode_is_not_a_mechanical_kernel_witness():
    s=toy();q=np.array([0.,0.,0.,1.])
    out=exact_work_witness(s.k,s.c,q,resultant_indices=(3,))
    assert out['status']=='UNRESOLVED' and not out['exact_resultant_stationarity'] and out['exact_constraint_zero']
    stationary=np.array([-1.,1.,0.,-.5]);stationary[3]+=1e-14
    out=exact_work_witness(toy(True).k,s.c,stationary,resultant_indices=(3,))
    assert out['status']=='UNRESOLVED' and 'nonstationary' in out['blocker']
    assert exact_work_witness(toy(True).k,s.c,np.array([-1.,1.,0.,-.5]),resultant_indices=(3,),fixed_indices=(0,))['status']=='UNRESOLVED'

def test_positive_probe_does_not_attempt_exact_arithmetic(monkeypatch):
    import scripts.r8_f03_sparse_signed_family_dev as module
    def forbidden(*args,**kwargs):raise AssertionError('positive probe attempted exact witness')
    monkeypatch.setattr(module,'exact_work_witness',forbidden)
    assert SparseSignedPreparation(toy()).probe(np.ones(2))['status']=='UNRESOLVED'

@pytest.mark.parametrize('mutation,pattern',[
    ('positive_D','negative'),('duplicate_blocks','overlapping'),('constraint_D','resultants'),
    ('zero_C','matching'),('singular_C','factorization'),('nan','nonfinite'),('free','unique')])
def test_refusals(mutation,pattern):
    s=toy()
    if mutation=='positive_D':s.k[3,3]=2.
    if mutation=='duplicate_blocks':s.blocks=((3,),(3,))
    if mutation=='constraint_D':s.c=sparse.csr_matrix([[1.,1.,0.,1.]])
    if mutation=='zero_C':s.c=sparse.csr_matrix((1,4))
    if mutation=='singular_C':s.c=sparse.csr_matrix([[1.,1.,0.,0.],[1.,1.,0.,0.]])
    if mutation=='nan':s.k[0,0]=np.nan
    if mutation=='free':s.free=np.array([0,1,1,3])
    with pytest.raises((Refused,Unresolved),match=pattern):SparseSignedPreparation(s)

def test_family_identity_refuses_old_doubled_beam_before_owner_access():
    wrong=dict(beam_halfcells=64,span_halfcells=4,c_lambda=2,basis='p0',shell_trace_segments=32)
    with pytest.raises(Refused,match='span2'):extract_owner(object(),32,wrong)
    with pytest.raises(Refused,match='literal'):extract_owner(object(),128,wrong)

def test_all_physical_roles_and_zero_parity_are_retained():
    s=toy();q=np.ones(4);r=q.copy();r[3]+=1e-8
    out=role_parity(s,q,r)
    assert set(out)=={'shell','beam_nodes','retained_rotations','resultants'} and out['resultants']>1e-10
    assert all(v==0 for v in role_parity(s,np.zeros(4),np.zeros(4)).values())

def test_no_global_dense_preparation(monkeypatch):
    old=sparse.csr_matrix.toarray
    def local_only(a,*args,**kwargs):
        assert max(a.shape)<=18
        return old(a,*args,**kwargs)
    monkeypatch.setattr(sparse.csr_matrix,'toarray',local_only)
    p=SparseSignedPreparation(toy());p.action(np.ones(2));p.transpose_action(np.ones(2));p.probe(np.ones(2))

def test_owned_preparation_detaches_mutable_inputs():
    s=toy();p=SparseSignedPreparation(s);v=np.ones(2);before=p.action(v)
    s.k.data[:]=0.;s.c.data[:]=0.;s.free[:]=0;s.coordinate_scale[:]=0;s.identity['modified']=True
    np.testing.assert_array_equal(p.action(v),before)
    assert p.snapshot.coordinate_scale.min()>0 and 'modified' not in p.snapshot.identity

def test_literal_extraction_without_family_assembly():
    from types import SimpleNamespace as NS
    n=804;free=np.arange(35,n);fixed=np.arange(35)
    k=sparse.eye(n,format='csr',dtype=float);c=sparse.csr_matrix((40,n),dtype=float)
    owner=NS(_K=k,_C=c,_free=free,_fixed=fixed,_shell_count=510,_beam_ids=tuple(range(101,118)),
        _beam_parts=tuple(range(8)),_identity='MANUFACTURED_LAYOUT_ONLY',_coordinate_scale=np.ones(n),_constraint_scale=np.ones(40),
        _definition=NS(lines=(NS(multiplier_space='BL5_P0_span_2'),)))
    owner._A=sparse.bmat([[k[free][:,free],c[:,free].T],[c[:,free],None]],format='csc')
    policy=dict(beam_halfcells=16,span_halfcells=2,c_lambda=2,shell_trace_segments=16)
    snapshot=extract_owner(owner,16,policy)
    assert snapshot.identity['family']==(16,4,16,2) and len(snapshot.roles['resultants'])==144
    k.data[:]=2.;assert np.all(snapshot.k.data==1.)
    owner._definition.lines=(NS(multiplier_space='BL5_P1_continuous'),)
    with pytest.raises(Refused,match='multiplier'):extract_owner(owner,16,policy)

def test_reduced_state_and_physical_metrics_against_tiny_independent_solve():
    from scripts.r8_f03_sparse_signed_family_dev import reduced_state,state_metrics
    s=toy();p=SparseSignedPreparation(s);load=np.array([0.,1.,1.,0.])
    q,lm=reduced_state(p,load)
    a=np.block([[s.k.toarray(),s.c.toarray().T],[s.c.toarray(),np.zeros((1,1))]])
    expected=np.linalg.solve(a,np.r_[load,0.])
    np.testing.assert_allclose(q,expected[:4],atol=1e-15,rtol=0)
    np.testing.assert_allclose(lm,expected[4:],atol=1e-15,rtol=0)
    measured=state_metrics(s,q,lm,load)
    assert measured['equilibrium']<=1e-8 and measured['weak_gap']<=1e-9
    assert max(measured[k] for k in ('energy_work_relative','compatible_work_relative','ground_work_relative'))<=1e-8
    with pytest.raises(Refused,match='resultant'):reduced_state(p,np.array([0.,1.,1.,1.]))
    with pytest.raises(Refused,match='work denominator'):state_metrics(s,np.zeros(4),np.zeros(1),np.zeros(4))

@pytest.mark.parametrize('physical,parity,negative,expected',[
    (True,True,False,'UNRESOLVED_KERNEL'),(False,True,False,'FAILED_PHYSICAL_OR_PARITY'),
    (True,True,True,'NEGATIVE_STORED_WORK_WITNESS'),(False,False,True,'NEGATIVE_STORED_WORK_WITNESS')])
def test_caller_disposition_preserves_negative_witness(physical,parity,negative,expected):
    from scripts.r8_f03_sparse_signed_family_dev import case_disposition
    witness=exact_work_witness(toy(True).k,toy(True).c,np.array([-1.,1.,0.,-.5]),resultant_indices=(3,))
    probes=[witness] if negative else [dict(status='UNRESOLVED')]
    assert case_disposition(physical,parity,probes)==expected

def test_underflow_skew_is_not_exact_symmetry():
    from scripts.r8_f03_sparse_signed_family_dev import exact_symmetry_label
    k=sparse.csr_matrix([[1.,1e-200],[0.,1.]])
    assert matrices(k,sparse.csr_matrix([[1.,0.]]))==0.
    assert not exact_symmetry_label(k)
    assert exact_symmetry_label(sparse.eye(2,format='csr'))

def test_actual_fixture_case_provenance_binding_without_import_or_execution(monkeypatch):
    import sys
    from types import FunctionType,SimpleNamespace as NS
    from pathlib import Path
    import scripts.r8_f03_sparse_signed_family_dev as module
    root=Path(module.__file__).resolve().parents[1]
    fixture=lambda:None;fixture.__module__='test_stiffened_shell_l1_assembly_experimental'
    model=FunctionType((lambda:None).__code__,{'_bl5_joint_mesh_definition':fixture});model.__module__='scripts.b3_e4_bl5_coherent_aggregation_dev'
    monkeypatch.setitem(sys.modules,model.__module__,NS(__file__=str(root/'scripts/b3_e4_bl5_coherent_aggregation_dev.py')))
    monkeypatch.setitem(sys.modules,fixture.__module__,NS(__file__=str(root/'tests/test_stiffened_shell_l1_assembly_experimental.py')))
    output=module.construction_provenance(model)
    assert set(output)=={'candidate','case','fixture'} and all(len(v['sha256'])==64 for v in output.values())
    monkeypatch.setitem(sys.modules,fixture.__module__,NS(__file__=str(root/'scripts/b3_e4_bl5_coherent_aggregation_dev.py')))
    with pytest.raises(Refused,match='escaped'):module.construction_provenance(model)

def test_exact_nonnegative_work_has_accurate_blocker():
    k=sparse.block_diag((np.array([[1e16,1.,-1e16],[1.,-2.,0.],[-1e16,0.,1e16]]),np.array([[-1.]])),format='csr')
    c=sparse.csr_matrix([[1.,0.,-1.,0.]])
    q=np.array([1.,1.,1.,0.]);assert q@(k@q)==-1.
    out=exact_work_witness(k,c,q,resultant_indices=(3,))
    assert out['status']=='UNRESOLVED' and out['work_sign']==0 and out['exact_constraint_zero'] and out['exact_resultant_stationarity']
    assert out['blocker']=='exact work is nonnegative'

@pytest.mark.parametrize('failure_phase',['preparation','reduction','public_result'])
def test_native_prefix_survives_reduced_failure(monkeypatch,tmp_path,failure_phase):
    import json,sys
    from types import SimpleNamespace as NS
    import scripts.r8_f03_sparse_signed_family_dev as module
    snapshot=toy();closed=[];solves=[]
    definition=NS(shell=NS(node_ids=(1,2,3,4)))
    def mock_model(*args):return definition,dict(beam_halfcells=16,span_halfcells=2,c_lambda=2,shell_trace_segments=16)
    def mock_solve(*args):
        solves.append(args)
        if failure_phase=='public_result':return NS(status='failed',failure='MOCK_PUBLIC_FAILURE',message='unchanged failed prefix')
        return NS(status='completed',shell_displacement=np.ones(2),beam_coordinates=np.ones(2),multipliers=np.ones(1))
    owner=NS(_beam_ids=tuple(range(101,118)),solve_linear=mock_solve,close=lambda:closed.append(True),_accepted=(None,np.ones(4)))
    monkeypatch.setitem(sys.modules,'scripts.b3_e4_bl5_coherent_aggregation_dev',NS(model=mock_model))
    monkeypatch.setitem(sys.modules,'anyfesolver',NS(stiffened_shell=NS(create_analysis=lambda **kwargs:owner,NodalAction=lambda *args:args)))
    monkeypatch.setattr(module,'construction_provenance',lambda model:dict(mock='no real model/fixture execution'))
    monkeypatch.setattr(module,'extract_owner',lambda *args:snapshot)
    def mock_prepare(s):
        if failure_phase=='preparation':raise Unresolved('MOCK_LOCAL_D_REFUSAL')
        return NS()
    monkeypatch.setattr(module,'SparseSignedPreparation',mock_prepare)
    def mock_reduced(*args):raise Refused('MOCK_REDUCED_REFUSAL')
    monkeypatch.setattr(module,'reduced_state',mock_reduced)
    out=tmp_path/'mock-only'
    with pytest.raises((Unresolved,Refused)):module.family_case(16,out)
    record=json.loads((out/'result.json').read_text());inputs=json.loads((out/'inputs.json').read_text())
    assert (out/'K.npz').is_file() and (out/'C.npz').is_file() and (out/'compiled-layout.npz').is_file()
    assert (sparse.load_npz(out/'K.npz')-snapshot.k).nnz==0 and inputs['endpoint_node']==117
    assert inputs['literal_inputs']['action_node']==117 and closed==[True] and 'traceback' in record
    assert len(solves)==1
    if failure_phase=='public_result':
        assert record['status']=='FAILED_PUBLIC_SOLVE' and record['public_result']['status']=='failed' and not (out/'owner-state.npz').exists()
    else:
        assert record['public_result']['status']=='completed' and (out/'owner-state.npz').exists() and not (out/'states.npz').exists()
        with np.load(out/'owner-state.npz') as z:np.testing.assert_array_equal(z['owner_q'],np.ones(4))
        if failure_phase=='preparation':assert record['status']=='UNRESOLVED_LOCAL_OPERATOR'
        else:assert record['status']=='FAILED' and record['stage']=='stationary-reduced-equilibrium'

@pytest.mark.parametrize('level,arguments,endpoint',[(16,(16,4,16,2),117),(32,(32,8,32,2),133),(64,(64,16,64,2),165)])
def test_literal_parameterization_and_actual_endpoint(level,arguments,endpoint):
    from types import SimpleNamespace as NS
    from scripts.r8_f03_sparse_signed_family_dev import case_spec,declared_endpoint
    spec=case_spec(level)
    assert spec['model_arguments']==arguments and spec['span_halfcells']==2
    assert spec['multiplier_space']=='BL5_P0_span_2' and spec['action_domain']=='beam'
    assert spec['action']==(0.,100.,0.,0.,0.,0.)
    owner=NS(_beam_ids=tuple(range(101,endpoint+1)))
    ids,actual=declared_endpoint(owner,arguments[2]);assert actual==ids[-1]==endpoint
    with pytest.raises(Refused,match='native node IDs'):declared_endpoint(NS(_beam_ids=tuple(range(101,118))),32)

@pytest.mark.parametrize('level',[32.,True,20,'32'])
def test_case_parameter_refusal_before_model_work(level):
    from scripts.r8_f03_sparse_signed_family_dev import case_spec
    with pytest.raises(Refused,match='builtin literal'):case_spec(level)

@pytest.mark.parametrize('phase',['model','owner'])
def test_parameterized_construction_failure_preserved_before_physical_work(monkeypatch,tmp_path,phase):
    import sys,json
    from types import SimpleNamespace as NS
    import scripts.r8_f03_sparse_signed_family_dev as module
    calls=[]
    def model(*args):
        calls.append(args)
        if phase=='model':raise Refused('MOCK_MODEL_INPUT_REFUSAL')
        return object(),{}
    def create(**kw):raise Refused('MOCK_OWNER_CONSTRUCTION_REFUSAL')
    monkeypatch.setitem(sys.modules,'scripts.b3_e4_bl5_coherent_aggregation_dev',NS(model=model))
    monkeypatch.setitem(sys.modules,'anyfesolver',NS(stiffened_shell=NS(create_analysis=create)))
    monkeypatch.setattr(module,'construction_provenance',lambda model:{'mock':'no model/owner execution'})
    out=tmp_path/'failed-construction'
    with pytest.raises(Refused,match='MOCK'):module.family_case(32,out)
    record=json.loads((out/'result.json').read_text())
    assert calls==[(32,8,32,2)] and record['requested_case']['action']==[0.,100.,0.,0.,0.,0.]
    assert record['stage']==('model-construction' if phase=='model' else 'owner-construction')
    assert 'traceback' in record and not (out/'states.npz').exists()
