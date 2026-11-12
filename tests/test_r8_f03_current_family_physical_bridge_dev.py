"""Manufactured/saved binding controls; no real family compile or equilibrium."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy import sparse
from scripts.r8_f03_current_family_physical_bridge_dev import load_bound,trace_norm,common_profiles,sha,CONSUMED,source_reconciliation,reference_linear_weak_gaps,verify_full_action,exerted_from_density

def view(cells):
    # Complete flat two-row Q4 grid; all support and off-line fields retained.
    xyz=np.array([(x,y,0.) for y in (0.,.2) for x in np.linspace(0.,1.,cells+1)])
    ids=tuple(range(1,len(xyz)+1));beamids=tuple(range(101,102+cells));ns=6*len(ids);nb=6*len(beamids)+24*(cells//2)
    fixed=np.r_[np.arange(6),6*(cells+1)+np.arange(6),ns+np.arange(5)]
    shell=NS(q4_elements=tuple((i+1,i+2,i+cells+3,i+cells+2) for i in range(cells)),s3_elements=())
    supports=tuple(NS(domain='shell',node=n,component=a) for n in (1,cells+2) for a in range(6))
    return NS(_definition=NS(shell=shell,fixed=supports),_shell_xyz=xyz,_shell_ids=ids,_shell_index={n:i for i,n in enumerate(ids)},
        _beam_ids=beamids,_beam_count=nb,_shell_count=ns,_K=sparse.eye(ns+nb,format='csr'),_C=sparse.csr_matrix((5*cells//2,ns+nb)),
        _fixed=fixed,_free=np.setdiff1d(np.arange(ns+nb),fixed))

def test_same_adopted_norm_retains_offline_fields_and_native_directors():
    a=view(2);h,e,w,absent=trace_norm(a,np.array((0.,0.,-.05)))
    assert h.shape==(5*4+3*2+3*2,)*2
    assert not e[a._fixed].nnz
    assert all(e[6*i].nnz==1 for i in (1,2,4,5))
    retained=a._shell_count+6*len(a._beam_ids)
    assert e[retained:retained+6].nnz==6 and not e[retained+6:].nnz
    assert set(range(retained+6,a._K.shape[0])).issubset(set(absent))
    assert np.min(np.linalg.eigvalsh(h.toarray()))>0

def test_omitted_work_not_silently_dropped():
    a=view(2);a._C[0,-1]=1.
    with pytest.raises(ValueError,match='omitted'):trace_norm(a,np.array((0.,0.,-.05)))

def test_reference_linear_director_sign_and_sided_cell_average():
    a,b=view(2),view(4);qa=np.zeros(a._K.shape[0]);qb=np.zeros(b._K.shape[0])
    qa[3::6][:len(a._shell_ids)]=2.
    qb[3::6][:len(b._shell_ids)]=3.
    ba=a._shell_count+6*len(a._beam_ids);bb=b._shell_count+6*len(b._beam_ids)
    qa[ba]=4.;qa[ba+3]=8.
    for j,v in enumerate((2.,6.,7.,9.)):qb[bb+24*(j//2)+3*(j%2)]=v
    p=common_profiles((a,qa),(b,qb))
    assert np.array(p['shell_director_change']['signed_change'])[0,1]==-1.
    assert p['beam_cell_director_average']['maximum_absolute_change']==0.
    assert 'not pointwise' in p['beam_cell_director_average']['meaning']

def test_no_interpolation_at_unmatched_material_nodes():
    a,b=view(2),view(4);b._shell_xyz[0,0]=.123
    with pytest.raises(ValueError,match='common'):common_profiles((a,np.zeros(a._K.shape[0])),(b,np.zeros(b._K.shape[0])))

@pytest.mark.parametrize('mode',('missing','changed','escaped'))
def test_saved_input_binding_refusals_before_numeric_load(tmp_path,mode):
    files={n:'0'*64 for n in CONSUMED}
    if mode=='missing':files.pop('K.npz')
    if mode=='escaped':files['../K.npz']=files.pop('K.npz')
    with pytest.raises(ValueError,match='binding|identity'):load_bound(tmp_path,files,16)

@pytest.mark.parametrize('level',(True,128,16.))
def test_only_saved16_32_admission(tmp_path,level):
    with pytest.raises(ValueError,match='16/32'):load_bound(tmp_path,{},level)

def test_saved64_admission_and_scope(tmp_path):
    from scripts.r8_f03_current_family_physical_bridge_dev import recovery_levels
    # Valid64 reaches mandatory hash binding, not an old level refusal.
    with pytest.raises(ValueError,match='complete saved input binding'):load_bound(tmp_path,{},64)
    assert recovery_levels([16,32])==(16,32) and recovery_levels([64])==(64,)
    for bad in ([64.],[True],[16,32,64],[128],[],[64,64]):
        with pytest.raises(ValueError,match='recovery'):recovery_levels(bad)

def test_three_level_observation_no_accuracy_claim():
    from scripts.r8_f03_current_family_physical_bridge_dev import three_level_observations
    records={str(n):dict(applied_work=float(i),root_support_work=0.,wrenches={
        'support':{'all':[0.,-100.,0.,6.75,0.,-100.]},
        'interface':{'shell':[x,100.,0.,0.,0.,100.],'beam':[-x,-100.,0.,0.,0.,-100.]}})
        for n,i,x in ((16,1,4.),(32,2,6.),(64,3,5.))}
    r=three_level_observations(records,{'mock':'not evaluated'})
    assert r['quantities']['interface_shell']['signed_increments'][1][0]==-1.
    assert r['quantities']['interface_shell']['absolute_increments'][1][0]==1.
    assert r['accuracy_estimate'] is None and r['convergence_acceptance'] is None and not r['qualification']
    with pytest.raises(ValueError,match='complete'):three_level_observations({'16':records['16']},{})

def test_old_source_difference_never_transfers_floor(tmp_path):
    p=tmp_path/'old.json';p.write_text(json.dumps({'source_sha256':{'a.py':'0'*64}}));(tmp_path/'a.py').write_text('pass\n')
    r=source_reconciliation(tmp_path,p)
    assert not r['old_floor_reused'] and not r['sources']['a.py']['byte_equal']

def test_independent_density_translation_weights_arm_and_units():
    from scripts.b3_e4_bl5_reference_actions_dev import physical_loads
    beam,shell,w=physical_loads(np.array(((0.,1.,0.,0.,0.),)),np.eye(3),np.array((0.,0.,1.)),
        np.array(((1.,0.,0.),(0.,1.,0.))),np.array((0.,0.,-.05)),np.array((0.,0.,.0175)),
        np.array((0.,.5,1.)),np.array((0.,.5,1.)),1,2,2)
    np.testing.assert_array_equal(beam[:9].reshape(3,3)[:,1],(.25,.5,.25))
    np.testing.assert_array_equal(shell.reshape(5,6)[1,:3],(-.25,-.5,-.25))
    fb,mb,fs,ms=w
    np.testing.assert_allclose(mb+ms+np.cross((0.,0.,.0675),fb),0.,atol=2e-17)
    assert float(np.sum(beam[9:].reshape(2,3)[:,0]))==.05
    # Wrong reference arm is a physical moment mismatch, not a scalar sign fix.
    assert np.max(np.abs(mb+ms+np.cross((0.,0.,.06),fb)))>.007

def test_weak_gap_label_identity_with_force_equivalent_density_conversion():
    c=np.array(((1.,-1.,0.),(0.,2.,-3.),(4.,0.,0.),(0.,5.,0.),(0.,0.,6.)))
    lam=np.diag((1.,1.,1.,.4,.4));q=np.array((.02,-.03,.01))
    loads=-c.T@lam
    np.testing.assert_allclose(reference_linear_weak_gaps(loads,q,lam),c@q,rtol=0.,atol=3e-17)
    assert 'not a sided transfer/compliance' in reference_linear_weak_gaps.__doc__

def test_full_action_refuses_fixed_row_error_orthogonal_to_saved_multiplier():
    c=np.arange(30,dtype=float).reshape(5,6)/30
    lam=np.diag((1.,1.,1.,.4,.4));loads=-c.T@lam
    assert verify_full_action(sparse.csr_matrix(c),loads,lam)==0.
    corrupted=c.copy();corrupted[1,0]+=1e-4
    e=np.eye(6)[:,1:] # fixed physical row0 disappears from supported work
    np.testing.assert_allclose(e.T@loads,-(corrupted@e).T@lam,rtol=0.,atol=0.)
    saved_lm=np.array((1.,0.,0.,0.,0.))
    np.testing.assert_array_equal(loads@saved_lm,-corrupted.T@saved_lm)
    with pytest.raises(ValueError,match='full independent'):
        verify_full_action(sparse.csr_matrix(corrupted),loads,lam)

def test_nonzero_exerted_convention_and_caller(tmp_path,monkeypatch):
    """Actual central caller with manufactured providers, no physical fixture."""
    from scripts import r8_f03_current_family_physical_bridge_dev as module
    from scripts import b3_e4_bl5_coherent_aggregation_dev as provider
    from scripts import r8_bl5_actual_primal_work_dev as normative
    from anyfesolver import stiffened_shell
    from scripts.r8_f03_sparse_signed_family_dev import sha_matrix
    k=sparse.eye(3,format='csr');c=sparse.csr_matrix(np.array(((1.,2.,-1.),)));lam=np.diag((.4,));lm=np.array((3.,));q=np.array((.1,.2,.3))
    loads=-np.asarray(c.T@lam)
    np.testing.assert_allclose(exerted_from_density(loads,lm,np.diag(lam)),-c.T@lm,rtol=0.,atol=1e-15)
    assert not np.allclose(-loads@(lm/np.diag(lam)),-c.T@lm,rtol=1e-11,atol=2e-12)
    definition=NS(shell=NS(thickness_m=.035))
    local=NS(port=NS(frame=np.eye(3)),reference_director=np.array((0.,0.,1.)),director_basis=np.array(((1.,0.,0.),(0.,1.,0.))),reference_arm=np.array((0.,0.,-.05)))
    owner=NS(_K=k,_C=c,_identity='manufactured',_line_samples=((local,),),close=lambda:None)
    monkeypatch.setattr(stiffened_shell,'create_analysis',lambda **kw:owner)
    monkeypatch.setattr(provider,'model',lambda *args:(definition,{}))
    monkeypatch.setattr(module,'load_bound',lambda *args:({'operator_identity':{'owner_identity':'manufactured'}},k,c,{'resultants':np.array([],int)},{'owner_q':q,'owner_lm':lm}))
    monkeypatch.setattr(module,'geometry_view',lambda *args:owner)
    monkeypatch.setattr(module,'trace_norm',lambda *args:(k,k,c,np.array([],int)))
    monkeypatch.setattr(normative,'actual_flat_primal_work',lambda *args:(k,k,c))
    monkeypatch.setattr(module,'independent_action',lambda *args:({'columns':1},loads,lam))
    monkeypatch.setattr(module,'common_profiles',lambda *args:{'manufactured':'no real profile'})
    monkeypatch.setattr(module,'source_reconciliation',lambda *args:{'old_floor_reused':False})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'root':str(tmp_path),'source_inputs':{},'saved':{str(n):{'directory':'mock','files':{}} for n in (16,32)},'old_manifest':'mock'}))
    out=tmp_path/'result';r=module.run_bridge(manifest,out)
    assert r['status']=='PHYSICAL_ACTION_NORM_BRIDGE_PASS_REFERENCE_UNRESOLVED'
    with np.load(out/'action16.npz') as z:
        np.testing.assert_allclose(z['physical_action'],-c.T@lm,rtol=0.,atol=1e-15)
        assert not np.any(np.abs(z['signed_delta'])>z['threshold'])
    assert (out/'action32.npz').exists()

def wrench_view():
    return NS(_K=sparse.eye(36,format='csr'),_shell_count=6,_shell_xyz=np.array(((0.,0.,0.),)),
        _beam_ids=(101,),_beam_nodes={101:(np.array((1.,0.,.1)),np.eye(3))},_beam_parts=(object(),))

def test_wrench_recovery_retained_torque_origin_work():
    from scripts.r8_f03_current_family_physical_bridge_dev import rigid_reference_generators,reference_wrench
    v=wrench_view();a=np.zeros(36);a[7]=100.;a[12:18]=(2.,3.,5.,7.,11.,13.);a[18:]=1e9
    g=rigid_reference_generators(v,np.zeros(3));w=reference_wrench(v,a,np.zeros(3))
    np.testing.assert_array_equal(w,(0.,100.,0.,-1.,14.,118.))
    np.testing.assert_allclose(g.T@a,w,rtol=0.,atol=1e-14)
    shift=np.array((.3,-.2,.4));ws=reference_wrench(v,a,shift)
    np.testing.assert_allclose(ws,np.r_[w[:3],w[3:]-np.cross(shift,w[:3])],rtol=0.,atol=1e-14)
    velocity=np.array((.2,-.3,.1,.4,.5,-.7))
    assert abs((g@velocity)@a-velocity@w)<1e-12
    wrong=g.copy();wrong[12:18]=0.
    assert np.max(np.abs(wrong.T@a-w))>1.
    assert not np.any(g[18:]) # no force-like resultant display map

def test_wrench_recovery_all_column_objectivity_wrong_retained_map():
    from scripts.r8_f03_current_family_physical_bridge_dev import rigid_reference_generators,reference_objectivity
    v=wrench_view();g=rigid_reference_generators(v,np.zeros(3))
    # Translation-cellrotation coupling, invariant under the correct generator.
    row=np.zeros(36);row[3]=1.;row[12]=-1.;k=sparse.csr_matrix(np.outer(row,row));c=sparse.csr_matrix(row[None,:])
    metrics,_=reference_objectivity(k,c,g,np.linspace(.2,3.7,36))
    assert max(metrics.values())<1e-14
    wrong=g.copy();wrong[12:18]=0.
    metrics,_=reference_objectivity(k,c,wrong,np.ones(36))
    assert metrics['right']>.9 and metrics['left']>.9 and metrics['constraint']>.9
    # Tiny accepted q/residual would not identify this wrong map: all-column check does.
    np.testing.assert_array_equal(k@np.zeros(36),np.zeros(36))

def test_wrench_recovery_no_floor_refusals():
    from scripts.r8_f03_current_family_physical_bridge_dev import absolute_product_ratio,reference_objectivity,reference_wrench
    assert absolute_product_ratio(np.zeros((2,3)),np.zeros((2,3)))==0.
    with pytest.raises(ValueError,match='zero absolute'):
        absolute_product_ratio(np.array((1e-200,)),np.zeros(1))
    v=wrench_view()
    with pytest.raises(ValueError,match='positive compiled'):
        reference_objectivity(v._K,sparse.csr_matrix((1,36)),np.zeros((36,6)),np.zeros(36))
    with pytest.raises(ValueError,match='complete action'):
        reference_wrench(v,np.full(36,np.nan),np.zeros(3))

def regional_policy(cells):
    return dict(beam_halfcells=cells,span_halfcells=2,c_lambda=2,
        support_membership=[list(range(j,j+2)) for j in range(0,cells,2)],support_endpoints_m=(np.arange(cells//2+1)/(cells//2)).tolist())

def test_regional_p0_constant_density_retained_torque_origin_shared_node():
    from scripts.r8_f03_current_family_physical_bridge_dev import p0_region_masks,p0_regional_actions,reference_wrench
    from scripts.b3_e4_bl5_reference_actions_dev import physical_loads
    from scripts.r8_bl5_actual_mref_dev import _candidate_rhs
    a=view(16);a._beam_parts=(None,)*8;a._beam_nodes={n:(np.array((i/16,0.,.0675)),None) for i,n in enumerate(a._beam_ids)}
    sx=np.arange(17)/16;sy=np.array((0.,.2));scales=np.repeat((1.,1.,1.,.4,.4),8);columns=[]
    for col in np.diag(scales).T:
        br,sr,_=physical_loads(col.reshape(5,8).T,np.eye(3),np.array((0.,0.,1.)),np.array(((1.,0.,0.),(0.,1.,0.))),
            np.array((0.,0.,-.05)),np.array((0.,0.,.0175)),sx,sx,1,16,2)
        columns.append(_candidate_rhs(a,sx,sy,br,sr))
    loads=np.column_stack(columns);lm=np.zeros(40);lm[8:16]=1.
    masks=p0_region_masks(regional_policy(16),16);actions=p0_regional_actions(loads,lm,scales,masks)
    np.testing.assert_allclose(sum(actions[n] for n in ('Q1','Q2','Q3','Q4')),actions['whole'],rtol=0.,atol=1e-16)
    np.testing.assert_allclose(actions['Q2']+actions['Q3'],actions['interior'],rtol=0.,atol=1e-16)
    for i,name in enumerate(('Q1','Q2','Q3','Q4')):
        w=reference_wrench(a,actions[name],np.zeros(3),'beam')
        np.testing.assert_allclose(w,(0.,.25,0.,-.0175*.25,0.,(2*i+1)/32),rtol=0.,atol=1e-16)
        shift=np.array((.3,-.2,.4));ws=reference_wrench(a,actions[name],shift,'beam')
        np.testing.assert_allclose(ws,np.r_[w[:3],w[3:]-np.cross(shift,w[:3])],rtol=0.,atol=1e-16)
        assert np.max(np.abs(reference_wrench(a,actions[name],np.zeros(3))))<1e-16
    retained=a._shell_count+6*len(a._beam_ids)
    assert np.sum(actions['Q1'][retained:].reshape(8,24)[:,:6])>0. # actual nonzero retained moment
    # Cropping globally assembled beam nodal rows incorrectly includes the outside
    # half of each shared boundary node. It overcounts the true interior force.
    whole=actions['whole'];node_rows=a._shell_count+6*np.arange(4,13)+1
    assert np.sum(whole[node_rows])>.5
    assert reference_wrench(a,actions['interior'],np.zeros(3),'beam')[1]==.5

@pytest.mark.parametrize('cells',(16,32,64))
def test_regional_p0_row_order_and_refusals(cells):
    from scripts.r8_f03_current_family_physical_bridge_dev import p0_region_masks,p0_regional_actions
    policy=regional_policy(cells);m=p0_region_masks(policy,cells);supports=cells//2
    assert np.flatnonzero(m['Q1']).tolist()==[c*supports+j for c in range(5) for j in range(supports//4)]
    assert np.array_equal(m['Q2']|m['Q3'],m['interior'])
    bad=dict(policy,support_membership=list(reversed(policy['support_membership'])))
    with pytest.raises(ValueError,match='membership'):p0_region_masks(bad,cells)
    bad=dict(policy,support_endpoints_m=policy['support_endpoints_m'].copy());bad['support_endpoints_m'][1]+=1e-6
    with pytest.raises(ValueError,match='endpoints'):p0_region_masks(bad,cells)
    with pytest.raises(ValueError,match='finite regional'):p0_regional_actions(np.zeros((2,5*supports)),np.zeros(5*supports),np.zeros(5*supports),m)

CURRENT16_ARCHIVE=Path(__file__).resolve().parents[1]/'docs/agent_plans/development/r8-f03-sparse-signed-20261009/raw-current16-v1.zip'
CURRENT16_ARCHIVE_SHA='4cd598d482a0bd6e6eb4f67fccfe99bb2c8380c547a9e3392c134da86048f8e2'
CURRENT16_MEMBER='.development/r8-f03-sparse-signed-20261009/current16-v1/result.json'
CURRENT16_MEMBER_SHA='a940525e3652e3b706893bc43fd9a2fbda8ef04b1ea752532e2d1905dc883515'

def historical16_record(tmp_path,archive=None):
    """Read-only current16 reader through a pytest tmp fixture. The protected
    archive and the absent .development member are never rewritten; member
    metadata is verified against the committed archive manifest, not invented.
    An absent archive skips only saved-data controls; an existing bad archive
    or member path fails."""
    import hashlib,zipfile
    location=Path(archive) if archive is not None else CURRENT16_ARCHIVE
    if not location.exists():
        pytest.skip('current16 archive absent: %s'%location)
    assert sha(location)==CURRENT16_ARCHIVE_SHA,'current16 archive identity drift'
    manifest=json.loads((Path(__file__).resolve().parents[1]/'docs/agent_plans/development/r8-f03-sparse-signed-20261009/archive-members-current16-v1.json').read_text())
    with zipfile.ZipFile(location) as z:
        names=set(z.namelist())
        assert names==set(manifest['members']) and manifest['count']==len(names),'current16 archive member set drift'
        for name,digest in manifest['members'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest,'current16 archive member drift: %s'%name
        z.extract(CURRENT16_MEMBER,tmp_path)
    record=tmp_path/CURRENT16_MEMBER
    assert sha(record)==CURRENT16_MEMBER_SHA,'current16 record identity drift'
    return json.loads(record.read_text())

def test_current16_reader_skip_and_bad_archive_contract(tmp_path):
    # Absent archive skips only saved-data controls; an existing bad archive
    # fails; the protected archive is never rewritten.
    with pytest.raises(pytest.skip.Exception,match='current16 archive absent'):
        historical16_record(tmp_path,tmp_path/'absent-raw-current16-v1.zip')
    bad=tmp_path/'bad-raw-current16-v1.zip';bad.write_bytes(b'not the exact current16 archive')
    with pytest.raises(AssertionError,match='archive identity drift'):
        historical16_record(tmp_path,bad)
    if not CURRENT16_ARCHIVE.exists():
        pytest.skip('current16 archive absent; protected-read check is saved-data only')
    before=sha(CURRENT16_ARCHIVE)
    historical16_record(tmp_path)
    assert sha(CURRENT16_ARCHIVE)==before==CURRENT16_ARCHIVE_SHA

def test_saved_record_schema_real16_summary_cannot_mask_metrics(tmp_path):
    from copy import deepcopy
    from scripts.r8_f03_current_family_physical_bridge_dev import admit_saved_record
    r=historical16_record(tmp_path);assert 'physical_parity_pass' not in r
    assert admit_saved_record(r,16)
    good=deepcopy(r);good['physical_parity_pass']=True;assert admit_saved_record(good,16)
    bad=deepcopy(good);bad['owner_metrics']['equilibrium']=1.1e-8
    with pytest.raises(ValueError,match='guards'):admit_saved_record(bad,16)
    bad=deepcopy(good);bad['parity']['multipliers']=1.1e-10
    with pytest.raises(ValueError,match='guards'):admit_saved_record(bad,16)
    for summary in (False,None,1,'true'):
        bad=deepcopy(r);bad['physical_parity_pass']=summary
        with pytest.raises(ValueError,match='summary'):admit_saved_record(bad,16)

@pytest.mark.parametrize('mode',('parity_nan','metric_inf','metric_nan','negative_metric','missing_metric','missing_group','missing_role','invalid_public','wrong_load','wrong_space','wrong_domain','wrong_node'))
def test_saved_record_schema_missing_corrupt_nonfinite_refuses(tmp_path,mode):
    from scripts.r8_f03_current_family_physical_bridge_dev import admit_saved_record
    r=historical16_record(tmp_path);r['physical_parity_pass']=True
    if mode=='parity_nan':r['parity']['shell']=float('nan')
    elif mode=='metric_inf':r['reduced_metrics']['weak_gap']=float('inf')
    elif mode=='metric_nan':r['owner_metrics']['compatible_work_relative']=float('nan')
    elif mode=='negative_metric':r['owner_metrics']['equilibrium']=-1.
    elif mode=='missing_metric':del r['owner_metrics']['ground_work_relative']
    elif mode=='missing_group':del r['reduced_metrics']
    elif mode=='missing_role':del r['parity']['resultants']
    elif mode=='invalid_public':r['public_result']=None
    elif mode=='wrong_load':r['literal_inputs']['action'][1]=1.
    elif mode=='wrong_space':r['operator_identity']['actual_multiplier_space']='BL5_P0_span_4'
    elif mode=='wrong_domain':r['literal_inputs']['action_domain']='shell'
    elif mode=='wrong_node':r['literal_inputs']['action_node']=116
    with pytest.raises(ValueError):admit_saved_record(r,16)

def test_saved_record_schema_real16_actual_caller_reaches_array_boundary(tmp_path,monkeypatch):
    from scripts import r8_f03_current_family_physical_bridge_dev as module
    records=historical16_record(tmp_path);directory=tmp_path/'native';directory.mkdir()
    (directory/'result.json').write_text(json.dumps(records))
    for name in CONSUMED:
        if name!='result.json':(directory/name).write_bytes(b'inert-unloaded-control')
    dummy=tmp_path/'inert.npz';dummy.write_bytes(b'inert-unloaded-control')
    saved={str(n):dict(directory=str(directory),files={name:sha(directory/name) for name in CONSUMED}) for n in (16,32,64)}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(source_inputs={},saved=saved,
        wrench_arrays={str(n):dict(path=str(dummy),sha256=sha(dummy)) for n in (16,32,64)})))
    admitted=[];original=module.admit_saved_record
    def guard(record,level):
        answer=original(record,level);admitted.append(level);return answer
    monkeypatch.setattr(module,'admit_saved_record',guard)
    def stop_before_array(*args,**kwargs):raise RuntimeError('MANUFACTURED_ARRAY_BOUNDARY_NO_EVALUATION')
    monkeypatch.setattr(module.np,'load',stop_before_array)
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='MANUFACTURED_ARRAY_BOUNDARY'):module.run_regional_wrenches(manifest,out)
    assert admitted==[16]
    r=json.loads((out/'result.json').read_text());assert r['status']=='FAILED' and r['cases']=={} and 'MANUFACTURED_ARRAY_BOUNDARY' in r['exception']
