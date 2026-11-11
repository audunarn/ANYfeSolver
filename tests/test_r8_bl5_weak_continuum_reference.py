"""Full-body weak reference development checks; no external/native rerun."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from scripts import r8_bl5_weak_continuum_reference as tool
CHECKED=set()


@pytest.fixture(scope='module',autouse=True)
def prior_prerequisite_evidence():
    path=os.environ.get('ANYSOLVER_WEAK_REUSE')
    if path:
        record=json.loads(Path(path).read_text())
        assert record['state']=='EXACT_PRIOR_PASSES_REUSED_BY_IMPACT'
        CHECKED.update(record['checked'])


@pytest.fixture(scope='module')
def assembled():
    if 'ANYSOLVER_WEAK_RAW' not in os.environ:
        pytest.skip('ANYSOLVER_WEAK_RAW raw-array output binding absent; '
            'the weak-continuum reference fixture writes its raw arrays '
            'only to the designated bound location')
    model=tool.specimen();k,elements=tool.assemble_body(model)
    cs,_=tool.constraints(model,'strong');cw,_=tool.constraints(model,'weak')
    np.savez(Path(os.environ['ANYSOLVER_WEAK_RAW'],'operators-v1.npz'),
        K_data=k.data,K_indices=k.indices,K_indptr=k.indptr,K_shape=k.shape,C_strong=cs.toarray(),C_weak=cw.toarray())
    save('input-v1.json',model)
    return model,k,elements


def save(name,value):
    Path(os.environ['ANYSOLVER_WEAK_RAW'],name).write_text(json.dumps(value,default=tool.jsonable,indent=2)+'\n')


@pytest.mark.parametrize('beam_slot',(55,58,130))
def test_vertical_midside_beam_slot_has_exact_zero_weak_face_trace(beam_slot):
    model=tool.specimen()
    _,index,*_=tool.layout(model)
    beam_nodes=model['beam_nodes']
    slot=index[beam_nodes[beam_slot//3]]+beam_slot%3
    assert model['physical_nodes'][beam_nodes[beam_slot//3]][2]==0.04
    for edge in range(4):
        for alpha in (.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)):
            assert np.array_equal(tool.five_rows(model,(edge+alpha)*.25)[:,slot],np.zeros(5))
    c,_=tool.constraints(model,'weak')
    assert np.array_equal(c.getcol(slot).toarray(),np.zeros((20,1)))


def test_exact_specimen_original_strong_rows(assembled):
    model,k,_=assembled;c,_=tool.constraints(model,'strong');order,index,fixed,free,load=tool.layout(model)
    independent=np.zeros((25,306))
    for i,row in enumerate(row for station in model['stations'] for row in tool.original.physical_rows(station)):
        for (n,j),v in row.items():independent[i,index[n]+j]=v
    np.testing.assert_allclose(c.toarray(),independent,rtol=0,atol=1e-13)
    assert model['model_sha256']=='a22170179b6e257a47288ff825a3b69ad503e5e3c4a040fcd5f3b53933c86463'
    assert k.shape==(306,306) and len(order)==102 and len(fixed)==39 and len(free)==267
    assert np.flatnonzero(load).tolist()==[index[model['load_node']]+2]
    assert load[index[model['load_node']]+2]==1.
    CHECKED.add('geometry')


def test_body_operator_matches_previous_independent_internal_covectors(assembled):
    model,k,_=assembled;order,index,*_=tool.layout(model)
    u={n:1e-7*np.array((p[0]*p[1],p[1]*p[2],p[0]**2+p[2])) for n,p in model['physical_nodes'].items()}
    elements=tool.original.internal_covectors(model,u);forces={n:np.zeros(3) for n in order}
    for field in elements.values():
        for n,f in field.items():forces[n]+=f
    expected=np.concatenate([forces[n] for n in order]);actual=k@np.concatenate([u[n] for n in order])
    assert np.max(abs(actual-expected))/max(np.max(abs(expected)),1.)<=1e-12
    skew=k-k.T
    assert (np.max(abs(skew.data),initial=0)/max(float(abs(k).max()),1.))<=1e-12
    CHECKED.add('body')


def test_weak_free_rank_and_all_six_bar_rigid_motions(assembled):
    model,k,_=assembled;c,scale=tool.constraints(model,'weak');result=tool.rank_checks(model,k,c,scale)
    save('rank-v1.json',result)
    assert c.shape==(20,306) and result['free_row_rank']==20 and result['bar_rigid_motion_rank']==6
    assert max(result['body_rigid_relative_residual'].values())<=1e-12
    CHECKED.add('rank')


def test_actual_common_native_weak_work_and_station_dual(assembled):
    model,k,_=assembled;c,scale=tool.constraints(model,'weak')
    owner=ss.create_analysis(definition=tool.matched.definition())
    try:
        sa=np.zeros((3,2));sa[:,1]=(.13,-.21,.37)
        ba=np.array(((.09,-.11),(-.08,.17),(.02,-.31)))
        q,other,field=tool.matched.global_common_field(owner,sa,np.zeros(3),ba,(.09,.14,-.04),supported=True)
        assert other==model
        ur=np.concatenate([field[n] for n in sorted(field)]);density=np.arange(1.,21.)
        np.testing.assert_allclose(c@ur,-owner._C@q,rtol=0,atol=3e-15)
        native_work=float(density@(owner._C@q));reference_work=float((-density)@(c@ur))
        assert abs(native_work-reference_work)<=1e-13
        cs,_=tool.constraints(model,'strong');concentrated=tool.matched.concentrated_from_density(density.reshape(5,4))
        assert abs(float(concentrated.ravel()@(cs@ur))-native_work)<=1e-13
        save('common-work-v1.json',dict(native_reference_gap_error=float(np.max(abs(c@ur+owner._C@q))),
            native_work=native_work,reference_work=reference_work,external_station_work=float(concentrated.ravel()@(cs@ur)),
            native_rows=20,reference_rows=20,external_strong_rows=25,
            domain='exact globally affine supported material/director common subspace only; arbitrary broken native rotations unmatched'))
        CHECKED.add('common')
    finally:owner.close()


@pytest.mark.parametrize('origin',((0.,0.,0.),(7.,-11.,3.)))
def test_transpose_work_both_body_wrenches_and_normal_spin(assembled,origin):
    model,k,_=assembled;c,scale=tool.constraints(model,'weak');lam=np.arange(1.,21.)
    force=-c.T@lam;nodes=model['physical_nodes'];fields=tool.field_dict(model,force)
    bodywrench={body:tool.matched.common_origin_wrench(nodes,{n:fields[n] for n in model[body+'_nodes']},origin) for body in ('shell','beam')}
    raw=bodywrench['shell']+bodywrench['beam']
    # Root-adopted physical1N/1Nm gate, fixed BEFORE V1. Retain stricter V1 failure.
    assert np.max(abs(raw))<=1e-8
    rigid=tool.rigid_fields(model,'shell')+tool.rigid_fields(model,'beam')
    at_origin=rigid.copy()
    for j in range(3):at_origin[:,3+j]-=rigid[:,:3]@np.cross(np.eye(3)[j],origin)
    coefficient_pair=-(c@at_origin).T@lam;transpose_pair=at_origin.T@force
    assert np.max(abs(raw-coefficient_pair))<=1e-8
    assert np.max(abs(transpose_pair-coefficient_pair))<=1e-8
    assert np.max(abs(c@at_origin))<=1e-8
    eps=np.finfo(float).eps;n=306;gamma=n*eps/(1-n*eps)
    reduction_scale=gamma*sum(float(np.sum(abs(np.cross(np.asarray(nodes[node])-origin,f)))) for node,f in fields.items())
    save('transpose-'+str(origin)+'.json',dict(raw_wrench_residual=raw,direct_rigid_transpose_pairing=transpose_pair,
        stored_coefficient_rigid_pairing=coefficient_pair,
        physical_gate_N_Nm=1e-8,stored_C_common_rigid_residual=float(np.max(abs(c@at_origin))),
        reduction_only_gamma_n_estimate=reduction_scale,
        estimate_scope='diagnostic reduction estimate ONLY; NOT certified whole-chain bound, C/action construction rounding excluded',
        prior_V1_helper_failure_preserved=True))
    u=np.arange(306.)/1024
    assert abs(float(force@u)+float(lam@(c@u)))<=1e-12
    # Compensated normal spin about each material point; normal arm makes position unchanged.
    _,index,*_=tool.layout(model);spin=np.zeros(306)
    for n in model['beam_nodes']:
        _,y,z=nodes[n];spin[index[n]:index[n]+3]=np.cross((0.,0.,.17),(0.,y,z-.04))
    np.testing.assert_allclose(c@spin,0.,atol=1e-15)
    assert abs(float(force@spin))<=1e-13
    CHECKED.add('transpose'+str(origin))


def test_supported_independent_director_shear(assembled):
    model,k,_=assembled;c,_=tool.constraints(model,'weak');order,index,fixed,*_=tool.layout(model)
    u=np.zeros(306)
    for n,p in model['physical_nodes'].items():u[index[n]+2]=.2*p[0]
    assert not np.any(u[fixed])
    np.testing.assert_allclose(c@u,0.,atol=1e-15)
    # Thickness material pairs stay vertical although geometric normal tilts.
    for station in model['stations']:
        bottom,top=station['shell_bottom'],station['shell_top']
        assert np.array_equal(u[index[top]:index[top]+3]-u[index[bottom]:index[bottom]+3],np.zeros(3))
    assert float(u[index[model['load_node']]+2])==.2
    CHECKED.add('shear')


@pytest.fixture(scope='module')
def strong_verified(assembled):
    assert {'geometry','body','rank','common','shear','transpose(0.0, 0.0, 0.0)','transpose(7.0, -11.0, 3.0)'}<=CHECKED,'operator/work prerequisites incomplete; no solve executed'
    model,k,_=assembled;c,scale=tool.constraints(model,'strong')
    saved,u,rf=tool.saved_strong(os.environ['ANYSOLVER_WEAK_SAVED_STRONG'],model)
    result,q,lam=tool.solve(model,k,c,scale,diagnostics=Path(os.environ['ANYSOLVER_WEAK_RAW'],'strong-calculated-not-validated.json'));save('strong-v1.json',result)
    floor_u=max(float(np.max(abs(saved))),1e-12)
    uerror=float(np.max(abs(q-saved)))/floor_u
    # Prior recovered internal actions derive from the saved actual binary64 CCX U.
    prior=json.loads(Path(os.environ['ANYSOLVER_WEAK_SAVED_STRONG'],'parsed-v2.json').read_text())
    saved_g={n:np.zeros(3) for n in u}
    for element in prior['element_internal_covectors'].values():
        for n,f in element.items():saved_g[int(n)]+=np.asarray(f)
    g0=np.concatenate([saved_g[n] for n in sorted(saved_g)]);g=k@q
    gerror=float(np.max(abs(g-g0)))/max(float(np.max(abs(g0))),1.)
    report=dict(saved_U_relative_error=uerror,U_denominator_m=floor_u,
        saved_internal_relative_error=gerror,internal_denominator_N=max(float(np.max(abs(g0))),1.),
        saved_RF_matches_internal_root_max=max(float(np.max(abs((k@saved)[tool.layout(model)[1][n]:tool.layout(model)[1][n]+3]-rf[n]))) for n in rf),
        interpretation='DEV reconstruction of same strong body operator; not external accuracy or equality of raw RF and physical support')
    save('strong-reconstruction-v1.json',report)
    assert uerror<=1e-8 and gerror<=1e-8
    return result


def test_strong_reconstruction_before_weak_solve(strong_verified):
    assert strong_verified['equation_ratio']<=1e-8 and strong_verified['constraint_max']<=1e-9
    CHECKED.add('strong')


def test_one_weak_solve_and_saved_native_diagnostic(assembled,strong_verified):
    assert 'strong' in CHECKED,'strong reconstruction incomplete; weak solve withheld'
    model,k,_=assembled;c,scale=tool.constraints(model,'weak')
    result,q,lam=tool.solve(model,k,c,scale,diagnostics=Path(os.environ['ANYSOLVER_WEAK_RAW'],'weak-calculated-not-validated.json'));save('weak-v1.json',result)
    native=json.loads(Path(os.environ['ANYSOLVER_WEAK_SAVED_NATIVE']).read_text())
    assert tool.SHA(os.environ['ANYSOLVER_WEAK_SAVED_NATIVE'])==tool.NATIVE_CALIBRATION_SHA
    assert native['status']=='completed' and native['identity']=='R8_BL5_MATCHED_NATIVE_REFERENCE_LINEAR_DEV_V1'
    report=dict(interpretation='diagnostic engineering observables, no accuracy score; common weak affine work demonstrated but different body/primal and broken-side spaces remain',
        strong_tip_uz_m=strong_verified['tip_centroid_uz_m'],weak_tip_uz_m=result['tip_centroid_uz_m'],
        saved_native_tip_uz_m=native['applied_virtual_work'],native_load_N=1.,native_calibration_reexecuted=False,
        response_accuracy_accepted=False,remaining='body/primal discretization and arbitrary broken native director trace coverage/convergence')
    save('diagnostic-v1.json',report)
    assert result['equation_ratio']<=1e-8 and result['constraint_max']<=1e-9
    assert result['internal_work_relative_error']<=1e-8 and result['interface_work_relative_error']<=1e-8
    for body,cut in result['domain_cuts'].items():
        residual=np.asarray(cut['internal_wrench'])-np.asarray(cut['interface_wrench'])-np.asarray(cut['applied_wrench'])-np.asarray(cut['support_wrench'])
        assert np.max(abs(residual))<=1e-8,(body,residual)
