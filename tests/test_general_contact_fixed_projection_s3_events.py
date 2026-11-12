"""Independent fixed-projection pressure integrals and constrained general work."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import FEModel,create_shell_element,ContactSurface,ContactDefinition,ContactInteraction
from anyfesolver.general_contact.core import ContactSolverSettings
from anyfesolver.general_contact.fixed_projection_s3_events import prepare_fixed_projection_s3_events
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.shell_pair_hermite_sweep import shell_pair_hermite_lower

GAP=.0002;KN=3.e8
SUPPORT=np.array([1/3,1/6,1/3,1/6])
X_BAR=np.array([.375,.75,.625,.25])-1/3
B=np.column_stack((np.ones(4),-np.ones(4),-X_BAR))


def build():
    model=FEModel('fixed-projection-nonuniform-target')
    model.add_material('elastic',1.e7,.3,density=1000.)
    coordinates=((0,0,.0202),(1,0,.0202),(1,1,.0202),(0,1,.0202),
        (-2,-2,0),(5,-2,0),(-2,5,0))
    for node,point in enumerate(coordinates,1):model.add_node(node,*point)
    for eid,nodes in ((1,[1,2,3]),(2,[1,3,4]),(3,[5,6,7])):
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.02,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.)))
    definition=ContactDefinition((ContactSurface('source',(1,2),side='bottom'),
        ContactSurface('target',(3,),side='top')),(),
        (ContactInteraction('pair','source','target',KN,2.e-5,normal_law='projected_penalty'),))
    settings=ContactSolverSettings(pair_motion='general',general_s3_difference_step=1.e-5,
        general_s3_differentiation='jet2_s3_dev')
    manager=GeneralS3PatchManager(model,definition,settings)
    modes=np.zeros((42,3))
    for node,point in enumerate(coordinates,1):
        normal=model.mesh.dof_manager.get_node_dofs(node)[2]
        if node<=4:modes[normal,0]=1.
        else:modes[normal,1:]=[1.,point[0]-1/3]
    return model,manager,modes


@pytest.mark.parametrize('sparse_modes',[False,True])
def test_shared_integrals_and_pressure_modes_have_independent_affine_oracle(sparse_modes):
    model,manager,modes=build()
    contract=prepare_fixed_projection_s3_events(model,manager,sparse.csr_matrix(modes) if sparse_modes else modes)
    np.testing.assert_allclose(contract.support,SUPPORT,rtol=2.e-13,atol=1.e-15)
    np.testing.assert_allclose(contract.reference_gap,GAP,rtol=0.,atol=1.e-17)
    np.testing.assert_allclose(contract.reduced_gradient,B,rtol=1.e-13,atol=1.e-14)
    assert contract.features==tuple(('pair',node) for node in (1,2,3,4))
    assert not contract.support.flags.writeable and not contract.reduced_gradient.flags.writeable


@pytest.mark.parametrize('q',[(0.,0.,0.),(-.00023,-.00001,1.e-5),(-.00019,0.,4.e-5)])
def test_nonuniform_target_preserves_complete_general_constrained_work(q):
    model,manager,modes=build();q=np.array(q)
    contract=prepare_fixed_projection_s3_events(model,manager,modes)
    residual,tangent,result=manager.evaluate(modes@q,tangent=True)
    contract.require_candidate(modes@q,q,result)
    gap=GAP+B@q;closing=np.minimum(gap,0.)
    np.testing.assert_allclose(modes.T@residual,KN*B.T@(SUPPORT*closing),rtol=2.e-11,atol=1.e-8)
    np.testing.assert_allclose(modes.T@tangent@modes,KN*B.T@np.diag(SUPPORT*(gap<0.))@B,rtol=2.e-12,atol=1.e-6)
    assert result.penalty_energy==0. # Existing projected work is not a full-space potential.
    assert result.regularization_storage_j==pytest.approx(.5*KN*float(SUPPORT@closing**2),rel=1.e-11,abs=1.e-13)
    np.testing.assert_allclose(np.array(result.integration_force)+result.target_force,0.,atol=2.e-8)
    np.testing.assert_allclose(np.array(result.integration_moment)+result.target_moment,0.,atol=2.e-8)
    if np.any(closing):
        direction=np.array([.1,-.2,.3]);step=1.e-8
        plus=manager.evaluate(modes@(q+step*direction),tangent=False)[0]
        minus=manager.evaluate(modes@(q-step*direction),tangent=False)[0]
        actual=modes.T@(plus-minus)/(2*step)
        expected=modes.T@tangent@modes@direction
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1.e-5


@pytest.mark.parametrize('kind',['source','tangent','rotation'])
def test_unadmitted_constraint_generator_is_rejected(kind):
    model,manager,modes=build();modes=modes.copy()
    if kind=='source':modes[2,2]=.1
    elif kind=='tangent':modes[0,2]=.1
    else:modes[3,2]=.1
    with pytest.raises(ValueError,match='fixed-projection'):
        prepare_fixed_projection_s3_events(model,manager,modes)


def test_equivalent_mode_scales_and_prescribed_normal_translation():
    model,manager,modes=build();scales=np.array([1.e-25,1.e25,3.])
    ordinary=prepare_fixed_projection_s3_events(model,manager,modes)
    q=np.array([-.00023,-.00001,1.e-5]);prescribed=modes@np.array([1.e-6,-2.e-6,0.])
    scaled=prepare_fixed_projection_s3_events(model,manager,modes*scales,prescribed=prescribed)
    np.testing.assert_allclose(scaled.reduced_gradient/scales,ordinary.reduced_gradient,atol=1.e-14)
    np.testing.assert_allclose(scaled.gaps(q/scales),ordinary.gaps(q)+3.e-6,atol=1.e-17)


@pytest.mark.parametrize('field',['identity','gap','support','motion'])
def test_same_pose_pressure_agreement_fails_closed(field):
    model,manager,modes=build();contract=prepare_fixed_projection_s3_events(model,manager,modes)
    q=np.array([-.00023,-.00001,1.e-5]);u=modes@q
    result=manager.evaluate(u,tangent=False)[2]
    if field=='identity':result=replace(result,source_pressure_node_ids=(2,1,3,4))
    elif field=='gap':result=replace(result,normalized_weighted_gap_m=tuple(np.array(result.normalized_weighted_gap_m)+1.e-8))
    elif field=='support':result=replace(result,reference_support_m2=tuple(np.array(result.reference_support_m2)*1.001))
    else:u[2]+=.00001
    with pytest.raises(ValueError,match='fixed-projection'):
        contract.require_candidate(u,q,result)


@pytest.mark.parametrize('kind',['nan','complex','boolean','empty','zeros','shape'])
def test_invalid_constraint_data_rejected(kind):
    model,manager,modes=build()
    invalid={'nan':modes+np.nan,'complex':modes.astype(complex),'boolean':modes.astype(bool),
        'empty':modes[:,:0],'zeros':modes*0.,'shape':modes[:-1]}
    with pytest.raises(ValueError,match='constraint subspace'):
        prepare_fixed_projection_s3_events(model,manager,invalid[kind])


def test_epsilon_near_source_generators_do_not_establish_exact_affine_events():
    model,manager,modes=build();first=modes[:,0];second=first.copy();second[2]+=1.e-14
    represented=np.column_stack((first,second))
    with pytest.raises(ValueError,match='uniform source'):
        prepare_fixed_projection_s3_events(model,manager,represented)


def test_resource_cap_and_wrong_owner_are_not_inferred_capabilities():
    model,manager,modes=build()
    with pytest.raises(ValueError,match='Jet2 projected penalty'):
        prepare_fixed_projection_s3_events(model,SimpleNamespace(),modes)
    manager.settings=replace(manager.settings,maximum_patch_normal_dofs=3)
    with pytest.raises(MemoryError,match='no positive support'):
        prepare_fixed_projection_s3_events(model,manager,modes)


@pytest.mark.parametrize('kind',['nonfinite_gradient','overflow_gap','overflow_scale'])
def test_nonfinite_derived_pressure_cannot_pass_candidate_comparisons(kind):
    model,manager,modes=build();contract=prepare_fixed_projection_s3_events(model,manager,modes)
    q=np.zeros(3);result=manager.evaluate(modes@q,tangent=False)[2]
    gradient=contract.reduced_gradient.copy()
    if kind=='nonfinite_gradient':gradient[0,0]=np.inf
    elif kind=='overflow_gap':gradient[:,0]=1.e308;q[0]=2.
    else:gradient[:,0]=1.e308;gradient[:,1]=-1.e308;q[:2]=1.
    broken=replace(contract,reduced_gradient=gradient)
    with pytest.raises(ValueError,match='not finite'):
        broken.require_candidate(modes@q,q,result)


def test_finite_generators_with_unrepresentable_pullback_fail_closed():
    model,manager,modes=build()
    huge=modes.copy();huge[:,0]*=1.e308
    # A finite physical generator must not make inf/NaN integral gradients pass
    # through comparisons or into later pressure-event logic.
    try:
        result=prepare_fixed_projection_s3_events(model,manager,huge)
    except ValueError as exc:
        assert 'not finite' in str(exc)
    else:
        assert np.all(np.isfinite(result.reduced_gradient))


def test_negligible_source_tangent_can_hide_large_quadratic_gap():
    model,manager,modes=build()
    delta=2.**-46
    mode=modes[:,2].copy()+X_BAR[0]*modes[:,0]
    for node in (1,2,3,4):mode[model.mesh.dof_manager.get_node_dofs(node)[0]]=delta
    # With q(t)=4Q*t*(1-t), Q=2**19, the exact pressure gap of row0 is
    # GAP-delta*q(t)**2. Its endpoints agree with the affine prediction, but
    # its midpoint decreases 2**-8 m. No epsilon membership may certify it.
    assert GAP-delta*(2.**19)**2<0.
    with pytest.raises(ValueError,match='exact normal'):
        prepare_fixed_projection_s3_events(model,manager,mode[:,None])


@pytest.mark.parametrize('column',[0,3,2])
def test_integer_minimum_motion_cannot_wrap_into_zero_scale(column):
    model,manager,modes=build();contract=prepare_fixed_projection_s3_events(model,manager,modes)
    value=np.zeros(42,dtype=np.int64);value[column]=np.iinfo(np.int64).min
    with pytest.raises(ValueError,match='fixed-projection'):
        contract.require_motion(value)


def test_internal_array_ownership_and_invalid_stored_support():
    model,manager,modes=build();contract=prepare_fixed_projection_s3_events(model,manager,modes)
    with pytest.raises(ValueError):contract.support.setflags(write=True)
    result=manager.evaluate(np.zeros(42),tangent=False)[2]
    broken=replace(contract,support=np.full(4,np.nan))
    with pytest.raises(ValueError,match='agreement'):
        broken.require_candidate(np.zeros(42),np.zeros(3),result)


def test_nearly_coplanar_source_is_not_an_exact_affine_event_space():
    model,manager,modes=build()
    source=manager.reference[0].copy();source[3,2]+=1.e-16
    manager.reference=(source,manager.reference[1])
    with pytest.raises(ValueError,match='exact common source plane'):
        prepare_fixed_projection_s3_events(model,manager,modes)


def test_nonuniform_skin_path_integrates_existing_work_pressure_and_complete_gap():
    model,manager,modes=build();contract=prepare_fixed_projection_s3_events(model,manager,modes)
    first=np.zeros(3);last=np.array([-.00021,0.,2.e-6])
    v0=np.array([-.1,0.,0.]);v1=np.array([-.1,0.,.002]);dt=.0021
    path=contract.skin_path(manager,modes@first,modes@last,modes@v0,modes@v1)
    source0,source1,target0,target1,av0,av1,bv0,bv1=path
    bounds=[]
    for face in manager.law.source_face_indices:
        result=shell_pair_hermite_lower(source0[face],source1[face],target0,target1,
            av0[face],av1[face],bv0,bv1,dt,gap_tolerance_m=2.e-5)
        assert result.success
        bounds.append(result.lower_gap)
    # Source translates linearly; target tilt grows monotonically to 2e-6.
    # All-pairs conservative minimum is at source end/target x=5.
    expected=GAP+last[0]-last[2]*(5-1/3)
    assert min(bounds)==pytest.approx(expected,abs=1.e-17)
    _,_,row=manager.evaluate(modes@last,tangent=True)
    contract.require_candidate(modes@last,last,row)
    np.testing.assert_allclose(row.normalized_weighted_gap_m,GAP+B@last,atol=1.e-17)


@pytest.mark.parametrize('change',['position','rate'])
def test_actual_skin_rounding_or_work_map_change_cannot_use_affine_events(change,monkeypatch):
    model,manager,modes=build();contract=prepare_fixed_projection_s3_events(model,manager,modes)
    zero=np.zeros(42);last=modes@np.array([-.00021,0.,2.e-6])
    if change=='position':
        original=manager._positions
        def perturbed(value):
            source,target=original(value);source=source.copy();source[0,2]+=1.e-16
            return source,target
        monkeypatch.setattr(manager,'_positions',perturbed)
    else:
        mapping=manager.map.tolil();mapping[0,2]=1.e-16;manager.map=mapping.tocsr()
    with pytest.raises(ValueError,match='represented'):
        contract.skin_path(manager,zero,last,modes@np.array([-.1,0.,0.]),zero)
