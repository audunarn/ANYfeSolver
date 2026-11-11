"""Independent exact target-force integration with unchanged normal pressure space."""
import numpy as np
import pytest
from dataclasses import replace
from scipy.integrate import quad
from scipy import sparse

from test_general_contact_s3_sliding_friction import _manager, _affine
from test_general_contact_s3_friction_static import _mode
from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionManager
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def _refined(level):
    model,definition,settings,friction,_unused=_manager()
    friction=replace(friction,source_quadrature_refinement=level)
    return model,S3SlidingFrictionManager(model,GeneralS3PatchManager(model,definition,settings),friction)


def test_shifted_triangle_target_force_against_independent_exact_integral(record_property):
    model,manager=_refined(3)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    press=-.02*z
    manager.commit(manager.prepare_commit(press))
    force=manager._friction(press+.39*x)[0]
    # Target node7's shape is max(A-B*bary0,0). On the reference triangle,
    # bary0 has density2(1-bary0). No production projection/integration oracle.
    A,B=.1885,.4
    closed=A*A/B-A**3/(3*B*B)
    numerical=quad(lambda b:2*(1-b)*max(A-B*b,0),0,1,
                   points=(A/B,),epsabs=1e-14,epsrel=1e-13)[0]
    assert numerical==pytest.approx(closed,rel=1e-13)
    expected=-.08*1.5*closed
    actual=force[model.mesh.dof_manager.get_node_dofs(7)[0]]
    relative=abs(actual/expected-1)
    record_property("target_node7_relative_error",relative)
    assert relative <= .002, (actual,expected,relative)
    mean_x,mean_y=1/3+.3885,1/3
    moments=np.array((1-mean_x-mean_y+closed,mean_x-closed,mean_y-closed,closed))
    expected_vector=-.08*1.5*moments
    actual_vector=np.array([force[model.mesh.dof_manager.get_node_dofs(n)[0]] for n in (4,5,6,7)])
    assert np.linalg.norm(actual_vector-expected_vector)/np.linalg.norm(expected_vector)<=.002


def test_reference_quadrature_positive_second_order_moments_and_pressure_space():
    for level in (0,1,2,3):
        model,manager=_refined(level)
        b=np.array([p.barycentric for p in manager.points])
        w=np.array(manager.weights)/.08
        assert len(manager.normal.law.source_node_ids)==3
        assert np.min(b)>0 and np.min(w)>0
        assert w.sum()==pytest.approx(1.,abs=1e-13)
        np.testing.assert_allclose(w@b,np.full(3,1/3),atol=1e-13)
        gram=np.einsum('i,ij,ik->jk',w,b,b)
        expected=np.full((3,3),1/12)
        np.fill_diagonal(expected,1/6)
        np.testing.assert_allclose(gram,expected,atol=1e-13)


def test_refinement_error_and_restart_point_identity_are_explicit(record_property):
    errors=[]
    exact=-.08*1.5*(.1885**2/.4-.1885**3/(3*.4**2))
    for level in (0,1,2,3):
        model,manager=_refined(level)
        x,z=(_mode(model,(1,2,3),a) for a in (0,2))
        press=-.02*z
        manager.commit(manager.prepare_commit(press))
        force=manager._friction(press+.39*x)[0]
        errors.append(abs(force[model.mesh.dof_manager.get_node_dofs(7)[0]]/exact-1))
        state=manager.prepare_commit(press+.39*x)
        manager.commit(state)
        restarted=S3SlidingFrictionManager(model,
            GeneralS3PatchManager(model,manager.normal.definition,manager.normal.settings),manager.friction_settings)
        restarted.restore(manager.checkpoint_state(),state[0])
        np.testing.assert_allclose(restarted._friction(state[0])[0],manager._friction(state[0])[0],atol=1e-12)
        assert restarted.restart_contract()==manager.restart_contract()
    record_property("relative_errors_levels_0_1_2_3",str(errors))
    assert errors[0]>.08
    assert errors[3]<=.002 and errors[3]<errors[2]<errors[0],errors
    # A cut can occupy the same relative position on successive reference grids;
    # the observed levels1/2 plateau is not a strict convergence-order claim.
    assert np.max(np.diff(errors))<=1e-12,errors
    model,definition,settings,friction,_unused=_manager()
    with pytest.raises(MemoryError,match="material-point cap"):
        S3SlidingFrictionManager(model,GeneralS3PatchManager(model,definition,settings),
            replace(friction,source_quadrature_refinement=3,maximum_material_points=100))


def test_local_work_maps_keep_unrelated_coordinates_and_sparse_blocks_empty():
    model,definition,settings,friction,_unused=_manager()
    model.add_node(8,10.,10.,10.)
    friction=replace(friction,source_quadrature_refinement=1,
        steps_by_dof=(2e-7,)*model.mesh.dof_manager.total_dofs)
    manager=S3SlidingFrictionManager(model,GeneralS3PatchManager(model,definition,settings),friction)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    press=-.02*z
    manager.commit(manager.prepare_commit(press))
    force,K,_certificate=manager.evaluate(press+.39*x)
    unused=model.mesh.dof_manager.get_node_dofs(8)
    np.testing.assert_array_equal(force[unused],0)
    assert K[unused].nnz==0 and K[:,unused].nnz==0
    assert all(p.shape==(3,len(manager.work_columns)) for p in manager.point_maps)


def test_cached_work_matches_full_maps_and_restriction_preserves_reactions():
    model,manager=_refined(1)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    press=-.02*z
    manager.commit(manager.prepare_commit(press))
    u=press+.39*x+.01*_affine(model,np.array(((1.1,.2),(.05,1.03))))
    corners,n,_frame,_origin=manager._patch(u)
    full_dn=manager._normal_variation(corners,n)
    local_dn=manager._normal_variation(corners,n,local=True)
    np.testing.assert_allclose(local_dn,full_dn[:,manager.work_columns],atol=1e-13)
    force,_states,records=manager._friction(u)
    for j,(point,record) in enumerate(zip(manager.points,records)):
        element,bary=record["target_element"],record["target_barycentric"]
        full=manager._work(point,u,element,bary,n,full_dn)
        local=manager._work(point,u,element,bary,n,local_dn,local_index=j,corners=corners)
        np.testing.assert_allclose(local[0],full[0],atol=1e-13)
        np.testing.assert_allclose(local[1],full[1][:,manager.work_columns],atol=1e-13)
        np.testing.assert_allclose(local[2],full[2],atol=1e-13)
    constrained=model.mesh.dof_manager.get_node_dofs(7)[0]
    active=np.array([i for i in range(manager.n_dof) if i!=constrained])
    transform=sparse.coo_matrix((np.ones(len(active)),(active,np.arange(len(active)))),
        shape=(manager.n_dof,len(active))).tocsr()
    manager.restrict_tangent_to_reduced_variations(transform)
    residual,K,_certificate=manager.evaluate(u)
    np.testing.assert_allclose(residual,manager.normal.evaluate(u,tangent=False)[0]+force,atol=1e-12)
    normal_K=manager.normal.evaluate(u)[1]
    friction_K=K-normal_K
    friction_K.eliminate_zeros()
    # The normal owner restricts physical-coordinate derivatives, then the
    # driver applies T.T@K@T. Only friction's explicit global-column restriction
    # guarantees an empty global column here; normal columns can remain.
    assert abs(residual[constrained])>1e-5 and friction_K[:,constrained].nnz==0
    translations=[_mode(model,nodes,0) for nodes in ((1,2,3),(4,5,6,7))]
    assert abs(translations[0]@force)>1e-3
    assert translations[0]@force==pytest.approx(-translations[1]@force,abs=1e-12)
