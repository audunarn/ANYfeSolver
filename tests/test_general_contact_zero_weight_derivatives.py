"""Zero local weights remove Hessians, never activation or shared support."""
import numpy as np
import pytest
from test_general_contact_nodal_force_linearization import _case


def case(warped):
    law, points = _case()
    if warped:
        points[2,2] += .008
        points[5,2] -= .006
        points[-1,2] += .002
    return law, points


@pytest.mark.parametrize('warped',[False,True])
@pytest.mark.parametrize('mixed',[False,True])
def test_complete_operator_and_activation_limit(warped,mixed):
    law, points=case(warped)
    count=len(law.source_node_ids)
    source,target=points[:count],points[count:]
    z=np.zeros(count)
    if mixed: z[0]=.007 # Only left Q4 weighted; shared nodes stay in both maps.
    law.zero_weight_first_order=False
    full=law.nodal_force_linearization(source,target,z)
    law.zero_weight_first_order=True
    first=law.nodal_force_linearization(source,target,z)
    assert first.gradient_only_pairs==(1 if mixed else 2)
    np.testing.assert_array_equal(first.state.normalized_gap,full.state.normalized_gap)
    for name in ('residual','gap_gradient','tangent'):
        actual,expected=getattr(first,name),getattr(full,name)
        assert np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12)<1e-5
    # At z=0 an increment uses full derivatives. The mixed u/z derivative
    # from the zero-weight branch must match that one-sided activation limit.
    dz=np.zeros(count);dz[4]=1.
    predicted=-first.gap_gradient.T @ dz
    for step in (1e-4,1e-6,1e-8):
        plus=law.nodal_force_linearization(source,target,z+step*dz,tangent=False)
        actual=(plus.residual-first.residual)/step
        assert np.linalg.norm(actual-predicted)/np.linalg.norm(predicted)<1e-5


def test_mixed_pair_force_and_gap_tangent_over_perturbation_interval():
    law,points=case(True);count=len(law.source_node_ids)
    z=np.zeros(count);z[0]=.007
    at=law.nodal_force_linearization(points[:count],points[count:],z)
    direction=np.random.default_rng(91).normal(size=points.size)
    direction/=np.linalg.norm(direction)
    predicted=at.tangent @ direction
    dg=at.gap_gradient @ direction
    for step in (1e-6,3e-7,1e-7):
        plus=(points.ravel()+step*direction).reshape(-1,3)
        minus=(points.ravel()-step*direction).reshape(-1,3)
        rp=law.nodal_force_linearization(plus[:count],plus[count:],z,tangent=False)
        rm=law.nodal_force_linearization(minus[:count],minus[count:],z,tangent=False)
        fd=(rp.residual-rm.residual)/(2*step)
        gapfd=(rp.state.normalized_gap-rm.state.normalized_gap)/(2*step)
        assert np.linalg.norm(fd-predicted)/np.linalg.norm(predicted)<1e-5
        assert np.linalg.norm(gapfd-dg)/np.linalg.norm(dg)<1e-5


def test_recovery_neighbor_retains_full_derivatives_even_without_pair_id():
    law,points=case(False);count=len(law.source_node_ids)
    z=np.zeros(count)
    raw=law._nodal_force_unresolved(points[:count],points[count:],z,
        tangent=True,tiny_wedge_pair=None,allow_gradient_only=False)
    assert raw.gradient_only_pairs==0
    assert np.linalg.norm(raw.gap_gradient)>0.


def test_coupled_coordinate_force_direction_through_zero_weight_activation():
    law,points=case(True);count=len(law.source_node_ids)
    z=np.zeros(count);z[0]=.007
    raw=law.nodal_force_linearization(points[:count],points[count:],z)
    du=np.random.default_rng(27).normal(size=points.size)
    du/=np.linalg.norm(du)
    dz=np.zeros(count);dz[4]=.03
    predicted=raw.tangent @ du-raw.gap_gradient.T @ dz
    for step in (1e-6,3e-7,1e-7):
        plus=(points.ravel()+step*du).reshape(-1,3)
        minus=(points.ravel()-step*du).reshape(-1,3)
        rp=law.nodal_force_linearization(plus[:count],plus[count:],z+step*dz,tangent=False)
        rm=law.nodal_force_linearization(minus[:count],minus[count:],z-step*dz,tangent=False)
        actual=(rp.residual-rm.residual)/(2*step)
        assert np.linalg.norm(actual-predicted)/np.linalg.norm(predicted)<1e-5


def test_complete_physical_skin_map_including_rotations_and_target_motion():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize
    from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3
    from test_general_contact_q4_s3_jet2_static_owner import _settings
    model,definition,settings=_two_q4_two_s3()
    manager=GeneralS3PatchManager(model,definition,_settings(settings))
    manager.law.zero_weight_first_order=True
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(6)[2]]=-.006
    u[model.mesh.dof_manager.get_node_dofs(9)[2]]=.002
    z=np.zeros(len(manager.law.pressure_mode_ids));z[0]=.007
    raw=linearize(manager,u,z)
    assert raw.gradient_only_pairs>0
    du=np.random.default_rng(12).normal(size=u.size)
    du/=np.linalg.norm(du)
    dz=np.zeros_like(z)
    dz[manager.law.pressure_mode_ids.index(3)]=.03
    predicted=raw.tangent @ du-raw.gap_gradient.T @ dz
    assert np.linalg.norm(predicted)>1e-6
    for step in (1e-6,3e-7,1e-7):
        rp=linearize(manager,u+step*du,z+step*dz,tangent=False)
        rm=linearize(manager,u-step*du,z-step*dz,tangent=False)
        actual=(rp.force-rm.force)/(2*step)
        assert np.linalg.norm(actual-predicted)/np.linalg.norm(predicted)<1e-5
