"""Cancellation-resistant elastic S3 pullback, with unchanged local mechanics."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_variational_shell import deformation, response
from anyfesolver._ge_beam3_fibre_section import canonical
from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
from test_stiffened_shell_ge_physical_fibre_linear import _definition


@pytest.mark.parametrize('epsilon', (1e-14,1e-16))
def test_stable_elastic_response_resolves_subspacing_membrane_strain(epsilon):
    owner = ExperimentalGEAnalysis(_definition(False,True))
    try:
        _eid,element,_slots,origin = owner._shell_parts[0]
        reference = np.array([owner._shell_model.mesh.nodes[node].coords()
                              for node in element.node_ids])
        u = np.zeros((3,6));u[:,:3]=epsilon*(reference-reference.mean(axis=0))
        local,_,_ = deformation(reference,u.ravel(),stable_translation=True)
        np.testing.assert_allclose(local,u.ravel(),rtol=1e-12,atol=epsilon*1e-14)
        force,tangent,state = response(owner._shell_model,element,u.ravel(),origin,3,
                                      stable_translation=True)
        material=owner._shell_model.get_material('shell')
        expected=element.compute_stiffness_matrix(owner._shell_model.mesh,material)@u.ravel()
        assert np.linalg.norm(expected)>0.
        assert np.linalg.norm(force-expected)/np.linalg.norm(expected)<1e-12
        assert np.isfinite(tangent).all()
        np.testing.assert_array_equal(state['alpha'],np.zeros_like(state['alpha']))
    finally:
        owner.close()


def test_opt_in_preserves_legacy_response_and_complete_finite_chart_map():
    owner = ExperimentalGEAnalysis(_definition(False,True))
    try:
        _eid,element,_slots,origin = owner._shell_parts[0]
        reference=np.array([owner._shell_model.mesh.nodes[node].coords()
                            for node in element.node_ids])
        u=.01*np.sin(np.arange(18)+.4)
        transform=Rotation.from_rotvec((.7,-.4,.5))
        moved=u.reshape(3,6).copy()
        moved[:,:3]=transform.apply(reference+moved[:,:3])+(.3,-.2,.7)-reference
        moved[:,3:]=(transform*Rotation.from_rotvec(moved[:,3:])).as_rotvec()
        moved=moved.ravel()
        default=response(owner._shell_model,element,moved,origin,3)
        explicit=response(owner._shell_model,element,moved,origin,3,stable_translation=False)
        assert canonical(default)==canonical(explicit)
        stable=response(owner._shell_model,element,moved,origin,3,stable_translation=True)
        for actual,expected in zip(stable[:2],default[:2]):
            assert np.linalg.norm(actual-expected)/max(1.,np.linalg.norm(expected))<1e-10
        for actual,expected in zip(deformation(reference,moved,stable_translation=True),
                                   deformation(reference,moved)):
            assert np.linalg.norm(actual-expected)/max(1.,np.linalg.norm(expected))<1e-12
        with pytest.raises(ValueError,match='translation arithmetic policy'):
            response(owner._shell_model,element,moved,origin,3,stable_translation=1)
    finally:
        owner.close()
