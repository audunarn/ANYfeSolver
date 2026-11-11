"""Independent small coupled response control for differentiated Q4/S3 overlap."""

from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.optimize import root

from anyfesolver.general_contact.linearized_pivot import _constrained_gap_response
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3


def test_jet2_partial_patch_gap_response_matches_reequilibration():
    model, definition, settings = _two_q4_two_s3()
    for node in (7, 8, 9, 10):
        model.mesh.get_node(node).x -= 0.5
    settings = replace(
        settings,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    base_u = np.zeros(manager.n_dof)
    for node in range(1, 7):
        base_u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    base_lambda = np.full(6, 10.0)
    manager.set_normal_multipliers(base_lambda)
    free = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1, 5)]
    transform = sparse.eye(manager.n_dof, format="csr")[:, free]
    manager.restrict_tangent_to_reduced_variations(transform)
    base_force, contact_tangent, state = manager.evaluate(base_u)
    supported = np.flatnonzero(np.asarray(state.reference_support_m2) > 0.0)
    active = np.flatnonzero(np.asarray(state.numerical_pressure_pa) > 0.0)
    np.testing.assert_array_equal(active, supported)
    assert 2 <= len(active) <= 4
    spring = np.diag([2.0e4, 3.0e4])
    spring_full = transform @ sparse.csr_matrix(spring) @ transform.T
    response = _constrained_gap_response(
        manager, base_u, contact_tangent+spring_full, transform,
        np.zeros((0, len(free))), supported, active,
        difference_step=2e-6,
    )
    assert response is not None

    def equilibrated_gap(multipliers):
        manager.set_normal_multipliers(multipliers)

        def equilibrium(free_u):
            u = base_u + np.asarray(transform @ free_u).reshape(-1)
            force, _, result = manager.evaluate(u, tangent=False)
            assert result.active_pressure_dofs == len(active)
            return spring @ free_u + np.asarray(transform.T @ (force-base_force)).reshape(-1)

        solved = root(equilibrium, np.zeros(len(free)), tol=1e-10)
        assert solved.success, solved.message
        u = base_u + np.asarray(transform @ solved.x).reshape(-1)
        source, target = manager._positions(u)
        return manager.law.evaluate(source, target, multipliers).normalized_gap[active]

    step = 1e-3
    columns = []
    for index in active:
        axis = np.zeros_like(base_lambda)
        axis[index] = step
        columns.append((equilibrated_gap(base_lambda+axis)
                        - equilibrated_gap(base_lambda-axis))/(2*step))
    np.testing.assert_allclose(response, np.column_stack(columns),
                               rtol=2e-3, atol=2e-8)
