"""Independent equilibrated-response control on a partial Q4/S3 shell patch."""

import numpy as np
from scipy import sparse
from scipy.optimize import root

from anyfesolver.general_contact import linearized_pivot
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3


def test_partial_patch_schur_matches_independent_reequilibration(monkeypatch):
    model, definition, settings = _two_q4_two_s3()
    # 0.45 keeps the same partial overlap while staying clear of the
    # exactly corner-aligned 0.5 shift, where the projected clip
    # classification is marginal and the response topology guard refuses.
    for node in (7, 8, 9, 10):
        model.mesh.get_node(node).x -= 0.45
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
    assert 2 <= len(active) <= 4
    np.testing.assert_array_equal(active, supported)
    spring = np.diag([2.0e4, 3.0e4])
    spring_full = transform @ sparse.csr_matrix(spring) @ transform.T
    captured = {}

    def capture(_pressure, _gap, _active, response, **_kwargs):
        captured["schur"] = response
        np.testing.assert_array_equal(_active, active)
        return None

    monkeypatch.setattr(linearized_pivot, "propose_active_set_pivot", capture)
    linearized_pivot.refined_patch_pivot(
        manager, base_u, contact_tangent + spring_full,
        transform, np.zeros((0, len(free))),
        difference_step=2.0e-6, maximum_pressure=100.0,
        pressure_tolerance=0.02,
    )
    assert "schur" in captured

    def equilibrated_gap(multipliers):
        manager.set_normal_multipliers(multipliers)

        def equilibrium(free_u):
            u = base_u + np.asarray(transform @ free_u).reshape(-1)
            force, _, result = manager.evaluate(u, tangent=False)
            assert result.active_pressure_dofs == len(active)
            return spring @ free_u + np.asarray(transform.T @ (force-base_force)).reshape(-1)

        solve = root(equilibrium, np.zeros(len(free)), tol=1.0e-10)
        assert solve.success, solve.message
        u = base_u + np.asarray(transform @ solve.x).reshape(-1)
        source, target = manager._positions(u)
        return manager.law.evaluate(source, target, multipliers).normalized_gap[active]

    step = 1.0e-3
    columns = []
    for index in active:
        axis = np.zeros_like(base_lambda)
        axis[index] = step
        columns.append((equilibrated_gap(base_lambda+axis)
                        - equilibrated_gap(base_lambda-axis))/(2.0*step))
    np.testing.assert_allclose(captured["schur"], np.column_stack(columns),
                               rtol=2.0e-3, atol=2.0e-8)


def test_marginal_clip_pose_refuses_response_before_pivot(monkeypatch):
    model, definition, settings = _two_q4_two_s3()
    # The exactly corner-aligned 0.5 shift leaves the projected clip
    # classification marginal, so the production topology guard must refuse
    # the finite-difference response before any pivot is proposed.
    for node in (7, 8, 9, 10):
        model.mesh.get_node(node).x -= 0.5
    manager = GeneralS3PatchManager(model, definition, settings)
    base_u = np.zeros(manager.n_dof)
    for node in range(1, 7):
        base_u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    manager.set_normal_multipliers(np.full(6, 10.0))
    free = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1, 5)]
    transform = sparse.eye(manager.n_dof, format="csr")[:, free]
    manager.restrict_tangent_to_reduced_variations(transform)
    _force, contact_tangent, state = manager.evaluate(base_u)
    active = np.flatnonzero(np.asarray(state.numerical_pressure_pa) > 0.0)
    assert 2 <= len(active) <= 4
    captured = {}

    def capture(*_args, **_kwargs):
        captured["pivot"] = True

    monkeypatch.setattr(linearized_pivot, "propose_active_set_pivot", capture)
    pivot = linearized_pivot.refined_patch_pivot(
        manager, base_u, contact_tangent, transform, np.zeros((0, len(free))),
        difference_step=2.0e-6, maximum_pressure=100.0,
        pressure_tolerance=0.02,
    )
    assert pivot is None
    assert not captured
