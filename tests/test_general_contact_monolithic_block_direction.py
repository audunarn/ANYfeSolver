"""Independent full-matrix check of the frozen active-set block solve."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact.semismooth_block import prepare_frozen_block_response
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


@pytest.mark.parametrize("active_ids", [(), (0,), (1, 3), (0, 1, 2, 3, 4)])
def test_block_direction_matches_full_monolithic_sparse_system(active_ids):
    rng = np.random.default_rng(20260928)
    q_size, mu_size, modes = 8, 2, 5
    basis = rng.normal(size=(q_size, q_size))
    kqq = sparse.csr_matrix(basis.T @ basis + 4*np.eye(q_size))
    controls = rng.normal(size=(mu_size, q_size))
    reactions = controls.T.copy()
    gap_q = rng.normal(size=(modes, q_size))
    kqz = sparse.csr_matrix(-gap_q.T)
    residual = rng.normal(size=q_size)
    constraint = rng.normal(size=mu_size)
    gap = rng.normal(size=modes)
    z = np.abs(rng.normal(size=modes))
    c = np.linspace(0.7, 2.1, modes)
    active = np.zeros(modes, dtype=bool)
    active[list(active_ids)] = True

    block = prepare_frozen_block_response(
        kqq=kqq, control_reactions=reactions, control_rows=controls,
        kqz=kqz, gap_q=gap_q, residual=residual,
        constraint=constraint, signature="test:monolithic_block",
    )
    dq, dmu, dz = block.direction(active, z, gap)

    branch_q = np.zeros_like(gap_q)
    branch_q[active] = c[active, None]*gap_q[active]
    branch_z = np.ones(modes)
    branch_z[active] = 0.0
    branch_residual = np.where(active, c*gap, z)
    direct_matrix = sparse.bmat((
        (kqq, sparse.csr_matrix(-reactions), kqz),
        (sparse.csr_matrix(controls), sparse.csr_matrix((mu_size, mu_size)),
         sparse.csr_matrix((mu_size, modes))),
        (sparse.csr_matrix(branch_q), sparse.csr_matrix((modes, mu_size)),
         sparse.diags(branch_z)),
    ), format="csr")
    direct = spsolve(direct_matrix,
                    np.r_[residual, constraint, -branch_residual])
    np.testing.assert_allclose(np.r_[dq, dmu, dz], direct,
                               rtol=1e-11, atol=1e-11)


def test_block_response_fails_before_dense_resource_overrun():
    with pytest.raises(MemoryError, match="dense budget"):
        prepare_frozen_block_response(
            kqq=sparse.eye(2, format="csr"),
            control_reactions=np.empty((2, 0)),
            control_rows=np.empty((0, 2)),
            kqz=sparse.csr_matrix((2, 3)), gap_q=np.zeros((3, 2)),
            residual=np.zeros(2), constraint=np.zeros(0),
            signature="test:budget", maximum_modes=2,
        )


def test_block_response_rejects_corrupted_coupled_direction():
    block = prepare_frozen_block_response(
        kqq=sparse.eye(2, format="csr"),
        control_reactions=np.empty((2, 0)),
        control_rows=np.empty((0, 2)),
        kqz=sparse.csr_matrix([[-1.0], [0.0]]),
        gap_q=np.array([[1.0, 0.0]]),
        residual=np.array([0.25, 0.0]), constraint=np.empty(0),
        signature="test:backward_error",
    )
    corrupted = replace(block, force_response=2*block.force_response)
    with pytest.raises(ArithmeticError, match="backward-error"):
        corrupted.direction(np.array([True]), np.array([0.1]),
                            np.array([-0.2]))


def test_block_response_factor_memory_bound():
    with pytest.raises(MemoryError, match="dense budget"):
        prepare_frozen_block_response(
            kqq=sparse.eye(4, format="csr"),
            control_reactions=np.empty((4, 0)),
            control_rows=np.empty((0, 4)),
            kqz=sparse.csr_matrix((4, 1)), gap_q=np.zeros((1, 4)),
            residual=np.zeros(4), constraint=np.zeros(0),
            signature="test:factor_budget", maximum_factor_bytes=255,
        )


def test_block_response_matches_direct_on_coupled_shell_steps():
    def run(linear_solver):
        model, definition, settings, program = build_case(4)
        settings = replace(
            settings, enforcement_solver="monolithic_active_set",
            complementarity_force_scale_n=0.1,
            monolithic_linear_solver=linear_solver,
            augmentation_acceleration="none",
            augmentation_schur_maximum_modes=None,
            augmentation_schur_maximum_pressure_pa=None,
            augmentation_schur_maximum_step_pa=None,
            augmentation_schur_difference_step_m=None,
            augmentation_schur_maximum_attempts=1,
            augmentation_schur_maximum_relinearizations=0,
        )
        return solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            emit_restart_checkpoint=True,
        )

    direct = run("direct")
    blocked = run("block_response_dev")
    assert direct.status == blocked.status == "completed"
    assert len(direct.steps) == len(blocked.steps) == 2
    np.testing.assert_allclose(direct.displacements, blocked.displacements,
                               rtol=0, atol=1e-11)
    direct_contact = direct.info["general_contact"]
    block_contact = blocked.info["general_contact"]
    for name in ("integration_force", "minimum_gap", "numerical_pressure_pa"):
        np.testing.assert_allclose(direct_contact[name], block_contact[name],
                                   rtol=0, atol=1e-10)
    assert direct.restart_checkpoint["analysis_fingerprint"] != (
        blocked.restart_checkpoint["analysis_fingerprint"]
    )
