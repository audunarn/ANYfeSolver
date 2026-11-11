"""Verification of SO(3)-objective corotational frames and work-conjugate pullback."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.corotational import (
    _CorotationalReference,
    _corotational_cache,
    _corotational_deformation_state,
    corotational_deformation_jacobian,
    corotational_element_response,
    corotational_work_conjugate_response,
    resolve_corotational_tangent_mode,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
    spatial_force_and_moment_resultants,
)
from anyfesolver.elements import BeamElement, create_shell_element
from anyfesolver.fe_core import FEModel


def _make_beam_model() -> tuple[FEModel, int, BeamElement]:
    model = FEModel("beam_3d")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.4, 0.25, -0.15)
    # Distinct principal second moments Iy != Iz (Astra counterexample regime)
    elem = BeamElement(
        1,
        [1, 2],
        material_name="steel",
        cross_section={
            "area": 0.04 * 0.12,
            "Iy": 0.04 * (0.12**3) / 12.0,
            "Iz": 0.12 * (0.04**3) / 12.0,
            "J": 1.8e-6,
            "shape": "rectangular",
            "b": 0.04,
            "h": 0.12,
        },
    )
    model.add_element(1, elem)
    return model, 1, elem


def _make_q4_shell_model() -> tuple[FEModel, int, object]:
    model = FEModel("shell_q4_e4pl")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.2, 0.05, 0.02)
    model.add_node(3, 1.15, 0.85, 0.04)
    model.add_node(4, -0.05, 0.8, 0.01)
    elem = create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.012)
    model.add_element(1, elem)
    return model, 1, elem


def _make_s3_shell_model() -> tuple[FEModel, int, object]:
    model = FEModel("shell_s3")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.1, 0.1, -0.02)
    model.add_node(3, 0.25, 0.9, 0.03)
    elem = create_shell_element(
        1, [1, 2, 3], "steel", thickness=0.012, formulation="legacy-s3"
    )
    model.add_element(1, elem)
    return model, 1, elem


def _superimpose_se3_motion(
    coords_ref: np.ndarray,
    u_base: np.ndarray,
    rot_vec_global: np.ndarray,
    translation_global: np.ndarray,
) -> np.ndarray:
    """Superimpose a global rigid body motion (Q = exp(rot_vec_global), c) on u_base."""
    num_nodes = coords_ref.shape[0]
    u = np.asarray(u_base, dtype=float).reshape(num_nodes, 6)
    Q = rotation_matrix_from_vector(rot_vec_global)
    c = np.asarray(translation_global, dtype=float).reshape(3)
    x_def = coords_ref + u[:, :3]
    u_rot = np.zeros_like(u)
    for i in range(num_nodes):
        x_new = Q @ x_def[i] + c
        u_rot[i, :3] = x_new - coords_ref[i]
        R_node = rotation_matrix_from_vector(u[i, 3:])
        u_rot[i, 3:] = rotation_vector_from_matrix(Q @ R_node)
    return u_rot.reshape(-1)


def test_astra_beam_mixed_bending_chord_rotation_objectivity() -> None:
    """Directly verify Astra's Iy != Iz mixed-bending + chord-rotation counterexample."""
    model = FEModel("astra_beam")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    elem = BeamElement(
        1,
        [1, 2],
        material_name="steel",
        cross_section={
            "area": 0.04 * 0.12,
            "Iy": 0.04 * (0.12**3) / 12.0,
            "Iz": 0.12 * (0.04**3) / 12.0,
            "J": 1.8e-6,
        },
    )
    model.add_element(1, elem)
    mesh = model.mesh
    material = model.get_material("steel")

    coords = np.asarray(elem.get_node_coordinates(mesh), dtype=float)
    ref = _CorotationalReference(elem, model)

    # Equal mixed bending rotations R1 = R2 = exp((0, b, c)) with chord along e_x
    u_base = np.zeros((2, 6), dtype=float)
    u_base[0, 3:] = np.array([0.0, 0.08, -0.06])
    u_base[1, 3:] = np.array([0.0, 0.08, -0.06])
    u_base_flat = u_base.reshape(-1)

    u_d_0, R_rig_0, _ = _corotational_deformation_state(ref, elem, u_base_flat)
    f_ref_0, k_ref_0, _ = elem.compute_nonlinear_response(
        mesh, material, u_d_0, None, 5, True
    )
    energy_0 = 0.5 * float(u_d_0 @ k_ref_0 @ u_d_0)

    # Superimpose rigid chord rotation Q = R_x(gamma) for 30 deg and 45 deg
    # plus arbitrary 3D rotations
    for rot_vec in [
        np.array([np.deg2rad(30.0), 0.0, 0.0]),
        np.array([np.deg2rad(45.0), 0.0, 0.0]),
        np.array([0.42, -0.35, 0.51]),
        np.array([-0.61, 0.48, -0.29]),
    ]:
        Q = rotation_matrix_from_vector(rot_vec)
        u_rotated = _superimpose_se3_motion(
            coords, u_base_flat, rot_vec, np.array([0.7, -1.2, 0.4])
        )
        u_d_rot, R_rig_rot, _ = _corotational_deformation_state(ref, elem, u_rotated)
        f_ref_rot, k_ref_rot, _ = elem.compute_nonlinear_response(
            mesh, material, u_d_rot, None, 5, True
        )
        energy_rot = 0.5 * float(u_d_rot @ k_ref_rot @ u_d_rot)

        assert np.linalg.norm(R_rig_rot - Q @ R_rig_0) < 1.0e-14
        assert np.linalg.norm(u_d_rot - u_d_0) < 1.0e-14
        assert np.linalg.norm(f_ref_rot - f_ref_0) / np.linalg.norm(f_ref_0) < 1.0e-12
        assert abs(energy_rot - energy_0) / energy_0 < 1.0e-14


@pytest.mark.parametrize(
    "builder",
    [_make_beam_model, _make_q4_shell_model, _make_s3_shell_model],
    ids=["beam_3d", "shell_q4_e4pl", "shell_s3"],
)
def test_so3_invariance_and_spatial_self_equilibrium(builder) -> None:
    """Verify <1e-13 SO(3) deformation/energy invariance and <1e-11 self-equilibrium."""
    model, elem_id, elem = builder()
    mesh = model.mesh
    material = model.get_material(elem.material_name)
    coords = np.asarray(elem.get_node_coordinates(mesh), dtype=float)
    num_nodes = int(elem.num_nodes)
    ref = _corotational_cache(model)[elem_id]

    rng = np.random.default_rng(20261017)
    u_base = np.zeros((num_nodes, 6), dtype=float)
    u_base[:, :3] = 0.008 * rng.standard_normal((num_nodes, 3))
    u_base[:, 3:] = 0.04 * rng.standard_normal((num_nodes, 3))
    u_base_flat = u_base.reshape(-1)

    u_d_base, R_rig_base, _ = _corotational_deformation_state(ref, elem, u_base_flat)
    f_ref_base, _, _ = elem.compute_nonlinear_response(
        mesh, material, u_d_base, None, 5, True
    )
    energy_base = 0.5 * float(u_d_base @ f_ref_base)

    for rot_deg in [15.0, 30.0, 45.0]:
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        rot_vec = np.deg2rad(rot_deg) * axis
        Q = rotation_matrix_from_vector(rot_vec)
        shift = np.array([1.25, -0.85, 0.65])

        u_rot = _superimpose_se3_motion(coords, u_base_flat, rot_vec, shift)
        u_d_rot, R_rig_rot, deformed_rot = _corotational_deformation_state(
            ref, elem, u_rot
        )
        f_ref_rot, _, _ = elem.compute_nonlinear_response(
            mesh, material, u_d_rot, None, 5, False
        )
        energy_rot = 0.5 * float(u_d_rot @ f_ref_rot)

        assert np.linalg.norm(R_rig_rot - Q @ R_rig_base) < 1.0e-13
        assert np.linalg.norm(u_d_rot - u_d_base) < 1.0e-13
        assert abs(energy_rot - energy_base) / max(abs(energy_base), 1.0) < 1.0e-13

        p_global, k_global, _ = corotational_work_conjugate_response(
            model, elem_id, elem, u_rot, tangent=True
        )
        assert p_global is not None and k_global is not None

        rotations_rot = u_rot.reshape(num_nodes, 6)[:, 3:]
        for origin in [np.zeros(3), deformed_rot.mean(axis=0), np.array([2.0, -1.0, 0.5])]:
            total_force, total_moment = spatial_force_and_moment_resultants(
                deformed_rot, rotations_rot, p_global, origin=origin
            )
            scale = max(float(np.linalg.norm(p_global)), 1.0)
            assert np.linalg.norm(total_force) / scale < 1.0e-12
            assert np.linalg.norm(total_moment) / scale < 1.0e-11


@pytest.mark.parametrize(
    "builder",
    [_make_beam_model, _make_q4_shell_model, _make_s3_shell_model],
    ids=["beam_3d", "shell_q4_e4pl", "shell_s3"],
)
def test_work_conjugate_jacobian_and_consistent_symmetric_tangent(builder) -> None:
    """Verify B(u) = du_d/du, K(u) = dp/du (<1e-6 FD error), and K == K.T (<1e-12)."""
    model, elem_id, elem = builder()
    mesh = model.mesh
    coords = np.asarray(elem.get_node_coordinates(mesh), dtype=float)
    num_nodes = int(elem.num_nodes)
    ref = _corotational_cache(model)[elem_id]

    rng = np.random.default_rng(42)
    u_local = np.zeros((num_nodes, 6), dtype=float)
    u_local[:, :3] = 0.006 * rng.standard_normal((num_nodes, 3))
    u_local[:, 3:] = 0.035 * rng.standard_normal((num_nodes, 3))
    rot_vec = np.deg2rad(35.0) * np.array([0.6, -0.5, 0.6244997998398398])
    u_elem = _superimpose_se3_motion(
        coords, u_local.reshape(-1), rot_vec, np.array([0.3, -0.4, 0.2])
    )

    u_d, R_rig, deformed = _corotational_deformation_state(ref, elem, u_elem)
    rotations = u_elem.reshape(num_nodes, 6)[:, 3:]
    B = corotational_deformation_jacobian(ref, elem, deformed, rotations, R_rig)

    # 1. Verify B(u) against central differences of _corotational_deformation_state
    eps_b = 1.0e-6
    B_fd = np.zeros_like(B)
    for j in range(u_elem.size):
        u_plus = u_elem.copy()
        u_plus[j] += eps_b
        u_minus = u_elem.copy()
        u_minus[j] -= eps_b
        u_d_plus, _, _ = _corotational_deformation_state(ref, elem, u_plus)
        u_d_minus, _, _ = _corotational_deformation_state(ref, elem, u_minus)
        B_fd[:, j] = (u_d_plus - u_d_minus) / (2.0 * eps_b)

    rel_b_err = np.linalg.norm(B - B_fd) / np.linalg.norm(B_fd)
    assert rel_b_err < 1.0e-9

    # 2. Verify p_global and K_global via corotational_element_response(tangent_mode="work_conjugate")
    p_global, k_global, _ = corotational_element_response(
        model, elem_id, elem, u_elem, tangent=True, tangent_mode="work_conjugate"
    )
    assert p_global is not None and k_global is not None
    assert resolve_corotational_tangent_mode("corotational", "work_conjugate") == "work_conjugate"

    # Symmetry check for hyperelastic local response
    sym_err = np.linalg.norm(k_global - k_global.T) / np.linalg.norm(k_global)
    assert sym_err < 1.0e-12

    # Central finite-difference check of K_global = dp_global / du
    eps_k = 1.0e-5
    k_fd = np.zeros_like(k_global)
    for j in range(u_elem.size):
        u_plus = u_elem.copy()
        u_plus[j] += eps_k
        u_minus = u_elem.copy()
        u_minus[j] -= eps_k
        p_plus, _, _ = corotational_work_conjugate_response(
            model, elem_id, elem, u_plus, tangent=False
        )
        p_minus, _, _ = corotational_work_conjugate_response(
            model, elem_id, elem, u_minus, tangent=False
        )
        assert p_plus is not None and p_minus is not None
        k_fd[:, j] = (p_plus - p_minus) / (2.0 * eps_k)

    rel_k_err = np.linalg.norm(k_global - k_fd) / np.linalg.norm(k_fd)
    assert rel_k_err < 1.0e-7
