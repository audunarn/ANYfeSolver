from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.elements import Element, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_mixed_element import (
    GE_BEAM3_MIXED_CANDIDATE_ID,
    GE_BEAM3_MIXED_FORMULATION_ID,
    GeBeam3MixedStateError,
    GeometricallyExactBeam3D3NElement,
)


def _section(*, coupled: bool = True) -> GeneralizedBeamSection:
    lower = np.array(
        (
            (2, 0, 0, 0, 0, 0),
            (1, 3, 0, 0, 0, 0),
            (0, 1, 2, 0, 0, 0),
            (1, 0, 0, 3, 0, 0),
            (0, 1, 0, 1, 2, 0),
            (0, 0, 1, 0, 1, 2),
        ),
        dtype=float,
    )
    stiffness = lower @ lower.T
    if not coupled:
        stiffness[:3, 3:] = 0.0
        stiffness[3:, :3] = 0.0
    return GeneralizedBeamSection(stiffness, name="mixed-test")


def _model(*, reversed_nodes: bool = False, coupled: bool = True):
    model = FEModel("ge-beam3-mixed")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.5, 0.0, 0.0)
    model.add_node(3, 1.0, 0.0, 0.0)
    element = GeometricallyExactBeam3D3NElement(
        1,
        [3, 2, 1] if reversed_nodes else [1, 2, 3],
        "mat",
        section=_section(coupled=coupled),
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )
    return model, element


def _permute(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(3, 6)[::-1].reshape(18)


def test_candidate_is_private_and_not_a_legacy_quadratic_beam() -> None:
    _model_instance, element = _model()
    assert isinstance(element, Element)
    assert not isinstance(element, QuadraticBeamElement)
    assert element.formulation_id == GE_BEAM3_MIXED_CANDIDATE_ID
    assert GE_BEAM3_MIXED_FORMULATION_ID == GE_BEAM3_MIXED_CANDIDATE_ID


def test_reference_tangent_has_rank_twelve_and_six_rigid_modes() -> None:
    model, element = _model()
    tangent = element.compute_candidate_stiffness_matrix(model.mesh)
    scale = np.sqrt(np.maximum(np.abs(np.diag(tangent)), 1.0))
    normalized = tangent / scale[:, None] / scale[None, :]
    eigenvalues = np.linalg.eigvalsh(0.5 * (normalized + normalized.T))
    assert np.linalg.matrix_rank(normalized, tol=2.0e-10) == 12
    assert np.count_nonzero(np.abs(eigenvalues) < 2.0e-10) == 6
    assert eigenvalues[6] > 1.0e-8
    np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=2.0e-13)


def test_common_finite_rigid_motion_has_zero_energy_and_force() -> None:
    model, element = _model()
    vector = np.array((0.37, -0.21, 0.19))
    from anyfesolver._ge_beam3_mixed_ad import rotation_exponential

    rotation = rotation_exponential(vector)
    translation = np.array((0.8, -0.4, 0.6))
    reference = element.get_node_coordinates(model.mesh)
    displacement = np.zeros((3, 6))
    displacement[:, :3] = reference @ rotation.T + translation - reference
    displacement[:, 3:] = vector
    matrices = np.repeat((rotation @ element.reference_triad(model.mesh))[None, :, :], 3, axis=0)
    displacement[:, 3:] = 0.0
    energy, force, tangent, fields = element.evaluate_candidate(
        model.mesh, displacement.reshape(18), rotation_matrices=matrices, tangent=True
    )
    assert abs(energy) < 2.0e-24
    assert np.linalg.norm(force, ord=np.inf) < 2.0e-11
    assert tangent is not None
    assert fields["local_residual_norm"] < 2.0e-11


def test_reversal_preserves_coupled_section_energy_force_and_tangent() -> None:
    forward_model, forward = _model()
    reverse_model, reverse = _model(reversed_nodes=True)
    values = np.array(
        (
            0.01, -0.03, 0.02, 0.07, -0.04, 0.03,
            0.02, 0.01, -0.01, 0.02, 0.03, -0.05,
            0.05, 0.04, 0.03, -0.01, 0.06, 0.04,
        )
    )
    from anyfesolver._ge_beam3_mixed_ad import rotation_exponential

    forward_frame = forward.reference_triad(forward_model.mesh)
    forward_nodal = values.reshape(3, 6)
    forward_rotations = np.asarray(
        [rotation_exponential(row[3:]) @ forward_frame for row in forward_nodal]
    )
    forward_values = np.array(values, copy=True).reshape(3, 6)
    forward_values[:, 3:] = 0.0
    reverse_frame = reverse.reference_triad(reverse_model.mesh)
    reversed_nodal = values.reshape(3, 6)[::-1]
    reverse_rotations = np.asarray(
        [rotation_exponential(row[3:]) @ reverse_frame for row in reversed_nodal]
    )
    reverse_values = forward_values[::-1].reshape(18)
    ef, ff, kf, _ = forward.evaluate_candidate(
        forward_model.mesh, forward_values.reshape(18), rotation_matrices=forward_rotations
    )
    er, fr, kr, _ = reverse.evaluate_candidate(
        reverse_model.mesh, reverse_values, rotation_matrices=reverse_rotations
    )
    assert kf is not None and kr is not None
    np.testing.assert_allclose(er, ef, rtol=2.0e-12, atol=2.0e-13)
    np.testing.assert_allclose(_permute(fr), ff, rtol=2.0e-10, atol=2.0e-11)
    permutation = np.zeros((18, 18))
    for new_node, old_node in enumerate((2, 1, 0)):
        permutation[new_node * 6 : new_node * 6 + 6, old_node * 6 : old_node * 6 + 6] = np.eye(6)
    np.testing.assert_allclose(permutation.T @ kr @ permutation, kf, rtol=2.0e-9, atol=2.0e-10)


def test_condensed_tangent_agrees_with_directional_energy_second_variation() -> None:
    model, element = _model(coupled=True)
    values = np.array(
        (
            0.002, -0.006, 0.004, 0.03, -0.02, 0.01,
            0.004, 0.008, -0.005, -0.01, 0.025, -0.035,
            0.011, 0.019, 0.013, 0.02, 0.045, 0.03,
        )
    )
    direction = np.linspace(-0.7, 0.8, 18)
    direction /= np.linalg.norm(direction)
    from anyfesolver._ge_beam3_mixed_ad import rotation_exponential

    nodal = values.reshape(3, 6)
    base_rotations = np.asarray([rotation_exponential(row[3:]) for row in nodal])
    base_values = np.array(values, copy=True).reshape(3, 6)
    base_values[:, 3:] = 0.0
    energy, _force, tangent, fields = element.evaluate_candidate(
        model.mesh, base_values.reshape(18), rotation_matrices=base_rotations
    )
    assert tangent is not None
    step = 2.0e-5

    def sampled(sign: float) -> tuple[float, np.ndarray]:
        moved = np.array(base_values, copy=True)
        moved[:, :3] += sign * step * direction.reshape(3, 6)[:, :3]
        rotations = np.asarray(
            [
                rotation_exponential(sign * step * direction.reshape(3, 6)[node, 3:])
                @ base_rotations[node]
                for node in range(3)
            ]
        )
        made_energy, made_force, _made_tangent, _made_fields = element.evaluate_candidate(
            model.mesh,
            moved.reshape(18),
            rotation_matrices=rotations,
            tangent=False,
        )
        return made_energy, made_force

    plus_energy, _plus_force = sampled(1.0)
    minus_energy, _minus_force = sampled(-1.0)
    finite = (plus_energy - 2.0 * energy + minus_energy) / step**2
    analytic = float(direction @ tangent @ direction)
    relative = abs(finite - analytic) / max(1.0, abs(finite))
    assert relative < 1.0e-6
    assert fields["generalized_strain"].shape == (4, 6)
    assert fields["generalized_resultant"].shape == (4, 6)


def test_additive_or_accumulated_rotation_coordinates_fail_closed() -> None:
    model, element = _model()
    values = np.zeros(18)
    values[4] = 1.0e-6
    with np.testing.assert_raises_regex(GeBeam3MixedStateError, "increments, not accumulated"):
        element.evaluate_candidate(model.mesh, values)
    with pytest.raises(GeBeam3MixedStateError, match="solver integration is not authorized"):
        element.compute_internal_forces(model.mesh, np.zeros(18), None)
    with pytest.raises(GeBeam3MixedStateError, match="linear solver integration is not authorized"):
        element.compute_stiffness_matrix(model.mesh, None)
    with pytest.raises(GeBeam3MixedStateError, match="transaction is not implemented"):
        element.compute_nonlinear_response(model.mesh, None, np.zeros(18))


def test_reversal_sensitive_section_requires_physical_axis_authority() -> None:
    model = FEModel("ge-beam3-mixed-missing-axis")
    model.add_material("mat", 1.0, 0.25)
    for node, x in enumerate((0.0, 0.5, 1.0), start=1):
        model.add_node(node, x, 0.0, 0.0)
    from anyfesolver.ge_beam3_mixed_element import GeBeam3MixedGeometryError

    with pytest.raises(GeBeam3MixedGeometryError, match="reference_axis_direction"):
        GeometricallyExactBeam3D3NElement(
            1,
            (1, 2, 3),
            "mat",
            section=_section(coupled=True),
            reference_orientation=(0.0, 1.0, 0.0),
        )


def test_cell_vertex_logarithm_uses_frozen_cutback_domain() -> None:
    from anyfesolver._ge_beam3_mixed_ad import (
        Jet2,
        RotationDomainError,
        constant_matrix,
        rotation_exponential,
        so3_log,
    )

    outside = rotation_exponential((0.0, 0.0, 0.91 * np.pi))
    with pytest.raises(RotationDomainError, match="0.9\\*pi"):
        so3_log(constant_matrix(outside, 1))
    admitted = rotation_exponential((0.0, 0.0, 0.89 * np.pi))
    values = so3_log(constant_matrix(admitted, 1))
    assert all(isinstance(value, Jet2) for value in values)
