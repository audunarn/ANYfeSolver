"""Mass-property summaries for pre-integrated shell and beam sections."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    BeamElement,
    FEModel,
    GeneralizedBeamSection,
    GeneralizedShellSection,
    QuadraticBeamElement,
    calculate_mass_properties,
    create_shell_element,
    element_mass_points,
)


def _shell_section(
    *,
    mass_per_area: float,
    rotary_inertia_per_area: float,
) -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.diag((2.0e8, 1.5e8, 6.0e7)),
        B=np.zeros((3, 3)),
        D=np.diag((2.0e5, 1.5e5, 6.0e4)),
        As=np.diag((4.0e7, 3.0e7)),
        mass_per_area=mass_per_area,
        rotary_inertia_per_area=rotary_inertia_per_area,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=float,
    )


def _physical_section_mass(
    line_mass: float,
    offset: np.ndarray,
    central_inertia: np.ndarray,
) -> np.ndarray:
    first_moment = line_mass * np.asarray(offset, dtype=float)
    coupling = -_skew(first_moment)
    offset_skew = _skew(np.asarray(offset, dtype=float))
    inertia_at_axis = (
        np.asarray(central_inertia, dtype=float)
        - line_mass * (offset_skew @ offset_skew)
    )
    return np.block(
        [
            [line_mass * np.eye(3), coupling],
            [coupling.T, inertia_at_axis],
        ]
    )


def test_shell_section_mass_and_rotary_metadata_drive_all_summaries() -> None:
    model = FEModel("generalized_shell_mass_properties")
    model.add_material("dummy", 210.0e9, 0.3, density=9999.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    model.add_node(3, 2.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    section = _shell_section(
        mass_per_area=7.0,
        rotary_inertia_per_area=0.3,
    )
    model.add_element(
        1,
        create_shell_element(
            1,
            [1, 2, 3, 4],
            "dummy",
            thickness=0.123,
            shell_section=section,
        ),
    )

    props = calculate_mass_properties(
        model,
        reference_point=(0.0, 0.0, 0.0),
    )
    points, skipped = element_mass_points(model)

    assert sum(mass for mass, _ in points) == pytest.approx(14.0)
    assert skipped == []
    assert props.total_mass == pytest.approx(14.0)
    np.testing.assert_allclose(
        props.center_of_mass,
        (1.0, 0.5, 0.0),
        atol=2.0e-14,
    )
    for value in props.assembled_translation_masses.values():
        assert value == pytest.approx(14.0)

    distributed = np.diag(
        (
            14.0 * 1.0**2 / 12.0,
            14.0 * 2.0**2 / 12.0,
            14.0 * (2.0**2 + 1.0**2) / 12.0,
        )
    )
    expected_center_inertia = distributed + 0.3 * 2.0 * np.eye(3)
    np.testing.assert_allclose(
        props.inertia_tensor_center_of_mass,
        expected_center_inertia,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        props.rigid_body_mass_matrix[3:, 3:],
        props.inertia_tensor_origin,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize("element_type", [BeamElement, QuadraticBeamElement])
def test_physical_coupled_beam_mass_sets_mass_com_and_inertia(
    element_type,
) -> None:
    length = 2.0
    line_mass = 10.0
    offset = np.array((0.0, 0.2, -0.1))
    central_inertia = np.diag((1.0, 2.0, 3.0))
    section_mass = _physical_section_mass(
        line_mass,
        offset,
        central_inertia,
    )
    section = GeneralizedBeamSection(
        stiffness=np.diag((1.0e8, 2.0e7, 3.0e7, 4.0e6, 5.0e6, 6.0e6)),
        mass_matrix=section_mass,
    )

    model = FEModel("generalized_beam_mass_properties")
    model.add_material("dummy", 210.0e9, 0.3, density=9999.0)
    positions = (0.0, length) if element_type is BeamElement else (
        0.0,
        length / 2.0,
        length,
    )
    for node_id, x in enumerate(positions, start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    model.add_element(
        1,
        element_type(
            1,
            list(range(1, len(positions) + 1)),
            "dummy",
            section=section,
        ),
    )

    props = calculate_mass_properties(
        model,
        reference_point=(0.0, 0.0, 0.0),
    )
    points, skipped = element_mass_points(model)

    assert skipped == []
    assert sum(mass for mass, _ in points) == pytest.approx(20.0)
    assert props.total_mass == pytest.approx(20.0)
    np.testing.assert_allclose(
        props.center_of_mass,
        (1.0, 0.2, -0.1),
        rtol=0.0,
        atol=2.0e-14,
    )
    expected_center_inertia = (
        length * central_inertia
        + np.diag(
            (
                0.0,
                line_mass * length**3 / 12.0,
                line_mass * length**3 / 12.0,
            )
        )
    )
    np.testing.assert_allclose(
        props.inertia_tensor_center_of_mass,
        expected_center_inertia,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        props.rigid_body_mass_matrix[3:, 3:],
        props.inertia_tensor_origin,
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_non_spatial_generalized_beam_inertia_fails_scalar_summary() -> None:
    section = GeneralizedBeamSection(
        stiffness=np.diag((1.0e8, 2.0e7, 3.0e7, 4.0e6, 5.0e6, 6.0e6)),
        mass_matrix=np.diag((10.0, 11.0, 12.0, 1.0, 2.0, 3.0)),
    )
    model = FEModel("non_spatial_beam_mass")
    model.add_material("dummy", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(1, [1, 2], "dummy", section=section),
    )

    with pytest.raises(ValueError, match="physical spatial inertia"):
        calculate_mass_properties(model)


def test_beam_section_offset_and_inertia_follow_local_frame_and_reference() -> None:
    length = 2.0
    line_mass = 10.0
    local_offset = np.array((0.0, 0.2, -0.1))
    local_central_inertia = np.diag((1.0, 2.0, 3.0))
    section = GeneralizedBeamSection(
        stiffness=np.diag((1.0e8, 2.0e7, 3.0e7, 4.0e6, 5.0e6, 6.0e6)),
        mass_matrix=_physical_section_mass(
            line_mass,
            local_offset,
            local_central_inertia,
        ),
    )
    model = FEModel("rotated_generalized_beam_mass")
    model.add_material("dummy", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, length, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "dummy",
            {"orientation": (0.0, 1.0, 0.0)},
            section=section,
        ),
    )
    reference = np.array((0.3, -0.4, 0.7))
    props = calculate_mass_properties(model, reference_point=reference)

    # For a beam along global x and requested local z along global y:
    # local (x, y, z) maps to global (x, -z, y).
    rotation = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    expected_center = (
        np.array((length / 2.0, 0.0, 0.0))
        + rotation @ local_offset
    )
    np.testing.assert_allclose(
        props.center_of_mass,
        expected_center,
        rtol=0.0,
        atol=2.0e-14,
    )

    local_line_inertia = (
        length * local_central_inertia
        + np.diag(
            (
                0.0,
                line_mass * length**3 / 12.0,
                line_mass * length**3 / 12.0,
            )
        )
    )
    expected_center_inertia = (
        rotation @ local_line_inertia @ rotation.T
    )
    np.testing.assert_allclose(
        props.inertia_tensor_center_of_mass,
        expected_center_inertia,
        rtol=3.0e-13,
        atol=3.0e-13,
    )

    total_mass = line_mass * length
    offset_from_reference = expected_center - reference
    expected_reference_inertia = (
        expected_center_inertia
        - total_mass
        * (_skew(offset_from_reference) @ _skew(offset_from_reference))
    )
    np.testing.assert_allclose(
        props.rigid_body_mass_matrix[:3, 3:],
        -_skew(total_mass * offset_from_reference),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        props.rigid_body_mass_matrix[3:, 3:],
        expected_reference_inertia,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
