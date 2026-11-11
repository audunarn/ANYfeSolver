from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.corotational as corotational
from anyfesolver.corotational import corotational_element_response, rotation_matrix_from_vector
from anyfesolver.corotational_performance import (
    corotational_performance_status,
    reset_corotational_performance_status,
    rotate_corotational_force_blocks,
    rotate_corotational_force_tangent_blocks,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement, create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_performance_bootstrap import nonlinear_performance_status


def _dense_rotation(rotation: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=float)
    for start in range(0, size, 3):
        result[start : start + 3, start : start + 3] = rotation
    return result


@pytest.mark.parametrize("num_nodes", [2, 3, 4])
def test_direct_block_rotation_matches_dense_oracle(num_nodes: int) -> None:
    rng = np.random.default_rng(100 + num_nodes)
    size = 6 * num_nodes
    rotation = rotation_matrix_from_vector(np.asarray([0.4, -0.2, 0.7]))
    force = rng.normal(size=size)
    tangent = rng.normal(size=(size, size))
    dense = _dense_rotation(rotation, size)

    actual_force = rotate_corotational_force_blocks(force, rotation)
    combined_force, actual_tangent = rotate_corotational_force_tangent_blocks(
        force,
        tangent,
        rotation,
    )

    np.testing.assert_allclose(actual_force, dense @ force, rtol=1.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(combined_force, dense @ force, rtol=1.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(
        actual_tangent,
        dense @ tangent @ dense.T,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    "force_shape,tangent_shape",
    [((5,), (5, 5)), ((6,), (5, 5)), ((0,), (0, 0))],
)
def test_direct_block_rotation_rejects_invalid_shapes(
    force_shape: tuple[int, ...],
    tangent_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        rotate_corotational_force_tangent_blocks(
            np.zeros(force_shape),
            np.zeros(tangent_shape),
            np.eye(3),
        )


def _models() -> tuple[FEModel, FEModel, FEModel]:
    beam = FEModel("block_beam")
    beam.add_material("steel", 210.0e9, 0.3)
    beam.add_node(1, 0.0, 0.0, 0.0)
    beam.add_node(2, 1.0, 0.0, 0.0)
    beam.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )

    quadratic = FEModel("block_beam3")
    quadratic.add_material("steel", 210.0e9, 0.3)
    quadratic.add_node(1, 0.0, 0.0, 0.0)
    quadratic.add_node(2, 0.5, 0.0, 0.0)
    quadratic.add_node(3, 1.0, 0.0, 0.0)
    quadratic.add_element(
        1,
        QuadraticBeamElement(
            1,
            [1, 2, 3],
            "steel",
            {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )

    shell = FEModel("block_shell")
    shell.add_material("steel", 210.0e9, 0.3)
    for node_id, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), 1):
        shell.add_node(node_id, float(x), float(y), 0.0)
    shell.add_element(
        1,
        create_element("shell", 1, [1, 2, 3, 4], "steel", thickness=0.01),
    )
    return beam, quadratic, shell


@pytest.mark.parametrize("model_index", [0, 1, 2])
def test_element_rotated_path_reports_direct_blocks(model_index: int) -> None:
    model = _models()[model_index]
    element = model.mesh.elements[1]
    displacement = np.zeros(element.total_dofs, dtype=float)
    displacement[-6] = 1.0e-4
    displacement[-4] = 2.0e-4
    reset_corotational_performance_status()

    force, tangent, _state = corotational_element_response(
        model,
        1,
        element,
        displacement,
        tangent=True,
        tangent_mode="rotated",
    )

    assert force is not None and tangent is not None
    status = corotational_performance_status()
    assert status["fast_path_name"] == "corotational_direct_3x3_blocks"
    assert status["force_block_rotations"] == 1
    assert status["tangent_block_rotations"] == 1
    assert status["dense_consistent_rotations"] == 0


def test_consistent_path_remains_observable_dense_fallback() -> None:
    model = _models()[0]
    element = model.mesh.elements[1]
    displacement = np.zeros(element.total_dofs, dtype=float)
    displacement[-6] = 1.0e-4
    reset_corotational_performance_status()

    force, tangent, _state = corotational_element_response(
        model,
        1,
        element,
        displacement,
        tangent=True,
        tangent_mode="consistent",
    )

    assert force is not None and tangent is not None
    status = corotational_performance_status()
    assert status["force_block_rotations"] == 1
    assert status["tangent_block_rotations"] == 0
    assert status["dense_consistent_rotations"] == 1


def test_fast_path_status_is_included_in_nonlinear_diagnostics() -> None:
    status = nonlinear_performance_status()["corotational"]

    assert status["fast_path_name"] == "corotational_direct_3x3_blocks"
    assert status["backend"]


def test_reference_cache_ignores_load_revision_but_tracks_geometry() -> None:
    model = _models()[2]
    first = corotational._corotational_cache(model)
    centered_first = corotational._corotational_cache(model, centered_shell=True)
    assert centered_first is not first
    model.bump_revision("load")
    assert corotational._corotational_cache(model) is first
    assert corotational._corotational_cache(model, centered_shell=True) is centered_first

    model.set_node_coordinates(2, 1.0, 0.0, 0.01)
    second = corotational._corotational_cache(model)
    assert second is not first
    assert corotational._corotational_cache(model, centered_shell=True) is not centered_first
