"""Cylinder benchmark tests."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    CylinderBenchmarkConfig,
    build_cylindrical_shell_benchmark_model,
    nominal_cylinder_membrane_stress,
    run_cylindrical_shell_benchmark,
)
from anyfesolver.elements import ShellElement


def test_nominal_cylinder_membrane_stress_reports_hoop_axial_and_von_mises() -> None:
    nominal = nominal_cylinder_membrane_stress(radius=3.0, thickness=0.01, pressure=100_000.0)

    assert nominal.hoop_stress == pytest.approx(-30.0e6)
    assert nominal.axial_stress == pytest.approx(-15.0e6)
    assert nominal.von_mises_stress == pytest.approx(np.sqrt((30.0e6) ** 2 + (15.0e6) ** 2 - 30.0e6 * 15.0e6))
    assert nominal.to_dict()["hoop_stress"] == nominal.hoop_stress


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_cylindrical_shell_benchmark_mesh_places_nodes_on_radius(use_8node_elements: bool) -> None:
    config = CylinderBenchmarkConfig(
        radius=2.0,
        height=3.0,
        thickness=0.02,
        pressure=10_000.0,
        num_circumferential=8,
        num_height=3,
        use_8node_elements=use_8node_elements,
    )
    model, load_case = build_cylindrical_shell_benchmark_model(config)

    assert len(load_case.pressure_loads) == config.num_circumferential * config.num_height
    assert model.mesh.num_elements == config.num_circumferential * config.num_height
    assert all(isinstance(element, ShellElement) for element in model.mesh.elements.values())

    radii = [np.linalg.norm(node.coords()[:2]) for node in model.mesh.nodes.values()]
    np.testing.assert_allclose(radii, config.radius, rtol=0.0, atol=1.0e-12)

    expected_corner_nodes = (config.num_height + 1) * config.num_circumferential
    if use_8node_elements:
        expected_nodes = expected_corner_nodes
        expected_nodes += (config.num_height + 1) * config.num_circumferential
        expected_nodes += config.num_height * config.num_circumferential
    else:
        expected_nodes = expected_corner_nodes
    assert model.mesh.num_nodes == expected_nodes


def test_cylinder_benchmark_returns_percentile_stress_reporting() -> None:
    config = CylinderBenchmarkConfig(
        radius=1.5,
        height=2.0,
        thickness=0.02,
        pressure=20_000.0,
        num_circumferential=8,
        num_height=4,
        use_8node_elements=False,
    )

    result = run_cylindrical_shell_benchmark(config)

    assert result.solver_status == "converged"
    assert result.node_count == (config.num_height + 1) * config.num_circumferential
    assert result.shell_element_count == config.num_height * config.num_circumferential
    assert result.nominal.hoop_stress == pytest.approx(-config.pressure * config.radius / config.thickness)
    assert result.nominal.axial_stress == pytest.approx(-config.pressure * config.radius / (2.0 * config.thickness))
    assert result.all_von_mises.count == config.num_height * config.num_circumferential * 4
    assert result.mid_height_von_mises.count > 0
    assert result.fe_max_von_mises >= result.fe_p95_von_mises >= 0.0
    assert result.mid_height_von_mises.maximum >= result.fe_mid_height_p95_von_mises >= 0.0
    assert np.isfinite(result.max_radial_displacement)
    assert np.isfinite(result.relative_rigid_body_load_imbalance)
    assert result.to_dict()["all_von_mises"]["p95"] == result.fe_p95_von_mises
