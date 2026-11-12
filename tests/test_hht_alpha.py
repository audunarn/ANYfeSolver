"""HHT-alpha time integration: config, dissipation, and regression behavior."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.assembly import solve_linear
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.contact import (
    NonlinearTransientConfig,
    RigidSphereImpact,
    SphereContactConfig,
    _verification_contact_panel,
    solve_transient_sphere_impact,
)
from anyfesolver.dynamics import TransientConfig, solve_transient_newmark
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


def test_hht_alpha_config_validation_and_parameter_derivation() -> None:
    with pytest.raises(ValueError):
        TransientConfig(dt=0.01, t_end=0.1, hht_alpha=0.1)
    with pytest.raises(ValueError):
        TransientConfig(dt=0.01, t_end=0.1, hht_alpha=-0.5)

    default = TransientConfig(dt=0.01, t_end=0.1)
    assert default.integration_parameters() == (0.0, 0.25, 0.5)

    hht = TransientConfig(dt=0.01, t_end=0.1, hht_alpha=-0.1)
    alpha, beta, gamma = hht.integration_parameters()
    assert alpha == -0.1
    assert beta == pytest.approx(0.25 * 1.1**2)
    assert gamma == pytest.approx(0.6)

    custom = TransientConfig(dt=0.01, t_end=0.1, beta=0.3, gamma=0.55, hht_alpha=-0.1)
    assert custom.integration_parameters() == (-0.1, 0.3, 0.55)


def _clamped_plate(n: int = 4) -> FEModel:
    model = FEModel("hht_plate")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    ids = {}
    node_id = 1
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(node_id, i / n, j / n, 0.0)
            ids[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(n):
        for i in range(n):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [
                        ids[(i, j)],
                        ids[(i + 1, j)],
                        ids[(i + 1, j + 1)],
                        ids[(i, j + 1)],
                    ],
                    "steel",
                    thickness=0.01,
                ),
            )
            element_id += 1
    edge = sorted(
        {ids[(i, 0)] for i in range(n + 1)}
        | {ids[(i, n)] for i in range(n + 1)}
        | {ids[(0, j)] for j in range(n + 1)}
        | {ids[(n, j)] for j in range(n + 1)}
    )
    model.add_boundary_condition(BoundaryCondition("clamped", edge, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _free_vibration(hht_alpha: float) -> dict:
    model = _clamped_plate()
    load_case = LoadCase("initial_shape")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 5.0e4)
    static_u, _info = solve_linear(model, load_case)
    config = TransientConfig(dt=2.0e-5, t_end=4.0e-3, hht_alpha=hht_alpha, initial_displacement=static_u)
    result = solve_transient_newmark(model, config)
    return {
        "status": result.status,
        "drift": float(result.diagnostics["max_relative_energy_drift"]),
        "method": result.diagnostics["method"],
        "displacements": result.displacements,
    }


def test_hht_alpha_zero_matches_default_newmark_exactly() -> None:
    reference = _free_vibration(0.0)
    assert reference["method"] == "newmark"
    explicit = _free_vibration(0.0)
    np.testing.assert_array_equal(reference["displacements"], explicit["displacements"])


def test_hht_alpha_dissipates_free_vibration_energy_monotonically_with_alpha() -> None:
    conservative = _free_vibration(0.0)
    mild = _free_vibration(-0.05)
    strong = _free_vibration(-1.0 / 3.0)

    assert conservative["status"] == "completed"
    assert mild["status"] == "completed"
    assert strong["status"] == "completed"
    assert mild["method"] == "hht_alpha"

    # Average-acceleration Newmark conserves energy in undamped free vibration.
    assert conservative["drift"] < 1.0e-6
    # HHT-alpha removes energy, monotonically with |alpha|.
    assert mild["drift"] > 10.0 * conservative["drift"]
    assert strong["drift"] > mild["drift"]

    # The dissipation shows in the final response amplitude as well.
    tail = slice(-10, None)
    amp_conservative = float(np.max(np.abs(conservative["displacements"][tail])))
    amp_strong = float(np.max(np.abs(strong["displacements"][tail])))
    assert amp_strong < amp_conservative


def _panel_impact(hht_alpha: float):
    return solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12, output_nodes=[1], hht_alpha=hht_alpha),
        RigidSphereImpact("hht_hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
    )


def test_hht_alpha_sphere_impact_damps_post_impact_ringing() -> None:
    newmark = _panel_impact(0.0)
    damped = _panel_impact(-0.1)

    assert newmark.status == "completed"
    assert damped.status == "completed"
    assert damped.peak_contact_force > 0.0
    # The contact event itself is comparable...
    assert damped.peak_contact_force == pytest.approx(newmark.peak_contact_force, rel=0.35)
    assert damped.diagnostics["contact_config"]["penalty_stiffness"] == 4000.0
    # ...but the retained structural vibration energy after impact is lower.
    energy_newmark = float(newmark.diagnostics["kinetic_energy"][-1] + newmark.diagnostics["strain_energy"][-1])
    energy_damped = float(damped.diagnostics["kinetic_energy"][-1] + damped.diagnostics["strain_energy"][-1])
    assert energy_damped < energy_newmark
    assert damped.result_case["analysis_case"]["settings"]["hht_alpha"] == -0.1


def test_hht_alpha_nonlinear_impact_smoke() -> None:
    model = _verification_contact_panel()
    model.materials["soft"].hardening_curve = DNVC208MaterialCurve(
        sigma_prop=800.0,
        sigma_yield=1000.0,
        sigma_yield_2=1200.0,
        eps_p_y1=1.0e-5,
        eps_p_y2=1.0e-3,
        K=2000.0,
        n=0.1,
    )
    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0025, t_end=0.05, hht_alpha=-0.05),
        RigidSphereImpact("hht_nl", radius=0.2, mass=5.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=1.0),
        SphereContactConfig(penalty_stiffness=500.0, max_contact_iterations=10),
        nonlinear_config=NonlinearTransientConfig(enabled=True, max_iterations=10, max_cutbacks=2),
    )
    assert result.status in {"completed", "no_contact"}
    assert result.diagnostics["method"] == "nonlinear_newmark_sphere_penalty_contact"
    assert result.result_case["analysis_case"]["settings"]["hht_alpha"] == -0.05
    assert result.displacements.shape[0] == len(result.times)
