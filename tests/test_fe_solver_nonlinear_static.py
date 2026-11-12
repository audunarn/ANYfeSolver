"""Validation of the incremental geometric/material nonlinear solver.

Closed-form anchors:

* DNV-RP-C208 curve: exact knot values and Part-3 continuity.
* Uniaxial tension past yield: FE stress/strain pair must lie on the curve.
* Pure bending of a plate strip with a near-perfectly-plastic curve: the
  capacity approaches the plastic moment of the through-thickness quadrature
  (shape factor ~1.5).
* Beam-column at P = Pcr/2: lateral deflection amplification ~2x.
* Membrane stiffening: a transversely loaded strip with immovable ends
  responds clearly sub-linearly.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, create_shell_element
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.assembly import solve_linear
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.plasticity import lobatto_layers
from anyfesolver.nonlinear_static import NonlinearConvergenceSettings, solve_static_nonlinear
from anyfesolver.recovery import ResourceConfig

E = 210.0e9
NU = 0.3

# S355, t <= 16 (RP-C208 Table 4-4), in Pa.
S355_CURVE = DNVC208MaterialCurve(
    sigma_prop=320.0e6,
    sigma_yield=357.0e6,
    sigma_yield_2=363.3e6,
    eps_p_y1=0.004,
    eps_p_y2=0.015,
    K=740.0e6,
    n=0.166,
)

# Nearly perfectly plastic curve for limit-load tests.
EPP_CURVE = DNVC208MaterialCurve(
    sigma_prop=354.0e6,
    sigma_yield=355.0e6,
    sigma_yield_2=355.5e6,
    eps_p_y1=0.004,
    eps_p_y2=0.1,
    K=400.0e6,
    n=0.2,
)


def test_rp_c208_curve_knots_and_continuity():
    assert S355_CURVE.flow_stress(np.array([0.0]))[0] == pytest.approx(320.0e6)
    assert S355_CURVE.flow_stress(np.array([0.004]))[0] == pytest.approx(357.0e6)
    assert S355_CURVE.flow_stress(np.array([0.015]))[0] == pytest.approx(363.3e6)
    # Part 3 continuity at eps_p_y2 and monotonic hardening beyond it.
    just_after = S355_CURVE.flow_stress(np.array([0.015 + 1.0e-9]))[0]
    assert just_after == pytest.approx(363.3e6, rel=1.0e-6)
    grid = np.linspace(0.0, 0.2, 400)
    flow = S355_CURVE.flow_stress(grid)
    assert np.all(np.diff(flow) >= -1.0e-6)
    assert np.all(S355_CURVE.hardening_modulus(grid) >= 0.0)


def _strip_model(nx, ny, length, width, t, curve):
    model = FEModel(name="strip")
    model.add_material("steel", E, NU, hardening_curve=curve)
    nid = {}
    k = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            model.add_node(k, length * i / nx, width * j / ny, 0.0)
            nid[(i, j)] = k
            k += 1
    eidx = 1
    for j in range(ny):
        for i in range(nx):
            model.add_element(
                eidx,
                create_shell_element(
                    eidx,
                    [
                        nid[(i, j)],
                        nid[(i + 1, j)],
                        nid[(i + 1, j + 1)],
                        nid[(i, j + 1)],
                    ],
                    "steel",
                    thickness=t,
                ),
            )
            eidx += 1
    return model, nid


def test_uniaxial_tension_follows_rp_c208_curve():
    """Pull a strip to ~340 MPa (inside the Part-1 hardening range) and check
    that the FE strain decomposes onto the material curve."""
    length, width, t = 1.0, 0.2, 0.01
    nx, ny = 4, 1
    model, nid = _strip_model(nx, ny, length, width, t, S355_CURVE)

    left = [nid[(0, j)] for j in range(ny + 1)]
    right = [nid[(nx, j)] for j in range(ny + 1)]
    model.add_boundary_condition(BoundaryCondition("clamp_x", left, {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_y", [nid[(0, 0)]], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition("plane", list(nid.values()), {"uz": 0.0, "rx": 0.0, "ry": 0.0})
    )

    sigma_target = 340.0e6
    F_total = sigma_target * width * t
    load_case = LoadCase(name="pull")
    for node in right:
        share = 0.5 if node in (nid[(nx, 0)], nid[(nx, ny)]) else 1.0
        load_case.add_nodal_load(node, load_vector=[F_total * share / ny, 0, 0, 0, 0, 0])

    result = solve_static_nonlinear(model, load_case, num_steps=8, num_layers=3)
    assert result.status == "completed"
    assert result.load_factor == pytest.approx(1.0)

    ux_end = np.mean([result.displacements[model.mesh.get_node(n).dofs[0]] for n in right])
    eps_total = ux_end / length
    eps_plastic = eps_total - sigma_target / E
    assert eps_plastic > 1.0e-4  # well past first yield (sigma_prop = 320 MPa)
    # The (stress, plastic strain) pair must lie on the RP-C208 curve.
    assert S355_CURVE.flow_stress(np.array([eps_plastic]))[0] == pytest.approx(
        sigma_target, rel=0.01
    )


def test_pure_bending_reaches_plastic_moment():
    """Ramp end moments on a strip; capacity ~= the quadrature plastic moment."""
    length, width, t = 0.5, 0.1, 0.01
    nx, ny = 5, 1
    model, nid = _strip_model(nx, ny, length, width, t, EPP_CURVE)

    left = [nid[(0, j)] for j in range(ny + 1)]
    right = [nid[(nx, j)] for j in range(ny + 1)]
    model.add_boundary_condition(FixedSupport("clamp", left))
    # Transverse contraction stays free so the strip is in uniaxial bending;
    # constraining uy would put it in plane strain and legitimately raise the
    # capacity by ~2/sqrt(3).

    sigma_y = 355.0e6
    M_yield = sigma_y * width * t**2 / 6.0
    num_layers = 9
    z, w = lobatto_layers(num_layers, t)
    M_plastic_quadrature = sigma_y * width * float(np.sum(w * np.abs(z)))
    shape_factor = M_plastic_quadrature / M_yield
    assert 1.4 < shape_factor < 1.5

    target_moment = 1.15 * M_plastic_quadrature
    load_case = LoadCase(name="moment")
    for node in right:
        load_case.add_nodal_load(node, load_vector=[0, 0, 0, 0, target_moment / len(right), 0])

    result = solve_static_nonlinear(
        model, load_case, num_steps=12, num_layers=num_layers, max_iterations=30
    )
    # The target is 15% above the section capacity: the solver must stop at
    # the limit, and the last converged factor times the applied moment must
    # match the quadrature plastic moment.
    assert result.status == "stopped_at_limit"
    capacity_moment = result.load_factor * target_moment
    assert capacity_moment == pytest.approx(M_plastic_quadrature, rel=0.06)
    assert result.steps[-1].max_equivalent_plastic_strain > 1.0e-3


def test_beam_column_amplification_at_half_critical_load():
    """Lateral deflection at P = Pcr/2 amplifies ~2x over the linear answer."""
    L = 3.0
    A, I = 0.01, 1.0e-6
    n_elem = 16
    section = {"area": A, "Iy": I, "Iz": I, "J": 1e-6, "orientation": (0.0, 0.0, 1.0)}
    P_cr = np.pi**2 * E * I / L**2

    def build():
        model = FEModel(name="beam_column")
        model.add_material("steel", E, NU)
        for i in range(n_elem + 1):
            model.add_node(i + 1, L * i / n_elem, 0.0, 0.0)
        for e in range(n_elem):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
        model.add_boundary_condition(
            BoundaryCondition("p1", [1], {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0})
        )
        model.add_boundary_condition(
            BoundaryCondition("p2", [n_elem + 1], {"uy": 0.0, "uz": 0.0})
        )
        return model

    Q = 100.0  # small lateral load at midspan
    mid = 1 + n_elem // 2

    lateral = LoadCase(name="lateral")
    lateral.add_nodal_load(mid, load_vector=[0.0, 0.0, Q, 0.0, 0.0, 0.0])

    model = build()
    u_lin, _ = solve_linear(model, lateral)
    w_linear = u_lin[model.mesh.get_node(mid).dofs[2]]

    axial_and_lateral = LoadCase(name="combined")
    axial_and_lateral.add_nodal_load(n_elem + 1, load_vector=[-0.5 * P_cr, 0, 0, 0, 0, 0])

    model2 = build()
    result = solve_static_nonlinear(
        model2, axial_and_lateral, constant_load_case=lateral, num_steps=10
    )
    assert result.status == "completed"
    w_nl = result.displacements[model2.mesh.get_node(mid).dofs[2]]
    amplification = w_nl / w_linear
    assert amplification == pytest.approx(2.0, rel=0.10)


def test_membrane_stiffening_of_immovable_strip():
    """With immovable in-plane edges, transverse response stiffens strongly."""
    length, width, t = 1.0, 0.1, 0.002
    nx, ny = 10, 1
    model, nid = _strip_model(nx, ny, length, width, t, None)

    left = [nid[(0, j)] for j in range(ny + 1)]
    right = [nid[(nx, j)] for j in range(ny + 1)]
    model.add_boundary_condition(
        BoundaryCondition("pin_ends", left + right, {"ux": 0.0, "uy": 0.0, "uz": 0.0})
    )

    P = 2000.0
    mid_nodes = [nid[(nx // 2, j)] for j in range(ny + 1)]
    load_case = LoadCase(name="press")
    for node in mid_nodes:
        load_case.add_nodal_load(node, load_vector=[0, 0, P / len(mid_nodes), 0, 0, 0])

    model_lin, nid_lin = _strip_model(nx, ny, length, width, t, None)
    model_lin.add_boundary_condition(
        BoundaryCondition(
            "pin_ends",
            [nid_lin[(0, j)] for j in range(ny + 1)] + [nid_lin[(nx, j)] for j in range(ny + 1)],
            {"ux": 0.0, "uy": 0.0, "uz": 0.0},
        )
    )
    lc_lin = LoadCase(name="press")
    for j in range(ny + 1):
        lc_lin.add_nodal_load(nid_lin[(nx // 2, j)], load_vector=[0, 0, P / (ny + 1), 0, 0, 0])
    u_lin, _ = solve_linear(model_lin, lc_lin)
    w_linear = abs(u_lin[model_lin.mesh.get_node(nid_lin[(nx // 2, 0)]).dofs[2]])

    result = solve_static_nonlinear(model, load_case, num_steps=10)
    assert result.status == "completed"
    w_nl = abs(result.displacements[model.mesh.get_node(mid_nodes[0]).dofs[2]])

    # Bending-only deflection would exceed the thickness many times; the von
    # Karman membrane stretching must cut it down substantially.
    assert w_linear > 5.0 * t
    assert w_nl < 0.5 * w_linear
    assert w_nl > 0.5 * t


def test_fast_convergence_settings_grow_load_steps_on_easy_elastic_case():
    model = FEModel(name="fast_nonlinear_settings")
    model.add_material("steel", E, NU)
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6, "orientation": (0.0, 0.0, 1.0)}
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))

    load = LoadCase("tip")
    load.add_nodal_load(2, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(model, load, num_steps=8, convergence_settings="fast")

    assert result.status == "completed"
    assert len(result.steps) < 8
    assert result.info["convergence_settings"]["profile"] == "fast"
    base_step = 1.0 / 8.0
    assert max(row["step_size"] for row in result.info["force_displacement_history"]) > base_step
    assert any(row["action"] == "grow" for row in result.info["convergence_adaptation"])


def test_legacy_convergence_settings_keep_original_step_cap():
    model = FEModel(name="legacy_nonlinear_settings")
    model.add_material("steel", E, NU)
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6, "orientation": (0.0, 0.0, 1.0)}
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))

    load = LoadCase("tip")
    load.add_nodal_load(2, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=4,
        convergence_settings=NonlinearConvergenceSettings.for_profile("legacy"),
    )

    assert result.status == "completed"
    assert len(result.steps) == 4
    assert max(row["step_size"] for row in result.info["force_displacement_history"]) == pytest.approx(0.25)


def test_legacy_positional_parameter_order_remains_compatible():
    model = FEModel(name="legacy_positional_nonlinear_api")
    model.add_material("steel", E, NU)
    section = {
        "area": 0.01,
        "Iy": 1.0e-6,
        "Iz": 1.0e-6,
        "J": 1.0e-6,
        "orientation": (0.0, 0.0, 1.0),
    }
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))

    load = LoadCase("tip")
    load.add_nodal_load(2, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    settings = NonlinearConvergenceSettings.for_profile("legacy")

    result = solve_static_nonlinear(
        model,
        load,
        None,
        1.0,
        4,
        25,
        1.0e-6,
        5,
        1.0 / 1024.0,
        None,
        None,
        "force",
        None,
        None,
        settings,
        ResourceConfig(),
        None,
        "von_karman",
        "auto",
        None,
        None,
    )

    assert result.status == "completed"
    assert len(result.steps) == 4
    assert result.info["convergence_settings"]["profile"] == "legacy"


def test_resource_config_assembly_threads_enters_nonlinear_numba_scope(monkeypatch):
    calls = []

    class DummyScope:
        def __init__(self, thread_count):
            self.thread_count = thread_count

        def __enter__(self):
            calls.append(("enter", self.thread_count))

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", self.thread_count))
            return False

    import anyfesolver.nonlinear_static as nonlinear_static_module

    monkeypatch.setattr(
        nonlinear_static_module,
        "numba_thread_scope",
        lambda thread_count: DummyScope(thread_count),
    )

    model = FEModel(name="resource_threads")
    model.add_material("steel", E, NU)
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6, "orientation": (0.0, 0.0, 1.0)}
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    load = LoadCase("tip")
    load.add_nodal_load(2, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        resource_config=ResourceConfig(assembly_threads=2),
    )

    assert result.status == "completed"
    assert ("enter", 2) in calls
    assert ("exit", 2) in calls
    assert result.info["resource_config"]["assembly_threads"] == 2


def test_nonlinear_result_reports_status_category_and_stop_reason():
    model = FEModel(name="nonlinear_status_completed")
    model.add_material("steel", E, NU)
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6, "orientation": (0.0, 0.0, 1.0)}
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))

    load = LoadCase("tip")
    load.add_nodal_load(2, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(model, load, num_steps=2)
    payload = result.to_dict()

    assert result.status == "completed"
    assert result.status_category == "converged"
    assert result.stop_reason == "target_load_factor_reached"
    assert payload["status_category"] == "converged"
    assert payload["stop_reason"] == "target_load_factor_reached"
