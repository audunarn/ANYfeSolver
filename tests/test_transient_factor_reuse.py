"""A cached effective-matrix factorization must not serve a different step size."""
import warnings

import numpy as np

from anyfesolver import (
    BoundaryCondition,
    LoadCase,
    RigidSphereImpact,
    SphereContactConfig,
    TransientConfig,
    create_shell_element,
    solve_transient_newmark,
    solve_transient_sphere_impact,
)
from anyfesolver.dynamics import same_step_size
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel

SEC = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}
DOFS = ["ux", "uy", "uz", "rx", "ry", "rz"]


def _cantilever(n=4):
    model = FEModel("cantilever")
    model.add_material("mat", 2.1e11, 0.3, density=7850.0)
    for i in range(n + 1):
        model.add_node(i + 1, 10.0 * i / n, 0.0, 0.0)
    for i in range(n):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(SEC)))
    model.add_boundary_condition(BoundaryCondition("fix", [1], {d: 0.0 for d in DOFS}))
    load = LoadCase("tip")
    load.add_nodal_load(n + 1, [0, 0, -1.0e4, 0, 0, 0])
    return model, load


def _panel():
    model = FEModel("panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        model.add_node(i + 1, float(x), float(y), 0.0)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "soft", thickness=0.05))
    model.add_boundary_condition(
        BoundaryCondition(
            "r", [1, 2, 3, 4], {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
        )
    )
    load = LoadCase("push")
    load.add_nodal_load(3, [0, 0, -50.0, 0, 0, 0])
    return model, load


def _far_sphere():
    return RigidSphereImpact(
        "far", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 50.0),
        travel_direction=(0.0, 0.0, -1.0), speed=0.0,
    )


def _check_last_step(full, first, second):
    # the first step is unaffected (guards against an unrelated failure)
    assert np.allclose(full.displacements[1], first.displacements[-1], rtol=1e-12, atol=0.0)
    expected = second.displacements[-1]
    scale = np.linalg.norm(expected)
    assert scale > 0.0
    assert np.linalg.norm(full.displacements[-1] - expected) <= 1.0e-6 * scale


def test_transient_final_short_step_must_not_reuse_stale_factorization():
    dt, t_end = 1.0e-8, 1.5e-8  # grid [0, 1e-8, 1.5e-8]: last dt = 5e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, load = _cantilever()
        full = solve_transient_newmark(
            model, TransientConfig(dt=dt, t_end=t_end), base_load_case=load
        )
        assert np.allclose(full.times, [0.0, 1.0e-8, 1.5e-8])
        assert full.diagnostics["factorization_count"] == 2
        m1, l1 = _cantilever()
        first = solve_transient_newmark(m1, TransientConfig(dt=dt, t_end=dt), base_load_case=l1)
        m2, l2 = _cantilever()
        second = solve_transient_newmark(
            m2,
            TransientConfig(
                dt=5.0e-9, t_end=5.0e-9,
                initial_displacement=first.displacements[-1],
                initial_velocity=first.velocities[-1],
            ),
            base_load_case=l2,
        )
    _check_last_step(full, first, second)


def test_sphere_impact_final_short_step_must_not_reuse_stale_factorization():
    # Sphere far away and at rest: the panel responds to a step load, and the
    # sphere route uses the same cached-factor logic as the plain transient.
    dt, t_end = 1.0e-8, 1.5e-8
    cfg = SphereContactConfig(penalty_stiffness=4000.0, contact_damping=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, load = _panel()
        full = solve_transient_sphere_impact(
            model, TransientConfig(dt=dt, t_end=t_end), _far_sphere(), cfg, base_load_case=load
        )
        assert full.diagnostics["factorization_count"] == 2
        m1, l1 = _panel()
        first = solve_transient_sphere_impact(
            m1, TransientConfig(dt=dt, t_end=dt), _far_sphere(), cfg, base_load_case=l1
        )
        m2, l2 = _panel()
        second = solve_transient_sphere_impact(
            m2,
            TransientConfig(
                dt=5.0e-9, t_end=5.0e-9,
                initial_displacement=first.displacements[-1],
                initial_velocity=first.velocities[-1],
            ),
            _far_sphere(), cfg, base_load_case=l2,
        )
    _check_last_step(full, first, second)


def test_grid_rounding_in_step_size_still_reuses_the_factorization():
    # ordinary uniform grid: dt differs between steps only by subtraction rounding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, load = _cantilever()
        result = solve_transient_newmark(
            model, TransientConfig(dt=0.0013, t_end=0.0731), base_load_case=load
        )
    assert result.diagnostics["factorization_count"] == 2  # uniform steps + final short step
    assert same_step_size(0.0013, 0.0013 + 2.0e-19, 0.07)
    assert not same_step_size(1.0e-8, 5.0e-9, 1.5e-8)
