"""A cached effective-matrix factorization must not serve a different step size."""
import warnings

import numpy as np
import pytest

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
    # ordinary uniform grid: interior steps use the nominal dt, so one factorization serves them all
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, load = _cantilever()
        result = solve_transient_newmark(
            model, TransientConfig(dt=0.0013, t_end=0.0731), base_load_case=load
        )
    assert result.diagnostics["factorization_count"] == 2  # uniform steps + final short step


def _two_step_factorizations(t_end):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, load = _cantilever()
        result = solve_transient_newmark(
            model, TransientConfig(dt=0.5, t_end=t_end), base_load_case=load
        )
    assert len(result.times) == 3
    return result.diagnostics["factorization_count"], result.times[2] - result.times[1]


def test_transient_route_reuse_boundary_end_to_end():
    # Uniform-end rule: the final step keeps the nominal dt (no refactorization) while the end
    # time is within 8 ulp of n * dt, tolerance = 8 * max(ulp(t_end), ulp(n dt), n ulp(dt)).
    # times [0, 0.5, t_end] with t_end in [0.5, 1): n dt = 1.0, tolerance = 8 * 2**-52 = 2**-49.
    t_at = 1.0 - 2.0**-49
    count, last_dt = _two_step_factorizations(t_at)
    assert 0.5 - last_dt == 2.0**-49
    assert count == 1
    t_above = float(np.nextafter(t_at, 0.0))  # one representable value further
    count, last_dt = _two_step_factorizations(t_above)
    assert 0.5 - last_dt > 2.0**-49
    assert count == 2


def test_reuse_at_the_threshold_is_identical_to_the_nominal_run():
    # With consistent-dt integration a last step shortened by rounding only (here exactly on
    # the threshold) integrates with the nominal dt, so the whole history is bit-identical to
    # the run that ends exactly on the nominal grid.
    dt = 2.0**-30

    def run(t_end):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, load = _cantilever()
            result = solve_transient_newmark(
                model, TransientConfig(dt=dt, t_end=t_end), base_load_case=load
            )
        return (result.diagnostics["factorization_count"], np.asarray(result.displacements),
                np.asarray(result.accelerations))

    n_nominal, u_nominal, a_nominal = run(2.0**-29)
    n_edge, u_edge, a_edge = run(2.0**-29 - 2.0**-79)  # last step shortened by 8 ulp
    assert (n_nominal, n_edge) == (1, 1)
    assert np.array_equal(u_edge, u_nominal)
    assert np.array_equal(a_edge, a_nominal)


def _unit_oscillator():
    """One free axial DOF with K = EA/L = 1 and M = rho*A*L/3 = 1 (consistent bar mass)."""
    model = FEModel("m_k_1")
    model.add_material("unit", 1.0, 0.3, density=3.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "unit", {"area": 1.0, "Iy": 1.0, "Iz": 1.0, "J": 1.0}))
    model.add_boundary_condition(BoundaryCondition("fix1", [1], {d: 0.0 for d in DOFS}))
    model.add_boundary_condition(
        BoundaryCondition("axial", [2], {d: 0.0 for d in DOFS if d != "ux"})
    )
    load = LoadCase("tip")
    load.add_nodal_load(2, [1.0, 0, 0, 0, 0, 0])
    return model, load


def test_reused_step_on_a_fine_grid_matches_exact_refactorization():
    # Codex P2 (dt round 2), M = K = 1 at static equilibrium (u = 1, v = a = 0, F = 1): the
    # exact response is u = 1, a = 0 for every step.  With dt = 2**-20 and the last step
    # shortened by exactly 8 ulp(t), the cached-matrix/grid-dt mismatch gave a spurious
    # acceleration of ~8 m/s^2 (Codex's 2**-4 end time: ~512).  Consistent-dt integration
    # integrates that step with the nominal dt on the single factorization.
    dt = 2.0**-20
    t_end = 2.0**-10 - 2.0**-60
    model, load = _unit_oscillator()
    node = model.mesh.get_node(2)
    u_start = np.zeros(12)
    u_start[node.dofs[0]] = 1.0
    result = solve_transient_newmark(
        model, TransientConfig(dt=dt, t_end=t_end, initial_displacement=u_start), base_load_case=load,
    )
    assert result.diagnostics["factorization_count"] == 1
    u = np.asarray(result.displacements)[:, node.dofs[0]]
    a = np.asarray(result.accelerations)[:, node.dofs[0]]
    assert np.abs(u - 1.0).max() <= 1.0e-13
    assert np.abs(a).max() <= 64.0 * np.finfo(float).eps * (4.0 / dt**2)
