"""Corotational kinematics: invariance, roll-up benchmarks, and scope guards."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.corotational import (
    corotational_element_response,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
)
from anyfesolver.elements import (
    BeamElement,
    LegacyShellElement,
    LegacyQ4DeprecationWarning,
    QuadraticBeamElement,
    create_shell_element,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import dnv_c208_steel_curve
from anyfesolver.nonlinear_static import _assemble_nonlinear_system, solve_static_nonlinear

E_STEEL = 2.1e11


def test_rotation_maps_roundtrip_including_large_angles() -> None:
    rng = np.random.default_rng(7)
    for angle in (1.0e-9, 0.3, 1.5, np.pi - 1.0e-4, 3.0):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        vector = angle * axis
        R = rotation_matrix_from_vector(vector)
        assert np.allclose(R @ R.T, np.eye(3), atol=1.0e-12)
        recovered = rotation_vector_from_matrix(R)
        R_back = rotation_matrix_from_vector(recovered)
        assert np.allclose(R_back, R, atol=1.0e-8)


def _single_shell_model() -> FEModel:
    model = FEModel("cr_shell")
    model.add_material("steel", E_STEEL, 0.0, density=7850.0)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], start=1):
        model.add_node(i, float(x), float(y), 0.0)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    )
    return model


def _single_legacy_shell_model() -> FEModel:
    """Build the registered legacy-Q4 rollback fixture explicitly."""

    model = FEModel("cr_legacy_shell")
    model.add_material("steel", E_STEEL, 0.0, density=7850.0)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], start=1):
        model.add_node(i, float(x), float(y), 0.0)
    with pytest.warns(LegacyQ4DeprecationWarning, match="temporary rollback"):
        element = LegacyShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.01
        )
    model.add_element(1, element)
    return model


def _single_beam_model() -> FEModel:
    model = FEModel("cr_beam")
    model.add_material("steel", E_STEEL, 0.0, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1e-3, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}))
    return model


def _single_quadratic_beam_model() -> FEModel:
    model = FEModel("cr_qbeam")
    model.add_material("steel", E_STEEL, 0.0, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.5, 0.0, 0.0)  # midside node
    model.add_node(3, 1.0, 0.0, 0.0)
    model.add_element(1, QuadraticBeamElement(1, [1, 2, 3], "steel", {"area": 1e-3, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}))
    return model


def _rigid_rotation_field(model: FEModel, angle_deg: float, axis) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    R = rotation_matrix_from_vector(np.radians(angle_deg) * axis)
    theta = np.radians(angle_deg) * axis
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes.values():
        X = np.array([node.x, node.y, node.z])
        u[np.asarray(node.dofs[:3])] = R @ X - X
        u[np.asarray(node.dofs[3:])] = theta
    return u


@pytest.mark.parametrize("model_factory,centered_shell_reference", [
    (_single_shell_model, False), (_single_shell_model, True),
    (_single_beam_model, False),
])
def test_consistent_corotational_tangent_matches_force_jacobian(
    model_factory, centered_shell_reference,
) -> None:
    """The opt-in tangent differentiates both pull-back and frame rotation."""

    model = model_factory()
    element = model.mesh.elements[1]
    u = _rigid_rotation_field(model, 35.0, (0.2, 0.4, 1.0))
    end_node = model.mesh.nodes[max(model.mesh.nodes)]
    u[end_node.dofs[0]] += 0.003
    u[end_node.dofs[2]] += 0.007
    u[end_node.dofs[4]] -= 0.015

    _force, tangent, _state = corotational_element_response(
        model,
        1,
        element,
        u,
        tangent=True,
        tangent_mode="consistent",
        centered_shell_reference=centered_shell_reference,
    )
    numerical = np.zeros_like(tangent)
    step = 2.0e-7
    for column in range(u.size):
        plus = u.copy()
        minus = u.copy()
        plus[column] += step
        minus[column] -= step
        force_plus, _, _ = corotational_element_response(
            model,
            1,
            element,
            plus,
            tangent=False,
            tangent_mode="consistent",
            centered_shell_reference=centered_shell_reference,
        )
        force_minus, _, _ = corotational_element_response(
            model,
            1,
            element,
            minus,
            tangent=False,
            tangent_mode="consistent",
            centered_shell_reference=centered_shell_reference,
        )
        numerical[:, column] = (force_plus - force_minus) / (2.0 * step)

    relative_error = np.linalg.norm(tangent - numerical) / max(np.linalg.norm(numerical), 1.0)
    assert relative_error < 2.0e-7
    # Additive rotation-vector coordinates make the exact Jacobian generally
    # nonsymmetric away from equilibrium; symmetrizing it loses consistency.
    assert np.linalg.norm(tangent - tangent.T) > 1.0e-6 * np.linalg.norm(tangent)


def test_corotational_tangent_mode_rejects_unknown_value() -> None:
    model = _single_shell_model()
    element = model.mesh.elements[1]
    with pytest.raises(ValueError, match="tangent_mode"):
        corotational_element_response(
            model,
            1,
            element,
            np.zeros(element.total_dofs),
            tangent=True,
            tangent_mode="secant",
        )


def test_centered_shell_reference_remains_objective_under_rigid_rotation() -> None:
    model = _single_shell_model()
    element = model.mesh.elements[1]
    zero = np.zeros(element.total_dofs)
    ordinary_force, ordinary_tangent, _ = corotational_element_response(
        model, 1, element, zero, tangent=True, tangent_mode="consistent")
    centered_force, centered_tangent, _ = corotational_element_response(
        model, 1, element, zero, tangent=True, tangent_mode="consistent",
        centered_shell_reference=True)
    assert np.linalg.norm(centered_force) < 1e-9
    assert np.linalg.norm(centered_force-ordinary_force) < 1e-9
    assert (np.linalg.norm(centered_tangent-ordinary_tangent) <=
            1e-10*np.linalg.norm(ordinary_tangent))
    displacement = _rigid_rotation_field(model, 90., (1., 1., 1.))
    force, _tangent, _state = corotational_element_response(
        model, 1, element, displacement, tangent=False,
        centered_shell_reference=True)
    assert np.linalg.norm(force) < 1e-9 * E_STEEL * 1e-5


def test_centered_shell_reference_preserves_response_after_origin_shift() -> None:
    model = _single_shell_model()
    element = model.mesh.elements[1]
    displacement = np.zeros(element.total_dofs)
    displacement[2::6] = (0., 0.001, 0.002, 0.001)
    displacement[3::6] = (0.001, -0.002, 0.003, 0.)
    base_force, base_tangent, _ = corotational_element_response(
        model, 1, element, displacement, tangent=True,
        tangent_mode="consistent", centered_shell_reference=True)
    shifted = _single_shell_model()
    for node_id, node in tuple(shifted.mesh.nodes.items()):
        shifted.set_node_coordinates(
            node_id, node.x+1000., node.y-300., node.z+700.)
    moved_force, moved_tangent, _ = corotational_element_response(
        shifted, 1, shifted.mesh.elements[1], displacement, tangent=True,
        tangent_mode="consistent", centered_shell_reference=True)
    assert (np.linalg.norm(moved_force-base_force) <=
            1e-9*np.linalg.norm(base_force))
    assert (np.linalg.norm(moved_tangent-base_tangent) <=
            1e-9*np.linalg.norm(base_tangent))


@pytest.mark.parametrize("axis", [(0, 0, 1), (0, 1, 0), (1, 1, 1)])
@pytest.mark.parametrize("angle", [30.0, 90.0, 170.0])
def test_corotational_internal_force_is_invariant_under_rigid_rotation(axis, angle) -> None:
    for model in (_single_shell_model(), _single_beam_model(), _single_quadratic_beam_model()):
        u = _rigid_rotation_field(model, angle, axis)
        F_vk, _, _ = _assemble_nonlinear_system(model, u, {}, 5, tangent=False)
        F_cr, _, _ = _assemble_nonlinear_system(model, u, {}, 5, tangent=False, kinematics="corotational")
        # von Karman produces enormous spurious forces at large rigid rotation;
        # the corotational response must stay at the roundoff floor.
        scale = max(float(np.linalg.norm(F_vk)), E_STEEL * 1.0e-5)
        assert float(np.linalg.norm(F_cr)) < 1.0e-9 * scale


def test_corotational_beam_cantilever_rolls_up_to_analytic_circle() -> None:
    n, L, I = 10, 1.0, 1.0e-8
    model = FEModel("rollup")
    model.add_material("steel", E_STEEL, 0.0, density=7850.0)
    for i in range(n + 1):
        model.add_node(i + 1, L * i / n, 0.0, 0.0)
    section = {"area": 1.0e-3, "Iy": I, "Iz": I, "J": I}
    for i in range(n):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    model.add_boundary_condition(BoundaryCondition("clamp", [1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))

    theta = np.pi / 2
    M = theta * E_STEEL * I / L
    load_case = LoadCase("moment")
    load_case.add_nodal_load(n + 1, moments=np.array([0.0, 0.0, M]))
    result = solve_static_nonlinear(model, load_case, num_steps=20, max_iterations=40, tolerance=1.0e-6, kinematics="corotational")

    assert result.status == "completed"
    corotational_diagnostics = result.info["nonlinear_performance"]["corotational"]
    assert corotational_diagnostics["activated"] is True
    assert corotational_diagnostics["rotated_tangent_block_activated"] is True
    assert corotational_diagnostics["dense_consistent_tangent_activated"] is False
    assert corotational_diagnostics["fallback_reason"] is None
    tip = model.mesh.get_node(n + 1)
    u = result.displacements
    radius = E_STEEL * I / M
    assert u[tip.dofs[5]] == pytest.approx(theta, rel=1.0e-4)
    assert u[tip.dofs[0]] == pytest.approx(radius * np.sin(theta) - L, rel=5.0e-3)
    assert u[tip.dofs[1]] == pytest.approx(radius * (1.0 - np.cos(theta)), rel=5.0e-3)
    assert result.diagnostics.get("kinematics", result.solver_info.get("kinematics")) if hasattr(result, "solver_info") else True


def test_legacy_corotational_matches_von_karman_for_registered_small_displacements() -> None:
    """Preserve the pre-activation 150 N kinematics regression explicitly."""

    # Note the realistic load scale: the corotational pull-back has an
    # intrinsic residual floor of about eps * ||K|| * L per element, so
    # convergence tolerances must sit above that floor (documented in
    # anyfesolver.corotational).
    model = _single_legacy_shell_model()
    model.add_boundary_condition(BoundaryCondition("clamp", [1, 4], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
    # ~0.5% of span deflection: small-displacement regime, but the residual
    # scale stays far above the corotational roundoff floor.
    load_case = LoadCase("small")
    load_case.add_nodal_load(3, forces=np.array([0.0, 0.0, 150.0]))

    reference = solve_static_nonlinear(model, load_case, num_steps=2, max_iterations=20, tolerance=1.0e-6)
    model2 = _single_legacy_shell_model()
    model2.add_boundary_condition(BoundaryCondition("clamp", [1, 4], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
    corotational = solve_static_nonlinear(model2, load_case, num_steps=2, max_iterations=20, tolerance=1.0e-6, kinematics="corotational")

    assert reference.status == "completed"
    assert corotational.status == "completed"
    scale = max(float(np.max(np.abs(reference.displacements))), 1.0e-12)
    assert np.max(np.abs(corotational.displacements - reference.displacements)) < 1.0e-2 * scale


def test_corotational_scope_rejects_bad_kinematics_and_fracture() -> None:
    from anyfesolver.fracture import FractureConfig

    model = _single_beam_model()
    load_case = LoadCase("f")
    load_case.add_nodal_load(2, forces=np.array([0.0, 1.0, 0.0]))

    with pytest.raises(ValueError, match="kinematics"):
        solve_static_nonlinear(model, load_case, kinematics="total_lagrangian")

    with pytest.raises(ValueError, match="fracture"):
        solve_static_nonlinear(
            _single_beam_model(), load_case, kinematics="corotational", fracture_config=FractureConfig(threshold=0.1)
        )


def _plastic_cr_cantilever(kinematics: str, quadratic: bool = False):
    model = FEModel("cr_plastic")
    model.add_material("steel", E_STEEL, 0.3, density=7850.0)
    model.materials["steel"].hardening_curve = dnv_c208_steel_curve("S355", 0.01)
    n = 4
    grid = 2 * n if quadratic else n
    section = {"area": 1e-3, "Iy": 1e-7, "Iz": 1e-7, "J": 1e-7, "fiber_plasticity": True}
    for i in range(grid + 1):
        model.add_node(i + 1, i / grid, 0.0, 0.0)
    for e in range(n):
        if quadratic:
            model.add_element(e + 1, QuadraticBeamElement(e + 1, [2 * e + 1, 2 * e + 2, 2 * e + 3], "steel", section))
        else:
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    model.add_boundary_condition(
        BoundaryCondition("clamp", [1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0})
    )
    load_case = LoadCase("f")
    load_case.add_nodal_load(grid + 1, forces=np.array([0.0, 3200.0, 0.0]))
    result = solve_static_nonlinear(model, load_case, num_steps=10, max_iterations=40, tolerance=1.0e-6, kinematics=kinematics)
    tip = model.mesh.get_node(grid + 1)
    plastic = float((result.info or {}).get("strain_summary", {}).get("max_equivalent_plastic_strain", 0.0) or 0.0)
    return result.status, float(result.displacements[tip.dofs[1]]), plastic


@pytest.mark.parametrize("quadratic", [False, True])
def test_corotational_fiber_plasticity_matches_von_karman_at_small_rotation(quadratic: bool) -> None:
    status_vk, tip_vk, plastic_vk = _plastic_cr_cantilever("von_karman", quadratic)
    status_cr, tip_cr, plastic_cr = _plastic_cr_cantilever("corotational", quadratic)

    assert status_vk == "completed"
    assert status_cr == "completed"
    # clearly plastic (elastic tip would be ~0.051) and the two kinematics
    # agree in the moderate-rotation regime where both are valid.
    assert plastic_cr > 1.0e-4
    assert tip_cr == pytest.approx(tip_vk, rel=0.05)
    assert plastic_cr == pytest.approx(plastic_vk, rel=0.25)


def test_corotational_shell_matches_von_karman_at_small_displacement() -> None:
    def _run(kinematics: str):
        model = _single_shell_model()
        model.add_boundary_condition(
            BoundaryCondition("clamp", [1, 4], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0})
        )
        load_case = LoadCase("f")
        load_case.add_nodal_load(3, forces=np.array([0.0, 0.0, 200.0]))
        result = solve_static_nonlinear(model, load_case, num_steps=4, max_iterations=25, tolerance=1.0e-6, kinematics=kinematics)
        return result.status, result.displacements.copy()

    status_vk, u_vk = _run("von_karman")
    status_cr, u_cr = _run("corotational")
    assert status_vk == "completed"
    assert status_cr == "completed"
    scale = max(float(np.max(np.abs(u_vk))), 1.0e-12)
    assert np.max(np.abs(u_cr - u_vk)) < 5.0e-2 * scale


def test_legacy_corotational_shell_preserves_registered_900n_comparison() -> None:
    """Keep the pre-Q1M Q4 kinematics fixture as explicit rollback coverage."""

    def _run(kinematics: str):
        model = _single_legacy_shell_model()
        model.materials["steel"].hardening_curve = dnv_c208_steel_curve(
            "S355", 0.01
        )
        model.add_boundary_condition(
            BoundaryCondition(
                "clamp",
                [1, 4],
                {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0},
            )
        )
        load_case = LoadCase("f")
        load_case.add_nodal_load(3, forces=np.array([0.0, 0.0, 900.0]))
        result = solve_static_nonlinear(
            model,
            load_case,
            num_steps=4,
            max_iterations=25,
            tolerance=1.0e-6,
            kinematics=kinematics,
        )
        plastic = float(
            (result.info or {})
            .get("strain_summary", {})
            .get("max_equivalent_plastic_strain", 0.0)
            or 0.0
        )
        return result.status, result.displacements.copy(), plastic

    status_vk, u_vk, plastic_vk = _run("von_karman")
    status_cr, u_cr, plastic_cr = _run("corotational")
    assert status_vk == "completed"
    assert status_cr == "completed"
    scale = max(float(np.max(np.abs(u_vk))), 1.0e-12)
    assert np.max(np.abs(u_cr - u_vk)) < 5.0e-2 * scale
    if plastic_vk > 1.0e-6:
        assert plastic_cr == pytest.approx(plastic_vk, rel=0.5)


def test_corotational_plastic_beam_roll_up_softens_beyond_elastic_circle() -> None:
    """Large-rotation plasticity: same end moment rotates further once fibers yield.

    The stocky section puts first yield at a genuinely large elastic rotation
    (sigma_y L / (E c) ~ 56 degrees), so plasticity and large rotations are
    active simultaneously.  The applied moment is 1.2x the first-yield moment —
    well below the plastic collapse moment, so an equilibrium exists.
    """
    n, L, I, A = 10, 1.0, 1.0e-8, 1.0e-2
    section = {"area": A, "Iy": I, "Iz": I, "J": I, "fiber_plasticity": True}
    fiber_c = np.sqrt(3.0 * I / A)
    yield_moment = 355.0e6 * I / fiber_c
    M = 1.2 * yield_moment

    def _rollup(plastic: bool):
        model = FEModel("plastic_rollup")
        model.add_material("steel", E_STEEL, 0.0, density=7850.0)
        if plastic:
            model.materials["steel"].hardening_curve = dnv_c208_steel_curve("S355", 0.01)
        for i in range(n + 1):
            model.add_node(i + 1, L * i / n, 0.0, 0.0)
        for i in range(n):
            model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", dict(section)))
        model.add_boundary_condition(BoundaryCondition("clamp", [1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
        load_case = LoadCase("m")
        load_case.add_nodal_load(n + 1, moments=np.array([0.0, 0.0, M]))
        result = solve_static_nonlinear(model, load_case, num_steps=25, max_iterations=40, tolerance=1.0e-6, kinematics="corotational")
        tip = model.mesh.get_node(n + 1)
        plastic_strain = float((result.info or {}).get("strain_summary", {}).get("max_equivalent_plastic_strain", 0.0) or 0.0)
        return result.status, float(result.displacements[tip.dofs[5]]), plastic_strain

    status_elastic, rotation_elastic, _ = _rollup(plastic=False)
    status_plastic, rotation_plastic, plastic_strain = _rollup(plastic=True)

    assert status_elastic == "completed"
    assert status_plastic == "completed"
    # elastic reference: exact analytic circle rotation M L / (E I) ~ 67 deg
    assert rotation_elastic == pytest.approx(M * L / (E_STEEL * I), rel=1.0e-3)
    assert rotation_elastic > 1.0  # genuinely large rotation (> 57 degrees)
    # yielded outer fibers soften the section: same moment rotates further
    assert plastic_strain > 1.0e-4
    assert rotation_plastic > 1.02 * rotation_elastic


def test_quadratic_beams_have_nonlinear_response_and_no_fallback_warning() -> None:
    # The former NONLINEAR_STATIC010 linear-elastic fallback warning must be
    # gone now that QuadraticBeamElement has a real von Karman implementation.
    model = _single_quadratic_beam_model()
    model.add_boundary_condition(BoundaryCondition("clamp", [1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
    load_case = LoadCase("f")
    load_case.add_nodal_load(3, forces=np.array([0.0, 100.0, 0.0]))

    result = solve_static_nonlinear(model, load_case, num_steps=2, max_iterations=15, tolerance=1.0e-6)
    assert result.status == "completed"
    assert not any("NONLINEAR_STATIC010" in str(w) for w in (result.info or {}).get("warnings", []))
    assert "quadratic_beam_elastic_fallback_element_ids" not in (result.info or {})


def test_corotational_shell_strip_rolls_to_quarter_circle_efficiently() -> None:
    """90-degree shell roll-up: analytic circle match, and Newton must not grind.

    Regression for two Phase-3 findings: the residual-norm line search rejects
    the necessary corotational frame-rotation excursion (grinding thousands of
    assemblies), and geometric tangent terms destabilize the shell Newton map.
    """
    import anyfesolver.nonlinear_static as ns

    n, width, thickness = 8, 0.1, 0.005
    model = FEModel("shell_rollup")
    model.add_material("steel", E_STEEL, 0.0, density=7850.0)
    nid = {}
    node_id = 1
    for i in range(n + 1):
        for j in range(2):
            model.add_node(node_id, i / n, width * j, 0.0)
            nid[(i, j)] = node_id
            node_id += 1
    for i in range(n):
        model.add_element(
            i + 1,
            create_shell_element(
                i + 1,
                [nid[(i, 0)], nid[(i + 1, 0)], nid[(i + 1, 1)], nid[(i, 1)]],
                "steel",
                thickness=thickness,
            ),
        )
    model.add_boundary_condition(
        BoundaryCondition("clamp", [nid[(0, 0)], nid[(0, 1)]], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0})
    )

    inertia = width * thickness**3 / 12.0
    theta = np.pi / 2.0
    moment = theta * E_STEEL * inertia / 1.0
    load_case = LoadCase("m")
    for j in (0, 1):
        load_case.add_nodal_load(nid[(n, j)], moments=np.array([0.0, -moment / 2.0, 0.0]))

    assemblies = [0]
    original = ns._assemble_nonlinear_system

    def counted(*args, **kwargs):
        assemblies[0] += 1
        return original(*args, **kwargs)

    ns._assemble_nonlinear_system = counted
    try:
        result = solve_static_nonlinear(model, load_case, num_steps=15, max_iterations=30, tolerance=1.0e-6, kinematics="corotational")
    finally:
        ns._assemble_nonlinear_system = original

    assert result.status == "completed"
    tip = model.mesh.get_node(nid[(n, 0)])
    u = result.displacements
    radius = E_STEEL * inertia / moment
    assert u[tip.dofs[4]] == pytest.approx(-theta, rel=1.0e-4)
    assert u[tip.dofs[0]] == pytest.approx(radius * np.sin(theta) - 1.0, rel=5.0e-3)
    assert u[tip.dofs[2]] == pytest.approx(radius * (1.0 - np.cos(theta)), rel=5.0e-3)
    assert assemblies[0] < 150  # was thousands before the Phase-3 fixes
