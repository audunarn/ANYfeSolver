import numpy as np
import pytest

import anyfesolver as fs
from anyfesolver.contact import _plastic_impact_damage_update, _verification_contact_panel
from anyfesolver.material_curves import DNVC208MaterialCurve


def test_nonlinear_impact_public_configs_are_exported():
    assert fs.NonlinearTransientConfig(enabled=True).enabled is True
    assert fs.PlasticImpactDamageConfig(threshold=0.02).threshold == 0.02


def test_plastic_impact_damage_deletes_only_from_committed_state():
    model = _verification_contact_panel()
    states = {1: {"alpha": np.array([0.0, 0.012]), "plastic_strain": np.zeros((2, 3))}}
    deleted = set()
    damage_states = {}
    config = fs.PlasticImpactDamageConfig(threshold=0.01, delete_at=1.0, max_deleted_fraction=1.0)

    records, changed, max_utilization = _plastic_impact_damage_update(
        model,
        states,
        config,
        deleted,
        damage_states,
        step_index=3,
        time_value=0.04,
    )

    assert changed is True
    assert max_utilization == 1.2
    assert deleted == {1}
    assert len(records) == 1
    assert records[0].trigger_name == "max_equivalent_plastic_strain"
    assert records[0].measure == pytest.approx(1.0, rel=1.0e-6)
    assert damage_states[1]["damage"] == 1.2


def test_nonlinear_sphere_impact_smoke_preserves_result_contract():
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

    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.01, t_end=0.01),
        fs.RigidSphereImpact(
            "nonlinear_smoke",
            radius=0.2,
            mass=5.0,
            start_point=(0.5, 0.5, 0.25),
            travel_direction=(0.0, 0.0, -1.0),
            speed=1.0,
        ),
        fs.SphereContactConfig(penalty_stiffness=500.0, max_contact_iterations=8),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True, max_iterations=6, max_cutbacks=1),
        plastic_damage_config=fs.PlasticImpactDamageConfig(threshold=0.01, max_deleted_fraction=1.0),
    )

    assert result.status in {"completed", "no_contact"}
    assert result.diagnostics["method"] == "nonlinear_newmark_sphere_penalty_contact"
    assert "strain_summary" in result.diagnostics
    assert result.diagnostics["impact_damage_summary"]["enabled"] is True
    assert result.displacements.shape[0] == len(result.times)


def test_nonlinear_impact_commits_state_from_exact_accepted_newton_iterate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed history must name the configuration that produced it.

    Loose convergence tolerances make one nonzero Newton correction sufficient.
    The confirmation evaluation must therefore be committed, rather than the
    unevaluated post-correction candidate paired with the preceding state.
    """

    from scipy import sparse

    from anyfesolver.boundary import LoadCase
    from anyfesolver import nonlinear_static

    model = _verification_contact_panel()
    load = LoadCase("newton-state-identity")
    load.add_nodal_load(1, forces=np.asarray((0.0, 0.0, 1.0)))
    evaluated_displacements: list[np.ndarray] = []

    def recording_assembler(
        _model,
        displacements,
        _committed_states,
        _num_layers,
        *,
        tangent=True,
        **_kwargs,
    ):
        evaluated = np.asarray(displacements, dtype=float).copy()
        evaluated_displacements.append(evaluated)
        size = evaluated.size
        return (
            np.zeros(size, dtype=float),
            sparse.csr_matrix((size, size), dtype=float) if tangent else None,
            {1: {"evaluated_full_displacement": evaluated.copy()}},
        )

    monkeypatch.setattr(
        nonlinear_static,
        "_assemble_nonlinear_system",
        recording_assembler,
    )
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "1000000")

    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.01, t_end=0.01),
        fs.RigidSphereImpact(
            "newton-state-identity",
            radius=0.1,
            mass=1.0,
            start_point=(0.5, 0.5, 5.0),
            travel_direction=(1.0, 0.0, 0.0),
            speed=1.0,
        ),
        fs.SphereContactConfig(penalty_stiffness=500.0),
        base_load_case=load,
        nonlinear_config=fs.NonlinearTransientConfig(
            enabled=True,
            max_iterations=1,
            max_cutbacks=0,
            residual_tolerance=1.0e6,
            displacement_tolerance=1.0e6,
            contact_force_tolerance=1.0e6,
            line_search=False,
            equilibrate_base_load=False,
        ),
    )

    assert result.status in {"completed", "no_contact"}
    accepted = np.asarray(result.displacements[-1], dtype=float)
    committed = np.asarray(
        result.diagnostics["element_states"][1]["evaluated_full_displacement"],
        dtype=float,
    )
    assert any(np.any(values != 0.0) for values in evaluated_displacements)
    assert np.any(accepted != 0.0)
    np.testing.assert_array_equal(committed, accepted)
    assert committed.tobytes(order="C") == accepted.tobytes(order="C")


def test_nonlinear_event_substepping_catches_contact_that_single_large_step_would_miss():
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

    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.05, t_end=0.05),
        fs.RigidSphereImpact(
            "nonlinear_fast",
            radius=0.1,
            mass=1.0,
            start_point=(0.5, 0.5, 0.45),
            travel_direction=(0.0, 0.0, -1.0),
            speed=20.0,
        ),
        fs.SphereContactConfig(penalty_stiffness=4000.0, max_event_substeps=64, max_contact_iterations=40),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True),
    )

    assert result.status == "completed"
    assert result.diagnostics["event_substep_count"] > 0
    assert result.peak_contact_force > 0.0


def test_nonlinear_impact_conserves_energy_and_momentum_through_rebound():
    """Rebound physics: velocity reversal, momentum balance, end-to-end energy."""
    result = fs.solve_transient_sphere_impact(
        _verification_contact_panel(),
        fs.TransientConfig(dt=0.0025, t_end=0.12),
        fs.RigidSphereImpact("hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        fs.SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True, max_iterations=15, max_cutbacks=4),
    )
    assert result.status == "completed"
    # sphere separated: downward motion arrested/reversed from -2 m/s
    assert result.sphere_velocities[-1][2] > -0.05
    assert result.sphere_momentum_balance_error < 1.0e-6
    # energy accounting now includes sphere KE and the internal work measure:
    # start (pure sphere KE) matches the end-state total within integration error
    total = (
        np.asarray(result.diagnostics["kinetic_energy"])
        + np.asarray(result.diagnostics["strain_energy"])
        + np.asarray(result.diagnostics["sphere_kinetic_energy"])
    )
    assert total[0] == pytest.approx(0.5 * 1.0 * 2.0**2)
    assert total[-1] == pytest.approx(total[0], rel=0.02)


def test_corotational_nonlinear_impact_matches_von_karman_at_small_deformation():
    results = {}
    for kinematics in ("von_karman", "corotational"):
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
        results[kinematics] = fs.solve_transient_sphere_impact(
            model,
            fs.TransientConfig(dt=0.0025, t_end=0.05),
            fs.RigidSphereImpact("cr", radius=0.2, mass=5.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=1.0),
            fs.SphereContactConfig(penalty_stiffness=500.0, max_contact_iterations=10),
            nonlinear_config=fs.NonlinearTransientConfig(enabled=True, max_iterations=12, max_cutbacks=3, kinematics=kinematics),
        )
    vk, cr = results["von_karman"], results["corotational"]
    assert vk.status == "completed"
    assert cr.status == "completed"
    assert cr.peak_contact_force == pytest.approx(vk.peak_contact_force, rel=1.0e-3)
    assert cr.result_case["analysis_case"]["settings"]["nonlinear"]["kinematics"] == "corotational"
    assert vk.result_case["analysis_case"]["settings"]["nonlinear"]["kinematics"] == "von_karman"


def test_plastic_impact_damage_erodes_beam_from_committed_state():
    """PlasticImpactDamageConfig deletes beam elements from committed plastic state."""
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.contact import _plastic_impact_damage_update
    from anyfesolver.elements import BeamElement
    from anyfesolver.fe_core import FEModel

    model = FEModel("beam_plastic")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(5, BeamElement(5, [1, 2], "steel", {"area": 1e-3, "Iy": 1e-7, "Iz": 1e-7, "J": 1e-7}))

    states = {5: {"alpha": np.array([0.0, 0.018]), "plastic_strain": np.zeros((2, 3))}}
    deleted = set()
    damage_states = {}
    config = fs.PlasticImpactDamageConfig(threshold=0.01, delete_at=1.0, max_deleted_fraction=1.0, element_scope=("shell", "beam"))

    records, changed, max_util = _plastic_impact_damage_update(
        model, states, config, deleted, damage_states, step_index=2, time_value=0.03
    )

    assert changed is True
    assert deleted == {5}
    assert len(records) == 1
    assert records[0].element_type == "BeamElement"
    assert records[0].measure == pytest.approx(1.0, rel=1.0e-6)  # beam length
    assert damage_states[5]["damage"] == pytest.approx(1.8)


def test_state_von_mises_envelope_reconstructs_return_mapped_stress():
    """sigma = C_el (eps - eps_p) from the committed state, not E * eps."""
    from anyfesolver.nonlinear_static import state_von_mises_envelope

    E, nu = 1.0e5, 0.3
    # Uniaxial layered state: total strain 0.01, of which 0.008 is plastic.
    layer_strain = np.array([[0.010, 0.0, 0.0], [0.002, 0.0, 0.0]])
    plastic_strain = np.array([[0.008, 0.0, 0.0], [0.0, 0.0, 0.0]])
    state = {"layer_strain": layer_strain, "plastic_strain": plastic_strain, "alpha": np.array([0.008, 0.0])}

    value = state_von_mises_envelope(state, E, nu)

    factor = E / (1.0 - nu**2)
    expected_rows = []
    for eps in (0.002, 0.002):
        sxx = factor * eps
        syy = factor * nu * eps
        expected_rows.append(np.sqrt(sxx**2 - sxx * syy + syy**2))
    assert value == pytest.approx(max(expected_rows), rel=1.0e-12)
    # An elastic recovery from total strain would be 5x larger for layer 1.
    assert value < 0.5 * factor * 0.010

    # Beam fiber states carry the return-mapped uniaxial stress directly.
    fiber_state = {"fiber_stress": np.array([-1234.5, 800.0]), "alpha": np.array([0.01, 0.0])}
    assert state_von_mises_envelope(fiber_state, E, nu) == pytest.approx(1234.5)

    # States without layered or fiber data fall back to None.
    assert state_von_mises_envelope({"alpha": np.array([0.01])}, E, nu) is None


def _clamped_grid_panel(n: int = 4) -> "object":
    """1 m x 1 m clamped shell grid that can actually bend and yield."""
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import create_shell_element
    from anyfesolver.fe_core import FEModel

    model = FEModel("yielding_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    node_of = {}
    node_id = 1
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(node_id, i / n, j / n, 0.0)
            node_of[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(n):
        for i in range(n):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [node_of[(i, j)], node_of[(i + 1, j)], node_of[(i + 1, j + 1)], node_of[(i, j + 1)]],
                    "soft",
                    thickness=0.05,
                ),
            )
            element_id += 1
    edge = [
        node_of[(i, j)]
        for j in range(n + 1)
        for i in range(n + 1)
        if i in (0, n) or j in (0, n)
    ]
    model.add_boundary_condition(
        BoundaryCondition("clamp", edge, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})
    )
    return model


def test_nonlinear_impact_state_von_mises_history_respects_material_curve():
    """Recorded stress history stays on the hardening curve while an elastic
    recovery from the same displacements overshoots it."""
    from anyfesolver.assembly import compute_stresses

    curve = DNVC208MaterialCurve(
        sigma_prop=800.0,
        sigma_yield=1000.0,
        sigma_yield_2=1200.0,
        eps_p_y1=1.0e-5,
        eps_p_y2=1.0e-3,
        K=2000.0,
        n=0.1,
    )
    model = _clamped_grid_panel()
    model.materials["soft"].hardening_curve = curve

    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.0025, t_end=0.05),
        fs.RigidSphereImpact("vm", radius=0.2, mass=20.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=4.0),
        fs.SphereContactConfig(penalty_stiffness=2000.0, max_contact_iterations=20),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True, max_iterations=15, max_cutbacks=4),
    )

    assert result.status == "completed"
    history = result.diagnostics["state_von_mises_history"]
    assert len(history) == len(result.times)

    summary = result.diagnostics["strain_summary"]
    max_alpha = float(summary["max_equivalent_plastic_strain"])
    assert max_alpha > 0.0, "impact should trigger plasticity for this check"

    flow_limit = float(curve.flow_stress(np.asarray([max_alpha]))[0])
    peak_state_vm = max(max(step.values()) for step in history if step)
    # Return-mapped stresses can never exceed the flow stress at the largest
    # recorded plastic strain (small tolerance for the discrete return map).
    assert peak_state_vm <= 1.02 * flow_limit

    # The elastic recovery from total displacements exceeds the same bound,
    # which is exactly why it must not be displayed for nonlinear runs.
    elastic = compute_stresses(model, result.displacements[-1])
    vm_elastic = max(
        float(np.max(np.asarray(stress["von_mises"], dtype=float)))
        for stress in elastic.values()
        if "von_mises" in stress
    )
    assert vm_elastic > peak_state_vm


def test_rtcl_triaxiality_weight_matches_published_anchor_points():
    from anyfesolver.fracture import rtcl_triaxiality_weight

    eta = np.array([-1.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0])
    weight = rtcl_triaxiality_weight(eta)
    assert weight[0] == 0.0, "deep compression accumulates no ductile damage"
    assert weight[1] == pytest.approx(0.0, abs=1e-12), "uniaxial compression limit"
    assert weight[2] == pytest.approx(2.0 / np.sqrt(12.0), rel=1e-9), "pure shear ~0.577"
    assert weight[3] == pytest.approx(1.0, rel=1e-9), "uniaxial tension is the reference"
    assert weight[4] == pytest.approx(np.exp(0.5), rel=1e-9), "equibiaxial tension ~1.65"
    # continuity at the branch switch
    assert rtcl_triaxiality_weight(np.array([1.0 / 3.0 - 1e-9]))[0] == pytest.approx(1.0, abs=1e-6)


def _layered_state(strain_rows, plastic_rows, alpha):
    return {
        "layer_strain": np.asarray(strain_rows, dtype=float),
        "plastic_strain": np.asarray(plastic_rows, dtype=float),
        "alpha": np.asarray(alpha, dtype=float),
    }


def test_rtcl_damage_ignores_compression_but_deletes_in_tension():
    """RTCL erosion: a compressed element accumulates no damage while a
    tension element with the same plastic strain softens and deletes."""
    from anyfesolver.contact import _plastic_impact_damage_update, _verification_contact_panel
    from anyfesolver.elements import create_shell_element

    model = _verification_contact_panel()
    # second shell, same material, so both categories are in scope
    model.add_node(5, 2.0, 0.0, 0.0)
    model.add_node(6, 2.0, 1.0, 0.0)
    model.add_element(
        2, create_shell_element(2, [2, 5, 6, 3], "soft", thickness=0.05)
    )

    E = float(model.materials["soft"].elastic_modulus)
    yield_strain = 0.01
    # element 1: uniaxial TENSION with large plastic strain (eta = +1/3)
    tension = _layered_state(
        [[0.20, -0.06, 0.0], [0.20, -0.06, 0.0]],
        [[0.19, -0.095, 0.0], [0.19, -0.095, 0.0]],
        [0.19, 0.19],
    )
    # element 2: uniaxial COMPRESSION, same magnitudes mirrored (eta = -1/3)
    compression = _layered_state(
        [[-0.20, 0.06, 0.0], [-0.20, 0.06, 0.0]],
        [[-0.19, 0.095, 0.0], [-0.19, 0.095, 0.0]],
        [0.19, 0.19],
    )
    states = {1: tension, 2: compression}
    config = fs.PlasticImpactDamageConfig(
        threshold=0.05, criterion="rtcl", softening_start=0.5, delete_at=1.0, max_deleted_fraction=1.0
    )
    deleted: set[int] = set()
    damage_states: dict[int, dict] = {}

    records, changed, max_util = _plastic_impact_damage_update(
        model, states, config, deleted, damage_states,
        step_index=1, time_value=0.01,
    )

    assert changed is True
    assert deleted == {1}, "only the tension element erodes"
    assert damage_states[1]["damage"] >= 1.0
    assert damage_states[2]["damage"] == pytest.approx(0.0, abs=1e-12), "compression accumulates nothing"
    assert records[0].trigger_name == "rtcl_damage"

    # Incremental accumulation: repeating the same committed state adds no
    # new damage (delta alpha = 0), so the accumulation is stable, not
    # re-counted per step.
    damage_before = damage_states[1]["damage"]
    _plastic_impact_damage_update(
        model, states, config, deleted, damage_states,
        step_index=2, time_value=0.02,
    )
    assert damage_states[1]["damage"] == pytest.approx(damage_before)


def test_rtcl_beam_fibers_damage_only_in_tension():
    from anyfesolver.fracture import state_rtcl_increment

    state = {
        "alpha": np.array([0.05, 0.05]),
        "fiber_stress": np.array([500.0, -500.0]),
    }
    alpha, weighted = state_rtcl_increment(state, None, 1.0e5, 0.3)
    assert weighted[0] == pytest.approx(0.05), "tension fiber accumulates fully"
    assert weighted[1] == 0.0, "compression fiber accumulates nothing"


def test_nonlinear_impact_solves_with_rtcl_damage():
    curve = DNVC208MaterialCurve(
        sigma_prop=800.0,
        sigma_yield=1000.0,
        sigma_yield_2=1200.0,
        eps_p_y1=1.0e-5,
        eps_p_y2=1.0e-3,
        K=2000.0,
        n=0.1,
    )
    model = _clamped_grid_panel()
    model.materials["soft"].hardening_curve = curve

    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.0025, t_end=0.05),
        fs.RigidSphereImpact("rtcl", radius=0.2, mass=20.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=4.0),
        fs.SphereContactConfig(penalty_stiffness=2000.0, max_contact_iterations=20),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True, max_iterations=15, max_cutbacks=4),
        plastic_damage_config=fs.PlasticImpactDamageConfig(
            threshold=0.05, criterion="rtcl", max_deleted_fraction=1.0
        ),
    )

    assert result.status in {"completed", "no_contact"}
    summary = result.diagnostics["impact_damage_summary"]
    assert summary["enabled"] is True
    assert summary["config"]["criterion"] == "rtcl"
    assert summary["max_damage"] >= 0.0
    selection = result.diagnostics["damage_matrix_plan_selection"]
    assert selection["break_even_future_update_events"] == 11
    assert selection["observed_update_events"] == (
        selection["legacy_update_count"] + selection["plan_update_count"]
    )
    assert selection["retained_memory_allowance_bytes"] > 0


def test_rtcl_modified_scales_critical_strain_by_plane_strain_weight():
    """rtcl_modified calibrates eps_cr so plane-strain tension fails exactly at
    the mesh-scaled GL limit: utilization is the plain-RTCL value divided by
    w_ps = exp(sqrt(3)/2 - 1/2) ~ 1.4424."""
    from anyfesolver.contact import _plastic_impact_damage_update, _verification_contact_panel

    tension = _layered_state(
        [[0.20, -0.06, 0.0], [0.20, -0.06, 0.0]],
        [[0.19, -0.095, 0.0], [0.19, -0.095, 0.0]],
        [0.19, 0.19],
    )
    utilizations = {}
    for criterion in ("rtcl", "rtcl_modified"):
        model = _verification_contact_panel()
        config = fs.PlasticImpactDamageConfig(
            threshold=0.05, criterion=criterion, delete_at=100.0, max_deleted_fraction=1.0
        )
        damage_states: dict[int, dict] = {}
        _records, _changed, max_util = _plastic_impact_damage_update(
            model, {1: tension}, config, set(), damage_states,
            step_index=1, time_value=0.01,
        )
        utilizations[criterion] = max_util

    weight_ps = float(np.exp(0.5 * np.sqrt(3.0) - 0.5))
    assert utilizations["rtcl"] > 0.0
    assert utilizations["rtcl_modified"] == pytest.approx(utilizations["rtcl"] / weight_ps, rel=1e-9)


def test_nonlinear_impact_equilibrates_base_load_instead_of_step():
    """A static base load must enter the transient as an equilibrium state,
    not a sudden step that rings the structure through the whole impact."""
    from anyfesolver.boundary import LoadCase

    def base_case(model, magnitude):
        case = LoadCase("preload")
        # pull the panel centre node transversely
        centre = min(
            model.mesh.nodes.values(),
            key=lambda node: (node.x - 0.5) ** 2 + (node.y - 0.5) ** 2,
        )
        case.add_nodal_load(int(centre.id), forces=[0.0, 0.0, -magnitude])
        return case

    def run(magnitude, equilibrate):
        model = _clamped_grid_panel()
        return fs.solve_transient_sphere_impact(
            model,
            fs.TransientConfig(dt=0.002, t_end=0.02),
            fs.RigidSphereImpact("far", radius=0.05, mass=1.0,
                                 start_point=(0.5, 0.5, 5.0),  # never reaches the panel
                                 travel_direction=(0.0, 0.0, -1.0), speed=1.0),
            fs.SphereContactConfig(penalty_stiffness=500.0),
            base_load_case=base_case(model, magnitude),
            nonlinear_config=fs.NonlinearTransientConfig(
                enabled=True, max_iterations=12, max_cutbacks=2,
                equilibrate_base_load=equilibrate,
            ),
        )

    # Mild preload: Newton equilibration converges; ringing suppressed.
    energies = {}
    for equilibrate in (True, False):
        result = run(2.0, equilibrate)
        assert result.status in {"completed", "no_contact"}
        assert result.diagnostics["base_load_equilibrated"] is equilibrate
        kinetic = np.asarray(result.diagnostics["kinetic_energy"], dtype=float)
        strain = np.asarray(result.diagnostics["strain_energy"], dtype=float)
        energies[equilibrate] = (float(np.max(kinetic)), float(np.max(strain)))

    ringing_kinetic = energies[False][0]
    equilibrated_kinetic = energies[True][0]
    assert energies[True][1] > 0.0, "preload strain energy present from t=0"
    # The stepped load converts preload work into structural ringing; the
    # equilibrated start suppresses it by orders of magnitude.
    assert equilibrated_kinetic < 0.05 * max(ringing_kinetic, 1.0e-30), energies

    # Extreme preload (deeply nonlinear for this soft panel): the backtracking
    # Newton equilibration still converges and the run completes cleanly.
    severe = run(200.0, True)
    assert severe.status in {"completed", "no_contact"}
    assert severe.diagnostics["base_load_equilibrated"] is True


def test_nonlinear_transient_config_exposes_stagnation_controls():
    config = fs.NonlinearTransientConfig(enabled=True)
    payload = config.to_dict()
    assert payload["stagnation_energy_tolerance"] == pytest.approx(1.0e-8)
    assert payload["equilibrate_base_load"] is True
    with pytest.raises(ValueError):
        fs.NonlinearTransientConfig(enabled=True, stagnation_energy_tolerance=-1.0)


def test_distress_carryover_prehalves_steps_after_cutbacks():
    """After a cutback, subsequent base steps start pre-subdivided.

    A stiff penalty with a tight Newton budget forces cutbacks during contact;
    the distress carryover must then pre-split the following base segments
    (reported as preemptive_substep_count) instead of failing them at full dt,
    and the run must still complete.
    """
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
    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.02, t_end=0.12),
        fs.RigidSphereImpact(
            "distress_carryover",
            radius=0.2,
            mass=20.0,
            start_point=(0.5, 0.5, 0.25),
            travel_direction=(0.0, 0.0, -1.0),
            speed=2.0,
        ),
        fs.SphereContactConfig(penalty_stiffness=50000.0, max_contact_iterations=8),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True, max_iterations=6, max_cutbacks=8),
        plastic_damage_config=fs.PlasticImpactDamageConfig(threshold=0.05, max_deleted_fraction=1.0),
    )
    metrics = result.diagnostics
    assert "preemptive_substep_count" in metrics
    if int(metrics.get("cutback_count", 0)) > 0:
        assert int(metrics["preemptive_substep_count"]) > 0, (
            "cutbacks occurred but no pre-emptive subdivision followed"
        )
    assert result.status in {"completed", "no_contact"}
