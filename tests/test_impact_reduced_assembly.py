from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import anyfesolver as fs
from anyfesolver import impact_reduced_assembly as impact_reduced_module
from anyfesolver import nonlinear_performance_bootstrap
from anyfesolver.contact import _verification_contact_panel
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.impact_reduced_assembly import prepare_impact_reduced_assembly
from anyfesolver.jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED
from anyfesolver.nonlinear_reduced_assembly import ReducedAssemblyPlanLimit


def _weighted_mpc_panel():
    model = _verification_contact_panel()
    slave = int(model.mesh.get_node(4).dofs[2])
    master_a = int(model.mesh.get_node(1).dofs[2])
    master_b = int(model.mesh.get_node(2).dofs[2])
    model.add_constraint_equation(
        terms=((slave, 1.0), (master_a, -0.75), (master_b, -0.25)),
        rhs=0.0,
        source_id="weighted-impact-uz",
        dependent_dof=slave,
    )
    return model


def _run_case(model):
    return fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=0.0025, t_end=0.04, hht_alpha=-0.05),
        fs.RigidSphereImpact(
            "direct_reduced",
            radius=0.1,
            mass=1.0,
            start_point=(0.5, 0.5, 0.12),
            travel_direction=(0.0, 0.0, -1.0),
            speed=2.0,
        ),
        fs.SphereContactConfig(
            penalty_stiffness=4000.0,
            max_contact_iterations=40,
        ),
        nonlinear_config=fs.NonlinearTransientConfig(
            enabled=True,
            max_iterations=15,
            max_cutbacks=4,
            tangent_reuse_iterations=2,
        ),
    )


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Direct reduced impact assembly requires Numba ({JIT_DISABLED_REASON})",
)
@pytest.mark.parametrize(
    ("model_factory", "mapping_kind"),
    [
        (_verification_contact_panel, "selector"),
        (_weighted_mpc_panel, "weighted_mpc"),
    ],
    ids=("selector", "weighted-mpc"),
)
def test_direct_reduced_impact_matches_full_coordinate_hht_history(
    monkeypatch,
    model_factory,
    mapping_kind,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "1000000")
    full = _run_case(model_factory())
    full_diagnostics = full.diagnostics["impact_reduced_assembly"]
    assert full_diagnostics["activated"] is False
    assert full_diagnostics["fallback_reason"] == "estimated_assembly_budget_below_threshold"
    assert full.diagnostics["direct_reduced_assembly_count"] == 0

    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    direct = _run_case(model_factory())
    direct_diagnostics = direct.diagnostics["impact_reduced_assembly"]

    assert direct.status == full.status == "completed"
    assert direct_diagnostics["activated"] is True
    assert direct_diagnostics["fallback_reason"] is None
    assert direct_diagnostics["plan"]["mapping_kind"] == mapping_kind
    assert direct.diagnostics["direct_reduced_assembly_count"] > 0
    assert direct_diagnostics["direct_reduced_residual_assembly_count"] > 0
    assert direct.diagnostics["full_coordinate_assembly_count"] < full.diagnostics[
        "full_coordinate_assembly_count"
    ]

    np.testing.assert_allclose(direct.times, full.times, rtol=0.0, atol=0.0)
    for direct_values, full_values in (
        (direct.displacements, full.displacements),
        (direct.velocities, full.velocities),
        (direct.accelerations, full.accelerations),
        (direct.sphere_positions, full.sphere_positions),
        (direct.sphere_velocities, full.sphere_velocities),
        (direct.contact_force_history, full.contact_force_history),
        (
            direct.diagnostics["strain_energy"],
            full.diagnostics["strain_energy"],
        ),
    ):
        np.testing.assert_allclose(
            direct_values,
            full_values,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
    assert direct.diagnostics["iteration_counts"] == full.diagnostics[
        "iteration_counts"
    ]
    assert direct.diagnostics["cutback_count"] == full.diagnostics["cutback_count"]


def test_direct_reduced_impact_selector_reports_all_static_exclusions(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "144")
    transformation = sparse.csr_matrix(np.asarray([[1.0], [0.5]], dtype=float))
    controller = prepare_impact_reduced_assembly(
        object(),
        transformation,
        np.asarray([0.0, 0.25]),
        num_layers=5,
        kinematics="corotational",
        plastic_damage_enabled=True,
        num_steps=1,
        max_iterations=20,
    )

    assert controller.active is False
    expected_exclusions = (
        "unsupported_kinematics",
        "plastic_damage_or_erosion_enabled",
        "affine_constraint_offset_nonzero",
        "estimated_assembly_budget_below_threshold",
    )
    if not JIT_ENABLED:
        expected_exclusions = ("jit_unavailable",) + expected_exclusions
    assert controller.fallback_reason == expected_exclusions[0]
    assert controller.exclusion_reasons == expected_exclusions

    identity = prepare_impact_reduced_assembly(
        object(),
        sparse.eye(2, format="csr"),
        np.zeros(2),
        num_layers=5,
        kinematics="von_karman",
        plastic_damage_enabled=False,
        num_steps=100,
        max_iterations=20,
    )
    assert identity.fallback_reason == "identity_constraint_transformation"


@pytest.mark.parametrize("plastic_attribute", ["hardening_curve", "hill_yield"])
def test_direct_reduced_impact_excludes_ordinary_plastic_material_history(
    monkeypatch,
    plastic_attribute,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    model = _verification_contact_panel()
    setattr(model.materials["soft"], plastic_attribute, object())

    controller = prepare_impact_reduced_assembly(
        model,
        sparse.csr_matrix(np.asarray([[1.0], [0.5]], dtype=float)),
        np.zeros(2),
        num_layers=5,
        kinematics="von_karman",
        plastic_damage_enabled=False,
        num_steps=100,
        max_iterations=20,
    )

    assert controller.active is False
    assert controller.fallback_reason == "plastic_material_history_unqualified"
    assert controller.exclusion_reasons == (
        "plastic_material_history_unqualified",
    )


@pytest.mark.parametrize(
    ("element_type", "node_ids"),
    [
        (BeamElement, [5, 6]),
        (QuadraticBeamElement, [5, 6, 7]),
    ],
    ids=("beam2", "beam3"),
)
def test_direct_reduced_impact_excludes_element_owned_fiber_plasticity(
    monkeypatch,
    element_type,
    node_ids,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    model = _verification_contact_panel()
    for node_id in node_ids:
        model.add_node(node_id, float(node_id - 5), 2.0, 0.0)
    model.add_element(
        2,
        element_type(
            2,
            node_ids,
            "soft",
            {
                "area": 1.0e-3,
                "Iy": 1.0e-7,
                "Iz": 1.0e-7,
                "J": 1.0e-7,
                "fiber_plasticity": True,
            },
        ),
    )

    controller = prepare_impact_reduced_assembly(
        model,
        sparse.csr_matrix(np.asarray([[1.0], [0.5]], dtype=float)),
        np.zeros(2),
        num_layers=5,
        kinematics="von_karman",
        plastic_damage_enabled=False,
        num_steps=100,
        max_iterations=20,
    )

    assert controller.active is False
    assert controller.fallback_reason == "beam_fiber_plasticity_unqualified"
    assert controller.exclusion_reasons == (
        "beam_fiber_plasticity_unqualified",
    )


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Direct reduced impact plan preparation requires Numba ({JIT_DISABLED_REASON})",
)
@pytest.mark.parametrize(
    ("exception", "expected_reason"),
    [
        (ReducedAssemblyPlanLimit("qualified memory cap"), "reduced_map_memory_limit"),
        (ValueError("qualified plan failure"), "reduced_plan_build_failed"),
    ],
)
def test_direct_reduced_impact_plan_failures_fall_back_observably(
    monkeypatch,
    exception,
    expected_reason,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    model = _verification_contact_panel()
    total_dofs = model.mesh.dof_manager.total_dofs
    reduced_dofs = total_dofs - 1
    transformation = sparse.csr_matrix(
        (
            np.ones(reduced_dofs),
            (np.arange(reduced_dofs), np.arange(reduced_dofs)),
        ),
        shape=(total_dofs, reduced_dofs),
    )

    def reject_plan(*_args, **_kwargs):
        raise exception

    monkeypatch.setattr(
        impact_reduced_module,
        "build_reduced_assembly_plan",
        reject_plan,
    )
    controller = prepare_impact_reduced_assembly(
        model,
        transformation,
        np.zeros(total_dofs),
        num_layers=5,
        kinematics="von_karman",
        plastic_damage_enabled=False,
        num_steps=100,
        max_iterations=20,
    )

    assert controller.active is False
    assert controller.fallback_reason == expected_reason
    assert "qualified" in str(controller.fallback_detail)


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Direct reduced impact plan preparation requires Numba ({JIT_DISABLED_REASON})",
)
def test_direct_reduced_impact_unavailable_plan_falls_back(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    monkeypatch.setattr(
        nonlinear_performance_bootstrap,
        "install_nonlinear_performance_optimizations",
        lambda: False,
    )
    controller = prepare_impact_reduced_assembly(
        object(),
        sparse.csr_matrix(np.asarray([[1.0], [0.5]], dtype=float)),
        np.zeros(2),
        num_layers=5,
        kinematics="von_karman",
        plastic_damage_enabled=False,
        num_steps=100,
        max_iterations=20,
    )

    assert controller.active is False
    assert controller.fallback_reason == "nonlinear_plan_unavailable"
