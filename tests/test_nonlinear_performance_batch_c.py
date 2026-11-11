from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import arc_length, nonlinear_performance_batch_c, nonlinear_static
from anyfesolver.arc_length import ArcLengthControl
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import Element
from anyfesolver.fe_core import FEModel
from anyfesolver.jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.nonlinear_performance_batch_c import (
    assemble_reduced_system,
    batch_c_status,
    build_reduced_assembly_plan,
    reset_batch_c_counters,
)
from anyfesolver.nonlinear_performance_bootstrap import (
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
    uninstall_nonlinear_performance_optimizations,
)
from anyfesolver.nonlinear_static import DisplacementControl, solve_static_nonlinear
from anyfesolver.nonlinear_static import _ensure_nonlinear_acceleration

_ensure_nonlinear_acceleration()


def _weighted_transformation(total_dofs: int) -> sparse.csr_matrix:
    reduced_dofs = total_dofs - 2
    rows = list(range(reduced_dofs))
    columns = list(range(reduced_dofs))
    values = [1.0] * reduced_dofs
    rows.extend((reduced_dofs, reduced_dofs))
    columns.extend((0, 1))
    values.extend((0.75, 0.25))
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(total_dofs, reduced_dofs),
    )


def _selector_transformation(total_dofs: int) -> sparse.csr_matrix:
    reduced_dofs = total_dofs - 2
    active = np.arange(reduced_dofs, dtype=np.intp)
    return sparse.csr_matrix(
        (np.ones(reduced_dofs), (active, active)),
        shape=(total_dofs, reduced_dofs),
    )


def test_direct_reduced_scatter_matches_full_projection_with_mpc_coefficients() -> None:
    model = generate_simple_panel_mesh(
        1.2,
        0.8,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
    )
    nonlinear_plan = get_nonlinear_assembly_plan(model, 5)
    transformation = _weighted_transformation(nonlinear_plan.total_dofs)
    reduced_plan = build_reduced_assembly_plan(nonlinear_plan, transformation)
    assert reduced_plan.mapping_kind == "weighted_mpc"
    assert reduced_plan.force_weighted_sources.size > 0
    assert reduced_plan.tangent_weighted_sources.size > 0

    rng = np.random.default_rng(20260618)
    q = rng.normal(scale=1.0e-5, size=transformation.shape[1])
    displacement = np.asarray(transformation @ q, dtype=float).reshape(-1)

    force_full, tangent_full, states_full = nonlinear_plan.assemble(
        displacement,
        {},
        tangent=True,
    )
    force_reduced, tangent_reduced, states_reduced = assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        displacement,
        {},
        tangent=True,
    )

    np.testing.assert_allclose(
        force_reduced,
        np.asarray(transformation.T @ force_full, dtype=float).reshape(-1),
        rtol=2.0e-11,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        tangent_reduced.toarray(),
        (transformation.T @ tangent_full @ transformation).toarray(),
        rtol=2.0e-11,
        atol=1.0e-5,
    )
    assert set(states_reduced) == set(states_full)


def test_selector_reduced_plan_reuses_csr_and_work_buffers() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        0.6,
        0.01,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    nonlinear_plan = get_nonlinear_assembly_plan(model, 5)
    transformation = _selector_transformation(nonlinear_plan.total_dofs)
    reduced_plan = build_reduced_assembly_plan(nonlinear_plan, transformation)
    assert reduced_plan.mapping_kind == "selector"
    assert reduced_plan.force_weighted_sources.size == 0
    assert reduced_plan.tangent_weighted_sources.size == 0

    identities = (
        id(reduced_plan.force_buffer),
        id(reduced_plan.tangent_buffer),
        id(reduced_plan.tangent_matrix),
        id(reduced_plan.tangent_matrix.data),
    )
    displacement = np.zeros(nonlinear_plan.total_dofs, dtype=float)
    first_force, first_tangent, _ = assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        displacement,
        {},
        tangent=True,
    )
    first_force_copy = first_force.copy()
    first_tangent_copy = first_tangent.copy()
    second_force, second_tangent, _ = assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        displacement,
        {},
        tangent=True,
    )

    assert identities == (
        id(reduced_plan.force_buffer),
        id(reduced_plan.tangent_buffer),
        id(reduced_plan.tangent_matrix),
        id(reduced_plan.tangent_matrix.data),
    )
    assert second_force is reduced_plan.force_buffer
    assert second_tangent is reduced_plan.tangent_matrix
    np.testing.assert_allclose(second_force, first_force_copy)
    np.testing.assert_allclose(
        second_tangent.toarray(),
        first_tangent_copy.toarray(),
    )
    for row in range(reduced_plan.reduced_dofs):
        start = int(reduced_plan.csr_indptr[row])
        stop = int(reduced_plan.csr_indptr[row + 1])
        row_indices = reduced_plan.csr_indices[start:stop]
        assert np.all(row_indices[:-1] < row_indices[1:])


def test_lazy_full_force_payload_rejects_replaced_plan_buffers() -> None:
    model, _load = _spring_model()
    plan = get_nonlinear_assembly_plan(model, 5)
    displacement = np.zeros(plan.total_dofs, dtype=float)
    expected, _tangent, _states = plan.assemble(
        displacement,
        {},
        tangent=True,
    )
    payload = nonlinear_performance_batch_c._ReducedVectorPayload(
        np.empty(0, dtype=float),
        object(),
        plan,
        int(plan.timings.calls),
    )

    np.testing.assert_allclose(
        nonlinear_performance_batch_c.materialize_full_internal_force(
            payload,
            plan.total_dofs,
        ),
        expected,
    )
    plan.assemble(
        np.full(plan.total_dofs, 1.0e-4, dtype=float),
        {},
        tangent=True,
    )
    assert (
        nonlinear_performance_batch_c.materialize_full_internal_force(
            payload,
            plan.total_dofs,
        )
        is None
    )


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C requires Batch B's Numba kernel ({JIT_DISABLED_REASON})",
)
def test_direct_reduced_assembly_retains_batch_b_elastic_kernel(monkeypatch) -> None:
    # Single elastic law (owner decision 2026-10-23, docs/
    # Q4_ASSUMED_STRAIN_PLASTICITY_20261023.md): planar isotropic qualified Q4
    # runs the assumed-strain batch and the batch-B elastic fast path is retired
    # for it; it remains in use for legacy shells.  Pin both facts, and keep the
    # Batch C retention check on the legacy panel (same retarget as
    # tests/test_nonlinear_performance_batch_b.py::_legacy_panel).
    from anyfesolver.elements import create_shell_element

    qualified = generate_simple_panel_mesh(
        1.0, 0.6, 0.01, num_divisions_x=2, num_divisions_y=1
    )
    qualified_plan = get_nonlinear_assembly_plan(qualified, 5)
    assert qualified_plan.shell_batches
    assert all(
        batch.key[0] == "isotropic_q4_assumed_strain_plastic"
        and getattr(batch, "_batch_b_elastic", False) is False
        for batch in qualified_plan.shell_batches
    )

    model = generate_simple_panel_mesh(
        1.0,
        0.6,
        0.01,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    for element_id, element in list(model.mesh.elements.items()):
        model.mesh.elements[element_id] = create_shell_element(
            element_id, list(element.node_ids), element.material_name,
            thickness=element.thickness, formulation="legacy")
    nonlinear_plan = get_nonlinear_assembly_plan(model, 5)
    assert nonlinear_plan.shell_batches
    assert all(
        getattr(batch, "_batch_b_elastic", False)
        for batch in nonlinear_plan.shell_batches
    )
    transformation = _selector_transformation(nonlinear_plan.total_dofs)
    reduced_plan = build_reduced_assembly_plan(nonlinear_plan, transformation)

    def fail_allocating_batch_path(*_args, **_kwargs):
        raise AssertionError("Batch C bypassed the Batch B in-place shell kernel")

    for batch in nonlinear_plan.shell_batches:
        monkeypatch.setattr(batch, "evaluate", fail_allocating_batch_path)

    force, tangent, states = assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        np.zeros(nonlinear_plan.total_dofs, dtype=float),
        {},
        tangent=True,
    )
    assert force.shape == (reduced_plan.reduced_dofs,)
    assert tangent.shape == (
        reduced_plan.reduced_dofs,
        reduced_plan.reduced_dofs,
    )
    assert len(states) == model.mesh.num_elements
    assert batch_c_status()["batch_b_local_kernel_retained"] is True


class _LinearSpringElement(Element):
    def __init__(self, element_id: int, node_id: int, stiffness: float = 2.0):
        super().__init__(element_id, [node_id], "default")
        self.stiffness = float(stiffness)

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray([mesh.get_node(self.node_ids[0]).coords()], dtype=float)

    def compute_stiffness_matrix(self, mesh, material):
        matrix = np.eye(6, dtype=float)
        matrix[0, 0] = self.stiffness
        return matrix

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        displacement = np.asarray(u_elem, dtype=float)
        force = displacement.copy()
        force[0] = self.stiffness * displacement[0]
        stiffness = None
        if tangent:
            stiffness = np.eye(6, dtype=float)
            stiffness[0, 0] = self.stiffness
        return force, stiffness, {
            "spring_displacement": float(displacement[0]),
        }


def _spring_model():
    model = FEModel("batch_c_linear_spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, _LinearSpringElement(1, 1))
    model.add_boundary_condition(
        BoundaryCondition(
            "one_dof",
            [1],
            {
                "uy": 0.0,
                "uz": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
            },
        )
    )
    load = LoadCase("reference")
    load.add_nodal_load(
        1,
        load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return model, load


class _AxialSpringElement(Element):
    def __init__(self, element_id: int, node_ids, stiffness: float = 2.0):
        super().__init__(element_id, node_ids, "default")
        self.stiffness = float(stiffness)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray(
            [mesh.get_node(node_id).coords() for node_id in self.node_ids],
            dtype=float,
        )

    def compute_stiffness_matrix(self, mesh, material):
        matrix = np.zeros((12, 12), dtype=float)
        matrix[0, 0] = self.stiffness
        matrix[0, 6] = -self.stiffness
        matrix[6, 0] = -self.stiffness
        matrix[6, 6] = self.stiffness
        return matrix

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        stiffness = self.compute_stiffness_matrix(mesh, material)
        displacement = np.asarray(u_elem, dtype=float)
        force = stiffness @ displacement
        return force, stiffness if tangent else None, {
            "spring_extension": float(displacement[6] - displacement[0]),
        }


def _supported_axial_spring_model():
    model = FEModel("batch_c_supported_axial_spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, _AxialSpringElement(1, [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition(
            "fixed",
            [1],
            {
                "ux": 0.0,
                "uy": 0.0,
                "uz": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
            },
        )
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "guide",
            [2],
            {
                "uy": 0.0,
                "uz": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
            },
        )
    )
    load = LoadCase("unit_pull")
    load.add_nodal_load(
        2,
        load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return model, load


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_nonlinear_solver_uses_direct_reduced_assembly(monkeypatch) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    reset_batch_c_counters()
    model, load = _spring_model()
    result = solve_static_nonlinear(
        model,
        load_case=load,
        max_load_factor=0.25,
        num_steps=2,
        max_iterations=8,
        tolerance=1.0e-10,
    )
    status = batch_c_status()
    assert result.status == "completed"
    assert status["installed"] is True
    assert status["contexts_entered"] >= 1
    assert status["reduced_plan_builds"] == 1
    assert status["reduced_assemblies"] > 0
    assert status["full_coordinate_fallbacks"] == 0
    assert status["last_plan"]["mapping_kind"] == "selector"
    assert status["active_context_depth"] == 0
    performance = result.info["nonlinear_performance"]
    assert performance["scope"] == "analysis_local_inclusive"
    assert performance["assembly"]["path_counts"]["direct_reduced"] > 0
    direct = performance["direct_reduced_assembly"]
    assert direct["context_active"] is True
    assert direct["activated"] is True
    assert direct["fallback_reason"] is None
    assert direct["assembly_count"] == status["reduced_assemblies"]
    assert result.info["reaction_force_recovery"] == {
        "accepted_force_reuse_count": len(result.steps),
        "full_reassembly_count": 0,
    }
    solver = performance["solver"]
    assert solver["reaction_force_reuse_count"] == len(result.steps)
    assert solver["reaction_force_reassembly_count"] == 0


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_force_control_reaction_reuse_falls_back_when_payload_is_stale(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    monkeypatch.setattr(
        nonlinear_performance_batch_c,
        "materialize_full_internal_force",
        lambda *_args, **_kwargs: None,
    )
    model, load = _spring_model()
    result = solve_static_nonlinear(
        model,
        load_case=load,
        max_load_factor=0.25,
        num_steps=2,
        max_iterations=8,
        tolerance=1.0e-10,
    )

    assert result.status == "completed"
    assert result.info["reaction_force_recovery"] == {
        "accepted_force_reuse_count": 0,
        "full_reassembly_count": len(result.steps),
    }
    solver = result.info["nonlinear_performance"]["solver"]
    assert solver["reaction_force_reuse_count"] == 0
    assert solver["reaction_force_reassembly_count"] == len(result.steps)


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_block_displacement_reactions_reuse_accepted_direct_force_with_oracle_parity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    reset_batch_c_counters()
    model, load = _supported_axial_spring_model()
    progress = []
    result = solve_static_nonlinear(
        model,
        load_case=load,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2,
            dof="ux",
            target_displacement=0.02,
        ),
        num_steps=3,
        max_iterations=8,
        tolerance=1.0e-12,
        progress_callback=progress.append,
        record_increment_snapshots=True,
    )

    assert result.status == "completed"
    assert result.info["displacement_control_linearization"] == "block_elimination"
    direct = result.info["nonlinear_performance"]["direct_reduced_assembly"]
    assert direct["context_active"] is True
    assert direct["activated"] is True
    assert direct["fallback_reason"] is None
    assert result.info["reaction_force_recovery"] == {
        "accepted_force_reuse_count": len(result.steps),
        "full_reassembly_count": 0,
    }
    assert result.info["constraint_postcheck"]["status"] == "passed"
    assert len(result.snapshots) == len(result.steps)

    final_reactions = result.steps[-1].support_reactions
    fixed_x = final_reactions["fixed"][0]
    applied_x = result.load_factor
    assert abs(fixed_x) > 0.0
    assert fixed_x == pytest.approx(-applied_x, rel=1.0e-12, abs=1.0e-12)
    assert result.info["force_displacement_history"][-1]["support_reactions"] == {
        name: list(values) for name, values in final_reactions.items()
    }
    assert progress[-1]["support_reactions"] == {
        name: list(values) for name, values in final_reactions.items()
    }

    # Force the conservative full-coordinate recovery and use it as the
    # reaction oracle for the generation-qualified accepted-force payload.
    monkeypatch.setattr(
        nonlinear_performance_batch_c,
        "materialize_full_internal_force",
        lambda *_args, **_kwargs: None,
    )
    oracle_model, oracle_load = _supported_axial_spring_model()
    oracle = solve_static_nonlinear(
        oracle_model,
        load_case=oracle_load,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2,
            dof="ux",
            target_displacement=0.02,
        ),
        num_steps=3,
        max_iterations=8,
        tolerance=1.0e-12,
    )
    assert oracle.info["reaction_force_recovery"] == {
        "accepted_force_reuse_count": 0,
        "full_reassembly_count": len(oracle.steps),
    }
    assert oracle.info["constraint_postcheck"] == result.info[
        "constraint_postcheck"
    ]
    assert len(result.steps) == len(oracle.steps)
    for optimized_step, oracle_step in zip(result.steps, oracle.steps):
        assert optimized_step.load_factor == pytest.approx(
            oracle_step.load_factor,
            rel=1.0e-12,
            abs=1.0e-13,
        )
        assert optimized_step.support_reactions.keys() == (
            oracle_step.support_reactions.keys()
        )
        for name, optimized_values in optimized_step.support_reactions.items():
            np.testing.assert_allclose(
                optimized_values,
                oracle_step.support_reactions[name],
                rtol=1.0e-12,
                atol=1.0e-12,
            )


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_arc_reactions_reuse_accepted_direct_force_without_reassembly(
    monkeypatch,
) -> None:
    """Reaction output must not repeat every accepted element evaluation."""

    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    reset_batch_c_counters()
    model, load = _spring_model()
    result = arc_length.solve_static_arc_length(
        model,
        load,
        control=ArcLengthControl(
            initial_load_increment=0.05,
            minimum_load_increment=1.0e-5,
            maximum_load_increment=0.05,
            max_steps=4,
            maximum_absolute_load_factor=0.15,
        ),
        max_iterations=8,
        tolerance=1.0e-10,
        arc_tolerance=1.0e-10,
    )

    assert result.steps
    recovery = result.info["reaction_force_recovery"]
    assert recovery == {
        "accepted_force_reuse_count": len(result.steps),
        "full_reassembly_count": 0,
    }
    performance = result.info["nonlinear_performance"]
    assert result.info["corrector_tangent_projection_count"] == result.info[
        "corrector_solve_many_count"
    ]
    assert result.info["corrector_tangent_projection_count"] < result.info[
        "total_newton_iterations"
    ]
    assert performance["direct_reduced_assembly"]["activated"] is True
    assert performance["assembly"]["residual_only_calls"] == 0
    assert "persistent_full_coordinate" not in performance["assembly"][
        "path_counts"
    ]
    assert all(
        np.all(np.isfinite(values))
        for step in result.steps
        for values in step.support_reactions.values()
    )

    # Force the conservative recovery path and use it as the reaction oracle.
    monkeypatch.setattr(
        nonlinear_performance_batch_c,
        "materialize_full_internal_force",
        lambda *_args, **_kwargs: None,
    )
    fallback_model, fallback_load = _spring_model()
    fallback = arc_length.solve_static_arc_length(
        fallback_model,
        fallback_load,
        control=ArcLengthControl(
            initial_load_increment=0.05,
            minimum_load_increment=1.0e-5,
            maximum_load_increment=0.05,
            max_steps=4,
            maximum_absolute_load_factor=0.15,
        ),
        max_iterations=8,
        tolerance=1.0e-10,
        arc_tolerance=1.0e-10,
    )
    assert fallback.info["reaction_force_recovery"] == {
        "accepted_force_reuse_count": 0,
        "full_reassembly_count": len(fallback.steps),
    }
    assert len(result.steps) == len(fallback.steps)
    for optimized_step, fallback_step in zip(result.steps, fallback.steps):
        assert optimized_step.load_factor == pytest.approx(
            fallback_step.load_factor,
            rel=1.0e-12,
            abs=1.0e-13,
        )
        assert optimized_step.support_reactions.keys() == (
            fallback_step.support_reactions.keys()
        )
        for name, optimized_values in optimized_step.support_reactions.items():
            np.testing.assert_allclose(
                optimized_values,
                fallback_step.support_reactions[name],
                rtol=1.0e-12,
                atol=1.0e-12,
            )


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_result_direct_counts_stay_local_when_source_plan_is_reused(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "0")
    reset_batch_c_counters()
    model, load = _spring_model()

    first = solve_static_nonlinear(
        model,
        load_case=load,
        max_load_factor=0.25,
        num_steps=2,
        max_iterations=8,
        tolerance=1.0e-10,
    )
    second = solve_static_nonlinear(
        model,
        load_case=load,
        max_load_factor=0.25,
        num_steps=2,
        max_iterations=8,
        tolerance=1.0e-10,
    )

    first_performance = first.info["nonlinear_performance"]
    second_performance = second.info["nonlinear_performance"]
    first_direct = first_performance["direct_reduced_assembly"]
    second_direct = second_performance["direct_reduced_assembly"]
    assert first_direct["assembly_count"] > 0
    assert second_direct["assembly_count"] > 0
    assert first_direct["assembly_count"] == first_performance["assembly"][
        "path_counts"
    ]["direct_reduced"]
    assert second_direct["assembly_count"] == second_performance["assembly"][
        "path_counts"
    ]["direct_reduced"]
    assert first_direct["plan_reused"] is True
    assert second_direct["plan_reused"] is True
    assert first_performance["assembly"]["plan_reused"] is True
    assert second_performance["assembly"]["plan_reused"] is True
    assert batch_c_status()["reduced_assemblies"] == (
        first_direct["assembly_count"] + second_direct["assembly_count"]
    )


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_short_nonlinear_solve_skips_direct_reduction_setup(monkeypatch) -> None:
    monkeypatch.delenv("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", raising=False)
    reset_batch_c_counters()
    model, load = _spring_model()
    result = solve_static_nonlinear(
        model,
        load_case=load,
        max_load_factor=0.25,
        num_steps=2,
        max_iterations=8,
        tolerance=1.0e-10,
    )
    status = batch_c_status()
    assert result.status == "completed"
    assert status["reduced_plan_builds"] == 0
    assert status["reduced_assemblies"] == 0
    assert status["cost_gate_skips"] >= 1
    assert status["last_cost_gate"] == {
        "estimated_assemblies": 8,
        "activation_threshold": 144,
        "activated": False,
        "reason": "estimated_assembly_budget_below_threshold",
    }
    performance = result.info["nonlinear_performance"]
    assert performance["assembly"]["path_counts"]["persistent_full_coordinate"] > 0
    direct = performance["direct_reduced_assembly"]
    assert direct["context_active"] is True
    assert direct["activated"] is False
    assert direct["fallback_reason"] == "estimated_assembly_budget_below_threshold"


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch C installation requires Numba ({JIT_DISABLED_REASON})",
)
def test_lazy_install_updates_and_restores_prebound_consumer_aliases() -> None:
    """A caller may import either solver before the lazy installer runs."""

    consumer_name = "_anyfesolver_batch_c_prebound_consumer"
    consumer = ModuleType(consumer_name)
    uninstall_nonlinear_performance_optimizations()
    original_static = nonlinear_static.solve_static_nonlinear
    original_arc = arc_length.solve_static_arc_length
    consumer.static_solver = original_static
    consumer.arc_solver = original_arc
    sys.modules[consumer_name] = consumer

    try:
        assert install_nonlinear_performance_optimizations() is True
        assert consumer.static_solver is nonlinear_static.solve_static_nonlinear
        assert consumer.arc_solver is arc_length.solve_static_arc_length
        assert consumer.static_solver is not original_static
        assert consumer.arc_solver is not original_arc
        assert consumer.static_solver._batch_c_original is original_static
        assert consumer.arc_solver._batch_c_original is original_arc

        reset_batch_c_counters()
        model, load = _spring_model()
        static_result = consumer.static_solver(
            model,
            load_case=load,
            max_load_factor=0.25,
            num_steps=2,
            max_iterations=8,
            tolerance=1.0e-10,
        )
        zero_model, _ = _spring_model()
        arc_result = consumer.arc_solver(zero_model, LoadCase("zero"))
        status = batch_c_status()
        assert static_result.status == "completed"
        assert arc_result.status == "zero_reference_load"
        assert status["contexts_entered"] >= 2
        assert status["active_context_depth"] == 0

        uninstall_nonlinear_performance_optimizations()
        assert consumer.static_solver is original_static
        assert consumer.arc_solver is original_arc
        assert batch_c_status()["installed"] is False
        assert batch_c_status()["active_context_depth"] == 0
    finally:
        uninstall_nonlinear_performance_optimizations()
        sys.modules.pop(consumer_name, None)
        install_nonlinear_performance_optimizations()
