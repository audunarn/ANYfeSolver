from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from anyfesolver import nonlinear_performance, nonlinear_static
from anyfesolver.boundary import FixedSupport, LoadCase
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED, jit_diagnostics
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.nonlinear_performance_batch_b import batch_b_status
from anyfesolver.nonlinear_performance_bootstrap import (
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    nonlinear_performance_status,
)
from anyfesolver.runtime import _runtime_display_stresses

pytestmark = pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"Batch B in-place shell kernel requires Numba ({JIT_DISABLED_REASON})",
)

nonlinear_static._ensure_nonlinear_acceleration()


def _tilted_shell_model() -> FEModel:
    model = FEModel("tilted_elastic_shell")
    model.add_material("steel", 210.0e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.1, 0.2, 0.4)
    model.add_node(3, 1.0, 1.1, 0.9)
    model.add_node(4, -0.1, 0.9, 0.5)
    model.add_element(
        1,
        create_element("shell", 1, [1, 2, 3, 4], "steel", thickness=0.012),
    )
    return model


def _legacy_tilted_shell_model() -> FEModel:
    """Tilted shell on the legacy element (batch-B elastic path; see _legacy_panel)."""

    model = _tilted_shell_model()
    element = model.mesh.elements[1]
    from anyfesolver.elements import create_shell_element

    model.mesh.elements[1] = create_shell_element(
        1, list(element.node_ids), element.material_name, thickness=element.thickness,
        formulation="legacy")
    return model


def _legacy_panel(*args, **kwargs) -> FEModel:
    """Panel on the legacy compatible shell, which keeps the batch-B elastic path.

    Single elastic law (owner decision 2026-10-23): planar isotropic qualified
    Q4 elements run the assumed-strain kernel and no longer use the batch-B
    elastic fast path, so its coverage is exercised on legacy shells.
    """

    from anyfesolver.elements import create_shell_element

    model = generate_simple_panel_mesh(*args, **kwargs)
    for element_id, element in list(model.mesh.elements.items()):
        model.mesh.elements[element_id] = create_shell_element(
            element_id, list(element.node_ids), element.material_name,
            thickness=element.thickness, formulation="legacy")
    return model


def test_planar_qualified_q4_uses_single_law_batch_not_batch_b() -> None:
    model = generate_simple_panel_mesh(1.0, 0.6, 0.01, num_divisions_x=2, num_divisions_y=1)
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    assert plan.shell_batches
    for batch in plan.shell_batches:
        assert batch.key[0] == "isotropic_q4_assumed_strain_plastic"
        assert batch.has_plasticity is False
        assert getattr(batch, "_batch_b_elastic", False) is False
    # Execution check: the single-law batch kernel body runs, and its result is
    # what the plan assembles (equal to the scalar element responses).
    import sys
    from anyfesolver import _q4_assumed_strain_plasticity as assumed

    rng = np.random.default_rng(7)
    displacement = 1e-3 * rng.standard_normal(model.mesh.dof_manager.total_dofs)
    calls = []
    code = assumed.kernel_batch.__code__

    def observe(frame, event, value):
        if event == "return" and frame.f_code is code:
            calls.append(int(np.asarray(frame.f_locals["u_elem_batch"]).shape[0]))

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        force, _tangent, _states = plan.assemble(displacement, {}, tangent=True)
    finally:
        sys.setprofile(previous)
    assert calls == [model.mesh.num_elements]
    scalar = np.zeros_like(force)
    for element in model.mesh.elements.values():
        dofs = np.asarray(element.get_dof_mapping(model.mesh))
        material = model.get_material(element.material_name)
        scalar[dofs] += np.asarray(element.compute_nonlinear_response(
            model.mesh, material, displacement[dofs], None, 5, False)[0])
    np.testing.assert_allclose(force, scalar, rtol=0.0, atol=1e-12 * np.abs(scalar).max())

def test_batch_b_is_active() -> None:
    status = nonlinear_performance_status()
    assert status["installed"] is True
    assert status["batch_b"]["eligible"] is True
    assert status["batch_b"]["installed"] is True
    assert batch_b_status()["installed"] is True
    assert status["batch_b"]["parallel_kernel"] is True
    assert jit_diagnostics()["num_threads"] is not None


def test_tilted_elastic_shell_matches_legacy() -> None:
    model = _tilted_shell_model()
    rng = np.random.default_rng(20260618)
    displacement = rng.normal(scale=1.0e-5, size=model.mesh.dof_manager.total_dofs)
    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    force_ref, tangent_ref, states_ref = legacy(model, displacement, {}, 5, tangent=True)
    force_fast, tangent_fast, states_fast = nonlinear_static._assemble_nonlinear_system(
        model, displacement, {}, 5, tangent=True
    )
    np.testing.assert_allclose(force_fast, force_ref, rtol=2.0e-11, atol=1.0e-8)
    np.testing.assert_allclose(
        tangent_fast.toarray(), tangent_ref.toarray(), rtol=2.0e-11, atol=1.0e-5
    )
    assert set(states_fast) == set(states_ref)


def test_elastic_batch_releases_history_work_arrays() -> None:
    model = _legacy_panel(
        1.2, 0.8, 0.01, num_divisions_x=2, num_divisions_y=2
    )
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    assert plan.shell_batches
    for batch in plan.shell_batches:
        assert batch.has_plasticity is False
        assert batch.plastic_work.size == 0
        assert batch.alpha_work.size == 0
        assert getattr(batch, "_batch_b_elastic", False) is True
        first_state = batch.elastic_states[0]
        assert first_state["plastic_strain"].flags.writeable is False
        assert first_state["alpha"].flags.writeable is False
        # Elastic recovery must fall back to displacement-based stresses; a
        # fabricated zero layer_strain would overwrite them with zero stress.
        assert "layer_strain" not in first_state


def test_isotropic_batch_build_reuses_one_constitutive_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anyfesolver.nonlinear_performance_batch_b as batch_b

    model = _legacy_panel(
        1.2, 0.8, 0.01, num_divisions_x=4, num_divisions_y=2
    )
    calls = 0
    original = batch_b._shell_material_matrices

    def counted(material, angle_rad):
        nonlocal calls
        calls += 1
        return original(material, angle_rad)

    monkeypatch.setattr(batch_b, "_shell_material_matrices", counted)
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)

    assert len(plan.shell_batches) == 1
    assert calls == 1
    batch = plan.shell_batches[0]
    np.testing.assert_array_equal(
        batch._batch_b_membrane_matrix,
        np.broadcast_to(
            batch._batch_b_membrane_matrix[0],
            batch._batch_b_membrane_matrix.shape,
        ),
    )
    np.testing.assert_array_equal(
        batch._batch_b_shear_matrix,
        np.broadcast_to(
            batch._batch_b_shear_matrix[0],
            batch._batch_b_shear_matrix.shape,
        ),
    )


def test_elastic_batch_does_not_overwrite_recovered_stress_with_zero() -> None:
    model = _legacy_tilted_shell_model()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    displacement[model.mesh.get_node(2).dofs[0]] = 2.0e-4
    _force, _tangent, states = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    stresses, state_based, provenance = _runtime_display_stresses(
        model,
        displacement,
        SimpleNamespace(element_states=states),
    )
    assert state_based == set()
    assert provenance["history_aware_element_ids"] == []
    assert float(np.max(np.asarray(stresses[1]["von_mises"], dtype=float))) > 0.0


def test_final_result_recovers_elastic_layer_state_once() -> None:
    model = generate_simple_panel_mesh(
        1.0, 0.6, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    left_nodes = [
        node_id
        for node_id, node in model.mesh.nodes.items()
        if np.isclose(node.x, 0.0)
    ]
    right_nodes = [
        node_id
        for node_id, node in model.mesh.nodes.items()
        if np.isclose(node.x, 1.0)
    ]
    model.add_boundary_condition(FixedSupport("fixed", left_nodes))
    load = LoadCase("membrane")
    for node_id in right_nodes:
        load.add_nodal_load(node_id, [5.0e4, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        num_layers=5,
    )
    assert result.status == "completed"

    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    _force, _tangent, reference_states = legacy(
        model,
        result.displacements,
        {},
        5,
        tangent=False,
    )
    assert set(result.element_states) == set(reference_states)
    for element_id, reference_state in reference_states.items():
        assert "layer_strain" in result.element_states[element_id]
        np.testing.assert_allclose(
            result.element_states[element_id]["layer_strain"],
            reference_state["layer_strain"],
            rtol=2.0e-12,
            atol=1.0e-15,
        )

    reference_summary = nonlinear_static._nonlinear_state_summary(reference_states)
    assert result.info["strain_summary"]["layer_strain_min"] == pytest.approx(
        reference_summary["layer_strain_min"],
        rel=2.0e-12,
        abs=1.0e-15,
    )
    assert result.info["strain_summary"]["layer_strain_max"] == pytest.approx(
        reference_summary["layer_strain_max"],
        rel=2.0e-12,
        abs=1.0e-15,
    )
    assert nonlinear_static.states_von_mises_map(
        model,
        result.element_states,
    ) == pytest.approx(
        nonlinear_static.states_von_mises_map(model, reference_states),
        rel=2.0e-12,
        abs=1.0e-6,
    )


def test_batch_b_reuses_plan_buffers() -> None:
    model = _legacy_panel(
        1.0, 0.6, 0.01, num_divisions_x=2, num_divisions_y=1
    )
    plan = get_nonlinear_assembly_plan(model, 5)
    buffer_ids = (
        id(plan.force_values),
        id(plan.tangent_values),
        tuple(id(batch.u_work) for batch in plan.shell_batches),
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    nonlinear_static._assemble_nonlinear_system(model, displacement, {}, 5, tangent=True)
    nonlinear_static._assemble_nonlinear_system(model, displacement, {}, 5, tangent=True)
    assert buffer_ids == (
        id(plan.force_values),
        id(plan.tangent_values),
        tuple(id(batch.u_work) for batch in plan.shell_batches),
    )
    diagnostics = plan.diagnostics()
    assert diagnostics["batch_b_installed"] is True
    assert diagnostics["elastic_fast_path_element_count"] == model.mesh.num_elements
    assert diagnostics["plastic_batch_count"] == 0
