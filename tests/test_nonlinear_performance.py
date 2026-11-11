from __future__ import annotations

import copy
import threading

import numpy as np

from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver import nonlinear_static
from anyfesolver import nonlinear_performance
from anyfesolver import nonlinear_performance_bootstrap
from anyfesolver.jit_compiler import JIT_ENABLED
from anyfesolver import nonlinear_performance_batch_c
from anyfesolver.material_curves import dnv_c208_steel_curve
from anyfesolver.elements import QuadraticBeamElement, create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.nonlinear_performance_bootstrap import (
    MAX_NONLINEAR_LAYER_PLANS_PER_MODEL,
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    nonlinear_assembly_diagnostics,
    nonlinear_performance_status,
)
from anyfesolver.nonlinear_static import ShellInitialField, _prepare_initial_states


def _panel_model():
    return generate_simple_panel_mesh(
        1.2,
        0.8,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
    )


def test_performance_layer_installs_on_first_nonlinear_use() -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    status = nonlinear_performance_status()
    assert status["installed"] is True
    if JIT_ENABLED:
        assert status["batch_c"]["installed"] is True
        assert (
            nonlinear_static._assemble_nonlinear_system
            is nonlinear_performance_batch_c._batch_c_assemble_nonlinear_system
        )
    else:
        assert status["batch_c"]["installed"] is False
        assert (
            nonlinear_static._assemble_nonlinear_system
            is nonlinear_performance._optimized_assemble_nonlinear_system
        )


def test_first_use_bootstrap_waits_for_one_complete_install(monkeypatch) -> None:
    install_started = threading.Event()
    release_install = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    errors = []

    def fake_install():
        nonlocal call_count
        with call_lock:
            call_count += 1
        install_started.set()
        assert release_install.wait(timeout=2.0)
        return True

    monkeypatch.setattr(
        nonlinear_performance_bootstrap,
        "install_nonlinear_performance_optimizations",
        fake_install,
    )
    monkeypatch.setattr(nonlinear_static, "_FAST_NL_BOOTSTRAPPED", False)
    monkeypatch.setattr(nonlinear_static, "_FAST_NL_BOOTSTRAP_ERROR", None)

    def ensure():
        try:
            nonlinear_static._ensure_nonlinear_acceleration()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first = threading.Thread(target=ensure)
    second = threading.Thread(target=ensure)
    first.start()
    assert install_started.wait(timeout=2.0)
    second.start()
    release_install.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert call_count == 1
    assert nonlinear_static._FAST_NL_BOOTSTRAPPED is True
    assert errors == []


def test_composite_install_and_uninstall_are_serialized(monkeypatch) -> None:
    install_entered = threading.Event()
    release_install = threading.Event()
    uninstall_entered = threading.Event()
    errors = []

    def fake_base_install():
        install_entered.set()
        assert release_install.wait(timeout=2.0)
        return True

    def fake_base_uninstall():
        uninstall_entered.set()

    monkeypatch.setattr(
        nonlinear_performance_bootstrap, "_BASE_INSTALL", fake_base_install
    )
    monkeypatch.setattr(
        nonlinear_performance_bootstrap, "_BASE_UNINSTALL", fake_base_uninstall
    )
    monkeypatch.setattr(
        nonlinear_performance_bootstrap._batch_b,
        "install_batch_b_optimizations",
        lambda: None,
    )
    monkeypatch.setattr(
        nonlinear_performance_bootstrap._batch_b,
        "uninstall_batch_b_optimizations",
        lambda: None,
    )
    monkeypatch.setattr(
        nonlinear_performance_bootstrap._batch_c,
        "install_batch_c_optimizations",
        lambda: None,
    )
    monkeypatch.setattr(
        nonlinear_performance_bootstrap._batch_c,
        "uninstall_batch_c_optimizations",
        lambda: None,
    )

    def run(action):
        try:
            action()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    installer = threading.Thread(
        target=run,
        args=(
            nonlinear_performance_bootstrap.install_nonlinear_performance_optimizations,
        ),
    )
    uninstaller = threading.Thread(
        target=run,
        args=(
            nonlinear_performance_bootstrap.uninstall_nonlinear_performance_optimizations,
        ),
    )
    installer.start()
    assert install_entered.wait(timeout=2.0)
    uninstaller.start()
    assert not uninstall_entered.wait(timeout=0.1)
    release_install.set()
    installer.join(timeout=2.0)
    uninstaller.join(timeout=2.0)

    assert not installer.is_alive()
    assert not uninstaller.is_alive()
    assert uninstall_entered.is_set()
    assert errors == []


def test_cached_assembly_matches_legacy_shell_assembly() -> None:
    model = _panel_model()
    rng = np.random.default_rng(1042)
    displacement = rng.normal(scale=2.0e-5, size=model.mesh.dof_manager.total_dofs)
    committed = {}

    assert nonlinear_performance._ORIGINAL_ASSEMBLER is not None
    force_reference, tangent_reference, states_reference = (
        nonlinear_performance._ORIGINAL_ASSEMBLER(
            model,
            displacement,
            committed,
            5,
            tangent=True,
        )
    )
    force_fast, tangent_fast, states_fast = (
        nonlinear_static._assemble_nonlinear_system(
            model,
            displacement,
            committed,
            5,
            tangent=True,
        )
    )

    np.testing.assert_allclose(
        force_fast,
        force_reference,
        rtol=1.0e-11,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        tangent_fast.toarray(),
        tangent_reference.toarray(),
        rtol=1.0e-11,
        atol=1.0e-5,
    )
    assert set(states_fast) == set(states_reference)
    for element_id in states_reference:
        np.testing.assert_allclose(
            states_fast[element_id]["plastic_strain"],
            states_reference[element_id]["plastic_strain"],
        )
        np.testing.assert_allclose(
            states_fast[element_id]["alpha"],
            states_reference[element_id]["alpha"],
        )


def test_mixed_initial_field_shell_batch_accelerates_initialized_elastic_element() -> None:
    model = generate_simple_panel_mesh(
        1.2,
        0.8,
        0.01,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    committed, _provenance = _prepare_initial_states(
        model,
        None,
        {
            1: ShellInitialField(
                membrane_stress=[65.0e6, -8.0e6, 3.0e6],
                bending_stress=[12.0e6, -3.0e6, 1.0e6],
                membrane_prestrain=[2.0e-5, -4.0e-6, 1.0e-6],
                curvature_prestrain=[8.0e-4, -2.0e-4, 1.0e-4],
            )
        },
        5,
    )
    displacement = np.linspace(
        -2.0e-5,
        3.0e-5,
        model.mesh.dof_manager.total_dofs,
    )
    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    force_reference, tangent_reference, states_reference = legacy(
        model, displacement, committed, 5, tangent=True
    )
    force_fast, tangent_fast, states_fast = nonlinear_static._assemble_nonlinear_system(
        model, displacement, committed, 5, tangent=True
    )
    np.testing.assert_allclose(force_fast, force_reference, rtol=2.0e-11, atol=1.0e-6)
    np.testing.assert_allclose(
        tangent_fast.toarray(), tangent_reference.toarray(), rtol=2.0e-11, atol=1.0e-3
    )
    assert states_fast[1]["initial_field_provenance"] == states_reference[1]["initial_field_provenance"]
    plan = get_nonlinear_assembly_plan(model, 5)
    assert plan.timings.initial_field_accelerated_elements >= 1
    assert plan.timings.initial_field_override_elements == 0
    assert plan.diagnostics()["shell_element_count"] == 2


def test_quadratic_beam_batch_matches_scalar_rotated_elements() -> None:
    model = FEModel("beam3_batch_qualification")
    model.add_material("steel", 210.0e9, 0.29, density=7850.0)
    points = (
        (0.0, 0.0, 0.0),
        (0.5, 0.2, 0.1),
        (1.0, 0.4, 0.2),
        (0.2, 1.0, -0.1),
        (0.5, 1.4, 0.25),
        (0.8, 1.8, 0.6),
    )
    for node_id, point in enumerate(points, start=1):
        model.add_node(node_id, *point)
    first_section = {"area": 1.1e-3, "Iy": 1.7e-6, "Iz": 2.3e-6, "J": 8.0e-7}
    second_section = {
        "area": 8.0e-4,
        "Iy": 9.0e-7,
        "Iz": 1.4e-6,
        "J": 5.0e-7,
        "shear_factor_y": 0.72,
        "shear_factor_z": 0.81,
    }
    model.add_element(1, QuadraticBeamElement(1, [1, 2, 3], "steel", first_section))
    model.add_element(2, QuadraticBeamElement(2, [4, 5, 6], "steel", second_section))
    displacement = np.random.default_rng(8102026).normal(
        scale=8.0e-4,
        size=model.mesh.dof_manager.total_dofs,
    )
    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    force_reference, tangent_reference, states_reference = legacy(
        model, displacement, {}, 5, tangent=True
    )
    force_fast, tangent_fast, states_fast = nonlinear_static._assemble_nonlinear_system(
        model, displacement, {}, 5, tangent=True
    )
    np.testing.assert_allclose(force_fast, force_reference, rtol=2.0e-12, atol=1.0e-7)
    np.testing.assert_allclose(
        tangent_fast.toarray(), tangent_reference.toarray(), rtol=2.0e-12, atol=1.0e-4
    )
    assert states_fast == states_reference == {}
    diagnostics = get_nonlinear_assembly_plan(model, 5).diagnostics()
    assert diagnostics["quadratic_beam_batch_count"] == 1
    assert diagnostics["quadratic_beam_element_count"] == 2
    assert diagnostics["non_shell_element_count"] == 0


def test_plastic_shell_batch_matches_scalar_algorithmic_tangent() -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    model = _panel_model()
    first_element = next(iter(model.mesh.elements.values()))
    material = model.get_material(first_element.material_name)
    material.hardening_curve = dnv_c208_steel_curve("S355", 0.01)
    model.bump_revision("material")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 0.004 * node.x
        displacement[node.dofs[1]] = -0.001 * node.y
        displacement[node.dofs[4]] = 0.003 * node.x

    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    force_reference, tangent_reference, states_reference = legacy(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    force_fast, tangent_fast, states_fast = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    np.testing.assert_allclose(force_fast, force_reference, rtol=2.0e-11, atol=1.0e-6)
    np.testing.assert_allclose(
        tangent_fast.toarray(),
        tangent_reference.toarray(),
        rtol=2.0e-10,
        atol=1.0e-3,
    )
    for element_id in states_reference:
        np.testing.assert_allclose(states_fast[element_id]["alpha"], states_reference[element_id]["alpha"])
        np.testing.assert_allclose(
            states_fast[element_id]["plastic_strain"],
            states_reference[element_id]["plastic_strain"],
        )


def test_deleted_q4_initial_field_override_is_scaled_and_frozen_exactly() -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    model = FEModel("deleted-qualified-q4-initial-field")
    model.add_material(
        "steel",
        210.0e9,
        0.3,
        hardening_curve=dnv_c208_steel_curve("S355", 0.01),
    )
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.2, 0.0),
            (0.0, 0.2, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.01,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    committed, _provenance = _prepare_initial_states(
        model,
        None,
        {
            1: ShellInitialField(
                membrane_stress=[50.0e6, 0.0, 0.0],
            )
        },
        3,
    )
    displacement = np.linspace(
        0.0,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
        dtype=np.float64,
    )
    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    arguments = {
        "tangent": True,
        "deleted_element_ids": (1,),
        "residual_stiffness_fraction": 0.2,
    }
    # Owner decision 2026-10-23: plastic qualified Q4 outside the assumed-strain
    # route scope is refused (the compatible-strain plastic route with the
    # qualified correction is unstable once layers yield).
    import pytest
    from anyfesolver._q4_assumed_strain_plasticity import UnsupportedQ4PlasticRouteError
    with pytest.raises(UnsupportedQ4PlasticRouteError, match="active initial"):
        legacy(model, displacement, committed, 3, **arguments)
    nonlinear_performance.clear_nonlinear_assembly_cache(model)
    with pytest.raises(UnsupportedQ4PlasticRouteError, match="active initial"):
        nonlinear_performance._optimized_assemble_nonlinear_system(
            model, displacement, committed, 3, **arguments
        )


def test_q8r_uses_scalar_nonlinear_path_with_hourglass_parity() -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    model = FEModel("q8r_nonlinear")
    model.add_material("steel", 210.0e9, 0.3)
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 0.0),
    )
    for node_id, coordinate in enumerate(coordinates, start=1):
        model.add_node(node_id, *coordinate)
    element = create_element(
        "shell",
        1,
        list(range(1, 9)),
        "steel",
        thickness=0.01,
        reduced_integration=True,
    )
    model.add_element(1, element)
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    assert plan.shell_batches == ()
    assert [record.element_id for record in plan.non_shell_elements] == [1]

    displacement = np.linspace(-2.0e-5, 3.0e-5, model.mesh.dof_manager.total_dofs)
    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert legacy is not None
    force_reference, tangent_reference, _states_reference = legacy(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    force_fast, tangent_fast, _states_fast = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    np.testing.assert_allclose(force_fast, force_reference, rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(tangent_fast.toarray(), tangent_reference.toarray(), rtol=1.0e-12, atol=1.0e-5)
    assert getattr(element, "_hourglass_stiffness_matrix", None) is not None


def test_residual_only_path_matches_tangent_path_force() -> None:
    model = _panel_model()
    displacement = np.linspace(
        -1.0e-5,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
        dtype=float,
    )
    force_tangent, tangent, _ = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=True,
    )
    force_only, no_tangent, _ = nonlinear_static._assemble_nonlinear_system(
        model,
        displacement,
        {},
        5,
        tangent=False,
    )
    assert tangent is not None
    assert no_tangent is None
    np.testing.assert_allclose(
        force_only,
        force_tangent,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_plan_is_reused_until_model_revision_changes() -> None:
    model = _panel_model()
    clear_nonlinear_assembly_cache(model)
    first = get_nonlinear_assembly_plan(model, 5)
    second = get_nonlinear_assembly_plan(model, 5)
    assert first is second

    node = next(iter(model.mesh.nodes.values()))
    model.mesh.set_node_coordinates(node.id, node.x, node.y, node.z + 1.0e-6)
    third = get_nonlinear_assembly_plan(model, 5)
    assert third is not first


def test_qualified_q4_plan_rebinds_all_direct_mutable_inputs() -> None:
    model = FEModel("qualified-q4-plan-inputs")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    clear_nonlinear_assembly_cache(model)

    plan = get_nonlinear_assembly_plan(model, 3)
    assert get_nonlinear_assembly_plan(model, 3) is plan

    model.mesh.nodes[2].x += 0.125
    rebuilt_node = get_nonlinear_assembly_plan(model, 3)
    assert rebuilt_node is not plan
    assert get_nonlinear_assembly_plan(model, 3) is rebuilt_node

    model.materials["steel"].elastic_modulus *= 0.95
    rebuilt_material = get_nonlinear_assembly_plan(model, 3)
    assert rebuilt_material is not rebuilt_node
    assert get_nonlinear_assembly_plan(model, 3) is rebuilt_material

    element.thickness *= 1.10
    rebuilt_thickness = get_nonlinear_assembly_plan(model, 3)
    assert rebuilt_thickness is not rebuilt_material
    assert get_nonlinear_assembly_plan(model, 3) is rebuilt_thickness

    element.director_polarity = -1
    rebuilt_director = get_nonlinear_assembly_plan(model, 3)
    assert rebuilt_director is not rebuilt_thickness
    assert get_nonlinear_assembly_plan(model, 3) is rebuilt_director


def test_qualified_q4_plan_rebinds_wholesale_mesh_mapping_replacements() -> None:
    model = FEModel("qualified-q4-wholesale-mapping-inputs")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.01,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    clear_nonlinear_assembly_cache(model)

    original = get_nonlinear_assembly_plan(model, 3)
    original_geometry = original.shell_batches[0].B_m.copy()
    replacement_nodes = {
        node_id: copy.deepcopy(node) for node_id, node in model.mesh.nodes.items()
    }
    replacement_nodes[2].x = 0.75
    token_before_nodes = model.mesh._qualified_direct_state_token[0]
    model.mesh.nodes = replacement_nodes
    assert model.mesh._qualified_direct_state_token[0] == token_before_nodes + 1
    assert replacement_nodes[2]._qualified_direct_state_token is (
        model.mesh._qualified_direct_state_token
    )

    rebuilt_nodes = get_nonlinear_assembly_plan(model, 3)
    assert rebuilt_nodes is not original
    assert not np.array_equal(rebuilt_nodes.shell_batches[0].B_m, original_geometry)

    replacement = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.04,
        reference_normal=(0.0, 0.0, 1.0),
    )
    token_before_elements = model.mesh._qualified_direct_state_token[0]
    model.mesh.elements = {1: replacement}
    assert model.mesh._qualified_direct_state_token[0] == token_before_elements + 1
    assert replacement._qualified_direct_state_token is (
        model.mesh._qualified_direct_state_token
    )

    rebuilt_elements = get_nonlinear_assembly_plan(model, 3)
    assert rebuilt_elements is not rebuilt_nodes
    assert rebuilt_elements.shell_batches[0].elements == (replacement,)
    assert rebuilt_elements.shell_batches[0].thickness == 0.04


def test_layer_plan_cache_is_bounded_lru_and_reports_evictions() -> None:
    model = _panel_model()
    clear_nonlinear_assembly_cache()
    built = {}
    requested_layers = list(range(1, MAX_NONLINEAR_LAYER_PLANS_PER_MODEL + 3))
    for layers in requested_layers:
        built[layers] = get_nonlinear_assembly_plan(model, layers)

    diagnostics = nonlinear_assembly_diagnostics(model)
    retained_layers = [int(layers) for layers in diagnostics]
    assert retained_layers == requested_layers[-MAX_NONLINEAR_LAYER_PLANS_PER_MODEL:]
    status = nonlinear_performance_status()
    assert (
        status["cache_policy"]["max_layer_plans_per_model"]
        == MAX_NONLINEAR_LAYER_PLANS_PER_MODEL
    )
    assert status["cache_policy"]["layer_plan_evictions"] == 2

    rebuilt = get_nonlinear_assembly_plan(model, requested_layers[0])
    assert rebuilt is not built[requested_layers[0]]
    assert (
        len(nonlinear_assembly_diagnostics(model))
        == MAX_NONLINEAR_LAYER_PLANS_PER_MODEL
    )
    assert nonlinear_performance_status()["cache_policy"]["layer_plan_evictions"] == 3


def test_csr_pattern_is_reused_and_contains_unique_entries() -> None:
    model = _panel_model()
    plan = get_nonlinear_assembly_plan(model, 5)
    assert plan.csr_indices.size == plan.nnz
    assert plan.csr_indptr.size == model.mesh.dof_manager.total_dofs + 1
    for row in range(plan.total_dofs):
        start = int(plan.csr_indptr[row])
        stop = int(plan.csr_indptr[row + 1])
        row_indices = plan.csr_indices[start:stop]
        assert np.all(row_indices[:-1] < row_indices[1:])


def test_legacy_path_can_be_restored_for_ab_measurements() -> None:
    nonlinear_performance.uninstall_nonlinear_performance_optimizations()
    try:
        assert (
            nonlinear_static._assemble_nonlinear_system
            is nonlinear_performance._ORIGINAL_ASSEMBLER
        )
    finally:
        nonlinear_performance.install_nonlinear_performance_optimizations()
    if JIT_ENABLED:
        assert (
            nonlinear_static._assemble_nonlinear_system
            is nonlinear_performance_batch_c._batch_c_assemble_nonlinear_system
        )
    else:
        assert (
            nonlinear_static._assemble_nonlinear_system
            is nonlinear_performance._optimized_assemble_nonlinear_system
        )
