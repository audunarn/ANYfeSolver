from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest

import anyfesolver.assembly as assembly_module
import anyfesolver.elements as elements_module
import anyfesolver.matrix_assembly as matrix_assembly_module
import anyfesolver.recovery_batches as recovery_batches_module
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel, FEMesh
from anyfesolver.matrix_assembly import AssemblyError


def _qualified_q4_model() -> tuple[FEModel, QualifiedE4PLShellElement]:
    model = FEModel("trusted-loop-authority")
    model.add_material("steel", 210.0e9, 0.3)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    element = QualifiedE4PLShellElement(
        1,
        (1, 2, 3, 4),
        "steel",
        thickness=0.05,
    )
    model.add_element(1, element)
    return model, element


def _assert_recovery_equal(actual: Any, expected: Any) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[name], value)
        else:
            assert actual[name] == value


def test_qualified_direct_state_epoch_is_monotonic_fixed_and_copyable() -> None:
    mesh = FEMesh()
    token = mesh._qualified_direct_state_token
    assert isinstance(token, list)
    assert token == [0]

    mesh.add_node(1, 0.0, 0.0, 0.0)
    current = token[0]
    assert current > 0

    with pytest.raises(ValueError, match="only advance by one"):
        token[0] = current - 1
    with pytest.raises(ValueError, match="only advance by one"):
        token[0] = current + 2
    with pytest.raises(TypeError, match="fixed length"):
        token.append(current + 1)
    assert token == [current]

    token[0] = current + 1
    assert token == [current + 1]

    copied = copy.deepcopy(token)
    restored = pickle.loads(pickle.dumps(token))
    assert copied == token and copied is not token
    assert restored == token and restored is not token
    copied[0] = copied[0] + 1
    restored[0] = restored[0] + 1


def test_all_qualified_lease_exposes_constant_time_aba_safe_check() -> None:
    model, element = _qualified_q4_model()
    lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="trusted-loop test",
    )
    trusted = vars(lease).get("_qualified_trusted_require")
    assert callable(trusted)

    for _ in range(1_000):
        trusted(model, context="trusted-loop repeat")

    original = element.thickness
    element.thickness = original * 2.0
    element.thickness = original
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        trusted(model, context="trusted-loop element ABA")


def test_trusted_check_rejects_module_aba_and_mixed_model_has_no_fast_path() -> None:
    model, _element = _qualified_q4_model()
    lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="trusted-loop module test",
    )
    trusted = vars(lease).get("_qualified_trusted_require")
    assert callable(trusted)
    original = matrix_assembly_module._assemble_element_matrix_under_lease

    def replacement(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("replacement must never run")

    setattr(
        matrix_assembly_module,
        "_assemble_element_matrix_under_lease",
        replacement,
    )
    setattr(
        matrix_assembly_module,
        "_assemble_element_matrix_under_lease",
        original,
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        trusted(model, context="trusted-loop module ABA")

    class GenericElement:
        pass

    mixed, _mixed_q4 = _qualified_q4_model()
    mixed.add_element(2, GenericElement())
    mixed_lease = (
        matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
            mixed,
            context="mixed-model test",
        )
    )
    assert vars(mixed_lease).get("_qualified_trusted_require") is None


def test_recovery_lease_forwards_only_the_all_qualified_trusted_check() -> None:
    model, element = _qualified_q4_model()
    observed: dict[str, Any] = {}

    def inspect(guard: Any) -> None:
        trusted = vars(guard).get("_qualified_trusted_recovery_require")
        assert callable(trusted)
        observed["trusted"] = trusted
        for _ in range(1_000):
            trusted(stage="recovery trusted-loop repeat")

    recovery_batches_module._run_with_qualified_recovery_runtime_lease(
        model,
        context="trusted recovery test",
        operation=inspect,
    )
    assert callable(observed["trusted"])

    original = element.thickness

    def mutate_then_check(guard: Any) -> None:
        element.thickness = original * 2.0
        element.thickness = original
        trusted = vars(guard).get("_qualified_trusted_recovery_require")
        assert callable(trusted)
        trusted(stage="recovery element ABA")

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        recovery_batches_module._run_with_qualified_recovery_runtime_lease(
            model,
            context="trusted recovery ABA test",
            operation=mutate_then_check,
        )


def test_recovery_failure_cleanup_uses_no_module_builtin_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element = _qualified_q4_model()
    callbacks: list[str] = []

    class TrapObject:
        @staticmethod
        def __getattribute__(*args: object) -> object:
            del args
            callbacks.append("object.__getattribute__")
            raise AssertionError("module-global object must not run in cleanup")

    original_thickness = element.thickness

    def mutate_then_fail(guard: Any) -> None:
        vars(model.mesh)["_recovery_batch_plan"] = {"stale": True}
        monkeypatch.setattr(
            recovery_batches_module,
            "object",
            TrapObject,
            raising=False,
        )
        element.thickness = original_thickness * 2.0
        guard(stage="recovery cleanup hostile builtin")

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        recovery_batches_module._run_with_qualified_recovery_runtime_lease(
            model,
            context="recovery cleanup test",
            operation=mutate_then_fail,
        )

    assert callbacks == []
    assert "_recovery_batch_plan" not in vars(model.mesh)


@pytest.mark.parametrize(
    "shadow_name",
    ("object", "type", "dict", "id", "len", "int", "list"),
)
def test_trusted_element_recovery_rejects_builtin_shadows_without_callbacks(
    shadow_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    displacements = np.zeros(
        model.mesh.dof_manager.total_dofs,
        dtype=np.float64,
    )
    assembly_module.compute_stresses(model, displacements)
    callbacks: list[str] = []

    def trap(*_args: object, **_kwargs: object) -> object:
        callbacks.append(shadow_name)
        raise AssertionError(f"module-global {shadow_name} must not run")

    class TrapObject:
        __getattribute__ = staticmethod(trap)

    class TrapDict:
        get = staticmethod(trap)

    class TrapList:
        __getitem__ = staticmethod(trap)

    replacement: object = {
        "object": TrapObject,
        "dict": TrapDict,
        "list": TrapList,
    }.get(shadow_name, trap)
    monkeypatch.setattr(
        matrix_assembly_module,
        shadow_name,
        replacement,
        raising=False,
    )

    with pytest.raises(AssemblyError, match="authority"):
        assembly_module.compute_stresses(model, displacements)
    assert callbacks == []


def test_stress_recovery_uses_three_full_boundaries_not_two_per_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    for element_id in range(2, 9):
        model.add_element(
            element_id,
            QualifiedE4PLShellElement(
                element_id,
                (1, 2, 3, 4),
                "steel",
                thickness=0.05,
            ),
        )

    original_capture = (
        recovery_batches_module._CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE
    )
    full_checks: list[str] = []

    def counted_capture(*args: Any, **kwargs: Any) -> Any:
        lease = original_capture(*args, **kwargs)

        def counted_lease(expected_model: Any, *, context: str, **options: Any) -> None:
            full_checks.append(context)
            lease(expected_model, context=context, **options)

        vars(counted_lease).update(vars(lease))
        return counted_lease

    monkeypatch.setattr(
        recovery_batches_module,
        "_CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE",
        counted_capture,
    )
    stresses = assembly_module.compute_stresses(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
    )
    assert tuple(stresses) == tuple(range(1, 9))
    assert len(full_checks) == 3
    assert full_checks == [
        "stress computation stress-recovery preflight",
        "stress computation constraint-force preflight",
        "stress computation output",
    ]


def test_mixed_stress_recovery_bounds_qualified_checks_and_keeps_generic_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    for element_id in range(2, 9):
        model.add_element(
            element_id,
            QualifiedE4PLShellElement(
                element_id,
                (1, 2, 3, 4),
                "steel",
                thickness=0.05,
            ),
        )

    class GenericElement:
        material_name = "steel"
        recovery_errors_fail_closed = False

        @staticmethod
        def get_dof_mapping(_mesh: Any) -> Any:
            return np.empty(0, dtype=np.intp)

    model.add_element(9, GenericElement())
    original_capture = (
        recovery_batches_module._CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE
    )
    full_checks: list[str] = []

    def counted_capture(*args: Any, **kwargs: Any) -> Any:
        lease = original_capture(*args, **kwargs)

        def counted_lease(expected_model: Any, *, context: str, **options: Any) -> None:
            full_checks.append(context)
            lease(expected_model, context=context, **options)

        vars(counted_lease).update(vars(lease))
        return counted_lease

    monkeypatch.setattr(
        recovery_batches_module,
        "_CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE",
        counted_capture,
    )
    stresses = assembly_module.compute_stresses(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
    )
    assert tuple(stresses) == tuple(range(1, 9))
    assert len(full_checks) == 4
    assert full_checks[-2:] == [
        "stress computation stress material observation for element 9",
        "stress computation output",
    ]


def test_exact_builtin_beam_recovery_is_equivalent_and_constant_full_guard_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    model.add_node(5, 0.5, 0.0, 0.0)
    beam = BeamElement(
        2,
        [1, 2],
        "steel",
        {"area": 0.02, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
    )
    quadratic = QuadraticBeamElement(
        3,
        [1, 5, 2],
        "steel",
        {"area": 0.02, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
    )
    model.add_element(2, beam)
    model.add_element(3, quadratic)
    displacement = np.linspace(
        0.0,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
        dtype=np.float64,
    )
    material = model.get_material("steel")
    expected = {
        2: beam.compute_stresses(
            model.mesh,
            displacement[np.asarray(beam.get_dof_mapping(model.mesh))],
            material,
        ),
        3: quadratic.compute_stresses(
            model.mesh,
            displacement[np.asarray(quadratic.get_dof_mapping(model.mesh))],
            material,
        ),
    }

    original_capture = (
        recovery_batches_module._CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE
    )
    full_checks: list[str] = []

    def counted_capture(*args: Any, **kwargs: Any) -> Any:
        lease = original_capture(*args, **kwargs)

        def counted_lease(expected_model: Any, *, context: str, **options: Any) -> None:
            full_checks.append(context)
            lease(expected_model, context=context, **options)

        vars(counted_lease).update(vars(lease))
        return counted_lease

    monkeypatch.setattr(
        recovery_batches_module,
        "_CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE",
        counted_capture,
    )
    actual = assembly_module.compute_stresses(model, displacement)
    for element_id in (2, 3):
        assert actual[element_id].keys() == expected[element_id].keys()
        for name, value in expected[element_id].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[element_id][name], value)
            else:
                assert actual[element_id][name] == value
    assert len(full_checks) == 5
    assert full_checks[-3:] == [
        "stress computation exact built-in beam recovery batch preflight",
        "stress computation exact built-in beam recovery batch output",
        "stress computation output",
    ]


def test_exact_beam_snapshot_never_calls_dynamic_copy_or_ignored_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    beam = BeamElement(
        2,
        [1, 2],
        "steel",
        {"area": 0.02, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
    )
    model.add_element(2, beam)
    displacement = np.linspace(
        0.0,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
        dtype=np.float64,
    )
    material = model.get_material("steel")
    expected = beam.compute_stresses(
        model.mesh,
        displacement[np.asarray(beam.get_dof_mapping(model.mesh))],
        material,
    )
    copy_callbacks: list[str] = []
    cache_callbacks: list[str] = []
    original_deepcopy = elements_module.copy.deepcopy

    class HostileIgnoredCache:
        def __deepcopy__(self, _memo: Any) -> Any:
            cache_callbacks.append("ignored cache copied")
            raise AssertionError("ignored recovery cache must not be copied")

    beam._nl_cache = HostileIgnoredCache()

    def forged_deepcopy(value: Any, memo: Any = None) -> Any:
        copy_callbacks.append(type(value).__name__)
        if value is beam.node_ids:
            return [1, 4]
        return original_deepcopy(value, memo)

    monkeypatch.setattr(elements_module.copy, "deepcopy", forged_deepcopy)
    actual = assembly_module.compute_stresses(
        model,
        displacement,
        element_ids=[2],
    )
    _assert_recovery_equal(actual[2], expected)
    assert copy_callbacks == []
    assert cache_callbacks == []


def test_exact_beam_snapshot_never_calls_shadowed_object_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    beam = BeamElement(
        2,
        [1, 2],
        "steel",
        {"area": 0.02, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
    )
    model.add_element(2, beam)
    displacement = np.linspace(
        0.0,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
        dtype=np.float64,
    )
    material = model.get_material("steel")
    expected = beam.compute_stresses(
        model.mesh,
        displacement[np.asarray(beam.get_dof_mapping(model.mesh))],
        material,
    )
    callbacks: list[str] = []
    real_object = object
    real_getattribute = object.__getattribute__
    forged_namespace = dict(real_getattribute(beam, "__dict__"))
    forged_namespace["node_ids"] = [1, 4]

    class TrapObject:
        @staticmethod
        def __getattribute__(value: Any, name: str) -> Any:
            callbacks.append(f"getattribute:{name}")
            if value is beam and name == "__dict__":
                return forged_namespace
            return real_getattribute(value, name)

        @staticmethod
        def __new__(owner: type[Any]) -> Any:
            callbacks.append(f"new:{owner.__name__}")
            return real_object.__new__(owner)

    monkeypatch.setattr(
        recovery_batches_module,
        "object",
        TrapObject,
        raising=False,
    )
    actual = assembly_module.compute_stresses(
        model,
        displacement,
        element_ids=[2],
    )
    _assert_recovery_equal(actual[2], expected)
    assert callbacks == []


def test_exact_builtin_beam_recovery_rejects_persistent_input_mutation() -> None:
    model, _element = _qualified_q4_model()
    beam = BeamElement(
        2,
        [1, 2],
        "steel",
        {"area": 0.02, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
    )
    model.add_element(2, beam)

    def mutate(guard: Any) -> None:
        candidate = vars(guard).get(
            "_qualified_exact_builtin_beam_candidate"
        )
        require = vars(guard).get(
            "_qualified_trusted_recovery_beam_require"
        )
        assert callable(candidate) and candidate(beam)
        assert callable(require)
        beam._A *= 2.0
        require(
            beam,
            model.get_material("steel"),
            stage="mutated beam observation",
        )

    with pytest.raises(
        AssemblyError,
        match="authority",
    ):
        recovery_batches_module._run_with_qualified_recovery_runtime_lease(
            model,
            context="beam mutation test",
            operation=mutate,
        )


def test_exact_builtin_beam_recovery_uses_captured_node_dofs_across_aba() -> None:
    model, _element = _qualified_q4_model()
    beam = BeamElement(2, [1, 2], "steel")
    model.add_element(2, beam)
    later_node = model.mesh.get_node(2)
    assert later_node is not None
    material = model.get_material("steel")
    local_displacements = np.linspace(0.0, 1.0e-5, 12)
    expected = beam.compute_stresses(model.mesh, local_displacements, material)

    def mutate(guard: Any) -> None:
        require = vars(guard).get(
            "_qualified_trusted_recovery_beam_require"
        )
        dof_mapping = vars(guard).get(
            "_qualified_captured_beam_dof_mapping"
        )
        recover = vars(guard).get("_qualified_recover_captured_beam")
        assert callable(require) and callable(dof_mapping) and callable(recover)
        captured = dof_mapping(beam, material)
        original_dofs = tuple(later_node.dofs)
        later_node.dofs[:] = [999] * 6
        later_node.dofs[:] = original_dofs
        assert require(beam, material, stage="node DOF ABA") is True
        assert dof_mapping(beam, material) == captured
        actual = recover(
            beam,
            model.mesh,
            local_displacements,
            material,
            False,
        )
        _assert_recovery_equal(actual, expected)

    recovery_batches_module._run_with_qualified_recovery_runtime_lease(
        model,
        context="beam node DOF ABA test",
        operation=mutate,
    )


@pytest.mark.parametrize("field", ["node_ids", "cross_section", "orientation"])
def test_exact_builtin_beam_recovery_isolates_mutable_input_aba(field: str) -> None:
    model, _element = _qualified_q4_model()
    beam = BeamElement(
        2,
        [1, 2],
        "steel",
        {
            "area": 0.02,
            "Iy": 2.0e-6,
            "Iz": 3.0e-6,
            "J": 4.0e-6,
            "orientation": (0.0, 0.0, 1.0),
        },
    )
    model.add_element(2, beam)
    material = model.get_material("steel")
    local_displacements = np.linspace(0.0, 1.0e-5, 12)
    expected = beam.compute_stresses(model.mesh, local_displacements, material)

    def mutate(guard: Any) -> None:
        require = vars(guard).get(
            "_qualified_trusted_recovery_beam_require"
        )
        recover = vars(guard).get("_qualified_recover_captured_beam")
        assert callable(require) and callable(recover)
        if field == "node_ids":
            original = tuple(beam.node_ids)
            list.reverse(beam.node_ids)
            list.reverse(beam.node_ids)
            assert tuple(beam.node_ids) == original
        elif field == "cross_section":
            original = beam.cross_section["area"]
            dict.__setitem__(beam.cross_section, "area", original * 2.0)
            dict.__setitem__(beam.cross_section, "area", original)
        else:
            original = beam._orientation.copy()
            beam._orientation[:] = (1.0, 0.0, 0.0)
            beam._orientation[:] = original
        assert require(
            beam,
            material,
            stage=f"{field} ABA",
        ) is True
        actual = recover(
            beam,
            model.mesh,
            local_displacements,
            material,
            False,
        )
        _assert_recovery_equal(actual, expected)

    recovery_batches_module._run_with_qualified_recovery_runtime_lease(
        model,
        context=f"beam {field} ABA test",
        operation=mutate,
    )


def test_exact_beam_recovery_isolates_material_aba() -> None:
    model, _element = _qualified_q4_model()
    beam = BeamElement(2, [1, 2], "steel")
    model.add_element(2, beam)
    material = model.get_material("steel")
    local_displacements = np.linspace(0.0, 1.0e-5, 12)
    expected = beam.compute_stresses(model.mesh, local_displacements, material)

    def mutate(guard: Any) -> None:
        require = vars(guard).get(
            "_qualified_trusted_recovery_beam_require"
        )
        recover = vars(guard).get("_qualified_recover_captured_beam")
        assert callable(require) and callable(recover)
        original_modulus = material.elastic_modulus
        original_poisson = material.poisson_ratio
        object.__setattr__(material, "elastic_modulus", original_modulus * 3.0)
        object.__setattr__(material, "poisson_ratio", 0.2)
        object.__setattr__(material, "elastic_modulus", original_modulus)
        object.__setattr__(material, "poisson_ratio", original_poisson)
        assert require(beam, material, stage="material ABA") is True
        actual = recover(
            beam,
            model.mesh,
            local_displacements,
            material,
            False,
        )
        _assert_recovery_equal(actual, expected)

    recovery_batches_module._run_with_qualified_recovery_runtime_lease(
        model,
        context="material ABA isolation test",
        operation=mutate,
    )


def test_exact_beam_rejects_node_attribute_dispatch_patch_before_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    model.add_node(5, 2.0, 0.0, 0.0)
    beam = BeamElement(2, [1, 2], "steel")
    later_beam = BeamElement(3, [3, 4], "steel")
    model.add_element(2, beam)
    model.add_element(3, later_beam)
    node_type = type(model.mesh.get_node(1))
    later_node = model.mesh.get_node(3)
    forged_node = model.mesh.get_node(5)
    assert later_node is not None and forged_node is not None
    dispatches: list[str] = []
    original_getattribute = node_type.__getattribute__

    def forged_getattribute(node: Any, name: str) -> Any:
        if node is later_node and name == "dofs":
            dispatches.append("forged routing")
            return original_getattribute(forged_node, "dofs")
        return original_getattribute(node, name)

    monkeypatch.setattr(node_type, "__getattribute__", forged_getattribute)
    with pytest.raises(
        AssemblyError,
        match="beam recovery attribute authority changed",
    ):
        assembly_module.compute_stresses(
            model,
            np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
        )
    assert dispatches == []


def test_exact_beam_rejects_node_mapping_dispatch_patch_before_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_q4_model()
    model.add_node(5, 2.0, 0.0, 0.0)
    model.add_element(2, BeamElement(2, [1, 2], "steel"))
    model.add_element(3, BeamElement(3, [3, 4], "steel"))
    node_mapping = model.mesh.nodes
    mapping_type = type(node_mapping)
    forged_node = model.mesh.get_node(5)
    assert forged_node is not None
    dispatches: list[str] = []
    original_getattribute = mapping_type.__getattribute__

    def forged_getattribute(mapping: Any, name: str) -> Any:
        if mapping is node_mapping and name == "get":
            dispatches.append("forged node mapping")

            def forged_get(node_id: Any, default: Any = None) -> Any:
                if node_id == 3:
                    return forged_node
                return dict.get(mapping, node_id, default)

            return forged_get
        return original_getattribute(mapping, name)

    monkeypatch.setattr(
        mapping_type,
        "__getattribute__",
        forged_getattribute,
    )
    with pytest.raises(
        AssemblyError,
        match="beam recovery attribute authority changed",
    ):
        assembly_module.compute_stresses(
            model,
            np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
        )
    assert dispatches == []


@pytest.mark.parametrize("restore_dofs", [False, True], ids=["persistent", "aba"])
def test_exact_beam_object_array_is_rejected_without_invoking_node_mutation(
    restore_dofs: bool,
) -> None:
    model, _element = _qualified_q4_model()
    first_beam = BeamElement(2, [1, 2], "steel")
    later_beam = BeamElement(3, [3, 4], "steel")
    model.add_element(2, first_beam)
    model.add_element(3, later_beam)
    later_node = model.mesh.get_node(3)
    assert later_node is not None
    original_dofs = tuple(later_node.dofs)
    callbacks: list[str] = []

    class NodeDofMutator:
        def _mutate(self) -> float:
            callbacks.append("object arithmetic")
            later_node.dofs[:] = [999] * 6
            if restore_dofs:
                later_node.dofs[:] = original_dofs
            return 0.0

        def __float__(self) -> float:
            return self._mutate()

        def __mul__(self, _other: Any) -> float:
            return self._mutate()

        def __rmul__(self, _other: Any) -> float:
            return self._mutate()

        def __sub__(self, _other: Any) -> float:
            return self._mutate()

        def __rsub__(self, _other: Any) -> float:
            return self._mutate()

    first_beam._orientation = np.array(
        [NodeDofMutator(), NodeDofMutator(), NodeDofMutator()],
        dtype=object,
    )

    with pytest.raises(
        AssemblyError,
        match="beam recovery input can dispatch caller code",
    ):
        assembly_module.compute_stresses(
            model,
            np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
        )
    assert callbacks == []
    assert tuple(later_node.dofs) == original_dofs


def test_beam_subclasses_and_custom_materials_never_enter_exact_fast_path() -> None:
    model, _element = _qualified_q4_model()
    exact = BeamElement(2, [1, 2], "steel")

    class CustomBeam(BeamElement):
        pass

    custom_beam = CustomBeam(3, [1, 2], "steel")
    material_type = type(model.get_material("steel"))

    class CustomMaterial(material_type):
        pass

    custom_material = CustomMaterial("custom", 210.0e9, 0.3)
    model.register_material(custom_material)
    custom_material_beam = BeamElement(4, [1, 2], "custom")
    model.add_element(2, exact)
    model.add_element(3, custom_beam)
    model.add_element(4, custom_material_beam)

    def inspect(guard: Any) -> None:
        candidate = vars(guard).get(
            "_qualified_exact_builtin_beam_candidate"
        )
        assert callable(candidate)
        assert candidate(exact) is True
        assert candidate(custom_beam) is False
        assert candidate(custom_material_beam) is False

    recovery_batches_module._run_with_qualified_recovery_runtime_lease(
        model,
        context="beam eligibility test",
        operation=inspect,
    )


def test_recovery_rejects_displacement_provider_material_swap_before_mechanics() -> None:
    model, _element = _qualified_q4_model()
    material_type = type(model.get_material("steel"))
    mechanics_observations: list[str] = []

    class TrapFloat:
        def __float__(self) -> float:
            mechanics_observations.append("elastic_modulus")
            return 210.0e9

    replacement = object.__new__(material_type)
    object.__setattr__(replacement, "name", "steel")
    object.__setattr__(replacement, "elastic_modulus", TrapFloat())
    object.__setattr__(replacement, "poisson_ratio", 0.3)
    object.__setattr__(replacement, "density", 0.0)
    object.__setattr__(replacement, "yield_stress", 0.0)
    object.__setattr__(replacement, "hardening_curve", None)

    class MutatingDisplacement:
        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            model.materials["steel"] = replacement
            made = np.zeros(
                model.mesh.dof_manager.total_dofs,
                dtype=np.float64,
            )
            if dtype is not None:
                made = made.astype(dtype, copy=False)
            if copy is True:
                made = made.copy()
            return made

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        assembly_module.compute_stresses(model, MutatingDisplacement())
    assert mechanics_observations == []


def test_alternating_q4_beam_recovery_keeps_one_bracket_and_rejects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _first_q4 = _qualified_q4_model()
    first_beam = BeamElement(2, [1, 2], "steel")
    second_q4 = QualifiedE4PLShellElement(
        3,
        (1, 2, 3, 4),
        "steel",
        thickness=0.05,
    )
    second_beam = BeamElement(4, [3, 4], "steel")
    model.add_element(2, first_beam)
    model.add_element(3, second_q4)
    model.add_element(4, second_beam)

    original_capture = (
        recovery_batches_module._CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE
    )
    full_checks: list[str] = []

    def counted_capture(*args: Any, **kwargs: Any) -> Any:
        lease = original_capture(*args, **kwargs)

        def counted_lease(expected_model: Any, *, context: str, **options: Any) -> None:
            full_checks.append(context)
            lease(expected_model, context=context, **options)

        vars(counted_lease).update(vars(lease))
        return counted_lease

    monkeypatch.setattr(
        recovery_batches_module,
        "_CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE",
        counted_capture,
    )
    recovered = assembly_module.compute_stresses(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
    )
    assert tuple(recovered) == (1, 2, 3, 4)
    assert len(full_checks) == 5
    assert full_checks.count(
        "stress computation exact built-in beam recovery batch preflight"
    ) == 1
    assert full_checks.count(
        "stress computation exact built-in beam recovery batch output"
    ) == 1

    def mutate(guard: Any) -> None:
        require = vars(guard).get(
            "_qualified_trusted_recovery_beam_require"
        )
        assert callable(require)
        second_beam._J *= 2.0
        require(
            second_beam,
            model.get_material("steel"),
            stage="alternating beam mutation",
        )

    with pytest.raises(
        AssemblyError,
        match="authority",
    ):
        recovery_batches_module._run_with_qualified_recovery_runtime_lease(
            model,
            context="alternating beam mutation test",
            operation=mutate,
        )


def test_large_alternating_q4_beam_recovery_has_constant_full_guard_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _first_q4 = _qualified_q4_model()
    pair_count = 128
    for pair_index in range(pair_count):
        beam_id = 2 + 2 * pair_index
        q4_id = beam_id + 1
        model.add_element(
            beam_id,
            BeamElement(beam_id, [1, 2], "steel"),
        )
        model.add_element(
            q4_id,
            QualifiedE4PLShellElement(
                q4_id,
                (1, 2, 3, 4),
                "steel",
                thickness=0.05,
            ),
        )

    original_capture = (
        recovery_batches_module._CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE
    )
    full_checks: list[str] = []

    def counted_capture(*args: Any, **kwargs: Any) -> Any:
        lease = original_capture(*args, **kwargs)

        def counted_lease(expected_model: Any, *, context: str, **options: Any) -> None:
            full_checks.append(context)
            lease(expected_model, context=context, **options)

        vars(counted_lease).update(vars(lease))
        return counted_lease

    monkeypatch.setattr(
        recovery_batches_module,
        "_CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE",
        counted_capture,
    )
    recovered = assembly_module.compute_stresses(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
    )
    assert len(recovered) == 1 + 2 * pair_count
    assert len(full_checks) == 5
    assert full_checks.count(
        "stress computation exact built-in beam recovery batch preflight"
    ) == 1
    assert full_checks.count(
        "stress computation exact built-in beam recovery batch output"
    ) == 1
