"""Load-vector assembly keeps every guard that foreign code could require.

``_assemble_load_vector_under_lease`` brackets spans in which only this
package's kernels run with the lease's constant-time trusted guard, and keeps
the complete lifecycle-plus-lease guard wherever user-supplied or generic code
may have run.  These tests pin both halves: the amortisation, and that no
mutation by foreign code goes unnoticed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import anyfesolver.current_state_tangent as current_state_tangent
from anyfesolver import boundary
from anyfesolver.boundary import LoadCase
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import BeamElement, create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import (
    AssemblyError,
    _assemble_load_vector_under_lease,
    _run_with_qualified_assembly_runtime_lease,
    assemble_load_matrix,
    assemble_load_vector,
)

_GUARD_ERRORS = (ElementCapabilityError, AssemblyError)


def _model() -> tuple[FEModel, dict[str, int]]:
    """A 4x3 node plate: one Q4, two S3 shells and one generic beam."""

    model = FEModel("load-vector-guard-amortization")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for j in range(3):
        for i in range(4):
            model.add_node(j * 4 + i + 1, float(i), float(j), 0.0)
    q4 = create_shell_element(1, [1, 2, 6, 5], "steel", thickness=0.01, formulation="e4-pl")
    s3_a = QualifiedE4PLS3ShellElement(
        2, [2, 3, 7], "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)
    )
    s3_b = QualifiedE4PLS3ShellElement(
        3, [2, 7, 6], "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)
    )
    beam = BeamElement(
        4, [3, 4], "steel", cross_section={"area": 0.004, "Iy": 1e-5, "Iz": 1e-5, "J": 1e-5}
    )
    for element in (q4, s3_a, s3_b, beam):
        model.add_element(element.element_id, element)
    return model, {"q4": 1, "s3": 2, "s3b": 3, "beam": 4}


def _cases(ids: dict[str, int]) -> list[LoadCase]:
    nodal = LoadCase("nodal")
    nodal.add_nodal_load(5, forces=[1.0e3, -2.0e3, 3.0e3], moments=[1.0, 2.0, 3.0])
    pressure = LoadCase("pressure")
    pressure.add_pressure_load(ids["q4"], 2.5e4)
    pressure.add_pressure_load(ids["s3"], 1.5e4)
    pressure.add_pressure_load(ids["s3b"], 9.0e3)
    follower = LoadCase("follower")
    follower.follower_pressure = True
    follower.add_pressure_load(ids["q4"], 1.0e4)
    element = LoadCase("element-load")
    element.element_loads[ids["q4"]] = np.arange(24, dtype=float)
    mixed = LoadCase("mixed")
    mixed.add_nodal_load(7, forces=[0.0, 0.0, -5.0e3])
    mixed.add_pressure_load(ids["q4"], 1.0e4)
    mixed.element_loads[ids["s3"]] = np.arange(18, dtype=float) * 0.5
    return [nodal, pressure, follower, element, mixed]


def _count_lifecycle_guard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    original = current_state_tangent.require_exact_qualified_component_lifecycle_api

    def counting(model: Any, *, context: str) -> Any:
        calls.append(context)
        return original(model, context=context)

    monkeypatch.setattr(
        current_state_tangent,
        "require_exact_qualified_component_lifecycle_api",
        counting,
    )
    return calls


def test_untouched_load_cases_are_bit_identical_to_the_public_kernel() -> None:
    model, ids = _model()
    displacements = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
    for case in _cases(ids):
        for u in (None, displacements):
            owned = boundary._run_foreign_code_free_load_vector(
                case,
                model.mesh,
                model.mesh.dof_manager,
                model.get_material,
                u,
                None,
                stage_guard=lambda *, stage: None,
            )
            assert owned is not None, case.name
            public = case.get_load_vector(
                model.mesh,
                model.mesh.dof_manager,
                model.get_material,
                displacements=u,
            )
            assert owned.tobytes() == np.asarray(public, dtype=float).tobytes()
            assembled, info = assemble_load_vector(model, case, u)
            assert assembled.tobytes() == owned.tobytes()
            assert info["load_case"] == case.name


def test_complete_lifecycle_guard_runs_once_per_lease_not_per_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, ids = _model()
    cases = _cases(ids)
    calls = _count_lifecycle_guard(monkeypatch)
    _, info = assemble_load_matrix(model, cases)
    assert info["load_cases"] == [case.name for case in cases]
    assert len(calls) == 1, calls
    calls.clear()
    assemble_load_vector(model, cases[1])
    assert len(calls) == 1, calls


def test_foreign_load_case_subclass_keeps_every_complete_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Observed(LoadCase):
        pass

    model, ids = _model()
    case = Observed("observed")
    case.add_pressure_load(ids["q4"], 1.0e4)
    calls = _count_lifecycle_guard(monkeypatch)
    assemble_load_vector(model, case)
    assert len(calls) >= 6, calls


def test_generic_element_pressure_keeps_the_complete_closing_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, ids = _model()
    case = LoadCase("generic")
    case.pressure_loads[ids["beam"]] = 1.0e3
    assert (
        boundary._run_foreign_code_free_load_vector(
            case,
            model.mesh,
            model.mesh.dof_manager,
            model.get_material,
            None,
            None,
            stage_guard=lambda *, stage: None,
        )
        is None
    )
    calls = _count_lifecycle_guard(monkeypatch)
    assemble_load_vector(model, case)
    assert len(calls) >= 3, calls


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda case, ids: case.nodal_loads.__setitem__(5, [1.0, 2.0, 3.0]),
            id="list-nodal-load",
        ),
        pytest.param(
            lambda case, ids: case.pressure_loads.__setitem__(
                ids["q4"], np.float64(1.0)
            ),
            id="numpy-scalar-pressure",
        ),
        pytest.param(
            lambda case, ids: case.pressure_loads.__setitem__(
                np.int64(ids["q4"]), 1.0
            ),
            id="numpy-key",
        ),
    ],
)
def test_runner_refuses_inputs_that_could_execute_foreign_code(mutate: Any) -> None:
    model, ids = _model()
    case = LoadCase("refused")
    case.add_nodal_load(5, forces=[1.0, 2.0, 3.0])
    mutate(case, ids)
    assert (
        boundary._run_foreign_code_free_load_vector(
            case,
            model.mesh,
            model.mesh.dof_manager,
            model.get_material,
            None,
            None,
            stage_guard=lambda *, stage: None,
        )
        is None
    )


def test_runner_refuses_activity_provider_and_non_exact_displacements() -> None:
    model, ids = _model()
    case = _cases(ids)[1]
    common = (case, model.mesh, model.mesh.dof_manager, model.get_material)
    guard = lambda *, stage: None  # noqa: E731
    assert (
        boundary._run_foreign_code_free_load_vector(
            *common, None, object(), stage_guard=guard
        )
        is None
    )
    assert (
        boundary._run_foreign_code_free_load_vector(
            *common, [0.0] * model.mesh.dof_manager.total_dofs, None, stage_guard=guard
        )
        is None
    )


def test_replaced_load_case_class_member_falls_back_to_the_complete_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, ids = _model()
    case = _cases(ids)[1]
    original = LoadCase._consistent_pressure_load
    seen: list[int] = []

    def hostile(self: Any, element: Any, mesh: Any, pressure: Any, coords: Any = None):
        seen.append(1)
        object.__getattribute__(element, "__dict__")[
            "compute_stiffness_matrix"
        ] = lambda *args, **kwargs: None
        return original(self, element, mesh, pressure, coords)

    monkeypatch.setattr(LoadCase, "_consistent_pressure_load", hostile)
    assert boundary._exact_load_case_instance_namespace(case) is None
    with pytest.raises(_GUARD_ERRORS):
        assemble_load_vector(model, case)
    assert seen


def test_model_mutation_between_two_assemblies_in_one_lease_raises() -> None:
    model, ids = _model()
    case = _cases(ids)[1]
    q4 = model.mesh.elements[ids["q4"]]

    def operation(lease: Any) -> Any:
        first, _ = _assemble_load_vector_under_lease(
            model, case, qualified_runtime_guard=lease
        )
        # An unsupported-surface write that no mutation epoch observes: only a
        # complete guard can see it, so the second assembly must run one.
        object.__getattribute__(q4, "__dict__")[
            "compute_stiffness_matrix"
        ] = lambda *args, **kwargs: None
        second, _ = _assemble_load_vector_under_lease(
            model, case, qualified_runtime_guard=lease
        )
        return first, second

    with pytest.raises(_GUARD_ERRORS):
        _run_with_qualified_assembly_runtime_lease(
            model, context="two assemblies", operation=operation
        )


def test_supported_mutation_between_two_assemblies_in_one_lease_raises() -> None:
    model, ids = _model()
    case = _cases(ids)[0]
    q4 = model.mesh.elements[ids["q4"]]

    def operation(lease: Any) -> Any:
        _assemble_load_vector_under_lease(model, case, qualified_runtime_guard=lease)
        q4.material_name = "steel"
        return _assemble_load_vector_under_lease(
            model, case, qualified_runtime_guard=lease, preflight_verified=True
        )

    with pytest.raises(_GUARD_ERRORS):
        _run_with_qualified_assembly_runtime_lease(
            model, context="supported mutation", operation=operation
        )


def test_foreign_mutation_inside_a_batch_is_caught_at_that_case() -> None:
    model, ids = _model()
    cases = _cases(ids)
    q4 = model.mesh.elements[ids["q4"]]

    class Hostile(LoadCase):
        def get_load_vector(self, *args: Any, **kwargs: Any) -> np.ndarray:
            object.__getattribute__(q4, "__dict__")[
                "compute_stiffness_matrix"
            ] = lambda *a, **k: None
            return super().get_load_vector(*args, **kwargs)

    hostile = Hostile("hostile")
    hostile.add_nodal_load(5, forces=[1.0, 0.0, 0.0])
    with pytest.raises(_GUARD_ERRORS):
        assemble_load_matrix(model, [cases[0], hostile, cases[1]])


def test_foreign_name_observation_is_guarded_in_a_batch() -> None:
    model, ids = _model()
    q4 = model.mesh.elements[ids["q4"]]

    class NamedByCallback(LoadCase):
        @property
        def name(self) -> str:  # type: ignore[override]
            object.__getattribute__(q4, "__dict__")[
                "compute_stiffness_matrix"
            ] = lambda *a, **k: None
            return "callback"

        @name.setter
        def name(self, value: str) -> None:
            pass

    case = NamedByCallback("ignored")
    case.add_nodal_load(5, forces=[1.0, 0.0, 0.0])
    with pytest.raises(_GUARD_ERRORS):
        assemble_load_matrix(model, [case])


def test_gravity_and_generic_routes_are_bit_identical_to_the_public_kernel() -> None:
    model, ids = _model()
    gravity = LoadCase("gravity")
    gravity.set_gravity(0.0, 0.0, -9.81)
    gravity.add_node_mass(5, 12.0)
    gravity.add_pressure_load(ids["q4"], 1.0e4)
    generic = LoadCase("generic")
    generic.pressure_loads[ids["beam"]] = 1.0e3
    for case in (gravity, generic):
        public = case.get_load_vector(
            model.mesh,
            model.mesh.dof_manager,
            model.get_material,
        )
        assembled, _ = assemble_load_vector(model, case)
        batch, _ = assemble_load_matrix(model, [case, case])
        assert assembled.tobytes() == np.asarray(public, dtype=float).tobytes()
        assert batch[:, 0].tobytes() == assembled.tobytes()
        assert batch[:, 1].tobytes() == assembled.tobytes()


@pytest.mark.parametrize("member", ["get_load_vector", "_get_load_vector_core", "_consistent_pressure_load"])
def test_instance_method_shadow_refuses_owned_span(member: str) -> None:
    model, ids = _model()
    case = _cases(ids)[1]
    setattr(case, member, lambda *args, **kwargs: None)
    assert boundary._exact_load_case_instance_namespace(case) is None


@pytest.mark.parametrize("source", ["nodal_loads", "element_loads", "displacements"])
def test_object_arrays_refuse_owned_span_without_conversion(source: str) -> None:
    model, ids = _model()
    case = _cases(ids)[1]

    class ForeignFloat:
        def __float__(self):
            pytest.fail("admission must not convert object-array members")

    values = np.full(model.mesh.dof_manager.total_dofs, ForeignFloat(), dtype=object)
    if source != "displacements":
        getattr(case, source)[5 if source == "nodal_loads" else ids["q4"]] = values
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material,
        values if source == "displacements" else None, None,
        stage_guard=lambda *, stage: None,
    ) is None


@pytest.mark.parametrize("kind", ["subclass", "dofs", "coordinate", "coords-provider"])
def test_foreign_node_state_refuses_owned_span(kind: str) -> None:
    from anyfesolver.fe_core import Node

    model, ids = _model()
    case = _cases(ids)[1]
    node = model.mesh.nodes[1]
    namespace = object.__getattribute__(node, "__dict__")
    if kind == "subclass":
        class ForeignNode(Node):
            pass
        foreign = ForeignNode(node.id, node.x, node.y, node.z, node.dofs)
        dict.__setitem__(model.mesh.nodes, 1, foreign)
    elif kind == "dofs":
        class ForeignDofs(list):
            pass
        namespace["dofs"] = ForeignDofs(node.dofs)
    elif kind == "coordinate":
        namespace["x"] = object()
    else:
        namespace["coords"] = lambda: np.zeros(3)
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
        stage_guard=lambda *, stage: None,
    ) is None


def test_object_displacement_callback_gets_complete_guard_before_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    model, ids = _model()
    case = _cases(ids)[2]
    seen = []
    original = LoadCase._get_load_vector_core

    def observed(self, *args, **kwargs):
        seen.append("kernel")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoadCase, "_get_load_vector_core", observed)

    class ForeignFloat:
        def __float__(self):
            object.__getattribute__(model.mesh.elements[ids["q4"]], "__dict__")[
                "compute_stiffness_matrix"
            ] = lambda *args, **kwargs: None
            return 0.0

    values = np.full(model.mesh.dof_manager.total_dofs, ForeignFloat(), dtype=object)
    with pytest.raises(_GUARD_ERRORS):
        assemble_load_vector(model, case, values)
    assert seen == []


def test_activity_descriptor_cannot_return_before_complete_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from anyfesolver.fe_core import FEMesh

    model, ids = _model()
    observed = []

    def activity(self):
        observed.append("callback")
        object.__getattribute__(model.mesh.elements[ids["q4"]], "__dict__")[
            "compute_stiffness_matrix"
        ] = lambda *args, **kwargs: None
        return None

    monkeypatch.setattr(FEMesh, "element_activity", property(activity))

    def operation(lease):
        _assemble_load_vector_under_lease(model, None, qualified_runtime_guard=lease)
        observed.append("returned")

    with pytest.raises(_GUARD_ERRORS):
        _run_with_qualified_assembly_runtime_lease(model, context="activity descriptor", operation=operation)
    assert "returned" not in observed


def test_foreign_batch_iterator_is_guarded_before_next_case_kernel() -> None:
    from anyfesolver.matrix_assembly import _assemble_load_matrix_under_lease

    model, ids = _model()
    seen = []

    class Observed(LoadCase):
        def get_load_vector(self, *args, **kwargs):
            seen.append("kernel")
            return super().get_load_vector(*args, **kwargs)

    class Cases:
        def __iter__(self):
            yield _cases(ids)[0]
            object.__getattribute__(model.mesh.elements[ids["q4"]], "__dict__")[
                "compute_stiffness_matrix"
            ] = lambda *args, **kwargs: None
            yield Observed("second")

    with pytest.raises(_GUARD_ERRORS):
        _run_with_qualified_assembly_runtime_lease(
            model, context="foreign iterator", operation=lambda lease:
            _assemble_load_matrix_under_lease(model, Cases(), qualified_runtime_guard=lease),
        )
    assert seen == []


@pytest.mark.parametrize("kind", ["mesh-subclass", "mapping-subclass", "native-fibre", "mesh-hook", "node-descriptor"])
def test_runner_refuses_foreign_routing_and_native_fibre(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from anyfesolver.fe_core import FEMesh, Node

    model, ids = _model()
    case = _cases(ids)[0]
    if kind == "mesh-subclass":
        class ForeignMesh(FEMesh):
            pass
        model.mesh.__class__ = ForeignMesh
    elif kind == "mapping-subclass":
        class ForeignMapping(dict):
            pass
        object.__getattribute__(model.mesh, "__dict__")["nodes"] = ForeignMapping(model.mesh.nodes)
    elif kind == "native-fibre":
        native = type("Native", (), {"__module__": "anyfesolver._ge_beam3_native_fibre_static_element"})()
        dict.__setitem__(model.mesh.elements, 9, native)
    elif kind == "mesh-hook":
        monkeypatch.setattr(FEMesh, "__getattr__", lambda self, name: None, raising=False)
    else:
        monkeypatch.setattr(Node, "dofs", property(lambda self: pytest.fail("descriptor invoked")), raising=False)
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
        stage_guard=lambda *, stage: None,
    ) is None


@pytest.mark.parametrize("kind", ["foreign-load-case", "object-displacements"])
def test_foreign_observation_cannot_revive_trusted_routing(kind: str) -> None:
    model, ids = _model()

    class Foreign(LoadCase):
        pass

    class ForeignFloat:
        def __float__(self):
            return 0.0

    case = Foreign("foreign") if kind == "foreign-load-case" else _cases(ids)[1]
    displacements = (
        np.full(model.mesh.dof_manager.total_dofs, ForeignFloat(), dtype=object)
        if kind == "object-displacements" else None
    )
    calls = []

    def operation(lease):
        namespace = object.__getattribute__(lease, "__dict__")
        for name in ("_qualified_trusted_state_require", "_qualified_trusted_input_require"):
            original = namespace[name]

            def observed(*args, _original=original, **kwargs):
                calls.append(kwargs["context"])
                return _original(*args, **kwargs)

            namespace[name] = observed
        return _assemble_load_vector_under_lease(
            model, case, displacements, qualified_runtime_guard=lease,
        )

    _run_with_qualified_assembly_runtime_lease(model, context="foreign proof lifetime", operation=operation)
    assert set(calls).issubset({
        "load-vector LoadCase name observation",
        "load-vector LoadCase pressure-policy observation",
    }), calls


@pytest.mark.parametrize("source", ["nodal_loads", "displacements"])
def test_complex_arrays_refuse_owned_span(source: str) -> None:
    model, ids = _model()
    case = _cases(ids)[1]
    values = np.zeros(model.mesh.dof_manager.total_dofs, dtype=complex)
    if source == "nodal_loads":
        case.nodal_loads[5] = values
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material,
        values if source == "displacements" else None, None,
        stage_guard=lambda *, stage: None,
    ) is None


def test_complex_conversion_warning_is_guarded_before_pressure_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    import warnings

    model, ids = _model()
    case = _cases(ids)[1]
    state = object.__getattribute__(model.mesh.elements[ids["q4"]], "__dict__")
    seen = []

    def foreign(mesh, pressure):
        seen.append("foreign kernel")
        state.pop("compute_native_dead_pressure_load", None)
        return np.zeros(24)

    def warning(*args, **kwargs):
        state["compute_native_dead_pressure_load"] = foreign

    warnings.simplefilter("always")
    monkeypatch.setattr(warnings, "showwarning", warning)
    with pytest.raises(_GUARD_ERRORS):
        assemble_load_vector(model, case, np.zeros(model.mesh.dof_manager.total_dofs, dtype=complex))
    assert seen == []


@pytest.mark.parametrize("mode", ["call", "log"])
def test_numpy_error_callbacks_refuse_owned_span(mode: str) -> None:
    model, ids = _model()
    case = LoadCase("overflow callback")
    case.add_pressure_load(ids["q4"], 1.0e308)
    for node in model.mesh.nodes.values():
        node.x *= 1000.0
        node.y *= 1000.0
    seen = []

    class Logger:
        def write(self, message):
            seen.append(message)

    previous = np.geterrcall()
    try:
        np.seterrcall((lambda error, flags: seen.append(error)) if mode == "call" else Logger())
        with np.errstate(over=mode, invalid=mode):
            result = boundary._run_foreign_code_free_load_vector(
                case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
                stage_guard=lambda *, stage: None,
            )
        assert result is None
        assert seen == []
    finally:
        np.seterrcall(previous)


@pytest.mark.parametrize("owner", ["load_case", "mesh", "dof_manager"])
def test_in_place_routing_function_code_change_refuses_owned_span(monkeypatch: pytest.MonkeyPatch, owner: str) -> None:
    case = LoadCase("patched code")
    if owner == "load_case":
        member = LoadCase._get_load_vector_core
    elif owner == "mesh":
        member = boundary._FEMesh.get_element
    else:
        member = boundary._DOFManager.total_dofs.fget
    replacement = lambda *args, **kwargs: None
    monkeypatch.setattr(member, "__code__", replacement.__code__)
    if owner == "load_case":
        assert boundary._exact_load_case_instance_namespace(case) is None
    else:
        assert not boundary._load_routing_classes_are_untouched()


def test_numpy_warning_handler_refuses_owned_span(monkeypatch: pytest.MonkeyPatch) -> None:
    import warnings

    model, ids = _model()
    case = LoadCase("overflow warning callback")
    case.add_pressure_load(ids["q4"], 1.0e308)
    for node in model.mesh.nodes.values():
        node.x *= 1000.0
        node.y *= 1000.0
    seen = []
    monkeypatch.setattr(warnings, "showwarning", lambda *args, **kwargs: seen.append("warning"))
    with warnings.catch_warnings(), np.errstate(over="warn", invalid="warn"):
        warnings.simplefilter("always")
        result = boundary._run_foreign_code_free_load_vector(
            case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
            stage_guard=lambda *, stage: None,
        )
    assert result is None
    assert seen == []


@pytest.mark.parametrize("consistent_beam", [False, True])
def test_acceleration_fast_batch_preserves_original_kernel(monkeypatch, consistent_beam):
    model, ids = _model()
    model.mesh.elements[ids["beam"]].cross_section["consistent_mass"] = consistent_beam
    model.add_point_mass(3, 8.0)
    cases = []
    expected = []
    for index, acceleration in enumerate(((0.0, 0.0, -9.81), (1.7, -0.8, 3.3))):
        case = LoadCase(f"acceleration-{index}")
        case.set_acceleration(*acceleration)
        case.add_node_mass(3, 4.0)
        case.add_pressure_load(ids["s3"], 1234.5)
        case.add_nodal_load(5, forces=[11.0, 22.0, 33.0])
        cases.append(case)
        expected.append(case.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material))
        owned = boundary._run_foreign_code_free_load_vector(
            case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
            stage_guard=lambda *, stage: None,
        )
        assert owned is not None
        assert owned.tobytes() == expected[-1].tobytes()
    calls = _count_lifecycle_guard(monkeypatch)
    assembled, _ = assemble_load_matrix(model, cases)
    assert assembled.tobytes() == np.column_stack(expected).tobytes()
    assert len(calls) == 1, calls


@pytest.mark.parametrize("target", ["material", "beam_section", "mass_function", "mass_helper", "point_mass"])
def test_acceleration_foreign_inputs_refuse_owned_span(monkeypatch, target):
    model, ids = _model()
    case = LoadCase("foreign acceleration")
    case.set_acceleration(1.0, 2.0, 3.0)
    if target == "material":
        model.materials["steel"].density = object()
    elif target == "beam_section":
        model.mesh.elements[ids["beam"]].generalized_section = object()
    elif target == "mass_function":
        monkeypatch.setattr(BeamElement.compute_mass_matrix, "__code__", (lambda *args: None).__code__)
    elif target == "mass_helper":
        monkeypatch.setattr(boundary._element_module._beam_rotation_matrix, "__code__", (lambda *args: None).__code__)
    else:
        model.mesh.point_masses[3] = object()
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
        stage_guard=lambda *, stage: None,
    ) is None


@pytest.mark.parametrize("owner", ["load_case", "mesh", "dof_manager"])
def test_in_place_function_defaults_change_refuses_owned_span(monkeypatch, owner):
    member = (LoadCase._get_load_vector_core if owner == "load_case" else
              boundary._FEMesh.get_element if owner == "mesh" else boundary._DOFManager.total_dofs.fget)
    monkeypatch.setattr(member, "__defaults__", (object(),))
    if owner == "load_case":
        assert boundary._exact_load_case_instance_namespace(LoadCase("defaults")) is None
    else:
        assert not boundary._load_routing_classes_are_untouched()



def test_replaced_numpy_error_getter_is_not_executed(monkeypatch):
    seen = []
    monkeypatch.setattr(np, "geterr", lambda: seen.append("foreign") or {})
    assert not boundary._numpy_error_routes_are_builtin()
    assert seen == []


def test_wide_float_conversion_is_not_owned():
    if np.dtype(np.longdouble).itemsize > 8:
        assert not boundary._plain_numeric_array(np.zeros(3, dtype=np.longdouble))


def test_exact_text_wrapper_with_foreign_buffer_is_not_owned():
    import io
    class Buffer(io.BytesIO):
        pass
    stream = io.TextIOWrapper(Buffer())
    try:
        assert not boundary._builtin_warning_stream(stream)
    finally:
        stream.close()


def test_acceleration_reaches_public_solve_linear_many(monkeypatch):
    from anyfesolver.assembly import solve_linear_many, solve_linear
    from anyfesolver.boundary import FixedSupport
    model, ids = _model()
    model.add_boundary_condition(FixedSupport("root", [node_id for node_id in model.mesh.nodes if node_id != 4]))
    cases = []
    for acceleration in ((0.0, 0.0, -9.81), (1.0, -2.0, 3.0)):
        case = LoadCase(str(acceleration))
        case.set_acceleration(*acceleration)
        cases.append(case)
    expected = np.column_stack([solve_linear(model, case)[0] for case in cases])
    owned = []
    original = boundary._run_foreign_code_free_load_vector
    def observed(*args, **kwargs):
        vector = original(*args, **kwargs)
        owned.append(vector is not None)
        return vector
    monkeypatch.setattr(boundary, "_run_foreign_code_free_load_vector", observed)
    actual, info = solve_linear_many(model, cases)
    assert owned == [True, True]
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-18)



def test_acceleration_rejects_retained_mass_original_code_change(monkeypatch):
    model, ids = _model()
    case = LoadCase("retained mass original")
    case.set_gravity()
    wrapper = QualifiedE4PLS3ShellElement.compute_mass_matrix
    original = next(cell.cell_contents for cell in wrapper.__closure__
                    if callable(cell.cell_contents) and getattr(cell.cell_contents, "__name__", "") == "compute_mass_matrix")
    monkeypatch.setattr(original, "__code__", (lambda *args: None).__code__)
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
        stage_guard=lambda *, stage: None,
    ) is None


def test_acceleration_rejects_material_accessor_code_change(monkeypatch):
    model, ids = _model()
    case = LoadCase("material accessor")
    case.set_gravity()
    monkeypatch.setattr(boundary._IsotropicMaterial.shear_modulus.fget, "__code__", (lambda *args: 0.0).__code__)
    assert boundary._run_foreign_code_free_load_vector(
        case, model.mesh, model.mesh.dof_manager, model.get_material, None, None,
        stage_guard=lambda *, stage: None,
    ) is None


def test_in_place_kwdefaults_change_refuses_owned_span(monkeypatch):
    function = LoadCase._get_load_vector_core
    # This kernel's required keyword normally has no default; a replacement
    # dictionary is also an authority change.
    monkeypatch.setattr(function, "__kwdefaults__", {"qualified_runtime_guard": object()})
    assert boundary._exact_load_case_instance_namespace(LoadCase("kwdefaults")) is None



def test_builtin_warning_capture_admission_keeps_foreign_handlers_out(monkeypatch):
    import warnings
    with warnings.catch_warnings(record=True):
        assert boundary._numpy_error_routes_are_builtin()
        monkeypatch.setattr(warnings, "_showwarnmsg_impl", lambda message: None)
        assert not boundary._numpy_error_routes_are_builtin()



def test_mapping_proxy_over_foreign_mapping_is_refused_without_observation():
    from collections import UserDict
    from types import MappingProxyType
    seen = []
    class ForeignMapping(UserDict):
        def items(self):
            seen.append("items")
            return super().items()
        def __iter__(self):
            seen.append("iteration")
            return super().__iter__()
    assert not boundary._plain_mass_input(MappingProxyType(ForeignMapping({"x": 1.0})))
    assert seen == []



def test_warning_handler_installed_before_import_is_not_self_authorized():
    import os
    import subprocess
    import sys
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """import warnings
seen = []
warnings.showwarning = lambda *args, **kwargs: seen.append('foreign')
import anyfesolver.boundary as boundary
assert not boundary._numpy_error_routes_are_builtin()
assert seen == []
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr



def test_wrapped_runtime_manager_cell_is_not_replaceable(monkeypatch):
    wrapper = QualifiedE4PLShellElement.compute_mass_matrix
    cell = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))["runtime_manager"]
    monkeypatch.setattr(cell, "cell_contents", object())
    assert not boundary._mass_classes_are_untouched()


def test_registered_codec_error_callback_refuses_builtin_writer(tmp_path):
    import codecs
    import io
    seen = []
    original = codecs.lookup_error("strict")
    stream = io.TextIOWrapper(io.FileIO(tmp_path / "stream", "w"), encoding="utf8", errors="strict")
    try:
        codecs.register_error("strict", lambda error: seen.append("handler"))
        assert not boundary._builtin_warning_stream(stream)
        assert seen == []
    finally:
        codecs.register_error("strict", original)
        stream.close()


def test_nonbuiltin_fast_encoding_is_refused(tmp_path):
    import io
    stream = io.TextIOWrapper(io.FileIO(tmp_path / "stream", "w"), encoding="utf16")
    try:
        assert not boundary._builtin_warning_stream(stream)
    finally:
        stream.close()


@pytest.mark.parametrize("provider", ["warning_message", "linecache"])
def test_preimport_warning_providers_are_not_self_authorized(provider):
    import os
    import subprocess
    import sys
    patch = ("warnings.WarningMessage = type('WarningMessage', (), {'__init__': lambda *args, **kwargs: seen.append('foreign')})"
             if provider == "warning_message" else
             "import linecache; linecache.getline = lambda *args, **kwargs: seen.append('foreign') or ''")
    script = "import warnings\nseen = []\n" + patch + "\nimport anyfesolver.boundary as boundary\nassert not boundary._numpy_error_routes_are_builtin()\nassert seen == []\n"
    completed = subprocess.run([sys.executable, "-B", "-c", script],
                               env=os.environ.copy(), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_mass_material_helper_function_metadata_is_authoritative(monkeypatch):
    namespace, name, function, authority = next(
        item for item in boundary._EXACT_MASS_FUNCTION_DEPENDENCIES
        if item[1] == "_canonical_material_symmetry")
    monkeypatch.setattr(function, "__code__", (lambda *args, **kwargs: None).__code__)
    assert not boundary._mass_classes_are_untouched()



def test_cp1252_builtin_encoder_admission_and_mutation_refusal(monkeypatch, tmp_path):
    import io
    stream = io.TextIOWrapper(io.FileIO(tmp_path / "cp1252", "w"), encoding="cp1252", errors="backslashreplace")
    try:
        assert boundary._builtin_warning_stream(stream)
        monkeypatch.setattr(boundary._CP1252_CODEC.IncrementalEncoder.encode, "__code__", (lambda *args: b'').__code__)
        assert not boundary._builtin_warning_stream(stream)
    finally:
        stream.close()


def test_retained_encoder_instance_shadow_refuses_warning_writer(tmp_path):
    import io
    stream = io.TextIOWrapper(io.FileIO(tmp_path / "cp1252", "w"), encoding="cp1252", errors="backslashreplace")
    encoder = next(value for value in boundary._GET_REFERENTS(stream)
                   if type(value) is boundary._CP1252_CODEC.IncrementalEncoder)
    try:
        encoder.encode = lambda *args: b''
        assert not boundary._builtin_warning_stream(stream)
    finally:
        del encoder.encode
        stream.close()



def test_preimport_referent_provider_is_not_self_authorized():
    import os
    import subprocess
    import sys
    script = """import gc
seen = []
gc.get_referents = lambda *args: seen.append('foreign') or []
import anyfesolver.boundary as boundary
from types import MappingProxyType
assert not boundary._plain_mass_input(MappingProxyType({}))
assert not boundary._numpy_error_routes_are_builtin()
assert seen == []
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script],
                               env=os.environ.copy(), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr



def test_cp1252_foreign_encoding_table_refuses_writer(monkeypatch, tmp_path):
    import io
    seen = []
    class ForeignTable:
        def __getitem__(self, key):
            seen.append(key)
            return key
    stream = io.TextIOWrapper(io.FileIO(tmp_path / "cp1252", "w"), encoding="cp1252", errors="backslashreplace")
    try:
        monkeypatch.setattr(boundary._CP1252_CODEC, "encoding_table", ForeignTable())
        assert not boundary._builtin_warning_stream(stream)
        assert seen == []
    finally:
        stream.close()
