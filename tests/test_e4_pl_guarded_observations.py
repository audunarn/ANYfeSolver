from __future__ import annotations

import copy
from abc import ABC
from typing import Any

import numpy as np
import pytest

from anymaterial import DNVC208MaterialCurve, LinearHardeningCurve
from anyfesolver import FEModel, QualifiedE4PLS3ShellElement, QualifiedE4PLShellElement
from anyfesolver._native_rotation_state import (
    NativeElementRotationView,
    create_native_rotation_state_store,
)
from anyfesolver.current_state_tangent import (
    require_exact_qualified_component_lifecycle_api,
)
from anyfesolver.e4_pl_element import _QUALIFIED_Q4_CLASS_NAMESPACE_AUTHORITY
from anyfesolver.e4_pl_s3_element import _QUALIFIED_S3_CLASS_NAMESPACE_AUTHORITY
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import Element
from anyfesolver.fe_core import FEMesh
from anyfesolver.material_curves import FiberSectionPlasticityConfig
from anyfesolver.shell_sections import GeneralizedShellSection


def _qualified_case(family: str):
    model = FEModel(f"guarded-observation-{family}")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    if family == "q4":
        coordinates = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        element = QualifiedE4PLShellElement(
            1,
            (1, 2, 3, 4),
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        )
    else:
        coordinates = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.5 * np.sqrt(3.0), 0.0),
        )
        element = QualifiedE4PLS3ShellElement(
            1,
            (1, 2, 3),
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        )
    for node_id, point in enumerate(coordinates, start=1):
        model.add_node(node_id, *point)
    model.add_element(1, element)
    return model, element, model.get_material("steel")


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_lazy_abc_annotations_are_captured_before_qualified_call(family: str) -> None:
    authority = (
        _QUALIFIED_Q4_CLASS_NAMESPACE_AUTHORITY
        if family == "q4"
        else _QUALIFIED_S3_CLASS_NAMESPACE_AUTHORITY
    )
    type.__getattribute__(ABC, "__annotations__")
    namespace = type.__getattribute__(ABC, "__dict__")
    assert set(namespace).issubset(authority[ABC])

    model, element, material = _qualified_case(family)
    stiffness = element.compute_stiffness_matrix(model.mesh, material)
    assert stiffness.shape == ((24, 24) if family == "q4" else (18, 18))
    assert np.all(np.isfinite(stiffness))


def test_s3_material_deepcopy_slotnames_cache_is_benign_before_qualified_call(
) -> None:
    model, element, material = _qualified_case("s3")
    material_type = type(material)
    material_namespace = type.__getattribute__(material_type, "__dict__")
    had_slotnames = "__slotnames__" in material_namespace
    original_slotnames = material_namespace.get("__slotnames__")
    if had_slotnames:
        delattr(material_type, "__slotnames__")

    try:
        copied = copy.deepcopy(material)
        assert copied is not material
        assert "__slotnames__" in type.__getattribute__(material_type, "__dict__")
        stiffness = element.compute_stiffness_matrix(model.mesh, material)
        assert stiffness.shape == (18, 18)
        assert np.all(np.isfinite(stiffness))
    finally:
        if had_slotnames:
            setattr(material_type, "__slotnames__", original_slotnames)
        elif "__slotnames__" in type.__getattribute__(material_type, "__dict__"):
            delattr(material_type, "__slotnames__")


def test_q4_material_deepcopy_slotnames_cache_is_benign_before_qualified_call(
) -> None:
    model, element, material = _qualified_case("q4")
    material_type = type(material)
    material_namespace = type.__getattribute__(material_type, "__dict__")
    had_slotnames = "__slotnames__" in material_namespace
    original_slotnames = material_namespace.get("__slotnames__")
    if had_slotnames:
        delattr(material_type, "__slotnames__")

    try:
        copied = copy.deepcopy(material)
        assert copied is not material
        assert "__slotnames__" in type.__getattribute__(material_type, "__dict__")
        stiffness = element.compute_stiffness_matrix(model.mesh, material)
        assert stiffness.shape == (24, 24)
        assert np.all(np.isfinite(stiffness))
    finally:
        if had_slotnames:
            setattr(material_type, "__slotnames__", original_slotnames)
        elif "__slotnames__" in type.__getattribute__(material_type, "__dict__"):
            delattr(material_type, "__slotnames__")


@pytest.mark.parametrize(
    ("family", "dependency"),
    (
        (
            "q4",
            DNVC208MaterialCurve(
                sigma_prop=320.0e6,
                sigma_yield=357.0e6,
                sigma_yield_2=363.3e6,
                eps_p_y1=0.004,
                eps_p_y2=0.015,
                K=740.0e6,
                n=0.166,
            ),
        ),
        ("s3", FiberSectionPlasticityConfig()),
    ),
    ids=("q4-anymaterial-curve", "s3-solver-fiber-config"),
)
def test_model_guard_allows_only_the_stdlib_slotnames_dependency_cache(
    family: str,
    dependency: Any,
) -> None:
    model, _, _ = _qualified_case(family)
    dependency_type = type(dependency)
    namespace = type.__getattribute__(dependency_type, "__dict__")
    had_slotnames = "__slotnames__" in namespace
    original_slotnames = namespace.get("__slotnames__")
    if had_slotnames:
        delattr(dependency_type, "__slotnames__")

    try:
        copied = copy.deepcopy(dependency)
        assert copied is not dependency
        assert "__slotnames__" in type.__getattribute__(dependency_type, "__dict__")
        result = require_exact_qualified_component_lifecycle_api(
            model,
            context="stdlib slotnames dependency cache regression",
        )
        assert result == {"qualified_element_ids": [1], "guarded": True}

        setattr(dependency_type, "__slotnames___", ())
        with pytest.raises(
            ElementCapabilityError,
            match="DEPENDENCY_AUTHORITY_MISMATCH=.*__slotnames___",
        ):
            require_exact_qualified_component_lifecycle_api(
                model,
                context="near-miss dependency cache mutation",
            )
    finally:
        if "__slotnames___" in type.__getattribute__(dependency_type, "__dict__"):
            delattr(dependency_type, "__slotnames___")
        if had_slotnames:
            setattr(dependency_type, "__slotnames__", original_slotnames)
        elif "__slotnames__" in type.__getattribute__(dependency_type, "__dict__"):
            delattr(dependency_type, "__slotnames__")


def test_q4_cold_stiffness_rejects_femesh_provider_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case("q4")
    original = FEMesh.get_node
    reached: list[int] = []

    def changed(mesh: FEMesh, node_id: int):
        reached.append(node_id)
        return original(mesh, node_id)

    monkeypatch.setattr(FEMesh, "get_node", changed)
    with pytest.raises(ValueError, match="FEMesh class namespace changed"):
        element.compute_stiffness_matrix(model.mesh, material)
    assert reached == []


def test_q4_internal_force_rejects_element_provider_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case("q4")
    original = Element._get_element_displacements
    reached: list[bool] = []

    def changed(*args: Any, **kwargs: Any):
        reached.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(Element, "_get_element_displacements", changed)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    with pytest.raises(ValueError, match="Element class namespace changed"):
        element.compute_internal_forces(model.mesh, displacement, material)
    assert reached == []


def test_q4_warm_stiffness_rejects_femesh_provider_before_cache_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case("q4")
    baseline = element.compute_stiffness_matrix(model.mesh, material)
    reached: list[int] = []
    original = FEMesh.get_node

    def changed(mesh: FEMesh, node_id: int):
        reached.append(node_id)
        return original(mesh, node_id)

    with monkeypatch.context() as scoped:
        scoped.setattr(FEMesh, "get_node", changed)
        with pytest.raises(ValueError, match="builtin mesh class authority changed"):
            element.compute_stiffness_matrix(model.mesh, material)
    assert reached == []
    assert element._qualified_components is None
    recovered = element.compute_stiffness_matrix(model.mesh, material)
    np.testing.assert_array_equal(recovered, baseline)


def test_q4_material_class_rejects_changed_provider_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case("q4")
    reached: list[bool] = []

    def hardening_curve(_self: Any) -> None:
        reached.append(True)
        return None

    monkeypatch.setattr(
        type(material),
        "hardening_curve",
        property(hardening_curve),
    )
    with pytest.raises(
        ValueError,
        match="IsotropicMaterial class namespace changed",
    ):
        element.compute_stiffness_matrix(model.mesh, material)
    assert reached == []


def _s3_native_view(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    committed_state: dict[str, object],
) -> NativeElementRotationView:
    reference = element.get_node_coordinates(model.mesh)
    committed_u = np.asarray(
        committed_state["committed_total_u"], dtype=np.float64
    ).reshape(18)
    committed_q = np.asarray(
        committed_state["committed_nodal_rotation_matrices"],
        dtype=np.float64,
    ).reshape(3, 3, 3)
    node_ids = tuple(element.node_ids)
    store = create_native_rotation_state_store(
        node_ids,
        rotational_dofs={
            node_id: (6 * row + 3, 6 * row + 4, 6 * row + 5)
            for row, node_id in enumerate(node_ids)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(node_ids)},
        coordinate_node_ids=node_ids,
        committed_full_displacement=committed_u,
        committed_full_coordinates=(
            reference + committed_u.reshape(3, 6)[:, :3]
        ),
        committed_rotation_matrices={
            node_id: committed_q[row] for row, node_id in enumerate(node_ids)
        },
    )
    assert store is not None
    trial_u = np.asarray(total_u, dtype=np.float64).reshape(18)
    token = store.begin_trial(
        trial_u,
        reference + trial_u.reshape(3, 6)[:, :3],
    )
    return store.element_view(
        element.element_id,
        node_ids,
        element.native_reference_directors(model.mesh),
        trial_token=token,
    )


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_revision_mapping_callback_rechecks_before_cache_arithmetic(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case(family)
    element.compute_stiffness_matrix(model.mesh, material)
    original = np.ascontiguousarray
    reached: list[str] = []

    def changed(*args: Any, **kwargs: Any):
        reached.append("ascontiguousarray")
        return original(*args, **kwargs)

    class RevisionResult:
        def items(self):
            monkeypatch.setattr(np, "ascontiguousarray", changed)
            return {"geometry": 0, "material": 0}.items()

    object.__setattr__(model.mesh, "revision_signature", lambda: RevisionResult())
    expected_error = (
        "cached stiffness route authority"
        if family == "s3"
        else "exact numerical runtime"
    )
    with pytest.raises(ValueError, match=expected_error):
        element.compute_stiffness_matrix(model.mesh, material)
    assert reached == []


@pytest.mark.parametrize("family", ("q4", "s3"))
@pytest.mark.parametrize("route", ("stiffness", "recovery"))
def test_material_property_callback_rechecks_before_formulation_mechanics(
    family: str,
    route: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, _material = _qualified_case(family)
    original = np.array
    reached: list[str] = []

    def changed(*args: Any, **kwargs: Any):
        reached.append("array")
        return original(*args, **kwargs)

    class Material:
        elastic_symmetry = "isotropic"
        poisson_ratio = 0.3
        shear_modulus = 210.0e9 / 2.6
        density = 7850.0
        hardening_curve = None
        hill_yield = None
        yield_stress = 0.0

        @property
        def elastic_modulus(self) -> float:
            monkeypatch.setattr(np, "array", changed)
            return 210.0e9

    material = Material()
    with pytest.raises(ValueError, match="exact numerical runtime"):
        if route == "stiffness":
            element.compute_stiffness_matrix(model.mesh, material)
        else:
            element.compute_stresses(
                model.mesh,
                np.zeros(element.total_dofs, dtype=np.float64),
                material,
            )
    assert reached == []


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_generalized_section_assignment_rechecks_before_array_conversion(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model, element, _material = _qualified_case(family)
    original = np.asarray
    reached: list[str] = []

    def changed(*args: Any, **kwargs: Any):
        reached.append("asarray")
        return original(*args, **kwargs)

    class SectionProtocol:
        B = np.zeros((3, 3), dtype=np.float64)
        D = np.eye(3, dtype=np.float64)
        As = np.eye(2, dtype=np.float64)
        name = "callback-section"
        mass_per_area = 1.0
        rotary_inertia_per_area = 0.01

        @property
        def A(self):
            monkeypatch.setattr(np, "asarray", changed)
            return np.eye(3, dtype=np.float64)

    with pytest.raises(ValueError, match="exact numerical runtime"):
        element.shell_section = SectionProtocol()
    assert reached == []


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_density_callback_rechecks_before_mass_arithmetic(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, _material = _qualified_case(family)
    original = np.zeros
    reached: list[str] = []

    def changed(*args: Any, **kwargs: Any):
        reached.append("zeros")
        return original(*args, **kwargs)

    class Material:
        elastic_symmetry = "isotropic"
        elastic_modulus = 210.0e9
        poisson_ratio = 0.3
        shear_modulus = 210.0e9 / 2.6

        @property
        def density(self) -> float:
            monkeypatch.setattr(np, "zeros", changed)
            return 7850.0

    with pytest.raises(ValueError, match="exact numerical runtime"):
        element.compute_mass_matrix(model.mesh, Material())
    assert reached == []


def test_q4_nested_state_array_rechecks_before_state_field_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case("q4")
    state = element.init_nonlinear_state(3)
    original = np.broadcast_to
    reached: list[str] = []

    def changed(*args: Any, **kwargs: Any):
        reached.append("broadcast_to")
        return original(*args, **kwargs)

    class StateArray:
        def __array__(self, dtype=None, copy=None):
            del copy
            monkeypatch.setattr(np, "broadcast_to", changed)
            return np.zeros(3, dtype=dtype)

    state["initial_membrane_stress"] = StateArray()
    with pytest.raises(ValueError, match="exact numerical runtime"):
        element.compute_nonlinear_response(
            model.mesh,
            material,
            np.zeros(24, dtype=np.float64),
            state,
            3,
        )
    assert reached == []


def test_s3_native_view_subclass_rejects_before_field_observation() -> None:
    model, element, material = _qualified_case("s3")
    reached: list[str] = []

    class ForeignView(NativeElementRotationView):
        def __getattribute__(self, name: str):
            reached.append(name)
            return super().__getattribute__(name)

    foreign = object.__new__(ForeignView)
    with pytest.raises(TypeError, match="exact NativeElementRotationView"):
        element.compute_nonlinear_response(
            model.mesh,
            material,
            np.zeros(18, dtype=np.float64),
            None,
            3,
            native_rotation_trial=foreign,
        )
    assert reached == []


def test_s3_material_class_rejects_changed_hardening_descriptor_before_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, element, material = _qualified_case("s3")
    state = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    total_u = np.zeros(18, dtype=np.float64)
    view = _s3_native_view(model, element, total_u, state)
    original = np.array
    reached: list[str] = []
    curve = LinearHardeningCurve(355.0e6, 1.0e9)

    def changed(*args: Any, **kwargs: Any):
        reached.append("array")
        return original(*args, **kwargs)

    def hardening_curve(_self: Any) -> LinearHardeningCurve:
        monkeypatch.setattr(np, "array", changed)
        return curve

    monkeypatch.setattr(
        type(material),
        "hardening_curve",
        property(hardening_curve),
    )
    with pytest.raises(ValueError, match="IsotropicMaterial class namespace changed"):
        element.compute_nonlinear_response(
            model.mesh,
            material,
            total_u,
            state,
            3,
            native_rotation_trial=view,
        )
    assert reached == []


def test_s3_native_layered_response_uses_owned_hardening_curve() -> None:
    model, element, _material = _qualified_case("s3")
    curve = LinearHardeningCurve(355.0e6, 1.0e9)
    reached: list[str] = []

    def shadowed_flow_stress(value: Any) -> np.ndarray:
        reached.append("flow_stress")
        return np.full_like(np.asarray(value, dtype=np.float64), 1.0)

    object.__setattr__(curve, "flow_stress", shadowed_flow_stress)
    material = model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
        yield_stress=355.0e6,
        hardening_curve=curve,
    )
    state = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    total_u = np.zeros(18, dtype=np.float64)
    view = _s3_native_view(model, element, total_u, state)
    force, tangent, _trial = element.compute_nonlinear_response(
        model.mesh,
        material,
        total_u,
        state,
        3,
        native_rotation_trial=view,
    )
    assert reached == []
    assert np.all(np.isfinite(force))
    assert tangent is not None and np.all(np.isfinite(tangent))


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_exact_generalized_section_constructor_remains_supported(family: str) -> None:
    section = GeneralizedShellSection(
        A=np.eye(3, dtype=np.float64),
        B=np.zeros((3, 3), dtype=np.float64),
        D=np.eye(3, dtype=np.float64),
        As=np.eye(2, dtype=np.float64),
        name="constructor-section",
    )
    if family == "q4":
        element = QualifiedE4PLShellElement(
            1,
            (1, 2, 3, 4),
            "section",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
            shell_section=section,
        )
    else:
        element = QualifiedE4PLS3ShellElement(
            1,
            (1, 2, 3),
            "section",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
            shell_section=section,
        )
    assert type(element.shell_section) is GeneralizedShellSection
