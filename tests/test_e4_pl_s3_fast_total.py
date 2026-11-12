from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3_module
import anyfesolver.fe_core as fe_core_module
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.elements import Element, ShellElement
from anyfesolver.fe_core import FEMesh, Material


def _exact_case() -> tuple[FEMesh, QualifiedE4PLS3ShellElement, Material]:
    mesh = FEMesh()
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0, 0.0)
    mesh.add_node(3, 0.15, 0.85, 0.0)
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.08,
        reference_normal=[0.0, 0.0, 1.0],
    )
    mesh.add_element(1, element)
    material = Material(
        "steel",
        elastic_modulus=210.0e9,
        poisson_ratio=0.3,
        density=7850.0,
    )
    return mesh, element, material


def _terminal_base(value: np.ndarray) -> object:
    current: object = value
    while type(current) is np.ndarray:
        current = current.base
    return current


def test_s3_warm_total_is_exact_fresh_and_closure_backed() -> None:
    mesh, element, material = _exact_case()
    cold = element.compute_stiffness_matrix(mesh, material)
    warm_a = element.compute_stiffness_matrix(mesh, material)
    warm_b = element.compute_stiffness_matrix(mesh, material)

    assert np.array_equal(warm_a, cold)
    assert np.array_equal(warm_b, cold)
    assert warm_a is not warm_b
    assert warm_a.flags.writeable is False
    assert warm_b.flags.writeable is False
    assert type(_terminal_base(warm_a)) is bytes

    # Public ndarray metadata is disposable and cannot poison the next view.
    warm_a.shape = (9, 36)
    next_warm = element.compute_stiffness_matrix(mesh, material)
    assert next_warm.shape == (18, 18)
    assert np.array_equal(next_warm, cold)


def test_s3_cold_total_rejects_femesh_get_node_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, element, material = _exact_case()
    reached: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> object:
        reached.append("get_node")
        raise AssertionError("changed FEMesh.get_node was invoked")

    monkeypatch.setattr(FEMesh, "get_node", tripwire)
    with pytest.raises(ValueError, match="FEMesh class namespace changed"):
        element.compute_stiffness_matrix(mesh, material)
    assert reached == []


def test_s3_internal_force_rejects_inherited_displacement_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, element, material = _exact_case()
    reached: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> object:
        reached.append("_get_element_displacements")
        raise AssertionError("changed Element displacement provider was invoked")

    monkeypatch.setattr(Element, "_get_element_displacements", tripwire)
    with pytest.raises(ValueError, match="Element class namespace changed"):
        element.compute_internal_forces(
            mesh,
            np.zeros(mesh.dof_manager.total_dofs, dtype=np.float64),
            material,
        )
    assert reached == []


def test_s3_warm_total_rejects_femesh_get_node_before_cached_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, element, material = _exact_case()
    baseline = element.compute_stiffness_matrix(mesh, material)
    reached: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> object:
        reached.append("get_node")
        raise AssertionError("changed FEMesh.get_node was invoked")

    with monkeypatch.context() as scoped:
        scoped.setattr(FEMesh, "get_node", tripwire)
        with pytest.raises(
            ValueError,
            match="cached stiffness class authority changed",
        ):
            element.compute_stiffness_matrix(mesh, material)
    assert reached == []
    assert element._qualified_components is None
    recovered = element.compute_stiffness_matrix(mesh, material)
    np.testing.assert_array_equal(recovered, baseline)


def test_s3_assembly_accessor_returns_exact_closure_owned_bytes() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)

    s3_module._require_s3_cached_stiffness_runtime_epoch_authority()
    s3_module._require_s3_fast_base_authority()
    payload = s3_module._try_s3_fast_assembly_cached_stiffness(
        element,
        mesh,
        material,
    )

    assert type(payload) is bytes
    assert len(payload) == 18 * 18 * 8
    actual = np.ndarray((18, 18), dtype=np.float64, buffer=payload)
    assert actual.flags.writeable is False
    assert np.array_equal(actual, expected)


def test_s3_assembly_accessor_rejects_raw_node_coordinate_change() -> None:
    mesh, element, material = _exact_case()
    element.compute_stiffness_matrix(mesh, material)
    node = mesh.nodes[1]
    original = node.x

    object.__setattr__(node, "x", 0.125)
    try:
        assert (
            s3_module._try_s3_fast_assembly_cached_stiffness(
                element,
                mesh,
                material,
            )
            is None
        )
    finally:
        object.__setattr__(node, "x", original)


def test_s3_assembly_accessor_rejects_raw_connectivity_change() -> None:
    mesh, element, material = _exact_case()
    element.compute_stiffness_matrix(mesh, material)
    original = element.node_ids

    object.__setattr__(element, "node_ids", (1, 2, 1))
    try:
        assert (
            s3_module._try_s3_fast_assembly_cached_stiffness(
                element,
                mesh,
                material,
            )
            is None
        )
    finally:
        object.__setattr__(element, "node_ids", original)


def test_s3_warm_total_rebuilds_after_owned_geometry_and_material_changes() -> None:
    mesh, element, material = _exact_case()
    baseline = element.compute_stiffness_matrix(mesh, material)
    element.compute_stiffness_matrix(mesh, material)

    mesh.nodes[2].x = 1.2
    geometry_changed = element.compute_stiffness_matrix(mesh, material)
    assert geometry_changed.shape == (18, 18)
    assert not np.array_equal(geometry_changed, baseline)

    material.elastic_modulus = 175.0e9
    material_changed = element.compute_stiffness_matrix(mesh, material)
    assert material_changed.shape == (18, 18)
    assert not np.array_equal(material_changed, geometry_changed)
    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        material_changed,
    )


@pytest.mark.parametrize("provider", ["mesh", "mapping_instance", "mapping_class"])
def test_s3_warm_total_rejects_provider_changes_without_invocation(
    provider: str,
) -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    element.compute_stiffness_matrix(mesh, material)
    reached: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> object:
        reached.append(provider)
        raise AssertionError("changed provider was invoked")

    mapping_type = fe_core_module._QualifiedStateMapping
    if provider == "mesh":
        mesh.__dict__["get_node"] = tripwire
    elif provider == "mapping_instance":
        mesh.nodes.__dict__["get"] = tripwire
    else:
        setattr(mapping_type, "get", tripwire)
    try:
        with pytest.raises((RuntimeError, ValueError), match="qualified S3"):
            element.compute_stiffness_matrix(mesh, material)
    finally:
        if provider == "mesh":
            mesh.__dict__.pop("get_node", None)
        elif provider == "mapping_instance":
            mesh.nodes.__dict__.pop("get", None)
        else:
            delattr(mapping_type, "get")
    assert reached == []
    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )


def test_s3_warm_total_rejects_lost_element_routing_then_recovers() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    element.compute_stiffness_matrix(mesh, material)

    dict.__delitem__(mesh.elements, 1)
    try:
        with pytest.raises((RuntimeError, ValueError), match="routing"):
            element.compute_stiffness_matrix(mesh, material)
    finally:
        dict.__setitem__(mesh.elements, 1, element)
    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )


def test_s3_total_only_ignores_non_total_metadata_until_component_use() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    element.compute_stiffness_matrix(mesh, material)
    components = element._qualified_components
    assert components is not None
    pl = components["pl"]
    original_shape = pl.shape
    pl.shape = (9, 36)
    try:
        assert np.array_equal(
            element.compute_stiffness_matrix(mesh, material),
            expected,
        )
        with pytest.raises((RuntimeError, ValueError), match="component|authority"):
            element.numerical_internal_force(np.zeros(18, dtype=np.float64))
    finally:
        pl.shape = original_shape


def test_s3_warm_total_rejects_total_and_scientific_metadata_changes() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    components = element._qualified_components
    assert components is not None
    total = components["total"]
    original_total_shape = total.shape
    total.shape = (9, 36)
    try:
        with pytest.raises((RuntimeError, ValueError), match="authority"):
            element.compute_stiffness_matrix(mesh, material)
    finally:
        total.shape = original_total_shape
    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )

    weights = s3_module._S3_QUADRATURE_WEIGHTS
    original_weights_shape = weights.shape
    weights.shape = (1, 7)
    try:
        with pytest.raises((RuntimeError, ValueError), match="authority"):
            element.compute_stiffness_matrix(mesh, material)
    finally:
        weights.shape = original_weights_shape
    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )


def test_s3_warm_total_never_invokes_shadowed_inherited_kernel() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    element.compute_stiffness_matrix(mesh, material)
    original = ShellElement._material_angle
    reached: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> float:
        reached.append("base")
        raise AssertionError("changed base kernel was invoked")

    ShellElement._material_angle = tripwire
    try:
        actual = element.compute_stiffness_matrix(mesh, material)
    finally:
        ShellElement._material_angle = original
    assert reached == []
    assert np.array_equal(actual, expected)
    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )


def test_s3_warm_total_rejects_changed_fast_helper_code() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    helper = s3_module._try_s3_fast_cached_stiffness
    original_code = helper.__code__
    changed_code = original_code.replace(co_name="changed_fast_total")

    helper.__code__ = changed_code
    try:
        with pytest.raises(
            ValueError,
            match="warm runtime authority changed: cached stiffness",
        ):
            element.compute_stiffness_matrix(mesh, material)
    finally:
        helper.__code__ = original_code

    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )


def test_s3_warm_total_does_not_invoke_restore_on_use_helper_code() -> None:
    mesh, element, material = _exact_case()
    expected = element.compute_stiffness_matrix(mesh, material)
    helper = s3_module._try_s3_fast_cached_stiffness
    original_code = helper.__code__

    def replacement_factory(original: object, saved: object):
        def replacement(
            _element: object,
            _mesh: object,
            _material: object,
        ) -> np.ndarray:
            original.__code__ = saved  # type: ignore[attr-defined]
            return np.zeros((18, 18), dtype=np.float64)

        return replacement

    replacement_code = replacement_factory(helper, original_code).__code__
    assert len(replacement_code.co_freevars) == len(original_code.co_freevars)
    helper.__code__ = replacement_code
    try:
        with pytest.raises(
            ValueError,
            match="warm runtime authority changed: cached stiffness",
        ):
            element.compute_stiffness_matrix(mesh, material)
        assert helper.__code__ is replacement_code
    finally:
        helper.__code__ = original_code

    assert np.array_equal(
        element.compute_stiffness_matrix(mesh, material),
        expected,
    )


def test_s3_non_builtin_mesh_preserves_the_full_guarded_route() -> None:
    class CustomMesh(FEMesh):
        pass

    mesh = CustomMesh()
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0, 0.0)
    mesh.add_node(3, 0.15, 0.85, 0.0)
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.08,
        reference_normal=[0.0, 0.0, 1.0],
    )
    mesh.add_element(1, element)
    material = Material("steel", 210.0e9, 0.3, 7850.0)

    first = element.compute_stiffness_matrix(mesh, material)
    second = element.compute_stiffness_matrix(mesh, material)
    assert first.shape == (18, 18)
    assert np.array_equal(first, second)
