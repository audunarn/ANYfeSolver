"""Closed grouping tests for mixed shell quadrature authorities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import anyfesolver.matrix_assembly as matrix_assembly_module
import anyfesolver.nonlinear_performance as nonlinear_performance_module
from anyfesolver import (
    FEModel,
    GeneralizedShellSection,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from anyfesolver.activity import ElementActivity
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import LegacyShellElement, ShellElement
from anyfesolver.matrix_assembly import (
    assemble_external_load_tangent,
    assemble_geometric_stiffness_matrix,
    assemble_load_vector,
)


_REFERENCE_SPARSITY_GETTER = matrix_assembly_module._get_cached_sparsity_pattern


class _VariableQuadratureLegacyQ4(LegacyShellElement):
    """Legal legacy/custom Q4 whose full-integration rule is instance-owned."""

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        material_name: str,
        *,
        quadrature: str,
        **kwargs: object,
    ) -> None:
        if quadrature == "one_float64":
            points = np.asarray(((0.0, 0.0),), dtype=np.float64)
            weights = np.asarray((4.0,), dtype=np.float64)
        elif quadrature in {
            "four_float64",
            "four_weighted_float64",
            "four_float32",
        }:
            dtype = np.float64 if quadrature.endswith("float64") else np.float32
            root = np.asarray(np.sqrt(3.0), dtype=dtype)
            points = np.asarray(
                ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
                dtype=dtype,
            ) / root
            if quadrature == "four_weighted_float64":
                weights = np.asarray((0.7, 0.9, 1.1, 1.3), dtype=dtype)
            else:
                weights = np.ones(4, dtype=dtype)
        else:
            raise ValueError(f"unknown test quadrature {quadrature!r}")
        self._test_gauss_points = np.ascontiguousarray(points)
        self._test_gauss_weights = np.ascontiguousarray(weights)
        super().__init__(element_id, list(node_ids), material_name, **kwargs)

    @property
    def gauss_points(self) -> np.ndarray:
        return self._test_gauss_points

    @property
    def gauss_weights(self) -> np.ndarray:
        return self._test_gauss_weights


class _AlternateVariableQuadratureLegacyQ4(_VariableQuadratureLegacyQ4):
    """Second concrete custom type with an otherwise identical rule."""


_ELEMENT_ORDER = (
    "qualified_q4",
    "qualified_s3",
    "custom_one_float64",
    "custom_four_float64",
    "custom_four_weighted_float64",
    "custom_four_float32",
    "alternate_four_float64",
)


def _section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.asarray(
            ((2.4e8, 0.7e8, 0.0), (0.7e8, 2.1e8, 0.0), (0.0, 0.0, 0.8e8))
        ),
        B=np.zeros((3, 3), dtype=float),
        D=np.asarray(
            ((1.9e4, 0.5e4, 0.0), (0.5e4, 1.7e4, 0.0), (0.0, 0.0, 0.6e4))
        ),
        As=np.asarray(((7.0e7, 0.0), (0.0, 6.0e7))),
        mass_per_area=37.0,
        rotary_inertia_per_area=0.021,
    )


def _mixed_model(
    order: Sequence[str],
    *,
    generalized_mass: bool,
) -> FEModel:
    model = FEModel("mixed-shell-quadrature-grouping")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    section = _section() if generalized_mass else None
    definitions: dict[str, tuple[int, tuple[tuple[float, float, float], ...]]] = {
        "qualified_q4": (
            1,
            ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (1.0, 0.9, 0.0), (0.0, 1.0, 0.0)),
        ),
        "qualified_s3": (
            2,
            ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.5, 0.866025403784, 0.0)),
        ),
        "custom_one_float64": (
            3,
            ((6.0, 0.0, 0.0), (7.2, 0.0, 0.0), (7.0, 0.8, 0.0), (6.1, 1.0, 0.0)),
        ),
        "custom_four_float64": (
            4,
            ((9.0, 0.0, 0.0), (10.0, 0.1, 0.0), (10.1, 1.0, 0.0), (8.9, 0.9, 0.0)),
        ),
        "custom_four_float32": (
            5,
            ((12.0, 0.0, 0.0), (13.1, -0.1, 0.0), (13.0, 1.1, 0.0), (12.1, 0.9, 0.0)),
        ),
        "custom_four_weighted_float64": (
            6,
            ((15.0, 0.0, 0.0), (16.0, 0.1, 0.0), (16.2, 1.0, 0.0), (14.9, 0.9, 0.0)),
        ),
        "alternate_four_float64": (
            7,
            ((18.0, 0.0, 0.0), (19.1, 0.0, 0.0), (19.0, 1.0, 0.0), (18.0, 0.9, 0.0)),
        ),
    }
    node_ids_by_name: dict[str, list[int]] = {}
    next_node = 1
    for name in _ELEMENT_ORDER:
        _element_id, coordinates = definitions[name]
        node_ids: list[int] = []
        for point in coordinates:
            model.add_node(next_node, *point)
            node_ids.append(next_node)
            next_node += 1
        node_ids_by_name[name] = node_ids

    for name in order:
        element_id, _coordinates = definitions[name]
        node_ids = node_ids_by_name[name]
        common = {
            "thickness": 0.02,
            "shell_section": section,
        }
        if name == "qualified_q4":
            element = QualifiedE4PLShellElement(
                element_id,
                node_ids,
                "steel",
                reference_normal=(0.0, 0.0, 1.0),
                material_direction=(1.0, 0.0, 0.0),
                **common,
            )
        elif name == "qualified_s3":
            element = QualifiedE4PLS3ShellElement(
                element_id,
                node_ids,
                "steel",
                reference_normal=(0.0, 0.0, 1.0),
                material_direction=(1.0, 0.0, 0.0),
                **common,
            )
        else:
            element_type = (
                _AlternateVariableQuadratureLegacyQ4
                if name == "alternate_four_float64"
                else _VariableQuadratureLegacyQ4
            )
            quadrature = (
                "four_float64"
                if name == "alternate_four_float64"
                else name.removeprefix("custom_")
            )
            element = element_type(
                element_id,
                node_ids,
                "steel",
                quadrature=quadrature,
                **common,
            )
        model.add_element(element_id, element)
    return model


def _direct_global(model: FEModel, matrix_type: str) -> np.ndarray:
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    result = np.zeros((total_dofs, total_dofs), dtype=float)
    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        if matrix_type == "mass":
            local = element.compute_mass_matrix(model.mesh, material)
        else:
            local = element.compute_stiffness_matrix(model.mesh, material)
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        result[np.ix_(dofs, dofs)] += np.asarray(local, dtype=float)
    return result


@pytest.mark.parametrize("optimized_sparsity", (False, True))
def test_direct_qualified_q4_connectivity_change_invalidates_warm_sparsity(
    optimized_sparsity: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        matrix_assembly_module,
        "_get_cached_sparsity_pattern",
        (
            nonlinear_performance_module._revision_cached_sparsity_pattern
            if optimized_sparsity
            else _REFERENCE_SPARSITY_GETTER
        ),
    )
    model = FEModel("qualified-q4-direct-connectivity")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)

    _warm, warm_info = assemble_stiffness_matrix(model)
    token_before = model.mesh._qualified_direct_state_token[0]
    element.node_ids = (5, 6, 7, 8)
    assert model.mesh._qualified_direct_state_token[0] == token_before + 1

    changed, changed_info = assemble_stiffness_matrix(model)
    local = element.compute_stiffness_matrix(
        model.mesh, model.get_material(element.material_name)
    )
    dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    expected = np.zeros(changed.shape, dtype=float)
    expected[np.ix_(dofs, dofs)] = np.asarray(local, dtype=float)

    assert changed_info["sparsity_signature"] != warm_info["sparsity_signature"]
    np.testing.assert_allclose(
        changed.toarray(), expected, rtol=3.0e-12, atol=2.0e-7
    )
    assert np.linalg.norm(changed.toarray()[:24]) == 0.0
    assert np.linalg.norm(changed.toarray()[24:]) > 0.0


@pytest.mark.parametrize("order", (_ELEMENT_ORDER, tuple(reversed(_ELEMENT_ORDER))))
def test_generic_mixed_shell_groups_bind_type_dtype_shape_and_bytes(
    order: Sequence[str],
) -> None:
    mass_model = _mixed_model(order, generalized_mass=False)
    stiffness_model = _mixed_model(order, generalized_mass=False)
    expected_mass = _direct_global(
        _mixed_model(order, generalized_mass=False), "mass"
    )
    expected_stiffness = _direct_global(
        _mixed_model(order, generalized_mass=False), "stiffness"
    )

    mass, mass_info = assemble_mass_matrix(mass_model)
    stiffness, stiffness_info = assemble_stiffness_matrix(stiffness_model)

    np.testing.assert_allclose(
        mass.toarray(), expected_mass, rtol=3.0e-12, atol=2.0e-13
    )
    np.testing.assert_allclose(
        stiffness.toarray(), expected_stiffness, rtol=1.0e-10, atol=1.0e-3
    )
    mass_groups = [
        group
        for group in mass_info["diagnostics"]["vectorized_shell_groups"]
        if group["kernel"] == "compute_shell_mass_matrices_jit"
    ]
    stiffness_groups = [
        group
        for group in stiffness_info["diagnostics"]["vectorized_shell_groups"]
        if group["kernel"] == "compute_shell_stiffness_matrices_jit"
    ]
    assert [group["num_elements"] for group in mass_groups] == [1, 1, 1, 1, 1, 1]
    assert [group["num_elements"] for group in stiffness_groups] == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("order", (_ELEMENT_ORDER, tuple(reversed(_ELEMENT_ORDER))))
def test_generalized_section_mass_and_stiffness_groups_are_order_independent(
    order: Sequence[str],
) -> None:
    mass_model = _mixed_model(order, generalized_mass=True)
    stiffness_model = _mixed_model(order, generalized_mass=True)
    expected_mass = _direct_global(
        _mixed_model(order, generalized_mass=True), "mass"
    )
    expected_stiffness = _direct_global(
        _mixed_model(order, generalized_mass=True), "stiffness"
    )

    mass, mass_info = assemble_mass_matrix(mass_model)
    stiffness, stiffness_info = assemble_stiffness_matrix(stiffness_model)

    np.testing.assert_allclose(
        mass.toarray(), expected_mass, rtol=3.0e-12, atol=2.0e-13
    )
    np.testing.assert_allclose(
        stiffness.toarray(), expected_stiffness, rtol=3.0e-12, atol=2.0e-7
    )
    mass_groups = [
        group
        for group in mass_info["diagnostics"]["vectorized_shell_groups"]
        if group["kernel"] == "compute_s4_section_mass_matrices_jit"
    ]
    stiffness_groups = [
        group
        for group in stiffness_info["diagnostics"]["vectorized_shell_groups"]
        if group["kernel"] == "compute_s4_generalized_stiffness_matrices_jit"
    ]
    assert [group["num_elements"] for group in mass_groups] == [1, 1, 1, 1, 1, 1]
    assert [group["num_elements"] for group in stiffness_groups] == [1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "assembler",
    (assemble_stiffness_matrix, assemble_mass_matrix),
)
def test_activity_scale_callback_is_guarded_before_numeric_validation(
    assembler: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(("qualified_q4",), generalized_mass=False)
    activity = ElementActivity([1])
    model.set_element_activity(activity)
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> bool:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    def observed_scales(_quantity: str, _element_ids: object) -> np.ndarray:
        reached.append("scales")
        monkeypatch.setattr(np, "all", forbidden_numeric)
        return np.ones(1, dtype=float)

    monkeypatch.setattr(activity, "scales", observed_scales)
    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        assembler(model)  # type: ignore[operator]
    assert reached == ["scales"]


def test_geometric_state_provider_is_guarded_before_shell_state_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(("qualified_q4",), generalized_mass=False)
    reached: list[str] = []

    def forbidden_state_mechanics(*_args: object, **_kwargs: object) -> np.ndarray:
        reached.append("state_mechanics")
        raise AssertionError("changed shell state helper was invoked")

    def provider(_element_id: int, _element: object) -> dict[str, object]:
        reached.append("provider")
        monkeypatch.setattr(
            ShellElement,
            "_membrane_compression_samples",
            forbidden_state_mechanics,
        )
        return {"membrane_compression": [1.0e4, 0.0, 0.0]}

    with pytest.raises(ElementCapabilityError, match="DEPENDENCY_AUTHORITY_MISMATCH"):
        assemble_geometric_stiffness_matrix(model, provider)
    assert reached == ["provider"]


@pytest.mark.parametrize(
    "assembler",
    (assemble_load_vector, assemble_external_load_tangent),
)
def test_load_displacement_array_protocol_is_guarded_before_finite_check(
    assembler: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(("qualified_q4",), generalized_mass=False)
    values = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> bool:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    class ObservedArray:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            del copy
            reached.append("array")
            monkeypatch.setattr(np, "all", forbidden_numeric)
            return values.astype(dtype, copy=False)

    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        assembler(model, None, ObservedArray())  # type: ignore[operator]
    assert reached == ["array"]


def test_trusted_scope_cleanup_termination_outranks_ordinary_operation_error() -> None:
    assert matrix_assembly_module._scoped_operation_error_has_precedence(
        ValueError("operation"),
        RuntimeError("cleanup"),
    )
    assert not matrix_assembly_module._scoped_operation_error_has_precedence(
        ValueError("operation"),
        KeyboardInterrupt(),
    )
    assert matrix_assembly_module._scoped_operation_error_has_precedence(
        KeyboardInterrupt(),
        RuntimeError("cleanup"),
    )
