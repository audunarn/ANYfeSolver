from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.e4_pl_element as e4_pl_module
import anyfesolver.matrix_assembly as matrix_assembly_module
from anyfesolver import FEModel, assemble_stiffness_matrix
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.fe_core import _QualifiedStateMapping
from anyfesolver.matrix_assembly import AssemblyError


def _model() -> tuple[FEModel, QualifiedE4PLShellElement]:
    model = FEModel("qualified-q4-assembly-authority")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_material("soft", 70.0e9, 0.3, density=2700.0)
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
        [1, 2, 3, 4],
        "steel",
        thickness=0.05,
    )
    model.add_element(1, element)
    return model, element


def _warm(model: FEModel) -> np.ndarray:
    first, _ = assemble_stiffness_matrix(model)
    second, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    return second.toarray()


@pytest.mark.parametrize("provider_name", ("items", "values"))
def test_exact_element_mapping_provider_shadow_fails_before_assembly(
    provider_name: str,
) -> None:
    model, _element = _model()
    baseline = _warm(model)
    mapping = model.mesh.elements
    assert type(mapping) is _QualifiedStateMapping
    object.__setattr__(mapping, provider_name, lambda: iter(()))
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        object.__delattr__(mapping, provider_name)
    recovered, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(recovered.toarray(), baseline)


def test_exact_element_mapping_class_provider_shadow_fails_before_assembly() -> None:
    model, _element = _model()
    baseline = _warm(model)
    type.__setattr__(_QualifiedStateMapping, "items", lambda _self: iter(()))
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        type.__delattr__(_QualifiedStateMapping, "items")
    recovered, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(recovered.toarray(), baseline)


def test_exact_model_material_provider_shadow_fails_before_assembly() -> None:
    model, _element = _model()
    baseline = _warm(model)
    soft = model.materials["soft"]
    object.__setattr__(model, "get_material", lambda _name=None: soft)
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        object.__delattr__(model, "get_material")
    recovered, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(recovered.toarray(), baseline)


def test_public_total_metadata_cannot_poison_warm_assembly() -> None:
    model, element = _model()
    material = model.materials["steel"]
    exposed = element.compute_stiffness_matrix(model.mesh, material)
    expected = exposed.copy()
    exposed.shape = (576,)
    assembled, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(assembled.toarray(), expected)
    repeated = element.compute_stiffness_matrix(model.mesh, material)
    assert repeated.shape == (24, 24)
    np.testing.assert_array_equal(repeated, expected)


def test_sparse_module_and_coo_conversion_dict_surgery_cannot_influence_output() -> None:
    model, _element = _model()
    baseline = _warm(model)
    exact_coo = vars(sparse)["coo_matrix"]
    exact_tocsr = next(
        type.__getattribute__(base, "__dict__")["tocsr"]
        for base in type.__getattribute__(exact_coo, "__mro__")
        if "tocsr" in type.__getattribute__(base, "__dict__")
    )
    reached: list[str] = []

    def forged_coo(*args: Any, **kwargs: Any) -> Any:
        reached.append("coo")
        return exact_coo(*args, **kwargs)

    def forged_tocsr(instance: Any, *args: Any, **kwargs: Any) -> Any:
        reached.append("tocsr")
        made = exact_tocsr(instance, *args, **kwargs)
        made.data[:] = 0.0
        return made

    # These low-level writes intentionally bypass ordinary module/type
    # setattr tracking.  Qualified stiffness assembly remains unaffected
    # because it invokes closure-captured constructor/conversion authority.
    vars(sparse)["coo_matrix"] = forged_coo
    type.__setattr__(exact_coo, "tocsr", forged_tocsr)
    try:
        made, _ = assemble_stiffness_matrix(model)
    finally:
        type.__setattr__(exact_coo, "tocsr", exact_tocsr)
        vars(sparse)["coo_matrix"] = exact_coo
    assert reached == []
    np.testing.assert_array_equal(made.toarray(), baseline)


def test_supported_sparse_setattr_rejects_and_next_clean_recovers() -> None:
    model, _element = _model()
    baseline = _warm(model)
    exact_coo = vars(sparse)["coo_matrix"]

    def forged_coo(*args: Any, **kwargs: Any) -> Any:
        return exact_coo(*args, **kwargs)

    setattr(sparse, "coo_matrix", forged_coo)
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        setattr(sparse, "coo_matrix", exact_coo)
    recovered, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(recovered.toarray(), baseline)


def test_protected_q4_method_replacement_never_dispatches_attacker() -> None:
    model, element = _model()
    material = model.materials["steel"]
    baseline = element.compute_stiffness_matrix(model.mesh, material)
    owner = QualifiedE4PLShellElement
    original = type.__getattribute__(owner, "__dict__")[
        "compute_stiffness_matrix"
    ]
    reached: list[str] = []

    def attacker(*_args: Any, **_kwargs: Any) -> np.ndarray:
        reached.append("attacker")
        return np.zeros((24, 24), dtype=np.float64)

    setattr(owner, "compute_stiffness_matrix", attacker)
    try:
        with pytest.raises(ValueError, match="class authority was replaced"):
            element.compute_stiffness_matrix(model.mesh, material)
    finally:
        setattr(owner, "compute_stiffness_matrix", original)
    assert reached == []
    recovered = element.compute_stiffness_matrix(model.mesh, material)
    np.testing.assert_array_equal(recovered, baseline)


def test_protected_q4_property_replacement_raises_on_lookup() -> None:
    _model_value, element = _model()
    owner = QualifiedE4PLShellElement
    original = type.__getattribute__(owner, "__dict__")[
        "physical_reference_director"
    ]
    baseline = element.physical_reference_director
    reached: list[str] = []

    def attacker(_element: Any) -> np.ndarray:
        reached.append("attacker")
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    setattr(owner, "physical_reference_director", property(attacker))
    try:
        with pytest.raises(ValueError, match="class authority was replaced"):
            _ = element.physical_reference_director
    finally:
        setattr(owner, "physical_reference_director", original)
    assert reached == []
    assert element.physical_reference_director is baseline


def test_bound_q4_method_is_safe_after_replace_then_restore() -> None:
    model, element = _model()
    material = model.materials["steel"]
    bound = element.compute_stiffness_matrix
    baseline = bound(model.mesh, material)
    owner = QualifiedE4PLShellElement
    original = type.__getattribute__(owner, "__dict__")[
        "compute_stiffness_matrix"
    ]
    setattr(owner, "compute_stiffness_matrix", lambda *_args, **_kwargs: None)
    setattr(owner, "compute_stiffness_matrix", original)
    recovered = bound(model.mesh, material)
    np.testing.assert_array_equal(recovered, baseline)


def test_assembly_rejects_bound_kernel_kwdefaults_before_callback() -> None:
    model, _element = _model()
    baseline = _warm(model)
    kernel = matrix_assembly_module._assemble_element_matrix_under_lease
    original = kernel.__kwdefaults__
    assert original is not None
    reached: list[str] = []

    def forged_symmetry(value: Any) -> float:
        reached.append(type(value).__name__)
        return 0.0

    changed = dict(original)
    changed["_relative_symmetry_kernel"] = forged_symmetry
    kernel.__kwdefaults__ = changed
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        kernel.__kwdefaults__ = original
    assert reached == []
    recovered, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(recovered.toarray(), baseline)


def test_direct_q4_ignores_exposed_warm_cache_helper_code_replacement() -> None:
    model, element = _model()
    material = model.materials["steel"]
    baseline = element.compute_stiffness_matrix(model.mesh, material)
    helper = e4_pl_module._try_q4_fast_cached_stiffness
    original = helper.__code__

    reached: list[str] = []

    def forged_factory() -> Any:
        records = reached

        def forged(_element: Any, _mesh: Any, _material: Any) -> np.ndarray:
            records.append("forged")
            return np.zeros((24, 24), dtype=np.float64)

        return forged

    forged = forged_factory()
    assert forged.__code__.co_freevars == original.co_freevars
    helper.__code__ = forged.__code__
    try:
        recovered = element.compute_stiffness_matrix(model.mesh, material)
        np.testing.assert_array_equal(recovered, baseline)
    finally:
        helper.__code__ = original
    assert reached == []
    recovered = element.compute_stiffness_matrix(model.mesh, material)
    np.testing.assert_array_equal(recovered, baseline)


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    (("__defaults__", ()), ("__kwdefaults__", {})),
)
def test_direct_q4_ignores_exposed_warm_cache_helper_default_metadata_replacement(
    attribute: str,
    replacement: Any,
) -> None:
    model, element = _model()
    material = model.materials["steel"]
    baseline = element.compute_stiffness_matrix(model.mesh, material)
    helper = e4_pl_module._try_q4_fast_cached_stiffness
    original = getattr(helper, attribute)
    setattr(helper, attribute, replacement)
    try:
        recovered = element.compute_stiffness_matrix(model.mesh, material)
        np.testing.assert_array_equal(recovered, baseline)
    finally:
        setattr(helper, attribute, original)
    recovered = element.compute_stiffness_matrix(model.mesh, material)
    np.testing.assert_array_equal(recovered, baseline)


@pytest.mark.parametrize(
    "helper_name",
    (
        "_require_q4_fast_array_authority",
        "_require_q4_fast_base_authority",
        "_q4_fast_input_snapshot_matches",
    ),
)
def test_direct_q4_ignores_exposed_transitive_warm_helper_code_replacement(
    helper_name: str,
) -> None:
    model, element = _model()
    material = model.materials["steel"]
    baseline = element.compute_stiffness_matrix(model.mesh, material)
    helper = getattr(e4_pl_module, helper_name)
    original = helper.__code__

    def forged(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("exposed helper reached warm mechanics")

    assert forged.__code__.co_freevars == original.co_freevars
    helper.__code__ = forged.__code__
    try:
        recovered = element.compute_stiffness_matrix(model.mesh, material)
        np.testing.assert_array_equal(recovered, baseline)
    finally:
        helper.__code__ = original
    recovered = element.compute_stiffness_matrix(model.mesh, material)
    np.testing.assert_array_equal(recovered, baseline)
