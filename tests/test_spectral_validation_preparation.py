from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

import anyfesolver.current_state_tangent as tangent
from anyfesolver import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import ShellElement, create_shell_element


def _q4(element_id: int) -> QualifiedE4PLShellElement:
    return QualifiedE4PLShellElement(
        element_id,
        (1, 2, 3, 4),
        "steel",
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )


def _model(*elements: QualifiedE4PLShellElement) -> SimpleNamespace:
    return SimpleNamespace(
        mesh=SimpleNamespace(
            elements={element.element_id: element for element in elements}
        )
    )


def _delete_material_name(element: QualifiedE4PLShellElement) -> None:
    object.__delattr__(element, "material_name")


def _replace_connectivity(element: QualifiedE4PLShellElement) -> None:
    object.__setattr__(element, "node_ids", [1, 2, 3, 4])


def _shadow_formulation(element: QualifiedE4PLShellElement) -> None:
    object.__setattr__(
        element,
        "formulation_id",
        tangent.QUALIFIED_Q4_FORMULATION_ID,
    )


def _add_callable_override(element: QualifiedE4PLShellElement) -> None:
    object.__setattr__(element, "attacker_callback", lambda: None)


def test_captured_class_names_are_exact_for_every_immutable_profile() -> None:
    for formulation_id, profile in tangent._QUALIFIED_PROFILES.items():
        expected = frozenset().union(
            *(
                frozenset(namespace)
                for namespace in profile["class_namespace_authority"].values()
            )
        )
        assert (
            tangent._QUALIFIED_PROFILE_CAPTURED_CLASS_NAMES[formulation_id]
            == expected
        )


def test_live_namespace_lookup_observes_mutation_deletion_and_mro_change() -> None:
    class First:
        member = object()

    class Second:
        member = object()

    class Child(First):
        pass

    lookup = tangent._call_local_static_lookup()
    assert lookup(Child, "member") is First.member
    First.member = object()
    assert lookup(Child, "member") is First.member
    Child.member = object()
    assert lookup(Child, "member") is Child.member
    del Child.member
    assert lookup(Child, "member") is First.member
    Child.__bases__ = (Second,)
    assert lookup(Child, "member") is Second.member
    del Second.member
    assert lookup(Child, "member") is None


def test_live_namespace_lookup_does_not_invoke_descriptors() -> None:
    class Subject:
        @property
        def member(self):
            raise AssertionError("descriptor evaluated")

        @staticmethod
        def static():
            raise AssertionError("static method called")

        @classmethod
        def class_method(cls):
            raise AssertionError("class method called")

    lookup = tangent._call_local_static_lookup()
    assert lookup(Subject, "member") is vars(Subject)["member"]
    for name in ("static", "class_method"):
        assert lookup(Subject, name) is vars(Subject)[name].__func__


@pytest.mark.parametrize("mutate", [None, _delete_material_name, _replace_connectivity,
                                  _shadow_formulation, _add_callable_override])
def test_live_lookup_and_serializer_match_original_predicate(mutate) -> None:
    element = _q4(1)
    if mutate is not None:
        mutate(element)
    profile = tangent._QUALIFIED_PROFILES[tangent.QUALIFIED_Q4_FORMULATION_ID]
    lookup = tangent._call_local_static_lookup()
    assert tangent._qualified_profile_api_failure(
        element, profile, _static_lookup=lookup, _serialization_static_lookup=lookup,
    ) == tangent._qualified_profile_api_failure(element, profile)


@pytest.mark.parametrize(
    "mutate",
    (
        None,
        _delete_material_name,
        _replace_connectivity,
        _shadow_formulation,
        _add_callable_override,
    ),
    ids=(
        "clean",
        "missing-data",
        "bad-connectivity",
        "identity-shadow",
        "callable-shadow",
    ),
)
def test_prepared_class_names_preserve_standalone_instance_predicate(
    mutate: Callable[[QualifiedE4PLShellElement], None] | None,
) -> None:
    profile = tangent._QUALIFIED_PROFILES[
        tangent.QUALIFIED_Q4_FORMULATION_ID
    ]
    element = _q4(1)
    if mutate is not None:
        mutate(element)

    standalone = tangent._qualified_profile_api_failure(element, profile)
    prepared = tangent._qualified_profile_api_failure(
        element,
        profile,
        _captured_class_names=(
            tangent._QUALIFIED_PROFILE_CAPTURED_CLASS_NAMES[
                tangent.QUALIFIED_Q4_FORMULATION_ID
            ]
        ),
    )

    assert prepared == standalone


def test_prepared_class_names_preserve_standalone_class_mutation_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tangent._QUALIFIED_PROFILES[
        tangent.QUALIFIED_Q4_FORMULATION_ID
    ]
    element = _q4(1)
    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "compute_mass_matrix",
        lambda self, mesh, material: None,
    )

    standalone = tangent._qualified_profile_api_failure(element, profile)
    prepared = tangent._qualified_profile_api_failure(
        element,
        profile,
        _captured_class_names=(
            tangent._QUALIFIED_PROFILE_CAPTURED_CLASS_NAMES[
                tangent.QUALIFIED_Q4_FORMULATION_ID
            ]
        ),
    )

    assert prepared == standalone
    assert standalone is not None
    assert "CRITICAL_API_MISMATCH=compute_mass_matrix" in standalone


def test_guard_rechecks_class_authority_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(_q4(1), _q4(2))
    assert tangent.require_exact_qualified_component_lifecycle_api(
        model,
        context="first validation",
    ) == {"qualified_element_ids": [1, 2], "guarded": True}

    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "compute_mass_matrix",
        lambda self, mesh, material: None,
    )

    with pytest.raises(
        ElementCapabilityError,
        match="CLASS_NAMESPACE_MISMATCH.*compute_mass_matrix",
    ):
        tangent.require_exact_qualified_component_lifecycle_api(
            model,
            context="class mutation revalidation",
        )


def test_guard_validates_elements_added_after_an_earlier_success() -> None:
    model = _model(_q4(1))
    assert tangent.require_exact_qualified_component_lifecycle_api(
        model,
        context="initial element",
    )["qualified_element_ids"] == [1]

    model.mesh.elements[2] = _q4(2)
    assert tangent.require_exact_qualified_component_lifecycle_api(
        model,
        context="late valid element",
    )["qualified_element_ids"] == [1, 2]

    object.__setattr__(model.mesh.elements[2], "element_id", 99)
    with pytest.raises(
        ElementCapabilityError,
        match="2 .*ELEMENT_MAPPING_ID_MISMATCH",
    ):
        tangent.require_exact_qualified_component_lifecycle_api(
            model,
            context="late mutated element",
        )


def test_guard_rejects_late_exact_class_substitution() -> None:
    class Q4Subclass(QualifiedE4PLShellElement):
        pass

    model = _model(_q4(1))
    substituted = _q4(2)
    object.__setattr__(substituted, "__class__", Q4Subclass)
    model.mesh.elements[2] = substituted

    with pytest.raises(
        ElementCapabilityError,
        match="2 .*FORMULATION_ID_CLASS_MISMATCH",
    ):
        tangent.require_exact_qualified_component_lifecycle_api(
            model,
            context="late subclass",
        )


def test_custom_profile_failure_callback_keeps_per_element_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(_q4(1), _q4(2))
    calls: list[int] = []

    def mutating_failure(element: Any, profile: Any) -> str | None:
        calls.append(element.element_id)
        failure = tangent._qualified_profile_api_failure(element, profile)
        if element.element_id == 1:
            monkeypatch.setattr(
                QualifiedE4PLShellElement,
                "compute_mass_matrix",
                lambda self, mesh, material: None,
            )
        return failure

    with pytest.raises(
        ElementCapabilityError,
        match="2 .*CRITICAL_API_MISMATCH=compute_mass_matrix",
    ):
        tangent._require_exact_qualified_component_lifecycle_api_implementation(
            model,
            context="callback boundary",
            _profile_failure=mutating_failure,
        )

    assert calls == [1, 2]


# ---------------------------------------------------------------------------
# per-call class snapshot (class-level facts read once per validation call)
# ---------------------------------------------------------------------------
def _shell(kind: str, element_id: int = 1) -> Any:
    if kind == "s3":
        return QualifiedE4PLS3ShellElement(
            element_id,
            (1, 2, 3),
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        )
    formulation, nodes = {
        "q4": ("e4-pl", (1, 2, 3, 4)),
        "v2d": ("e4-pl-s3-v2d", (1, 2, 3)),
    }[kind]
    return create_shell_element(
        element_id,
        nodes,
        "steel",
        formulation=formulation,
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )


def _profile_of(element: Any) -> Any:
    return next(
        profile
        for profile in tangent._QUALIFIED_PROFILES.values()
        if profile["element_type"] is type(element)
    )


def _fresh_snapshot() -> Any:
    return tangent._ScanClassSnapshot(tangent._call_local_static_lookup())


def _guard_style_failure(element: Any, profile: Any, snapshot: Any) -> Any:
    """The profile check exactly as the lifecycle guard calls it by default."""

    return tangent._qualified_profile_api_failure(
        element,
        profile,
        _captured_class_names=(
            tangent._QUALIFIED_PROFILE_CAPTURED_CLASS_NAMES[
                str(profile["formulation_id"])
            ]
        ),
        _static_lookup=snapshot.lookup,
        _serialization_static_lookup=(
            snapshot.lookup
            if profile["family"] in {"qualified_q4", "qualified_s3"}
            else None
        ),
        _snapshot=snapshot,
    )


def _critical_name(profile: Any) -> str:
    names = profile["critical_apis"]
    return (
        "compute_stiffness_matrix"
        if "compute_stiffness_matrix" in names
        else next(iter(names))
    )


def _mutation_missing_data(element, profile, monkeypatch):
    object.__delattr__(element, "material_name")


def _mutation_bad_connectivity(element, profile, monkeypatch):
    object.__setattr__(element, "node_ids", list(element.node_ids))


def _mutation_identity_shadow(element, profile, monkeypatch):
    object.__setattr__(element, "formulation_id", profile["formulation_id"])


def _mutation_callable_shadow(element, profile, monkeypatch):
    object.__setattr__(element, "attacker_callback", lambda: None)


def _mutation_critical_shadow(element, profile, monkeypatch):
    object.__setattr__(
        element, _critical_name(profile), lambda *args, **kwargs: None
    )


def _mutation_offset_instance(element, profile, monkeypatch):
    bad = 0.5 if profile["family"] == "qualified_q4" else float("nan")
    object.__setattr__(element, "reference_surface_offset", bad)


def _mutation_class_critical(element, profile, monkeypatch):
    monkeypatch.setattr(
        type(element),
        _critical_name(profile),
        lambda *args, **kwargs: None,
    )


def _mutation_class_identity(element, profile, monkeypatch):
    name = next(iter(profile["class_identity"]))
    monkeypatch.setattr(type(element), name, "tampered")


def _mutation_base_critical(element, profile, monkeypatch):
    names = profile["base_critical_apis"]
    if not names:
        pytest.skip("this profile has no base critical APIs")
    monkeypatch.setattr(
        ShellElement, next(iter(names)), lambda *args, **kwargs: None
    )


def _mutation_class_substitution(element, profile, monkeypatch):
    substitute = type("Substitute", (type(element),), {})
    object.__setattr__(element, "__class__", substitute)


def _mutation_class_data_shadow(element, profile, monkeypatch):
    monkeypatch.setattr(type(element), "element_id", 0, raising=False)


def _mutation_class_offset_shadow(element, profile, monkeypatch):
    monkeypatch.setattr(
        type(element), "reference_surface_offset", 0.0, raising=False
    )


_SNAPSHOT_MUTATIONS = {
    "clean": None,
    "missing-data": _mutation_missing_data,
    "bad-connectivity": _mutation_bad_connectivity,
    "identity-shadow": _mutation_identity_shadow,
    "callable-shadow": _mutation_callable_shadow,
    "critical-shadow": _mutation_critical_shadow,
    "offset-instance": _mutation_offset_instance,
    "class-critical": _mutation_class_critical,
    "class-identity": _mutation_class_identity,
    "base-critical": _mutation_base_critical,
    "class-substitution": _mutation_class_substitution,
    "class-data-shadow": _mutation_class_data_shadow,
    "class-offset-shadow": _mutation_class_offset_shadow,
}


@pytest.mark.parametrize("name", tuple(_SNAPSHOT_MUTATIONS))
@pytest.mark.parametrize("kind", ("q4", "s3", "v2d"))
def test_snapshot_failure_matches_the_standalone_predicate(
    kind: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    element = _shell(kind)
    profile = _profile_of(element)
    mutate = _SNAPSHOT_MUTATIONS[name]
    if mutate is not None:
        mutate(element, profile, monkeypatch)

    standalone = tangent._qualified_profile_api_failure(element, profile)
    snapshotted = _guard_style_failure(element, profile, _fresh_snapshot())

    assert snapshotted == standalone
    assert (standalone is None) == (name == "clean")


def test_snapshot_critical_api_comparison_is_read_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile_of(_shell("q4"))
    snapshot = _fresh_snapshot()
    assert snapshot.changed_critical(QualifiedE4PLShellElement, profile) == ()

    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        _critical_name(profile),
        lambda *args, **kwargs: None,
    )

    # The snapshot keeps what it read; a new call reads the class again.
    assert snapshot.changed_critical(QualifiedE4PLShellElement, profile) == ()
    fresh = _fresh_snapshot().changed_critical(
        QualifiedE4PLShellElement, profile
    )
    assert fresh == (_critical_name(profile),)


def test_snapshot_facts_are_kept_per_class() -> None:
    profile = _profile_of(_shell("q4"))
    name = _critical_name(profile)
    snapshot = _fresh_snapshot()
    assert snapshot.changed_critical(QualifiedE4PLShellElement, profile) == ()

    tampered = type(
        "Tampered",
        (QualifiedE4PLShellElement,),
        {name: lambda *args, **kwargs: None},
    )

    assert snapshot.changed_critical(tampered, profile) == (name,)
    assert snapshot.lookup(tampered, name) is not snapshot.lookup(
        QualifiedE4PLShellElement, name
    )


@pytest.mark.parametrize("memo", ("lookup", "changed_critical"))
def test_snapshot_memos_keep_the_keyed_class_alive(memo: str) -> None:
    """An ``id()`` key is safe only while the keyed class cannot be collected."""

    import gc
    import weakref

    # A stub lookup and a profile without critical APIs (no lookups at all)
    # pin nothing, so only the memo entry itself can hold the class.
    snapshot = tangent._ScanClassSnapshot(lambda owner, name: None)
    owner = type("Transient", (QualifiedE4PLShellElement,), {})
    if memo == "lookup":
        snapshot.lookup(owner, "formulation_id")
    else:
        snapshot.changed_critical(owner, {"critical_apis": {}})

    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is not None


@pytest.mark.parametrize("kind", ("q4", "s3"))
def test_change_to_the_class_itself_is_still_reported_by_the_remaining_elements(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Q4 and S3 validators re-read their own class authority per element."""

    profile = _profile_of(_shell(kind))
    first, second = _shell(kind, 1), _shell(kind, 2)
    snapshot = _fresh_snapshot()
    assert _guard_style_failure(first, profile, snapshot) is None

    monkeypatch.setattr(
        type(first), _critical_name(profile), lambda *args, **kwargs: None
    )

    failure = _guard_style_failure(second, profile, snapshot)
    assert failure is not None and "class authority" in failure


@pytest.mark.parametrize("kind", ("q4", "s3"))
def test_base_class_change_is_reported_by_the_next_call(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented relaxation: base-class facts are read once per call."""

    profile = _profile_of(_shell(kind))
    name = next(iter(profile["base_critical_apis"]))
    first, second = _shell(kind, 1), _shell(kind, 2)
    snapshot = _fresh_snapshot()
    assert _guard_style_failure(first, profile, snapshot) is None

    monkeypatch.setattr(ShellElement, name, lambda *args, **kwargs: None)

    assert _guard_style_failure(second, profile, snapshot) is None
    failure = _guard_style_failure(second, profile, _fresh_snapshot())
    assert failure is not None and "BASE_CRITICAL_API_MISMATCH" in failure


def test_class_change_made_while_a_v2d_element_is_checked_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Class facts of S3 V2D elements are read after the instance-level work.

    A key that collides with a class attribute name makes the guard call its
    ``__eq__`` while it intersects the instance namespace with the class
    names.  The class change made there must still be seen by the same
    element's check, not only by a later element or the next call.
    """

    elements = [_shell("v2d", element_id) for element_id in (1, 2, 3)]
    owner = type(elements[0])
    name = _critical_name(_profile_of(elements[0]))
    armed: list[bool] = []
    planted: list[int] = []

    class CollidingKey(str):
        def __hash__(self) -> int:
            return hash("formulation_id")

        def __eq__(self, other: object) -> bool:
            # Armed only for the guard: instance dicts share a key table per
            # class, so an unarmed comparison could run while inserting.
            if armed and not planted:
                planted.append(1)
                monkeypatch.setattr(owner, name, lambda *args, **kwargs: None)
            return False

        def __ne__(self, other: object) -> bool:
            return True

    elements[-1].__dict__[CollidingKey("marker")] = 1
    armed.append(True)

    with pytest.raises(ElementCapabilityError, match=r"3 \(.*CRITICAL_API_MISMATCH"):
        tangent.require_exact_qualified_component_lifecycle_api(
            _model(*elements), context="colliding key"
        )
    assert planted == [1]


def test_v2d_elements_never_take_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families: list[str] = []
    real = tangent._ScanClassSnapshot.changed_critical

    def recording(self: Any, owner: type, profile: Any) -> Any:
        families.append(str(profile["family"]))
        return real(self, owner, profile)

    monkeypatch.setattr(
        tangent._ScanClassSnapshot, "changed_critical", recording
    )
    model = _model(_q4(1), _shell("v2d", 2), _shell("v2d", 3), _shell("s3", 4))
    tangent.require_exact_qualified_component_lifecycle_api(
        model, context="mixed families"
    )

    assert set(families) == {"qualified_q4", "qualified_s3"}


@pytest.mark.parametrize("kind", ("q4", "s3", "v2d"))
def test_snapshot_keeps_the_instance_checks_per_element(kind: str) -> None:
    profile = _profile_of(_shell(kind))
    first, second = _shell(kind, 1), _shell(kind, 2)
    snapshot = _fresh_snapshot()
    assert _guard_style_failure(first, profile, snapshot) is None

    object.__setattr__(second, "attacker_callback", lambda: None)

    failure = _guard_style_failure(second, profile, snapshot)
    assert failure is not None and "CALLABLE_INSTANCE_OVERRIDE" in failure


def test_class_level_lookups_do_not_grow_with_the_element_count() -> None:
    def underlying_lookups(count: int) -> int:
        calls: list[tuple[int, str]] = []

        def factory() -> Any:
            real = tangent._call_local_static_lookup()

            def counting(owner: type, name: str) -> Any:
                calls.append((id(owner), name))
                return real(owner, name)

            return counting

        model = _model(*(_q4(index) for index in range(1, count + 1)))
        tangent._require_exact_qualified_component_lifecycle_api_implementation(
            model, context="counted", _lookup_factory=factory
        )
        return len(calls)

    assert underlying_lookups(3) == underlying_lookups(30)


def test_serialization_module_guard_runs_once_per_validation_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tangent._QUALIFIED_PROFILES[tangent.QUALIFIED_Q4_FORMULATION_ID]
    validator = profile["serialization_validator"]
    real = tangent._SERIALIZATION_MODULE_GUARDS[validator]
    calls: list[int] = []

    def counting(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        real(*args, **kwargs)

    monkeypatch.setattr(
        tangent, "_SERIALIZATION_MODULE_GUARDS", {validator: counting}
    )
    model = _model(_q4(1), _q4(2), _q4(3))
    tangent.require_exact_qualified_component_lifecycle_api(
        model, context="first call"
    )
    assert len(calls) == 1
    tangent.require_exact_qualified_component_lifecycle_api(
        model, context="second call"
    )
    assert len(calls) == 2


def test_failing_serialization_module_guard_is_reported_for_every_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tangent._QUALIFIED_PROFILES[tangent.QUALIFIED_Q4_FORMULATION_ID]
    validator = profile["serialization_validator"]
    attempts: list[int] = []

    def failing(*args: Any, **kwargs: Any) -> None:
        attempts.append(1)
        raise ValueError("module authority is incompatible")

    monkeypatch.setattr(
        tangent, "_SERIALIZATION_MODULE_GUARDS", {validator: failing}
    )
    model = _model(_q4(1), _q4(2), _q4(3))

    with pytest.raises(ElementCapabilityError) as caught:
        tangent.require_exact_qualified_component_lifecycle_api(
            model, context="failing module guard"
        )

    message = str(caught.value)
    for element_id in (1, 2, 3):
        assert f"{element_id} (" in message
    assert "CONFIGURATION_AUTHORITY=module authority is incompatible" in message
    assert len(attempts) == 3  # a failed guard is never marked done


def test_custom_profile_failure_callbacks_never_receive_a_snapshot() -> None:
    model = _model(_q4(1), _q4(2))
    received: list[list[str]] = []

    def custom(element: Any, profile: Any, **kwargs: Any) -> str | None:
        received.append(sorted(kwargs))
        return None

    tangent._require_exact_qualified_component_lifecycle_api_implementation(
        model,
        context="custom callback",
        _profile_failure=custom,
        _preparable_profile_failure=custom,
    )

    assert len(received) == 2
    assert all("_snapshot" not in keywords for keywords in received)
