"""Constant-time Newton-loop guards for owned S3 V2D elements and tokens.

The trusted path must be result-neutral and must keep failing closed for every
supported mutation surface that the complete lifecycle scan observes.  Its own
inputs (the cancellation token, the V2D write counters, the disk-cache default)
are covered here as well.
"""

from __future__ import annotations

import gc
import sys
import threading

import numpy as np
import pytest

import anyfesolver.current_state_tangent as current_state_tangent
import anyfesolver.e4_pl_s3_v2d_element as v2d_module
import anyfesolver.jit_compiler as jit_compiler
import anyfesolver.nonlinear_static as nonlinear_static_module
from anyfesolver import elements as elements_module
from anyfesolver import plasticity, vectorized_nonlinear
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.control import CancellationToken
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_v2d_element import (
    NativeParityCapabilityError,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.jit_compiler import JIT_ENABLED
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver.nonlinear_static import solve_static_nonlinear


NEWTON_LOOP_CONTEXTS = (
    "nonlinear static force tangent assembly",
    "nonlinear static force external load",
    "nonlinear static force iteration cancellation",
    "nonlinear static step external load",
)
CAUGHT = (AssemblyError, ElementCapabilityError, NativeParityCapabilityError)
_FIXED = {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}


def _mixed_model() -> tuple[FEModel, LoadCase]:
    """One qualified Q4 beside one exact S3 V2D triangle."""

    model = FEModel("mixed-q4-v2d")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    quad = create_shell_element(
        1, (1, 2, 3, 4), "steel", formulation="e4-pl", thickness=0.01
    )
    triangle = create_shell_element(
        2,
        (2, 5, 3),
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=0.01,
        reference_normal=(0.0, 0.0, 1.0),
    )
    assert type(quad) is QualifiedE4PLShellElement
    assert type(triangle) is NativeParityE4PLS3V2DShellElement
    model.add_element(1, quad)
    model.add_element(2, triangle)
    model.add_boundary_condition(BoundaryCondition("fixed", [1, 4], _FIXED))
    load = LoadCase("tip")
    load.add_nodal_load(5, [0.0, 0.0, 2.0e2, 0.0, 0.0, 0.0])
    return model, load


def _two_v2d_model() -> tuple[FEModel, LoadCase]:
    """One qualified Q4 beside two exact S3 V2D triangles."""

    model = FEModel("mixed-q4-two-v2d")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
            (2.0, 1.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        create_shell_element(
            1, (1, 2, 3, 4), "steel", formulation="e4-pl", thickness=0.01
        ),
    )
    for element_id, nodes in ((2, (2, 5, 3)), (3, (3, 5, 6))):
        model.add_element(
            element_id,
            create_shell_element(
                element_id,
                nodes,
                "steel",
                formulation="e4-pl-s3-v2d",
                thickness=0.01,
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
    model.add_boundary_condition(BoundaryCondition("fixed", [1, 4], _FIXED))
    load = LoadCase("tip")
    load.add_nodal_load(6, [0.0, 0.0, 2.0e2, 0.0, 0.0, 0.0])
    return model, load


def _two_q4_model() -> tuple[FEModel, LoadCase]:
    """Two qualified Q4 elements and no V2D element."""

    model = FEModel("two-q4")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    for element_id, nodes in ((1, (1, 2, 3, 4)), (2, (2, 5, 6, 3))):
        model.add_element(
            element_id,
            create_shell_element(
                element_id, nodes, "steel", formulation="e4-pl", thickness=0.01
            ),
        )
    model.add_boundary_condition(BoundaryCondition("fixed", [1, 4], _FIXED))
    load = LoadCase("tip")
    load.add_nodal_load(6, [0.0, 0.0, 2.0e1, 0.0, 0.0, 0.0])
    return model, load


def _solve(model: FEModel, load: LoadCase, **options: object):
    return solve_static_nonlinear(
        model, load, max_load_factor=1.0, num_steps=2, num_layers=5, **options
    )


def _events(result: object) -> dict[str, int]:
    return dict(
        result.info["nonlinear_performance"]["solver"]["event_counts"]  # type: ignore[attr-defined]
    )


def _count_complete_scans(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    original = nonlinear_static_module._EXACT_QUALIFIED_LIFECYCLE_GUARD
    contexts: list[str] = []

    def counted_guard(observed_model: object, *, context: str) -> object:
        contexts.append(context)
        return original(observed_model, context=context)

    monkeypatch.setattr(
        nonlinear_static_module,
        "_EXACT_QUALIFIED_LIFECYCLE_GUARD",
        counted_guard,
    )
    return contexts


def _v2d_elements(model: FEModel) -> list[NativeParityE4PLS3V2DShellElement]:
    return [
        element
        for element in model.mesh.elements.values()
        if type(element) is NativeParityE4PLS3V2DShellElement
    ]


def _capture(model: FEModel):
    return current_state_tangent._capture_qualified_v2d_trusted_authority(
        tuple(model.mesh.elements.items()), context="test"
    )


def _inject_in_trusted_window(
    monkeypatch: pytest.MonkeyPatch, action: object, *, call: int = 2
) -> list[int]:
    """Run *action* inside the second assembly call of the Newton loop.

    That window is followed only by the constant-time trusted guard, unlike a
    status callback, which is always followed by a complete scan.
    """

    calls: list[int] = []
    original = nonlinear_static_module._assemble_nonlinear_system

    def wrapped(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == call:
            action()  # type: ignore[operator]
        return original(*args, **kwargs)

    monkeypatch.setattr(nonlinear_static_module, "_assemble_nonlinear_system", wrapped)
    return calls


# ---------------------------------------------------------------------------
# result neutrality and the trusted loop
# ---------------------------------------------------------------------------
def test_mixed_q4_v2d_newton_loop_uses_trusted_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _count_complete_scans(monkeypatch)
    result = _solve(*_mixed_model())

    assert result.status == "completed"
    assert "solve_static_nonlinear preflight" in contexts
    assert "nonlinear static constraint postcheck" in contexts
    for context in NEWTON_LOOP_CONTEXTS:
        assert context not in contexts


def test_two_v2d_newton_loop_uses_trusted_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _count_complete_scans(monkeypatch)
    result = _solve(*_two_v2d_model())

    assert result.status == "completed"
    for context in NEWTON_LOOP_CONTEXTS:
        assert context not in contexts


def test_trusted_v2d_path_is_bitwise_result_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = _solve(*_mixed_model())
    monkeypatch.setattr(
        nonlinear_static_module,
        "_CAPTURE_QUALIFIED_V2D_TRUSTED_AUTHORITY",
        lambda *_args, **_kwargs: None,
    )
    contexts = _count_complete_scans(monkeypatch)
    complete = _solve(*_mixed_model())

    assert "nonlinear static force tangent assembly" in contexts
    assert trusted.status == complete.status == "completed"
    assert np.asarray(trusted.displacements).tobytes() == np.asarray(
        complete.displacements
    ).tobytes()
    assert trusted.load_factor == complete.load_factor
    assert len(trusted.steps) == len(complete.steps)


def test_solver_cancellation_token_keeps_the_trusted_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _count_complete_scans(monkeypatch)
    result = _solve(*_mixed_model(), cancellation_token=CancellationToken())

    assert result.status == "completed"
    assert "nonlinear static force iteration cancellation" not in contexts
    assert "nonlinear static force increment cancellation" not in contexts


# ---------------------------------------------------------------------------
# reporting: a solve that cannot use the constant-time path says so
# ---------------------------------------------------------------------------
class _ForeignToken:
    def raise_if_cancelled(self, stage: str = "") -> None:
        del stage


def test_trusted_solve_reports_its_guard_path() -> None:
    events = _events(_solve(*_mixed_model(), cancellation_token=CancellationToken()))

    assert events["qualified_guard_trusted"] > 0
    assert "qualified_v2d_trusted_unavailable" not in events
    assert "qualified_guard_untrusted_token" not in events


def test_ineligible_v2d_solve_reports_the_complete_scan_fallback() -> None:
    model, load = _mixed_model()
    object.__setattr__(_v2d_elements(model)[0], "writable_marker", np.zeros(3))
    result = _solve(model, load)
    events = _events(result)

    assert result.status == "completed"
    assert events["qualified_v2d_trusted_unavailable"] == 1
    assert events["qualified_guard_complete_scan"] > 0
    assert "qualified_guard_trusted" not in events


def test_untrusted_token_solve_reports_the_fallback() -> None:
    events = _events(_solve(*_mixed_model(), cancellation_token=_ForeignToken()))

    assert events["qualified_guard_untrusted_token"] > 0


# ---------------------------------------------------------------------------
# cancellation token trust
# ---------------------------------------------------------------------------
class _SubclassedToken(CancellationToken):
    pass


def _token_with_instance_override() -> CancellationToken:
    token = CancellationToken()
    token.raise_if_cancelled = lambda stage="": None  # type: ignore[method-assign]
    return token


def _token_with_event_override() -> CancellationToken:
    token = CancellationToken()
    token._event.is_set = lambda: False  # type: ignore[method-assign]
    return token


class _HostileFlag:
    def __bool__(self) -> bool:
        return False


def _token_with_hostile_flag() -> CancellationToken:
    token = CancellationToken()
    token._event._flag = _HostileFlag()  # type: ignore[assignment]
    return token


@pytest.mark.parametrize(
    "make_token",
    (
        _ForeignToken,
        _SubclassedToken,
        _token_with_instance_override,
        _token_with_event_override,
        _token_with_hostile_flag,
    ),
    ids=(
        "foreign",
        "subclass",
        "instance-override",
        "event-override",
        "hostile-flag",
    ),
)
def test_untrusted_tokens_keep_the_complete_scan(
    monkeypatch: pytest.MonkeyPatch,
    make_token: object,
) -> None:
    token = make_token()  # type: ignore[operator]
    assert not nonlinear_static_module._is_known_cancellation_checkpoint(token)
    contexts = _count_complete_scans(monkeypatch)
    _solve(*_mixed_model(), cancellation_token=token)

    assert "nonlinear static force iteration cancellation" in contexts


def test_modified_token_class_keeps_the_complete_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = CancellationToken()
    monkeypatch.setattr(
        CancellationToken, "raise_if_cancelled", lambda self, stage="": None
    )
    assert not nonlinear_static_module._is_known_cancellation_checkpoint(token)
    # Without a token the checkpoint runs no token code at all.
    assert nonlinear_static_module._is_known_cancellation_checkpoint(None)


def test_replaced_checkpoint_keeps_the_complete_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = nonlinear_static_module.cancellation_safe_point
    monkeypatch.setattr(
        nonlinear_static_module,
        "cancellation_safe_point",
        lambda token, stage="": original(token, stage),
    )
    assert not nonlinear_static_module._is_known_cancellation_checkpoint(None)
    contexts = _count_complete_scans(monkeypatch)
    _solve(*_mixed_model())

    assert "nonlinear static force iteration cancellation" in contexts


class _SelfHealingFlag:
    """An event flag whose truth test runs hostile code, then repairs itself.

    The token is a plain exact ``CancellationToken`` again by the time the
    solver looks at it after the checkpoint, so only a check made *before* the
    checkpoint can see that the checkpoint ran foreign code.
    """

    def __init__(self, token: CancellationToken, plant: object) -> None:
        self.token = token
        self.plant = plant
        self.calls = 0

    def __bool__(self) -> bool:
        self.calls += 1
        if self.calls == 2:  # the first checkpoint that only the token guards
            self.plant()  # type: ignore[operator]
            self.token._event._flag = False
        return False


def _plant_override(element: object, hits: list[int]):
    def hostile(*_args: object, **_kwargs: object) -> object:
        element.__dict__.pop("compute_nonlinear_response", None)  # type: ignore[attr-defined]
        hits.append(1)
        raise AssertionError("the planted override must never run")

    def plant() -> None:
        # ``object.__setattr__`` is outside every counted mutation surface, so
        # only the complete lifecycle scan can see the override.
        object.__setattr__(element, "compute_nonlinear_response", hostile)

    return plant


def test_self_healing_token_is_rejected_at_its_checkpoint() -> None:
    model, load = _two_v2d_model()
    hits: list[int] = []
    token = CancellationToken()
    token._event._flag = _SelfHealingFlag(
        token, _plant_override(_v2d_elements(model)[1], hits)
    )

    with pytest.raises(ElementCapabilityError, match="force step cancellation"):
        _solve(model, load, cancellation_token=token)
    assert hits == []


def test_self_healing_token_is_rejected_for_q4_only_models() -> None:
    model, load = _two_q4_model()
    hits: list[int] = []
    token = CancellationToken()
    token._event._flag = _SelfHealingFlag(
        token, _plant_override(model.mesh.elements[2], hits)
    )

    with pytest.raises(ElementCapabilityError, match="force step cancellation"):
        _solve(model, load, cancellation_token=token)
    assert hits == []


# ---------------------------------------------------------------------------
# V2D write counters
# ---------------------------------------------------------------------------
def test_v2d_writes_do_not_touch_the_mesh_epoch() -> None:
    model, _load = _mixed_model()
    element = _v2d_elements(model)[0]
    token = model.mesh._qualified_direct_state_token
    start = int(token[0])

    element.thickness = element.thickness
    element.material_name = element.material_name
    element.extra_marker = 1
    del element.extra_marker
    element._stiffness_matrix = None

    assert int(token[0]) == start


def test_v2d_writes_are_not_counted_without_a_live_capture() -> None:
    element = _v2d_elements(_mixed_model()[0])[0]

    element.thickness = element.thickness
    assert v2d_module._V2D_LEASE_EPOCHS_KEY not in element.__dict__


def test_v2d_capture_counts_ordinary_writes_and_deletions() -> None:
    model, _load = _mixed_model()
    element = _v2d_elements(model)[0]
    require = _capture(model)
    assert require is not None
    require(context="clean")

    # Solver-owned derived-cache resets are not writes to an authority fact.
    element._stiffness_matrix = None
    element._mass_matrix = np.zeros((18, 18))
    require(context="derived cache reset")

    element.thickness = element.thickness
    with pytest.raises(ElementCapabilityError, match="modified during the trusted"):
        require(context="thickness write")
    # The count only ever grows, so the capture stays failed.
    with pytest.raises(ElementCapabilityError, match="modified during the trusted"):
        require(context="still failed")


@pytest.mark.parametrize(
    "mutate",
    (
        lambda element: setattr(element, "thickness", 0.02),
        lambda element: setattr(element, "material_name", "steel"),
        lambda element: setattr(element, "node_ids", tuple(element.node_ids)),
        lambda element: setattr(element, "extra_marker", 1),
        lambda element: setattr(element, "compute_nonlinear_response", len),
        lambda element: setattr(element, "_nl_cache", lambda: None),
        lambda element: setattr(element, "_stiffness_matrix", "not an array"),
        lambda element: delattr(element, "material_direction"),
    ),
    ids=(
        "thickness",
        "material-name",
        "node-ids",
        "new-attribute",
        "method-override",
        "callable-cache",
        "non-array-cache",
        "delete-attribute",
    ),
)
def test_v2d_capture_counts_every_kind_of_write(mutate: object) -> None:
    model, _load = _mixed_model()
    element = _v2d_elements(model)[0]
    require = _capture(model)
    assert require is not None

    mutate(element)  # type: ignore[operator]

    with pytest.raises(ElementCapabilityError):
        require(context="after write")


def test_v2d_capture_covers_elements_other_than_its_representative() -> None:
    model, _load = _two_v2d_model()
    require = _capture(model)
    assert require is not None

    _v2d_elements(model)[1].thickness = 0.02

    with pytest.raises(ElementCapabilityError, match="modified during the trusted"):
        require(context="second element")


def test_v2d_capture_subscription_ends_with_the_capture() -> None:
    model, _load = _mixed_model()
    element = _v2d_elements(model)[0]
    key = v2d_module._V2D_LEASE_EPOCHS_KEY

    first = _capture(model)
    assert first is not None
    del first
    gc.collect()
    element.thickness = element.thickness  # nobody is listening any more
    second = _capture(model)
    assert second is not None
    second(context="fresh capture")

    references = element.__dict__[key]
    assert len(references) == 1 and references[0]() is not None


def test_v2d_write_counts_are_atomic_under_contention() -> None:
    model, _load = _two_v2d_model()
    require = _capture(model)
    assert require is not None
    elements = _v2d_elements(model)
    key = v2d_module._V2D_LEASE_EPOCHS_KEY
    epoch = elements[0].__dict__[key][0]()
    writes_per_thread, errors = 1500, []

    def hammer(element: object) -> None:
        try:
            for _ in range(writes_per_thread):
                element.extra_marker = 1  # type: ignore[attr-defined]
        except Exception as error:  # pragma: no cover - failure path
            errors.append(error)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0e-6)
    try:
        threads = [
            threading.Thread(target=hammer, args=(elements[index % 2],))
            for index in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(interval)

    assert errors == []
    assert epoch.value == 4 * writes_per_thread
    with pytest.raises(ElementCapabilityError, match="modified during the trusted"):
        require(context="after contention")


def test_v2d_capture_rejects_formulation_and_class_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _load = _mixed_model()
    require = _capture(model)
    assert require is not None
    require(context="clean")

    # A replaced module helper is formulation-level authority.
    original_helper = v2d_module._advance_v2d_lease_epochs
    monkeypatch.setattr(
        v2d_module,
        "_advance_v2d_lease_epochs",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ElementCapabilityError):
        require(context="replaced helper")
    monkeypatch.setattr(v2d_module, "_advance_v2d_lease_epochs", original_helper)

    # Swapping a protected hook installs a raising stub; restoring it still
    # leaves the class epoch changed for this capture.
    fresh = _capture(model)
    assert fresh is not None
    original_hook = NativeParityE4PLS3V2DShellElement.__dict__["__setattr__"]
    NativeParityE4PLS3V2DShellElement.__setattr__ = object.__setattr__
    try:
        with pytest.raises(ValueError, match="class authority was replaced"):
            _v2d_elements(model)[0].extra_marker = 1
    finally:
        NativeParityE4PLS3V2DShellElement.__setattr__ = original_hook
    assert NativeParityE4PLS3V2DShellElement.__dict__["__setattr__"] is original_hook
    with pytest.raises(ElementCapabilityError, match="class authority changed"):
        fresh(context="restored hook")
    assert _capture(model) is not None


def test_v2d_capture_requires_immutable_instance_values() -> None:
    model, _load = _mixed_model()
    object.__setattr__(_v2d_elements(model)[0], "writable_marker", np.zeros(3))

    assert _capture(model) is None


def test_v2d_capture_declines_an_inadmissible_element() -> None:
    model, _load = _mixed_model()
    element = _v2d_elements(model)[0]
    # A shadow of class authority fails the exact per-instance validation.
    object.__setattr__(element, "formulation_id", element.formulation_id)

    assert _capture(model) is None


# ---------------------------------------------------------------------------
# mutations inside a solve
# ---------------------------------------------------------------------------
class _V2DSubclass(NativeParityE4PLS3V2DShellElement):
    pass


def _install_delegating_override(element: object, _model: object) -> None:
    """A behaviourally transparent override: only the guards can notice it."""

    real = type(element).compute_nonlinear_response  # type: ignore[attr-defined]
    element.compute_nonlinear_response = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: real(element, *args, **kwargs)
    )


_WINDOW_MUTATIONS = {
    "thickness-write": lambda element, model: setattr(element, "thickness", 0.02),
    "material-name-write": lambda element, model: setattr(
        element, "material_name", element.material_name
    ),
    "node-ids-write": lambda element, model: setattr(
        element, "node_ids", tuple(element.node_ids)
    ),
    "method-override": _install_delegating_override,
    "class-swap": lambda element, model: setattr(element, "__class__", _V2DSubclass),
    "mapping-replace": lambda element, model: model.mesh.elements.__setitem__(
        3, element
    ),
}


@pytest.mark.parametrize("name", tuple(_WINDOW_MUTATIONS))
def test_non_representative_v2d_mutation_fails_closed_in_a_trusted_window(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Only the first V2D element is re-scanned by the trusted check; the
    others are protected by the write counters, so mutate the second one at a
    point that is followed by nothing but the constant-time guard."""

    model, load = _two_v2d_model()
    target = _v2d_elements(model)[1]
    calls = _inject_in_trusted_window(
        monkeypatch, lambda: _WINDOW_MUTATIONS[name](target, model)
    )

    with pytest.raises(CAUGHT):
        _solve(model, load)
    assert len(calls) >= 2


def test_v2d_write_from_a_status_callback_fails_closed() -> None:
    model, load = _mixed_model()
    element = _v2d_elements(model)[0]
    calls: list[int] = []

    def mutate(_message: str) -> None:
        calls.append(1)
        if len(calls) == 1:
            element.thickness = element.thickness

    with pytest.raises(CAUGHT):
        _solve(model, load, status_callback=mutate)
    assert calls == [1]


def test_v2d_write_after_a_solve_is_not_an_error() -> None:
    model, load = _mixed_model()
    assert _solve(model, load).status == "completed"

    _v2d_elements(model)[0].thickness = 0.01  # the capture ended with the solve


# ---------------------------------------------------------------------------
# Numba disk cache default
# ---------------------------------------------------------------------------
def test_jit_decorators_enable_the_disk_cache_unless_told_otherwise() -> None:
    seen: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake(*args: object, **kwargs: object) -> str:
        seen.append((args, kwargs))
        return "decorated"

    wrapped = jit_compiler._cached_by_default(fake)
    marker = object()

    assert wrapped(marker) == "decorated"  # bare @njit
    assert wrapped(parallel=True) == "decorated"  # @njit(parallel=True)
    assert wrapped(cache=False) == "decorated"  # explicit opt-out
    assert wrapped(cache=True, parallel=True) == "decorated"

    assert seen[0] == ((marker,), {"cache": True})
    assert seen[1] == ((), {"parallel": True, "cache": True})
    assert seen[2] == ((), {"cache": False})
    assert seen[3] == ((), {"cache": True, "parallel": True})


@pytest.mark.skipif(not JIT_ENABLED, reason="Numba is not active")
@pytest.mark.parametrize(
    ("module", "name"),
    (
        (plasticity, "_jit_plane_stress_return_map"),
        (plasticity, "_jit_flow_stress"),
        (elements_module, "_jit_integrate_nonlinear_response"),
        (elements_module, "_jit_compute_4node_shape_functions"),
        (vectorized_nonlinear, "_jit_batch_integrate_nonlinear_response"),
    ),
)
def test_hot_nonlinear_kernels_use_the_numba_disk_cache(
    module: object, name: str
) -> None:
    dispatcher = getattr(module, name)
    if not hasattr(dispatcher, "_cache"):
        pytest.skip("Numba compilation is disabled (NUMBA_DISABLE_JIT)")
    assert type(dispatcher._cache).__name__ != "NullCache"
