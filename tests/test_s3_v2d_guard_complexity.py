"""Guard work on S3 V2D (default triangle) models must stay linear in mesh size.

Spot check 2026-11-24: a triangle benchmark set did not finish in 40 minutes
where the equivalent Q4 sets took minutes.  Two guard paths were quadratic:

* ``_qualified_s3_pressure_surface_records`` ran the complete model-wide
  lifecycle scan after every pressure record whenever the lease had no
  constant-time input guard, which is the case for every pure-V2D and mixed
  V2D model on the ``solve_linear``/``assemble_system`` route (plate 16x16
  triangles with pressure: 81 s, the same plate with a point load: 1 s);
* ``NativeParityE4PLS3V2DShellElement._validate_model_scope`` scanned every
  mesh element on every per-element geometry evaluation (stress recovery of
  1,024 triangles: 7.3 million scan steps).

These tests count the work, not the time, so they are deterministic, and they
check that results and refusals are unchanged.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, solve_linear
from anyfesolver.assembly import compute_stresses
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.e4_pl_s3_v2d_element import (
    NativeParityCapabilityError,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.elements import BeamElement, ShellElement, create_shell_element

_SCAN = "_require_exact_qualified_component_lifecycle_api_implementation"
_V2D_FILE = "e4_pl_s3_v2d_element.py"


def _triangle_plate(n: int, *, quads_every: int = 0) -> tuple[FEModel, list[int]]:
    """Clamped n x n plate split into triangles (optionally some Q4 cells)."""

    model = FEModel("v2d-guard-complexity")
    model.add_material("steel", 210.0e9, 0.3)
    ids = {}
    for i in range(n + 1):
        for j in range(n + 1):
            ids[i, j] = len(ids) + 1
            model.add_node(ids[i, j], 0.1 * i, 0.1 * j, 0.0)
    element_id = 0
    cell = 0
    for i in range(n):
        for j in range(n):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            cell += 1
            if quads_every and cell % quads_every == 0:
                element_id += 1
                model.add_element(
                    element_id,
                    create_shell_element(element_id, [a, b, c, d], "steel", thickness=0.01),
                )
                continue
            for tri in ((a, b, c), (a, c, d)):
                element_id += 1
                model.add_element(
                    element_id,
                    create_shell_element(
                        element_id,
                        list(tri),
                        "steel",
                        thickness=0.01,
                        reference_normal=(0.0, 0.0, 1.0),
                    ),
                )
    edge = [ids[i, j] for (i, j) in ids if i in (0, n) or j in (0, n)]
    model.add_boundary_condition(
        BoundaryCondition(
            "clamp", edge, {k: 0.0 for k in ("ux", "uy", "uz", "rx", "ry", "rz")}
        )
    )
    return model, sorted(model.mesh.elements)


def _pressure(element_ids: list[int]) -> LoadCase:
    case = LoadCase("pressure")
    for element_id in element_ids:
        case.add_pressure_load(int(element_id), 1.0e3)
    return case


def _profile(call):
    """Return (complete multi-element lifecycle scans, V2D generator steps, result)."""

    scans = 0
    v2d_generator_steps = 0

    def profiler(frame, event, arg):
        nonlocal scans, v2d_generator_steps
        if event != "call":
            return
        code = frame.f_code
        if code.co_name == _SCAN:
            probe = frame.f_locals.get("model")
            if len(probe.mesh.elements) > 1:
                scans += 1
        elif code.co_name == "<genexpr>" and code.co_filename.endswith(_V2D_FILE):
            v2d_generator_steps += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        result = call()
    finally:
        sys.setprofile(previous)
    return scans, v2d_generator_steps, result


@pytest.mark.parametrize("quads_every", [0, 3])
def test_solve_linear_pressure_scans_do_not_grow_with_pressured_elements(quads_every):
    counts = []
    for n in (2, 4):
        model, element_ids = _triangle_plate(n, quads_every=quads_every)
        scans, _steps, (u, info) = _profile(
            lambda: solve_linear(model, _pressure(element_ids))
        )
        assert info["convergence_info"]["status"] == "converged"
        assert np.all(np.isfinite(u)) and np.any(u)
        counts.append(scans)
    # Four times the pressured elements: one complete scan per record (the old
    # route) would add about 3 * (number of small-mesh records) scans.
    assert counts[1] <= counts[0] + 2, counts


def _records(model, element_ids):
    from anyfesolver.matrix_assembly import _run_with_qualified_assembly_runtime_lease
    import anyfesolver.matrix_assembly as matrix_assembly

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="test pressure-surface records",
        # A plain wrapper exposes no trusted capability, as on the
        # assemble_system route that loads triangles.
        operation=lambda lease: matrix_assembly._qualified_s3_pressure_surface_records(
            model,
            _pressure(element_ids),
            qualified_runtime_guard=lambda m, *, context: lease(m, context=context),
        ),
    )


def test_v2d_pressure_surface_records_are_unchanged(monkeypatch):
    import anyfesolver.matrix_assembly as matrix_assembly

    model, element_ids = _triangle_plate(2, quads_every=3)
    scans, _steps, records = _profile(lambda: _records(model, element_ids))
    monkeypatch.setattr(
        matrix_assembly, "_foreign_code_free_v2d_element_mapping", lambda model: None
    )
    guarded_scans, _steps, guarded = _profile(lambda: _records(model, element_ids))
    assert records == guarded
    assert scans < guarded_scans
    v2d_ids = [
        e for e in element_ids
        if type(model.mesh.elements[e]) is NativeParityE4PLS3V2DShellElement
    ]
    assert [r["element_id"] for r in records] == v2d_ids
    for record in records:
        assert record == {
            "element_id": record["element_id"],
            "pressure_surface_id": "ELEMENT_NODAL_REFERENCE_SURFACE_V1",
            "reference_surface_offset": 0.0,
            "resultant_and_reaction_reference": "GLOBAL_NODAL_REFERENCE_COORDINATES",
            "section_origin_offset_from_reference": -0.0,
            "virtual_work": "TRANSLATIONAL_NODAL_REFERENCE_SURFACE_ONLY",
        }


def test_stress_recovery_scope_scan_is_linear_in_mesh_size():
    steps = []
    for n in (2, 4):
        model, element_ids = _triangle_plate(n)
        u, _info = solve_linear(model, _pressure(element_ids))
        _scans, generator_steps, stresses = _profile(lambda: compute_stresses(model, u))
        assert sorted(stresses) == element_ids
        steps.append((len(element_ids), generator_steps))
    (small_n, small), (large_n, large) = steps
    # Linear work grows with the element ratio (4x); the old per-evaluation
    # scan grew with its square (16x).
    assert large <= 1.5 * (large_n / small_n) * small, steps


# --- the scope check still refuses exactly as before ---------------------


class _LegacyShell(ShellElement):
    """A shell class outside the V2D mixed-shell scope."""


def _v2d_with_beam() -> tuple[FEModel, NativeParityE4PLS3V2DShellElement, BeamElement]:
    model, element_ids = _triangle_plate(1)
    beam = BeamElement(
        99, [1, 2], "steel", {"A": 1.0e-4, "Iy": 1.0e-8, "Iz": 1.0e-8, "J": 1.0e-8}
    )
    model.add_element(99, beam)
    return model, model.mesh.elements[element_ids[0]], beam


def test_scope_refusal_is_unchanged_outside_an_operation():
    model, v2d, _beam = _v2d_with_beam()
    v2d._validate_model_scope(model.mesh)
    legacy = _LegacyShell.__new__(_LegacyShell)
    model.mesh.elements[100] = legacy
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        v2d._validate_model_scope(model.mesh)


def _count_q4_boundary_guards(call):
    calls = 0

    def profiler(frame, event, arg):
        nonlocal calls
        if event == "call" and frame.f_code.co_name == "boundary_guard":
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        result = call()
    finally:
        sys.setprofile(previous)
    return calls, result


def test_mixed_q4_v2d_recovery_guard_work_is_linear():
    """Mixed Q4/V2D recovery ran a complete Q4 guard per V2D observation."""

    counts = []
    for n in (3, 6):
        model, element_ids = _triangle_plate(n, quads_every=3)
        u, _info = solve_linear(model, _pressure(element_ids))
        calls, stresses = _count_q4_boundary_guards(lambda: compute_stresses(model, u))
        assert sorted(stresses) == element_ids
        counts.append((len(element_ids), calls))
    (small_n, small), (large_n, large) = counts
    # Linear work grows with the element ratio (~4x); one complete Q4 guard
    # per V2D observation grew with its square (~16x).
    assert small > 0 and large <= 1.5 * (large_n / small_n) * small, counts


# --- review round 1 (Codex, 2026-11-24) -------------------------------------


def test_pressure_records_follow_an_instance_get_element_override(monkeypatch):
    import anyfesolver.matrix_assembly as matrix_assembly

    model, element_ids = _triangle_plate(2)
    hidden = element_ids[0]
    original = model.mesh.get_element
    model.mesh.get_element = lambda element_id: (
        None if element_id == hidden else original(element_id)
    )
    records = _records(model, element_ids)
    monkeypatch.setattr(
        matrix_assembly, "_foreign_code_free_v2d_element_mapping", lambda model: None
    )
    guarded = _records(model, element_ids)
    assert records == guarded
    assert hidden not in [record["element_id"] for record in records]


# --- review round 2 (Codex, 2026-11-24): the scope test stays per-evaluation --


class _ForeignElement(BeamElement):
    """A subclass is not package-owned: its methods may be foreign code."""


# test_class_swap_inside_an_element_callback_is_refused_at_once (round 2) was
# removed with the owner's guard-shortcut contract of 2026-11-24: __class__
# reassignment made and undone inside an operation is outside supported use,
# and the per-operation census memo (Codex supported-use review, P2) relies on
# that.  A persistent change is still refused by the next operation.


def test_a_persistent_class_change_is_refused_by_the_next_operation():
    model, element_ids = _triangle_plate(2)
    beam = BeamElement(
        99, [1, 2], "steel", {"A": 1.0e-4, "Iy": 1.0e-8, "Iz": 1.0e-8, "J": 1.0e-8}
    )
    model.add_element(99, beam)
    u, _info = solve_linear(model, _pressure(element_ids))
    compute_stresses(model, u)
    beam.__class__ = _LegacyShell
    with pytest.raises(Exception) as caught:
        compute_stresses(model, u)
    assert "scope" in str(caught.value) or "authority" in str(caught.value)


def test_census_answers_like_isinstance_for_every_class_shape():
    from anyfesolver.e4_pl_s3_v2d_element import _require_v2d_mixed_shell_scope

    model, v2d, beam = _v2d_with_beam()
    elements = model.mesh.elements
    _require_v2d_mixed_shell_scope(elements)

    class _MasqueradingBeam(BeamElement):
        @property
        def __class__(self):  # isinstance consults __class__
            return _LegacyShell

    beam.__class__ = _MasqueradingBeam
    assert isinstance(beam, ShellElement)
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        _require_v2d_mixed_shell_scope(elements)
    object.__dict__["__class__"].__set__(beam, _ForeignElement)
    _require_v2d_mixed_shell_scope(elements)
    ShellElement.register(_ForeignElement)  # virtual subclass
    try:
        with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
            _require_v2d_mixed_shell_scope(elements)
    finally:
        beam.__class__ = BeamElement
    _require_v2d_mixed_shell_scope(elements)
    plain = dict(elements)
    plain[100] = _LegacyShell.__new__(_LegacyShell)
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        _require_v2d_mixed_shell_scope(plain)


def test_provider_change_from_a_displacement_callback_is_refused():
    """Codex round 2: a callback installs providers and removes them later."""

    model, element_ids = _triangle_plate(2, quads_every=3)
    u, _info = solve_linear(model, _pressure(element_ids))
    material = model.get_material("steel")
    mesh = model.mesh
    calls = {"material": 0}

    def get_material(name):
        calls["material"] += 1
        if calls["material"] == 2:
            # Remove both overrides before the operation's final guard.
            del model.__dict__["get_material"]
            del mesh.__dict__["get_node"]
        return material

    class _Displacements:
        def __array__(self, dtype=None, copy=None):
            original_get_node = mesh.get_node
            model.get_material = get_material
            mesh.get_node = lambda node_id: original_get_node(node_id)
            return np.asarray(u, dtype=dtype)

    with pytest.raises(Exception) as caught:
        compute_stresses(model, _Displacements())
    assert calls["material"] <= 1, calls
    assert "authority" in str(caught.value), caught.value


# --- review round 3 (Codex, 2026-11-24) -------------------------------------


def test_scope_shortcut_defers_to_isinstance_for_a_faked_class():
    from anyfesolver.e4_pl_s3_v2d_element import _require_v2d_mixed_shell_scope

    class _FakingBeam(BeamElement):
        def __getattribute__(self, name):
            if name == "__class__":
                return _LegacyShell
            return object.__getattribute__(self, name)

    model, v2d, beam = _v2d_with_beam()
    beam.__class__ = _FakingBeam
    assert isinstance(beam, ShellElement)
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        _require_v2d_mixed_shell_scope(model.mesh.elements)


def test_scope_shortcut_defers_when_owned_class_machinery_changed(monkeypatch):
    from anyfesolver.e4_pl_s3_v2d_element import _require_v2d_mixed_shell_scope

    model, v2d, beam = _v2d_with_beam()

    def faking(self, name):
        if name == "__class__":
            return _LegacyShell
        return object.__getattribute__(self, name)

    monkeypatch.setattr(BeamElement, "__getattribute__", faking, raising=False)
    assert isinstance(beam, ShellElement)
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        _require_v2d_mixed_shell_scope(model.mesh.elements)


def test_pressure_records_drop_the_fast_path_after_caller_code(monkeypatch):
    import anyfesolver.matrix_assembly as matrix_assembly
    from anyfesolver.matrix_assembly import _run_with_qualified_assembly_runtime_lease

    model, element_ids = _triangle_plate(2)
    mesh = model.mesh
    hidden = element_ids[-1]

    class _InstallingId:
        def __int__(self):
            original = FEMeshGetElement
            mesh.get_element = lambda element_id: (
                None if element_id == hidden else original(mesh, element_id)
            )
            return 10_000

    class _RestoringId:
        def __int__(self):
            mesh.__dict__.pop("get_element", None)
            return 10_001

    from anyfesolver.fe_core import FEMesh

    FEMeshGetElement = FEMesh.get_element
    case = LoadCase("pressure")
    case.pressure_loads = {element_ids[0]: 1.0, _InstallingId(): 1.0, hidden: 1.0, _RestoringId(): 1.0}

    def records():
        return _run_with_qualified_assembly_runtime_lease(
            model,
            context="test pressure-surface records",
            operation=lambda lease: matrix_assembly._qualified_s3_pressure_surface_records(
                model,
                case,
                qualified_runtime_guard=lambda m, *, context: lease(m, context=context),
            ),
        )

    try:
        fast = records()
    except Exception as exc:  # the guarded route may refuse the provider change
        fast = exc
    monkeypatch.setattr(
        matrix_assembly, "_foreign_code_free_v2d_element_mapping", lambda model: None
    )
    mesh.__dict__.pop("get_element", None)
    try:
        guarded = records()
    except Exception as exc:
        guarded = exc
    if isinstance(guarded, Exception):
        assert type(fast) is type(guarded) and str(fast) == str(guarded)
    else:
        assert fast == guarded
        assert hidden not in [record["element_id"] for record in fast]


# --- review round 4 (Codex, 2026-11-24; reproduced before a content filter stopped it) ---


def test_scope_census_compares_classes_by_identity_not_metaclass_equality():
    from abc import ABCMeta

    from anyfesolver.e4_pl_s3_v2d_element import _require_v2d_mixed_shell_scope

    class _AliasMeta(ABCMeta):
        def __hash__(cls):
            return hash(BeamElement)

        def __eq__(cls, other):
            return other is BeamElement or type.__eq__(cls, other)

    class _AliasedBeam(BeamElement, metaclass=_AliasMeta):
        def __getattribute__(self, name):
            if name == "__class__":
                return _LegacyShell
            return object.__getattribute__(self, name)

    model, v2d, beam = _v2d_with_beam()
    aliased = _AliasedBeam.__new__(_AliasedBeam)
    dict.__setitem__(model.mesh.elements, 98, aliased)
    assert isinstance(aliased, ShellElement)
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        _require_v2d_mixed_shell_scope(model.mesh.elements)


def test_scope_census_defers_when_the_mapping_class_machinery_changed(monkeypatch):
    from anyfesolver.e4_pl_s3_v2d_element import _require_v2d_mixed_shell_scope
    from anyfesolver.fe_core import _QualifiedStateMapping

    model, v2d, beam = _v2d_with_beam()
    legacy = _LegacyShell.__new__(_LegacyShell)

    def faking(self, name):
        if name == "values":
            return lambda: [legacy]
        return object.__getattribute__(self, name)

    monkeypatch.setattr(_QualifiedStateMapping, "__getattribute__", faking, raising=False)
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        _require_v2d_mixed_shell_scope(model.mesh.elements)


# --- Codex supported-use review (2026-11-24): performance evidence ------------


def _count_calls(name, call):
    calls = 0

    def profiler(frame, event, arg):
        nonlocal calls
        code = frame.f_code
        if event == "call" and code.co_name == name and code.co_filename.endswith(_V2D_FILE):
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        result = call()
    finally:
        sys.setprofile(previous)
    return calls, result


def test_full_element_census_is_taken_a_bounded_number_of_times_per_operation():
    counts = []
    for n in (2, 4):
        model, element_ids = _triangle_plate(n)
        solve_calls, (u, _info) = _count_calls(
            "_census_ids", lambda: solve_linear(model, _pressure(element_ids))
        )
        recovery_calls, _stresses = _count_calls(
            "_census_ids", lambda: compute_stresses(model, u)
        )
        counts.append((solve_calls, recovery_calls))
    # Each walk over the whole element registry is linear; it must not be
    # repeated once per element (that made recovery quadratic).
    assert 0 < counts[1][1] <= counts[0][1] + 2, counts
    assert 0 < counts[1][0] <= counts[0][0] + 2, counts


def test_a_missing_pressure_id_keeps_the_pressure_record_fast_path():
    model, element_ids = _triangle_plate(4)
    first = LoadCase("missing first")
    first.pressure_loads = {987654: 1.0e3, **{e: 1.0e3 for e in element_ids}}
    last = LoadCase("missing last")
    last.pressure_loads = {**{e: 1.0e3 for e in element_ids}, 987654: 1.0e3}
    from anyfesolver.matrix_assembly import assemble_system

    scans_first, _steps, (k1, f1, _i1) = _profile(lambda: assemble_system(model, first))
    scans_last, _steps, (k2, f2, _i2) = _profile(lambda: assemble_system(model, last))
    assert np.array_equal(f1, f2)
    assert scans_first == scans_last, (scans_first, scans_last)


def test_census_memo_retakes_the_census_after_a_supported_mesh_mutation():
    from anyfesolver.e4_pl_s3_v2d_element import v2d_scope_census_memo

    model, v2d, beam = _v2d_with_beam()
    with v2d_scope_census_memo():
        v2d._validate_model_scope(model.mesh)
        model.mesh.elements[100] = _LegacyShell.__new__(_LegacyShell)
        with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
            v2d._validate_model_scope(model.mesh)


def test_census_memo_is_inert_in_a_context_copied_during_the_operation():
    import contextvars

    from anyfesolver.e4_pl_s3_v2d_element import _SCOPE_CENSUS_MEMO, v2d_scope_census_memo

    model, v2d, beam = _v2d_with_beam()
    with v2d_scope_census_memo():
        v2d._validate_model_scope(model.mesh)
        copied = contextvars.copy_context()
    memo = copied.run(_SCOPE_CENSUS_MEMO.get)
    assert memo is not None and not memo.active and not memo.entries


# --- follow-up 2026-11-25 (Codex on main): exact list/tuple by identity -------


def test_element_ids_with_a_list_equal_metaclass_are_not_treated_as_plain():
    class _ListEqualMeta(type):
        def __eq__(cls, other):
            return other is list or type.__eq__(cls, other)

        __hash__ = type.__hash__

    model, element_ids = _triangle_plate(2)
    u, _info = solve_linear(model, _pressure(element_ids))
    wanted = element_ids[:3]
    shared = iter(wanted)

    class _SharedIterator(metaclass=_ListEqualMeta):
        def __iter__(self):
            return shared  # one iterator: a second pass sees nothing

    assert type(_SharedIterator()) in (list, tuple)  # the old check's view
    stresses = compute_stresses(model, u, element_ids=_SharedIterator())
    assert sorted(stresses) == sorted(wanted)
