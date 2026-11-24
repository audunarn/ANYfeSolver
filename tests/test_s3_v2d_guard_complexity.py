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


# --- the memoized scope check still refuses ---------------------------------


class _LegacyShell(ShellElement):
    """A shell class outside the V2D mixed-shell scope."""


def _memo():
    from anyfesolver.e4_pl_s3_v2d_element import v2d_model_scope_memo

    return v2d_model_scope_memo()


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


def test_memo_sees_mapping_mutations_immediately():
    model, v2d, _beam = _v2d_with_beam()
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        with _memo():
            v2d._validate_model_scope(model.mesh)
            model.mesh.elements[100] = _LegacyShell.__new__(_LegacyShell)
            with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
                v2d._validate_model_scope(model.mesh)
            raise NativeParityCapabilityError("mixed shell scope (propagated)")


def test_memo_rescans_at_operation_end_for_unstamped_changes():
    model, v2d, beam = _v2d_with_beam()
    with pytest.raises(NativeParityCapabilityError, match="mixed shell scope"):
        with _memo():
            v2d._validate_model_scope(model.mesh)
            # __class__ assignment does not advance the mesh mutation epoch;
            # the memo may reuse its verdict inside the operation, but the
            # operation as a whole must still be refused when it completes.
            beam.__class__ = _LegacyShell
            v2d._validate_model_scope(model.mesh)


def test_memo_does_not_rescan_after_an_exception():
    model, v2d, beam = _v2d_with_beam()
    with pytest.raises(KeyError):
        with _memo():
            v2d._validate_model_scope(model.mesh)
            beam.__class__ = _LegacyShell
            raise KeyError("operation failure wins")
