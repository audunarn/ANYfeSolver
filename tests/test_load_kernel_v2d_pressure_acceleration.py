"""The owned load kernel takes exact S3 V2D pressure and acceleration loads.

Before this, a pressure or acceleration load on S3 V2D elements was refused by
``boundary._run_foreign_code_free_load_vector``.  Routing then fell back to a
complete model-wide lifecycle guard per element (quadratic: 2,192 s to prepare a
7,000-node bracketed barge).  The kernel now admits V2D under a constant-time V2D
authority check, so the number of complete lifecycle scans must not grow with
the number of loaded elements, the result must be bit-identical to the plain
arithmetic, and every refusal must still refuse.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import anyfesolver.boundary as boundary
from anyfesolver.boundary import LoadCase
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import (
    _capture_v2d_load_guard,
    assemble_load_vector,
)

_SCAN = "_require_exact_qualified_component_lifecycle_api_implementation"


def _mixed_model(pairs: int) -> tuple[FEModel, list[int], list[int]]:
    """Interleaved Q4 and V2D elements; returns (model, q4 ids, v2d ids)."""

    model = FEModel("v2d-load-kernel")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    nodes: dict[tuple[int, int], int] = {}

    def node(i: int, j: int) -> int:
        if (i, j) not in nodes:
            nodes[(i, j)] = len(nodes) + 1
            model.add_node(nodes[(i, j)], 0.1 * i, 0.1 * j, 0.0)
        return nodes[(i, j)]

    element_id = 0
    q4_ids: list[int] = []
    v2d_ids: list[int] = []
    for i in range(pairs):
        element_id += 1
        model.add_element(
            element_id,
            create_shell_element(
                element_id,
                (node(i, 0), node(i + 1, 0), node(i + 1, 1), node(i, 1)),
                "steel",
                thickness=0.01,
            ),
        )
        q4_ids.append(element_id)
        for corners in (
            ((i, 2), (i + 1, 2), (i + 1, 3)),
            ((i, 2), (i + 1, 3), (i, 3)),
        ):
            element_id += 1
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    tuple(node(*corner) for corner in corners),
                    "steel",
                    formulation="e4-pl-s3-v2d",
                    thickness=0.01,
                    reference_normal=(0.0, 0.0, 1.0),
                ),
            )
            v2d_ids.append(element_id)
    return model, q4_ids, v2d_ids


def _pressure_case(q4: list[int], v2d: list[int]) -> LoadCase:
    case = LoadCase("pressure")
    case.pressure_loads = {i: 1.0e3 + 7.0 * i for i in sorted(q4 + v2d)}
    return case


def _gravity_case(model: FEModel) -> LoadCase:
    case = LoadCase("gravity")
    case.set_gravity(0.3, -0.2, -9.81)
    case.added_node_masses = {n: 10.0 + n for n in sorted(model.mesh.nodes)[:4]}
    return case


def _oracle(model: FEModel, case: LoadCase, displacements=None) -> np.ndarray:
    """The unguarded load arithmetic the public route runs."""

    return case._get_load_vector_core(
        model.mesh,
        model.mesh.dof_manager,
        model.get_material,
        displacements,
        None,
        qualified_runtime_guard=lambda **_kw: None,
    )


def _count_scans(model: FEModel, case: LoadCase):
    scans = 0

    def profiler(frame: object, event: str, arg: object) -> None:
        nonlocal scans
        if event == "call" and frame.f_code.co_name == _SCAN:  # type: ignore[attr-defined]
            # The V2D authority check re-runs the guard on a one-element probe
            # (constant cost); count only scans of more than one element.
            probe = frame.f_locals.get("model")  # type: ignore[attr-defined]
            if len(probe.mesh.elements) > 1:
                scans += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        result = assemble_load_vector(model, case)
    finally:
        sys.setprofile(previous)
    return scans, result


@pytest.mark.parametrize("kind", ["pressure", "gravity"])
def test_complete_guard_scans_do_not_grow_with_v2d_elements(kind: str) -> None:
    counts = []
    for pairs in (3, 12):
        model, q4, v2d = _mixed_model(pairs)
        case = _pressure_case(q4, v2d) if kind == "pressure" else _gravity_case(model)
        scans, (vector, _info) = _count_scans(model, case)
        assert scans > 0 and np.all(np.isfinite(vector)) and np.any(vector)
        counts.append(scans)
    # Four times the elements: a per-element complete guard would quadruple it.
    assert counts[1] <= counts[0] + 2, counts


def test_v2d_loads_are_taken_by_the_owned_kernel(monkeypatch) -> None:
    taken = []
    original = boundary._run_foreign_code_free_load_vector

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        taken.append(result is not None)
        return result

    monkeypatch.setattr(boundary, "_run_foreign_code_free_load_vector", spy)
    model, q4, v2d = _mixed_model(3)
    assemble_load_vector(model, _pressure_case(q4, v2d))
    assemble_load_vector(model, _gravity_case(model))
    assert taken == [True, True]


def test_v2d_results_equal_the_plain_arithmetic_bit_for_bit() -> None:
    model, q4, v2d = _mixed_model(5)
    cases = [_pressure_case(q4, v2d), _gravity_case(model)]
    element_loaded = LoadCase("element")
    element_loaded.element_loads = {v2d[0]: np.arange(18, dtype=float)}
    cases.append(element_loaded)
    both = _gravity_case(model)
    both.pressure_loads = {i: 250.0 for i in v2d}
    cases.append(both)
    for case in cases:
        vector, _info = assemble_load_vector(model, case)
        assert np.array_equal(vector, _oracle(model, case)), case.name
    _vector, info = assemble_load_vector(model, cases[0])
    assert [r["element_id"] for r in info["qualified_s3_pressure_surfaces"]] == sorted(v2d)


def test_follower_pressure_on_v2d_matches_the_plain_arithmetic() -> None:
    model, q4, v2d = _mixed_model(3)
    case = LoadCase("follower")
    case.follower_pressure = True
    case.pressure_loads = {i: 500.0 for i in v2d}
    u = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
    vector, _ = assemble_load_vector(model, case, u)
    assert np.array_equal(vector, _oracle(model, case, u))


# --- adversarial: the refusals still fire -----------------------------------


def _call_kernel(model: FEModel, case: LoadCase, **kwargs):
    mesh = model.mesh
    return boundary._run_foreign_code_free_load_vector(
        case, mesh, mesh.dof_manager, model.get_material, None, None,
        stage_guard=lambda **_k: None, **kwargs,
    )


def test_kernel_refuses_v2d_without_the_authority_attestation() -> None:
    model, q4, v2d = _mixed_model(2)
    for case in (_pressure_case(q4, v2d), _gravity_case(model)):
        assert _call_kernel(model, case) is None
        assert _call_kernel(model, case, v2d_authority=True) is not None


def test_ineligible_v2d_instance_keeps_the_public_route(monkeypatch) -> None:
    model, q4, v2d = _mixed_model(2)
    case = _pressure_case(q4, v2d)
    expected = _oracle(model, case)
    element = model.mesh.elements[v2d[0]]
    object.__setattr__(element, "scratch_note", ["mutable"])
    assert _capture_v2d_load_guard(model, vars(case)) is None
    taken = []
    original = boundary._run_foreign_code_free_load_vector

    def spy(*args, **kwargs):
        taken.append(original(*args, **kwargs))
        return taken[-1]

    monkeypatch.setattr(boundary, "_run_foreign_code_free_load_vector", spy)
    # The complete guard may reject the foreign instance state; if it does not,
    # the kernel must not have taken the span and the arithmetic is unchanged.
    try:
        vector, _info = assemble_load_vector(model, case)
    except Exception:
        return
    assert taken == [None]
    assert np.array_equal(vector, expected)


def test_v2d_instance_write_during_the_span_is_refused(monkeypatch) -> None:
    model, q4, v2d = _mixed_model(2)
    case = _pressure_case(q4, v2d)
    target = model.mesh.elements[v2d[1]]
    real_core = boundary._EXACT_LOAD_CASE_GET_VECTOR_CORE

    def mutating_core(*args, **kwargs):
        target.thickness = float(target.thickness)  # ordinary write, same value
        return real_core(*args, **kwargs)

    monkeypatch.setattr(boundary, "_EXACT_LOAD_CASE_GET_VECTOR_CORE", mutating_core)
    with pytest.raises(Exception, match="(?i)v2d|qualified|modif"):
        assemble_load_vector(model, case)


def test_v2d_class_change_during_the_span_is_refused(monkeypatch) -> None:
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    model, q4, v2d = _mixed_model(2)
    case = _pressure_case(q4, v2d)
    real_core = boundary._EXACT_LOAD_CASE_GET_VECTOR_CORE

    def mutating_core(*args, **kwargs):
        NativeParityE4PLS3V2DShellElement.scratch_flag = 1
        try:
            return real_core(*args, **kwargs)
        finally:
            del NativeParityE4PLS3V2DShellElement.scratch_flag

    monkeypatch.setattr(boundary, "_EXACT_LOAD_CASE_GET_VECTOR_CORE", mutating_core)
    with pytest.raises(Exception, match="(?i)v2d|qualified|authority"):
        assemble_load_vector(model, case)


def test_subclassed_v2d_class_is_not_admitted() -> None:
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    model, q4, v2d = _mixed_model(2)
    element = model.mesh.elements[v2d[0]]
    forged = type("ForgedV2D", (NativeParityE4PLS3V2DShellElement,), {})
    element.__class__ = forged
    case = _pressure_case(q4, v2d)
    assert _call_kernel(model, case, v2d_authority=True) is None


# --- zero-mass beam-shell MPC elements ride along in mixed acceleration models


def _add_mpc(model: FEModel, shell_ids: list[int]) -> int:
    from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement

    element = model.mesh.elements[shell_ids[0]]
    beam_node = max(model.mesh.nodes) + 1
    model.add_node(beam_node, 0.05, 0.05, 0.1)
    element_id = max(model.mesh.elements) + 1
    model.add_element(
        element_id,
        InterpolatedBeamShellMPCElement(
            element_id,
            beam_node,
            list(element.node_ids),
            np.array([0.2, 0.3, 0.5]),
            np.array([0.0, 0.0, 0.1]),
            "steel",
        ),
    )
    return element_id


def test_mpc_element_acceleration_is_owned_and_bit_identical(monkeypatch) -> None:
    model, q4, v2d = _mixed_model(3)
    _add_mpc(model, v2d)
    case = _gravity_case(model)
    taken = []
    original = boundary._run_foreign_code_free_load_vector

    def spy(*args, **kwargs):
        taken.append(original(*args, **kwargs) is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary, "_run_foreign_code_free_load_vector", spy)
    vector, _info = assemble_load_vector(model, case)
    assert taken[0] is True
    assert np.array_equal(vector, _oracle(model, case))


def test_replaced_mpc_mass_method_is_not_admitted(monkeypatch) -> None:
    from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement

    model, q4, v2d = _mixed_model(2)
    _add_mpc(model, v2d)
    case = _gravity_case(model)
    assert _call_kernel(model, case, v2d_authority=True) is not None
    original = InterpolatedBeamShellMPCElement.compute_mass_matrix
    monkeypatch.setattr(
        InterpolatedBeamShellMPCElement,
        "compute_mass_matrix",
        lambda self, mesh, material: original(self, mesh, material),
    )
    assert _call_kernel(model, case, v2d_authority=True) is None


# --- pressure entries on two-node beam/coupling elements (exactly zero) ------


def _add_beam_and_coupling(model: FEModel) -> tuple[int, int]:
    from anyfesolver.elements import BeamElement, CoupledBeamShellElement

    nodes = sorted(model.mesh.nodes)
    beam_id = max(model.mesh.elements) + 1
    model.add_element(beam_id, BeamElement(beam_id, [nodes[0], nodes[1]], "steel"))
    coupling_id = beam_id + 1
    model.add_element(
        coupling_id,
        CoupledBeamShellElement(coupling_id, nodes[2], nodes[3], "steel", 1.0e6),
    )
    return beam_id, coupling_id


def test_pressure_on_beam_and_coupling_is_zero_and_stays_owned(monkeypatch) -> None:
    model, q4, v2d = _mixed_model(3)
    beam_id, coupling_id = _add_beam_and_coupling(model)
    case = _pressure_case(q4, v2d)
    probe = LoadCase("zero")
    for element_id in (beam_id, coupling_id):
        assert not np.any(
            probe._consistent_pressure_load(model.mesh.elements[element_id], model.mesh, 1.0, None)
        )
    case.pressure_loads[beam_id] = 5.0e3
    case.pressure_loads[coupling_id] = 5.0e3
    taken = []
    original = boundary._run_foreign_code_free_load_vector

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        taken.append(result is not None)
        return result

    monkeypatch.setattr(boundary, "_run_foreign_code_free_load_vector", spy)
    vector, _info = assemble_load_vector(model, case)
    assert taken == [True]
    assert np.array_equal(vector, _oracle(model, case))


def test_scans_stay_flat_with_beam_pressure_entries() -> None:
    counts = []
    for pairs in (3, 12):
        model, q4, v2d = _mixed_model(pairs)
        beam_id, coupling_id = _add_beam_and_coupling(model)
        case = _pressure_case(q4, v2d)
        case.pressure_loads[beam_id] = 1.0
        case.pressure_loads[coupling_id] = 1.0
        scans, _result = _count_scans(model, case)
        counts.append(scans)
    assert counts[1] <= counts[0] + 2, counts


def test_replaced_beam_class_member_keeps_the_public_route(monkeypatch) -> None:
    from anyfesolver.elements import BeamElement

    model, q4, v2d = _mixed_model(2)
    beam_id, _coupling_id = _add_beam_and_coupling(model)
    case = LoadCase("beam-pressure")
    case.pressure_loads = {beam_id: 1.0}
    assert _call_kernel(model, case, v2d_authority=True) is not None
    original = BeamElement.get_node_coordinates
    monkeypatch.setattr(
        BeamElement, "get_node_coordinates", lambda self, mesh: original(self, mesh)
    )
    assert _call_kernel(model, case, v2d_authority=True) is None


def test_beam_with_native_pressure_hook_is_not_admitted() -> None:
    model, q4, v2d = _mixed_model(2)
    beam_id, _coupling_id = _add_beam_and_coupling(model)
    case = LoadCase("beam-pressure")
    case.pressure_loads = {beam_id: 1.0}
    object.__setattr__(
        model.mesh.elements[beam_id], "compute_native_dead_pressure_load", lambda *a: None
    )
    assert _call_kernel(model, case, v2d_authority=True) is None


def test_v2d_trusted_capture_does_not_change_the_source_witness() -> None:
    """A load assembly subscribes V2D elements to a lease epoch; preparation's
    coherence witness must not read that bookkeeping as a model change."""

    from anyfesolver.prepared_static_snapshot import capture_source_witness

    model, q4, v2d = _mixed_model(3)
    before = capture_source_witness(model)
    assemble_load_vector(model, _pressure_case(q4, v2d))
    assert capture_source_witness(model) == before


# --- the admission must not be wider than the exact package classes ----------


def _pressure_on_beam_case(beam_id: int) -> LoadCase:
    case = LoadCase("beam-pressure")
    case.pressure_loads = {beam_id: 1.0}
    return case


def test_beam_subclass_is_refused_and_takes_the_public_route() -> None:
    from anyfesolver.elements import BeamElement

    model, q4, v2d = _mixed_model(2)
    beam_id, _c = _add_beam_and_coupling(model)

    class SubBeam(BeamElement):
        pass

    old = model.mesh.elements[beam_id]
    model.mesh.elements[beam_id] = SubBeam(
        beam_id, list(old.node_ids), "steel", cross_section=dict(old.cross_section)
    )
    assert _call_kernel(model, _pressure_on_beam_case(beam_id), v2d_authority=True) is None


def test_metaclass_equality_and_class_spoof_are_refused() -> None:
    from anyfesolver.elements import BeamElement

    class AlwaysEqual(type(BeamElement)):
        def __eq__(cls, other):  # noqa: D105
            return True

        __hash__ = type.__hash__

    model, q4, v2d = _mixed_model(2)
    beam_id, _c = _add_beam_and_coupling(model)
    old = model.mesh.elements[beam_id]
    forged_cls = AlwaysEqual("Forged", (BeamElement,), {})
    forged = forged_cls(beam_id, list(old.node_ids), "steel", cross_section=dict(old.cross_section))
    model.mesh.elements[beam_id] = forged
    case = _pressure_on_beam_case(beam_id)
    assert _call_kernel(model, case, v2d_authority=True) is None
    # A ``__class__`` property that claims to be the exact class is not the type.
    spoof_cls = type("Spoof", (BeamElement,), {"__class__": property(lambda self: BeamElement)})
    spoof = spoof_cls(beam_id, list(old.node_ids), "steel", cross_section=dict(old.cross_section))
    model.mesh.elements[beam_id] = spoof
    assert _call_kernel(model, case, v2d_authority=True) is None
    # The same metaclass trick against a V2D slot of the owned types.
    model2, q4b, v2db = _mixed_model(2)
    victim = model2.mesh.elements[v2db[0]]
    victim.__class__ = type(
        "ForgedV2D",
        (type(victim),),
        {},
        )
    assert _call_kernel(model2, _pressure_case(q4b, v2db), v2d_authority=True) is None


def test_beam_with_non_plain_instance_state_is_refused() -> None:
    model, q4, v2d = _mixed_model(2)
    beam_id, coupling_id = _add_beam_and_coupling(model)
    for element_id in (beam_id, coupling_id):
        case = _pressure_on_beam_case(element_id)
        assert _call_kernel(model, case, v2d_authority=True) is not None
        element = model.mesh.elements[element_id]
        object.__setattr__(element, "hostile", object())
        assert _call_kernel(model, case, v2d_authority=True) is None
        del element.__dict__["hostile"]
        assert _call_kernel(model, case, v2d_authority=True) is not None
        # A node-id list holding a non-int is refused as well.
        saved = element.node_ids
        object.__setattr__(element, "node_ids", [saved[0], float(saved[1])])
        assert _call_kernel(model, case, v2d_authority=True) is None
        object.__setattr__(element, "node_ids", saved)


def test_v2d_subclass_pressure_and_acceleration_are_refused() -> None:
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    model, q4, v2d = _mixed_model(2)
    forged = type("SubV2D", (NativeParityE4PLS3V2DShellElement,), {})
    model.mesh.elements[v2d[0]].__class__ = forged
    assert _call_kernel(model, _pressure_case(q4, v2d), v2d_authority=True) is None
    assert _call_kernel(model, _gravity_case(model), v2d_authority=True) is None


def test_mixed_q4_v2d_beam_coupling_hashes_equal_the_public_route() -> None:
    import hashlib

    model, q4, v2d = _mixed_model(4)
    beam_id, coupling_id = _add_beam_and_coupling(model)
    pressure = _pressure_case(q4, v2d)
    pressure.pressure_loads[beam_id] = 2.0e3
    pressure.pressure_loads[coupling_id] = 2.0e3
    gravity = _gravity_case(model)
    mixed = _gravity_case(model)
    mixed.pressure_loads = dict(pressure.pressure_loads)
    for case in (pressure, gravity, mixed):
        owned, _info = assemble_load_vector(model, case)
        public = case.get_load_vector(
            model.mesh, model.mesh.dof_manager, model.get_material
        )
        digest = lambda a: hashlib.sha256(np.asarray(a, dtype=float).tobytes()).hexdigest()
        assert digest(owned) == digest(public), case.name
