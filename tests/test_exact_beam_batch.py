"""Batched exact-beam stiffness/recovery (D1 guard bracket, D2 batch kernel).

The batch must be bit-identical to the scalar element methods and must take
the unchanged scalar loop (or reject identically) for every input outside its
admitted exact surface.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pytest

import anyfesolver.elements as elements_module
import anyfesolver.matrix_assembly as matrix_assembly
from anyfesolver import FEModel, LoadCase, solve_linear
from anyfesolver.assembly import _compute_stresses_under_lease, compute_stresses
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.elements import BeamElement, Element, create_shell_element
from anyfesolver.exact_beam_batch import (
    CHUNK_ELEMENTS,
    MIN_BATCH_ELEMENTS,
    disable_exact_beam_batch,
    enable_exact_beam_batch,
    exact_beam_batch_status,
)
from anyfesolver.fe_core import Material
from anyfesolver.matrix_assembly import AssemblyError, assemble_stiffness_matrix

DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")
SECTION = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}


def _random_model(count: int = 60, seed: int = 11) -> tuple[FEModel, LoadCase]:
    """Irregular 3D beams covering every scalar branch of the beam frame."""

    rng = np.random.default_rng(seed)
    model = FEModel("beam-batch")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    model.add_material("alu", 7.0e10, 0.33, density=2700.0)
    points = np.cumsum(rng.normal(size=(count + 1, 3)), axis=0)
    # Axis-aligned members and signed zeros.
    points[1] = points[0] + np.array([2.0, 0.0, 0.0])
    points[2] = points[1] + np.array([0.0, -1.5, 0.0])
    points[3] = points[2] + np.array([0.0, 0.0, 3.0])
    points[4] = np.array([-0.0, -0.0, points[3][2]])
    for index, point in enumerate(points):
        model.add_node(index + 1, *map(float, point))
    model.add_node(count + 2, -0.0, 7.0, -0.0)
    for index in range(count):
        section = dict(SECTION)
        section["area"] = float(0.01 + 0.01 * rng.random())
        if index % 3 == 0:
            section["orientation"] = list(map(float, rng.normal(size=3)))
        if index == 3:
            # Orientation parallel to the member axis: heuristic fallback.
            axis = points[index + 1] - points[index]
            section["orientation"] = list(map(float, 2.5 * axis))
        if index % 5 == 0:
            section["shear_factor_y"] = 0.5
            section["c_y"] = 0.2
            section["torsion_modulus"] = 1.0e-3
        if index % 7 == 0:
            section = {
                key: np.float64(value) if isinstance(value, float) else value
                for key, value in section.items()
            }
        if index % 11 == 0:
            section["web_direction"] = (0.0, 1.0, 0.0)
        if index % 13 == 0:
            section["c_z"] = 0
            section.pop("J", None)
        material = "alu" if index % 4 == 0 else "steel"
        model.add_element(
            index + 1,
            BeamElement(index + 1, [index + 1, index + 2], material, section),
        )
    model.add_element(
        count + 1,
        BeamElement(count + 1, [count + 1, count + 2], "steel", dict(SECTION)),
    )
    model.add_boundary_condition(
        BoundaryCondition("bc1", [1], {name: 0.0 for name in DOFS})
    )
    load_case = LoadCase("lc")
    for node_id in range(5, count + 2, 9):
        load_case.add_nodal_load(node_id, [1e3, -2e3, 3e3, 10.0, -5.0, 2.0])
    return model, load_case


@contextmanager
def _scalar_only() -> Iterator[None]:
    disable_exact_beam_batch("test: scalar reference")
    try:
        yield
    finally:
        enable_exact_beam_batch()


def _bits(value: Any) -> bytes:
    array = np.asarray(value)
    return array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()


def _csr_bits(matrix: Any) -> tuple[bytes, bytes, bytes, tuple[int, int]]:
    return (
        _bits(matrix.data),
        _bits(matrix.indices),
        _bits(matrix.indptr),
        matrix.shape,
    )


def _stress_signature(stresses: dict[int, dict[str, Any]]) -> list[Any]:
    signature = []
    for element_id, record in stresses.items():
        entries = []
        for key, value in record.items():
            if isinstance(value, str):
                entries.append((key, "str", value))
            else:
                entries.append((key, type(value).__name__, _bits(value)))
        signature.append(
            (
                type(element_id).__name__,
                element_id,
                tuple(entries),
                record.get("equivalent_stress") is record.get("von_mises"),
            )
        )
    return signature


def _comparable_info(info: dict[str, Any]) -> dict[str, Any]:
    made = dict(info)
    made["assembly_time"] = None
    made["element_times"] = list(info["element_times"])
    return made


def _batches() -> tuple[int, int]:
    status = exact_beam_batch_status()
    return status["stiffness_batches"], status["recovery_batches"]


def test_probe_passes_and_batches_run() -> None:
    model, load_case = _random_model()
    before = _batches()
    _matrix, info = assemble_stiffness_matrix(model)
    u, _info = solve_linear(model, load_case)
    compute_stresses(model, u)
    after = _batches()
    status = exact_beam_batch_status()
    assert status["probe"] is True
    assert status["disabled_reason"] is None
    assert after[0] - before[0] == 2
    assert after[1] - before[1] == 1
    assert info["num_elements"] == 61
    assert "exact_beam_stiffness_batch" not in info["diagnostics"]


def test_stiffness_bitwise_matches_scalar_methods_and_scalar_loop() -> None:
    model, _load_case = _random_model()
    before = _batches()[0]
    matrix, info = assemble_stiffness_matrix(model)
    assert _batches()[0] == before + 1
    batched_element = {
        element_id: element._stiffness_matrix
        for element_id, element in model.mesh.elements.items()
    }
    mesh = model.mesh
    for element_id, element in mesh.elements.items():
        stored = batched_element[element_id]
        assert type(stored) is np.ndarray
        assert stored.flags.c_contiguous and stored.flags.owndata
        oracle = element.compute_stiffness_matrix(
            mesh, model.get_material(element.material_name)
        )
        assert _bits(stored) == _bits(oracle), element_id
    with _scalar_only():
        scalar_matrix, scalar_info = assemble_stiffness_matrix(model)
    assert _csr_bits(matrix) == _csr_bits(scalar_matrix)
    assert _comparable_info(info) == _comparable_info(scalar_info)
    assert list(info) == list(scalar_info)


@pytest.mark.parametrize("return_global", [False, True])
@pytest.mark.parametrize("selection", [None, "every-third"])
def test_recovery_bitwise_matches_scalar_loop(
    return_global: bool,
    selection: str | None,
) -> None:
    model, load_case = _random_model()
    u, _info = solve_linear(model, load_case)
    element_ids = (
        None if selection is None else list(model.mesh.elements)[::3]
    )
    before = _batches()[1]
    batched = compute_stresses(
        model, u, return_global=return_global, element_ids=element_ids
    )
    assert _batches()[1] == before + 1
    with _scalar_only():
        scalar = compute_stresses(
            model, u, return_global=return_global, element_ids=element_ids
        )
    assert type(batched) is dict
    assert _stress_signature(batched) == _stress_signature(scalar)
    mesh = model.mesh
    for element_id, record in batched.items():
        element = mesh.elements[element_id]
        dofs = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        oracle = element.compute_stresses(
            mesh, u[dofs], model.get_material(element.material_name)
        )
        assert _stress_signature({element_id: record}) == _stress_signature(
            {element_id: oracle}
        )


def test_solve_linear_bitwise_identical() -> None:
    model, load_case = _random_model(count=80, seed=3)
    u, info = solve_linear(model, load_case)
    with _scalar_only():
        scalar_u, scalar_info = solve_linear(model, load_case)
    assert _bits(u) == _bits(scalar_u)
    assert info.get("convergence_info", {}).get("status") == scalar_info.get(
        "convergence_info", {}
    ).get("status")


def _counting(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[int]:
    original = owner.__dict__[name]
    calls = [0]

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


@pytest.mark.parametrize(
    "owner_name, name",
    [
        ("BeamElement", "_local_linear_stiffness"),
        ("BeamElement", "_beam_frame_and_transform"),
        ("BeamElement", "compute_stresses"),
        ("Element", "get_dof_mapping"),
        ("elements", "_beam_rotation_matrix"),
        ("elements", "_contiguous_vector_norm"),
        ("elements", "_frame_columns"),
        ("elements", "_beam_material_properties"),
    ],
)
def test_hostile_replacement_before_call_takes_scalar_loop(
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    name: str,
) -> None:
    owner = {
        "BeamElement": BeamElement,
        "Element": Element,
        "elements": elements_module,
    }[owner_name]

    def run() -> tuple[Any, ...]:
        model, load_case = _random_model(count=20)
        outcome: dict[str, Any] = {}

        def solve() -> None:
            outcome["u"] = _bits(solve_linear(model, load_case)[0])

        def recover() -> None:
            u = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
            outcome["S"] = _stress_signature(compute_stresses(model, u))

        return _outcome(solve), _outcome(recover), outcome

    calls = _counting(monkeypatch, owner, name)
    before = _batches()
    batched = run()
    assert _batches() == before
    # The replacement runs, unless the existing recovery authority rejects
    # the changed class before any element method is reached.
    assert calls[0] > 0 or batched[1][0] == "AssemblyError"
    with _scalar_only():
        scalar = run()
    # The scalar loop calls the replacement too, so results (or the existing
    # recovery-authority rejection) are identical.
    assert batched == scalar


def test_instance_attribute_shadowing_a_method_takes_scalar_loop() -> None:
    model, _load_case = _random_model(count=20)
    element = model.mesh.elements[5]
    calls = [0]
    original = BeamElement.compute_stiffness_matrix

    def shadow(mesh: Any, material: Any) -> np.ndarray:
        calls[0] += 1
        return original(element, mesh, material)

    element.compute_stiffness_matrix = shadow
    before = _batches()
    _matrix, info = assemble_stiffness_matrix(model)
    assert _batches() == before
    assert calls[0] == 1
    assert "exact_beam_stiffness_batch" not in info["diagnostics"]


def test_material_subclass_takes_scalar_loop() -> None:
    class SubMaterial(Material):
        pass

    model, load_case = _random_model(count=20)
    model.materials["steel"] = SubMaterial("steel", 2.1e11, 0.3, 7850.0)
    before = _batches()
    u, _info = solve_linear(model, load_case)
    compute_stresses(model, u)
    assert _batches() == before


def test_model_subclass_takes_scalar_loop() -> None:
    class SubModel(FEModel):
        pass

    model, load_case = _random_model(count=20)
    sub = SubModel.__new__(SubModel)
    sub.__dict__.update(model.__dict__)
    before = _batches()
    with _scalar_only():
        reference = assemble_stiffness_matrix(model)[0]
    matrix, info = assemble_stiffness_matrix(sub)
    assert _batches() == before
    assert "exact_beam_stiffness_batch" not in info["diagnostics"]
    assert _csr_bits(matrix) == _csr_bits(reference)


def test_mixed_shell_model_and_small_models_take_scalar_loop() -> None:
    model, load_case = _random_model(count=20)
    model.add_node(900, 0.0, 0.0, 50.0)
    model.add_node(901, 1.0, 0.0, 50.0)
    model.add_node(902, 1.0, 1.0, 50.0)
    model.add_node(903, 0.0, 1.0, 50.0)
    model.add_element(
        900,
        create_shell_element(900, [900, 901, 902, 903], "steel", thickness=0.01),
    )
    small, small_load = _random_model(count=MIN_BATCH_ELEMENTS - 2)
    before = _batches()
    for candidate, candidate_load in ((model, load_case), (small, small_load)):
        u, _info = solve_linear(candidate, candidate_load)
        compute_stresses(candidate, u)
    assert _batches() == before


def test_equivalent_lambda_getter_takes_scalar_loop() -> None:
    model, _load_case = _random_model(count=20)
    before = _batches()
    _matrix, info = matrix_assembly._assemble_element_matrix(
        model,
        "stiffness",
        lambda element, mesh, material: element.compute_stiffness_matrix(
            mesh, material
        ),
    )
    assert _batches() == before
    assert "exact_beam_stiffness_batch" not in info["diagnostics"]


def _outcome(call: Any) -> tuple[str, str]:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
    except Exception as error:  # noqa: BLE001 - outcome comparison
        return type(error).__name__, str(error)
    return "ok", repr(sorted({str(item.category) for item in caught}))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-default-material",
        "nan-coordinate",
        "overflowing-area",
        "zero-length",
        "orthotropic",
        "offsets",
    ],
)
def test_irregular_inputs_reject_or_pass_exactly_as_scalar_loop(
    mutation: str,
) -> None:
    def build() -> tuple[FEModel, LoadCase]:
        model, load_case = _random_model(count=20)
        if mutation == "missing-default-material":
            del model.materials["default"]
        elif mutation == "nan-coordinate":
            model.mesh.nodes[7].x = float("nan")
        elif mutation == "overflowing-area":
            model.mesh.elements[4]._A = 1.0e305
        elif mutation == "zero-length":
            model.mesh.nodes[9].x = model.mesh.nodes[8].x
            model.mesh.nodes[9].y = model.mesh.nodes[8].y
            model.mesh.nodes[9].z = model.mesh.nodes[8].z
        elif mutation == "orthotropic":
            model.add_orthotropic_material(
                "steel",
                2.1e11,
                1.0e11,
                1.0e11,
                0.3,
                0.3,
                0.3,
                8.0e10,
                8.0e10,
                4.0e10,
            )
            for element in model.mesh.elements.values():
                element.cross_section["torsional_rigidity"] = 1.0e3
        elif mutation == "offsets":
            model.mesh.elements[6].set_linear_nodal_offsets(
                [[0.0, 0.0, 0.1], [0.0, 0.05, 0.1]]
            )
        return model, load_case

    def run(model: FEModel, load_case: LoadCase) -> tuple[Any, ...]:
        outcome = {}

        def stiffness() -> None:
            outcome["K"] = _csr_bits(assemble_stiffness_matrix(model)[0])

        def recover() -> None:
            u = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
            outcome["S"] = _stress_signature(compute_stresses(model, u))

        return _outcome(stiffness), _outcome(recover), outcome

    batched = run(*build())
    with _scalar_only():
        scalar = run(*build())
    assert batched == scalar


def test_existing_hostile_state_rejected_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hostile() -> tuple[str, str]:
        model, _load_case = _random_model(count=20)
        published = matrix_assembly._assemble_element_matrix_under_lease
        monkeypatch.setattr(
            matrix_assembly,
            "_assemble_element_matrix_under_lease",
            lambda *args, **kwargs: published(*args, **kwargs),
        )
        try:
            return _outcome(lambda: assemble_stiffness_matrix(model))
        finally:
            monkeypatch.undo()

    batched = hostile()
    with _scalar_only():
        scalar = hostile()
    assert batched == scalar
    assert batched[0] != "ok"


class _GuardFailure(AssemblyError):
    pass


def _failing_assembly_guard(fail_on: str, seen: list[str]) -> Any:
    def guard(model: Any, *, context: str, final: bool = False) -> None:
        del model, final
        seen.append(context)
        if fail_on in context:
            raise _GuardFailure(f"{context} found incompatible qualified shell authority")

    return guard


def _assemble_with(model: Any, guard: Any, *, scalar: bool) -> None:
    if scalar:
        with _scalar_only():
            matrix_assembly._assemble_element_matrix_under_lease(
                model,
                "stiffness",
                matrix_assembly._standard_element_stiffness,
                guard,
            )
    else:
        matrix_assembly._assemble_element_matrix_under_lease(
            model,
            "stiffness",
            matrix_assembly._standard_element_stiffness,
            guard,
        )


def test_stiffness_bracket_uses_first_and_last_scalar_guard_contexts() -> None:
    # The bracket keeps the scalar loop's first per-element guard (before any
    # element is read) and its final guard, with identical context strings.
    model, _load_case = _random_model(count=20)
    scalar_seen: list[str] = []
    _assemble_with(model, _failing_assembly_guard("never", scalar_seen), scalar=True)
    assert sum("material observation" in c for c in scalar_seen) == 2 * 21
    batch_seen: list[str] = []
    _assemble_with(model, _failing_assembly_guard("never", batch_seen), scalar=False)
    assert batch_seen == [scalar_seen[0], scalar_seen[1], scalar_seen[-1]]
    assert batch_seen == [
        "stiffness assembly activity",
        "stiffness assembly material observation",
        "stiffness assembly completed elements",
    ]


@pytest.mark.parametrize(
    "broken_from", ["material observation", "completed elements"]
)
def test_stiffness_guard_failure_matches_scalar_path(broken_from: str) -> None:
    # A guard whose authority is broken from a stage on (single-threaded: the
    # state cannot change inside the bracket) fails both paths with the same
    # exception type and message, and leaves the same element side effects.
    outcomes = []
    for scalar in (True, False):
        model, _load_case = _random_model(count=20)
        seen: list[str] = []
        broken = [False]

        def guard(model: Any, *, context: str, final: bool = False) -> None:
            seen.append(context)
            if broken_from in context:
                broken[0] = True
            if broken[0]:
                raise _GuardFailure(
                    f"{context} found incompatible qualified shell authority"
                )

        with pytest.raises(_GuardFailure) as caught:
            _assemble_with(model, guard, scalar=scalar)
        stored = [
            None if e._stiffness_matrix is None else e._stiffness_matrix.tobytes()
            for e in model.mesh.elements.values()
        ]
        outcomes.append((type(caught.value), str(caught.value), stored))
    assert outcomes[0] == outcomes[1]


def test_recovery_bracket_uses_first_and_last_scalar_guard_stages() -> None:
    model, _load_case = _random_model(count=20)
    u = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
    sequences = []
    for scalar in (True, False):
        seen: list[str] = []

        def guard(*, stage: str) -> None:
            seen.append(stage)

        if scalar:
            with _scalar_only():
                _compute_stresses_under_lease(model, u, qualified_runtime_guard=guard)
        else:
            _compute_stresses_under_lease(model, u, qualified_runtime_guard=guard)
        sequences.append(seen)
    scalar_seen, batch_seen = sequences
    observations = [s for s in scalar_seen if "observation for element" in s]
    assert len(observations) == 2 * len(model.mesh.elements)
    assert batch_seen == (
        scalar_seen[:2] + [observations[0], observations[-1]]
    )


def test_recovery_guard_failure_matches_scalar_path() -> None:
    model, _load_case = _random_model(count=20)
    u = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
    outcomes = []
    for scalar in (True, False):

        def guard(*, stage: str) -> None:
            if "observation" in stage:
                raise _GuardFailure(stage)

        with pytest.raises(_GuardFailure) as caught:
            if scalar:
                with _scalar_only():
                    _compute_stresses_under_lease(
                        model, u, qualified_runtime_guard=guard
                    )
            else:
                _compute_stresses_under_lease(
                    model, u, qualified_runtime_guard=guard
                )
        outcomes.append((type(caught.value), str(caught.value)))
    assert outcomes[0] == outcomes[1]


def test_multi_chunk_batch_is_bitwise_identical() -> None:
    model, _load_case = _random_model(count=CHUNK_ELEMENTS + 36, seed=5)
    mesh = model.mesh
    before = _batches()
    _matrix, info = assemble_stiffness_matrix(model)
    assert info["num_elements"] > CHUNK_ELEMENTS
    for element in mesh.elements.values():
        stored = element._stiffness_matrix
        oracle = element.compute_stiffness_matrix(
            mesh, model.get_material(element.material_name)
        )
        assert _bits(stored) == _bits(oracle)
    u = np.random.default_rng(9).normal(size=mesh.dof_manager.total_dofs) * 1.0e-3
    batched = compute_stresses(model, u)
    assert _batches() == (before[0] + 1, before[1] + 1)
    with _scalar_only():
        scalar = compute_stresses(model, u)
    assert _stress_signature(batched) == _stress_signature(scalar)


_SPOT_CHECK_SCRIPT = r"""
import sys
import numpy as np
sys.path.insert(0, {tests!r})
import test_exact_beam_batch as helpers
from anymaterial import beam_material_properties
from anyfesolver.assembly import compute_stresses
from anyfesolver.exact_beam_batch import exact_beam_batch_status
from anyfesolver.matrix_assembly import assemble_stiffness_matrix

kind = {kind!r}
model, _load_case = helpers._random_model(count=20)
u = np.linspace(-1.0e-3, 1.0e-3, model.mesh.dof_manager.total_dofs)
with helpers._scalar_only():
    reference_k = helpers._csr_bits(assemble_stiffness_matrix(model)[0])
    reference_s = helpers._stress_signature(compute_stresses(model, u))
# Warm the batch: the process probe runs and passes.
assemble_stiffness_matrix(model)
compute_stresses(model, u)
assert exact_beam_batch_status()["probe"] is True
# Perturb the compliance seen by the next two material reductions only: the
# two materials of the batch.  The scalar spot check sees the true values.
namespace = beam_material_properties.__globals__
original = namespace["elastic_compliance_matrix"]
remaining = [2]

def perturbed(material):
    matrix = np.array(original(material), dtype=float)
    if remaining[0] > 0:
        remaining[0] -= 1
        matrix[0, 0] = np.nextafter(matrix[0, 0], np.inf)
    return matrix

namespace["elastic_compliance_matrix"] = perturbed
if kind == "stiffness":
    result = helpers._csr_bits(assemble_stiffness_matrix(model)[0])
    expected = reference_k
else:
    result = helpers._stress_signature(compute_stresses(model, u))
    expected = reference_s
namespace["elastic_compliance_matrix"] = original
status = exact_beam_batch_status()
assert result == expected, "spot-check fallback changed the result"
assert status["disabled_reason"] == kind + " spot check mismatch", status
# Disabled for the process: later calls take the scalar loop.
before = status["stiffness_batches"], status["recovery_batches"]
assemble_stiffness_matrix(model)
compute_stresses(model, u)
after = exact_beam_batch_status()
assert (after["stiffness_batches"], after["recovery_batches"]) == before
print("SPOT-CHECK-OK")
"""


@pytest.mark.parametrize("kind", ["stiffness", "recovery"])
def test_spot_check_mismatch_falls_back_and_disables(kind: str) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    import anyfesolver

    tests = str(Path(__file__).resolve().parent)
    source = str(Path(anyfesolver.__file__).resolve().parents[1])
    environment = dict(os.environ)
    # The child sees exactly the import path of this test process.
    environment["PYTHONPATH"] = os.pathsep.join(
        [source, *(entry for entry in sys.path if entry)]
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SPOT_CHECK_SCRIPT.format(tests=tests, kind=kind)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "SPOT-CHECK-OK" in completed.stdout


# ----------------------------------------------------------------------
# Codex review round 3 reproductions (each failed on the pre-fix tree).
# ----------------------------------------------------------------------


def _disjoint_model(count: int = 8) -> FEModel:
    """``count`` disjoint unit beams along X with sequential ids."""

    model = FEModel("beam-batch-disjoint")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    for index in range(count):
        first = 2 * index + 1
        model.add_node(first, 2.0 * index, 0.0, 0.0)
        model.add_node(first + 1, 2.0 * index + 1.0, 0.0, 0.0)
        model.add_element(
            index + 1,
            BeamElement(index + 1, [first, first + 1], "steel", dict(SECTION)),
        )
    return model


def _stored_matrices(model: FEModel) -> list[Any]:
    return [
        None
        if dict.get(vars(element), "_stiffness_matrix") is None
        else _bits(vars(element)["_stiffness_matrix"])
        for element in dict.values(model.mesh.elements)
    ]


def _both_paths(run: Any) -> tuple[Any, Any, tuple[int, int], tuple[int, int]]:
    """``run()`` on the batch-enabled path, then on the scalar path."""

    before = _batches()
    batched = run()
    after = _batches()
    with _scalar_only():
        scalar = run()
    return batched, scalar, before, after


def _stiffness_and_recovery(prepare: Any, count: int = 8) -> Any:
    """Outcomes of stiffness assembly and recovery after ``prepare(model)``."""

    def run() -> tuple[Any, ...]:
        model = _disjoint_model(count)
        u = np.arange(model.mesh.dof_manager.total_dofs, dtype=float) * 1.0e-3
        undo = prepare(model)
        outcome: dict[str, Any] = {}
        try:

            def stiffness() -> None:
                outcome["K"] = _csr_bits(assemble_stiffness_matrix(model)[0])

            stiffness_outcome = _outcome(stiffness)
            stored = _stored_matrices(model)

            def recover() -> None:
                outcome["S"] = _stress_signature(compute_stresses(model, u))

            recovery_outcome = _outcome(recover)
        finally:
            if undo is not None:
                undo()
        return stiffness_outcome, recovery_outcome, stored, outcome

    return run


@pytest.mark.parametrize("mapping_name", ["nodes", "elements"])
@pytest.mark.parametrize("method", ["get", "items", "keys", "values"])
def test_mapping_instance_shadow_runs_as_in_scalar_path(
    mapping_name: str, method: str
) -> None:
    # Codex r3 P1: ``model.mesh.nodes.get`` shadowed on the instance; the
    # batch read the raw dict while its endpoint checks called the shadow.
    def prepare(model: FEModel) -> None:
        mapping = vars(model.mesh)[mapping_name]
        original = getattr(dict, method)

        def shadow(*args: Any) -> Any:
            if method == "get" and args and args[0] == 8:
                return None
            result = original(mapping, *args)
            if method in {"items", "keys", "values"}:
                # A filtering wrapper: drop the fourth entry.
                return [entry for position, entry in enumerate(result) if position != 3]
            return result

        object.__setattr__(mapping, method, shadow)
        return None

    batched, scalar, before, after = _both_paths(_stiffness_and_recovery(prepare))
    assert batched == scalar
    assert after == before


@pytest.mark.parametrize(
    "method",
    ["__iter__", "__len__", "__contains__", "__getitem__", "get", "items", "keys", "values"],
)
def test_replaced_state_mapping_method_runs_as_in_scalar_path(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    # Codex r3 P2: a replaced ``_QualifiedStateMapping.__iter__`` was invoked
    # by recovery admission before any authority check.
    from anyfesolver.fe_core import _QualifiedStateMapping

    targets: list[Any] = []
    original = getattr(dict, method)

    def replaced(self: Any, *args: Any) -> Any:
        if any(self is target for target in targets):
            raise RuntimeError(f"nodes {method}")
        return original(self, *args)

    def prepare(model: FEModel) -> Any:
        targets[:] = [vars(model.mesh)["nodes"]]
        monkeypatch.setattr(_QualifiedStateMapping, method, replaced)
        return monkeypatch.undo

    batched, scalar, before, after = _both_paths(_stiffness_and_recovery(prepare))
    assert batched == scalar
    assert after == before


def test_oversized_dof_fails_and_leaves_caches_as_scalar_path() -> None:
    # Codex r3 P2: DOF 2**100; scalar stiffness raises in the sparsity kernel
    # before any element cache is written.
    def prepare(model: FEModel) -> None:
        model.mesh.nodes[8].dofs[0] = 2**100

    batched, scalar, _before, _after = _both_paths(_stiffness_and_recovery(prepare))
    assert batched == scalar
    assert batched[0][0] == "OverflowError"


@pytest.mark.parametrize("bad_dof", [False, True])
def test_overflowing_displacement_fails_as_scalar_path(bad_dof: bool) -> None:
    # Codex r3 P2: with over="raise", scalar recovery stops at element 1 with
    # FloatingPointError before it reaches the oversized DOF of element 4.
    def run() -> tuple[str, str]:
        model = _disjoint_model()
        if bad_dof:
            model.mesh.nodes[8].dofs[0] = 2**100
        u = np.arange(model.mesh.dof_manager.total_dofs, dtype=float) * 1.0e-3
        u[6] = 1.0e200
        with np.errstate(over="raise"):
            return _outcome(lambda: compute_stresses(model, u))

    batched, scalar, _before, _after = _both_paths(run)
    assert batched == scalar
    assert batched[0] == "FloatingPointError"


@pytest.mark.parametrize("dof", [-1, 96, 2**53, 2**100])
def test_out_of_range_dof_takes_scalar_path(dof: int) -> None:
    def prepare(model: FEModel) -> None:
        model.mesh.nodes[8].dofs[0] = dof

    batched, scalar, before, after = _both_paths(_stiffness_and_recovery(prepare))
    assert batched == scalar
    assert after == before


def test_successful_stiffness_info_equals_scalar_info() -> None:
    # Codex r3 P2: the batch added ``diagnostics["exact_beam_stiffness_batch"]``.
    model = _disjoint_model()
    before = _batches()
    matrix, info = assemble_stiffness_matrix(model)
    assert _batches()[0] == before[0] + 1
    with _scalar_only():
        scalar_matrix, scalar_info = assemble_stiffness_matrix(model)
    assert _csr_bits(matrix) == _csr_bits(scalar_matrix)
    timing_free = [
        {**item, "assembly_time": None, "element_times": list(item["element_times"])}
        for item in (info, scalar_info)
    ]
    assert timing_free[0] == timing_free[1]
    assert list(info) == list(scalar_info)
    assert list(info["diagnostics"]) == list(scalar_info["diagnostics"])


@pytest.mark.parametrize("kind", ["stiffness", "recovery"])
def test_material_reduction_runs_per_element_in_element_order(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # Codex r3 P1: the batch reduced each material once, so a stateful
    # dependency of ``_beam_material_properties`` changed an interior
    # element without reaching the probe or the endpoint checks.
    from anymaterial import beam_material_properties

    namespace = beam_material_properties.__globals__
    original = namespace["elastic_compliance_matrix"]

    def run() -> tuple[Any, ...]:
        model = _disjoint_model()
        model.add_material("own", 2.0e11, 0.3, density=7850.0)
        own = model.materials["own"]
        for element_id in (4, 5):
            model.mesh.elements[element_id].material_name = "own"
        seen = [0]
        per_reduction: list[int] = []

        def stateful(material: Any) -> Any:
            matrix = np.array(original(material), dtype=float)
            if material is own:
                seen[0] += 1
                # Every reduction of ``own`` after its first is perturbed.
                if per_reduction and seen[0] > per_reduction[0]:
                    matrix[0, 0] = 2.0 * matrix[0, 0]
            return matrix

        monkeypatch.setitem(namespace, "elastic_compliance_matrix", stateful)
        try:
            # Dependency calls per reduction, measured on the scalar function.
            elements_module._beam_material_properties(own, {})
            per_reduction.append(seen[0])
            seen[0] = 0
            assert per_reduction[0] > 0
            u = np.arange(model.mesh.dof_manager.total_dofs, dtype=float) * 1.0e-3
            if kind == "stiffness":
                return _csr_bits(assemble_stiffness_matrix(model)[0]), seen[0]
            return _stress_signature(compute_stresses(model, u)), seen[0]
        finally:
            monkeypatch.setitem(namespace, "elastic_compliance_matrix", original)

    run()  # warm the process probe outside the comparison
    batched, scalar, before, after = _both_paths(run)
    assert batched[0] == scalar[0]


@pytest.mark.parametrize(
    "module_name, name, replacement",
    [
        ("anyfesolver.s3_v2c_fast_assembly", "v2c_fast_candidate", "true"),
        ("anyfesolver.s3_v2d_fast_assembly", "v2d_fast_candidate", "true"),
        ("anyfesolver.s3_reference_batch", "reference_s3_candidate", "true"),
        ("anyfesolver.elements", "ShellElement", "beam-class"),
        ("anyfesolver.e4_pl_element", "FORMULATION_ID", "empty-id"),
        ("anyfesolver.e4_pl_element", "QualifiedE4PLShellElement", "beam-class"),
    ],
)
def test_classification_dispatch_matches_scalar_path(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    name: str,
    replacement: str,
) -> None:
    # The scalar classification loop dispatches through these module
    # attributes for every element; the batch must see the same state.
    import importlib

    module = importlib.import_module(module_name)
    counter = [0]

    def true_counter(*args: Any) -> bool:
        counter[0] += 1
        return True

    value = {
        "true": true_counter,
        "beam-class": BeamElement,
        "empty-id": "",
    }[replacement]

    def prepare(model: FEModel) -> Any:
        monkeypatch.setattr(module, name, value)
        return monkeypatch.undo

    def run() -> Any:
        counter[0] = 0
        result = _stiffness_and_recovery(prepare)()
        return result, counter[0]

    batched, scalar, _before, _after = _both_paths(run)
    assert batched == scalar



# ----------------------------------------------------------------------
# Codex review round 4 reproductions (in scope under the owner's trust
# boundary of 2026-11-24; each failed on cf30b7545 unless noted).
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, name, inner_name",
    [
        ("anyfesolver.s3_v2c_fast_assembly", "v2c_fast_candidate", "_candidate"),
        ("anyfesolver.s3_v2d_fast_assembly", "v2d_fast_candidate", "_candidate"),
        (
            "anyfesolver.s3_reference_batch",
            "reference_s3_candidate",
            "_is_exact_qualified_s3",
        ),
    ],
)
def test_replaced_shell_predicate_runs_as_in_scalar_path(
    monkeypatch: pytest.MonkeyPatch, module_name: str, name: str, inner_name: str
) -> None:
    # Codex r4 P2: a fresh predicate defined in its own module with the
    # original global names passed the structural check; scalar raises.
    import importlib

    module = importlib.import_module(module_name)
    namespace = vars(module)
    source = f"def {name}(element):\n    {inner_name}(element)\n    raise element\n"
    scratch: dict[str, Any] = {}
    # The fresh function's globals are the module namespace itself.
    exec(compile(source, module.__file__, "exec"), namespace, scratch)
    fresh = scratch[name]

    def prepare(model: FEModel) -> Any:
        monkeypatch.setitem(namespace, name, fresh)
        return monkeypatch.undo

    batched, scalar, before, after = _both_paths(_stiffness_and_recovery(prepare))
    assert batched == scalar
    assert batched[0][0] == "TypeError"
    assert after[0] == before[0]


def test_shell_predicate_name_methods_are_not_invoked() -> None:
    # Codex r4 P2: the structural check compared the genuine predicate's
    # ``__name__``, whose ``str`` subclass logged ``__ne__``/``__eq__``.
    from anyfesolver import s3_v2c_fast_assembly as module

    log: list[str] = []

    class LoggingName(str):
        def __ne__(self, other: object) -> bool:
            log.append("ne")
            return str.__ne__(self, other)

        def __eq__(self, other: object) -> bool:
            log.append("eq")
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    function = module.v2c_fast_candidate
    original = function.__name__
    function.__name__ = LoggingName(original)
    try:

        def run() -> Any:
            del log[:]
            result = _stiffness_and_recovery(lambda model: None)()
            return result, list(log)

        batched, scalar, before, after = _both_paths(run)
    finally:
        function.__name__ = original
    assert batched == scalar
    assert batched[1] == []
    assert after[0] == before[0] + 1


def test_replaced_observation_guard_factory_runs_as_in_scalar_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex r4 P2: the bracket called the raw lease guard instead of the
    # guard returned by ``assembly._qualified_recovery_observation_guard``.
    import anyfesolver.assembly as assembly_module

    original = assembly_module._qualified_recovery_observation_guard

    def factory(guard: Any) -> Any:
        delegate = original(guard)

        def observe(*, stage: str, element: Any = None, material: Any = None) -> None:
            if stage == "stress material observation for element 4":
                raise RuntimeError("fourth beam observation")
            delegate(stage=stage, element=element, material=material)

        return observe

    def run() -> tuple[str, str]:
        model = _disjoint_model()
        u = np.arange(model.mesh.dof_manager.total_dofs, dtype=float) * 1.0e-3
        monkeypatch.setattr(
            assembly_module, "_qualified_recovery_observation_guard", factory
        )
        try:
            return _outcome(lambda: compute_stresses(model, u))
        finally:
            monkeypatch.undo()

    batched, scalar, before, after = _both_paths(run)
    assert batched == scalar == ("RuntimeError", "fourth beam observation")
    assert after == before


def _direct_assembly(model: FEModel, **kernels: Any) -> Any:
    """The implementation called directly, with its kwdefaults or ``kernels``."""

    def guard(model: Any, *, context: str, final: bool = False) -> None:
        del model, context, final

    return matrix_assembly._assemble_element_matrix_under_lease_impl(
        model,
        "stiffness",
        matrix_assembly._standard_element_stiffness,
        guard,
        None,
        **kernels,
    )


@pytest.mark.parametrize("kind", ["check", "symmetry", "symmetry-defaults"])
def test_replaced_matrix_check_kernel_runs_as_in_scalar_path(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # Codex r4 P2 (ANYfeSolver part): the batch replaced the scalar loop's
    # per-element ``_check_matrix_kernel``/``_relative_symmetry_kernel``
    # calls by its own tests, so a different kernel was not honoured.
    check = matrix_assembly._check_element_matrix_shape
    symmetry = matrix_assembly._relative_symmetry_error
    dense_calls = [0]

    def raising_check(element_id: int, name: str, matrix: Any, size: int) -> Any:
        if element_id == 4:
            raise AssemblyError("element-4 matrix check")
        return check(element_id, name, matrix, size)

    def asymmetric_fourth(matrix: Any) -> float:
        if isinstance(matrix, np.ndarray):
            dense_calls[0] += 1
            if dense_calls[0] == 4:
                return 1.0
        return symmetry(matrix)

    issparse, sparse_norm = symmetry.__defaults__

    def raising_issparse(matrix: Any) -> bool:
        if isinstance(matrix, np.ndarray):
            dense_calls[0] += 1
            if dense_calls[0] == 4:
                raise LookupError("element-4 issparse")
        return issparse(matrix)

    def run() -> Any:
        dense_calls[0] = 0
        model = _disjoint_model()
        if kind == "check":
            kernels = {"_check_matrix_kernel": raising_check}
        elif kind == "symmetry":
            kernels = {"_relative_symmetry_kernel": asymmetric_fourth}
        else:
            kernels = {}
            monkeypatch.setattr(
                symmetry, "__defaults__", (raising_issparse, sparse_norm)
            )
        try:
            outcome = _outcome(lambda: _direct_assembly(model, **kernels))
        finally:
            monkeypatch.undo()
        return outcome, _stored_matrices(model)

    batched, scalar, before, after = _both_paths(run)
    assert batched == scalar
    assert batched[0][0] in {"AssemblyError", "LookupError"}
    assert after[0] == before[0]


def test_direct_implementation_call_with_default_kernels_still_batches() -> None:
    model = _disjoint_model()
    before = _batches()
    matrix, _info = _direct_assembly(model)
    assert _batches()[0] == before[0] + 1
    with _scalar_only():
        scalar_matrix, _scalar_info = _direct_assembly(_disjoint_model())
    assert _csr_bits(matrix) == _csr_bits(scalar_matrix)


def test_replaced_post_observation_guard_is_called_as_in_scalar_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex r4 (endpoint-check call counts): ``_elastic_symmetry`` calls the
    # elements-module ``_run_post_observation_guard`` once per material
    # reduction; the endpoint checks added two calls to the scalar count.
    original = elements_module._run_post_observation_guard
    calls = [0]

    def counting(post_observation: Any) -> None:
        calls[0] += 1
        original(post_observation)

    def run() -> Any:
        calls[0] = 0
        monkeypatch.setattr(elements_module, "_run_post_observation_guard", counting)
        try:
            result = _stiffness_and_recovery(lambda model: None)()
        finally:
            monkeypatch.undo()
        return result, calls[0]

    _stiffness_and_recovery(lambda model: None)()  # warm the process probe
    batched, scalar, _before, _after = _both_paths(run)
    assert batched == scalar
    assert scalar[1] == 2 * 8


@pytest.mark.parametrize("stored", ["none", "empty", "zeros"])
def test_any_stored_linear_nodal_offsets_take_scalar_path(stored: str) -> None:
    # Regression guard (already held on cf30b7545): the default rigid arm is
    # taken only when the attribute is absent (AttributeError, 30bcd9bf5);
    # any stored value takes the scalar loop.
    value = {"none": None, "empty": (), "zeros": np.zeros((2, 3))}[stored]

    def prepare(model: FEModel) -> None:
        vars(model.mesh.elements[5])["_linear_nodal_offsets"] = value

    batched, scalar, before, after = _both_paths(_stiffness_and_recovery(prepare))
    assert batched == scalar
    assert after == before



# ----------------------------------------------------------------------
# Codex review round 6 reproductions (finding 3; each failed on c95d1f20d).
# A declined batch must not touch hostile input before deciding, and must
# leave no side effect before it has decided to succeed.
# ----------------------------------------------------------------------


def _zero_length_fifth_member(model: FEModel) -> None:
    # Element 5 joins nodes 9 and 10; a zero-length member makes the batch
    # decline after admission, while the scalar loop returns zeros for it.
    node = model.mesh.nodes[10]
    node.x, node.y, node.z = model.mesh.nodes[9].x, 0.0, 0.0


class _RaisingGetitemCache(dict):
    def __getitem__(self, key: Any) -> Any:
        raise RuntimeError("hostile sparsity cache read")


def test_hostile_sparsity_cache_is_not_built_by_a_declined_batch() -> None:
    # Codex r6 finding 3(a): the batch built the empty hostile cache before
    # declining on the zero-length member; the scalar fallback then read the
    # new entry through ``__getitem__`` and raised.  Scalar-only succeeds.
    def prepare(model: FEModel) -> None:
        _zero_length_fifth_member(model)
        object.__setattr__(model.mesh, "_sparsity_cache", _RaisingGetitemCache())

    batched, scalar, before, after = _both_paths(_stiffness_and_recovery(prepare))
    assert scalar[0][0] == "ok"
    assert batched == scalar
    assert after[0] == before[0]


def _logging_cache(log: list[str]) -> dict:
    class LoggingCache(dict):
        def __contains__(self, key: Any) -> bool:
            log.append("contains")
            return dict.__contains__(self, key)

        def __getitem__(self, key: Any) -> Any:
            log.append("getitem")
            return dict.__getitem__(self, key)

        def __setitem__(self, key: Any, value: Any) -> None:
            log.append("setitem")
            dict.__setitem__(self, key, value)

        def get(self, key: Any, default: Any = None) -> Any:
            log.append("get")
            return dict.get(self, key, default)

    return LoggingCache()


@pytest.mark.parametrize("cache_name", ["_sparsity_cache", "_topology_signature_cache"])
@pytest.mark.parametrize("zero_length", [True, False])
def test_mesh_cache_protocol_calls_match_scalar_path(
    cache_name: str, zero_length: bool
) -> None:
    # Same class as 3(a): every protocol call on a caller-held mesh cache
    # (dict subclass) happens exactly as in the scalar path, whether the
    # batch would decline (zero-length member) or not.
    log: list[str] = []

    def run() -> Any:
        log.clear()
        model = _disjoint_model()
        if zero_length:
            _zero_length_fifth_member(model)
        object.__setattr__(model.mesh, cache_name, _logging_cache(log))
        outcome = _outcome(lambda: assemble_stiffness_matrix(model))
        return outcome, list(log), _stored_matrices(model)

    batched, scalar, _before, _after = _both_paths(run)
    assert scalar[0][0] == "ok"
    assert scalar[1]
    assert batched == scalar


def test_hostile_signature_cache_is_not_read_again_after_a_decline() -> None:
    # Same class as 3(a) for the topology-signature cache: a ``get`` that
    # raises on its third call.  Scalar-only assembly calls it twice.
    calls = [0]

    class ThirdGetRaises(dict):
        def get(self, key: Any, default: Any = None) -> Any:
            calls[0] += 1
            if calls[0] >= 3:
                raise RuntimeError("hostile signature cache read")
            return dict.get(self, key, default)

    def run() -> Any:
        calls[0] = 0
        model = _disjoint_model()
        _zero_length_fifth_member(model)
        object.__setattr__(model.mesh, "_topology_signature_cache", ThirdGetRaises())
        return _outcome(lambda: assemble_stiffness_matrix(model)), calls[0]

    batched, scalar, _before, _after = _both_paths(run)
    assert scalar[0][0] == "ok" and scalar[1] == 2
    assert batched == scalar


def _second_hash_raises() -> Any:
    class SecondHashRaises(str):
        hashes = 0

        def __hash__(self) -> int:
            type(self).hashes += 1
            if type(self).hashes > 1:
                raise RuntimeError("material name hashed twice")
            return str.__hash__(self)

    return SecondHashRaises("steel")


def test_recovery_bracket_does_not_look_up_a_hostile_material_name() -> None:
    # Codex r6 finding 3(b): the recovery bracket looked the first beam's
    # ``material_name`` up before admission; admission declined and the
    # scalar loop's lookup was the second hash, which raised.
    def run() -> Any:
        model = _disjoint_model()
        vars(model.mesh.elements[1])["material_name"] = _second_hash_raises()
        u = np.arange(model.mesh.dof_manager.total_dofs, dtype=float) * 1.0e-3
        outcome: dict[str, Any] = {}

        def recover() -> None:
            outcome["S"] = _stress_signature(compute_stresses(model, u))

        return _outcome(recover), outcome

    batched, scalar, before, after = _both_paths(run)
    assert scalar[0][0] == "ok"
    assert batched == scalar
    assert after[1] == before[1]


def test_recovery_material_name_protocol_calls_match_scalar_path() -> None:
    # Same class as 3(b): a ``str`` subclass that logs every hash, comparison
    # and truth test; the bracket must add none before the batch declines.
    log: list[str] = []

    class LoggingName(str):
        def __hash__(self) -> int:
            log.append("hash")
            return str.__hash__(self)

        def __eq__(self, other: Any) -> bool:
            log.append("eq")
            return str.__eq__(self, other)

        def __bool__(self) -> bool:
            log.append("bool")
            return True

    def run() -> Any:
        log.clear()
        model = _disjoint_model()
        vars(model.mesh.elements[1])["material_name"] = LoggingName("steel")
        u = np.arange(model.mesh.dof_manager.total_dofs, dtype=float) * 1.0e-3
        outcome = _outcome(lambda: compute_stresses(model, u))
        return outcome, list(log)

    batched, scalar, _before, _after = _both_paths(run)
    assert scalar[0][0] == "ok"
    assert scalar[1]
    assert batched == scalar


def test_stiffness_spot_checks_write_no_element_cache_before_success() -> None:
    # Same finding, write class: the endpoint spot checks called
    # ``compute_stiffness_matrix``, which stores ``_stiffness_matrix`` on the
    # first element before the batch had decided to succeed (and restores it
    # on a decline).  An observer in the material reduction (an ANYmaterial
    # internal: trusted, not identity-checked) records which elements hold a
    # stored matrix at every reduction the batch makes.
    from anymaterial import beam_material_properties

    namespace = beam_material_properties.__globals__
    original = namespace["elastic_compliance_matrix"]
    model = _disjoint_model()
    stored_at_reduction: list[tuple[int, ...]] = []

    def observing(material: Any) -> Any:
        stored_at_reduction.append(
            tuple(
                element_id
                for element_id, element in dict.items(model.mesh.elements)
                if dict.get(vars(element), "_stiffness_matrix") is not None
            )
        )
        return original(material)

    assemble_stiffness_matrix(_disjoint_model())  # warm the process probe
    before = _batches()
    namespace["elastic_compliance_matrix"] = observing
    try:
        matrix, _info = assemble_stiffness_matrix(model)
    finally:
        namespace["elastic_compliance_matrix"] = original
    assert _batches()[0] == before[0] + 1
    assert stored_at_reduction
    assert all(stored == () for stored in stored_at_reduction), stored_at_reduction
    with _scalar_only():
        scalar_model = _disjoint_model()
        scalar_matrix, _scalar_info = assemble_stiffness_matrix(scalar_model)
    assert _csr_bits(matrix) == _csr_bits(scalar_matrix)
    assert _stored_matrices(model) == _stored_matrices(scalar_model)


def test_hostile_entry_of_an_exact_sparsity_cache_is_read_once() -> None:
    # Same class as 3(a) with an exact-dict cache (admitted) holding a valid
    # entry of a logging dict subclass: the scalar sparsity kernel reads it
    # once; a declined batch must not have read it before.
    log: list[str] = []

    class LoggingEntry(dict):
        def get(self, key: Any, default: Any = None) -> Any:
            log.append("get")
            return dict.get(self, key, default)

        def __getitem__(self, key: Any) -> Any:
            log.append("getitem")
            return dict.__getitem__(self, key)

    def run() -> Any:
        model = _disjoint_model()
        _zero_length_fifth_member(model)
        # Populate the exact caches (the batch declines on the zero-length
        # member, so the scalar loop builds them).
        assemble_stiffness_matrix(model)
        cache = vars(model.mesh)["_sparsity_cache"]
        assert type(cache) is dict
        cache["stiffness"] = LoggingEntry(cache["stiffness"])
        log.clear()
        outcome = _outcome(lambda: assemble_stiffness_matrix(model))
        return outcome, list(log)

    batched, scalar, _before, _after = _both_paths(run)
    assert scalar == (("ok", "[]"), ["get", "getitem", "getitem"])
    assert batched == scalar
