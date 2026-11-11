from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_v2_element as candidate_module
from anyfesolver.e4_pl_s3_v2_element import (
    StrictFlatLinearCapabilityError,
    StrictFlatLinearE4PLS3V2ShellElement,
)
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEMesh, Material


NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
ARRAY_COMPONENTS = (
    "membrane",
    "bending",
    "shear",
    "physical",
    "pl",
    "numerical",
    "hourglass",
    "total",
    "frame",
    "phi",
)


def _mesh_element(
    coordinates: np.ndarray,
    *,
    node_ids: tuple[int, int, int] = (1, 2, 3),
    element_id: int = 1,
    register: bool = True,
) -> tuple[FEMesh, StrictFlatLinearE4PLS3V2ShellElement]:
    mesh = FEMesh()
    for node_id, coordinate in enumerate(
        np.asarray(coordinates, dtype=np.float64),
        start=1,
    ):
        mesh.add_node(node_id, *coordinate)
    element = StrictFlatLinearE4PLS3V2ShellElement(
        element_id,
        node_ids,
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    if register:
        mesh.add_element(element_id, element)
    return mesh, element


def _material(elastic_modulus: float = 210.0e9) -> Material:
    return Material("steel", elastic_modulus, 0.3, density=7850.0)


def _array_records(result: Mapping[str, object]) -> dict[str, tuple[object, ...]]:
    records = {}
    for name in ARRAY_COMPONENTS:
        array = np.asarray(result[name])
        records[name] = (array.dtype.str, array.shape, array.tobytes(order="C"))
    return records


def _operator_calls(action: Callable[[], object]) -> int:
    target = StrictFlatLinearE4PLS3V2ShellElement._operators.__code__
    count = 0
    previous = sys.getprofile()

    def observe(frame: object, event: str, _argument: object) -> None:
        nonlocal count
        if event == "call" and getattr(frame, "f_code", None) is target:
            count += 1

    sys.setprofile(observe)
    try:
        action()
    finally:
        sys.setprofile(previous)
    return count


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str((Path(__file__).resolve().parents[1] / "src").resolve())
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source + (os.pathsep + inherited if inherited else "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in (
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def test_cache_has_no_callable_key_lookup_store_or_factory_surface() -> None:
    forbidden = {
        "_component_cache_info",
        "_component_cache_key",
        "_component_cache_lookup",
        "_component_cache_store",
        "_exact_component_cache_info",
        "_exact_component_cache_key",
        "_exact_component_cache_lookup",
        "_exact_component_cache_store",
        "_make_exact_component_cache",
        "_make_exact_component_compute_wrapper",
    }
    assert forbidden.isdisjoint(vars(candidate_module))
    assert forbidden.isdisjoint(
        vars(StrictFlatLinearE4PLS3V2ShellElement)
    )


def test_cold_compute_then_exact_hit_is_fresh_and_mutation_isolated() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.75, 0.125, 0.0), (0.375, 1.625, 0.0)),
        dtype=np.float64,
    )
    mesh, element = _mesh_element(coordinates)
    material = _material()
    results: list[Mapping[str, object]] = []

    assert _operator_calls(
        lambda: results.extend(
            (
                element.compute_stiffness_components(mesh, material),
                element.compute_stiffness_components(mesh, material),
            )
        )
    ) == 1
    first, second = results
    expected = _array_records(first)
    assert _array_records(second) == expected
    assert list(second) == [
        "membrane",
        "bending",
        "shear",
        "physical",
        "pl",
        "numerical",
        "hourglass",
        "total",
        "frame",
        "area",
        "phi",
        "quadrature_authority_id",
        "pl_completion_policy_id",
    ]
    assert second["pl"] is not second["numerical"]

    for name in ARRAY_COMPONENTS:
        np.asarray(first[name]).reshape(-1)[0] += 123.0
    third = element.compute_stiffness_components(mesh, material)
    assert _array_records(third) == expected
    for name in ARRAY_COMPONENTS:
        assert second[name] is not third[name]
        assert not np.shares_memory(second[name], third[name])


def test_translation_reuses_exact_content_while_one_bit_inputs_do_not() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.25, 0.0, 0.0), (0.25, 1.5, 0.0)),
        dtype=np.float64,
    )
    translated = coordinates + np.asarray((8.0, 16.0, 0.0), dtype=np.float64)
    changed = coordinates.copy()
    changed[1, 0] = np.nextafter(changed[1, 0], np.inf)
    base_mesh, base = _mesh_element(coordinates, element_id=101)
    translated_mesh, translated_element = _mesh_element(
        translated, element_id=102
    )
    changed_mesh, changed_element = _mesh_element(changed, element_id=103)
    numbered_mesh, numbered = _mesh_element(
        coordinates,
        node_ids=(2, 3, 1),
        element_id=104,
    )
    material = _material()
    results: list[Mapping[str, object]] = []

    assert _operator_calls(
        lambda: results.extend(
            (
                base.compute_stiffness_components(base_mesh, material),
                translated_element.compute_stiffness_components(
                    translated_mesh, material
                ),
                changed_element.compute_stiffness_components(changed_mesh, material),
                numbered.compute_stiffness_components(numbered_mesh, material),
                base.compute_stiffness_components(
                    base_mesh,
                    _material(np.nextafter(210.0e9, np.inf)),
                ),
            )
        )
    ) == 4
    assert _array_records(results[0]) == _array_records(results[1])


def test_scope_guard_still_runs_before_a_warm_exact_hit() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.875, 0.0, 0.0), (0.25, 1.375, 0.0)),
        dtype=np.float64,
    )
    mesh, element = _mesh_element(coordinates)
    material = _material()
    element.compute_stiffness_components(mesh, material)

    mesh.add_element(2, ShellElement(2, (1, 2, 3), "steel", thickness=0.08))
    with pytest.raises(StrictFlatLinearCapabilityError, match="mixed element"):
        element.compute_stiffness_components(mesh, material)


def test_simultaneous_cold_callers_are_exact_and_cannot_poison_each_other() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.9375, 0.0625, 0.0), (0.1875, 1.4375, 0.0)),
        dtype=np.float64,
    )
    mesh, element = _mesh_element(coordinates, register=False)
    material = _material()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: element.compute_stiffness_components(mesh, material),
                range(24),
            )
        )
    expected = _array_records(results[0])
    assert all(_array_records(result) == expected for result in results)
    for name in ARRAY_COMPONENTS:
        np.asarray(results[0][name]).reshape(-1)[0] -= 456.0
    assert _array_records(
        element.compute_stiffness_components(mesh, material)
    ) == expected


def test_two_fresh_processes_have_one_cold_compute_and_identical_hits(
    tmp_path: Path,
) -> None:
    script = r'''
import hashlib
import json
import sys
import numpy as np
from anyfesolver.e4_pl_s3_v2_element import StrictFlatLinearE4PLS3V2ShellElement
from anyfesolver.fe_core import FEMesh, Material

mesh = FEMesh()
for node_id, coordinate in enumerate(((0.0, 0.0, 0.0), (1.8125, 0.0625, 0.0), (0.3125, 1.5625, 0.0)), 1):
    mesh.add_node(node_id, *coordinate)
element = StrictFlatLinearE4PLS3V2ShellElement(1, (1, 2, 3), "steel", thickness=0.08, reference_normal=(0.0, 0.0, 1.0))
material = Material("steel", 210.0e9, 0.3, density=7850.0)
target = type(element)._operators.__code__
calls = 0
def observe(frame, event, argument):
    global calls
    if event == "call" and frame.f_code is target:
        calls += 1
sys.setprofile(observe)
first = element.compute_stiffness_components(mesh, material)
second = element.compute_stiffness_components(mesh, material)
first["total"][0, 0] += 1.0
third = element.compute_stiffness_components(mesh, material)
sys.setprofile(None)
def digest(result):
    made = hashlib.sha256()
    for name in ("membrane", "bending", "shear", "physical", "pl", "numerical", "hourglass", "total", "frame", "phi"):
        made.update(np.asarray(result[name]).tobytes(order="C"))
    return made.hexdigest().upper()
print(json.dumps({"calls": calls, "second": digest(second), "third": digest(third)}, sort_keys=True, separators=(",", ":")))
'''
    outputs = []
    for _replica in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=_subprocess_environment(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.encode("utf-8"))
    assert outputs[0] == outputs[1]
    result = json.loads(outputs[0])
    assert result["calls"] == 1
    assert result["second"] == result["third"]


def test_content_key_eviction_is_bounded_and_insertion_order_independent(
    tmp_path: Path,
) -> None:
    script = r'''
import hashlib
import inspect
import json
import sys
from anyfesolver.e4_pl_s3_v2_element import StrictFlatLinearE4PLS3V2ShellElement
from anyfesolver.fe_core import FEMesh, Material

order = range(514) if sys.argv[1] == "forward" else range(513, -1, -1)
material = Material("steel", 210.0e9, 0.3, density=7850.0)
for index in order:
    mesh = FEMesh()
    width = 1.5 + index / 4096.0
    for node_id, coordinate in enumerate(((0.0, 0.0, 0.0), (width, 0.0, 0.0), (0.125, 1.25, 0.0)), 1):
        mesh.add_node(node_id, *coordinate)
    element = StrictFlatLinearE4PLS3V2ShellElement(index + 1, (1, 2, 3), "steel", thickness=0.08, reference_normal=(0.0, 0.0, 1.0))
    element.compute_stiffness_components(mesh, material)
cache = inspect.getclosurevars(StrictFlatLinearE4PLS3V2ShellElement.compute_stiffness_components).nonlocals["entries"]
digests = sorted(hashlib.sha256(repr(key).encode("utf-8")).hexdigest() for key in cache)
print(json.dumps({"keys": hashlib.sha256("".join(digests).encode("ascii")).hexdigest(), "size": len(cache)}, sort_keys=True, separators=(",", ":")))
'''
    outputs = []
    for order in ("forward", "reverse"):
        completed = subprocess.run(
            [sys.executable, "-c", script, order],
            cwd=tmp_path,
            env=_subprocess_environment(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.encode("utf-8"))
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["size"] == 512
