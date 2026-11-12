"""Independent regression gate for the public strict-flat S3 V2 candidate.

The reference implementation is loaded from its research path at run time so
this test does not turn it into a production dependency.  The comparison uses
only the reference's independently authored public assembly/field functions.
"""

from __future__ import annotations

import importlib.util
import itertools
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from anyfesolver.e4_pl_s3_v2_element import (
    StrictFlatLinearE4PLS3V2ShellElement,
)
from anyfesolver.fe_core import FEMesh, Material


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_v2_independent_reference.py"
)
YOUNG = 210.0e9
POISSON = 0.3
THICKNESS = 0.08
NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
GEOMETRIES = {
    "canonical": np.asarray(((0.0, 0.0), (2.0, 0.0), (0.3, 1.1))),
    "translated_skew": np.asarray(((-3.25, 2.0), (1.5, 2.0), (-1.1, 3.7))),
    "hostile_obtuse": np.asarray(((0.0, 0.0), (5.0, 0.0), (4.65, 0.72))),
    "hostile_slender": np.asarray(((0.0, 0.0), (4.2, 0.0), (0.18, 0.41))),
}
PROBE = np.asarray(
    (3, -5, 7, 11, -13, 17, -19, 23, 29, -31, 37, 41, 43, -47, 53, 59, -61, 67),
    dtype=np.float64,
) / 8192.0
VIRTUAL = np.asarray(
    (-71, 73, -79, 83, 89, -97, 101, -103, 107, 109, -113, 127, -131, 137, 139, -149, 151, 157),
    dtype=np.float64,
) / 16384.0


@lru_cache(maxsize=1)
def _reference_module() -> ModuleType:
    """Load the hash-bound research reference without importing it by name."""

    module_name = "_s3_v2_independent_reference_production_gate"
    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass decoration resolves the defining module through sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    assert module.REFERENCE_IMPLEMENTATION_ID == (
        "INDEPENDENT_S3_V2A_FLAT_DKMT_EQ12_41_V1"
    )
    return module


def _registered_case(
    coordinates_2d: np.ndarray,
    permutation: tuple[int, int, int],
    normal: np.ndarray = NORMAL,
) -> tuple[FEMesh, StrictFlatLinearE4PLS3V2ShellElement]:
    mesh = FEMesh()
    coordinates_3d = np.column_stack((coordinates_2d, np.zeros(3)))
    for node_id, coordinate in enumerate(coordinates_3d, start=1):
        mesh.add_node(node_id, *coordinate)
    element = StrictFlatLinearE4PLS3V2ShellElement(
        1,
        tuple(index + 1 for index in permutation),
        "steel",
        thickness=THICKNESS,
        reference_normal=normal,
    )
    mesh.add_element(element.element_id, element)
    return mesh, element


def _relative_inf(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected), ord=np.inf)
        / max(float(np.linalg.norm(np.asarray(expected), ord=np.inf)), 1.0)
    )


def _scaled_max(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
        / max(float(np.max(np.abs(np.asarray(expected)))), 1.0)
    )


def _numerical_rank(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    return int(np.count_nonzero(singular > singular[0] * 1.0e-9))


def _global_resultants(result: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = np.asarray(result["frame"], dtype=np.float64)[:, :2]

    def symmetric(rows: object) -> np.ndarray:
        tensors = []
        for xx, yy, xy in np.asarray(rows, dtype=np.float64):
            local = np.asarray(((xx, xy), (xy, yy)), dtype=np.float64)
            tensors.append(axes @ local @ axes.T)
        return np.asarray(tensors)

    membrane = symmetric(result["membrane_resultants"])
    bending = symmetric(result["bending_resultants"])
    shear = np.asarray(result["transverse_shear_resultants"]) @ axes.T
    return membrane, bending, shear


def _reference_resultants(reference: object, module: ModuleType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = [module.generalized_fields(reference, PROBE, station) for station in range(3)]

    def symmetric(key: str) -> np.ndarray:
        tensors = []
        for field in fields:
            xx, yy, xy = np.asarray(field[key], dtype=np.float64)
            tensors.append(((xx, xy, 0.0), (xy, yy, 0.0), (0.0, 0.0, 0.0)))
        return np.asarray(tensors)

    membrane = symmetric("N")
    bending = symmetric("M")
    shear = np.asarray(
        [(field["Q"][0], field["Q"][1], 0.0) for field in fields],
        dtype=np.float64,
    )
    return membrane, bending, shear


def _run_public_candidate_reference_gate() -> dict[str, float]:
    reference_module = _reference_module()
    section = reference_module.isotropic_generalized_section(
        YOUNG,
        POISSON,
        THICKNESS,
    )
    material = Material("steel", YOUNG, POISSON, density=7850.0)
    worst_component = 0.0
    worst_resultant = 0.0
    worst_work = 0.0

    for geometry_name, coordinates in GEOMETRIES.items():
        reference = reference_module.assemble_flat_reference(coordinates, section)
        expected_components = {
            "membrane": reference.membrane_stiffness,
            "bending": reference.bending_stiffness,
            "shear": reference.shear_stiffness,
            "physical": reference.physical_stiffness,
            "pl": reference.pl_stiffness,
            "total": reference.condensed_stiffness,
        }
        expected_resultants = _reference_resultants(reference, reference_module)
        coordinates_3d = np.column_stack((coordinates, np.zeros(3)))
        barycentric = np.asarray(
            [point for point, _weight in reference_module.TRIANGLE_RULE],
            dtype=np.float64,
        )
        expected_stations = barycentric @ coordinates_3d

        assert _numerical_rank(reference.physical_stiffness) == 9
        assert _numerical_rank(reference.pl_stiffness) == 3
        assert _numerical_rank(reference.condensed_stiffness) == 12
        assert np.linalg.matrix_rank(reference.rigid_modes) == 6

        for permutation in PERMUTATIONS:
            transport = reference_module.block_permutation(permutation)
            mesh, element = _registered_case(coordinates, permutation)
            produced = element.compute_stiffness_components(mesh, material)
            for component, expected_base in expected_components.items():
                expected = transport @ expected_base @ transport.T
                residual = _relative_inf(produced[component], expected)
                worst_component = max(worst_component, residual)
                assert residual <= 2.0e-13, (
                    geometry_name,
                    permutation,
                    component,
                    residual,
                )

            assert _numerical_rank(produced["physical"]) == 9
            assert _numerical_rank(produced["pl"]) == 3
            assert _numerical_rank(produced["total"]) == 12
            rigid = transport @ reference.rigid_modes
            rigid_scale = max(float(np.linalg.norm(produced["total"], ord=np.inf)), 1.0)
            assert float(np.linalg.norm(produced["total"] @ rigid, ord=np.inf)) <= (
                2.0e-13 * rigid_scale
            )

            displacement = transport @ PROBE
            virtual = transport @ VIRTUAL
            expected_force = transport @ (reference.condensed_stiffness @ PROBE)
            force = element.compute_internal_forces(mesh, displacement, material)
            force_residual = _scaled_max(force, expected_force)
            assert force_residual <= 3.0e-13, (
                geometry_name,
                permutation,
                "internal_force",
                force_residual,
            )
            expected_work = float(VIRTUAL @ reference.condensed_stiffness @ PROBE)
            produced_work = float(virtual @ force)
            work_residual = abs(produced_work - expected_work) / max(
                abs(expected_work), 1.0
            )
            worst_work = max(worst_work, work_residual)
            assert work_residual <= 3.0e-13, (
                geometry_name,
                permutation,
                "virtual_work",
                work_residual,
            )

            result = element.compute_variational_resultants(
                mesh,
                displacement,
                material,
            )
            produced_resultants = _global_resultants(result)
            produced_stations = np.asarray(
                result["physical_station_coordinates"], dtype=np.float64
            )
            for station, point in enumerate(expected_stations):
                produced_station = int(
                    np.argmin(np.linalg.norm(produced_stations - point, axis=1))
                )
                assert float(np.linalg.norm(produced_stations[produced_station] - point)) <= 5.0e-14
                for expected_rows, produced_rows in zip(
                    expected_resultants,
                    produced_resultants,
                    strict=True,
                ):
                    residual = _scaled_max(
                        produced_rows[produced_station],
                        expected_rows[station],
                    )
                    worst_resultant = max(worst_resultant, residual)
                    assert residual <= 3.0e-13, (
                        geometry_name,
                        permutation,
                        station,
                        residual,
                    )

        # Physical-director reversal is a separate operation from D3
        # re-numbering.  In this symmetric flat-isotropic scope it preserves
        # the stiffness, rigid kernel, and external virtual work exactly to
        # the frozen binary64 comparison budget.
        for permutation in PERMUTATIONS:
            transport = reference_module.block_permutation(permutation)
            mesh, element = _registered_case(coordinates, permutation, -NORMAL)
            reversed_components = element.compute_stiffness_components(mesh, material)
            expected = transport @ reference.condensed_stiffness @ transport.T
            residual = _relative_inf(reversed_components["total"], expected)
            worst_component = max(worst_component, residual)
            assert residual <= 2.0e-13, (
                geometry_name,
                permutation,
                "director_reversal",
                residual,
            )
            displacement = transport @ PROBE
            virtual = transport @ VIRTUAL
            reversed_force = element.compute_internal_forces(mesh, displacement, material)
            reversed_work = float(virtual @ reversed_force)
            expected_work = float(VIRTUAL @ reference.condensed_stiffness @ PROBE)
            residual = abs(reversed_work - expected_work) / max(abs(expected_work), 1.0)
            worst_work = max(worst_work, residual)
            assert residual <= 3.0e-13, (
                geometry_name,
                permutation,
                "director_reversal_work",
                residual,
            )

    # Keep the worst observed errors in assertion messages without making
    # normal test output or deterministic evidence depend on diagnostics.
    assert worst_component <= 2.0e-13, worst_component
    assert worst_resultant <= 3.0e-13, worst_resultant
    assert worst_work <= 3.0e-13, worst_work
    return {
        "component_relative_inf": worst_component,
        "resultant_scaled_max": worst_resultant,
        "work_scaled_absolute": worst_work,
    }


def test_public_candidate_matches_independent_reference_for_all_d3_numberings() -> None:
    metrics = _run_public_candidate_reference_gate()
    assert set(metrics) == {
        "component_relative_inf",
        "resultant_scaled_max",
        "work_scaled_absolute",
    }
