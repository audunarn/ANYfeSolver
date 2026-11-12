from __future__ import annotations

import copy
import inspect
import itertools
import math
import pickle

import numpy as np
import pytest

import anyfesolver
from anyfesolver.boundary import LoadCase
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.e4_pl_s3_v2_element import (
    EQUATION_MAP_SHA256,
    FORMULATION_ID,
    HAMMER_POINTS,
    IMPLEMENTATION_ID,
    RESULTANT_POLICY_ID,
    SOURCE_CONTRACT_SHA256,
    StrictFlatLinearCapabilityError,
    StrictFlatLinearE4PLS3V2ShellElement,
    _ALLOWED_INHERITED_CALLABLES,
    _SEALED_INHERITED_CALLABLES,
)
from anyfesolver.elements import (
    BeamElement,
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    ELEMENT_TYPES,
    Element,
    ShellElement,
    create_element,
    create_shell_element,
    shell_element_from_dict,
    shell_formulation_diagnostics,
)
from anyfesolver.fe_core import FEModel, FEMesh, Material
from anyfesolver.matrix_assembly import (
    assemble_external_load_tangent,
    assemble_load_vector,
)


COORDINATES = np.asarray(
    ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.3, 1.1, 0.0)),
    dtype=float,
)
NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=float)


def _mesh(coordinates: np.ndarray = COORDINATES) -> FEMesh:
    mesh = FEMesh()
    for node_id, coordinate in enumerate(np.asarray(coordinates, dtype=float), start=1):
        mesh.add_node(node_id, *coordinate)
    return mesh


def _material() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


def _element(
    node_ids: tuple[int, int, int] = (1, 2, 3),
    *,
    element_id: int = 1,
    reference_normal: np.ndarray = NORMAL,
) -> StrictFlatLinearE4PLS3V2ShellElement:
    return StrictFlatLinearE4PLS3V2ShellElement(
        element_id,
        node_ids,
        "steel",
        thickness=0.08,
        reference_normal=reference_normal,
    )


def _registered(
    node_ids: tuple[int, int, int] = (1, 2, 3),
) -> tuple[FEMesh, StrictFlatLinearE4PLS3V2ShellElement]:
    mesh = _mesh()
    element = _element(node_ids)
    mesh.add_element(element.element_id, element)
    return mesh, element


def _model() -> tuple[FEModel, StrictFlatLinearE4PLS3V2ShellElement]:
    model = FEModel("strict-flat-s3-v2-pressure")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    element = _element()
    model.add_element(element.element_id, element)
    return model, element


def _block_permutation(permutation: tuple[int, int, int]) -> np.ndarray:
    made = np.zeros((18, 18), dtype=float)
    for new, old in enumerate(permutation):
        made[6 * new : 6 * new + 6, 6 * old : 6 * old + 6] = np.eye(6)
    return made


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right, ord=np.inf)
        / max(np.linalg.norm(right, ord=np.inf), 1.0)
    )


def _rank(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular > singular[0] * 1.0e-9))


def _physical_resultants(result: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = np.asarray(result["frame"], dtype=float)[:, :2]

    def tensors(rows: object) -> np.ndarray:
        made = []
        for xx, yy, xy in np.asarray(rows, dtype=float):
            local = np.asarray(((xx, xy), (xy, yy)), dtype=float)
            made.append(frame @ local @ frame.T)
        return np.asarray(made)

    membrane = tensors(result["membrane_resultants"])
    bending = tensors(result["bending_resultants"])
    shear = np.asarray(result["transverse_shear_resultants"], dtype=float) @ frame.T
    return membrane, bending, shear


def _rigid_modes(coordinates: np.ndarray) -> np.ndarray:
    modes = np.zeros((18, 6), dtype=float)
    centre = np.mean(coordinates, axis=0)
    for node, coordinate in enumerate(coordinates):
        base = 6 * node
        modes[base : base + 3, :3] = np.eye(3)
        for axis in range(3):
            modes[base : base + 3, 3 + axis] = np.cross(
                np.eye(3)[axis], coordinate - centre
            )
            modes[base + 3 : base + 6, 3 + axis] = np.eye(3)[axis]
    return modes


def test_explicit_selector_preserves_every_existing_default_and_q4_route() -> None:
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "e4-pl-s3-v2d"
    # The registry retains the legacy implementation identity; public current
    # factories apply the independently activated topology-specific policy.
    assert ELEMENT_TYPES["s3"] is ShellElement
    assert type(create_shell_element(1, [1, 2, 3], "steel", reference_normal=NORMAL)) is NativeParityE4PLS3V2DShellElement
    assert type(create_element("s3", 2, [1, 2, 3], "steel", reference_normal=NORMAL)) is NativeParityE4PLS3V2DShellElement
    assert type(create_shell_element(11, [1, 2, 3], "steel", formulation="legacy-s3")) is ShellElement
    assert type(create_element("legacy-s3", 12, [1, 2, 3], "steel")) is ShellElement
    assert type(create_shell_element(3, [1, 2, 3, 4], "steel")) is (
        QualifiedE4PLShellElement
    )

    selected = create_shell_element(
        4,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    direct = create_element(
        "e4-pl-s3-v2",
        5,
        [1, 2, 3],
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    assert type(selected) is StrictFlatLinearE4PLS3V2ShellElement
    assert type(direct) is StrictFlatLinearE4PLS3V2ShellElement
    alias = create_element(
        "qualified-s3-v2",
        51,
        [1, 2, 3],
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    assert type(alias) is StrictFlatLinearE4PLS3V2ShellElement
    diagnostic = shell_formulation_diagnostics(
        node_count=3,
        formulation="e4-pl-s3-v2",
    )
    assert diagnostic["selected_formulation"] == "e4-pl-s3-v2"
    assert diagnostic["production_default"] is False
    assert diagnostic["topology_policy"] == (
        "STRICT_FLAT_LINEAR_E4_PL_S3_V2_OPT_IN"
    )

    for spelling in ("e4_pl_s3_v2", "qualified_s3_v2"):
        with pytest.raises(ValueError, match="canonical"):
            create_element(
                "s3",
                6,
                [1, 2, 3],
                "steel",
                formulation=spelling,
                reference_normal=NORMAL,
            )
    with pytest.raises(ValueError, match="only for three-node"):
        create_shell_element(
            7,
            [1, 2, 3, 4],
            "steel",
            formulation="e4-pl-s3-v2",
            reference_normal=NORMAL,
        )


def test_authority_identity_and_fixed_hammer_rule_are_exact() -> None:
    assert issubclass(StrictFlatLinearCapabilityError, ElementCapabilityError)
    assert anyfesolver.StrictFlatLinearE4PLS3V2ShellElement is (
        StrictFlatLinearE4PLS3V2ShellElement
    )
    assert anyfesolver.STRICT_FLAT_S3_V2_FORMULATION_ID == FORMULATION_ID
    assert anyfesolver.STRICT_FLAT_S3_V2_IMPLEMENTATION_ID == IMPLEMENTATION_ID
    assert anyfesolver.STRICT_FLAT_S3_V2_SELECTOR == "e4-pl-s3-v2"
    assert FORMULATION_ID == "CANDIDATE_E4_PL_S3_V2A_FLAT_LINEAR_V1"
    assert IMPLEMENTATION_ID == "E4_PL_S3_V2_DKMT_EQ12_41_CST_PL_HAMMER3_V1"
    assert SOURCE_CONTRACT_SHA256 == (
        "754A31C2B03FA3785274F30BF4F2A2FC8C66DF66A76C5D39CD9736E81679513A"
    )
    assert EQUATION_MAP_SHA256 == (
        "B527729C2F3AF482722ECB2D4635FB0FB165FB35F2EE952833D06740A68E0C4A"
    )
    assert np.array_equal(
        HAMMER_POINTS,
        np.asarray(
            ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
        ),
    )
    assert HAMMER_POINTS.flags.writeable is False
    element = _element()
    assert element.gauss_points is HAMMER_POINTS
    assert element.shear_gauss_points is HAMMER_POINTS
    weights = element.gauss_weights
    for power_x in range(3):
        for power_y in range(3 - power_x):
            integrated = float(
                np.sum(weights * HAMMER_POINTS[:, 0] ** power_x * HAMMER_POINTS[:, 1] ** power_y)
            )
            exact = (
                math.factorial(power_x)
                * math.factorial(power_y)
                / math.factorial(power_x + power_y + 2)
            )
            assert integrated == pytest.approx(exact, rel=0.0, abs=2.0e-16)


def test_rank_rigid_kernel_component_split_and_internal_force() -> None:
    mesh, element = _registered()
    components = element.compute_stiffness_components(mesh, _material())
    assert _rank(components["membrane"]) == 3
    assert _rank(components["bending"] + components["shear"]) == 6
    assert _rank(components["physical"]) == 9
    assert _rank(components["pl"]) == 3
    assert _rank(components["total"]) == 12
    np.testing.assert_allclose(
        components["total"],
        components["membrane"]
        + components["bending"]
        + components["shear"]
        + components["pl"],
        rtol=2.0e-15,
        atol=2.0e-5,
    )
    np.testing.assert_array_equal(components["numerical"], components["pl"])
    np.testing.assert_array_equal(components["hourglass"], np.zeros((18, 18)))

    rigid = _rigid_modes(COORDINATES)
    residual = components["total"] @ rigid
    scale = max(float(np.linalg.norm(components["total"], ord=np.inf)), 1.0)
    assert float(np.linalg.norm(residual, ord=np.inf)) <= 3.0e-15 * scale

    displacement = np.linspace(-2.0e-4, 3.0e-4, 18)
    np.testing.assert_allclose(
        element.compute_internal_forces(mesh, displacement, _material()),
        components["total"] @ displacement,
        rtol=2.0e-15,
        atol=1.0e-8,
    )


def test_all_d3_numberings_transport_every_component_force_and_pressure() -> None:
    material = _material()
    base_mesh, base_element = _registered()
    base = base_element.compute_stiffness_components(base_mesh, material)
    displacement = np.linspace(-0.004, 0.006, 18)
    base_force = base_element.compute_internal_forces(
        base_mesh,
        displacement,
        material,
    )
    base_pressure = base_element.compute_dead_transverse_pressure_load(base_mesh, 7.0)
    base_result = base_element.compute_variational_resultants(
        base_mesh, displacement, material
    )
    base_physical = _physical_resultants(base_result)
    base_stations = np.asarray(base_result["physical_station_coordinates"], dtype=float)

    for permutation in itertools.permutations(range(3)):
        node_ids = tuple(index + 1 for index in permutation)
        mesh, element = _registered(node_ids)
        made = element.compute_stiffness_components(mesh, material)
        transport = _block_permutation(permutation)
        for name in ("membrane", "bending", "shear", "physical", "pl", "total"):
            expected = transport @ base[name] @ transport.T
            assert _relative(made[name], expected) <= 8.0e-15
        permuted_displacement = transport @ displacement
        np.testing.assert_allclose(
            element.compute_internal_forces(mesh, permuted_displacement, material),
            transport @ base_force,
            rtol=5.0e-14,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            element.compute_dead_transverse_pressure_load(mesh, 7.0),
            transport @ base_pressure,
            rtol=0.0,
            atol=2.0e-15,
        )
        made_result = element.compute_variational_resultants(
            mesh, permuted_displacement, material
        )
        external_coordinates = COORDINATES[np.asarray(permutation)]
        expected_stations = np.column_stack(
            (
                1.0 - HAMMER_POINTS[:, 0] - HAMMER_POINTS[:, 1],
                HAMMER_POINTS[:, 0],
                HAMMER_POINTS[:, 1],
            )
        ) @ external_coordinates
        np.testing.assert_allclose(
            made_result["physical_station_coordinates"],
            expected_stations,
            rtol=0.0,
            atol=3.0e-16,
        )
        made_physical = _physical_resultants(made_result)
        made_stations = np.asarray(
            made_result["physical_station_coordinates"], dtype=float
        )
        for base_index, station in enumerate(base_stations):
            made_index = int(np.argmin(np.linalg.norm(made_stations - station, axis=1)))
            assert np.linalg.norm(made_stations[made_index] - station) < 1.0e-14
            for base_field, made_field in zip(base_physical, made_physical):
                np.testing.assert_allclose(
                    made_field[made_index],
                    base_field[base_index],
                    rtol=3.0e-13,
                    atol=2.0e-6,
                )
        assert float(
            permuted_displacement @ made["total"] @ permuted_displacement
        ) == pytest.approx(
            float(displacement @ base["total"] @ displacement),
            rel=3.0e-14,
        )


def test_raw_variational_resultants_close_physical_work_and_exclude_pl() -> None:
    mesh, element = _registered()
    material = _material()
    displacement = np.linspace(-0.003, 0.004, 18)
    components = element.compute_stiffness_components(mesh, material)
    result = element.compute_variational_resultants(mesh, displacement, material)
    assert result["recovery_scope"] == RESULTANT_POLICY_ID
    assert result["qualified_recovery"] is False
    assert result["numerical_fields_excluded"] is True
    physical_work = np.sum(
        result["physical_weights"]
        * (
            np.einsum(
                "ij,ij->i",
                result["membrane_strain"],
                result["membrane_resultants"],
            )
            + np.einsum(
                "ij,ij->i",
                result["curvature"],
                result["bending_resultants"],
            )
            + np.einsum(
                "ij,ij->i",
                result["transverse_shear_strain"],
                result["transverse_shear_resultants"],
            )
        )
    )
    assert physical_work == pytest.approx(
        float(displacement @ components["physical"] @ displacement),
        rel=3.0e-14,
    )

    pure_drill = np.zeros(18, dtype=float)
    pure_drill[5::6] = (1.0, -0.4, 0.7)
    drill_result = element.compute_variational_resultants(mesh, pure_drill, material)
    for name in (
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
    ):
        np.testing.assert_array_equal(drill_result[name], np.zeros_like(drill_result[name]))
    assert float(pure_drill @ components["physical"] @ pure_drill) == 0.0
    assert float(pure_drill @ components["pl"] @ pure_drill) > 0.0

    virtual = np.linspace(0.002, -0.001, 18)
    virtual_fields = element.compute_variational_resultants(mesh, virtual, material)
    physical_virtual_work = np.sum(
        result["physical_weights"]
        * (
            np.einsum(
                "ij,ij->i",
                virtual_fields["membrane_strain"],
                result["membrane_resultants"],
            )
            + np.einsum(
                "ij,ij->i",
                virtual_fields["curvature"],
                result["bending_resultants"],
            )
            + np.einsum(
                "ij,ij->i",
                virtual_fields["transverse_shear_strain"],
                result["transverse_shear_resultants"],
            )
        )
    )
    assert physical_virtual_work == pytest.approx(
        float(virtual @ components["physical"] @ displacement),
        rel=4.0e-14,
    )
    total_virtual_work = physical_virtual_work + float(
        virtual @ components["pl"] @ displacement
    )
    assert total_virtual_work == pytest.approx(
        float(
            virtual
            @ element.compute_internal_forces(mesh, displacement, material)
        ),
        rel=4.0e-14,
    )


def test_pl_completion_removes_drill_modes_but_preserves_rigid_planar_spin() -> None:
    mesh, element = _registered()
    total = element.compute_stiffness_components(mesh, _material())
    alpha = 0.07
    planar_spin = np.zeros(18, dtype=float)
    uniform_drill = np.zeros(18, dtype=float)
    for node, (x_coordinate, y_coordinate, _z_coordinate) in enumerate(COORDINATES):
        planar_spin[6 * node] = -alpha * y_coordinate
        planar_spin[6 * node + 1] = alpha * x_coordinate
        uniform_drill[6 * node + 5] = alpha
    assert float(planar_spin @ total["pl"] @ planar_spin) > 0.0
    assert float(uniform_drill @ total["pl"] @ uniform_drill) > 0.0
    rigid = planar_spin + uniform_drill
    cancellation_scale = max(
        abs(float(planar_spin @ total["pl"] @ planar_spin)),
        abs(float(uniform_drill @ total["pl"] @ uniform_drill)),
        1.0,
    )
    assert abs(float(rigid @ total["total"] @ rigid)) <= 2.0e-14 * cancellation_scale


def test_dead_transverse_pressure_is_consistent_and_has_no_moments() -> None:
    mesh, element = _registered()
    area = 1.1
    uniform = element.compute_dead_transverse_pressure_load(mesh, 6.0)
    expected = np.zeros(18, dtype=float)
    for node in range(3):
        expected[6 * node : 6 * node + 3] = 6.0 * area / 3.0 * NORMAL
    np.testing.assert_allclose(uniform, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_array_equal(uniform.reshape(3, 6)[:, 3:], np.zeros((3, 3)))

    nodal_pressure = np.asarray((2.0, 5.0, -1.0))
    affine = element.compute_dead_transverse_pressure_load(mesh, nodal_pressure)
    nodal_work = (area / 12.0) * np.asarray(
        ((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))
    ) @ nodal_pressure
    np.testing.assert_allclose(
        affine.reshape(3, 6)[:, :3],
        nodal_work[:, None] * NORMAL[None, :],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_load_case_dispatches_uniform_v2_dead_pressure_through_public_assembly() -> None:
    model, element = _model()
    load = LoadCase("strict-flat-v2-uniform-pressure")
    load.add_pressure_load(element.element_id, 6.0)

    assembled, info = assemble_load_vector(model, load)

    np.testing.assert_allclose(
        assembled,
        element.compute_dead_transverse_pressure_load(model.mesh, 6.0),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert info["pressure_configuration"] == "reference"


def test_load_case_owns_and_dispatches_affine_v2_dead_pressure() -> None:
    model, element = _model()
    pressure = np.asarray((2.0, 5.0, -1.0))
    expected_pressure = pressure.copy()
    load = LoadCase("strict-flat-v2-affine-pressure")
    load.add_pressure_load(element.element_id, pressure)
    pressure[:] = 900.0

    assert load.pressure_loads[element.element_id] == (2.0, 5.0, -1.0)
    assembled, _info = assemble_load_vector(model, load)
    expected = element.compute_dead_transverse_pressure_load(
        model.mesh,
        expected_pressure,
    )
    np.testing.assert_allclose(assembled, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_array_equal(
        assembled.reshape(3, 6)[:, 3:],
        np.zeros((3, 3)),
    )


def test_load_case_rejects_v2_follower_pressure_before_legacy_mechanics() -> None:
    model, element = _model()
    load = LoadCase("strict-flat-v2-follower-pressure", follower_pressure=True)
    load.add_pressure_load(element.element_id, (2.0, 5.0, -1.0))
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)

    with pytest.raises(StrictFlatLinearCapabilityError, match="follower_pressure"):
        assemble_load_vector(model, load, displacement)
    with pytest.raises(StrictFlatLinearCapabilityError, match="follower_pressure"):
        assemble_external_load_tangent(model, load, displacement)


def test_geometry_material_mixed_mesh_and_curved_scope_fail_closed() -> None:
    material = _material()
    wrong_normal = _element(reference_normal=np.asarray((0.0, 1.0, 0.0)))
    with pytest.raises(StrictFlatLinearCapabilityError, match="not normal"):
        wrong_normal.compute_stiffness_matrix(_mesh(), material)

    degenerate_mesh = _mesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    )
    with pytest.raises(ValueError, match="nondegenerate"):
        _element().compute_stiffness_matrix(degenerate_mesh, material)

    mesh, element = _registered()
    mesh.add_element(2, ShellElement(2, [1, 2, 3], "steel", thickness=0.08))
    with pytest.raises(StrictFlatLinearCapabilityError, match="mixed element"):
        element.compute_stiffness_matrix(mesh, material)

    beam_mesh, v2_with_beam = _registered()
    beam_mesh.add_element(2, BeamElement(2, [1, 2], "steel"))
    with pytest.raises(StrictFlatLinearCapabilityError, match="mixed element"):
        v2_with_beam.compute_stiffness_matrix(beam_mesh, material)

    malformed_mesh, malformed = _registered()
    malformed_mesh.elements[99] = malformed_mesh.elements.pop(1)
    with pytest.raises(StrictFlatLinearCapabilityError, match="registry identity"):
        malformed.compute_stiffness_matrix(malformed_mesh, material)

    duplicate_mesh, duplicate = _registered()
    duplicate_mesh.elements[2] = duplicate
    with pytest.raises(StrictFlatLinearCapabilityError, match="registry identity|duplicate"):
        duplicate.compute_stiffness_matrix(duplicate_mesh, material)

    curved_mesh = _mesh(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.1),
            )
        )
    )
    first = _element((1, 2, 3), element_id=1)
    second = StrictFlatLinearE4PLS3V2ShellElement(
        2,
        (2, 4, 3),
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    curved_mesh.add_element(1, first)
    curved_mesh.add_element(2, second)
    with pytest.raises(StrictFlatLinearCapabilityError, match="coplanar"):
        first.compute_stiffness_matrix(curved_mesh, material)

    translated = _mesh(
        np.asarray(
            (
                (1.0e12, -2.0e12, 3.0e12),
                (1.0e12 + 1.0, -2.0e12, 3.0e12),
                (1.0e12, -2.0e12 + 1.0, 3.0e12),
                (1.0e12 + 1.0, -2.0e12 + 1.0, 3.0e12 + 0.1),
            )
        )
    )
    shifted_first = _element((1, 2, 3), element_id=1)
    shifted_second = StrictFlatLinearE4PLS3V2ShellElement(
        2,
        (2, 4, 3),
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    translated.add_element(1, shifted_first)
    translated.add_element(2, shifted_second)
    with pytest.raises(StrictFlatLinearCapabilityError, match="coplanar"):
        shifted_first.compute_stiffness_matrix(translated, material)

    with pytest.raises(StrictFlatLinearCapabilityError, match="exact homogeneous"):
        element.compute_stiffness_matrix(_mesh(), object())
    with pytest.raises(TypeError):
        create_shell_element(
            9,
            [1, 2, 3],
            "steel",
            formulation="e4-pl-s3-v2",
            reference_normal=NORMAL,
            shell_section={"A": np.eye(3)},
        )


def test_every_unsupported_inherited_or_persistence_route_fails_closed() -> None:
    mesh, element = _registered()
    material = _material()
    vector = np.zeros(18)
    calls = (
        lambda: element.compute_mass_matrix(mesh, material),
        lambda: element.compute_geometric_stiffness_matrix(mesh, material),
        lambda: element.init_nonlinear_state(3),
        lambda: element.compute_nonlinear_response(mesh, material, vector),
        lambda: element.compute_stresses(mesh, vector, material),
        element.to_dict,
        lambda: type(element).from_dict({}),
        element.__getstate__,
        lambda: element.__setstate__({}),
        lambda: pickle.dumps(element),
        lambda: copy.copy(element),
        lambda: copy.deepcopy(element),
    )
    for call in calls:
        with pytest.raises(StrictFlatLinearCapabilityError):
            call()


def test_candidate_instance_and_module_authority_fail_closed(monkeypatch) -> None:
    mesh, element = _registered()
    material = _material()
    baseline = element.compute_stiffness_matrix(mesh, material)

    with pytest.raises(StrictFlatLinearCapabilityError, match="immutable"):
        element._constitutive = lambda unused: {}  # type: ignore[method-assign]

    namespace = object.__getattribute__(element, "__dict__")
    namespace["_constitutive"] = lambda unused: {}
    with pytest.raises(
        StrictFlatLinearCapabilityError, match="shadowing|unregistered state"
    ):
        element.compute_stiffness_matrix(mesh, material)
    del namespace["_constitutive"]

    namespace["unregistered_candidate_state"] = 1
    with pytest.raises(StrictFlatLinearCapabilityError, match="unregistered state"):
        element.compute_stiffness_matrix(mesh, material)
    del namespace["unregistered_candidate_state"]

    original_thickness = namespace["thickness"]
    namespace["thickness"] = original_thickness * 2.0
    with pytest.raises(StrictFlatLinearCapabilityError, match="construction authority"):
        element.compute_stiffness_matrix(mesh, material)
    namespace["thickness"] = original_thickness

    with pytest.raises(StrictFlatLinearCapabilityError, match="class authority"):
        setattr(
            StrictFlatLinearE4PLS3V2ShellElement,
            "_constitutive",
            lambda unused_self, unused_material: {},
        )

    import anyfesolver.e4_pl_s3_v2_element as candidate_module

    monkeypatch.setattr(candidate_module, "_validate_module_authority", lambda: None)
    monkeypatch.setattr(candidate_module, "SHEAR_CORRECTION", 0.0)
    with pytest.raises(StrictFlatLinearCapabilityError, match="module authority"):
        element.compute_stiffness_matrix(mesh, material)
    monkeypatch.undo()
    assert np.array_equal(element.compute_stiffness_matrix(mesh, material), baseline)

    monkeypatch.setattr(candidate_module, "_real_scalar", lambda value, label: 1.0)
    with pytest.raises(StrictFlatLinearCapabilityError, match="module authority"):
        element.compute_stiffness_matrix(mesh, material)
    monkeypatch.undo()

    monkeypatch.setattr(
        candidate_module,
        "_immutable_float64_array",
        lambda value: np.asarray(value, dtype=float),
    )
    with pytest.raises(StrictFlatLinearCapabilityError, match="module authority"):
        _element(element_id=42)
    monkeypatch.undo()

    with pytest.raises(TypeError):
        object.__setattr__(element, "__class__", ShellElement)

    import anyfesolver.elements as elements_module

    monkeypatch.setattr(
        elements_module,
        "_reject_strict_flat_s3_v2_legacy_dispatch",
        lambda unused_element, unused_operation: None,
    )
    with pytest.raises(ElementCapabilityError, match="explicit legacy"):
        Element.compute_mass_matrix(element, mesh, material)
    with pytest.raises(ElementCapabilityError, match="explicit legacy"):
        ShellElement.shear_gauss_points.fget(element)


def test_explicit_legacy_base_dispatch_cannot_bypass_candidate_capabilities() -> None:
    mesh, element = _registered()
    material = _material()
    vector = np.zeros(18)
    calls = (
        lambda: ShellElement.compute_mass_matrix(element, mesh, material),
        lambda: ShellElement.compute_geometric_stiffness_matrix(
            element, mesh, material
        ),
        lambda: ShellElement.init_nonlinear_state(element, 3),
        lambda: ShellElement.compute_nonlinear_response(
            element, mesh, material, vector
        ),
        lambda: ShellElement.compute_stresses(element, mesh, vector, material),
        lambda: ShellElement.to_dict(element),
        lambda: ShellElement._build_shell_b_matrices(
            element, np.ones(3), np.ones(3), np.ones(3)
        ),
        lambda: Element.to_dict(element),
        lambda: Element.__deepcopy__(element, {}),
        lambda: Element.compute_mass_matrix(element, mesh, material),
        lambda: Element.compute_geometric_stiffness_matrix(
            element, mesh, material
        ),
        lambda: Element.compute_internal_forces(element, mesh, vector, material),
        lambda: Element.compute_nonlinear_response(
            element, mesh, material, vector
        ),
        lambda: Element.compute_stresses(element, mesh, vector, material),
        lambda: ShellElement.shear_gauss_points.fget(element),
        lambda: ShellElement.shear_gauss_weights.fget(element),
    )
    for call in calls:
        with pytest.raises(ElementCapabilityError, match="explicit legacy"):
            call()
    guarded_base_routes = (
        Element.compute_mass_matrix,
        ShellElement.init_nonlinear_state,
        ShellElement.shear_gauss_points.fget,
        ShellElement.shear_gauss_weights.fget,
        ShellElement._compute_3node_shape_functions,
    )
    for guarded in guarded_base_routes:
        assert guarded is not None
        assert not hasattr(guarded, "__wrapped__")
        assert inspect.unwrap(guarded) is guarded


def test_closed_world_seals_every_unused_inherited_shell_callable() -> None:
    inherited = {
        name
        for name, value in inspect.getmembers(ShellElement, callable)
        if not name.startswith("__")
    }
    remaining = inherited - set(StrictFlatLinearE4PLS3V2ShellElement.__dict__)
    assert remaining == set(_ALLOWED_INHERITED_CALLABLES)
    assert _SEALED_INHERITED_CALLABLES <= set(
        StrictFlatLinearE4PLS3V2ShellElement.__dict__
    )
    element = _element()
    for name in sorted(_SEALED_INHERITED_CALLABLES):
        with pytest.raises(StrictFlatLinearCapabilityError, match="legacy_inherited"):
            getattr(element, name)()


@pytest.mark.parametrize(
    "marker,value",
    (
        ("implementation_id", IMPLEMENTATION_ID),
        ("selector", "e4-pl-s3-v2"),
        ("source_contract_sha256", SOURCE_CONTRACT_SHA256),
        ("equation_map_sha256", EQUATION_MAP_SHA256),
        ("resultant_policy_id", RESULTANT_POLICY_ID),
    ),
)
def test_deserialization_recognizes_v2_and_never_downgrades_missing_id(
    marker: str,
    value: str,
) -> None:
    exact = {
        "type": "StrictFlatLinearE4PLS3V2ShellElement",
        "element_id": 1,
        "node_ids": [1, 2, 3],
        "material_name": "steel",
        "formulation_id": FORMULATION_ID,
    }
    before = copy.deepcopy(exact)
    with pytest.raises(StrictFlatLinearCapabilityError, match="serialization/restart"):
        shell_element_from_dict(exact)
    assert exact == before

    stripped = {
        "type": "ShellElement",
        "element_id": 1,
        "node_ids": [1, 2, 3],
        "material_name": "steel",
        marker: value,
    }
    before_stripped = copy.deepcopy(stripped)
    with pytest.raises(ValueError, match="retains .*S3.* fingerprint"):
        shell_element_from_dict(stripped)
    assert stripped == before_stripped


@pytest.mark.parametrize(
    ("node_ids", "marker", "value"),
    (
        ([1, 2, 3, 4], "selector", "e4-pl-s3-v2"),
        ([1, 2, 3, 4, 5, 6], "source_contract_sha256", SOURCE_CONTRACT_SHA256),
        ([1, 2, 3], "selector", "E4-PL-S3-V2"),
        ([1, 2, 3], "selector", " e4_pl_s3_v2 "),
        ([1, 2, 3], "implementation_id", IMPLEMENTATION_ID.lower()),
        (
            [1, 2, 3, 4, 5, 6],
            "quadrature_authority_id",
            "S3_V2_DKMT_HAMMER3_DEGREE2_EXACT_V1",
        ),
    ),
)
def test_v2_fingerprint_never_downgrades_across_topology_or_case(
    node_ids: list[int], marker: str, value: str
) -> None:
    payload = {
        "type": "ShellElement",
        "element_id": 1,
        "node_ids": node_ids,
        "material_name": "steel",
        marker: value,
    }
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="strict-flat S3 V2 fingerprint"):
        shell_element_from_dict(payload)
    assert payload == before


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("selector", "legacy-s3"),
        ("selector", "unrelated-research-metadata"),
        ("implementation_id", "UNRELATED_LEGACY_IMPLEMENTATION"),
    ),
)
def test_generic_historical_metadata_does_not_become_a_v2_fingerprint(
    key: str, value: str
) -> None:
    payload = {
        "type": "ShellElement",
        "element_id": 1,
        "node_ids": [1, 2, 3],
        "material_name": "steel",
        key: value,
    }
    before = copy.deepcopy(payload)
    with pytest.warns(UserWarning, match="LEGACY_S3_MISSING_FORMULATION_ID"):
        made = shell_element_from_dict(payload)
    assert type(made) is ShellElement
    assert payload == before
