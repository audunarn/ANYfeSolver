from __future__ import annotations

import itertools

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    FEModel,
    FixedSupport,
    QualifiedE4PLS3ShellElement,
    TransientConfig,
    solve_transient_sphere_impact,
)
from anyfesolver._native_rotation_state import (
    create_native_rotation_state_store,
    rotation_exponential,
)
from anyfesolver.contact import (
    NonlinearTransientConfig,
    RigidSphereImpact,
    SphereContactConfig,
    assemble_sphere_contact_load_vector,
)
from anyfesolver.elements import create_shell_element
from anyfesolver.nonlinear_state import NonlinearStateStore
from anyfesolver.shell_sections import GeneralizedShellSection
from anyfesolver.validation import load_vector_resultant


_THICKNESS = 0.2


def _s3_model(
    *,
    node_order: tuple[int, int, int] = (1, 2, 3),
    coordinates: np.ndarray | None = None,
    reference_normal: np.ndarray | None = None,
    thickness: float = _THICKNESS,
    director_polarity: int = 1,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    points = (
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        if coordinates is None
        else np.asarray(coordinates, dtype=float)
    )
    normal = (
        np.asarray((0.0, 0.0, 1.0), dtype=float)
        if reference_normal is None
        else np.asarray(reference_normal, dtype=float)
    )
    model = FEModel("qualified-s3-contact")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for node_id, point in enumerate(points, start=1):
        model.add_node(node_id, *point)
    element = QualifiedE4PLS3ShellElement(
        1,
        list(node_order),
        "soft",
        thickness=thickness,
        reference_normal=normal,
        director_polarity=director_polarity,
    )
    model.add_element(1, element)
    return model, element


def _sphere(position: np.ndarray, *, radius: float = 0.2) -> RigidSphereImpact:
    return RigidSphereImpact(
        "qualified-s3-contact",
        radius=radius,
        mass=1.0,
        start_point=tuple(float(value) for value in position),
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )


def _assemble(
    model: FEModel,
    position: np.ndarray,
    surface: str,
    *,
    displacement: np.ndarray | None = None,
    native_rotation_context: NonlinearStateStore | None = None,
):
    return assemble_sphere_contact_load_vector(
        model,
        _sphere(position),
        SphereContactConfig(
            penalty_stiffness=1000.0,
            contact_surface=surface,
        ),
        sphere_position=position,
        sphere_velocity=np.zeros(3),
        structural_displacement=displacement,
        structural_velocity=(
            None if displacement is None else np.zeros_like(displacement)
        ),
        native_rotation_context=native_rotation_context,
    )


@pytest.mark.parametrize(
    ("surface", "position", "expected_z", "expected_penetration"),
    (
        ("midsurface", np.asarray((0.25, 0.25, 0.15)), 0.0, 0.05),
        ("top", np.asarray((0.25, 0.25, 0.15)), 0.1, 0.15),
        ("bottom", np.asarray((0.25, 0.25, -0.15)), -0.1, 0.15),
    ),
)
def test_qualified_s3_midsurface_top_and_bottom_use_physical_director(
    surface: str,
    position: np.ndarray,
    expected_z: float,
    expected_penetration: float,
) -> None:
    model, _element = _s3_model()
    load, sphere_force, records = _assemble(model, position, surface)

    assert len(records) == 1
    record = records[0]
    assert record.contact_point[2] == pytest.approx(expected_z, abs=1.0e-15)
    assert record.penetration == pytest.approx(expected_penetration)
    np.testing.assert_allclose(record.structure_force, -sphere_force, atol=1.0e-13)
    resultant = load_vector_resultant(model, load)
    np.testing.assert_allclose(resultant.force, record.structure_force, atol=1.0e-13)
    np.testing.assert_allclose(
        resultant.moment,
        np.cross(record.contact_point, record.structure_force),
        atol=2.0e-13,
    )


def test_linear_contact_uses_facet_director_not_oblique_owner_hint() -> None:
    model, _element = _s3_model(
        reference_normal=np.asarray((0.25, -0.1, 1.0), dtype=float)
    )
    position = np.asarray((0.25, 0.25, 0.15), dtype=float)

    _load, _force, records = _assemble(model, position, "top")

    assert len(records) == 1
    np.testing.assert_allclose(
        records[0].contact_point,
        np.asarray((0.25, 0.25, 0.1), dtype=float),
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    ("surface", "position", "expected_z"),
    (
        ("top", np.asarray((0.25, 0.25, -0.15)), -0.1),
        ("bottom", np.asarray((0.25, 0.25, 0.15)), 0.1),
    ),
)
def test_physical_director_reversal_reverses_top_and_bottom_contact_surfaces(
    surface: str,
    position: np.ndarray,
    expected_z: float,
) -> None:
    model, _element = _s3_model(director_polarity=-1)

    _load, _force, records = _assemble(model, position, surface)

    assert len(records) == 1
    assert records[0].contact_point[2] == pytest.approx(expected_z, abs=1.0e-15)


def test_qualified_s3_broad_phase_includes_physical_surface_offset() -> None:
    model, _element = _s3_model(
        coordinates=np.asarray(
            ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
            dtype=float,
        ),
        thickness=1.0,
    )
    position = np.asarray((0.03, 0.03, 0.55), dtype=float)

    _load, _force, records = assemble_sphere_contact_load_vector(
        model,
        _sphere(position, radius=0.1),
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="top"),
        sphere_position=position,
        sphere_velocity=np.zeros(3),
    )

    assert len(records) == 1
    assert records[0].contact_point[2] == pytest.approx(0.5, abs=1.0e-15)
    assert records[0].penetration == pytest.approx(0.05, abs=2.0e-14)


def test_qualified_s3_offset_scatter_is_exactly_work_and_moment_conjugate() -> None:
    model, element = _s3_model()
    position = np.asarray((-0.05, 0.25, 0.18), dtype=float)
    load, _sphere_force, records = _assemble(model, position, "top")

    assert len(records) == 1
    record = records[0]
    assert any(
        float(np.linalg.norm(moment)) > 0.0
        for moment in record.nodal_moments.values()
    )
    resultant = load_vector_resultant(model, load)
    np.testing.assert_allclose(resultant.force, record.structure_force, atol=2.0e-13)
    np.testing.assert_allclose(
        resultant.moment,
        np.cross(record.contact_point, record.structure_force),
        atol=2.0e-13,
    )

    r, s = record.local_coordinates
    shape = np.asarray((1.0 - r - s, r, s), dtype=float)
    rng = np.random.default_rng(20260825)
    virtual_translations = rng.normal(size=(3, 3))
    virtual_rotations = rng.normal(size=(3, 3))
    director = np.asarray(element.native_reference_directors(model.mesh)[0], dtype=float)
    offset_arm = 0.5 * _THICKNESS * director
    contact_variation = np.sum(
        shape[:, None]
        * (
            virtual_translations
            + np.cross(virtual_rotations, offset_arm[None, :])
        ),
        axis=0,
    )
    nodal_work = 0.0
    for local_node, node_id in enumerate(element.node_ids):
        nodal_work += float(
            record.nodal_forces[int(node_id)] @ virtual_translations[local_node]
            + record.nodal_moments[int(node_id)] @ virtual_rotations[local_node]
        )
    contact_work = float(record.structure_force @ contact_variation)
    assert nodal_work == pytest.approx(contact_work, rel=0.0, abs=2.0e-13)


def _native_trial_context(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    displacement: np.ndarray,
    trial_coordinates: np.ndarray,
) -> tuple[NonlinearStateStore, object]:
    nodes = tuple(int(node_id) for node_id in model.mesh.nodes)
    rotational_dofs = {
        node_id: tuple(int(value) for value in model.mesh.get_node(node_id).dofs[3:6])
        for node_id in nodes
    }
    coordinate_rows = {node_id: row for row, node_id in enumerate(nodes)}
    reference_coordinates = np.asarray(
        [model.mesh.get_node(node_id).coords() for node_id in nodes], dtype=float
    )
    native = create_native_rotation_state_store(
        element.node_ids,
        rotational_dofs=rotational_dofs,
        coordinate_rows=coordinate_rows,
        committed_full_displacement=np.zeros_like(displacement),
        committed_full_coordinates=reference_coordinates,
        coordinate_node_ids=nodes,
    )
    assert native is not None
    store = NonlinearStateStore()
    store.attach_native_rotation_store(native)
    token = store.begin_trial(
        full_displacement=displacement,
        full_coordinates=trial_coordinates,
    )
    return store, token


def test_nonlinear_qualified_s3_contact_consumes_exact_native_trial_and_is_objective() -> None:
    base_model, _base_element = _s3_model()
    base_position = np.asarray((-0.05, 0.25, 0.18), dtype=float)
    base_load, _base_sphere_force, base_records = _assemble(
        base_model, base_position, "top"
    )
    assert len(base_records) == 1

    rotation_vector = np.asarray((0.31, -0.23, 0.17), dtype=float)
    rotation = rotation_exponential(rotation_vector)
    translation = np.asarray((0.37, -0.29, 0.41), dtype=float)
    reference_coordinates = np.asarray(
        [base_model.mesh.get_node(node_id).coords() for node_id in (1, 2, 3)],
        dtype=float,
    )
    trial_coordinates = reference_coordinates @ rotation.T + translation
    model, element = _s3_model()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for row, node_id in enumerate((1, 2, 3)):
        node = model.mesh.get_node(node_id)
        displacement[np.asarray(node.dofs[:3], dtype=int)] = (
            trial_coordinates[row] - reference_coordinates[row]
        )
        displacement[np.asarray(node.dofs[3:6], dtype=int)] = rotation_vector
    store, token = _native_trial_context(
        model, element, displacement, trial_coordinates
    )
    rotated_position = rotation @ base_position + translation
    load, sphere_force, records = _assemble(
        model,
        rotated_position,
        "top",
        displacement=displacement,
        native_rotation_context=store,
    )

    assert len(records) == 1
    base_record = base_records[0]
    record = records[0]
    np.testing.assert_allclose(
        record.contact_point,
        rotation @ base_record.contact_point + translation,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        record.structure_force, rotation @ base_record.structure_force, atol=3.0e-12
    )
    np.testing.assert_allclose(
        sphere_force, rotation @ _base_sphere_force, atol=3.0e-12
    )
    for node_id in (1, 2, 3):
        node = model.mesh.get_node(node_id)
        base_node = base_model.mesh.get_node(node_id)
        np.testing.assert_allclose(
            load[np.asarray(node.dofs[:3], dtype=int)],
            rotation @ base_load[np.asarray(base_node.dofs[:3], dtype=int)],
            atol=3.0e-12,
        )
        np.testing.assert_allclose(
            load[np.asarray(node.dofs[3:6], dtype=int)],
            rotation @ base_load[np.asarray(base_node.dofs[3:6], dtype=int)],
            atol=3.0e-12,
        )

    native = store.native_rotation_store
    assert native is not None
    assert native.generation == 0
    store.discard_trial(token)
    assert native.generation == 0
    np.testing.assert_array_equal(
        native.committed_rotation_matrices,
        np.repeat(np.eye(3)[None, :, :], 3, axis=0),
    )


def test_native_contact_rejects_mismatched_trial_without_mutating_history() -> None:
    model, element = _s3_model()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    coordinates = np.asarray(
        [model.mesh.get_node(node_id).coords() for node_id in (1, 2, 3)],
        dtype=float,
    )
    store, token = _native_trial_context(model, element, displacement, coordinates)
    mismatched = displacement.copy()
    mismatched[model.mesh.get_node(1).dofs[0]] = 1.0e-6
    with pytest.raises(ValueError, match="disagree with the active solver-owned"):
        _assemble(
            model,
            np.asarray((0.25, 0.25, 0.15)),
            "top",
            displacement=mismatched,
            native_rotation_context=store,
        )
    native = store.native_rotation_store
    assert native is not None
    assert native.generation == 0
    assert store.has_active_trial
    store.discard_trial(token)
    assert native.generation == 0
    np.testing.assert_array_equal(native.committed_full_displacement, displacement)


def test_qualified_s3_contact_is_covariant_for_all_six_d3_numberings() -> None:
    position = np.asarray((-0.05, 0.25, 0.18), dtype=float)
    baseline_model, _baseline_element = _s3_model()
    baseline_load, baseline_sphere_force, baseline_records = _assemble(
        baseline_model, position, "top"
    )
    baseline = baseline_records[0]

    for ordering in itertools.permutations((1, 2, 3)):
        model, _element = _s3_model(node_order=ordering)
        load, sphere_force, records = _assemble(model, position, "top")
        assert len(records) == 1
        record = records[0]
        np.testing.assert_allclose(load, baseline_load, atol=2.0e-13)
        np.testing.assert_allclose(sphere_force, baseline_sphere_force, atol=2.0e-13)
        np.testing.assert_allclose(record.contact_point, baseline.contact_point, atol=2.0e-13)
        for node_id in (1, 2, 3):
            np.testing.assert_allclose(
                record.nodal_forces[node_id], baseline.nodal_forces[node_id], atol=2.0e-13
            )
            np.testing.assert_allclose(
                record.nodal_moments[node_id], baseline.nodal_moments[node_id], atol=2.0e-13
            )


def test_mixed_q4_s3_contact_keeps_q4_scatter_compatible() -> None:
    model, _element = _s3_model()
    for node_id, point in {
        4: (1.2, 0.0, 0.0),
        5: (2.2, 0.0, 0.0),
        6: (2.2, 1.0, 0.0),
        7: (1.2, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *point)
    model.add_element(
        2,
        create_shell_element(
            2,
            [4, 5, 6, 7],
            "soft",
            thickness=_THICKNESS,
        ),
    )

    s3_position = np.asarray((0.25, 0.25, 0.15), dtype=float)
    s3_load, _s3_force, s3_records = _assemble(model, s3_position, "top")
    assert tuple(record.element_id for record in s3_records) == (1,)
    q4_position = np.asarray((1.7, 0.5, 0.15), dtype=float)
    q4_load, _q4_force, q4_records = _assemble(model, q4_position, "top")
    assert tuple(record.element_id for record in q4_records) == (2,)
    q4_record = q4_records[0]
    for node_id in (4, 5, 6, 7):
        node = model.mesh.get_node(node_id)
        np.testing.assert_array_equal(
            q4_load[np.asarray(node.dofs[3:6], dtype=int)], np.zeros(3)
        )
    assert q4_record.nodal_moments == {}
    assert "nodal_moments" not in q4_record.to_dict()
    assert np.linalg.norm(s3_load) > 0.0


def _generalized_contact_section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="qualified-s3-contact-generalized",
        A=np.asarray(
            (
                (2.4e8, 0.42e8, 0.13e8),
                (0.42e8, 1.35e8, -0.08e8),
                (0.13e8, -0.08e8, 0.71e8),
            ),
            dtype=float,
        ),
        B=np.asarray(
            (
                (2.1e4, -0.7e4, 0.3e4),
                (0.4e4, 1.6e4, -0.2e4),
                (-0.5e4, 0.1e4, 0.8e4),
            ),
            dtype=float,
        ),
        D=np.asarray(
            (
                (3.2e4, 0.55e4, 0.12e4),
                (0.55e4, 2.1e4, -0.09e4),
                (0.12e4, -0.09e4, 1.15e4),
            ),
            dtype=float,
        ),
        As=np.asarray(((0.82e8, 0.11e8), (0.11e8, 0.59e8)), dtype=float),
        mass_per_area=41.0,
        rotary_inertia_per_area=0.014,
    )


def _nonlinear_contact_model(
    *, constrain_drill: bool = False, generalized: bool = False
) -> FEModel:
    model = FEModel("qualified-s3-contact-top-level")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, point in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *point)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
            shell_section=(
                _generalized_contact_section() if generalized else None
            ),
            material_direction=((1.0, 0.0, 0.0) if generalized else None),
        ),
    )
    model.add_boundary_condition(FixedSupport("edge-12", [1, 2]))
    guided = {"ux": 0.0, "uy": 0.0}
    if constrain_drill:
        guided["rz"] = 0.0
    model.add_boundary_condition(
        BoundaryCondition(
            "node-3-guided",
            [3],
            guided,
        )
    )
    return model


def _solve_native_contact_model(
    model: FEModel,
    *,
    surface: str = "top",
    z: float = 0.12,
    travel_z: float = -1.0,
    start_xy: tuple[float, float] = (-0.03, 0.3),
    dt: float = 5.0e-4,
    t_end: float = 5.0e-4,
    penalty: float = 100.0,
    max_iterations: int = 10,
    max_cutbacks: int = 2,
):
    return solve_transient_sphere_impact(
        model,
        TransientConfig(dt=dt, t_end=t_end),
        RigidSphereImpact(
            "qualified-s3-native-contact",
            radius=0.15,
            mass=1.0,
            start_point=(start_xy[0], start_xy[1], z),
            travel_direction=(0.0, 0.0, travel_z),
            speed=0.0,
        ),
        SphereContactConfig(
            penalty_stiffness=penalty,
            contact_surface=surface,
            max_contact_iterations=10,
        ),
        nonlinear_config=NonlinearTransientConfig(
            enabled=True,
            max_iterations=max_iterations,
            max_cutbacks=max_cutbacks,
            min_dt=1.0e-8,
            equilibrate_base_load=False,
        ),
    )


def _mixed_nonlinear_contact_model() -> FEModel:
    model = _nonlinear_contact_model()
    for node_id, point in {
        4: (3.0, 0.0, 0.0),
        5: (4.0, 0.0, 0.0),
        6: (4.0, 1.0, 0.0),
        7: (3.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *point)
    model.add_element(
        2,
        create_shell_element(
            2,
            [4, 5, 6, 7],
            "steel",
            thickness=0.02,
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed-q4-contact-target", [4, 5, 6, 7]))
    return model


@pytest.mark.parametrize(
    ("surface", "z", "travel_z", "moment_sign"),
    (("top", 0.12, -1.0, 1.0), ("bottom", -0.12, 1.0, -1.0)),
)
def test_top_level_nonlinear_impact_routes_exact_native_contact_trial(
    surface: str,
    z: float,
    travel_z: float,
    moment_sign: float,
) -> None:
    model = _nonlinear_contact_model()
    result = _solve_native_contact_model(
        model,
        surface=surface,
        z=z,
        travel_z=travel_z,
    )

    assert result.status == "completed"
    assert result.diagnostics["cutback_count"] == 0
    assert result.diagnostics["nonlinear_state_storage"][
        "native_rotation_activated"
    ] is True
    descriptor = result.diagnostics["declared_algebraic_contact_dynamics"]
    assert descriptor["enabled"] is True
    assert descriptor["mass_basis"]["compatible_global_nullity"] == 1
    assert descriptor["static_reduction"]["algebraic_dimension"] == 1
    assert descriptor["static_reduction"]["physical_dimension"] == 3
    assert result.active_contact_history
    record = result.active_contact_history[0][0]
    assert record["element_id"] == 1
    assert record["contact_point"][2] == pytest.approx(
        moment_sign * 0.01, abs=1.0e-14
    )
    moments = np.asarray(list(record["nodal_moments"].values()), dtype=float)
    assert float(np.linalg.norm(moments)) > 0.0
    assert moment_sign * float(np.sum(moments[:, 1])) > 0.0
    state = result.diagnostics["element_states"][1]
    element = model.mesh.get_element(1)
    assert element is not None
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    np.testing.assert_array_equal(
        state["committed_total_u"], result.displacements[-1][mapping]
    )


@pytest.mark.parametrize(
    ("surface", "z", "travel_z"),
    (("top", 0.12, -1.0), ("bottom", -0.12, 1.0)),
)
def test_generalized_section_impact_uses_the_same_native_contact_transaction(
    surface: str,
    z: float,
    travel_z: float,
) -> None:
    model = _nonlinear_contact_model(generalized=True)

    result = _solve_native_contact_model(
        model,
        surface=surface,
        z=z,
        travel_z=travel_z,
    )

    assert result.status == "completed"
    assert result.diagnostics["element_states"][1]["state_mode"] == (
        "stateless_generalized_section"
    )
    assert result.active_contact_history
    record = result.active_contact_history[0][0]
    assert record["nodal_moments"]
    assert float(np.linalg.norm(result.moment_impulse)) > 0.0
    assert result.diagnostics["declared_algebraic_contact_dynamics"][
        "static_reduction"
    ]["policy_id"] == "STATIC_ALGEBRAIC_TRANSIENT_REDUCTION_V1"


@pytest.mark.parametrize(
    ("target_element", "start_xy"),
    ((1, (-0.03, 0.3)), (2, (3.5, 0.5))),
)
def test_full_nonlinear_mixed_q4_s3_contact_preserves_target_semantics(
    target_element: int,
    start_xy: tuple[float, float],
) -> None:
    model = _mixed_nonlinear_contact_model()

    result = _solve_native_contact_model(model, start_xy=start_xy)

    assert result.status == "completed"
    assert result.active_contact_history
    record = result.active_contact_history[0][0]
    assert record["element_id"] == target_element
    if target_element == 1:
        assert record["nodal_moments"]
    else:
        assert "nodal_moments" not in record
    assert result.diagnostics["nonlinear_state_storage"][
        "native_rotation_activated"
    ] is True


def test_unconstrained_drill_matches_explicit_projected_contact_reference() -> None:
    free_model = _nonlinear_contact_model()
    projected_model = _nonlinear_contact_model(constrain_drill=True)
    free = _solve_native_contact_model(
        free_model,
        t_end=1.0e-3,
    )
    projected = _solve_native_contact_model(
        projected_model,
        t_end=1.0e-3,
    )

    assert free.status == projected.status == "completed"
    free_node = free_model.mesh.get_node(3)
    projected_node = projected_model.mesh.get_node(3)
    physical_local = (2, 3, 4)
    free_dofs = np.asarray([free_node.dofs[index] for index in physical_local])
    projected_dofs = np.asarray(
        [projected_node.dofs[index] for index in physical_local]
    )
    np.testing.assert_allclose(
        free.displacements[:, free_dofs],
        projected.displacements[:, projected_dofs],
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        free.velocities[:, free_dofs],
        projected.velocities[:, projected_dofs],
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        free.accelerations[:, free_dofs],
        projected.accelerations[:, projected_dofs],
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(
        free.contact_force_history, projected.contact_force_history
    )
    np.testing.assert_allclose(
        free.sphere_positions, projected.sphere_positions, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        free.force_impulse, projected.force_impulse, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        free.moment_impulse, projected.moment_impulse, rtol=0.0, atol=1.0e-15
    )
    free_drill = int(free_node.dofs[5])
    assert float(np.max(np.abs(free.displacements[:, free_drill]))) <= 1.0e-15
    assert free.diagnostics["declared_algebraic_contact_dynamics"][
        "static_reduction"
    ]["policy_id"] == "STATIC_ALGEBRAIC_TRANSIENT_REDUCTION_V1"
    assert projected.diagnostics["declared_algebraic_contact_dynamics"][
        "constrained_algebraic_dimension"
    ] == 0


def test_top_level_native_contact_cutbacks_do_not_commit_rejected_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anyfesolver.nonlinear_static as nonlinear_static

    captured: list[NonlinearStateStore] = []
    original_activate = nonlinear_static._activate_nonlinear_state_storage

    def capture_store(*args, **kwargs):
        result = original_activate(*args, **kwargs)
        if isinstance(result, NonlinearStateStore):
            captured.append(result)
        return result

    monkeypatch.setattr(
        nonlinear_static, "_activate_nonlinear_state_storage", capture_store
    )
    model = _nonlinear_contact_model()
    result = _solve_native_contact_model(
        model,
        dt=2.0e-3,
        t_end=2.0e-3,
        penalty=1.0e6,
        max_iterations=1,
        max_cutbacks=2,
    )

    assert result.status == "nonlinear_iteration_failed"
    assert result.diagnostics["cutback_count"] == 2
    assert len(captured) == 1
    store = captured[0]
    assert not store.has_active_trial
    native = store.native_rotation_store
    assert native is not None
    # Only the exact initial configuration was committed.  Neither failed
    # Newton candidate nor either cutback contact-only trial advanced history.
    assert native.generation == 1
    np.testing.assert_array_equal(
        native.committed_full_displacement,
        np.zeros(model.mesh.dof_manager.total_dofs),
    )
