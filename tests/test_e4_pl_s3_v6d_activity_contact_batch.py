from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.contact import (
    RigidSphereImpact,
    SphereContactConfig,
    assemble_sphere_contact_load_vector,
)
from anyfesolver.e4_pl_s3_v2d_element import (
    BATCH_POLICY_ID,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.e4_pl_s3_v2d_state import (
    V2DStateError,
    canonical_json_bytes,
    seal_v2d_state,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.fracture import FractureConfig
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.validation import load_vector_resultant
import anyfesolver.s3_v2d_fast_assembly as v2d_batch


COORDINATES = ((0.0, 0.0, 0.0), (1.2, 0.05, 0.0), (0.2, 0.95, 0.0))


def _plastic_model() -> tuple[FEModel, LoadCase]:
    model = FEModel("s3-v2d-v6d-activity")
    curve = DNVC208MaterialCurve(
        sigma_prop=80.0e6,
        sigma_yield=85.0e6,
        sigma_yield_2=100.0e6,
        eps_p_y1=0.002,
        eps_p_y2=0.02,
        K=280.0e6,
        n=0.18,
    )
    model.add_material("steel", 210.0e9, 0.3, density=7850.0, hardening_curve=curve)
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        NativeParityE4PLS3V2DShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.018,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "fixed-1",
            [1],
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "guided-2",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "fixed-3",
            [3],
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("s3-v2d-v6d-pull")
    load.add_nodal_load(2, [2.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _fracture_seed(model: FEModel) -> dict:
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    state = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    state["alpha"][:] = 0.01
    return seal_v2d_state(state)


def _fracture_config() -> FractureConfig:
    return FractureConfig(
        threshold=0.001,
        residual_stiffness_fraction=0.1,
        max_deleted_fraction=1.0,
    )


def test_v2d_activity_dispositions_are_closed_and_mutation_detected() -> None:
    model, _load = _plastic_model()
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    active = _fracture_seed(model)
    deleted = element.seal_noncurrent_deleted_state(
        model.mesh,
        material,
        np.zeros(18),
        active,
        3,
        deletion_step_index=1,
        deletion_load_factor=0.25,
        residual_stiffness_fraction=0.1,
        trigger_name="max_equivalent_plastic_strain",
    )
    assert deleted["qualified_s3_activity_disposition"]["status"] == (
        "DELETED_FROZEN_NONCURRENT"
    )
    element.validate_noncurrent_deleted_state(
        model.mesh,
        material,
        deleted,
        3,
        expected_deletion_step_index=1,
        expected_deletion_load_factor=0.25,
        expected_residual_stiffness_fraction=0.1,
        expected_trigger_name="max_equivalent_plastic_strain",
    )
    with pytest.raises(V2DStateError, match="noncurrent"):
        element.validate_model_bound_nonlinear_state(
            model.mesh, material, deleted, 3
        )
    mutated = copy.deepcopy(deleted)
    mutated["qualified_s3_activity_disposition"]["trigger_name"] = "wrong"
    with pytest.raises(V2DStateError):
        element.validate_noncurrent_deleted_state(
            model.mesh, material, mutated, 3
        )

    failed = element.mark_noncurrent_failed_state(
        model.mesh,
        material,
        np.full(18, 1.0e-6),
        active,
        3,
        failure_reason="bounded-test",
    )
    assert failed["qualified_s3_activity_disposition"]["status"] == (
        "FAILED_NONAUTHORITATIVE"
    )
    element.validate_noncurrent_failed_state(model.mesh, material, failed, 3)
    with pytest.raises(V2DStateError, match="noncurrent"):
        element.compute_nonlinear_response(
            model.mesh, material, np.zeros(18), failed, 3, True
        )


def test_v2d_deleted_checkpoint_replay_keeps_frozen_history_exact() -> None:
    model, load = _plastic_model()
    first = solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.10,
        num_steps=2,
        num_layers=3,
        initial_element_states={1: _fracture_seed(model)},
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert first.status == "completed"
    assert first.info["fracture_summary"]["deleted_element_ids"] == [1]
    frozen = canonical_json_bytes(first.element_states[1])

    resumed_model, resumed_load = _plastic_model()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.15,
        num_steps=1,
        num_layers=3,
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
        restart_checkpoint=first.restart_checkpoint_bytes(),
        emit_restart_checkpoint=True,
    )
    assert resumed.status == "completed"
    assert canonical_json_bytes(resumed.element_states[1]) == frozen
    assert resumed.info["qualified_s3_committed_state_lifecycle"][
        "restored_deleted_frozen_element_ids"
    ] == [1]


@pytest.mark.parametrize(
    ("polarity", "offset", "surface", "position", "expected_z"),
    (
        (1, 0.0, "top", (0.25, 0.25, 0.15), 0.1),
        (-1, 0.0, "top", (0.25, 0.25, -0.15), -0.1),
        (1, 0.03, "top", (0.25, 0.25, 0.12), 0.07),
    ),
)
def test_v2d_contact_uses_physical_director_surface_and_exact_work(
    polarity: int,
    offset: float,
    surface: str,
    position: tuple[float, float, float],
    expected_z: float,
) -> None:
    model = FEModel("s3-v2d-v6d-contact")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for node_id, point in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0)), start=1):
        model.add_node(node_id, *point)
    element = NativeParityE4PLS3V2DShellElement(
        1,
        [1, 2, 3],
        "soft",
        thickness=0.2,
        reference_normal=(0.0, 0.0, 1.0),
        director_polarity=polarity,
        reference_surface_offset=offset,
    )
    model.add_element(1, element)
    center = np.asarray(position, dtype=float)
    sphere = RigidSphereImpact(
        "v2d-contact",
        radius=0.2,
        mass=1.0,
        start_point=position,
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )
    load, sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface=surface),
        sphere_position=center,
        sphere_velocity=np.zeros(3),
    )
    assert len(records) == 1
    record = records[0]
    assert record.contact_point[2] == pytest.approx(expected_z, abs=2.0e-14)
    np.testing.assert_allclose(record.structure_force, -sphere_force, atol=1.0e-13)
    resultant = load_vector_resultant(model, load)
    np.testing.assert_allclose(resultant.force, record.structure_force, atol=2.0e-13)
    np.testing.assert_allclose(
        resultant.moment,
        np.cross(record.contact_point, record.structure_force),
        atol=3.0e-13,
    )


def _batch_model(count: int = 4) -> FEModel:
    model = FEModel("s3-v2d-v6d-batch")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    node_id = 1
    for index in range(count):
        x = 2.0 * index
        ids = [node_id, node_id + 1, node_id + 2]
        for current, point in zip(ids, ((x, 0, 0), (x + 1, 0, 0), (x, 1, 0))):
            model.add_node(current, *point)
        model.add_element(
            index + 1,
            NativeParityE4PLS3V2DShellElement(
                index + 1,
                ids,
                "steel",
                thickness=0.02,
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
        node_id += 3
    return model


def test_v2d_bounded_stiffness_plan_is_scalar_exact_and_mutation_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _batch_model()
    cold, cold_info = assemble_stiffness_matrix(model)
    warm, warm_info = assemble_stiffness_matrix(model)
    cold_diag = cold_info["diagnostics"]["s3_v2d_exact_stiffness"]
    warm_diag = warm_info["diagnostics"]["s3_v2d_exact_stiffness"]
    assert cold_diag["policy_id"] == BATCH_POLICY_ID
    assert cold_diag["plan_reused"] is False
    assert warm_diag["plan_reused"] is True
    np.testing.assert_array_equal(cold.toarray(), warm.toarray())

    scalar_model = _batch_model()
    monkeypatch.setattr(v2d_batch, "v2d_fast_candidate", lambda _element: False)
    scalar, scalar_info = assemble_stiffness_matrix(scalar_model)
    assert "s3_v2d_exact_stiffness" not in scalar_info["diagnostics"]
    np.testing.assert_array_equal(cold.toarray(), scalar.toarray())

    monkeypatch.undo()
    model.materials["steel"].elastic_modulus *= 1.01
    changed, changed_info = assemble_stiffness_matrix(model)
    assert changed_info["diagnostics"]["s3_v2d_exact_stiffness"][
        "plan_reused"
    ] is False
    assert not np.array_equal(warm.toarray(), changed.toarray())
