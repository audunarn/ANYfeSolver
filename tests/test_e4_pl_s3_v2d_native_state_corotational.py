from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver.corotational import (
    corotational_element_response,
    rotation_matrix_from_vector,
)
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.e4_pl_s3_v2d_state import (
    STATE_SCHEMA,
    V2DStateError,
    canonical_json_bytes,
)
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.shell_sections import GeneralizedShellSection


E = 210.0e9
NU = 0.3
H = 0.018
COORDINATES = np.asarray(
    ((0.0, 0.0, 0.0), (1.2, 0.05, 0.0), (0.2, 0.95, 0.0)),
    dtype=np.float64,
)


def _section() -> GeneralizedShellSection:
    scale = E / (1.0 - NU**2)
    plane = scale * np.asarray(
        ((1.0, NU, 0.0), (NU, 1.0, 0.0), (0.0, 0.0, 0.5 * (1.0 - NU)))
    )
    return GeneralizedShellSection(
        A=H * plane,
        B=1.0e-4 * H * plane,
        D=H**3 / 12.0 * plane,
        As=(5.0 / 6.0) * E * H / (2.0 * (1.0 + NU)) * np.eye(2),
        mass_per_area=7850.0 * H,
    )


def _model(*, generalized: bool = False, plastic: bool = False) -> tuple[FEModel, NativeParityE4PLS3V2DShellElement]:
    model = FEModel("s3-v2d-native-state")
    curve = (
        DNVC208MaterialCurve(
            sigma_prop=80.0e6,
            sigma_yield=85.0e6,
            sigma_yield_2=100.0e6,
            eps_p_y1=0.002,
            eps_p_y2=0.02,
            K=280.0e6,
            n=0.18,
        )
        if plastic
        else None
    )
    model.add_material(
        "steel", E, NU, density=7850.0, hardening_curve=curve
    )
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    element = create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=H,
        reference_normal=(0.0, 0.0, 1.0),
        material_direction=(1.0, 0.2, 0.0) if generalized else None,
        shell_section=_section() if generalized else None,
    )
    assert type(element) is NativeParityE4PLS3V2DShellElement
    model.add_element(1, element)
    return model, element


def test_v2d_model_bound_state_roundtrip_and_foreign_restart_rejection() -> None:
    model, element = _model(generalized=True)
    material = model.get_material("steel")
    committed = element.init_model_bound_nonlinear_state(
        model.mesh, material, 5
    )
    assert committed["schema"] == STATE_SCHEMA
    raw = element.serialize_nonlinear_state(model.mesh, material, committed, 5)
    restored = element.deserialize_nonlinear_state(model.mesh, material, raw, 5)
    assert canonical_json_bytes(restored) == raw
    assert canonical_json_bytes(committed) == raw

    malformed = raw.replace(
        b"CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
        b"E4_PL_QUALIFIED_S3_COMPANION_V1__________",
    )
    with pytest.raises(V2DStateError):
        element.deserialize_nonlinear_state(model.mesh, material, malformed, 5)
    changed_model, changed_element = _model(generalized=True)
    changed_model.mesh.nodes[2].x += 0.01
    with pytest.raises(V2DStateError, match="binding"):
        changed_element.deserialize_nonlinear_state(
            changed_model.mesh, changed_model.get_material("steel"), raw, 5
        )
    for invalid_layers in (True, 3.5, 0, -1):
        with pytest.raises(V2DStateError, match="num_layers"):
            element.init_model_bound_nonlinear_state(
                model.mesh, material, invalid_layers  # type: ignore[arg-type]
            )


def test_generalized_trial_is_deterministic_and_does_not_mutate_committed_state() -> None:
    model, element = _model(generalized=True)
    material = model.get_material("steel")
    committed = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    before = canonical_json_bytes(committed)
    displacement = np.arange(18, dtype=np.float64) / 200000.0
    first = element.compute_nonlinear_response(
        model.mesh, material, displacement, committed, 5, True
    )
    second = element.compute_nonlinear_response(
        model.mesh, material, displacement, committed, 5, True
    )
    assert canonical_json_bytes(committed) == before
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert canonical_json_bytes(first[2]) == canonical_json_bytes(second[2])
    assert not np.array_equal(
        first[2]["committed_total_u"], committed["committed_total_u"]
    )


def test_generalized_native_tangent_matches_independent_force_difference() -> None:
    model, element = _model(generalized=True)
    material = model.get_material("steel")
    committed = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    displacement = np.arange(18, dtype=np.float64) / 400000.0
    force, tangent, _trial = element.compute_nonlinear_response(
        model.mesh, material, displacement, committed, 5, True
    )
    assert tangent is not None and np.all(np.isfinite(force))
    numerical = np.zeros((18, 18), dtype=np.float64)
    step = 2.0e-8
    for column in range(18):
        plus = displacement.copy()
        minus = displacement.copy()
        plus[column] += step
        minus[column] -= step
        force_plus, _, _ = element.compute_nonlinear_response(
            model.mesh, material, plus, committed, 5, False
        )
        force_minus, _, _ = element.compute_nonlinear_response(
            model.mesh, material, minus, committed, 5, False
        )
        numerical[:, column] = (force_plus - force_minus) / (2.0 * step)
    relative = np.linalg.norm(tangent - numerical) / max(
        float(np.linalg.norm(numerical)), 1.0
    )
    assert relative <= 2.0e-9


def test_layered_trial_commit_revert_and_initial_field_work() -> None:
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    initial = element.init_model_bound_nonlinear_state(
        model.mesh,
        material,
        5,
        initial_fields={
            "initial_membrane_stress": (2.0e6, 1.0e6, 0.2e6),
            "initial_membrane_prestrain": (1.0e-6, -0.5e-6, 0.2e-6),
        },
        initial_field_provenance={"kind": "shell", "source": "v6b-test"},
    )
    initial_bytes = canonical_json_bytes(initial)
    residual, _, zero_trial = element.compute_nonlinear_response(
        model.mesh, material, np.zeros(18), initial, 5, True
    )
    assert float(np.linalg.norm(residual)) > 0.0
    assert zero_trial["initial_field_provenance"]["source"] == "v6b-test"

    load = np.zeros(18, dtype=np.float64)
    load[6] = 0.012
    load[12] = -0.003
    _force, tangent, plastic_trial = element.compute_nonlinear_response(
        model.mesh, material, load, initial, 5, True
    )
    assert tangent is not None
    assert float(np.max(plastic_trial["alpha"])) > 0.0
    assert canonical_json_bytes(initial) == initial_bytes
    # Re-evaluating from the same committed parent reproduces the discarded trial.
    _force_2, tangent_2, plastic_trial_2 = element.compute_nonlinear_response(
        model.mesh, material, load, initial, 5, True
    )
    np.testing.assert_array_equal(tangent_2, tangent)
    assert canonical_json_bytes(plastic_trial_2) == canonical_json_bytes(plastic_trial)

    unload = 0.4 * load
    _unload_force, _unload_tangent, unload_trial = element.compute_nonlinear_response(
        model.mesh, material, unload, plastic_trial, 5, True
    )
    assert np.all(unload_trial["alpha"] >= plastic_trial["alpha"])


def test_layered_algorithmic_tangent_matches_plastic_force_difference() -> None:
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    committed_displacement = np.zeros(18, dtype=np.float64)
    committed_displacement[6] = 0.008
    committed_displacement[12] = -0.002
    _force, _tangent, committed = element.compute_nonlinear_response(
        model.mesh, material, committed_displacement, virgin, 5, True
    )
    assert float(np.max(committed["alpha"])) > 0.0

    trial_displacement = committed_displacement.copy()
    trial_displacement[6] += 2.0e-4
    trial_displacement[12] -= 5.0e-5
    _trial_force, tangent, _trial = element.compute_nonlinear_response(
        model.mesh, material, trial_displacement, committed, 5, True
    )
    assert tangent is not None
    numerical = np.zeros((18, 18), dtype=np.float64)
    step = 2.0e-8
    for column in range(18):
        plus = trial_displacement.copy()
        minus = trial_displacement.copy()
        plus[column] += step
        minus[column] -= step
        force_plus, _, _ = element.compute_nonlinear_response(
            model.mesh, material, plus, committed, 5, False
        )
        force_minus, _, _ = element.compute_nonlinear_response(
            model.mesh, material, minus, committed, 5, False
        )
        numerical[:, column] = (force_plus - force_minus) / (2.0 * step)
    relative = np.linalg.norm(tangent - numerical) / max(
        float(np.linalg.norm(numerical)), 1.0
    )
    assert relative <= 2.0e-6


def test_yielded_s3_corotational_tangent_uses_fixed_predecessor() -> None:
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    rigid = _rigid_rotation_field(model, np.radians(20.), (0.2, 0.6, 1.0))
    committed_u = rigid.copy()
    committed_u[6] += .012
    committed_u[12] -= .003
    _force, _tangent, committed = corotational_element_response(
        model, 1, element, committed_u, tangent=True,
        committed_state=virgin, num_layers=5, tangent_mode="consistent")
    assert float(np.max(committed["alpha"])) > 0.
    predecessor_bytes = canonical_json_bytes(committed)

    trial_u = committed_u.copy()
    trial_u[6] += .0002
    trial_u[12] -= .00005
    force, tangent, candidate = corotational_element_response(
        model, 1, element, trial_u, tangent=True,
        committed_state=committed, num_layers=5, tangent_mode="consistent")
    assert float(np.max(candidate["alpha"])) > float(np.max(committed["alpha"]))
    assert np.isfinite(force).all() and np.isfinite(tangent).all()
    rng = np.random.default_rng(20260929)
    direction = rng.normal(size=18)
    direction /= np.linalg.norm(direction)
    active = candidate["alpha"]-committed["alpha"] > 1e-9
    assert np.any(active)
    force_rows = np.array([6*node+axis for node in range(3)
                           for axis in range(3)])
    moment_rows = np.array([6*node+axis for node in range(3)
                            for axis in range(3, 6)])
    errors = {"force": [], "moment": []}
    for step in (2e-7, 1e-7, 5e-8):
        plus, _, plus_state = corotational_element_response(
            model, 1, element, trial_u+step*direction, tangent=False,
            committed_state=committed, num_layers=5, tangent_mode="consistent")
        minus, _, minus_state = corotational_element_response(
            model, 1, element, trial_u-step*direction, tangent=False,
            committed_state=committed, num_layers=5, tangent_mode="consistent")
        for state in (plus_state, minus_state):
            np.testing.assert_array_equal(
                state["alpha"]-committed["alpha"] > 1e-9, active)
        observed = (plus-minus)/(2*step)
        predicted = tangent@direction
        for name, rows in (("force", force_rows), ("moment", moment_rows)):
            scale = float(np.linalg.norm(observed[rows]))
            assert scale > 1.
            errors[name].append(float(np.linalg.norm(
                observed[rows]-predicted[rows])/scale))
    assert max(errors["force"]) < 1e-5
    assert max(errors["moment"]) < 1e-5
    assert canonical_json_bytes(committed) == predecessor_bytes


def test_yielded_s3_solver_state_round_trip_preserves_next_trial() -> None:
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    global_u = _rigid_rotation_field(model, np.radians(20.), (0.2, 0.6, 1.0))
    global_u[6] += .012
    global_u[12] -= .003
    _force, _tangent, candidate = corotational_element_response(
        model, 1, element, global_u, tangent=True,
        committed_state=virgin, num_layers=5, tangent_mode="consistent")
    assert float(np.max(candidate["alpha"])) > 0.
    accepted = element.seal_solver_integrated_nonlinear_state(
        model.mesh, material, candidate, 5, global_u,
        kinematics="COROTATIONAL")
    raw = element.serialize_nonlinear_state(model.mesh, material, accepted, 5)
    restored = element.deserialize_nonlinear_state(
        model.mesh, material, raw, 5,
        expected_committed_total_u=global_u)
    np.testing.assert_array_equal(restored["committed_total_u"], global_u)
    assert restored["solver_kinematics"] == "COROTATIONAL"
    trial = global_u.copy()
    trial[6] += .0002
    trial[12] -= .00005
    first = corotational_element_response(
        model, 1, element, trial, tangent=True,
        committed_state=accepted, num_layers=5, tangent_mode="consistent")
    repeated = corotational_element_response(
        model, 1, element, trial, tangent=True,
        committed_state=restored, num_layers=5, tangent_mode="consistent")
    for original, resumed in zip(first[:2], repeated[:2]):
        np.testing.assert_array_equal(original, resumed)
    np.testing.assert_array_equal(first[2]["alpha"], repeated[2]["alpha"])


def _rigid_rotation_field(model: FEModel, angle: float, axis: tuple[float, float, float]) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= float(np.linalg.norm(direction))
    rotation = rotation_matrix_from_vector(float(angle) * direction)
    vector = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    for node in model.mesh.nodes.values():
        coordinates = np.asarray((node.x, node.y, node.z), dtype=np.float64)
        vector[np.asarray(node.dofs[:3])] = rotation @ coordinates - coordinates
        vector[np.asarray(node.dofs[3:6])] = float(angle) * direction
    return vector


def test_v2d_corotational_rigid_rotation_objectivity_and_consistent_tangent() -> None:
    model, element = _model()
    rigid = _rigid_rotation_field(model, np.radians(70.0), (0.2, 0.6, 1.0))
    force, _tangent, _state = corotational_element_response(
        model, 1, element, rigid, tangent=True, tangent_mode="consistent"
    )
    assert force is not None
    scale = E * H * max(float(np.linalg.norm(rigid)), 1.0)
    assert float(np.linalg.norm(force)) <= 2.0e-12 * scale

    displacement = rigid.copy()
    displacement[model.mesh.nodes[2].dofs[0]] += 0.001
    displacement[model.mesh.nodes[3].dofs[2]] -= 0.0015
    displacement[model.mesh.nodes[3].dofs[4]] += 0.004
    force, tangent, _state = corotational_element_response(
        model, 1, element, displacement, tangent=True, tangent_mode="consistent"
    )
    assert force is not None and tangent is not None
    numerical = np.zeros_like(tangent)
    step = 2.0e-7
    for column in range(displacement.size):
        plus = displacement.copy()
        minus = displacement.copy()
        plus[column] += step
        minus[column] -= step
        force_plus, _, _ = corotational_element_response(
            model, 1, element, plus, tangent=False, tangent_mode="consistent"
        )
        force_minus, _, _ = corotational_element_response(
            model, 1, element, minus, tangent=False, tangent_mode="consistent"
        )
        numerical[:, column] = (force_plus - force_minus) / (2.0 * step)
    relative = np.linalg.norm(tangent - numerical) / max(
        float(np.linalg.norm(numerical)), 1.0
    )
    assert relative <= 5.0e-7
