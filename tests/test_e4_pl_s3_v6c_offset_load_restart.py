from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.e4_pl_s3_v2d_element import (
    FORMULATION_ID,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.e4_pl_s3_v2d_state import V2DStateError, canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.matrix_assembly import (
    assemble_external_load_tangent,
    assemble_load_vector,
)
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.shell_sections import GeneralizedShellSection


E = 210.0e9
NU = 0.3
H = 0.018
OFFSET = 0.004
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
        B=0.07 * H**2 * plane,
        D=H**3 / 12.0 * plane,
        As=(5.0 / 6.0) * E * H / (2.0 * (1.0 + NU)) * np.eye(2),
        mass_per_area=7850.0 * H,
    )


def _model(
    *,
    polarity: int = 1,
    offset: float = 0.0,
    generalized: bool = False,
    plastic: bool = False,
    order: tuple[int, int, int] = (1, 2, 3),
) -> tuple[FEModel, NativeParityE4PLS3V2DShellElement]:
    model = FEModel("s3-v2d-v6c")
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
    model.add_material("steel", E, NU, density=7850.0, hardening_curve=curve)
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    element = create_shell_element(
        1,
        order,
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=H,
        reference_normal=(0.0, 0.0, 1.0),
        director_polarity=polarity,
        reference_surface_offset=offset,
        material_direction=(1.0, 0.2, 0.0) if generalized else None,
        shell_section=_section() if generalized else None,
    )
    assert type(element) is NativeParityE4PLS3V2DShellElement
    model.add_element(1, element)
    return model, element


def test_v2d_offset_director_serialization_surface_coordinates_and_validation() -> None:
    model, element = _model(polarity=-1, offset=-OFFSET, generalized=True)
    assert element.physical_reference_director.tolist() == [0.0, 0.0, -1.0]
    assert element.material_mid_surface_offset_from_reference == OFFSET
    assert element.material_surface_offset_from_reference(-1.0) == pytest.approx(
        OFFSET - 0.5 * H
    )
    assert element.material_surface_offset_from_reference(1.0) == pytest.approx(
        OFFSET + 0.5 * H
    )
    payload = element.to_dict()
    restored = create_shell_element(
        payload["element_id"],
        payload["node_ids"],
        payload["material_name"],
        formulation="e4-pl-s3-v2d",
        thickness=payload["thickness"],
        reference_normal=payload["reference_normal"],
        director_polarity=payload["director_polarity"],
        reference_surface_offset=payload["reference_surface_offset"],
        material_direction=payload["material_direction"],
        material_angle_deg=payload["material_angle_deg"],
        shell_section=element.shell_section,
    )
    assert restored.director_polarity == -1
    assert restored.reference_surface_offset == -OFFSET
    for invalid in (True, 0, 2, -2, 1.0):
        with pytest.raises(ValueError, match="director_polarity"):
            _model(polarity=invalid)  # type: ignore[arg-type]
    for invalid_offset in (True, np.nan, np.inf):
        with pytest.raises((TypeError, ValueError), match="reference_surface_offset"):
            _model(offset=invalid_offset)  # type: ignore[arg-type]
    assert model.mesh.get_element(1) is element


def test_v2d_offset_director_operator_covariance_d3_and_virtual_work() -> None:
    plus_model, plus = _model(polarity=1, offset=OFFSET, generalized=True)
    minus_model, minus = _model(polarity=-1, offset=-OFFSET, generalized=True)
    material_plus = plus_model.get_material("steel")
    material_minus = minus_model.get_material("steel")
    plus_components = plus.compute_stiffness_components(plus_model.mesh, material_plus)
    minus_components = minus.compute_stiffness_components(minus_model.mesh, material_minus)
    for name in ("physical", "pl", "total"):
        np.testing.assert_allclose(
            minus_components[name], plus_components[name], rtol=3.0e-14, atol=3.0e-6
        )

    rng = np.random.default_rng(2026090201)
    displacement = 2.0e-4 * rng.standard_normal(18)
    recovered = plus.compute_variational_resultants(
        plus_model.mesh, displacement, material_plus
    )
    strain = np.concatenate(
        (
            recovered["membrane_strain"],
            recovered["curvature"],
            recovered["transverse_shear_strain"],
        ),
        axis=1,
    )
    resultant = np.concatenate(
        (
            recovered["membrane_resultants"],
            recovered["bending_resultants"],
            recovered["transverse_shear_resultants"],
        ),
        axis=1,
    )
    station_work = float(
        np.sum(recovered["physical_weights"] * np.einsum("gi,gi->g", strain, resultant))
    )
    matrix_work = float(displacement @ plus_components["physical"] @ displacement)
    assert station_work == pytest.approx(matrix_work, rel=3.0e-12, abs=2.0e-7)

    base = np.asarray(plus_components["total"])
    eye = np.eye(6)
    for order_zero in (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
        (0, 2, 1),
        (2, 1, 0),
        (1, 0, 2),
    ):
        order = tuple(value + 1 for value in order_zero)
        model, element = _model(
            polarity=1, offset=OFFSET, generalized=True, order=order
        )
        made = element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))
        permutation = np.zeros((18, 18))
        for external, canonical in enumerate(order_zero):
            permutation[
                6 * external : 6 * external + 6,
                6 * canonical : 6 * canonical + 6,
            ] = eye
        np.testing.assert_allclose(
            permutation.T @ made @ permutation,
            base,
            rtol=3.0e-14,
            atol=5.0e-6,
        )


@pytest.mark.parametrize("generalized,plastic", ((True, False), (False, True)))
def test_v2d_offset_director_nonlinear_tangent_matches_force_difference(
    generalized: bool, plastic: bool
) -> None:
    model, element = _model(
        polarity=-1,
        offset=-OFFSET,
        generalized=generalized,
        plastic=plastic,
    )
    material = model.get_material("steel")
    committed = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    displacement = np.arange(18, dtype=np.float64) / (2500.0 if plastic else 300000.0)
    if plastic:
        _force, _tangent, committed = element.compute_nonlinear_response(
            model.mesh, material, 0.75 * displacement, committed, 5, True
        )
        assert float(np.max(committed["alpha"])) > 0.0
    _force, tangent, _trial = element.compute_nonlinear_response(
        model.mesh, material, displacement, committed, 5, True
    )
    assert tangent is not None
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
    assert relative <= (3.0e-6 if plastic else 3.0e-9)


def test_v2d_dead_follower_pressure_tangent_and_distributed_couple_work() -> None:
    baseline: tuple[np.ndarray, np.ndarray] | None = None
    rng = np.random.default_rng(2026090202)
    displacement = 2.0e-3 * rng.standard_normal(18)
    direction = rng.standard_normal(18)
    direction /= np.linalg.norm(direction)
    pressure = 731.0
    for polarity, offset in ((1, OFFSET), (-1, -OFFSET)):
        model, element = _model(polarity=polarity, offset=offset)
        dead = LoadCase("v2d-dead")
        dead.add_pressure_load(1, pressure)
        dead_force, dead_info = assemble_load_vector(model, dead)
        np.testing.assert_array_equal(
            dead_force,
            element.compute_native_dead_pressure_load(model.mesh, pressure),
        )
        assert dead_info["qualified_s3_pressure_surfaces"][0][
            "pressure_surface_id"
        ] == "ELEMENT_NODAL_REFERENCE_SURFACE_V1"

        follower = LoadCase("v2d-follower", follower_pressure=True)
        follower.add_pressure_load(1, pressure)
        force, info = assemble_load_vector(model, follower, displacement)
        tangent, tangent_info = assemble_external_load_tangent(
            model, follower, displacement
        )
        tangent_array = tangent.toarray()
        step = 2.0e-7
        plus, _ = assemble_load_vector(model, follower, displacement + step * direction)
        minus, _ = assemble_load_vector(model, follower, displacement - step * direction)
        numerical = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(
            tangent_array @ direction, numerical, rtol=2.0e-9, atol=2.0e-7
        )
        assert info["qualified_s3_pressure_surfaces"] == tangent_info[
            "qualified_s3_pressure_surfaces"
        ]
        np.testing.assert_array_equal(force.reshape(3, 6)[:, 3:], 0.0)

        couple = np.zeros(18)
        couple.reshape(3, 6)[:, 3:] = np.asarray(
            ((2.0, -1.0, 0.7), (-3.0, 5.0, -2.0), (4.0, 0.5, 1.0))
        )
        couple_case = LoadCase("v2d-couple")
        couple_case.element_loads[1] = couple.copy()
        assembled, _ = assemble_load_vector(model, couple_case)
        np.testing.assert_array_equal(assembled, couple)
        virtual = rng.standard_normal(18)
        assert float(virtual @ assembled) == float(assembled @ virtual)
        if baseline is None:
            baseline = force, tangent_array
        else:
            np.testing.assert_array_equal(force, baseline[0])
            np.testing.assert_array_equal(tangent_array, baseline[1])


def _restart_model() -> tuple[FEModel, LoadCase]:
    model, _element = _model(plastic=True)
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
    load = LoadCase("v2d-restart-load")
    load.add_nodal_load(2, [2.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def test_v2d_solver_checkpoint_exact_split_and_state_mutation_rejection() -> None:
    common = {
        "max_iterations": 12,
        "tolerance": 1.0e-10,
        "convergence_settings": "legacy",
        "kinematics": "corotational",
        "corotational_tangent": "consistent",
    }
    full_model, full_load = _restart_model()
    full = solve_static_nonlinear(
        full_model,
        full_load,
        max_load_factor=0.20,
        num_steps=4,
        emit_restart_checkpoint=True,
        **common,
    )
    first_model, first_load = _restart_model()
    first = solve_static_nonlinear(
        first_model,
        first_load,
        max_load_factor=0.10,
        num_steps=2,
        emit_restart_checkpoint=True,
        **common,
    )
    resumed_model, resumed_load = _restart_model()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.20,
        num_steps=2,
        restart_checkpoint=first.restart_checkpoint_bytes(),
        emit_restart_checkpoint=True,
        **common,
    )
    assert full.status == resumed.status == "completed"
    np.testing.assert_array_equal(resumed.displacements, full.displacements)
    assert resumed.restart_checkpoint_bytes() == full.restart_checkpoint_bytes()
    state = resumed.element_states[1]
    assert state["formulation_id"] == FORMULATION_ID
    assert state["solver_kinematics"] == "COROTATIONAL"
    dofs = resumed_model.mesh.get_element(1).get_dof_mapping(resumed_model.mesh)
    np.testing.assert_array_equal(
        state["committed_total_u"], resumed.displacements[np.asarray(dofs)]
    )

    element = resumed_model.mesh.get_element(1)
    material = resumed_model.get_material("steel")
    raw = element.serialize_nonlinear_state(resumed_model.mesh, material, state, 5)
    old = copy.deepcopy(state)
    old["schema"] = "anysolver.e4-pl-s3-v2d-native-committed-state-v1"
    with pytest.raises(V2DStateError, match="schema"):
        element.deserialize_nonlinear_state(
            resumed_model.mesh, material, canonical_json_bytes(old), 5
        )
    foreign = raw.replace(FORMULATION_ID.encode(), b"E4_PL_QUALIFIED_S3_COMPANION_V1____")
    with pytest.raises(V2DStateError):
        element.deserialize_nonlinear_state(resumed_model.mesh, material, foreign, 5)
