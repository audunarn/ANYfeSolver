from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver._native_rotation_state import create_native_rotation_state_store
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_state import (
    GENERALIZED_INITIAL_FIELD_POLICY_ID,
    GENERALIZED_NONLINEAR_STATE_SCHEMA,
    S3CommittedStateError,
    canonical_json_bytes,
    seal_committed_s3_state,
    strict_canonical_json_loads,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import ShellInitialField, solve_static_nonlinear
from anyfesolver.recovery import _recover_qualified_s3_committed_state
from anyfesolver.shell_sections import GeneralizedShellSection


REFERENCE = np.asarray(
    ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (0.24, 0.94, 0.0)),
    dtype=np.float64,
)
OWNER = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
THICKNESS = 0.027
MATERIAL_DIRECTION = np.asarray((0.91, 0.37, 0.0), dtype=np.float64)


def _section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="B-coupled-generalized-initial-field",
        A=np.asarray(
            (
                (2.4e8, 0.42e8, 0.13e8),
                (0.42e8, 1.35e8, -0.08e8),
                (0.13e8, -0.08e8, 0.71e8),
            )
        ),
        B=np.asarray(
            (
                (2.1e4, -0.7e4, 0.3e4),
                (0.4e4, 1.6e4, -0.2e4),
                (-0.5e4, 0.1e4, 0.8e4),
            )
        ),
        D=np.asarray(
            (
                (3.2e4, 0.55e4, 0.12e4),
                (0.55e4, 2.1e4, -0.09e4),
                (0.12e4, -0.09e4, 1.15e4),
            )
        ),
        As=np.asarray(((0.82e8, 0.11e8), (0.11e8, 0.59e8))),
        mass_per_area=41.0,
        rotary_inertia_per_area=0.014,
    )


def _model(
    *,
    polarity: int = 1,
    node_order: tuple[int, int, int] = (1, 2, 3),
    constrained: bool = False,
    reference_surface_offset: float = 0.0,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified-s3-generalized-initial-fields")
    model.add_material("carrier", 70.0e9, 0.31, density=2700.0)
    for node_id, coordinates in enumerate(REFERENCE, start=1):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLS3ShellElement(
        1,
        list(node_order),
        "carrier",
        thickness=THICKNESS,
        material_direction=MATERIAL_DIRECTION,
        material_angle_deg=13.0,
        shell_section=_section(),
        reference_normal=OWNER,
        director_polarity=polarity,
        reference_surface_offset=reference_surface_offset,
    )
    model.add_element(1, element)
    if constrained:
        model.add_boundary_condition(FixedSupport("node-1", [1]))
        model.add_boundary_condition(
            BoundaryCondition(
                "node-2-axial-only",
                [2],
                {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            )
        )
        model.add_boundary_condition(FixedSupport("node-3", [3]))
    return model, element


def _load(value: float = 25.0) -> LoadCase:
    load = LoadCase("generalized-initial-field-load")
    load.add_nodal_load(2, [float(value), 0.0, 0.0, 0.0, 0.0, 0.0])
    return load


def _view(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object] | None,
):
    total = np.asarray(total_u, dtype=np.float64).reshape(18)
    if state is None:
        committed_u = np.zeros(18, dtype=np.float64)
        committed_q = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    else:
        committed_u = np.asarray(state["committed_total_u"], dtype=np.float64)
        committed_q = np.asarray(
            state["committed_nodal_rotation_matrices"], dtype=np.float64
        )
    node_ids = tuple(element.node_ids)
    store = create_native_rotation_state_store(
        node_ids,
        rotational_dofs={
            node_id: (6 * row + 3, 6 * row + 4, 6 * row + 5)
            for row, node_id in enumerate(node_ids)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(node_ids)},
        coordinate_node_ids=node_ids,
        committed_full_displacement=committed_u,
        committed_full_coordinates=(
            element.get_node_coordinates(model.mesh)
            + committed_u.reshape(3, 6)[:, :3]
        ),
        committed_rotation_matrices={
            node_id: committed_q[row] for row, node_id in enumerate(node_ids)
        },
    )
    assert store is not None
    reference = element.get_node_coordinates(model.mesh)
    token = store.begin_trial(total, reference + total.reshape(3, 6)[:, :3])
    return store.element_view(
        element.element_id,
        node_ids,
        element.native_reference_directors(model.mesh),
        trial_token=token,
    )


def _response(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object],
):
    return element.compute_nonlinear_response(
        model.mesh,
        model.get_material("carrier"),
        total_u,
        state,
        3,
        True,
        native_rotation_trial=_view(model, element, total_u, state),
    )


def _seven_station_fields() -> dict[str, np.ndarray]:
    station = np.arange(7, dtype=np.float64)[:, None]
    return {
        "initial_membrane_stress": np.asarray((2.1e6, -0.7e6, 0.3e6))
        + station * np.asarray((1.1e4, -0.7e4, 0.5e4)),
        "initial_bending_stress": np.asarray((0.8e6, 0.2e6, -0.4e6))
        + station * np.asarray((-0.6e4, 0.3e4, 0.2e4)),
        "initial_membrane_prestrain": np.asarray((1.4e-5, -0.9e-5, 0.5e-5))
        + station * np.asarray((0.04e-5, -0.02e-5, 0.03e-5)),
        "initial_curvature_prestrain": np.asarray((2.2e-3, -1.1e-3, 0.7e-3))
        + station * np.asarray((-0.03e-3, 0.02e-3, 0.01e-3)),
    }


def _expected_offsets(
    fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    prestrain = np.zeros((7, 8), dtype=np.float64)
    prestrain[:, :3] = fields["initial_membrane_prestrain"]
    prestrain[:, 3:6] = fields["initial_curvature_prestrain"]
    resultant = np.zeros((7, 8), dtype=np.float64)
    resultant[:, :3] = THICKNESS * fields["initial_membrane_stress"]
    resultant[:, 3:6] = (
        THICKNESS * THICKNESS / 6.0
    ) * fields["initial_bending_stress"]
    return prestrain, resultant


def test_generalized_initial_fields_are_integrated_without_fabricated_layers() -> None:
    model, element = _model()
    fields = _seven_station_fields()
    provenance = {
        "kind": "shell",
        "source": "seven-station-generalized-field",
        "components": sorted(fields),
    }
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        11,
        initial_fields=fields,
        initial_field_provenance=provenance,
    )
    expected_prestrain, expected_resultant = _expected_offsets(fields)
    constitutive, _membrane = element._constitutive(
        model.get_material("carrier"), np.asarray(state["reference_frame"])
        if "reference_frame" in state
        else element.compute_stiffness_components(
            model.mesh, model.get_material("carrier")
        )["frame"],
    )

    assert state["state_schema"] == GENERALIZED_NONLINEAR_STATE_SCHEMA
    assert state["generalized_initial_field_policy_id"] == (
        GENERALIZED_INITIAL_FIELD_POLICY_ID
    )
    assert state["initial_field_provenance"] == provenance
    assert isinstance(state["initial_fields_fingerprint"], str)
    for name, values in fields.items():
        np.testing.assert_array_equal(state[name], values)
    np.testing.assert_array_equal(
        state["initial_generalized_prestrain"], expected_prestrain
    )
    np.testing.assert_array_equal(
        state["initial_generalized_resultant"], expected_resultant
    )
    np.testing.assert_array_equal(
        state["station_generalized_resultant"],
        -expected_prestrain @ constitutive.T + expected_resultant,
    )
    assert state["physical_layer_recovery_available"] is False
    for forbidden in (
        "plastic_strain",
        "alpha",
        "layer_strain",
        "layer_stress",
        "layer_stress_material",
    ):
        assert forbidden not in state
    assert "initial_fields" not in element.capability_gaps

    raw = canonical_json_bytes(state)
    decoded = strict_canonical_json_loads(raw)
    validated = element.validate_model_bound_nonlinear_state(
        model.mesh, model.get_material("carrier"), decoded, 3
    )
    assert canonical_json_bytes(validated) == raw


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("initial_membrane_stress", "initial-fields fingerprint"),
        ("initial_generalized_prestrain", "initial generalized prestrain"),
        ("initial_generalized_resultant", "initial generalized resultant"),
        ("station_generalized_resultant", "resultants contradict"),
    ),
)
def test_generalized_initial_field_state_mutations_fail_closed(
    field: str, message: str
) -> None:
    model, element = _model()
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        3,
        initial_fields=_seven_station_fields(),
        initial_field_provenance={"source": "mutation-authority"},
    )
    mutated = copy.deepcopy(state)
    mutated[field][0, 0] += 1.0e-9
    mutated = seal_committed_s3_state(mutated)
    with pytest.raises(S3CommittedStateError, match=message):
        element.validate_model_bound_nonlinear_state(
            model.mesh, model.get_material("carrier"), mutated, 3
        )


def test_generalized_V3_hot_restart_fails_closed_and_V4_binds_offset_and_fields() -> None:
    fields = _seven_station_fields()
    model, element = _model()
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        3,
        initial_fields=fields,
        initial_field_provenance={"source": "V4-binding"},
    )
    repeated = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        3,
        initial_fields=copy.deepcopy(fields),
        initial_field_provenance={"source": "V4-binding"},
    )
    assert canonical_json_bytes(repeated) == canonical_json_bytes(state)
    old = copy.deepcopy(state)
    old["state_schema"] = "anysolver.e4_pl_s3.committed_state.generalized_section.v3"
    old["state_version"] = 3
    old["nonlinear_state_layout_id"] = (
        "S3_TL_Q18_NODE_SO3_PL3_TRIADS4_BUBBLE2_STATION7_GENERALIZED_STATELESS_V3"
    )
    old = seal_committed_s3_state(old)
    with pytest.raises(S3CommittedStateError, match="incompatible state_schema"):
        element.validate_model_bound_nonlinear_state(
            model.mesh, model.get_material("carrier"), old, 3
        )

    offset_model, offset_element = _model(reference_surface_offset=0.004)
    offset_state = offset_element.init_model_bound_nonlinear_state(
        offset_model.mesh,
        offset_model.get_material("carrier"),
        3,
        initial_fields=fields,
        initial_field_provenance={"source": "V4-binding"},
    )
    assert state["initial_fields_fingerprint"] == (
        offset_state["initial_fields_fingerprint"]
    )
    assert state["element_configuration_fingerprint"] != (
        offset_state["element_configuration_fingerprint"]
    )
    assert state["generalized_formulation_fingerprint"] == (
        offset_state["generalized_formulation_fingerprint"]
    )

    changed_fields = copy.deepcopy(fields)
    changed_fields["initial_membrane_stress"][0, 0] += 1.0
    changed = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        3,
        initial_fields=changed_fields,
        initial_field_provenance={"source": "V4-binding"},
    )
    assert changed["initial_fields_fingerprint"] != state["initial_fields_fingerprint"]
    assert changed["element_configuration_fingerprint"] == (
        state["element_configuration_fingerprint"]
    )


def test_generalized_initial_resultants_drive_force_and_consistent_tangent() -> None:
    model, element = _model()
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        3,
        initial_fields=_seven_station_fields(),
        initial_field_provenance={"source": "force-tangent-work"},
    )
    rng = np.random.default_rng(20260825)
    centre = 8.0e-6 * rng.standard_normal(18)
    direction = rng.standard_normal(18)
    direction /= np.linalg.norm(direction)
    step = 5.0e-8
    force, tangent, trial = _response(model, element, centre, state)
    plus = _response(model, element, centre + step * direction, state)[0]
    minus = _response(model, element, centre - step * direction, state)[0]
    assert tangent is not None
    np.testing.assert_allclose(
        tangent @ direction,
        (plus - minus) / (2.0 * step),
        rtol=7.0e-6,
        atol=0.15,
    )
    constitutive, _membrane = element._constitutive(
        model.get_material("carrier"),
        element.compute_stiffness_components(
            model.mesh, model.get_material("carrier")
        )["frame"],
    )
    expected = (
        (
            np.asarray(trial["station_generalized_strain"])
            - np.asarray(trial["initial_generalized_prestrain"])
        )
        @ constitutive.T
        + np.asarray(trial["initial_generalized_resultant"])
    )
    np.testing.assert_array_equal(trial["station_generalized_resultant"], expected)
    virtual = rng.standard_normal(18)
    np.testing.assert_allclose(
        float(virtual @ tangent @ direction),
        float(direction @ tangent @ virtual),
        rtol=3.0e-13,
        atol=2.0e-6,
    )
    assert float(np.linalg.norm(force)) > 0.0


def test_generalized_initial_resultants_enter_geometric_tangent_and_virtual_work() -> None:
    from _e4_pl_s3_native_trial import native_trial_for_increment

    model, element = _model()
    frame = element.compute_stiffness_components(
        model.mesh, model.get_material("carrier")
    )["frame"]
    constitutive, _membrane = element._constitutive(
        model.get_material("carrier"), frame
    )
    triads = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 4, axis=0)
    increment = np.linspace(-8.0e-6, 9.0e-6, 20)
    view, exact, _store = native_trial_for_increment(REFERENCE, triads, increment)
    zero = np.zeros((7, 8), dtype=np.float64)
    fields = _seven_station_fields()
    prestrain, initial_resultant = _expected_offsets(fields)
    plain = s3._native_generalized_uncondensed_response_components(
        REFERENCE,
        triads,
        exact,
        REFERENCE,
        frame,
        constitutive,
        zero,
        0.0,
        zero,
        zero,
        native_rotation_trial=view,
    )
    with_initial = s3._native_generalized_uncondensed_response_components(
        REFERENCE,
        triads,
        exact,
        REFERENCE,
        frame,
        constitutive,
        zero,
        0.0,
        prestrain,
        initial_resultant,
        native_rotation_trial=view,
    )
    np.testing.assert_array_equal(
        plain[3]["material"], with_initial[3]["material"]
    )
    assert not np.array_equal(plain[3]["geometric"], with_initial[3]["geometric"])
    expected_resultant = (
        (np.asarray(with_initial[2]["station_generalized_strain"]) - prestrain)
        @ constitutive.T
        + initial_resultant
    )
    np.testing.assert_array_equal(
        with_initial[2]["station_generalized_resultant"], expected_resultant
    )

    _values, gradients, _hessians = s3._native_station_kinematics(
        REFERENCE,
        triads,
        exact,
        REFERENCE,
        frame,
        native_rotation_trial=view,
    )
    local = (REFERENCE - REFERENCE[0]) @ frame[:, :2]
    _jacobian, _inverse, determinant = s3._jacobian(local)
    virtual = np.linspace(0.9, -0.7, 20)
    integrated_work = 0.0
    for station, (_r, _s, weight) in enumerate(s3.TRIANGLE_QUADRATURE):
        integrated_work += (
            abs(determinant)
            * float(weight)
            * float((gradients[station] @ virtual) @ expected_resultant[station])
        )
    assert float(virtual @ with_initial[0]) == pytest.approx(
        integrated_work, rel=3.0e-13, abs=3.0e-9
    )


def _numbered_frame(nodes: np.ndarray) -> np.ndarray:
    first = np.asarray(nodes[1] - nodes[0], dtype=np.float64)
    first -= float(first @ OWNER) * OWNER
    first /= np.linalg.norm(first)
    second = np.cross(OWNER, first)
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, OWNER))


def _local_tensor_vector(
    tensor: np.ndarray, frame: np.ndarray, *, engineering: bool
) -> np.ndarray:
    local = frame[:, :2].T @ np.asarray(tensor, dtype=np.float64) @ frame[:, :2]
    shear = (2.0 if engineering else 1.0) * local[0, 1]
    return np.asarray((local[0, 0], local[1, 1], shear), dtype=np.float64)


def _permutation_matrix(order: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((18, 18), dtype=np.float64)
    for row, source in enumerate(order):
        result[6 * row : 6 * row + 6, 6 * source : 6 * source + 6] = np.eye(6)
    return result


def test_generalized_initial_fields_transport_over_all_six_D3_actions() -> None:
    section = _section()
    direction = MATERIAL_DIRECTION / np.linalg.norm(MATERIAL_DIRECTION)
    global_membrane_stress = np.asarray(
        ((2.0e6, 0.35e6, 0.0), (0.35e6, -0.8e6, 0.0), (0.0, 0.0, 0.0))
    )
    global_bending_stress = np.asarray(
        ((0.7e6, -0.2e6, 0.0), (-0.2e6, 0.4e6, 0.0), (0.0, 0.0, 0.0))
    )
    global_membrane_prestrain = np.asarray(
        ((1.2e-5, 0.3e-5, 0.0), (0.3e-5, -0.5e-5, 0.0), (0.0, 0.0, 0.0))
    )
    global_curvature_prestrain = np.asarray(
        ((1.7e-3, -0.4e-3, 0.0), (-0.4e-3, 0.8e-3, 0.0), (0.0, 0.0, 0.0))
    )
    rng = np.random.default_rng(7419)
    external = 9.0e-6 * rng.standard_normal(18)
    virtual = rng.standard_normal(18)
    baseline: tuple[np.ndarray, np.ndarray, float] | None = None

    for order in itertools.permutations(range(3)):
        operator = _permutation_matrix(order)
        nodes = REFERENCE[np.asarray(order)]
        frame = _numbered_frame(nodes)
        angle = float(
            np.arctan2(frame[:, 1] @ direction, frame[:, 0] @ direction)
        )
        rotated = section.rotated(angle)
        constitutive = np.zeros((8, 8), dtype=np.float64)
        constitutive[:3, :3] = rotated.A
        constitutive[:3, 3:6] = rotated.B
        constitutive[3:6, :3] = rotated.B.T
        constitutive[3:6, 3:6] = rotated.D
        constitutive[6:, 6:] = rotated.As
        prestrain = np.zeros((7, 8), dtype=np.float64)
        resultant = np.zeros((7, 8), dtype=np.float64)
        prestrain[:, :3] = _local_tensor_vector(
            global_membrane_prestrain, frame, engineering=True
        )
        prestrain[:, 3:6] = _local_tensor_vector(
            global_curvature_prestrain, frame, engineering=True
        )
        resultant[:, :3] = THICKNESS * _local_tensor_vector(
            global_membrane_stress, frame, engineering=False
        )
        resultant[:, 3:6] = (THICKNESS**2 / 6.0) * _local_tensor_vector(
            global_bending_stress, frame, engineering=False
        )
        triads = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 4, axis=0)

        def builder(increment: np.ndarray):
            from _e4_pl_s3_native_trial import native_trial_for_increment

            view, exact, _store = native_trial_for_increment(nodes, triads, increment)
            return s3._native_generalized_uncondensed_response_components(
                nodes,
                triads,
                exact,
                nodes,
                frame,
                constitutive,
                np.zeros((7, 8)),
                0.0,
                prestrain,
                resultant,
                native_rotation_trial=view,
            )

        force, tangent, _trial, metadata = s3._solve_native_bubble_equilibrium(
            operator @ external, np.zeros(2), builder
        )
        work = float((operator @ virtual) @ force)
        if baseline is None:
            baseline = (force, tangent, work)
        else:
            np.testing.assert_allclose(
                force, operator @ baseline[0], rtol=4.0e-10, atol=3.0e-5
            )
            np.testing.assert_allclose(
                tangent,
                operator @ baseline[1] @ operator.T,
                rtol=4.0e-10,
                atol=3.0e-2,
            )
            assert work == pytest.approx(baseline[2], rel=4.0e-10, abs=3.0e-7)
        assert np.linalg.norm(np.asarray(metadata["bubble_increment"])) > 0.0


def test_generalized_initial_fields_preserve_physical_director_reversal() -> None:
    fields = _seven_station_fields()
    responses: dict[int, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    total = np.linspace(-7.0e-6, 8.0e-6, 18)
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity)
        state = element.init_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            3,
            initial_fields=fields,
            initial_field_provenance={"source": "director-reversal-field"},
        )
        responses[polarity] = _response(model, element, total, state)
    force_plus, tangent_plus, state_plus = responses[1]
    force_minus, tangent_minus, state_minus = responses[-1]
    np.testing.assert_allclose(force_minus, force_plus, rtol=4.0e-10, atol=3.0e-5)
    np.testing.assert_allclose(
        tangent_minus, tangent_plus, rtol=4.0e-10, atol=3.0e-2
    )
    np.testing.assert_array_equal(
        state_minus["initial_membrane_stress"],
        state_plus["initial_membrane_stress"],
    )
    np.testing.assert_array_equal(
        state_minus["initial_membrane_prestrain"],
        state_plus["initial_membrane_prestrain"],
    )
    for name in (
        "initial_bending_stress",
        "initial_curvature_prestrain",
    ):
        np.testing.assert_array_equal(state_minus[name], -np.asarray(state_plus[name]))
    np.testing.assert_allclose(
        state_minus["station_generalized_resultant"][:, :3],
        state_plus["station_generalized_resultant"][:, :3],
        rtol=4.0e-10,
        atol=3.0e-5,
    )
    np.testing.assert_allclose(
        state_minus["station_generalized_resultant"][:, 3:],
        -np.asarray(state_plus["station_generalized_resultant"][:, 3:]),
        rtol=4.0e-10,
        atol=3.0e-5,
    )


def _self_equilibrated_field(
    model: FEModel, element: QualifiedE4PLS3ShellElement
) -> ShellInitialField:
    frame = element.compute_stiffness_components(
        model.mesh, model.get_material("carrier")
    )["frame"]
    constitutive, _membrane = element._constitutive(
        model.get_material("carrier"), frame
    )
    eigenstrain = np.asarray(
        (1.1e-5, -0.6e-5, 0.4e-5, 1.3e-3, -0.8e-3, 0.5e-3)
    )
    initial = constitutive[:6, :6] @ eigenstrain
    return ShellInitialField(
        membrane_stress=initial[:3] / THICKNESS,
        bending_stress=6.0 * initial[3:6] / (THICKNESS**2),
        membrane_prestrain=eigenstrain[:3],
        curvature_prestrain=eigenstrain[3:6],
        source="self-equilibrated-B-coupled-field",
    )


def test_generalized_initial_fields_run_static_restart_arc_and_recovery() -> None:
    full_model, full_element = _model(constrained=True)
    initial = _self_equilibrated_field(full_model, full_element)
    full = solve_static_nonlinear(
        full_model,
        _load(),
        num_steps=2,
        num_layers=3,
        initial_fields={1: initial},
        emit_restart_checkpoint=True,
        convergence_settings="legacy",
    )
    first_model, first_element = _model(constrained=True)
    first_initial = _self_equilibrated_field(first_model, first_element)
    first = solve_static_nonlinear(
        first_model,
        _load(),
        max_load_factor=0.5,
        num_steps=1,
        num_layers=3,
        initial_fields={1: first_initial},
        emit_restart_checkpoint=True,
        convergence_settings="legacy",
    )
    resumed_model, _resumed_element = _model(constrained=True)
    resumed = solve_static_nonlinear(
        resumed_model,
        _load(),
        max_load_factor=1.0,
        num_steps=1,
        num_layers=3,
        restart_checkpoint=first.restart_checkpoint_bytes(),
        convergence_settings="legacy",
    )

    assert full.status == first.status == resumed.status == "completed"
    np.testing.assert_array_equal(full.displacements, resumed.displacements)
    assert canonical_json_bytes(full.element_states[1]) == canonical_json_bytes(
        resumed.element_states[1]
    )
    assert full.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()
    state = full.element_states[1]
    assert state["initial_field_provenance"]["source"] == initial.source
    assert full.info["initial_state_equilibration"]["converged"] is True
    assert full.restart_checkpoint is not None
    assert resumed.element_states[1]["initial_fields_fingerprint"] == (
        state["initial_fields_fingerprint"]
    )

    arc_model, arc_element = _model(constrained=True)
    virgin = arc_element.init_model_bound_nonlinear_state(
        arc_model.mesh,
        arc_model.get_material("carrier"),
        3,
        initial_fields=initial.state_values(),
        initial_field_provenance={
            "kind": "shell",
            "source": initial.source,
            "components": sorted(initial.state_values()),
        },
    )
    np.testing.assert_allclose(
        virgin["station_generalized_resultant"],
        np.zeros((7, 8)),
        rtol=0.0,
        atol=1.0e-10,
    )
    arc = solve_static_arc_length(
        arc_model,
        _load(),
        control=ArcLengthControl(
            initial_load_increment=0.05,
            minimum_load_increment=0.01,
            maximum_load_increment=0.05,
            maximum_absolute_load_factor=0.10,
            max_steps=2,
        ),
        num_layers=3,
        initial_element_states={1: virgin},
        emit_restart_checkpoint=True,
    )
    assert arc.status == "load_factor_limit_reached"
    assert arc.element_states[1]["initial_fields_fingerprint"] == (
        virgin["initial_fields_fingerprint"]
    )

    recovered, reason, sources = _recover_qualified_s3_committed_state(
        full_model,
        1,
        full_element,
        state,
        displacements=np.asarray(full.displacements),
        return_global=True,
    )
    assert reason == ""
    assert recovered is not None
    assert recovered["physical_stress_available"] is False
    assert recovered["physical_layer_recovery_available"] is False
    assert recovered["initial_field_provenance"] == state["initial_field_provenance"]
    np.testing.assert_array_equal(
        recovered["initial_generalized_prestrain"],
        state["initial_generalized_prestrain"],
    )
    np.testing.assert_array_equal(
        recovered["initial_generalized_resultant"],
        state["initial_generalized_resultant"],
    )
    assert sources["initial_fields"] == "committed_s3_generalized_initial_fields"
