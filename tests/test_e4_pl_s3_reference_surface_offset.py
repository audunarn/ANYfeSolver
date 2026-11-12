from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    FEModel,
    FixedSupport,
    LoadCase,
    QualifiedE4PLS3ShellElement,
    assemble_external_load_tangent,
    assemble_load_vector,
    shell_element_from_dict,
    solve_eigenvalue_buckling,
    solve_free_vibration,
)
from anyfesolver._native_rotation_state import create_native_rotation_state_store
from anyfesolver.contact import SphereContactConfig, _surface_offset
from anyfesolver.e4_pl_s3_element import (
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
)
from anyfesolver.e4_pl_s3_state import (
    REFERENCE_SURFACE_MASS_SHIFT_ID,
    REFERENCE_SURFACE_OFFSET_POLICY_ID,
    REFERENCE_SURFACE_STRAIN_TRANSFORM_ID,
    S3CommittedStateError,
)
from anyfesolver.shell_sections import GeneralizedShellSection


REFERENCE = np.asarray(
    ((0.0, 0.0, 0.0), (1.10, 0.0, 0.0), (0.25, 0.90, 0.0)),
    dtype=np.float64,
)
OWNER = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
THICKNESS = 0.027
OFFSET = 0.006


def _section(
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    D: np.ndarray | None = None,
    mass_per_area: float = 41.0,
    rotary_inertia_per_area: float = 0.014,
) -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="offset-B-coupled",
        A=(
            np.asarray(
                (
                    (2.4e8, 0.42e8, 0.13e8),
                    (0.42e8, 1.35e8, -0.08e8),
                    (0.13e8, -0.08e8, 0.71e8),
                )
            )
            if A is None
            else A
        ),
        B=(
            np.asarray(
                (
                    (2.1e4, -0.7e4, 0.3e4),
                    (0.4e4, 1.6e4, -0.2e4),
                    (-0.5e4, 0.1e4, 0.8e4),
                )
            )
            if B is None
            else B
        ),
        D=(
            np.asarray(
                (
                    (3.2e4, 0.55e4, 0.12e4),
                    (0.55e4, 2.1e4, -0.09e4),
                    (0.12e4, -0.09e4, 1.15e4),
                )
            )
            if D is None
            else D
        ),
        As=np.asarray(((0.82e8, 0.11e8), (0.11e8, 0.59e8))),
        mass_per_area=mass_per_area,
        rotary_inertia_per_area=rotary_inertia_per_area,
    )


def _shifted_section(section: GeneralizedShellSection, offset: float) -> GeneralizedShellSection:
    e = float(offset)
    return _section(
        A=np.asarray(section.A),
        B=np.asarray(section.B) - e * np.asarray(section.A),
        D=(
            np.asarray(section.D)
            - e * (np.asarray(section.B) + np.asarray(section.B).T)
            + e * e * np.asarray(section.A)
        ),
        mass_per_area=float(section.mass_per_area),
        rotary_inertia_per_area=float(section.rotary_inertia_per_area),
    )


def _model(
    *,
    offset: float = OFFSET,
    polarity: int = 1,
    node_order: tuple[int, int, int] = (1, 2, 3),
    generalized: bool = True,
    section: GeneralizedShellSection | None = None,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified-s3-reference-surface-offset")
    model.add_material("carrier", 70.0e9, 0.31, density=2700.0)
    for node_id, point in enumerate(REFERENCE, start=1):
        model.add_node(node_id, *point)
    element = QualifiedE4PLS3ShellElement(
        1,
        list(node_order),
        "carrier",
        thickness=THICKNESS,
        material_direction=(0.91, 0.37, 0.0),
        material_angle_deg=13.0,
        shell_section=(section or _section()) if generalized else None,
        reference_normal=OWNER,
        director_polarity=polarity,
        reference_surface_offset=offset,
    )
    model.add_element(1, element)
    return model, element


def _native_response(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object] | None = None,
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
    view = store.element_view(
        element.element_id,
        node_ids,
        element.native_reference_directors(model.mesh),
        trial_token=token,
    )
    return element.compute_nonlinear_response(
        model.mesh,
        model.get_material("carrier"),
        total,
        state,
        3,
        True,
        native_rotation_trial=view,
    )


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected, ord=np.inf)) / max(
        float(np.linalg.norm(actual, ord=np.inf)),
        float(np.linalg.norm(expected, ord=np.inf)),
        1.0,
    )


def test_offset_is_strict_additive_serialized_and_bound_to_state_identity() -> None:
    for invalid in (True, False, "0", None, np.nan, np.inf, -np.inf):
        with pytest.raises((TypeError, ValueError), match="reference_surface_offset"):
            _model(offset=invalid)  # type: ignore[arg-type]

    for generalized in (False, True):
        model, element = _model(generalized=generalized)
        payload = json.loads(json.dumps(element.to_dict()))
        assert payload["reference_surface_offset"] == OFFSET
        assert payload["reference_surface_offset_policy_id"] == (
            REFERENCE_SURFACE_OFFSET_POLICY_ID
        )
        assert payload["reference_surface_strain_transform_id"] == (
            REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
        )
        assert payload["reference_surface_mass_shift_id"] == (
            REFERENCE_SURFACE_MASS_SHIFT_ID
        )
        rebuilt = shell_element_from_dict(payload)
        assert rebuilt.to_dict() == payload
        state = element.init_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            3,
        )
        zero_model, zero_element = _model(offset=0.0, generalized=generalized)
        with pytest.raises(S3CommittedStateError, match="identity does not match"):
            zero_element.validate_model_bound_nonlinear_state(
                zero_model.mesh,
                zero_model.get_material("carrier"),
                state,
                3,
            )
        for key in (
            "reference_surface_offset",
            "reference_surface_offset_policy_id",
            "reference_surface_strain_transform_id",
            "reference_surface_mass_shift_id",
        ):
            malformed = dict(payload)
            malformed.pop(key)
            with pytest.raises(ValueError, match="reference-surface|reference_surface"):
                shell_element_from_dict(malformed)


def test_zero_offset_preserves_linear_and_mass_arrays_exactly() -> None:
    model_default, default = _model(offset=0.0)
    model_explicit, explicit = _model(offset=-0.0)
    material_default = model_default.get_material("carrier")
    material_explicit = model_explicit.get_material("carrier")
    for key in ("physical", "pl", "total", "uncondensed_physical", "bubble_map"):
        np.testing.assert_array_equal(
            default.compute_stiffness_components(model_default.mesh, material_default)[key],
            explicit.compute_stiffness_components(model_explicit.mesh, material_explicit)[key],
        )
    for key in ("full_local", "condensed_local", "global", "guyan"):
        np.testing.assert_array_equal(
            default.compute_mass_components(model_default.mesh, material_default)[key],
            explicit.compute_mass_components(model_explicit.mesh, material_explicit)[key],
        )


def test_linear_offset_is_exact_section_congruence_with_B_coupling_and_virtual_work() -> None:
    source = _section()
    shifted = _shifted_section(source, OFFSET)
    offset_model, offset_element = _model(section=source)
    shifted_model, shifted_element = _model(offset=0.0, section=shifted)
    offset_components = offset_element.compute_stiffness_components(
        offset_model.mesh, offset_model.get_material("carrier")
    )
    shifted_components = shifted_element.compute_stiffness_components(
        shifted_model.mesh, shifted_model.get_material("carrier")
    )
    for key in (
        "uncondensed_physical",
        "bubble_block",
        "bubble_map",
        "condensed_physical_15",
        "physical",
        "pl",
        "total",
    ):
        np.testing.assert_allclose(
            offset_components[key], shifted_components[key], rtol=2.0e-13, atol=2.0e-7
        )

    rng = np.random.default_rng(2026082501)
    displacement = 2.0e-5 * rng.standard_normal(18)
    virtual = rng.standard_normal(18)
    force = offset_element.compute_internal_forces(
        offset_model.mesh,
        displacement,
        offset_model.get_material("carrier"),
    )
    stiffness = np.asarray(offset_components["total"])
    np.testing.assert_allclose(force, stiffness @ displacement, rtol=0.0, atol=0.0)
    assert float(virtual @ force) == pytest.approx(
        float(displacement @ stiffness @ virtual), rel=2.0e-13, abs=1.0e-10
    )


def test_offset_native_generalized_tangent_matches_directional_finite_difference() -> None:
    model, element = _model()
    rng = np.random.default_rng(2026082502)
    total = 8.0e-6 * rng.standard_normal(18)
    direction = rng.standard_normal(18)
    direction /= np.linalg.norm(direction)
    force, tangent, state = _native_response(model, element, total)
    assert tangent is not None
    step = 2.0e-7
    plus_force, _plus_tangent, _plus_state = _native_response(
        model, element, total + step * direction
    )
    minus_force, _minus_tangent, _minus_state = _native_response(
        model, element, total - step * direction
    )
    numerical = (plus_force - minus_force) / (2.0 * step)
    analytical = np.asarray(tangent) @ direction
    assert _relative(numerical, analytical) <= 8.0e-7
    assert float(direction @ force) == pytest.approx(
        float(force @ direction), rel=0.0, abs=0.0
    )
    assert state["station_generalized_strain"].shape == (7, 8)


def test_all_six_D3_actions_and_signed_polarity_reversal_preserve_offset_operators() -> None:
    for polarity, offset in ((1, OFFSET), (-1, -OFFSET)):
        base_model, base_element = _model(polarity=polarity, offset=offset)
        base_material = base_model.get_material("carrier")
        base_stiffness = np.asarray(
            base_element._compute_stiffness_components(
                base_model.mesh, base_material, enforce_positive_winding=False
            )["total"]
        )
        for ordering in itertools.permutations((1, 2, 3)):
            model, element = _model(
                polarity=polarity,
                offset=offset,
                node_order=ordering,
            )
            actual = np.asarray(
                element._compute_stiffness_components(
                    model.mesh,
                    model.get_material("carrier"),
                    enforce_positive_winding=False,
                )["total"]
            )
            dofs = [
                6 * (node_id - 1) + component
                for node_id in ordering
                for component in range(6)
            ]
            assert _relative(actual, base_stiffness[np.ix_(dofs, dofs)]) <= 4.0e-13

    plus_model, plus = _model(polarity=1, offset=OFFSET)
    minus_model, minus = _model(polarity=-1, offset=-OFFSET)
    plus_material = plus_model.get_material("carrier")
    minus_material = minus_model.get_material("carrier")
    np.testing.assert_allclose(
        minus.compute_stiffness_matrix(minus_model.mesh, minus_material),
        plus.compute_stiffness_matrix(plus_model.mesh, plus_material),
        rtol=3.0e-13,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        minus.compute_mass_matrix(minus_model.mesh, minus_material),
        plus.compute_mass_matrix(plus_model.mesh, plus_material),
        rtol=3.0e-13,
        atol=2.0e-12,
    )


def test_offset_mass_contains_first_moment_coupling_is_symmetric_psd_and_guyan_consistent() -> None:
    model, element = _model()
    components = element.compute_mass_components(
        model.mesh, model.get_material("carrier")
    )
    m0 = float(components["mass_per_area"])
    expected_m1_physical = -OFFSET * m0
    expected_m1_owner = expected_m1_physical
    expected_m2 = float(components["section_rotary_inertia_per_area"]) + OFFSET**2 * m0
    assert components["physical_first_mass_moment_per_area"] == pytest.approx(
        expected_m1_physical, rel=0.0, abs=0.0
    )
    assert components["owner_first_mass_moment_per_area"] == pytest.approx(
        expected_m1_owner, rel=0.0, abs=0.0
    )
    assert components["rotary_inertia_per_area"] == pytest.approx(
        expected_m2, rel=0.0, abs=0.0
    )
    full = np.asarray(components["full_local"])
    np.testing.assert_array_equal(full, full.T)
    scale = max(float(np.linalg.norm(full, ord=2)), 1.0)
    assert float(np.linalg.eigvalsh(full)[0]) >= -2.0e-13 * scale
    corner = np.asarray(components["corner_moment"])
    corner_bubble = np.asarray(components["corner_bubble_moment"])
    expected = np.column_stack((corner, corner_bubble)) * expected_m1_owner
    np.testing.assert_allclose(full[np.ix_((0, 6, 12), (4, 10, 16, 19))], expected)
    np.testing.assert_allclose(full[np.ix_((1, 7, 13), (3, 9, 15, 18))], -expected)
    guyan = np.asarray(components["guyan"])
    np.testing.assert_allclose(
        components["condensed_local"],
        guyan.T @ full @ guyan,
        rtol=2.0e-14,
        atol=2.0e-12,
    )


def test_offset_geometric_operator_matches_N_M_H_translation_for_modal_and_buckling() -> None:
    source = _section()
    shifted = _shifted_section(source, OFFSET)
    offset_model, offset_element = _model(section=source)
    shifted_model, shifted_element = _model(offset=0.0, section=shifted)
    rng = np.random.default_rng(2026082503)
    membrane = 2.0e5 + 8.0e4 * rng.random((7, 3))
    bending = 2.0e3 * rng.standard_normal((7, 3))
    second = 120.0 + 40.0 * rng.random((7, 3))
    common = {
        "membrane_compression_at_gauss": membrane,
        "bending_compression_at_gauss": bending,
        "stress_second_moment_at_gauss": second,
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    }
    shifted_state = {
        "membrane_compression_at_gauss": membrane,
        "bending_compression_at_gauss": bending - OFFSET * membrane,
        "stress_second_moment_at_gauss": (
            second - 2.0 * OFFSET * bending + OFFSET**2 * membrane
        ),
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    }
    offset_geometric = offset_element.compute_geometric_stiffness_components(
        offset_model.mesh,
        offset_model.get_material("carrier"),
        common,
    )
    shifted_geometric = shifted_element.compute_geometric_stiffness_components(
        shifted_model.mesh,
        shifted_model.get_material("carrier"),
        shifted_state,
    )
    for key in ("full_local", "condensed_local", "global"):
        np.testing.assert_allclose(
            offset_geometric[key], shifted_geometric[key], rtol=3.0e-13, atol=2.0e-7
        )


def test_signed_offset_mass_and_geometric_operators_reach_modal_and_buckling_workflows() -> None:
    results = {}
    for polarity, offset in ((1, OFFSET), (-1, -OFFSET)):
        model, _element = _model(
            polarity=polarity,
            offset=offset,
            generalized=False,
        )
        model.add_boundary_condition(FixedSupport("fixed-edge", [1, 2]))
        model.add_boundary_condition(
            BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
        )
        compression = 1.8e5
        prestress = {
            "bubble_linearization_policy": (
                REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
            ),
            "membrane_compression": [compression, compression, 0.0],
            "bending_compression": [0.0, 0.0, 0.0],
            "stress_second_moment": [
                compression * THICKNESS**2 / 12.0,
                compression * THICKNESS**2 / 12.0,
                0.0,
            ],
        }
        modal = solve_free_vibration(
            model,
            num_modes=3,
            prestress_states={1: prestress},
        )
        buckling = solve_eigenvalue_buckling(
            model,
            {1: prestress},
            num_modes=3,
            dense_size_limit=1000,
            allow_free_mechanisms=True,
            reference_elastic_only=True,
        )
        assert modal.solver_status == "ok", modal.diagnostics
        assert buckling.solver_status == "ok", buckling.diagnostics
        results[polarity] = (
            np.asarray(modal.frequencies_hz, dtype=np.float64),
            np.asarray(
                [mode.load_factor for mode in buckling.modes], dtype=np.float64
            ),
        )
    np.testing.assert_allclose(
        results[-1][0], results[1][0], rtol=4.0e-12, atol=2.0e-9
    )
    np.testing.assert_allclose(
        results[-1][1], results[1][1], rtol=4.0e-12, atol=2.0e-9
    )


def test_layer_recovery_and_contact_surfaces_follow_physical_offset_convention() -> None:
    rng = np.random.default_rng(2026082504)
    displacement = 4.0e-6 * rng.standard_normal(18)
    recoveries: dict[int, dict[str, object]] = {}
    for polarity, offset in ((1, OFFSET), (-1, -OFFSET)):
        model, element = _model(
            generalized=False,
            polarity=polarity,
            offset=offset,
        )
        recoveries[polarity] = element.compute_stresses(
            model.mesh,
            displacement,
            model.get_material("carrier"),
            return_global=True,
        )
    plus = recoveries[1]
    minus = recoveries[-1]
    for component in ("xx", "yy", "xy"):
        np.testing.assert_allclose(
            minus[f"local_{component}_top"],
            plus[f"local_{component}_bot"],
            rtol=3.0e-12,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            minus[f"local_{component}_bot"],
            plus[f"local_{component}_top"],
            rtol=3.0e-12,
            atol=2.0e-7,
        )

    model, element = _model()
    assert _surface_offset(
        element, SphereContactConfig(contact_surface="midsurface")
    ) == pytest.approx(-OFFSET)
    assert _surface_offset(
        element, SphereContactConfig(contact_surface="top")
    ) == pytest.approx(0.5 * THICKNESS - OFFSET)
    assert _surface_offset(
        element, SphereContactConfig(contact_surface="bottom")
    ) == pytest.approx(-0.5 * THICKNESS - OFFSET)
    assert _surface_offset(
        element,
        SphereContactConfig(contact_surface="signed", signed_surface_offset=0.4),
    ) == pytest.approx(0.2 * THICKNESS - OFFSET)


def test_pressure_follower_tangent_and_distributed_couples_use_the_nodal_reference_surface() -> None:
    rng = np.random.default_rng(2026082505)
    total = 2.0e-3 * rng.standard_normal(18)
    direction = rng.standard_normal(18)
    direction /= np.linalg.norm(direction)
    virtual = rng.standard_normal(18)
    pressure = 731.0
    applied_couples = np.zeros(18, dtype=np.float64)
    applied_couples.reshape(3, 6)[:, 3:] = np.asarray(
        ((2.0, -1.0, 0.7), (-3.0, 5.0, -2.0), (4.0, 0.5, 1.0)),
        dtype=np.float64,
    )
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for polarity, offset in ((1, OFFSET), (-1, -OFFSET)):
        model, _element = _model(polarity=polarity, offset=offset)
        load = LoadCase("qualified-s3-offset-follower")
        load.add_pressure_load(1, pressure)
        load.follower_pressure = True
        force, load_info = assemble_load_vector(model, load, total)
        tangent, tangent_info = assemble_external_load_tangent(model, load, total)
        tangent_array = tangent.toarray()
        record = {
            "element_id": 1,
            "pressure_surface_id": "ELEMENT_NODAL_REFERENCE_SURFACE_V1",
            "reference_surface_offset": offset,
            "resultant_and_reaction_reference": (
                "GLOBAL_NODAL_REFERENCE_COORDINATES"
            ),
            "section_origin_offset_from_reference": -offset,
            "virtual_work": "TRANSLATIONAL_NODAL_REFERENCE_SURFACE_ONLY",
        }
        assert load_info["qualified_s3_pressure_surfaces"] == [record]
        assert tangent_info["qualified_s3_pressure_surfaces"] == [record]
        np.testing.assert_array_equal(force.reshape(3, 6)[:, 3:], 0.0)

        step = 2.0e-7
        plus, _ = assemble_load_vector(model, load, total + step * direction)
        minus, _ = assemble_load_vector(model, load, total - step * direction)
        numerical = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(
            tangent_array @ direction,
            numerical,
            rtol=2.0e-9,
            atol=2.0e-7,
        )
        assert float(virtual @ force) == pytest.approx(
            float(force @ virtual), rel=0.0, abs=0.0
        )

        couple_load = LoadCase("qualified-s3-offset-distributed-couples")
        couple_load.element_loads[1] = applied_couples.copy()
        couples, couple_info = assemble_load_vector(model, couple_load)
        assert "qualified_s3_pressure_surfaces" not in couple_info
        np.testing.assert_array_equal(couples, applied_couples)
        assert float(virtual @ couples) == pytest.approx(
            float(couples @ virtual), rel=0.0, abs=0.0
        )
        if baseline is None:
            baseline = (force, tangent_array, couples)
        else:
            np.testing.assert_array_equal(force, baseline[0])
            np.testing.assert_array_equal(tangent_array, baseline[1])
            np.testing.assert_array_equal(couples, baseline[2])


def test_signed_polarity_pair_preserves_native_layered_work_and_swaps_layer_history() -> None:
    rng = np.random.default_rng(2026082506)
    total = 5.0e-6 * rng.standard_normal(18)
    responses = {}
    for polarity, offset in ((1, OFFSET), (-1, -OFFSET)):
        model, element = _model(
            generalized=False,
            polarity=polarity,
            offset=offset,
        )
        responses[polarity] = _native_response(model, element, total)
    force_plus, tangent_plus, state_plus = responses[1]
    force_minus, tangent_minus, state_minus = responses[-1]
    np.testing.assert_allclose(force_minus, force_plus, rtol=5.0e-10, atol=3.0e-4)
    np.testing.assert_allclose(tangent_minus, tangent_plus, rtol=5.0e-10, atol=0.3)
    plus_layers = np.asarray(state_plus["layer_stress"]).reshape(7, 3, 3)
    minus_layers = np.asarray(state_minus["layer_stress"]).reshape(7, 3, 3)
    np.testing.assert_allclose(
        minus_layers,
        plus_layers[:, ::-1, :],
        rtol=5.0e-10,
        atol=3.0e-4,
    )
