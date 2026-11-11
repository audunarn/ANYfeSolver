from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import (
    AnyStructureFEMConfig,
    FEModel,
    QualifiedE4PLS3ShellElement,
    build_fe_model_from_generated_geometry,
    build_symmetric_load_case,
    idealize_generated_geometry_members,
    load_case_resultant,
    recover_prestress_from_static_result,
    run_anystructure_fem_mode,
)
from anyfesolver.assembly import solve_linear
from anyfesolver.e4_pl_s3_element import (
    HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID,
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    RESULTANT_SUMMARY_POLICY_ID,
)
from anyfesolver.elements import BeamElement, ShellElement
from anyfesolver.matrix_assembly import assemble_geometric_stiffness_matrix


class _Part:
    span = 1.0
    spacing = 1.0
    t = 0.01
    sigma_x1 = 5.0
    sigma_x2 = 5.0
    sigma_y1 = 0.0
    sigma_y2 = 0.0
    tau_xy = 0.0
    pressure = 0.0


class _Calc:
    Plate = _Part()
    Stiffener = None


def _square_shell_geometry():
    return {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [2.0, 0.0, 0.0]},
            {"id": 3, "coords": [2.0, 1.0, 0.0]},
            {"id": 4, "coords": [0.0, 1.0, 0.0]},
        ],
        "shells": [{"id": 1, "node_ids": [1, 2, 3, 4], "thickness": 0.02}],
    }


def _qualified_s3_model() -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified-s3-prestress-consumer")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, np.sqrt(3.0) / 2.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=0.1,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    return model, element


def _cylinder_geometry(num_circ=8):
    radius = 1.0
    height = 1.0
    nodes = []
    for iz, z in enumerate([0.0, height]):
        for itheta in range(num_circ):
            theta = 2.0 * np.pi * itheta / num_circ
            nodes.append({"id": iz * num_circ + itheta + 1, "coords": [radius * np.cos(theta), radius * np.sin(theta), z]})
    shells = []
    for itheta in range(num_circ):
        next_theta = (itheta + 1) % num_circ
        shells.append(
            {
                "id": itheta + 1,
                "node_ids": [itheta + 1, next_theta + 1, num_circ + next_theta + 1, num_circ + itheta + 1],
                "thickness": 0.02,
            }
        )
    return {
        "nodes": nodes,
        "shells": shells,
        "supports": [
            {
                "name": "bottom_fixed",
                "node_ids": list(range(1, num_circ + 1)),
                "dof_constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            }
        ],
    }


def test_generated_shell_geometry_converts_to_fe_model():
    model = build_fe_model_from_generated_geometry(_square_shell_geometry())

    assert model.mesh.num_nodes == 4
    assert model.mesh.num_elements == 1
    assert isinstance(model.mesh.get_element(1), ShellElement)


def test_generated_beam_shell_coupling_metadata_is_preserved():
    geometry = _square_shell_geometry()
    geometry["nodes"].extend(
        [
            {"id": 100, "coords": [0.5, 0.5, 0.2]},
            {"id": 101, "coords": [1.5, 0.5, 0.2]},
        ]
    )
    geometry["beams"] = [{"id": 20, "node_ids": [100, 101], "cross_section": {"area": 0.01, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}}]
    geometry["couplings"] = [
        {"id": 30, "beam_node_id": 100, "shell_node_ids": [1, 2, 3, 4], "shape_weights": [0.25, 0.25, 0.25, 0.25], "eccentricity": [0.0, 0.0, 0.2]}
    ]

    model = build_fe_model_from_generated_geometry(geometry)
    coupling = model.mesh.get_element(30)

    assert model.mesh.num_elements == 3
    assert coupling is not None
    assert len(coupling.get_mpc_constraints(model.mesh)) == 6


def test_generated_stiffeners_and_girders_are_created_as_beams():
    geometry = _square_shell_geometry()
    geometry["nodes"].extend(
        [
            {"id": 100, "coords": [0.0, 0.5, 0.2]},
            {"id": 101, "coords": [2.0, 0.5, 0.2]},
            {"id": 200, "coords": [1.0, 0.0, 0.3]},
            {"id": 201, "coords": [1.0, 1.0, 0.3]},
        ]
    )
    section = {"area": 0.01, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}
    geometry["stiffeners"] = [{"id": 20, "node_ids": [100, 101], "cross_section": section}]
    geometry["girders"] = [{"id": 21, "node_ids": [200, 201], "cross_section": section}]

    model = build_fe_model_from_generated_geometry(geometry)
    beams = [element for element in model.mesh.elements.values() if isinstance(element, BeamElement)]

    assert len(beams) == 2
    assert {getattr(beam, "structural_role", None) for beam in beams} == {"stiffener", "girder"}


def test_generated_beam_collections_are_additive_not_first_alias_only():
    geometry = _square_shell_geometry()
    geometry["nodes"].extend(
        [
            {"id": 100, "coords": [0.0, 0.25, 0.2]},
            {"id": 101, "coords": [2.0, 0.25, 0.2]},
            {"id": 110, "coords": [0.0, 0.75, 0.2]},
            {"id": 111, "coords": [2.0, 0.75, 0.2]},
        ]
    )
    section = {"area": 0.01, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}
    geometry["beams"] = [{"id": 20, "node_ids": [100, 101], "cross_section": section}]
    geometry["stiffeners"] = [{"id": 21, "node_ids": [110, 111], "cross_section": section}]

    model = build_fe_model_from_generated_geometry(geometry)
    beam_ids = {element.element_id for element in model.mesh.elements.values() if isinstance(element, BeamElement)}

    assert beam_ids == {20, 21}


def _square_shell_with_tagged_member_plate(role="stiffener_web"):
    geometry = _square_shell_geometry()
    geometry["nodes"].extend(
        [
            {"id": 100, "coords": [0.0, 0.5, 0.0]},
            {"id": 101, "coords": [2.0, 0.5, 0.0]},
            {"id": 102, "coords": [2.0, 0.5, 0.2]},
            {"id": 103, "coords": [0.0, 0.5, 0.2]},
        ]
    )
    geometry["plates"] = [{"id": 2, "node_ids": [100, 101, 102, 103], "thickness": 0.01, "role": role, "member_id": "S1"}]
    return geometry


@pytest.mark.parametrize(("tag", "expected_role"), [("stiffener_web", "stiffener"), ("girder_web", "girder")])
def test_tagged_member_plates_are_auto_idealized_as_beams_by_default(tag, expected_role):
    idealized = idealize_generated_geometry_members(_square_shell_with_tagged_member_plate(tag))

    assert len(idealized["shells"]) == 1
    assert len(idealized["beams"]) == 1
    assert idealized["beams"][0]["role"] == expected_role
    np.testing.assert_allclose(idealized["beams"][0]["cross_section"]["area"], 0.002)

    model = build_fe_model_from_generated_geometry(_square_shell_with_tagged_member_plate(tag))
    beams = [element for element in model.mesh.elements.values() if isinstance(element, BeamElement)]

    assert model.mesh.num_elements == 2
    assert len(beams) == 1
    assert getattr(beams[0], "structural_role", None) == expected_role


def test_tagged_stiffener_or_girder_plates_fail_closed_when_auto_idealization_is_disabled():
    config = AnyStructureFEMConfig(auto_idealize_member_plates_as_beams=False)

    with pytest.raises(ValueError, match="idealized-member-plates-require-beam-members-stiffener"):
        build_fe_model_from_generated_geometry(_square_shell_with_tagged_member_plate(), config)


def test_symmetric_pressure_load_resultant_matches_generated_area():
    model = build_fe_model_from_generated_geometry(_square_shell_geometry())
    load_case = build_symmetric_load_case(_Calc(), model, AnyStructureFEMConfig(pressure_pa=5.0, add_inplane_edge_loads=False))

    resultant = load_case_resultant(model, load_case)

    np.testing.assert_allclose(resultant.force, [0.0, 0.0, 10.0], atol=1.0e-12)


def test_symmetric_pressure_load_sign_controls_normal_direction():
    model = build_fe_model_from_generated_geometry(_square_shell_geometry())
    positive = build_symmetric_load_case(
        _Calc(),
        model,
        AnyStructureFEMConfig(pressure_pa=5.0, pressure_sign=1.0, add_inplane_edge_loads=False),
    )
    negative = build_symmetric_load_case(
        _Calc(),
        model,
        AnyStructureFEMConfig(pressure_pa=5.0, pressure_sign=-1.0, add_inplane_edge_loads=False),
    )

    positive_resultant = load_case_resultant(model, positive)
    negative_resultant = load_case_resultant(model, negative)

    np.testing.assert_allclose(positive_resultant.force, [0.0, 0.0, 10.0], atol=1.0e-12)
    np.testing.assert_allclose(negative_resultant.force, [0.0, 0.0, -10.0], atol=1.0e-12)
    np.testing.assert_allclose(positive_resultant.force, -negative_resultant.force, atol=1.0e-12)


def test_shell_geometric_stiffness_from_membrane_compression_is_symmetric_and_scales():
    model = build_fe_model_from_generated_geometry(_square_shell_geometry())
    KG_1, _ = assemble_geometric_stiffness_matrix(model, {1: {"membrane_compression_x": 10.0, "membrane_compression_y": 4.0}})
    KG_3, _ = assemble_geometric_stiffness_matrix(model, {1: {"membrane_compression_x": 30.0, "membrane_compression_y": 12.0}})

    assert KG_1.nnz > 0
    np.testing.assert_allclose(KG_1.toarray(), KG_1.toarray().T, atol=1.0e-12)
    np.testing.assert_allclose(KG_3.toarray(), 3.0 * KG_1.toarray(), rtol=1.0e-12, atol=1.0e-12)


def test_recovered_static_prestress_drives_buckling_workflow_for_full_generated_cylinder():
    config = AnyStructureFEMConfig(pressure_pa=2_000.0, pressure_sign=-1.0, add_inplane_edge_loads=False, num_buckling_modes=2)

    result = run_anystructure_fem_mode(_Calc(), _cylinder_geometry(), config)

    assert result.static_solver_status == "converged"
    assert result.valid, result.to_dict()
    assert result.critical_load_factor is not None
    assert result.critical_load_factor > 0.0
    assert result.buckling_load_factors == sorted(result.buckling_load_factors)
    assert result.prestress_summary["shell_elements"] > 0
    assert result.stress_max > 0.0


def test_prestress_recovery_returns_shell_states_from_static_result():
    geometry = _square_shell_geometry()
    geometry["supports"] = [
        {"node_ids": [1, 4], "dof_constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}},
        {"node_ids": [2, 3], "dof_constraints": {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}},
    ]
    model = build_fe_model_from_generated_geometry(geometry)
    load_case = build_symmetric_load_case(_Calc(), model, AnyStructureFEMConfig(pressure_pa=0.0))

    displacements, solver_info = solve_linear(model, load_case)
    states, summary = recover_prestress_from_static_result(model, displacements)

    assert solver_info["convergence_info"]["status"] == "converged"
    assert 1 in states
    assert summary["shell_elements"] == 1
    assert np.asarray(states[1]["membrane_forces_at_gauss"]).shape == (4, 3)
    assert np.asarray(states[1]["bending_moments_at_gauss"]).shape == (4, 3)


def test_qualified_s3_prestress_consumer_preserves_all_seven_native_stations():
    model, element = _qualified_s3_model()
    displacement = 2.0e-5 * np.sin(
        np.arange(model.mesh.dof_manager.total_dofs, dtype=float) + 0.23
    )
    direct = element.compute_stresses(
        model.mesh,
        displacement,
        model.get_material("steel"),
    )

    states, summary = recover_prestress_from_static_result(model, displacement)

    membrane = np.asarray(states[1]["membrane_forces_at_gauss"])
    bending = np.asarray(states[1]["bending_moments_at_gauss"])
    assert membrane.shape == bending.shape == (len(element.gauss_points), 3) == (7, 3)
    np.testing.assert_array_equal(membrane, direct["membrane_resultants"])
    np.testing.assert_array_equal(bending, direct["bending_resultants"])
    np.testing.assert_array_equal(
        (
            states[1]["membrane_force_x"],
            states[1]["membrane_force_y"],
            states[1]["membrane_force_xy"],
        ),
        np.average(membrane, axis=0, weights=element.gauss_weights),
    )
    assert states[1]["resultant_summary_policy"] == RESULTANT_SUMMARY_POLICY_ID
    assert states[1]["bubble_linearization_policy"] == (
        REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
    )
    assert states[1]["through_thickness_stress_profile"] == (
        HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID
    )
    assert "transverse_shear_resultants" not in states[1]
    assert summary["shell_elements"] == 1
    assert summary["max_shell_compression_resultant"] == pytest.approx(
        max(float(np.max(-membrane[:, :2])), 0.0)
    )
    geometric = element.compute_geometric_stiffness_matrix(
        model.mesh,
        model.get_material("steel"),
        states[1],
    )
    assert np.all(np.isfinite(geometric))
    np.testing.assert_array_equal(geometric, geometric.T)
    assert np.linalg.norm(geometric) > 0.0


@pytest.mark.parametrize(
    "mutation",
    (
        "six_stations",
        "eight_stations",
        "mismatched",
        "nan",
        "infinite",
        "missing_membrane",
        "missing_bending",
    ),
)
def test_shell_prestress_consumer_rejects_wrong_or_nonfinite_station_fields(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    model, _element = _qualified_s3_model()
    membrane = np.zeros((7, 3), dtype=float)
    bending = np.zeros_like(membrane)
    if mutation == "six_stations":
        membrane = membrane[:6]
        bending = bending[:6]
    elif mutation == "eight_stations":
        membrane = np.vstack((membrane, np.zeros((1, 3))))
        bending = np.vstack((bending, np.zeros((1, 3))))
    elif mutation == "mismatched":
        bending = bending[:6]
    elif mutation == "nan":
        membrane[3, 1] = np.nan
    else:
        if mutation == "infinite":
            bending[5, 2] = np.inf

    stress = {}
    if mutation != "missing_membrane":
        stress["membrane_resultants"] = membrane
    if mutation != "missing_bending":
        stress["bending_resultants"] = bending

    monkeypatch.setattr(
        "anyfesolver.anystructure_fem_mode.compute_stresses",
        lambda *_args, **_kwargs: {1: stress},
    )
    with pytest.raises(
        ValueError,
        match=(
            "requires paired membrane and bending"
            if mutation.startswith("missing")
            else "inconsistent or non-finite generalized shell resultant"
        ),
    ):
        recover_prestress_from_static_result(
            model,
            np.zeros(model.mesh.dof_manager.total_dofs),
        )


def test_shell_prestress_consumer_prefers_native_resultants_over_stress_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = _qualified_s3_model()
    membrane = np.arange(21, dtype=float).reshape(7, 3) - 9.0
    bending = 0.1 * membrane
    monkeypatch.setattr(
        "anyfesolver.anystructure_fem_mode.compute_stresses",
        lambda *_args, **_kwargs: {
            1: {
                "membrane_resultants": membrane,
                "bending_resultants": bending,
                "membrane_xx": np.full(7, 9.9e20),
                "membrane_yy": np.full(7, -9.9e20),
                "membrane_xy": np.full(7, 4.2e20),
            }
        },
    )

    states, _summary = recover_prestress_from_static_result(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs),
    )

    np.testing.assert_array_equal(states[1]["membrane_forces_at_gauss"], membrane)
    np.testing.assert_array_equal(states[1]["bending_moments_at_gauss"], bending)
    weighted = np.average(
        membrane,
        axis=0,
        weights=model.mesh.elements[1].gauss_weights,
    )
    np.testing.assert_array_equal(
        (
            states[1]["membrane_force_x"],
            states[1]["membrane_force_y"],
            states[1]["membrane_force_xy"],
        ),
        weighted,
    )
    assert not np.array_equal(weighted, np.mean(membrane, axis=0))


def test_prestress_recovery_preserves_pure_bending_gauss_resultants():
    model = build_fe_model_from_generated_geometry(_square_shell_geometry())
    material = model.get_material("steel")
    element = model.mesh.get_element(1)
    kappa_x = 1.1e-3
    displacements = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacements[node.dofs[4]] = kappa_x * float(node.x)

    states, summary = recover_prestress_from_static_result(model, displacements)

    membrane = np.asarray(states[1]["membrane_forces_at_gauss"], dtype=float)
    bending = np.asarray(states[1]["bending_moments_at_gauss"], dtype=float)
    expected_mx = (
        material.elastic_modulus
        * float(element.thickness) ** 3
        * kappa_x
        / (12.0 * (1.0 - material.poisson_ratio**2))
    )
    np.testing.assert_allclose(membrane, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(bending[:, 0], expected_mx, rtol=1.0e-10, atol=1.0e-9)
    np.testing.assert_allclose(
        bending[:, 1],
        material.poisson_ratio * expected_mx,
        rtol=1.0e-10,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(bending[:, 2], 0.0, atol=1.0e-9)
    assert summary["shell_elements"] == 1


def test_prestress_recovery_uses_committed_nonlinear_shell_state():
    model = build_fe_model_from_generated_geometry(_square_shell_geometry())
    element = model.mesh.get_element(1)
    num_layers = 3
    num_points = len(element.gauss_points) * num_layers
    committed_stress = np.array([-250.0e6, -50.0e6, -10.0e6])
    committed_state = {
        "layer_strain": np.zeros((num_points, 3), dtype=float),
        "plastic_strain": np.zeros((num_points, 3), dtype=float),
        "layer_stress": np.tile(committed_stress, (num_points, 1)),
        "alpha": np.zeros(num_points, dtype=float),
    }
    displacements = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    nonlinear_result = SimpleNamespace(
        element_states={1: committed_state},
        displacements=displacements,
        info={"kinematics": "von_karman"},
        status="completed",
    )

    states, summary = recover_prestress_from_static_result(
        model,
        displacements,
        nonlinear_result=nonlinear_result,
    )

    membrane = np.asarray(states[1]["membrane_forces_at_gauss"], dtype=float)
    expected = committed_stress * float(element.thickness)
    np.testing.assert_allclose(membrane, np.tile(expected, (4, 1)))
    assert np.max(np.abs(membrane)) > 0.0
    assert summary["max_shell_compression_resultant"] == pytest.approx(
        -expected[0]
    )
    provenance = summary["stress_recovery"]
    assert provenance["mode"] == "material_history"
    assert provenance["state_source"] == "SimpleNamespace.element_states"
    assert provenance["history_aware_element_ids"] == [1]
    assert provenance["per_element_source"][1] == "committed_shell_layer_state"
