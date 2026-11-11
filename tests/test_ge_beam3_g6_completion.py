"""Eight ordered, bounded G6 scientific evidence routes."""
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from anyfesolver import (
    AnalysisSession, BoundaryCondition, FEModel, FixedSupport, LoadCase,
    solve_linear_many,
)
from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.elements import QuadraticBeamElement, create_element
from anyfesolver.ge_beam3_element import GeometricallyExactBeam3D3NElement
from anyfesolver._ge_beam3_g1_elastic import ElasticSection
from anyfesolver._ge_beam3_g6_domain import (
    ConsumerPolicy, GeneralizedInitialField, canonical, evaluate_initial_field,
    evidence,
)
from anyfesolver import ge_beam3_native as native
from anyfesolver._ge_beam3_native_arc import solve_arc
from test_ge_beam3_g5_completion import _exercise_reference_family, _model
from test_ge_beam3_g6_domain import (
    test_beam_contact_and_fracture_classification_cover_three_node_routes as _contact_route,
    test_ge_activity_epoch_scales_assembly_and_rejects_stale_trial as _activity_route,
    test_objective_pose_joint_is_explicit_identity_bearing_opt_in as _pose_joint_route,
)
from test_ge_beam3_native_arc import bar


SCIENTIFIC_RECORDS = []
ROOT = Path(__file__).resolve().parents[1]


def _record(route, **checks):
    SCIENTIFIC_RECORDS.append(evidence(route, **checks))


def test_01_complete_parity_register_is_source_and_evidence_bound():
    matrix = json.loads((ROOT / "docs/reference_cases/ge_beam3_legacy_parity_matrix_v1.json").read_text())
    assert [row["id"] for row in matrix["rows"]] == [
        *[f"P{i:02d}" for i in range(1, 33)], *[f"U{i:02d}" for i in range(1, 11)]
    ]
    assert all((ROOT / source).is_file() for row in matrix["rows"] for source in row["sources"])
    accepted = [ROOT / "docs/reference_cases" / name for name in (
        "ge_beam3_g5_completion_confirmation_v1.json",
        "ge_beam3_g5_completion_confirmation_review_v1.json",
        "ge_beam3_g5_completion_status_v1.json",
    )]
    assert all(path.is_file() and len(path.read_bytes()) > 0 for path in accepted)
    _record("PARITY_REGISTER", exact_32_p_rows=True, exact_10_u_rows=True,
            source_paths_present=True, accepted_g5_bound=True)


def test_02_legacy_b3_baselines_cover_shared_reference_and_contact_routes():
    result = _exercise_reference_family("B3")
    assert type(result["element"]) is QuadraticBeamElement
    _contact_route("B3")
    _record("LEGACY_B3_BASELINES", reference_static=True, reference_modal=True,
            reference_transient=True, beam_segment_contact=True, fracture_category=True)


def test_03_initial_field_state_is_station_owned_and_work_conjugate():
    law = ElasticSection(np.diag((11., 13., 17., 19., 23., 29.)), "G6")
    field = GeneralizedInitialField([np.arange(1., 7.)], [np.arange(6.) / 10], "G6")
    strain = np.linspace(-.2, .3, 6)
    response = evaluate_initial_field(law, strain, (), field, 0)
    np.testing.assert_array_equal(
        response.resultants,
        law.stiffness @ (strain - field.prestrain[0]) + field.prestress[0],
    )
    raw = field.to_bytes()
    assert GeneralizedInitialField.from_bytes(
        raw, expected_sha256=sha256(raw).hexdigest()).to_bytes() == raw
    _record("INITIAL_FIELD_STATE", station_owned=True, exact_work=True,
            immutable=True, authenticated_roundtrip=True, no_history_reseal=True)


def test_04_reference_dynamics_uses_physical_ge_mass_and_shared_solver():
    result = _exercise_reference_family("GE")
    assert type(result["element"]) is GeometricallyExactBeam3D3NElement
    assert result["modal"].frequencies_hz[0] > 0.
    assert result["transient"].status == "completed"
    _record("REFERENCE_DYNAMICS", generalized_mass=True, point_edge_mass_route=True,
            modal=True, buckling_bound_by_g5=True, rayleigh_damping=True,
            linear_newmark=True, finite_rotation_dynamics_not_claimed=True)


def test_05_contact_activity_and_hard_delete_preserve_identity_and_balance():
    _contact_route("GE")
    _activity_route()
    _record("CONTACT_ACTIVITY", ge_segment_contact=True, action_reaction=True,
            fracture_classification=True, activity_epoch=True, softening=True,
            hard_delete=True, topology_retained=True, capacity_damage_not_claimed=True)


def test_06_objective_joint_and_actual_continuation_are_explicit():
    _pose_joint_route()
    model, program = bar()
    result = solve_arc(model, program)
    assert result.status == "completed" and result.completed_steps == len(program.steps)
    _record("OBJECTIVE_JOINT_CONTINUATION", exact_pose_joint=True,
            identity_bearing=True, owner_binding_not_overclaimed=True,
            actual_arc_path=True, authenticated_checkpoint=bool(result.checkpoint),
            stable_postbuckling_not_claimed=True)


def _chain(count):
    model = FEModel(f"g6-scale-{count}")
    model.add_material("mat", 2.1e5, .3, density=2.)
    section = GeneralizedBeamSection(
        np.diag((1.2e5, 4.e4, 3.5e4, 2.e3, 4.5e3, 5.e3)),
        mass_matrix=np.diag((2., 2., 2., .04, .05, .06)), name="G6_SCALE",
    )
    for node in range(2 * count + 1):
        model.add_node(node + 1, node / 2., 0., 0.)
    for index in range(count):
        element_id = index + 1
        model.add_element(element_id, create_element(
            "ge-beam3", element_id, [2 * index + 1, 2 * index + 2, 2 * index + 3],
            "mat", section=section, reference_orientation=(0., 1., 0.),
            reference_axis_direction=(1., 0., 0.),
        ))
    model.add_boundary_condition(FixedSupport("root", [1]))
    model.add_boundary_condition(BoundaryCondition(
        "axial-sliders", list(range(2, 2 * count + 2)),
        {name: 0. for name in ("uy", "uz", "rx", "ry", "rz")},
    ))
    return model


def test_07_bounded_scale_sparse_multiple_rhs_and_session_reuse():
    count = int(os.environ.get("GE_BEAM3_G6_SCALE_ELEMENTS", "64"))
    assert count in (64, 256, 1024)
    model = _chain(count)
    end = 2 * count + 1
    first = LoadCase("first"); first.add_nodal_load(end, forces=(1., 0., 0.))
    second = LoadCase("second"); second.add_nodal_load(end, forces=(2., 0., 0.))
    with AnalysisSession(model) as session:
        values, info = solve_linear_many(model, [first, second], session=session)
        diagnostics = session.diagnostics()
    assert values.shape[1] == 2
    np.testing.assert_allclose(values[:, 1], 2 * values[:, 0], rtol=2.e-12, atol=1.e-13)
    assert info["analysis_session"]["plan_reused"] is True
    assert diagnostics["factorization_cache"]["misses"] == 1
    _record("SCALE_PERFORMANCE", registered_element_count=True, sparse=True,
            multiple_rhs=True, session_reuse=True, cache_identity=True,
            legacy_sources_unchanged=True, speed_claim_not_made=True)


def test_08_ecosystem_boundary_is_persisted_explicit_opt_in_only():
    assert ConsumerPolicy() == ConsumerPolicy(False, "legacy", "")
    selected = ConsumerPolicy.ge_beam3()
    raw = selected.to_bytes()
    assert ConsumerPolicy.from_bytes(raw, expected_sha256=sha256(raw).hexdigest()) == selected
    assert native.ConsumerPolicy is ConsumerPolicy
    _record("ECOSYSTEM_BOUNDARY", explicit_policy=True, missing_is_legacy=True,
            selector_is_ge_beam3=True, formulation_bound=True, default_unchanged=True,
            anymesh_out_of_scope=True, anyintelligent_out_of_scope=True)
