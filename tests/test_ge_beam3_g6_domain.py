"""G6 remaining-domain integration tests; accepted G1--G5 is not rerun."""
from hashlib import sha256

import numpy as np
import pytest

from anyfesolver import (
    ContributionPolicy, ElementActivity, ElementActivityPolicy,
    assemble_damping_matrix, assemble_mass_matrix, assemble_stiffness_matrix,
)
from anyfesolver._ge_beam3_g1_elastic import ElasticSection
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_g6_domain import (
    ActivityEpochLease, ConsumerPolicy, GeneralizedInitialField, PARITY_DISPOSITIONS,
    ROUTES, adjudicate, canonical,
    evaluate_initial_field, evidence,
)
from anyfesolver.contact import (
    RigidSphereImpact, SphereContactConfig, assemble_sphere_contact_load_vector,
)
from anyfesolver.fracture import element_fracture_category, element_measure
from anyfesolver.validation import load_vector_resultant
from anyfesolver import ge_beam3_native as native
from test_ge_beam3_g5_completion import _model


def _all_activity_policy():
    return ElementActivityPolicy(**{
        name: ContributionPolicy.ACTIVITY
        for name in ("stiffness", "mass", "damping", "load", "contact")
    })


def test_station_owned_initial_field_is_exact_work_and_strictly_serialized():
    stiffness = np.diag((11., 13., 17., 19., 23., 29.))
    law = ElasticSection(stiffness, "G6 initial-field section")
    field = GeneralizedInitialField(
        [[1., -2., 3., -4., 5., -6.], [0., 0., 0., 0., 0., 0.]],
        [[.1, -.2, .3, -.4, .5, -.6], [0., 0., 0., 0., 0., 0.]],
        "frozen-manufactured-field",
    )
    strain = np.array((.7, -.1, .2, -.3, .4, -.5))
    response = evaluate_initial_field(law, strain, (), field, 0)
    expected = stiffness @ (strain - field.prestrain[0]) + field.prestress[0]
    np.testing.assert_array_equal(response.resultants, expected)
    np.testing.assert_array_equal(response.tangent, stiffness)
    assert response.potential == pytest.approx(
        .5 * (strain - field.prestrain[0]) @ stiffness
        @ (strain - field.prestrain[0]) + field.prestress[0] @ strain
    )
    direction = np.array((-.3, .2, .1, .4, -.2, .5))
    step = 1.e-7
    plus = evaluate_initial_field(law, strain + step * direction, (), field, 0)
    minus = evaluate_initial_field(law, strain - step * direction, (), field, 0)
    assert (plus.potential - minus.potential) / (2 * step) == pytest.approx(
        response.resultants @ direction, rel=2.e-9, abs=2.e-9
    )
    raw = field.to_bytes()
    restored = GeneralizedInitialField.from_bytes(raw, expected_sha256=sha256(raw).hexdigest())
    assert restored.to_bytes() == raw
    assert not restored.prestress.flags.writeable and not restored.prestrain.flags.writeable
    with pytest.raises(ValueError, match="duplicate"):
        GeneralizedInitialField.from_bytes(
            raw.replace(b'"identity":', b'"identity":"bad","identity":', 1),
            expected_sha256=sha256(raw.replace(b'"identity":', b'"identity":"bad","identity":', 1)).hexdigest(),
        )


def test_initial_field_composes_without_committing_stateful_section_origin():
    law = EllipsoidalGeneralizedSection(np.diag((11., 13., 17., 19., 23., 29.)),
                                        np.eye(6), 1000., 10.)
    origin = law.virgin()
    field = GeneralizedInitialField([[1., 2., 3., 4., 5., 6.]],
                                    [[.01, .02, .03, .04, .05, .06]], "stateful")
    before = canonical(origin)
    response = evaluate_initial_field(law, np.arange(6.) / 10., origin, field, 0)
    assert canonical(origin) == before
    assert response.constitutive_response.origin == origin
    np.testing.assert_array_equal(
        response.resultants,
        response.constitutive_response.resultants + field.prestress[0],
    )


@pytest.mark.parametrize("family", ["B3", "GE"])
def test_beam_contact_and_fracture_classification_cover_three_node_routes(family):
    model, element, _end = _model(family)
    element.cross_section["contact_radius"] = .05
    sphere = RigidSphereImpact(
        "g6-touch", radius=.1, mass=1., start_point=(1., 0., .5),
        travel_direction=(0., 0., -1.), speed=0.,
    )
    position = np.array((1., 0., .13))
    load, sphere_force, records = assemble_sphere_contact_load_vector(
        model, sphere, SphereContactConfig(penalty_stiffness=1000., beam_contact=True),
        position, np.zeros(3),
    )
    assert len(records) == 1
    assert records[0].contact_classification == "beam"
    assert records[0].normal_force == pytest.approx(20.)
    resultant = load_vector_resultant(model, load)
    np.testing.assert_allclose(resultant.force + sphere_force, 0., rtol=0., atol=1.e-12)
    assert element_fracture_category(element) == "beam"
    assert element_measure(model.mesh, element) == pytest.approx(2.)


def test_ge_activity_epoch_scales_assembly_and_rejects_stale_trial():
    model, _element, _end = _model("GE")
    baseline_k, _ = assemble_stiffness_matrix(model)
    baseline_m, _ = assemble_mass_matrix(model)
    baseline_c, _ = assemble_damping_matrix(model, .2, .3)
    activity = ElementActivity([1], policy=_all_activity_policy())
    model.set_element_activity(activity)
    first = ActivityEpochLease.capture(activity)
    first.validate(activity)
    activity.set_activity([1], [.25], reason="g6-softening")
    with pytest.raises(RuntimeError, match="changed during open"):
        first.validate(activity)
    second = ActivityEpochLease.capture(activity)
    k, _ = assemble_stiffness_matrix(model)
    m, _ = assemble_mass_matrix(model)
    c, _ = assemble_damping_matrix(model, .2, .3)
    np.testing.assert_allclose(k.toarray(), .25 * baseline_k.toarray(), rtol=0., atol=0.)
    np.testing.assert_allclose(m.toarray(), .25 * baseline_m.toarray(), rtol=0., atol=0.)
    np.testing.assert_allclose(c.toarray(), .25 * baseline_c.toarray(), rtol=0., atol=0.)
    restored = ElementActivity.from_restart(activity.to_restart(include_history=True))
    assert ActivityEpochLease.capture(restored) == second
    activity.hard_delete([1], reason="g6-hard-delete")
    zero_k, _ = assemble_stiffness_matrix(model)
    zero_m, _ = assemble_mass_matrix(model)
    assert zero_k.nnz == zero_m.nnz == 0
    assert tuple(model.mesh.elements) == (1,)


def test_objective_pose_joint_is_explicit_identity_bearing_opt_in():
    positions = np.array(((0., 0., 0.), (1., .2, -.1)), dtype=float)
    frames = np.array((np.eye(3), np.eye(3)), dtype=float)
    joint = native.ObjectivePoseJoint(positions, frames)
    raw = joint.to_bytes()
    restored = native.ObjectivePoseJoint.from_bytes(raw, expected_sha256=sha256(raw).hexdigest())
    result = restored.evaluate(positions, frames, np.zeros(6))
    np.testing.assert_array_equal(result.constraints, np.zeros(6))
    np.testing.assert_array_equal(result.residual, np.zeros(18))
    assert result.production_qualified is False
    assert result.owner_binding_authorized is False


def test_ecosystem_policy_is_explicit_hash_bound_and_missing_means_legacy():
    assert ConsumerPolicy() == ConsumerPolicy(False, "legacy", "")
    selected = ConsumerPolicy.ge_beam3()
    raw = selected.to_bytes()
    assert ConsumerPolicy.from_bytes(raw, expected_sha256=sha256(raw).hexdigest()) == selected
    with pytest.raises(ValueError):
        ConsumerPolicy(True, "ge-beam3", "wrong-formulation")


def test_g6_records_and_terminal_are_closed_world_and_mutation_sensitive():
    records = [evidence(route, complete=True) for route in ROUTES]
    rows = dict(PARITY_DISPOSITIONS)
    installed = {
        "wheel_sha256": "a" * 64,
        "probe_sha256": "b" * 64,
        "replicas_byte_identical": True,
        "anyfem_probe_sha256": "c" * 64,
        "anystructure_probe_sha256": "d" * 64,
    }
    result = adjudicate(records, parity_rows=rows, installed_artifact=installed)
    assert result["terminal"] == "PROVISIONAL_GO_GE_BEAM3_FULL_LEGACY_DOMAIN_PARITY_OPT_IN"
    changed = [dict(row) for row in records]
    changed[0] = {**changed[0], "production_default_qualified": True}
    with pytest.raises(ValueError):
        adjudicate(changed, parity_rows=rows, installed_artifact=installed)
    unresolved = dict(rows); unresolved["P29"] = "PARTIAL"
    with pytest.raises(ValueError, match="unresolved"):
        adjudicate(records, parity_rows=unresolved, installed_artifact=installed)
