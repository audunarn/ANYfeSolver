"""Saved-state KKT algebra and identity rejection, without optimization."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from fixtures import d1_radial_linear_saved_assessment as assessment
from docs.d1_review_support.d1_radial_two_body_reference import RadialTwoBodyReference


@pytest.fixture
def fixture(monkeypatch):
    # Exact miniature joint two-body normal algebra. Production-size geometry
    # preparation is assessed once against the existing saved 12/12 QP.
    xy = ((.1, .1), (1.1, .1), (1.1, 1.1), (.1, 1.1))
    nodes = {n: SimpleNamespace(x=x, y=y) for n, (x, y) in enumerate(xy+xy, 1)}
    dofs = lambda n: tuple(range(6*(n-1), 6*n))
    model = SimpleNamespace(mesh=SimpleNamespace(nodes=nodes,
        elements={1: SimpleNamespace(node_ids=(1, 2, 3, 4)),
                  2: SimpleNamespace(node_ids=(5, 6, 7)),
                  3: SimpleNamespace(node_ids=(5, 7, 8))},
        dof_manager=SimpleNamespace(total_dofs=48, get_node_dofs=dofs)))
    definition = SimpleNamespace(surfaces=(SimpleNamespace(element_ids=(1,)),
                                          SimpleNamespace(element_ids=(2, 3))),
        interactions=(SimpleNamespace(normal_stiffness=1e6),))
    reference = RadialTwoBodyReference(assessment.EXPECTED["source_guided_shear_n_m"],
                                      assessment.EXPECTED["target_guided_shear_n_m"])
    transform = np.zeros((48, 8))
    for i in range(8):
        transform[dofs(i+1)[2], i] = 1.
    controls = transform[:, (0, 4)].T
    free_map = transform[:, (1, 2, 3, 5, 6, 7)]
    gap_map = transform[:, :4].T-transform[:, 4:].T
    prepared = assessment._Prepared(np.eye(48), transform, free_map, controls,
                                   (1, 2, 3, 4), np.full(4, .25), gap_map, 1.)
    monkeypatch.setattr(assessment, "_prepare", lambda *args: prepared)
    program = SimpleNamespace(controls=(SimpleNamespace(terms=((1, "uz", 1.),)),
                                        SimpleNamespace(terms=((5, "uz", 1.),))),
        nodal_forces=(), force_values_at=lambda step: (), target_at=lambda step: (0., 0.))
    metadata = deepcopy(assessment.EXPECTED)
    saved = deepcopy(metadata)
    saved["steps"] = [dict(step=step, displacements=np.zeros(48).tolist(),
                            nodal_force_n=np.zeros(4).tolist()) for step in (1, 2)]
    return model, definition, program, reference, metadata, saved


def test_open_saved_state_and_complete_joint_span(fixture):
    result = assessment.assess_saved_radial_linear(*fixture)
    assert result["kkt_passed"]
    assert result["source_node_ids"] == [1, 2, 3, 4]
    assert result["joint_normal_free_dimensions"] == 6
    assert result["joint_normal_control_dimensions"] == 2
    assert all(row["equilibrium_relative_force"] == 0. for row in result["steps"])
    assert result["final_reference_pressure_observables"]["total"]["force_n"] == 0.
    assert result["final_reference_pressure_observables"]["total"]["resultant_xy_m"] is None


def test_loaded_contact_kkt_observables_and_independent_both_body_samples(fixture):
    model, definition, program, reference, metadata, saved = fixture
    g = reference.gap_m
    program.nodal_forces = tuple(SimpleNamespace(node_id=n, dof="uz") for n in range(1, 5))
    program.force_values_at = lambda step: (-2*g,)*4
    program.target_at = lambda step: (-1.5*g, -.5*g)
    pose = np.zeros(48)
    for n in range(1, 9):
        pose[model.mesh.dof_manager.get_node_dofs(n)[2]] = (-1.5*g if n <= 4 else -.5*g)
    for row in saved["steps"]:
        row.update(displacements=pose.tolist(), nodal_force_n=[.5*g]*4)
    result = assessment.assess_saved_radial_linear(*fixture)
    assert result["kkt_passed"]
    observable = result["final_reference_pressure_observables"]
    assert observable["total"]["force_n"] == pytest.approx(2*g)
    assert np.asarray([[r["force_n"] for r in row] for row in observable["bins"]]) == pytest.approx(np.full((2, 2), .5*g))
    assert observable["total"]["moment_nm"] == pytest.approx((0., 0., 0.), abs=1e-20)
    assert observable["total"]["resultant_xy_m"] == pytest.approx((.6, .6))
    for sample in result["steps"][-1]["samples"]:
        assert sample["source_uz_m"] == pytest.approx(-1.5*g)
        assert sample["target_uz_m"] == pytest.approx(-.5*g)
        assert sample["parallel_skin_gap_m"] == pytest.approx(0., abs=1e-20)


@pytest.mark.parametrize("name", ["target_cells", "target_right_x_m", "target_y0_m",
                                  "source_guided_shear_n_m", "target_guided_shear_n_m",
                                  "target_phi_squared_range", "gap_m", "exact_force_n"])
def test_changed_saved_fixture_identity_rejected(fixture, name):
    saved = fixture[-1]
    if name == "target_phi_squared_range":
        saved[name] = (.4, .4)
    else:
        saved[name] *= 1.01
    with pytest.raises(ValueError, match="identity mismatch"):
        assessment.assess_saved_radial_linear(*fixture)


def test_missing_saved_state_rejected(fixture):
    del fixture[-1]["steps"][1]["nodal_force_n"]
    with pytest.raises(ValueError, match="missing saved"):
        assessment.assess_saved_radial_linear(*fixture)


def test_negative_nodal_force_is_a_failed_kkt_diagnostic(fixture):
    fixture[-1]["steps"][1]["nodal_force_n"][1] = -1e-6
    result = assessment.assess_saved_radial_linear(*fixture)
    assert not result["kkt_passed"]
    assert not result["steps"][1]["nonnegative_nodal_force"]


def test_free_equilibrium_and_constraint_defects_are_distinct(fixture):
    fixture[-1]["steps"][1]["displacements"][8] = 1e-5
    result = assessment.assess_saved_radial_linear(*fixture)
    assert not result["kkt_passed"]
    assert result["steps"][1]["equilibrium_relative_force"] == pytest.approx(1e-5/.08)
    assert result["steps"][1]["structural_constraint_relative_gap"] == 0.
    fixture[-1]["steps"][1]["displacements"][0] = 1e-5
    result = assessment.assess_saved_radial_linear(*fixture)
    assert result["steps"][1]["structural_constraint_relative_gap"] == pytest.approx(1e-5/fixture[3].gap_m)


def test_prescribed_control_defect_is_rejected(fixture):
    fixture[-1]["steps"][1]["displacements"][2] = 1e-5
    result = assessment.assess_saved_radial_linear(*fixture)
    assert not result["kkt_passed"]
    assert result["steps"][1]["equilibrium_relative_force"] == 0.
    assert result["steps"][1]["control_constraint_relative_gap"] > 1.


def test_nonfinite_and_wrong_order_saved_states_rejected(fixture):
    fixture[-1]["source_node_ids"] = [4, 3, 2, 1]
    with pytest.raises(ValueError, match="order"):
        assessment.assess_saved_radial_linear(*fixture)
    fixture[-1]["source_node_ids"] = [1, 2, 3, 4]
    fixture[-1]["steps"][1]["displacements"][8] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assessment.assess_saved_radial_linear(*fixture)


def test_exact_rational_coefficient_audit_covers_original_gauss_arithmetic(fixture):
    model, definition, program, reference, _, _ = fixture
    ids, support, work, area = assessment.independent_gap_work(model, definition, 0.)
    prepared = replace(assessment._prepare(model, definition, program, reference),
                       source_ids=ids, support=support, gap_map=work/support[:, None], area=area)
    gap_error, support_error, record = assessment._audit_gap_coefficients(model, definition, prepared)
    assert record["clipped_pairs"] == 2
    assert np.max(gap_error) < 1e-12
    assert np.max(support_error) < 1e-12
    assert abs(Fraction(float(prepared.gap_map[0, 2]))-Fraction(4, 9)) <= Fraction(float(gap_error[0, 2]))
    assert assessment._triangle_moments(((Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))))[(2, 1)] == Fraction(1, 60)
    definition.surfaces[1].element_ids = (2,)
    with pytest.raises(ValueError, match="coverage"):
        assessment._audit_gap_coefficients(model, definition, prepared)


def test_fixed_branch_correction_retains_saved_values_and_bounds_observables(fixture, monkeypatch):
    model, definition, program, reference, _, saved = fixture
    g = reference.gap_m
    program.nodal_forces = tuple(SimpleNamespace(node_id=n, dof="uz") for n in range(2, 5))
    program.force_values_at = lambda step: ((0.,)*3 if step == 1 else (-2*g,)*3)
    pose = np.zeros(48)
    for source, target in zip(range(2, 5), range(6, 9), strict=True):
        pose[model.mesh.dof_manager.get_node_dofs(source)[2]] = -1.5*g
        pose[model.mesh.dof_manager.get_node_dofs(target)[2]] = -.5*g
    saved["steps"][1].update(displacements=pose.tolist(), nodal_force_n=[0., .5*g+1e-8, .5*g, .5*g])
    monkeypatch.setattr(assessment, "_audit_gap_coefficients", lambda m, d, p:
        (np.zeros_like(p.gap_map), np.zeros_like(p.support), {"kind": "exact test coefficients"}))
    original = float(sum(saved["steps"][1]["nodal_force_n"]))
    result = assessment.assess_saved_radial_linear(*fixture)
    assert result["kkt_passed"] and result["solution_uncertainty_assessed"]
    assert result["final_observables"]["force_n"] == pytest.approx(original, abs=1e-20)
    assert result["final_observables"]["force_solution_uncertainty_n"] >= 1e-8-1e-20
    diagnostic = result["solution_uncertainty"]["diagnostics"]
    assert diagnostic["active_modes"] == 3
    assert diagnostic["correction_force_l1_n"] == pytest.approx(1e-8, rel=1e-6)
    assert diagnostic["scaled_remaining_residual_infinity"] < 1e-10
    assert diagnostic["inverse_residual_infinity_bound"] < .01
    assert all(s["solution_uncertainty_m"] >= 0. for s in result["final_observables"]["source_samples"])
    assert np.all(np.asarray(result["final_observables"]["bin_solution_uncertainty_n"]) >= 0.)


def test_changed_or_ambiguous_active_branch_rejected():
    reference = SimpleNamespace(gap_m=1., load_pa=.08)
    prepared = assessment._Prepared(np.eye(2), np.eye(2), np.eye(2), np.zeros((0, 2)),
        (1,), np.ones(1), np.array(((1., -1.),)), 1.)
    with pytest.raises(ValueError, match="branch"):
        assessment._fixed_active_correction(prepared, np.array((-1., 0.)), np.array((.1,)),
            np.array((2., 0.)), np.zeros(0), reference, np.zeros((1, 2)))
    duplicate = replace(prepared, source_ids=(1, 2), support=np.ones(2),
                        gap_map=np.array(((1., -1.), (1., -1.))))
    with pytest.raises(ValueError, match="branch"):
        assessment._fixed_active_correction(duplicate, np.array((-1.5, -.5)), np.array((.5, 0.)),
            np.array((-2., 0.)), np.zeros(0), reference, np.zeros((2, 2)))


def test_singular_and_excess_dimension_inverse_rejected():
    with pytest.raises((ValueError, assessment.linalg.LinAlgWarning)):
        assessment._checked_inverse(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="budget"):
        assessment._checked_inverse(np.eye(assessment.MAXIMUM_CORRECTION_DIMENSION+1))


def test_offdiagonal_coefficient_uncertainty_transfers_large_component():
    base = np.array((1e-12, 1.))
    perturbation = np.array(((0., -.08), (0., 0.)))
    actual = np.linalg.solve(np.eye(2)+perturbation, base)
    eta = np.linalg.norm(perturbation, np.inf)
    assert actual[0] > (base/(1.-eta))[0]*1e9
    envelope = assessment._coefficient_perturbation_envelope(base, eta)
    assert np.all(actual <= envelope)


def test_inactive_gap_includes_coefficient_error_on_remaining_displacement():
    reference = SimpleNamespace(gap_m=1., load_pa=.08)
    prepared = assessment._Prepared(np.eye(3), np.eye(3), np.eye(3), np.zeros((0, 3)),
        (1, 2), np.ones(2), np.array(((1., -1., 0.), (0., 0., 0.))), 1.)
    args = (prepared, np.array((-1.5, -.5, 0.)), np.array((.5, 0.)),
            np.array((-2., 0., 0.)), np.zeros(0), reference)
    corrected_u, _, remaining_u, _, diagnostic = assessment._fixed_active_correction(*args, np.zeros((2, 3)))
    assert diagnostic["minimum_inactive_gap_margin_m"] > .99
    assert corrected_u[2] == 0. and remaining_u[2] > 0.
    coefficient_error = np.zeros((2, 3))
    coefficient_error[1, 2] = 2./remaining_u[2]
    # The nominal inactive row and evaluated state have zero third-coordinate
    # work, but its uncertain coefficient acting on remaining displacement
    # makes the positive-gap claim unresolved.
    assert (coefficient_error @ abs(corrected_u))[1] == 0.
    with pytest.raises(ValueError, match="branch"):
        assessment._fixed_active_correction(*args, coefficient_error)
