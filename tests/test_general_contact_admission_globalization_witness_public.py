"""Focused admission/globalization checks: witness math, wiring, public selection.

Small owner module for the D1 admission/globalization work: directed interval
division and witness enclosures (negative numerators, rational enclosure,
endpoint error boxes, varying actual normals, refusal-only soundness), the
decision-preserving witness wiring at both production gap-budget sites, and
the default/enabled public settings contract, request identity and off-path
refusal behavior. No large eight-macro fixture is run here.
"""
from dataclasses import replace
from fractions import Fraction
import math

import numpy as np
import pytest

from anyfesolver import (ContactRequestIdentityError, LoadCase,
                       identify_general_contact_request)
from anyfesolver.general_contact import validate_general_contact
from anyfesolver.general_contact import s3_patch_static
from anyfesolver.general_contact.core import _contact_settings_contract
from anyfesolver.general_contact.q4_overlap_local_bound import (
    _divide_positive,
    overlap_violating_point_witness,
)
from anyfesolver.general_contact.request_identity import (
    verify_general_contact_request,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_interior_press_dev import build_interior_press
from test_general_contact_q4_s3_patch_static import _case

_CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION = "declared_macro_path_physical_substeps_v1"
_CONTINUATION_FIELDS = ("path_continuation_maximum_halvings",
                         "path_stagnation_patience",
                         "path_stagnation_fraction",
                         "path_stagnation_merit_rate")


def _d1_case():
    return build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", force_n=0.04,
    )


def _enabled(settings):
    return replace(settings, q4_source_gap_certificate="overlap_local_dev",
                   path_continuation_maximum_halvings=6)


@pytest.mark.parametrize("numerator,denominator", [
    ((-1., -1.), (1., 2.)),
    ((-2., -1.), (.5, 2.)),
    ((-1., 1.), (.25, 4.)),
    ((3., 5.), (1., 1.)),
    ((-7., -3.), (2., 9.)),
    ((-1., 2.), (.125, .5)),
])
def test_divide_positive_encloses_all_four_directed_quotients(numerator, denominator):
    lower, upper = _divide_positive(numerator, denominator)
    exact = [Fraction(numerator[a]) / Fraction(denominator[b])
             for a in (0, 1) for b in (0, 1)]
    assert Fraction(float(lower)) <= min(exact)
    assert max(exact) <= Fraction(float(upper))
    assert float(lower) <= float(upper)
    # The historical inverted pairing: a negative numerator over a positive
    # denominator attains its minimum at the denominator's lower endpoint.
    if numerator[1] < 0.:
        assert float(lower) <= min(numerator) / max(denominator) <= float(upper)


_TARGET = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
_PLANAR = np.array([[.2, .2, -.2], [.6, .2, -.2],
                     [.6, .6, -.2], [.2, .6, -.2]])
_TILTED = np.array([[.2, .2, -.1], [.7, .2, -.3],
                    [.7, .7, -.1], [.2, .7, -.3]])


def test_witness_planar_penetration_reports_enclosed_physical_gap():
    diagnostics = {}
    found, checked = overlap_violating_point_witness(
        _PLANAR, _PLANAR, _TARGET, _TARGET, gap_tolerance=0.01,
        diagnostics=diagnostics)
    assert found is True and checked >= 1
    np.testing.assert_allclose(diagnostics["physical_gap_upper_m"], -0.2,
                               rtol=0., atol=1e-12)
    assert min(diagnostics["foot_barycentric_lower"]) >= 0.
    assert sum(diagnostics["foot_barycentric_upper"]) <= 1.


def test_witness_uses_actual_varying_normal_not_target_plane_height():
    diagnostics = {}
    found, _ = overlap_violating_point_witness(
        _TILTED, _TILTED, _TARGET, _TARGET, gap_tolerance=0.01,
        sample_params=((0.0, 0.5),), sample_times=(1.0,),
        diagnostics=diagnostics)
    assert found is True
    # r = N/(P.T) = -0.2/0.25; gap = r*(P.Ns)/|Ns| = -0.8*sqrt(0.0725).
    expected = -0.8 * math.sqrt(0.0725)
    np.testing.assert_allclose(diagnostics["physical_gap_upper_m"], expected,
                               rtol=0., atol=1e-12)
    # The target-plane height at the same point is -0.2: the varying actual
    # parametric normal changes the physical gap, and the witness reports the
    # physical quantity, never the plane height.
    assert abs(diagnostics["physical_gap_upper_m"] - (-0.2)) > 1e-3
    assert min(diagnostics["foot_barycentric_lower"]) >= 0.
    assert sum(diagnostics["foot_barycentric_upper"]) <= 1.


def test_witness_requires_foot_inside_target_triangle():
    outside = np.array([[2., 2., -.2], [2.5, 2., -.2],
                        [2.5, 2.5, -.2], [2., 2.5, -.2]])
    diagnostics = {}
    found, checked = overlap_violating_point_witness(
        outside, outside, _TARGET, _TARGET, gap_tolerance=0.01,
        diagnostics=diagnostics)
    assert found is False and checked > 0
    assert diagnostics["found"] is False
    assert diagnostics["physical_gap_upper_m"] is None
    assert diagnostics["foot_barycentric_lower"] is None
    assert diagnostics["foot_barycentric_upper"] is None


def test_witness_never_admits_and_checks_every_sample_when_clear():
    above = _PLANAR + np.array([0., 0., .4])
    diagnostics = {}
    found, checked = overlap_violating_point_witness(
        above, above, _TARGET, _TARGET, gap_tolerance=0.01,
        diagnostics=diagnostics)
    assert found is False and checked == 27
    assert diagnostics["found"] is False
    assert diagnostics["physical_gap_upper_m"] is None


def test_witness_error_boxes_widen_and_keep_the_violating_enclosure():
    base = {}
    found, _ = overlap_violating_point_witness(
        _PLANAR, _PLANAR, _TARGET, _TARGET, gap_tolerance=0.01,
        diagnostics=base)
    assert found is True
    widened = {}
    found, _ = overlap_violating_point_witness(
        _PLANAR, _PLANAR, _TARGET, _TARGET, gap_tolerance=0.01,
        endpoint_errors=(1e-9, 1e-9, 1e-9, 1e-9), diagnostics=widened)
    assert found is True
    assert (widened["physical_gap_upper_m"]
            >= base["physical_gap_upper_m"] - 1e-15)
    huge = {}
    found, _checked = overlap_violating_point_witness(
        _PLANAR, _PLANAR, _TARGET, _TARGET, gap_tolerance=0.01,
        endpoint_errors=(.5, .5, .5, .5), diagnostics=huge)
    assert isinstance(found, bool)


@pytest.mark.parametrize("kwargs", [
    {"gap_tolerance": -0.1},
    {"gap_tolerance": float("nan")},
    {"gap_tolerance": 0.01, "endpoint_errors": (0.,) * 3},
    {"gap_tolerance": 0.01, "sample_params": ((1.5, 0.),)},
    {"gap_tolerance": 0.01, "sample_times": (2.,)},
])
def test_witness_rejects_malformed_inputs(kwargs):
    with pytest.raises(ValueError):
        overlap_violating_point_witness(
            _PLANAR, _PLANAR, _TARGET, _TARGET, **kwargs)
    with pytest.raises(ValueError):
        overlap_violating_point_witness(
            _TARGET, _PLANAR, _TARGET, _TARGET, gap_tolerance=0.01)


def _press_manager():
    model, definition, settings, _program = _d1_case()
    settings = replace(settings, q4_source_gap_certificate="overlap_local_dev")
    manager = GeneralS3PatchManager(model, definition, settings)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in manager.law.source_node_ids:
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.2
    return manager, displacement


def _capped_matching(monkeypatch, calls):
    original = s3_patch_static.matching_projection_swept_lower

    def capped(*args, **kwargs):
        # Cap every production call shape with its minimum valid budget so
        # this unit wiring check fits the bounded job: the production initial
        # depth is 3 and adaptive validation requires adaptive_depth>=depth.
        # Admission-shaped calls (explicit gap_budget) accept node_budget=1,
        # a valid conservative finite-budget refusal for the deep test pose.
        # Diagnostic refinement calls (gap_budget None, refinement budget)
        # must cover the complete initial partition, so their minimum valid
        # budget is (4**4-1)//3 = 85; the production 1024-node search on the
        # grossly violating subdivision middle pose is the measured cost
        # driver and cannot fit the cap. Only the admission-shaped calls the
        # witness wiring bypasses are counted; the refinement calls are the
        # existing subdivision refusal mechanics, not the witness's target.
        if kwargs.get("gap_budget") is not None:
            kwargs["adaptive_depth"] = 3
            kwargs["node_budget"] = 1
            calls.append(1)
        else:
            kwargs["adaptive_depth"] = 3
            kwargs["node_budget"] = 85
        return original(*args, **kwargs)

    monkeypatch.setattr(s3_patch_static, "matching_projection_swept_lower",
                        capped)


def test_point_gap_budget_witness_skips_matching_without_admitting(monkeypatch):
    manager, displacement = _press_manager()
    positions = manager._positions(displacement)
    source, target = positions
    geometry = (manager.law.geometry if manager.projected_work
                else manager.law)
    candidates = geometry._candidate_pair_indices(source, target)
    target_count = len(geometry.target_face_indices)
    pairs = (range(len(geometry.pair_laws)) if candidates is None
             else sorted(candidates))
    first = source[geometry.source_face_indices[pairs[0] // target_count]]
    second = target[geometry.target_face_indices[pairs[0] % target_count]]
    assert overlap_violating_point_witness(
        first, first, second, second,
        gap_tolerance=manager.gap_tolerance)[0] is True

    calls = []
    _capped_matching(monkeypatch, calls)
    assert manager._point_gap_budget_admissible(positions) is False
    assert calls == []

    witness_calls = []

    def no_witness(*args, **kwargs):
        witness_calls.append(1)
        return (False, 0)

    monkeypatch.setattr(s3_patch_static, "overlap_violating_point_witness",
                        no_witness)
    assert manager._point_gap_budget_admissible(positions) is False
    assert len(witness_calls) >= 1
    assert len(calls) >= 1


def test_swept_certify_witness_skips_matching_without_admitting(monkeypatch):
    manager, displacement = _press_manager()
    start = np.zeros_like(displacement)
    # The production endpoint admissibility gate refuses this deep pose before
    # the swept pair certification; neutralize only that gate so the swept
    # witness wiring is reached with the real certify branch.
    monkeypatch.setattr(manager, "_point_gap_budget_admissible",
                        lambda positions: True)
    calls = []
    _capped_matching(monkeypatch, calls)
    assert manager.swept_increment_admissible(start, displacement, 0., 1.) is False
    assert calls == []

    witness_calls = []

    def no_witness(*args, **kwargs):
        witness_calls.append(1)
        return (False, 0)

    monkeypatch.setattr(s3_patch_static, "overlap_violating_point_witness",
                        no_witness)
    assert manager.swept_increment_admissible(start, displacement, 0., 1.) is False
    assert len(witness_calls) >= 1
    assert len(calls) >= 1


def test_settings_contract_omits_default_and_binds_enabled():
    _model, _definition, settings, _program = _d1_case()
    contract = _contact_settings_contract(settings)
    for name in _CONTINUATION_FIELDS:
        assert name not in contract
    enabled = _contact_settings_contract(_enabled(settings))
    assert enabled["path_continuation_maximum_halvings"] == 6
    assert enabled["path_stagnation_patience"] == 2
    assert enabled["path_stagnation_fraction"] == .125
    assert enabled["path_stagnation_merit_rate"] == .05


def test_contact_path_selection_admits_enabled_continuation():
    model, definition, settings, _program = _d1_case()
    report = validate_general_contact(model, definition, _enabled(settings),
                                      analysis_control="contact_path")
    assert report.valid


@pytest.mark.parametrize("control", ["force", "displacement", "arc_length"])
def test_real_public_validation_refuses_enabled_continuation_off_path(control):
    model, definition, settings, _program = _d1_case()
    report = validate_general_contact(model, definition, _enabled(settings),
                                      analysis_control=control)
    assert not report.valid
    assert any("path_continuation requires contact_path_program" in issue
               for issue in report.issues)
    # The refusal is new and opt-in only: the same ordinary force request
    # stays admissible with continuation off.
    ordinary_model, ordinary_definition, ordinary_settings = _case()
    assert validate_general_contact(
        ordinary_model, ordinary_definition, ordinary_settings,
        analysis_control="force").valid


def test_request_identity_default_shape_and_enabled_revision():
    model, definition, settings, program = _d1_case()
    default_identity = identify_general_contact_request(
        model, definition, settings, contact_path_program=program)
    request = default_identity.to_dict()["request"]
    for name in _CONTINUATION_FIELDS:
        assert name not in request["settings"]
    assert "contact_path_continuation_revision" not in request

    enabled = _enabled(settings)
    enabled_identity = identify_general_contact_request(
        model, definition, enabled, contact_path_program=program)
    enabled_request = enabled_identity.to_dict()["request"]
    assert (enabled_request["contact_path_continuation_revision"]
            == _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION)
    assert enabled_request["settings"]["path_continuation_maximum_halvings"] == 6
    verify_general_contact_request(enabled_identity, model, definition,
                                    enabled, contact_path_program=program)
    with pytest.raises(ContactRequestIdentityError) as caught:
        verify_general_contact_request(
            enabled_identity, model, definition,
            replace(enabled, path_continuation_maximum_halvings=4),
            contact_path_program=program)
    assert caught.value.code == "request_mismatch"


def test_dead_load_request_refuses_enabled_continuation_before_observation():
    model, definition, settings, _program = _d1_case()
    load = LoadCase("off-path dead load")
    load.add_nodal_load(1, forces=(0., 0., -0.01))
    with pytest.raises(ContactRequestIdentityError) as caught:
        identify_general_contact_request(
            model, definition, _enabled(settings), load_case=load, num_steps=2)
    assert caught.value.code == "unsupported_profile"
    assert "path_continuation requires contact_path_program" in str(caught.value)
    # The same ordinary dead-load request stays observable with the default
    # settings; only the enabled continuation is refused off-path.
    ordinary_model, ordinary_definition, ordinary_settings = _case()
    ordinary_load = LoadCase("ordinary dead load")
    ordinary_load.add_nodal_load(1, forces=(0., 0., -0.1))
    identify_general_contact_request(
        ordinary_model, ordinary_definition, ordinary_settings,
        load_case=ordinary_load)
