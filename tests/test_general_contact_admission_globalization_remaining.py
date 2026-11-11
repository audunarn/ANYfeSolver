"""Remaining admission/globalization caller/public checks plus one adversarial
physical-normal witness case.

Owner module for the v5 narrowed repair: the sixteen unaffected passed witness
math checks are NOT rerun here. This module imports only the nine remaining
caller/public test functions from the witness-public module (parameter markers
counted) after the capped-matching spy repair, and adds one genuinely
adversarial physical-normal case where the target-plane height exceeds the gap
tolerance but the actual physical source-normal gap does not, so a
target-plane-height criterion would falsely refuse and the physical witness
must not.
"""
import math
from fractions import Fraction

import numpy as np

from anyfesolver.general_contact.q4_overlap_local_bound import (
    overlap_violating_point_witness,
)
from test_general_contact_admission_globalization_witness_public import (
    test_contact_path_selection_admits_enabled_continuation,
    test_dead_load_request_refuses_enabled_continuation_before_observation,
    test_point_gap_budget_witness_skips_matching_without_admitting,
    test_real_public_validation_refuses_enabled_continuation_off_path,
    test_request_identity_default_shape_and_enabled_revision,
    test_settings_contract_omits_default_and_binds_enabled,
    test_swept_certify_witness_skips_matching_without_admitting,
)

_ADVERSARIAL_SOURCE = np.array([[0., 0., -.1], [1., 0., -.1],
                                [1., 1., .9], [0., 1., -.1]])
_ADVERSARIAL_TARGET = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])


def _adversarial_physical_gap_squared_rational():
    """Independent rational derivation of the squared physical gap.

    Bilinear source X(u,v) with corners A,B,C,D; at u=v=1/4 the exact
    derivatives are dX/du = (1,0,1/4) and dX/dv = (0,1,1/4), so the actual
    parametric normal is Ns = dX/du x dX/dv = (-1/4,-1/4,1) with
    |Ns|^2 = 9/8. The parent corner-0 normal is P = (0,0,1) and the target
    normal is T = (0,0,1), so the foot radius is r = N/(P.T) = -3/80 and the
    physical gap is g = r (P.Ns)/|Ns| with (P.Ns) = 1, giving
    g^2 = (9/6400)/(9/8) = 1/800. The target-plane height at the same point
    is -3/80 = -0.0375, which is more negative than g = -sqrt(1/800).
    """
    du = (Fraction(1), Fraction(0), Fraction(1, 4))
    dv = (Fraction(0), Fraction(1), Fraction(1, 4))
    ns = (du[1] * dv[2] - du[2] * dv[1],
          du[2] * dv[0] - du[0] * dv[2],
          du[0] * dv[1] - du[1] * dv[0])
    assert ns == (Fraction(-1, 4), Fraction(-1, 4), Fraction(1))
    ns_squared = sum(component * component for component in ns)
    assert ns_squared == Fraction(9, 8)
    radius_squared = Fraction(9, 6400)
    parent_actual = Fraction(1)
    return radius_squared * parent_actual * parent_actual / ns_squared


def test_witness_adversarial_physical_normal_within_tolerance_no_refusal():
    gap_squared = _adversarial_physical_gap_squared_rational()
    assert gap_squared == Fraction(1, 800)
    gap = -math.sqrt(1. / 800.)
    plane_height = -0.0375
    # The adversarial ordering: the target-plane height violates the 0.036
    # tolerance but the actual physical source-normal gap does not.
    assert plane_height < -0.036 < gap
    diagnostics = {}
    found, checked = overlap_violating_point_witness(
        _ADVERSARIAL_SOURCE, _ADVERSARIAL_SOURCE,
        _ADVERSARIAL_TARGET, _ADVERSARIAL_TARGET,
        gap_tolerance=.036,
        sample_params=((.25, .25),), sample_times=(1.,),
        diagnostics=diagnostics)
    assert found is False
    assert checked == 1
    assert diagnostics["found"] is False
    assert diagnostics["physical_gap_upper_m"] is None
    # The same exact sample with a tolerance below the physical gap magnitude
    # must refuse, proving the not-found result above is a genuine
    # within-tolerance decision on the physical gap and not a skipped chart.
    live = {}
    found, checked = overlap_violating_point_witness(
        _ADVERSARIAL_SOURCE, _ADVERSARIAL_SOURCE,
        _ADVERSARIAL_TARGET, _ADVERSARIAL_TARGET,
        gap_tolerance=.035,
        sample_params=((.25, .25),), sample_times=(1.,),
        diagnostics=live)
    assert found is True and checked == 1
    np.testing.assert_allclose(live["physical_gap_upper_m"], gap,
                               rtol=0., atol=1e-12)
    # The reported physical gap differs from the target-plane height: the
    # witness measures the actual normal quantity, never the plane height.
    assert abs(live["physical_gap_upper_m"] - plane_height) > 1e-4
    assert min(live["foot_barycentric_lower"]) >= 0.
    assert sum(live["foot_barycentric_upper"]) <= 1.
