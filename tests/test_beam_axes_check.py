"""Default beam-axis switch report (spot check 2026-11-24, finding 2, owner option c)."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, solve_linear
from anyfesolver.beam_axes_check import (
    DEFAULT_AXIS_SWITCH_THRESHOLD,
    check_default_beam_axes,
    section_is_orientation_sensitive,
)
from anyfesolver.boundary import FixedSupport
from anyfesolver.elements import BeamElement, _beam_rotation_matrix

_W, _T = 0.2, 0.1
_STRIP = {"area": _W * _T, "Iy": _W * _T**3 / 12, "Iz": _T * _W**3 / 12, "J": 0.229 * _W * _T**3}
_ROUND = {"area": 0.02, "Iy": 3.0e-5, "Iz": 3.0e-5, "J": 6.0e-5}
_CURVED_OUT_OF_PLANE_REFERENCE = 0.5022  # MacNeal & Harder 1985


def _curved_beam(section, n=24):
    """MacNeal-Harder curved beam (R=4.12+0.1, 90 deg arc in the XY plane, tip load Z)."""
    model = FEModel("curved_beam")
    model.add_material("mat", 1.0e7, 0.25)
    for i in range(n + 1):
        s = np.pi / 2 * i / n
        model.add_node(i + 1, 4.22 * np.cos(s), 4.22 * np.sin(s), 0.0)
    for i in range(n):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(section)))
    model.add_boundary_condition(FixedSupport("root", [1]))
    return model


def _tip_out_of_plane(model, n=24):
    load_case = LoadCase("tip")
    load_case.add_nodal_load(n + 1, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    u, info = solve_linear(model, load_case)
    assert info["convergence_info"]["status"] == "converged"
    return np.asarray(u, dtype=float), float(u[model.mesh.nodes[n + 1].dofs][2])


def test_curved_beam_reports_the_switch_between_elements_5_and_6():
    # Element i (1-based) spans 3.75 deg; its chord is the arc tangent at the
    # midpoint, |e1.Y| = cos((i - 0.5) * 3.75 deg) > 0.95 for i <= 5 only.
    report = check_default_beam_axes(_curved_beam(_STRIP))
    assert report.has_findings
    assert dict(report.switch_nodes) == {6: (5, 6)}
    assert report.z_trial_elements == (1, 2, 3, 4, 5)
    assert report.oriented_elements == ()
    for record in report.records:
        assert record.orientation_sensitive
        expected = "global_Z" if record.alignment_with_global_y > DEFAULT_AXIS_SWITCH_THRESHOLD else "global_Y"
        assert record.trial_vector == expected
        local_z = _beam_rotation_matrix(np.asarray(record.axis), None)[:, 2]
        assert np.array_equal(np.asarray(record.local_z), local_z)
        # The switch is the 90 degree turn: Z-trial elements keep local z in the XY plane.
        assert (abs(local_z[2]) < 1e-12) == (record.trial_vector == "global_Z")


def test_orientation_clears_the_report_and_gives_the_reference_result():
    model = _curved_beam({**_STRIP, "orientation": [0.0, 0.0, 1.0]})
    report = check_default_beam_axes(model)
    assert not report.has_findings
    assert report.records == ()
    assert report.oriented_elements == tuple(range(1, 25))
    _u, tip = _tip_out_of_plane(model)
    assert tip == pytest.approx(_CURVED_OUT_OF_PLANE_REFERENCE, rel=0.03)


def test_symmetric_section_has_no_findings():
    report = check_default_beam_axes(_curved_beam(_ROUND))
    assert not report.has_findings
    assert report.z_trial_elements == ()
    assert all(not r.orientation_sensitive for r in report.records)


def test_straight_member_along_global_y_is_consistent_not_a_finding():
    model = FEModel("column")
    model.add_material("mat", 1.0e7, 0.25)
    for i in range(4):
        model.add_node(i + 1, 0.0, float(i), 0.0)
    for i in range(3):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(_STRIP)))
    report = check_default_beam_axes(model)
    assert not report.has_findings
    assert report.z_trial_elements == (1, 2, 3)


def test_check_is_read_only():
    model = _curved_beam(_STRIP)
    before, tip_before = _tip_out_of_plane(model)
    check_default_beam_axes(model)
    after, tip_after = _tip_out_of_plane(model)
    assert np.array_equal(before, after)
    # Characterisation of the unchanged default result (measured -21.7 %).
    assert -0.30 < tip_before / _CURVED_OUT_OF_PLANE_REFERENCE - 1.0 < -0.15
    assert tip_after == tip_before


@pytest.mark.parametrize(
    "section, sensitive",
    [
        (_STRIP, True),
        (_ROUND, False),
        ({"area": 0.01}, False),  # element defaults Iy = Iz
        ({**_ROUND, "shear_factor_y": 0.5}, True),
        ({**_ROUND, "c_y": 0.1}, True),
        ({**_ROUND, "c_y": 0.1, "c_z": 0.1}, False),
        ({"generalized_stiffness": np.eye(6)}, True),  # fails closed
    ],
)
def test_section_sensitivity(section, sensitive):
    got, reason = section_is_orientation_sensitive(section)
    assert got is sensitive, reason
