"""Actual-model discriminator: whole-face certificate feasibility first.

The sphere is the undeformed reference placed rigidly at the geometric first
touch (and at the existing 5 mm open clearance) above the accepted gravity-
preloaded panel. The existing whole-face certificate must establish at least
-50 micrometres without a new certificate or a weakened criterion.
"""
import json

import numpy as np

from anysolver.general_contact.corner_contact_certificate import (
    PENETRATION_LIMIT_M, first_touch_lift, whole_face_certificates,
)
from anysolver.general_contact.s3_q4_swept import s3_q4_affine_lower
from examples.elastic_sphere_panel_contact_work import prepare_work_case


def _affine_report(sphere_map, sphere_pose, panel_map, panel_pose):
    """Second existing certificate, same endpoint; reported, not substituted."""
    sp, pp = sphere_pose.positions, panel_pose.positions
    se, pe = sphere_pose.position_errors, panel_pose.position_errors
    values = []
    order = np.array((0, 2, 1))
    for i, (sa, sb) in enumerate(sphere_map.face_slices):
        s = sp[sa:sb][order]
        lo, hi = (sp[sa:sb] - se[sa:sb]).min(axis=0), (sp[sa:sb] + se[sa:sb]).max(axis=0)
        for j, (pa, pb) in enumerate(panel_map.face_slices):
            p_lo = (pp[pa:pb] - pe[pa:pb]).min(axis=0)
            p_hi = (pp[pa:pb] + pe[pa:pb]).max(axis=0)
            if lo[2] > p_hi[2] or np.any((lo[:2] > p_hi[:2]) | (hi[:2] < p_lo[:2])):
                continue
            t = pp[pa:pb]
            values.append(float(s3_q4_affine_lower(
                s, s, t, t, endpoint_errors=(se[sa:sb][order], se[sa:sb][order],
                                             pe[pa:pb], pe[pa:pb]))))
    return min(values) if values else 0., len(values)


def test_whole_face_certificate_feasible_at_open_and_first_touch(record_property):
    case = prepare_work_case()
    panel_pose = case.panel_map.pose(case.panel_q)
    zero = case.sphere_map.pose(np.zeros(case.sphere_map.n_dof))
    lift, worst_corners, worst_lifts = first_touch_lift(
        zero.positions, panel_pose.positions, case.panel_map.face_slices)
    report = dict(first_touch_lift_m=lift, lowest_corner_indices=worst_corners,
                  lowest_corner_lifts_m=worst_lifts, states={})
    for name, clearance in (("open_5mm", 5e-3), ("first_touch", 0.0)):
        q = np.zeros(case.sphere_map.n_dof)
        for node in case.sphere.mesh.nodes.values():
            q[node.dofs[2]] = lift + clearance
        pose = case.sphere_map.pose(q)
        cert = whole_face_certificates(case.sphere_map, pose, case.panel_map, panel_pose)
        affine, affine_pairs = _affine_report(case.sphere_map, pose, case.panel_map, panel_pose)
        report["states"][name] = dict(
            clearance_m=clearance, certificate=cert,
            s3_q4_affine_lower_minimum_m=affine, affine_pairs=affine_pairs)
    record_property("certificate_report", json.dumps(report, default=float))
    failures = {
        name: s for name, s in report["states"].items()
        if s["certificate"]["unavailable_pairs"]
        or s["certificate"]["minimum_bound_m"] < -PENETRATION_LIMIT_M}
    assert not failures, "whole-face certificate infeasible: " + json.dumps(report, default=float)


def _snapshot(sim):
    return (sim.q.copy(), sim.mu.copy(), dict(sim.lam), sim.target.copy(),
            sim.bridge.accepted().displacement.copy())


def _same(sim, snapshot):
    now = _snapshot(sim)
    return (np.array_equal(now[0], snapshot[0]) and np.array_equal(now[1], snapshot[1])
            and now[2] == snapshot[2] and np.array_equal(now[3], snapshot[3])
            and np.array_equal(now[4], snapshot[4]))


def test_actual_coupled_increment_and_rollback(record_property):
    """Open equilibrium (both gravities), touch, ONE genuinely coupled loaded increment,
    then exact rollback after cancellation, a rejected trial and foreign/nonfinite state."""
    import pytest
    from anysolver.general_contact.corner_contact_indentation import (
        CoupledIndentation, GateFailure,
    )
    from anysolver.general_contact.corner_contact_solver import Cancelled
    from anysolver.general_contact.structural_state_bridge import BridgeState

    import time
    started = time.perf_counter()
    case = prepare_work_case()
    prepared = time.perf_counter() - started

    holder = {}

    def stage(label):
        timing = holder["sim"].timing if "sim" in holder else {}
        print("[stage] %-30s t=%7.1f s timing=%s" % (
            label, time.perf_counter() - started, timing), flush=True)
    stage("gravity prepared")
    sim = holder["sim"] = CoupledIndentation(case)
    stage("simulator built")
    # Self-limiting: report measured timing instead of being killed silently.
    sim.cancel = lambda: time.perf_counter() - started > 440.0
    phase = "open"
    try:
        open_record = sim.prepare_open()
        stage("open accepted")
        open_deformation = sim.nonrigid_sphere_deformation_m()
        z_touch, corners, lifts = sim.first_touch_height()
        assert z_touch < sim.z_open
        phase = "touch"
        touch = sim.advance(z_touch)[-1]
        stage("touch accepted")
        phase = "loaded"
        loaded = sim.advance(z_touch - 0.25e-3)[-1]
        stage("loaded accepted (all gates)")
    except Cancelled:
        pytest.fail("deadline in phase %s after %.1f s (case preparation %.1f s): %s"
                    % (phase, time.perf_counter() - started, prepared, sim.timing))
    record_property("timing", json.dumps(dict(prepared_s=prepared, **sim.timing)))
    summary = dict(open={k: v for k, v in open_record.items() if k != "newton_trace"},
                   open_nonrigid_m=open_deformation, z_open=sim.z_open, z_touch=z_touch,
                   touch={k: v for k, v in touch.items() if k != "newton_trace"},
                   loaded={k: v for k, v in loaded.items() if k != "newton_trace"},
                   weights=sim.weight, f_ref=sim.f_ref, gram_condition=sim.gram_condition)
    record_property("coupled_report", json.dumps(summary, default=float))
    assert loaded["active_corners"] >= 1
    assert all(v > 0 for v in loaded["multipliers_N"].values())
    assert all(abs(g) <= 1e-12 for g in loaded["accepted_gaps_m"].values())
    assert loaded["minimum_candidate_gap_m"] >= -1e-12
    assert loaded["contact_resultant_N"][2] > 0
    assert loaded["whole_face_certificate"]["unavailable"] == 0
    assert loaded["whole_face_certificate"]["minimum_bound_m"] >= -50e-6
    assert loaded["nonrigid_sphere_deformation_m"] > open_deformation
    assert loaded["panel_vertical_change_from_gravity_m"] > 1e-9
    accepted = _snapshot(sim)
    # Cancellation between iterations leaves the accepted state exactly unchanged.
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] >= 2
    sim.cancel = cancel
    with pytest.raises(Cancelled):
        sim.solve_to(z_touch - 0.5e-3)
    sim.cancel = None
    assert _same(sim, accepted)
    # A rejected trial (iteration budget exhausted) also rolls back exactly.
    from anysolver.general_contact.corner_contact_solver import NewtonFailure
    sim.max_iterations = 0
    with pytest.raises(NewtonFailure):
        sim.solve_to(z_touch - 0.5e-3)
    sim.max_iterations = 20
    assert _same(sim, accepted)
    # A certificate gate that cannot be satisfied refuses the converged loaded result.
    sim.penetration_limit = -1e-9
    with pytest.raises(GateFailure):
        sim._gates(sim.last_result)
    sim.penetration_limit = 50e-6
    # Foreign and nonfinite states cannot be committed.
    good = sim.bridge.accepted()
    with pytest.raises(ValueError):
        sim.bridge.commit(BridgeState(object(), good.displacement, {}, {}))
    bad = sim.bridge.accepted()
    bad.displacement[0] = np.nan
    with pytest.raises(ValueError):
        sim.bridge.commit(bad)
    assert _same(sim, accepted)
    stage("rollback checks done")
