"""Physical gap discrimination, independent scalar geometry, and query guards."""

import json
import runpy
import time
from pathlib import Path

import numpy as np
import pytest
from dataclasses import replace
from scipy.optimize import minimize_scalar
from anyfesolver.control import CancellationToken
from anyfesolver.general_contact.native_curve_union import resolved_physical_gap_lower

from test_general_contact_native_curved_pair_ownership import _physical_curves, _scalar_projection, _point


def _saved_fixture(shift=0.):
    root = Path(__file__).parents[1]
    saved = json.loads((root/"tests/data/general_contact_native_curved_pair_quadrature_pose_v1.json").read_text())
    builder = runpy.run_path(str(root/"examples/general_contact_native_retained_whole_pair.py"))["analysis"]
    owner = builder(physical_fibres=True, plastic_fibres=True, targets=(.05,),
                    line_stiffness_n_per_m2=4000., gap_step_control=True,
                    gap_tolerance_m=saved["gap_tolerance_m"],
                    reference_rise_m=.005, certify_curved_ownership=True)
    pose = saved["pose"]
    p, low, rotations = (np.array(pose[key]) for key in ("positions", "position_low", "cell_rotations"))
    p[3:, 1] += shift
    arguments = (p[:3], low[:3], rotations[0], p[3:], low[3:], rotations[1])
    return owner, arguments


def _independent_gap(owner, arguments):
    references = (owner.contact.cells_a[0].reference, owner.contact.cells_b[0].reference)
    curves = tuple(curve for ref, pose in zip(references, (arguments[:3], arguments[3:]))
                   for curve in _physical_curves(ref, pose))
    candidates = []
    # Complete stationary polynomial projection at each queried source station.
    # Bounded source minimization is an independent oracle, not the certificate.
    for source in curves[:2]:
        for target in curves[2:]:
            distance = lambda s: _scalar_projection(_point(source, s), target)[0]
            candidates.extend((distance(0.), distance(1.)))
            for left, right in zip(np.linspace(0., 1., 33)[:-1], np.linspace(0., 1., 33)[1:]):
                candidates.append(minimize_scalar(distance, bounds=(left, right), method="bounded",
                                                 options={"xatol": 1e-14}).fun)
    return min(candidates)-.1


def test_saved_curved_gap_discriminator():
    rows = []
    for shift in (0., .00036):
        owner, arguments = _saved_fixture(shift)
        start = time.perf_counter()
        lower = owner.contact.geometry_gap_lower(*arguments)
        elapsed = time.perf_counter()-start
        oracle = _independent_gap(owner, arguments)
        assert lower <= oracle+1e-12
        start = time.perf_counter()
        resolved = owner.contact.physical_gap_certificate(*arguments)
        cost = time.perf_counter()-start
        assert resolved.lower_gap_m <= oracle+1e-12
        assert oracle <= resolved.upper_gap_m+1e-12
        assert resolved.upper_gap_m-resolved.lower_gap_m <= owner.contact.distance_tolerance
        rows.append(dict(shift_m=shift, scan_lower_m=lower, independent_gap_m=oracle,
                         old_gap_query_s=elapsed, resolved_lower_m=resolved.lower_gap_m,
                         resolved_upper_m=resolved.upper_gap_m,
                         visited=resolved.visited_cells, resolved_query_s=cost))
    assert rows[0]["independent_gap_m"] < -50e-6
    assert rows[1]["independent_gap_m"] >= -50e-6
    # This saved pose already receives a tight chord lower. It demonstrates
    # real penetration, not falsely rejected clearance or large gap distortion.
    assert abs(rows[0]["scan_lower_m"]-rows[0]["independent_gap_m"]) < 2e-9
    print("curved_gap_discriminator", rows)


def test_physical_query_refuses_budget_precision_mutation_and_malformed_geometry():
    owner, arguments = _saved_fixture()
    searches = owner.contact._geometry_searches(arguments)
    searches[0].maximum_depth = 0
    with pytest.raises(MemoryError, match="depth budget"):
        resolved_physical_gap_lower(searches, 1e-10)
    searches = owner.contact._geometry_searches(arguments)
    searches[0].maximum_cells = 8
    with pytest.raises(MemoryError, match="cell budget"):
        resolved_physical_gap_lower(searches, 1e-10)
    with pytest.raises(MemoryError, match="evaluation floor"):
        resolved_physical_gap_lower(owner.contact._geometry_searches(arguments), 1e-16)
    with pytest.raises(ValueError):
        owner.contact.physical_gap_lower(np.full((3, 3), np.nan), *arguments[1:])
    owner.contact.maximum_search_depth += 1
    with pytest.raises(ValueError, match="definition changed"):
        owner.contact.physical_gap_lower(*arguments)


def _mechanical(owner, arguments):
    p, low, rotations = (part.copy() for part in (owner.initial.mechanical.positions,
                                                owner.initial.mechanical.position_low,
                                                owner.initial.mechanical.cell_rotations))
    for index, pose in zip(owner.indices, (arguments[:3], arguments[3:])):
        p[owner.context.nodes[index]] = pose[0]
        low[owner.context.nodes[index]] = pose[1]
        rotations[index] = pose[2]
    return replace(owner.initial.mechanical, positions=p, position_low=low, cell_rotations=rotations)


def test_endpoint_prefilter_and_swept_query_use_resolved_physical_gap(monkeypatch):
    owner, arguments = _saved_fixture()
    mechanical = _mechanical(owner, arguments)
    original_contact = owner._contact
    def forbidden(_):
        raise AssertionError("penetrated candidate entered full contact mechanics")
    monkeypatch.setattr(owner, "_contact", forbidden)
    result, gap = owner._endpoint_contact(mechanical)
    assert result is None and gap < -owner.gap_tolerance
    assert owner._swept_gap(mechanical) == gap
    # The accepted endpoint uses the same immutable pose and conservative
    # physical query; a coarse mechanical scan value cannot replace it.
    shifted = mechanical.positions.copy()
    shifted[owner.context.nodes[owner.indices[1]], 1] += .00036
    admitted = replace(mechanical, positions=shifted)
    calls = []
    monkeypatch.setattr(owner, "_contact", lambda pose: calls.append(pose) or owner.initial.contact)
    result, _ = owner._endpoint_contact(admitted)
    assert result is owner.initial.contact and calls == [admitted]
    monkeypatch.setattr(owner.contact, "evaluate", lambda *poses: replace(
        owner.initial.contact, conservative_lower_gap_m=-99.))
    result = original_contact(admitted)
    assert result.conservative_lower_gap_m >= -owner.gap_tolerance
    assert result.potential_j == owner.initial.contact.potential_j


def test_query_cancellation_and_postquery_mutation_refuse_consumption(monkeypatch):
    owner, arguments = _saved_fixture()
    mechanical = _mechanical(owner, arguments)
    actual = owner.contact.physical_gap_lower
    def mutate(*poses):
        value = actual(*poses)
        owner.contact.maximum_search_depth += 1
        return value
    monkeypatch.setattr(owner.contact, "physical_gap_lower", mutate)
    with pytest.raises(ValueError, match="definition"):
        owner._physical_gap(mechanical)
    owner, arguments = _saved_fixture()
    mechanical = _mechanical(owner, arguments)
    token = CancellationToken()
    with owner.context.operation(token):
        owner.context.activate_cancellation(token)
        token.cancel()
        with pytest.raises(Exception, match="cancel"):
            owner._physical_gap(mechanical)
    actual = owner.contact.physical_gap_lower
    token = CancellationToken()
    def cancel_after(*poses):
        value = actual(*poses)
        token.cancel()
        return value
    monkeypatch.setattr(owner.contact, "physical_gap_lower", cancel_after)
    with owner.context.operation(token):
        owner.context.activate_cancellation(token)
        with pytest.raises(Exception, match="cancel"):
            owner._physical_gap(mechanical)
