"""Independent geometry and work checks for opt-in curved pair ownership."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
import os
import json
from pathlib import Path

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver.general_contact.beam_curve_pair import _curve, closest_on_b3
from anyfesolver.general_contact.native_retained_whole_pair import (
    NativeRetainedWholeCurvePair, _CurvedOwnershipUnionSearch, _joined_owned_pieces,
    _active_gap_correspondence_upper,
    _exact_tiny_cell_omission_bound,
)
from anyfesolver.general_contact.native_curved_pair_partition import (
    curved_ownership_partition, physical_point_map, _near_junction_band,
)


def _reference(y, xs):
    return CenteredCurvedBeam3ReferenceGeometry(
        np.array([(x, y, 0.0) for x in xs]), np.tile(np.eye(3), (3, 1, 1)),
    )


def _pose(reference, angle=0.0):
    rotations = np.tile(np.eye(3), (2, 1, 1))
    rotations[0] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
    return [reference.coordinates.copy(), np.zeros((3, 3)), rotations]


def _kernel(a, b, **options):
    return NativeRetainedWholeCurvePair(
        11, a, (1, 2, 3), .05, 12, b, (4, 5, 6), .05,
        line_stiffness_n_per_m2=100.0, distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9, search_parameter_tolerance=1e-2,
        maximum_search_depth=20, maximum_search_cells=10000,
        energy_tolerance_j=1e-8, force_tolerance_n=1e-7,
        tangent_tolerance_n_per_m=1e-6, maximum_integration_intervals=2000,
        **options,
    )


def test_joined_pieces_preserve_roots_bands_support_and_length_budget():
    pieces = [(0., .1, 0, False), (.1, .2, 0, False),
              (.2, .3, 1, False), (.3, .4, 1, False),
              (.5, .6, 1, False), (.6, .7, 1, True)]
    joined = _joined_owned_pieces(pieces)
    assert joined == [(0., .2, 0, False), (.2, .4, 1, False),
                      (.5, .6, 1, False), (.6, .7, 1, True)]
    assert sum(b-a for a, b, _, _ in pieces) == pytest.approx(
        sum(b-a for a, b, _, _ in joined))


def test_complete_active_correspondence_rejects_unsampled_open_strip():
    target = np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.)))
    active = target+np.array((0., .099999, 0.))
    assert _active_gap_correspondence_upper(active, target, 0., 1., .1) < 0.
    # The strip 0<=s<~1e-4 is open; both Gauss8 and Gauss16 miss it.
    source = target.copy()
    source[:, 1] = (.10000001, .099950005, .0999)
    for count in (8, 16):
        stations = .5*(np.polynomial.legendre.leggauss(count)[0]+1.)
        assert all(.10000001-station*.00010001 < .1 for station in stations)
    assert _active_gap_correspondence_upper(source, target, 0., 1., .1) > 0.
    assert _active_gap_correspondence_upper(target+np.array((0., .1, 0.)),
                                            target, 0., 1., .1) > 0.


def test_exact_open_strip_has_zero_energy_work_and_tangent_bound():
    target = np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.)))
    source = target+np.array((0., .10000000001, 0.))
    bound = _exact_tiny_cell_omission_bound(source, target, source, .1, 4000.,
                                           .5-1e-13, .5+1e-13)
    np.testing.assert_array_equal(bound, np.zeros(3))


def test_any_rounded_open_jet_discards_partial_rule_and_preserves_analytic_work(monkeypatch):
    from dataclasses import replace
    import anyfesolver.general_contact.native_retained_whole_pair as product
    a, b = _reference(.09999, (0., 1., 2.24)), _reference(0., (0., 1.12, 2.24))
    kernel = _kernel(a, b, certify_open_intervals=True, certify_curved_ownership=True)
    actual = product.b3_distance_jet
    calls = [0]
    def rounded(*args, **kwargs):
        value = actual(*args, **kwargs)
        calls[0] += 1
        return replace(value, gap=0.) if calls[0] == 2 else value
    monkeypatch.setattr(product, "b3_distance_jet", rounded)
    result=kernel.evaluate(*_pose(a), *_pose(b))
    # The entire partial rule must be discarded and reevaluated/bounded;
    # omitting just the contradictory sample would lose a finite tangent.
    length=2.24;gap=.09999-.1;stiffness=100.
    assert result.potential_j==pytest.approx(.5*stiffness*length*gap**2,rel=1e-8)
    np.testing.assert_allclose(result.beam_a_force_n,(0.,-stiffness*length*gap,0.),
        rtol=1e-8,atol=1e-12)
    direction=np.zeros(84);direction[[1,7,13]]=1.
    assert direction@result.tangent@direction==pytest.approx(stiffness*length,rel=1e-8)
    assert calls[0]>2


@pytest.mark.skipif(not os.environ.get("CONTACT_CURVED_SAVED_POSE"),
                    reason="explicit bounded saved-pose check; no coupled replay")
def test_saved_curved_pose_quadrature_inventory(monkeypatch):
    import inspect
    import runpy
    import sys
    import time
    import anyfesolver.general_contact.native_retained_whole_pair as product
    saved = json.loads(Path(os.environ["CONTACT_CURVED_SAVED_POSE"]).read_text())
    builder = runpy.run_path(str(Path(__file__).parents[1]/"examples"
                                /"general_contact_native_retained_whole_pair.py"))["analysis"]
    owner = builder(physical_fibres=True, plastic_fibres=True, targets=(.05,),
                    line_stiffness_n_per_m2=saved["line_stiffness_n_per_m2"],
                    gap_step_control=True, reference_rise_m=saved["reference_rise_m"],
                    certify_curved_ownership=True)
    pose = saved["pose"]
    positions, low, rotations = (np.array(pose[key]) for key in
                                  ("positions", "position_low", "cell_rotations"))
    arguments = (positions[:3], low[:3], rotations[0],
                 positions[3:], low[3:], rotations[1])
    counts = dict(integration_calls=0, inactive_quadrature=0,
                  active_quadrature=0, error_refinement=0, scan={})
    lines, start = inspect.getsourcelines(product.NativeRetainedWholeCurvePair.evaluate)
    acceptance_line = start+next(i for i, line in enumerate(lines)
                                if "if (coarse_active or refined_active)" in line)
    original_scan = product._CurvedOwnershipUnionSearch.scan
    def scan(search):
        result = original_scan(search)
        for interval in result.intervals:
            counts["scan"][interval.status] = counts["scan"].get(interval.status, 0)+1
        return result
    monkeypatch.setattr(product._CurvedOwnershipUnionSearch, "scan", scan)
    def trace(frame, event, arg):
        if frame.f_code.co_name != "integrate":
            return None
        if event == "call":
            counts["integration_calls"] += 1
        elif event == "line" and frame.f_lineno == acceptance_line:
            active = frame.f_locals["coarse_active"] or frame.f_locals["refined_active"]
            counts["active_quadrature" if active else "inactive_quadrature"] += 1
            if np.any(frame.f_locals["measured"] > frame.f_locals["budget"]):
                counts["error_refinement"] += 1
        return trace
    previous = sys.gettrace()
    start_time = time.perf_counter()
    failure, result = None, None
    try:
        sys.settrace(trace)
        result = owner.contact.evaluate(*arguments)
    except (ValueError, MemoryError) as exc:
        failure = str(exc)
    finally:
        sys.settrace(previous)
    print("saved_curved_quadrature", dict(counts, elapsed_s=time.perf_counter()-start_time,
                                           failure=failure))
    if os.environ.get("CONTACT_CURVED_DIAGNOSTIC_ONLY") == "1":
        assert failure is not None
        return
    assert failure is None, failure
    a, b = owner.contact.cells_a[0].reference, owner.contact.cells_b[0].reference
    oracle = _scalar_energy(a, b, list(arguments[:3]), list(arguments[3:]),
                            split_contact=True)*40.0
    assert result.potential_j == pytest.approx(oracle, rel=1e-8, abs=1e-13)
    assert result.tangent_error_n_per_m <= 1e-6
    print("saved_curved_energy", dict(energy_j=result.potential_j, scalar_j=oracle,
                                      tangent_error=result.tangent_error_n_per_m,
                                      integration_intervals=result.integration_intervals,
                                      integration_evaluations=result.integration_evaluations))
    direction = np.zeros(84)
    direction[[43, 49, 55]] = .001  # rigid translation of body B
    step = 1e-4
    plus_b = _moved(list(arguments[3:]), direction[42:], step)
    minus_b = _moved(list(arguments[3:]), direction[42:], -step)
    plus = owner.contact.evaluate(*arguments[:3], *plus_b)
    minus = owner.contact.evaluate(*arguments[:3], *minus_b)
    e_plus = _scalar_energy(a, b, list(arguments[:3]), plus_b, split_contact=True)*40.0
    e_minus = _scalar_energy(a, b, list(arguments[:3]), minus_b, split_contact=True)*40.0
    scalar_work = (e_plus-e_minus)/(2*step)
    scalar_h = (e_plus-2*oracle+e_minus)/step**2
    native_h = direction@result.tangent@direction
    print("saved_curved_work", dict(step=step, scalar_work=scalar_work,
                                    native_work=result.residual@direction,
                                    scalar_h=scalar_h, native_h=native_h))
    assert result.residual@direction == pytest.approx(scalar_work, rel=1e-6, abs=1e-12)
    assert native_h == pytest.approx(scalar_h, rel=1e-5, abs=1e-10)
    assert direction@((plus.residual-minus.residual)/(2*step)) == pytest.approx(
        native_h, rel=1e-5, abs=1e-10)


@pytest.mark.parametrize("slope, plateau", [(1e-10, False), (-1e-3, True)])
def test_junction_band_complete_inventory_and_error_budget(slope, plateau):
    """Independent linear endpoint polynomials enumerate both band topologies."""
    source = np.array(((0., .1, 0.), (1., .1, 0.), (2., .1, 0.)))
    targets = (np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.))),
               np.array(((1., 0., 0.), (1.5, .5*slope, 0.), (2., slope, 0.))))
    search = _CurvedOwnershipUnionSearch(
        source, targets, ((1, 2), (2, 3)), combined_radius_m=.11,
        distance_tolerance_m=1e-10, feature_tolerance_m=1e-9,
        parameter_tolerance=.01, maximum_depth=20, maximum_cells=10000,
    )
    budget = np.array((1e-8, 1e-7, 1e-6))
    result = _near_junction_band(search, .4, .6, (2., 2.), source, 100., budget)
    assert result is not None
    pieces, bands = result
    spans = sorted([(a, b) for a, b, _ in pieces]
                   +[(a, b) for a, b, _, _ in bands])
    assert spans[0][0] == .4 and spans[-1][1] == .6
    assert all(first[1] == second[0] for first, second in zip(spans[:-1], spans[1:]))
    # phiL=1-2s; phiR=1-2s-.1*slope. These are derived from
    # linear endpoint geometry, independently of production curve/projection.
    roots = (.5, .5-.05*slope)
    assert all(any(a <= root <= b for a, b, _, _ in bands) for root in roots)
    if plateau:
        assert len(bands) == 2 and all(root_h == 0. for _, _, _, root_h in bands)
        assert pieces[1][1]-pieces[1][0] > 1e-5  # wide plateau stays integrated
    else:
        assert len(bands) == 1 and bands[0][3] > 0.
    for begin, stop, owner in pieces:
        station = .5*(begin+stop)
        # Finite straight segment projection, independently computed.
        distances = []
        point = np.array((2.*station, .1, 0.))
        for target in targets:
            tangent = target[2]-target[0]
            t = np.clip((point-target[0])@tangent/(tangent@tangent), 0., 1.)
            distances.append(np.linalg.norm(point-target[0]-t*tangent))
        assert distances[owner] <= min(distances)+1e-14
    assert _near_junction_band(search, .4, .6, (2., 2.), source, 100.,
                               np.zeros(3)) is None
    print("junction_band", {"plateau": plateau, "bands": [
        {"interval": (a, b), "bulk_bound": bulk.tolist(), "root_h_bound": root_h}
        for a, b, bulk, root_h in bands]})


def _polynomial(points):
    return (points[0], -3*points[0]+4*points[1]-points[2],
            2*points[0]-4*points[1]+2*points[2])


def _point(points, station):
    constant, linear, quadratic = _polynomial(points)
    return constant+station*(linear+station*quadratic)


def _scalar_projection(point, target):
    """Independent polynomial stationarity enumeration, no product projector."""
    constant, linear, quadratic = _polynomial(target)
    relative = constant-point
    coefficients = np.array((relative@linear,
                             linear@linear+2*relative@quadratic,
                             3*linear@quadratic, 2*quadratic@quadratic))
    nonzero = np.flatnonzero(coefficients)
    roots = (np.roots(coefficients[:nonzero[-1]+1][::-1])
             if len(nonzero) else ())
    stations = [0.0, 1.0]
    stations.extend(float(root.real) for root in roots
                    if abs(root.imag) < 1e-12 and 0.0 < root.real < 1.0)
    distance, station = min((float(np.linalg.norm(point-_point(target, t))), t)
                            for t in stations)
    return distance, station


def _physical_curves(reference, pose):
    # Independently reconstruct the adopted rotated analytic half-cell lift.
    lift = -(reference.coordinates[0]-2*reference.coordinates[1]
             +reference.coordinates[2])/8.0
    points, low, rotations = pose
    endpoints = points+low
    return tuple(np.array((endpoints[cell],
                           .5*(endpoints[cell]+endpoints[cell+1])+rotations[cell]@lift,
                           endpoints[cell+1])) for cell in (0, 1))


def _scalar_energy(a, b, pose_a, pose_b, *, split_contact=False):
    current = (*_physical_curves(a, pose_a), *_physical_curves(b, pose_b))
    reference = (*_physical_curves(a, _pose(a)), *_physical_curves(b, _pose(b)))
    value = 0.0
    for source_base, target_base in ((0, 2), (2, 0)):
        for cell in (0, 1):
            source, ref = current[source_base+cell], reference[source_base+cell]
            _, first, second = _polynomial(ref)

            def density(station):
                point = _point(source, station)
                distance = min(_scalar_projection(point, current[target_base+j])[0]
                               for j in (0, 1))
                gap = min(distance-.1, 0.0)
                return 25.0*np.linalg.norm(first+2*station*second)*gap*gap

            cuts = []
            if split_contact:
                # Independent quadratic endpoint KKT polynomials enumerate
                # projection-regime boundaries; no production partition calls.
                constant, linear, quadratic = _polynomial(source)
                for target in current[target_base:target_base+2]:
                    origin, velocity, acceleration = _polynomial(target)
                    for t in (0., 1.):
                        tangent = velocity+2*t*acceleration
                        endpoint = origin+t*(velocity+t*acceleration)
                        coefficients = np.array(((endpoint-constant)@tangent,
                                                  -linear@tangent, -quadratic@tangent))
                        nonzero = np.flatnonzero(coefficients)
                        roots = np.roots(coefficients[:nonzero[-1]+1][::-1])
                        cuts.extend(float(root.real) for root in roots
                                    if abs(root.imag) < 1e-12 and 0. < root.real < 1.)
                def gap(station):
                    point = _point(source, station)
                    return min(_scalar_projection(point, target)[0]
                               for target in current[target_base:target_base+2])-.1
                # Numerical oracle cuts, not a support completeness certificate.
                stations = sorted(set((0., 1., *cuts, *np.linspace(0., 1., 129))))
                for first_s, last_s in zip(stations[:-1], stations[1:]):
                    if gap(first_s)*gap(last_s) < 0.:
                        cuts.append(brentq(gap, first_s, last_s, xtol=1e-14, rtol=1e-14))
            value += quad(density, 0.0, 1.0, points=sorted(set(cuts)),
                          epsabs=1e-19, epsrel=1e-10, limit=400)[0]
    return value


def _small_fixture():
    a = _reference(.09999, (0.0, 1.0, 2.24))
    b = _reference(0.0, (0.0, 1.12, 2.24))
    return a, b, _pose(a, .05), _pose(b)


def _moved(pose, direction, step):
    result = [part.copy() for part in pose]
    for node in range(3):
        result[0][node] += step*direction[6*node:6*node+3]
    for cell in (0, 1):
        result[2][cell] = Rotation.from_rotvec(
            step*direction[18+3*cell:21+3*cell],
        ).as_matrix() @ result[2][cell]
    return result


def test_archived_curved_switch_has_distinct_remote_minima():
    a = _reference(.08, (0.0, 1.0, 2.24))
    b = _reference(0.0, (0.0, 1.12, 2.24))
    kernel = _kernel(a, b)
    targets = tuple(cell.evaluate(*_pose(a, 30.0)).positions
                    for cell in kernel.cells_a)

    def distances(x):
        point = np.array((x, 0.0, 0.0))
        return tuple(closest_on_b3(point, target) for target in targets)

    root = brentq(lambda x: distances(x)[0].distance-distances(x)[1].distance,
                  1.0, 1.005, xtol=1e-14, rtol=1e-14)
    projections = distances(root)
    points = tuple(_curve(target, projection.minimizers[0])[0]
                   for target, projection in zip(targets, projections))
    assert all(projection.unique for projection in projections)
    assert 0.0 < projections[0].minimizers[0] < 1.0
    assert 0.0 < projections[1].minimizers[0] < 1.0
    assert np.linalg.norm(points[0]-points[1]) > 1e-3
    assert projections[0].distance == pytest.approx(projections[1].distance,
                                                   abs=1e-13)
    assert all(np.linalg.norm(point-targets[0][2]) > 1e-3 for point in points)
    independent = tuple(_scalar_projection(np.array((root, 0.0, 0.0)), target)
                        for target in targets)
    assert independent[0][0] == pytest.approx(independent[1][0], abs=1e-13)
    assert all(abs(item[1]-projection.minimizers[0]) < 1e-12
               for item, projection in zip(independent, projections))
    with pytest.raises(ValueError, match="target ownership is unresolved"):
        kernel.evaluate(*_pose(a, 30.0), *_pose(b))
    print("archived_remote_switch", dict(
        source_x=root,
        target_stations=[p.minimizers[0] for p in projections],
        target_separation_m=float(np.linalg.norm(points[0]-points[1])),
        distance_m=projections[0].distance,
    ))


def test_small_curved_remote_switch_energy_work_and_directional_tangent():
    a, b, pose_a, pose_b = _small_fixture()
    kernel = _kernel(a, b, certify_open_intervals=True,
                     certify_curved_ownership=True)
    result = kernel.evaluate(*pose_a, *pose_b)
    # All A points are above this exact Q2 ordinate minimum; B is straight
    # at y=0. This bounds the complete pair, including the reverse direction.
    true_gap = .09999-.03*np.sin(np.deg2rad(.05))-.1
    assert -50e-6 < true_gap < 0.0
    assert result.conservative_lower_gap_m <= true_gap
    oracle = _scalar_energy(a, b, pose_a, pose_b)
    assert result.potential_j == pytest.approx(oracle, rel=1e-8, abs=1e-16)
    direction = np.zeros(84)
    direction[[1, 7, 13]] = .001
    direction[[43, 49, 55]] = -.0005
    direction[20] = .001
    step = 1e-4
    plus_a, minus_a = (_moved(pose_a, direction[:42], value)
                       for value in (step, -step))
    plus_b, minus_b = (_moved(pose_b, direction[42:], value)
                       for value in (step, -step))
    plus = kernel.evaluate(*plus_a, *plus_b)
    minus = kernel.evaluate(*minus_a, *minus_b)
    scalar_plus = _scalar_energy(a, b, plus_a, plus_b)
    scalar_minus = _scalar_energy(a, b, minus_a, minus_b)
    work = (scalar_plus-scalar_minus)/(2*step)
    stiffness = direction@result.tangent@direction
    scalar_stiffness = (scalar_plus-2*oracle+scalar_minus)/step**2
    assert work == pytest.approx(result.residual@direction, rel=1e-6, abs=1e-12)
    assert scalar_stiffness == pytest.approx(stiffness, rel=1e-5, abs=1e-10)
    assert direction@((plus.residual-minus.residual)/(2*step)) == pytest.approx(
        stiffness, rel=1e-5, abs=1e-10,
    )
    assert (scalar_plus-oracle)/step == pytest.approx(work, rel=1e-2, abs=1e-12)
    assert (oracle-scalar_minus)/step == pytest.approx(work, rel=1e-2, abs=1e-12)
    assert result.tangent_error_n_per_m <= 1e-6
    assert np.linalg.norm(result.tangent[:42, 42:]) > 0.0
    print("small_curved_work", dict(true_gap_m=true_gap, energy_j=oracle,
                                    virtual_work=work, directional_tangent=stiffness,
                                    scalar_tangent=scalar_stiffness,
                                    tangent_error_bound=result.tangent_error_n_per_m))


def test_small_curved_wrench_and_rigid_transform():
    a, b, pose_a, pose_b = _small_fixture()
    kernel = _kernel(a, b, certify_open_intervals=True,
                     certify_curved_ownership=True)
    result = kernel.evaluate(*pose_a, *pose_b)
    np.testing.assert_allclose(result.beam_a_force_n+result.beam_b_force_n, 0.0,
                               atol=1e-10)
    np.testing.assert_allclose(result.beam_a_moment_nm+result.beam_b_moment_nm, 0.0,
                               atol=1e-10)
    turn = Rotation.from_rotvec((.2, -.1, .15)).as_matrix()
    shift = np.array((.7, -.4, .2))
    transformed = [
        [pose[0]@turn.T+shift, pose[1]@turn.T,
         np.array([turn@rotation for rotation in pose[2]])]
        for pose in (pose_a, pose_b)
    ]
    moved = kernel.evaluate(*transformed[0], *transformed[1])
    assert moved.potential_j == pytest.approx(result.potential_j, rel=1e-8, abs=1e-16)
    np.testing.assert_allclose(moved.beam_a_force_n, turn@result.beam_a_force_n,
                               rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(moved.beam_b_force_n, turn@result.beam_b_force_n,
                               rtol=1e-7, atol=1e-10)


def test_partition_covers_unique_interior_and_shared_junction_owners():
    a = _reference(.09999, (0.0, 1.0, 2.24))
    b = _reference(0.0, (0.0, 1.12, 2.24))
    pose_a = _pose(a, -.05)
    targets = _physical_curves(a, pose_a)
    source = _physical_curves(b, _pose(b))[0]
    search = _CurvedOwnershipUnionSearch(
        source, targets, ((1, 2), (2, 3)), combined_radius_m=.1,
        distance_tolerance_m=1e-10, feature_tolerance_m=1e-9,
        parameter_tolerance=.01, maximum_depth=20, maximum_cells=10000,
    )
    partition = curved_ownership_partition(search, 0.0, 1.0, remaining_cells=10000)
    assert not partition.switches  # an actual common-endpoint region
    assert partition.pieces[0][0] == 0.0 and partition.pieces[-1][1] == 1.0
    assert all(left[1] == right[0]
               for left, right in zip(partition.pieces[:-1], partition.pieces[1:]))
    for station in np.linspace(0.0, 1.0, 129):
        piece = next(item for item in partition.pieces if item[0] <= station <= item[1])
        distances = tuple(_scalar_projection(_point(source, station), target)[0]
                          for target in targets)
        assert distances[piece[2]] <= min(distances)+1e-12


def test_persistent_remote_tie_and_endpoint_switch_fail_closed():
    targets = (np.array(((-1., .2, 0.), (-.5, .6, 0.), (0., 1., 0.))),
               np.array(((0., 1., 0.), (.5, .6, 0.), (1., .2, 0.))))
    for source in (np.array(((0., -.5, 0.), (0., -.25, 0.), (0., 0., 0.))),
                   np.array(((0., 0., 0.), (.1, 0., 0.), (.2, 0., 0.)))):
        search = _CurvedOwnershipUnionSearch(
            source, targets, ((1, 2), (2, 3)), combined_radius_m=2.0,
            distance_tolerance_m=1e-10, feature_tolerance_m=1e-9,
            parameter_tolerance=.01, maximum_depth=4, maximum_cells=1000,
        )
        with pytest.raises(MemoryError, match="depth budget|boundary unresolved"):
            curved_ownership_partition(search, 0.0, 1.0, remaining_cells=1000)


def test_opt_in_identity_and_definition_guard_preserve_default():
    a, b, pose_a, pose_b = _small_fixture()
    default = _kernel(a, b)
    explicit_default = _kernel(a, b, certify_curved_ownership=False)
    enabled = _kernel(a, b, certify_curved_ownership=True)
    assert default.identity == explicit_default.identity
    assert enabled.identity != default.identity
    default.certify_curved_ownership = True
    with pytest.raises(ValueError, match="definition changed"):
        default.evaluate(*pose_a, *pose_b)
    with pytest.raises(ValueError, match="curved ownership selection must be bool"):
        _kernel(a, b, certify_curved_ownership=1)


def test_physical_length_chart_removes_only_material_endpoint_duplicates():
    point_map = physical_point_map()
    assert point_map.shape == (36, 30)
    assert np.max(np.sum(np.abs(point_map), axis=0)) == 2.0
    assert np.linalg.norm(point_map, 2) == pytest.approx(np.sqrt(2.0))
    shared_endpoint_jump = np.zeros(36)
    shared_endpoint_jump[6:9] = (1., 2., 3.)
    shared_endpoint_jump[9:12] = (-1., -2., -3.)
    np.testing.assert_array_equal(point_map.T @ shared_endpoint_jump, 0.0)
    a, b, pose_a, pose_b = _small_fixture()
    kernel = _kernel(a, b)
    states = [cell.evaluate(*pose) for cells, pose in
              ((kernel.cells_a, pose_a), (kernel.cells_b, pose_b)) for cell in cells]
    positions = np.concatenate([state.positions.ravel() for state in states])
    native_map = np.zeros((36, 84))
    for index, state in enumerate(states):
        native_map[9*index:9*index+9, 42*(index//2):42*(index//2)+42] = state.geometric_jacobian
    independent_rows = np.ravel([np.arange(3*row, 3*row+3)
                                for row in (0, 1, 2, 4, 5, 6, 7, 8, 10, 11)])
    np.testing.assert_array_equal(point_map @ positions[independent_rows], positions)
    np.testing.assert_array_equal(point_map @ native_map[independent_rows], native_map)
    gradient = np.arange(36, dtype=float)/36.0
    direction = np.arange(84, dtype=float)/84.0
    assert gradient@native_map@direction == pytest.approx(
        (point_map.T@gradient)@native_map[independent_rows]@direction,
    )
