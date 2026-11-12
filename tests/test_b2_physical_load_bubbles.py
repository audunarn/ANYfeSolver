"""Independent local controls for experimental fixed-end B2 load bubbles."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anyfesolver._b2_physical_flexibility_linear import (
    LinearOperator, LinearSection, RepresentabilityError,
)
from anyfesolver._b2_physical_load_bubbles import FixedEndLoadBubbleMaps


def _section(coupled=False):
    diagonal = np.diag((17., 8., 9., 3., 4., 5.))
    transform = np.eye(6)
    if coupled:
        transform[0, 4], transform[1, 5], transform[2, 3] = .2, -.3, .15
    return LinearSection.generalized(transform.T@diagonal@transform)


def test_loaded_midpoint_elastic_carrier_requires_compliance_kkt():
    # An axial member with both ends fixed and an elastic carrier at L/2.
    # Independent continuum solution: b=pL²/(8EA), G=L/(4EA),
    # r=b/(1+kG), lambda=-k*r. The homogeneous trace alone predicts r=0.
    length, section, load, carrier_k = 2., _section(), 1.3, 9.
    operator = LinearOperator(length, section)
    xi, wi = np.polynomial.legendre.leggauss(6)
    stations = np.r_[.25*(xi+1.), .5+.25*(xi+1.), .5]
    maps = FixedEndLoadBubbleMaps(operator, stations=stations, point_stations=(.5,))
    uniform = np.array((load,0.,0.,0.,0.,0.))
    body = maps.evaluate(uniform)
    bubble = body.pose[-1,0]
    green = maps.point_compliance[0,0]
    displacement, multiplier = np.linalg.solve(
        np.array(((carrier_k,1.),(1.,-green))), np.array((0.,bubble)))
    ea = section.stiffness[0][0]
    expected = (load*length**2/(8*ea))/(1.+carrier_k*length/(4*ea))
    assert displacement == pytest.approx(expected, rel=2e-13)
    assert multiplier == pytest.approx(-carrier_k*expected, rel=2e-13)
    loaded = maps.evaluate(uniform,point_wrenches=((multiplier,0.,0.,0.,0.,0.),))
    assert loaded.pose[-1,0] == pytest.approx(displacement, rel=2e-13)
    assert operator.trace(np.zeros(12),stations=(.5,)).pose[0,0] == 0.
    assert displacement > 0.
    # Global body work balances beam strain work plus carrier elastic work;
    # interface multiplier work cancels, rather than becoming an external load.
    weights = np.r_[length*wi/4., length*wi/4.]
    _,strain = _independent_bvp(section,length,uniform,(.5,),((multiplier,0.,0.,0.,0.,0.),))
    eps = strain(stations[:-1])
    beam_work = weights@np.einsum('ni,ij,nj->n',eps,section.stiffness,eps)
    body_work = weights@(loaded.pose[:-1]@uniform)
    assert body_work == pytest.approx(beam_work+carrier_k*displacement**2, rel=2e-13)


def _quadrature(length, point_stations, stop=1.):
    knots = [0.] + sorted({float(s) for s in point_stations if 0. < s < stop}) + [stop]
    nodes, weights = np.polynomial.legendre.leggauss(6)
    places, measure = [], []
    for left, right in zip(knots, knots[1:]):
        places.extend(left+(right-left)*(nodes+1.)/2.)
        measure.extend(length*(right-left)*weights/2.)
    return np.asarray(places), np.asarray(measure)


def _independent_bvp(section, length, uniform, point_stations, point_wrenches,
                     point_offsets=None):
    """Initial-resultant BVP with segmented quadrature, no candidate trace/maps."""
    compliance = np.linalg.inv(np.asarray(section.stiffness))
    uniform = np.asarray(uniform)
    offsets = np.zeros((len(point_stations), 3)) if point_offsets is None else np.asarray(point_offsets)
    actions = np.asarray(point_wrenches, dtype=float).reshape(-1, 6).copy()
    if len(actions):
        actions[:, 3:] += np.cross(offsets, actions[:, :3])

    def field(initial, normalized):
        x = length*np.asarray(normalized)
        force = initial[:3]-x[:, None]*uniform[:3]
        moment = (initial[3:]-x[:, None]*np.cross((1., 0., 0.), initial[:3])
                  -x[:, None]*uniform[3:]+.5*x[:, None]**2*np.cross((1., 0., 0.), uniform[:3]))
        for station, action in zip(point_stations, actions):
            crossed = x > length*station
            force[crossed] -= action[:3]
            moment[crossed] += (-action[3:] + (x[crossed]-length*station)[:, None]
                                *np.cross((1., 0., 0.), action[:3]))
        return np.column_stack((force, moment))@compliance.T

    def pose(initial, station):
        places, weights = _quadrature(length, point_stations, station)
        strain = field(initial, places)
        ordinary = weights@strain
        weighted = (weights*(length*station-length*places))@strain
        return np.r_[ordinary[0], ordinary[1]+weighted[5], ordinary[2]-weighted[4], ordinary[3:]]

    base = pose(np.zeros(6), 1.)
    boundary = np.column_stack([pose(np.eye(6)[i], 1.)-base for i in range(6)])
    initial = np.linalg.solve(boundary, -base)
    return lambda station: pose(initial, station), lambda places: field(initial, places)


@pytest.mark.parametrize("channel", (0, 1, 2, 3))
def test_uniform_fixed_end_axial_timoshenko_and_torsion_controls(channel):
    length = 2.
    section = LinearSection.isotropic(E=200., nu=.25, area=.1, Iy=.02, Iz=.03,
                                      J=.04, ky=.7, kz=.8)
    operator = LinearOperator(length, section)
    stations = np.linspace(0., 1., 9)
    wrench = np.zeros(6); wrench[channel] = 1.3
    response = FixedEndLoadBubbleMaps(operator, stations=stations).evaluate(wrench)
    x = length*stations
    expected = np.zeros((len(x), 6))
    stiffness = np.asarray(section.stiffness)
    if channel in (0, 3):
        expected[:, channel] = wrench[channel]*x*(length-x)/(2.*stiffness[channel, channel])
    else:
        bend = 5 if channel == 1 else 4
        expected[:, channel] = (wrench[channel]*x**2*(length-x)**2/(24.*stiffness[bend, bend])
            + wrench[channel]*x*(length-x)/(2.*stiffness[channel, channel]))
        expected[:, bend] = ((1. if channel == 1 else -1.)*wrench[channel]
            *x*(length-x)*(length-2.*x)/(12.*stiffness[bend, bend]))
    np.testing.assert_allclose(response.pose, expected, rtol=3e-13, atol=3e-15)
    np.testing.assert_array_equal(response.pose[[0, -1]], 0.)
    assert np.any(response.pose[1:-1] != 0.)
    np.testing.assert_array_equal(operator.trace(np.zeros(12), stations=stations).pose, 0.)


@pytest.mark.parametrize("coupled", (False, True))
def test_combined_loads_match_independent_piecewise_boundary_value(coupled):
    length, section = 1.7, _section(coupled)
    sources = (.35, .6)
    actions = np.array(((.8, -.2, -.7, .1, .5, -.3), (-.2, .3, .6, -.4, .2, .1)))
    offsets = np.array(((.04, -.03, .02), (-.02, .01, .03)))
    uniform = np.array((.6, -.4, .3, -.5, .7, .8))
    stations = (0., .12, .35, .51, .6, .83, 1.)
    observation_offsets = np.tile((.03, -.04, .02), (len(stations), 1))
    maps = FixedEndLoadBubbleMaps(LinearOperator(length, section), stations=stations,
        point_stations=sources, observation_offsets_local_m=observation_offsets,
        point_offsets_local_m=offsets)
    response = maps.evaluate(uniform, point_wrenches=actions)
    reference, _ = _independent_bvp(section, length, uniform, sources, actions, offsets)
    expected = np.array([reference(station) for station in stations])
    expected[:, :3] += np.cross(expected[:, 3:], observation_offsets)
    np.testing.assert_allclose(response.pose, expected, rtol=2e-12, atol=3e-14)
    np.testing.assert_array_equal(response.pose[[0, -1]], 0.)
    assert maps.production_qualified is response.production_qualified is False


def test_concentrated_axial_green_and_midpoint_timoshenko_compliance():
    length, section = 2., _section()
    stations = (0., .2, .5, .8, 1.)
    maps = FixedEndLoadBubbleMaps(LinearOperator(length, section), stations=stations,
                                point_stations=(.5,))
    axial = maps.evaluate(point_wrenches=((1., 0., 0., 0., 0., 0.),))
    expected = length*np.minimum(stations, .5)*(1.-np.maximum(stations, .5))/17.
    np.testing.assert_allclose(axial.pose[:, 0], expected, rtol=2e-14, atol=0.)
    np.testing.assert_array_equal(axial.pose[:, 1:], 0.)
    expected_midpoint = length**3/(192.*5.)+length/(4.*8.)
    assert maps.point_compliance[1, 1] == pytest.approx(expected_midpoint, rel=2e-14)


def test_point_pose_is_continuous_and_endpoint_point_loads_have_no_bubble():
    station = .375
    nearby = (np.nextafter(station, 0.), station, np.nextafter(station, 1.))
    operator = LinearOperator(2., _section(True))
    maps = FixedEndLoadBubbleMaps(operator, stations=nearby, point_stations=(station,))
    action = ((.4, -.3, .6, -.5, .2, .7),)
    pose = maps.evaluate(point_wrenches=action).pose
    np.testing.assert_allclose(pose, np.tile(pose[1], (3, 1)), rtol=3e-14, atol=3e-15)
    endpoints = FixedEndLoadBubbleMaps(operator, stations=(0., .2, .5, .8, 1.),
        point_stations=(0., 1.), point_offsets_local_m=((.02, -.03, .04), (-.01, .05, .03)))
    response = endpoints.evaluate(point_wrenches=(action[0], action[0]))
    np.testing.assert_array_equal(response.pose, 0.)
    np.testing.assert_array_equal(endpoints.point_compliance, 0.)
    assert np.any(response.equivalent_nodal_load != 0.)


def test_coupled_offset_green_reciprocity_positive_work_and_strain_energy():
    length, section = 1.7, _section(True)
    sources = (.2, .65)
    offsets = np.array(((.04, -.03, .02), (-.02, .01, .03)))
    actions = np.array(((.8, -.2, -.7, .1, .5, -.3), (-.2, .3, .6, -.4, .2, .1)))
    uniform = np.array((.6, -.4, .3, -.5, .7, .8))
    places, weights = _quadrature(length, sources)
    stations = tuple(places)+sources
    observations = np.vstack((np.zeros((len(places), 3)), offsets))
    maps = FixedEndLoadBubbleMaps(LinearOperator(length, section), stations=stations,
        point_stations=sources, observation_offsets_local_m=observations,
        point_offsets_local_m=offsets)
    np.testing.assert_array_equal(maps.point_compliance, maps.point_compliance.T)
    assert np.all(np.linalg.eigvalsh(maps.point_compliance) > 0.)
    assert actions.ravel()@maps.point_compliance@actions.ravel() > 0.
    response = maps.evaluate(uniform, point_wrenches=actions)
    _, strains = _independent_bvp(section, length, uniform, sources, actions, offsets)
    strain = strains(places)
    elastic_work = weights@np.einsum("ni,ij,nj->n", strain, section.stiffness, strain)
    external_work = (weights@(response.pose[:len(places)]@uniform)
                     + np.sum(actions*response.pose[len(places):]))
    assert external_work == pytest.approx(elastic_work, rel=2e-12, abs=1e-14)
    body = maps.evaluate(uniform)
    point = maps.evaluate(point_wrenches=actions)
    assert np.sum(actions*body.pose[len(places):]) == pytest.approx(
        weights@(point.pose[:len(places)]@uniform), rel=2e-12, abs=1e-14)


def test_equivalent_nodal_load_preserves_offset_work_force_and_moment():
    length = 1.7
    operator = LinearOperator(length, _section(True))
    sources = (.2, .65)
    offsets = np.array(((.04, -.03, .02), (-.02, .01, .03)))
    actions = np.array(((.8, -.2, -.7, .1, .5, -.3), (-.2, .3, .6, -.4, .2, .1)))
    uniform = np.array((.6, -.4, .3, -.5, .7, .8))
    maps = FixedEndLoadBubbleMaps(operator, point_stations=sources, point_offsets_local_m=offsets)
    response = maps.evaluate(uniform, point_wrenches=actions)
    nodal = np.linspace(-.03, .02, 12)
    places, weights = _quadrature(length, sources)
    trace = operator.trace(nodal, stations=places)
    work = weights@(trace.pose@uniform)
    for station, offset, action in zip(sources, offsets, actions):
        work += operator.trace(nodal, stations=(station,), offset_local_m=offset).pose[0]@action
    assert response.equivalent_nodal_load@nodal == pytest.approx(work, rel=2e-13, abs=1e-15)
    end = response.equivalent_nodal_load.reshape(2, 6)
    np.testing.assert_allclose(end[:, :3].sum(axis=0), length*uniform[:3]+actions[:, :3].sum(axis=0),
                               rtol=2e-13, atol=1e-14)
    moment = end[:, 3:].sum(axis=0)+np.cross((length, 0., 0.), end[1, :3])
    expected = length*uniform[3:]+.5*length**2*np.cross((1., 0., 0.), uniform[:3])
    for station, offset, action in zip(sources, offsets, actions):
        expected += action[3:]+np.cross(np.array((length*station, 0., 0.))+offset, action[:3])
    np.testing.assert_allclose(moment, expected, rtol=2e-13, atol=1e-14)


def test_consistent_equivalent_length_and_force_units():
    length, section = 1.7, _section(True)
    stations, sources = (.15, .5, .85), (.25, .7)
    offset = np.tile((.04, -.03, .02), (3, 1))
    source_offset = offset[:2]
    uniform = np.array((.6, -.4, .3, -.5, .7, .8))
    points = np.array(((.8, -.2, -.7, .1, .5, -.3), (-.2, .3, .6, -.4, .2, .1)))
    original = FixedEndLoadBubbleMaps(LinearOperator(length, section), stations=stations,
        point_stations=sources, observation_offsets_local_m=offset, point_offsets_local_m=source_offset)
    reference = original.evaluate(uniform, point_wrenches=points)
    # Consistent m/N -> mm/kN conversion. Strains scale inversely in curvature.
    scale_l, scale_f = 1000., .001
    curvature_scale = np.array((1., 1., 1., scale_l, scale_l, scale_l))
    converted_section = LinearSection.generalized(scale_f*np.asarray(section.stiffness)
                                                   *np.outer(curvature_scale, curvature_scale))
    converted = FixedEndLoadBubbleMaps(LinearOperator(scale_l*length, converted_section),
        stations=stations, point_stations=sources, observation_offsets_local_m=scale_l*offset,
        point_offsets_local_m=scale_l*source_offset)
    force_scale = np.array((scale_f,)*3+(scale_f*scale_l,)*3)
    response = converted.evaluate(uniform*force_scale/scale_l, point_wrenches=points*force_scale)
    np.testing.assert_allclose(response.pose, reference.pose*np.array((scale_l,)*3+(1.,)*3),
                               rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(response.equivalent_nodal_load,
                               reference.equivalent_nodal_load*np.tile(force_scale, 2),
                               rtol=3e-12, atol=3e-14)


def test_compiled_maps_and_responses_own_immutable_inputs_and_arrays():
    stations = np.array((0., .5, 1.))
    sources = np.array((.3, .7))
    offsets = np.tile((.02, -.03, .04), (2, 1))
    maps = FixedEndLoadBubbleMaps(LinearOperator(2., _section(True)), stations=stations,
                                point_stations=sources, point_offsets_local_m=offsets)
    uniform = np.array((.6, -.4, .3, -.5, .7, .8))
    point = np.ones((2, 6))
    response = maps.evaluate(uniform, point_wrenches=point)
    expected = response.pose.copy()
    stations[:] = .8; sources[:] = .9; offsets[:] = 8.; uniform[:] = 0.; point[:] = 0.
    assert maps.stations == (0., .5, 1.) and maps.point_stations == (.3, .7)
    np.testing.assert_array_equal(response.pose, expected)
    with pytest.raises(FrozenInstanceError): maps.production_qualified = True
    for array in (maps.uniform_pose_map, maps.point_pose_map, maps.point_compliance,
                  maps.uniform_equivalent_load_map, maps.point_equivalent_load_map,
                  response.pose, response.equivalent_nodal_load):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.setflags(write=True)
        with pytest.raises(ValueError): array.flat[0] = 0.


def test_empty_observations_and_repeated_sources_are_admitted():
    operator = LinearOperator(2., _section())
    empty = FixedEndLoadBubbleMaps(operator, stations=(), point_stations=())
    assert empty.uniform_pose_map.shape == (0, 6, 6)
    assert empty.point_compliance.shape == (0, 0)
    assert empty.evaluate().pose.shape == (0, 6)
    maps = FixedEndLoadBubbleMaps(operator, point_stations=(.5, .5))
    np.testing.assert_array_equal(maps.point_compliance[:6, :6], maps.point_compliance[6:, 6:])
    response = maps.evaluate(point_wrenches=(np.ones(6), -np.ones(6)))
    np.testing.assert_array_equal(response.pose, 0.)
    np.testing.assert_array_equal(response.equivalent_nodal_load, 0.)


@pytest.mark.parametrize("options", (
    {"stations": .5}, {"stations": (-.01,)}, {"stations": (1.01,)},
    {"stations": (float("nan"),)}, {"stations": (True,)},
    {"point_stations": (False,)}, {"point_stations": (float("inf"),)},
    {"observation_offsets_local_m": ((0., 0.),)*3},
    {"observation_offsets_local_m": ((0., 0., True),)*3},
    {"point_stations": (.5,), "point_offsets_local_m": ()},
    {"point_stations": (.5,), "point_offsets_local_m": ((0., 0., float("nan")),)},
))
def test_invalid_compilation_inputs_fail_closed(options):
    with pytest.raises(ValueError):
        FixedEndLoadBubbleMaps(LinearOperator(2., _section()), **options)


@pytest.mark.parametrize("uniform,points", (
    ((0.,)*5, None), ((True, 0., 0., 0., 0., 0.), None),
    ((float("inf"), 0., 0., 0., 0., 0.), None), ((0.,)*6, np.zeros(6)),
    ((0.,)*6, ((0.,)*5,)), ((0.,)*6, ((True, 0., 0., 0., 0., 0.),)),
))
def test_invalid_applied_actions_fail_closed(uniform, points):
    maps = FixedEndLoadBubbleMaps(LinearOperator(2., _section()), point_stations=(.5,))
    with pytest.raises(ValueError): maps.evaluate(uniform, point_wrenches=points)


def test_unrepresentable_outputs_fail_and_tiny_resolved_loads_are_preserved():
    section = LinearSection.generalized(np.eye(6))
    with pytest.raises(ValueError, match="LinearOperator"):
        FixedEndLoadBubbleMaps(object())
    maps = FixedEndLoadBubbleMaps(LinearOperator(10., section), stations=(.5,))
    with pytest.raises(RepresentabilityError, match="overflow"):
        maps.evaluate((1e308, 0., 0., 0., 0., 0.))
    maps = FixedEndLoadBubbleMaps(LinearOperator(1., section), stations=(.5,))
    with pytest.raises(RepresentabilityError, match="underflow|relative precision"):
        maps.evaluate((np.nextafter(0., 1.), 0., 0., 0., 0., 0.))
    tiny = maps.evaluate((1e-250, 0., 0., 0., 0., 0.))
    assert tiny.pose[0, 0] == pytest.approx(1e-250/8., rel=2e-14, abs=0.)
