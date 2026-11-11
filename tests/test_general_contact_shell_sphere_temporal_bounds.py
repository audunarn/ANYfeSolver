"""Independent geometry fixtures for the Q4/sphere represented cubic path."""

from fractions import Fraction
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.general_contact.beam_temporal_bounds import BeamHermiteGapBoundsError
from anyfesolver.general_contact.shell_sphere_temporal_bounds import (
    _face_controls, _split_tensor, _squared_controls,
    q4_sphere_hermite_gap_bounds,
)
from anyfesolver.general_contact.sphere import _SphereFace


def _shape(r, s):
    return (np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)])/4,
            None, None)


def _face(*, height=0., offset=0., half_width=1., element_id=1, dof_start=0):
    reference = np.array([[-half_width, -half_width, height],
                          [half_width, -half_width, height],
                          [half_width, half_width, height],
                          [-half_width, half_width, height]])
    return _SphereFace(element_id, tuple(range(4)),
                       np.arange(dof_start, dof_start+24).reshape(4, 6),
                       reference, np.array([0., 0., 1.]), offset,
                       SimpleNamespace(compute_shape_functions=_shape), ())


def _manager(*, faces=None, radius=.25, **face_options):
    faces = (_face(**face_options),) if faces is None else tuple(faces)
    return SimpleNamespace(faces=faces, radius=radius, kind="sphere", metric=np.eye(3))


def _bounds(manager, **options):
    size = int(max((np.max(face.dofs) for face in manager.faces), default=23))+1
    zeros = np.zeros(size)
    return q4_sphere_hermite_gap_bounds(
        manager, options.pop("start", zeros), options.pop("end", zeros),
        options.pop("start_velocity", zeros), options.pop("end_velocity", zeros),
        options.pop("dt", 1.),
        sphere_start_point=options.pop("sphere_start_point", [0., 0., 0.]),
        sphere_end_point=options.pop("sphere_end_point", [0., 0., 0.]),
        sphere_start_velocity=options.pop("sphere_start_velocity", [0., 0., 0.]),
        sphere_end_velocity=options.pop("sphere_end_velocity", [0., 0., 0.]),
        tolerance_m=options.pop("tolerance_m", 1.e-8),
        maximum_subdivisions=options.pop("maximum_subdivisions", 1024), **options)


def _encloses(report, expected, tolerance=1.e-8):
    assert report.lower_gap <= expected <= report.upper_gap
    assert report.upper_gap-report.lower_gap <= tolerance


def _evaluate(tensor, coordinates):
    degrees = (len(tensor)-1, len(tensor[0])-1, len(tensor[0][0])-1)
    return sum(value*math.prod(math.comb(degree, index)*coordinate**index
                              *(1-coordinate)**(degree-index)
                              for degree, index, coordinate in zip(degrees, (i, j, k), coordinates))
               for i, slab in enumerate(tensor) for j, row in enumerate(slab)
               for k, value in enumerate(row))


def test_squared_tensor_retains_both_spatial_parameters_and_all_mixed_products():
    controls = tuple(tuple(tuple(tuple(Fraction((axis+1)*(i+2)-3*j+(k-2)**2, 16)
                                       for k in range(4)) for j in range(2))
                            for i in range(2)) for axis in range(3))
    squared = _squared_controls(controls)
    assert (len(squared), len(squared[0]), len(squared[0][0])) == (3, 3, 7)
    for coordinates in itertools.product((Fraction(2, 7), Fraction(3, 5)), repeat=3):
        expected = sum(_evaluate(lane, coordinates)**2 for lane in controls)
        assert _evaluate(squared, coordinates) == expected


@pytest.mark.parametrize("axis", range(3))
def test_tensor_subdivision_preserves_independent_polynomial_on_each_axis(axis):
    tensor = tuple(tuple(tuple(Fraction(7*i-3*j+k*k, 32) for k in range(4))
                         for j in range(2)) for i in range(2))
    children = _split_tensor(tensor, axis)
    point = [Fraction(2, 7), Fraction(3, 5), Fraction(4, 11)]
    for side, child in enumerate(children):
        parent = list(point)
        parent[axis] = (parent[axis]+side)/2
        assert _evaluate(child, point) == _evaluate(tensor, parent)


def test_physical_skin_interior_collides_while_every_vertex_is_distant():
    manager = _manager(height=.125, offset=.0625, half_width=10.)
    report = _bounds(manager, maximum_subdivisions=0)
    _encloses(report, -.0625)
    assert all(np.linalg.norm(point+[0., 0., .0625])-.25 > 10
               for point in manager.faces[0].reference)


@pytest.mark.parametrize("offset", [-.0625, .0625])
def test_physical_top_bottom_offset_is_applied_once(offset):
    report = _bounds(_manager(height=.5, offset=offset, radius=.125))
    _encloses(report, .5+offset-.125)


@pytest.mark.parametrize("moving", ["sphere", "shell", "both"])
def test_cubic_interior_crossing_uses_both_bodies_endpoint_velocities(moving):
    # Relative z(t)=16*(t-1/4)^2*(1+t), with endpoints 1,18
    # and dimensionless derivatives -7,57. All endpoint gaps are positive.
    # dt=2 checks that velocities have units per physical time.
    if moving == "sphere":
        manager = _manager(height=-.0625, offset=.0625)
        report = _bounds(manager, dt=2., sphere_start_point=[0., 0., 1.],
                         sphere_end_point=[0., 0., 18.],
                         sphere_start_velocity=[0., 0., -3.5],
                         sphere_end_velocity=[0., 0., 28.5])
    else:
        manager = _manager(height=.9375, offset=.0625)
        end, v0, v1 = (np.zeros(24) for _ in range(3))
        if moving == "shell":
            end[2::6], v0[2::6], v1[2::6] = 17., -3.5, 28.5
            options = {}
        else:
            # Any independent cubic shell path minus this sphere path gives
            # the same analytic relative polynomial, requiring both motions.
            end[2::6], v0[2::6], v1[2::6] = 8., 3., 13.
            options = dict(sphere_end_point=[0., 0., -9.],
                           sphere_start_velocity=[0., 0., 6.5],
                           sphere_end_velocity=[0., 0., -15.5])
        report = _bounds(manager, end=end, start_velocity=v0, end_velocity=v1,
                         dt=2., **options)
    _encloses(report, -.25)
    assert report.subdivisions > 0


@pytest.mark.parametrize("body,endpoint", itertools.product(("shell", "sphere"), ("start", "end")))
def test_each_endpoint_velocity_can_create_a_hidden_crossing(body, endpoint):
    manager = _manager(height=1.)
    _encloses(_bounds(manager), .75)
    sign = -1 if endpoint == "start" else 1
    if body == "shell":
        velocity = np.zeros(24)
        velocity[2::6] = sign*8
        options = {endpoint+"_velocity": velocity}
    else:
        options = {"sphere_"+endpoint+"_velocity": [0., 0., -sign*8]}
    _encloses(_bounds(manager, **options), -.25)


def test_joint_two_space_coordinates_and_nondyadic_temporal_turning_minimum():
    # Surface minimum z(t)=d+a*(3*t-1)^2*(1+t), attained at
    # t=1/3, u=3/8, v=5/8. Endpoints reconstruct this exactly.
    d, a, offset = .5, 2.**-15, .0625
    manager = _manager(height=d+a-offset, offset=offset, radius=.25)
    end, v0, v1 = (np.zeros(24) for _ in range(3))
    end[2::6], v0[2::6], v1[2::6] = 7*a, -5*a, 28*a
    report = _bounds(manager, end=end, start_velocity=v0, end_velocity=v1,
                     sphere_start_point=[-.25, .25, 0.], sphere_end_point=[-.25, .25, 0.],
                     maximum_subdivisions=512)
    _encloses(report, .25)
    assert report.subdivisions <= 512


def test_exact_controls_match_independent_physical_face_and_hermite_evaluation():
    face = _face(height=.5, offset=.125)
    states = tuple(np.array([((i+1)*(j+2)%11-5)/32 for j in range(24)]) for i in range(4))
    sphere_states = (np.array([.25, -.5, 1.]), np.array([-.125, .25, .75]),
                     np.array([.5, -.125, .25]), np.array([-.25, .5, -.125]))
    step = Fraction(2)
    sphere = tuple((Fraction(float(sphere_states[0][i])),
                    Fraction(float(sphere_states[0][i]))+step*Fraction(float(sphere_states[2][i]))/3,
                    Fraction(float(sphere_states[1][i]))-step*Fraction(float(sphere_states[3][i]))/3,
                    Fraction(float(sphere_states[1][i]))) for i in range(3))
    controls = _face_controls(face, states, step, sphere)
    for u, v, t in itertools.product((.125, .375, .75), repeat=3):
        weights = np.array([2*t**3-3*t*t+1, -2*t**3+3*t*t,
                            2*(t**3-2*t*t+t), 2*(t**3-t*t)])
        shell = sum(weight*state for weight, state in zip(weights, states))
        centre = sum(weight*state for weight, state in zip(weights, sphere_states))
        actual = face.point((2*u-1, 2*v-1), shell)-centre
        exact = np.array([float(_evaluate(lane, tuple(map(Fraction, (u, v, t))))) for lane in controls])
        np.testing.assert_allclose(exact, actual, rtol=0., atol=1.e-14)


def test_exhaustive_small_static_rectangles_match_analytic_distance():
    # Exact planar closest-point distance: clamp independently in x and y.
    for width, offset, centre in itertools.product((.5, 1.), (-.0625, .0625),
                                                   ((0., 0., 1.), (.25, -.5, .75), (1.25, -.875, .5))):
        manager = _manager(height=.125, offset=offset, half_width=width)
        dx, dy = max(abs(centre[0])-width, 0.), max(abs(centre[1])-width, 0.)
        dz = centre[2]-.125-offset
        expected = math.sqrt(dx*dx+dy*dy+dz*dz)-.25
        report = _bounds(manager, sphere_start_point=centre, sphere_end_point=centre)
        _encloses(report, expected)


@pytest.mark.parametrize("height,saddle", itertools.product((.5, 1.), (-.25, .25)))
def test_bilinear_normal_deformation_and_cubic_time_have_analytic_interior_minimum(height, saddle):
    # z=d+a*x*y+b*(3*t-1)^2*(1+t). Here z>=0 and |d*a|<1,
    # so x²+y²+(d+a*x*y)² >= d² and the extra positive cubic
    # term only increases the norm. Equality requires x=y=0,t=1/3.
    manager = _manager(offset=.0625)
    start, end, v0, v1 = (np.zeros(24) for _ in range(4))
    b = 2.**-10
    points = manager.faces[0].reference
    start[2::6] = height-.0625+saddle*points[:, 0]*points[:, 1]+b
    end[2::6] = start[2::6]+7*b
    v0[2::6], v1[2::6] = -5*b, 28*b
    report = _bounds(manager, start=start, end=end, start_velocity=v0, end_velocity=v1)
    _encloses(report, height-.25)


@pytest.mark.parametrize("scale,tolerance", [(1.e200, 1.e-8), (1.e-200, 1.e-210)])
def test_extreme_finite_squared_distance_and_radius_cancellation_are_exact(scale, tolerance):
    report = _bounds(_manager(height=scale, radius=scale), tolerance_m=tolerance,
                     maximum_subdivisions=0)
    _encloses(report, 0., tolerance)


def test_every_face_contributes_and_subdivision_budget_is_shared():
    faces = (_face(height=10., element_id=1), _face(height=0., element_id=2, dof_start=24))
    _encloses(_bounds(_manager(faces=faces), maximum_subdivisions=0), -.25)
    manager = _manager(faces=(_face(height=1.), _face(height=1., dof_start=24)))
    with pytest.raises(BeamHermiteGapBoundsError, match="unresolved after 1 subdivisions"):
        _bounds(manager, sphere_start_velocity=[0., 0., 6.], maximum_subdivisions=1)


def test_resource_exhaustion_roundoff_and_cancellation_fail_explicitly():
    manager = _manager(height=1.)
    with pytest.raises(BeamHermiteGapBoundsError, match="unresolved after 0"):
        _bounds(manager, sphere_start_velocity=[0., 0., 6.], maximum_subdivisions=0)
    with pytest.raises(BeamHermiteGapBoundsError, match="roundoff"):
        _bounds(manager, tolerance_m=1.e-30)
    with pytest.raises(BeamHermiteGapBoundsError, match="cancelled"):
        _bounds(manager, check_cancel=lambda: True)
    calls = []
    def later_cancel():
        calls.append(True)
        return len(calls) == 4
    with pytest.raises(BeamHermiteGapBoundsError, match="cancelled"):
        _bounds(manager, sphere_start_velocity=[0., 0., 6.], check_cancel=later_cancel)
    assert len(calls) == 4
    def own_stop():
        raise InterruptedError("owned stop")
    with pytest.raises(InterruptedError, match="owned stop"):
        _bounds(manager, check_cancel=own_stop)
    with pytest.raises(BeamHermiteGapBoundsError, match="finite float range"):
        _bounds(_manager(height=np.finfo(float).max, radius=0.))


@pytest.mark.parametrize("options", [dict(dt=0.), dict(dt=float("nan")),
    dict(tolerance_m=-1.), dict(maximum_subdivisions=True), dict(maximum_subdivisions=-1),
    dict(check_cancel=False), dict(sphere_start_point=[0., 0.]),
    dict(sphere_end_velocity=[0., float("inf"), 0.]), dict(end=np.zeros(23))])
def test_invalid_motion_and_budget_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        _bounds(_manager(), **options)


@pytest.mark.parametrize("invalid", ["triangle", "warp", "dofs", "director", "empty", "cylinder", "metric", "radius"])
def test_unsupported_or_invalid_geometry_is_rejected(invalid):
    manager = _manager()
    if invalid in ("triangle", "warp", "dofs", "director"):
        face = manager.faces[0]
        from dataclasses import replace
        if invalid == "triangle":
            face = replace(face, node_ids=(0, 1, 2), reference=face.reference[:3], dofs=face.dofs[:3])
        elif invalid == "warp":
            reference = face.reference.copy()
            reference[2, 2] = .125
            face = replace(face, reference=reference)
        elif invalid == "dofs":
            face = replace(face, dofs=face.dofs.astype(float))
        else:
            face = replace(face, director=np.zeros(3))
        manager.faces = (face,)
    elif invalid == "empty":
        manager.faces = ()
    elif invalid == "cylinder":
        manager.kind = "cylinder"
    elif invalid == "metric":
        manager.metric[2, 2] = 0.
    else:
        manager.radius = -1.
    with pytest.raises(ValueError):
        _bounds(manager)


def test_real_q4_sphere_manager_physical_skin_agrees_with_analytic_gap():
    from anyfesolver import (ContactDefinition, ContactInteraction, ContactSolverSettings,
                           ContactSurface, RigidContactBody, create_shell_element)
    from anyfesolver.fe_core import FEModel
    from anyfesolver.general_contact.sphere import _SphereShellManager
    model = FEModel("represented_q4_sphere")
    model.add_material("elastic", 1.e5, .3, density=1.)
    for node, (x, y) in enumerate(((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)), 1):
        model.add_node(node, x, y, -.0625)
    model.add_element(1, create_shell_element(1, (1, 2, 3, 4), "elastic", thickness=.125))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody("ball", "sphere", (0., 0., 1.), radius=.25),),
        interactions=(ContactInteraction("pair", "skin", "ball", 1.e5, 1.e-4),))
    manager = _SphereShellManager(model, definition, ContactSolverSettings(
        pair_motion="general", maximum_shell_tilt=.05))
    report = _bounds(manager, sphere_start_point=[0., 0., 1.], sphere_end_point=[0., 0., 1.])
    _encloses(report, .75)
