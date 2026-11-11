"""Closed G1 piecewise-quadratic ring: topology, rigid modes and pressure response.

Independent circular-ring reference: equilibrium gives N=pR for radial dead
force per reference length, while uniform expansion gives epsilon=u/R. Hence
u=pR**2/EA. A rigid translation fixes the reference node without changing this
solution. Applied loads are explicit nodal dead forces from a Q2 circular
pressure quadrature, not a claim of exact native distributed-pressure work.
Both that load approximation and the quadratic ring converge to the circle;
the geometry's tangent intersection
defines each quadratic Bezier control point, not a fitted mechanical parameter.
No production qualification or general ring-buckling claim follows from this test.
"""
from hashlib import sha256
import math
import numpy as np
import pytest

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_retained_nodal_loading import Program, NodalDeadForces
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from test_ge_beam3_native_analysis import analysis

EA = 1000.
RADIUS = 1.
PRESSURE = .1


def ring(macros, *, supported=True, transform=None, offset=None, reversed=False):
    assert type(macros) is int and 4 <= macros <= 16
    rotation = np.eye(3) if transform is None else transform
    shift = np.zeros(3) if offset is None else offset
    reversal = np.diag([-1., 1., -1.]) if reversed else np.eye(3)
    h = math.pi / macros
    midpoint_radius = RADIUS * (math.cos(h) + 1 / math.cos(h)) / 2
    model = FEModel('native-closed-ring')
    frames = []
    for i in range(2 * macros):
        angle = i * h
        radial = np.array([math.cos(angle), math.sin(angle), 0.])
        tangent = np.array([-math.sin(angle), math.cos(angle), 0.])
        second = np.array([0., 0., 1.])
        frames.append(rotation @ np.column_stack((tangent, second, np.cross(tangent, second))) @ reversal)
        model.add_node(i + 1, *(rotation @ ((midpoint_radius if i % 2 else RADIUS) * radial) + shift))
    law = EllipsoidalGeneralizedSection(np.diag([EA, 400., 350., .8, 1., 1.2]), np.eye(6), 1e6, 1.)
    for i in range(macros):
        ids = (2 * i + 1, 2 * i + 2, (2 * i + 2) % (2 * macros) + 1)
        if reversed:
            ids = ids[::-1]
        reference = CenteredCurvedBeam3ReferenceGeometry(
            np.array([model.mesh.nodes[node].coords() for node in ids]),
            np.array([frames[node - 1] for node in ids]))
        element = NativeGeneralizedStaticElement(i + 1, ids, reference, law, order=4)
        model.add_element(i + 1, element)
        model.materials[element.material_name] = element.section
    if supported:
        model.add_boundary_condition(BoundaryCondition('remove-ring-rigids', [1],
            {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    return model


def pressure_program(macros, pressure=PRESSURE, *, transform=None):
    """Analytic circular pressure integrated against all three Q2 node shapes."""
    rotation = np.eye(3) if transform is None else transform
    h = math.pi / macros
    i0 = 2 * math.sin(h) / h
    i1 = 2 * (math.sin(h) - h * math.cos(h)) / h**2
    i2 = 2 * ((h*h - 2) * math.sin(h) + 2*h*math.cos(h)) / h**3
    local = np.array([[i2/2, -i1/2, 0.], [i0-i2, 0., 0.], [i2/2, i1/2, 0.]])
    force = np.zeros((2 * macros, 3))
    for i in range(macros):
        angle = (2*i+1)*h
        basis = np.array([[math.cos(angle), -math.sin(angle), 0.],
                          [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        loads = local @ basis.T @ rotation.T * (pressure * RADIUS * h)
        for node, row in zip((2*i, 2*i+1, (2*i+2) % (2*macros)), loads):
            force[node] += row
    return Program((.5, 1.), DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(tuple((i+1, *map(float, row)) for i, row in enumerate(force))))


def test_pressure_load_matches_independent_quadrature_and_balance():
    for n in (4, 8, 16):
        p = pressure_program(n)
        force = np.array([row[1:] for row in p.nodal_forces.rows])
        independent = np.zeros_like(force)
        x, w = np.polynomial.legendre.leggauss(64)
        h = math.pi/n
        for i in range(n):
            angle = (2*i+1)*h+h*x
            field = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(x)))
            shapes = np.column_stack((x*(x-1)/2, 1-x*x, x*(x+1)/2))
            loads = shapes.T @ (w[:, None]*field) * PRESSURE*RADIUS*h
            for node, row in zip((2*i, 2*i+1, (2*i+2) % (2*n)), loads):
                independent[node] += row
        assert np.linalg.norm(force-independent) <= 1e-11
        assert np.linalg.norm(np.sum(force, axis=0)) <= 1e-11


@pytest.mark.parametrize('macros,pressure', ((4, PRESSURE), (8, PRESSURE),
    (16, PRESSURE), (16, -PRESSURE)))
def test_closed_ring_pressure_response(macros, pressure, tmp_path):
    model = ring(macros)
    made = analysis(model)
    p = pressure_program(macros, pressure)
    result = made.solve_nodal_program(p)
    assert result.status == 'completed', result.backend_result.failure
    state = result.backend_result.state
    reference = np.array([node.coords() for node in model.mesh.nodes.values()])
    displacement = state.mechanical.positions-reference+state.mechanical.position_low
    angles = np.arange(2*macros)*math.pi/macros
    radial = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(2*macros)))
    expected = pressure*RADIUS**2/EA*(radial-radial[0])
    error = float(np.linalg.norm(displacement-expected)/np.linalg.norm(expected))
    # The engineering threshold is assessed on the finest mesh, not tuned here.
    if macros == 16:
        assert error < .02
    before = canonical(state)
    recovery = made.recover_nodal_program(p, result.checkpoint, expected_sha256=result.checkpoint_sha256)
    assert canonical(state) == before
    with (tmp_path/'ring.json').open('xb') as stream:
        stream.write(canonical(dict(macros=macros, pressure=pressure,
            relative_displacement_error=error,
            displacements=displacement, circular_reference=expected,
            checkpoint_sha256=result.checkpoint_sha256, recovery=recovery,
            topology_nodes=2*macros, topology_elements=macros, production_qualified=False)))
    with (tmp_path/'checkpoint.json').open('xb') as stream:
        stream.write(result.checkpoint)
    if macros == 4:
        prefix = made.nodal_program_checkpoint_prefix(p, result.checkpoint, 1,
            expected_sha256=result.checkpoint_sha256)
        resumed = made.solve_nodal_program(p, checkpoint=prefix, expected_sha256=sha256(prefix).hexdigest())
        assert resumed.status == 'completed' and resumed.checkpoint == result.checkpoint


def test_free_ring_has_exactly_six_numerical_rigid_modes(tmp_path):
    model = ring(4, supported=False)
    made = analysis(model)
    packet, modes = made.reference_modes(num_modes=12)
    scale = max(1., np.linalg.norm(packet.stiffness)/np.linalg.norm(packet.mass))
    assert np.max(np.abs(modes.eigenvalues[:6])) <= 1e-11*scale
    assert np.min(modes.eigenvalues[6:]) > 1e-11*scale
    assert modes.normalized_residual <= 1e-11
    with (tmp_path/'free-ring.json').open('xb') as stream:
        stream.write(canonical(dict(packet=packet, modes=modes, rigid_modes=6,
            production_qualified=False)))


@pytest.mark.parametrize('kind', ('proper-motion', 'reversed-connectivity'))
def test_ring_pressure_covariance(kind, tmp_path):
    vector = np.array([.3, -.7, .4])
    angle = np.linalg.norm(vector)
    x, y, z = vector/angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*cross@cross
    transform = rotation if kind == 'proper-motion' else np.eye(3)
    models = (ring(4), ring(4, transform=transform, offset=np.array([1.3, -.7, 2.1]),
        reversed=kind == 'reversed-connectivity'))
    values = []
    for index, model in enumerate(models):
        made = analysis(model)
        p = pressure_program(4, transform=transform if index else None)
        result = made.solve_nodal_program(p)
        assert result.status == 'completed', result.backend_result.failure
        state = result.backend_result.state.mechanical
        reference = np.array([node.coords() for node in model.mesh.nodes.values()])
        values.append(state.positions-reference+state.position_low)
    expected = values[0]@transform.T
    error = float(np.linalg.norm(values[1]-expected)/np.linalg.norm(expected))
    assert error <= 1e-11
    with (tmp_path/'covariance.json').open('xb') as stream:
        stream.write(canonical(dict(kind=kind, relative_error=error, displacements=values,
            production_qualified=False)))
