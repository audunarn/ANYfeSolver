"""Accepted axial states versus a separate two-cell thin-limit pencil.

One macro only, three frozen loads, two slenderness values. This is numerical
development, not a continuum buckling or general prestressed-mode gate.
"""

import numpy as np
import pytest
from scipy import linalg
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver._ge_beam3_loaded_modal import prepare
from docs.reference_cases.ge_beam3_loaded_factor_split_probe import split_accepted_elastic_operator
from docs.reference_cases.ge_beam3_signed_spectrum_probe import reduce_signed_split, bracket_lowest
from docs.reference_cases.ge_beam3_straight_discrete_signed_reference import bending_roots
from test_ge_beam3_loaded_modal import accepted
from test_ge_beam3_native_load_state import problem
from test_ge_beam3_curved_p5_mass_probe import section_mass
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum


def make(slenderness):
    h = 2./slenderness; ea = 12./h**2; shear = (5./6)*ea/2.6
    model = FEModel('signed-loaded-conditioning-probe')
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    for i, p in enumerate(points, 1): model.add_node(i, *p)
    reference = CenteredCurvedBeam3ReferenceGeometry(points, np.tile(np.eye(3), (3, 1, 1)))
    section = DirectedHardeningSection(np.diag([ea, shear, shear, 2., 1., 1.]),
        np.array([1., 0., 0., 0., 0., 0.]), 1e6, 1.)
    element = NativeP5BeamElement(1, (1, 2, 3), reference, section, line_force=np.zeros(3))
    model.add_element(1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    return model, np.diag([1., 1., 1., h*h/6, h*h/12, h*h/12])


def thin_limit(axial):
    # Reference bending Schur from exact endpoint-moment elimination (EI=1).
    # Inextensibility makes the two cell spins equal their transverse slopes;
    # axial initial-stress work is N*((v_mid)^2+(v_tip-v_mid)^2)/2.
    k = np.array([[96., -30.], [-30., 12.]])/7
    k += axial*np.array([[2., -1.], [-1., 1.]])
    m = np.array([[4., 1.], [1., 2.]])/6
    return np.repeat(linalg.eigvalsh(k, m), 2)


@pytest.mark.parametrize('slenderness', [100., 1000000.])
@pytest.mark.parametrize('axial', [-1., -.5, .5])
def test_actual_loaded_state_retains_signed_discrete_bending_roots(slenderness, axial, tmp_path):
    model, inertia = make(slenderness)
    result, states, force = accepted(model, nodal=((3, axial, 0., 0.),))
    with (tmp_path/'accepted-state.json').open('xb') as stream: stream.write(result.checkpoint)
    saved = canonical(states)
    packet, guard = prepare(model, states, result.displacements, {1: inertia}, load_parameter=result.parameter)
    residual = packet.net_residual-np.r_[force, np.zeros(6)]
    assert np.linalg.norm(residual[list(packet.free_dofs)]) <= 1e-11*max(1., np.linalg.norm(force))
    element = model.mesh.elements[1]
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    f, g = split_accepted_elastic_operator(element, inner)
    assert np.linalg.norm(f.T@f+g-packet.stiffness) <= 1e-11*np.linalg.norm(packet.stiffness)
    d, signed, vectors = reduce_signed_split(f, g, packet.mass, packet.free_dofs, packet.algebraic_dofs)
    intervals = bracket_lowest(d, signed, (-10., 100.), 4)
    values = intervals.mean(axis=1); expected = thin_limit(axial)
    finite = np.repeat(bending_roots(int(slenderness), axial), 2)
    row = dict(slenderness=slenderness, axial=axial, intervals=intervals,
        eigenvalues=values, thin_limit=expected, finite_shear_reference=finite,
        material_diagonal=d, signed_correction=signed,
        packet_identity=packet.identity, production_qualified=False, certified_intervals=False)
    with (tmp_path/'signed-comparison.json').open('xb') as stream: stream.write(canonical(row))
    print(canonical(dict(slenderness=slenderness, axial=axial, values=values, thin=expected)).decode(), flush=True)
    assert np.max(np.abs((values-expected)/expected)) < (.002 if slenderness == 100. else 1e-8)
    assert np.max(np.abs(values-finite)) < 1e-9
    assert np.all(np.sign(values) == np.sign(expected))
    assert canonical(states) == saved
    guard()


def test_loaded_curved_coupled_signed_roots_match_moderate_dense_pencil(tmp_path):
    model = problem()
    result, states, force = accepted(model, nodal=((55, .025, -.01, .005),))
    saved = canonical(states)
    packet, guard = prepare(model, states, result.displacements, {1: section_mass()}, load_parameter=result.parameter)
    element = model.mesh.elements[1]
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    f, g = split_accepted_elastic_operator(element, inner)
    assert np.linalg.norm(g) > .01
    d, signed, vectors = reduce_signed_split(f, g, packet.mass, packet.free_dofs, packet.algebraic_dofs)
    intervals = bracket_lowest(d, signed, (-10000., 1000000.), 6)
    dense = solve_stationary_spectrum(packet.stiffness, packet.mass, packet.free_dofs,
        packet.algebraic_dofs, num_modes=6)
    assert np.max(np.abs(intervals.mean(axis=1)-dense.eigenvalues)) < 1e-9
    # This checks the actual signed trace correction, not just root agreement.
    reconstructed = vectors.T@packet.stiffness@vectors
    assert np.linalg.norm(reconstructed-np.diag(d)-signed) <= 1e-11*np.linalg.norm(reconstructed)
    assert np.linalg.norm(vectors.T@packet.mass@vectors-np.eye(len(d))) <= 1e-11
    with (tmp_path/'curved-signed.json').open('xb') as stream:
        stream.write(canonical(dict(intervals=intervals, dense_eigenvalues=dense.eigenvalues,
            packet_identity=packet.identity, production_qualified=False)))
    assert canonical(states) == saved
    guard()
