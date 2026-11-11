"""Actual V4 reference adapter; keep loaded/history paths explicitly separate."""

import numpy as np
import pytest

from anyfesolver._ge_beam3_relative_virgin_reference_modes import solve_relative_virgin_reference_modes
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from test_ge_beam3_native_load_state import problem


def initial(model):
    return {i: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1) for i, e in model.mesh.elements.items()}


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_native_virgin_reference_uses_retained_factors(slenderness, tmp_path):
    model = FEModel('virgin-slender-reference')
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    for i, p in enumerate(points, 1): model.add_node(i, *p)
    h = 2/slenderness; ea = 12/h**2; shear = (5./6)*ea/2.6
    geometry = CenteredCurvedBeam3ReferenceGeometry(points, np.tile(np.eye(3), (3, 1, 1)))
    section = DirectedHardeningSection(np.diag([ea, shear, shear, 2., 1., 1.]),
        np.array([1., 0., 0., 0., 0., 0.]), 1e6, 1.)
    element = NativeP5BeamElement(1, (1, 2, 3), geometry, section, line_force=np.zeros(3))
    model.add_element(1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    states = initial(model); old = canonical(states)
    packet, modes = solve_relative_virgin_reference_modes(model, states, np.zeros(18),
        {1: np.diag([1., 1., 1., h*h/6, h*h/12, h*h/12])})
    assert canonical(states) == old
    assert not modes.production_qualified and not modes.prestressed_tangent_authorized
    ratios = modes.eigenvalues[:4]/np.array([36/49, 36/49, 36., 36.])
    with (tmp_path/'relative-native-reference.json').open('xb') as stream:
        stream.write(canonical(dict(slenderness=slenderness, eigenvalues=modes.eigenvalues,
            packet_identity=packet.identity, normalized_residual=modes.normalized_residual,
            production_qualified=False)))
    assert np.all(ratios > 0.) and np.max(np.abs(np.sqrt(ratios)-1.)) < .02
    for first, second in ((0, 1), (2, 3)):
        assert abs(modes.eigenvalues[first]-modes.eigenvalues[second]) <= 1e-11*max(1., modes.eigenvalues[second])


def test_curved_coupled_reference_keeps_six_rigid_modes():
    model = problem(); model.boundary_conditions.clear(); states = initial(model)
    packet, modes = solve_relative_virgin_reference_modes(model, states, np.zeros(18),
        {1: np.diag([1., 1., 1., .002, .001, .001])}, num_modes=9)
    scale = max(1., np.linalg.norm(packet.stiffness)/np.linalg.norm(packet.mass))
    assert np.max(modes.eigenvalues[:6]) < 1e-11*scale
    assert np.min(modes.eigenvalues[6:]) > 1e-11*scale


@pytest.mark.parametrize('kind', ['displaced', 'loaded_state', 'missing_state', 'bad_inertia'])
def test_nonvirgin_or_invalid_inputs_cannot_enter_reference_kernel(kind, monkeypatch):
    import anyfesolver._ge_beam3_relative_virgin_reference_modes as adapter
    from test_ge_beam3_loaded_modal import accepted
    model = problem(); states = initial(model); total = np.zeros(18)
    inertias = {1: np.eye(6)}
    if kind == 'displaced': total[6] = .01
    elif kind == 'loaded_state':
        _, states, _ = accepted(model)
    elif kind == 'missing_state': states = {}
    elif kind == 'bad_inertia': inertias = {1: -np.eye(6)}
    def forbidden(*args, **kwargs): raise AssertionError('reference factor kernel must not run')
    monkeypatch.setattr(adapter, 'solve_relative_reference_factor_spectrum', forbidden)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve_relative_virgin_reference_modes(model, states, total, inertias)


def test_model_changed_inside_reference_solver_is_rejected_after_return(monkeypatch):
    import anyfesolver._ge_beam3_relative_virgin_reference_modes as adapter
    model = problem(); states = initial(model); solve = adapter.solve_relative_reference_factor_spectrum
    def changed(*args, **kwargs):
        result = solve(*args, **kwargs)
        model.mesh.dof_manager._total_dofs += 6
        return result
    monkeypatch.setattr(adapter, 'solve_relative_reference_factor_spectrum', changed)
    with pytest.raises(ValueError):
        solve_relative_virgin_reference_modes(model, states, np.zeros(18), {1: np.eye(6)})


def test_rotated_curved_reference_frequencies_and_factors_agree():
    from copy import deepcopy
    old = problem(); rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    new = FEModel('rotated-virgin-reference')
    for i, node in old.mesh.nodes.items(): new.add_node(i, *(rotation@node.coords()))
    for i, e in old.mesh.elements.items():
        r = e.core.reference
        geometry = CenteredCurvedBeam3ReferenceGeometry(r.coordinates@rotation.T, rotation@r.nodal_triads)
        moved = NativeP5BeamElement(i, e.node_ids, geometry, e.core.section, line_force=rotation@e.core.line_force)
        new.add_element(i, moved); new.materials[moved.material_name] = moved.core.section
    for bc in old.boundary_conditions: new.add_boundary_condition(deepcopy(bc))
    results = [solve_relative_virgin_reference_modes(m, initial(m), np.zeros(18), {1: np.eye(6)}) for m in (old, new)]
    np.testing.assert_allclose(results[0][1].eigenvalues, results[1][1].eigenvalues, rtol=1e-11, atol=1e-11)
