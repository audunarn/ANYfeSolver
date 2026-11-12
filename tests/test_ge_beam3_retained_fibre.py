"""Finite-rotation physical-fibre potential and FEModel container checks."""
import json
import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_curved_contrast_probe import make_curved
from test_ge_beam3_fibre_section import section
from test_ge_beam3_retained_plastic import args


def make_model(contrast=1., *, order=4):
    old, _ = make_curved(100.)
    model = FEModel('native-retained-physical-fibre')
    for node_id, node in old.mesh.nodes.items(): model.add_node(node_id, *node.coords())
    for element_id, element in old.mesh.elements.items():
        made = NativeRetainedFibreElement(element_id, tuple(element.node_ids), element.core.reference,
            section(contrast), order=order)
        model.add_element(element_id, made); model.materials[made.material_name] = made.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    return model


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_full_fibre_potential_first_second_variations(contrast, tmp_path):
    model = make_model(contrast); probe = model.mesh.elements[1].operator; state = args(probe)
    origin = probe.cell.response((state[-1]*.5).tolist())['history']; saved = canonical(origin)
    direction = .1*np.cos(np.arange(42)); centre = .01*np.sin(np.arange(42)); step = 1e-5
    value = probe.evaluate(*state, origin=origin, increment=centre)
    plus = probe.evaluate(*state, origin=origin, increment=centre+step*direction)
    minus = probe.evaluate(*state, origin=origin, increment=centre-step*direction)
    tangent = value.hessian+value.hessian_low
    second = float(np.linalg.norm((plus.residual-minus.residual)/(2*step)-tangent@direction)/max(1., np.linalg.norm(tangent@direction)))
    first = abs((plus.potential-minus.potential)/(2*step)-value.residual@direction)
    symmetry = float(np.linalg.norm(tangent-tangent.T)/max(1., np.linalg.norm(tangent)))
    assert first <= 1e-7 and second <= 1e-7 and symmetry <= 1e-11
    assert canonical(origin) == saved and not value.production_qualified
    assert value.material == probe.evaluate(*state, origin=origin, increment=centre).material
    for array in (value.residual, value.hessian, value.hessian_low, value.kinematics):
        with pytest.raises(ValueError): array.setflags(write=True)
    with (tmp_path/'fibre-variations.json').open('xb') as stream:
        stream.write(canonical(dict(first=first, second=second, symmetry=symmetry, material=json.loads(value.material))))


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_arbitrary_common_rigid_motion_and_physical_fibre_recovery(contrast, tmp_path):
    probe = make_model(contrast).mesh.elements[1].operator; x, low, q, u, p = args(probe)
    origin = probe.cell.response((p*.5).tolist())['history']
    common = rotation([2.6, .8, -.3]); shift = [1000., -2000., 3000.]
    moved = np.zeros_like(x); moved_low = np.zeros_like(low)
    for node in range(3):
        for axis in range(3):
            moved[node, axis], moved_low[node, axis] = split_sum([shift[axis],
                *(float(common[axis, j]*x[node, j]) for j in range(3))])
    first = probe.evaluate(x, low, q, u, p, origin=origin)
    second = probe.evaluate(moved, moved_low, common@q, common@u, p, origin=origin)
    transform = np.eye(42); transform[:24, :24] = np.kron(np.eye(8), common)
    errors = dict(energy=abs(first.potential-second.potential),
        kinematics=float(np.linalg.norm(first.kinematics-second.kinematics)),
        residual=float(np.linalg.norm(transform@first.residual-second.residual)),
        hessian=float(np.linalg.norm(transform@first.hessian@transform.T-second.hessian)/max(1., np.linalg.norm(first.hessian))))
    assert max(errors.values()) <= 1e-11 and first.material == second.material
    before = probe.recover(u, p, origin=origin); after = probe.recover(common@u, p, origin=origin)
    for a, b in zip(before, after):
        for key in ('strain', 'strain_low', 'resultants', 'resultants_low', 'fibres', 'history'):
            assert canonical(a[key]) == canonical(b[key])
        np.testing.assert_allclose(common@a['current_frame'], b['current_frame'], rtol=1e-11, atol=1e-11)
    with (tmp_path/'fibre-objectivity.json').open('xb') as stream: stream.write(canonical(dict(errors=errors, recovery=before)))


@pytest.mark.parametrize('method', ['compute_stiffness_matrix', 'compute_mass_matrix', 'compute_geometric_stiffness_matrix',
    'compute_internal_forces', 'compute_nonlinear_response', 'compute_stresses'])
def test_unqualified_ordinary_routes_fail_closed(method):
    model = make_model(); element = model.mesh.elements[1]
    element.check(model.mesh)
    assert element.num_nodes == 3 and element.dofs_per_node == 6 and element.total_dofs == 18
    assert len(element.get_dof_mapping(model.mesh)) == 18
    with pytest.raises(ValueError, match='requires the retained-fibre driver'): getattr(element, method)(model.mesh)


def test_invalid_chart_origin_and_model_mutations_rejected():
    model = make_model(); element = model.mesh.elements[1]; probe = element.operator; state = args(probe)
    delta = np.zeros(42); delta[3] = np.pi
    with pytest.raises(ValueError): probe.evaluate(*state, increment=delta)
    with pytest.raises(ValueError): probe.evaluate(*state, origin=make_model(2.).mesh.elements[1].operator.cell.virgin())
    model.mesh.nodes[1].x += .01
    with pytest.raises(ValueError, match='node/reference mismatch'): element.check(model.mesh)
    with pytest.raises(AttributeError): probe.cell = None


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_order_eight_station_coverage_and_consistent_recovery(contrast):
    probe = make_model(contrast, order=8).mesh.elements[1].operator; state = args(probe)
    value = probe.evaluate(*state); recovery = probe.recover(state[3], state[4])
    assert len(recovery) == 16 and len(value.history.stations) == 16
    material = json.loads(value.material)
    for row, fields in zip(recovery, material['stations']):
        assert canonical(row['fibres']) == canonical(fields['fibres'])
        np.testing.assert_array_equal(row['resultants'], fields['resultants'][0])


@pytest.mark.parametrize('name,value', [('formulation_id', 'legacy-b3'), ('production_qualified', True),
                                     ('num_nodes', 3.), ('dofs_per_node', 5)])
def test_mutated_element_provenance_is_rejected(name, value):
    model = make_model(); element = model.mesh.elements[1]
    setattr(element, name, value)
    with pytest.raises(ValueError, match='formulation/provenance traits changed'): element.check(model.mesh)
