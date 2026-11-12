"""Small straight refinement samples with a separate continuum ODE reference.

Frozen before execution: L/h=100 and1e6, four/eight macros, first three
bending frequencies in both planes; finest engineering discrepancy <2%.
No fitted coefficients, default change or full-domain qualification claim.
"""

import numpy as np
import pytest

from anyfesolver._ge_beam3_relative_virgin_reference_modes import solve_relative_virgin_reference_modes
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from docs.reference_cases.ge_beam3_straight_prestress_reference import StraightPrestress


def sample(slenderness, count):
    h = 2/slenderness; ea = 12/h**2; shear = (5/6)*ea/2.6; rotary = h*h/12
    reference = StraightPrestress(2., ea, shear, 1., 1., rotary)
    target = np.repeat([reference.bracketed_squared_frequency(b) for b in
        ((-2., 4.), (10., 100.), (100., 400.))], 2)
    model = FEModel('relative-spectrum-engineering')
    points = np.column_stack((np.linspace(0., 2., 2*count+1), np.zeros((2*count+1, 2))))
    for i, p in enumerate(points, 1): model.add_node(i, *p)
    for e in range(count):
        geometry = CenteredCurvedBeam3ReferenceGeometry(points[2*e:2*e+3], np.tile(np.eye(3), (3, 1, 1)))
        section = DirectedHardeningSection(np.diag([ea, shear, shear, 2., 1., 1.]),
            np.array([1., 0., 0., 0., 0., 0.]), 1e6, 1.)
        element = NativeP5BeamElement(e+1, (2*e+1, 2*e+2, 2*e+3), geometry, section, line_force=np.zeros(3))
        model.add_element(e+1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    states = {i: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1)
        for i, e in model.mesh.elements.items()}
    before = canonical(states)
    inertias = {i: np.diag([1., 1., 1., 2*rotary, rotary, rotary]) for i in model.mesh.elements}
    packet, modes = solve_relative_virgin_reference_modes(model, states,
        np.zeros(model.mesh.dof_manager.total_dofs), inertias)
    assert canonical(states) == before
    return dict(slenderness=slenderness, macros=count, continuum_eigenvalues=target,
        native_eigenvalues=modes.eigenvalues, normalized_residual=modes.normalized_residual,
        packet_identity=packet.identity, production_qualified=False)


@pytest.mark.parametrize('slenderness', [100., 1000000.])
def test_first_six_native_frequencies_approach_continuum(slenderness, tmp_path):
    errors = []
    for count in (4, 8):
        print(f'relative engineering initialized: slenderness={slenderness}, macros={count}', flush=True)
        row = sample(slenderness, count)
        with (tmp_path/f'{count}-macro.json').open('xb') as stream: stream.write(canonical(row))
        error = np.sqrt(row['native_eigenvalues']/row['continuum_eigenvalues'])-1.
        errors.append(error)
        assert row['normalized_residual'] <= 1e-11
        for first, second in ((0, 1), (2, 3), (4, 5)):
            assert abs(row['native_eigenvalues'][first]-row['native_eigenvalues'][second]) <= 1e-11*max(1., row['native_eigenvalues'][second])
        print(f'relative engineering completed: slenderness={slenderness}, macros={count}, errors={error.tolist()}', flush=True)
    assert np.max(np.abs(errors[-1])) < .02
    assert np.all(np.abs(errors[-1]) < np.abs(errors[0]))
