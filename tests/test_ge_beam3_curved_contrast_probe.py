"""Small two-macro curved/coupled contrast diagnosis, not qualification.

Freeze slenderness parameters 100, 10000, 1000000 and one exact coordinate
permutation rotation. No loads yet: distinguish reference-state construction
and spectral covariance from nonlinear equilibrium defects.
"""

import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver._ge_beam3_seeded_signed_modes import solve_signed_loaded_modes


ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def make_curved(slenderness, rotation=np.eye(3), element_type=NativeP5BeamElement):
    model = FEModel('two-macro-curved-contrast'); parameters = np.linspace(-1., 1., 5)
    nodes = np.array([[x, .25*(1-x*x), .125*(1-x*x)] for x in parameters])
    frames = []
    for x in parameters:
        a = np.array([1., -.5*x, -.25*x]); a /= np.linalg.norm(a)
        b = np.array([0., 0., 1.]); b -= a*(a@b); b /= np.linalg.norm(b)
        frames.append(np.column_stack((a, b, np.cross(a, b))))
    nodes = nodes@rotation.T; frames = rotation@np.array(frames)
    for i, point in enumerate(nodes, 1): model.add_node(i, *point)
    ea = 3*slenderness**2; h = 2/slenderness
    base = np.array([[1., .05, -.08, .12, -.04, .03], [0., 1., .06, -.05, .08, .04],
        [0., 0., 1., .07, .02, -.05], [0., 0., 0., 1., .1, .05],
        [0., 0., 0., 0., 1., -.1], [0., 0., 0., 0., 0., 1.]])
    factor = base@np.diag(np.sqrt([ea, ea/3, ea/3.5, 2., 1., 1.5]))
    for index in range(2):
        section = DirectedHardeningSection(factor.T@factor, np.array([1., .2, -.1, .3, -.4, .5]), 1e6, 1.)
        reference = CenteredCurvedBeam3ReferenceGeometry(nodes[2*index:2*index+3], frames[2*index:2*index+3])
        e = element_type(index+1, (2*index+1, 2*index+2, 2*index+3), reference, section,
            line_force=np.zeros(3))
        model.add_element(index+1, e); model.materials[e.material_name] = e.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    inertia = np.diag([1., 1., 1., h*h/6, h*h/12, h*h/12])
    return model, {1: inertia, 2: inertia.copy()}


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_stress_free_curved_coupled_modes_are_frame_covariant(slenderness, tmp_path):
    outputs = []
    for case, rotation in (('E', np.eye(3)), ('R90', ROTATION)):
        stage = 'construction'
        try:
            model, inertias = make_curved(slenderness, rotation)
            stage = 'initial-state'
            states = {i: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1)
                for i, e in model.mesh.elements.items()}
            saved = canonical(states); stage = 'signed-modes'
            packet, modes = solve_signed_loaded_modes(model, states, np.zeros(30), inertias, np.zeros(30),
                load_parameter=0., bounds=(-10000., 100000000.), num_modes=6)
            assert canonical(states) == saved
            row = dict(slenderness=slenderness, case=case, status='returned', stage=stage,
                eigenvalues=modes.eigenvalues, spectral_residual=modes.spectral_residual,
                modes=modes.full_modes, mass=packet.mass, production_qualified=False)
        except Exception as exc:
            with (tmp_path/(case+'-failure.json')).open('xb') as stream:
                stream.write(canonical(dict(slenderness=slenderness, case=case, stage=stage,
                    status='failed', exception_type=type(exc).__name__, message=str(exc), production_qualified=False)))
            raise
        with (tmp_path/(case+'-modes.json')).open('xb') as stream: stream.write(canonical(row))
        outputs.append((packet, modes))
    first, second = outputs
    np.testing.assert_allclose(second[1].eigenvalues, first[1].eigenvalues, rtol=1e-11, atol=1e-11)
    transform = np.kron(np.eye(14), ROTATION)
    correlation = (transform@first[1].full_modes).T@second[0].mass@second[1].full_modes
    np.testing.assert_allclose(np.abs(correlation), np.eye(6), rtol=1e-11, atol=1e-11)
