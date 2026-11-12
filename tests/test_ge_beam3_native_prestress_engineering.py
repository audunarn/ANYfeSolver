"""Small native/continuum engineering comparisons, not a qualification wave."""

from dataclasses import replace
import json

import numpy as np
import pytest

from anyfesolver._ge_beam3_loaded_modal import solve_elastic_modes
from anyfesolver._ge_beam3_load_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.fe_core import FEModel
from docs.reference_cases.ge_beam3_straight_prestress_reference import StraightPrestress


def model_for(count):
    assert count in (2, 4)
    model = FEModel('straight-prestress-engineering-development')
    points = np.column_stack((np.linspace(0., 2., 2*count+1), np.zeros((2*count+1, 2))))
    for i, point in enumerate(points, 1): model.add_node(i, *point)
    for e in range(count):
        reference = CenteredCurvedBeam3ReferenceGeometry(points[2*e:2*e+3], np.tile(np.eye(3), (3, 1, 1)))
        section = DirectedHardeningSection(np.diag([3000., 1000., 1000., 2., 1., 1.]),
            np.array([1., 0., 0., 0., 0., 0.]), 1e6, 1.)
        element = NativeP5BeamElement(e+1, (2*e+1, 2*e+2, 2*e+3), reference, section, line_force=np.zeros(3))
        model.add_element(e+1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    return model


@pytest.fixture(scope='module')
def samples(tmp_path_factory):
    directory = tmp_path_factory.mktemp('native-prestress-engineering')
    reference = StraightPrestress(2., 3000., 1000., 1., 1., .001)
    pressure = reference.critical_compression(); records = []
    for count, ratio in ((2, 0.), (4, 0.), (4, .5), (4, .98), (4, 1.02)):
        tension = -ratio*pressure; model = model_for(count)
        print(f'prestress fixture initialized: macros={count}, compression_ratio={ratio}', flush=True)
        result = solve_force_program(model, ForceProgram((1.,), ((2*count+1, tension, 0., 0.),)))
        # Preserve the actual response before interpreting it, including failures.
        with (directory/f'{count}-{ratio}-state.json').open('xb') as stream:
            stream.write(result.checkpoint)
        assert result.status == 'completed', result.failure
        states = {r['element_id']: r['state'] for r in json.loads(result.checkpoint)['element_states']}
        saved = canonical(states); force = np.zeros(len(result.displacements))
        force[model.mesh.dof_manager.get_node_dofs(2*count+1)[0]] = tension
        inertias = {e: np.diag([1., 1., 1., .002, .001, .001]) for e in model.mesh.elements}
        print(f'prestress equilibrium accepted: macros={count}, compression_ratio={ratio}', flush=True)
        packet, modes = solve_elastic_modes(model, states, result.displacements, inertias, force,
            load_parameter=1., num_modes=6)
        exact = replace(reference, tension=tension).bracketed_squared_frequency((-2., 4.))
        record = dict(macros=count, compression_ratio=ratio, tension=tension,
            critical_compression=pressure, reference_squared_frequency=exact,
            native_squared_frequencies=modes.eigenvalues, operator_identity=packet.identity,
            normalized_residual=modes.normalized_residual, production_qualified=False)
        with (directory/f'{count}-{ratio}-comparison.json').open('xb') as stream:
            stream.write(canonical(record))
        assert saved == canonical(states)
        x = np.array([node.coords()[0] for node in model.mesh.nodes.values()])
        expected = np.zeros_like(result.displacements).reshape(-1, 6)
        expected[:, 0] = tension*x/3000.
        np.testing.assert_allclose(result.displacements.reshape(-1, 6), expected, rtol=1e-11, atol=1e-11)
        assert modes.normalized_residual <= 1e-11
        assert modes.negative_eigenvalues_retained and not modes.buckling_factor_authorized
        assert not modes.production_qualified and not packet.production_qualified
        records.append(record)
        print(f'prestress comparison preserved: macros={count}, compression_ratio={ratio}', flush=True)
    return records


def test_both_bending_frequencies_meet_frozen_engineering_check(samples):
    for row in samples[:3]:
        error = np.sqrt(row['native_squared_frequencies'][:2]/row['reference_squared_frequency'])-1.
        if row['macros'] == 4: assert np.max(np.abs(error)) < .02
    coarse, fine = samples[:2]
    for i in (0, 1):
        assert abs(fine['native_squared_frequencies'][i]/fine['reference_squared_frequency']-1.) < abs(
            coarse['native_squared_frequencies'][i]/coarse['reference_squared_frequency']-1.)


def test_native_instability_signs_straddle_continuum_onset_neighborhood(samples):
    below, above = samples[-2:]
    assert below['reference_squared_frequency'] > 0. > above['reference_squared_frequency']
    assert np.all(below['native_squared_frequencies'][:2] > 0.)
    assert np.all(above['native_squared_frequencies'][:2] < 0.)
    # Two spatial bending planes only; no unseen unstable mode is clipped.
    assert np.all(above['native_squared_frequencies'][2:] > 0.)
