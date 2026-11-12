"""Small accepted-state split identities; no prestress qualification claim."""

import numpy as np
import pytest

from docs.reference_cases.ge_beam3_loaded_factor_split_probe import split_accepted_elastic_operator
from anyfesolver._ge_beam3_loaded_modal import prepare
from anyfesolver._ge_beam3_p5_loads.core import canonical
from test_ge_beam3_native_load_state import problem
from test_ge_beam3_loaded_modal import accepted, straight_model
from test_ge_beam3_curved_p5_mass_probe import section_mass


@pytest.mark.parametrize('case', ['virgin_curved', 'loaded_curved', 'tension', 'compression'])
def test_retained_split_reconstructs_actual_native_hessian(case, tmp_path):
    model = problem() if 'curved' in case else straight_model()
    if case == 'virgin_curved':
        element = model.mesh.elements[1]; parameter = 0.; total = np.zeros(18)
        states = {1: element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    else:
        nodal = ((3, .5 if case == 'tension' else -.5, 0., 0.),) if 'curved' not in case else ()
        result, states, _ = accepted(model, nodal=nodal)
        total, parameter = result.displacements, result.parameter
    before = canonical(states); element = model.mesh.elements[1]
    packet, _ = prepare(model, states, total, {1: section_mass()}, load_parameter=parameter)
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    factor, geometric = split_accepted_elastic_operator(element, inner)
    reconstructed = factor.T@factor+geometric
    assert np.linalg.norm(reconstructed-packet.stiffness) <= 1e-11*max(1., np.linalg.norm(packet.stiffness))
    assert np.linalg.norm(geometric-geometric.T) <= 1e-11*max(1., np.linalg.norm(geometric))
    assert canonical(states) == before
    # The native curved reference replay has roundoff-level station stress.
    # Preserve it rather than clipping it to manufacture an exact zero.
    if case == 'virgin_curved': assert np.linalg.norm(geometric) <= 1e-11
    else: assert np.linalg.norm(geometric) > 1e-5
    with (tmp_path/'split.json').open('xb') as stream:
        stream.write(canonical(dict(case=case, factor=factor, geometric=geometric,
            native_stiffness=packet.stiffness, production_qualified=False)))


def test_active_plastic_increment_is_not_silently_given_elastic_spectrum():
    model = problem(plastic=True); result, states, _ = accepted(model)
    element = model.mesh.elements[1]
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    with pytest.raises(ValueError, match='elastic accepted'):
        split_accepted_elastic_operator(element, inner)
