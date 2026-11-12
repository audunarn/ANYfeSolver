"""Native accepted states, private signed kernel and physical mode vectors."""

import numpy as np
import pytest
from anyfesolver._ge_beam3_loaded_modal import prepare
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver._native_signed_factor_modes import solve_signed_factor_modes
from anyfesolver._ge_beam3_signed_loaded_modes import solve_signed_loaded_modes
from docs.reference_cases.ge_beam3_loaded_factor_split_probe import split_accepted_elastic_operator
from docs.reference_cases.ge_beam3_straight_discrete_signed_reference import bending_roots
from test_ge_beam3_signed_loaded_spectrum import make
from test_ge_beam3_loaded_modal import accepted


@pytest.mark.parametrize('axial', [-1., .5])
def test_loaded_slender_modes_preserve_physical_mass_and_signed_reference(axial, tmp_path):
    model, inertia = make(1000000.)
    result, states, force = accepted(model, nodal=((3, axial, 0., 0.),))
    saved = canonical(states)
    packet, guard = prepare(model, states, result.displacements, {1: inertia}, load_parameter=result.parameter)
    e = model.mesh.elements[1]
    inner = e.validate_model_bound_nonlinear_state(model.mesh, e.core.section, states[1], 1)['material_state']
    f, g = split_accepted_elastic_operator(e, inner)
    modes = solve_signed_factor_modes(f, g, packet.mass, packet.free_dofs, packet.algebraic_dofs,
        bounds=(-10., 100.), num_modes=4)
    native_packet, native = solve_signed_loaded_modes(model, states, result.displacements,
        {1: inertia}, force, load_parameter=result.parameter, bounds=(-10., 100.), num_modes=4)
    assert canonical(native.eigenvalues) == canonical(modes.eigenvalues)
    assert canonical(native.full_modes) == canonical(modes.full_modes)
    assert native.operator_identity == native_packet.identity == packet.identity
    expected = np.repeat(bending_roots(1000000, axial), 2)
    np.testing.assert_allclose(modes.eigenvalues, expected, atol=1e-9, rtol=1e-11)
    np.testing.assert_allclose(modes.full_modes.T@packet.mass@modes.full_modes, np.eye(4), atol=1e-11, rtol=1e-11)
    assert np.all(np.sign(modes.eigenvalues) == np.sign(expected))
    assert modes.spectral_residual <= 1e-11
    assert canonical(states) == saved; guard()
    with (tmp_path/'signed-modes.json').open('xb') as stream:
        stream.write(canonical(dict(axial=axial, eigenvalues=modes.eigenvalues, modes=modes.full_modes,
            spectral_residual=modes.spectral_residual, backward_residual=modes.full_backward_residual,
            packet_identity=packet.identity, production_qualified=False)))
