"""Default globalization must not cripple Newton on non-monotone-convergent elastic problems."""
import numpy as np
import pytest

import anyfesolver as fs
from anyfesolver.boundary import FixedSupport, LoadCase
from anyfesolver.elements import BeamElement


def _cantilever(elements, axis, load):
    model = fs.FEModel("cantilever")
    model.add_material("mat", 210e9, 0.3, density=7850.0)
    for i in range(elements + 1):
        model.add_node(i + 1, i / elements, 0.0, 0.0)
    for i in range(elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(
            area=1e-4, Iy=8.33e-10, Iz=8.33e-10, J=1.67e-9, orientation=(0.0, 0.0, 1.0))))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    case = LoadCase("tip")
    vector = np.zeros(6)
    vector[axis] = load
    case.add_nodal_load(elements + 1, vector)
    model.apply_boundary_conditions()
    return model, case


@pytest.mark.parametrize("elements,axis,load", [(1, 2, 500.0), (4, 2, 500.0), (4, 4, 200.0)])
def test_default_settings_converge_without_crawling(elements, axis, load):
    model, case = _cantilever(elements, axis, load)
    stepped = fs.solve_static_nonlinear(model, case, num_steps=5, max_iterations=25)
    single = fs.solve_static_nonlinear(model, case, num_steps=1, max_iterations=25)
    assert stepped.converged and stepped.load_factor == 1.0
    # Previously the eager residual-decrease line search needed 565 to 47521
    # Newton iterations here; plain Newton with rescue needs a few dozen.
    assert stepped.info["total_newton_iterations"] < 60
    np.testing.assert_allclose(stepped.displacements, single.displacements, rtol=1e-6, atol=1e-9)
