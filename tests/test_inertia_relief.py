"""Analytical mechanics and adversarial algebra for the public relief solve."""
import json

import numpy as np
import pytest

from anyfesolver import (BeamElement, FEModel, GeneralizedBeamSection, InertiaReliefError,
                       LoadCase, solve_inertia_relief)
from anyfesolver.matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix


def bar(n=3, consistent=True):
    model = FEModel("relief-bar")
    model.add_material("steel", 1.e7, .3, density=5.)
    for i, x in enumerate(np.linspace(0., 2., n+1), 1):
        model.add_node(i, x, 0., 0.)
    for i in range(1, n+1):
        model.add_element(i, BeamElement(i, [i, i+1], "steel",
            {"area": .2, "Iy": .001, "Iz": .002, "J": .001,
             "consistent_mass": consistent}))
    return model


def load_vector(model, values):
    load = LoadCase("vector")
    for node in model.mesh.nodes.values():
        load.add_nodal_load(node.id, np.asarray(values)[node.dofs])
    return load


def rigid(model, origin):
    """Independent kinematics: each angular column is omega cross r."""
    matrix = np.zeros((model.mesh.dof_manager.total_dofs, 6))
    for node in model.mesh.nodes.values():
        matrix[np.ix_(node.dofs[:3], range(3))] = np.eye(3)
        for j in range(3):
            matrix[node.dofs[:3], j+3] = np.cross(np.eye(3)[j], node.coords()-origin)
        matrix[np.ix_(node.dofs[3:], range(3, 6))] = np.eye(3)
    return matrix


@pytest.mark.parametrize("consistent", [True, False])
def test_prescribed_rigid_acceleration_has_no_elastic_response(consistent):
    model = bar(consistent=consistent)
    M, _ = assemble_mass_matrix(model)
    origin = np.array([1., 0., 0.])
    a = np.array([.3, -.7, .1, .2, .4, -.5])
    expected = rigid(model, origin) @ a
    result = solve_inertia_relief(model, load_vector(model, M @ expected))
    assert result.converged, result.to_dict()
    np.testing.assert_allclose(result.acceleration_at_center_of_mass, a[:3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.angular_acceleration, a[3:], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.rigid_acceleration, expected, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(result.displacements) < 1e-14
    assert result.diagnostics["force_balance_error"] <= 1e-10
    assert result.diagnostics["moment_balance_error"] <= 1e-10
    recovered = result.recover()
    assert recovered.reactions == {}
    assert recovered.element_stresses
    json.dumps(result.to_dict(), allow_nan=False)
    assert not result.displacements.flags.writeable


@pytest.mark.parametrize("consistent", [True, False])
def test_uniform_bar_end_force_matches_analytical_axial_solution(consistent):
    model = bar(n=4, consistent=consistent)
    load = LoadCase("end-force")
    load.add_nodal_load(5, forces=[12., 0., 0.])
    result = solve_inertia_relief(model, load)
    assert result.converged, result.to_dict()
    np.testing.assert_allclose(result.acceleration_at_center_of_mass, [6., 0., 0.], atol=1e-10)
    x = np.linspace(0., 2., 5)
    u = 12.*x*x/(2.*1.e7*.2*2.)
    u -= np.mean(u)
    np.testing.assert_allclose(result.displacements[::6], u, rtol=1e-8, atol=1e-13)
    forces = 1.e7*.2*np.diff(result.displacements[::6])/.5
    np.testing.assert_allclose(forces, 12.*(x[:-1]+.25)/2., rtol=1e-8, atol=1e-9)


def test_torque_uses_section_polar_inertia_not_torsional_stiffness_constant():
    model = bar()
    load = LoadCase("torque")
    load.add_nodal_load(4, moments=[.3, 0., 0.])
    result = solve_inertia_relief(model, load)
    assert result.converged, result.to_dict()
    # J_mass=rho*(Iy+Iz)*L; J_stiffness=.001 is deliberately different.
    np.testing.assert_allclose(result.angular_acceleration, [.3/(5.*.003*2.), 0., 0.], atol=1e-10)
    np.testing.assert_allclose(result.acceleration_at_center_of_mass, 0., atol=1e-11)


def test_eccentric_generalized_spatial_inertia_and_reference_transform():
    model = FEModel("offset-section")
    model.add_material("dummy", 1.e7, .3, density=999.)
    model.add_node(1, 0., 0., 0.)
    model.add_node(2, 2., 0., 0.)
    offset = np.array([0., .2, -.1])
    h = 10.*offset
    skew = np.array([[0., -h[2], h[1]], [h[2], 0., -h[0]], [-h[1], h[0], 0.]])
    j = np.diag([1., 2., 3.])+10.*((offset @ offset)*np.eye(3)-np.outer(offset, offset))
    section = GeneralizedBeamSection(stiffness=np.diag([1e7,2e6,3e6,4e5,5e5,6e5]),
                                     mass_matrix=np.block([[10.*np.eye(3), -skew], [skew, j]]))
    model.add_element(1, BeamElement(1, [1, 2], "dummy", section=section))
    load = LoadCase("eccentric-force")
    load.add_nodal_load(2, forces=[3., 4., 5.], moments=[.6, .7, .8])
    result = solve_inertia_relief(model, load, reference_point=[0., 0., 0.])
    assert result.converged, result.to_dict()
    center = np.array([1., .2, -.1])
    np.testing.assert_allclose(result.center_of_mass, center, atol=1e-12)
    inertia = np.diag([2., 4.+10.*2.**3/12., 6.+10.*2.**3/12.])
    alpha = np.linalg.solve(inertia, np.cross(np.array([2.,0.,0.])-center, [3.,4.,5.])+[.6,.7,.8])
    acom = np.array([3.,4.,5.])/20.
    np.testing.assert_allclose(result.angular_acceleration, alpha, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.acceleration_at_reference, acom+np.cross(alpha,-center), rtol=1e-10, atol=1e-12)
    other = solve_inertia_relief(model, load, reference_point=[4., -3., 2.])
    assert other.converged
    np.testing.assert_allclose(result.inertial_loads, other.inertial_loads, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.displacements, other.displacements, rtol=1e-8, atol=1e-12)
    a = np.r_[result.acceleration_at_reference, result.angular_acceleration]
    np.testing.assert_allclose(result.rigid_body_mass_matrix @ a,
                               rigid(model, np.zeros(3)).T @ result.applied_loads, atol=1e-10)


def test_zero_load_and_self_equilibrated_load():
    model = bar()
    zero = solve_inertia_relief(model)
    assert zero.converged, zero.to_dict()
    np.testing.assert_array_equal(zero.displacements, 0.)
    load = LoadCase("tension")
    load.add_nodal_load(1, forces=[-2.,0.,0.])
    load.add_nodal_load(4, forces=[2.,0.,0.])
    result = solve_inertia_relief(model, load)
    assert result.converged, result.to_dict()
    np.testing.assert_allclose(result.angular_acceleration, 0., atol=1e-12)
    np.testing.assert_allclose(result.acceleration_at_reference, 0., atol=1e-12)
    np.testing.assert_allclose(np.diff(result.displacements[::6]), 2./(1e7*.2)*2./3., rtol=1e-8)


class AxialOnlyBeam(BeamElement):
    def compute_stiffness_matrix(self, mesh, material):
        matrix = np.zeros((12, 12))
        matrix[0, 0] = matrix[6, 6] = 1.
        matrix[0, 6] = matrix[6, 0] = -1.
        return matrix


class ReleasedBeam(BeamElement):
    def compute_stiffness_matrix(self, mesh, material):
        matrix = super().compute_stiffness_matrix(mesh, material)
        values, vectors = np.linalg.eigh(matrix)
        index = np.flatnonzero(values > np.max(values)*1e-8)[0]
        return matrix-values[index]*np.outer(vectors[:, index], vectors[:, index])


class NegativeBeam(BeamElement):
    def compute_stiffness_matrix(self, mesh, material):
        matrix = super().compute_stiffness_matrix(mesh, material)
        values, vectors = np.linalg.eigh(matrix)
        index = np.flatnonzero(values > np.max(values)*1e-8)[0]
        return matrix-2.*values[index]*np.outer(vectors[:, index], vectors[:, index])


@pytest.mark.parametrize("element_type", [AxialOnlyBeam, ReleasedBeam, NegativeBeam])
def test_unloaded_extra_mechanism_is_not_accepted(element_type):
    model = bar(n=1)
    old = model.mesh.elements[1]
    model.mesh.elements[1] = element_type(1, [1,2], "steel", dict(old.cross_section))
    result = solve_inertia_relief(model)
    assert result.status == "unresolved_elastic_stiffness", result.to_dict()
    assert result.displacements is None
    with pytest.raises(InertiaReliefError, match="non-successful"):
        result.recover()


@pytest.mark.parametrize("mass", [-1., np.nan, np.inf])
def test_direct_invalid_point_mass_is_rejected(mass):
    model = bar()
    model.mesh.point_masses[1] = mass
    with pytest.raises(InertiaReliefError):
        solve_inertia_relief(model)


def test_point_mass_missing_node_and_zero_total_mass_are_rejected():
    model = bar()
    model.mesh.point_masses[99] = 1.
    with pytest.raises(InertiaReliefError, match="missing node"):
        solve_inertia_relief(model)
    model.mesh.point_masses.clear()
    model.materials["steel"].density = 0.
    model.bump_revision("material")
    with pytest.raises(InertiaReliefError, match="positive total mass"):
        solve_inertia_relief(model)


def test_nonphysical_generalized_inertia_is_rejected():
    model = bar(n=1)
    section = GeneralizedBeamSection(stiffness=np.eye(6)*1.e6,
                                     mass_matrix=np.diag([1.,2.,3.,1.,1.,1.]))
    model.mesh.elements[1] = BeamElement(1, [1,2], "steel", section=section)
    with pytest.raises(InertiaReliefError, match="scalar mass times"):
        solve_inertia_relief(model)


@pytest.mark.parametrize("reference", [[1.,2.], [0.,0.,np.nan], [[0.,0.,0.]]])
def test_invalid_reference_fails_early(reference):
    with pytest.raises(InertiaReliefError, match="reference_point"):
        solve_inertia_relief(bar(), reference_point=reference)


def test_singular_rotational_inertia_is_not_pseudoinverted():
    class NoRotaryMass(BeamElement):
        def compute_mass_matrix(self, mesh, material):
            matrix = super().compute_mass_matrix(mesh, material).copy()
            rotations = [3,4,5,9,10,11]
            matrix[rotations, :] = 0.
            matrix[:, rotations] = 0.
            return matrix
    model = bar(n=1)
    model.mesh.elements[1] = NoRotaryMass(1, [1,2], "steel", dict(model.mesh.elements[1].cross_section))
    with pytest.raises(InertiaReliefError, match="singular"):
        solve_inertia_relief(model)


def test_refined_beam_keeps_converged_spectral_classification():
    result = solve_inertia_relief(bar(n=512))
    assert result.converged, result.to_dict()
    mechanism = result.diagnostics["mechanism"]
    assert mechanism["elastic_dimension"] == 3072
    assert mechanism["eigenvalue_lower_margin"] > 1e-12
    assert mechanism["eigenpair_residual"] <= 1e-13


@pytest.mark.parametrize("element_type", [ReleasedBeam, NegativeBeam])
def test_unloaded_mechanism_in_refined_body_is_never_accepted(element_type):
    model = bar(n=512)
    old = model.mesh.elements[256]
    model.mesh.elements[256] = element_type(256, list(old.node_ids), "steel", dict(old.cross_section))
    result = solve_inertia_relief(model)
    assert result.status == "unresolved_elastic_stiffness", result.to_dict()
    assert result.displacements is None


@pytest.mark.parametrize("amplitude", [1e-9, 1e12])
def test_non_gravity_rigid_load_amplitude_is_scale_invariant(amplitude):
    model = bar()
    M, _ = assemble_mass_matrix(model)
    a = amplitude*np.array([.3, -.7, .1, .2, .4, -.5])
    acceleration = rigid(model, np.array([1.,0.,0.])) @ a
    result = solve_inertia_relief(model, load_vector(model, M @ acceleration))
    assert result.converged, result.to_dict()
    np.testing.assert_allclose(result.acceleration_at_center_of_mass, a[:3], rtol=1e-10, atol=amplitude*1e-12)
    np.testing.assert_allclose(result.angular_acceleration, a[3:], rtol=1e-10, atol=amplitude*1e-12)
