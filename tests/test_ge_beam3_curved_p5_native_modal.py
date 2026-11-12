"""Native FEModel reference modal integration; not engineering qualification."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._native_reference_modal import solve_reference_modal
from docs.reference_cases.ge_beam3_curved_p5_native_driver_probe import DriverP5Element
from docs.reference_cases.ge_beam3_curved_p5_native_modal_probe import solve, prepare
from docs.reference_cases.ge_beam3_curved_p5_native_material_probe import canonical
from docs.reference_cases.ge_beam3_curved_p5_continuum_probe import parabolic_references
from docs.reference_cases.ge_beam3_curved_p5_modal_chain_probe import clamped_full_inertia_chain
from docs.reference_cases.ge_beam3_curved_p5_mass_probe import symmetric_modes, reference_kinetic_factors
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from test_ge_beam3_curved_p5_mass_probe import section_mass
from test_ge_beam3_curved_p5_algebra_probe import assert_scaled_close


def problem(count=1, *, clamped=True, height=.4, reverse_insert=False):
    model = FEModel('private-native-modal')
    refs = parabolic_references(height, count)
    node_ids = (11, 23, 47, 61, 83)[:2*count+1]
    points = np.vstack((refs[0].coordinates, *(r.coordinates[1:] for r in refs[1:])))
    for i, point in zip(node_ids, points): model.add_node(i, *point)
    elements = []
    for index, ref in enumerate(refs):
        element = DriverP5Element(10+10*index, tuple(node_ids[2*index:2*index+3]), ref, law(), order=24)
        elements.append(element)
    for element in (elements[::-1] if reverse_insert else elements):
        model.add_element(element.element_id, element)
        model.materials[element.material_name] = element.core.section
    if clamped:
        model.add_boundary_condition(BoundaryCondition('root', [11],
            {name: 0. for name in ('ux','uy','uz','rx','ry','rz')}))
    return model, refs, {e.element_id: section_mass() for e in elements}


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('height', [0., .4])
def test_native_clamped_full_inertia_matches_preserved_pencil(count, height):
    model, refs, inertia = problem(count, height=height)
    made = solve(model, inertia, num_modes=6)
    section = next(iter(model.mesh.elements.values())).core.section._elastic
    expected = clamped_full_inertia_chain(refs, section, section_mass())
    values, _ = symmetric_modes(expected.stiffness, expected.mass)
    assert_scaled_close(made.full_stiffness, expected.full_stiffness)
    assert_scaled_close(made.full_mass, expected.full_mass)
    assert_scaled_close(made.eigenvalues, values[:6])
    assert made.nodal_modes.shape == (6*(2*count+1), 6)
    assert len(made.internal_modes) == count
    assert all(mode.shape == (6, 6) for _, mode in made.internal_modes)
    assert len(made.algebraic_dofs) == 6*count
    assert made.dynamic_map.shape[1] == 12*count
    assert np.array_equal(made.nodal_modes[:6], np.zeros((6, 6)))
    assert not made.production_qualified
    assert made.normalized_residual <= 1e-11


def test_shared_trace_is_balanced_after_global_assembly_not_per_element():
    model, _, inertia = problem(2)
    blocks, guard = prepare(model, inertia)
    made = solve_reference_modal(model, blocks, check_inputs=guard, num_modes=6)
    incident = []
    for index, block in enumerate(blocks):
        local = np.vstack((made.nodal_modes[list(block.nodal_dofs)], made.internal_modes[index][1]))
        force = block.elastic_factor.T @ (block.elastic_factor @ local)
        incident.append(force[15:18] if index == 0 else force[3:6])
    assert np.linalg.norm(incident[0]) > 1e-4
    assert_scaled_close(incident[0] + incident[1], np.zeros((3, 6)))


@pytest.mark.parametrize('count', [1, 2])
def test_free_body_six_rigid_modes_and_all_cell_inertia(count):
    model, _, inertia = problem(count, clamped=False)
    made = solve(model, inertia, num_modes=9)
    scale = np.linalg.norm(made.full_stiffness)
    assert np.all(np.abs(made.eigenvalues[:6]) < 1e-11*scale)
    assert np.all(made.eigenvalues[6:] > 1e-11*scale)
    assert made.dynamic_map.shape[1] == 12*count+3
    assert_scaled_close(made.full_modes.T @ made.full_mass @ made.full_modes, np.eye(9))
    for _, internal in made.internal_modes:
        assert np.linalg.norm(internal[:, :6]) > 0.


def test_all_nodal_coordinates_supported_still_retains_cell_spin_spectrum():
    model, _, inertia = problem()
    model.boundary_conditions[0].node_ids = [11, 23, 47]
    made = solve(model, inertia)
    assert made.dynamic_map.shape == (24, 6)
    assert np.all(made.eigenvalues > 0.)
    assert np.array_equal(made.nodal_modes, np.zeros((18, 6)))
    assert np.linalg.norm(made.internal_modes[0][1]) > 0.


def test_deterministic_owned_output_and_no_mass_api_fallback():
    model, _, inertia = problem(2)
    first = solve(model, inertia)
    other, _, other_inertia = problem(2, reverse_insert=True)
    second = solve(other, other_inertia)
    assert canonical(first) == canonical(second)
    with pytest.raises(ValueError): first.full_modes.setflags(write=True)
    with pytest.raises(ValueError): first.internal_modes[0][1][0, 0] = 0.
    for element in model.mesh.elements.values():
        with pytest.raises(ValueError, match='not yet authorized'):
            element.compute_mass_matrix(model.mesh, element.core.section)


@pytest.mark.parametrize('target', ['geometry', 'section', 'boundary', 'ownership', 'block', 'block_hash'])
def test_bound_model_and_factor_mutations_fail(target):
    model, _, inertia = problem()
    blocks, guard = prepare(model, inertia)
    element = model.mesh.elements[10]
    if target == 'geometry': model.mesh.nodes[23].x += .01
    elif target == 'section': element.core.section._elastic[0, 0] += .01
    elif target == 'boundary': model.boundary_conditions[0].dof_constraints.pop('rx')
    elif target == 'ownership': model.materials[element.material_name] = law()
    elif target == 'block': blocks = (replace(blocks[0], elastic_factor=blocks[0].elastic_factor*1.01),)
    else: blocks = (replace(blocks[0], identity='0'*64),)
    with pytest.raises(ValueError): solve_reference_modal(model, blocks, check_inputs=guard)


@pytest.mark.parametrize('cancel_stage', ['algebraic_equilibrium', 'output'])
def test_cancellation_before_factors_and_before_output(cancel_stage):
    model, _, inertia = problem()
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled): solve(model, inertia, cancellation_token=token)
    token = CancellationToken()
    blocks, original = prepare(model, inertia)
    def guard(stage, observed):
        original(stage, observed)
        if stage == cancel_stage: token.cancel()
    with pytest.raises(SolveCancelled):
        solve_reference_modal(model, blocks, check_inputs=guard, cancellation_token=token)


@pytest.mark.parametrize('target', ['nonzero_support', 'mpc', 'activity', 'point_mass', 'missing_inertia', 'bad_inertia', 'too_many_modes'])
def test_unsupported_inputs_fail_closed(target):
    model, _, inertia = problem()
    if target == 'nonzero_support': model.boundary_conditions[0].dof_constraints['ux'] = .01
    elif target == 'mpc': model.constraint_equations = [object()]
    elif target == 'activity': model.mesh.element_activity = object()
    elif target == 'point_mass': model.add_point_mass(47, 1.)
    elif target == 'missing_inertia': inertia = {}
    elif target == 'bad_inertia': inertia[10][0, 0] = -1.
    with pytest.raises(ValueError): solve(model, inertia, num_modes=256 if target == 'too_many_modes' else 6)


def test_declared_mass_kernel_and_layout_are_not_inferred_from_eigenvalues():
    model, _, inertia = problem()
    blocks, _ = prepare(model, inertia)
    block = blocks[0]; bad = block.kinetic_factor.copy(); bad[0, 3] = 1e-100
    with pytest.raises(ValueError, match='exactly zero'): replace(block, kinetic_factor=bad)
    with pytest.raises(ValueError): replace(block, nodal_dofs=block.nodal_dofs[:-1]+(0,))
    with pytest.raises(ValueError): replace(block, identity='not-a-hash')


def test_stiffness_condensation_is_not_used_for_dynamic_internal_coordinates():
    model, refs, inertia = problem()
    element = model.mesh.elements[10]
    factors = reference_kinetic_factors(refs[0], element.core.section._elastic, inertia[10], order=24)
    made = solve(model, inertia)
    for _, internal in made.internal_modes:
        guyan = factors.static_map[18:] @ made.nodal_modes
        assert np.linalg.norm(internal-guyan) > 1e-3


def test_packet_stiffness_matches_uneliminated_stationary_moment_schur():
    from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import reference_hessians
    model, _, inertia = problem()
    element = model.mesh.elements[10]
    full, _, _ = reference_hessians(element.core.reference, element.core.section._elastic, 24)
    expected = full[:24, :24] - full[:24, 24:] @ np.linalg.solve(full[24:, 24:], full[24:, :24])
    blocks, _ = prepare(model, inertia)
    factor = blocks[0].elastic_factor
    assert_scaled_close(factor.T @ factor, expected)


def test_packet_kinetic_energy_matches_direct_station_field_without_static_map():
    model, refs, inertia = problem()
    blocks, _ = prepare(model, inertia)
    velocity = np.sin(np.arange(24)+.2)
    nodal, spin = velocity[:18].reshape(3, 6), velocity[18:].reshape(2, 3)
    ref = refs[0]; total = 0.
    points, weights = np.polynomial.legendre.leggauss(48)
    for cell in (0, 1):
        for point, weight in zip(points, weights):
            t = (point+1)/2; xi = cell-1+t
            offset = ref.position(xi)-(1-t)*ref.coordinates[cell]-t*ref.coordinates[cell+1]
            v = (1-t)*nodal[cell, :3]+t*nodal[cell+1, :3]+np.cross(spin[cell], offset)
            frame = ref.frame(xi)
            material = np.r_[frame.T@v, frame.T@spin[cell]]
            total += weight*ref.jacobian(xi)*(material@inertia[10]@material)/4
    made = blocks[0].kinetic_factor@velocity
    assert abs(made@made/2-total) <= 1e-11*max(1., abs(total))


def test_caller_inertia_is_copied_not_a_live_mutable_input():
    model, _, inertia = problem()
    blocks, guard = prepare(model, inertia)
    before = solve_reference_modal(model, blocks, check_inputs=guard)
    inertia[10][:] = np.nan
    after = solve_reference_modal(model, blocks, check_inputs=guard)
    assert canonical(before) == canonical(after)


def test_no_research_mechanics_imports_in_native_assembler():
    import ast
    from pathlib import Path
    import anyfesolver._native_reference_modal as native
    tree = ast.parse(Path(native.__file__).read_text(encoding='utf-8'))
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert set(imports) == {'dataclasses', 'scipy', 'assembly', 'control'}
    assert not any('reference_cases' in name for name in imports)
