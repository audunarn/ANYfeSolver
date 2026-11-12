"""Reference-frame/node-numbering controls, not finite-motion admission."""
import importlib
import json
import os
from pathlib import Path

import numpy as np

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSurface, ContactSolverSettings, LoadCase, create_shell_element,
    solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import TriangulatedRigidContactBody
from anyfesolver.general_contact.triangulated_rigid_static import TriangulatedRigidManager

R = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
SHIFT = np.array((1/8, -1/4, 1/2))
FORCE_SCALE = .15  # N, declared three .05 N nodal loads.
MOMENT_SCALE = .15  # N*m, same load and 1 m tool footprint scale.
DISPLACEMENT_SCALE = 1.e-5  # m, declared initial physical clearance.
WORK_RATE_SCALE = FORCE_SCALE*1.+MOMENT_SCALE*1.  # J per unit path parameter.


def _case(transformed, renumbered=False):
    rotation = R if transformed else np.eye(3)
    shift = SHIFT if transformed else np.zeros(3)
    numbering = {1: 41, 2: 7, 3: 105, 4: 23} if renumbered else {i: i for i in range(1, 5)}
    positions = ((.2, .15, .00501), (.85, .2, .00501),
                 (.3, .8, .00501), (.8, .85, .00501))
    model = FEModel('reference_invariance_patch')
    model.add_material('elastic', 1.e5, .3, density=1.)
    for original in ((3, 1, 4, 2) if renumbered else (1, 2, 3, 4)):
        model.add_node(numbering[original], *(rotation@positions[original-1]+shift))
    normal = tuple(rotation@np.array((0., 0., 1.)))
    for eid, nodes in ((1, (1, 2, 3)), (2, (2, 4, 3))):
        model.add_element(eid, create_shell_element(eid, tuple(numbering[i] for i in nodes),
            'elastic', thickness=.01, formulation='e4-pl-s3-v2d', reference_normal=normal))
    normal_axis = int(np.argmax(np.abs(normal)))
    translation_names = ('ux', 'uy', 'uz')
    guide = {translation_names[i]: 0. for i in range(3) if i != normal_axis}
    guide.update(dict.fromkeys(('rx', 'ry', 'rz'), 0.))
    model.add_boundary_condition(BoundaryCondition('guide', tuple(numbering.values()), guide))
    model.add_boundary_condition(BoundaryCondition('root', (numbering[1],),
        {translation_names[normal_axis]: 0.}))
    vertices = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))
    tool = TriangulatedRigidContactBody('tool',
        tuple(tuple(rotation@point+shift) for point in vertices),
        ((1, (0, 1, 2)), (2, (0, 2, 3))))
    definition = ContactDefinition((ContactSurface('skin', (1, 2), 'bottom'),),
        (tool,), (ContactInteraction('patch_tool', 'skin', 'tool', 1.e6, 5.e-5),))
    settings = ContactSolverSettings(pair_motion='general', normal_discretization='primal_mortar',
        general_s3_difference_step=1.e-6)
    return model, definition, settings, numbering, rotation, shift


def _map(model, moved, numbering, rotation):
    G = np.zeros((moved.mesh.dof_manager.total_dofs, model.mesh.dof_manager.total_dofs))
    for original in range(1, 5):
        old = model.mesh.dof_manager.get_node_dofs(original)
        new = moved.mesh.dof_manager.get_node_dofs(numbering[original])
        for start in (0, 3):
            G[np.ix_(new[start:start+3], old[start:start+3])] = rotation
    assert np.array_equal(G.T@G, np.eye(len(G)))
    return G


def _wrench_errors(old, new, rotation=R, shift=SHIFT):
    errors = {}
    for prefix in ('integration', 'target'):
        force = rotation@np.array(old[prefix+'_force'])
        moment = rotation@np.array(old[prefix+'_moment'])+np.cross(shift, force)
        errors[prefix+'_force'] = float(np.linalg.norm(force-new[prefix+'_force'])/FORCE_SCALE)
        errors[prefix+'_moment'] = float(np.linalg.norm(moment-new[prefix+'_moment'])/MOMENT_SCALE)
    return errors


def _equilibrium(owner, result, captured, numbering, rotation):
    residual = captured['residual']
    normal_axis = int(np.argmax(abs(rotation@np.array((0., 0., 1.)))))
    force, moment = np.zeros(3), np.zeros(3)
    for original, node in numbering.items():
        ids = owner.mesh.dof_manager.get_node_dofs(node)
        reference = owner.mesh.nodes[node]
        position = np.array((reference.x, reference.y, reference.z))+result.displacements[ids[:3]]
        support = residual[ids].copy()
        if original != 1:
            support[normal_axis] = 0.
        force += support[:3]
        moment += support[3:]+np.cross(position, support[:3])
        if original != 1:
            load = rotation@np.array((0., 0., -.05))
            force += load
            moment += np.cross(position, load)
    contact = result.info['general_contact']
    force += contact['integration_force']
    moment += contact['integration_moment']
    return dict(force=float(np.linalg.norm(force)/FORCE_SCALE),
                moment=float(np.linalg.norm(moment)/MOMENT_SCALE))


def _rows(vector, translation, rotation):
    return dict(force=float(np.linalg.norm(vector[translation])/FORCE_SCALE),
                moment=float(np.linalg.norm(vector[rotation])/MOMENT_SCALE))


def test_connected_patch_reference_and_numbering_invariance(monkeypatch):
    baseline = _case(False)
    variants = [baseline, _case(True), _case(False, True), _case(True, True)]
    transformed = variants[-1]
    mappings = [_map(baseline[0], entry[0], entry[3], entry[4]) for entry in variants]
    G = mappings[-1]
    from scipy import sparse
    from anyfesolver.assembly import build_constraint_transformation
    charts = []
    for model, *_ in variants:
        model.apply_boundary_conditions()
        n = model.mesh.dof_manager.total_dofs
        _, _, T, _, _, _ = build_constraint_transformation(sparse.eye(n, format='csr'), np.zeros(n), model)
        charts.append(T.toarray())
    for mapping, chart in zip(mappings, charts):
        assert np.array_equal(chart@chart.T, mapping@charts[0]@charts[0].T@mapping.T)
    solver = importlib.import_module('anyfesolver.nonlinear_static')
    original_recovery = solver._support_reaction_resultants
    captured = {}
    def capture(owner, imbalance, **kwargs):
        captured.update(model=owner, residual=np.array(imbalance, copy=True))
        return original_recovery(owner, imbalance, **kwargs)
    monkeypatch.setattr(solver, '_support_reaction_resultants', capture)
    results, equilibrium = [], []
    for model, definition, settings, numbers, rotation, shift in variants:
        load = LoadCase('patch_press')
        for node in (2, 3, 4):
            load.add_nodal_load(numbers[node], forces=tuple(rotation@np.array((0., 0., -.05))))
        result = solve_static_nonlinear(model, load, contact=definition, contact_settings=settings,
            kinematics='von_karman', num_steps=2, max_iterations=15)
        assert result.status == 'completed', result.failure_reason
        assert result.info['general_contact']['active_pressure_dofs'] > 0
        assert result.info['general_contact']['gap_tolerance_met']
        results.append(result)
        equilibrium.append(_equilibrium(captured['model'], result, captured, numbers, rotation))
    old, new = results[0], results[-1]
    response_error = float(np.linalg.norm(new.displacements-G@old.displacements)/DISPLACEMENT_SCALE)
    wrench = _wrench_errors(old.info['general_contact'], new.info['general_contact'])
    independent_comparisons = []
    for label, entry, result, mapping in zip(('reference', 'frame_only', 'numbering_only', 'combined'), variants, results, mappings):
        independent_comparisons.append(dict(case=label,
            response_error=float(np.linalg.norm(result.displacements-mapping@old.displacements)/DISPLACEMENT_SCALE),
            wrench_errors=_wrench_errors(old.info['general_contact'], result.info['general_contact'], entry[4], entry[5])))
    # Complete FE skin-map work/H, at a distinct active partial-footprint pose.
    managers = [TriangulatedRigidManager(m, d, s) for m, d, s, *_ in (baseline, transformed)]
    q = np.zeros(len(old.displacements))
    v = np.zeros_like(q)
    for node, dz in zip((1, 2, 3, 4), (.1, -.2, .3, .4)):
        ids = baseline[0].mesh.dof_manager.get_node_dofs(node)
        q[ids[0]] = .22
        q[ids[2]] = -3.e-5
        q[ids[4]] = .001
        v[ids[:3]] = (.1, -.07, dz)
        v[ids[3:6]] = (.03, -.02, .01)
    r, K, summary = managers[0].evaluate(q)
    r1, K1, summary1 = managers[1].evaluate(G@q)
    summary, summary1 = summary.to_dict(), summary1.to_dict()
    translation = [i for node in range(1, 5) for i in baseline[0].mesh.dof_manager.get_node_dofs(node)[:3]]
    rotation = [i for node in range(1, 5) for i in baseline[0].mesh.dof_manager.get_node_dofs(node)[3:6]]
    residual_error = G.T@r1-r
    tangent_error = G.T@K1.toarray()@G-K.toarray()
    energy_scale = .5*1.e6*(2.e-5)**2  # J at declared 1m²/20um scales.
    local = dict(energy=float(abs(summary['penalty_energy']-summary1['penalty_energy'])/energy_scale),
        residual=_rows(residual_error, translation, rotation),
        tangent_blocks={
            'force_translation_N_per_m':float(np.linalg.norm(tangent_error[np.ix_(translation, translation)])/FORCE_SCALE),
            'force_rotation_N_per_rad':float(np.linalg.norm(tangent_error[np.ix_(translation, rotation)])/FORCE_SCALE),
            'moment_translation_Nm_per_m':float(np.linalg.norm(tangent_error[np.ix_(rotation, translation)])/MOMENT_SCALE),
            'moment_rotation_Nm_per_rad':float(np.linalg.norm(tangent_error[np.ix_(rotation, rotation)])/MOMENT_SCALE)},
        current_overlap_area_m2=float(summary['current_overlap_area']),
        partial_footprint_boundary_guard='ACTUAL_MANAGER_EVALUATE_PASSED',
        wrenches=_wrench_errors(summary, summary1))
    work = []
    for h in (1.e-7, 1.e-8):
        values = [managers[0].evaluate(q+sign*h*v) for sign in (-1, 1)]
        dE = (values[1][2].penalty_energy-values[0][2].penalty_energy)/(2*h)
        dR = (values[1][0]-values[0][0])/(2*h)
        work.append(dict(h=h, work=float(abs(dE-r@v)/WORK_RATE_SCALE),
                         tangent=_rows(dR-K@v, translation, rotation)))
    report = dict(reference_rotation=R.tolist(), translation_m=SHIFT.tolist(),
        node_numbering=transformed[3], normal_changed_z_to_x=True, actual_force_equilibria=4,
        independent_frame_and_numbering=independent_comparisons, exact_constraint_subspace_map=True,
        physical_gaps_m=[float(result.info['general_contact']['minimum_gap']) for result in results],
        gap_limit_m=5.e-5, force_scale_N=FORCE_SCALE, moment_scale_Nm=MOMENT_SCALE,
        force_rate_scale_N_per_path_parameter=FORCE_SCALE,
        moment_rate_scale_Nm_per_path_parameter=MOMENT_SCALE,
        work_rate_scale_J_per_path_parameter=WORK_RATE_SCALE,
        response_error=response_error, wrench_errors=wrench, equilibrium=equilibrium,
        local_partial_footprint=local, work_tangent=work,
        scope='REFERENCE_TRANSFORMATION_ONLY_NOT_FINITE_MOTION_ADMISSION')
    if os.environ.get('TRIANGULATED_RIGID_RESULT'):
        Path(os.environ['TRIANGULATED_RIGID_RESULT']).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    assert response_error <= 1.e-6
    assert all(e['response_error'] <= 1.e-6 and max(e['wrench_errors'].values()) <= 1.e-6 for e in independent_comparisons)
    assert max(wrench.values()) <= 1.e-6
    assert all(max(e.values()) <= 1.e-6 for e in equilibrium)
    assert max(local['energy'], *local['residual'].values(), *local['tangent_blocks'].values()) <= 1.e-6
    assert max(local['wrenches'].values()) <= 1.e-6
    assert all(max(e['work'], *e['tangent'].values()) <= 1.e-5 for e in work)
