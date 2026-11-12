"""Frozen reverse-load spherical continuation from physical contact preload."""
import importlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from anyfesolver import ArcLengthControl, LoadCase, solve_static_arc_length
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.runtime import active_contact_manager
from examples.general_contact_arc_length_affine import build
from s3_von_karman_oracle import model_triangle_von_karman

FORCE_SCALE = .15
MOMENT_SCALE = .15
DISPLACEMENT_SCALE = 1.e-5


def _numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('unsupported release evidence value')


def _case():
    model, contact, settings, _, constant = build(True, .01)
    load = LoadCase('reverse_proportional_opening')
    for node in (2, 3, 4):
        load.add_nodal_load(node, forces=(0., 0., .05))
    return model, contact, settings, load, constant


def _run(steps, restart=None):
    model, contact, settings, load, constant = _case()
    control = ArcLengthControl(initial_load_increment=.05, minimum_load_increment=5.e-7,
        maximum_load_increment=.05, load_scaling=.01, growth_factor=1.,
        max_steps=steps, preload_steps=2)
    # Under the adopted S3 VK law the open plate response is nonlinear, so the
    # default 1e-6 residual tolerance (exact for the former linear one-step
    # solve) left 1.1e-7 N residuals that the soft open mode (~30 m/N) turned
    # into 1.5e-6 m response errors. Tighten the solve; gates are unchanged.
    return solve_static_arc_length(model, load, constant_load_case=constant,
        control=control, contact=contact, contact_settings=settings, max_iterations=15, tolerance=1.e-12,
        emit_restart_checkpoint=True, record_increment_snapshots=True,
        restart_checkpoint=restart)


def _reference(model):
    K, _ = assemble_stiffness_matrix(model)
    ids = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1, 2, 3, 4)]
    K = K.toarray()[np.ix_(ids, ids)]
    M, S = np.zeros((4, 4)), np.zeros(4)
    for nodes in ((1, 2, 3), (2, 4, 3)):
        xy = np.array([(model.mesh.nodes[node].x, model.mesh.nodes[node].y) for node in nodes])
        area = abs(np.linalg.det(np.array((xy[1]-xy[0], xy[2]-xy[0]))))/2
        indices = np.array(nodes)-1
        M[np.ix_(indices, indices)] += area/12*(np.eye(3)+np.ones((3, 3)))
        S[indices] += area/3
    # Retarget to the adopted S3 VK law (precedent 706bf011c): every in-plane
    # translation and rotation is guided to zero, so the skin adds only the
    # independent VK membrane force of each triangle (fixture E=1e5, nu=.3,
    # t=.01) to the linear stiffness. Solved per enumerated active set.
    def membrane(u):
        force, hessian = np.zeros(4), np.zeros((4, 4))
        for nodes in ((1, 2, 3), (2, 4, 3)):
            indices = np.array(nodes)-1
            _, f, h = model_triangle_von_karman(model, nodes, (0., 0., 1.), u[indices], 1.e5, .3, .01)
            force[indices] += f
            hessian[np.ix_(indices, indices)] += h
        return force, hessian
    def oracle(factor):
        load = np.array((0., -.01+.05*factor, -.01+.05*factor, -.01+.05*factor))
        imposed = -2.e-6*(1.+factor)
        admitted = []
        for bits in itertools.product((False, True), repeat=4):
            active = np.array(bits)
            C = 1.e6*M[active].T@np.diag(1/S[active])@M[active]
            H = K+C
            rhs = load-C@np.full(4, 1.e-5)
            u = np.empty(4)
            u[0] = imposed
            u[1:] = np.linalg.solve(H[1:, 1:], rhs[1:]-H[1:, 0]*imposed)
            for _ in range(40):  # Newton on the VK cubic, independent of the solver.
                force, hessian = membrane(u)
                u[1:] -= np.linalg.solve((H+hessian)[1:, 1:], (H@u+force-rhs)[1:])
            force, _ = membrane(u)
            scale = np.linalg.norm(abs(H)@abs(u))+np.linalg.norm(rhs)+np.linalg.norm(force)
            assert np.linalg.norm((H@u+force-rhs)[1:]) <= 1.e-12*scale, 'independent VK oracle'
            weighted = M@(np.full(4, 1.e-5)+u)
            if np.all(weighted[active] < 0.) and np.all(weighted[~active] >= 0.):
                admitted.append(u)
        assert len(admitted) == 1
        return admitted[0]
    return ids, oracle


def _wrench(model, pose, residual, factor, contact):
    force, moment = np.zeros(3), np.zeros(3)
    for node in (1, 2, 3, 4):
        ids = model.mesh.dof_manager.get_node_dofs(node)
        point = model.mesh.nodes[node]
        position = np.array((point.x, point.y, point.z))+pose[ids[:3]]
        support = residual[ids].copy()
        if node != 1:
            support[2] = 0.
        force += support[:3]
        moment += support[3:]+np.cross(position, support[:3])
        if node != 1:
            load = np.array((0., 0., -.01+.05*factor))
            force += load
            moment += np.cross(position, load)
    force += contact['integration_force']
    moment += contact['integration_moment']
    return dict(force=float(np.linalg.norm(force)/FORCE_SCALE),
                moment=float(np.linalg.norm(moment)/MOMENT_SCALE))


def test_active_preload_to_open_reverse_arc_complete_restart(monkeypatch):
    arc = importlib.import_module('anyfesolver.arc_length')
    static = importlib.import_module('anyfesolver.nonlinear_static')
    actual_arc, actual_static = arc._support_reaction_resultants_from_forces, static._support_reaction_resultants
    arc_rows, preload_rows = [], []
    def capture_arc(model, internal, constant, proportional, factor, **kwargs):
        arc_rows.append((model, np.asarray(internal)-np.asarray(constant)-factor*np.asarray(proportional)))
        return actual_arc(model, internal, constant, proportional, factor, **kwargs)
    def capture_static(model, imbalance, **kwargs):
        preload_rows.append((model, np.array(imbalance, copy=True)))
        return actual_static(model, imbalance, **kwargs)
    monkeypatch.setattr(arc, '_support_reaction_resultants_from_forces', capture_arc)
    monkeypatch.setattr(static, '_support_reaction_resultants', capture_static)
    full = _run(6)
    assert full.status == 'maximum_steps_reached' and len(full.steps) == 6, full.failure_reason
    assert active_contact_manager() is None
    output = os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:
        # Preserve accepted production state before any consumer/oracle failure.
        Path(output).with_name('accepted_full.checkpoint').write_bytes(full.restart_checkpoint_bytes())
    model, *_ = _case()
    ids, oracle = _reference(model)
    from anyfesolver.general_contact import transient_results_archive as codec
    base = codec._decode(full.to_restart_checkpoint()['path_state']['contact_state']['base'],
                         codec._Budget(codec.MAX_BYTES), {})
    previous = base['displacements'].copy()
    base_contact = base['history'][-1]['contact']
    base_response = float(np.linalg.norm(previous[ids]-oracle(0.))/DISPLACEMENT_SCALE)
    base_wrench = _wrench(preload_rows[-1][0], previous, preload_rows[-1][1], 0., base_contact)
    history = full.info['general_contact_history']
    response, sphere, equilibrium, pair, gaps = [], [], [], [], []
    last = 0.
    assert len(arc_rows) == len(history) == 6
    for row, step, (owner, residual) in zip(history, full.steps, arc_rows):
        pose, factor = row['displacements'], row['load_factor']
        contact = row['contact']
        response.append(float(np.linalg.norm(pose[ids]-oracle(factor))/DISPLACEMENT_SCALE))
        sphere.append(float(abs(np.dot(pose-previous, pose-previous)+.01**2*(factor-last)**2-step.arc_radius**2)/step.arc_radius**2))
        equilibrium.append(_wrench(owner, pose, residual, factor, contact))
        pair.append(dict(force=float(np.linalg.norm(np.array(contact['integration_force'])+contact['target_force'])/FORCE_SCALE),
            moment=float(np.linalg.norm(np.array(contact['integration_moment'])+contact['target_moment'])/MOMENT_SCALE)))
        gaps.append(float(contact['minimum_gap']))
        assert contact['gap_tolerance_met'] and contact['minimum_gap'] >= -5.e-5
        previous, last = pose, factor
    report = dict(status=full.status, steps=6, preload_active=base_contact['active_pressure_dofs'] > 0,
        final_open=(history[-1]['contact']['active_pressure_dofs'] == 0
                    and history[-1]['contact']['minimum_gap'] > 0.),
        preload_contact=base_contact, preload_response=base_response, preload_equilibrium=base_wrench,
        load_factors=[float(row['load_factor']) for row in history], gaps_m=gaps,
        response=response, spherical=sphere, equilibrium=equilibrium, pair_wrenches=pair,
        force_scale_N=FORCE_SCALE, moment_scale_Nm=MOMENT_SCALE,
        clearance_scale_m=DISPLACEMENT_SCALE, restart_exact=False)
    if output:
        Path(output).write_text(json.dumps(report, indent=2, default=_numpy_scalar)+'\n', encoding='utf-8')
    assert report['preload_active'] and report['final_open']
    assert base_response <= 1.e-6 and max(response) <= 1.e-6
    assert max(sphere) <= 1.e-6
    assert max(base_wrench.values()) <= 1.e-6
    assert all(max(row.values()) <= 1.e-6 for row in equilibrium+pair)
    monkeypatch.setattr(arc, '_support_reaction_resultants_from_forces', actual_arc)
    monkeypatch.setattr(static, '_support_reaction_resultants', actual_static)
    prefix = _run(3)
    if output:
        Path(output).with_name('accepted_prefix.checkpoint').write_bytes(prefix.restart_checkpoint_bytes())
    resumed = _run(3, prefix.to_restart_checkpoint())
    report['restart_exact'] = resumed.restart_checkpoint_bytes() == full.restart_checkpoint_bytes()
    if output:
        Path(output).write_text(json.dumps(report, indent=2, default=_numpy_scalar)+'\n', encoding='utf-8')
    assert report['restart_exact']
    assert active_contact_manager() is None
