"""Registered installed-artifact W6 rehearsal; never promotes qualification.

SPDX-License-Identifier: MPL-2.0
Run only under the prospective task allowance and external process supervisor.
"""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from collections.abc import Mapping

import numpy as np

from oracles import catenary, contact_response, morison_endpoint


ROOT = Path(__file__).resolve().parent


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(path, data):
    path.write_text(json.dumps(plain(data), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inf(value):
    return float(np.max(np.abs(value), initial=0.))


def rodrigues(vector):
    x, y, z = vector
    s = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = float(np.linalg.norm(vector))
    return np.eye(3)+math.sin(angle)/angle*s+(1-math.cos(angle))/angle**2*s@s


class Rehearsal:
    def __init__(self, out, contract, limit):
        self.out, self.c, self.start = out, contract, time.perf_counter()
        self.deadline = self.start+limit
        self.records, self.failed, self.required = {}, [], []
        self.counts = dict(static_solves=0, dynamic_steps_requested=0, dynamic_steps_accepted=0)

    def cancelled(self):
        return time.perf_counter() >= self.deadline

    def predicate(self, record, name, value, limit):
        passed = math.isfinite(float(value)) and float(value) <= float(limit)
        record.setdefault('predicates', []).append(dict(name=name, value=float(value), limit=float(limit), passed=passed))
        if not passed:
            self.failed.append(dict(case=record['id'], predicate=name, value=float(value), limit=float(limit)))

    def run_case(self, case_id, function):
        record = dict(id=case_id, status='running', phase='rehearsal')
        self.records[case_id] = record
        path = self.out/case_id
        path.mkdir()
        save(path/'record.json', record)
        if self.cancelled():
            record['status'] = 'not_run_deadline'
        else:
            try:
                function(record, path)
                record['status'] = 'passed_rehearsal' if all(p['passed'] for p in record.get('predicates', [])) else 'failed_predicates'
            except Exception as error:
                record['status'] = 'failed_execution'
                record['error'] = f'{type(error).__name__}: {error}'
                (path/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
                # Product failures may carry their immutable accepted prefix.
                partial = getattr(error, 'partial', None)
                if partial is not None:
                    from anyfesolver.marine_mechanical import result_to_dict
                    save(path/'accepted-prefix.json', result_to_dict(partial))
        save(path/'record.json', record)
        self.summary()
        print(f'{case_id}: {record["status"]}', flush=True)
        return record

    def summary(self):
        missing = sorted(set(self.required)-set(self.records))
        bad = [key for key, record in self.records.items() if record['status'] != 'passed_rehearsal']
        save(self.out/'summary.json', dict(phase='rehearsal', qualification='NOT_ASSESSED',
             status='complete_rehearsal' if not missing and not bad else 'incomplete_or_failed',
             required=self.required, observed=list(self.records), missing=missing, failed=bad,
             failed_predicates=self.failed, counts=self.counts,
             process_elapsed_seconds=time.perf_counter()-self.start,
             remaining_requirements=self.c['remaining_W6'],
             application_replay_inventory=self.c['application_replay_inventory']))

    def static(self, record, path, name, segments, *, displacement=None, translation=None, body=False):
        from anyfesolver.marine_equilibrium import MooringStaticSpec, solve_mooring_equilibrium
        from anyfesolver.marine_mooring import (FixedEndpoint, DynamicLineNodeEndpoint,
             RigidBodyAttachmentEndpoint, MooringLine, MooringLineAssembler, data_to_dict)
        from anyfesolver.marine_rigid import RigidBody6D
        from anyfesolver.marine_seabed import SeabedContact
        inputs = {**self.c['static'], **self.c['static_case_inputs'][name]}
        first, last = np.array(inputs['first'], float), np.array(inputs['last'], float)
        if displacement is not None:
            last += displacement
        if translation is not None:
            first += translation
            last += translation
        length, ea = inputs['length'], inputs['EA']
        weight = (inputs['mass_per_length']-inputs['water_density']*inputs['displaced_area'])*inputs['gravity']
        material = np.linspace(0., length, segments+1)
        reference = catenary(first, last, length, ea, weight, material, grounded=name=='touchdown')
        force_scale = max(inf(reference['last_force']), abs(weight)*length, 1.)
        states, bodies = {}, ()
        terminal = FixedEndpoint(tuple(last))
        if body:
            spec = self.c['body_fairlead']
            b = RigidBody6D('fairlead', spec['body_mass'], (0., 0., 0.), np.diag(spec['body_inertia_diagonal']))
            rotation = rodrigues(spec['rotation_vector'])
            origin = last-rotation@spec['point_body']
            states = {b.body_id: b.state(position=origin, rotation_vector=spec['rotation_vector'])}
            bodies = (b,)
            terminal = RigidBodyAttachmentEndpoint(b.body_id, tuple(spec['point_body']))
        endpoints = (FixedEndpoint(tuple(first)),)+tuple(DynamicLineNodeEndpoint(i) for i in range(segments-1))+(terminal,)
        line = MooringLine(name, endpoints, np.full(segments, length/segments), np.full(segments, ea),
             np.zeros(segments), np.full(segments, inputs['mass_per_length']), np.full(segments, inputs['displaced_area']),
             contact_diameters=np.full(segments, self.c['touchdown']['diameter']) if name=='touchdown' else None)
        contact = None
        if name=='touchdown':
            t = self.c['touchdown']
            contact = SeabedContact(t['floor_z'], t['pressure_stiffness'], t['pressure_damping'])
        model = MooringLineAssembler(bodies, (line,), segments-1, (0., 0., -inputs['gravity']), inputs['water_density'], 0., seabed_contact=contact)
        controls = MooringStaticSpec(force_scale=force_scale, force_atol=inputs['force_atol'], force_rtol=inputs['force_rtol'],
                                    max_iterations=inputs['max_iterations'], max_line_search=inputs['max_line_search'])
        save(path/'input.json', dict(model=data_to_dict(model), states=states, controls=controls, inputs=inputs,
             first=first, last=last, reference=reference, force_scale=force_scale))
        self.counts['static_solves'] += 1
        result = solve_mooring_equilibrium(model, states, spec=controls, cancel=self.cancelled)
        save(path/'accepted.json', result)
        if result.status != 'converged':
            raise RuntimeError(f'static {result.status}: {result.message}')
        response = model.evaluate(states, result.nodes)
        positions = np.vstack((first, result.nodes.positions, last))
        chord = first+material[:, None]/length*(last-first)
        geometry_scale = length if name=='taut' else max(inf(reference['positions']-chord), .01*length)
        lump = np.full(segments+1, length/segments)
        lump[[0, -1]] *= .5
        loads = np.zeros_like(positions)
        loads[:, 2] = -weight*lump
        if contact is not None:
            penetration = np.maximum(contact.elevation+.5*t['diameter']-positions[:, 2], 0.)
            loads[:, 2] += t['pressure_stiffness']*t['diameter']*lump*penetration
            self.predicate(record, 'maximum_penetration_m', inf(penetration), t['penetration_limit_m'])
        criteria = self.c['criteria']
        if segments == max(self.c['mesh_segments']):
            self.predicate(record, 'reference_fairlead_force', inf(response.endpoint_forces[-1]-reference['last_force'])/force_scale, criteria['catenary_force_geometry'])
            self.predicate(record, 'reference_geometry', inf(positions-reference['positions'])/geometry_scale, criteria['catenary_force_geometry'])
        self.predicate(record, 'compression', max(0., -float(np.min(response.tensions)))/force_scale, criteria['compression_relative'])
        self.predicate(record, 'distributed_force_balance_N', inf(np.sum(response.endpoint_forces-loads, axis=0)), criteria['balance_atol']+criteria['balance_rtol']*force_scale)
        # Use positions relative to first terminal, avoiding origin-dependent roundoff.
        self.predicate(record, 'distributed_moment_balance_Nm', inf(np.sum(np.cross(positions-first, response.endpoint_forces-loads), axis=0)), criteria['balance_atol']+criteria['balance_rtol']*force_scale*length)
        free = response.force[6*len(bodies):]
        self.predicate(record, 'free_node_residual_N', inf(free), controls.force_limit)
        if body:
            actual = np.linalg.solve(states['fairlead'].rate_map.T, response.force[:6])
            expected = np.r_[response.endpoint_forces[-1], np.cross(last-origin, response.endpoint_forces[-1])]
            self.predicate(record, 'body_force_work_map_N', inf(actual[:3]-expected[:3]), criteria['balance_atol']+criteria['balance_rtol']*force_scale)
            self.predicate(record, 'body_moment_work_map_Nm', inf(actual[3:]-expected[3:]), criteria['balance_atol']+criteria['balance_rtol']*force_scale*length)
        save(path/'arrays.json', dict(positions=positions, fairlead_force=response.endpoint_forces[-1],
             tensions=response.tensions, endpoint_forces=response.endpoint_forces, reference=reference,
             force_scale=force_scale, geometry_scale=geometry_scale, force_components=response.force_components,
             anchor_reactions=response.anchor_reactions, body_states=states))
        record['metrics'] = dict(force_scale=force_scale, geometry_scale=geometry_scale, reference_closure=reference['closure'],
                                 peak_tension=float(np.max(response.tensions)), accepted_stages=len(result.stages))

    def refinement(self, record, path, family, coarse, fine):
        a = json.loads((self.out/coarse/'arrays.json').read_text())
        b = json.loads((self.out/fine/'arrays.json').read_text())
        criteria = self.c['criteria']
        if family=='mesh':
            scale = b['force_scale']
            self.predicate(record, 'fairlead_mesh', inf(np.array(a['fairlead_force'])-b['fairlead_force'])/scale, criteria['mesh_tension_fairlead'])
            self.predicate(record, 'peak_tension_mesh', abs(max(a['tensions'])-max(b['tensions']))/scale, criteria['mesh_tension_fairlead'])
        else:
            # Registered nested fixed steps: exact common-time sample, no interpolated solution.
            ta, tb = np.asarray(a['time']), np.asarray(b['time'])
            stride = int(round((ta[1]-ta[0])/(tb[1]-tb[0])))
            self.predicate(record, 'nested_time_inventory', inf(ta-tb[::stride]), 1e-12)
            self.predicate(record, 'position_time_refinement', inf(np.array(a['q'])-np.array(b['q'])[::stride])/b['position_scale'], criteria['time_response'])
            self.predicate(record, 'velocity_time_refinement', inf(np.array(a['v'])-np.array(b['v'])[::stride])/b['velocity_scale'], criteria['time_response'])
            for name in ('tensions', 'anchor_reactions'):
                self.predicate(record, name+'_time_refinement', inf(np.array(a[name])-np.array(b[name])[::stride])/b['line_force_scale'], criteria['time_response'])
            if family=='contact':
                self.predicate(record, 'contact_energy_defect_decrease', b['energy_defect'], criteria['contact_energy_refinement_factor']*a['energy_defect']+criteria['energy_roundoff_floor_relative'])
        save(path/'comparison.json', dict(coarse=coarse, fine=fine, predicates=record['predicates']))

    def translation(self, record, path):
        self.static(record, path, 'suspended', self.c['rigid_translation']['mesh'], translation=np.array(self.c['rigid_translation']['vector']))
        a = json.loads((self.out/'suspended-64'/'arrays.json').read_text())
        b = json.loads((path/'arrays.json').read_text())
        spec = self.c['rigid_translation']
        self.predicate(record, 'translation_force', inf(np.array(a['fairlead_force'])-b['fairlead_force'])/a['force_scale'], spec['relative_force_tolerance'])
        self.predicate(record, 'translation_geometry_m', inf(np.array(b['positions'])-np.array(a['positions'])-spec['vector']), spec['position_atol'])

    def dynamics(self, record, path, name, dt, damping=None):
        from anyfesolver.marine_mooring import FixedEndpoint, DynamicLineNodeEndpoint, MooringNodeState, MooringLine, MooringLineAssembler
        from anyfesolver.marine_rigid import RigidBody6D
        from anyfesolver.marine_seabed import SeabedContact
        from anyfesolver.marine_mechanical import MarineMechanicalModel, MarineMechanicalRunSpec, solve_marine_mechanical, model_to_dict, result_to_dict
        common, fixture = self.c['dynamic_fixture'], self.c[name]
        b = RigidBody6D(common['body_id'], common['body_mass'], (0., 0., 0.), np.diag(common['body_inertia_diagonal']))
        state = b.state(position=common['body_position'])
        contact = None
        position = np.array(common['center'], float)
        velocity = np.zeros(3)
        axis = 0 if name=='axial_vibration' else 2
        q0 = fixture['initial_displacement'] if axis==0 else fixture['initial_gap']
        position[axis] += q0
        velocity[axis] = fixture['initial_velocity']
        if axis==2:
            contact = SeabedContact(fixture['floor_z'], fixture['pressure_stiffness'], damping,
                                   activation_depth_m=fixture.get('activation_depth_m'))
            position[2] = fixture['floor_z']+.5*fixture['diameter']+q0
        nodes = MooringNodeState(position.reshape(1, 3), velocity.reshape(1, 3), np.array(common['initial_node_acceleration']).reshape(1, 3))
        endpoints = (FixedEndpoint(tuple(common['anchors'][0])), DynamicLineNodeEndpoint(0), FixedEndpoint(tuple(common['anchors'][1])))
        line = MooringLine(name, endpoints, np.full(2, fixture['rest_segment']), np.full(2, fixture['EA']),
             np.zeros(2), np.full(2, fixture['mass_per_length']), np.zeros(2),
             contact_diameters=np.full(2, fixture['diameter']) if contact is not None else None)
        assembler = MooringLineAssembler((b,), (line,), 1, tuple(common['gravity_world']), common['water_density'], 0., seabed_contact=contact)
        model = MarineMechanicalModel((b,), mooring=assembler)
        steps = int(round(fixture['duration']/dt))
        if steps > self.c['execution']['max_dynamic_steps_per_case']:
            raise ValueError('registered step cap exceeded')
        run_spec = MarineMechanicalRunSpec(dt, steps, common['native_residual_tolerance'], common['max_newton_iterations'])
        save(path/'input.json', dict(model=model_to_dict(model), states={b.body_id: state}, nodes=nodes, run_spec=run_spec))
        self.counts['dynamic_steps_requested'] += steps
        result = solve_marine_mechanical(model, {b.body_id: state}, run_spec, cancel=self.cancelled, initial_nodes=nodes)
        save(path/'accepted.json', result_to_dict(result))
        self.counts['dynamic_steps_accepted'] += max(0, len(result.accepted)-1)
        if result.status != 'completed':
            raise RuntimeError(f'dynamic {result.status}')
        times = np.array([frame.time for frame in result.accepted])
        q = np.array([frame.nodes.positions[0, axis] for frame in result.accepted])
        q -= common['center'][axis] if axis==0 else fixture['floor_z']+.5*fixture['diameter']
        v = np.array([frame.nodes.velocities[0, axis] for frame in result.accepted])
        mass = fixture['mass_per_length']*fixture['rest_segment']
        criteria = self.c['criteria']
        if axis==0:
            stiffness = 2*fixture['EA']/fixture['rest_segment']
            omega = math.sqrt(stiffness/mass)
            rq, rv = q0*np.cos(omega*times), -q0*omega*np.sin(omega*times)
            scale = abs(q0)
            baseline = fixture['EA']/fixture['rest_segment']*(1-fixture['rest_segment'])**2
            energy = np.array([frame.kinetic_energy+frame.mooring_diagnostics['elastic_energy']-baseline for frame in result.accepted])
            e0 = .5*stiffness*q0*q0
            defect = inf(energy-e0)/e0
            self.predicate(record, 'axial_energy_relative', defect, criteria['axial_energy_relative'])
            loss = np.zeros_like(times)
        else:
            stiffness = fixture['diameter']*fixture['rest_segment']*fixture['pressure_stiffness']
            c = fixture['diameter']*fixture['rest_segment']*damping
            omega = math.sqrt(stiffness/mass)
            reference = contact_response(times, q0, fixture['initial_velocity'], mass, stiffness, c,
                                         activation_depth_m=fixture.get('activation_depth_m'))
            rq, rv = reference['q'], reference['v']
            scale = math.hypot(q0, fixture['initial_velocity']/omega)
            energy = np.array([frame.kinetic_energy+.5*stiffness*max(-gap, 0.)**2 for frame, gap in zip(result.accepted, q)])
            loss = np.array([frame.mooring_diagnostics['contact']['dissipation'] for frame in result.accepted])
            e0 = .5*mass*fixture['initial_velocity']**2+.5*stiffness*max(-q0, 0.)**2
            integral = np.r_[0., np.cumsum(.5*(loss[:-1]+loss[1:])*np.diff(times))]
            defect = inf(energy-e0+integral)/e0
            self.predicate(record, 'energy_growth_relative', max(0., float(np.max(energy-e0)))/e0, criteria['energy_growth_relative'])
            self.predicate(record, 'energy_defect_relative', defect, criteria['contact_energy_defect_relative'])
            self.predicate(record, 'maximum_penetration_m', max(0., -float(np.min(q))), fixture['penetration_limit_m'])
            self.predicate(record, 'negative_dissipation', max(0., -float(np.min(loss))), criteria['balance_atol'])
            save(path/'reference.json', reference)
        if dt==min(fixture['dt']):
            self.predicate(record, 'reference_position', inf(q-rq)/scale, criteria['dynamic_reference'])
            self.predicate(record, 'reference_velocity', inf(v-rv)/(scale*omega), criteria['dynamic_reference'])
        tensions = np.concatenate([frame.mooring_diagnostics['tensions'] for frame in result.accepted])
        tension_history = np.array([frame.mooring_diagnostics['tensions'] for frame in result.accepted])
        anchor_history = np.array([frame.mooring_diagnostics['anchor_reactions'] for frame in result.accepted])
        if axis==0:
            preload = fixture['EA']*(1-fixture['rest_segment'])/fixture['rest_segment']
            reference_tensions = np.column_stack((preload+fixture['EA']/fixture['rest_segment']*rq,
                                                  preload-fixture['EA']/fixture['rest_segment']*rq))
            reference_anchors = np.zeros_like(anchor_history)
            reference_anchors[:, 0, 0] = -reference_tensions[:, 0]
            reference_anchors[:, 1, 0] = reference_tensions[:, 1]
            line_force_scale = max(fixture['EA']*scale/fixture['rest_segment'], 1.)
            if dt==min(fixture['dt']):
                for label, actual, reference_values in (('tensions', tension_history, reference_tensions), ('anchor_loads', anchor_history, reference_anchors)):
                    self.predicate(record, 'reference_'+label, inf(actual-reference_values)/line_force_scale, criteria['dynamic_reference'])
        else:
            reference_tensions, reference_anchors = np.zeros_like(tension_history), np.zeros_like(anchor_history)
            line_force_scale = 1.
            self.predicate(record, 'slack_tension_zero_N', inf(tension_history), criteria['balance_atol'])
            self.predicate(record, 'slack_anchor_load_zero_N', inf(anchor_history), criteria['balance_atol'])
        self.predicate(record, 'compression', max(0., -float(np.min(tensions))), criteria['compression_relative'])
        self.predicate(record, 'accepted_time_inventory', inf(times-np.arange(steps+1)*dt), 1e-12)
        force_error, moment_error, body_error, transverse_error = 0., 0., 0., 0.
        force_scale = max(stiffness*scale, 1.)  # force units; registered limits unchanged
        for frame in result.accepted:
            points = np.vstack((common['anchors'][0], frame.nodes.positions, common['anchors'][1]))
            external = np.zeros((3, 3))
            external[1] = -mass*frame.nodes.accelerations[0]
            if axis==2:
                gap = frame.nodes.positions[0, 2]-fixture['floor_z']-.5*fixture['diameter']
                pen=max(-gap,0.)
                factor=min(pen/fixture['activation_depth_m'],1.) if 'activation_depth_m' in fixture else 1.
                raw = stiffness*pen-c*factor*frame.nodes.velocities[0, 2]
                external[1, 2] += max(raw, 0.) if gap < 0 else 0.
            difference = np.asarray(frame.mooring_diagnostics['endpoint_forces'])-external
            force_error = max(force_error, inf(np.sum(difference, axis=0)))
            moment_error = max(moment_error, inf(np.sum(np.cross(points-points[0], difference), axis=0)))
            body_error = max(body_error, inf(np.array(frame.positions[b.body_id])-common['body_position']),
                 inf(np.array(frame.rotations[b.body_id])-np.eye(3)), inf(frame.world_velocities[b.body_id]))
            transverse = [i for i in range(3) if i != axis]
            transverse_error = max(transverse_error, inf(frame.nodes.positions[0, transverse]-np.array(common['center'])[transverse]))
        self.predicate(record, 'endpoint_dynamic_force_balance_N', force_error, criteria['balance_atol']+criteria['balance_rtol']*force_scale)
        self.predicate(record, 'endpoint_dynamic_moment_balance_Nm', moment_error, criteria['balance_atol']+criteria['balance_rtol']*force_scale*2*fixture['rest_segment'])
        self.predicate(record, 'unloaded_body_and_transverse_invariance', max(body_error, transverse_error), 1e-10)
        save(path/'arrays.json', dict(time=times, q=q, v=v, reference_q=rq, reference_v=rv,
             energy=energy, dissipation=loss, energy_defect=defect,
             position_scale=scale, velocity_scale=scale*omega,
             tensions=tension_history, anchor_reactions=anchor_history, reference_tensions=reference_tensions,
             reference_anchors=reference_anchors, line_force_scale=line_force_scale))
        record['metrics'] = dict(energy_defect=defect, steps=steps, accepted=len(result.accepted), position_scale=scale)

    def point_contact(self, record, path):
        from anyfesolver.marine_seabed import SeabedContact, seabed_contact_force
        f = self.c['contact_point_passivity']
        contact = SeabedContact(0., f['pressure_stiffness'], f['pressure_damping'],activation_depth_m=f.get('activation_depth_m'))
        gaps, velocities = np.meshgrid(f['gaps'], f['velocities'])
        gaps, velocities = gaps.ravel(), velocities.ravel()
        p, v = np.zeros((len(gaps), 3)), np.zeros((len(gaps), 3))
        p[:, 2] = gaps+.5*f['diameter']
        v[:, 2] = velocities
        actual = seabed_contact_force(contact, f['diameter'], f['reference_length'], p, v)
        k = f['diameter']*f['reference_length']*f['pressure_stiffness']
        c = f['diameter']*f['reference_length']*f['pressure_damping']
        fe = k*np.maximum(-gaps, 0.)
        factor=np.minimum(np.maximum(-gaps,0.)/f['activation_depth_m'],1.) if 'activation_depth_m' in f else 1.
        expected = np.where(gaps < 0, np.maximum(fe-c*factor*velocities, 0.), 0.)
        for name, value in dict(force=actual.force[:, 2]-expected,
            storage=actual.potential_energy-.5*k*np.maximum(-gaps, 0.)**2,
            power=actual.force[:, 2]*velocities-fe*velocities+actual.dissipation,
            tensile=np.maximum(-actual.force[:, 2], 0.), negative_loss=np.maximum(-actual.dissipation, 0.)).items():
            self.predicate(record, name, inf(value), f['balance_atol'])
        save(path/'arrays.json', dict(gaps=gaps, velocities=velocities, actual=actual, expected_force=expected))

    def harmonic(self, record, path, prescribed=False):
        from anyloads.marine import AiryEnvironment, AiryComponent, CurrentProfile
        from anyloads.marine_line import LineHydrodynamics
        from anyfesolver.marine_mooring import FixedEndpoint, RigidBodyAttachmentEndpoint, DynamicLineNodeEndpoint, MooringNodeState, MooringLine, MooringLineAssembler
        from anyfesolver.marine_rigid import RigidBody6D
        f = self.c['harmonic']
        kinematics = self.c['prescribed_kinematics']
        heading = kinematics['heading_degrees'] if prescribed else f['heading_degrees']
        environment = AiryEnvironment(f['water_depth'], (AiryComponent(f['amplitude'], f['omega'], heading, f['phase']),),
             CurrentProfile.uniform(f['water_depth'], f['current']), f['water_density'], f['gravity'])
        section = LineHydrodynamics(*(f[key] for key in ('diameter', 'drag_normal', 'drag_tangential', 'added_mass_normal', 'added_mass_tangential', 'inertia_normal', 'inertia_tangential')))
        grid = np.arange(f['phase_samples'])*2*math.pi/(f['phase_samples']*f['omega'])
        samples = []
        for n in f['segments']:
            p = np.linspace(f['anchors'][0], f['anchors'][1], n+1)
            represented = np.full(n+1, f['length']/n)
            represented[[0, -1]] *= .5
            bodies, states = (), {}
            terminal_first, terminal_last = FixedEndpoint(tuple(p[0])), FixedEndpoint(tuple(p[-1]))
            velocity, acceleration = np.zeros_like(p), np.zeros_like(p)
            if prescribed:
                b = RigidBody6D(kinematics['body_id'], kinematics['body_mass'], (0., 0., 0.), np.diag(kinematics['body_inertia_diagonal']))
                origin = np.asarray(kinematics['body_position'])
                velocity[:] = kinematics['structure_velocity']
                acceleration[:] = kinematics['structure_acceleration']
                bodies = (b,)
                states = {b.body_id: b.state(position=origin, velocity=np.r_[velocity[0], kinematics['angular_velocity']], acceleration=np.r_[acceleration[0], kinematics['angular_acceleration']])}
                terminal_first = RigidBodyAttachmentEndpoint(b.body_id, tuple(p[0]-origin))
                terminal_last = RigidBodyAttachmentEndpoint(b.body_id, tuple(p[-1]-origin))
            ends = (terminal_first,)+tuple(DynamicLineNodeEndpoint(i) for i in range(n-1))+(terminal_last,)
            line = MooringLine('harmonic', ends, np.full(n, f['length']/n), np.full(n, f['EA']), np.zeros(n),
                 np.full(n, f['mass_per_length']), np.full(n, math.pi*f['diameter']**2/4), hydrodynamics=(section,)*n)
            model = MooringLineAssembler(bodies, (line,), n-1, tuple(f['gravity_world']), f['water_density'], 0., environment=environment)
            nodes = MooringNodeState(p[1:-1], velocity[1:-1], acceleration[1:-1])
            for t in grid:
                actual = model.evaluate(states, nodes, time=float(t))
                expected, ca = morison_endpoint(p, float(t), f, heading, velocity, acceleration, represented)
                expected['inertia'] = -f['mass_per_length']*represented[:, None]*acceleration
                total = sum(expected.values())
                tolerance = f['force_atol']+f['force_rtol']*max(inf(total), 1.)
                self.predicate(record, f'endpoint_force_n{n}_t{t:.9g}', inf(actual.endpoint_forces-total), tolerance)
                if prescribed:
                    for component, values in expected.items():
                        generalized = np.r_[np.sum(values[[0, -1]], axis=0), np.sum(np.cross(p[[0, -1]]-origin, values[[0, -1]]), axis=0), values[1:-1].ravel()]
                        self.predicate(record, f'component_{component}_n{n}_t{t:.9g}', inf(actual.force_components[component]-generalized), tolerance)
                    expected_nodes_mass = (f['mass_per_length']*represented[1:-1, None]+ca[1:-1]).ravel()
                    self.predicate(record, f'node_mass_n{n}_t{t:.9g}', inf(actual.mass.diagonal()[6:]-expected_nodes_mass), tolerance)
                samples.append(dict(segments=n, time=t, endpoint_forces=actual.endpoint_forces, expected=total,
                     expected_components=expected, force_components=actual.force_components))
        save(path/'arrays.json', dict(samples=samples, times=grid, prescribed=prescribed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['rehearsal'], required=True)
    parser.add_argument('--internal-seconds', type=float, default=80.)
    parser.add_argument('--group', choices=['all', 'static', 'loads', 'dynamic'], default='all',
                        help='bounded independent group; full missing inventory remains in summary')
    args = parser.parse_args()
    c = json.loads((ROOT/'contract.json').read_text())
    if not 0 < args.internal_seconds <= c['execution']['per_process_seconds']-5:
        parser.error('internal deadline must retain at least five seconds of process reserve')
    args.out.mkdir(parents=True, exist_ok=False)
    packages = {}
    import anyfesolver, anyloads
    for name, module in [('anyfesolver', anyfesolver), ('anyloads', anyloads)]:
        base = Path(module.__file__).resolve().parent
        if 'site-packages' not in str(base).lower():
            raise RuntimeError(f'{name} must be installed, found {base}')
        packages[name] = dict(version=importlib.metadata.version(name), root=str(base),
                             files={str(p.relative_to(base)): sha(p) for p in sorted(base.rglob('*.py'))})
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError(f'{key}=1 is required')
    save(args.out/'identity.json', dict(contract_sha256=sha(ROOT/'contract.json'), runner_sha256=sha(Path(__file__)),
        oracle_sha256=sha(ROOT/'oracles.py'), packages=packages, python=sys.version, executable=sys.executable,
        environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONPATH', 'ANYMARINE_SOLVER_PATH')},
        command=sys.argv, phase=args.phase))
    save(args.out/'contract.json', c)
    run = Rehearsal(args.out, c, args.internal_seconds)
    tasks = []
    for name in c['static_cases']:
        for n in c['mesh_segments']:
            tasks.append((f'{name}-{n}', lambda r, p, name=name, n=n: run.static(r, p, name, n)))
        tasks.append((f'{name}-mesh', lambda r, p, name=name: run.refinement(r, p, 'mesh', f'{name}-32', f'{name}-64')))
    for i, displacement in enumerate(c['fairlead_curve']['displacements']):
        if displacement != [0., 0., 0.]:
            tasks.append((f'fairlead-curve-{i}', lambda r, p, d=displacement: run.static(r, p, 'suspended', 64, displacement=d)))
    tasks += [('translation', run.translation), ('body-fairlead', lambda r, p: run.static(r, p, 'suspended', 64, body=True)),
              ('point-contact', run.point_contact), ('harmonic', run.harmonic),
              ('prescribed-kinematics', lambda r, p: run.harmonic(r, p, True))]
    for name in ('axial_vibration', 'contact_release', 'contact_landing', 'contact_clipping'):
        fixture = c[name]
        dampings = [None] if name=='axial_vibration' else fixture['pressure_damping']
        for damping in dampings:
            prefix = name if damping is None else f'{name}-c{damping:g}'
            ids = []
            for i, dt in enumerate(fixture['dt']):
                case_id = f'{prefix}-level{i}'
                ids.append(case_id)
                tasks.append((case_id, lambda r, p, name=name, dt=dt, damping=damping: run.dynamics(r, p, name, dt, damping)))
            tasks.append((f'{prefix}-refinement', lambda r, p, a=ids[-2], b=ids[-1], kind='time' if damping is None else 'contact': run.refinement(r, p, kind, a, b)))
    run.required = [case_id for case_id, _ in tasks]
    run.summary()
    selected = lambda case_id: (args.group=='all' or
        args.group=='dynamic' and case_id.startswith(('axial_vibration', 'contact_release', 'contact_landing', 'contact_clipping')) or
        args.group=='loads' and case_id in ('point-contact', 'harmonic', 'prescribed-kinematics') or
        args.group=='static' and not case_id.startswith(('axial_vibration', 'contact_release', 'contact_landing', 'contact_clipping')) and case_id not in ('point-contact', 'harmonic', 'prescribed-kinematics'))
    for case_id, function in tasks:
        if not selected(case_id):
            continue
        result = run.run_case(case_id, function)
        if result['status'] != 'passed_rehearsal':
            # Preserve the complete required inventory; no expensive blind continuation after a failure.
            break
    run.summary()
    summary = json.loads((args.out/'summary.json').read_text())
    return 0 if summary['status']=='complete_rehearsal' else 2


if __name__=='__main__':
    raise SystemExit(main())
