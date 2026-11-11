"""Loaded noncommuting native fibre chart transport; scoped development only."""
from dataclasses import replace
from math import fsum
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_ge_physical_fibre_linear import _definition
from test_stiffened_shell_staged_material_history import _capture, _same, _canonical
from test_stiffened_shell_fibre_chart_boundary_comparison import (
    cell_operators, cyclic_fibre_boundary_hook, fibre_sections)


PREFIX = (.125, .5, .75, 1.)
FOLLOWING = (1.125, 1.5, 1.75, 2., 2.5, 3., 3.5, 4.)
OPTIONS = dict(max_iterations=20, max_backtracks=12)
CONTROL = ss.AdaptiveIncrementControl(4, 64)
FORCE_SCALE, LENGTH = 1_750_000., 2.


def definition():
    # Same reference geometry, elastic stiffness and force levels as the passed
    # large-elastic staged point fixture. The physical fibres have an explicitly
    # chosen strong-hardening constitutive control, not an engineering capacity law.
    base = _definition(mixed=True)
    beams = []
    for native in base.beams:
        element, inertia = native.instantiate()
        if type(element.section) is not b3_ge.PhysicalFibreSection:
            beams.append(native)
            continue
        section = b3_ge.PhysicalFibreSection(tuple(replace(fibre,
            curve=b3_ge.FlowCurve((0.,), (250e6,), 210e9))
            for fibre in element.section.fibres), element.section.background_factor)
        ref = element.operator.reference
        beams.append(b3_ge.define_beam('b3-ge', element.element_id, element.node_ids,
            ref.coordinates, ref.nodal_triads, section, inertia,
            quadrature=element.operator.order))
    return replace(base, beams=tuple(beams), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def programme(owner):
    bending = ss.NodalAction('beam', 105, (0., 0., 1e6, 0., 0., 0.))
    combined = (bending, ss.NodalAction('shell', 3, (0., 0., -375e3, 0., 0., 0.)),
        ss.NodalAction('shell', 6, (0., 0., 375e3, 0., 0., 0.)))
    unload = tuple(replace(row, wrench=tuple(.25*v for v in row.wrench)) for row in combined)
    return ss.StagedDeadLoadProgramme(owner.preflight()['identity'],
        ((bending,), combined, unload, combined))


def physical_forces(position):
    # Independent analytic path, not the production interpolation helper.
    if position <= 1.:
        return position*1e6, 0.
    if position <= 2.:
        return 1e6, (position-1.)*375e3
    scale = 1.-.75*(position-2.) if position <= 3. else .25+.75*(position-3.)
    return scale*1e6, scale*375e3


def check(snapshot, previous, owner):
    result = snapshot['result']
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
    bending, couple = physical_forces(result.load_position)
    expected = np.zeros_like(result.applied_dead_load)
    expected[owner._shell_count+6*owner._beam_index[105]+2] = bending
    expected[6*owner._shell_index[3]+2] = -couple
    expected[6*owner._shell_index[6]+2] = couple
    np.testing.assert_allclose(result.applied_dead_load/FORCE_SCALE,
        expected/FORCE_SCALE, rtol=0., atol=1e-15)
    shell_pos = np.asarray(owner._definition.shell.coordinates_m)+snapshot['q'][:owner._shell_count].reshape(-1,6)[:,:3]
    pos = np.vstack((shell_pos, result.beam_positions))
    support = result.support_reactions[:6*len(pos)].reshape(-1,6)
    applied = expected[:6*len(pos)].reshape(-1,6)
    force = (support[:,:3]+applied[:,:3]).sum(axis=0)
    moment = (support[:,3:]+np.cross(pos,support[:,:3]+applied[:,:3])).sum(axis=0)
    assert np.linalg.norm(force)/FORCE_SCALE <= 1e-8
    assert np.linalg.norm(moment)/(FORCE_SCALE*LENGTH) <= 1e-8
    for port in result.point_wrenches:
        actions = np.array((port.shell_action,port.beam_action))
        positions = np.array((port.shell_position_m,port.beam_position_m))
        positions_low = np.array((port.shell_position_low_m,port.beam_position_low_m))
        assert np.linalg.norm(actions[:,:3].sum(axis=0))/FORCE_SCALE <= 1e-8
        moment_at_origin = (actions[:,3:]+np.cross(positions,actions[:,:3])+
            np.cross(positions_low,actions[:,:3])).sum(axis=0)
        assert np.linalg.norm(moment_at_origin)/(FORCE_SCALE*LENGTH) <= 1e-8
        assert np.max(np.abs(port.gap)/np.array((LENGTH,)*3+(1.,)*3)) <= 1e-9
    for energy in result.beam_fibre_energy:
        assert fsum(energy.incremental_dissipation_j) >= 0.
    fibres = snapshot['beam']['accepted_fibres']
    accumulated = np.array([fsum(row[2:]) for row in fibres])
    old = np.zeros_like(accumulated) if previous is None else np.array([
        fsum(row[2:]) for row in previous['beam']['accepted_fibres']])
    assert np.all(accumulated >= old-64*np.finfo(float).eps*np.maximum(abs(accumulated),abs(old)))
    (snapshot['path']/'balance.json').write_bytes(_canonical(dict(
        force_N=force, moment_Nm=moment, fibre_accumulated=accumulated)))


def same(a, b, model, *, exact=False, predecessors=None, sections=None,
        operators=None):
    # The default route stays the uniform comparator. Only the explicit
    # cross-chart comparison passes each branch's own accepted predecessor
    # capture and the model's actual retained-fibre operators; the hook then
    # re-verifies every fibre tangent/branch against the unchanged law - by
    # exact station-pair reconstruction, or by exact local cell replay from
    # the capture's own retained resultants when the published pairs lose
    # the 80-digit tail - before comparing the paired records.
    hook = None
    if not exact and predecessors is not None:
        hook = cyclic_fibre_boundary_hook(sections, operators, a, b,
            predecessors[0], predecessors[1])
    _same(a,b,model,FORCE_SCALE,exact=exact,fibre_material=hook)
    x,y = a['result'].point_wrenches,b['result'].point_wrenches
    assert len(x) == len(y)
    for left,right in zip(x,y):
        assert (left.point_index,left.shell_node_id,left.beam_node_id,left.configuration) == (
            right.point_index,right.shell_node_id,right.beam_node_id,right.configuration)
        for name,scale in (('shell_action',(FORCE_SCALE,)*3+(FORCE_SCALE*LENGTH,)*3),
                           ('beam_action',(FORCE_SCALE,)*3+(FORCE_SCALE*LENGTH,)*3),
                           ('gap',(LENGTH,)*3+(1.,)*3),
                           ('shell_position_m',LENGTH),('beam_position_m',LENGTH),
                           ('shell_position_low_m',LENGTH),('beam_position_low_m',LENGTH)):
            np.testing.assert_allclose(np.asarray(getattr(left,name))/scale,
                np.asarray(getattr(right,name))/scale,rtol=0.,atol=1e-9)


def coverage(history, field):
    frames = [row['shell_frames'] if field == 'shell' else
        row['result'].beam_frames if field == 'beam' else
        row['result'].beam_frames[2:] if field == 'fibre_nodes' else
        row['result'].cell_frames[1].reshape(-1,3,3) if field == 'fibre_cells' else
        row['result'].cell_frames.reshape(-1,3,3) for row in history]
    frames = [np.tile(np.eye(3),(len(frames[0]),1,1))]+frames
    increments = [b@a.transpose(0,2,1) for a,b in zip(frames,frames[1:])]
    angle = max(float(np.max(Rotation.from_matrix(a).magnitude())) for a in frames)
    comm = max(float(np.max(Rotation.from_matrix(b@a@b.transpose(0,2,1)@a.transpose(0,2,1)).magnitude()))
        for a,b in zip(increments,increments[1:]))
    return dict(rotation_rad=angle, commutator_rad=comm)


def has_flow(result):
    return any(fsum(f.plastic_increment) != 0. for station in result.beam_fibre_material
        for f in station.fibres)


def active_noncommuting_positions(history):
    # Both consecutive rotations belong to the physical-fibre member's actual
    # retained cell frames; an elastic neighbouring member cannot supply coverage.
    frames = [np.tile(np.eye(3),(2,1,1))]+[
        row['result'].cell_frames[1].reshape(-1,3,3) for row in history]
    increments = [b@a.transpose(0,2,1) for a,b in zip(frames,frames[1:])]
    positions = []
    for index,(a,b) in enumerate(zip(increments,increments[1:]),start=1):
        angles = Rotation.from_matrix(b@a@b.transpose(0,2,1)@a.transpose(0,2,1)).magnitude()
        result = history[index]['result']
        if result.load_position <= 1.:
            continue
        for cell,angle in enumerate(angles):
            active = any(station.element_id == 2 and station.cell == cell and
                any(fsum(f.plastic_increment) != 0. for f in station.fibres)
                for station in result.beam_fibre_material)
            if angle >= 1e-5 and active:
                positions.append(dict(position=result.load_position, element_id=2,
                    cell=cell, fibre_cell_commutator_rad=float(angle)))
    return positions


def test_loaded_native_fibre_cyclic_chart_transport(request, tmp_path, record_property):
    model = definition()
    def owner():
        made = ss.create_analysis(definition=model)
        request.addfinalizer(made.close)
        return made
    control = owner()
    loads = programme(control)
    (tmp_path/'input.json').write_bytes(_canonical(dict(
        programme=loads.to_payload(), model_identity=control.preflight()['identity'],
        prefix=PREFIX, following=FOLLOWING, solver_options=OPTIONS,
        max_cutbacks=4, max_attempts=64, length_m=LENGTH, force_scale_N=FORCE_SCALE,
        fibre_yield_pa=250e6, fibre_hardening_pa=210e9,
        rotation_floor_rad=.05, commutator_floor_rad=1e-5,
        scope='elastic Q4/mixed elastic-generalized and yielding biaxial-fibre point owner only')))
    histories = {}
    def run(branch, label, targets, history):
        histories[label] = history
        def observe(result):
            captured = _capture(branch, result, tmp_path/(label+'-'+str(len(history))))
            check(captured, history[-1] if history else None, branch)
            history.append(captured)
        try:
            return branch.solve(loads, targets, increment_control=CONTROL,
                on_accepted=observe, **OPTIONS)
        except Exception as error:
            if branch._finite_driver is not None and branch._finite_driver._accepted is not None:
                (tmp_path/(label+'-failed-prefix.checkpoint')).write_bytes(branch.checkpoint())
            (tmp_path/(label+'-failure.json')).write_bytes(_canonical(dict(
                type=type(error).__name__, message=str(error),
                accepted_positions=[row['result'].load_position for row in history])))
            raise
    initial = []
    prefix = run(control, 'prefix', PREFIX, initial)
    assert has_flow(prefix), 'no active fibre branch at the rebase predecessor'
    assert any(fsum(energy.incremental_dissipation_j) > 0. for row in initial
        for energy in row['result'].beam_fibre_energy)
    raw = control.checkpoint()
    reset = owner(); reset.restore(raw)
    before = _capture(reset, reset.recover(), tmp_path/'before-rebase')
    same(initial[-1], before, model, exact=True)
    token = CancellationToken(); token.cancel('before plastic chart reset')
    with pytest.raises(SolveCancelled):
        reset.rebase_rotations(prefix.generation, cancellation_token=token)
    assert reset.checkpoint() == raw
    with pytest.raises(ValueError, match='stale'):
        reset.rebase_rotations(prefix.generation-1)
    assert reset.checkpoint() == raw
    origins = reset._finite_driver._shell_origins, reset._finite_driver._beam_origins
    event = reset.rebase_rotations(prefix.generation)
    assert event.generation == prefix.generation+1
    after = _capture(reset, event, tmp_path/'after-rebase')
    assert reset._finite_driver._shell_origins is origins[0]
    assert reset._finite_driver._beam_origins is origins[1]
    same(before, after, model, exact=True)
    np.testing.assert_array_equal(event.shell_chart_coordinates.reshape(-1,6)[:,3:], 0.)
    chart = owner(); chart.restore(reset.checkpoint())
    restored = _capture(chart, chart.recover(), tmp_path/'chart-restored')
    same(after, restored, model, exact=True)
    assert chart.checkpoint() == reset.checkpoint()
    assert control.checkpoint() == raw
    for label, branch in (('control',control),('reset',reset),('restored',chart)):
        following = []
        run(branch, label, FOLLOWING, following)
    reference = histories['control']
    # Each branch's actual accepted predecessor chain. on_accepted captures
    # every accepted substep (adaptive control may insert steps), and each
    # capture snapshots the driver's immutable accepted origins at capture
    # time, so the capture immediately preceding a compared capture is the
    # actual predecessor its records were recovered from. The control chain
    # therefore starts at the prefix-final capture, and the reset/restored
    # chains at their rebase/restore captures. The hook refuses a tampered,
    # missing or cross-station predecessor and any recovery inconsistency.
    chains = {'control': initial[-1:]+histories['control'],
              'reset': [after]+histories['reset'],
              'restored': [restored]+histories['restored']}
    sections = fibre_sections(model)
    operators = cell_operators(model)
    for label in ('reset','restored'):
        assert len(histories[label]) == len(reference)
        for index,(a,b) in enumerate(zip(reference,histories[label])):
            same(a,b,model,predecessors=(chains['control'][index],chains[label][index]),
                 sections=sections, operators=operators)
    measured = {label:{field:coverage(initial+history,field)
        for field in ('shell','beam','cell','fibre_nodes','fibre_cells')}
        for label,history in histories.items() if label != 'prefix'}
    active_positions = {label:active_noncommuting_positions(initial+history)
        for label,history in histories.items() if label != 'prefix'}
    (tmp_path/'coverage.json').write_bytes(_canonical(measured))
    (tmp_path/'active-noncommuting.json').write_bytes(_canonical(active_positions))
    for branch in measured.values():
        for field,value in branch.items():
            assert value['rotation_rad'] >= .05, (field,value)
            assert value['commutator_rad'] >= 1e-5, (field,value)
    assert all(active_positions.values()), 'no active fibre branch on a post-reset noncommuting increment'
    assert all(v[-1]['result'].load_position == 4. for label,v in histories.items() if label != 'prefix')
    record_property('native_plastic_chart', json.dumps(dict(
        coverage=measured, active_noncommuting_positions=active_positions,
        accepted_positions=list(control.recover().accepted_factors),
        actual_prefix_plastic_flow=True, unchanged_rebase_histories=True,
        material_and_energy_successor_parity=True, state_path=str(tmp_path),
        scope='scoped fibre chart transport; no shell-J2, zero-load, capacity or total-energy qualification')))
