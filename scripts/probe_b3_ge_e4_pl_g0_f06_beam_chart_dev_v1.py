"""Loaded native B3 beam-chart description discriminator; not an owner rebase."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('explicit writable NUMBA_CACHE_DIR required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

import anyfesolver
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from scripts.probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4 import OwnedRebasePrototype


TOLERANCE = 1e-8
FLOORS = dict(translation=1e-9, rotation=1e-9, force=1e-9,
    moment=1e-9, strain=1e-12, curvature=1e-12, mixed_solver=1e-9)


def metric(left, right, unit):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'{unit} shape/nonfinite mismatch')
    absolute = float(np.linalg.norm(a-b))
    scale = max(FLOORS[unit], float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    return dict(unit=unit, absolute=absolute, scale=scale, relative=absolute/scale)


def angle_metric(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.shape[-2:] != (3, 3):
        raise ValueError('rotation frame shape mismatch')
    angles = np.array([Rotation.from_matrix(x@y.T).magnitude()
        for x, y in zip(a.reshape(-1, 3, 3), b.reshape(-1, 3, 3))])
    absolute = float(np.linalg.norm(angles))
    return dict(unit='rotation', absolute=absolute, scale=1.,
        relative=absolute)


def describe_frames(physical, anchors, fixed=()):
    coordinates = []
    reconstructed = []
    for i, (q, anchor) in enumerate(zip(physical, anchors)):
        if i in fixed:
            theta = np.zeros(3)
            made = np.array(q, copy=True)
        else:
            theta = Rotation.from_matrix(q@anchor.T).as_rotvec()
            made = Rotation.from_rotvec(theta).as_matrix()@anchor
        coordinates.append(theta)
        reconstructed.append(made)
    return np.array(coordinates), np.array(reconstructed)


def beam_description(beam, base, rebase):
    original = base.mechanical
    data = {name: np.array(value, copy=True)
        for name, value in original.descriptor().items()}
    nodal_anchors = (original.nodal_frames if rebase else beam.reference_frames)
    cell_anchors = (original.cell_rotations if rebase else
        np.tile(np.eye(3), (len(beam.elements), 2, 1, 1)))
    fixed_nodes = tuple(sorted({int(d)//6 for d in beam.fixed if int(d)%6 >= 3}))
    nodal_theta, data['nodal_frames'] = describe_frames(
        original.nodal_frames, nodal_anchors, fixed_nodes)
    cell_theta, cell_frames = describe_frames(original.cell_rotations.reshape(-1, 3, 3),
        cell_anchors.reshape(-1, 3, 3))
    data['cell_rotations'] = cell_frames.reshape(original.cell_rotations.shape)
    made = beam.make(data)
    return made, nodal_anchors, nodal_theta, cell_anchors, cell_theta


def section_values(owner, state, mechanical):
    values = {name: [] for name in ('strain', 'strain_low', 'resultants', 'resultants_low')}
    histories = []
    for i, (_, element) in enumerate(owner.assembly.beam.elements):
        stations = element.operator.recover(mechanical.cell_rotations[i],
            mechanical.resultants[i], origin=state.beam_origins[i])
        for station in stations:
            for name in values:
                values[name].append(station[name])
            histories.append(canonical(station['history']))
    return {name: np.asarray(rows) for name, rows in values.items()}, histories


def port(owner, state, mechanical, beam_anchor, beam_theta):
    a = owner.assembly
    bi = a.beam.index[a.beam_node]
    shell = state.shell_u.reshape(-1, 6)[a.shell_node]
    positions = np.array((a.coordinates[a.shell_node]+shell[:3], mechanical.positions[bi]))
    frames = np.array((a.frame, beam_anchor[bi]))
    coordinate = np.array((shell[3:], beam_theta[bi]))
    return a.joint.evaluate_ports(positions, frames, state.multipliers,
        rotation_coordinates=coordinate, charted_ports=(True, True),
        position_low=np.array((np.zeros(3), mechanical.position_low[bi])))


def measure():
    prototype = OwnedRebasePrototype()
    accepted = prototype.legacy_prefix()
    if accepted.status != 'paused' or accepted.state.cursor != 2:
        raise ValueError('loaded two-step accepted prefix required')
    owner, base = prototype.owner, accepted.state
    beam = owner.assembly.beam
    prefix_raw = canonical(base.descriptor())
    checkpoint_raw = bytes(accepted.checkpoint)
    old, old_anchor, old_theta, old_cell_anchor, old_cell_theta = beam_description(
        beam, base, False)
    new, new_anchor, new_theta, new_cell_anchor, new_cell_theta = beam_description(
        beam, base, True)
    if (np.max(np.linalg.norm(old_theta, axis=1)) <= 1e-5 or
            np.linalg.norm(base.multipliers) <= 1e-6):
        raise ValueError('nontrivial loaded rotation and multiplier required')
    original_response, original_loaded, _, _ = owner._evaluate(base,
        base.mechanical, base.shell_u, base.multipliers, 2)
    left_response, left_loaded, _, _ = owner._evaluate(base,
        old, base.shell_u, base.multipliers, 2)
    right_response, right_loaded, _, _ = owner._evaluate(base,
        new, base.shell_u, base.multipliers, 2)
    old_port = port(owner, base, old, old_anchor, old_theta)
    new_port = port(owner, base, new, new_anchor, new_theta)
    original_sections, original_histories = section_values(owner, base, base.mechanical)
    left_sections, left_histories = section_values(owner, base, old)
    right_sections, right_histories = section_values(owner, base, new)
    metrics = {}
    metrics['nodal_frame'] = angle_metric(old.nodal_frames, new.nodal_frames)
    metrics['cell_frame'] = angle_metric(old.cell_rotations, new.cell_rotations)
    metrics['positions'] = metric(old.positions, new.positions, 'translation')
    metrics['retained_force'] = metric(old.resultants[:, :6], new.resultants[:, :6], 'force')
    metrics['retained_moment'] = metric(old.resultants[:, 6:], new.resultants[:, 6:], 'moment')
    metrics['gap_translation'] = metric(old_port.pose.constraints[:3],
        new_port.pose.constraints[:3], 'translation')
    metrics['gap_rotation'] = metric(old_port.pose.constraints[3:],
        new_port.pose.constraints[3:], 'rotation')
    metrics['pose_spatial_force'] = metric(old_port.pose.residual[:12].reshape(2, 6)[:, :3],
        new_port.pose.residual[:12].reshape(2, 6)[:, :3], 'force')
    metrics['pose_spatial_moment'] = metric(old_port.pose.residual[:12].reshape(2, 6)[:, 3:],
        new_port.pose.residual[:12].reshape(2, 6)[:, 3:], 'moment')
    metrics['loaded_residual'] = metric(left_loaded, right_loaded, 'mixed_solver')
    metrics['baseline_loaded_residual'] = metric(original_loaded, right_loaded, 'mixed_solver')
    metrics['baseline_gap_translation'] = metric(original_response.constraints[:3],
        right_response.constraints[:3], 'translation')
    metrics['baseline_gap_rotation'] = metric(original_response.constraints[3:],
        right_response.constraints[3:], 'rotation')
    shell_count = owner.assembly.shell_count
    fixed = sorted(set(range(shell_count+beam.nodal_count))-set(int(i) for i in owner.free))
    if not fixed:
        raise ValueError('physical fixed support reaction rows required')
    force_rows = [i for i in fixed if i%6 < 3]
    moment_rows = [i for i in fixed if i%6 >= 3]
    metrics['support_force'] = metric(left_loaded[force_rows], right_loaded[force_rows], 'force')
    metrics['support_moment'] = metric(left_loaded[moment_rows], right_loaded[moment_rows], 'moment')
    if max(np.linalg.norm(left_loaded[force_rows]),
           np.linalg.norm(left_loaded[moment_rows])) <= 1e-6:
        raise ValueError('nontrivial beam reaction required')
    for name in ('strain', 'strain_low', 'resultants', 'resultants_low'):
        units = ('strain', 'curvature') if name.startswith('strain') else ('force', 'moment')
        for part, unit in enumerate(units):
            rows = slice(3*part, 3*part+3)
            key = name+'_'+unit
            metrics['section_'+key] = metric(left_sections[name][:, rows],
                right_sections[name][:, rows], unit)
            metrics['baseline_section_'+key] = metric(original_sections[name][:, rows],
                right_sections[name][:, rows], unit)
    if (original_histories != left_histories or original_histories != right_histories):
        # Float-level reconstruction may differ; history ownership remains
        # accepted-state data and is never replaced by these trial recoveries.
        history_trial_identical = False
    else:
        history_trial_identical = True
    bi = beam.index[owner.assembly.beam_node]
    spatial_direction = np.array((.17, -.11, .23))
    old_chart_direction = np.linalg.solve(_exp_terms(old_theta[bi])[1], spatial_direction)
    old_work = float(old_port.residual[9:12]@old_chart_direction)
    new_work = float(new_port.residual[9:12]@spatial_direction)
    work_error = abs(old_work-new_work)/max(FLOORS['moment'], abs(old_work), abs(new_work))
    prefix_unchanged = (canonical(base.descriptor()) == prefix_raw
        and accepted.checkpoint == checkpoint_raw)
    finite = all(np.isfinite(m['relative']) for m in metrics.values())
    failed_metrics = [name for name, m in metrics.items()
        if m['absolute'] > FLOORS[m['unit']] and m['relative'] > TOLERANCE]
    passed = (prefix_unchanged and finite and work_error <= TOLERANCE
        and not failed_metrics)
    return dict(status='DEV_F06_BEAM_CHART_PASS' if passed else 'DEV_F06_BEAM_CHART_FAIL',
        accepted_prefix_sha256=sha256(prefix_raw).hexdigest(),
        accepted_checkpoint_sha256=sha256(checkpoint_raw).hexdigest(),
        accepted_prefix_unchanged=prefix_unchanged,
        beam_nodal_chart_max_angle=float(np.max(np.linalg.norm(old_theta, axis=1))),
        beam_cell_chart_max_angle=float(np.max(np.linalg.norm(old_cell_theta, axis=1))),
        rebased_nodal_coordinate_max=float(np.max(np.abs(new_theta))),
        rebased_cell_coordinate_max=float(np.max(np.abs(new_cell_theta))),
        multiplier_norm=float(np.linalg.norm(base.multipliers)),
        support_force_norm=float(np.linalg.norm(left_loaded[force_rows])),
        support_moment_norm=float(np.linalg.norm(left_loaded[moment_rows])),
        chart_virtual_work_relative_error=work_error,
        recovered_history_bits_identical=history_trial_identical,
        accepted_beam_history_unchanged=(base.beam_histories == accepted.state.beam_histories),
        failed_metrics=failed_metrics,
        metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    modules = ('_ge_beam3_retained_generalized_state.py',
        '_ge_beam3_shell_joint_state.py', '_ge_beam3_shell_joint_trial.py',
        '_ge_beam3_pose_joint.py', '_ge_beam3_generalized_static_boundary.py')
    manifest = dict(purpose='F06_DEV_LOADED_BEAM_CHART_DESCRIPTION_NOT_ACCEPTANCE',
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256={name: sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
            for name in ('probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py',
                'probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3.py')},
        module_sha256={name: sha256((ROOT/'src'/'anyfesolver'/name).read_bytes()).hexdigest()
            for name in modules}, imported_package=str(Path(anyfesolver.__file__).resolve()),
        python=sys.version, numpy=np.__version__,
        numba_jit_disabled=os.environ.get('NUMBA_DISABLE_JIT') == '1',
        tolerance=TOLERANCE, floors=FLOORS)
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    try:
        with threadpool_limits(limits=1):
            result = measure()
    except Exception as error:
        result = dict(status='DEV_F06_BEAM_CHART_EXCEPTION',
            failure_type=type(error).__name__, failure=str(error))
    (args.root/'result.json').write_text(json.dumps(result, indent=2,
        sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],
        max_metric=max((m['relative'] for m in result.get('metrics', {}).values()), default=None),
        failure=result.get('failure'))))
    return 0 if result['status'] == 'DEV_F06_BEAM_CHART_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
