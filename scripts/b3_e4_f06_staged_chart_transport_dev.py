"""Frozen staged point chart diagnostic; outside CI and not qualification.

Stage one bends through beam105 Fz=1 MN. At its endpoint the shell chart is
reset/restored. Stage two retains that force and adds the independent shell
3/6 z-force couple of 375 kN. Actual accepted increments supply coverage.
Run under the root's installed-import/resource supervisor; no source path is
inserted here. The unchanged proportional diagnostic supplies shared oracles.
"""
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_checkpoint_binary import CHART_MAGIC
from anyfesolver._stiffened_shell_increment_control import decode_control
from anyfesolver._stiffened_shell_staged_load_experimental import (
    MAGIC as STAGED_MAGIC, decode_checkpoint,
)
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
import b3_e4_f06_loaded_chart_transport_dev as oracle


BENDING_FORCE = 1_000_000.0
COUPLE_FORCE = 375_000.0
STAGE1_POSITIONS = (.125, .5, .75, 1.)
STAGE2_POSITIONS = (1.125, 1.5, 1.75, 2.)
ROTATION_FLOOR = .05
COMMUTATOR_FLOOR = 1e-5


def _programme(owner):
    bending = stiffened_shell.NodalAction(
        "beam", 105, (0., 0., BENDING_FORCE, 0., 0., 0.))
    return stiffened_shell.StagedDeadLoadProgramme(owner._identity, (
        (bending,),
        (bending,
         stiffened_shell.NodalAction("shell", 3, (0., 0., -COUPLE_FORCE, 0., 0., 0.)),
         stiffened_shell.NodalAction("shell", 6, (0., 0., COUPLE_FORCE, 0., 0., 0.))),
    ))


def _physical_actions(position):
    # Independent known force path, not the compiled interpolation helper.
    assert 0. <= position <= 2.
    bending = min(position, 1.)*BENDING_FORCE
    couple = max(position-1., 0.)*COUPLE_FORCE
    return (("beam", 105, np.array((0., 0., bending))),
            ("shell", 3, np.array((0., 0., -couple))),
            ("shell", 6, np.array((0., 0., couple))))


def _snapshot(owner, record_path):
    """Staged force accounting; retain the old physical/material comparisons."""
    driver, result = owner._finite_driver, owner.recover()
    state = driver._state
    reference = np.asarray(owner._definition.shell.coordinates_m)
    rows = state.q[:owner._shell_count].reshape(-1, 6)
    shell_positions = reference+rows[:, :3]
    shell_frames = Rotation.from_rotvec(rows[:, 3:].copy()).as_matrix()
    if state.shell_anchors is not None:
        shell_frames = shell_frames @ state.shell_anchors
    beam_reference = np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
    reference_frames = np.array([owner._beam_nodes[n][1] for n in owner._beam_ids])
    beam_positions = result.beam_positions.copy()
    gaps = []
    for point in owner._definition.points:
        s, b = owner._shell_index[point.shell_node], owner._beam_index[point.beam_node]
        rs, rb = shell_frames[s], result.beam_frames[b]
        translation = rs.T @ (beam_positions[b]-shell_positions[s])
        translation -= beam_reference[b]-reference[s]
        orientation = Rotation.from_matrix(reference_frames[b].T @ rs.T @ rb).as_rotvec()
        gaps.append(np.r_[translation, orientation])
    support = result.support_reactions[:owner._shell_count].reshape(-1, 6).copy()
    applied_force, applied_moment, potential = np.zeros(3), np.zeros(3), 0.
    expected_applied = np.zeros(owner._K.shape[0])
    for domain, node, force in _physical_actions(result.load_position):
        if domain == "shell":
            index, offset = owner._shell_index[node], 0
            position, initial = shell_positions[index], reference[index]
        else:
            index, offset = owner._beam_index[node], owner._shell_count
            position, initial = beam_positions[index], beam_reference[index]
        expected_applied[offset+6*index:offset+6*index+3] = force
        applied_force += force
        applied_moment += np.cross(position, force)
        potential -= float(force @ (position-initial))
    reaction_force = support[:, :3].sum(axis=0)
    reaction_moment = (support[:, 3:]+np.cross(shell_positions, support[:, :3])).sum(axis=0)
    fixed = [owner._shell_index[n] for n in (1, 4)]
    shell_material, beam_material = oracle._plain(driver._shell_origins), oracle._plain(driver._beam_origins)
    snapshot = dict(
        shell_positions=shell_positions, shell_frames=shell_frames,
        beam_positions=beam_positions, beam_frames=result.beam_frames.copy(),
        cell_frames=result.cell_frames.copy(), resultants=result.cell_resultants.copy(),
        multipliers=result.multipliers.copy(), support=support, gaps=np.asarray(gaps),
        potential=potential,
        point_actions=np.array([record.shell_action+record.beam_action for record in result.point_wrenches]),
        material_sha256=sha256(canonical_json_bytes((shell_material, beam_material))).hexdigest(),
        shell_material=shell_material, beam_material=beam_material,
        material_inputs=dict(E=owner._definition.shell.young_pa,
                             nu=owner._definition.shell.poisson, t=owner._definition.shell.thickness_m),
        accepted_factors=result.accepted_factors,
        coordinate_scale=owner._coordinate_scale.copy(), constraint_scale=owner._constraint_scale.copy(),
        status=result.status, generation=result.generation,
        free_equilibrium_ratio=result.free_equilibrium_ratio, constraint_max=result.constraint_max,
        shell_chart_rows=rows.copy(), force_balance=reaction_force+applied_force,
        moment_balance=reaction_moment+applied_moment,
        load_programme_identity=result.load_programme_identity, load_position=result.load_position,
        load_stage_cursor=result.load_stage_cursor, load_stage_fraction=result.load_stage_fraction,
        load_programme_complete=result.load_programme_complete,
        applied_dead_load=result.applied_dead_load.copy(), expected_applied_dead_load=expected_applied,
    )
    # Complete detached origins and arrays precede every comparison/assertion.
    record_path.write_text(json.dumps(oracle._plain(snapshot), sort_keys=True, allow_nan=False), encoding="utf-8")
    assert result.status in ("accepted_prefix", "completed")
    assert result.load_factor == result.load_position
    assert result.load_programme_identity == driver._programme.identity
    cursor = int(np.floor(result.load_position))
    assert result.load_stage_cursor == cursor
    assert result.load_stage_fraction == result.load_position-cursor
    assert result.load_programme_complete == (result.load_position == 2.)
    np.testing.assert_allclose(result.applied_dead_load/oracle.FORCE_SCALE,
                               expected_applied/oracle.FORCE_SCALE, rtol=0., atol=1e-15)
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    assert np.max(np.abs(gaps)/np.array((oracle.LENGTH,)*3+(1.,)*3)) <= 1e-9
    assert np.linalg.norm(reaction_force+applied_force)/oracle.FORCE_SCALE <= 1e-8
    assert np.linalg.norm(reaction_moment+applied_moment)/oracle.MOMENT_SCALE <= 1e-8
    np.testing.assert_allclose(shell_positions[fixed], reference[fixed], rtol=0., atol=1e-12)
    np.testing.assert_allclose(shell_frames[fixed], np.broadcast_to(np.eye(3), shell_frames[fixed].shape), rtol=0., atol=1e-12)
    for frames in (shell_frames, result.beam_frames, result.cell_frames.reshape(-1, 3, 3)):
        np.testing.assert_allclose(frames @ frames.transpose(0, 2, 1),
                                   np.broadcast_to(np.eye(3), frames.shape), rtol=0., atol=1e-9)
        assert np.all(np.linalg.det(frames) > 0.)
    assert np.max(np.linalg.norm(rows[:, 3:], axis=1)) < .9*np.pi
    assert len(result.point_wrenches) == len(owner._definition.points)
    oracle._assert_elastic_material(owner, snapshot)
    return snapshot


def _same(left, right, *, new_increment=False):
    oracle._assert_same_physical(left, right, new_increment=new_increment)
    for name in ("load_programme_identity", "load_position", "load_stage_cursor",
                 "load_stage_fraction", "load_programme_complete"):
        assert left[name] == right[name]
    np.testing.assert_array_equal(left["applied_dead_load"], right["applied_dead_load"])


def _advance(owner, programme, targets, folder, label):
    accepted = []
    predecessor = owner._finite_driver
    prefix = () if predecessor is None else predecessor._accepted_factors
    def observe(result):
        assert result.generation == owner.recover().generation
        assert result.accepted_factors == prefix+tuple(
            item["load_position"] for item in accepted)+(result.load_position,)
        accepted.append(_snapshot(owner, folder/f"accepted-{label}-{len(accepted)+1}.json"))
    result = owner.solve(programme, targets, max_iterations=20, max_backtracks=12,
        increment_control=stiffened_shell.AdaptiveIncrementControl(), on_accepted=observe)
    assert result.status == "completed"
    assert accepted and result.load_position == targets[-1]
    assert result.accepted_factors == prefix+tuple(item["load_position"] for item in accepted)
    assert set(targets).issubset(result.accepted_factors)
    return accepted


def _retain_first_failure(function):
    @wraps(function)
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            # pytest passes fixture arguments by keyword; direct caller may use
            # the declared positional signature. Never replace the first error.
            folder = kwargs.get("tmp_path", args[3] if len(args) > 3 else None)
            if folder is not None:
                path = Path(folder)/"first-failure.json"
                if not path.exists():
                    try:
                        path.write_text(json.dumps(dict(type=type(error).__name__, message=str(error),
                            traceback=traceback.format_exc()), sort_keys=True), encoding="utf-8")
                    except Exception as recording_error:
                        if hasattr(error, "add_note"):
                            error.add_note(f"first-failure recorder also failed: {recording_error!r}")
            raise
    return run


@pytest.mark.parametrize("triangles", (False, True), ids=("selected-q4", "all-s3"))
@_retain_first_failure
def test_staged_noncommuting_point_chart_reset_restore_continue(
        triangles, record_property, request, tmp_path):
    assert (BENDING_FORCE, COUPLE_FORCE, ROTATION_FLOOR, COMMUTATOR_FLOOR) == (
        oracle.BENDING_FORCE, oracle.COUPLE_FORCE, oracle.ROTATION_FLOOR, oracle.COMMUTATOR_FLOOR)
    definition = oracle._fixture(triangles)
    control = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(control.close)
    programme = _programme(control)
    (tmp_path/"frozen-programme.json").write_bytes(programme.serialize())
    (tmp_path/"prospective-path.json").write_text(json.dumps(dict(
        stage1=STAGE1_POSITIONS, stage2=STAGE2_POSITIONS, reset_position=1.,
        rotation_floor=ROTATION_FLOOR, commutator_floor=COMMUTATOR_FLOOR,
        max_iterations=20, max_backtracks=12,
        adaptive=oracle._plain(stiffened_shell.AdaptiveIncrementControl())), sort_keys=True), encoding="utf-8")
    history = _advance(control, programme, STAGE1_POSITIONS, tmp_path, "stage1-control")
    prefix = control.checkpoint()
    (tmp_path/"stage1-prefix.checkpoint").write_bytes(prefix)
    assert prefix.startswith(STAGED_MAGIC)
    rebased = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(rebased.close)
    rebased.restore(prefix)
    before = _snapshot(rebased, tmp_path/"stage1-restored.json")
    _same(history[-1], before)
    driver = rebased._finite_driver
    predecessor, origins = driver._state, (driver._shell_origins, driver._beam_origins)
    result = rebased.rebase_rotations(rebased.recover().generation)
    assert result.generation == control.recover().generation+1
    assert driver._shell_origins is origins[0] and driver._beam_origins is origins[1]
    for name in ("frames", "cells", "resultants", "multipliers"):
        assert getattr(driver._state, name) is getattr(predecessor, name)
    np.testing.assert_array_equal(result.shell_chart_coordinates.reshape(-1, 6)[:, 3:], 0.)
    np.testing.assert_allclose(result.shell_rotation_anchors, before["shell_frames"], rtol=0., atol=1e-12)
    _same(before, _snapshot(rebased, tmp_path/"stage1-reset.json"))
    anchored = rebased.checkpoint()
    (tmp_path/"stage1-reset.checkpoint").write_bytes(anchored)
    descriptor, position, inner = decode_checkpoint(anchored)
    assert descriptor.identity == programme.identity and position == 1.
    controller, chart = decode_control(inner, descriptor.augmented_model_identity)
    assert controller.completed
    assert chart.startswith(CHART_MAGIC)
    resumed = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(resumed.close)
    resumed.restore(anchored)
    _same(before, _snapshot(resumed, tmp_path/"stage1-reset-restored.json"))
    assert control.checkpoint() == prefix
    successors, branch_histories = {}, {}
    for branch, owner in (("control", control), ("reset", rebased), ("restored", resumed)):
        accepted = _advance(owner, programme, STAGE2_POSITIONS, tmp_path, f"stage2-{branch}")
        branch_histories[branch] = history+accepted
        successors[branch] = _snapshot(owner, tmp_path/f"final-{branch}.json")
        (tmp_path/f"final-{branch}.checkpoint").write_bytes(owner.checkpoint())
    coverage = {}
    for branch in successors:
        for domain in ("shell_frames", "beam_frames"):
            rotation, commutator = oracle._coverage(branch_histories[branch], domain)
            record_property(f"{branch}_{domain}_rotation_rad", rotation)
            record_property(f"{branch}_{domain}_commutator_rad", commutator)
            coverage[f"{branch}_{domain}"] = (rotation, commutator)
    (tmp_path/"coverage.json").write_text(json.dumps(coverage, sort_keys=True, allow_nan=False), encoding="utf-8")
    for branch in ("reset", "restored"):
        _same(successors["control"], successors[branch], new_increment=True)
        assert len(branch_histories[branch]) == len(branch_histories["control"])
        for expected, actual in zip(branch_histories["control"][len(history):],
                                    branch_histories[branch][len(history):]):
            _same(expected, actual, new_increment=True)
    for branch_domain, (rotation, commutator) in coverage.items():
        assert rotation >= ROTATION_FLOOR, (branch_domain, rotation)
        assert commutator >= COMMUTATOR_FLOOR, (branch_domain, commutator)


if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), *sys.argv[1:]]))
