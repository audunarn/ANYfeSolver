"""Loaded point-owner chart transport development diagnostic; not a CI pass.

The accepted straight panel is driven by a physical tip force couple and beam
transverse force. This differs from the retained unsuccessful curved-arc case.
The finite owner keeps one immutable dead-load pattern. Actual accepted physical
increments must demonstrate noncommutation; chart shadows are never replayed.
"""

from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_checkpoint_binary import CHART_MAGIC
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from test_stiffened_shell_ge_experimental import _panel


LENGTH = 2.0
# One prospective stronger candidate after V1 lacked accepted noncommutation.
# This changes forcing, not the coverage floors or the nonlinear solve budget.
BENDING_FORCE = 1_000_000.0  # N at the beam tip, in global z.
COUPLE_FORCE = 375_000.0  # Opposite z forces, separated by the 1 m shell width.
FORCE_SCALE = BENDING_FORCE + 2.0 * COUPLE_FORCE
MOMENT_SCALE = FORCE_SCALE * LENGTH
PREFIX_FACTORS = (0.125, 0.5)
FINAL_FACTOR = 1.0
ROTATION_FLOOR = 0.05
COMMUTATOR_FLOOR = 1e-5
RECOMPUTED_FIELDS = (
    "committed_total_u", "committed_constitutive_u", "layer_strain",
    "layer_stress", "station_generalized_strain", "station_generalized_resultant",
)


def _fixture(triangles):
    base = _panel()
    shell = base.shell
    if triangles:
        shell = replace(shell, q4_elements=(),
                        s3_elements=((1, 2, 5), (1, 5, 4),
                                     (2, 3, 6), (2, 6, 5)))
    return replace(
        base, shell=shell,
        q4_pullback_policy=("historical_corotational" if triangles else
                            stiffened_shell.Q4_ALGORITHMIC_PULLBACK_V1))


def _actions():
    # Translation forces only: no unsupported finite nodal moment is requested.
    # The shell pair supplies a genuine global-x couple; beam force bends in y.
    return (
        stiffened_shell.NodalAction("beam", 105,
                                   (0., 0., BENDING_FORCE, 0., 0., 0.)),
        stiffened_shell.NodalAction("shell", 3,
                                   (0., 0., -COUPLE_FORCE, 0., 0., 0.)),
        stiffened_shell.NodalAction("shell", 6,
                                   (0., 0., COUPLE_FORCE, 0., 0., 0.)),
    )


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _snapshot(owner, record_path):
    driver, result = owner._finite_driver, owner.recover()
    state = driver._state
    reference = np.asarray(owner._definition.shell.coordinates_m)
    rows = state.q[:owner._shell_count].reshape(-1, 6)
    shell_positions = reference + rows[:, :3]
    # Independent physical reconstruction, not the production chart helper.
    shell_frames = Rotation.from_rotvec(rows[:, 3:].copy()).as_matrix()
    if state.shell_anchors is not None:
        shell_frames = shell_frames @ state.shell_anchors
    beam_reference = np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
    beam_reference_frames = np.array([owner._beam_nodes[n][1] for n in owner._beam_ids])
    beam_positions = result.beam_positions.copy()
    gaps = []
    for point in owner._definition.points:
        s, b = owner._shell_index[point.shell_node], owner._beam_index[point.beam_node]
        rs, rb = shell_frames[s], result.beam_frames[b]
        translation = rs.T @ (beam_positions[b] - shell_positions[s])
        translation -= beam_reference[b] - reference[s]
        orientation = Rotation.from_matrix(
            beam_reference_frames[b].T @ rs.T @ rb).as_rotvec()
        gaps.append(np.r_[translation, orientation])
    support = result.support_reactions[:owner._shell_count].reshape(-1, 6).copy()
    applied_force, applied_moment, potential = np.zeros(3), np.zeros(3), 0.
    for action in _actions():
        if action.domain == "shell":
            index = owner._shell_index[action.node]
            position, initial = shell_positions[index], reference[index]
        else:
            index = owner._beam_index[action.node]
            position, initial = beam_positions[index], beam_reference[index]
        force = result.load_factor * np.asarray(action.wrench[:3])
        applied_force += force
        applied_moment += np.cross(position, force)
        potential -= float(force @ (position - initial))
    reaction_force = support[:, :3].sum(axis=0)
    reaction_moment = (support[:, 3:] +
                       np.cross(shell_positions, support[:, :3])).sum(axis=0)
    fixed = [owner._shell_index[n] for n in (1, 4)]
    shell_material = _plain(driver._shell_origins)
    beam_material = _plain(driver._beam_origins)
    materials = canonical_json_bytes((shell_material, beam_material))
    snapshot = dict(
        shell_positions=shell_positions, shell_frames=shell_frames,
        beam_positions=beam_positions, beam_frames=result.beam_frames.copy(),
        cell_frames=result.cell_frames.copy(), resultants=result.cell_resultants.copy(),
        multipliers=result.multipliers.copy(), support=support,
        gaps=np.asarray(gaps), potential=potential,
        point_actions=np.array([record.shell_action + record.beam_action
                                for record in result.point_wrenches]),
        material_sha256=sha256(materials).hexdigest(),
        shell_material=shell_material, beam_material=beam_material,
        material_inputs=dict(E=owner._definition.shell.young_pa,
                             nu=owner._definition.shell.poisson,
                             t=owner._definition.shell.thickness_m),
        accepted_factors=result.accepted_factors,
        coordinate_scale=owner._coordinate_scale.copy(),
        constraint_scale=owner._constraint_scale.copy(),
        status=result.status, free_equilibrium_ratio=result.free_equilibrium_ratio,
        constraint_max=result.constraint_max, shell_chart_rows=rows.copy(),
        force_balance=reaction_force + applied_force,
        moment_balance=reaction_moment + applied_moment,
    )
    # Detached complete origins and physical arrays precede every comparison.
    record_path.write_text(json.dumps(_plain(snapshot), sort_keys=True, allow_nan=False),
                           encoding="utf-8")
    assert result.status in ("accepted_prefix", "completed")
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    assert np.max(np.abs(gaps) / np.array((LENGTH,) * 3 + (1.,) * 3)) <= 1e-9
    assert np.linalg.norm(reaction_force + applied_force) / FORCE_SCALE <= 1e-8
    assert np.linalg.norm(reaction_moment + applied_moment) / MOMENT_SCALE <= 1e-8
    np.testing.assert_allclose(shell_positions[fixed], reference[fixed], rtol=0., atol=1e-12)
    np.testing.assert_allclose(shell_frames[fixed],
                               np.broadcast_to(np.eye(3), shell_frames[fixed].shape),
                               rtol=0., atol=1e-12)
    for frames in (shell_frames, result.beam_frames, result.cell_frames.reshape(-1, 3, 3)):
        np.testing.assert_allclose(
            frames @ frames.transpose(0, 2, 1),
            np.broadcast_to(np.eye(3), frames.shape), rtol=0., atol=1e-9)
        assert np.all(np.linalg.det(frames) > 0.)
    assert np.max(np.linalg.norm(rows[:, 3:], axis=1)) < .9 * np.pi
    assert len(result.point_wrenches) == len(owner._definition.points)
    return snapshot


def _record_snapshot(owner, tmp_path, label):
    snapshot = _snapshot(owner, tmp_path / f"{label}.json")
    _assert_elastic_material(owner, snapshot)
    return snapshot


def _assert_elastic_material(owner, snapshot):
    assert len(snapshot["shell_material"]) == len(owner._shell_parts)
    material = owner._shell_model.get_material("shell")
    e, nu = snapshot["material_inputs"]["E"], snapshot["material_inputs"]["nu"]
    plane_stress = e / (1. - nu**2) * np.array(
        ((1., nu, 0.), (nu, 1., 0.), (0., 0., (1. - nu) / 2.)))
    for (_, element, slots, _), origin in zip(
            owner._shell_parts, snapshot["shell_material"]):
        np.testing.assert_array_equal(np.asarray(origin["plastic_strain"]), 0.)
        np.testing.assert_array_equal(np.asarray(origin["alpha"]), 0.)
        if len(slots) == 18:
            # Validates model identity, full schema, finite fields and integrity.
            element.validate_model_bound_nonlinear_state(
                owner._shell_model.mesh, material, origin, 3)
            np.testing.assert_allclose(
                np.asarray(origin["layer_stress"]),
                np.asarray(origin["layer_strain"]) @ plane_stress.T,
                rtol=1e-9, atol=2e-6)  # Pa; independent elastic material oracle.
    assert len(snapshot["beam_material"]) == len(owner._beam_parts)
    for cell in snapshot["beam_material"]:
        assert set(cell) == {"cell_identity", "stations"}
        for origin in cell["stations"]:
            assert set(origin) == {"section_identity", "plastic", "accumulated"}
            np.testing.assert_array_equal(np.asarray(origin["plastic"]), 0.)
            np.testing.assert_array_equal(np.asarray(origin["accumulated"]), 0.)


def _assert_successor_material(left, right):
    # Only after a NEW increment may derived elastic response fields differ.
    # All IDs, initial inputs, schemas and irreversible histories remain exact.
    assert left["material_inputs"] == right["material_inputs"]
    assert canonical_json_bytes(left["beam_material"]) == canonical_json_bytes(right["beam_material"])
    assert len(left["shell_material"]) == len(right["shell_material"])
    e, nu, t = (left["material_inputs"][key] for key in ("E", "nu", "t"))
    g = e / (2. * (1. + nu))
    # SI: displacement m/rad; strains 1 and curvature 1/m; stresses Pa;
    # generalized resultants N/m (membrane/shear) and N (bending moments).
    scales = {
        "committed_total_u": np.tile((LENGTH,) * 3 + (1.,) * 3, 3),
        "committed_constitutive_u": np.tile((LENGTH,) * 3 + (1.,) * 3, 3),
        "layer_strain": 1., "layer_stress": e,
        "station_generalized_strain": np.array((1.,) * 3 + (1. / LENGTH,) * 3 + (1.,) * 2),
        "station_generalized_resultant": np.array((e * t,) * 3 + (e * t**3 / LENGTH,) * 3 + (g * t,) * 2),
    }
    for a, b in zip(left["shell_material"], right["shell_material"]):
        assert set(a) == set(b)
        if not set(RECOMPUTED_FIELDS).issubset(a):
            # Q4 origins contain irreversible history only.
            assert canonical_json_bytes(a) == canonical_json_bytes(b)
            continue
        exact_keys = set(a) - set(RECOMPUTED_FIELDS) - {"state_integrity_sha256"}
        assert canonical_json_bytes({key: a[key] for key in exact_keys}) == canonical_json_bytes(
            {key: b[key] for key in exact_keys})
        changed = False
        for key in RECOMPUTED_FIELDS:
            aa, bb = np.asarray(a[key]), np.asarray(b[key])
            assert aa.shape == bb.shape
            assert np.all(np.isfinite(aa)) and np.all(np.isfinite(bb))
            np.testing.assert_allclose(aa / scales[key], bb / scales[key], rtol=0., atol=1e-9)
            changed |= canonical_json_bytes(a[key]) != canonical_json_bytes(b[key])
        # A validated integrity digest is derived, not an immutable identity.
        # It may differ only when at least one explicitly declared field differs.
        if not changed:
            assert a["state_integrity_sha256"] == b["state_integrity_sha256"]


def _assert_same_physical(left, right, *, new_increment=False):
    assert left["accepted_factors"] == right["accepted_factors"]
    if new_increment:
        _assert_successor_material(left, right)
    else:
        assert left["material_sha256"] == right["material_sha256"]
    for name in ("shell_positions", "beam_positions"):
        np.testing.assert_allclose(left[name] / LENGTH, right[name] / LENGTH,
                                   rtol=0., atol=1e-9)
    for name in ("shell_frames", "beam_frames", "cell_frames", "gaps"):
        np.testing.assert_allclose(left[name], right[name], rtol=0., atol=1e-9)
    wrench_scale = np.array((FORCE_SCALE,) * 3 + (MOMENT_SCALE,) * 3)
    for name in ("resultants", "multipliers", "support", "point_actions"):
        np.testing.assert_allclose(left[name].reshape(-1, 6) / wrench_scale,
                                   right[name].reshape(-1, 6) / wrench_scale,
                                   rtol=0., atol=1e-9)
    assert abs(left["potential"] - right["potential"]) / MOMENT_SCALE <= 1e-8
    for name in ("coordinate_scale", "constraint_scale"):
        np.testing.assert_array_equal(left[name], right[name])


def _coverage(history, domain):
    frames = [np.tile(np.eye(3), (len(history[0][domain]), 1, 1))]
    frames.extend(item[domain] for item in history)
    rotation = max(float(np.max(Rotation.from_matrix(
        made @ frames[0].transpose(0, 2, 1)).magnitude())) for made in frames[1:])
    increments = [b @ a.transpose(0, 2, 1) for a, b in zip(frames, frames[1:])]
    commutator = max(float(np.max(Rotation.from_matrix(
        b @ a @ b.transpose(0, 2, 1) @ a.transpose(0, 2, 1)).magnitude()))
        for a, b in zip(increments, increments[1:]))
    return rotation, commutator


@pytest.mark.parametrize("triangles", (False, True), ids=("selected-q4", "all-s3"))
def test_loaded_noncommuting_point_chart_reset_restore_continue(
        triangles, record_property, request, tmp_path):
    definition = _fixture(triangles)
    control = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(control.close)
    history = []
    for factor in PREFIX_FACTORS:
        result = control.solve(_actions(), (factor,), max_iterations=20, max_backtracks=12)
        assert result.accepted_factors == PREFIX_FACTORS[:len(history) + 1]
        history.append(_record_snapshot(control, tmp_path, f"prefix-{len(history) + 1}"))
    prefix = control.checkpoint()
    rebased = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(rebased.close)
    rebased.restore(prefix)
    before = _record_snapshot(rebased, tmp_path, "prefix-restored")
    _assert_same_physical(history[-1], before)
    driver = rebased._finite_driver
    predecessor, origins = driver._state, (driver._shell_origins, driver._beam_origins)
    result = rebased.rebase_rotations(rebased.recover().generation)
    assert result.generation == control.recover().generation + 1
    assert driver._shell_origins is origins[0] and driver._beam_origins is origins[1]
    for name in ("frames", "cells", "resultants", "multipliers"):
        assert getattr(driver._state, name) is getattr(predecessor, name)
    np.testing.assert_array_equal(result.shell_chart_coordinates.reshape(-1, 6)[:, 3:], 0.)
    np.testing.assert_allclose(result.shell_rotation_anchors, before["shell_frames"],
                               rtol=0., atol=1e-12)
    after_reset = _record_snapshot(rebased, tmp_path, "prefix-reset")
    _assert_same_physical(before, after_reset)
    anchored = rebased.checkpoint()
    assert anchored.startswith(CHART_MAGIC)
    resumed = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(resumed.close)
    resumed.restore(anchored)
    after_restore = _record_snapshot(resumed, tmp_path, "prefix-reset-restored")
    _assert_same_physical(before, after_restore)
    assert control.checkpoint() == prefix
    successors = {}
    branch_histories = {}
    for branch, owner in (("control", control), ("reset", rebased), ("restored", resumed)):
        accepted = []

        def observe(result):
            assert result.generation == owner.recover().generation
            accepted.append(_record_snapshot(
                owner, tmp_path, f"accepted-{branch}-{len(accepted) + 1}"))

        result = owner.solve(
            _actions(), (FINAL_FACTOR,), max_iterations=20, max_backtracks=12,
            increment_control=stiffened_shell.AdaptiveIncrementControl(),
            on_accepted=observe)
        assert result.status == "completed"
        assert result.accepted_factors == PREFIX_FACTORS + tuple(
            item["accepted_factors"][-1] for item in accepted)
        assert result.accepted_factors[-1] == FINAL_FACTOR
        branch_histories[branch] = history + accepted
        successors[branch] = _record_snapshot(owner, tmp_path, f"final-{branch}")
    coverage = {}
    for branch, last in successors.items():
        for domain in ("shell_frames", "beam_frames"):
            rotation, commutator = _coverage(branch_histories[branch], domain)
            record_property(f"{branch}_{domain}_rotation_rad", rotation)
            record_property(f"{branch}_{domain}_commutator_rad", commutator)
            coverage[f"{branch}_{domain}"] = (rotation, commutator)
    (tmp_path / "coverage.json").write_text(
        json.dumps(coverage, sort_keys=True, allow_nan=False), encoding="utf-8")
    final = successors["control"]
    _assert_same_physical(final, successors["reset"], new_increment=True)
    _assert_same_physical(final, successors["restored"], new_increment=True)
    for branch in ("reset", "restored"):
        assert len(branch_histories[branch]) == len(branch_histories["control"])
        for expected, actual in zip(branch_histories["control"][len(history):],
                                    branch_histories[branch][len(history):]):
            _assert_same_physical(expected, actual, new_increment=True)
    for branch_domain, (rotation, commutator) in coverage.items():
        assert rotation >= ROTATION_FLOOR, (branch_domain, rotation)
        assert commutator >= COMMUTATOR_FLOOR, (branch_domain, commutator)
