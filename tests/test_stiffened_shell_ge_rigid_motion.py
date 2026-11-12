"""V03 global finite rigid-motion control for the experimental GE owner."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anymaterial import DNVC208MaterialCurve

from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, _State, _position_pair,
)

from test_stiffened_shell_ge_experimental import _panel


@pytest.mark.parametrize("mixed", (False, True))
@pytest.mark.parametrize("shell_j2", (False, True))
def test_global_common_rigid_motion_annihilates_internal_and_point_work(mixed, shell_j2):
    definition = _panel()
    if mixed:
        definition = replace(definition, shell=replace(
            definition.shell, q4_elements=((1, 2, 5, 4),),
            s3_elements=((2, 3, 6), (2, 6, 5))))
    if shell_j2:
        curve = DNVC208MaterialCurve(80e6, 85e6, 100e6, .002, .02, 280e6, .18)
        definition = replace(definition, shell=replace(definition.shell, j2_curve=curve))
    owner = ExperimentalGEAnalysis(definition)
    driver = FinitePointDriver(owner)
    vector = np.array((.13, -.21, .17))
    rotation = Rotation.from_rotvec(vector).as_matrix()
    shift = np.array((.07, -.05, .09))

    q = driver._state.q.copy()
    for node, reference in enumerate(owner._shell_xyz):
        q[6*node:6*node+3] = rotation@reference+shift-reference
        q[6*node+3:6*node+6] = vector
    for node, identity in enumerate(owner._beam_ids):
        reference = owner._beam_nodes[identity][0]
        slot = owner._shell_count+6*node
        q[slot:slot+3] = rotation@reference+shift-reference
    frames = np.array([rotation@owner._beam_nodes[node][1]
                       for node in owner._beam_ids])
    cells = rotation@driver._state.cells
    state = _State(q, frames, cells, driver._state.resultants.copy(),
                   driver._state.multipliers.copy())
    residual, _, _, _, shell_candidates, _ = driver._assemble(
        state, 1., driver._shell_origins, driver._beam_origins,
        pattern=driver._load(()))
    physical = owner._K.shape[0]
    row_scale = np.asarray(np.abs(owner._K)@owner._coordinate_scale).ravel()
    scaled = np.abs(residual[:physical])/row_scale
    assert np.max(scaled) <= 1e-9, {
        "shell": float(np.max(scaled[:owner._shell_count])),
        "beam": float(np.max(scaled[owner._shell_count:])),
    }
    assert np.max(np.abs(residual[physical:])) <= 1e-9
    if shell_j2:
        assert all(not np.any(candidate["alpha"]) for candidate in shell_candidates)


def test_finite_point_rows_remain_spatial_at_nonzero_shell_chart():
    owner = ExperimentalGEAnalysis(_panel())
    driver = FinitePointDriver(owner)
    slots = owner._joint_slots[0]
    q = driver._state.q.copy()
    q[slots[3:6]] = (.22, -.13, .19)
    zero = _State(q, driver._state.frames, driver._state.cells,
                  driver._state.resultants, driver._state.multipliers)
    multipliers = np.zeros_like(zero.multipliers)
    multipliers[:6] = (2., -3., 4., 5., -6., 7.)
    loaded = replace(zero, multipliers=multipliers)
    arguments = (0., driver._shell_origins, driver._beam_origins)
    pattern = driver._load(())
    base_residual, base_tangent, *_ = driver._assemble(
        zero, *arguments, pattern=pattern)
    actual_residual, actual_tangent, *_ = driver._assemble(
        loaded, *arguments, pattern=pattern)

    shell_node = slots[0]//6
    beam_node = (slots[6]-owner._shell_count)//6
    shell_position, shell_low = _position_pair(
        owner._shell_xyz[shell_node], q[slots[:3]])
    beam_position, beam_low = _position_pair(
        driver._reference_positions[beam_node],
        q[slots[6:9]])
    trial = owner._connections[0].evaluate_ports(
        np.array((shell_position, beam_position)),
        np.array((np.eye(3), loaded.frames[beam_node])),
        multipliers[:6],
        rotation_coordinates=np.array((q[slots[3:6]], np.zeros(3))),
        charted_ports=(True, False),
        position_low=np.array((shell_low, beam_low)))
    virgin = owner._connections[0].evaluate_ports(
        np.array((shell_position, beam_position)),
        np.array((np.eye(3), zero.frames[beam_node])),
        zero.multipliers[:6],
        rotation_coordinates=np.array((q[slots[3:6]], np.zeros(3))),
        charted_ports=(True, False),
        position_low=np.array((shell_low, beam_low)))
    count = owner._K.shape[0]
    extended = np.r_[slots, np.arange(count, count+6)]
    # Isolate O(1) joint terms by subtracting two assembled shell states;
    # the absolute floor covers cancellation of much larger shell entries.
    np.testing.assert_allclose(
        (actual_residual-base_residual)[extended],
        trial.pose.residual-virgin.pose.residual, rtol=1e-12, atol=3e-8)
    np.testing.assert_allclose(
        (actual_tangent-base_tangent).toarray()[np.ix_(extended, extended)],
        (trial.pose.spatial_jacobian-virgin.pose.spatial_jacobian)
        @trial.coordinate_map, rtol=1e-11, atol=1e-7)
    assert np.linalg.norm(trial.residual-trial.pose.residual) > 1e-3
