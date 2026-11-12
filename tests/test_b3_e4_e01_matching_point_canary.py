"""Check the external point equations against the native reference pose law."""

import numpy as np
import pytest

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from scripts.b3_e4_e01_matching_point_canary import (
    SHELL, STATIONS, _beam_nodes, bond_rows, validate_work,
)


@pytest.mark.parametrize("sign", (-1, 1))
def test_matching_point_deck_rows_are_native_reference_jacobian(sign):
    beam_nodes = _beam_nodes(sign)
    rows = bond_rows(sign)
    for i, (shell, beam) in enumerate(STATIONS):
        positions = np.array((SHELL[shell], beam_nodes[beam]))
        frames = np.tile(np.eye(3), (2, 1, 1))
        evaluated = RigidPoseJoint(positions, frames).evaluate(
            positions, frames, np.zeros(6))
        expected = np.zeros((6, 12))
        for local_row, terms in enumerate(rows[6*i:6*i+6]):
            for node, dof, coefficient in terms:
                slot = (0 if node == shell else 6)+dof-1
                expected[local_row, slot] += coefficient
        np.testing.assert_allclose(evaluated.constraints, 0., atol=1e-15)
        np.testing.assert_allclose(
            evaluated.constraint_jacobian, expected, rtol=0., atol=1e-13)
    work = validate_work(sign)
    assert work["rows"] == 18
    assert work["maximum_connection_wrench_imbalance"] < 1e-13
