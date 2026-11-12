"""Focused parity and admission checks for physical footprint compilation."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import (
    B2PhysicalRigidFootprint, _node, compile_physical_footprints,
)


def _dense_hessians(n, e, arm):
    """Reference expansion of the original all-pairs five-row Hessian."""
    h = np.zeros((5, 12, 12))
    eye = np.eye(12)
    for a in range(12):
        for b in range(12):
            sa = eye[3:6, a]
            sb = eye[3:6, b]
            ba = eye[9:, a]
            bb = eye[9:, b]
            da = eye[:3, a] - eye[6:9, a]
            db = eye[:3, b] - eye[6:9, b]
            h[:3, a, b] = -np.cross(ba, db) - np.cross(bb, da) + .5 * (
                np.cross(ba, np.cross(bb, arm)) + np.cross(bb, np.cross(ba, arm)))
            h[3:, a, b] = e.T @ (.5 * (
                np.cross(ba, np.cross(bb, n)) + np.cross(bb, np.cross(ba, n))
                + np.cross(sa, np.cross(sb, n)) + np.cross(sb, np.cross(sa, n)))
                - np.cross(ba, np.cross(sb, n)) - np.cross(bb, np.cross(sa, n)))
    return h


@pytest.mark.parametrize("angle,arm", [
    (0., (0., 0., 0.)),
    (.37, (.25, -1.3, 2.1)),
    (-.71, (1e6, -3e-4, -.8)),
])
def test_structural_hessian_matches_dense_law_exactly(angle, arm):
    frame = Rotation.from_rotvec((angle, -.2, .31)).as_matrix()
    n = frame[:, 2]
    e = frame[:, :2]
    item = SimpleNamespace(shell_node=1, beam_node=10,
                           shell_slots=tuple(range(6)), beam_slots=tuple(range(6, 12)),
                           shell_position=tuple(arm), beam_position=(0., 0., 0.))
    group = SimpleNamespace(footprint_id="pad", reference_director=n, tangent_basis=e)
    made = _node(group, item, 7)
    np.testing.assert_array_equal(made.constraint_hessians, _dense_hessians(n, e, np.asarray(arm)))
    assert made.row_start == 7
    assert made.constraint_hessians.flags.writeable is False


def _case(extra_shell=None, moved=False):
    positions = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (1., 1., 0.), 4: (0., 1., 0.)}
    if moved:
        positions[3] = (1., 1., .01)
    group = B2PhysicalRigidFootprint("pad", (1, 2, 3, 4), 10, (0., 0., 1.),
                                     ((1., 0.), (0., 1.), (0., 0.)), ((1, 2, 3, 4),))
    shells = [SimpleNamespace(shell_id="patch", node_ids=(1, 2, 3, 4),
                              reference_normal=(0., 0., 1.))]
    if extra_shell is not None:
        shells.append(extra_shell)
    args = ((group,), positions, tuple(shells), {10: (.2, .4, .3)},
            {node: i for i, node in enumerate(positions)}, {10: 0}, ())
    return args


def test_indexed_incidence_keeps_row_order_and_detaches_coordinates():
    extra = SimpleNamespace(shell_id="adjacent", node_ids=(1, 5, 6, 7),
                            reference_normal=(0., 0., 1.))
    args = _case(extra)
    _, nodes = compile_physical_footprints(*args)
    assert tuple(node.shell_node for node in nodes) == (1, 2, 3, 4)
    assert tuple(node.row_start for node in nodes) == (0, 5, 10, 15)
    before = nodes[2].constraint.copy(), nodes[2].constraint_hessians.copy()
    args[1][3] = (999., 999., 999.)
    np.testing.assert_array_equal(nodes[2].constraint, before[0])
    np.testing.assert_array_equal(nodes[2].constraint_hessians, before[1])


@pytest.mark.parametrize("extra,moved,message", [
    (SimpleNamespace(shell_id="seam", node_ids=(1, 5, 6, 7),
                     reference_normal=(0., 0., -1.)), False, "ambiguous director seam"),
    (SimpleNamespace(shell_id="pentagon", node_ids=(1, 5, 6, 7, 8),
                     reference_normal=(0., 0., 1.)), False,
     "physical footprint requires native three/four-node shells"),
    (None, True, "coplanar with owned director"),
])
def test_indexed_incidence_preserves_refusals(extra, moved, message):
    with pytest.raises(ValueError, match=message):
        compile_physical_footprints(*_case(extra, moved))

def test_indexed_incidence_admits_native_three_node_shell():
    # Mixed S3/Q4 admission: a native three-node shell with the owned
    # director is admitted and leaves the compiled group unchanged.
    extra = SimpleNamespace(shell_id="triangle", node_ids=(1, 5, 6),
                            reference_normal=(0., 0., 1.))
    _, nodes = compile_physical_footprints(*_case(extra))
    assert tuple(node.shell_node for node in nodes) == (1, 2, 3, 4)
    assert tuple(node.row_start for node in nodes) == (0, 5, 10, 15)
