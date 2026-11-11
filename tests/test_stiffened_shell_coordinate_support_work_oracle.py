"""Independent spatial-variation check of original-Log support reactions.

The oracle differentiates the declared constraint through physical matrices,
without using the implementation's Exp differential or coordinate pullback.
It verifies a local law, not the complete F06 material/history programme.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._stiffened_shell_coordinate_support import reactions


@pytest.mark.parametrize("anchor_vector", [(0., 0., 0.), (.2, -.1, .05)])
def test_physical_reaction_is_constraint_normal_and_does_no_admissible_work(anchor_vector):
    psi = np.array((0., .7, .35))
    physical = Rotation.from_rotvec(psi).as_matrix()
    anchor = Rotation.from_rotvec(anchor_vector).as_matrix()
    q = np.zeros(6)
    q[3:] = Rotation.from_matrix(physical @ anchor.T).as_rotvec()
    multiplier = 350.  # N m; constraint psi_x is dimensionless.
    covector = np.zeros(6)
    covector[3] = multiplier
    moment = reactions(q, anchor[None, :, :], 6, [3], covector, 6)[3:]

    h = 1e-5
    normal = np.empty(3)
    for axis in range(3):
        perturbation = np.eye(3)[axis] * h
        plus = Rotation.from_matrix(
            Rotation.from_rotvec(perturbation).as_matrix() @ physical).as_rotvec()[0]
        minus = Rotation.from_matrix(
            Rotation.from_rotvec(-perturbation).as_matrix() @ physical).as_rotvec()[0]
        normal[axis] = (plus - minus) / (2 * h)
    assert np.linalg.norm(moment - multiplier * normal) / np.linalg.norm(moment) <= 1e-8
    # A partial coordinate restraint can exert physical moments on every axis.
    assert np.linalg.norm(moment[1:]) / np.linalg.norm(moment) > .1

    for axis in (1, 2):
        variation = np.eye(3)[axis] * h
        plus = Rotation.from_rotvec(psi + variation).as_matrix()
        minus = Rotation.from_rotvec(psi - variation).as_matrix()
        spatial_velocity = Rotation.from_matrix(plus @ minus.T).as_rotvec() / (2 * h)
        normalized_work = abs(moment @ spatial_velocity) / (
            np.linalg.norm(moment) * np.linalg.norm(spatial_velocity))
        assert normalized_work <= 1e-8
