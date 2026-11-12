"""G0 F04 rigid material footprint work and bounded support degeneracy."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_mixed_ad import Jet2, matvec, so3_exp, transpose


REF = np.array([[-.5, -.5, 0.], [.5, -.5, 0.], [.5, .5, 0.], [-.5, .5, 0.]])
BEAM_REF = np.array([0., 0., .2])
SIZE = 42


def multipliers() -> np.ndarray:
    inertia = np.sum(REF[:, :2]**2)
    return -np.cross(np.array([0., 0., 1.]), REF).ravel() / inertia


def trial() -> np.ndarray:
    q = np.zeros(SIZE)
    q[:24].reshape(4, 6)[:, :3] = .01 * np.sin(np.arange(12).reshape(4, 3) + .3)
    q[24:27] = (.02, -.01, .03)
    q[27:30] = (.04, -.03, .05)
    q[30:] = multipliers()
    return q


def independent(q: np.ndarray, *, motion: tuple[np.ndarray, np.ndarray] | None = None
                ) -> tuple[float, np.ndarray]:
    shell = REF + q[:24].reshape(4, 6)[:, :3]
    beam = BEAM_REF + q[24:27]
    frame = Rotation.from_rotvec(q[27:30]).as_matrix()
    if motion is not None:
        rotation, translation = motion
        shell = shell @ rotation.T + translation
        beam = rotation @ beam + translation
        frame = rotation @ frame
    gaps = (shell - beam) @ frame - (REF - BEAM_REF)
    return float(q[30:] @ gaps.ravel()), gaps.ravel()


def evaluate(q: np.ndarray) -> tuple[Jet2, np.ndarray]:
    jets = [Jet2.variable(float(value), i, SIZE) for i, value in enumerate(q)]
    frame = transpose(so3_exp(jets[27:30]))
    beam = [BEAM_REF[axis] + jets[24 + axis] for axis in range(3)]
    constraints = []
    for node in range(4):
        shell = [REF[node, axis] + jets[6 * node + axis] for axis in range(3)]
        distance = matvec(frame, [shell[axis] - beam[axis] for axis in range(3)])
        constraints.extend(distance[axis] - (REF[node, axis] - BEAM_REF[axis])
                           for axis in range(3))
    work = sum((jets[30 + i] * c for i, c in enumerate(constraints)),
               Jet2.constant(0., SIZE))
    return work, np.array([c.value for c in constraints])


def support_rank(points: np.ndarray) -> int:
    rows = []
    for x, y in points[:, :2]:
        rows.extend(([0., 0., -y], [0., 0., x], [y, -x, 0.]))
    # Equivalent to -skew(x,y,0) up to a fixed row sign/permutation;
    # the bounded local rank is the only quantity used here.
    return int(np.linalg.matrix_rank(np.asarray(rows)))


def main() -> int:
    reference = np.zeros(SIZE)
    reference[30:] = multipliers()
    ref_work, ref_gap = evaluate(reference)
    force = ref_work.gradient[:24].reshape(4, 6)[:, :3]
    force_resultant = float(np.linalg.norm(force.sum(axis=0)))
    shell_moment = float(np.cross(REF, force).sum(axis=0)[2])
    beam_moment = float(ref_work.gradient[29])
    shell_rotation_gradient = float(np.linalg.norm(ref_work.gradient[:24].reshape(4, 6)[:, 3:]))
    q = trial()
    ad, gap = evaluate(q)
    value, expected_gap = independent(q)
    h = 2e-6
    directions = np.eye(SIZE)
    gradient = np.array([(independent(q + h * v)[0] - independent(q - h * v)[0]) / (2*h)
                         for v in directions])
    first_error = float(np.linalg.norm(ad.gradient - gradient) / max(1., np.linalg.norm(gradient)))
    direction = np.sin(np.arange(SIZE) + .3)
    direction /= np.linalg.norm(direction)
    expected_second = float(direction @ ad.hessian @ direction)
    second_errors = []
    for step in (2e-4, 1e-4):
        plus, minus = independent(q + step * direction)[0], independent(q - step * direction)[0]
        second_errors.append(abs((plus - 2*value + minus) / step**2 - expected_second)
                             / max(1., abs(expected_second)))
    skew = float(np.linalg.norm(ad.hessian - ad.hessian.T) / max(1., np.linalg.norm(ad.hessian)))
    motion = (Rotation.from_rotvec([.8, -.4, .6]).as_matrix(), np.array([2., -3., 1.]))
    objective_error = float(np.linalg.norm(independent(q, motion=motion)[1] - expected_gap))
    full_rank = support_rank(REF)
    collinear_rank = support_rank(np.array([[-.5, 0., 0.], [0., 0., 0.], [.5, 0., 0.]]))
    passed = (abs(ref_work.value) <= 1e-12 and np.linalg.norm(ref_gap) <= 1e-12
              and force_resultant <= 1e-10 and abs(shell_moment + 1.) <= 1e-10
              and abs(beam_moment - 1.) <= 1e-10 and shell_rotation_gradient == 0.
              and abs(ad.value - value) <= 1e-10 and np.linalg.norm(gap - expected_gap) <= 1e-10
              and first_error <= 1e-7 and max(second_errors) <= 1e-5
              and skew <= 1e-10 and objective_error <= 1e-10
              and full_rank == 3 and collinear_rank == 2)
    print(json.dumps({"case": "G0_F04_RIGID_FOOTPRINT_V1", "status": "diagnostic_pass" if passed else "failed",
                      "reference_gap": float(np.linalg.norm(ref_gap)), "force_resultant": force_resultant,
                      "shell_moment_z": shell_moment, "beam_moment_z": beam_moment,
                      "shell_rotation_gradient": shell_rotation_gradient,
                      "first_error": first_error, "second_errors": second_errors,
                      "skew": skew, "objective_error": objective_error,
                      "full_rank": full_rank, "collinear_rank": collinear_rank}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
