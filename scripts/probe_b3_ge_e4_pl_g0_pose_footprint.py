"""G0 F04: bounded four-node distributing pose and moment work candidate."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, matmul, matvec, so3_exp, so3_log, transpose,
)
from anyfesolver._ge_beam3_variational_shell import rotation_jets


REF = np.array([[-.5, -.5, 0.], [.5, -.5, 0.], [.5, .5, 0.], [-.5, .5, 0.]])
BEAM_REF = np.array([0., 0., .2])
SIZE = 36  # shell: 4x6, beam: 6, multipliers: 6


def trial() -> np.ndarray:
    q = np.zeros(SIZE)
    q[:24].reshape(4, 6)[:, :3] = .01 * np.sin(np.arange(12).reshape(4, 3) + .3)
    q[24:27] = (.02, -.01, .03)
    q[27:30] = (.04, -.03, .05)
    q[30:] = (.6, -.3, .5, -.2, .7, -.4)
    return q


def independent(q: np.ndarray, *, motion: tuple[np.ndarray, np.ndarray] | None = None
                ) -> tuple[float, np.ndarray]:
    shell = REF + q[:24].reshape(4, 6)[:, :3]
    centroid = shell.mean(axis=0)
    fit = Rotation.align_vectors(shell - centroid, REF)[0].as_matrix()
    beam = BEAM_REF + q[24:27]
    beam_frame = Rotation.from_rotvec(q[27:30]).as_matrix()
    if motion is not None:
        rotation, translation = motion
        shell = shell @ rotation.T + translation
        centroid = shell.mean(axis=0)
        fit = rotation @ fit
        beam = rotation @ beam + translation
        beam_frame = rotation @ beam_frame
    gap = np.r_[fit.T @ (beam - centroid) - BEAM_REF,
                Rotation.from_matrix(fit.T @ beam_frame).as_rotvec()]
    return float(q[30:] @ gap), gap


def _embedded(jet: Jet2) -> Jet2:
    gradient = np.zeros(SIZE)
    hessian = np.zeros((SIZE, SIZE))
    gradient[:24] = jet.gradient
    hessian[:24, :24] = jet.hessian
    return Jet2(jet.value, gradient, hessian)


def evaluate(q: np.ndarray) -> tuple[Jet2, np.ndarray]:
    shell = REF + q[:24].reshape(4, 6)[:, :3]
    fitted = [[_embedded(jet) for jet in row] for row in rotation_jets(REF, shell)]
    variables = [Jet2.variable(float(value), i, SIZE) for i, value in enumerate(q)]
    centroid = [sum((REF[i, axis] + variables[6 * i + axis] for i in range(4)),
                    Jet2.constant(0., SIZE)) / 4. for axis in range(3)]
    beam = [BEAM_REF[axis] + variables[24 + axis] for axis in range(3)]
    frame = so3_exp(variables[27:30])
    translation = [v - BEAM_REF[axis] for axis, v in enumerate(
        matvec(transpose(fitted), [beam[i] - centroid[i] for i in range(3)]))]
    relative = so3_log(matmul(transpose(fitted), frame))
    constraints = translation + relative
    work = sum((variables[30 + i] * value for i, value in enumerate(constraints)),
               Jet2.constant(0., SIZE))
    return work, np.array([value.value for value in constraints])


def main() -> int:
    reference = np.zeros(SIZE)
    ref_work, ref_gap = evaluate(reference)
    reference[35] = 1.
    moment_work, _ = evaluate(reference)
    shell_force = moment_work.gradient[:24].reshape(4, 6)[:, :3]
    force_resultant = float(np.linalg.norm(shell_force.sum(axis=0)))
    shell_moment = float(np.cross(REF, shell_force).sum(axis=0)[2])
    beam_moment = float(moment_work.gradient[29])
    drill_work = float(np.linalg.norm(moment_work.gradient[:24].reshape(4, 6)[:, 3:]))

    q = trial()
    ad, constraints = evaluate(q)
    independent_value, independent_gap = independent(q)
    h = 2e-6
    numerical_gradient = np.array([(
        independent(q + h * np.eye(SIZE)[i])[0]
        - independent(q - h * np.eye(SIZE)[i])[0]) / (2 * h)
        for i in range(SIZE)])
    gradient_error = float(np.linalg.norm(ad.gradient - numerical_gradient) /
                           max(1., np.linalg.norm(numerical_gradient)))
    direction = np.sin(np.arange(SIZE) + .3)
    direction /= np.linalg.norm(direction)
    expected_second = float(direction @ ad.hessian @ direction)
    second_errors = []
    for step in (2e-4, 1e-4):
        plus = independent(q + step * direction)[0]
        minus = independent(q - step * direction)[0]
        observed = (plus - 2 * independent_value + minus) / step**2
        second_errors.append(abs(observed - expected_second) / max(1., abs(expected_second)))
    skew = float(np.linalg.norm(ad.hessian - ad.hessian.T) /
                 max(1., np.linalg.norm(ad.hessian)))
    common = (Rotation.from_rotvec([.8, -.4, .6]).as_matrix(), np.array([2., -3., 1.]))
    common_error = float(np.linalg.norm(independent(q, motion=common)[1] - independent_gap))
    value_error = abs(ad.value - independent_value)
    gap_error = float(np.linalg.norm(constraints - independent_gap))
    status = (abs(ref_work.value) <= 1e-12 and np.linalg.norm(ref_gap) <= 1e-12
              and force_resultant <= 1e-10 and abs(shell_moment + 1.) <= 1e-10
              and abs(beam_moment - 1.) <= 1e-10 and drill_work == 0.
              and value_error <= 1e-10 and gap_error <= 1e-10
              and gradient_error <= 1e-7 and max(second_errors) <= 1e-5
              and skew <= 1e-10 and common_error <= 1e-10)
    print(json.dumps({"case": "G0_F04_DISTRIBUTING_POSE_V1",
                      "status": "diagnostic_pass" if status else "failed",
                      "reference_gap": float(np.linalg.norm(ref_gap)),
                      "shell_force_resultant": force_resultant,
                      "shell_moment_z": shell_moment, "beam_moment_z": beam_moment,
                      "shell_rotation_gradient": drill_work,
                      "value_error": value_error, "gap_error": gap_error,
                      "gradient_error": gradient_error, "second_errors": second_errors,
                      "hessian_skew": skew, "common_motion_gap_error": common_error},
                     sort_keys=True))
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
