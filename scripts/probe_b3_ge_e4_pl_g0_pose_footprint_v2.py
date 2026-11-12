"""Independent-review correction: refit after common motion and check wrenches."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_pose_joint import _exp_terms

import probe_b3_ge_e4_pl_g0_pose_footprint as prior


def refitted_oracle(q: np.ndarray, rotation: np.ndarray, translation: np.ndarray
                    ) -> tuple[float, np.ndarray]:
    shell = (prior.REF + q[:24].reshape(4, 6)[:, :3]) @ rotation.T + translation
    centroid = shell.mean(axis=0)
    fit = Rotation.align_vectors(shell - centroid, prior.REF)[0].as_matrix()
    beam = rotation @ (prior.BEAM_REF + q[24:27]) + translation
    frame = rotation @ Rotation.from_rotvec(q[27:30]).as_matrix()
    gap = np.r_[fit.T @ (beam - centroid) - prior.BEAM_REF,
                Rotation.from_matrix(fit.T @ frame).as_rotvec()]
    return float(q[30:] @ gap), gap


def spatial_wrench(q: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shell = prior.REF + q[:24].reshape(4, 6)[:, :3]
    beam = prior.BEAM_REF + q[24:27]
    shell_forces = gradient[:24].reshape(4, 6)[:, :3]
    beam_force = gradient[24:27]
    _, left, _ = _exp_terms(q[27:30])
    beam_moment = np.linalg.solve(left.T, gradient[27:30])
    total_force = shell_forces.sum(axis=0) + beam_force
    total_moment = np.cross(shell, shell_forces).sum(axis=0) + np.cross(beam, beam_force) + beam_moment
    return total_force, total_moment, beam_moment


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1., np.linalg.norm(b)))


def main() -> int:
    q = prior.trial()
    w = Rotation.from_rotvec([.8, -.4, .6]).as_matrix()
    t = np.array([2., -3., 1.])
    q_moved = q.copy()
    shell = prior.REF + q[:24].reshape(4, 6)[:, :3]
    q_moved[:24].reshape(4, 6)[:, :3] = shell @ w.T + t - prior.REF
    beam = prior.BEAM_REF + q[24:27]
    q_moved[24:27] = w @ beam + t - prior.BEAM_REF
    q_moved[27:30] = Rotation.from_matrix(w @ Rotation.from_rotvec(q[27:30]).as_matrix()).as_rotvec()
    original, gap_original = prior.evaluate(q)
    moved, gap_moved = prior.evaluate(q_moved)
    oracle_value, oracle_gap = refitted_oracle(q, w, t)
    source_force, source_moment, source_beam_moment = spatial_wrench(q, original.gradient)
    moved_force, moved_moment, moved_beam_moment = spatial_wrench(q_moved, moved.gradient)
    f0 = original.gradient[:24].reshape(4, 6)[:, :3]
    f1 = moved.gradient[:24].reshape(4, 6)[:, :3]
    translation_error = max(relative(f1, f0 @ w.T),
                            relative(moved.gradient[24:27], w @ original.gradient[24:27]))
    moment_transport = relative(moved_beam_moment, w @ source_beam_moment)
    balance = max(float(np.linalg.norm(source_force)), float(np.linalg.norm(source_moment)),
                  float(np.linalg.norm(moved_force)), float(np.linalg.norm(moved_moment)))
    gap_error = max(relative(gap_moved, gap_original), relative(gap_moved, oracle_gap))
    potential_error = max(abs(moved.value - original.value), abs(moved.value - oracle_value))
    passed = (gap_error <= 1e-10 and potential_error <= 1e-10
              and translation_error <= 1e-10 and moment_transport <= 1e-10
              and balance <= 1e-10)
    print(json.dumps({"case": "G0_F04_DISTRIBUTING_POSE_REFIT_V2",
                      "status": "diagnostic_pass" if passed else "failed",
                      "gap_error": gap_error, "potential_error": potential_error,
                      "translation_force_transport": translation_error,
                      "beam_spatial_moment_transport": moment_transport,
                      "global_wrench_balance": balance}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
