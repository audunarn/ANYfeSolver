"""Bounded yielded S3-V2D algorithmic pullback plus eccentric pose-joint check.

Both trial actions start from one accepted plastic predecessor. The existing
elastic variational adapter guard is untouched. This is not G0B acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_variational_shell import deformation
from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


REFERENCE = np.array(((0., 0., 0.), (1.2, .05, 0.), (.2, .95, 0.)))
BEAM_REFERENCE = REFERENCE[1] + np.array((.18, -.11, .07))
PORT = np.r_[np.arange(6, 12), np.arange(18, 30)]
STEPS = (2e-6, 2e-7, 2e-8)
TOLERANCE = 1e-5


def _skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _left_jacobian(theta):
    a = np.linalg.norm(theta)
    s = _skew(theta)
    if a < 1e-4:
        a2 = a*a
        first = .5-a2/24+a2*a2/720
        second = 1/6-a2/120+a2*a2/5040
    else:
        first = (1-np.cos(a))/(a*a)
        second = (a-np.sin(a))/(a*a*a)
    return np.eye(3)+first*s+second*s@s


def _independent_joint(q, joint):
    """Independent SciPy pose and closed first variation in additive charts."""
    xs = REFERENCE[1]+q[6:9]
    xb = BEAM_REFERENCE+q[18:21]
    rs = Rotation.from_rotvec(q[9:12]).as_matrix()
    rb = Rotation.from_rotvec(q[21:24]).as_matrix()
    d = xb-xs
    a = joint.reference_frames[0].T@joint.reference_frames[1]
    e = joint.reference_frames[0].T@(BEAM_REFERENCE-REFERENCE[1])
    phi = Rotation.from_matrix(a.T@rs.T@rb).as_rotvec()
    angle = np.linalg.norm(phi)
    if angle < 1e-4:
        factor = 1/12+angle**2/720+angle**4/30240
    else:
        factor = 1/angle**2-(1+np.cos(angle))/(2*angle*np.sin(angle))
    p = _skew(phi)
    inverse = np.eye(3)-.5*p+factor*p@p
    b = inverse@a.T@rs.T
    jac = np.zeros((6, 12))
    jac[:3, :3] = -rs.T
    jac[:3, 3:6] = rs.T@_skew(d)
    jac[:3, 6:9] = rs.T
    jac[3:, 3:6] = -b
    jac[3:, 9:12] = b
    chart = np.eye(18)
    chart[3:6, 3:6] = _left_jacobian(q[9:12])
    chart[9:12, 9:12] = _left_jacobian(q[21:24])
    return chart.T@np.r_[jac.T@q[24:30], rs.T@d-e, phi]


def _fixture():
    model = FEModel("g0-f05-s3-pose-pullback-dev-v1")
    curve = DNVC208MaterialCurve(
        sigma_prop=80e6, sigma_yield=85e6, sigma_yield_2=100e6,
        eps_p_y1=.002, eps_p_y2=.02, K=280e6, n=.18)
    material = model.add_material("steel", 210e9, .3, density=7850.,
                                  hardening_curve=curve)
    for node_id, point in enumerate(REFERENCE, start=1):
        model.add_node(node_id, *point)
    element = create_shell_element(
        1, [1, 2, 3], "steel", formulation="e4-pl-s3-v2d",
        thickness=.018, reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    accepted = np.zeros(18)
    accepted[6], accepted[12] = .008, -.002
    _, _, predecessor = element.compute_nonlinear_response(
        model.mesh, material, accepted, virgin, 5, True)
    if float(np.max(predecessor["alpha"])) <= 0:
        raise ArithmeticError("yielded S3 accepted predecessor required")
    q = np.zeros(30)
    q[:18] = accepted
    q[6] += .0002
    q[12] -= .00005
    q[:18].reshape(3, 6)[:, 3:] = np.array((
        (.003, -.002, .001), (-.001, .002, -.003), (.001, .001, .002)))
    q[18:24] = (.00003, -.00007, .00002, .015, -.011, .008)
    q[24:30] = (2e6, -1e6, .5e6, 2e4, -3e4, 1e4)
    frames = np.array((np.eye(3), np.eye(3)))
    joint = RigidPoseJoint(np.array((REFERENCE[1], BEAM_REFERENCE)), frames)
    return model, material, element, predecessor, q, joint, frames


def _material(q, model, material, element, predecessor, tangent):
    local, differential, second = deformation(REFERENCE, q[:18])
    force, k, candidate = element.compute_nonlinear_response(
        model.mesh, material, local, predecessor, 5, tangent)
    force = np.asarray(force)
    result = np.zeros(30)
    result[:18] = differential.T@force
    matrix = None
    if tangent:
        matrix = np.zeros((30, 30))
        matrix[:18, :18] = differential.T@np.asarray(k)@differential + \
            np.einsum("i,ijk->jk", force, second)
    if not np.isfinite(result).all() or (matrix is not None and not np.isfinite(matrix).all()):
        raise ArithmeticError("finite S3 material pullback required")
    return result, matrix, candidate


def _joint(q, joint, frames):
    positions = np.array((REFERENCE[1]+q[6:9], BEAM_REFERENCE+q[18:21]))
    coordinates = np.array((q[9:12], q[21:24]))
    response = joint.evaluate_ports(
        positions, frames, q[24:30], rotation_coordinates=coordinates,
        charted_ports=(True, True))
    force = np.zeros(30)
    force[PORT] = response.residual
    tangent = np.zeros((30, 30))
    tangent[np.ix_(PORT, PORT)] = response.tangent
    return force, tangent


def _relative(a, b):
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))


def evaluate() -> dict:
    model, material, element, predecessor, q, joint, frames = _fixture()
    before = canonical_json_bytes(predecessor)
    parent_alpha = np.asarray(predecessor["alpha"])
    mf, mk, candidate = _material(q, model, material, element, predecessor, True)
    jf, jk = _joint(q, joint, frames)
    independent = _independent_joint(q, joint)
    first_error = _relative(jf[PORT], independent)
    mask = np.asarray(candidate["alpha"]) > parent_alpha
    if not np.any(mask):
        raise ArithmeticError("active S3 trial branch required")
    directions = []
    for phase, scale in ((.3, 1.), (.7, .73)):
        v = np.zeros(30)
        v[:24] = np.sin(np.arange(24)*scale+phase)
        v[:24] /= np.linalg.norm(v[:24])
        v[24:] = 1e5*np.cos(np.arange(6)*.53+phase)
        expected_material, expected_joint = mk@v, jk@v
        measurements = []
        for h in STEPS:
            samples, stable = [], []
            for factor in (-1., -.5, .5, 1.):
                trial = q+factor*h*v
                force, _, state = _material(
                    trial, model, material, element, predecessor, False)
                pose = np.zeros(30)
                pose[PORT] = _independent_joint(trial, joint)
                samples.append((force, pose))
                stable.append(np.array_equal(
                    np.asarray(state["alpha"]) > parent_alpha, mask))
            measured_material = (samples[3][0]-samples[0][0])/(2*h)
            measured_joint = (samples[3][1]-samples[0][1])/(2*h)
            measurements.append({
                "material_error": _relative(expected_material, measured_material),
                "joint_error": _relative(expected_joint, measured_joint),
                "combined_error": _relative(expected_material+expected_joint,
                                            measured_material+measured_joint),
                "active_mask_stable": all(stable),
            })
        plateau = any(all(measurements[i][key] <= TOLERANCE and
                          measurements[i+1][key] <= TOLERANCE
                          for key in ("material_error", "joint_error", "combined_error"))
                      and measurements[i]["active_mask_stable"] and
                      measurements[i+1]["active_mask_stable"]
                      for i in range(len(STEPS)-1))
        directions.append({"phase": phase, "measurements": measurements,
                           "adjacent_plateau": plateau})
    immutable = canonical_json_bytes(predecessor) == before
    passed = first_error <= 1e-10 and immutable and all(
        d["adjacent_plateau"] for d in directions)
    return {
        "classification": "F05_S3_POSE_PULLBACK_DEV_NOT_ACCEPTANCE",
        "status": "DEV_F05_S3_PASS" if passed else "DEV_F05_S3_FAIL",
        "first_variation_relative_error": first_error,
        "accepted_alpha_max": float(parent_alpha.max()),
        "trial_alpha_max": float(np.asarray(candidate["alpha"]).max()),
        "active_stations": int(mask.sum()),
        "predecessor_unchanged": immutable,
        "joint_cross_block_norm": float(np.linalg.norm(
            jk[np.ix_(PORT[:6], PORT[6:])])),
        "material_force_norm": float(np.linalg.norm(mf)),
        "joint_force_norm": float(np.linalg.norm(jf)),
        "steps": STEPS,
        "tolerance": TOLERANCE,
        "directions": directions,
        "limits": ["one yielded S3 triangle and one eccentric nodal pose joint",
                   "additive chart direction and same accepted material predecessor only",
                   "not a mixed owner, generalized shell material, restart or F05/G0B acceptance"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "first_error": result["first_variation_relative_error"],
                      "active_stations": result["active_stations"],
                      "maximum_direction_error": max(
                          max(m[k] for k in ("material_error", "joint_error", "combined_error"))
                          for d in result["directions"] for m in d["measurements"])}))
    if result["status"] != "DEV_F05_S3_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
