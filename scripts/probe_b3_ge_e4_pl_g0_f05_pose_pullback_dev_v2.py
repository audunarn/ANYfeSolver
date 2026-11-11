"""Bounded yielded-Q4 plus exact pose-chart F05 development probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY/'src'))

import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

import anyfesolver
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_variational_shell import deformation
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


REFERENCE = np.array(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
BEAM_REFERENCE = REFERENCE[1]+np.array((.18, -.11, .07))
STEPS = (2e-6, 2e-7, 2e-8)
PORT = np.r_[np.arange(6, 12), np.arange(24, 36)]
TOLERANCE = 1e-5


def skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def left_jacobian(theta):
    """Independent Rodrigues differential, with its small-angle limit."""
    a = np.linalg.norm(theta)
    s = skew(theta)
    if a < 1e-4:
        a2 = a*a
        first = .5-a2/24+a2*a2/720
        second = 1/6-a2/120+a2*a2/5040
    else:
        first = (1-np.cos(a))/(a*a)
        second = (a-np.sin(a))/(a*a*a)
    return np.eye(3)+first*s+second*s@s


def independent_joint_residual(q, joint):
    """SciPy pose and closed spatial first variation; no owner pose jets."""
    xs = REFERENCE[1]+q[6:9]
    xb = BEAM_REFERENCE+q[24:27]
    rs = Rotation.from_rotvec(q[9:12]).as_matrix()
    rb = Rotation.from_rotvec(q[27:30]).as_matrix()
    d = xb-xs
    e = joint.reference_frames[0].T@(BEAM_REFERENCE-REFERENCE[1])
    a = joint.reference_frames[0].T@joint.reference_frames[1]
    phi = Rotation.from_matrix(a.T@rs.T@rb).as_rotvec()
    angle = np.linalg.norm(phi)
    if angle < 1e-4:
        factor = 1/12+angle**2/720+angle**4/30240
    else:
        factor = 1/angle**2-(1+np.cos(angle))/(2*angle*np.sin(angle))
    p = skew(phi)
    inverse = np.eye(3)-.5*p+factor*p@p
    b = inverse@a.T@rs.T
    jac = np.zeros((6, 12))
    jac[:3, :3] = -rs.T
    jac[:3, 3:6] = rs.T@skew(d)
    jac[:3, 6:9] = rs.T
    jac[3:, 3:6] = -b
    jac[3:, 9:12] = b
    constraints = np.r_[rs.T@d-e, phi]
    spatial = np.r_[jac.T@q[30:36], constraints]
    chart = np.eye(18)
    chart[3:6, 3:6] = left_jacobian(q[9:12])
    chart[9:12, 9:12] = left_jacobian(q[27:30])
    return chart.T@spatial


def fixture():
    model = FEModel('g0-f05-q4-pose-pullback-dev-v2')
    curve = DNVC208MaterialCurve(sigma_prop=100e6, sigma_yield=105e6,
        sigma_yield_2=110e6, eps_p_y1=.005, eps_p_y2=.010, K=400e6, n=.20)
    material = model.add_material('steel', 210e9, .3, density=7850.,
        yield_stress=100e6, hardening_curve=curve)
    for node_id, point in enumerate(REFERENCE, start=1):
        model.add_node(node_id, *point)
    element = create_shell_element(1, [1, 2, 3, 4], 'steel', formulation='e4-pl',
        thickness=.02, reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    accepted = np.zeros(24)
    accepted[6] = accepted[12] = .004
    accepted[8] = .0004
    accepted[14] = .0007
    local = deformation(REFERENCE, accepted)[0]
    _, _, parent_trial = element.compute_nonlinear_response(
        model.mesh, material, local, None, 3, True)
    predecessor = element.seal_committed_current_tangent_state(
        model.mesh, material, local, parent_trial, 3)
    q = np.zeros(36)
    q[:24] = accepted
    q[6] += .0002
    q[12] += .00015
    q[8] += .00004
    q[14] += .00005
    q[:24].reshape(4, 6)[:, 3:] = np.array(((.003, -.002, .001),
        (-.001, .002, -.003), (.001, .001, .002), (.002, -.001, .003)))
    q[24:30] = (.00003, -.00007, .00002, .015, -.011, .008)
    q[30:36] = (2e6, -1e6, .5e6, 2e4, -3e4, 1e4)
    frames = np.array((np.eye(3), np.eye(3)))
    joint = RigidPoseJoint(np.array((REFERENCE[1], BEAM_REFERENCE)), frames)
    return model, material, element, predecessor, q, joint, frames


def material_response(q, model, material, element, predecessor, tangent):
    local, differential, second = deformation(REFERENCE, q[:24])
    force, k, state = element.compute_nonlinear_response(
        model.mesh, material, local, predecessor, 3, tangent)
    force = np.asarray(force)
    result = np.zeros(36)
    result[:24] = differential.T@force
    matrix = None
    if tangent:
        matrix = np.zeros((36, 36))
        matrix[:24, :24] = (differential.T@np.asarray(k)@differential
            +np.einsum('i,ijk->jk', force, second))
    if not np.isfinite(result).all() or (matrix is not None and not np.isfinite(matrix).all()):
        raise ValueError('nonfinite material force or tangent')
    return result, matrix, state


def owner_joint(q, joint, frames):
    positions = np.array((REFERENCE[1]+q[6:9], BEAM_REFERENCE+q[24:27]))
    coordinates = np.array((q[9:12], q[27:30]))
    got = joint.evaluate_ports(positions, frames, np.array(q[30:36]),
        rotation_coordinates=coordinates, charted_ports=(True, True))
    force = np.zeros(36)
    force[PORT] = got.residual
    tangent = np.zeros((36, 36))
    tangent[np.ix_(PORT, PORT)] = got.tangent
    return force, tangent


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))


def measure():
    model, material, element, predecessor, q, joint, frames = fixture()
    sealed = canonical_json_bytes(predecessor)
    parent_alpha = np.asarray(predecessor['alpha'])
    mf, mk, state = material_response(q, model, material, element, predecessor, True)
    jf, jk = owner_joint(q, joint, frames)
    independent = independent_joint_residual(q, joint)
    first_error = relative(jf[PORT], independent)
    base_mask = np.asarray(state['alpha']) > parent_alpha
    if not np.any(base_mask):
        raise ValueError('yielded Q4 trial has no active plastic station')
    vectors = []
    for phase in (.3, .7):
        v = np.zeros(36)
        v[:30] = np.sin(np.arange(30)*(.73 if phase == .7 else 1.)+phase)
        v[:30] /= np.linalg.norm(v[:30])
        v[30:36] = 1e5*np.cos(np.arange(6)*.53+phase)
        vectors.append(v)
    directions = []
    for v in vectors:
        material_expected = mk@v
        joint_expected = jk@v
        material_errors = []
        joint_errors = []
        combined_errors = []
        masks = []
        for h in STEPS:
            sampled = []
            sampled_masks = []
            for factor in (-1., -.5, .5, 1.):
                trial = q+factor*h*v
                force, _, candidate = material_response(
                    trial, model, material, element, predecessor, False)
                pose = np.zeros(36)
                pose[PORT] = independent_joint_residual(trial, joint)
                sampled.append((force, pose))
                sampled_masks.append(np.asarray(candidate['alpha']) > parent_alpha)
            md = (sampled[3][0]-sampled[0][0])/(2*h)
            jd = (sampled[3][1]-sampled[0][1])/(2*h)
            material_errors.append(relative(material_expected, md))
            joint_errors.append(relative(joint_expected, jd))
            combined_errors.append(relative(material_expected+joint_expected, md+jd))
            masks.append(all(np.array_equal(mask, base_mask) for mask in sampled_masks))
        plateau = any(all(errors[i] <= TOLERANCE and errors[i+1] <= TOLERANCE
            for errors in (material_errors, joint_errors, combined_errors))
            and masks[i] and masks[i+1] for i in range(len(STEPS)-1))
        directions.append(dict(material_errors=material_errors, joint_errors=joint_errors,
            combined_errors=combined_errors, sampled_active_masks_stable=masks,
            adjacent_plateau=plateau))
    immutable = canonical_json_bytes(predecessor) == sealed
    finite = all(np.isfinite(x).all() for x in (mf, mk, jf, jk, independent))
    pass_check = (first_error <= 1e-10 and immutable and finite
        and all(item['adjacent_plateau'] for item in directions))
    return dict(status='DEV_F05_POSE_PASS' if pass_check else 'DEV_F05_POSE_FAIL',
        first_variation_relative_error=first_error, active_stations=int(base_mask.sum()),
        parent_alpha_max=float(parent_alpha.max()),
        trial_alpha_max=float(np.asarray(state['alpha']).max()),
        material_force_norm=float(np.linalg.norm(mf)),
        joint_force_norm=float(np.linalg.norm(jf)),
        joint_cross_block_norm=float(np.linalg.norm(jk[np.ix_(PORT[:6], PORT[6:])])),
        predecessor_unchanged=immutable, finite=finite, steps=STEPS,
        directions=directions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    modules = ('_ge_beam3_variational_shell.py', '_ge_beam3_pose_joint.py',
        '_ge_beam3_mixed_ad.py', 'e4_pl_element.py', 'e4_pl_s3_state.py',
        'material_curves.py', 'plasticity.py')
    manifest = dict(purpose='F05_DEV_Q4_POSE_NOT_G0B_ACCEPTANCE',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        modules_sha256={name: hashlib.sha256((REPOSITORY/'src'/'anyfesolver'/name).read_bytes()).hexdigest()
            for name in modules}, python=sys.version, numpy=np.__version__,
        anyfesolver_import=str(Path(anyfesolver.__file__).resolve()),
        numba_jit_disabled=os.environ.get('NUMBA_DISABLE_JIT') == '1',
        numba_cache=os.environ['NUMBA_CACHE_DIR'], steps=STEPS,
        tolerance=TOLERANCE, predecessor='Q4 v1 loaded sealed state')
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    try:
        with threadpool_limits(limits=1):
            result = measure()
    except Exception as error:
        result = dict(status='DEV_F05_POSE_EXCEPTION', failure_type=type(error).__name__,
            failure=str(error))
    (args.root/'result.json').write_text(json.dumps(result, indent=2,
        sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],
        first_variation_error=result.get('first_variation_relative_error'),
        directions=result.get('directions'), failure=result.get('failure'))))
    return 0 if result['status'] == 'DEV_F05_POSE_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
