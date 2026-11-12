"""Independent analytic first variation and finite-difference spatial derivative.

The oracle uses scipy Rotation and closed-form relative-pose derivatives, not
the production SO(3) jets. This is independent implementation, not independent
authorship or beam/shell owner integration qualification.
"""
from hashlib import sha256
import json
from math import fsum
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled


def exp(v):
    return Rotation.from_rotvec(v).as_matrix()


def cross(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def specimen(kind='eccentric'):
    positions = np.array([[.2, -.4, .7], [.4, .1, 1.]])
    frames = np.array([exp([.2, -.1, .3]), exp([-.7, .4, .6])])
    if kind == 'coincident': positions[1] = positions[0]
    joint = RigidPoseJoint(positions, frames)
    qs = exp([1.1, -.4, .3]) @ frames[0]
    relative = frames[0].T @ frames[1]
    phi = np.array([2.6, .1, -.05]) if kind == 'large-relative' else np.array([.4, -.5, .2])
    gap = np.array([.04, -.07, .02])
    if kind == 'satisfied': phi *= 0.; gap *= 0.
    e = frames[0].T @ (positions[1]-positions[0])
    x = np.array([[1.3, -.7, 2.1], [0., 0., 0.]])
    x[1] = x[0]+qs@(e+gap)
    q = np.array([qs, qs@relative@exp(phi)])
    return joint, x, q, np.array([.6, -.3, .5, -.2, .7, -.4])


def oracle(joint, positions, frames, multipliers):
    qs, qb = frames
    d = positions[1]-positions[0]
    e = joint.reference_frames[0].T @ (joint.reference_positions[1]-joint.reference_positions[0])
    a = joint.reference_frames[0].T @ joint.reference_frames[1]
    phi = Rotation.from_matrix(a.T@qs.T@qb).as_rotvec()
    angle = np.linalg.norm(phi)
    factor = (1/12+angle**2/720+angle**4/30240) if angle < 1e-5 else (
        1/angle**2-(1+np.cos(angle))/(2*angle*np.sin(angle)))
    inverse = np.eye(3)-.5*cross(phi)+factor*cross(phi)@cross(phi)
    b = inverse@a.T@qs.T
    jac = np.zeros((6, 12))
    jac[:3, :3] = -qs.T
    jac[:3, 3:6] = qs.T@cross(d)
    jac[:3, 6:9] = qs.T
    jac[3:, 3:6] = -b
    jac[3:, 9:12] = b
    c = np.r_[qs.T@d-e, phi]
    return float(multipliers@c), c, jac, np.r_[jac.T@multipliers, c]


def perturb(x, q, force, step):
    return (x+step[:12].reshape(2, 6)[:, :3],
        np.array([exp(step[6*i+3:6*i+6])@q[i] for i in range(2)]), force+step[12:])


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))


@pytest.mark.parametrize('kind', ('coincident', 'satisfied', 'eccentric', 'large-relative'))
def test_first_variation_spatial_jacobian_and_second_work(kind, tmp_path):
    joint, x, q, force = specimen(kind)
    before = canonical(dict(x=x, q=q, force=force))
    got = joint.evaluate(x, q, force)
    value, c, jac, r = oracle(joint, x, q, force)
    assert abs(got.potential-value) <= 1e-11
    assert relative(got.constraints, c) <= 1e-11
    assert relative(got.constraint_jacobian, jac) <= 1e-11
    assert relative(got.residual, r) <= 1e-11
    assert relative(got.chart_hessian, got.chart_hessian.T) <= 1e-11
    derivative = np.empty((18, 18))
    h = 2e-6
    for i in range(18):
        step = np.zeros(18); step[i] = h
        rp = oracle(joint, *perturb(x, q, force, step))[-1]
        rm = oracle(joint, *perturb(x, q, force, -step))[-1]
        derivative[:, i] = (rp-rm)/(2*h)
    error = relative(got.spatial_jacobian, derivative)
    assert error <= 1e-7
    assert np.linalg.norm(got.spatial_jacobian-got.chart_hessian) > .1
    direction = np.sin(np.arange(18)+.3)/3
    h = 2e-4
    vp = oracle(joint, *perturb(x, q, force, h*direction))[0]
    vm = oracle(joint, *perturb(x, q, force, -h*direction))[0]
    second = (vp-2*value+vm)/h**2
    expected = float(direction@got.chart_hessian@direction)
    assert abs(second-expected)/max(1., abs(expected)) <= 1e-7
    assert canonical(dict(x=x, q=q, force=force)) == before
    assert not got.production_qualified and not got.owner_binding_authorized
    for name in ('constraints', 'constraint_jacobian', 'residual', 'chart_hessian', 'spatial_jacobian'):
        with pytest.raises(ValueError): getattr(got, name).setflags(write=True)
    (tmp_path/'work.json').write_bytes(canonical(dict(kind=kind, result=got,
        oracle_residual=r, spatial_derivative_error=error, second_work=second)))


@pytest.mark.parametrize('vector', ([.4, -.9, 1.2], [0., np.pi, 0.], [2.8, -.7, .3]))
def test_common_finite_motion_objectivity_and_wrench_balance(vector, tmp_path):
    joint, x, q, force = specimen()
    old = joint.evaluate(x, q, force)
    s = exp(vector)
    new = joint.evaluate(x@s.T+np.array([.7, -1.2, .9]), s@q, force)
    transport = np.eye(18); transport[:12, :12] = np.kron(np.eye(4), s)
    assert relative(new.constraints, old.constraints) <= 1e-11
    assert relative(new.residual, transport@old.residual) <= 1e-11
    assert relative(new.chart_hessian, transport@old.chart_hessian@transport.T) <= 1e-11
    assert relative(new.spatial_jacobian, transport@old.spatial_jacobian@transport.T) <= 1e-11
    assert abs(new.potential-old.potential) <= 1e-11
    f0, m0, f1, m1 = old.residual[:12].reshape(4, 3)
    assert np.linalg.norm(f0+f1) <= 1e-11
    assert np.linalg.norm(np.cross(x[1]-x[0], f1)+m0+m1) <= 1e-11
    (tmp_path/'objectivity.json').write_bytes(canonical(dict(rotation=s,
        original=old, transformed=new, balanced_wrench=True)))


def test_reference_saddle_rank_rigids_and_linear_offset_limit(tmp_path):
    positions = np.array([[0., 0., 0.], [.2, -.3, .4]])
    frames = np.array([np.eye(3), np.eye(3)])
    joint = RigidPoseJoint(positions, frames)
    got = joint.evaluate(positions, frames, np.zeros(6))
    expected = np.zeros((6, 12))
    expected[:3, :3] = -np.eye(3); expected[:3, 3:6] = cross(positions[1])
    expected[:3, 6:9] = np.eye(3)
    expected[3:, 3:6] = -np.eye(3); expected[3:, 9:12] = np.eye(3)
    np.testing.assert_array_equal(got.constraint_jacobian, expected)
    assert np.linalg.norm(got.residual) == 0.
    assert np.linalg.matrix_rank(got.constraint_jacobian) == 6
    eigen = np.linalg.eigvalsh(got.chart_hessian)
    assert (np.sum(eigen > 1e-11), np.sum(eigen < -1e-11), np.sum(abs(eigen) <= 1e-11)) == (6, 6, 6)
    rigid = np.zeros((12, 6))
    for i, point in enumerate(positions):
        rigid[6*i:6*i+3, :3] = np.eye(3)
        rigid[6*i:6*i+3, 3:] = -cross(point)
        rigid[6*i+3:6*i+6, 3:] = np.eye(3)
    assert np.linalg.norm(got.constraint_jacobian@rigid) <= 1e-11
    (tmp_path/'saddle.json').write_bytes(canonical(dict(result=got, eigenvalues=eigen, rigid=rigid)))


def test_large_common_translation_uses_position_low():
    joint, x, q, force = specimen('satisfied')
    old = joint.evaluate(x, q, force)
    shift = np.array([1e9, -2e9, 3e9])
    moved = x+shift
    low = np.array([[fsum((float(x[i, j]), float(shift[j]), -float(moved[i, j])))
        for j in range(3)] for i in range(2)])
    got = joint.evaluate(moved, q, force, position_low=low)
    assert relative(got.constraints, old.constraints) <= 1e-11
    assert relative(got.spatial_jacobian, old.spatial_jacobian) <= 1e-11


def test_successive_noncommuting_rotations_preserve_rigid_attachment(tmp_path):
    joint, _, _, _ = specimen()
    a = joint.reference_frames[0].T@joint.reference_frames[1]
    e = joint.reference_frames[0].T@(joint.reference_positions[1]-joint.reference_positions[0])
    qs = joint.reference_frames[0].copy(); rows = []
    for i in range(12):
        qs = exp([.8, -.4 if i % 2 else .6, .5])@qs
        xs = np.array([.1*i, -.04*i, .03*i])
        got = joint.evaluate(np.array([xs, xs+qs@e]), np.array([qs, qs@a]), np.zeros(6))
        assert np.linalg.norm(got.constraints) <= 1e-11
        rows.append(got.constraints)
    (tmp_path/'finite-path.json').write_bytes(canonical(dict(constraints=rows, final_master_frame=qs)))


@pytest.mark.parametrize('mutation', ('schema', 'policy', 'ports', 'update', 'qualification',
    'basis', 'duplicate', 'nonfinite', 'overflow', 'integer', 'boolean', 'extra', 'whitespace', 'hash'))
def test_strict_definition_roundtrip_and_mutation(mutation):
    joint, _, _, _ = specimen()
    raw = joint.to_bytes()
    assert RigidPoseJoint.from_bytes(raw, expected_sha256=sha256(raw).hexdigest()).to_bytes() == raw
    data = json.loads(raw)
    if mutation == 'schema': data['schema'] = 'legacy'
    elif mutation == 'policy': data['policy'] = 'penalty'
    elif mutation == 'ports': data['port_order'].reverse()
    elif mutation == 'update': data['update'] = 'ADDITIVE_ROTATION_VECTOR'
    elif mutation == 'qualification': data['production_qualified'] = True
    elif mutation == 'basis': data['multiplier_basis'] = 'SPATIAL'
    elif mutation == 'integer': data['reference_positions'][0][0] = 0
    elif mutation == 'boolean': data['reference_positions'][0][0] = False
    elif mutation == 'extra': data['extra'] = 0
    raw = canonical(data)
    if mutation == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    elif mutation == 'nonfinite': raw = raw.replace(b'false', b'NaN', 1)
    elif mutation == 'overflow': raw = raw.replace(b'false', b'1e9999', 1)
    elif mutation == 'whitespace': raw += b' '
    with pytest.raises(ValueError):
        RigidPoseJoint.from_bytes(raw, expected_sha256='0'*64 if mutation == 'hash' else sha256(raw).hexdigest())


@pytest.mark.parametrize('kind', ('reflection', 'nonorthogonal', 'bool', 'shape', 'nan',
    'relative-cutback', 'cancel', 'compiled', 'immutable'))
def test_invalid_pose_authority_and_unsupported_chart(kind):
    joint, x, q, force = specimen()
    token = None
    if kind == 'reflection': q[0, :, 0] *= -1
    elif kind == 'nonorthogonal': q[0, 0, 0] += .1
    elif kind == 'bool': force = np.zeros(6, dtype=bool)
    elif kind == 'shape': x = x[:1]
    elif kind == 'nan': force[0] = float('nan')
    elif kind == 'relative-cutback':
        a = joint.reference_frames[0].T@joint.reference_frames[1]
        q[1] = q[0]@a@exp([.95*np.pi, 0., 0.])
    elif kind == 'cancel': token = CancellationToken(); token.cancel()
    elif kind == 'compiled': object.__setattr__(joint, '_offset', joint._offset+1.)
    elif kind == 'immutable':
        with pytest.raises(AttributeError): joint.identity = '0'*64
        with pytest.raises(ValueError): joint.reference_frames.setflags(write=True)
        return
    with pytest.raises((ValueError, SolveCancelled)):
        joint.evaluate(x, q, force, cancellation_token=token)


def left_jacobian(theta):
    angle = np.linalg.norm(theta); k = cross(theta)
    if angle < 1e-6:
        return np.eye(3)+(.5-angle**2/24)*k+(1/6-angle**2/120)*k@k
    return np.eye(3)+(1-np.cos(angle))/angle**2*k+(angle-np.sin(angle))/angle**3*k@k


def port_oracle(joint, x, anchors, force, coordinates, mask):
    frames = np.array([exp(coordinates[i])@anchors[i] if mask[i] else anchors[i] for i in range(2)])
    result = oracle(joint, x, frames, force)
    transform = np.eye(18)
    for i in range(2):
        if mask[i]: transform[6*i+3:6*i+6, 6*i+3:6*i+6] = left_jacobian(coordinates[i])
    return transform.T@result[-1]


@pytest.mark.parametrize('mask', ((False, False), (True, False), (False, True), (True, True)))
def test_exact_mixed_port_chart_derivative(mask, tmp_path):
    joint, x, q, force = specimen()
    coordinates = np.array([[3.2, -.5, .2], [-.7, .4, .2]])
    anchors = q.copy()
    for i in range(2):
        if mask[i]: anchors[i] = exp(-coordinates[i])@q[i]
        else: coordinates[i] = 0.
    result = joint.evaluate_ports(x, anchors, force, rotation_coordinates=coordinates, charted_ports=mask)
    assert relative(result.frames, q) <= 1e-11
    expected = port_oracle(joint, x, anchors, force, coordinates, mask)
    assert relative(result.residual, expected) <= 1e-11
    derivative = np.empty((18, 18)); h = 2e-6
    for index in range(18):
        values = []
        for sign in (1., -1.):
            step = np.zeros(18); step[index] = sign*h
            moved = anchors.copy(); changed = coordinates.copy()
            for i in range(2):
                if mask[i]: changed[i] += step[6*i+3:6*i+6]
                else: moved[i] = exp(step[6*i+3:6*i+6])@anchors[i]
            values.append(port_oracle(joint, x+step[:12].reshape(2, 6)[:, :3], moved,
                force+step[12:], changed, mask))
        derivative[:, index] = (values[0]-values[1])/(2*h)
    error = relative(result.tangent, derivative)
    assert error <= 1e-7
    if all(mask): assert relative(result.tangent, result.tangent.T) <= 1e-11
    else: assert np.linalg.norm(result.tangent-result.tangent.T) > .1
    (tmp_path/'port-work.json').write_bytes(canonical(dict(mask=mask, result=result,
        derivative_error=error)))


def test_half_turn_is_admissible_but_singular_total_chart_is_not():
    joint, _, _, force = specimen()
    coordinates = np.array([[0., np.pi, 0.], [0., np.pi, 0.]])
    q = exp(coordinates[0])
    x = joint.reference_positions@q.T
    result = joint.evaluate_ports(x, joint.reference_frames, force,
        rotation_coordinates=coordinates, charted_ports=(True, True))
    assert np.linalg.norm(result.pose.constraints) <= 1e-11
    assert np.linalg.matrix_rank(result.coordinate_map) == 18
    with pytest.raises(ValueError, match='singular additive'):
        joint.evaluate_ports(x, joint.reference_frames, force,
            rotation_coordinates=2*coordinates, charted_ports=(True, True))
    # A common 2*pi motion remains admissible in matrix coordinates.
    assert np.linalg.norm(joint.evaluate(joint.reference_positions,
        joint.reference_frames, force).constraints) <= 1e-11


@pytest.mark.parametrize('kind', ('mask-type', 'mask-size', 'spatial-coordinate', 'boolean', 'cancel'))
def test_mixed_port_controls(kind):
    joint, x, q, force = specimen()
    mask = (True, False); coordinates = np.zeros((2, 3)); token = None
    if kind == 'mask-type': mask = (1, False)
    elif kind == 'mask-size': mask = (True,)
    elif kind == 'spatial-coordinate': coordinates[1, 0] = .1
    elif kind == 'boolean': coordinates = coordinates.astype(bool)
    elif kind == 'cancel': token = CancellationToken(); token.cancel()
    with pytest.raises((ValueError, SolveCancelled)):
        joint.evaluate_ports(x, q, force, rotation_coordinates=coordinates,
            charted_ports=mask, cancellation_token=token)


@pytest.mark.parametrize('raw', (b'\xff', b'['*2000+b'0'+b']'*2000, b'', b' '*16385))
def test_malformed_bounded_decoder(raw):
    with pytest.raises(ValueError):
        RigidPoseJoint.from_bytes(raw, expected_sha256=sha256(raw).hexdigest())
