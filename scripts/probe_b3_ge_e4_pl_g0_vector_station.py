"""G0 F03-V: bounded vector pose of a shell strip and native lifted B3 station."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, so3_exp, so3_log, transpose
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_variational_shell import rotation_jets


COUNT = 57
SHELL = 36
BEAM = 36
CELLS = 45
MULTIPLIERS = 51
STATION = -.25
WINDOW = .2


def fixture(topology, curved):
    nodes = np.array([(x, y, .06*(1-x*x) if curved else 0.)
                      for y in (-.5, .5) for x in (-1., 0., 1.)])
    beam = np.array([(x, 0., (.06*(1-x*x) if curved else 0.)+.2)
                     for x in (-1., 0., 1.)])
    frames = []
    for x in (-1., 0., 1.):
        tangent = np.array((1., 0., -.12*x if curved else 0.))
        tangent /= np.linalg.norm(tangent)
        width = np.array((0., 1., 0.))
        frames.append(np.column_stack((tangent, width, np.cross(tangent, width))))
    reference = CenteredCurvedBeam3ReferenceGeometry(beam, np.array(frames))
    weights = []
    for x,y in ((STATION-WINDOW,-.25), (STATION+WINDOW,-.25),
                (STATION+WINDOW,.25), (STATION-WINDOW,.25)):
        xi, eta = x+1., y+.5
        out = np.zeros(6)
        if topology == "Q4":
            out[[0,1,4,3]] = ((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta)
        else:
            tri = (0,1,4) if eta <= xi else (0,4,3)
            corners = np.array([nodes[i, :2] for i in tri])
            bary = np.linalg.solve(np.vstack((corners.T, np.ones(3))), np.array([x,y,1.]))
            out[list(tri)] = bary
        assert abs(sum(out)-1.) < 1e-14 and np.count_nonzero(out) <= (4 if topology == "Q4" else 3)
        weights.append(out)
    weights = np.array(weights)
    samples = weights @ nodes
    shell_ref = Rotation.align_vectors(samples-samples.mean(axis=0),
                                       samples-samples.mean(axis=0))[0].as_matrix()
    station_ref = reference.position_pair(STATION)
    beam_ref = station_ref[0]+station_ref[1]
    beam_frame = reference.frame(STATION)
    e0 = shell_ref.T@(beam_ref-samples.mean(axis=0))
    a0 = shell_ref.T@beam_frame
    return nodes, reference, weights, samples, shell_ref, e0, a0


def independent(q, fixture_data, motion=None):
    nodes, reference, weights, samples0, _, e0, a0 = fixture_data
    current_shell = nodes+q[:SHELL].reshape(6,6)[:,:3]
    shell = weights @ current_shell
    x = reference.coordinates+q[BEAM:CELLS].reshape(3,3)
    u = [Rotation.from_rotvec(q[CELLS+3*c:CELLS+3*c+3]).as_matrix() for c in (0,1)]
    if motion is not None:
        w,t = motion
        shell = shell@w.T+t
        x = x@w.T+t
        u = [w@v for v in u]
    centroid = shell.mean(axis=0)
    fitted = Rotation.align_vectors(shell-centroid, samples0-samples0.mean(axis=0))[0].as_matrix()
    tau = STATION+1.
    lift = reference.half_cell_lift(0, tau)
    beam_position = (1-tau)*x[0]+tau*x[1]+u[0]@lift
    beam_frame = u[0]@reference.frame(STATION)
    gap = np.r_[fitted.T@(beam_position-centroid)-e0,
                Rotation.from_matrix(a0.T@fitted.T@beam_frame).as_rotvec()]
    return float(q[MULTIPLIERS:]@gap), gap


def evaluate(q, fixture_data):
    nodes, reference, weights, samples0, _, e0, a0 = fixture_data
    variables = [Jet2.variable(float(value), i, COUNT) for i,value in enumerate(q)]
    current = nodes+q[:SHELL].reshape(6,6)[:,:3]
    sample_now = weights @ current
    sample_fit = rotation_jets(samples0, sample_now)
    transform = np.zeros((24,COUNT))
    for sample in range(4):
        for node in range(6):
            for axis in range(3):
                transform[6*sample+axis,6*node+axis] = weights[sample,node]
    fitted = [[Jet2(v.value, transform.T@v.gradient,
                    transform.T@v.hessian@transform) for v in row] for row in sample_fit]
    shell = [[Jet2.constant(float(samples0[i,j]),COUNT)+sum(
        (weights[i,n]*variables[6*n+j] for n in range(6)),Jet2.constant(0.,COUNT))
        for j in range(3)] for i in range(4)]
    centroid = [sum((shell[i][j] for i in range(4)),Jet2.constant(0.,COUNT))/4.
                for j in range(3)]
    tau = STATION+1.
    lift = reference.half_cell_lift(0,tau)
    cell = so3_exp(variables[CELLS:CELLS+3])
    beam = [(1-tau)*(float(reference.coordinates[0,j])+variables[BEAM+j])
            +tau*(float(reference.coordinates[1,j])+variables[BEAM+3+j])
            +matvec(cell, [Jet2.constant(float(v),COUNT) for v in lift])[j]
            for j in range(3)]
    frame = matmul(cell, constant_matrix(reference.frame(STATION),COUNT))
    ct = [v-float(e0[j]) for j,v in enumerate(
        matvec(transpose(fitted), [beam[j]-centroid[j] for j in range(3)]))]
    cr = so3_log(matmul(constant_matrix(a0.T,COUNT),matmul(transpose(fitted),frame)))
    gaps = ct+cr
    work = sum((variables[MULTIPLIERS+i]*gaps[i] for i in range(6)),Jet2.constant(0.,COUNT))
    return work, np.array([v.value for v in gaps])


def trial():
    q = np.zeros(COUNT)
    q[:SHELL].reshape(6,6)[:,:3] = .008*np.sin(np.arange(18).reshape(6,3)+.3)
    q[:SHELL].reshape(6,6)[:,3:] = .004*np.cos(np.arange(18).reshape(6,3)+.7)
    q[BEAM:CELLS] = .012*np.sin(np.arange(9)+.21)
    q[CELLS:CELLS+6] = (.04,-.03,.05,-.02,.04,.01)
    q[MULTIPLIERS:] = (.6,-.3,.5,-.2,.7,-.4)
    return q


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b)))


def row(topology, curved):
    data = fixture(topology, curved)
    reference = np.zeros(COUNT)
    _, reference_gap = evaluate(reference, data)
    normal = data[4][:,2]
    ref_beam_frame = data[1].frame(STATION)
    reference[MULTIPLIERS+3:] = ref_beam_frame.T@normal
    moment_work, _ = evaluate(reference, data)
    shell_force = moment_work.gradient[:SHELL].reshape(6,6)[:,:3]
    shell_rotations = float(np.linalg.norm(moment_work.gradient[:SHELL].reshape(6,6)[:,3:]))
    shell_force_balance = float(np.linalg.norm(shell_force.sum(axis=0)))
    shell_moment = np.cross(data[0],shell_force).sum(axis=0)
    beam_moment = moment_work.gradient[CELLS:CELLS+3]
    reference_moment_balance = float(np.linalg.norm(shell_moment+beam_moment))

    q = trial()
    work, gaps = evaluate(q,data)
    independent_value, independent_gap = independent(q,data)
    h = 2e-6
    derivative = np.zeros(COUNT)
    eye = np.eye(COUNT)
    for i in range(COUNT):
        derivative[i] = (independent(q+h*eye[i],data)[0]
                         -independent(q-h*eye[i],data)[0])/(2*h)
    gradient_error = relative(work.gradient,derivative)
    direction = np.sin(np.arange(COUNT)+.3)
    direction /= np.linalg.norm(direction)
    expected = float(direction@work.hessian@direction)
    second_errors = []
    for step in (2e-4,1e-4):
        plus = independent(q+step*direction,data)[0]
        minus = independent(q-step*direction,data)[0]
        second_errors.append(abs((plus-2*independent_value+minus)/step**2-expected)/
                             max(1.,abs(expected)))
    skew = relative(work.hessian,work.hessian.T)
    w = Rotation.from_rotvec((.5,-.3,.4)).as_matrix()
    t = np.array((1.,-.7,.9))
    moved = q.copy()
    moved[:SHELL].reshape(6,6)[:,:3] = (data[0]+q[:SHELL].reshape(6,6)[:,:3])@w.T+t-data[0]
    x = data[1].coordinates+q[BEAM:CELLS].reshape(3,3)
    moved[BEAM:CELLS] = (x@w.T+t-data[1].coordinates).ravel()
    for c in (0,1):
        u = Rotation.from_rotvec(q[CELLS+3*c:CELLS+3*c+3]).as_matrix()
        moved[CELLS+3*c:CELLS+3*c+3] = Rotation.from_matrix(w@u).as_rotvec()
    moved_work,moved_gap = evaluate(moved,data)
    motion_work,motion_gap = independent(q,data,(w,t))
    objectivity = max(abs(moved_work.value-work.value),abs(motion_work-independent_value),
                      relative(moved_gap,gaps),relative(motion_gap,independent_gap))
    shell = data[0]+q[:SHELL].reshape(6,6)[:,:3]
    beam_nodes = data[1].coordinates+q[BEAM:CELLS].reshape(3,3)
    sf = work.gradient[:SHELL].reshape(6,6)[:,:3]
    bf = work.gradient[BEAM:CELLS].reshape(3,3)
    cell_moment = np.zeros(3)
    for c in (0,1):
        _,jac,_ = _exp_terms(q[CELLS+3*c:CELLS+3*c+3])
        cell_moment += np.linalg.solve(jac.T,work.gradient[CELLS+3*c:CELLS+3*c+3])
    balance = max(float(np.linalg.norm(sf.sum(axis=0)+bf.sum(axis=0))),
                  float(np.linalg.norm(np.cross(shell,sf).sum(axis=0)+
                                       np.cross(beam_nodes,bf).sum(axis=0)+cell_moment)))
    passed = (np.linalg.norm(reference_gap)<=1e-12 and shell_rotations==0.
              and shell_force_balance<=1e-10 and reference_moment_balance<=1e-10
              and abs(work.value-independent_value)<=1e-10
              and relative(gaps,independent_gap)<=1e-10
              and gradient_error<=1e-7 and max(second_errors)<=1e-5
              and skew<=1e-10 and objectivity<=1e-10 and balance<=1e-8)
    return dict(topology=topology,curved=curved,status="diagnostic_pass" if passed else "failed",
                reference_gap=float(np.linalg.norm(reference_gap)),
                reference_shell_force_balance=shell_force_balance,
                reference_moment_balance=reference_moment_balance,
                shell_rotation_gradient=shell_rotations,
                value_error=abs(work.value-independent_value),
                gap_error=relative(gaps,independent_gap),gradient_error=gradient_error,
                directional_hessian_errors=second_errors,hessian_skew=skew,
                common_motion_error=objectivity,wrench_balance=balance)


if __name__ == "__main__":
    results = [row(topology,curved) for topology in ("Q4","S3-V2D")
               for curved in (False,True)]
    passed = all(case["status"]=="diagnostic_pass" for case in results)
    print(json.dumps(dict(profile="G0-F03-V-v1",status="diagnostic_pass" if passed else "failed",
                          results=results),sort_keys=True))
    raise SystemExit(0 if passed else 1)
