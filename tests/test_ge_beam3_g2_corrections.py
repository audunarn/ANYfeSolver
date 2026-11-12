"""G2-IR regressions: independent SO(3) work/transport, actual Newton seam."""
from hashlib import sha256
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from scipy.linalg import block_diag, null_space
from scipy.spatial.transform import Rotation
from anyfesolver._ge_beam3_g1_analysis import ElasticAnalysis
from anyfesolver._ge_beam3_g2_analysis import ConstrainedAnalysis
from anyfesolver._ge_beam3_g2_constraints import ConstraintSet
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_elastic import canonical
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from test_ge_beam3_g2_constraints import make, snapshot, invariant, FIXTURE
from test_ge_beam3_g1_elastic import reference, section


def record(name, value):
    directory = os.environ.get("G2_CORRECTION_RECORDS")
    if directory:
        with (Path(directory)/(name+".json")).open("xb") as stream:
            stream.write(canonical(value))


def spatial_exp_jacobian(v):
    # Independent Rodrigues derivative, not the production chart helper.
    x, y, z = v; S = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm(v)
    if angle == 0: return np.eye(3)
    return np.eye(3)+(1-np.cos(angle))/angle**2*S+(angle-np.sin(angle))/angle**3*(S @ S)


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("route", ["unbound", "super"])
def test_inherited_solve_rejects_without_state_change(prefix, route):
    descriptor = make().constraints.descriptor()
    descriptor["affine"][0]["offset"] = .001
    owner = ConstrainedAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),),
                                ConstraintSet.from_descriptor(descriptor))
    if prefix: owner.solve(np.zeros(18))
    before = snapshot(owner); checkpoint = owner.checkpoint()
    invocation = (lambda: ElasticAnalysis.solve(owner, np.zeros(18))) if route == "unbound" else (
        lambda: super(ConstrainedAnalysis, owner).solve(np.zeros(18)))
    # Rejection must occur before guard/evaluation/callback/commit activity.
    with patch.object(owner, "_evaluate", side_effect=AssertionError("evaluated")), \
         patch.object(owner.store, "commit", side_effect=AssertionError("committed")):
        with pytest.raises(ValueError, match="exact G1 owner"): invocation()
    assert snapshot(owner) == before and owner.checkpoint() == checkpoint
    assert not owner.store.has_active_trial
    restored = ConstrainedAnalysis.resume(checkpoint, sha256(checkpoint).hexdigest())
    assert restored.checkpoint() == checkpoint
    result = restored.solve(np.zeros(18))
    invariant(result["u"][0], .001)
    record(f"bypass-{route}-{prefix}", dict(rejected=True, prefix=prefix, route=route,
                                           checkpoint_sha256=sha256(checkpoint).hexdigest()))


def test_invariant_rejects_default_rtol_sized_error():
    with pytest.raises(AssertionError): invariant(np.array([1.+5e-8]), np.ones(1))
    with pytest.raises(AssertionError): invariant(np.array([float("nan")]), np.ones(1))
    invariant([1.+1e-12], [1.])


def test_prescribed_target_work_at_noncommuting_loaded_states():
    owner = make(); force = np.zeros(18)
    force[12:15] = FIXTURE["tip_force"]; force[15:18] = FIXTURE["tip_moment"]
    target = np.eye(3); results = []
    spatial_direction = np.array([.2, -.3, .1])
    for vector in FIXTURE["orientation_vectors"]:
        base = owner.store.native_rotation_store.committed_full_displacement.copy()
        rotations = owner.store.native_rotation_store.committed_rotation_matrices.copy()
        target = Rotation.from_rotvec(vector).as_matrix() @ target
        actual = owner.solve(force, targets=[target]); u = actual["u"]
        A = spatial_exp_jacobian(u[3:6]-base[3:6])
        dtheta = np.linalg.solve(A, spatial_direction)
        g, J, _, _ = owner.constraints.evaluate(u, base, rotations, targets=[target])
        mu = actual["multipliers"]; moment = actual["spatial_reactions"][3:6]
        invariant(g, np.zeros(6))
        invariant(actual["reactions"][3:6], A.T @ moment)
        invariant(J[3:, 3:6] @ dtheta, target.T @ spatial_direction)
        prescribed_work = -mu[3:] @ (target.T @ spatial_direction)
        invariant(prescribed_work, moment @ spatial_direction)
        invariant(prescribed_work, actual["reactions"][3:6] @ dtheta)
        assert abs(prescribed_work) > 1e-5  # Nontrivial work, not the old zero-load case.
        errors = []
        for h in FIXTURE["derivative_steps"]:
            plus = Rotation.from_rotvec(h*spatial_direction).as_matrix() @ target
            minus = Rotation.from_rotvec(-h*spatial_direction).as_matrix() @ target
            gp = owner.constraints.evaluate(u, base, rotations, targets=[plus])[0]
            gm = owner.constraints.evaluate(u, base, rotations, targets=[minus])[0]
            derivative = (gp-gm)/(2*h)
            error = abs(mu @ derivative-prescribed_work)/max(1., abs(prescribed_work))
            assert error <= 1e-7; errors.append(float(error))
        results.append(dict(target=target, chart_moment=actual["reactions"][3:6],
                            spatial_moment=moment, work=prescribed_work, derivative_errors=errors))
    record("target-work", results)


@pytest.mark.parametrize("kind", ["global", "renumber", "reverse"])
@pytest.mark.parametrize("counts", [(1, 0), (2, 2), (3, 3)])
def test_constraint_rows_pose_and_work_transport(kind, counts):
    selected, relative = counts
    a = np.array(FIXTURE["relative_offset"])
    C = Rotation.from_rotvec(FIXTURE["relative_frame_vector"]).as_matrix()
    X = np.array([np.zeros(3), a/2, a]); frames = np.array([np.eye(3), np.eye(3), C])
    B = Rotation.from_rotvec([.1, .2, -.1]).as_matrix()[:, :selected]
    Br = Rotation.from_rotvec([-.2, .1, .05]).as_matrix()[:, :relative]
    target = Rotation.from_rotvec(FIXTURE["orientation_vectors"][0]).as_matrix()
    def construct(ids, x, R, root, tip, target):
        return ConstraintSet(tuple(ids), x, R,
            orientations=[dict(node=root, axes=B, target=target)],
            poses=[dict(master=root, slave=tip, offset=a, frame=C, axes=Br)])
    c = construct((1, 2, 3), X, frames, 1, 3, target)
    q = np.linspace(-.01, .02, 18); committed = np.linspace(.001, -.002, 18)
    Q = Rotation.from_rotvec(np.array([[.02, -.01, .01], [.01, .02, -.01], [-.01, .01, .02]])).as_matrix()
    g, J, H, _ = c.evaluate(q, committed, Q)
    W = Rotation.from_rotvec(FIXTURE["common_rotation"]).as_matrix() if kind == "global" else np.eye(3)
    ids = (7, 2, 9) if kind == "renumber" else (1, 2, 3)
    permutation = [ids.index(n) for n in sorted(ids)] if kind != "reverse" else [2, 1, 0]
    ordered = [ids[i] for i in permutation]
    D = np.zeros((18, 18))
    for j, i in enumerate(permutation): D[6*j:6*j+6, 6*i:6*i+6] = block_diag(W, W)
    moved = construct(ordered, (X @ W.T)[permutation], (W @ frames)[permutation], ids[0], ids[2], W @ target)
    gm, Jm, Hm, _ = moved.evaluate(D @ q, D @ committed, (W @ Q @ W.T)[permutation])
    S = block_diag(np.eye(selected), W, np.eye(relative))
    invariant(gm, S @ g); invariant(Jm @ D, S @ J)
    transformed = np.einsum("ia,aef,je,kf->ijk", S, H, D, D)
    invariant(Hm, transformed)
    mu = np.linspace(.1, .3, len(g)); moved_mu = S @ mu
    invariant(-Jm.T @ moved_mu, D @ (-J.T @ mu))
    direction = np.linspace(-.3, .2, 18)
    invariant((-Jm.T @ moved_mu) @ (D @ direction), (-J.T @ mu) @ direction)
    for j, i in enumerate(permutation):
        base_pose = Rotation.from_rotvec((q-committed)[6*i+3:6*i+6]).as_matrix() @ Q[i] @ frames[i]
        new_pose = Rotation.from_rotvec((D @ (q-committed))[6*j+3:6*j+6]).as_matrix() @ (W @ Q[i] @ W.T) @ W @ frames[i]
        invariant(new_pose, W @ base_pose)
    record(f"transport-{kind}-{selected}-{relative}", dict(values=gm, jacobian=Jm,
           hessian=Hm, work=float((-Jm.T @ moved_mu) @ (D @ direction))))


def test_nonzero_multiplier_actual_newton_curvature_and_nullspace():
    owner = make(); owner.solve(np.zeros(18), targets=[Rotation.from_rotvec(FIXTURE["orientation_vectors"][0]).as_matrix()])
    before = snapshot(owner); u = owner.store.native_rotation_store.committed_full_displacement.copy()
    u += np.linspace(-.002, .003, 18)
    target = owner.constraints.target_frames(); mu = np.linspace(.1, .3, 6)
    force = np.zeros(18); lines = np.zeros((1, 3)); couples = lines.copy()
    direction = np.linspace(-.3, .2, 24)
    try:
        residual, tangent, _, J, _ = owner._system(u, mu, force, lines, couples, 0., target)
        _, _, H, _ = owner._constraint_trial(u, 0., target)
        curvature = np.einsum("i,ijk->jk", mu, H)
        assert np.linalg.norm(curvature) > 1e-5
        errors = []
        for h in FIXTURE["derivative_steps"]:
            plus = owner._system(u+h*direction[:18], mu+h*direction[18:], force, lines, couples, 0., target)[0]
            minus = owner._system(u-h*direction[:18], mu-h*direction[18:], force, lines, couples, 0., target)[0]
            fd = (plus-minus)/(2*h); expected = tangent @ direction
            error = np.linalg.norm(fd-expected)/max(1., np.linalg.norm(fd), np.linalg.norm(expected))
            assert error <= 1e-7; errors.append(float(error))
        # Independent constrained elimination, including the nonzero g RHS.
        rhs = -residual; N = null_space(J)
        particular = J.T @ np.linalg.solve(J @ J.T, rhs[18:])
        K = tangent[:18, :18]
        du = particular+N @ np.linalg.solve(N.T @ K @ N, N.T @ (rhs[:18]-K @ particular))
        dm = np.linalg.solve(J @ J.T, J @ (rhs[:18]-K @ du))
        full = np.linalg.solve(tangent, rhs)
        invariant(full, np.r_[du, dm]); invariant(tangent @ full, rhs)
        # Omission mutation must be observable, not a zero-curvature test.
        bad = tangent.copy(); bad[:18, :18] -= curvature
        assert np.linalg.norm((bad-tangent) @ direction) > 1e-6
        record("newton-curvature", dict(residual=residual, tangent=tangent, increment=full, errors=errors))
    finally:
        if owner.store.has_active_trial: owner.store.discard_trial(owner.store.active_trial_token())
    assert snapshot(owner) == before and not owner.store.has_active_trial


@pytest.mark.parametrize("kind", ["global", "renumber", "reverse"])
def test_actual_owner_selected_pose_operator_transport(kind):
    # Actual element connectivity (not merely constraint storage) is reversed.
    ref = reference(); C = Rotation.from_rotvec(FIXTURE["relative_frame_vector"]).as_matrix()
    R = np.array([np.eye(3), np.eye(3), C])
    B = Rotation.from_rotvec([.1, .2, -.1]).as_matrix()[:, :2]
    target = Rotation.from_rotvec(FIXTURE["orientation_vectors"][0]).as_matrix()
    def owner_for(ids, W, reverse):
        order = sorted(ids); permutation = [ids.index(n) for n in order]
        nodes = ids[::-1] if reverse else ids
        x = ref.coordinates @ W.T; triads = W @ ref.nodal_triads
        if reverse: x = x[::-1]; triads = triads[::-1] @ np.diag([-1., 1., -1.])
        element = ElasticElement(1, tuple(nodes), Reference(x, triads), section())
        c = ConstraintSet(tuple(order), (ref.coordinates @ W.T)[permutation], (W @ R)[permutation],
            orientations=[dict(node=ids[0], axes=B, target=W @ target)],
            poses=[dict(master=ids[0], slave=ids[2], offset=FIXTURE["relative_offset"], frame=C, axes=B)])
        return ConstrainedAnalysis((element,), c), permutation
    base, _ = owner_for((1, 2, 3), np.eye(3), False)
    W = Rotation.from_rotvec(FIXTURE["common_rotation"]).as_matrix() if kind == "global" else np.eye(3)
    ids = (7, 2, 9) if kind == "renumber" else (1, 2, 3)
    moved, permutation = owner_for(ids, W, kind == "reverse")
    D = np.zeros((18, 18))
    for j, i in enumerate(permutation): D[6*j:6*j+6, 6*i:6*i+6] = block_diag(W, W)
    S = block_diag(np.eye(2), W, np.eye(2)); P = block_diag(D, S)
    q = np.linspace(-.002, .003, 18); mu = np.linspace(.1, .3, 7)
    zero = np.zeros(18); lines = np.zeros((1, 3))
    before = [snapshot(o) for o in (base, moved)]
    try:
        r, K, _, _, _ = base._system(q, mu, zero, lines, lines, 0., base.constraints.target_frames())
        rm, Km, _, _, _ = moved._system(D @ q, S @ mu, zero, lines, lines, 0., moved.constraints.target_frames())
        invariant(rm, P @ r); invariant(Km, P @ K @ P.T)
        g, J, H, _ = base._constraint_trial(q, 0., base.constraints.target_frames())
        gm, Jm, Hm, _ = moved._constraint_trial(D @ q, 0., moved.constraints.target_frames())
        invariant(gm, S @ g); invariant(Jm @ D, S @ J)
        invariant(Hm, np.einsum("ia,aef,je,kf->ijk", S, H, D, D))
        record("actual-owner-"+kind, dict(residual=rm, tangent=Km, values=gm, jacobian=Jm))
    finally:
        for owner in (base, moved):
            if owner.store.has_active_trial: owner.store.discard_trial(owner.store.active_trial_token())
    assert [snapshot(o) for o in (base, moved)] == before
