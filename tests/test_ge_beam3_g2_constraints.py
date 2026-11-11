"""Frozen G2 S09-S14 development fixtures; independent value/work checks."""
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation
from anyfesolver._ge_beam3_g2_constraints import ConstraintSet, compile_affine
from anyfesolver._ge_beam3_g2_analysis import ConstrainedAnalysis
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_elastic import canonical
from anyfesolver._ge_beam3_mixed_ad import RotationDomainError
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from test_ge_beam3_g1_elastic import reference, section

FIXTURE = json.loads((Path(__file__).resolve().parents[1]/"docs/reference_cases/ge_beam3_g2_fixtures_v1.json").read_text())


def invariant(actual, expected):
    """Frozen dimensionless invariant criterion; never NumPy default rtol."""
    a, b = np.asarray(actual), np.asarray(expected)
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b))
    assert np.isfinite(error) and error <= FIXTURE["tolerances"]["invariant"]
    return float(error)


def affine(d, masters=(), offset=0., rate=0.):
    return dict(dependent=d, masters=list(masters), offset=offset, rate=rate)


def make(*, prescribed=False, ids=(1, 2, 3), transform=None, reverse=False, two=False):
    W = np.eye(3) if transform is None else transform
    ref = reference(); positions = ref.coordinates @ W.T; triads = W @ ref.nodal_triads
    nodes = ids; material = section()
    if reverse:
        nodes = ids[::-1]; positions = positions[::-1]; triads = triads[::-1] @ np.diag([-1., 1., -1.])
    e = ElasticElement(1, tuple(nodes), Reference(positions, triads), material)
    elements = (e,)
    points = dict(zip(ids, ref.coordinates @ W.T))
    if two:
        e2 = ElasticElement(2, (3, 4, 5), reference(1., .37), section(True)); elements += (e2,)
        points.update(dict(zip(e2.node_ids, e2.operator.reference.coordinates)))
    ordered = sorted(points); start = 6*ordered.index(ids[0]); tip = 6*ordered.index(ids[-1])
    rows = [affine(start+j) for j in range(3)]
    if prescribed: rows.append(affine(tip, rate=.001))
    c = ConstraintSet(tuple(ordered), [points[n] for n in ordered], np.tile(W, (len(ordered), 1, 1)),
                      affine=rows, orientations=[dict(node=ids[0], axes=np.eye(3), target=W)])
    return ConstrainedAnalysis(elements, c)


def snapshot(owner):
    r = owner.store.native_rotation_store
    return canonical(dict(state=owner.store.materialize(), u=r.committed_full_displacement,
                          Q=r.committed_rotation_matrices, generation=owner.store.generation,
                          rotation_generation=r.generation, history=owner._accepted, journal=owner._journal_digest))


def test_S09_exact_nested_affine():
    rows = [affine(d, m, b, c) for d, m, b, c in FIXTURE["affine_nested"]]
    T, b, c, order = compile_affine(30, rows)
    assert order == (12, 6)
    free = [i for i in range(30) if i not in (6, 12)]
    assert T[12, free.index(24)] == .25
    assert T[6, free.index(24)] == .125 and T[6, free.index(18)] == .5
    assert b[6] == .001 and c[6] == .002
    z = np.arange(len(free), dtype=float)/128
    for control in (0., 1., 2.):
        q = T @ z+b+c*control
        assert abs(q[6]-.5*q[12]-.5*q[18]-.001-.002*control) < 1e-16
        assert q[12] == .25*q[24]
    with pytest.raises(ValueError): T.setflags(write=True)
    # Exercise the same nested/multimaster rows in the actual native assembler.
    points = np.array(FIXTURE["coordinates"])
    constraints = ConstraintSet((1, 2, 3, 4, 5), points, np.tile(np.eye(3), (5, 1, 1)),
                                affine=[affine(j) for j in range(3)]+rows,
                                orientations=[dict(node=1, axes=np.eye(3), target=np.eye(3))])
    owner = ConstrainedAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),
                                 ElasticElement(2, (3, 4, 5), reference(1.), section())), constraints)
    # Independent axial station equilibrium: four EA/(L/2) axial cells.
    K = np.zeros((5, 5))
    for i in range(4): K[i:i+2, i:i+2] += 200*np.array([[1., -1.], [-1., 1.]])
    A = np.array([[1., 0., 0., 0., 0.], [0., 1., -.5, -.5, 0.], [0., 0., 1., 0., -.25]])
    for load in (0., 1.):
        rhs = np.r_[np.zeros(5), 0., .001+.002*load, 0.]
        independent = np.linalg.solve(np.block([[K, A.T], [A, np.zeros((3, 3))]]), rhs)
        result = owner.solve(np.zeros(30), control=load)
        invariant(result["u"][::6], independent[:5])


@pytest.mark.parametrize("rows", [
    [affine(0), affine(0)], [affine(0, [(6, 1)]), affine(6, [(0, 1)])],
    [affine(3)], [affine(0, [(3, 1)])], [affine(0, [(6, 1), (6, 2)])],
    [affine(True)], [affine(0, rate=float("nan"))], [affine(0, rate=True)]])
def test_S09_invalid_affine(rows):
    with pytest.raises((ValueError, TypeError)): compile_affine(18, rows)


def test_S09_prescribed_translation_and_reaction():
    owner = make(prescribed=True)
    for control in (1., 2.):
        result = owner.solve(np.zeros(18), control=control)
        assert abs(result["u"][12]-.001*control) < 1e-11
        assert abs(result["u"][6]-.0005*control) < 1e-11
        assert abs(result["reactions"][12]-.1*control) < 1e-11
        assert abs(result["reactions"][0]+.1*control) < 1e-11
        assert abs(result["control_work_rate"]-.0001*control) < 1e-11


def derivative_checks(constraints, q, committed, Q, expected_values):
    g, J, H, _ = constraints.evaluate(q, committed, Q)
    invariant(g, expected_values(q))
    d = np.linspace(-.3, .2, len(q)); mu = np.linspace(.1, .3, len(g))
    assert np.linalg.norm(H-H.transpose(0, 2, 1)) < 1e-11
    for h in FIXTURE["derivative_steps"]:
        plus = constraints.evaluate(q+h*d, committed, Q); minus = constraints.evaluate(q-h*d, committed, Q)
        assert np.linalg.norm((plus[0]-minus[0])/(2*h)-J @ d) < 1e-7
        assert np.linalg.norm(((plus[1]-minus[1])/(2*h)).T @ mu-np.einsum("i,ijk,k->j", mu, H, d)) < 1e-7


@pytest.mark.parametrize("count", [1, 2, 3])
def test_S10_partial_orientation_derivatives(count):
    B = Rotation.from_rotvec([.1, .2, -.1]).as_matrix()[:, :count]
    target = Rotation.from_rotvec([.02, -.01, .015]).as_matrix()
    c = ConstraintSet((1, 2, 3), reference().coordinates, np.tile(np.eye(3), (3, 1, 1)),
                      orientations=[dict(node=2, axes=B, target=target)])
    q = np.linspace(-.02, .03, 18); committed = np.zeros(18); Q = np.tile(np.eye(3), (3, 1, 1))
    def values(u): return B.T @ Rotation.from_matrix(target.T @ Rotation.from_rotvec(u[9:12]).as_matrix()).as_rotvec()
    derivative_checks(c, q, committed, Q, values)
    _, J, _, _ = c.evaluate(np.zeros(18), committed, Q, targets=[np.eye(3)])
    assert np.linalg.matrix_rank(J) == count
    invariant(J @ null_space(J), 0.)
    rows = [affine(j) for j in range(3)]
    if count == 1: rows += [affine(13), affine(14)]
    elif count == 2: rows += [affine(13)]
    support = ConstraintSet((1, 2, 3), reference().coordinates, Q, affine=rows,
                            orientations=[dict(node=1, axes=np.eye(3)[:, :count], target=np.eye(3))])
    owner = ConstrainedAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),), support)
    force = np.zeros(18); force[17] = .001; owner.solve(force)
    root = Rotation.from_matrix(owner.store.native_rotation_store.committed_rotation_matrices[0]).as_rotvec()
    invariant(root[:count], 0.)
    if count < 3: assert abs(root[2]) > 1e-8


def test_S11_noncommuting_targets_and_restart(tmp_path):
    owner = make(); target1 = Rotation.from_rotvec(FIXTURE["orientation_vectors"][0]).as_matrix()
    target2 = Rotation.from_rotvec(FIXTURE["orientation_vectors"][1]).as_matrix() @ target1
    owner.solve(np.zeros(18), targets=[target1]); data = owner.checkpoint()
    restored = ConstrainedAnalysis.resume(data, sha256(data).hexdigest())
    a = owner.solve(np.zeros(18), targets=[target2]); b = restored.solve(np.zeros(18), targets=[target2])
    assert canonical(a) == canonical(b) and owner.checkpoint() == restored.checkpoint()
    invariant(owner.store.native_rotation_store.committed_rotation_matrices[0], target2)
    path = tmp_path/"restart.json"; owner.publish(path); before = path.read_bytes()
    with pytest.raises(ValueError): owner.publish(path)
    assert path.read_bytes() == before
    with patch("anyfesolver._ge_beam3_g1_analysis.os.link", side_effect=OSError("publication failure")):
        with pytest.raises(OSError): owner.publish(tmp_path/"failed.json")
    assert not (tmp_path/"failed.json").exists()


def test_S11_chart_cutback_and_common_rotation():
    owner = make(); before = snapshot(owner)
    with pytest.raises(RotationDomainError): owner.solve(np.zeros(18), targets=[Rotation.from_rotvec([.91*np.pi, 0, 0]).as_matrix()])
    assert snapshot(owner) == before and not owner.store.has_active_trial
    W = Rotation.from_rotvec(FIXTURE["common_rotation"]).as_matrix()
    moved = make(transform=W); result = moved.solve(np.zeros(18))
    assert result["residual_norm"] < 1e-11


def test_S12_relative_pose_derivatives_and_work():
    offset = np.array(FIXTURE["relative_offset"]); C = Rotation.from_rotvec(FIXTURE["relative_frame_vector"]).as_matrix()
    X = np.array([np.zeros(3), offset/2, offset]); R = np.array([np.eye(3), np.eye(3), C])
    c = ConstraintSet((1, 2, 3), X, R, poses=[dict(master=1, slave=3, offset=offset, frame=C, axes=np.eye(3))])
    q = np.linspace(-.01, .02, 18); Q = np.tile(np.eye(3), (3, 1, 1))
    def values(u):
        Dm = Rotation.from_rotvec(u[3:6]).as_matrix(); Ds = Rotation.from_rotvec(u[15:18]).as_matrix() @ C
        return np.r_[X[2]+u[12:15]-X[0]-u[:3]-Dm @ offset,
                     Rotation.from_matrix(C.T @ Dm.T @ Ds).as_rotvec()]
    derivative_checks(c, q, np.zeros(18), Q, values)
    _, J, _, _ = c.evaluate(np.zeros(18), np.zeros(18), Q)
    mu = np.array([.2, -.1, .3, .01, .02, -.03]); force = -J.T @ mu
    invariant(force[:3]+force[12:15], 0.)
    invariant(force[3:6]+force[15:18]+np.cross(offset, force[12:15]), 0.)
    for d in null_space(J).T: assert abs(force @ d) < 1e-11
    # Actual elastic element with a rigid end-to-end pose constraint. The tie
    # transmits external work; no replacement stiffness or fabricated mass.
    tie = ConstraintSet((1, 2, 3), reference().coordinates, Q,
                        affine=[affine(j) for j in range(3)],
                        orientations=[dict(node=1, axes=np.eye(3), target=np.eye(3))],
                        poses=[dict(master=1, slave=3, offset=[1., 0., 0.], frame=np.eye(3), axes=np.eye(3))])
    owner = ConstrainedAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),), tie)
    applied = np.zeros(18); applied[12:15] = [.01, -.02, .015]
    actual = owner.solve(applied)
    invariant(actual["u"], 0.)
    invariant(actual["reactions"]+applied, 0.)
    data = owner.checkpoint()
    assert ConstrainedAnalysis.resume(data, sha256(data).hexdigest()).checkpoint() == data


def test_S13_nullspace_and_multiplier_agree():
    owner = make(two=True); total = np.zeros(owner.size); f = total.copy(); f[-6:] = [.01, -.02, .015, .002, .003, -.001]
    try:
        g, J, H, _ = owner._constraint_trial(total, 0., owner.constraints.target_frames())
        r, K, _ = owner._evaluate(total, f, np.zeros((2, 3)), np.zeros((2, 3)))
        N = null_space(J); reduced = -N @ np.linalg.solve(N.T @ K @ N, N.T @ r)
        full = np.linalg.solve(np.block([[K, J.T], [J, np.zeros((len(g), len(g)))]]), -np.r_[r, g])
        invariant(full[:owner.size], reduced)
    finally:
        if owner.store.has_active_trial: owner.store.discard_trial(owner.store.active_trial_token())
    result = owner.solve(f)
    assert result["residual_norm"] < 1e-11
    invariant(result["reactions"][:3]+f[-6:-3], 0.)


def test_S13_redundant_constraints_reject():
    r = dict(node=1, axes=np.eye(3), target=np.eye(3))
    c = ConstraintSet((1, 2, 3), reference().coordinates, np.tile(np.eye(3), (3, 1, 1)), orientations=[r, r])
    with pytest.raises(ValueError, match="rank-deficient"): ConstrainedAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),), c)
    for invalid in (dict(node=1, axes=2*np.eye(3), target=np.eye(3)),
                    dict(node=99, axes=np.eye(3), target=np.eye(3)),
                    dict(node=1, axes=np.eye(3), target=-np.eye(3))):
        with pytest.raises(ValueError):
            ConstraintSet((1, 2, 3), reference().coordinates, np.tile(np.eye(3), (3, 1, 1)), orientations=[invalid])


@pytest.mark.parametrize("kind", ["global", "renumber", "reverse"])
def test_S14_numbering_and_global_covariance(kind):
    base = make(); f = np.zeros(18); f[12:15] = FIXTURE["tip_force"]; f[15:18] = FIXTURE["tip_moment"]
    expected = base.solve(f)
    W = Rotation.from_rotvec(FIXTURE["common_rotation"]).as_matrix() if kind == "global" else np.eye(3)
    ids = (7, 2, 9) if kind == "renumber" else (1, 2, 3)
    owner = make(ids=ids, transform=W, reverse=kind == "reverse")
    order = sorted(ids); force = np.zeros(18); tip = 6*order.index(ids[2])
    force[tip:tip+3] = W @ f[12:15]; force[tip+3:tip+6] = W @ f[15:18]
    actual = owner.solve(force)
    for i, n in enumerate(ids):
        j = order.index(n)
        invariant(actual["u"][6*j:6*j+3], W @ expected["u"][6*i:6*i+3])
        invariant(actual["spatial_reactions"][6*j:6*j+3], W @ expected["spatial_reactions"][6*i:6*i+3])
        invariant(actual["spatial_reactions"][6*j+3:6*j+6], W @ expected["spatial_reactions"][6*i+3:6*i+6])
        invariant(owner.store.native_rotation_store.committed_rotation_matrices[j] @ W,
                  W @ base.store.native_rotation_store.committed_rotation_matrices[i])


@pytest.mark.parametrize("kind", ["callback", "prepare", "cancel", "model_load"])
def test_g2_mutation_and_prepare_failure_preserve_prefix(kind):
    owner = make(two=True); force = np.zeros(owner.size); owner.solve(force); before = snapshot(owner)
    if kind == "prepare":
        victim = owner.elements[-1]; original = victim._validate; commit = owner.store.commit; active = []
        def validation(*args, **kwargs):
            original(*args, **kwargs)
            if active: owner.model.constraint_equations.append("invalid")
        def committing(*args, **kwargs):
            active.append(True)
            try: return commit(*args, **kwargs)
            finally: active.pop()
        with patch.object(victim, "_validate", side_effect=validation), patch.object(owner.store, "commit", side_effect=committing):
            with pytest.raises(ValueError): owner.solve(force)
    else:
        calls = []
        def callback():
            calls.append(True)
            if kind == "cancel": return True
            if len(calls) == 2:
                if kind == "model_load": owner.model.load_cases.append("invalid")
                else:
                    d = owner.constraints.descriptor(); d["affine"][0]["rate"] = .01
                    owner.constraints = ConstraintSet.from_descriptor(d)
            return False
        with pytest.raises((ValueError, InterruptedError)): owner.solve(force, cancel=callback)
    assert snapshot(owner) == before and not owner.store.has_active_trial


@pytest.mark.parametrize("field", ["schema", "constraints", "control", "multipliers", "target", "state", "duplicate", "nonfinite"])
def test_g2_restart_tamper(field):
    owner = make(); owner.solve(np.zeros(18)); raw = owner.checkpoint(); body = json.loads(raw)
    if field == "schema": body["schema"] = "GE_BEAM3_G1_ELASTIC_RESTART_V1"
    elif field == "constraints": body["constraints"]["affine"][0]["offset"] += .01
    elif field == "control": body["history"][0]["control"] = 1.
    elif field == "multipliers": body["history"][0]["multipliers"][0] += .1
    elif field == "target": body["history"][0]["targets"][0] = Rotation.from_rotvec([.01, 0., 0.]).as_matrix().tolist()
    elif field == "state": body["state"]["1"]["response"]["resultants"][0] += .1
    changed = canonical(body)
    if field == "duplicate": changed = b'{"schema":"duplicate",'+raw[1:]
    if field == "nonfinite": changed = raw.replace(b'"control":0.0', b'"control":NaN')
    # A changed no-op control still changes authenticated history; it is not
    # necessarily a mechanics error. Authenticate against the original digest.
    expected = sha256(raw).hexdigest() if field == "control" else sha256(changed).hexdigest()
    with pytest.raises(ValueError): ConstrainedAnalysis.resume(changed, expected)


def test_g2_deterministic_packet(tmp_path):
    owner = make(prescribed=True); a = owner.solve(np.zeros(18), control=1.)
    raw = owner.checkpoint(); restored = ConstrainedAnalysis.resume(raw, sha256(raw).hexdigest())
    b = owner.solve(np.zeros(18), control=2.); replay = restored.solve(np.zeros(18), control=2.)
    assert canonical(b) == canonical(replay) and owner.checkpoint() == restored.checkpoint()
    result = canonical(dict(schema="GE_BEAM3_G2_DEVELOPMENT_PACKET_V1", qualification=False, first=a, second=b,
                            recovery=owner.recover(), checkpoint_sha256=sha256(owner.checkpoint()).hexdigest()))
    target = Path(os.environ.get("G2_DIAGNOSTIC_PAYLOAD", str(tmp_path/"packet.json")))
    with target.open("xb") as stream: stream.write(result)


def test_g2_contract_and_boundary():
    assert FIXTURE["parent"] == "c8408eb509fdd2346f5aec2aa53ed262a3d07e7c"
    assert set(FIXTURE["tests"]) == {n for n in globals() if n.startswith("test_")}
    assert FIXTURE["tolerances"] == dict(invariant=1e-11, directional=1e-7)
