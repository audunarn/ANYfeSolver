"""G3a native graph development fixtures; not formal qualification."""
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver._ge_beam3_g1_analysis import ElasticAnalysis
from anyfesolver._ge_beam3_g2_analysis import ConstrainedAnalysis
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_elastic import ElasticSection, canonical
from anyfesolver._ge_beam3_g3_analysis import NativeGraphAnalysis, rigid_basis
from anyfesolver._ge_beam3_g3_constraints import GraphConstraintSet, compile_affine
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT/"docs/reference_cases/ge_beam3_g3_graph_fixtures_v1.json").read_text())
GRAPHS = {g["id"]: g for g in FIXTURE["native_graphs"]}


def record(name, body):
    import os
    directory = os.environ.get("G3_CORRECTION_RECORDS")
    if directory:
        with (Path(directory)/(name+".json")).open("xb") as stream:
            stream.write(canonical(body))


def verify_prefix_and_continue(owner, checkpoint, force):
    assert not owner.store.has_active_trial and owner.checkpoint() == checkpoint
    restored = NativeGraphAnalysis.resume(checkpoint, sha256(checkpoint).hexdigest())
    assert restored.checkpoint() == checkpoint
    assert canonical(owner.solve(force)) == canonical(restored.solve(force))
    assert owner.checkpoint() == restored.checkpoint()


def invariant(a, b):
    a, b = np.asarray(a), np.asarray(b)
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b))
    assert np.isfinite(error) and error <= 1e-11
    return float(error)


def affine(d, masters=(), offset=0., rate=0.):
    return dict(dependent=d, masters=list(masters), offset=offset, rate=rate)


def parts(name="N_BRANCH3", *, shuffle=False, W=None, shift=None, renumber=False, reverse=False):
    g = GRAPHS[name]
    W = np.eye(3) if W is None else W
    shift = np.zeros(3) if shift is None else shift
    ids = sorted(n for n, _ in g["nodes"])
    mapping = {n: (1001+7*(len(ids)-1-i) if renumber else n) for i, n in enumerate(ids)}
    points = {mapping[n]: W @ np.array(x)+shift for n, x in g["nodes"]}
    # Preserve the exact-midpoint admission after binary64 coordinate transport.
    for e in g["elements"]:
        a, m, b = [mapping[n] for n in e["nodes"]]; points[m] = (points[a]+points[b])/2
    elements = []
    for d in g["elements"]:
        nodes = tuple(mapping[n] for n in d["nodes"]); x = np.array([points[n] for n in nodes])
        tangent = x[2]-x[0]; tangent /= np.linalg.norm(tangent)
        v = W @ np.array(d["orientation"]); v -= tangent*(tangent @ v); v /= np.linalg.norm(v)
        frame = np.column_stack([tangent, v, np.cross(tangent, v)]) @ Rotation.from_rotvec([d["roll"], 0., 0.]).as_matrix()
        frames = np.tile(frame, (3, 1, 1))
        if reverse:
            nodes = nodes[::-1]; x = x[::-1]; frames = frames[::-1] @ np.diag([-1., 1., -1.])
        material = ElasticSection.isotropic(**FIXTURE["materials"]["native_isotropic"])
        elements.append(ElasticElement(d["id"], nodes, Reference(x, frames), material))
    ordered = sorted(points)
    fixed = sorted(mapping[n] for n in g["fixed_nodes"])
    rows = [affine(6*ordered.index(n)+j) for n in fixed for j in range(3)]
    orientations = [dict(node=n, axes=np.eye(3), target=W) for n in fixed]
    c = GraphConstraintSet(tuple(ordered), [points[n] for n in ordered],
                           np.tile(W, (len(ordered), 1, 1)), affine=rows, orientations=orientations)
    return tuple(elements[::-1] if shuffle else elements), c, mapping


def make(name="N_BRANCH3", **kwargs):
    elements, constraints, _ = parts(name, **kwargs)
    return NativeGraphAnalysis(elements, constraints)


def loads(owner, name="N_BRANCH3", factor=1., W=None, mapping=None):
    W = np.eye(3) if W is None else W
    mapping = {} if mapping is None else mapping
    f = np.zeros(owner.size); ids = owner.inventory["node_ids"]
    for n in GRAPHS[name]["load_nodes"]:
        i = ids.index(mapping.get(n, n)); start = 6*i
        f[start:start+3] = factor*W @ np.array(FIXTURE["programs"]["force"])
        f[start+3:start+6] = factor*W @ np.array(FIXTURE["programs"]["moment"])
    return f


def snapshot(owner):
    r = owner.store.native_rotation_store
    return canonical(dict(state=owner.store.materialize(), u=r.committed_full_displacement,
                          Q=r.committed_rotation_matrices, generation=owner.store.generation,
                          rotation_generation=r.generation, history=owner._accepted, journal=owner._journal_digest))


def balance(owner, result, f):
    points = np.array([n.coords() for n in owner.model.mesh.nodes.values()])+result["u"].reshape(-1, 6)[:, :3]
    work = f.reshape(-1, 6)+result["spatial_reactions"].reshape(-1, 6)
    invariant(work[:, :3].sum(axis=0), np.zeros(3))
    invariant((work[:, 3:]+np.cross(points, work[:, :3])).sum(axis=0), np.zeros(3))


def test_three_arm_branch_smoke():
    owner = make()
    assert owner.size == 42 and len(owner.elements) == 3
    assert owner.inventory["components"] == [sorted(n for n, _ in GRAPHS["N_BRANCH3"]["nodes"])]
    assert owner.inventory["cycle_rank"] == 0
    assert dict(owner.inventory["incidence"])[100] == [1, 2, 3]
    before = snapshot(owner)
    f = loads(owner); result = owner.solve(f)
    assert result["residual_norm"] <= 1e-11 and owner.store.generation == 1
    assert not owner.store.has_active_trial and snapshot(owner) != before
    balance(owner, result, f)
    state = owner.store.materialize(); ids = owner.inventory["node_ids"]
    Q = owner.store.native_rotation_store.committed_rotation_matrices
    triads = []
    for e in owner.elements:
        stored = state[e.element_id]["committed_nodal_rotation_matrices"]
        for j, n in enumerate(e.node_ids): np.testing.assert_array_equal(stored[j], Q[ids.index(n)])
        triads.append(e.operator.reference.nodal_triads[0])
    assert not np.array_equal(triads[0], triads[1])
    assert not np.array_equal(triads[1], triads[2])
    recovery = owner.recover()
    assert list(recovery) == [1, 2, 3] and all(len(rows) == 8 for rows in recovery.values())
    accepted = snapshot(owner)
    with pytest.raises(InterruptedError): owner.solve(2*f, cancel=lambda: True)
    assert snapshot(owner) == accepted and not owner.store.has_active_trial


def discard(owner):
    if owner.store.has_active_trial: owner.store.discard_trial(owner.store.active_trial_token())


def assembled_reference(owner):
    z = np.zeros(owner.size); shape = (len(owner.elements), 3)
    try:
        return owner._evaluate(z, z, np.zeros(shape), np.zeros(shape))[:2]
    finally:
        discard(owner)


def complement(R):
    # Independent deterministic leftmost-column RREF of R.T.
    A = R.T.copy(); pivots = []; row = 0
    floor = 64*np.finfo(float).eps*max(A.shape)*max(1., np.linalg.svd(A, compute_uv=False)[0])
    for col in range(A.shape[1]):
        candidates = np.flatnonzero(np.abs(A[row:, col]) > floor)
        if not len(candidates): continue
        p = row+int(candidates[0]); A[[row, p]] = A[[p, row]]; A[row] /= A[row, col]
        for i in range(len(A)):
            if i != row: A[i] -= A[i, col]*A[row]
        pivots.append(col); row += 1
        if row == len(A): break
    assert row == len(A)
    free = [i for i in range(A.shape[1]) if i not in pivots]
    Z = np.zeros((A.shape[1], len(free))); Z[free, :] = np.eye(len(free))
    Z[pivots, :] = -A[:, free]
    invariant(R.T @ Z, np.zeros((R.shape[1], len(free))))
    return Z


@pytest.mark.parametrize("name", list(GRAPHS))
def test_reference_graph_rigid_spaces_and_sorted_assembly(name):
    owner = make(name); g = GRAPHS[name]; inv = owner.inventory
    assert len(inv["components"]) == g["components"] and inv["cycle_rank"] == g["cycle_rank"]
    r, K = assembled_reference(owner)
    R = rigid_basis(owner.constraints, inv["components"]); Z = complement(R)
    invariant(r, np.zeros(owner.size)); invariant(K, K.T)
    invariant(K @ R, np.zeros_like(R))
    eigen = np.linalg.eigvalsh(K)
    floor = 64*np.finfo(float).eps*max(K.shape)*max(1., np.max(abs(eigen)))
    assert np.count_nonzero(abs(eigen) <= floor) == 6*g["components"]
    assert np.min(np.linalg.eigvalsh(Z.T @ K @ Z)) > floor
    np.linalg.cholesky(Z.T @ K @ Z)
    # Independent scatter over unchanged local operators, not graph assembly.
    independent = np.zeros_like(K)
    for e in owner.elements:
        indices = [d for n in e.node_ids for d in owner.model.mesh.dof_manager.get_node_dofs(n)]
        independent[np.ix_(indices, indices)] += owner.store[e.element_id]["response"]["tangent"]
    invariant(K, independent)
    shuffled = make(name, shuffle=True); _, changed = assembled_reference(shuffled)
    np.testing.assert_array_equal(K, changed)
    from scipy.sparse import coo_matrix
    patterns = []
    for graph in (owner, shuffled):
        rows = graph.model.mesh._sparsity_cache["tangent_stiffness"]
        csr = coo_matrix((np.ones(len(rows["rows"])), (rows["rows"], rows["cols"])),
                         shape=(graph.size, graph.size)).tocsr()
        assert csr.has_sorted_indices and csr.has_canonical_format
        patterns.append((csr.indptr, csr.indices))
    for a, b in zip(*patterns): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_native_load_program_recovery_and_checkpoint(name):
    owner = make(name)
    triads = {e.element_id: e.operator.reference.nodal_triads.copy() for e in owner.elements}
    pose_records = []
    for factor in FIXTURE["programs"]["parameters"]:
        f = loads(owner, name, factor); result = owner.solve(f)
        assert result["residual_norm"] <= 1e-11
        balance(owner, result, f)
        assert sum(map(len, owner.recover().values())) == 8*len(owner.elements)
        Q = owner.store.native_rotation_store.committed_rotation_matrices
        state = owner.store.materialize(); ids = owner.inventory["node_ids"]
        for e in owner.elements:
            np.testing.assert_array_equal(e.operator.reference.nodal_triads, triads[e.element_id])
            for j, n in enumerate(e.node_ids):
                np.testing.assert_array_equal(state[e.element_id]["committed_nodal_rotation_matrices"][j], Q[ids.index(n)])
        pose_records.append(dict(factor=factor, Q=Q.copy(), result=result))
    assert owner.store.generation == 5 and not owner.store.has_active_trial
    data = owner.checkpoint()
    restored = NativeGraphAnalysis.resume(data, sha256(data).hexdigest())
    assert restored.checkpoint() == data
    assert canonical(restored.recover()) == canonical(owner.recover())
    f = loads(owner, name, .5)
    assert canonical(owner.solve(f)) == canonical(restored.solve(f))
    record("poses-"+name, dict(incidence=owner.inventory["incidence"], triads=triads, steps=pose_records,
                               checkpoint_sha256=sha256(data).hexdigest()))


@pytest.mark.parametrize("name", FIXTURE["programs"]["orientation_graphs"])
def test_noncommuting_shared_pose_and_support_work(name):
    owner = make(name); ids = owner.inventory["node_ids"]
    support = GRAPHS[name]["fixed_nodes"][0]; i = ids.index(support); d = 6*i+3
    rows = owner.constraints.descriptor()["orientations"]
    target = owner.constraints.target_frames().copy(); index = next(i for i, r in enumerate(rows) if r["node"] == support)
    direction = np.array([.2, -.3, .1])
    for vector in FIXTURE["programs"]["orientation_vectors"]:
        committed = owner.store.native_rotation_store.committed_full_displacement.copy()
        Q = owner.store.native_rotation_store.committed_rotation_matrices.copy()
        target[index] = Rotation.from_rotvec(vector).as_matrix() @ target[index]
        result = owner.solve(loads(owner, name), targets=target)
        current = owner.store.native_rotation_store.committed_rotation_matrices
        invariant(current[i], target[index])
        moment = result["spatial_reactions"][d:d+3]
        assert abs(moment @ direction) > 1e-5
        for h in FIXTURE["acceptance"]["derivative_steps"]:
            plus = target.copy(); minus = target.copy()
            plus[index] = Rotation.from_rotvec(h*direction).as_matrix() @ target[index]
            minus[index] = Rotation.from_rotvec(-h*direction).as_matrix() @ target[index]
            gp = owner.constraints.evaluate(result["u"], committed, Q, targets=plus)[0]
            gm = owner.constraints.evaluate(result["u"], committed, Q, targets=minus)[0]
            assert abs(result["multipliers"] @ ((gp-gm)/(2*h))-moment @ direction) <= 1e-7
        balance(owner, result, loads(owner, name))


@pytest.mark.parametrize("name", list(GRAPHS))
def test_full_stationary_schur_nonzero_internal_residual_and_load_work(name):
    from anyfesolver._ge_beam3_g1_operator import schur
    owner = make(name); n = owner.size; size = n+24*len(owner.elements)
    H = np.zeros((size, size)); r = np.zeros(size)
    condensed = np.zeros((n, n)); residual = np.zeros(n)
    line = np.array(FIXTURE["programs"]["native_line_force"])
    couple = np.array(FIXTURE["programs"]["native_line_couple"])
    for k, e in enumerate(owner.elements):
        ref = e.operator.reference
        rotations = np.tile(Rotation.from_rotvec([.0001, -.0002, .00015]).as_matrix(), (2, 1, 1))
        p = np.linspace(-.003, .004, 18)
        full = e.operator.evaluate(ref.coordinates, np.zeros((3, 3)), ref.nodal_triads, rotations, p, line=line, couple=couple)
        local_r, local_H = full["residual"], full["jacobian"]
        assert np.linalg.norm(local_r[18:]) > 1e-5
        external = [d for node in e.node_ids for d in owner.model.mesh.dof_manager.get_node_dofs(node)]
        indices = external+list(range(n+24*k, n+24*(k+1)))
        H[np.ix_(indices, indices)] += local_H; r[indices] += local_r
        fr, fk, _, _ = schur(local_r, local_H)
        condensed[np.ix_(external, external)] += fk; residual[external] += fr
    _, J, _, _ = owner._constraint_trial(np.zeros(n), 0., None)
    Jfull = np.pad(J, ((0, 0), (0, size-n)))
    actual = np.linalg.solve(np.block([[H, Jfull.T], [Jfull, np.zeros((len(J), len(J)))]]), -np.r_[r, np.zeros(len(J))])
    reduced = np.linalg.solve(np.block([[condensed, J.T], [J, np.zeros((len(J), len(J)))]]), -np.r_[residual, np.zeros(len(J))])
    invariant(actual[:n], reduced[:n]); invariant(actual[size:], reduced[n:])
    # Schur internal reconstruction includes its nonzero affine correction.
    internal = np.linalg.solve(H[n:, n:], -r[n:]-H[n:, :n] @ reduced[:n])
    invariant(actual[n:size], internal)
    lines = np.tile(line, (len(owner.elements), 1)); couples = np.tile(couple, (len(owner.elements), 1))
    result = owner.solve(np.zeros(n), lines=lines, couples=couples)
    length = sum(np.linalg.norm(e.operator.reference.coordinates[2]-e.operator.reference.coordinates[0]) for e in owner.elements)
    invariant(result["spatial_reactions"].reshape(-1, 6)[:, :3].sum(axis=0), -length*line)
    data = owner.checkpoint()
    assert NativeGraphAnalysis.resume(data, sha256(data).hexdigest()).checkpoint() == data
    record("schur-"+name, dict(full=actual, reduced=reduced, internal=internal, reactions=result["spatial_reactions"]))


@pytest.mark.parametrize("route", [ElasticAnalysis.solve, ConstrainedAnalysis.solve])
@pytest.mark.parametrize("accepted", [False, True])
def test_base_dispatch_rejected_before_mutation(route, accepted):
    owner = make()
    if accepted: owner.solve(loads(owner))
    before = snapshot(owner)
    with patch.object(owner, "_evaluate", side_effect=AssertionError("must not evaluate")):
        with pytest.raises(ValueError): route(owner, loads(owner))
    assert snapshot(owner) == before and not owner.store.has_active_trial


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("accepted", [False, True])
def test_last_element_prepare_failure_retains_complete_graph(accepted, name):
    owner = make(name); f = loads(owner, name)
    if accepted: owner.solve(f)
    before = snapshot(owner); checkpoint = owner.checkpoint(); victim = owner.elements[-1]
    original = victim._validate; commit = owner.store.commit
    preparing = []; triggered = []
    def publication(*args, **kwargs):
        preparing.append(True)
        try: return commit(*args, **kwargs)
        finally: preparing.pop()
    def validate(*args, **kwargs):
        original(*args, **kwargs)
        if preparing:
            triggered.append(True); owner.model.constraint_equations.append("mutated in final prepare")
    with patch.object(owner.store, "commit", side_effect=publication), patch.object(victim, "_validate", side_effect=validate):
        with pytest.raises(ValueError): owner.solve(.5*f)
    assert triggered and snapshot(owner) == before and not owner.store.has_active_trial
    # Remove only the deliberately injected unregistered authority marker.
    assert owner.model.constraint_equations == ["mutated in final prepare"]
    owner.model.constraint_equations.pop()
    verify_prefix_and_continue(owner, checkpoint, .25*f)
    record("prepare-"+name+"-"+str(accepted), dict(prefix_sha256=sha256(before).hexdigest(), replay=True))


@pytest.mark.parametrize("kind", ["cache", "free", "constraint", "node", "section", "load", "order", "callback", "activity"])
def test_graph_mutation_rejects_before_work(kind):
    owner = make(); before = snapshot(owner)
    if kind == "cache": owner.model.mesh._sparsity_cache["tangent_stiffness"]["rows"][0] = 1
    elif kind == "free": owner._free = owner.free.copy()
    elif kind == "constraint": object.__setattr__(owner.constraints, "_raw", owner.constraints._raw.replace(b'"offset":0.0', b'"offset":0.1', 1))
    elif kind == "node": owner.model.mesh.nodes[100].x += .01
    elif kind == "section": owner.model.materials[owner.elements[0].material_name] = object()
    elif kind == "load": owner.elements[0]._load = (owner.store, np.ones(3), np.zeros(3))
    elif kind == "order": owner.model.mesh.elements = dict(reversed(list(owner.model.mesh.elements.items())))
    elif kind == "callback": owner.elements[-1]._owner_guard = lambda: None
    else: owner.model.mesh.element_activity = object()
    with patch.object(owner, "_evaluate", side_effect=AssertionError("must reject before work")):
        with pytest.raises(ValueError): owner.solve(loads(owner))
    assert snapshot(owner) == before and not owner.store.has_active_trial


def test_reference_multi_rhs_is_ephemeral_and_epoch_isolated():
    owner = make("N_BRACED5"); other = make("N_BRACED5")
    f = loads(owner, "N_BRACED5"); before = snapshot(owner)
    together = owner.reference_rhs(np.column_stack([f, -.5*f]))
    one = owner.reference_rhs(f[:, None]); two = owner.reference_rhs((-.5*f)[:, None])
    invariant(together, np.column_stack([one[:, 0], two[:, 0]]))
    assert snapshot(owner) == before == snapshot(other)
    invariant(together, other.reference_rhs(np.column_stack([f, -.5*f])))
    owner.solve(f)
    with pytest.raises(ValueError): owner.reference_rhs(f[:, None])
    assert snapshot(other) == before
    invariant(together, other.reference_rhs(np.column_stack([f, -.5*f])))


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("kind", ["global", "renumber", "reverse"])
def test_loaded_graph_covariance(kind, name):
    a = make(name)
    W = Rotation.from_rotvec([2.9, -1.2, .7]).as_matrix() if kind == "global" else np.eye(3)
    shift = np.array([.7, -1.2, .9]) if kind == "global" else np.zeros(3)
    elements, constraints, mapping = parts(name, W=W, shift=shift, renumber=kind == "renumber", reverse=kind == "reverse")
    b = NativeGraphAnalysis(elements, constraints)
    x = a.solve(loads(a, name)); y = b.solve(loads(b, name, W=W, mapping=mapping))
    ai = a.inventory["node_ids"]; bi = b.inventory["node_ids"]
    for n in ai:
        i, j = ai.index(n), bi.index(mapping[n])
        invariant(y["u"][6*j:6*j+3], W @ x["u"][6*i:6*i+3])
        invariant(y["spatial_reactions"][6*j:6*j+3], W @ x["spatial_reactions"][6*i:6*i+3])
        invariant(y["spatial_reactions"][6*j+3:6*j+6], W @ x["spatial_reactions"][6*i+3:6*i+6])
        invariant(b.store.native_rotation_store.committed_rotation_matrices[j],
                  W @ a.store.native_rotation_store.committed_rotation_matrices[i] @ W.T)
    balance(b, y, loads(b, name, W=W, mapping=mapping))
    record("covariance-"+name+"-"+kind, dict(original=x, transported=y, mapping=mapping))


@pytest.mark.parametrize("kind", ["duplicate_element", "unknown", "orphan", "unsupported", "midpoint", "subclass", "element_bound", "owned"])
def test_graph_admission_before_state_work(kind):
    elements, c, _ = parts("N_DISJOINT2")
    if kind == "duplicate_element": elements = (elements[0], elements[0])
    elif kind == "unknown":
        d = c.descriptor(); d["node_ids"][-1] = 999
        c = GraphConstraintSet(tuple(d["node_ids"]), d["positions"], d["frames"], affine=d["affine"], orientations=d["orientations"])
    elif kind == "orphan":
        d = c.descriptor()
        c = GraphConstraintSet(tuple(d["node_ids"]+[999]), d["positions"]+[[3., 3., 3.]], d["frames"]+[np.eye(3).tolist()], affine=d["affine"], orientations=d["orientations"])
    elif kind == "unsupported":
        d = c.descriptor(); c = GraphConstraintSet(tuple(d["node_ids"]), d["positions"], d["frames"],
                                                  affine=d["affine"][:3], orientations=d["orientations"][:1])
    elif kind == "midpoint":
        e = elements[0]; x = e.operator.reference.coordinates.copy(); x[1, 0] += .01
        elements = (ElasticElement(e.element_id, e.node_ids, Reference(x, e.operator.reference.nodal_triads), e.section), elements[1])
    elif kind == "subclass":
        class Unexpected(ElasticElement): pass
        e = elements[0]; elements = (Unexpected(e.element_id, e.node_ids, e.operator.reference, e.section), elements[1])
    elif kind == "element_bound": elements = elements*5
    else: NativeGraphAnalysis(elements, c)
    with patch.object(ElasticElement, "init_model_bound_nonlinear_state", side_effect=AssertionError("must reject before state")):
        with pytest.raises(ValueError): NativeGraphAnalysis(elements, c)


def test_constraint_bounds_and_bad_native_ties():
    with pytest.raises(ValueError): GraphConstraintSet(tuple(range(1, 34)), np.zeros((33, 3)), np.tile(np.eye(3), (33, 1, 1)))
    with pytest.raises(ValueError): GraphConstraintSet((1, 1, 2), np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)))
    for rows in ([affine(0), affine(0)], [affine(0, [(6, 1)]), affine(6, [(0, 1)])], [affine(3)], [affine(0, [(3, 1)])]):
        with pytest.raises(ValueError): compile_affine(54, rows)


@pytest.mark.parametrize("field", ["graph", "state", "history", "constraints", "schema", "hash", "duplicate", "nonfinite", "count"])
def test_strict_graph_restart_mutation(field):
    owner = make(); data = owner.checkpoint(); body = json.loads(data)
    if field == "graph": body["graph"]["policy"] = "MIXED"
    elif field == "state": body["state"]["1"]["response"]["resultants"][0] += .1
    elif field == "history": body["history"] = [{}]*129
    elif field == "constraints": body["constraints"]["orientations"][0]["target"][0][0] = 2.
    elif field == "schema": body["schema"] = "GE_BEAM3_G2_ELASTIC_RESTART_V1"
    elif field == "count": body["definition"]["elements"] *= 4
    mutated = canonical(body)
    if field == "duplicate": mutated = mutated.replace(b'{"constraints":', b'{"schema":"foreign","constraints":', 1)
    if field == "nonfinite": mutated = mutated.replace(b'"offset":0.0', b'"offset":NaN', 1)
    digest = "0"*64 if field == "hash" else sha256(mutated).hexdigest()
    with pytest.raises((ValueError, TypeError)): NativeGraphAnalysis.resume(mutated, digest)


def test_actual_graph_kkt_directional_operator():
    owner = make("N_RING4")
    owner.solve(loads(owner, "N_RING4"))
    u = owner.store.native_rotation_store.committed_full_displacement.copy()
    u += np.linspace(-.0001, .0001, owner.size)
    target = owner.constraints.target_frames()
    mu = np.linspace(.1, .3, len(owner._constraint_trial(u, 0., target)[0]))
    shape = (len(owner.elements), 3); f = loads(owner, "N_RING4")
    lines = np.tile(FIXTURE["programs"]["native_line_force"], (len(owner.elements), 1))
    couples = np.tile(FIXTURE["programs"]["native_line_couple"], (len(owner.elements), 1))
    d = np.linspace(-.2, .3, owner.size+len(mu))
    before = snapshot(owner)
    try:
        r, K, _, _, _ = owner._system(u, mu, f, lines, couples, 0., target)
        for h in FIXTURE["acceptance"]["derivative_steps"]:
            plus = owner._system(u+h*d[:owner.size], mu+h*d[owner.size:], f, lines, couples, 0., target)[0]
            minus = owner._system(u-h*d[:owner.size], mu-h*d[owner.size:], f, lines, couples, 0., target)[0]
            error = np.linalg.norm((plus-minus)/(2*h)-K @ d)/max(1., np.linalg.norm(K @ d))
            assert error <= 1e-7
    finally: discard(owner)
    assert snapshot(owner) == before


def test_native_remote_pose_matches_shared_three_arm_junction():
    shared = make(); original, c, _ = parts()
    body = c.descriptor(); points = dict(zip(body["node_ids"], body["positions"])); del points[100]
    new = []
    for e in original:
        hub = 1000*e.element_id; points[hub] = [0., 0., 0.]
        new.append(ElasticElement(e.element_id, (hub, *e.node_ids[1:]), e.operator.reference, e.section))
    ids = sorted(points); fixed = GRAPHS["N_BRANCH3"]["fixed_nodes"]
    remote = GraphConstraintSet(tuple(ids), [points[n] for n in ids], np.tile(np.eye(3), (len(ids), 1, 1)),
        affine=[affine(6*ids.index(n)+j) for n in fixed for j in range(3)],
        orientations=[dict(node=n, axes=np.eye(3), target=np.eye(3)) for n in fixed],
        poses=[dict(master=1000, slave=n, offset=[0., 0., 0.], frame=np.eye(3), axes=np.eye(3)) for n in (2000, 3000)])
    owner = NativeGraphAnalysis(tuple(new), remote)
    assert len(owner.inventory["components"]) == 3
    target = np.tile(np.eye(3), (3, 1, 1)); program = FIXTURE["programs"]
    for factor in (1., .25):
        f = loads(shared, factor=factor); rhs = np.zeros(owner.size)
        start = 6*ids.index(1000); rhs[start:start+6] = f[:6]
        target[0] = Rotation.from_rotvec(program["orientation_vectors"][0]).as_matrix() @ target[0]
        x = shared.solve(f, targets=target); y = owner.solve(rhs, targets=target)
        for n in shared.inventory["node_ids"]:
            i = shared.inventory["node_ids"].index(n)
            for replacement in ((1000, 2000, 3000) if n == 100 else (n,)):
                j = ids.index(replacement)
                invariant(y["u"][6*j:6*j+3], x["u"][6*i:6*i+3])
                invariant(owner.store.native_rotation_store.committed_rotation_matrices[j],
                          shared.store.native_rotation_store.committed_rotation_matrices[i])
        for n in fixed:
            i, j = shared.inventory["node_ids"].index(n), ids.index(n)
            invariant(y["spatial_reactions"][6*j:6*j+6], x["spatial_reactions"][6*i:6*i+6])
        balance(owner, y, rhs)
    data = owner.checkpoint()
    assert NativeGraphAnalysis.resume(data, sha256(data).hexdigest()).checkpoint() == data


@pytest.mark.parametrize("kind", ["store", "diagnostic", "final_cancel"])
def test_publication_guard_and_fallible_diagnostics_are_atomic(kind):
    owner = make(); owner.solve(loads(owner)); before = snapshot(owner)
    if kind == "store":
        other = make(); other.solve(loads(other))
        owner.store = other.store
        with pytest.raises(ValueError, match="shared state"): owner.solve(loads(owner))
        assert snapshot(other) == before
        owner.store = owner._owned_store
    elif kind == "diagnostic":
        def fail(stage):
            if stage == "acceptance prepare": raise OSError("broken progress destination")
        with patch("anyfesolver._ge_beam3_g3_analysis.progress", side_effect=fail):
            with pytest.raises(OSError): owner.solve(.5*loads(owner))
    else:
        cancel_at_final_acceptance(owner, loads(owner))
    assert snapshot(owner) == before and not owner.store.has_active_trial


def test_exclusive_graph_publication_and_deterministic_development_packet(tmp_path):
    import os
    a = make(); b = make(shuffle=True)
    result_a = a.solve(loads(a)); result_b = b.solve(loads(b))
    assert canonical(result_a) == canonical(result_b)
    data = a.checkpoint(); assert data == b.checkpoint()
    out = tmp_path/"checkpoint.json"; a.publish(out)
    assert out.read_bytes() == data
    with pytest.raises(ValueError): a.publish(out)
    failed = tmp_path/"failed.json"
    with patch("anyfesolver._ge_beam3_g1_analysis.os.link", side_effect=OSError("publication fault")):
        with pytest.raises(OSError): a.publish(failed)
    assert not failed.exists() and out.read_bytes() == data
    assert not list(tmp_path.glob("*.pending"))
    packet = canonical(dict(schema="GE_BEAM3_G3A_DEVELOPMENT_PACKET_V1", qualification=False,
                            inventory=a.inventory, result=result_a, recovery=a.recover(),
                            checkpoint_sha256=sha256(data).hexdigest()))
    output = Path(os.environ.get("G3_DIAGNOSTIC_PAYLOAD", str(tmp_path/"development.json")))
    with output.open("xb") as stream: stream.write(packet)


@pytest.mark.parametrize("accepted", [False, True])
def test_final_callback_constraint_mutation_is_atomic(accepted):
    owner = make()
    if accepted: owner.solve(np.zeros(owner.size))
    before = snapshot(owner); calls = []
    def callback():
        calls.append(True)
        if len(calls) == 2: owner.model.constraint_equations.append("late mutation")
        return False
    with pytest.raises(ValueError): owner.solve(np.zeros(owner.size), cancel=callback)
    assert len(calls) == 2 and snapshot(owner) == before and not owner.store.has_active_trial


def cancel_at_final_acceptance(owner, force):
    import sys
    original = owner._system; systems = []; calls = []; triggered = []
    def observe(u, multipliers, *args, **kwargs):
        result = original(u, multipliers, *args, **kwargs)
        systems.append((u.copy(), multipliers.copy(), float(np.linalg.norm(result[0]))))
        return result
    def cancel():
        calls.append(True)
        if (len(systems) < 2 or any(row[2] > 1e-11 for row in systems[-2:]) or
                not np.array_equal(systems[-2][0], systems[-1][0]) or
                not np.array_equal(systems[-2][1], systems[-1][1])):
            return False
        # The accepted line search is re-evaluated at the next main iteration.
        # Prove this callback is solve -> check -> cancel, not system -> check.
        caller = sys._getframe(2).f_code
        assert caller.co_name == "solve" and caller.co_filename == original.__func__.__code__.co_filename
        assert owner.store.has_active_trial
        triggered.append(True)
        return True
    with patch.object(owner, "_system", side_effect=observe):
        with pytest.raises(InterruptedError): owner.solve(force, cancel=cancel)
    assert triggered == [True] and len(systems) >= 3
    # Multipliers reset at solve entry: a loaded accepted prefix is not an
    # initially converged KKT system even when force is unchanged.
    assert systems[0][2] > 1e-11
    assert all(row[2] <= 1e-11 for row in systems[-2:])
    return dict(callback_count=len(calls), system_count=len(systems),
                initial_residual_norm=systems[0][2], final_residual_norms=[row[2] for row in systems[-2:]],
                converged_state_repeated=True, final_acceptance_callback=True)


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("phase", ["entry", "prepublication"])
def test_per_graph_cancel_preserves_accepted_prefix(name, phase):
    owner = make(name); f = loads(owner, name); owner.solve(f)
    before = snapshot(owner); checkpoint = owner.checkpoint()
    if phase == "entry":
        calls = []
        def cancel():
            calls.append(True)
            assert not owner.store.has_active_trial
            return True
        with pytest.raises(InterruptedError): owner.solve(f, cancel=cancel)
        assert calls == [True]
        observation = dict(callback_count=1, system_count=0)
    else:
        observation = cancel_at_final_acceptance(owner, f)
    assert snapshot(owner) == before and not owner.store.has_active_trial
    verify_prefix_and_continue(owner, checkpoint, .5*f)
    record("cancel-"+name+"-"+phase, dict(prefix_sha256=sha256(before).hexdigest(), replay=True,
                                       observation=observation))


@pytest.mark.parametrize("kind", ["stale", "foreign"])
@pytest.mark.parametrize("operation", ["commit", "discard", "set_state"])
def test_braced_graph_issued_token_isolation(kind, operation):
    from anyfesolver.nonlinear_state import StateTransactionError
    owner = make("N_BRACED5"); f = loads(owner, "N_BRACED5"); owner.solve(f)
    other = make("N_BRACED5")
    before, foreign_before = snapshot(owner), snapshot(other); checkpoint = owner.checkpoint()
    def issue(target):
        q = target.store.native_rotation_store.committed_full_displacement
        shape = (len(target.elements), 3)
        target._evaluate(q, np.zeros(target.size), np.zeros(shape), np.zeros(shape))
        return target.store.active_trial_token()
    if kind == "stale":
        wrong = issue(owner); owner.store.discard_trial(wrong)
    else: wrong = issue(other)
    live = issue(owner)
    pending = canonical(owner.store.materialize(trial_token=live))
    try:
        with pytest.raises(StateTransactionError):
            if operation == "commit": owner.store.commit(wrong)
            elif operation == "discard": owner.store.discard_trial(wrong)
            else: owner.store.set_trial_state(wrong, owner.elements[0].element_id, owner.store[owner.elements[0].element_id])
        assert owner.store.active_trial_token() is live
        assert canonical(owner.store.materialize(trial_token=live)) == pending
        assert snapshot(owner) == before and snapshot(other) == foreign_before
    finally:
        discard(owner); discard(other)
    assert not other.store.has_active_trial
    verify_prefix_and_continue(owner, checkpoint, .25*f)
    assert snapshot(other) == foreign_before
    record("token-"+kind+"-"+operation, dict(prefix_sha256=sha256(before).hexdigest(),
                                            other_sha256=sha256(foreign_before).hexdigest(), replay=True))


@pytest.mark.parametrize("field", ["policy", "adapter_allowlist", "node_ids", "element_ids",
                                  "components", "cycle_rank", "incidence", "external_dofs",
                                  "internal_coordinates", "missing", "extra", "boolean"])
def test_restart_graph_authority_precedes_native_construction(field):
    owner = make(); data = owner.checkpoint(); body = json.loads(data); before = snapshot(owner)
    graph = body["graph"]
    if field == "policy": graph[field] = "MIXED"
    elif field == "adapter_allowlist": graph[field] = ["unregistered"]
    elif field in ("node_ids", "element_ids"): graph[field] = graph[field][::-1]
    elif field in ("components", "incidence"): graph[field] = []
    elif field == "missing": del graph["policy"]
    elif field == "extra": graph["extra"] = 0
    elif field == "boolean": graph["cycle_rank"] = False
    else: graph[field] += 1
    raw = canonical(body)
    with patch.object(ElasticElement, "__init__", side_effect=AssertionError("native construction forbidden")), \
         patch.object(ElasticElement, "init_model_bound_nonlinear_state", side_effect=AssertionError("native state forbidden")):
        with pytest.raises(ValueError, match="preflight"): NativeGraphAnalysis.resume(raw, sha256(raw).hexdigest())
    assert snapshot(owner) == before and owner.checkpoint() == data and not owner.store.has_active_trial
    record("preflight-"+field, dict(native_work=False, prefix_sha256=sha256(before).hexdigest()))
