"""Conservative successor map; old shell operators/evidence remain unchanged."""
from hashlib import sha256
from itertools import permutations
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_variational_shell import deformation, rotation_jets, response
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis
from test_ge_beam3_coupled_free_beam import build
from test_ge_beam3_shell_joint_trial import specimen, norm_error


def fit_oracle(ref, u):
    """Independent Kabsch implementation; no producer frame/AD imported here."""
    x = ref+u.reshape(-1, 6)[:, :3]
    rc = ref-ref.mean(axis=0); xc = x-x.mean(axis=0)
    q = Rotation.align_vectors(xc, rc)[0]
    local = np.empty((len(ref), 6))
    local[:, :3] = q.inv().apply(xc)-rc
    local[:, 3:] = (q.inv()*Rotation.from_rotvec(u.reshape(-1, 6)[:, 3:])).as_rotvec()
    return local.ravel(), q.as_matrix()


def moved(ref, u, s):
    v = u.reshape(-1, 6).copy()
    v[:, :3] = (ref+v[:, :3])@s.T+np.array([.3, -.2, .7])-ref
    v[:, 3:] = (Rotation.from_matrix(s)*Rotation.from_rotvec(v[:, 3:])).as_rotvec()
    return v.ravel()


@pytest.mark.parametrize('count', (3, 4))
def test_fit_map_independent_values_and_analytic_derivatives(count, tmp_path):
    ref = np.array([[0., 0., .2], [1.2, .1, .2], [.9, 1.1, .2], [-.1, .9, .2]])[:count]
    u = .035*np.sin(np.arange(6*count)+.4)
    s = Rotation.from_rotvec([1.3, -.7, 2.1]).as_matrix()
    u = moved(ref, u, s)
    value, d, h = deformation(ref, u)
    expected, q = fit_oracle(ref, u)
    actual_q = np.array([[x.value for x in row] for row in rotation_jets(ref, ref+u.reshape(-1, 6)[:, :3])])
    errors = dict(value=norm_error(value, expected), rotation=norm_error(actual_q, q))
    direction = np.cos(np.arange(len(u))+.6); eps = 2e-6
    dp = deformation(ref, u+eps*direction); dm = deformation(ref, u-eps*direction)
    errors['first'] = norm_error((dp[0]-dm[0])/(2*eps), d@direction)
    errors['second'] = norm_error((dp[1]-dm[1])/(2*eps), np.einsum('ijk,k->ij', h, direction))
    assert errors['value'] < 1e-11 and errors['rotation'] < 1e-11, errors
    assert max(errors.values()) < 1e-7, errors
    assert norm_error(h, h.transpose(0, 2, 1)) < 1e-11
    for p in permutations(range(count)):
        ids = np.array([[6*i+j for j in range(6)] for i in p]).ravel()
        reordered = deformation(ref[list(p)], u[ids])
        assert norm_error(reordered[0], value[ids]) < 1e-11
        assert norm_error(reordered[1], d[np.ix_(ids, ids)]) < 1e-11
    (tmp_path/'fit.json').write_bytes(canonical(dict(errors=errors, local=value, differential=d,
        production_qualified=False)))


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_real_shell_energy_work_tangent_and_objectivity(topology, tmp_path):
    a = specimen(topology); ref = a.coordinates
    u = .015*np.sin(np.arange(a.shell_count)+.4)
    s = Rotation.from_rotvec([1.3, -.7, 2.1]).as_matrix()
    v = moved(ref, u, s)
    def evaluate(w):
        return response(a.model, a.element, w, a.shell_origin, a.layers)
    def energy(w):
        local, _ = fit_oracle(ref, w)
        # Independent path integral of the unchanged local elastic operator.
        x, weights = np.polynomial.legendre.leggauss(8)
        return sum(.5*weight*float(local@a.element.compute_nonlinear_response(a.model.mesh,
            a.model.get_material('joint-shell'), .5*(point+1)*local, a.shell_origin, 3, False)[0])
            for point, weight in zip(x, weights))
    r, j, candidate = evaluate(v); base = evaluate(u)
    p = np.eye(len(v)); correction = np.zeros_like(j)
    for i, row in enumerate(v.reshape(-1, 6)):
        _, ai, dai = _exp_terms(row[3:]); sl = slice(6*i+3, 6*i+6)
        p[sl, sl] = ai; correction[sl, sl] = np.einsum('ijk,i->jk', dai, r[sl])
    h = p.T@j+correction
    direction = np.cos(np.arange(len(v))+.3); eps = 2e-6
    rp = evaluate(v+eps*direction)[0]; rm = evaluate(v-eps*direction)[0]
    errors = dict(tangent=norm_error((rp-rm)/(2*eps), j@direction),
        symmetry=norm_error(h, h.T), energy_objectivity=abs(energy(v)-energy(u)),
        work=abs((energy(v+eps*direction)-energy(v-eps*direction))/(2*eps)-direction@p.T@r),
        wrench_objectivity=norm_error(r.reshape(-1, 3), base[0].reshape(-1, 3)@s.T))
    assert errors['tangent'] < 1e-7 and errors['work'] < 1e-7, errors
    assert max(errors[k] for k in ('symmetry', 'energy_objectivity', 'wrench_objectivity')) < 1e-11, errors
    wrench = r.reshape(-1, 6); x = ref+v.reshape(-1, 6)[:, :3]
    assert np.linalg.norm(wrench[:, :3].sum(axis=0)) < 1e-11
    assert np.linalg.norm((wrench[:, 3:]+np.cross(x, wrench[:, :3])).sum(axis=0)) < 1e-11
    zero = evaluate(np.zeros_like(v))
    k = a.element.compute_stiffness_matrix(a.model.mesh, a.model.get_material('joint-shell'))
    assert norm_error(zero[1], k) < 1e-11
    (tmp_path/'shell.json').write_bytes(canonical(dict(errors=errors, residual=r, tangent=j,
        state=candidate, production_qualified=False)))


def successor(topology, curved=True):
    old = build(topology, curved); a = old.assembly
    new = ShellBeamTrialAssembly(a.beam, topology=topology, coordinates=a.coordinates,
        reference_normal=a.normal, thickness=.2, elastic_modulus=1000., poisson_ratio=.3,
        shell_node=a.shell_node, beam_node=a.beam_node, variational_shell=True)
    owner = CoupledShellBeamAnalysis(new, targets=old.targets, shell_fixed=old.shell_fixed,
        nodal_forces=old.forces)
    return old, owner


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_loaded_coupled_symmetry_restart_and_policy_separation(topology, tmp_path):
    old, made = successor(topology)
    result = made.solve(stop_after=2)
    assert result.status == 'paused', result.failure
    state = result.state
    response_, residual, _, _ = made._evaluate(state, state.mechanical, state.shell_u,
        state.multipliers, state.cursor)
    p = np.eye(made.assembly.count)
    for i, row in enumerate(state.shell_u.reshape(-1, 6)):
        p[6*i+3:6*i+6, 6*i+3:6*i+6] = _exp_terms(row[3:])[1]
    h = (p.T@response_.tangent)[np.ix_(made.free, made.free)]
    assert norm_error(h, h.T) < 1e-11
    assert np.linalg.norm(residual[list(made.free)]) < 1e-11
    with pytest.raises(ValueError):
        old.solve(checkpoint=result.checkpoint, expected_sha256=sha256(result.checkpoint).hexdigest())
    _, other = successor(topology)
    resumed = other.solve(checkpoint=result.checkpoint, expected_sha256=sha256(result.checkpoint).hexdigest())
    _, fresh = successor(topology)
    full = fresh.solve()
    assert resumed.status == full.status == 'completed', (resumed.failure, full.failure)
    assert resumed.checkpoint == full.checkpoint
    (tmp_path/'coupled.json').write_bytes(canonical(dict(checkpoint=full.checkpoint.decode('ascii'),
        symmetry=norm_error(h, h.T), production_qualified=False)))


@pytest.mark.parametrize('kind', ('coincident', 'collinear', 'nonfinite'))
def test_invalid_fit_fails_closed(kind):
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    x = ref.copy()
    if kind == 'coincident': x[:] = 0.
    elif kind == 'collinear': x[:, 1:] = 0.
    else: x[0, 0] = np.nan
    with pytest.raises(ValueError): rotation_jets(ref, x)


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_real_numbering_covariance(topology, tmp_path):
    a = specimen(topology); count = len(a.coordinates)
    u = moved(a.coordinates, .02*np.sin(np.arange(a.shell_count)+.2),
        Rotation.from_rotvec([1.1, -.2, .7]).as_matrix())
    r, j, _ = response(a.model, a.element, u, a.shell_origin, 3)
    orders = list(permutations(range(3))) if count == 3 else [
        tuple(np.roll(np.arange(4)[::sign], offset)) for sign in (1, -1) for offset in range(4)]
    errors = []
    for order in orders:
        order = tuple(int(i) for i in order)
        b = ShellBeamTrialAssembly(a.beam, topology=topology, coordinates=a.coordinates[list(order)],
            reference_normal=a.normal, thickness=.2, elastic_modulus=1000., poisson_ratio=.3,
            shell_node=order.index(a.shell_node), beam_node=a.beam_node, variational_shell=True)
        ids = np.array([[6*i+k for k in range(6)] for i in order]).ravel()
        rr, jj, _ = response(b.model, b.element, u[ids], b.shell_origin, 3)
        errors.append(dict(order=order, force=norm_error(rr, r[ids]), tangent=norm_error(jj, j[np.ix_(ids, ids)])))
    assert max(max(x['force'], x['tangent']) for x in errors) < 1e-11, errors
    (tmp_path/'numbering.json').write_bytes(canonical(dict(errors=errors, production_qualified=False)))


@pytest.mark.parametrize('kind', ('plastic', 'curve', 'singular-chart', 'policy', 'policy-type'))
def test_unsupported_state_and_policy_mutations_rejected(kind):
    a = specimen(); u = np.zeros(a.shell_count)
    if kind == 'plastic': a.model.get_material('joint-shell').yield_stress = 1.
    elif kind == 'curve': a.model.get_material('joint-shell').hardening_curve = object()
    elif kind == 'singular-chart': u[3] = 2*np.pi
    else:
        a.variational_shell = True if kind == 'policy' else 0
        with pytest.raises(ValueError): a.guard()
        return
    with pytest.raises(ValueError): response(a.model, a.element, u, a.shell_origin, 3)
