"""Accepted-state conservative point pencil; numerical checks require supervision."""
from dataclasses import replace
from hashlib import sha256
import pickle
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import eig
from scipy.sparse.linalg import ArpackNoConvergence

from anyfesolver import stiffened_shell as public
from anyfesolver import _stiffened_shell_ge_point_buckling as buckling
from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis, NodalAction, FixedCoordinate
from anyfesolver._stiffened_shell_ge_finite_experimental import _position_pair
from anyfesolver._ge_beam3_variational_shell import deformation
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log
from anyfesolver.control import CancellationToken
from test_stiffened_shell_ge_experimental import _panel


def definition():
    return replace(_panel(), q4_pullback_policy=public.Q4_ALGORITHMIC_PULLBACK_V1)


@pytest.fixture(scope='module')
def accepted_owner():
    owner = ExperimentalGEAnalysis(definition())
    result = owner.solve((NodalAction('beam', 105, (-1000., 0., 100., 0., 0., 0.)),),
                         (1.,), max_iterations=24, max_backtracks=12)
    assert result.status == 'completed'
    yield owner
    owner.close()


def dense_full_hessian(owner, packet):
    """Independent dense scatter of actual local scalar-potential derivatives.

    Point AD uses the actual additive shell chart and zero-Exp beam chart,
    avoiding the production spatial-row point conversion entirely.
    """
    snap = packet.snapshot; driver = owner._finite_driver
    size = packet.material.shape[0]
    assert size <= 256, 'dense oracle is bounded and never a production solve'
    h = np.zeros((size, size))
    material = np.zeros_like(h)
    for (_eid, element, slots, _), origin in zip(owner._shell_parts, pickle.loads(snap.shell_origins), strict=True):
        reference = np.array([owner._shell_model.mesh.nodes[node].coords() for node in element.node_ids])
        anchors = None if snap.shell_anchors is None else snap.shell_anchors[slots[::6]//6]
        local, d, dd = deformation(reference, snap.q[slots], stable_translation=True, rotation_anchors=anchors)
        f, k, _, parts = element.compute_nonlinear_response(owner._shell_model.mesh,
            owner._shell_model.get_material(element.material_name), local, origin, 3, True,
            _return_tangent_components=True)
        h[np.ix_(slots, slots)] += d.T@k@d+np.einsum('i,ijk->jk', f, dd)
        material[np.ix_(slots, slots)] += d.T@parts['material']@d
    u = snap.q[owner._shell_count:owner._shell_count+driver._beam_nodal].reshape(-1, 6)[:, :3]
    positions, low = _position_pair(driver._reference_positions, u)
    for index, ((element, slots, _), origin) in enumerate(zip(owner._beam_parts, pickle.loads(snap.beam_origins), strict=True)):
        nodes = np.array([owner._beam_index[node] for node in element.node_ids])
        response = element.operator.evaluate(positions[nodes], low[nodes], snap.frames[nodes],
            snap.cells[index], snap.resultants[index], origin=origin)
        slots = slots+owner._shell_count
        h[np.ix_(slots, slots)] += response.hessian+response.hessian_low
        mixed = response.hessian+response.hessian_low
        local_material = mixed.copy(); local_material[:24, :24] = 0.
        material[np.ix_(slots, slots)] += local_material
    for index, (joint, slots) in enumerate(zip(owner._connections, owner._joint_slots, strict=True)):
        sn = slots[0]//6; bn = (slots[6]-owner._shell_count)//6
        sx, sl = _position_pair(owner._shell_xyz[sn], snap.q[slots[:3]])
        distance = (positions[bn]-sx)+(low[bn]-sl)
        v = [Jet2.variable(0., i, 18) for i in range(18)]
        theta = snap.q[slots[3:6]]
        anchor = np.eye(3) if snap.shell_anchors is None else snap.shell_anchors[sn]
        rs = matmul(so3_exp([v[3+j]+float(theta[j]) for j in range(3)]), constant_matrix(anchor, 18))
        rb = matmul(so3_exp(v[9:12]), constant_matrix(snap.frames[bn], 18))
        separation = [v[6+j]-v[j]+float(distance[j]) for j in range(3)]
        ct = [value-float(offset) for value, offset in zip(matvec(transpose(rs), separation), joint._offset)]
        cr = so3_log(matmul(constant_matrix(joint._relative.T, 18), matmul(transpose(rs), rb)))
        potential = sum((value*(v[12+j]+float(snap.multipliers[6*index+j])) for j, value in enumerate(ct+cr)), Jet2.constant(0., 18))
        slots = np.r_[slots, np.arange(packet.primal_count+6*index, packet.primal_count+6*index+6)]
        h[np.ix_(slots, slots)] += potential.hessian
        local_material = potential.hessian.copy(); local_material[:12, :12] = 0.
        material[np.ix_(slots, slots)] += local_material
    return h, material


def test_full_mixed_split_matches_independent_dense_oracle_and_owned_state(accepted_owner):
    owner = accepted_owner; before = owner.checkpoint()
    packet, guard = buckling.capture(owner, owner._generation)
    assert len(owner._shell_parts) >= 2 and len(owner._beam_parts) >= 2
    assert np.linalg.norm(packet.snapshot.multipliers) > 0
    assert np.linalg.norm(packet.snapshot.q[:owner._shell_count].reshape(-1, 6)[:, 3:]) > 0
    assert packet.primal_count == owner._shell_count+6*len(owner._beam_ids)+24*len(owner._beam_parts)
    oracle, material = dense_full_hessian(owner, packet)
    np.testing.assert_allclose(packet.hessian.toarray(), oracle, rtol=1e-11, atol=2e-6)
    np.testing.assert_allclose(packet.material.toarray(), material, rtol=1e-11, atol=2e-6)
    np.testing.assert_allclose(packet.geometric.toarray(), oracle-material, rtol=1e-11, atol=2e-6)
    np.testing.assert_allclose((packet.material+packet.geometric).toarray(), oracle, rtol=1e-11, atol=2e-6)
    free = np.array(packet.free)
    np.testing.assert_allclose((packet.material+packet.geometric)[free, :][:, free].toarray(),
                               oracle[np.ix_(free, free)], rtol=1e-11, atol=2e-6)
    assert packet.multiplier_curvature.nnz > 0
    retained = owner._shell_count+6*len(owner._beam_ids)
    for i in range(len(owner._beam_parts)):
        p = np.arange(retained+24*i+6, retained+24*i+24)
        assert np.all(np.linalg.eigvalsh(packet.material[p, :][:, p].toarray()) < 0)
    for value in (packet.snapshot.q, packet.snapshot.frames, packet.material.data):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert isinstance(packet.snapshot.shell_origins, bytes)
    assert owner.checkpoint() == before
    guard()


def test_actual_advance_covector_derivative_includes_chart_and_multiplier_curvature(accepted_owner):
    owner = accepted_owner; driver = owner._finite_driver
    packet, _ = buckling.capture(owner, owner._generation)
    state = driver._state; n = packet.primal_count
    direction = np.sin(np.arange(packet.material.shape[0])+1.)
    direction[:n] *= owner._coordinate_scale
    direction[n:] *= owner._constraint_scale
    direction[owner._fixed] = 0.
    # Three fixed steps form the registered derivative plateau; no adaptation
    # or history commit changes the accepted parent or constitutive update.
    steps = (3e-6, 1e-6, 3e-7)
    cache_names = ('_line_trial_recovery','_point_trial_recovery','_distributing_trial_recovery','_generalized_trial_recovery')
    caches = {name: getattr(driver, name) for name in cache_names}
    before = owner.checkpoint()
    def covector(amount):
        trial = driver._advance(state, direction, amount)
        residual = driver._assemble(trial, driver._load_factor,
            pickle.loads(packet.snapshot.shell_origins), pickle.loads(packet.snapshot.beam_origins),
            applied_load=packet.snapshot.applied_dead_load)[0]
        result = residual.copy()
        for node, row in enumerate(trial.q[:owner._shell_count].reshape(-1, 6)):
            s = slice(6*node+3, 6*node+6)
            result[s] = _exp_terms(row[3:])[1].T@residual[s]
        starts = [owner._shell_count+6*node+3 for node in range(len(owner._beam_ids))]
        retained = owner._shell_count+6*len(owner._beam_ids)
        starts += [retained+24*i+3*c for i in range(len(owner._beam_parts)) for c in range(2)]
        for start in starts:
            s = slice(start, start+3)
            result[s] = _exp_terms(amount*direction[s])[1].T@residual[s]
        return result
    try:
        finite_differences = [(covector(step)-covector(-step))/(2*step) for step in steps]
    finally:
        for name, value in caches.items():
            setattr(driver, name, value)
    exact = packet.hessian@direction
    row_scale = np.r_[owner._coordinate_scale, owner._constraint_scale]
    errors = [np.max(abs((finite_difference-exact)*row_scale))/max(1., np.max(abs(exact*row_scale)))
              for finite_difference in finite_differences]
    assert max(errors) <= 1e-6, dict(steps=steps, errors=errors)
    assert owner.checkpoint() == before


def test_sparse_finite_pairs_match_bounded_dense_full_mixed_pencil(accepted_owner):
    owner = accepted_owner; before = owner.checkpoint()
    packet, _ = buckling.capture(owner, owner._generation)
    free = np.array(packet.free)
    # LAPACK handles singular B; infinity is kept distinct from finite factors.
    oracle, material = dense_full_hessian(owner, packet)
    _, _, scale = buckling._scaled_pencil(owner, packet)
    congruence = np.outer(scale, scale)
    values = eig(material[np.ix_(free, free)]*congruence,
                 -(oracle-material)[np.ix_(free, free)]*congruence, right=False)
    finite = [float(value.real) for value in values if np.isfinite(value) and value.real > 0
              and abs(value.imag) <= 1e-8*abs(value.real)]
    assert finite
    result = owner.buckling(public.GEPointBucklingRequest(owner._generation, count=2))
    assert result.status == 'completed', result.failure_reason
    for value in result.prestress_multipliers:
        assert min(abs(value-ref)/max(1., abs(ref)) for ref in finite) < 1e-7
    assert max(result.equation_residuals) <= 1e-8 and max(result.constraint_residuals) <= 1e-8
    assert result.resultant_modes[0].shape == (len(owner._beam_parts), 18)
    assert result.cell_rotation_modes[0].shape == (len(owner._beam_parts), 6)
    for coordinate, spatial_mode in zip(result.shell_coordinate_modes, result.shell_spatial_modes, strict=True):
        expected = (packet.chart[:owner._shell_count, :owner._shell_count]@coordinate.ravel()).reshape(-1, 6)
        np.testing.assert_array_equal(spatial_mode, expected)
    for coordinate, physical in zip(result.coordinate_support_reactions, result.physical_support_wrenches, strict=True):
        np.testing.assert_allclose(packet.chart[:packet.primal_count, :packet.primal_count].T@physical,
                                   coordinate, rtol=1e-12, atol=1e-9)
    assert result.checkpoint_sha256 == sha256(before).hexdigest() and owner.checkpoint() == before


def test_capture_guards_stale_generation_and_pencil_mutation(accepted_owner):
    owner = accepted_owner
    with pytest.raises(ValueError, match='matching generation'):
        buckling.capture(owner, owner._generation+1)
    packet, guard = buckling.capture(owner, owner._generation)
    packet.geometric.data = packet.geometric.data.copy()
    packet.geometric.data[0] += 1.
    with pytest.raises(ValueError, match='pencil changed'):
        guard()


def test_terminal_factor_failure_and_cancellation_preserve_checkpoint(accepted_owner, monkeypatch):
    owner = accepted_owner; before = owner.checkpoint()
    class Broken:
        status = 'failed'; failure_reason = 'injected singular shift'
        factorization_count = 1; solve_count = 0
    monkeypatch.setattr(buckling, 'factorize_cached', lambda *args: Broken())
    result = owner.buckling(public.GEPointBucklingRequest(owner._generation))
    assert result.status == 'failed' and not result.full_coordinates
    token = CancellationToken(); token.cancel('test cancellation')
    result = owner.buckling(public.GEPointBucklingRequest(owner._generation), cancellation_token=token)
    assert result.status == 'cancelled' and not result.full_coordinates
    assert owner.checkpoint() == before


def test_infinite_descriptor_pairs_are_failure_not_silent_mode_removal(accepted_owner, monkeypatch):
    owner = accepted_owner
    monkeypatch.setattr(buckling, 'eigs', lambda operator, **kwargs:
        (np.zeros(kwargs['k']), np.ones((operator.shape[0], kwargs['k']))))
    result = owner.buckling(public.GEPointBucklingRequest(owner._generation, count=1))
    assert result.status == 'failed' and 'infinite' in result.failure_reason


def test_direct_elastic_beam_history_branch_and_yield_margin_admission(accepted_owner):
    core = accepted_owner._beam_parts[0][0].operator
    origin = core.cell.virgin()
    stations = [dict(branch='ELASTIC', resultants=[[0.]*6, [0.]*6])
                for _ in origin.stations]
    def response(history=origin):
        return SimpleNamespace(history=history, material=json.dumps(dict(stations=stations)))
    buckling._elastic_beam(core, response(), origin)
    changed = replace(origin, cell_identity=origin.cell_identity+'-changed')
    with pytest.raises(ValueError, match='must not advance generalized history'):
        buckling._elastic_beam(core, response(changed), origin)
    stations[0]['branch'] = 'PLASTIC'
    with pytest.raises(ValueError, match='elastic-interior'):
        buckling._elastic_beam(core, response(), origin)
    stations[0]['branch'] = 'ELASTIC'
    # Exact unit metric isolates the production Decimal yield-margin predicate;
    # it does not modify the accepted model or its material law.
    metric = tuple(tuple(buckling.Decimal(int(i == j)) for j in range(6)) for i in range(6))
    section = SimpleNamespace(yield_force=1., hardening=0., _m=metric,
                              _origin=lambda history: (None, 0))
    margin_core = SimpleNamespace(section=section)
    stations[0]['resultants'][0][0] = 1.-128*np.finfo(float).eps
    buckling._elastic_beam(margin_core, response(), origin)
    for value in (1.-32*np.finfo(float).eps, 1., 1.+np.finfo(float).eps):
        stations[0]['resultants'][0][0] = value
        with pytest.raises(ValueError, match='unresolved yield boundary'):
            buckling._elastic_beam(margin_core, response(), origin)


class _MockInverse:
    status = 'ok'; failure_reason = None
    factorization_count = 1
    def __init__(self):
        self.solve_count = 0
    def solve(self, rhs):
        self.solve_count += 1
        return np.asarray(rhs)


def test_mid_iteration_cancellation_has_no_partial_modes(accepted_owner, monkeypatch):
    owner = accepted_owner; before = owner.checkpoint()
    token = CancellationToken(); handle = _MockInverse()
    monkeypatch.setattr(buckling, 'factorize_cached', lambda *args: handle)
    def interrupted(operator, **kwargs):
        operator.matvec(np.ones(operator.shape[0]))
        token.cancel('injected between inverse actions')
        operator.matvec(np.ones(operator.shape[0]))
        pytest.fail('iteration cancellation was ignored')
    monkeypatch.setattr(buckling, 'eigs', interrupted)
    result = owner.buckling(public.GEPointBucklingRequest(owner._generation), cancellation_token=token)
    assert result.status == 'cancelled' and 'iteration' in result.failure_reason
    assert handle.solve_count == 1 and result.solve_count == 1
    assert not result.full_coordinates and not result.prestress_multipliers
    assert owner.checkpoint() == before


@pytest.mark.parametrize('failure', ['arpack', 'backend'])
def test_iteration_numerical_failures_discard_partial_eigenpairs(accepted_owner, monkeypatch, failure):
    owner = accepted_owner; before = owner.checkpoint(); handle = _MockInverse()
    monkeypatch.setattr(buckling, 'factorize_cached', lambda *args: handle)
    def failed(operator, **kwargs):
        if failure == 'arpack':
            raise ArpackNoConvergence('injected incomplete Arnoldi', np.array([1.]),
                                     np.ones((operator.shape[0], 1)))
        operator.matvec(np.ones(operator.shape[0]))
        pytest.fail('backend failure was ignored')
    if failure == 'backend':
        def broken(rhs):
            handle.solve_count += 1
            raise RuntimeError('Sparse solve produced NaN/Inf values')
        handle.solve = broken
    monkeypatch.setattr(buckling, 'eigs', failed)
    result = owner.buckling(public.GEPointBucklingRequest(owner._generation))
    assert result.status == 'failed' and result.failure_reason
    assert not result.full_coordinates and not result.prestress_multipliers
    assert owner.checkpoint() == before


def test_model_provenance_mutation_is_rejected_at_boundary(accepted_owner):
    owner = accepted_owner; before = owner.checkpoint()
    _, guard = buckling.capture(owner, owner._generation)
    saved = owner._coordinate_scale
    owner._coordinate_scale = saved.copy()
    owner._coordinate_scale[0] *= 2.
    try:
        # Cheap iterative guards deliberately defer model-wide identity checks.
        guard('iteration', full=False)
        with pytest.raises(ValueError, match='model/support provenance changed'):
            guard('final')
    finally:
        owner._coordinate_scale = saved
    guard('restored')
    assert owner.checkpoint() == before


def test_explicit_admission_and_legacy_namespace_are_preserved(accepted_owner):
    assert accepted_owner.point_buckling_capabilities['definition_admitted']
    assert accepted_owner.point_buckling_capabilities['material_capture_validation_pending']
    assert accepted_owner.point_buckling_capabilities['eligible'] is None
    assert public.GEPointBucklingRequest.__name__ in public.__all__
    fresh = ExperimentalGEAnalysis(definition())
    try:
        assert not fresh.point_buckling_capabilities['eligible']
        with pytest.raises(ValueError, match='accepted finite'):
            fresh.buckling(public.GEPointBucklingRequest(1))
    finally:
        fresh.close()
    for key, value in (('q4_pullback_policy', 'historical_corotational'),):
        old = getattr(accepted_owner._definition, key)
        object.__setattr__(accepted_owner._definition, key, value)
        try:
            with pytest.raises(ValueError, match='objective'):
                accepted_owner.buckling(public.GEPointBucklingRequest(accepted_owner._generation))
        finally:
            object.__setattr__(accepted_owner._definition, key, old)


def test_partial_rotational_support_rejected_without_rotating_support_equations(accepted_owner):
    saved = accepted_owner._fixed
    accepted_owner._fixed = np.array([i for i in saved if i != 5])
    try:
        with pytest.raises(ValueError, match='full fixed triples'):
            buckling.capture(accepted_owner, accepted_owner._generation)
    finally:
        accepted_owner._fixed = saved


def test_member_geometric_block_changes_sign_with_stress_without_condensing(accepted_owner):
    owner = accepted_owner; core = owner._beam_parts[0][0].operator
    state = owner._finite_driver._state
    element = owner._beam_parts[0][0]
    nodes = np.array([owner._beam_index[node] for node in element.node_ids])
    u = state.q[owner._shell_count:owner._shell_count+6*len(owner._beam_ids)].reshape(-1, 6)[:, :3]
    positions, low = _position_pair(owner._finite_driver._reference_positions, u)
    origin = core.cell.virgin()
    p = np.zeros(18); p[:2] = 100.
    positive = core.evaluate(positions[nodes], low[nodes], state.frames[nodes], state.cells[0], p, origin=origin)
    negative = core.evaluate(positions[nodes], low[nodes], state.frames[nodes], state.cells[0], -p, origin=origin)
    np.testing.assert_array_equal(positive.hessian[:24, :24], -negative.hessian[:24, :24])
    assert positive.hessian.shape == (42, 42)


@pytest.mark.parametrize('kwargs', [dict(expected_generation=0), dict(expected_generation=1, count=0),
    dict(expected_generation=1, shift=float('nan')), dict(expected_generation=1, bounds=(2.,1.))])
def test_typed_request_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        public.GEPointBucklingRequest(**kwargs)
