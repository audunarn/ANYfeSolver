"""Bounded independent physical rigid-work and free descriptor controls.

The dense quotient is test-only (160 rows); it removes only the analytically
known six rigid fields from the complete singular pencil. Native algebraic
coordinates and multipliers remain, without static condensation or new inertia.
"""
from dataclasses import replace
from math import fsum
import numpy as np
import pytest
from scipy import linalg
from anyfesolver import stiffened_shell as api
from test_stiffened_shell_ge_reference_modal_oracle import (
    _definition, _independent_mass, _cross, PARTICLES, WEIGHTS)


def _rigid_fields(owner, definition, origin):
    """Physical infinitesimal common motion; derive incidence independently."""
    R = np.zeros((owner._K.shape[0], 6))
    for i, xyz in enumerate(definition.shell.coordinates_m):
        R[6*i:6*i+3] = np.c_[np.eye(3), -_cross(np.asarray(xyz)-origin)]
        R[6*i+3:6*i+6, 3:] = np.eye(3)
    ids = sorted({n for d in definition.beams for n in d.instantiate()[0].node_ids})
    coordinates = {}
    for d in definition.beams:
        e, _ = d.instantiate()
        coordinates.update(zip(e.node_ids, e.operator.reference.coordinates))
    start = 6*len(definition.shell.node_ids)
    for i, node in enumerate(ids):
        R[start+6*i:start+6*i+3] = np.c_[np.eye(3), -_cross(coordinates[node]-origin)]
        R[start+6*i+3:start+6*i+6, 3:] = np.eye(3)
    start += 6*len(ids)
    for i in range(len(definition.beams)):
        R[start+24*i:start+24*i+3, 3:] = np.eye(3)
        R[start+24*i+3:start+24*i+6, 3:] = np.eye(3)
    return R


def _volume_gram(definition, origin):
    """Physical shell volume and beam particles, no FE mass or trace helper."""
    G = np.zeros((6, 6))
    def add(position, weight):
        velocity = np.c_[np.eye(3), -_cross(position-origin)]
        G[:] += weight*(velocity.T@velocity)
    nodes = dict(zip(definition.shell.node_ids, np.asarray(definition.shell.coordinates_m)))
    quadrature, weights = np.polynomial.legendre.leggauss(3)
    shell = definition.shell
    for ids in shell.q4_elements:
        p = np.array([nodes[n] for n in ids])
        normal = np.cross(p[1]-p[0], p[3]-p[0])
        area = np.linalg.norm(normal); normal /= area
        for u, wu in zip(quadrature, weights):
            for v, wv in zip(quadrature, weights):
                shape = np.array(((1-u)*(1-v), (1+u)*(1-v),
                                  (1+u)*(1+v), (1-u)*(1+v)))/4.
                for z, wz in zip(quadrature, weights):
                    position = shape@p + z*shell.thickness_m/2.*normal
                    add(position, shell.density_kg_m3*area*shell.thickness_m/8.*wu*wv*wz)
    for d in definition.beams:
        e, _ = d.instantiate(); ref = e.operator.reference
        frame = ref.nodal_triads[0]
        for a, b in ((0, 1), (1, 2)):
            length = np.linalg.norm(ref.coordinates[b]-ref.coordinates[a])
            for s, w in zip(quadrature, weights):
                axis = (1-s)/2.*ref.coordinates[a]+(1+s)/2.*ref.coordinates[b]
                for p, density in zip(PARTICLES, WEIGHTS):
                    add(axis+frame@p, length/2.*w*density)
    return G


def _flexible_qz(owner, mass, R):
    n = len(mass); m = owner._C.shape[0]
    A = np.block([[owner._K.toarray(), owner._C.T.toarray()],
                  [owner._C.toarray(), np.zeros((m,m))]])
    B = linalg.block_diag(mass, np.zeros((m,m)))
    assert n+m <= 160
    rigid = np.vstack((R, np.zeros((m,6))))
    # B-orthogonal rigid complement retains all algebraic coordinates. This
    # dense basis exists solely as a bounded independent test oracle.
    Z = linalg.null_space(rigid.T@B)
    aa, bb = Z.T@A@Z, Z.T@B@Z
    scale = np.ones(len(aa))
    for _ in range(24):
        magnitude = np.maximum(np.max(abs(aa), axis=1), np.max(abs(bb), axis=1))
        assert np.all(magnitude > 0.)
        d = 1./np.sqrt(magnitude)
        aa = d[:,None]*aa*d[None,:]; bb = d[:,None]*bb*d[None,:]; scale *= d
    time_scale = np.max(abs(aa))/np.max(abs(bb))
    raw, vectors = linalg.eig(aa, time_scale*bb, homogeneous_eigvals=True)
    pairs = []
    for (alpha, beta), vector in zip(raw.T, vectors.T):
        if beta == 0.: continue
        lam = time_scale*alpha/beta
        if abs(lam.imag) > 1e-8*abs(lam.real) or lam.real <= 0.: continue
        factor = linalg.lu_factor(aa-lam.real*(1.+1e-4)*bb)
        v = vector.real
        for _ in range(4):
            v = linalg.lu_solve(factor, bb@v); v /= np.linalg.norm(v)
        outer = np.outer(v,v)
        denominator = fsum((outer*bb).ravel())
        if denominator <= 0.: continue
        value = fsum((outer*aa).ravel())/denominator
        x = Z@(scale*v)
        x /= np.sqrt(x@B@x)
        residual = A@x-value*(B@x)
        bound = abs(A)@abs(x)+abs(value)*(abs(B)@abs(x))
        if np.linalg.norm(residual)/np.linalg.norm(bound) > 1e-8: continue
        pairs.append((value,x))
    assert len(pairs) >= 3
    return A, B, pairs, raw


@pytest.mark.parametrize('angle', (0., .63))
def test_six_rigid_fields_have_independent_physical_volume_work(angle, tmp_path):
    definition = replace(_definition(angle=angle), fixed=(),
                         modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)
    owner = api.create_analysis(definition=definition)
    try:
        result = owner.modes(api.GEReferenceModesRequest(3, 1.))
        assert result.status == 'completed', result.failure_reason
        origin = np.mean(definition.shell.coordinates_m, axis=0)
        R = _rigid_fields(owner, definition, origin)
        mass = _independent_mass(owner, definition)
        gram = _volume_gram(definition, origin)
        np.testing.assert_allclose(np.column_stack(result.canonical_rigid_coordinates), R,
                                   rtol=0., atol=1e-14)
        np.testing.assert_allclose(result.rigid_kinetic_gram, gram, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(R.T@mass@R, gram, rtol=2e-13, atol=2e-13)
        assert np.linalg.eigvalsh(gram)[0] > 0.
        for matrix in (owner._K, owner._C):
            for r in R.T:
                error = np.linalg.norm(matrix@r)
                scale = np.linalg.norm(abs(matrix)@abs(r))
                assert error <= 1e-9*scale + 1e-12
        np.savez_compressed(tmp_path/'rigid-physical-work.npz', R=R,
                            physical_volume_gram=gram, mass=mass,
                            K=owner._K.toarray(), C=owner._C.toarray())
    finally: owner.close()


def test_resolved_extra_physical_null_mode_is_rejected_without_zeroing_stiffness(tmp_path):
    """Inject one positive-kinetic compatible mechanism, not a zero operator.

    This controls the near-zero branch; a finite requested Krylov space cannot
    certify the absence of every unexcited mechanism or spectrum completeness.
    """
    from scipy import sparse
    definition = replace(_definition(), fixed=(),
                         modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)
    owner = api.create_analysis(definition=definition)
    try:
        original = owner.modes(api.GEReferenceModesRequest(1, 1.))
        assert original.status == 'completed', original.failure_reason
        q = np.asarray(original.flexible.full_coordinates[0])
        mass = _independent_mass(owner, definition)
        R = _rigid_fields(owner, definition, np.mean(definition.shell.coordinates_m, axis=0))
        np.testing.assert_allclose(owner._C@q, 0., atol=1e-9)
        np.testing.assert_allclose(R.T@mass@q, 0., atol=1e-7)
        assert q@mass@q > 0.
        old = owner._K.toarray()
        action = old@q
        work = float(q@action)
        assert work > 0.
        modified = old-np.outer(action, action)/work
        assert np.linalg.norm(modified) > .5*np.linalg.norm(old)
        residual = modified@q
        assert np.linalg.norm(residual) <= 1e-8*np.linalg.norm(abs(modified)@abs(q))
        for r in R.T:
            assert np.linalg.norm(modified@r) <= 1e-9*np.linalg.norm(abs(modified)@abs(r))+1e-12
            assert np.linalg.norm(owner._C@r) <= 1e-9*np.linalg.norm(abs(owner._C)@abs(r))+1e-12
        owner._K = sparse.csr_matrix(modified)
        owner._A = sparse.bmat([[owner._K, owner._C.T], [owner._C,None]], format='csc')
        result = owner.modes(api.GEReferenceModesRequest(1,1.))
        np.savez_compressed(tmp_path/'injected-physical-mechanism.npz',
            original_K=old, modified_K=modified, null_field=q, mass=mass,
            C=owner._C.toarray(), null_residual=residual)
        assert result.status == 'failed', result
        assert 'mechanism' in result.failure_reason, result.failure_reason
        assert not result.rigid_coordinates and not result.flexible.full_coordinates
        assert owner._generation == 0 and owner._accepted is None and owner._finite_driver is None
    finally: owner.close()


@pytest.mark.parametrize('shift_hz', (1., 80.))
def test_flexible_pairs_match_independent_full_descriptor_quotient(shift_hz, tmp_path):
    definition = replace(_definition(), fixed=(),
                         modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)
    owner = api.create_analysis(definition=definition)
    try:
        mass = _independent_mass(owner, definition)
        R = _rigid_fields(owner, definition, np.mean(definition.shell.coordinates_m,axis=0))
        A,B,pairs,raw = _flexible_qz(owner,mass,R)
        result = owner.modes(api.GEReferenceModesRequest(3,shift_hz))
        assert result.status == 'completed', result.failure_reason
        selected = sorted(pairs,key=lambda p:p[0])[:3]
        values = result.flexible.eigenvalues_rad2_s2
        np.savez_compressed(tmp_path/'free-descriptor-oracle.npz', A=A,B=B,
            raw_quotient_qz=raw, reference_values=np.array([p[0] for p in selected]),
            sparse_values=np.array(values),
            sparse_coordinates=np.array(result.flexible.full_coordinates))
        np.testing.assert_allclose(values,[p[0] for p in selected],rtol=1e-10,atol=1e-10)
        V = np.column_stack([np.r_[q,l] for q,l in zip(
            result.flexible.full_coordinates,result.flexible.multiplier_modes)])
        np.testing.assert_allclose(V.T@B@V,np.eye(3),rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(R.T@mass@V[:len(mass)],0.,rtol=0.,atol=1e-7)
        # Subspace comparison uses physical kinetic work; sign is arbitrary.
        U = np.column_stack([p[1] for p in selected])
        np.testing.assert_allclose(linalg.svdvals(U.T@B@V),np.ones(3),rtol=1e-7,atol=1e-7)
        for value,x in zip(values,V.T):
            residual = A@x-value*(B@x)
            bound = abs(A)@abs(x)+abs(value)*(abs(B)@abs(x))
            assert np.linalg.norm(residual)/np.linalg.norm(bound) <= 1e-8
    finally: owner.close()


def test_near_zero_screen_decisions_are_invariant_under_equivalent_units_and_frames():
    """Same physical panel in equivalent unit systems and frames.

    The injected mechanism must be refused and the genuine flexible pairs
    resolved in every equivalent description: a mixed-unit spectral floor
    would decide from the numeric scale instead of the physics.
    """
    from scipy import sparse
    resolved = []
    for angle in (0., .63):
        for length, mass in ((1., 1.), (1000., .001)):
            definition = replace(_definition(angle=angle, length_unit=length,
                mass_unit=mass), fixed=(),
                modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)
            owner = api.create_analysis(definition=definition)
            try:
                genuine = owner.modes(api.GEReferenceModesRequest(3, 1.))
                assert genuine.status == 'completed', genuine.failure_reason
                assert max(genuine.flexible.equation_residuals) <= 1e-8
                assert max(genuine.flexible.constraint_residuals) <= 1e-9
                resolved.append(np.asarray(genuine.flexible.eigenvalues_rad2_s2))
                q = np.asarray(genuine.flexible.full_coordinates[0])
                old = owner._K.toarray()
                action = old@q
                work = float(q@action)
                assert work > 0.
                owner._K = sparse.csr_matrix(old-np.outer(action, action)/work)
                owner._A = sparse.bmat([[owner._K, owner._C.T], [owner._C, None]],
                    format='csc')
                refused = owner.modes(api.GEReferenceModesRequest(1, 1.))
                assert refused.status == 'failed', refused
                assert 'mechanism' in refused.failure_reason, refused.failure_reason
                assert not refused.rigid_coordinates
                assert not refused.flexible.full_coordinates
            finally:
                owner.close()
    for values in resolved[1:]:
        np.testing.assert_allclose(values, resolved[0], rtol=1e-8, atol=1e-8)


def test_genuinely_soft_positive_mode_is_resolved_by_the_physical_work_screen():
    """A uniformly softer panel has positive modes far below any fixed
    spectral floor; the physical work screen must resolve them, not refuse.
    """
    values = {}
    for label, definition in (
            ('base', replace(_definition(), fixed=(),
                modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)),
            ('soft', replace(_definition(stiffness_unit=1e-11), fixed=(),
                modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)),
            ('soft-scaled', replace(_definition(stiffness_unit=1e-11,
                length_unit=1000., mass_unit=.001), fixed=(),
                modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1))):
        # The declared shift is a public request control; it is placed near
        # the soft spectrum so sigma+1/mu reconstruction stays well above
        # its own cancellation roundoff. The work screen itself uses no shift.
        shift = 1. if label == 'base' else .01
        owner = api.create_analysis(definition=definition)
        try:
            result = owner.modes(api.GEReferenceModesRequest(3, shift))
            assert result.status == 'completed', result.failure_reason
            assert max(result.flexible.equation_residuals) <= 1e-8
            assert max(result.flexible.constraint_residuals) <= 1e-9
            values[label] = np.asarray(result.flexible.eigenvalues_rad2_s2)
        finally:
            owner.close()
    assert np.all(values['soft'] > 0.)
    np.testing.assert_allclose(values['soft'], 1e-11*values['base'],
        rtol=1e-8, atol=1e-18)
    np.testing.assert_allclose(values['soft-scaled'], values['soft'],
        rtol=1e-8, atol=1e-18)
