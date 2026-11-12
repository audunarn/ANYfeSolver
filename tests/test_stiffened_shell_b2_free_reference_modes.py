"""Bounded free B2 descriptor oracles; no continuum or release qualification."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.spatial.transform import Rotation

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
import anyfesolver._stiffened_shell_b2_q4_panel_experimental as owner_module
from test_stiffened_shell_b2_mixed_reference_modes import _definition, _dense_mass
from test_stiffened_shell_b2_mixed_panel_experimental import _dense_operators


def _free_definition(kind="mixed"):
    definition, _ = _definition(kind)
    return replace(definition, transformation=sparse.eye(54, format="csr"),
                   boundary_policy=api.B2_FREE_FREE_MODAL_ONLY_V1)


def _canonical_fields(definition, origin):
    """Global translations and infinitesimal rotations about the published origin."""
    positions = dict(definition.shell_nodes_m)
    for member in definition.members:
        for node, position in zip(member.node_ids, member.reference_points_m):
            if node in positions:
                np.testing.assert_array_equal(positions[node], position)
            positions[node] = position
    fields = np.zeros((54, 6))
    for index, node in enumerate(sorted(positions)):
        radius = np.asarray(positions[node])-origin
        fields[6*index:6*index+3, :3] = np.eye(3)
        fields[6*index+3:6*index+6, 3:] = np.eye(3)
        for axis in range(3):
            fields[6*index:6*index+3, axis+3] = np.cross(np.eye(3)[axis], radius)
    return fields


def _dense_finite_positive_pairs(K, C, M, count):
    """Full homogeneous QZ pencil, retaining finite positive kinetic pairs.

    Singular massless angular DOFs may add zero/indeterminate algebraic pairs.
    No assertion of six and only six zero descriptor eigenvalues is made.
    """
    A = np.block([[K, C.T], [C, np.zeros((len(C), len(C)))]])
    B = linalg.block_diag(M, np.zeros((len(C), len(C))))
    homogeneous, vectors = linalg.eig(A, B, homogeneous_eigvals=True)
    alpha, beta = homogeneous
    eps = np.finfo(float).eps
    finite = abs(beta) > 100*eps*(abs(alpha)+abs(beta))
    values = np.full(len(alpha), np.nan+0j)
    values[finite] = alpha[finite]/beta[finite]
    # Roundoff in exact rigid fields is bounded using the independent pencil's
    # physical stiffness/inertia scale rather than a presumed nullity.
    zero_bound = 1e-8*np.linalg.norm(K, 2)/np.linalg.norm(M, 2)
    indices = []
    for i in np.flatnonzero(finite):
        value, primal = values[i], vectors[:len(K), i]
        if value.real <= zero_bound:
            continue
        assert abs(value.imag) <= 1e-8*abs(value.real)
        assert np.linalg.norm(primal.imag) <= 1e-8*np.linalg.norm(primal.real)
        primal = primal.real
        kinetic = primal@M@primal
        if kinetic > 1e-10*np.linalg.norm(M, 2)*np.dot(primal, primal):
            indices.append(i)
    order = sorted(indices, key=lambda i: values[i].real)
    assert len(order) >= count
    chosen = order[:count]
    primal = vectors[:len(K), chosen].real
    d, basis = linalg.eigh(primal.T@M@primal)
    assert np.min(d) > 0.
    normalized = primal@basis@np.diag(1./np.sqrt(d))
    return np.sqrt(values[chosen].real)/(2*np.pi), normalized


def _assert_tuple_tree(value):
    assert type(value) is tuple
    for item in value:
        if isinstance(item, (tuple, list, np.ndarray)):
            _assert_tuple_tree(item)
        else:
            assert isinstance(item, (float, int, np.floating))


@pytest.mark.parametrize("kind", ("q4", "mixed", "s3"))
def test_free_rigid_fields_and_flexible_pairs_against_independent_dense_pencil(kind):
    definition = _free_definition(kind)
    K, C = _dense_operators(definition)
    M = _dense_mass(definition)
    expected_frequencies, expected_span = _dense_finite_positive_pairs(K, C, M, 3)
    owner = api.create_analysis(definition=definition)
    try:
        before = tuple(matrix.copy() for matrix in (owner._K, owner._C, owner._M))
        result = owner.modes(api.B2ReferenceModesRequest(count=3))
        assert type(result) is api.B2FreeReferenceModesResult
        assert result.status == result.flexible.status == "ok", result.failure_reason
        assert result.policy == api.B2_FREE_FREE_MODAL_ONLY_V1
        assert result.production_qualified is result.flexible.production_qualified is False
        assert result.spectral_shift_rad2_s2 < 0.
        assert result.flexible.factorization_count == 1
        assert result.flexible.solve_count > 0
        np.testing.assert_allclose(result.flexible.frequencies_hz, expected_frequencies, rtol=2e-8)
        canonical = np.column_stack(result.canonical_rigid_modes)
        rigid = np.column_stack(result.rigid_modes)
        flexible = np.column_stack(result.flexible.physical_modes)
        assert canonical.shape == rigid.shape == (54, 6)
        assert flexible.shape == (54, 3)
        assert np.asarray(result.rigid_multiplier_modes).shape == (6, len(C))
        origin = np.asarray(result.rigid_origin_m)
        assert origin.shape == (3,) and np.isfinite(origin).all()
        np.testing.assert_allclose(canonical, _canonical_fields(definition, origin), rtol=0., atol=2e-15)
        np.testing.assert_allclose(K@canonical, 0., rtol=0., atol=2e-11*np.linalg.norm(K, np.inf))
        np.testing.assert_allclose(C@canonical, 0., rtol=0., atol=2e-13)
        np.testing.assert_allclose(result.rigid_kinetic_gram, canonical.T@M@canonical, rtol=2e-13, atol=2e-15)
        assert np.min(linalg.eigvalsh(np.asarray(result.rigid_kinetic_gram))) > 0.
        np.testing.assert_allclose(rigid.T@M@rigid, np.eye(6), rtol=0., atol=1e-7)
        np.testing.assert_allclose(flexible.T@M@flexible, np.eye(3), rtol=0., atol=1e-7)
        np.testing.assert_allclose(rigid.T@M@flexible, 0., rtol=0., atol=1e-7)
        np.testing.assert_allclose(linalg.svdvals(expected_span.T@M@flexible), np.ones(3), atol=2e-7)
        assert result.rigid_flexible_mass_overlap <= 1e-7
        assert len(result.rigid_equation_residuals) == len(result.rigid_constraint_residuals) == 6
        assert max(result.rigid_equation_residuals) <= 1e-8
        assert max(result.rigid_constraint_residuals) <= 1e-9
        for u, multiplier in zip(rigid.T, result.rigid_multiplier_modes):
            residual = K@u+C.T@multiplier
            bound = np.linalg.norm(K, 2)*np.linalg.norm(u)+np.linalg.norm(C.T@multiplier)
            assert np.linalg.norm(residual) <= 1e-8*bound
            assert np.linalg.norm(C@u) <= 1e-9*np.linalg.norm(abs(C)@abs(u))
        for frequency, u, multiplier in zip(result.flexible.frequencies_hz, flexible.T,
                                           result.flexible.point_multiplier_modes):
            inertial = (2*np.pi*frequency)**2*(M@u)
            balance = K@u+C.T@multiplier-inertial
            bound = np.linalg.norm(K@u)+np.linalg.norm(C.T@multiplier)+np.linalg.norm(inertial)
            assert np.linalg.norm(balance) <= 1e-8*bound
            assert np.linalg.norm(C@u) <= 1e-9*np.linalg.norm(abs(C)@abs(u))
        for field in ("rigid_origin_m", "canonical_rigid_modes", "rigid_modes",
                      "rigid_multiplier_modes", "rigid_kinetic_gram",
                      "rigid_equation_residuals", "rigid_constraint_residuals"):
            _assert_tuple_tree(getattr(result, field))
        for field in ("frequencies_hz", "physical_modes", "point_multiplier_modes"):
            _assert_tuple_tree(getattr(result.flexible, field))
        with pytest.raises(FrozenInstanceError):
            result.policy = "changed"
        for actual, original in zip((owner._K, owner._C, owner._M), before):
            for field in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))
        assert owner._handle is None
    finally:
        owner.close()


def test_explicit_opt_in_preserves_legacy_free_rejection_and_defers_factorization(monkeypatch):
    definition = _free_definition()
    with pytest.raises(ValueError):
        api.create_analysis(definition=replace(definition, boundary_policy=None))
    def unexpected(*args, **kwargs):
        raise AssertionError("free modal construction factorized a singular static system")
    with monkeypatch.context() as guard:
        guard.setattr(owner_module, "factorize_cached", unexpected)
        owner = api.create_analysis(definition=definition)
    try:
        assert owner._handle is None
        assert owner.capabilities.operations == ("modes", "close")
        assert not owner.capabilities.production_qualified
        with pytest.raises(FrozenInstanceError):
            definition.boundary_policy = None
        for operation in (lambda: owner.solve_linear(np.zeros(54)),
                          lambda: owner.solve_linear_many((("case", np.zeros(54)),), selected_dofs=(0,)),
                          lambda: owner.solve_linear_many((), selected_dofs=(0,)),
                          lambda: owner.body_acceleration_load((0., 0., -9.81))):
            with pytest.raises(ValueError, match="modal-only"):
                operation()
        assert owner._handle is None
    finally:
        owner.close()
    owner.close()
    with pytest.raises(owner_module.ExperimentalB2Q4PanelError, match="closed"):
        owner.modes(api.B2ReferenceModesRequest())


@pytest.mark.parametrize("bad", ("partial-support", "scaled", "permuted", "mpc", "offset",
                                  "policy", "missing-mass", "prestress"))
def test_explicit_free_policy_rejects_unsupported_definitions(bad):
    definition = _free_definition()
    if bad == "partial-support":
        definition = replace(definition, transformation=definition.transformation[:, 1:])
    elif bad == "scaled":
        definition = replace(definition, transformation=2*definition.transformation)
    elif bad == "permuted":
        definition = replace(definition, transformation=definition.transformation[:, np.arange(53, -1, -1)])
    elif bad == "mpc":
        T = definition.transformation.tolil(); T[1, 0] = .2
        definition = replace(definition, transformation=T.tocsr())
    elif bad == "offset":
        offset = np.zeros(54); offset[0] = 1e-6
        definition = replace(definition, prescribed_offset=offset)
    elif bad == "policy":
        definition = replace(definition, boundary_policy="free-free")
    elif bad == "missing-mass":
        definition = replace(definition, members=(replace(definition.members[0], mass=None),)+definition.members[1:])
    elif bad == "prestress":
        from anyfesolver._b2_physical_linear_definition import GeometricMetadata
        metadata = GeometricMetadata("geometry", (0., 0., 0.), (0., 0., 0.), (0., 0., 0.))
        definition = replace(definition, members=(replace(definition.members[0], geometric=metadata),)+definition.members[1:])
    with pytest.raises(ValueError):
        api.create_analysis(definition=definition)


@pytest.mark.parametrize("failure", ("cancel-before", "cancel-action", "arpack", "backend"))
def test_free_failures_publish_no_partial_rigid_or_flexible_pairs(failure, monkeypatch):
    owner = api.create_analysis(definition=_free_definition("s3"))
    token = CancellationToken()
    if failure == "cancel-before":
        token.cancel("before free reference modes")
    elif failure == "cancel-action":
        def interrupt(operator, **kwargs):
            token.cancel("during free reference modes")
            operator@np.ones(operator.shape[0])
            raise AssertionError("cancelled action returned")
        monkeypatch.setattr(owner_module, "eigs", interrupt)
    elif failure == "arpack":
        def partial(*args, **kwargs):
            raise ArpackNoConvergence("partial free eigensolve", np.array((1.,)), np.ones((66, 1)))
        monkeypatch.setattr(owner_module, "eigs", partial)
    else:
        original = owner_module.factorize_cached
        def broken(*args, **kwargs):
            handle = original(*args, **kwargs)
            def fail(_rhs):
                raise RuntimeError("Sparse solve produced NaN/Inf values")
            monkeypatch.setattr(handle, "solve", fail)
            return handle
        monkeypatch.setattr(owner_module, "factorize_cached", broken)
    try:
        result = owner.modes(api.B2ReferenceModesRequest(), cancellation=token)
        expected = "cancelled" if failure.startswith("cancel") else "failed"
        assert result.status == result.flexible.status == expected
        assert result.failure_reason
        assert result.rigid_modes == result.canonical_rigid_modes == result.rigid_multiplier_modes == ()
        assert result.flexible.physical_modes == result.flexible.frequencies_hz == ()
        assert result.flexible.factorization_count == (0 if failure == "cancel-before" else 1)
        assert owner._handle is None
    finally:
        owner.close()


def test_free_unclassified_backend_error_propagates(monkeypatch):
    owner = api.create_analysis(definition=_free_definition())
    original = owner_module.factorize_cached
    error = RuntimeError("unclassified free modal programming failure")
    def broken(*args, **kwargs):
        handle = original(*args, **kwargs)
        def fail(_rhs):
            raise error
        monkeypatch.setattr(handle, "solve", fail)
        return handle
    monkeypatch.setattr(owner_module, "factorize_cached", broken)
    try:
        with pytest.raises(RuntimeError, match="unclassified free modal programming failure") as caught:
            owner.modes(api.B2ReferenceModesRequest())
        assert caught.value is error
        assert owner._handle is None
    finally:
        owner.close()


@pytest.mark.parametrize("operator", ("K", "C"))
def test_actual_operator_rigid_field_corruption_is_not_projected_away(operator):
    owner = api.create_analysis(definition=_free_definition())
    try:
        if operator == "K":
            altered = owner._K.tolil(); altered[0, 0] += 1.
            owner._K = altered.tocsc()
        else:
            altered = owner._C.tolil(); altered[0, 0] += 1.
            owner._C = altered.tocsc()
        result = owner.modes(api.B2ReferenceModesRequest(count=1))
        assert result.status == result.flexible.status == "failed"
        assert "rigid field" in result.failure_reason
        assert result.flexible.factorization_count == 0
        assert result.rigid_modes == result.flexible.physical_modes == ()
    finally:
        owner.close()


def test_extra_positive_inertia_mechanism_is_not_hidden_by_rigid_projection():
    definition = _free_definition("q4")
    K, C = _dense_operators(definition)
    M = _dense_mass(definition)
    rigid = _canonical_fields(definition, np.zeros(3))
    # Produce a physical field satisfying the same point laws and orthogonal
    # to all six rigid fields. Retain a direction with actual positive inertia.
    basis = linalg.null_space(np.vstack((C, rigid.T)))
    kinetic = np.diag(basis.T@M@basis)
    mechanism = basis[:, np.argmax(kinetic)]
    assert mechanism@M@mechanism > 1e-8*np.linalg.norm(M, 2)
    np.testing.assert_allclose(C@mechanism, 0., atol=2e-13)
    np.testing.assert_allclose(rigid.T@mechanism, 0., atol=2e-13)
    projector = np.eye(54)-np.outer(mechanism, mechanism)/np.dot(mechanism, mechanism)
    corrupted = projector.T@K@projector
    np.testing.assert_allclose(corrupted@mechanism, 0., atol=2e-12*np.linalg.norm(K, 2))
    np.testing.assert_allclose(corrupted@rigid, K@rigid, atol=2e-12*np.linalg.norm(K, 2))
    owner = api.create_analysis(definition=definition)
    try:
        owner._K = sparse.csc_matrix(corrupted)
        result = owner.modes(api.B2ReferenceModesRequest(count=1))
        assert result.status == result.flexible.status == "failed"
        assert any(word in result.failure_reason for word in ("mechanism", "near-zero", "factorization", "scaling"))
        assert result.rigid_modes == result.flexible.physical_modes == ()
    finally:
        owner.close()


def _assert_mapped_modal_subspaces(reference, changed, mass, map_to_reference):
    assert reference.status == changed.status == "ok", (reference.failure_reason, changed.failure_reason)
    assert reference.flexible.status == changed.flexible.status == "ok"
    assert len(reference.flexible.physical_modes) == len(changed.flexible.physical_modes) == 3
    assert reference.flexible.factorization_count == changed.flexible.factorization_count == 1
    np.testing.assert_allclose(changed.flexible.frequencies_hz, reference.flexible.frequencies_hz,
                               rtol=2e-8)
    for field, count in (("rigid_modes", 6), ("physical_modes", 3)):
        base_result = reference if field == "rigid_modes" else reference.flexible
        changed_result = changed if field == "rigid_modes" else changed.flexible
        base = np.column_stack(getattr(base_result, field))
        mapped = map_to_reference@np.column_stack(getattr(changed_result, field))
        np.testing.assert_allclose(mapped.T@mass@mapped, np.eye(count), atol=1e-7)
        np.testing.assert_allclose(linalg.svdvals(base.T@mass@mapped), np.ones(count), atol=2e-7)
    assert changed.rigid_flexible_mass_overlap <= 1e-7


@pytest.mark.parametrize("kind", ("mixed", "s3"))
def test_free_finite_pairs_and_rigid_inertia_are_covariant_under_frame_and_triangle_reversal(kind):
    definition = _free_definition(kind)
    reverse = replace(definition, shells=tuple(
        replace(shell, node_ids=(shell.node_ids[0], shell.node_ids[2], shell.node_ids[1]))
        if type(shell) is api.S3V2DPanelShell else shell for shell in definition.shells))
    rotation = Rotation.from_rotvec((.2, -.4, .3)).as_matrix()
    translation = np.array((.4, -.2, .7))
    moved = replace(definition,
        shell_nodes_m={node: tuple(rotation@np.asarray(point)+translation)
                       for node, point in definition.shell_nodes_m.items()},
        shells=tuple(replace(shell, reference_normal=tuple(rotation@np.asarray(shell.reference_normal)))
                     for shell in definition.shells),
        members=tuple(replace(member,
            reference_points_m=tuple(tuple(rotation@np.asarray(point)+translation)
                                     for point in member.reference_points_m),
            local_y_global=tuple(rotation@np.asarray(member.local_y_global)))
                      for member in definition.members))
    owners = []
    try:
        for item in (definition, reverse, moved):
            owners.append(api.create_analysis(definition=item))
        base, reversed_result, moved_result = tuple(
            owner.modes(api.B2ReferenceModesRequest(count=3)) for owner in owners)
        mass = _dense_mass(definition)
        frame = linalg.block_diag(rotation, rotation)
        transform = np.kron(np.eye(9), frame)
        _assert_mapped_modal_subspaces(base, reversed_result, mass, np.eye(54))
        _assert_mapped_modal_subspaces(base, moved_result, mass, transform.T)
        np.testing.assert_allclose(reversed_result.rigid_origin_m, base.rigid_origin_m, atol=2e-15)
        np.testing.assert_allclose(moved_result.rigid_origin_m,
                                   rotation@np.asarray(base.rigid_origin_m)+translation, atol=2e-15)
        np.testing.assert_allclose(reversed_result.canonical_rigid_modes, base.canonical_rigid_modes,
                                   rtol=0., atol=2e-15)
        np.testing.assert_allclose(np.column_stack(moved_result.canonical_rigid_modes),
                                   transform@np.column_stack(base.canonical_rigid_modes)@frame.T,
                                   rtol=0., atol=2e-15)
        np.testing.assert_allclose(reversed_result.rigid_kinetic_gram, base.rigid_kinetic_gram,
                                   rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(moved_result.rigid_kinetic_gram,
                                   frame@np.asarray(base.rigid_kinetic_gram)@frame.T,
                                   rtol=2e-13, atol=2e-15)
        for item, result in ((reverse, reversed_result), (moved, moved_result)):
            canonical = np.column_stack(result.canonical_rigid_modes)
            np.testing.assert_allclose(result.rigid_kinetic_gram,
                                       canonical.T@_dense_mass(item)@canonical,
                                       rtol=2e-13, atol=2e-15)
    finally:
        for owner in owners:
            owner.close()


def test_free_finite_pairs_and_rigid_inertia_in_equivalent_m_n_and_mm_kn_units():
    definition = _free_definition("mixed")
    length_scale, force_scale = 1000., .001
    section_scale = np.array((1., 1., 1., length_scale, length_scale, length_scale))
    section_outer = np.outer(section_scale, section_scale)
    members = tuple(replace(member,
        reference_points_m=tuple(tuple(length_scale*np.asarray(point))
                                 for point in member.reference_points_m),
        section=api.LinearSection.generalized(
            force_scale*np.asarray(member.section.stiffness)*section_outer),
        mass=api.MassMetadata(member.mass.source_id,
            tuple(length_scale*np.asarray(member.mass.origin_local_m)),
            force_scale/length_scale**2*np.asarray(member.mass.matrix_per_m)*section_outer))
                    for member in definition.members)
    changed = replace(definition,
        shell_nodes_m={node: tuple(length_scale*np.asarray(point))
                       for node, point in definition.shell_nodes_m.items()},
        shells=tuple(replace(shell, E=shell.E*force_scale/length_scale**2,
            thickness_m=shell.thickness_m*length_scale,
            density_kg_m3=shell.density_kg_m3*force_scale/length_scale**4)
                     for shell in definition.shells),
        members=members,
        points=tuple(replace(point, offset_local_m=tuple(length_scale*np.asarray(point.offset_local_m)))
                     for point in definition.points))
    # Both definitions keep identity T. Unit conversion changes physical input
    # quantities and the numerical normalization of the resulting modes.
    pose_scale = np.tile((length_scale,)*3+(1.,)*3, 9)
    work_scale = length_scale*force_scale
    owners = []
    try:
        for item in (definition, changed):
            owners.append(api.create_analysis(definition=item))
        base, converted = tuple(owner.modes(api.B2ReferenceModesRequest(count=3)) for owner in owners)
        _assert_mapped_modal_subspaces(base, converted, _dense_mass(definition),
                                      np.diag(np.sqrt(work_scale)/pose_scale))
        np.testing.assert_allclose(converted.rigid_origin_m,
                                   length_scale*np.asarray(base.rigid_origin_m), atol=2e-12)
        canonical_scale = np.diag((1/length_scale,)*3+(1.,)*3)
        np.testing.assert_allclose(np.column_stack(converted.canonical_rigid_modes),
            pose_scale[:, None]*np.column_stack(base.canonical_rigid_modes)@canonical_scale,
            rtol=0., atol=2e-12)
        np.testing.assert_allclose(converted.rigid_kinetic_gram,
            work_scale*canonical_scale.T@np.asarray(base.rigid_kinetic_gram)@canonical_scale,
            rtol=2e-13, atol=2e-15)
        canonical = np.column_stack(converted.canonical_rigid_modes)
        np.testing.assert_allclose(converted.rigid_kinetic_gram,
                                   canonical.T@_dense_mass(changed)@canonical,
                                   rtol=2e-13, atol=2e-15)
    finally:
        for owner in owners:
            owner.close()
