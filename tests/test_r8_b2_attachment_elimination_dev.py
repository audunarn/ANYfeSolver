"""Prepared independent bounded full-KKT and actual physical-caller controls."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver._b2_attachment_elimination_dev import prepare_attachment_elimination
import anyfesolver._b2_attachment_elimination_dev as elimination


def raw_fixture(*, groups=1, nodes=3, nonsymmetric=False):
    """New mathematical physical patch rows, independently assembled by hand."""
    each = 6*nodes+6
    n, m = groups*each, groups*nodes*5
    k = np.diag(np.arange(1., n+1.))
    c = np.zeros((m, n))
    items = []
    for group in range(groups):
        beam = tuple(range(group*each+6*nodes, (group+1)*each))
        for node in range(nodes):
            arm = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))[node % 3]
            x, y, z = arm
            skew = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
            shell = tuple(range(group*each+6*node, group*each+6*node+6))
            row = group*5*nodes+5*node
            c[row:row+3, shell[:3]] = np.eye(3)
            c[row:row+3, beam[:3]] = -np.eye(3)
            c[row:row+3, beam[3:]] = skew
            director = np.array(((0., 1., 0.), (-1., 0., 0.)))
            c[row+3:row+5, shell[3:]] = director
            c[row+3:row+5, beam[3:]] = -director
            items.append(SimpleNamespace(footprint_id=f"patch-{group}", shell_node=group*nodes+node+1,
                row_count=5, row_start=row, shell_slots=shell, beam_slots=beam))
    j = c.T.copy()
    if nonsymmetric:
        j[0, 0] += 1e-7
        j[6*nodes, 0] -= 2e-7
    a = np.block([[k, j], [c, np.zeros((m, m))]])
    return sparse.csc_matrix(a), n, sparse.eye(n, format="csr"), tuple(items)


def physical_definition(*, affine=False, supported_attachment=False, quiet_affine=False, topology="q4"):
    """New two-Q4/two-B2 connected fixture attached to fullyfree node11."""
    nodes = {1:(0.,0.,0.), 2:(1.,0.,0.), 3:(2.,0.,0.),
             4:(0.,1.,0.), 5:(1.,1.,0.), 6:(2.,1.,0.)}
    shells = tuple(api.Q4PanelShell(f"new-shell-{i}", ids, 1000., .3, .15)
                   for i, ids in enumerate(((1,2,5,4), (2,3,6,5))))
    patch_cells = ((1,2,5,4),)
    if topology != "q4":
        assert topology in ("s3", "mixed")
        left = tuple(api.S3V2DPanelShell(f"new-left-{i}", ids, 1000., .3, .15, (0.,0.,1.))
                     for i, ids in enumerate(((1,2,5), (1,5,4))))
        right = (shells[1],) if topology == "mixed" else tuple(
            api.S3V2DPanelShell(f"new-right-{i}", ids, 1000., .3, .15, (0.,0.,1.))
            for i, ids in enumerate(((2,3,6), (2,6,5))))
        shells = left+right
        patch_cells = ((1,2,5), (1,5,4))
    section = api.LinearSection.isotropic(E=2000., nu=.25, area=.04,
        Iy=.04*.2**2/12., Iz=.04*.2**2/12., J=2*.04*.2**2/12., ky=.8, kz=.8)
    members = tuple(api.PhysicalB2Member(f"new-beam-{i}", f"new-square-{i}",
        (10+i,11+i), ((.5+i,.5,0.),(1.5+i,.5,0.)), (0.,1.,0.), "square", section)
        for i in range(2))
    group = api.B2PhysicalRigidFootprint("new-patch", (1,2,5,4),
        10 if supported_attachment else 11, (0.,0.,1.), ((1.,0.),(0.,1.),(0.,0.)), patch_cells)
    free = [i for i in range(54) if i not in range(36,42)]
    t = sparse.eye(54, format="lil")[:, free]
    u0 = np.zeros(54)
    if affine:
        t[1,0] = .5
        u0[36], u0[40] = 1e-4, -2e-4
        if not quiet_affine:
            u0[37] = 1e-4
    return api.B2PhysicalRigidFootprintDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, (group,), t.tocsr(), u0)


@pytest.mark.parametrize("nonsymmetric", (False, True))
def test_raw_general_kkt_arbitrary_rhs_and_beta_recovery(nonsymmetric):
    a, n, t, items = raw_fixture(nonsymmetric=nonsymmetric)
    maps, reason = prepare_attachment_elimination(a, n, t, items)
    assert reason == "eligible" and maps is not None
    assert maps.info["eliminated_primal"] == maps.info["eliminated_multipliers"] == 6
    assert maps.info["retained_constraints"] == 9
    if nonsymmetric:
        assert np.max(abs((a-a.T).data)) > 0
        assert np.max(abs((maps.left-maps.right.T).data)) > 0
    rhs = np.arange(1., a.shape[0]+1.)*1e-3
    rhs[n:] *= -1.  # Nonzero alpha AND beta RHS, including correction roles.
    handle = maps.factorize()
    actual = handle.solve(rhs)
    expected = np.linalg.solve(a.toarray(), rhs)  # Independent full, bounded oracle.
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(a @ actual, rhs, rtol=1e-11, atol=1e-13)
    assert handle.factorization_count == handle.solve_count == 1
    both = np.column_stack((rhs, -.3*rhs))
    np.testing.assert_allclose(handle.solve_many(both), np.linalg.solve(a.toarray(), both), rtol=1e-11, atol=1e-13)
    assert handle.factorization_count == 1 and handle.solve_count == 3


@pytest.mark.parametrize("side", ("right", "left"))
def test_actual_foreign_cross_group_coefficients_refused(side):
    a, n, t, items = raw_fixture(groups=2)
    a = a.tolil()
    if side == "right":
        a[n+15, 18] = 1e-3
    else:
        a[18, n+15] = 1e-3
    maps, reason = prepare_attachment_elimination(a.tocsc(), n, t, items)
    assert maps is None and reason == f"foreign_{side}_attachment_coupling"


@pytest.mark.parametrize("kind", ("right_rank", "left_rank", "partial_support", "shared_column", "compliance"))
def test_local_ineligibility_falls_back_before_global_factor(kind):
    a, n, t, items = raw_fixture()
    a = a.tolil()
    if kind == "right_rank":
        a[n:, 23] = 0.
    elif kind == "left_rank":
        a[18, n:] = 0.
    elif kind == "partial_support":
        # The helper sees a partially supported physical beam row.
        t = t.tolil(); t[23,23] = 0.; t = t.tocsr()
    elif kind == "shared_column":
        t = t.tolil(); t[0,18] = .5; t = t.tocsr()
    else:
        a[n,n] = -1.
    maps, reason = prepare_attachment_elimination(a.tocsc(), n, t, items)
    assert maps is None and reason != "eligible"


def test_patch_bound_is_eligibility_only_and_overlapping_slots_refused():
    a, n, t, items = raw_fixture(nodes=17)
    maps, reason = prepare_attachment_elimination(a, n, t, items)
    assert maps is None and reason == "bounded_patch_incidence_required"
    a, n, t, items = raw_fixture(groups=2)
    changed = list(items)
    changed[3] = SimpleNamespace(**{**vars(changed[3]), "shell_slots": changed[0].shell_slots})
    maps, reason = prepare_attachment_elimination(a, n, t, tuple(changed))
    assert maps is None and reason == "overlapping_attachment_blocks"


def test_compiled_operator_and_support_binding_owns_inputs():
    a, n, t, items = raw_fixture()
    maps, _ = prepare_attachment_elimination(a, n, t, items)
    original = a.toarray().copy()
    signature = maps.info["operator_support_signature"]
    a.data[:] *= 2.; t.data[:] = .5
    rhs = np.ones(original.shape[0])
    np.testing.assert_allclose(maps.factorize().solve(rhs), np.linalg.solve(original, rhs), rtol=1e-11, atol=1e-13)
    altered, reason = prepare_attachment_elimination(a, n, t, items)
    assert altered is None and reason == "fully_free_beam_selector_required"
    assert maps.info["operator_support_signature"] == signature


@pytest.mark.parametrize("topology,affine", (("q4",False), ("q4",True), ("s3",False), ("mixed",False)))
def test_new_fullyfree_endpoint_caller_matches_independent_full_kkt(topology, affine, monkeypatch):
    definition = physical_definition(affine=affine, topology=topology)
    owner = api.create_analysis(definition=definition)
    try:
        assert owner._attachment_elimination_info["status"] == "eligible"
        load = np.zeros(54); load[[14,32,50]] = (-1.,-.5,-.25)
        before = load.copy()
        # Independent full raw stored solve, not the reduced maps or handle.
        rhs = np.r_[owner._T.T @ (load-owner._K @ owner._u0), -owner._C @ owner._u0]
        z = np.linalg.solve(owner._physical_scaled_kkt.toarray(), owner._factor_scale*rhs)
        answer = owner._factor_scale*z
        u = owner._T @ answer[:owner._T.shape[1]]+owner._u0
        candidates = []
        original_correct = owner._correct_physical_equilibrium
        def capture(*args):
            candidate = original_correct(*args)
            candidates.append(candidate.copy())
            return candidate
        monkeypatch.setattr(owner, "_correct_physical_equilibrium", capture)
        result = owner.solve_linear(load, recover_members={},
            selected_footprints=api.B2PhysicalRigidFootprintSelection(("new-patch",)))
        if result.status != "completed" and candidates:
            cu = owner._T @ candidates[-1][:owner._T.shape[1]]+owner._u0
            gap = owner._C @ cu
            scale = owner._abs_C @ abs(cu)
            bad = np.flatnonzero(abs(gap) > 1e-9*scale)
            details = [(int(i), float(gap[i]), float(scale[i])) for i in bad]
        else:
            details = []
        assert result.status == "completed", (result.failure_reason, details,
            owner._physical_equilibrium_ratio(candidates[-1], load) if candidates else None)
        np.testing.assert_allclose(result.displacements, u, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(result.constraint_multipliers, answer[owner._T.shape[1]:], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result.affine_reactions, owner._K @ np.asarray(result.displacements)+
            owner._C.T @ np.asarray(result.constraint_multipliers)-load, rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(load, before)
        assert result.factorization_count == 1 and result.footprint_wrenches
        q = np.linalg.solve((owner._T.T @ owner._T).toarray(), owner._T.T @ (np.asarray(result.displacements)-owner._u0))
        physical = np.r_[q, result.constraint_multipliers]
        assert owner._physical_equilibrium_ratio(physical, load) <= 1.
        assert owner._physical_work_ratio(physical, load) <= 1.
        # Prescribed work identity, with actual recovered multipliers/gap.
        reaction = np.asarray(result.affine_reactions)
        defect = np.asarray(result.displacements) @ (owner._K @ result.displacements)-np.asarray(result.displacements) @ load
        identity = q @ (owner._T.T @ reaction)-np.asarray(result.constraint_multipliers) @ result.constraint_gaps+owner._u0 @ reaction
        assert abs(defect-identity) <= 1e-11
    finally:
        owner.close()


class BiasedFirstAttachmentSolve:
    def __init__(self, handle, scale, *, always=False):
        self.handle, self.scale, self.always = handle, scale, always
        self.calls, self.right_hand_sides = 0, []

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def solve(self, rhs):
        self.calls += 1
        self.right_hand_sides.append(np.asarray(rhs).copy())
        value = self.handle.solve(rhs).copy()
        if self.calls == 1 or self.always:
            # Disturb a selected beam displacement, producing an actual gap
            # as well as force residual; singleton restoration cannot mask it.
            at = int(self.handle.maps.beam[0])
            value[at] += 1e-4/self.scale[at]
        return value


def test_new_batch_prefix_reuse_refusal_and_original_correction_rhs():
    owner = api.create_analysis(definition=physical_definition())
    load = np.zeros(54); load[[14,32,50]] = (-1.,-.5,-.25)
    token = CancellationToken()
    before = load.copy()
    try:
        assert owner._attachment_elimination_info["status"] == "eligible"
        owner._handle = BiasedFirstAttachmentSolve(owner._handle, owner._factor_scale)
        result = owner.solve_linear_many((("first",load), ("later",2*load)), selected_dofs=(14,32,50),
            recover_members={}, cancellation_token=token, on_case=lambda *_: token.cancel("completed prefix"))
        assert result.status == "cancelled" and len(result.cases) == 1 and result.cases[0].case_id == "first"
        assert result.factorization_count == 1
        assert owner._handle.calls == 2 and result.solve_count == 2
        nq = owner._T.shape[1]
        assert np.count_nonzero(owner._handle.right_hand_sides[1][nq:])
        count = owner._handle.solve_count
        # Nonzero alpha RHS of an original-target correction uses SAME factor.
        target = np.arange(1., len(owner._factor_scale)+1.)*1e-7
        actual = owner._handle.solve(-owner._factor_scale*target)
        expected = np.linalg.solve(owner._physical_scaled_kkt.toarray(), -owner._factor_scale*target)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        assert owner._handle.factorization_count == 1 and owner._handle.solve_count == count+1
        later = owner.solve_linear(load, recover_members={})
        assert later.status == "completed", later.failure_reason
        np.testing.assert_array_equal(load, before)
        with pytest.raises(RuntimeError):
            owner._handle.solve(np.full_like(owner._factor_scale, np.nan))
    finally:
        owner.close()
    assert owner._attachment_elimination is None
    with pytest.raises(RuntimeError, match="closed"):
        owner.solve_linear(load)


def test_unimproved_eligible_candidate_stays_failed_without_new_factor():
    owner = api.create_analysis(definition=physical_definition())
    load = np.zeros(54); load[14] = -1.
    before = load.copy()
    try:
        owner._handle = BiasedFirstAttachmentSolve(owner._handle, owner._factor_scale, always=True)
        result = owner.solve_linear(load, recover_members={})
        assert result.status == "failed" and result.failure_reason
        assert owner._handle.factorization_count == 1 and owner._handle.calls == 2
        np.testing.assert_array_equal(load, before)
    finally:
        owner.close()


def test_sparse_budget_uses_bounded_kp_intermediate(monkeypatch):
    # Independent support sets: L@K has 12 entries, K@P has only four.
    # Re-associating the actual allocation would invalidate the K@P screen.
    products = []
    class Support:
        def __init__(self, name, rows):
            self.name, self.rows = name, rows
        def __matmul__(self, other):
            rows = [set().union(*(other.rows[j] for j in row)) for row in self.rows]
            products.append((self.name, other.name, sum(map(len, rows))))
            return Support(f"({self.name}@{other.name})", rows)
    k = Support("K", [{0,1,2},{3,4,5},{0,1,2},{3,4,5}])
    p = Support("P", [{0},{0},{0},{1},{1},{1}])
    l = Support("L", [{0,1},{2,3}])
    j = Support("J", [{0},{1},{0},{1}])
    c = Support("C", [{0,1,2,3,4,5}])
    monkeypatch.setattr(elimination.sparse, "bmat", lambda blocks, **_: blocks)
    elimination._reduced_operator(l, k, p, j, c)
    assert products[0] == ("K", "P", 4)
    assert products[1] == ("L", "(K@P)", 4)
    assert all(pair[:2] != ("L", "K") for pair in products)


def test_supported_endpoint_public_owner_keeps_legacy_factor_path():
    owner = api.create_analysis(definition=physical_definition(supported_attachment=True))
    try:
        assert owner._attachment_elimination_info == {"status":"legacy_fallback", "reason":"fully_free_beam_selector_required"}
        load = np.zeros(54); load[14] = -1.
        assert owner.solve_linear(load, recover_members={}).status == "completed"
    finally:
        owner.close()


def test_quiet_y_affine_input_preserves_typed_numerical_refusal():
    # Preserved source2/3 input: nonzero u0 excludes the existing exact gap
    # certificate; cancellation in a physically quiet y field remains refused.
    owner = api.create_analysis(definition=physical_definition(affine=True, quiet_affine=True))
    load = np.zeros(54); load[[14,32,50]] = (-1.,-.5,-.25)
    try:
        assert owner._attachment_elimination_info["status"] == "eligible"
        result = owner.solve_linear(load, recover_members={})
        assert result.status == "failed" and result.failure_reason == "panel point gap or equilibrium failed"
        assert owner._handle.factorization_count == 1
    finally:
        owner.close()
